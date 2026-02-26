import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import random

from reward_model import RewardComputer


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for text in texts:
            tokens = tokenizer.encode(text)
            for i in range(0, len(tokens) - max_length, max_length // 2):
                chunk = tokens[i:i + max_length]
                if len(chunk) == max_length:
                    self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


class AZRTrainer:
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu',
                 status_callback=None, rl_weight=0.1):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.training_history = []
        self.iteration = 0
        self.status_callback = status_callback
        self.current_loss = 0.0
        self.current_reward = 0.0
        # Вес для policy gradient loss (0 = только supervised, 1 = только RL)
        self.rl_weight = rl_weight
        # RewardComputer будет инициализирован после загрузки данных
        self.reward_computer = None

    def _init_reward_computer(self, texts):
        """Инициализируем RewardComputer с реальными данными для статистики биграмм"""
        self.reward_computer = RewardComputer(self.tokenizer, reference_texts=texts)

    def train_step(self, batch):
        self.model.train()
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        logits, loss = self.model(x, y)
        return loss

    @torch.no_grad()
    def _val_loss(self, val_loader):
        """Вычислить loss на валидационной выборке"""
        self.model.eval()
        total_loss = 0.0
        n = 0
        for batch in val_loader:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            if loss is not None:
                total_loss += loss.item()
                n += 1
        return total_loss / max(n, 1)

    def self_play_with_logprobs(self, prompts, max_length=30, temperature=0.8):
        """
        Генерация с сохранением log-вероятностей для REINFORCE.
        Возвращает: список (текст, sum_log_prob) для каждого промпта.
        """
        self.model.train()  # train режим — нужны градиенты
        results = []

        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt)
            if len(tokens) == 0:
                continue
            idx = torch.tensor([tokens], dtype=torch.long, device=self.device)

            sum_log_prob = torch.tensor(0.0, device=self.device, requires_grad=False)
            log_probs_list = []
            generated_tokens = list(tokens)

            for _ in range(max_length):
                idx_cond = idx if idx.size(1) <= self.model.max_seq_len else idx[:, -self.model.max_seq_len:]
                logits, _ = self.model(idx_cond)
                logits = logits[:, -1, :] / temperature

                # Подавляем служебные токены
                logits[:, 0] = -float('Inf')  # <PAD>
                logits[:, 1] = -float('Inf')  # <UNK>

                log_prob_dist = F.log_softmax(logits, dim=-1)
                probs = log_prob_dist.exp()

                idx_next = torch.multinomial(probs, num_samples=1)
                token_id = idx_next.item()

                # Собираем log-вероятность выбранного токена
                log_probs_list.append(log_prob_dist[0, token_id])
                generated_tokens.append(token_id)
                idx = torch.cat((idx, idx_next), dim=1)

            generated_text = self.tokenizer.decode(generated_tokens)
            # sum log probs — дифференцируемая величина
            if log_probs_list:
                sum_log_prob = torch.stack(log_probs_list).sum()
            results.append((generated_text, sum_log_prob))

        return results

    def compute_reward(self, generated_texts):
        """Вычислить награды через RewardComputer (6 компонентов)"""
        if self.reward_computer is None:
            # Fallback если reward_computer не инициализирован
            rewards = []
            for text in generated_texts:
                tokens = self.tokenizer.encode(text)
                unique_ratio = len(set(tokens)) / max(len(tokens), 1)
                length_score = min(len(tokens) / 100.0, 1.0)
                rewards.append(unique_ratio * 0.5 + length_score * 0.5)
            return np.array(rewards) if rewards else np.array([0.0])

        rewards = []
        for text in generated_texts:
            result = self.reward_computer.compute_reward(text)
            rewards.append(result["total"])
        return np.array(rewards) if rewards else np.array([0.0])

    def reinforce_step(self, optimizer, prompts, num_self_play=5):
        """
        REINFORCE: генерируем текст, считаем награду, делаем policy gradient step.
        Возвращает (policy_loss float, avg_reward float).
        """
        selected = random.sample(prompts, min(num_self_play, len(prompts)))

        try:
            play_results = self.self_play_with_logprobs(selected, max_length=30)
        except Exception as e:
            print(f"Self-play error: {e}")
            return 0.0, 0.0

        if not play_results:
            return 0.0, 0.0

        texts = [r[0] for r in play_results]
        log_prob_sums = [r[1] for r in play_results]

        rewards_np = self.compute_reward(texts)
        rewards_tensor = torch.tensor(rewards_np, dtype=torch.float32, device=self.device)

        # Normalize rewards (baseline) — стабилизирует обучение
        if len(rewards_tensor) > 1:
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        # Policy gradient loss: -log_prob * reward (хотим максимизировать reward)
        policy_losses = []
        for log_prob, reward in zip(log_prob_sums, rewards_tensor):
            if isinstance(log_prob, torch.Tensor) and log_prob.requires_grad:
                policy_losses.append(-log_prob * reward)

        if not policy_losses:
            return 0.0, float(rewards_np.mean())

        policy_loss = torch.stack(policy_losses).mean()

        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()

        return float(policy_loss.item()), float(rewards_np.mean())

    def azr_train_epoch(self, dataloader, optimizer, num_self_play=5, max_iterations=0):
        epoch_loss = 0
        epoch_reward = 0
        num_batches = 0
        reward_count = 0

        sample_texts = [
            "Once upon a time",
            "The future of AI",
            "In a distant galaxy",
            "The secret to happiness",
            "Technology has changed"
        ]

        for batch_idx, batch in enumerate(dataloader):
            # --- Supervised loss step ---
            loss = self.train_step(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            self.current_loss = loss.item()

            # --- REINFORCE step каждые 10 батчей ---
            if batch_idx % 10 == 0:
                try:
                    policy_loss, avg_reward = self.reinforce_step(
                        optimizer, sample_texts, num_self_play=num_self_play
                    )
                    self.current_reward = avg_reward
                    epoch_reward += avg_reward
                    reward_count += 1

                    if batch_idx % 50 == 0 and policy_loss != 0.0:
                        print(f"  [RL] policy_loss={policy_loss:.4f} reward={avg_reward:.4f}")
                except Exception as e:
                    print(f"REINFORCE error: {e}")
                    self.current_reward = 0.0

            num_batches += 1
            self.iteration += 1

            if self.status_callback and batch_idx % 5 == 0:
                try:
                    self.status_callback({
                        'current_iteration': self.iteration,
                        'max_iterations': max_iterations,
                        'current_loss': float(self.current_loss),
                        'current_reward': float(self.current_reward),
                        'is_training': True
                    })
                except Exception as e:
                    print(f"Status callback error: {e}")

        avg_loss = epoch_loss / num_batches
        avg_reward = epoch_reward / max(reward_count, 1)

        return avg_loss, avg_reward

    def train_continuous(self, texts, max_iterations=1000000, batch_size=16, lr=3e-4,
                         save_every=1000, checkpoint_dir='checkpoints', val_split=0.1):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # Инициализируем RewardComputer на обучающих данных
        self._init_reward_computer(texts)

        full_dataset = TextDataset(texts, self.tokenizer, max_length=128)
        if len(full_dataset) == 0:
            print("ERROR: No samples in dataset! Check your text data.")
            return []

        # Train / validation split
        val_size = max(1, int(len(full_dataset) * val_split))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iterations)

        print(f"Starting AZR training: {train_size} train / {val_size} val samples")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        print(f"REINFORCE RL weight: {self.rl_weight}")

        epoch = 0
        best_val_loss = float('inf')

        try:
            while self.iteration < max_iterations:
                epoch += 1
                avg_loss, avg_reward = self.azr_train_epoch(
                    dataloader, optimizer,
                    max_iterations=max_iterations
                )
                scheduler.step()

                # Validation loss
                val_loss = self._val_loss(val_loader)
                overfit_flag = "⚠ OVERFIT" if val_loss > best_val_loss * 1.05 else ""
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                self.training_history.append({
                    'epoch': epoch,
                    'iteration': self.iteration,
                    'loss': avg_loss,
                    'val_loss': val_loss,
                    'reward': avg_reward,
                    'lr': optimizer.param_groups[0]['lr'],
                    'timestamp': datetime.now().isoformat()
                })

                print(f"Epoch {epoch} | Iter {self.iteration}/{max_iterations} | "
                      f"Loss: {avg_loss:.4f} | Val: {val_loss:.4f} {overfit_flag} | "
                      f"Reward: {avg_reward:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

                if self.iteration % save_every == 0:
                    self.save_checkpoint(checkpoint_dir / f"model_iter_{self.iteration}.pt")

            print("Training completed!")

            if self.status_callback:
                self.status_callback({
                    'current_iteration': self.iteration,
                    'max_iterations': max_iterations,
                    'current_loss': float(self.current_loss),
                    'current_reward': float(self.current_reward),
                    'is_training': False
                })

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()

        return self.training_history

    def save_checkpoint(self, path):
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'iteration': self.iteration,
                'training_history': self.training_history[-100:],
                'model_config': {
                    'vocab_size': self.model.vocab_size,
                    'd_model': self.model.d_model,
                    'max_seq_len': self.model.max_seq_len
                }
            }
            torch.save(checkpoint, path, _use_new_zipfile_serialization=False)
            print(f"✓ Checkpoint saved: {path.name}")
        except Exception as e:
            print(f"✗ Error saving checkpoint: {e}")
            try:
                fallback_path = str(path).replace('.pt', '_state.pt')
                torch.save(self.model.state_dict(), fallback_path)
                print(f"✓ Model state saved as fallback: {Path(fallback_path).name}")
            except Exception as e2:
                print(f"✗ Fallback also failed: {e2}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.iteration = checkpoint.get('iteration', 0)
        self.training_history = checkpoint.get('training_history', [])
        return checkpoint
