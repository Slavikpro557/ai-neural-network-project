import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import random


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
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu', status_callback=None):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.training_history = []
        self.iteration = 0
        self.status_callback = status_callback
        self.current_loss = 0.0
        self.current_reward = 0.0
        self.optimizer = None
        self.scheduler = None
        self.should_stop = False  # Флаг для остановки
        
    def train_step(self, batch):
        self.model.train()
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
        logits, loss = self.model(x, y)
        
        return loss
    
    def self_play_step(self, prompts, max_length=50, temperature=0.8):
        self.model.eval()
        generated_texts = []
        
        with torch.no_grad():
            for prompt in prompts:
                tokens = self.tokenizer.encode(prompt)
                if len(tokens) == 0:
                    continue
                idx = torch.tensor([tokens], dtype=torch.long, device=self.device)
                
                generated = self.model.generate(idx, max_new_tokens=max_length, 
                                               temperature=temperature, top_k=40)
                
                generated_tokens = generated[0].cpu().tolist()
                generated_text = self.tokenizer.decode(generated_tokens)
                generated_texts.append(generated_text)
        
        return generated_texts
    
    def compute_reward(self, generated_texts):
        rewards = []
        for text in generated_texts:
            tokens = self.tokenizer.encode(text)
            unique_ratio = len(set(tokens)) / max(len(tokens), 1)
            length_score = min(len(tokens) / 100.0, 1.0)
            reward = unique_ratio * 0.5 + length_score * 0.5
            rewards.append(reward)
        
        return np.array(rewards) if len(rewards) > 0 else np.array([0.0])
    
    def azr_train_epoch(self, dataloader, optimizer, num_self_play=5, max_iterations=0):
        epoch_loss = 0
        epoch_reward = 0
        num_batches = 0
        reward_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if self.should_stop:
                print("Training stopped by user")
                break
                
            loss = self.train_step(batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            self.current_loss = loss.item()
            
            if batch_idx % 10 == 0:
                sample_texts = [
                    "Once upon a time",
                    "The future of AI",
                    "In a distant galaxy",
                    "The secret to happiness",
                    "Technology has changed"
                ]
                
                try:
                    prompts = random.sample(sample_texts, min(num_self_play, len(sample_texts)))
                    generated = self.self_play_step(prompts, max_length=30)
                    if len(generated) > 0:
                        rewards = self.compute_reward(generated)
                        self.current_reward = float(rewards.mean())
                        epoch_reward += self.current_reward
                        reward_count += 1
                except Exception as e:
                    print(f"Self-play error: {e}")
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
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        avg_reward = epoch_reward / max(reward_count, 1)
        
        return avg_loss, avg_reward
    
    def train_continuous(self, texts, max_iterations=1000000, batch_size=16, lr=3e-4, 
                        save_every=1000, checkpoint_dir='checkpoints', resume_from=None):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        dataset = TextDataset(texts, self.tokenizer, max_length=128)
        if len(dataset) == 0:
            print("ERROR: No samples in dataset! Check your text data.")
            return []
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                               num_workers=0, pin_memory=False)
        
        # Resume from checkpoint if provided
        if resume_from and Path(resume_from).exists():
            print(f"Resuming from checkpoint: {resume_from}")
            checkpoint = self.load_checkpoint(resume_from, load_optimizer=True)
            print(f"Resumed from iteration {self.iteration}")
        else:
            # Create new optimizer and scheduler
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_iterations)
        
        print(f"Starting AZR training with {len(dataset)} samples")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        print(f"Starting from iteration: {self.iteration}")
        print(f"Target iterations: {max_iterations}")
        
        epoch = 0
        self.should_stop = False
        
        try:
            while self.iteration < max_iterations and not self.should_stop:
                epoch += 1
                avg_loss, avg_reward = self.azr_train_epoch(dataloader, self.optimizer, max_iterations=max_iterations)
                
                if self.should_stop:
                    break
                    
                self.scheduler.step()
                
                self.training_history.append({
                    'epoch': epoch,
                    'iteration': self.iteration,
                    'loss': avg_loss,
                    'reward': avg_reward,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'timestamp': datetime.now().isoformat()
                })
                
                print(f"Epoch {epoch} | Iter {self.iteration}/{max_iterations} | "
                      f"Loss: {avg_loss:.4f} | Reward: {avg_reward:.4f} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                if self.iteration % save_every == 0:
                    self.save_checkpoint(checkpoint_dir / f"model_iter_{self.iteration}.pt", save_optimizer=True)
                    # Автоочистка старых чекпоинтов (оставляем только 7 последних)
                    self.cleanup_old_checkpoints(checkpoint_dir, keep_last=7)
            
            if self.should_stop:
                print("Training paused. Save checkpoint to resume later.")
                self.save_checkpoint(checkpoint_dir / f"model_paused_{self.iteration}.pt", save_optimizer=True)
            else:
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
            print("\nTraining interrupted by user (Ctrl+C)")
            print("Saving checkpoint...")
            self.save_checkpoint(checkpoint_dir / f"model_interrupted_{self.iteration}.pt", save_optimizer=True)
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            print("Saving checkpoint...")
            self.save_checkpoint(checkpoint_dir / f"model_error_{self.iteration}.pt", save_optimizer=True)
        
        return self.training_history
    
    def cleanup_old_checkpoints(self, checkpoint_dir, keep_last=5):
        """
        Удаляет старые чекпоинты, оставляя только последние keep_last
        """
        try:
            import glob
            # Находим все чекпоинты model_iter_*.pt
            pattern = str(checkpoint_dir / "model_iter_*.pt")
            checkpoints = sorted(glob.glob(pattern), key=lambda x: Path(x).stat().st_mtime)
            
            if len(checkpoints) > keep_last:
                to_delete = checkpoints[:-keep_last]
                for cp in to_delete:
                    try:
                        Path(cp).unlink()
                        print(f"   🗑️ Deleted old checkpoint: {Path(cp).name}")
                    except Exception as e:
                        print(f"   ⚠️ Failed to delete {Path(cp).name}: {e}")
        except Exception as e:
            print(f"   ⚠️ Cleanup failed: {e}")
    
    def stop_training(self):
        """Остановить обучение (вызывается извне)"""
        self.should_stop = True
        print("Stop signal received, will pause after current batch...")
    
    def save_checkpoint(self, path, save_optimizer=False):
        """
        Сохранить checkpoint с возможностью сохранения optimizer state
        """
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'iteration': self.iteration,
                'training_history': self.training_history[-100:],
                'model_config': {
                    'vocab_size': self.model.vocab_size,
                    'd_model': self.model.d_model,
                    'max_seq_len': self.model.max_seq_len
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Сохраняем optimizer и scheduler для точного возобновления
            if save_optimizer and self.optimizer is not None:
                checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
                if self.scheduler is not None:
                    checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
            torch.save(checkpoint, path, _use_new_zipfile_serialization=False)
            print(f"✓ Checkpoint saved: {path.name}")
            return True
        except Exception as e:
            print(f"✗ Error saving checkpoint: {e}")
            try:
                fallback_path = str(path).replace('.pt', '_state.pt')
                torch.save(self.model.state_dict(), fallback_path)
                print(f"✓ Model state saved as fallback: {Path(fallback_path).name}")
                return True
            except Exception as e2:
                print(f"✗ Fallback also failed: {e2}")
                return False
    
    def load_checkpoint(self, path, load_optimizer=False):
        """
        Загрузить checkpoint с возможностью восстановления optimizer state
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.iteration = checkpoint.get('iteration', 0)
        self.training_history = checkpoint.get('training_history', [])
        
        # Восстанавливаем optimizer и scheduler если нужно
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✓ Optimizer state restored")
            
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("✓ Scheduler state restored")
        
        print(f"✓ Checkpoint loaded from iteration {self.iteration}")
        return checkpoint
