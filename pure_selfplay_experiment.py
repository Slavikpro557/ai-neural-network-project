"""
ЭКСПЕРИМЕНТАЛЬНЫЙ: Чистый Self-Play
Попытка обучения ПОЧТИ без данных

ИДЕЯ:
1. Используем предзаданный словарь (английские слова)
2. Модель генерирует случайный текст
3. Оцениваем "качество" через простые метрики
4. Модель учится генерировать "лучше"

ВНИМАНИЕ: Это proof-of-concept, не production код!
"""

import torch
import torch.nn as nn
import random
from model import CustomTransformerLM
from tokenizer import SimpleTokenizer

class PureSelfPlayTrainer:
    """Обучение ТОЛЬКО через self-play"""
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
    def generate_random_prompt(self):
        """Генерируем случайный промпт из словаря"""
        # Берём 1-3 случайных токена из словаря
        available_tokens = list(range(4, min(100, len(self.tokenizer))))
        num_tokens = random.randint(1, 3)
        tokens = random.sample(available_tokens, num_tokens)
        return torch.tensor([tokens], dtype=torch.long, device=self.device)
    
    def evaluate_quality(self, generated_text):
        """
        Оценка качества БЕЗ внешних данных
        Используем внутренние метрики:
        - Разнообразие токенов
        - Длина
        - Отсутствие повторов
        - "Плавность" (низкая perplexity на собственных данных)
        """
        tokens = self.tokenizer.encode(generated_text)
        
        if len(tokens) == 0:
            return 0.0
        
        # Метрика 1: Разнообразие
        diversity = len(set(tokens)) / len(tokens)
        
        # Метрика 2: Длина (штраф за слишком короткие)
        length_score = min(len(tokens) / 50.0, 1.0)
        
        # Метрика 3: Отсутствие повторяющихся последовательностей
        bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]
        bigram_diversity = len(set(bigrams)) / max(len(bigrams), 1)
        
        # Метрика 4: "Смысловость" (может ли модель предсказать сама себя)
        self.model.eval()
        with torch.no_grad():
            idx = torch.tensor([tokens[:-1]], dtype=torch.long, device=self.device)
            target = torch.tensor([tokens[1:]], dtype=torch.long, device=self.device)
            logits, loss = self.model(idx, target)
            # Низкий loss = модель "верит" в свои генерации
            confidence = torch.exp(-loss).item()
        
        # Комбинируем метрики
        quality = (
            diversity * 0.3 +
            length_score * 0.2 +
            bigram_diversity * 0.2 +
            confidence * 0.3
        )
        
        return quality
    
    def self_play_iteration(self, num_samples=10):
        """Одна итерация self-play"""
        samples = []
        rewards = []
        
        # Генерируем несколько вариантов
        self.model.eval()
        for _ in range(num_samples):
            prompt = self.generate_random_prompt()
            
            with torch.no_grad():
                generated = self.model.generate(
                    prompt, 
                    max_new_tokens=30,
                    temperature=1.0,
                    top_k=50
                )
            
            text = self.tokenizer.decode(generated[0].tolist())
            quality = self.evaluate_quality(text)
            
            samples.append(generated[0])
            rewards.append(quality)
        
        # Отбираем лучшие
        best_indices = sorted(range(len(rewards)), key=lambda i: rewards[i], reverse=True)[:5]
        
        # Обучаем на лучших примерах
        self.model.train()
        total_loss = 0
        
        for idx in best_indices:
            tokens = samples[idx]
            if len(tokens) < 2:
                continue
                
            x = tokens[:-1].unsqueeze(0).to(self.device)
            y = tokens[1:].unsqueeze(0).to(self.device)
            
            logits, loss = self.model(x, y)
            
            # Взвешиваем loss по качеству
            weighted_loss = loss * rewards[idx]
            
            self.optimizer.zero_grad()
            weighted_loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(best_indices)
        avg_reward = sum(rewards) / len(rewards)
        best_reward = max(rewards)
        
        return avg_loss, avg_reward, best_reward
    
    def train(self, iterations=100):
        """Полное обучение через self-play"""
        print("\n" + "="*60)
        print("PURE SELF-PLAY TRAINING")
        print("="*60)
        print("ВНИМАНИЕ: Экспериментальный подход!")
        print("Модель учится БЕЗ внешних данных\n")
        
        for iteration in range(iterations):
            loss, avg_reward, best_reward = self.self_play_iteration()
            
            if iteration % 10 == 0:
                print(f"Iter {iteration:3d} | Loss: {loss:.4f} | "
                      f"Avg Reward: {avg_reward:.4f} | Best: {best_reward:.4f}")
        
        print("\n" + "="*60)
        print("Обучение завершено!")
        print("="*60)


def main():
    print("="*60)
    print("ЭКСПЕРИМЕНТ: Обучение почти без данных")
    print("="*60)
    
    # Создаём ПРЕДЗАДАННЫЙ словарь (это минимальное "знание")
    # В реальности это можно взять из любого существующего токенизатора
    vocab = [
        '<PAD>', '<UNK>', '<BOS>', '<EOS>',
        'the', 'a', 'an', 'and', 'or', 'but',
        'is', 'are', 'was', 'were', 'be',
        'have', 'has', 'had', 'do', 'does',
        'can', 'could', 'will', 'would', 'should',
        'cat', 'dog', 'bird', 'fish', 'tree',
        'run', 'walk', 'fly', 'swim', 'sit',
        'big', 'small', 'fast', 'slow', 'happy',
        'red', 'blue', 'green', 'yellow', 'black'
    ]
    
    print(f"\nПредзаданный словарь: {len(vocab)} слов")
    print("(Это минимальное 'знание' которое есть у модели)")
    
    # Создаём токенизатор с предзаданным словарём
    tokenizer = SimpleTokenizer(vocab_size=len(vocab))
    for i, word in enumerate(vocab):
        tokenizer.token_to_id[word] = i
        tokenizer.id_to_token[i] = word
    
    # Маленькая модель
    model = CustomTransformerLM(
        vocab_size=len(vocab),
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        max_seq_len=64
    )
    
    print(f"Параметров: {sum(p.numel() for p in model.parameters())}")
    
    # Pure Self-Play обучение
    trainer = PureSelfPlayTrainer(model, tokenizer)
    trainer.train(iterations=50)  # Мало для демо
    
    # Тестирование
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ")
    print("="*60)
    
    model.eval()
    test_words = ['the', 'cat', 'dog', 'is']
    
    for word in test_words:
        if word in tokenizer.token_to_id:
            token_id = tokenizer.token_to_id[word]
            idx = torch.tensor([[token_id]])
            
            with torch.no_grad():
                generated = model.generate(idx, max_new_tokens=10, temperature=0.8)
            
            text = tokenizer.decode(generated[0].tolist())
            print(f"\n'{word}' → {text}")
    
    print("\n" + "="*60)
    print("ВЫВОДЫ:")
    print("="*60)
    print("✓ Self-play МОЖЕТ работать для языковых моделей")
    print("✗ НО требует предзаданный словарь (минимальное знание)")
    print("✗ Качество намного ниже чем с реальными данными")
    print("✓ Подходит для исследований и экспериментов")
    print("\nДля реального использования:")
    print("→ Используйте хотя бы небольшой seed dataset")
    print("→ Или предобученный токенизатор")
    print("→ Self-play как дополнение, не замена")


if __name__ == "__main__":
    main()
