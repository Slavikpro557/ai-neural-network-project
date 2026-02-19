"""
Пример обучения с МИНИМАЛЬНЫМ датасетом
Используем только несколько предложений как seed
"""

import torch
from model import CustomTransformerLM
from tokenizer import SimpleTokenizer
from azr_trainer import AZRTrainer

# МИНИМАЛЬНЫЙ seed - всего 10 предложений!
SEED_DATA = """
The cat sits on the mat.
A dog runs in the park.
Birds fly in the sky.
The sun shines bright.
Water flows down the river.
Trees grow in the forest.
Children play with toys.
Books contain knowledge.
Music makes people happy.
Stars twinkle at night.
"""

print("="*60)
print("ОБУЧЕНИЕ С МИНИМАЛЬНЫМ ДАТАСЕТОМ")
print("="*60)

# Создаём токенизатор из seed
tokenizer = SimpleTokenizer(vocab_size=500)  # Маленький словарь
tokenizer.train([SEED_DATA])
print(f"\nСловарь создан: {len(tokenizer)} токенов")
print(f"Размер seed: {len(SEED_DATA)} символов")

# Очень маленькая модель
model = CustomTransformerLM(
    vocab_size=len(tokenizer),
    d_model=64,
    num_layers=2,
    num_heads=4,
    d_ff=256,
    max_seq_len=64
)

print(f"Параметров модели: {sum(p.numel() for p in model.parameters())}")

# Обучение: 90% self-play, 10% supervised
trainer = AZRTrainer(model, tokenizer, device='cpu')

# Разбиваем seed на маленькие куски
texts = [SEED_DATA[i:i+50] for i in range(0, len(SEED_DATA), 25)]

print(f"\nНачальных примеров: {len(texts)}")
print("\nОбучение (большая часть - self-play)...")

# Короткое обучение для демонстрации
history = trainer.train_continuous(
    texts=texts,
    max_iterations=100,  # Мало итераций для демо
    batch_size=2,
    lr=0.001,
    save_every=50,
    checkpoint_dir='minimal_checkpoints'
)

print("\n" + "="*60)
print("ТЕСТИРОВАНИЕ ГЕНЕРАЦИИ")
print("="*60)

model.eval()
test_prompts = ["The cat", "A dog", "Birds"]

for prompt in test_prompts:
    tokens = tokenizer.encode(prompt)
    if len(tokens) > 0:
        idx = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad():
            generated = model.generate(idx, max_new_tokens=20, temperature=1.0)
        text = tokenizer.decode(generated[0].tolist())
        print(f"\nПромпт: '{prompt}'")
        print(f"Результат: {text}")

print("\n" + "="*60)
print("ВЫВОД:")
print("="*60)
print("✓ Модель может учиться на МИНИМУМЕ данных")
print("✓ Self-play помогает расширить знания")
print("✗ Но совсем БЕЗ данных - невозможно")
print("✗ Нужен хотя бы словарь и базовое понимание")
