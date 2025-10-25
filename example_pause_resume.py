"""
Пример использования Pause/Resume функционала
Показывает как:
1. Начать обучение
2. Остановить
3. Возобновить с другим количеством итераций
"""

import torch
from model import CustomTransformerLM
from tokenizer import SimpleTokenizer
from azr_trainer_resume import AZRTrainer
from pathlib import Path
import time
import threading

print("="*70)
print("ДЕМОНСТРАЦИЯ PAUSE/RESUME ОБУЧЕНИЯ")
print("="*70)

# Создаём простую модель
print("\n1. Создание модели...")
tokenizer = SimpleTokenizer(vocab_size=1000)

# Минимальный seed
seed_text = """
The cat sits on the mat. A dog runs in the park.
Birds fly in the sky. The sun shines bright.
Water flows down the river. Trees grow tall.
"""

tokenizer.train([seed_text])
print(f"Словарь: {len(tokenizer)} токенов")

model = CustomTransformerLM(
    vocab_size=len(tokenizer),
    d_model=64,
    num_layers=2,
    num_heads=4,
    d_ff=256,
    max_seq_len=64
)

print(f"Параметров: {sum(p.numel() for p in model.parameters())}")

# Создаём trainer
trainer = AZRTrainer(model, tokenizer, device='cpu')

# Подготовка данных
texts = [seed_text[i:i+50] for i in range(0, len(seed_text), 25)]
checkpoint_dir = Path("pause_resume_demo")
checkpoint_dir.mkdir(exist_ok=True)

print("\n" + "="*70)
print("СЦЕНАРИЙ 1: Начальное обучение (100 итераций)")
print("="*70)

# Обучение 1: 100 итераций
print("\nНачинаем обучение на 100 итераций...")
history1 = trainer.train_continuous(
    texts=texts,
    max_iterations=100,
    batch_size=4,
    lr=0.001,
    save_every=50,
    checkpoint_dir=checkpoint_dir
)

first_checkpoint = checkpoint_dir / "model_iter_100.pt"
print(f"\n✓ Первый этап завершён. Checkpoint: {first_checkpoint}")
print(f"  Текущая итерация: {trainer.iteration}")

# Тестовая генерация после первого этапа
print("\n📝 Генерация после 100 итераций:")
model.eval()
tokens = tokenizer.encode("The cat")
if len(tokens) > 0:
    idx = torch.tensor([tokens])
    with torch.no_grad():
        gen1 = model.generate(idx, max_new_tokens=15)
    text1 = tokenizer.decode(gen1[0].tolist())
    print(f"   '{text1}'")

print("\n" + "="*70)
print("СЦЕНАРИЙ 2: Возобновление с ИЗМЕНЕНИЕМ итераций (100 → 250)")
print("="*70)

# Создаём НОВЫЙ trainer для возобновления
print("\nСоздаём новую модель и загружаем checkpoint...")
model2 = CustomTransformerLM(
    vocab_size=len(tokenizer),
    d_model=64,
    num_layers=2,
    num_heads=4,
    d_ff=256,
    max_seq_len=64
)

trainer2 = AZRTrainer(model2, tokenizer, device='cpu')

# Возобновляем с НОВЫМ количеством итераций
print(f"\nВозобновляем обучение...")
print(f"  Было: до 100 итераций")
print(f"  Стало: до 250 итераций (добавили +150)")

history2 = trainer2.train_continuous(
    texts=texts,
    max_iterations=250,  # ИЗМЕНИЛИ!
    batch_size=4,
    lr=0.001,
    save_every=50,
    checkpoint_dir=checkpoint_dir,
    resume_from=first_checkpoint  # Возобновляем
)

print(f"\n✓ Второй этап завершён")
print(f"  Финальная итерация: {trainer2.iteration}")

# Тестовая генерация после второго этапа
print("\n📝 Генерация после 250 итераций:")
model2.eval()
tokens = tokenizer.encode("The cat")
if len(tokens) > 0:
    idx = torch.tensor([tokens])
    with torch.no_grad():
        gen2 = model2.generate(idx, max_new_tokens=15)
    text2 = tokenizer.decode(gen2[0].tolist())
    print(f"   '{text2}'")

print("\n" + "="*70)
print("СЦЕНАРИЙ 3: Ручная остановка и возобновление")
print("="*70)

# Создаём новую модель для демонстрации ручной остановки
model3 = CustomTransformerLM(
    vocab_size=len(tokenizer),
    d_model=64,
    num_layers=2,
    num_heads=4,
    d_ff=256,
    max_seq_len=64
)

trainer3 = AZRTrainer(model3, tokenizer, device='cpu')

# Функция для автоостановки через 3 секунды
def auto_stop():
    time.sleep(3)
    print("\n⏸️  Автоостановка через 3 секунды...")
    trainer3.stop_training()

# Запускаем автоостановку в фоне
stop_thread = threading.Thread(target=auto_stop)
stop_thread.start()

print("\nЗапускаем обучение, которое остановится через 3 секунды...")
print("(симулируем нажатие кнопки 'Stop')")

history3 = trainer3.train_continuous(
    texts=texts,
    max_iterations=1000,  # Большое число
    batch_size=4,
    lr=0.001,
    save_every=50,
    checkpoint_dir=checkpoint_dir
)

stop_thread.join()

paused_checkpoint = checkpoint_dir / f"model_paused_{trainer3.iteration}.pt"
print(f"\n✓ Обучение приостановлено на итерации {trainer3.iteration}")
print(f"  Checkpoint сохранён: {paused_checkpoint}")

# Возобновляем с другим количеством итераций
print(f"\n🔄 Возобновляем с новой целью: {trainer3.iteration + 100} итераций")

model4 = CustomTransformerLM(
    vocab_size=len(tokenizer),
    d_model=64,
    num_layers=2,
    num_heads=4,
    d_ff=256,
    max_seq_len=64
)

trainer4 = AZRTrainer(model4, tokenizer, device='cpu')

history4 = trainer4.train_continuous(
    texts=texts,
    max_iterations=trainer3.iteration + 100,  # Добавляем 100 итераций
    batch_size=4,
    lr=0.001,
    save_every=50,
    checkpoint_dir=checkpoint_dir,
    resume_from=paused_checkpoint
)

print(f"\n✓ Финальная итерация: {trainer4.iteration}")

print("\n" + "="*70)
print("ИТОГИ ДЕМОНСТРАЦИИ")
print("="*70)
print("\n✅ Что продемонстрировано:")
print("  1. Обучение можно остановить в любой момент")
print("  2. При возобновлении можно изменить max_iterations")
print("  3. Модель продолжает с того же места")
print("  4. Optimizer и scheduler восстанавливаются")
print("  5. История обучения сохраняется")

print("\n📁 Созданные checkpoints:")
for checkpoint in checkpoint_dir.glob("*.pt"):
    size = checkpoint.stat().st_size / 1024
    print(f"  - {checkpoint.name} ({size:.1f} KB)")

print("\n🎯 Как использовать в вашем проекте:")
print("  1. Замените azr_trainer.py на azr_trainer_resume.py")
print("  2. Добавьте кнопку 'Stop' в UI")
print("  3. Добавьте поле для изменения max_iterations при resume")
print("  4. Используйте resume_from параметр")

print("\n" + "="*70)
