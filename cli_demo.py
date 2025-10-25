"""
CLI демонстрация AZR Model Trainer
Создание, обучение и использование модели без веб-интерфейса
"""

import torch
from pathlib import Path
from model import CustomTransformerLM, count_parameters
from tokenizer import SimpleTokenizer
from azr_trainer import AZRTrainer

def main():
    print("\n" + "="*70)
    print(" 🧠 AZR Model Trainer - CLI Demo ".center(70, "="))
    print("="*70 + "\n")
    
    # Настройки
    MODEL_NAME = "demo_model"
    BOOK_PATH = Path("books/example_book.txt")
    OUTPUT_DIR = Path("models") / MODEL_NAME
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Параметры модели
    VOCAB_SIZE = 5000
    D_MODEL = 256
    NUM_LAYERS = 6
    NUM_HEADS = 8
    MAX_SEQ_LEN = 256
    
    # Параметры обучения
    MAX_ITERATIONS = 100  # Для демо - маленькое число
    BATCH_SIZE = 8
    LEARNING_RATE = 3e-4
    
    print("📋 Конфигурация:")
    print(f"   Модель: {MODEL_NAME}")
    print(f"   Размерность: {D_MODEL}, Слоёв: {NUM_LAYERS}")
    print(f"   Книга: {BOOK_PATH}")
    print(f"   Итераций обучения: {MAX_ITERATIONS}")
    print()
    
    # Шаг 1: Загрузка текста
    print("📚 Шаг 1: Загрузка книги...")
    if not BOOK_PATH.exists():
        print(f"   ❌ Файл {BOOK_PATH} не найден!")
        print(f"   Создайте файл или измените BOOK_PATH в скрипте.")
        return
    
    with open(BOOK_PATH, 'r', encoding='utf-8') as f:
        book_text = f.read()
    
    print(f"   ✓ Загружено {len(book_text)} символов")
    print(f"   Первые 100 символов: '{book_text[:100]}...'")
    print()
    
    # Шаг 2: Создание и обучение токенизатора
    print("🔤 Шаг 2: Создание токенизатора...")
    tokenizer = SimpleTokenizer(vocab_size=VOCAB_SIZE)
    
    # Разбиваем текст на фрагменты для обучения
    chunks = [book_text[i:i+500] for i in range(0, len(book_text), 250)]
    tokenizer.train(chunks)
    
    print(f"   ✓ Токенизатор обучен. Словарь: {len(tokenizer)} токенов")
    
    # Сохраняем токенизатор
    tokenizer.save(OUTPUT_DIR / "tokenizer.pkl")
    print(f"   ✓ Токенизатор сохранён в {OUTPUT_DIR / 'tokenizer.pkl'}")
    print()
    
    # Шаг 3: Создание модели
    print("🏗️  Шаг 3: Создание модели...")
    model = CustomTransformerLM(
        vocab_size=len(tokenizer),
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_MODEL * 4,
        max_seq_len=MAX_SEQ_LEN,
        dropout=0.1
    )
    
    params = count_parameters(model)
    print(f"   ✓ Модель создана")
    print(f"   Параметров: {params:,}")
    print(f"   Память: ~{params * 4 / 1024 / 1024:.1f} MB (float32)")
    print()
    
    # Шаг 4: Обучение с AZR
    print("🚀 Шаг 4: Запуск AZR обучения...")
    print(f"   Это займёт несколько минут...")
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Устройство: {device.upper()}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    trainer = AZRTrainer(model, tokenizer, device=device)
    
    history = trainer.train_continuous(
        texts=chunks,
        max_iterations=MAX_ITERATIONS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        save_every=50,
        checkpoint_dir=OUTPUT_DIR / "checkpoints"
    )
    
    print(f"\n   ✓ Обучение завершено!")
    print(f"   Финальный loss: {history[-1]['loss']:.4f}")
    print(f"   Финальный reward: {history[-1]['reward']:.4f}")
    print()
    
    # Сохраняем обученную модель
    torch.save(model.state_dict(), OUTPUT_DIR / "model_trained.pt")
    print(f"   ✓ Модель сохранена в {OUTPUT_DIR / 'model_trained.pt'}")
    print()
    
    # Шаг 5: Тестирование генерации
    print("✨ Шаг 5: Тестирование генерации...")
    print()
    
    test_prompts = [
        "Искусственный интеллект",
        "Нейронные сети",
        "Будущее",
        "Обучение",
    ]
    
    for prompt in test_prompts:
        print(f"   Промпт: '{prompt}'")
        
        tokens = tokenizer.encode(prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=device)
        
        with torch.no_grad():
            generated = model.generate(
                idx, 
                max_new_tokens=50,
                temperature=0.8,
                top_k=40
            )
        
        generated_text = tokenizer.decode(generated[0].cpu().tolist())
        print(f"   → {generated_text}")
        print()
    
    print("="*70)
    print(" ✅ Демонстрация завершена! ".center(70, "="))
    print("="*70)
    print()
    print("📂 Результаты сохранены в:", OUTPUT_DIR)
    print()
    print("🎯 Следующие шаги:")
    print("   1. Запустите веб-интерфейс: python server.py")
    print("   2. Увеличьте MAX_ITERATIONS для лучших результатов")
    print("   3. Экспериментируйте с параметрами модели")
    print("   4. Попробуйте разные книги и стили текста")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Обучение прервано пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
