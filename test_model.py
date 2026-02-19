import torch
from model import CustomTransformerLM, count_parameters
from tokenizer import SimpleTokenizer
from azr_trainer import AZRTrainer
from pathlib import Path

def test_basic_functionality():
    print("="*60)
    print("Тест AZR Model Trainer")
    print("="*60)
    
    print("\n1. Создание токенизатора...")
    tokenizer = SimpleTokenizer(vocab_size=1000)
    
    texts = [
        "Это тестовый текст для обучения модели.",
        "Нейронные сети учатся на данных.",
        "Искусственный интеллект становится умнее.",
        "Трансформеры революционизировали NLP.",
    ]
    
    tokenizer.train(texts)
    print(f"   ✓ Токенизатор создан. Размер словаря: {len(tokenizer)}")
    
    print("\n2. Тестирование токенизации...")
    test_text = "Искусственный интеллект"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    print(f"   Исходный текст: '{test_text}'")
    print(f"   Токены: {tokens}")
    print(f"   Декодированный: '{decoded}'")
    print(f"   ✓ Токенизация работает")
    
    print("\n3. Создание модели...")
    model = CustomTransformerLM(
        vocab_size=len(tokenizer),
        d_model=128,
        num_layers=4,
        num_heads=4,
        d_ff=512,
        max_seq_len=128,
        dropout=0.1
    )
    
    params = count_parameters(model)
    print(f"   ✓ Модель создана. Параметров: {params:,}")
    
    print("\n4. Тестирование forward pass...")
    sample_input = torch.randint(0, len(tokenizer), (2, 10))
    sample_target = torch.randint(0, len(tokenizer), (2, 10))
    
    logits, loss = model(sample_input, sample_target)
    print(f"   Input shape: {sample_input.shape}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Loss: {loss.item():.4f}")
    print(f"   ✓ Forward pass работает")
    
    print("\n5. Тестирование генерации...")
    model.eval()
    prompt_tokens = tokenizer.encode("Искусственный")
    idx = torch.tensor([prompt_tokens], dtype=torch.long)
    
    generated = model.generate(idx, max_new_tokens=20, temperature=1.0)
    generated_text = tokenizer.decode(generated[0].tolist())
    print(f"   Промпт: 'Искусственный'")
    print(f"   Сгенерировано: '{generated_text}'")
    print(f"   ✓ Генерация работает")
    
    print("\n6. Тестирование AZR Trainer...")
    trainer = AZRTrainer(model, tokenizer, device='cpu')
    
    print("   Запуск мини-обучения (5 итераций)...")
    history = trainer.train_continuous(
        texts=texts * 10,
        max_iterations=5,
        batch_size=2,
        lr=1e-3,
        save_every=10,
        checkpoint_dir='test_checkpoints'
    )
    
    print(f"   ✓ Обучение завершено. Итераций: {len(history)}")
    
    print("\n7. Тестирование сохранения/загрузки...")
    Path('test_output').mkdir(exist_ok=True)
    
    torch.save(model.state_dict(), 'test_output/model.pt')
    tokenizer.save('test_output/tokenizer.pkl')
    print("   ✓ Модель и токенизатор сохранены")
    
    model2 = CustomTransformerLM(
        vocab_size=len(tokenizer),
        d_model=128,
        num_layers=4,
        num_heads=4,
        d_ff=512,
        max_seq_len=128
    )
    model2.load_state_dict(torch.load('test_output/model.pt'))
    tokenizer2 = SimpleTokenizer.load('test_output/tokenizer.pkl')
    
    print("   ✓ Модель и токенизатор загружены")
    
    print("\n" + "="*60)
    print("✅ Все тесты пройдены успешно!")
    print("="*60)
    print("\nСистема готова к использованию!")
    print("Запустите сервер: python server.py")
    print("Откройте браузер: http://localhost:8000")
    print("="*60)

if __name__ == "__main__":
    try:
        test_basic_functionality()
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
