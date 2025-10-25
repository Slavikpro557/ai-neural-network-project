# ⏸️ Руководство по Pause/Resume обучения

## 🎯 Что добавлено:

### ✅ Новые возможности:
1. **Pause** - остановка обучения в любой момент
2. **Resume** - возобновление с того же места
3. **Изменение итераций** - при возобновлении можно указать новое количество
4. **Сохранение состояния** - optimizer и scheduler восстанавливаются
5. **Обработка Ctrl+C** - автосохранение при прерывании

---

## 📝 Базовое использование:

### 1. Начальное обучение:

```python
from azr_trainer_resume import AZRTrainer

trainer = AZRTrainer(model, tokenizer)

# Обучение на 1000 итераций
history = trainer.train_continuous(
    texts=texts,
    max_iterations=1000,
    checkpoint_dir='checkpoints'
)
```

**Результат:** Checkpoint сохранён как `model_iter_1000.pt`

---

### 2. Возобновление с ТЕМ ЖЕ количеством:

```python
# Создаём новую модель
model2 = CustomTransformerLM(...)
trainer2 = AZRTrainer(model2, tokenizer)

# Возобновляем
history = trainer2.train_continuous(
    texts=texts,
    max_iterations=1000,  # То же значение
    checkpoint_dir='checkpoints',
    resume_from='checkpoints/model_iter_1000.pt'  # Загружаем
)
```

**Результат:** Модель уже на 1000 итерациях, обучение завершится сразу

---

### 3. Возобновление с ДРУГИМ количеством:

```python
# Продолжаем до 5000 итераций
history = trainer2.train_continuous(
    texts=texts,
    max_iterations=5000,  # ИЗМЕНИЛИ!
    checkpoint_dir='checkpoints',
    resume_from='checkpoints/model_iter_1000.pt'
)
```

**Результат:** Обучение продолжится с 1000 до 5000 итераций (+4000)

---

## ⏸️ Ручная остановка:

### Способ 1: Кнопка Stop (для UI)

```python
# В отдельном потоке
def background_training():
    trainer.train_continuous(...)

training_thread = threading.Thread(target=background_training)
training_thread.start()

# Когда пользователь нажимает "Stop"
def on_stop_button():
    trainer.stop_training()  # Остановит после текущего батча
```

### Способ 2: Ctrl+C (в терминале)

```bash
python server.py
# Во время обучения: Ctrl+C
```

**Результат:** Автоматически сохранится checkpoint `model_interrupted_XXX.pt`

---

## 🔄 Полный workflow:

### Сценарий 1: "Передумал, хочу обучать дольше"

```python
# День 1: Обучил на 10K итераций
trainer.train_continuous(max_iterations=10000)
# → model_iter_10000.pt

# День 2: Решил продолжить до 50K
trainer2.train_continuous(
    max_iterations=50000,  # Новая цель
    resume_from='checkpoints/model_iter_10000.pt'
)
# → Обучение с 10K до 50K
```

### Сценарий 2: "Нужно освободить компьютер"

```python
# Запустил обучение
trainer.train_continuous(max_iterations=100000)

# Через час: нужно закрыть компьютер
# Нажимаете Ctrl+C
# → Сохранится model_interrupted_5432.pt

# На следующий день
trainer2.train_continuous(
    max_iterations=100000,
    resume_from='checkpoints/model_interrupted_5432.pt'
)
# → Продолжит с итерации 5432
```

### Сценарий 3: "Хочу увеличить количество итераций по ходу"

```python
# Запускаю на 1000
trainer.train_continuous(max_iterations=1000)

# Смотрю результаты - хорошо! Продолжу
trainer.train_continuous(
    max_iterations=10000,  # x10
    resume_from='checkpoints/model_iter_1000.pt'
)

# Ещё лучше! Продолжу ещё
trainer.train_continuous(
    max_iterations=100000,  # x10
    resume_from='checkpoints/model_iter_10000.pt'
)
```

---

## 📊 Что сохраняется в checkpoint:

```python
checkpoint = {
    'model_state_dict': ...,      # Веса модели
    'optimizer_state_dict': ...,  # Состояние optimizer (Adam momentum, etc.)
    'scheduler_state_dict': ...,  # Learning rate schedule
    'iteration': 5432,             # Текущая итерация
    'training_history': [...],     # История loss/reward
    'timestamp': '2025-10-20...'   # Когда сохранён
}
```

**Это означает:**
- ✅ Обучение продолжится точно с того же места
- ✅ Learning rate будет правильным
- ✅ Optimizer momentum сохранён
- ✅ История не теряется

---

## 🎨 Интеграция в веб-интерфейс:

### Изменения в server.py:

```python
# Глобальная переменная для trainer
active_trainer = None

@app.post("/train")
async def start_training(config: TrainingConfig):
    global active_trainer
    
    # ... создание модели ...
    
    active_trainer = AZRTrainer(model, tokenizer, status_callback=update_status)
    
    # Запуск в фоне
    thread = threading.Thread(target=lambda: active_trainer.train_continuous(
        texts=texts,
        max_iterations=config.max_iterations,
        checkpoint_dir=checkpoint_dir,
        resume_from=config.resume_from if hasattr(config, 'resume_from') else None
    ))
    thread.start()

@app.post("/stop_training")
async def stop_training():
    global active_trainer
    if active_trainer:
        active_trainer.stop_training()
        return {"status": "stopping"}
    return {"status": "not training"}

@app.post("/resume_training")
async def resume_training(config: ResumeConfig):
    # config содержит checkpoint_path и new_max_iterations
    # ... аналогично start_training, но с resume_from
```

### Изменения в UI:

```html
<!-- Кнопка Stop -->
<button onclick="stopTraining()">⏸️  Остановить</button>

<!-- Форма Resume -->
<select id="checkpoint_to_resume">
    <!-- Список checkpoints -->
</select>
<input type="number" id="new_max_iterations" placeholder="Новое количество итераций">
<button onclick="resumeTraining()">▶️  Возобновить</button>

<script>
async function stopTraining() {
    await fetch('/stop_training', {method: 'POST'});
    alert('Обучение останавливается...');
}

async function resumeTraining() {
    const config = {
        checkpoint_path: document.getElementById('checkpoint_to_resume').value,
        new_max_iterations: parseInt(document.getElementById('new_max_iterations').value)
    };
    await fetch('/resume_training', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(config)
    });
}
</script>
```

---

## 🧪 Тестирование:

### Запустите demo:

```bash
python example_pause_resume.py
```

**Что происходит:**
1. Обучение на 100 итераций → checkpoint
2. Возобновление до 250 итераций
3. Ручная остановка через 3 секунды
4. Возобновление с новой целью

---

## ⚠️ Важные замечания:

### 1. Checkpoint совместимость:
- ✅ Можно продолжить на другом компьютере
- ✅ Можно с CPU checkpoint продолжить на GPU
- ❌ Нельзя если изменилась архитектура модели

### 2. Изменение параметров:
- ✅ `max_iterations` - можно менять
- ✅ `batch_size` - можно менять
- ✅ `learning_rate` - можно менять (но лучше не надо)
- ❌ `d_model`, `num_layers` - нельзя (модель другая)

### 3. Optimizer state:
- Если загружаете checkpoint с `load_optimizer=True`, optimizer восстановится
- Если с `load_optimizer=False`, optimizer создастся заново (небольшой скачок в обучении)

### 4. Learning rate:
- При resume scheduler продолжит с того же места
- Если хотите изменить LR, создайте новый scheduler

---

## 🔍 Отладка:

### Проверить что в checkpoint:

```python
checkpoint = torch.load('model_iter_1000.pt')
print(f"Iteration: {checkpoint['iteration']}")
print(f"Has optimizer: {'optimizer_state_dict' in checkpoint}")
print(f"History length: {len(checkpoint['training_history'])}")
```

### Список всех checkpoints:

```python
from pathlib import Path

for cp in Path('checkpoints').glob('*.pt'):
    checkpoint = torch.load(cp)
    print(f"{cp.name}: iter {checkpoint['iteration']}")
```

---

## 📚 Примеры команд:

### Обучить с нуля:
```bash
python cli_demo.py  # Использует обычный trainer
```

### Обучить с resume:
```python
from azr_trainer_resume import AZRTrainer

trainer = AZRTrainer(model, tokenizer)
trainer.train_continuous(
    texts=texts,
    max_iterations=50000,
    resume_from='checkpoints/model_iter_10000.pt'
)
```

### Программная остановка:
```python
import threading

def train_bg():
    trainer.train_continuous(...)

thread = threading.Thread(target=train_bg)
thread.start()

# Через 10 секунд остановить
time.sleep(10)
trainer.stop_training()
thread.join()
```

---

## 🎯 Итого:

### Теперь вы можете:
1. ✅ Останавливать обучение в любой момент
2. ✅ Возобновлять с того же места
3. ✅ Изменять max_iterations при возобновлении
4. ✅ Обучать итеративно (100 → 1000 → 10000 → ...)
5. ✅ Не бояться Ctrl+C (автосохранение)

### Файлы:
- `azr_trainer_resume.py` - новая версия trainer
- `example_pause_resume.py` - демонстрация
- `PAUSE_RESUME_GUIDE.md` - это руководство

### Следующий шаг:
Интегрируйте в ваш веб-интерфейс кнопки Stop/Resume!

---

**Happy training! ⏸️▶️🚀**
