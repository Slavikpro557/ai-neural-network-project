"""
Обновлённый сервер с поддержкой множественных датасетов
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from pathlib import Path
from typing import List
import torch
import json
import uvicorn
from datetime import datetime
import threading
import shutil

from model import CustomTransformerLM, count_parameters
from tokenizer import SimpleTokenizer
from azr_trainer_resume import AZRTrainer
from dataset_manager import DatasetManager

app = FastAPI(title="AZR Model Trainer with Datasets")

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
BOOKS_DIR = BASE_DIR / "books"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
TEMPLATES_DIR = BASE_DIR / "templates"

for dir_path in [MODELS_DIR, BOOKS_DIR, CHECKPOINTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Менеджер датасетов
dataset_manager = DatasetManager("datasets_db.json")

training_status = {
    "is_training": False,
    "current_iteration": 0,
    "max_iterations": 0,
    "current_loss": 0.0,
    "current_reward": 0.0,
    "model_name": None,
    "history": []
}

active_models = {}
active_trainer = None


class ModelConfig(BaseModel):
    name: str
    vocab_size: int = 8000
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 1024
    max_seq_len: int = 256


class TrainingConfig(BaseModel):
    model_name: str
    max_iterations: int = 1000000
    batch_size: int = 16
    learning_rate: float = 3e-4
    save_every: int = 1000
    resume_from: str = None
    device: str = "auto"  # "auto", "cpu", "cuda"


class AttachDatasetConfig(BaseModel):
    model_name: str
    dataset_name: str


class DetachDatasetConfig(BaseModel):
    model_name: str
    dataset_name: str


class GenerateConfig(BaseModel):
    model_name: str
    prompt: str
    max_length: int = 100
    temperature: float = 0.8
    top_k: int = 40


@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_file = TEMPLATES_DIR / "index_complete.html"
    if not html_file.exists():
        html_file = TEMPLATES_DIR / "index.html"
    if html_file.exists():
        return html_file.read_text(encoding='utf-8')
    return "<h1>AZR Model Trainer</h1><p>Template not found</p>"


@app.post("/create_model")
async def create_model(config: ModelConfig):
    try:
        tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
        
        model = CustomTransformerLM(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len
        )
        
        params = count_parameters(model)
        
        model_dir = MODELS_DIR / config.name
        model_dir.mkdir(exist_ok=True)
        
        torch.save(model.state_dict(), model_dir / "model.pt")
        tokenizer.save(model_dir / "tokenizer.pkl")
        
        with open(model_dir / "config.json", "w") as f:
            json.dump(config.dict(), f, indent=2)
        
        active_models[config.name] = {
            "model": model,
            "tokenizer": tokenizer,
            "config": config.dict()
        }
        
        return {
            "status": "success",
            "message": f"Model '{config.name}' created successfully",
            "parameters": params,
            "config": config.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_book")
async def upload_book(file: UploadFile = File(...)):
    try:
        file_path = BOOKS_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Регистрируем в менеджере датасетов
        dataset_manager.register_dataset(file.filename, str(file_path))
        
        return {
            "status": "success",
            "filename": file.filename,
            "path": str(file_path),
            "size": file_path.stat().st_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/books")
async def list_books():
    """Список всех датасетов"""
    datasets = dataset_manager.get_available_datasets()
    return {"books": datasets}


@app.get("/models")
async def list_models():
    models = []
    for model_dir in MODELS_DIR.iterdir():
        if model_dir.is_dir():
            config_file = model_dir / "config.json"
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
                
                # Получаем информацию о датасетах
                model_info = dataset_manager.get_model_info(model_dir.name)
                
                # Считаем размер модели
                model_size = 0
                trained_model = model_dir / "model_trained.pt"
                base_model = model_dir / "model.pt"
                
                if trained_model.exists():
                    model_size = trained_model.stat().st_size
                elif base_model.exists():
                    model_size = base_model.stat().st_size
                
                # Размер в MB
                model_size_mb = round(model_size / (1024 * 1024), 2)
                
                models.append({
                    "name": model_dir.name,
                    "config": config,
                    "datasets": model_info["datasets"] if model_info else [],
                    "total_datasets": model_info["total_datasets"] if model_info else 0,
                    "size_mb": model_size_mb,
                    "trained": trained_model.exists()
                })
    return {"models": models}


@app.post("/attach_dataset")
async def attach_dataset(config: AttachDatasetConfig):
    """Прикрепить датасет к модели"""
    try:
        success = dataset_manager.attach_dataset(config.model_name, config.dataset_name)
        if success:
            return {
                "status": "success",
                "message": f"Dataset '{config.dataset_name}' attached to '{config.model_name}'"
            }
        else:
            raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detach_dataset")
async def detach_dataset(config: DetachDatasetConfig):
    """Открепить датасет от модели"""
    try:
        success = dataset_manager.detach_dataset(config.model_name, config.dataset_name)
        if success:
            return {
                "status": "success",
                "message": f"Dataset '{config.dataset_name}' detached from '{config.model_name}'"
            }
        else:
            raise HTTPException(status_code=404, detail="Not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_datasets/{model_name}")
async def get_model_datasets(model_name: str):
    """Получить датасеты модели"""
    try:
        # Прикреплённые
        attached = dataset_manager.get_attached_datasets(model_name)
        attached_info = []
        for ds_name in attached:
            if ds_name in dataset_manager.db["datasets"]:
                info = dataset_manager.db["datasets"][ds_name]
                attached_info.append({
                    "name": ds_name,
                    "size": info["size"],
                    "path": info["path"]
                })
        
        # Доступные (не прикреплённые)
        available = dataset_manager.get_available_datasets(model_name)
        
        return {
            "attached": attached_info,
            "available": available
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def train_model_background(config: TrainingConfig):
    global training_status, active_trainer
    
    try:
        training_status["is_training"] = True
        training_status["model_name"] = config.model_name
        training_status["max_iterations"] = config.max_iterations
        
        model_dir = MODELS_DIR / config.model_name
        
        # Загружаем модель
        if config.model_name in active_models:
            model = active_models[config.model_name]["model"]
            tokenizer = active_models[config.model_name]["tokenizer"]
        else:
            with open(model_dir / "config.json") as f:
                model_config = json.load(f)
            
            model = CustomTransformerLM(
                vocab_size=model_config["vocab_size"],
                d_model=model_config["d_model"],
                num_layers=model_config["num_layers"],
                num_heads=model_config["num_heads"],
                d_ff=model_config["d_ff"],
                max_seq_len=model_config["max_seq_len"]
            )
            model.load_state_dict(torch.load(model_dir / "model.pt"))
            tokenizer = SimpleTokenizer.load(model_dir / "tokenizer.pkl")
        
        # Загружаем ВСЕ прикреплённые датасеты
        print(f"\nLoading attached datasets for model '{config.model_name}'...")
        texts = dataset_manager.load_attached_texts(config.model_name)
        
        if len(texts) == 0:
            print("ERROR: No attached datasets! Attach datasets first.")
            training_status["is_training"] = False
            return
        
        print(f"Loaded {len(texts)} text chunks from attached datasets")
        
        # Получаем список текущих прикреплённых датасетов
        current_datasets = set(dataset_manager.get_attached_datasets(config.model_name))
        
        # Проверяем нужно ли переобучить токенизатор
        tokenizer_datasets = set(tokenizer.get_trained_datasets())
        need_retrain = False
        preserve_mode = False  # По умолчанию
        
        # Проверка: если токенизатор уже большой (готовая модель типа Qwen)
        # и нет прикреплённых датасетов в истории - НЕ ТРОГАЕМ токенизатор!
        is_pretrained = (len(tokenizer.token_to_id) > 1000 and len(tokenizer_datasets) == 0)
        
        if is_pretrained:
            print(f"✅ Pretrained model detected ({len(tokenizer.token_to_id)} tokens)")
            print(f"   Tokenizer will NOT be retrained to preserve model knowledge")
            print(f"   Training will use existing vocabulary")
            need_retrain = False
        elif len(tokenizer.token_to_id) <= len(tokenizer.special_tokens):
            # Токенизатор пустой - нужно обучить с нуля
            need_retrain = True
            preserve_mode = False
            print("🔄 Tokenizer is empty, will train...")
        elif current_datasets != tokenizer_datasets:
            # Датасеты изменились - добавляем только новые слова
            need_retrain = True
            preserve_mode = True  # ВСЕГДА сохраняем старые ID!
            added = current_datasets - tokenizer_datasets
            removed = tokenizer_datasets - current_datasets
            
            # Инкрементальное обучение
            # Даже если датасет убрали - слова остаются в словаре
            print("🔄 Datasets changed, updating tokenizer...")
            if added:
                print(f"   📁 Added: {', '.join(added)}")
            if removed:
                print(f"   ℹ️  Removed: {', '.join(removed)} (but words remain in vocabulary)")
        else:
            print(f"✅ Tokenizer already trained on current datasets ({len(current_datasets)} datasets)")
            print(f"   Vocabulary size: {len(tokenizer.token_to_id)} tokens")
        
        # Обучаем токенизатор если нужно
        if need_retrain:
            print(f"📚 Training tokenizer on {len(texts)} text chunks...")
            
            # Используем ВСЕ тексты для обучения токенизатора
            # Объединяем в большие куски чтобы не было проблем с памятью
            batch_size = 100
            all_texts_combined = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                all_texts_combined.append(" ".join(batch))
            
            print(f"   Processing {len(all_texts_combined)} batches...")
            
            # Используем правильный режим обучения
            tokenizer.train(all_texts_combined, preserve_existing=preserve_mode)
            
            print(f"✅ Tokenizer trained! Vocabulary size: {len(tokenizer.token_to_id)} tokens")
            
            # Сохраняем с информацией о датасетах
            tokenizer.save(model_dir / "tokenizer.pkl", trained_on_datasets=list(current_datasets))
            print(f"💾 Tokenizer saved (trained on: {', '.join(current_datasets)})")
        
        # Определяем устройство
        if config.device == "auto":
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif config.device == "cuda":
            if not torch.cuda.is_available():
                print("WARNING: CUDA requested but not available, using CPU")
                device = 'cpu'
            else:
                device = 'cuda'
        else:
            device = 'cpu'
        
        print(f"Using device: {device}")
        model = model.to(device)
        training_status["device"] = device
        
        # Callback для обновления статуса
        def update_status(status_dict):
            training_status.update(status_dict)
        
        # Создаём trainer
        active_trainer = AZRTrainer(model, tokenizer, status_callback=update_status)
        
        # Обучение
        history = active_trainer.train_continuous(
            texts=texts,
            max_iterations=config.max_iterations,
            batch_size=config.batch_size,
            lr=config.learning_rate,
            save_every=config.save_every,
            checkpoint_dir=CHECKPOINTS_DIR / config.model_name,
            resume_from=config.resume_from
        )
        
        # Сохраняем финальную модель
        torch.save(model.state_dict(), model_dir / "model_trained.pt")
        
        training_status["is_training"] = False
        training_status["history"] = history
        
    except Exception as e:
        training_status["is_training"] = False
        training_status["error"] = str(e)
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()


@app.post("/train")
async def start_training(config: TrainingConfig):
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    model_dir = MODELS_DIR / config.model_name
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model '{config.model_name}' not found")
    
    # Проверяем что есть прикреплённые датасеты
    attached = dataset_manager.get_attached_datasets(config.model_name)
    if len(attached) == 0:
        raise HTTPException(status_code=400, detail="No datasets attached to model. Attach datasets first!")
    
    thread = threading.Thread(target=train_model_background, args=(config,))
    thread.start()
    
    return {
        "status": "success",
        "message": f"Training started with {len(attached)} datasets",
        "datasets": attached,
        "config": config.dict()
    }


@app.post("/stop_training")
async def stop_training():
    global active_trainer
    if active_trainer:
        active_trainer.stop_training()
        return {"status": "stopping", "message": "Training will pause after current batch"}
    return {"status": "not training"}


@app.get("/training_status")
async def get_training_status():
    return training_status


@app.post("/generate")
async def generate_text(config: GenerateConfig):
    try:
        model_dir = MODELS_DIR / config.model_name
        
        if config.model_name in active_models:
            model = active_models[config.model_name]["model"]
            tokenizer = active_models[config.model_name]["tokenizer"]
        else:
            with open(model_dir / "config.json") as f:
                model_config = json.load(f)
            
            model = CustomTransformerLM(
                vocab_size=model_config["vocab_size"],
                d_model=model_config["d_model"],
                num_layers=model_config["num_layers"],
                num_heads=model_config["num_heads"],
                d_ff=model_config["d_ff"],
                max_seq_len=model_config["max_seq_len"]
            )
            
            trained_model = model_dir / "model_trained.pt"
            if trained_model.exists():
                checkpoint = torch.load(trained_model, map_location='cpu')
                # Проверяем формат: чекпоинт или просто state_dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # Формат чекпоинта (из AZRTrainer)
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # Простой state_dict
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(torch.load(model_dir / "model.pt", map_location='cpu'))
            
            tokenizer = SimpleTokenizer.load(model_dir / "tokenizer.pkl")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        
        tokens = tokenizer.encode(config.prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=device)
        
        with torch.no_grad():
            generated = model.generate(
                idx, 
                max_new_tokens=config.max_length,
                temperature=config.temperature,
                top_k=config.top_k
            )
        
        generated_tokens = generated[0].cpu().tolist()
        generated_text = tokenizer.decode(generated_tokens)
        
        return {
            "status": "success",
            "prompt": config.prompt,
            "generated_text": generated_text
        }
    except Exception as e:
        print(f"Generation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download_model/{model_name}")
async def download_model(model_name: str):
    model_dir = MODELS_DIR / model_name
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    trained_model = model_dir / "model_trained.pt"
    if trained_model.exists():
        return FileResponse(
            trained_model,
            filename=f"{model_name}_trained.pt",
            media_type="application/octet-stream"
        )
    else:
        return FileResponse(
            model_dir / "model.pt",
            filename=f"{model_name}.pt",
            media_type="application/octet-stream"
        )


@app.post("/upload_model/{model_name}")
async def upload_model(model_name: str, file: UploadFile = File(...)):
    """Загрузить обученную модель обратно на сервер"""
    try:
        model_dir = MODELS_DIR / model_name
        if not model_dir.exists():
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        # Сохраняем файл модели
        model_path = model_dir / "model_trained.pt"
        with open(model_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Загружаем модель для проверки
        try:
            with open(model_dir / "config.json") as f:
                model_config = json.load(f)
            
            model = CustomTransformerLM(
                vocab_size=model_config["vocab_size"],
                d_model=model_config["d_model"],
                num_layers=model_config["num_layers"],
                num_heads=model_config["num_heads"],
                d_ff=model_config["d_ff"],
                max_seq_len=model_config["max_seq_len"]
            )
            model.load_state_dict(torch.load(model_path))
            
            # Обновляем кэш активных моделей
            if model_name in active_models:
                active_models[model_name]["model"] = model
            
            return {
                "status": "success",
                "message": f"Model '{model_name}' uploaded and loaded successfully",
                "path": str(model_path),
                "size": model_path.stat().st_size
            }
        except Exception as e:
            # Если модель не загрузилась, удаляем файл
            model_path.unlink()
            raise HTTPException(status_code=400, detail=f"Invalid model file: {str(e)}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dataset_stats")
async def get_dataset_stats():
    """Статистика системы датасетов"""
    return dataset_manager.get_stats()


@app.get("/device_info")
async def get_device_info():
    """Информация о доступных устройствах"""
    cuda_available = torch.cuda.is_available()
    
    info = {
        "cpu": {
            "available": True,
            "name": "CPU",
            "cores": torch.get_num_threads()
        },
        "cuda": {
            "available": cuda_available,
            "name": torch.cuda.get_device_name(0) if cuda_available else None,
            "count": torch.cuda.device_count() if cuda_available else 0,
            "memory": None
        },
        "current_device": "cuda" if cuda_available else "cpu",
        "recommendation": "GPU (CUDA)" if cuda_available else "CPU (медленно, рекомендуется GPU)"
    }
    
    if cuda_available:
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info["cuda"]["memory"] = f"{total_memory:.1f} GB"
        except:
            pass
    
    return info


if __name__ == "__main__":
    print("Starting AZR Model Trainer with Dataset Management...")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Books directory: {BOOKS_DIR}")
    print(f"Checkpoints directory: {CHECKPOINTS_DIR}")
    print(f"Dataset DB: datasets_db.json")
    uvicorn.run(app, host="0.0.0.0", port=8000)
