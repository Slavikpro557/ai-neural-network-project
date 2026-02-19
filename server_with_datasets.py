"""
Обновлённый сервер с поддержкой:
- Множественных датасетов
- Каталог датасетов + HuggingFace поиск
- Детальная аналитика обучения
- Сравнение итераций
- Генерация с метриками качества
- Рекомендации по конфигурации
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional
import torch
import torch.nn.functional as F
import json
import uvicorn
from datetime import datetime
import threading
import shutil

from model import CustomTransformerLM, count_parameters
from tokenizer import SimpleTokenizer
from azr_trainer_resume import AZRTrainer
from dataset_manager import DatasetManager
from dataset_catalog import DatasetCatalog
from reward_model import RewardComputer
from training_analytics import TrainingAnalytics

app = FastAPI(title="AZR Model Trainer v2")

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
BOOKS_DIR = BASE_DIR / "books"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
TEMPLATES_DIR = BASE_DIR / "templates"
REPORTS_DIR = BASE_DIR / "reports"

for dir_path in [MODELS_DIR, BOOKS_DIR, CHECKPOINTS_DIR, REPORTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Менеджер датасетов
dataset_manager = DatasetManager("datasets_db.json")

# Каталог датасетов
dataset_catalog = DatasetCatalog(BOOKS_DIR)

training_status = {
    "is_training": False,
    "current_iteration": 0,
    "max_iterations": 0,
    "current_loss": 0.0,
    "current_reward": 0.0,
    "model_name": None,
    "history": [],
    "perplexity": 0.0,
    "tokens_per_sec": 0.0,
    "eta_seconds": -1,
    "reward_components": {},
    "memory_mb": 0.0,
}

download_status = {
    "is_downloading": False,
    "dataset_id": None,
    "progress": 0,
    "message": ""
}

active_models = {}
active_trainer = None
active_analytics = None


# === Pydantic Models ===

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
    device: str = "auto"


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


class GenerateAtCheckpointConfig(BaseModel):
    model_name: str
    checkpoint_iteration: int
    prompt: str
    max_length: int = 100
    temperature: float = 0.8


class CompareConfig(BaseModel):
    model_name: str
    prompt: str
    iterations: List[int]
    max_length: int = 100


# === UI ===

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_file = TEMPLATES_DIR / "index_complete.html"
    if not html_file.exists():
        html_file = TEMPLATES_DIR / "index.html"
    if html_file.exists():
        return html_file.read_text(encoding='utf-8')
    return "<h1>AZR Model Trainer v2</h1><p>Template not found. Run build_complete_interface.py</p>"


# === Model CRUD ===

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


@app.get("/models")
async def list_models():
    models = []
    for model_dir in MODELS_DIR.iterdir():
        if model_dir.is_dir():
            config_file = model_dir / "config.json"
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)

                model_info = dataset_manager.get_model_info(model_dir.name)

                model_size = 0
                trained_model = model_dir / "model_trained.pt"
                base_model = model_dir / "model.pt"

                if trained_model.exists():
                    model_size = trained_model.stat().st_size
                elif base_model.exists():
                    model_size = base_model.stat().st_size

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


@app.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Удалить модель и все её файлы"""
    model_dir = MODELS_DIR / model_name
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    try:
        shutil.rmtree(model_dir)
        # Удаляем чекпоинты если есть
        checkpoint_dir = CHECKPOINTS_DIR / model_name
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        return {"status": "deleted", "model_name": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Dataset Upload & Management ===

@app.post("/upload_book")
async def upload_book(file: UploadFile = File(...)):
    try:
        file_path = BOOKS_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Определяем формат
        metadata = None
        converted_name = file.filename
        try:
            fmt_info = dataset_catalog.detect_format(file_path)
            metadata = fmt_info

            # Конвертируем если не txt
            if fmt_info["format"] != "txt":
                converted = dataset_catalog.convert_to_txt(file_path, fmt_info["format"])
                if converted != file_path:
                    file_path = converted
                    converted_name = converted.name
                    metadata["converted"] = True
        except Exception as conv_err:
            metadata = metadata or {}
            metadata["conversion_error"] = str(conv_err)

        dataset_manager.register_dataset(converted_name, str(file_path), metadata=metadata)

        return {
            "status": "success",
            "filename": converted_name,
            "original_filename": file.filename,
            "path": str(file_path),
            "size": file_path.stat().st_size,
            "format": metadata.get("format", "txt") if metadata else "txt"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/books")
async def list_books():
    datasets = dataset_manager.get_available_datasets()
    return {"books": datasets}


@app.post("/attach_dataset")
async def attach_dataset(config: AttachDatasetConfig):
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
    try:
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

        available = dataset_manager.get_available_datasets(model_name)

        return {
            "attached": attached_info,
            "available": available
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Dataset Catalog ===

@app.get("/dataset_catalog")
async def get_dataset_catalog(category: str = None, language: str = None):
    return {"datasets": dataset_catalog.get_catalog(category, language)}


@app.get("/dataset_catalog/categories")
async def get_catalog_categories():
    return {"categories": dataset_catalog.get_categories()}


@app.post("/dataset_catalog/download/{dataset_id}")
async def download_catalog_dataset(dataset_id: str):
    global download_status

    if download_status["is_downloading"]:
        raise HTTPException(status_code=400, detail="Another download in progress")

    info = dataset_catalog.get_dataset_info(dataset_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

    def do_download():
        global download_status
        download_status["is_downloading"] = True
        download_status["dataset_id"] = dataset_id

        def progress_cb(status):
            download_status["progress"] = status.get("progress", 0)
            download_status["message"] = status.get("message", "")

        try:
            file_path = dataset_catalog.download_dataset(dataset_id, progress_callback=progress_cb)
            if file_path:
                # Регистрируем в менеджере датасетов
                dataset_manager.register_dataset(
                    file_path.name, str(file_path),
                    metadata={
                        "source": "catalog",
                        "catalog_id": dataset_id,
                        "description": info.get("description", ""),
                        "format": "txt"
                    }
                )
                download_status["message"] = f"Downloaded: {file_path.name}"
        except Exception as e:
            download_status["message"] = f"Error: {e}"
            download_status["progress"] = -1
        finally:
            download_status["is_downloading"] = False

    thread = threading.Thread(target=do_download)
    thread.start()

    return {"status": "downloading", "dataset_id": dataset_id, "name": info["name"]}


@app.get("/dataset_catalog/download_status")
async def get_download_status():
    return download_status


@app.get("/dataset_catalog/preview/{dataset_id}")
async def preview_catalog_dataset(dataset_id: str, lines: int = 20):
    return dataset_catalog.preview_dataset(dataset_id, lines)


@app.get("/dataset_catalog/search_hf")
async def search_huggingface(query: str, limit: int = 10):
    results = dataset_catalog.search_huggingface(query, limit)
    return {"results": results}


@app.post("/dataset_catalog/add_custom")
async def add_custom_dataset(name: str = Form(...), url: str = Form(...), language: str = Form("auto")):
    """Добавить свой датасет по URL"""
    try:
        custom_entry = dataset_catalog.add_custom_url(name, url, language)
        return {"status": "added", "dataset": custom_entry}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Training ===

def train_model_background(config: TrainingConfig):
    global training_status, active_trainer, active_analytics

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

        # Проверяем нужно ли переобучить токенизатор
        current_datasets = set(dataset_manager.get_attached_datasets(config.model_name))
        tokenizer_datasets = set(tokenizer.get_trained_datasets())
        need_retrain = False
        preserve_mode = False

        is_pretrained = (len(tokenizer.token_to_id) > 1000 and len(tokenizer_datasets) == 0)

        if is_pretrained:
            print(f"Pretrained model detected ({len(tokenizer.token_to_id)} tokens)")
            need_retrain = False
        elif len(tokenizer.token_to_id) <= len(tokenizer.special_tokens):
            need_retrain = True
            preserve_mode = False
            print("Tokenizer is empty, will train...")
        elif current_datasets != tokenizer_datasets:
            need_retrain = True
            preserve_mode = True
            print("Datasets changed, updating tokenizer...")
        else:
            print(f"Tokenizer already trained on current datasets")

        if need_retrain:
            print(f"Training tokenizer on {len(texts)} text chunks...")
            batch_size_tok = 100
            all_texts_combined = []
            for i in range(0, len(texts), batch_size_tok):
                batch = texts[i:i + batch_size_tok]
                all_texts_combined.append(" ".join(batch))

            tokenizer.train(all_texts_combined, preserve_existing=preserve_mode)
            print(f"Tokenizer trained! Vocabulary size: {len(tokenizer.token_to_id)} tokens")
            tokenizer.save(model_dir / "tokenizer.pkl", trained_on_datasets=list(current_datasets))

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

        # Создаём RewardComputer с референсными текстами
        reward_computer = RewardComputer(tokenizer, reference_texts=texts[:100])

        # Создаём аналитику
        reports_dir = REPORTS_DIR / config.model_name
        analytics = TrainingAnalytics(reports_dir)
        active_analytics = analytics

        # Создаём trainer
        active_trainer = AZRTrainer(
            model, tokenizer,
            device=device,
            status_callback=update_status,
            reward_computer=reward_computer,
            analytics=analytics
        )

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


# === Training Analytics ===

@app.get("/training_analytics")
async def get_training_analytics():
    if active_analytics:
        return active_analytics.get_summary()
    return {"error": "No active analytics session"}


@app.get("/training_analytics/reports")
async def get_analytics_reports():
    if active_analytics:
        return {"reports": active_analytics.get_all_reports()}
    return {"reports": []}


@app.get("/training_analytics/benchmarks")
async def get_benchmark_results():
    if active_analytics:
        return {"benchmarks": active_analytics.get_benchmark_history()}
    return {"benchmarks": []}


@app.get("/training_analytics/compare/{iter_a}/{iter_b}")
async def compare_iterations(iter_a: int, iter_b: int):
    if active_analytics:
        comparison = active_analytics.compare_iterations(iter_a, iter_b)
        if comparison:
            return comparison
        return {"error": f"Iterations {iter_a} or {iter_b} not found in reports"}
    return {"error": "No active analytics session"}


# === Checkpoints & Comparison ===

@app.get("/checkpoints/{model_name}")
async def list_checkpoints(model_name: str):
    checkpoint_dir = CHECKPOINTS_DIR / model_name
    if not checkpoint_dir.exists():
        return {"checkpoints": []}

    checkpoints = []
    for f in sorted(checkpoint_dir.glob("model_iter_*.pt")):
        try:
            # Извлекаем номер итерации из имени файла
            iter_num = int(f.stem.split("_")[-1])
            checkpoints.append({
                "iteration": iter_num,
                "filename": f.name,
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                "timestamp": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
            })
        except (ValueError, IndexError):
            continue

    return {"checkpoints": sorted(checkpoints, key=lambda x: x["iteration"])}


@app.post("/generate_at_checkpoint")
async def generate_at_checkpoint(config: GenerateAtCheckpointConfig):
    try:
        model_dir = MODELS_DIR / config.model_name
        checkpoint_dir = CHECKPOINTS_DIR / config.model_name

        with open(model_dir / "config.json") as f:
            model_config = json.load(f)

        # Ищем чекпоинт
        checkpoint_file = checkpoint_dir / f"model_iter_{config.checkpoint_iteration}.pt"
        if not checkpoint_file.exists():
            raise HTTPException(status_code=404, detail=f"Checkpoint at iteration {config.checkpoint_iteration} not found")

        model = CustomTransformerLM(
            vocab_size=model_config["vocab_size"],
            d_model=model_config["d_model"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            d_ff=model_config["d_ff"],
            max_seq_len=model_config["max_seq_len"]
        )

        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        tokenizer = SimpleTokenizer.load(model_dir / "tokenizer.pkl")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()

        tokens = tokenizer.encode(config.prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            generated = model.generate(idx, max_new_tokens=config.max_length,
                                      temperature=config.temperature, top_k=40)

        gen_tokens = generated[0].cpu().tolist()
        gen_text = tokenizer.decode(gen_tokens)

        # Оценка качества
        reward_computer = RewardComputer(tokenizer)
        quality = reward_computer.compute_reward(gen_text)

        return {
            "status": "success",
            "iteration": config.checkpoint_iteration,
            "prompt": config.prompt,
            "generated_text": gen_text,
            "quality_metrics": quality
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare_generations")
async def compare_generations(config: CompareConfig):
    try:
        model_dir = MODELS_DIR / config.model_name
        checkpoint_dir = CHECKPOINTS_DIR / config.model_name

        with open(model_dir / "config.json") as f:
            model_config = json.load(f)

        tokenizer = SimpleTokenizer.load(model_dir / "tokenizer.pkl")
        reward_computer = RewardComputer(tokenizer)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        results = []

        for iteration in config.iterations[:5]:  # Макс 5
            checkpoint_file = checkpoint_dir / f"model_iter_{iteration}.pt"
            if not checkpoint_file.exists():
                results.append({
                    "iteration": iteration,
                    "error": "Checkpoint not found"
                })
                continue

            model = CustomTransformerLM(
                vocab_size=model_config["vocab_size"],
                d_model=model_config["d_model"],
                num_layers=model_config["num_layers"],
                num_heads=model_config["num_heads"],
                d_ff=model_config["d_ff"],
                max_seq_len=model_config["max_seq_len"]
            )

            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model = model.to(device)
            model.eval()

            tokens = tokenizer.encode(config.prompt)
            idx = torch.tensor([tokens], dtype=torch.long, device=device)

            with torch.no_grad():
                generated = model.generate(idx, max_new_tokens=config.max_length,
                                          temperature=0.8, top_k=40)

            gen_tokens = generated[0].cpu().tolist()
            gen_text = tokenizer.decode(gen_tokens)
            quality = reward_computer.compute_reward(gen_text)

            results.append({
                "iteration": iteration,
                "text": gen_text,
                "quality": quality
            })

            del model

        return {"prompt": config.prompt, "comparisons": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Generation ===

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
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(torch.load(model_dir / "model.pt", map_location='cpu'))

            tokenizer = SimpleTokenizer.load(model_dir / "tokenizer.pkl")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()

        input_tokens = tokenizer.encode(config.prompt)
        idx = torch.tensor([input_tokens], dtype=torch.long, device=device)

        # Генерация с отслеживанием confidence per token
        token_details = []
        current_idx = idx.clone()

        with torch.no_grad():
            for _ in range(config.max_length):
                idx_cond = current_idx if current_idx.size(1) <= model.max_seq_len else current_idx[:, -model.max_seq_len:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / config.temperature

                if config.top_k:
                    v, _ = torch.topk(logits, min(config.top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                chosen_prob = probs[0, idx_next[0, 0]].item()

                # Топ-3 альтернативы
                top3_probs, top3_ids = torch.topk(probs[0], min(3, probs.size(-1)))
                alternatives = []
                for p, tid in zip(top3_probs.tolist(), top3_ids.tolist()):
                    token_str = tokenizer.id_to_token.get(tid, f"<{tid}>") if hasattr(tokenizer, 'id_to_token') else str(tid)
                    alternatives.append({"token": token_str, "prob": round(p, 4)})

                token_id = idx_next[0, 0].item()
                token_str = tokenizer.id_to_token.get(token_id, f"<{token_id}>") if hasattr(tokenizer, 'id_to_token') else str(token_id)

                token_details.append({
                    "token": token_str,
                    "confidence": round(chosen_prob, 4),
                    "top_alternatives": alternatives
                })

                current_idx = torch.cat((current_idx, idx_next), dim=1)

        generated_tokens = current_idx[0].cpu().tolist()
        generated_text = tokenizer.decode(generated_tokens)

        # Метрики качества
        reward_computer = RewardComputer(tokenizer)
        quality = reward_computer.compute_reward(generated_text)

        return {
            "status": "success",
            "prompt": config.prompt,
            "generated_text": generated_text,
            "token_details": token_details,
            "quality_metrics": quality,
            "tokens_generated": len(token_details),
        }
    except Exception as e:
        print(f"Generation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_before_after")
async def generate_before_after(config: GenerateConfig):
    """Сравнение генерации до и после обучения"""
    try:
        model_dir = MODELS_DIR / config.model_name

        with open(model_dir / "config.json") as f:
            model_config = json.load(f)

        tokenizer = SimpleTokenizer.load(model_dir / "tokenizer.pkl")
        reward_computer = RewardComputer(tokenizer)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        def generate_from_weights(weights_path):
            m = CustomTransformerLM(
                vocab_size=model_config["vocab_size"],
                d_model=model_config["d_model"],
                num_layers=model_config["num_layers"],
                num_heads=model_config["num_heads"],
                d_ff=model_config["d_ff"],
                max_seq_len=model_config["max_seq_len"]
            )
            ckpt = torch.load(weights_path, map_location='cpu')
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                m.load_state_dict(ckpt['model_state_dict'])
            else:
                m.load_state_dict(ckpt)
            m = m.to(device)
            m.eval()

            tokens = tokenizer.encode(config.prompt)
            idx = torch.tensor([tokens], dtype=torch.long, device=device)
            with torch.no_grad():
                gen = m.generate(idx, max_new_tokens=config.max_length,
                                temperature=config.temperature, top_k=config.top_k)
            text = tokenizer.decode(gen[0].cpu().tolist())
            quality = reward_computer.compute_reward(text)
            del m
            return text, quality

        # До обучения
        before_text, before_quality = generate_from_weights(model_dir / "model.pt")

        # После обучения
        trained_path = model_dir / "model_trained.pt"
        if trained_path.exists():
            after_text, after_quality = generate_from_weights(trained_path)
        else:
            after_text = before_text
            after_quality = before_quality

        # Считаем улучшение
        improvement = {
            "reward_delta": round(after_quality["total"] - before_quality["total"], 4),
            "components_delta": {}
        }
        for key in after_quality.get("components", {}):
            before_val = before_quality.get("components", {}).get(key, 0)
            after_val = after_quality["components"][key]
            improvement["components_delta"][key] = round(after_val - before_val, 4)

        return {
            "prompt": config.prompt,
            "before": {"text": before_text, "quality": before_quality},
            "after": {"text": after_text, "quality": after_quality},
            "improvement": improvement
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Download / Upload Model ===

@app.get("/download_model/{model_name}")
async def download_model(model_name: str):
    model_dir = MODELS_DIR / model_name
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    trained_model = model_dir / "model_trained.pt"
    if trained_model.exists():
        return FileResponse(trained_model, filename=f"{model_name}_trained.pt",
                          media_type="application/octet-stream")
    else:
        return FileResponse(model_dir / "model.pt", filename=f"{model_name}.pt",
                          media_type="application/octet-stream")


@app.post("/upload_model/{model_name}")
async def upload_model(model_name: str, file: UploadFile = File(...)):
    try:
        model_dir = MODELS_DIR / model_name
        if not model_dir.exists():
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

        model_path = model_dir / "model_trained.pt"
        with open(model_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

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

            if model_name in active_models:
                active_models[model_name]["model"] = model

            return {
                "status": "success",
                "message": f"Model '{model_name}' uploaded successfully",
                "size": model_path.stat().st_size
            }
        except Exception as e:
            model_path.unlink()
            raise HTTPException(status_code=400, detail=f"Invalid model file: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === System Info ===

@app.get("/dataset_stats")
async def get_dataset_stats():
    return dataset_manager.get_stats()


@app.get("/device_info")
async def get_device_info():
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
        "recommendation": "GPU (CUDA)" if cuda_available else "CPU (slow, GPU recommended)"
    }

    if cuda_available:
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            info["cuda"]["memory"] = f"{total_memory:.1f} GB"
        except Exception:
            pass

    return info


@app.get("/hardware_recommendation")
async def get_hardware_recommendation():
    """Рекомендация конфигурации на основе оборудования"""
    has_gpu = torch.cuda.is_available()
    gpu_mem_gb = 0

    if has_gpu:
        try:
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            pass

    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        ram_gb = 8  # Default assumption

    if has_gpu and gpu_mem_gb >= 8:
        preset = "quality"
        reason = f"GPU {torch.cuda.get_device_name(0)} ({gpu_mem_gb:.0f} GB) — можно обучать большие модели"
    elif has_gpu and gpu_mem_gb >= 4:
        preset = "standard"
        reason = f"GPU {torch.cuda.get_device_name(0)} ({gpu_mem_gb:.0f} GB) — стандартные модели"
    elif has_gpu:
        preset = "quick"
        reason = f"GPU с {gpu_mem_gb:.0f} GB — только маленькие модели"
    elif ram_gb >= 16:
        preset = "standard"
        reason = f"{ram_gb:.0f} GB RAM, CPU — стандартные модели (медленно)"
    else:
        preset = "quick"
        reason = f"{ram_gb:.0f} GB RAM, CPU — только быстрые тесты"

    presets = {
        "quick": {
            "d_model": 128, "num_layers": 4, "num_heads": 4, "d_ff": 512,
            "max_seq_len": 128, "vocab_size": 10000,
            "max_iterations": 1000, "batch_size": 8, "learning_rate": 0.001
        },
        "standard": {
            "d_model": 256, "num_layers": 6, "num_heads": 8, "d_ff": 1024,
            "max_seq_len": 256, "vocab_size": 15000,
            "max_iterations": 10000, "batch_size": 16, "learning_rate": 0.0003
        },
        "quality": {
            "d_model": 512, "num_layers": 12, "num_heads": 16, "d_ff": 2048,
            "max_seq_len": 512, "vocab_size": 25000,
            "max_iterations": 100000, "batch_size": 32, "learning_rate": 0.0001
        }
    }

    return {
        "preset": preset,
        "reason": reason,
        "config": presets[preset],
        "has_gpu": has_gpu,
        "gpu_memory_gb": round(gpu_mem_gb, 1),
        "ram_gb": round(ram_gb, 1)
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  AZR Model Trainer v2")
    print("  Features: Catalog, Analytics, REINFORCE, Comparison")
    print("=" * 60)
    print(f"Models: {MODELS_DIR}")
    print(f"Books: {BOOKS_DIR}")
    print(f"Checkpoints: {CHECKPOINTS_DIR}")
    print(f"Reports: {REPORTS_DIR}")
    print(f"Catalog: {len(dataset_catalog.catalog)} datasets")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)
    print()
    print("  >>> Открой в браузере: http://localhost:8000 <<<")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000)
