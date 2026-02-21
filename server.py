"""
AZR Neural Network Trainer — единый сервер
Объединяет всё: датасеты, каталог, аналитику, REINFORCE, сравнение, GPU-активацию из UI
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
import subprocess
import sys

from model import CustomTransformerLM, count_parameters
from tokenizer import SimpleTokenizer
from azr_trainer_resume import AZRTrainer
from dataset_manager import DatasetManager
from dataset_catalog import DatasetCatalog
from reward_model import RewardComputer
from training_analytics import TrainingAnalytics

app = FastAPI(title="AZR Neural Network Trainer")

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
BOOKS_DIR = BASE_DIR / "books"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
TEMPLATES_DIR = BASE_DIR / "templates"
REPORTS_DIR = BASE_DIR / "reports"

for dir_path in [MODELS_DIR, BOOKS_DIR, CHECKPOINTS_DIR, REPORTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Менеджер и каталог датасетов
dataset_manager = DatasetManager("datasets_db.json")
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
    "device": "cpu",
}

download_status = {
    "is_downloading": False,
    "dataset_id": None,
    "progress": 0,
    "message": ""
}

gpu_install_status = {
    "is_installing": False,
    "progress": 0,
    "message": "",
    "success": False,
    "error": ""
}

active_models = {}
active_trainer = None
active_analytics = None


# ─────────────────────────────────────────
# Pydantic-схемы
# ─────────────────────────────────────────

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


# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def read_root():
    for name in ["index_complete.html", "index.html"]:
        f = TEMPLATES_DIR / name
        if f.exists():
            return f.read_text(encoding="utf-8")
    return "<h1>AZR Trainer</h1><p>Template not found</p>"


# ─────────────────────────────────────────
# Модели CRUD
# ─────────────────────────────────────────

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
        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config.dict(), f, indent=2)

        active_models[config.name] = {
            "model": model, "tokenizer": tokenizer, "config": config.dict()
        }

        return {
            "status": "success",
            "message": f"Модель '{config.name}' создана",
            "parameters": params,
            "config": config.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    models = []
    if not MODELS_DIR.exists():
        return {"models": []}
    for model_dir in MODELS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        config_file = model_dir / "config.json"
        if not config_file.exists():
            continue
        with open(config_file, encoding="utf-8") as f:
            config = json.load(f)

        model_info = dataset_manager.get_model_info(model_dir.name)
        trained_model = model_dir / "model_trained.pt"
        base_model = model_dir / "model.pt"
        size_bytes = trained_model.stat().st_size if trained_model.exists() else (
            base_model.stat().st_size if base_model.exists() else 0
        )

        models.append({
            "name": model_dir.name,
            "config": config,
            "datasets": model_info["datasets"] if model_info else [],
            "total_datasets": model_info["total_datasets"] if model_info else 0,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "trained": trained_model.exists()
        })
    return {"models": models}


@app.delete("/models/{model_name}")
async def delete_model(model_name: str):
    model_dir = MODELS_DIR / model_name
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Модель '{model_name}' не найдена")
    try:
        shutil.rmtree(model_dir)
        ckpt_dir = CHECKPOINTS_DIR / model_name
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        return {"status": "deleted", "model_name": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# Загрузка и управление датасетами
# ─────────────────────────────────────────

@app.post("/upload_book")
async def upload_book(file: UploadFile = File(...)):
    try:
        file_path = BOOKS_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        metadata = None
        converted_name = file.filename
        try:
            fmt_info = dataset_catalog.detect_format(file_path)
            metadata = fmt_info
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
            return {"status": "success",
                    "message": f"Датасет '{config.dataset_name}' подключён к '{config.model_name}'"}
        raise HTTPException(status_code=404, detail="Датасет не найден")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detach_dataset")
async def detach_dataset(config: DetachDatasetConfig):
    try:
        success = dataset_manager.detach_dataset(config.model_name, config.dataset_name)
        if success:
            return {"status": "success",
                    "message": f"Датасет '{config.dataset_name}' отключён от '{config.model_name}'"}
        raise HTTPException(status_code=404, detail="Не найдено")
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
                attached_info.append({"name": ds_name, "size": info["size"], "path": info["path"]})
        available = dataset_manager.get_available_datasets(model_name)
        return {"attached": attached_info, "available": available}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# Каталог датасетов
# ─────────────────────────────────────────

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
        raise HTTPException(status_code=400, detail="Уже идёт скачивание")

    info = dataset_catalog.get_dataset_info(dataset_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Датасет '{dataset_id}' не найден")

    def do_download():
        global download_status
        download_status.update({"is_downloading": True, "dataset_id": dataset_id, "progress": 0})

        def progress_cb(status):
            download_status["progress"] = status.get("progress", 0)
            download_status["message"] = status.get("message", "")

        try:
            file_path = dataset_catalog.download_dataset(dataset_id, progress_callback=progress_cb)
            if file_path:
                dataset_manager.register_dataset(
                    file_path.name, str(file_path),
                    metadata={"source": "catalog", "catalog_id": dataset_id,
                              "description": info.get("description", ""), "format": "txt"}
                )
                download_status["message"] = f"Скачано: {file_path.name}"
        except Exception as e:
            download_status["message"] = f"Ошибка: {e}"
            download_status["progress"] = -1
        finally:
            download_status["is_downloading"] = False

    threading.Thread(target=do_download).start()
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
    try:
        custom_entry = dataset_catalog.add_custom_url(name, url, language)
        return {"status": "added", "dataset": custom_entry}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# Обучение
# ─────────────────────────────────────────

def _load_model_and_tokenizer(model_name: str):
    """Вспомогательная: загрузить модель и токенизатор из диска или кэша."""
    if model_name in active_models:
        return active_models[model_name]["model"], active_models[model_name]["tokenizer"]

    model_dir = MODELS_DIR / model_name
    with open(model_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    model = CustomTransformerLM(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
        num_layers=cfg["num_layers"], num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"], max_seq_len=cfg["max_seq_len"]
    )
    ckpt = torch.load(model_dir / "model.pt", map_location="cpu")
    model.load_state_dict(ckpt if not isinstance(ckpt, dict) else ckpt.get("model_state_dict", ckpt))
    tokenizer = SimpleTokenizer.load(model_dir / "tokenizer.pkl")
    return model, tokenizer


def train_model_background(config: TrainingConfig):
    global training_status, active_trainer, active_analytics

    try:
        training_status.update({
            "is_training": True,
            "model_name": config.model_name,
            "max_iterations": config.max_iterations,
            "error": None
        })

        model_dir = MODELS_DIR / config.model_name
        model, tokenizer = _load_model_and_tokenizer(config.model_name)

        # Загружаем все прикреплённые датасеты
        texts = dataset_manager.load_attached_texts(config.model_name)
        if not texts:
            training_status.update({"is_training": False,
                                    "error": "Нет прикреплённых датасетов. Сначала прикрепи датасет."})
            return

        # Обновляем токенизатор если нужно
        current_datasets = set(dataset_manager.get_attached_datasets(config.model_name))
        tokenizer_datasets = set(tokenizer.get_trained_datasets())
        is_pretrained = len(tokenizer.token_to_id) > 1000 and len(tokenizer_datasets) == 0

        if not is_pretrained:
            if len(tokenizer.token_to_id) <= len(tokenizer.special_tokens):
                preserve = False
            elif current_datasets != tokenizer_datasets:
                preserve = True
            else:
                preserve = None  # не нужно

            if preserve is not None:
                combined = [" ".join(texts[i:i+100]) for i in range(0, len(texts), 100)]
                tokenizer.train(combined, preserve_existing=preserve)
                tokenizer.save(model_dir / "tokenizer.pkl",
                               trained_on_datasets=list(current_datasets))

        # Устройство
        if config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif config.device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        else:
            device = config.device

        training_status["device"] = device
        model = model.to(device)

        def update_status(d):
            training_status.update(d)

        reward_computer = RewardComputer(tokenizer, reference_texts=texts[:100])
        reports_dir = REPORTS_DIR / config.model_name
        analytics = TrainingAnalytics(reports_dir)
        active_analytics = analytics

        active_trainer = AZRTrainer(
            model, tokenizer,
            device=device,
            status_callback=update_status,
            reward_computer=reward_computer,
            analytics=analytics
        )

        history = active_trainer.train_continuous(
            texts=texts,
            max_iterations=config.max_iterations,
            batch_size=config.batch_size,
            lr=config.learning_rate,
            save_every=config.save_every,
            checkpoint_dir=CHECKPOINTS_DIR / config.model_name,
            resume_from=config.resume_from
        )

        torch.save(model.state_dict(), model_dir / "model_trained.pt")
        training_status.update({"is_training": False, "history": history})

    except Exception as e:
        training_status.update({"is_training": False, "error": str(e)})
        import traceback
        traceback.print_exc()


@app.post("/train")
async def start_training(config: TrainingConfig):
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Обучение уже идёт")

    model_dir = MODELS_DIR / config.model_name
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Модель '{config.model_name}' не найдена")

    attached = dataset_manager.get_attached_datasets(config.model_name)
    if not attached:
        raise HTTPException(status_code=400,
                            detail="Нет прикреплённых датасетов. Сначала прикрепи датасет!")

    threading.Thread(target=train_model_background, args=(config,)).start()
    return {
        "status": "success",
        "message": f"Обучение запущено на {len(attached)} датасет(ах)",
        "datasets": attached,
        "config": config.dict()
    }


@app.post("/stop_training")
async def stop_training():
    global active_trainer
    if active_trainer:
        active_trainer.stop_training()
        return {"status": "stopping", "message": "Обучение остановится после текущего батча"}
    return {"status": "not_training"}


@app.get("/training_status")
async def get_training_status():
    return training_status


# ─────────────────────────────────────────
# Аналитика обучения
# ─────────────────────────────────────────

@app.get("/training_analytics")
async def get_training_analytics():
    if active_analytics:
        return active_analytics.get_summary()
    return {"error": "Нет активной сессии аналитики"}


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
        return {"error": f"Итерации {iter_a} или {iter_b} не найдены"}
    return {"error": "Нет активной сессии аналитики"}


# ─────────────────────────────────────────
# Чекпоинты и сравнение
# ─────────────────────────────────────────

@app.get("/checkpoints/{model_name}")
async def list_checkpoints(model_name: str):
    ckpt_dir = CHECKPOINTS_DIR / model_name
    if not ckpt_dir.exists():
        return {"checkpoints": []}
    checkpoints = []
    for f in sorted(ckpt_dir.glob("model_iter_*.pt")):
        try:
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
        ckpt_file = CHECKPOINTS_DIR / config.model_name / f"model_iter_{config.checkpoint_iteration}.pt"
        if not ckpt_file.exists():
            raise HTTPException(status_code=404,
                                detail=f"Чекпоинт итерации {config.checkpoint_iteration} не найден")

        with open(model_dir / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)

        model = CustomTransformerLM(**{k: cfg[k] for k in
                                       ["vocab_size", "d_model", "num_layers", "num_heads", "d_ff", "max_seq_len"]})
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt)

        tokenizer = SimpleTokenizer.load(model_dir / "tokenizer.pkl")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        tokens = tokenizer.encode(config.prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            gen = model.generate(idx, max_new_tokens=config.max_length,
                                 temperature=config.temperature, top_k=40)

        gen_text = tokenizer.decode(gen[0].cpu().tolist())
        quality = RewardComputer(tokenizer).compute_reward(gen_text)

        return {"status": "success", "iteration": config.checkpoint_iteration,
                "prompt": config.prompt, "generated_text": gen_text, "quality_metrics": quality}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare_generations")
async def compare_generations(config: CompareConfig):
    try:
        model_dir = MODELS_DIR / config.model_name
        with open(model_dir / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)

        tokenizer = SimpleTokenizer.load(model_dir / "tokenizer.pkl")
        reward_computer = RewardComputer(tokenizer)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        results = []

        for iteration in config.iterations[:5]:
            ckpt_file = CHECKPOINTS_DIR / config.model_name / f"model_iter_{iteration}.pt"
            if not ckpt_file.exists():
                results.append({"iteration": iteration, "error": "Чекпоинт не найден"})
                continue

            model = CustomTransformerLM(**{k: cfg[k] for k in
                                           ["vocab_size", "d_model", "num_layers", "num_heads", "d_ff", "max_seq_len"]})
            ckpt = torch.load(ckpt_file, map_location="cpu")
            model.load_state_dict(ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt)
            model = model.to(device)
            model.eval()

            tokens = tokenizer.encode(config.prompt)
            idx = torch.tensor([tokens], dtype=torch.long, device=device)
            with torch.no_grad():
                gen = model.generate(idx, max_new_tokens=config.max_length, temperature=0.8, top_k=40)

            gen_text = tokenizer.decode(gen[0].cpu().tolist())
            quality = reward_computer.compute_reward(gen_text)
            results.append({"iteration": iteration, "text": gen_text, "quality": quality})
            del model

        return {"prompt": config.prompt, "comparisons": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# Генерация текста
# ─────────────────────────────────────────

@app.post("/generate")
async def generate_text(config: GenerateConfig):
    try:
        model_dir = MODELS_DIR / config.model_name

        if config.model_name in active_models:
            model = active_models[config.model_name]["model"]
            tokenizer = active_models[config.model_name]["tokenizer"]
        else:
            with open(model_dir / "config.json", encoding="utf-8") as f:
                cfg = json.load(f)
            model = CustomTransformerLM(**{k: cfg[k] for k in
                                           ["vocab_size", "d_model", "num_layers", "num_heads", "d_ff", "max_seq_len"]})
            trained = model_dir / "model_trained.pt"
            weights_path = trained if trained.exists() else model_dir / "model.pt"
            ckpt = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt)
            tokenizer = SimpleTokenizer.load(model_dir / "tokenizer.pkl")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        input_tokens = tokenizer.encode(config.prompt)
        current_idx = torch.tensor([input_tokens], dtype=torch.long, device=device)
        token_details = []

        with torch.no_grad():
            for _ in range(config.max_length):
                idx_cond = (current_idx if current_idx.size(1) <= model.max_seq_len
                            else current_idx[:, -model.max_seq_len:])
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / config.temperature

                if config.top_k:
                    v, _ = torch.topk(logits, min(config.top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                chosen_prob = probs[0, idx_next[0, 0]].item()

                top3_probs, top3_ids = torch.topk(probs[0], min(3, probs.size(-1)))
                alternatives = [
                    {"token": tokenizer.id_to_token.get(tid.item(), f"<{tid.item()}>"),
                     "prob": round(p.item(), 4)}
                    for p, tid in zip(top3_probs, top3_ids)
                ]

                token_id = idx_next[0, 0].item()
                token_str = tokenizer.id_to_token.get(token_id, f"<{token_id}>")
                token_details.append({
                    "token": token_str,
                    "confidence": round(chosen_prob, 4),
                    "top_alternatives": alternatives
                })
                current_idx = torch.cat((current_idx, idx_next), dim=1)

        generated_text = tokenizer.decode(current_idx[0].cpu().tolist())
        quality = RewardComputer(tokenizer).compute_reward(generated_text)

        return {
            "status": "success",
            "prompt": config.prompt,
            "generated_text": generated_text,
            "token_details": token_details,
            "quality_metrics": quality,
            "tokens_generated": len(token_details),
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_before_after")
async def generate_before_after(config: GenerateConfig):
    try:
        model_dir = MODELS_DIR / config.model_name
        with open(model_dir / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)

        tokenizer = SimpleTokenizer.load(model_dir / "tokenizer.pkl")
        reward_computer = RewardComputer(tokenizer)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def _gen(weights_path):
            m = CustomTransformerLM(**{k: cfg[k] for k in
                                       ["vocab_size", "d_model", "num_layers", "num_heads", "d_ff", "max_seq_len"]})
            ckpt = torch.load(weights_path, map_location="cpu")
            m.load_state_dict(ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt)
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

        before_text, before_quality = _gen(model_dir / "model.pt")
        trained_path = model_dir / "model_trained.pt"
        if trained_path.exists():
            after_text, after_quality = _gen(trained_path)
        else:
            after_text, after_quality = before_text, before_quality

        improvement = {
            "reward_delta": round(after_quality["total"] - before_quality["total"], 4),
            "components_delta": {
                k: round(after_quality.get("components", {}).get(k, 0) -
                         before_quality.get("components", {}).get(k, 0), 4)
                for k in after_quality.get("components", {})
            }
        }

        return {
            "prompt": config.prompt,
            "before": {"text": before_text, "quality": before_quality},
            "after": {"text": after_text, "quality": after_quality},
            "improvement": improvement
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# Скачивание / загрузка модели
# ─────────────────────────────────────────

@app.get("/download_model/{model_name}")
async def download_model(model_name: str):
    model_dir = MODELS_DIR / model_name
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail="Модель не найдена")
    trained = model_dir / "model_trained.pt"
    path = trained if trained.exists() else model_dir / "model.pt"
    return FileResponse(path, filename=f"{model_name}_trained.pt",
                        media_type="application/octet-stream")


@app.post("/upload_model/{model_name}")
async def upload_model(model_name: str, file: UploadFile = File(...)):
    try:
        model_dir = MODELS_DIR / model_name
        if not model_dir.exists():
            raise HTTPException(status_code=404, detail=f"Модель '{model_name}' не найдена")
        model_path = model_dir / "model_trained.pt"
        with open(model_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Проверяем что файл валидный
        try:
            with open(model_dir / "config.json", encoding="utf-8") as f:
                cfg = json.load(f)
            m = CustomTransformerLM(**{k: cfg[k] for k in
                                       ["vocab_size", "d_model", "num_layers", "num_heads", "d_ff", "max_seq_len"]})
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
            if model_name in active_models:
                active_models[model_name]["model"] = m
            return {"status": "success", "size": model_path.stat().st_size}
        except Exception as e:
            model_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Невалидный файл модели: {e}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# Системная информация
# ─────────────────────────────────────────

@app.get("/dataset_stats")
async def get_dataset_stats():
    return dataset_manager.get_stats()


@app.get("/device_info")
async def get_device_info():
    cuda_available = torch.cuda.is_available()
    info = {
        "cpu": {"available": True, "name": "CPU", "cores": torch.get_num_threads()},
        "cuda": {
            "available": cuda_available,
            "name": torch.cuda.get_device_name(0) if cuda_available else None,
            "count": torch.cuda.device_count() if cuda_available else 0,
            "memory": None
        },
        "current_device": "cuda" if cuda_available else "cpu",
        "recommendation": "GPU (CUDA)" if cuda_available else "CPU (медленно, рекомендуется GPU)",
        "pytorch_version": torch.__version__,
    }
    if cuda_available:
        try:
            mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            info["cuda"]["memory"] = f"{mem_gb:.1f} GB"
        except Exception:
            pass
    return info


@app.get("/hardware_recommendation")
async def get_hardware_recommendation():
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
        ram_gb = 8

    if has_gpu and gpu_mem_gb >= 8:
        preset, reason = "quality", f"GPU {torch.cuda.get_device_name(0)} ({gpu_mem_gb:.0f} ГБ) — большие модели"
    elif has_gpu and gpu_mem_gb >= 4:
        preset, reason = "standard", f"GPU {torch.cuda.get_device_name(0)} ({gpu_mem_gb:.0f} ГБ) — стандартные модели"
    elif has_gpu:
        preset, reason = "quick", f"GPU {gpu_mem_gb:.0f} ГБ — только маленькие модели"
    elif ram_gb >= 16:
        preset, reason = "standard", f"{ram_gb:.0f} ГБ RAM, CPU — стандартные модели (медленно)"
    else:
        preset, reason = "quick", f"{ram_gb:.0f} ГБ RAM, CPU — быстрые тесты"

    presets = {
        "quick":    {"d_model": 128, "num_layers": 4, "num_heads": 4, "d_ff": 512,
                     "max_seq_len": 128, "vocab_size": 10000,
                     "max_iterations": 1000, "batch_size": 8, "learning_rate": 0.001},
        "standard": {"d_model": 256, "num_layers": 6, "num_heads": 8, "d_ff": 1024,
                     "max_seq_len": 256, "vocab_size": 15000,
                     "max_iterations": 10000, "batch_size": 16, "learning_rate": 0.0003},
        "quality":  {"d_model": 512, "num_layers": 12, "num_heads": 16, "d_ff": 2048,
                     "max_seq_len": 512, "vocab_size": 25000,
                     "max_iterations": 100000, "batch_size": 32, "learning_rate": 0.0001}
    }

    return {
        "preset": preset, "reason": reason,
        "config": presets[preset],
        "has_gpu": has_gpu,
        "gpu_memory_gb": round(gpu_mem_gb, 1),
        "ram_gb": round(ram_gb, 1)
    }


# ─────────────────────────────────────────
# GPU — установка и активация из UI
# ─────────────────────────────────────────

CUDA_INDEXES = [
    "https://download.pytorch.org/whl/cu124",
    "https://download.pytorch.org/whl/cu121",
]

def _run_uv(args: list, timeout: int = 60) -> subprocess.CompletedProcess:
    """Запускает uv через текущий Python: python -m uv <args>"""
    return subprocess.run(
        [sys.executable, "-m", "uv"] + args,
        capture_output=True, text=True, timeout=timeout
    )

def _find_cuda_index_for_python(python_exe: str) -> str:
    """Проверяет, какой CUDA-индекс PyTorch поддерживает данный python_exe."""
    for idx in CUDA_INDEXES:
        try:
            r = subprocess.run(
                [python_exe, "-m", "pip", "index", "versions", "torch",
                 "--index-url", idx],
                capture_output=True, text=True, timeout=20
            )
            if r.returncode == 0 and "torch" in r.stdout.lower():
                return idx
        except Exception:
            continue
    return ""

def _get_venv_python(venv_dir: Path) -> str:
    """Путь к python.exe внутри venv (Windows / Linux)."""
    for p in [venv_dir / "Scripts" / "python.exe",
              venv_dir / "bin" / "python"]:
        if p.exists():
            return str(p)
    return ""


@app.get("/gpu_status")
async def get_gpu_status():
    """Полный статус GPU: CUDA, версия Python, наличие GPU-venv."""
    cuda_available = torch.cuda.is_available()
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    venv_ready = bool(_get_venv_python(BASE_DIR / ".venv-gpu"))
    return {
        "cuda_available": cuda_available,
        "pytorch_version": torch.__version__,
        "python_version": py_ver,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "venv_gpu_ready": venv_ready,
        "install_status": gpu_install_status,
    }


@app.post("/gpu_install")
async def install_gpu_pytorch():
    """
    Универсальная установка PyTorch CUDA для любой версии Python.
    Алгоритм:
      1. Если текущий Python поддерживает CUDA-колёса — ставим прямо в него.
      2. Иначе (Python 3.14+): через uv скачиваем Python 3.13, создаём .venv-gpu,
         ставим туда torch+CUDA, создаём start_gpu.bat / start_gpu.sh.
    """
    global gpu_install_status

    if gpu_install_status["is_installing"]:
        return {"status": "already_installing", "message": "Установка уже идёт"}
    if torch.cuda.is_available():
        return {"status": "already_available", "message": "GPU уже доступен!"}

    def _upd(progress: int, message: str):
        gpu_install_status["progress"] = progress
        gpu_install_status["message"] = message

    def do_install():
        global gpu_install_status
        gpu_install_status.update({
            "is_installing": True, "progress": 2,
            "message": "Запускаем...", "success": False, "error": ""
        })

        try:
            # ── Путь А: текущий Python поддерживает CUDA ────────────────────
            _upd(5, "Шаг 1/3: Проверяем CUDA-совместимость текущего Python...")
            cur_idx = _find_cuda_index_for_python(sys.executable)

            if cur_idx:
                cuda_ver = "12.4" if "cu124" in cur_idx else "12.1"
                _upd(15, f"Удаляем CPU-версию PyTorch...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "uninstall",
                     "torch", "torchvision", "torchaudio", "-y"],
                    capture_output=True, timeout=120
                )
                _upd(30, f"Скачиваем PyTorch CUDA {cuda_ver} (~2 ГБ)...")
                r = subprocess.run(
                    [sys.executable, "-m", "pip", "install",
                     "torch", "torchvision", "--index-url", cur_idx],
                    capture_output=True, text=True, timeout=900
                )
                if r.returncode == 0:
                    gpu_install_status.update({
                        "progress": 100,
                        "message": "✅ PyTorch CUDA установлен! Перезапусти сервер.",
                        "success": True
                    })
                else:
                    err = (r.stderr or r.stdout or "")[-600:]
                    gpu_install_status.update({
                        "progress": -1,
                        "message": "❌ Ошибка pip install. Смотри поле error.",
                        "error": err
                    })
                return  # Путь А завершён

            # ── Путь Б: несовместимый Python (3.14+) → uv → Python 3.13 ────
            _upd(8, "Текущий Python несовместим с CUDA PyTorch.\nШаг 1/5: Устанавливаем uv...")

            # Убеждаемся что uv доступен
            uv_check = _run_uv(["--version"], timeout=10)
            if uv_check.returncode != 0:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "uv", "-q"],
                    capture_output=True, timeout=90
                )

            _upd(15, "Шаг 2/5: Скачиваем Python 3.13 (~30 МБ)...")
            # Пробуем 3.13, при неудаче — 3.12
            py_target = "3.13"
            r = _run_uv(["python", "install", py_target], timeout=300)
            if r.returncode != 0:
                py_target = "3.12"
                r = _run_uv(["python", "install", py_target], timeout=300)
                if r.returncode != 0:
                    gpu_install_status.update({
                        "progress": -1,
                        "message": f"❌ Не удалось скачать Python {py_target} через uv.\n{r.stderr[-300:]}",
                        "error": r.stderr[-300:]
                    })
                    return

            _upd(35, f"Шаг 3/5: Создаём GPU-окружение (Python {py_target})...")
            venv_dir = BASE_DIR / ".venv-gpu"
            r = _run_uv(["venv", str(venv_dir), "--python", py_target, "--clear"], timeout=60)
            if r.returncode != 0:
                gpu_install_status.update({
                    "progress": -1,
                    "message": f"❌ Не удалось создать venv.\n{r.stderr[-300:]}",
                    "error": r.stderr[-300:]
                })
                return

            venv_python = _get_venv_python(venv_dir)
            if not venv_python:
                gpu_install_status.update({
                    "progress": -1,
                    "message": "❌ venv создан, но python.exe не найден.",
                    "error": "venv python not found"
                })
                return

            # Python 3.13 точно поддерживает cu124; используем uv pip (pip не нужен в venv)
            idx = CUDA_INDEXES[0]  # cu124
            cuda_ver = "12.4"
            _upd(45, f"Шаг 4/5: Скачиваем PyTorch CUDA {cuda_ver} (~2 ГБ)...")
            r = _run_uv(
                ["pip", "install", "torch", "torchvision",
                 "--python", venv_python,
                 "--index-url", idx],
                timeout=900
            )
            if r.returncode != 0:
                # Попробуем cu121
                idx = CUDA_INDEXES[1]
                cuda_ver = "12.1"
                _upd(50, f"Шаг 4/5: cu124 не сработал, пробуем CUDA {cuda_ver}...")
                r = _run_uv(
                    ["pip", "install", "torch", "torchvision",
                     "--python", venv_python,
                     "--index-url", idx],
                    timeout=900
                )
            if r.returncode != 0:
                err = (r.stderr or r.stdout or "")[-600:]
                gpu_install_status.update({
                    "progress": -1,
                    "message": "❌ Ошибка установки PyTorch в venv.",
                    "error": err
                })
                return

            _upd(88, "Шаг 5/5: Устанавливаем зависимости проекта в venv...")
            _run_uv(
                ["pip", "install", "fastapi", "uvicorn[standard]", "pydantic",
                 "--python", venv_python, "-q"],
                timeout=180
            )

            # Создаём скрипты запуска
            _upd(95, "Создаём start_gpu.bat / start_gpu.sh...")
            server_path = BASE_DIR / "server.py"

            bat = f'@echo off\necho Starting AZR Trainer with GPU support...\n"{venv_python}" "{server_path}"\npause\n'
            (BASE_DIR / "start_gpu.bat").write_text(bat, encoding="utf-8")

            venv_py_sh = (venv_dir / "bin" / "python")
            sh = f'#!/bin/bash\necho "Starting AZR Trainer with GPU support..."\n"{venv_py_sh}" "{server_path}"\n'
            sh_path = BASE_DIR / "start_gpu.sh"
            sh_path.write_text(sh, encoding="utf-8")
            try:
                sh_path.chmod(0o755)
            except Exception:
                pass

            gpu_install_status.update({
                "progress": 100,
                "message": (
                    "✅ Готово! GPU-окружение создано.\n"
                    "Закрой сервер и запусти через start_gpu.bat (Windows) "
                    "или start_gpu.sh (Linux/Mac)."
                ),
                "success": True
            })

        except subprocess.TimeoutExpired:
            gpu_install_status.update({
                "progress": -1,
                "message": "❌ Превышен таймаут. Установка заняла слишком долго.",
                "error": "timeout"
            })
        except Exception as e:
            gpu_install_status.update({
                "progress": -1,
                "message": f"❌ Ошибка: {e}",
                "error": str(e)
            })
        finally:
            gpu_install_status["is_installing"] = False

    threading.Thread(target=do_install, daemon=True).start()
    return {"status": "installing", "message": "Установка началась."}


# ─────────────────────────────────────────
# Запуск
# ─────────────────────────────────────────

if __name__ == "__main__":
    out = sys.stdout.buffer if hasattr(sys.stdout, 'buffer') else None
    def _p(s):
        if out:
            out.write((s + "\n").encode("utf-8"))
            out.flush()
        else:
            print(s)
    _p("=" * 60)
    _p("  AZR Neural Network Trainer")
    _p("=" * 60)
    cuda = torch.cuda.is_available()
    _p(f"  Device:    {'GPU - ' + torch.cuda.get_device_name(0) if cuda else 'CPU (slow)'}")
    _p(f"  PyTorch:   {torch.__version__}")
    _p(f"  Models:    {MODELS_DIR}")
    _p(f"  Datasets:  {BOOKS_DIR}")
    _p(f"  Catalog:   {len(dataset_catalog.catalog)} datasets")
    if not cuda:
        _p("")
        _p("  ! GPU not found. Open browser and click 'Activate GPU'")
        _p("    (NVIDIA GPU + internet required)")
    _p("=" * 60)
    _p("")
    _p("  >>> Open in browser: http://localhost:8000 <<<")
    _p("")
    uvicorn.run(app, host="0.0.0.0", port=8000)
