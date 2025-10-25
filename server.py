from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import torch
import json
import uvicorn
from datetime import datetime
import threading
import shutil

from model import CustomTransformerLM, count_parameters
from tokenizer import SimpleTokenizer
from azr_trainer import AZRTrainer

app = FastAPI(title="AZR Model Trainer")

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
BOOKS_DIR = BASE_DIR / "books"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
TEMPLATES_DIR = BASE_DIR / "templates"

for dir_path in [MODELS_DIR, BOOKS_DIR, CHECKPOINTS_DIR]:
    dir_path.mkdir(exist_ok=True)

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
    book_file: str
    max_iterations: int = 1000000
    batch_size: int = 16
    learning_rate: float = 3e-4
    save_every: int = 1000


class GenerateConfig(BaseModel):
    model_name: str
    prompt: str
    max_length: int = 100
    temperature: float = 0.8
    top_k: int = 40


@app.get("/", response_class=HTMLResponse)
async def read_root():
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
        
        return {
            "status": "success",
            "filename": file.filename,
            "path": str(file_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/books")
async def list_books():
    books = [f.name for f in BOOKS_DIR.iterdir() if f.is_file()]
    return {"books": books}


@app.get("/models")
async def list_models():
    models = []
    for model_dir in MODELS_DIR.iterdir():
        if model_dir.is_dir():
            config_file = model_dir / "config.json"
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
                models.append({
                    "name": model_dir.name,
                    "config": config
                })
    return {"models": models}


def train_model_background(config: TrainingConfig):
    global training_status
    
    try:
        training_status["is_training"] = True
        training_status["model_name"] = config.model_name
        training_status["max_iterations"] = config.max_iterations
        
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
            model.load_state_dict(torch.load(model_dir / "model.pt"))
            tokenizer = SimpleTokenizer.load(model_dir / "tokenizer.pkl")
        
        book_path = BOOKS_DIR / config.book_file
        with open(book_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if len(tokenizer.token_to_id) <= len(tokenizer.special_tokens):
            tokenizer.train([text])
            tokenizer.save(model_dir / "tokenizer.pkl")
        
        texts = [text[i:i+1000] for i in range(0, len(text), 500)]
        
        # Callback to update status in real-time
        def update_status(status_dict):
            training_status.update(status_dict)
        
        trainer = AZRTrainer(model, tokenizer, status_callback=update_status)
        
        history = trainer.train_continuous(
            texts=texts,
            max_iterations=config.max_iterations,
            batch_size=config.batch_size,
            lr=config.learning_rate,
            save_every=config.save_every,
            checkpoint_dir=CHECKPOINTS_DIR / config.model_name
        )
        
        torch.save(model.state_dict(), model_dir / "model_trained.pt")
        
        training_status["is_training"] = False
        training_status["history"] = history
        
    except Exception as e:
        training_status["is_training"] = False
        training_status["error"] = str(e)
        print(f"Training error: {e}")


@app.post("/train")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    model_dir = MODELS_DIR / config.model_name
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model '{config.model_name}' not found")
    
    book_path = BOOKS_DIR / config.book_file
    if not book_path.exists():
        raise HTTPException(status_code=404, detail=f"Book '{config.book_file}' not found")
    
    thread = threading.Thread(target=train_model_background, args=(config,))
    thread.start()
    
    return {
        "status": "success",
        "message": "Training started in background",
        "config": config.dict()
    }


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
                model.load_state_dict(torch.load(trained_model))
            else:
                model.load_state_dict(torch.load(model_dir / "model.pt"))
            
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


if __name__ == "__main__":
    print("Starting AZR Model Trainer Server...")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Books directory: {BOOKS_DIR}")
    print(f"Checkpoints directory: {CHECKPOINTS_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
