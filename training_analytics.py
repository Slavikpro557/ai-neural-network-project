"""
Система аналитики обучения: детальные отчёты по итерациям,
бенчмарки, перплексия, сравнение чекпоинтов.
"""

import json
import time
import math
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


# Фиксированные промпты для бенчмарков (RU + EN)
DEFAULT_BENCHMARK_PROMPTS = [
    "Искусственный интеллект",
    "В далёком будущем",
    "Секрет успеха заключается в",
    "Однажды в тёмном лесу",
    "The future of technology",
    "Once upon a time",
    "The most important thing in life",
    "In a world where machines",
]


class TrainingAnalytics:
    """Детальная аналитика процесса обучения"""

    def __init__(self, reports_dir: Path, benchmark_prompts: List[str] = None):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True, parents=True)

        self.benchmark_prompts = benchmark_prompts or DEFAULT_BENCHMARK_PROMPTS
        self.iteration_reports = []
        self.benchmark_history = []
        self.start_time = None
        self.tokens_processed = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._batch_times = []
        self._last_batch_time = None

    def start_session(self):
        """Начать сессию аналитики"""
        self.start_time = time.time()
        self.tokens_processed = 0
        self._batch_times = []

    def record_batch_time(self, batch_tokens: int):
        """Записать время обработки батча"""
        now = time.time()
        if self._last_batch_time is not None:
            elapsed = now - self._last_batch_time
            if elapsed > 0:
                self._batch_times.append({
                    "tokens": batch_tokens,
                    "time": elapsed,
                    "speed": batch_tokens / elapsed
                })
                # Храним только последние 100 замеров
                if len(self._batch_times) > 100:
                    self._batch_times.pop(0)
        self._last_batch_time = now
        self.tokens_processed += batch_tokens

    def get_tokens_per_sec(self) -> float:
        """Средняя скорость обработки (последние 50 батчей)"""
        if not self._batch_times:
            return 0.0
        recent = self._batch_times[-50:]
        total_tokens = sum(b["tokens"] for b in recent)
        total_time = sum(b["time"] for b in recent)
        if total_time == 0:
            return 0.0
        return total_tokens / total_time

    def record_iteration(self, iteration: int, loss: float, reward_breakdown: Dict,
                         generated_samples: List[str], perplexity: float,
                         learning_rate: float, tokens_per_sec: float,
                         vocab_usage: Dict = None):
        """Записать полный отчёт по итерации"""
        report = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "loss": round(loss, 6),
            "perplexity": round(perplexity, 2),
            "reward": {
                "total": round(reward_breakdown.get("total", 0), 4),
                "components": {
                    k: round(v, 4)
                    for k, v in reward_breakdown.get("components", {}).items()
                }
            },
            "learning_rate": learning_rate,
            "tokens_per_sec": round(tokens_per_sec, 1),
            "samples": generated_samples[:3],
            "vocab_usage": vocab_usage or {},
            "elapsed_seconds": round(time.time() - self.start_time, 1) if self.start_time else 0,
        }

        self.iteration_reports.append(report)

    def record_benchmarks(self, iteration: int, benchmark_results: List[Dict]):
        """Записать результаты бенчмарков"""
        entry = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "results": benchmark_results
        }
        self.benchmark_history.append(entry)

    @torch.no_grad()
    def compute_perplexity(self, model, eval_tokens: List[List[int]], device: str) -> float:
        """Вычислить перплексию на валидационных данных"""
        model.eval()
        total_loss = 0.0
        total_count = 0

        for tokens in eval_tokens[:50]:  # Макс 50 примеров для скорости
            if len(tokens) < 2:
                continue

            x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

            try:
                logits, loss = model(x, y)
                if loss is not None and not math.isnan(loss.item()):
                    total_loss += loss.item()
                    total_count += 1
            except Exception:
                continue

        if total_count == 0:
            return float('inf')

        avg_loss = total_loss / total_count
        perplexity = math.exp(min(avg_loss, 20))  # Ограничиваем чтобы не было overflow
        return round(perplexity, 2)

    @torch.no_grad()
    def run_benchmarks(self, model, tokenizer, device: str,
                       reward_computer=None, max_tokens: int = 50) -> List[Dict]:
        """Генерация по фиксированным промптам для отслеживания качества"""
        model.eval()
        results = []

        for prompt in self.benchmark_prompts:
            try:
                tokens = tokenizer.encode(prompt)
                if len(tokens) == 0:
                    continue

                idx = torch.tensor([tokens], dtype=torch.long, device=device)
                generated = model.generate(idx, max_new_tokens=max_tokens,
                                          temperature=0.8, top_k=40)

                gen_tokens = generated[0].cpu().tolist()
                gen_text = tokenizer.decode(gen_tokens)

                result = {
                    "prompt": prompt,
                    "text": gen_text,
                    "tokens_generated": len(gen_tokens) - len(tokens),
                }

                if reward_computer:
                    reward = reward_computer.compute_reward(gen_text)
                    result["reward"] = reward

                results.append(result)
            except Exception as e:
                results.append({
                    "prompt": prompt,
                    "text": f"[Error: {e}]",
                    "tokens_generated": 0,
                })

        return results

    def compute_vocab_usage(self, samples: List[str], tokenizer) -> Dict:
        """Анализ использования словаря в сгенерированных текстах"""
        all_tokens = []
        for text in samples:
            all_tokens.extend(tokenizer.encode(text))

        if not all_tokens:
            return {"unique_tokens": 0, "total_tokens": 0, "coverage_pct": 0, "top_tokens": []}

        counter = {}
        for t in all_tokens:
            counter[t] = counter.get(t, 0) + 1

        unique = len(counter)
        total_vocab = len(tokenizer.token_to_id) if hasattr(tokenizer, 'token_to_id') else 1
        coverage = round(unique / max(total_vocab, 1) * 100, 2)

        # Топ-20 токенов
        sorted_tokens = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:20]
        top_tokens = []
        for token_id, count in sorted_tokens:
            token_str = tokenizer.id_to_token.get(token_id, f"<{token_id}>") if hasattr(tokenizer, 'id_to_token') else str(token_id)
            top_tokens.append({"token": token_str, "count": count})

        return {
            "unique_tokens": unique,
            "total_tokens": len(all_tokens),
            "coverage_pct": coverage,
            "top_tokens": top_tokens
        }

    def compare_iterations(self, iter_a: int, iter_b: int) -> Optional[Dict]:
        """Сравнить две итерации"""
        report_a = None
        report_b = None

        for r in self.iteration_reports:
            if r["iteration"] == iter_a:
                report_a = r
            if r["iteration"] == iter_b:
                report_b = r

        if not report_a or not report_b:
            return None

        comparison = {
            "iteration_a": iter_a,
            "iteration_b": iter_b,
            "loss_delta": round(report_b["loss"] - report_a["loss"], 6),
            "perplexity_delta": round(report_b["perplexity"] - report_a["perplexity"], 2),
            "reward_delta": round(
                report_b["reward"]["total"] - report_a["reward"]["total"], 4
            ),
            "reward_components_delta": {},
            "speed_delta": round(
                report_b["tokens_per_sec"] - report_a["tokens_per_sec"], 1
            ),
            "samples_a": report_a.get("samples", []),
            "samples_b": report_b.get("samples", []),
        }

        # Дельта по каждому компоненту награды
        comp_a = report_a["reward"].get("components", {})
        comp_b = report_b["reward"].get("components", {})
        for key in comp_b:
            comparison["reward_components_delta"][key] = round(
                comp_b.get(key, 0) - comp_a.get(key, 0), 4
            )

        # Интерпретация изменений
        improvements = []
        degradations = []

        if comparison["loss_delta"] < -0.01:
            improvements.append(f"Loss улучшился на {abs(comparison['loss_delta']):.4f}")
        elif comparison["loss_delta"] > 0.01:
            degradations.append(f"Loss ухудшился на {comparison['loss_delta']:.4f}")

        if comparison["reward_delta"] > 0.005:
            improvements.append(f"Reward вырос на {comparison['reward_delta']:.4f}")
        elif comparison["reward_delta"] < -0.005:
            degradations.append(f"Reward упал на {abs(comparison['reward_delta']):.4f}")

        for key, delta in comparison["reward_components_delta"].items():
            if delta > 0.02:
                improvements.append(f"{key} улучшился на {delta:.4f}")
            elif delta < -0.02:
                degradations.append(f"{key} ухудшился на {abs(delta):.4f}")

        comparison["improvements"] = improvements
        comparison["degradations"] = degradations
        comparison["overall"] = "improved" if len(improvements) > len(degradations) else (
            "degraded" if len(degradations) > len(improvements) else "stable"
        )

        return comparison

    def get_eta(self, current_iteration: int, max_iterations: int) -> int:
        """Оценка оставшегося времени в секундах"""
        tokens_per_sec = self.get_tokens_per_sec()
        if tokens_per_sec <= 0 or not self._batch_times:
            return -1

        # Среднее время на итерацию (батч)
        recent = self._batch_times[-20:]
        avg_batch_time = sum(b["time"] for b in recent) / len(recent)

        remaining = max_iterations - current_iteration
        return int(remaining * avg_batch_time)

    def get_summary(self) -> Dict:
        """Сводка для API"""
        summary = {
            "session_id": self.session_id,
            "total_iterations_recorded": len(self.iteration_reports),
            "total_benchmarks": len(self.benchmark_history),
            "tokens_per_sec": round(self.get_tokens_per_sec(), 1),
            "elapsed_seconds": round(time.time() - self.start_time, 1) if self.start_time else 0,
        }

        if self.iteration_reports:
            latest = self.iteration_reports[-1]
            summary["latest"] = {
                "iteration": latest["iteration"],
                "loss": latest["loss"],
                "perplexity": latest["perplexity"],
                "reward": latest["reward"],
                "vocab_usage": latest.get("vocab_usage", {}),
            }

            # Тренд по последним 5 записям
            if len(self.iteration_reports) >= 2:
                recent = self.iteration_reports[-5:]
                summary["trend"] = {
                    "loss_direction": "decreasing" if recent[-1]["loss"] < recent[0]["loss"] else "increasing",
                    "reward_direction": "increasing" if recent[-1]["reward"]["total"] > recent[0]["reward"]["total"] else "decreasing",
                }

        return summary

    def get_all_reports(self) -> List[Dict]:
        """Все отчёты по итерациям"""
        return self.iteration_reports

    def get_benchmark_history(self) -> List[Dict]:
        """Вся история бенчмарков"""
        return self.benchmark_history

    def save_report(self, path: Path = None):
        """Сохранить полный отчёт на диск"""
        if path is None:
            path = self.reports_dir / f"report_{self.session_id}.json"

        report = {
            "session_id": self.session_id,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            "iterations": self.iteration_reports,
            "benchmarks": self.benchmark_history,
            "summary": self.get_summary(),
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return path
