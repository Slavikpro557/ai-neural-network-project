"""
Многокомпонентная система наград для оценки качества генерации текста.
Заменяет простую формулу unique_ratio * 0.5 + length_score * 0.5
на 6 компонентов с детальной разбивкой.
"""

import math
from collections import Counter
from typing import List, Dict, Optional


class RewardComputer:
    """Вычисляет детальную награду за сгенерированный текст"""

    # Веса компонентов
    WEIGHTS = {
        "diversity": 0.15,
        "coherence": 0.25,
        "repetition_penalty": 0.20,
        "length_score": 0.10,
        "vocabulary_richness": 0.15,
        "bigram_naturalness": 0.15,
    }

    def __init__(self, tokenizer, reference_texts: Optional[List[str]] = None):
        self.tokenizer = tokenizer
        self.bigram_probs = {}
        self.unigram_probs = {}
        self.total_bigrams = 0
        self.total_unigrams = 0

        if reference_texts:
            self._build_reference_stats(reference_texts)

    def _build_reference_stats(self, texts: List[str]):
        """Строим статистику биграмм и униграмм из обучающих текстов"""
        bigram_counts = Counter()
        unigram_counts = Counter()

        for text in texts:
            tokens = self.tokenizer.encode(text)
            for t in tokens:
                unigram_counts[t] += 1
            for i in range(len(tokens) - 1):
                bigram_counts[(tokens[i], tokens[i + 1])] += 1

        self.total_unigrams = sum(unigram_counts.values()) or 1
        self.total_bigrams = sum(bigram_counts.values()) or 1

        self.unigram_probs = {
            k: v / self.total_unigrams for k, v in unigram_counts.items()
        }
        self.bigram_probs = {
            k: v / self.total_bigrams for k, v in bigram_counts.items()
        }

    def update_reference(self, texts: List[str]):
        """Обновить референсную статистику новыми текстами"""
        self._build_reference_stats(texts)

    def compute_reward(self, text: str) -> Dict:
        """Вычислить детальную награду за текст"""
        tokens = self.tokenizer.encode(text)

        if len(tokens) < 2:
            return {
                "total": 0.0,
                "components": {k: 0.0 for k in self.WEIGHTS}
            }

        components = {
            "diversity": self._diversity_score(tokens),
            "coherence": self._coherence_score(tokens),
            "repetition_penalty": self._repetition_penalty(tokens),
            "length_score": self._length_score(tokens),
            "vocabulary_richness": self._vocabulary_richness(tokens),
            "bigram_naturalness": self._bigram_naturalness(tokens),
        }

        total = sum(components[k] * self.WEIGHTS[k] for k in components)
        total = max(0.0, min(1.0, total))

        return {"total": total, "components": components}

    def compute_batch_reward(self, texts: List[str]) -> List[Dict]:
        """Вычислить награду для батча текстов"""
        return [self.compute_reward(text) for text in texts]

    def _diversity_score(self, tokens: List[int]) -> float:
        """Уникальность токенов: unique / total"""
        if len(tokens) == 0:
            return 0.0
        return len(set(tokens)) / len(tokens)

    def _coherence_score(self, tokens: List[int]) -> float:
        """Когерентность: средняя log-вероятность биграмм по референсным данным"""
        if len(tokens) < 2 or not self.bigram_probs:
            return 0.5  # Нейтральная оценка если нет референса

        log_probs = []
        smoothing = 1e-8

        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i + 1])
            prob = self.bigram_probs.get(bigram, smoothing)
            log_probs.append(math.log(prob + smoothing))

        if not log_probs:
            return 0.5

        avg_log_prob = sum(log_probs) / len(log_probs)
        # Нормализуем в [0, 1]: типичный диапазон [-15, -2]
        score = (avg_log_prob + 15) / 13
        return max(0.0, min(1.0, score))

    def _repetition_penalty(self, tokens: List[int]) -> float:
        """Штраф за повторения n-грамм (1, 2, 3-граммы)"""
        if len(tokens) < 3:
            return 1.0

        penalties = []

        for n in [1, 2, 3]:
            ngrams = []
            for i in range(len(tokens) - n + 1):
                ngrams.append(tuple(tokens[i:i + n]))

            if not ngrams:
                continue

            total = len(ngrams)
            unique = len(set(ngrams))
            # Доля уникальных n-грамм
            ratio = unique / total
            penalties.append(ratio)

        if not penalties:
            return 1.0

        # Среднее по всем n
        return sum(penalties) / len(penalties)

    def _length_score(self, tokens: List[int]) -> float:
        """Оценка длины: оптимальный диапазон 20-200 токенов"""
        length = len(tokens)

        if length < 5:
            return 0.1
        elif length < 20:
            return 0.3 + 0.7 * (length - 5) / 15
        elif length <= 200:
            return 1.0
        elif length <= 500:
            return 1.0 - 0.5 * (length - 200) / 300
        else:
            return 0.5

    def _vocabulary_richness(self, tokens: List[int]) -> float:
        """Богатство словаря: type-token ratio + hapax legomena"""
        if len(tokens) == 0:
            return 0.0

        counter = Counter(tokens)
        types = len(counter)
        total = len(tokens)

        # Type-token ratio
        ttr = types / total

        # Hapax legomena (слова встретившиеся 1 раз)
        hapax = sum(1 for count in counter.values() if count == 1)
        hapax_ratio = hapax / types if types > 0 else 0

        # Комбинируем
        score = ttr * 0.6 + hapax_ratio * 0.4
        return max(0.0, min(1.0, score))

    def _bigram_naturalness(self, tokens: List[int]) -> float:
        """Естественность: насколько биграммы похожи на обучающие данные"""
        if len(tokens) < 2 or not self.bigram_probs:
            return 0.5

        # Считаем биграммы генерации
        gen_bigrams = Counter()
        for i in range(len(tokens) - 1):
            gen_bigrams[(tokens[i], tokens[i + 1])] += 1

        total_gen = sum(gen_bigrams.values()) or 1

        # Считаем долю биграмм которые встречаются в референсе
        known_count = 0
        for bigram, count in gen_bigrams.items():
            if bigram in self.bigram_probs:
                known_count += count

        coverage = known_count / total_gen
        return max(0.0, min(1.0, coverage))
