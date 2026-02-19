"""
Каталог датасетов: встроенный каталог популярных датасетов,
поиск по HuggingFace, скачивание, превью, определение формата.
"""

import json
import csv
import io
import os
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from typing import List, Dict, Optional, Callable
from datetime import datetime


# Встроенный каталог датасетов (публичные домены / свободные лицензии)
BUILT_IN_CATALOG = [
    # === ЛИТЕРАТУРА (EN) ===
    {
        "id": "gutenberg_alice",
        "name": "Alice in Wonderland",
        "name_ru": "Алиса в стране чудес",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
        "size_estimate": "~170 KB",
        "description": "Классика Льюиса Кэрролла. Отличный стартовый датасет для обучения на английском.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_sherlock",
        "name": "Adventures of Sherlock Holmes",
        "name_ru": "Приключения Шерлока Холмса",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
        "size_estimate": "~580 KB",
        "description": "Детективные рассказы Конан Дойла. Хороший датасет среднего размера.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_frankenstein",
        "name": "Frankenstein",
        "name_ru": "Франкенштейн",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/84/pg84.txt",
        "size_estimate": "~440 KB",
        "description": "Роман Мэри Шелли. Готический стиль, богатая лексика.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_pride",
        "name": "Pride and Prejudice",
        "name_ru": "Гордость и предубеждение",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
        "size_estimate": "~710 KB",
        "description": "Роман Джейн Остин. Элегантный английский стиль.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_moby_dick",
        "name": "Moby Dick",
        "name_ru": "Моби Дик",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
        "size_estimate": "~1.2 MB",
        "description": "Большой роман Мелвилла. Объёмный датасет для серьёзного обучения.",
        "difficulty": "intermediate",
    },
    {
        "id": "gutenberg_dracula",
        "name": "Dracula",
        "name_ru": "Дракула",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/345/pg345.txt",
        "size_estimate": "~860 KB",
        "description": "Роман Брэма Стокера. Эпистолярный стиль, разнообразная лексика.",
        "difficulty": "intermediate",
    },
    {
        "id": "gutenberg_grimm",
        "name": "Grimm's Fairy Tales",
        "name_ru": "Сказки братьев Гримм",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/2591/pg2591.txt",
        "size_estimate": "~530 KB",
        "description": "Коллекция сказок. Простой язык, много повторяющихся структур — идеально для обучения.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_dorian_gray",
        "name": "The Picture of Dorian Gray",
        "name_ru": "Портрет Дориана Грея",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/174/pg174.txt",
        "size_estimate": "~490 KB",
        "description": "Роман Оскара Уайльда. Богатый, выразительный язык.",
        "difficulty": "intermediate",
    },
    {
        "id": "gutenberg_great_expectations",
        "name": "Great Expectations",
        "name_ru": "Большие надежды",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",
        "size_estimate": "~1 MB",
        "description": "Роман Чарльза Диккенса. Большой объём, великолепный стиль.",
        "difficulty": "intermediate",
    },
    {
        "id": "gutenberg_war_of_worlds",
        "name": "The War of the Worlds",
        "name_ru": "Война миров",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/36/pg36.txt",
        "size_estimate": "~340 KB",
        "description": "Научная фантастика Герберта Уэллса. Динамичный нарратив.",
        "difficulty": "beginner",
    },
    # === ЛИТЕРАТУРА (RU) === (источник: github.com/d0rj/RusLit — русские оригиналы)
    {
        "id": "ruslit_karenina",
        "name": "Анна Каренина",
        "name_ru": "Анна Каренина",
        "category": "literature_ru",
        "language": "ru",
        "source": "github_ruslit",
        "url": "https://raw.githubusercontent.com/d0rj/RusLit/master/prose/Tolstoy/%D0%90%D0%BD%D0%BD%D0%B0%20%D0%9A%D0%B0%D1%80%D0%B5%D0%BD%D0%B8%D0%BD%D0%B0.txt",
        "size_estimate": "~1.8 MB",
        "description": "Роман Льва Толстого на русском языке. Огромный датасет, классический русский стиль.",
        "difficulty": "advanced",
    },
    {
        "id": "ruslit_war_and_peace_1",
        "name": "Война и мир (Том 1)",
        "name_ru": "Война и мир (Том 1)",
        "category": "literature_ru",
        "language": "ru",
        "source": "github_ruslit",
        "url": "https://raw.githubusercontent.com/d0rj/RusLit/master/prose/Tolstoy/%D0%92%D0%BE%D0%B9%D0%BD%D0%B0%20%D0%B8%20%D0%BC%D0%B8%D1%80.%20%D0%A2%D0%BE%D0%BC%201.txt",
        "size_estimate": "~800 KB",
        "description": "Первый том масштабного романа Толстого. Русский оригинал.",
        "difficulty": "advanced",
    },
    {
        "id": "ruslit_war_and_peace_2",
        "name": "Война и мир (Том 2)",
        "name_ru": "Война и мир (Том 2)",
        "category": "literature_ru",
        "language": "ru",
        "source": "github_ruslit",
        "url": "https://raw.githubusercontent.com/d0rj/RusLit/master/prose/Tolstoy/%D0%92%D0%BE%D0%B9%D0%BD%D0%B0%20%D0%B8%20%D0%BC%D0%B8%D1%80.%20%D0%A2%D0%BE%D0%BC%202.txt",
        "size_estimate": "~800 KB",
        "description": "Второй том масштабного романа Толстого. Русский оригинал.",
        "difficulty": "advanced",
    },
    {
        "id": "ruslit_war_and_peace_3",
        "name": "Война и мир (Том 3)",
        "name_ru": "Война и мир (Том 3)",
        "category": "literature_ru",
        "language": "ru",
        "source": "github_ruslit",
        "url": "https://raw.githubusercontent.com/d0rj/RusLit/master/prose/Tolstoy/%D0%92%D0%BE%D0%B9%D0%BD%D0%B0%20%D0%B8%20%D0%BC%D0%B8%D1%80.%20%D0%A2%D0%BE%D0%BC%203.txt",
        "size_estimate": "~800 KB",
        "description": "Третий том масштабного романа Толстого. Русский оригинал.",
        "difficulty": "advanced",
    },
    {
        "id": "ruslit_war_and_peace_4",
        "name": "Война и мир (Том 4)",
        "name_ru": "Война и мир (Том 4)",
        "category": "literature_ru",
        "language": "ru",
        "source": "github_ruslit",
        "url": "https://raw.githubusercontent.com/d0rj/RusLit/master/prose/Tolstoy/%D0%92%D0%BE%D0%B9%D0%BD%D0%B0%20%D0%B8%20%D0%BC%D0%B8%D1%80.%20%D0%A2%D0%BE%D0%BC%204.txt",
        "size_estimate": "~800 KB",
        "description": "Четвёртый том масштабного романа Толстого. Русский оригинал.",
        "difficulty": "advanced",
    },
    {
        "id": "ruslit_crime_punishment",
        "name": "Преступление и наказание",
        "name_ru": "Преступление и наказание",
        "category": "literature_ru",
        "language": "ru",
        "source": "github_ruslit",
        "url": "https://raw.githubusercontent.com/d0rj/RusLit/master/prose/Dostoevsky/%D0%91%D0%B5%D0%B4%D0%BD%D1%8B%D0%B5%20%D0%BB%D1%8E%D0%B4%D0%B8.txt",
        "size_estimate": "~400 KB",
        "description": "Роман Достоевского. Психологический стиль, сложные конструкции.",
        "difficulty": "intermediate",
    },
    {
        "id": "ruslit_brothers_karamazov",
        "name": "Братья Карамазовы",
        "name_ru": "Братья Карамазовы",
        "category": "literature_ru",
        "language": "ru",
        "source": "github_ruslit",
        "url": "https://raw.githubusercontent.com/d0rj/RusLit/master/prose/Dostoevsky/%D0%91%D1%80%D0%B0%D1%82%D1%8C%D1%8F%20%D0%9A%D0%B0%D1%80%D0%B0%D0%BC%D0%B0%D0%B7%D0%BE%D0%B2%D1%8B.txt",
        "size_estimate": "~1.5 MB",
        "description": "Последний великий роман Достоевского. Огромный датасет, глубокая проза.",
        "difficulty": "advanced",
    },
    {
        "id": "ruslit_idiot",
        "name": "Идиот",
        "name_ru": "Идиот",
        "category": "literature_ru",
        "language": "ru",
        "source": "github_ruslit",
        "url": "https://raw.githubusercontent.com/d0rj/RusLit/master/prose/Dostoevsky/%D0%98%D0%B4%D0%B8%D0%BE%D1%82.txt",
        "size_estimate": "~1.1 MB",
        "description": "Роман Достоевского. Психологическая глубина, русский стиль.",
        "difficulty": "intermediate",
    },
    {
        "id": "ruslit_dead_souls",
        "name": "Мёртвые души",
        "name_ru": "Мёртвые души",
        "category": "literature_ru",
        "language": "ru",
        "source": "github_ruslit",
        "url": "https://raw.githubusercontent.com/d0rj/RusLit/master/prose/Gogol/%D0%9C%D1%91%D1%80%D1%82%D0%B2%D1%8B%D0%B5%20%D0%B4%D1%83%D1%88%D0%B8.txt",
        "size_estimate": "~700 KB",
        "description": "Поэма Гоголя. Сатирический стиль, яркий русский язык.",
        "difficulty": "intermediate",
    },
    {
        "id": "ruslit_shinel",
        "name": "Шинель",
        "name_ru": "Шинель",
        "category": "literature_ru",
        "language": "ru",
        "source": "github_ruslit",
        "url": "https://raw.githubusercontent.com/d0rj/RusLit/master/prose/Gogol/%D0%A8%D0%B8%D0%BD%D0%B5%D0%BB%D1%8C.txt",
        "size_estimate": "~60 KB",
        "description": "Повесть Гоголя. Компактный датасет, идеален для начала обучения на русском.",
        "difficulty": "beginner",
    },
    {
        "id": "ruslit_taras_bulba",
        "name": "Тарас Бульба",
        "name_ru": "Тарас Бульба",
        "category": "literature_ru",
        "language": "ru",
        "source": "github_ruslit",
        "url": "https://raw.githubusercontent.com/d0rj/RusLit/master/prose/Gogol/%D0%A2%D0%B0%D1%80%D0%B0%D1%81%20%D0%91%D1%83%D0%BB%D1%8C%D0%B1%D0%B0.txt",
        "size_estimate": "~200 KB",
        "description": "Повесть Гоголя. Эпический стиль, яркие описания.",
        "difficulty": "beginner",
    },
    {
        "id": "ruslit_eugene_onegin",
        "name": "Евгений Онегин",
        "name_ru": "Евгений Онегин",
        "category": "poetry",
        "language": "ru",
        "source": "github_ruslit",
        "url": "https://raw.githubusercontent.com/d0rj/RusLit/master/poems/Pushkin/%D0%95%D0%B2%D0%B3%D0%B5%D0%BD%D0%B8%D0%B9%20%D0%9E%D0%BD%D0%B5%D0%B3%D0%B8%D0%BD.txt",
        "size_estimate": "~200 KB",
        "description": "Роман в стихах Пушкина. Поэтический стиль, рифма.",
        "difficulty": "intermediate",
    },
    {
        "id": "ruslit_ruslan_ludmila",
        "name": "Руслан и Людмила",
        "name_ru": "Руслан и Людмила",
        "category": "poetry",
        "language": "ru",
        "source": "github_ruslit",
        "url": "https://raw.githubusercontent.com/d0rj/RusLit/master/poems/Pushkin/%D0%A0%D1%83%D1%81%D0%BB%D0%B0%D0%BD%20%D0%B8%20%D0%9B%D1%8E%D0%B4%D0%BC%D0%B8%D0%BB%D0%B0.txt",
        "size_estimate": "~120 KB",
        "description": "Поэма Пушкина. Сказочный стиль, красивый русский язык.",
        "difficulty": "beginner",
    },
    {
        "id": "ruslit_resurrection",
        "name": "Воскресение",
        "name_ru": "Воскресение",
        "category": "literature_ru",
        "language": "ru",
        "source": "github_ruslit",
        "url": "https://raw.githubusercontent.com/d0rj/RusLit/master/prose/Tolstoy/%D0%92%D0%BE%D1%81%D0%BA%D1%80%D0%B5%D1%81%D0%B5%D0%BD%D0%B8%D0%B5.txt",
        "size_estimate": "~900 KB",
        "description": "Последний роман Толстого. Социальная тематика, зрелый стиль.",
        "difficulty": "intermediate",
    },
    {
        "id": "ruslit_demons",
        "name": "Бесы",
        "name_ru": "Бесы",
        "category": "literature_ru",
        "language": "ru",
        "source": "github_ruslit",
        "url": "https://raw.githubusercontent.com/d0rj/RusLit/master/prose/Dostoevsky/%D0%91%D0%B5%D1%81%D1%8B.txt",
        "size_estimate": "~1.2 MB",
        "description": "Роман Достоевского. Политическая проза, сложные диалоги.",
        "difficulty": "advanced",
    },
    # === КОД ===
    {
        "id": "code_python_snippets",
        "name": "Python Code Examples",
        "name_ru": "Примеры кода Python",
        "category": "code",
        "language": "en",
        "source": "url",
        "url": "https://raw.githubusercontent.com/TheAlgorithms/Python/master/DIRECTORY.md",
        "size_estimate": "~50 KB",
        "description": "Список алгоритмов на Python. Маленький но полезный для понимания кода.",
        "difficulty": "beginner",
    },
    # === НАУКА ===
    {
        "id": "gutenberg_origin_species",
        "name": "On the Origin of Species",
        "name_ru": "Происхождение видов",
        "category": "science",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/1228/pg1228.txt",
        "size_estimate": "~900 KB",
        "description": "Научный труд Дарвина. Научный стиль, сложные конструкции.",
        "difficulty": "advanced",
    },
    {
        "id": "gutenberg_art_of_war",
        "name": "The Art of War",
        "name_ru": "Искусство войны",
        "category": "science",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/132/pg132.txt",
        "size_estimate": "~100 KB",
        "description": "Трактат Сунь-Цзы. Компактный текст, афористичный стиль.",
        "difficulty": "beginner",
    },
    # === ПОЭЗИЯ ===
    {
        "id": "gutenberg_shakespeare_sonnets",
        "name": "Shakespeare's Sonnets",
        "name_ru": "Сонеты Шекспира",
        "category": "poetry",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/1041/pg1041.txt",
        "size_estimate": "~100 KB",
        "description": "154 сонета Шекспира. Поэзия, компактный датасет.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_leaves_of_grass",
        "name": "Leaves of Grass",
        "name_ru": "Листья травы",
        "category": "poetry",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/1322/pg1322.txt",
        "size_estimate": "~800 KB",
        "description": "Сборник поэзии Уитмена. Свободный стих, большой объём.",
        "difficulty": "intermediate",
    },
    # === ФИЛОСОФИЯ ===
    {
        "id": "gutenberg_republic",
        "name": "The Republic (Plato)",
        "name_ru": "Государство (Платон)",
        "category": "science",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/1497/pg1497.txt",
        "size_estimate": "~650 KB",
        "description": "Философский диалог Платона. Диалоговый формат, глубокие размышления.",
        "difficulty": "advanced",
    },
    {
        "id": "gutenberg_meditations",
        "name": "Meditations (Marcus Aurelius)",
        "name_ru": "Размышления (Марк Аврелий)",
        "category": "science",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/2680/pg2680.txt",
        "size_estimate": "~200 KB",
        "description": "Философские записки римского императора. Компактный, афористичный.",
        "difficulty": "beginner",
    },
    # === ПРИКЛЮЧЕНИЯ ===
    {
        "id": "gutenberg_treasure_island",
        "name": "Treasure Island",
        "name_ru": "Остров сокровищ",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/120/pg120.txt",
        "size_estimate": "~370 KB",
        "description": "Приключенческий роман Стивенсона. Динамичный сюжет, простой язык.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_jungle_book",
        "name": "The Jungle Book",
        "name_ru": "Книга джунглей",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/236/pg236.txt",
        "size_estimate": "~290 KB",
        "description": "Сказки Киплинга. Природа, животные, простой стиль.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_tom_sawyer",
        "name": "The Adventures of Tom Sawyer",
        "name_ru": "Приключения Тома Сойера",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/74/pg74.txt",
        "size_estimate": "~410 KB",
        "description": "Классика Марка Твена. Живой разговорный язык.",
        "difficulty": "beginner",
    },
    # === УЖАСЫ / МИСТИКА ===
    {
        "id": "gutenberg_call_of_cthulhu",
        "name": "The Call of Cthulhu",
        "name_ru": "Зов Ктулху (сборник)",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/68283/pg68283.txt",
        "size_estimate": "~80 KB",
        "description": "Рассказ Лавкрафта. Мистический стиль, атмосферное повествование.",
        "difficulty": "beginner",
    },
]

# Категории с отображаемыми именами
CATEGORIES = {
    "literature_en": {"name": "Literature (EN)", "name_ru": "Литература (EN)", "icon": "📚"},
    "literature_ru": {"name": "Literature (RU)", "name_ru": "Литература (RU)", "icon": "📖"},
    "code": {"name": "Code", "name_ru": "Код", "icon": "💻"},
    "science": {"name": "Science", "name_ru": "Наука", "icon": "🔬"},
    "poetry": {"name": "Poetry", "name_ru": "Поэзия", "icon": "🎭"},
    "conversations": {"name": "Conversations", "name_ru": "Диалоги", "icon": "💬"},
    "news": {"name": "News", "name_ru": "Новости", "icon": "📰"},
    "custom": {"name": "Custom", "name_ru": "Свои", "icon": "📎"},
}


class DatasetCatalog:
    """Каталог датасетов с поиском, скачиванием и конвертацией"""

    def __init__(self, books_dir: Path, cache_dir: Path = None):
        self.books_dir = Path(books_dir)
        self.cache_dir = cache_dir or (self.books_dir / ".cache")
        self.books_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.catalog = list(BUILT_IN_CATALOG)
        self._load_custom_catalog()

    def get_catalog(self, category: str = None, language: str = None) -> List[Dict]:
        """Получить каталог с фильтрацией"""
        results = self.catalog

        if category:
            results = [d for d in results if d["category"] == category]
        if language:
            results = [d for d in results if d["language"] == language]

        # Добавляем статус скачивания
        for item in results:
            file_path = self.books_dir / f"{item['id']}.txt"
            item["downloaded"] = file_path.exists()
            if file_path.exists():
                item["local_size"] = file_path.stat().st_size

        return results

    def get_categories(self) -> List[Dict]:
        """Получить список категорий с количеством датасетов"""
        result = []
        for cat_id, cat_info in CATEGORIES.items():
            count = sum(1 for d in self.catalog if d["category"] == cat_id)
            if count > 0:
                result.append({
                    "id": cat_id,
                    "name": cat_info["name"],
                    "name_ru": cat_info["name_ru"],
                    "icon": cat_info["icon"],
                    "count": count
                })
        return result

    def get_dataset_info(self, dataset_id: str) -> Optional[Dict]:
        """Информация о конкретном датасете"""
        for d in self.catalog:
            if d["id"] == dataset_id:
                return d
        return None

    def download_dataset(self, dataset_id: str,
                         progress_callback: Callable = None) -> Optional[Path]:
        """Скачать датасет в books_dir"""
        info = self.get_dataset_info(dataset_id)
        if not info:
            raise ValueError(f"Dataset '{dataset_id}' not found in catalog")

        file_path = self.books_dir / f"{info['id']}.txt"

        # Если уже скачан
        if file_path.exists():
            if progress_callback:
                progress_callback({"progress": 100, "message": "Already downloaded"})
            return file_path

        url = info["url"]

        if progress_callback:
            progress_callback({"progress": 0, "message": f"Downloading {info['name']}..."})

        try:
            # Скачиваем
            req = urllib.request.Request(url, headers={
                'User-Agent': 'AZR-Model-Trainer/1.0'
            })

            response = urllib.request.urlopen(req, timeout=60)
            total_size = int(response.headers.get('Content-Length', 0))

            data = b""
            downloaded = 0
            chunk_size = 8192

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                data += chunk
                downloaded += len(chunk)

                if progress_callback and total_size > 0:
                    pct = int(downloaded / total_size * 100)
                    progress_callback({
                        "progress": pct,
                        "message": f"Downloading... {downloaded // 1024} KB / {total_size // 1024} KB"
                    })

            # Определяем кодировку и декодируем
            text = self._decode_text(data)

            # Очищаем Gutenberg header/footer если источник Gutenberg
            if info.get("source") == "gutenberg":
                text = self._clean_gutenberg(text)

            # Проверяем язык скачанного текста
            lang_check = self._validate_language(text, info.get("language", ""))
            if not lang_check["valid"]:
                if progress_callback:
                    progress_callback({
                        "progress": 90,
                        "message": f"WARNING: {lang_check['message']}"
                    })

            # Сохраняем
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)

            if progress_callback:
                progress_callback({
                    "progress": 100,
                    "message": f"Downloaded: {file_path.name} ({file_path.stat().st_size // 1024} KB)"
                })

            return file_path

        except Exception as e:
            if progress_callback:
                progress_callback({"progress": -1, "message": f"Error: {str(e)}"})
            raise

    def preview_dataset(self, dataset_id: str, lines: int = 20) -> Dict:
        """Превью датасета (первые N строк)"""
        info = self.get_dataset_info(dataset_id)
        if not info:
            return {"error": "Dataset not found"}

        file_path = self.books_dir / f"{info['id']}.txt"

        # Если уже скачан — читаем с диска
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                preview_lines = []
                for i, line in enumerate(f):
                    if i >= lines:
                        break
                    preview_lines.append(line.rstrip())

            return {
                "id": dataset_id,
                "name": info["name"],
                "lines": preview_lines,
                "total_size": file_path.stat().st_size,
                "source": "local"
            }

        # Если не скачан — пробуем скачать первый кусок
        try:
            req = urllib.request.Request(info["url"], headers={
                'User-Agent': 'AZR-Model-Trainer/1.0',
                'Range': 'bytes=0-8192'
            })
            response = urllib.request.urlopen(req, timeout=30)
            data = response.read(8192)
            text = self._decode_text(data)
            preview_lines = text.split('\n')[:lines]

            return {
                "id": dataset_id,
                "name": info["name"],
                "lines": preview_lines,
                "source": "remote_preview"
            }
        except Exception as e:
            return {
                "id": dataset_id,
                "name": info["name"],
                "lines": [f"Preview not available: {e}"],
                "source": "error"
            }

    def detect_format(self, file_path: Path) -> Dict:
        """Определить формат и кодировку файла"""
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        # Определяем кодировку
        encoding = 'utf-8'
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(1024)
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='cp1251') as f:
                    f.read(1024)
                encoding = 'cp1251'
            except UnicodeDecodeError:
                encoding = 'latin-1'

        # Считаем строки
        line_count = 0
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            for _ in f:
                line_count += 1

        # Определяем формат
        fmt = "txt"
        if ext == '.pdf':
            fmt = "pdf"
            line_count = 0  # PDF не имеет текстовых строк до конвертации
        elif ext in ('.csv', '.tsv'):
            fmt = "csv"
        elif ext == '.json':
            fmt = "json"
        elif ext == '.jsonl':
            fmt = "jsonl"
        elif ext in ('.md', '.markdown'):
            fmt = "txt"
        elif ext in ('.jpg', '.jpeg', '.png', '.bmp', '.gif'):
            fmt = "image"
            line_count = 0

        return {
            "encoding": encoding,
            "format": fmt,
            "extension": ext,
            "lines": line_count,
            "size": file_path.stat().st_size
        }

    def convert_to_txt(self, file_path: Path, source_format: str = None) -> Path:
        """Конвертировать файл в .txt для обучения"""
        file_path = Path(file_path)

        if source_format is None:
            info = self.detect_format(file_path)
            source_format = info["format"]
            encoding = info["encoding"]
        else:
            encoding = "utf-8"

        if source_format == "txt":
            return file_path  # Уже txt

        output_path = file_path.with_suffix('.txt')

        if source_format == "pdf":
            text = self._pdf_to_text(file_path)
        elif source_format == "image":
            text = self._image_to_text(file_path)
        elif source_format == "csv":
            text = self._csv_to_text(file_path, encoding)
        elif source_format == "json":
            text = self._json_to_text(file_path, encoding)
        elif source_format == "jsonl":
            text = self._jsonl_to_text(file_path, encoding)
        else:
            # Просто копируем как текст
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                text = f.read()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)

        return output_path

    def search_huggingface(self, query: str, limit: int = 10) -> List[Dict]:
        """Поиск датасетов на HuggingFace (REST API, без библиотеки)"""
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://huggingface.co/api/datasets?search={encoded_query}&limit={limit}&sort=downloads&direction=-1"

            req = urllib.request.Request(url, headers={
                'User-Agent': 'AZR-Model-Trainer/1.0'
            })

            response = urllib.request.urlopen(req, timeout=15)
            data = json.loads(response.read().decode('utf-8'))

            results = []
            for item in data:
                results.append({
                    "id": f"hf_{item.get('id', '').replace('/', '_')}",
                    "name": item.get("id", "Unknown"),
                    "name_ru": item.get("id", "Unknown"),
                    "category": "huggingface",
                    "language": "multi",
                    "source": "huggingface",
                    "description": item.get("description", "")[:200] or "HuggingFace dataset",
                    "downloads": item.get("downloads", 0),
                    "likes": item.get("likes", 0),
                    "tags": item.get("tags", [])[:5],
                    "size_estimate": "varies",
                    "difficulty": "intermediate",
                    "hf_id": item.get("id", ""),
                })

            return results

        except Exception as e:
            return [{"error": str(e)}]

    def _decode_text(self, data: bytes) -> str:
        """Декодировать байты в текст, пробуя разные кодировки"""
        for encoding in ['utf-8', 'cp1251', 'latin-1', 'ascii']:
            try:
                return data.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                continue
        return data.decode('utf-8', errors='replace')

    def _validate_language(self, text: str, expected_lang: str) -> Dict:
        """Проверить, соответствует ли язык текста ожидаемому"""
        if not expected_lang or not text:
            return {"valid": True, "message": ""}

        # Берём сэмпл из середины текста (пропускаем заголовки)
        sample_start = len(text) // 4
        sample = text[sample_start:sample_start + 2000]

        if expected_lang == "ru":
            # Считаем долю кириллических символов
            cyrillic = sum(1 for c in sample if '\u0400' <= c <= '\u04FF')
            alpha = sum(1 for c in sample if c.isalpha())
            if alpha == 0:
                return {"valid": False, "message": "Текст не содержит букв"}
            ratio = cyrillic / alpha
            if ratio < 0.3:
                return {
                    "valid": False,
                    "message": f"Ожидался русский текст, но найдено только {ratio*100:.0f}% кириллицы. Возможно скачан английский перевод."
                }
        elif expected_lang == "en":
            # Считаем долю латиницы
            latin = sum(1 for c in sample if 'a' <= c.lower() <= 'z')
            alpha = sum(1 for c in sample if c.isalpha())
            if alpha == 0:
                return {"valid": False, "message": "Text contains no letters"}
            ratio = latin / alpha
            if ratio < 0.3:
                return {
                    "valid": False,
                    "message": f"Expected English text but found only {ratio*100:.0f}% Latin characters."
                }

        return {"valid": True, "message": ""}

    def _clean_gutenberg(self, text: str) -> str:
        """Удалить Gutenberg header/footer"""
        # Ищем начало текста
        start_markers = [
            "*** START OF THE PROJECT GUTENBERG",
            "*** START OF THIS PROJECT GUTENBERG",
            "***START OF THE PROJECT GUTENBERG",
        ]
        end_markers = [
            "*** END OF THE PROJECT GUTENBERG",
            "*** END OF THIS PROJECT GUTENBERG",
            "***END OF THE PROJECT GUTENBERG",
            "End of the Project Gutenberg",
            "End of Project Gutenberg",
        ]

        for marker in start_markers:
            idx = text.find(marker)
            if idx != -1:
                # Пропускаем до конца строки после маркера
                newline = text.find('\n', idx)
                if newline != -1:
                    text = text[newline + 1:]
                break

        for marker in end_markers:
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx]
                break

        return text.strip()

    def _csv_to_text(self, file_path: Path, encoding: str) -> str:
        """Извлечь текст из CSV"""
        texts = []
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            reader = csv.reader(f)
            header = next(reader, None)

            # Находим колонку с самым длинным текстом
            text_col = 0
            if header:
                for i, col in enumerate(header):
                    col_lower = col.lower()
                    if col_lower in ('text', 'content', 'body', 'message', 'sentence'):
                        text_col = i
                        break

            for row in reader:
                if len(row) > text_col:
                    texts.append(row[text_col])

        return '\n'.join(texts)

    def _json_to_text(self, file_path: Path, encoding: str) -> str:
        """Извлечь текст из JSON"""
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            data = json.load(f)

        texts = self._extract_texts_from_json(data)
        return '\n'.join(texts)

    def _jsonl_to_text(self, file_path: Path, encoding: str) -> str:
        """Извлечь текст из JSON Lines"""
        texts = []
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    texts.extend(self._extract_texts_from_json(obj))
                except json.JSONDecodeError:
                    continue
        return '\n'.join(texts)

    def _extract_texts_from_json(self, data) -> List[str]:
        """Рекурсивно извлечь текстовые поля из JSON"""
        text_keys = {'text', 'content', 'body', 'message', 'sentence',
                     'paragraph', 'document', 'input', 'output', 'response'}
        texts = []

        if isinstance(data, dict):
            for key, value in data.items():
                if key.lower() in text_keys and isinstance(value, str) and len(value) > 10:
                    texts.append(value)
                elif isinstance(value, (dict, list)):
                    texts.extend(self._extract_texts_from_json(value))
        elif isinstance(data, list):
            for item in data:
                texts.extend(self._extract_texts_from_json(item))

        return texts

    def _pdf_to_text(self, file_path: Path) -> str:
        """Извлечь текст из PDF через PyMuPDF"""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(file_path))
            text_parts = []
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():
                    text_parts.append(page_text)
            doc.close()
            if not text_parts:
                return f"[PDF file {file_path.name} contains no extractable text. It may be a scanned document.]"
            return '\n\n'.join(text_parts)
        except ImportError:
            return f"[PyMuPDF not installed. Run: pip install PyMuPDF]"
        except Exception as e:
            return f"[Error reading PDF: {e}]"

    def _image_to_text(self, file_path: Path) -> str:
        """Попытка OCR из изображения"""
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img, lang='rus+eng')
            if text.strip():
                return text
            return f"[Image {file_path.name}: no text recognized by OCR]"
        except ImportError:
            return f"[OCR requires: pip install Pillow pytesseract + Tesseract installed]"
        except Exception as e:
            return f"[Error processing image: {e}]"

    def add_custom_url(self, name: str, url: str, language: str = "auto") -> Dict:
        """Добавить пользовательский датасет по URL"""
        custom_id = f"custom_{name.lower().replace(' ', '_')}"

        # Определяем язык если auto
        if language == "auto":
            language = "multi"

        entry = {
            "id": custom_id,
            "name": name,
            "name_ru": name,
            "category": "custom",
            "language": language,
            "source": "custom_url",
            "url": url,
            "size_estimate": "unknown",
            "description": f"User-added dataset from {url[:50]}...",
            "difficulty": "intermediate",
        }

        # Добавляем в каталог
        self.catalog.append(entry)

        # Сохраняем в файл чтобы не потерять при перезапуске
        custom_file = self.books_dir / ".custom_catalog.json"
        custom_list = []
        if custom_file.exists():
            try:
                with open(custom_file, 'r', encoding='utf-8') as f:
                    custom_list = json.load(f)
            except Exception:
                custom_list = []

        custom_list.append(entry)
        with open(custom_file, 'w', encoding='utf-8') as f:
            json.dump(custom_list, f, ensure_ascii=False, indent=2)

        return entry

    def _load_custom_catalog(self):
        """Загрузить пользовательские датасеты при инициализации"""
        custom_file = self.books_dir / ".custom_catalog.json"
        if custom_file.exists():
            try:
                with open(custom_file, 'r', encoding='utf-8') as f:
                    custom_list = json.load(f)
                for entry in custom_list:
                    if entry["id"] not in [d["id"] for d in self.catalog]:
                        self.catalog.append(entry)
            except Exception:
                pass
