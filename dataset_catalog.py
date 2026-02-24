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
    # === ЛИТЕРАТУРА (EN) — Детская классика ===
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
        "description_en": "Lewis Carroll classic. Great starter dataset for training on English text.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_alice_looking_glass",
        "name": "Through the Looking-Glass",
        "name_ru": "Сквозь зеркало",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/12/pg12.txt",
        "size_estimate": "~180 KB",
        "description": "Продолжение Алисы. Льюис Кэрролл. Фантастический стиль.",
        "description_en": "Sequel to Alice. Lewis Carroll. Fantastical style.",
        "difficulty": "beginner",
    },
    # === ДЕТЕКТИВЫ ===
    {
        "id": "gutenberg_sherlock",
        "name": "Adventures of Sherlock Holmes",
        "name_ru": "Приключения Шерлока Холмса",
        "category": "detective",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
        "size_estimate": "~580 KB",
        "description": "Детективные рассказы Конан Дойла. Хороший датасет среднего размера.",
        "description_en": "Conan Doyle detective stories. Good medium-sized dataset.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_memoirs_sherlock",
        "name": "Memoirs of Sherlock Holmes",
        "name_ru": "Записки о Шерлоке Холмсе",
        "category": "detective",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/834/pg834.txt",
        "size_estimate": "~490 KB",
        "description": "Второй сборник рассказов Конан Дойла о Холмсе.",
        "description_en": "Second collection of Conan Doyle's Holmes stories.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_return_sherlock",
        "name": "The Return of Sherlock Holmes",
        "name_ru": "Возвращение Шерлока Холмса",
        "category": "detective",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/108/pg108.txt",
        "size_estimate": "~580 KB",
        "description": "Третий сборник рассказов о Шерлоке Холмсе.",
        "description_en": "Third collection of Sherlock Holmes stories.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_hound_baskervilles",
        "name": "The Hound of the Baskervilles",
        "name_ru": "Собака Баскервилей",
        "category": "detective",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/2852/pg2852.txt",
        "size_estimate": "~320 KB",
        "description": "Знаменитый роман Конан Дойла. Готическая детективная атмосфера.",
        "description_en": "Famous Conan Doyle novel. Gothic detective atmosphere.",
        "difficulty": "beginner",
    },
    # === GOTHIC / HORROR ===
    {
        "id": "gutenberg_frankenstein",
        "name": "Frankenstein",
        "name_ru": "Франкенштейн",
        "category": "horror",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/84/pg84.txt",
        "size_estimate": "~440 KB",
        "description": "Роман Мэри Шелли. Готический стиль, богатая лексика.",
        "description_en": "Mary Shelley novel. Gothic style, rich vocabulary.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_dracula",
        "name": "Dracula",
        "name_ru": "Дракула",
        "category": "horror",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/345/pg345.txt",
        "size_estimate": "~860 KB",
        "description": "Роман Брэма Стокера. Эпистолярный стиль, разнообразная лексика.",
        "description_en": "Bram Stoker novel. Epistolary style, diverse vocabulary.",
        "difficulty": "intermediate",
    },
    {
        "id": "gutenberg_call_of_cthulhu",
        "name": "The Call of Cthulhu",
        "name_ru": "Зов Ктулху",
        "category": "horror",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/68283/pg68283.txt",
        "size_estimate": "~80 KB",
        "description": "Рассказ Лавкрафта. Мистический стиль, атмосферное повествование.",
        "description_en": "Lovecraft story. Mystical style, atmospheric narrative.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_strange_case_jekyll",
        "name": "Strange Case of Dr Jekyll and Mr Hyde",
        "name_ru": "Странная история доктора Джекила",
        "category": "horror",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/43/pg43.txt",
        "size_estimate": "~130 KB",
        "description": "Повесть Стивенсона. Психологический хоррор, двойственность личности.",
        "description_en": "Stevenson novella. Psychological horror, duality of personality.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_turn_of_screw",
        "name": "The Turn of the Screw",
        "name_ru": "Поворот винта",
        "category": "horror",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/209/pg209.txt",
        "size_estimate": "~200 KB",
        "description": "Повесть Генри Джеймса. Призраки, психологическое напряжение.",
        "description_en": "Henry James novella. Ghosts, psychological tension.",
        "difficulty": "intermediate",
    },
    # === ROMANCE / CLASSICS EN ===
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
        "description_en": "Jane Austen novel. Elegant English style.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_sense_sensibility",
        "name": "Sense and Sensibility",
        "name_ru": "Разум и чувства",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/161/pg161.txt",
        "size_estimate": "~680 KB",
        "description": "Роман Джейн Остин. Изысканный викторианский стиль.",
        "description_en": "Jane Austen novel. Refined Victorian style.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_emma",
        "name": "Emma",
        "name_ru": "Эмма",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/158/pg158.txt",
        "size_estimate": "~880 KB",
        "description": "Роман Джейн Остин. Социальная комедия, живые диалоги.",
        "description_en": "Jane Austen novel. Social comedy, lively dialogues.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_jane_eyre",
        "name": "Jane Eyre",
        "name_ru": "Джейн Эйр",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/1260/pg1260.txt",
        "size_estimate": "~1.1 MB",
        "description": "Роман Шарлотты Бронте. Страстный, эмоциональный нарратив.",
        "description_en": "Charlotte Bronte novel. Passionate, emotional narrative.",
        "difficulty": "intermediate",
    },
    {
        "id": "gutenberg_wuthering_heights",
        "name": "Wuthering Heights",
        "name_ru": "Грозовой перевал",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/768/pg768.txt",
        "size_estimate": "~680 KB",
        "description": "Роман Эмили Бронте. Атмосферный романтизм, сильные эмоции.",
        "description_en": "Emily Bronte novel. Atmospheric romanticism, strong emotions.",
        "difficulty": "intermediate",
    },
    # === VICTORIAN CLASSICS ===
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
        "description_en": "Major Melville novel. Large dataset for serious training.",
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
        "description_en": "Charles Dickens novel. Large volume, magnificent style.",
        "difficulty": "intermediate",
    },
    {
        "id": "gutenberg_oliver_twist",
        "name": "Oliver Twist",
        "name_ru": "Оливер Твист",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/730/pg730.txt",
        "size_estimate": "~870 KB",
        "description": "Роман Диккенса. Социальная критика, живые персонажи.",
        "description_en": "Dickens novel. Social criticism, vivid characters.",
        "difficulty": "intermediate",
    },
    {
        "id": "gutenberg_tale_two_cities",
        "name": "A Tale of Two Cities",
        "name_ru": "Повесть о двух городах",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/98/pg98.txt",
        "size_estimate": "~790 KB",
        "description": "Исторический роман Диккенса. Французская революция, драма.",
        "description_en": "Dickens historical novel. French Revolution, drama.",
        "difficulty": "intermediate",
    },
    {
        "id": "gutenberg_david_copperfield",
        "name": "David Copperfield",
        "name_ru": "Дэвид Копперфилд",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/766/pg766.txt",
        "size_estimate": "~1.9 MB",
        "description": "Крупнейший роман Диккенса. Огромный датасет, автобиографический стиль.",
        "description_en": "Dickens' largest novel. Huge dataset, autobiographical style.",
        "difficulty": "intermediate",
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
        "description_en": "Oscar Wilde novel. Rich, expressive language.",
        "difficulty": "intermediate",
    },
    {
        "id": "gutenberg_importance_earnest",
        "name": "The Importance of Being Earnest",
        "name_ru": "Как важно быть серьёзным",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/844/pg844.txt",
        "size_estimate": "~90 KB",
        "description": "Пьеса Оскара Уайльда. Остроумные диалоги, комедия.",
        "description_en": "Oscar Wilde play. Witty dialogues, comedy.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_tess",
        "name": "Tess of the d'Urbervilles",
        "name_ru": "Тэсс из рода д'Эрбервиллей",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/110/pg110.txt",
        "size_estimate": "~820 KB",
        "description": "Роман Томаса Харди. Трагическая история, богатый стиль.",
        "description_en": "Thomas Hardy novel. Tragic story, rich style.",
        "difficulty": "intermediate",
    },
    {
        "id": "gutenberg_middlemarch",
        "name": "Middlemarch",
        "name_ru": "Мидлмарч",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/145/pg145.txt",
        "size_estimate": "~1.9 MB",
        "description": "Роман Джордж Элиот. Огромный датасет, глубокий психологизм.",
        "description_en": "George Eliot novel. Huge dataset, deep psychological insight.",
        "difficulty": "advanced",
    },
    # === AMERICAN CLASSICS ===
    {
        "id": "gutenberg_huckleberry_finn",
        "name": "Adventures of Huckleberry Finn",
        "name_ru": "Приключения Гекльберри Финна",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/76/pg76.txt",
        "size_estimate": "~600 KB",
        "description": "Роман Марка Твена. Живой разговорный американский стиль.",
        "description_en": "Mark Twain novel. Lively colloquial American style.",
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
        "description_en": "Mark Twain classic. Lively colloquial language.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_scarlet_letter",
        "name": "The Scarlet Letter",
        "name_ru": "Алая буква",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/25344/pg25344.txt",
        "size_estimate": "~490 KB",
        "description": "Роман Готорна. Ранняя американская классика, нравственная драма.",
        "description_en": "Hawthorne novel. Early American classic, moral drama.",
        "difficulty": "intermediate",
    },
    {
        "id": "gutenberg_call_wild",
        "name": "The Call of the Wild",
        "name_ru": "Зов предков",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/215/pg215.txt",
        "size_estimate": "~180 KB",
        "description": "Повесть Джека Лондона. Динамичная, живая проза.",
        "description_en": "Jack London novella. Dynamic, vivid prose.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_white_fang",
        "name": "White Fang",
        "name_ru": "Белый клык",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/910/pg910.txt",
        "size_estimate": "~370 KB",
        "description": "Роман Джека Лондона. Природа Аляски, приключения.",
        "description_en": "Jack London novel. Alaskan wilderness, adventure.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_red_badge",
        "name": "The Red Badge of Courage",
        "name_ru": "Алый знак доблести",
        "category": "literature_en",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/73/pg73.txt",
        "size_estimate": "~250 KB",
        "description": "Роман Стивена Крейна. Гражданская война, реализм.",
        "description_en": "Stephen Crane novel. Civil War, realism.",
        "difficulty": "intermediate",
    },
    # === SCIENCE FICTION ===
    {
        "id": "gutenberg_war_of_worlds",
        "name": "The War of the Worlds",
        "name_ru": "Война миров",
        "category": "scifi",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/36/pg36.txt",
        "size_estimate": "~340 KB",
        "description": "Научная фантастика Герберта Уэллса. Динамичный нарратив.",
        "description_en": "H.G. Wells science fiction. Dynamic narrative.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_time_machine",
        "name": "The Time Machine",
        "name_ru": "Машина времени",
        "category": "scifi",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/35/pg35.txt",
        "size_estimate": "~170 KB",
        "description": "Научная фантастика Уэллса. Путешествие во времени.",
        "description_en": "H.G. Wells science fiction. Time travel.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_invisible_man",
        "name": "The Invisible Man",
        "name_ru": "Человек-невидимка",
        "category": "scifi",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/5230/pg5230.txt",
        "size_estimate": "~320 KB",
        "description": "Фантастика Герберта Уэллса. Научный триллер.",
        "description_en": "H.G. Wells fiction. Science thriller.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_island_dr_moreau",
        "name": "The Island of Doctor Moreau",
        "name_ru": "Остров доктора Моро",
        "category": "scifi",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/159/pg159.txt",
        "size_estimate": "~220 KB",
        "description": "Роман Уэллса. Этические вопросы науки, остросюжетный.",
        "description_en": "Wells novel. Ethical questions of science, thrilling plot.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_first_men_moon",
        "name": "The First Men in the Moon",
        "name_ru": "Первые люди на Луне",
        "category": "scifi",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/1013/pg1013.txt",
        "size_estimate": "~340 KB",
        "description": "Роман Уэллса. Лунная фантастика, приключения.",
        "description_en": "Wells novel. Lunar fiction, adventure.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_20k_leagues",
        "name": "Twenty Thousand Leagues Under the Sea",
        "name_ru": "Двадцать тысяч лье под водой",
        "category": "scifi",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/164/pg164.txt",
        "size_estimate": "~630 KB",
        "description": "Жюль Верн. Классическая научная фантастика, подводные приключения.",
        "description_en": "Jules Verne. Classic science fiction, underwater adventures.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_journey_centre",
        "name": "Journey to the Center of the Earth",
        "name_ru": "Путешествие к центру Земли",
        "category": "scifi",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/18857/pg18857.txt",
        "size_estimate": "~380 KB",
        "description": "Жюль Верн. Подземные миры, научные приключения.",
        "description_en": "Jules Verne. Underground worlds, scientific adventures.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_around_world",
        "name": "Around the World in Eighty Days",
        "name_ru": "Вокруг света за 80 дней",
        "category": "scifi",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/103/pg103.txt",
        "size_estimate": "~370 KB",
        "description": "Жюль Верн. Путешествия, приключения, динамичный сюжет.",
        "description_en": "Jules Verne. Travel, adventure, dynamic plot.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_mysterious_island",
        "name": "The Mysterious Island",
        "name_ru": "Таинственный остров",
        "category": "scifi",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/1268/pg1268.txt",
        "size_estimate": "~1.1 MB",
        "description": "Жюль Верн. Большой роман, выживание на острове.",
        "description_en": "Jules Verne. Large novel, island survival.",
        "difficulty": "intermediate",
    },
    # === ADVENTURE ===
    {
        "id": "gutenberg_treasure_island",
        "name": "Treasure Island",
        "name_ru": "Остров сокровищ",
        "category": "adventure",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/120/pg120.txt",
        "size_estimate": "~370 KB",
        "description": "Приключенческий роман Стивенсона. Динамичный сюжет, простой язык.",
        "description_en": "Stevenson adventure novel. Dynamic plot, simple language.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_kidnapped",
        "name": "Kidnapped",
        "name_ru": "Похищенный",
        "category": "adventure",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/421/pg421.txt",
        "size_estimate": "~380 KB",
        "description": "Роман Стивенсона. Шотландские приключения.",
        "description_en": "Stevenson novel. Scottish adventures.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_jungle_book",
        "name": "The Jungle Book",
        "name_ru": "Книга джунглей",
        "category": "adventure",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/236/pg236.txt",
        "size_estimate": "~290 KB",
        "description": "Сказки Киплинга. Природа, животные, простой стиль.",
        "description_en": "Kipling tales. Nature, animals, simple style.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_second_jungle",
        "name": "The Second Jungle Book",
        "name_ru": "Вторая книга джунглей",
        "category": "adventure",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/1937/pg1937.txt",
        "size_estimate": "~260 KB",
        "description": "Продолжение книги джунглей Киплинга.",
        "description_en": "Sequel to Kipling's Jungle Book.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_three_musketeers",
        "name": "The Three Musketeers",
        "name_ru": "Три мушкетёра",
        "category": "adventure",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/1257/pg1257.txt",
        "size_estimate": "~1.2 MB",
        "description": "Александр Дюма. Исторические приключения, динамичный сюжет.",
        "description_en": "Alexandre Dumas. Historical adventures, dynamic plot.",
        "difficulty": "intermediate",
    },
    {
        "id": "gutenberg_count_monte_cristo",
        "name": "The Count of Monte Cristo",
        "name_ru": "Граф Монте-Кристо",
        "category": "adventure",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/1184/pg1184.txt",
        "size_estimate": "~2.9 MB",
        "description": "Александр Дюма. Огромный роман, приключения и месть.",
        "description_en": "Alexandre Dumas. Huge novel, adventure and revenge.",
        "difficulty": "intermediate",
    },
    {
        "id": "gutenberg_robinson_crusoe",
        "name": "Robinson Crusoe",
        "name_ru": "Робинзон Крузо",
        "category": "adventure",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/521/pg521.txt",
        "size_estimate": "~560 KB",
        "description": "Даниэль Дефо. Классика выживания, подробный реалистичный нарратив.",
        "description_en": "Daniel Defoe. Survival classic, detailed realistic narrative.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_swiss_family",
        "name": "The Swiss Family Robinson",
        "name_ru": "Швейцарский Робинзон",
        "category": "adventure",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/3852/pg3852.txt",
        "size_estimate": "~720 KB",
        "description": "Семья на необитаемом острове. Детские приключения.",
        "description_en": "Family on a deserted island. Children's adventure.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_gulliver",
        "name": "Gulliver's Travels",
        "name_ru": "Путешествия Гулливера",
        "category": "adventure",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/829/pg829.txt",
        "size_estimate": "~600 KB",
        "description": "Сатира Свифта. Фантастические путешествия.",
        "description_en": "Swift's satire. Fantastical voyages.",
        "difficulty": "intermediate",
    },
    {
        "id": "gutenberg_don_quixote",
        "name": "Don Quixote",
        "name_ru": "Дон Кихот",
        "category": "adventure",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/996/pg996.txt",
        "size_estimate": "~2.2 MB",
        "description": "Сервантес. Один из величайших романов. Огромный датасет.",
        "description_en": "Cervantes. One of the greatest novels ever. Huge dataset.",
        "difficulty": "advanced",
    },
    # === CHILDREN / FAIRY TALES ===
    {
        "id": "gutenberg_grimm",
        "name": "Grimm's Fairy Tales",
        "name_ru": "Сказки братьев Гримм",
        "category": "fairy_tales",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/2591/pg2591.txt",
        "size_estimate": "~530 KB",
        "description": "Коллекция сказок. Простой язык, много повторяющихся структур.",
        "description_en": "Collection of fairy tales. Simple language, many repeating structures.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_andersen",
        "name": "Andersen's Fairy Tales",
        "name_ru": "Сказки Андерсена",
        "category": "fairy_tales",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/1597/pg1597.txt",
        "size_estimate": "~390 KB",
        "description": "Сказки Ганса Христиана Андерсена. Поэтичный стиль.",
        "description_en": "Hans Christian Andersen fairy tales. Poetic style.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_wizard_oz",
        "name": "The Wonderful Wizard of Oz",
        "name_ru": "Волшебник страны Оз",
        "category": "fairy_tales",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/55/pg55.txt",
        "size_estimate": "~230 KB",
        "description": "Баум. Классическая американская сказка. Простой живой язык.",
        "description_en": "Baum. Classic American fairy tale. Simple lively language.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_peter_pan",
        "name": "Peter Pan",
        "name_ru": "Питер Пэн",
        "category": "fairy_tales",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/16/pg16.txt",
        "size_estimate": "~230 KB",
        "description": "Барри. Классическая детская фантазия. Лёгкий стиль.",
        "description_en": "Barrie. Classic children's fantasy. Light style.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_wind_willows",
        "name": "The Wind in the Willows",
        "name_ru": "Ветер в ивах",
        "category": "fairy_tales",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/289/pg289.txt",
        "size_estimate": "~290 KB",
        "description": "Грэм. Уютная детская классика про животных.",
        "description_en": "Grahame. Cozy children's classic about animals.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_pollyanna",
        "name": "Pollyanna",
        "name_ru": "Поллианна",
        "category": "fairy_tales",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/1450/pg1450.txt",
        "size_estimate": "~370 KB",
        "description": "Элинор Портер. Оптимистичная детская классика.",
        "description_en": "Eleanor Porter. Optimistic children's classic.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_secret_garden",
        "name": "The Secret Garden",
        "name_ru": "Таинственный сад",
        "category": "fairy_tales",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/113/pg113.txt",
        "size_estimate": "~460 KB",
        "description": "Бёрнетт. Классика детской литературы.",
        "description_en": "Burnett. Children's literature classic.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_little_women",
        "name": "Little Women",
        "name_ru": "Маленькие женщины",
        "category": "fairy_tales",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/514/pg514.txt",
        "size_estimate": "~1 MB",
        "description": "Луиза Олкотт. Тёплая семейная история.",
        "description_en": "Louisa May Alcott. Warm family story.",
        "difficulty": "beginner",
    },
    # === PHILOSOPHY / HISTORY ===
    {
        "id": "gutenberg_republic",
        "name": "The Republic (Plato)",
        "name_ru": "Государство (Платон)",
        "category": "philosophy",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/1497/pg1497.txt",
        "size_estimate": "~650 KB",
        "description": "Философский диалог Платона. Диалоговый формат.",
        "description_en": "Plato's philosophical dialogue. Dialogue format.",
        "difficulty": "advanced",
    },
    {
        "id": "gutenberg_meditations",
        "name": "Meditations (Marcus Aurelius)",
        "name_ru": "Размышления (Марк Аврелий)",
        "category": "philosophy",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/2680/pg2680.txt",
        "size_estimate": "~200 KB",
        "description": "Философские записки римского императора. Компактный, афористичный.",
        "description_en": "Philosophical notes of a Roman emperor. Compact, aphoristic.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_art_of_war",
        "name": "The Art of War",
        "name_ru": "Искусство войны",
        "category": "philosophy",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/132/pg132.txt",
        "size_estimate": "~100 KB",
        "description": "Трактат Сунь-Цзы. Афористичный стиль.",
        "description_en": "Sun Tzu treatise. Aphoristic style.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_nicomachean_ethics",
        "name": "Nicomachean Ethics (Aristotle)",
        "name_ru": "Никомахова этика (Аристотель)",
        "category": "philosophy",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/8438/pg8438.txt",
        "size_estimate": "~470 KB",
        "description": "Аристотель. Классическая философия счастья и добродетели.",
        "description_en": "Aristotle. Classical philosophy of happiness and virtue.",
        "difficulty": "advanced",
    },
    {
        "id": "gutenberg_beyond_good_evil",
        "name": "Beyond Good and Evil (Nietzsche)",
        "name_ru": "По ту сторону добра и зла",
        "category": "philosophy",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/4363/pg4363.txt",
        "size_estimate": "~340 KB",
        "description": "Ницше. Провокационная философия, афоризмы.",
        "description_en": "Nietzsche. Provocative philosophy, aphorisms.",
        "difficulty": "advanced",
    },
    {
        "id": "gutenberg_prince",
        "name": "The Prince (Machiavelli)",
        "name_ru": "Государь (Макиавелли)",
        "category": "philosophy",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/1232/pg1232.txt",
        "size_estimate": "~220 KB",
        "description": "Макиавелли. Политическая философия.",
        "description_en": "Machiavelli. Political philosophy.",
        "difficulty": "intermediate",
    },
    {
        "id": "gutenberg_leviathan",
        "name": "Leviathan (Hobbes)",
        "name_ru": "Левиафан (Гоббс)",
        "category": "philosophy",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/3207/pg3207.txt",
        "size_estimate": "~1.1 MB",
        "description": "Томас Гоббс. Основы политической философии.",
        "description_en": "Thomas Hobbes. Foundations of political philosophy.",
        "difficulty": "advanced",
    },
    {
        "id": "gutenberg_utopia",
        "name": "Utopia (Thomas More)",
        "name_ru": "Утопия (Томас Мор)",
        "category": "philosophy",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/2130/pg2130.txt",
        "size_estimate": "~190 KB",
        "description": "Томас Мор. Классическая философская утопия.",
        "description_en": "Thomas More. Classic philosophical utopia.",
        "difficulty": "intermediate",
    },
    # === SCIENCE ===
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
        "description_en": "Darwin's scientific work. Scientific style, complex constructions.",
        "difficulty": "advanced",
    },
    {
        "id": "gutenberg_descent_man",
        "name": "The Descent of Man (Darwin)",
        "name_ru": "Происхождение человека (Дарвин)",
        "category": "science",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/2300/pg2300.txt",
        "size_estimate": "~1.3 MB",
        "description": "Дарвин. Эволюция человека. Объёмный научный датасет.",
        "description_en": "Darwin. Human evolution. Large scientific dataset.",
        "difficulty": "advanced",
    },
    {
        "id": "gutenberg_relativity",
        "name": "Relativity (Einstein)",
        "name_ru": "Теория относительности (Эйнштейн)",
        "category": "science",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/5001/pg5001.txt",
        "size_estimate": "~200 KB",
        "description": "Эйнштейн. Популярное изложение теории относительности.",
        "description_en": "Einstein. Popular exposition of the theory of relativity.",
        "difficulty": "advanced",
    },
    # === POETRY ===
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
        "description_en": "Shakespeare's 154 sonnets. Poetry, compact dataset.",
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
        "description_en": "Whitman poetry collection. Free verse, large volume.",
        "difficulty": "intermediate",
    },
    {
        "id": "gutenberg_iliad",
        "name": "The Iliad (Homer)",
        "name_ru": "Илиада (Гомер)",
        "category": "poetry",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/2199/pg2199.txt",
        "size_estimate": "~900 KB",
        "description": "Гомер. Эпическая поэзия, героический стиль.",
        "description_en": "Homer. Epic poetry, heroic style.",
        "difficulty": "advanced",
    },
    {
        "id": "gutenberg_odyssey",
        "name": "The Odyssey (Homer)",
        "name_ru": "Одиссея (Гомер)",
        "category": "poetry",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/1727/pg1727.txt",
        "size_estimate": "~750 KB",
        "description": "Гомер. Странствия Одиссея. Эпические приключения.",
        "description_en": "Homer. Odysseus' wanderings. Epic adventures.",
        "difficulty": "advanced",
    },
    {
        "id": "gutenberg_divine_comedy",
        "name": "The Divine Comedy (Dante)",
        "name_ru": "Божественная комедия (Данте)",
        "category": "poetry",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/8800/pg8800.txt",
        "size_estimate": "~650 KB",
        "description": "Данте. Величайшая поэма Средневековья.",
        "description_en": "Dante. The greatest poem of the Middle Ages.",
        "difficulty": "advanced",
    },
    {
        "id": "gutenberg_paradise_lost",
        "name": "Paradise Lost (Milton)",
        "name_ru": "Потерянный рай (Мильтон)",
        "category": "poetry",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/26/pg26.txt",
        "size_estimate": "~490 KB",
        "description": "Мильтон. Религиозная эпопея, поэтический шедевр.",
        "description_en": "Milton. Religious epic, poetic masterpiece.",
        "difficulty": "advanced",
    },
    {
        "id": "gutenberg_aeneid",
        "name": "The Aeneid (Virgil)",
        "name_ru": "Энеида (Вергилий)",
        "category": "poetry",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/228/pg228.txt",
        "size_estimate": "~510 KB",
        "description": "Вергилий. Основа латинской эпической поэзии.",
        "description_en": "Virgil. Foundation of Latin epic poetry.",
        "difficulty": "advanced",
    },
    # === SHAKESPEARE PLAYS ===
    {
        "id": "gutenberg_hamlet",
        "name": "Hamlet",
        "name_ru": "Гамлет",
        "category": "drama",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/1524/pg1524.txt",
        "size_estimate": "~180 KB",
        "description": "Шекспир. Величайшая трагедия, богатые монологи.",
        "description_en": "Shakespeare. Greatest tragedy, rich monologues.",
        "difficulty": "intermediate",
    },
    {
        "id": "gutenberg_macbeth",
        "name": "Macbeth",
        "name_ru": "Макбет",
        "category": "drama",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/1533/pg1533.txt",
        "size_estimate": "~100 KB",
        "description": "Шекспир. Трагедия, власть и безумие.",
        "description_en": "Shakespeare. Tragedy, power and madness.",
        "difficulty": "intermediate",
    },
    {
        "id": "gutenberg_king_lear",
        "name": "King Lear",
        "name_ru": "Король Лир",
        "category": "drama",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/1532/pg1532.txt",
        "size_estimate": "~150 KB",
        "description": "Шекспир. Трагедия старости и предательства.",
        "description_en": "Shakespeare. Tragedy of old age and betrayal.",
        "difficulty": "intermediate",
    },
    {
        "id": "gutenberg_othello",
        "name": "Othello",
        "name_ru": "Отелло",
        "category": "drama",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/1531/pg1531.txt",
        "size_estimate": "~160 KB",
        "description": "Шекспир. Трагедия ревности.",
        "description_en": "Shakespeare. Tragedy of jealousy.",
        "difficulty": "intermediate",
    },
    {
        "id": "gutenberg_romeo_juliet",
        "name": "Romeo and Juliet",
        "name_ru": "Ромео и Джульетта",
        "category": "drama",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/1112/pg1112.txt",
        "size_estimate": "~140 KB",
        "description": "Шекспир. Великая трагедия любви.",
        "description_en": "Shakespeare. Great tragedy of love.",
        "difficulty": "intermediate",
    },
    {
        "id": "gutenberg_midsummer",
        "name": "A Midsummer Night's Dream",
        "name_ru": "Сон в летнюю ночь",
        "category": "drama",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/1514/pg1514.txt",
        "size_estimate": "~110 KB",
        "description": "Шекспир. Лёгкая комедия, поэтический стиль.",
        "description_en": "Shakespeare. Light comedy, poetic style.",
        "difficulty": "intermediate",
    },
    {
        "id": "gutenberg_tempest",
        "name": "The Tempest",
        "name_ru": "Буря",
        "category": "drama",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/2235/pg2235.txt",
        "size_estimate": "~110 KB",
        "description": "Шекспир. Последняя пьеса. Магия и примирение.",
        "description_en": "Shakespeare. Last play. Magic and reconciliation.",
        "difficulty": "intermediate",
    },
    # === MYTHOLOGY / ANCIENT ===
    {
        "id": "gutenberg_bulfinch_mythology",
        "name": "Bulfinch's Mythology",
        "name_ru": "Мифология Булфинча",
        "category": "mythology",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/4928/pg4928.txt",
        "size_estimate": "~1.3 MB",
        "description": "Классическое собрание мифов. Большой датасет, богатый стиль.",
        "description_en": "Classic collection of myths. Large dataset, rich style.",
        "difficulty": "intermediate",
    },
    {
        "id": "gutenberg_arabian_nights",
        "name": "One Thousand and One Nights",
        "name_ru": "Тысяча и одна ночь",
        "category": "mythology",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/128/pg128.txt",
        "size_estimate": "~600 KB",
        "description": "Арабские сказки. Сказочный восточный стиль.",
        "description_en": "Arabian tales. Magical Eastern style.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_aesop",
        "name": "Aesop's Fables",
        "name_ru": "Басни Эзопа",
        "category": "mythology",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/11339/pg11339.txt",
        "size_estimate": "~150 KB",
        "description": "Эзоп. Короткие поучительные истории. Отличный датасет для коротких текстов.",
        "description_en": "Aesop. Short moral stories. Great dataset for short texts.",
        "difficulty": "beginner",
    },
    # === BIOGRAPHY / AUTOBIOGRAPHY ===
    {
        "id": "gutenberg_autobiography_franklin",
        "name": "Autobiography of Benjamin Franklin",
        "name_ru": "Автобиография Бенджамина Франклина",
        "category": "biography",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/148/pg148.txt",
        "size_estimate": "~360 KB",
        "description": "Франклин. Яркая личная история, живой стиль.",
        "description_en": "Franklin. Vivid personal story, lively style.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_narrative_douglass",
        "name": "Narrative of the Life of Frederick Douglass",
        "name_ru": "Рассказ о жизни Фредерика Дугласса",
        "category": "biography",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/23/pg23.txt",
        "size_estimate": "~200 KB",
        "description": "Дуглас. Мемуары. Исторически важный документ.",
        "description_en": "Douglass. Memoirs. Historically important document.",
        "difficulty": "beginner",
    },
    {
        "id": "gutenberg_up_from_slavery",
        "name": "Up From Slavery",
        "name_ru": "Из рабства",
        "category": "biography",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/2376/pg2376.txt",
        "size_estimate": "~380 KB",
        "description": "Букер Вашингтон. Автобиография, борьба за образование.",
        "description_en": "Booker T. Washington. Autobiography, struggle for education.",
        "difficulty": "beginner",
    },
    # === SPIRITUAL / RELIGIOUS ===
    {
        "id": "gutenberg_bible_kjv",
        "name": "The King James Bible",
        "name_ru": "Библия (KJV)",
        "category": "spiritual",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/10/pg10.txt",
        "size_estimate": "~4.4 MB",
        "description": "Библия. Крупнейший датасет. Архаичный английский стиль.",
        "description_en": "The Bible. Largest dataset. Archaic English style.",
        "difficulty": "advanced",
    },
    {
        "id": "gutenberg_koran",
        "name": "The Koran",
        "name_ru": "Коран (EN)",
        "category": "spiritual",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/7440/pg7440.txt",
        "size_estimate": "~900 KB",
        "description": "Коран (перевод на английский). Ритмическая проза.",
        "description_en": "The Koran (English translation). Rhythmic prose.",
        "difficulty": "advanced",
    },
    # === HISTORY ===
    {
        "id": "gutenberg_decline_fall",
        "name": "The History of the Decline and Fall of the Roman Empire Vol 1",
        "name_ru": "История упадка Рима (Гиббон) T.1",
        "category": "history",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/731/pg731.txt",
        "size_estimate": "~1.2 MB",
        "description": "Гиббон. Классическая историческая проза. Большой датасет.",
        "description_en": "Gibbon. Classic historical prose. Large dataset.",
        "difficulty": "advanced",
    },
    {
        "id": "gutenberg_histories_herodotus",
        "name": "The Histories (Herodotus)",
        "name_ru": "История (Геродот)",
        "category": "history",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/2456/pg2456.txt",
        "size_estimate": "~1.1 MB",
        "description": "Геродот. Первый историк. Богатый нарратив.",
        "description_en": "Herodotus. The first historian. Rich narrative.",
        "difficulty": "advanced",
    },
    {
        "id": "gutenberg_plutarch_lives",
        "name": "Plutarch's Lives",
        "name_ru": "Сравнительные жизнеописания (Плутарх)",
        "category": "history",
        "language": "en",
        "source": "gutenberg",
        "url": "https://www.gutenberg.org/cache/epub/674/pg674.txt",
        "size_estimate": "~1.8 MB",
        "description": "Плутарх. Биографии великих. Огромный исторический датасет.",
        "description_en": "Plutarch. Biographies of the great. Huge historical dataset.",
        "difficulty": "advanced",
    },
    # === LITERATURE RU === (источник: github.com/d0rj/RusLit)
    {
        "id": "ruslit_karenina",
        "name": "Анна Каренина",
        "name_ru": "Анна Каренина",
        "category": "literature_ru",
        "language": "ru",
        "source": "github_ruslit",
        "url": "https://raw.githubusercontent.com/d0rj/RusLit/master/prose/Tolstoy/%D0%90%D0%BD%D0%BD%D0%B0%20%D0%9A%D0%B0%D1%80%D0%B5%D0%BD%D0%B8%D0%BD%D0%B0.txt",
        "size_estimate": "~1.8 MB",
        "description": "Роман Льва Толстого. Огромный датасет, классический русский стиль.",
        "description_en": "Leo Tolstoy novel. Huge dataset, classic Russian style.",
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
        "description": "Первый том масштабного романа Толстого.",
        "description_en": "First volume of Tolstoy's epic novel.",
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
        "description": "Второй том масштабного романа Толстого.",
        "description_en": "Second volume of Tolstoy's epic novel.",
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
        "description": "Третий том масштабного романа Толстого.",
        "description_en": "Third volume of Tolstoy's epic novel.",
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
        "description": "Четвёртый том масштабного романа Толстого.",
        "description_en": "Fourth volume of Tolstoy's epic novel.",
        "difficulty": "advanced",
    },
    {
        "id": "ruslit_poor_folk",
        "name": "Бедные люди",
        "name_ru": "Бедные люди",
        "category": "literature_ru",
        "language": "ru",
        "source": "github_ruslit",
        "url": "https://raw.githubusercontent.com/d0rj/RusLit/master/prose/Dostoevsky/%D0%91%D0%B5%D0%B4%D0%BD%D1%8B%D0%B5%20%D0%BB%D1%8E%D0%B4%D0%B8.txt",
        "size_estimate": "~400 KB",
        "description": "Первый роман Достоевского. Эпистолярный стиль.",
        "description_en": "Dostoevsky's first novel. Epistolary style.",
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
        "description": "Последний великий роман Достоевского. Огромный датасет.",
        "description_en": "Dostoevsky's last great novel. Huge dataset.",
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
        "description": "Роман Достоевского. Психологическая глубина.",
        "description_en": "Dostoevsky novel. Psychological depth.",
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
        "description": "Роман Достоевского. Политическая проза.",
        "description_en": "Dostoevsky novel. Political prose.",
        "difficulty": "advanced",
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
        "description_en": "Gogol's poem-novel. Satirical style, vivid Russian language.",
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
        "description": "Повесть Гоголя. Компактный датасет.",
        "description_en": "Gogol novella. Compact dataset.",
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
        "description_en": "Gogol novella. Epic style, vivid descriptions.",
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
        "description": "Последний роман Толстого. Социальная тематика.",
        "description_en": "Tolstoy's last novel. Social themes.",
        "difficulty": "intermediate",
    },
    # === POETRY RU ===
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
        "description_en": "Pushkin's novel in verse. Poetic style, rhyme.",
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
        "description_en": "Pushkin poem. Fairy-tale style, beautiful Russian language.",
        "difficulty": "beginner",
    },
    # === CODE ===
    {
        "id": "code_python_snippets",
        "name": "Python Code Examples",
        "name_ru": "Примеры кода Python",
        "category": "code",
        "language": "en",
        "source": "url",
        "url": "https://raw.githubusercontent.com/TheAlgorithms/Python/master/DIRECTORY.md",
        "size_estimate": "~50 KB",
        "description": "Список алгоритмов на Python. Маленький но полезный.",
        "description_en": "List of Python algorithms. Small but useful.",
        "difficulty": "beginner",
    },
]


# Категории с отображаемыми именами
CATEGORIES = {
    "literature_en": {"name": "Literature (EN)", "name_ru": "Классика (EN)", "icon": "📚"},
    "literature_ru": {"name": "Literature (RU)", "name_ru": "Классика (RU)", "icon": "📖"},
    "detective": {"name": "Detective", "name_ru": "Детективы", "icon": "🔍"},
    "horror": {"name": "Horror / Gothic", "name_ru": "Ужасы / Готика", "icon": "🦇"},
    "scifi": {"name": "Science Fiction", "name_ru": "Фантастика", "icon": "🚀"},
    "adventure": {"name": "Adventure", "name_ru": "Приключения", "icon": "⚔️"},
    "fairy_tales": {"name": "Fairy Tales", "name_ru": "Сказки", "icon": "🧚"},
    "drama": {"name": "Drama / Theatre", "name_ru": "Пьесы / Театр", "icon": "🎭"},
    "poetry": {"name": "Poetry", "name_ru": "Поэзия", "icon": "✍️"},
    "mythology": {"name": "Mythology", "name_ru": "Мифология", "icon": "⚡"},
    "philosophy": {"name": "Philosophy", "name_ru": "Философия", "icon": "🧠"},
    "science": {"name": "Science", "name_ru": "Наука", "icon": "🔬"},
    "history": {"name": "History", "name_ru": "История", "icon": "🏛️"},
    "biography": {"name": "Biography", "name_ru": "Биография", "icon": "👤"},
    "spiritual": {"name": "Spiritual / Religion", "name_ru": "Религия / Духовное", "icon": "🕊️"},
    "code": {"name": "Code", "name_ru": "Код", "icon": "💻"},
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
        import gzip as _gzip
        # Распаковать gzip если данные сжаты
        if data[:2] == b'\x1f\x8b':
            try:
                data = _gzip.decompress(data)
            except Exception:
                pass
        # Убрать UTF-8 BOM
        if data[:3] == b'\xef\xbb\xbf':
            data = data[3:]
        for enc in ['utf-8', 'cp1251', 'latin-1']:
            try:
                return data.decode(enc)
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
