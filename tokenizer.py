import re
import json
import pickle
from collections import Counter
from pathlib import Path


class SimpleTokenizer:
    def __init__(self, vocab_size=8000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }
        self.token_to_id.update(self.special_tokens)
        self.id_to_token.update({v: k for k, v in self.special_tokens.items()})
        
    def train(self, texts, preserve_existing=False):
        """
        Обучить токенизатор на текстах
        
        Args:
            texts: список текстов для обучения
            preserve_existing: если True, сохранить существующие ID слов (для добавления новых датасетов)
                              если False, пересоздать словарь с нуля (по умолчанию)
        """
        all_tokens = []
        for text in texts:
            tokens = self._tokenize_text(text)
            all_tokens.extend(tokens)
        
        counter = Counter(all_tokens)
        
        if preserve_existing:
            # Инкрементальное обучение: добавляем только новые слова
            # Старые ID сохраняются!
            existing_tokens = set(self.token_to_id.keys())
            new_tokens = set(counter.keys()) - existing_tokens
            
            # Находим следующий свободный ID
            current_id = max(self.token_to_id.values()) + 1 if self.token_to_id else len(self.special_tokens)
            
            # Добавляем только новые слова
            for token in new_tokens:
                if current_id >= self.vocab_size:
                    break  # Достигли лимита словаря
                self.token_to_id[token] = current_id
                self.id_to_token[current_id] = token
                current_id += 1
            
            print(f"   Added {len(new_tokens)} new tokens (preserving {len(existing_tokens)} existing)")
        else:
            # Полное переобучение: создаём словарь с нуля
            most_common = counter.most_common(self.vocab_size - len(self.special_tokens))
            
            # Очищаем старый словарь (кроме спецтокенов)
            self.token_to_id = dict(self.special_tokens)
            self.id_to_token = {v: k for k, v in self.special_tokens.items()}
            
            current_id = len(self.special_tokens)
            for token, _ in most_common:
                if token not in self.token_to_id:
                    self.token_to_id[token] = current_id
                    self.id_to_token[current_id] = token
                    current_id += 1
        
        return self
    
    def _tokenize_text(self, text):
        text = text.lower()
        tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
        return tokens
    
    def encode(self, text):
        tokens = self._tokenize_text(text)
        ids = [self.token_to_id.get(token, self.special_tokens['<UNK>']) for token in tokens]
        return ids
    
    def decode(self, ids):
        tokens = [self.id_to_token.get(id, '<UNK>') for id in ids]
        text = ' '.join(tokens)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text
    
    def save(self, path, trained_on_datasets=None):
        data = {
            'vocab_size': self.vocab_size,
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'special_tokens': self.special_tokens,
            'trained_on_datasets': trained_on_datasets or []  # Список датасетов
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = cls(data['vocab_size'])
        tokenizer.token_to_id = data['token_to_id']
        tokenizer.id_to_token = data['id_to_token']
        tokenizer.special_tokens = data['special_tokens']
        tokenizer.trained_on_datasets = data.get('trained_on_datasets', [])
        return tokenizer
    
    def get_trained_datasets(self):
        """Получить список датасетов на которых обучался токенизатор"""
        return getattr(self, 'trained_on_datasets', [])
    
    def __len__(self):
        return len(self.token_to_id)
