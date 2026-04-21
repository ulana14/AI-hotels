# -*- coding: utf-8 -*-
"""
ФИНАЛЬНАЯ РАБОЧАЯ ВЕРСИЯ КЛАССИФИКАТОРА
"""

import pandas as pd
import numpy as np
import torch
import random
import os
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Отключаем Arrow-типы в pandas
pd.options.mode.string_storage = "python"

# ==================== 1. КЛАСС ДАТАСЕТА ====================
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = list(texts)  # принудительно в list
        self.labels = list(labels)  # принудительно в list
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ==================== 2. АУГМЕНТАЦИЯ ====================
class TextAugmenter:
    def __init__(self):
        self.synonyms = {
            'отличный': ['превосходный', 'замечательный', 'великолепный'],
            'хороший': ['неплохой', 'достойный', 'качественный'],
            'чистый': ['опрятный', 'аккуратный', 'ухоженный'],
            'плохой': ['ужасный', 'отвратительный', 'некачественный'],
            'грязный': ['загрязненный', 'неопрятный', 'запущенный'],
            'ужасный': ['кошмарный', 'чудовищный', 'отвратительный'],
        }
        
    def synonym_replacement(self, text, probability=0.3):
        words = text.split()
        new_words = []
        for word in words:
            word_lower = word.lower().strip('.,!?()"-')
            if word_lower in self.synonyms and random.random() < probability:
                synonym = random.choice(self.synonyms[word_lower])
                if word[0].isupper():
                    synonym = synonym.capitalize()
                new_words.append(synonym)
            else:
                new_words.append(word)
        return ' '.join(new_words)
    
    def augment(self, text):
        return self.synonym_replacement(text, probability=0.3)

# ==================== 3. ПОДГОТОВКА ДАННЫХ ====================
def prepare_data(file_path):
    print("\n" + "="*60)
    print("ПОДГОТОВКА ДАННЫХ")
    print("="*60)
    
    # Загрузка с принудительным str типом
    df = pd.read_csv(file_path, dtype={'text': str, 'user_rating': int})
    df = df.dropna(subset=['text', 'user_rating'])
    print(f"Оригинальный датасет: {len(df)} отзывов")
    
    # Добавляем негативные отзывы
    negative_reviews = [
        {"city": "Москва", "hotel": "Тест", "user_rating": 2, "text": "Ужасный сервис! Грязный номер.", "landmarks": "Кремль"},
        {"city": "Москва", "hotel": "Тест", "user_rating": 1, "text": "Тараканы, воняет сыростью!", "landmarks": "Кремль"},
        {"city": "Санкт-Петербург", "hotel": "Тест", "user_rating": 2, "text": "Ждали заселения 2 часа.", "landmarks": "Эрмитаж"},
        {"city": "Санкт-Петербург", "hotel": "Тест", "user_rating": 1, "text": "Клопы в кровати!", "landmarks": "Эрмитаж"},
        {"city": "Казань", "hotel": "Тест", "user_rating": 2, "text": "Старые номера, мебель разваливается.", "landmarks": "Кремль"},
        {"city": "Сочи", "hotel": "Тест", "user_rating": 2, "text": "Огромный муравейник, грязно.", "landmarks": "Парк"},
        {"city": "Владивосток", "hotel": "Тест", "user_rating": 1, "text": "Номер-коробка, вид на помойку.", "landmarks": "Мост"},
        {"city": "Иркутск", "hotel": "Тест", "user_rating": 1, "text": "Шум, грязь, сомнительные личности.", "landmarks": "Байкал"},
        {"city": "Новосибирск", "hotel": "Тест", "user_rating": 2, "text": "Вокзальный отель, шум, грязь.", "landmarks": "Театр"},
        {"city": "Нижний Новгород", "hotel": "Тест", "user_rating": 1, "text": "Советский союз, текущие краны.", "landmarks": "Кремль"},
    ]
    
    df = pd.concat([df, pd.DataFrame(negative_reviews)], ignore_index=True)
    print(f"Добавлено негативных отзывов: {len(negative_reviews)}")
    
    # Аугментация
    augmenter = TextAugmenter()
    augmented_rows = []
    for idx, row in df.iterrows():
        aug_text = augmenter.augment(row['text'])
        if aug_text != row['text']:
            new_row = row.copy()
            new_row['text'] = aug_text
            augmented_rows.append(new_row)
    
    df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
    print(f"Размер после аугментации: {len(df)}")
    
    # Бинарная классификация
    df['label'] = (df['user_rating'] >= 8).astype(int)
    
    print(f"\nРаспределение классов:")
    print(f"  Не рекомендую (1-7): {(df['label']==0).sum()} ({(df['label']==0).sum()/len(df)*100:.1f}%)")
    print(f"  Рекомендую (8-10): {(df['label']==1).sum()} ({(df['label']==1).sum()/len(df)*100:.1f}%)")
    
    return df

# ==================== 4. ФУНКЦИЯ ПРЕДСКАЗАНИЯ ====================
def predict_review(text, model, tokenizer):
    class_names = ['❌ Не рекомендую', '✅ Рекомендую']
    model.eval()
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    return {'class': class_names[pred_class], 'confidence': probs[0][pred_class].item()}

# ==================== 5. MAIN ====================
def main():
    print("="*60)
    print("КЛАССИФИКАЦИЯ ОТЗЫВОВ НА ОТЕЛИ")
    print("="*60)
    
    # 1. Данные
    df = prepare_data('all_cities_raw.csv')
    
    # 2. Разделение - ПРЕОБРАЗУЕМ В ОБЫЧНЫЕ СПИСКИ
    print("\n[2] Разделение на train/test...")
    
    texts = df['text'].astype(str).tolist()  # ← ВОТ КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ
    labels = df['label'].astype(int).tolist()  # ← ВОТ КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ
    
    # Разделение через списки (работает 100%)
    from sklearn.model_selection import train_test_split
    
    # Сначала разделяем на train+val и test
    texts_train_val, texts_test, labels_train_val, labels_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Затем train_val на train и val
    texts_train, texts_val, labels_train, labels_val = train_test_split(
        texts_train_val, labels_train_val, test_size=0.1, random_state=42, stratify=labels_train_val
    )
    
    print(f"Train: {len(texts_train)}")
    print(f"Validation: {len(texts_val)}")
    print(f"Test: {len(texts_test)}")
    
    # 3. Загрузка модели
    print("\n[3] Загрузка модели...")
    model_name = 'cointegrated/rubert-tiny2'
    
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=2, ignore_mismatched_sizes=True
        )
        print(f"✅ Модель загружена")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return
    
    # 4. Датасеты
    print("\n[4] Подготовка датасетов...")
    train_dataset = ReviewDataset(texts_train, labels_train, tokenizer)
    val_dataset = ReviewDataset(texts_val, labels_val, tokenizer)
    test_dataset = ReviewDataset(texts_test, labels_test, tokenizer)
    
    # 5. Обучение
    print("\n[5] Настройка обучения...")
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        report_to='none',
        logging_steps=10,
    )
    
    def compute_metrics(pred):
        acc = accuracy_score(pred.label_ids, pred.predictions.argmax(-1))
        return {'accuracy': acc}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("\n[6] Обучение...")
    trainer.train()
    
    # 6. Оценка
    print("\n[7] Оценка...")
    preds = trainer.predict(test_dataset)
    y_pred = np.argmax(preds.predictions, axis=1)
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ")
    print("="*60)
    print(f"Accuracy: {accuracy_score(labels_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(labels_test, y_pred, target_names=['Не рекомендую', 'Рекомендую']))
    
    # 7. Сохранение
    trainer.save_model("./hotel_model")
    tokenizer.save_pretrained("./hotel_model")
    print("\n✅ Модель сохранена в ./hotel_model")
    
    # 8. Примеры
    print("\n" + "="*60)
    print("ПРИМЕРЫ")
    print("="*60)
    examples = [
        "Отель великолепный! Чисто, отличный сервис!",
        "Ужасный отель, грязно, персонал хамит.",
        "Нормальный отель за свои деньги.",
    ]
    for ex in examples:
        res = predict_review(ex, model, tokenizer)
        print(f"\n📝 {ex}")
        print(f"🏷️ {res['class']} ({res['confidence']:.2%})")

if __name__ == "__main__":
    main()
