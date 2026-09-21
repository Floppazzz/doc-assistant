import pandas as pd
from datasets import load_dataset
import os

# Создаем папку для данных
os.makedirs("data/processed", exist_ok=True)

print("Загрузка датасета с Hugging Face...")
dataset = load_dataset("zloelias/lenta-ru")
df = dataset["train"].to_pandas()

# Оставляем только нужные колонки
df = df[["text", "topic"]]

# Удаляем пустые строки
df = df.dropna()

print(f"Всего строк до обработки: {len(df)}")

# Перемешиваем
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Обрезаем до 1000 строк
df = df.head(1000)

# Сохраняем
df.to_csv("data/processed/clean.csv", index=False, encoding="utf-8")

print(f"✅ Датасет сохранен в 'data/processed/clean.csv'")
print(f"📊 Строк: {len(df)}")
print(f"📋 Распределение тем:\n{df['topic'].value_counts()}")
