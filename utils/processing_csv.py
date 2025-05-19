import pandas as pd
import os
import re

VECTOR_STORE_DIR = "vector_store"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

def clean_sentences(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def process_info_csv(df):
    for index, row in df.iterrows():
        exercise_name =  row[1]
        exercise_description = row[2]
        exercise_type = row[3]
        exercise_equipment = row[4]
        exercise_level = row[5]

    return

