import os
import glob
import sys
import torch
import json
import pandas as pd
import multiprocessing
import requests
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


# Load the pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

#Replace this part to to the connexion with the DB
cv_folder_path = "C:/Users/PatriciaWintrebert/Womenplusplus/extractenv/app/data/resume/"  
job_description_csv = "IT_entry_swiss_jobs_200.csv"  
job_descriptions_df = pd.read_csv(job_description_csv)

# Concatenation of columns to create free text to analyze
job_descriptions_df["description"] = job_descriptions_df["job_title"] + " " + job_descriptions_df["skills"]

similarities = []

try:

    for cv_file in glob.glob(os.path.join(cv_folder_path, "*.txt")):
        print(f"Processing CV file: {cv_file}")

        try:
            with open(cv_file, "r", encoding="utf-8") as file:
                cv_text = file.read()

            # Tokenize and encoding
            cv_tokens = tokenizer(cv_text, padding=True, truncation=True, return_tensors="pt")

            for index, job_description_row in job_descriptions_df.iterrows():
                job_description_text = job_description_row["description"]

                job_description_tokens = tokenizer(job_description_text, padding=True, truncation=True, return_tensors="pt")

                # Get embeddings BERT of phrases
                with torch.no_grad():
                    cv_outputs = model(**cv_tokens)
                    job_description_outputs = model(**job_description_tokens)

                # Get embedding
                cv_embeddings = cv_outputs.last_hidden_state.mean(dim=1)
                job_description_embeddings = job_description_outputs.last_hidden_state.mean(dim=1)

                # Calculate similarity
                similarity = cosine_similarity(cv_embeddings, job_description_embeddings)

                # Add similarity score
                similarities.append((os.path.basename(cv_file), job_description_row["job_title"], similarity[0][0]))

        except Exception as e:
            print(f"Error processing CV file {cv_file}: {str(e)}")

    # Sort candidates by similarity
    similarities.sort(key=lambda x: x[2], reverse=True)

    # Display candidates and similarity
    for cv_file, job_title, score in similarities:
        print(f"CV File: {cv_file}, Job Title: {job_title}, Similarity Score: {score}")

except Exception as e:
    print(f"An error occurred: {str(e)}")