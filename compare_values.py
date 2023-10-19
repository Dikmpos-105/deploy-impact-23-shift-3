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

api_candidates = 'https://django-backend-shift-enter-u53fbnjraa-oe.a.run.app/api/values/'
api_companies = 'https://django-backend-shift-enter-u53fbnjraa-oe.a.run.app/api/values/'

try:
    response = requests.get(api_candidates)

    # Check if request is successful
    if response.status_code == 200:
        data = response.json()

        # Request values from the API for candidates
        candidate_values = [item['list_values_name'] for item in data]

    else:
        print(f"Request for candidates failed, code: {response.status_code}")

except Exception as e:
    print(f"Request for candidates failed: {str(e)}")

try:
    response = requests.get(api_companies)

    # Check if request is successful
    if response.status_code == 200:
        data = response.json()

        # Request values from the API for companies
        company_values = [item['list_values_name'] for item in data]

    else:
        print(f"Request for companies failed, code: {response.status_code}")

except Exception as e:
    print(f"Request for companies failed: {str(e)}")

# List to store similarity scores
similarities = []

try:
    # Iterate through the candidate_values
    for candidate_val in candidate_values:
        print(f"Processing candidate value: {candidate_val}")

        try:
            # Tokenize and encode the candidate value
            candidates_token = tokenizer(candidate_val, padding=True, truncation=True, return_tensors="pt")

            # Iterate through the company_values
            for company_val in company_values:
                print(f"Processing company value: {company_val}")

                # Tokenize and encode the company value
                company_token = tokenizer(company_val, padding=True, truncation=True, return_tensors="pt")

                # Get BERT embeddings for the sentences
                with torch.no_grad():
                    candidate_output = model(**candidates_token)
                    company_output = model(**company_token)

                # Get sentence embeddings (using the output from the embedding layer)
                cv_embeddings = candidate_output.last_hidden_state.mean(dim=1)
                job_description_embeddings = company_output.last_hidden_state.mean(dim=1)

                # Calculate cosine similarity between sentences
                similarity = cosine_similarity(cv_embeddings, job_description_embeddings)

                # Add the similarity score and the candidate and company values to the list
                similarities.append((candidate_val, company_val, similarity[0][0]))

        except Exception as e:
            print(f"Error processing candidate value {candidate_val} or company value {company_val}: {str(e)}")

    # Sort candidates by similarity score (from most similar to least similar)
    similarities.sort(key=lambda x: x[2], reverse=True)

    # Display the sorted list of candidates and their best company value matches
    for candidate_val, company_val, score in similarities:
        print(f"Candidate Value: {candidate_val}, Company Value: {company_val}, Similarity Score: {score}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
