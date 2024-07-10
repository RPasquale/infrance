import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import PyPDF2
import docx
import pandas as pd
import faiss
import numpy as np
from werkzeug.utils import secure_filename
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import json
import requests
from bs4 import BeautifulSoup
import torch
import logging

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

PROMPTS_FILE = 'prompts.json'
PROMPT_COMBINATIONS_FILE = 'prompt_combinations.json'

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def save_prompts(prompts):
    with open(PROMPTS_FILE, 'w') as file:
        json.dump(prompts, file)

def load_prompts():
    if os.path.exists(PROMPTS_FILE):
        with open(PROMPTS_FILE, 'r') as file:
            return json.load(file)
    return {"global_system_prompt": "", "global_user_prompt": "", "local_system_prompt": "", "local_user_prompt": ""}

def save_prompt_combinations(prompt_combinations):
    with open(PROMPT_COMBINATIONS_FILE, 'w') as file:
        json.dump(prompt_combinations, file)

def load_prompt_combinations():
    if os.path.exists(PROMPT_COMBINATIONS_FILE):
        with open(PROMPT_COMBINATIONS_FILE, 'r') as file:
            return json.load(file)
    return []

@app.route('/save_prompts', methods=['POST'])
def save_prompts_route():
    prompts = request.json
    save_prompts(prompts)
    return jsonify({"message": "Prompts saved successfully"}), 200

@app.route('/load_prompts', methods=['GET'])
def load_prompts_route():
    prompts = load_prompts()
    return jsonify(prompts), 200

@app.route('/save_prompt_combination', methods=['POST'])
def save_prompt_combination_route():
    prompt_combination = request.json
    prompt_combinations = load_prompt_combinations()
    prompt_combinations.append(prompt_combination)
    save_prompt_combinations(prompt_combinations)
    return jsonify({"message": "Prompt combination saved successfully"}), 200

@app.route('/load_saved_prompt_combinations', methods=['GET'])
def load_saved_prompt_combinations_route():
    prompt_combinations = load_prompt_combinations()
    return jsonify(prompt_combinations), 200

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist('files')
    uploaded_files = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            uploaded_files.append(filename)

    return jsonify({"message": f"{len(uploaded_files)} files uploaded successfully", "files": uploaded_files}), 200

def extract_text_from_file(file_path):
    _, extension = os.path.splitext(file_path)
    text = ""
    
    if extension == '.pdf':
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    elif extension == '.docx':
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    elif extension == '.txt':
        with open(file_path, 'r') as file:
            text = file.read()
    elif extension == '.xlsx':
        df = pd.read_excel(file_path)
        text = df.to_string(index=False)
    return text

def embed_text(texts, model, hf_model_name=None, hf_api_key=None):
    if model == 'openai':
        response = openai.Embedding.create(
            input=texts,
            model="text-embedding-ada-002"
        )
        embeddings = np.array([data['embedding'] for data in response['data']])
    elif model == 'huggingface' and hf_model_name:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, use_auth_token=hf_api_key)
        model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_name, use_auth_token=hf_api_key)
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    else:
        raise ValueError("Invalid model or missing Hugging Face model name.")
    return embeddings

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_faiss_index(index, query_embedding, top_k=10):
    D, I = index.search(query_embedding, top_k)
    return I

def chunk_text(text, max_tokens=512):
    words = text.split()
    chunks = [' '.join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]
    return chunks

def summarize_text(text, model, hf_model_name=None, hf_api_key=None, max_tokens=512):
    chunks = chunk_text(text, max_tokens)
    summaries = []
    for chunk in chunks:
        if model == 'openai':
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": f"Summarize the following text:\n\n{chunk}"}],
                max_tokens=200
            )
            summaries.append(response.choices[0].message['content'].strip())
        elif model == 'huggingface' and hf_model_name:
            hf_pipeline = pipeline('summarization', model=hf_model_name, use_auth_token=hf_api_key)
            summary = hf_pipeline(chunk, max_new_tokens=200, truncation=True)
            summaries.append(summary[0]['summary_text'])
        else:
            raise ValueError("Invalid model or missing Hugging Face model name.")
    return ' '.join(summaries)

def perform_web_search(query):
    search_url = f"https://www.google.com/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    search_results = []
    for g in soup.find_all('div', class_='BVG0Nb'):
        result_title = g.find('h3').text if g.find('h3') else "No title"
        result_link = g.find('a')['href'] if g.find('a') else "No link"
        result_snippet = g.find('div', class_='IsZvec').text if g.find('div', class_='IsZvec') else "No snippet"
        search_results.append({
            "title": result_title,
            "link": result_link,
            "snippet": result_snippet
        })
    return search_results

@app.route('/web_search', methods=['POST'])
def web_search():
    data = request.json
    query = data.get('query')
    uploaded_files = data.get('uploadedFiles', [])
    model = data.get('model')
    hf_model_name = data.get('hfModelName')
    hf_api_key = data.get('hfApiKey')
    
    # Perform the web search
    search_results = perform_web_search(query)
    
    # Process uploaded files
    file_contents = []
    for filename in uploaded_files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file_contents.append(extract_text_from_file(file_path))
    
    # Summarize large document contents
    summarized_contents = [summarize_text(content, model, hf_model_name, hf_api_key) for content in file_contents]
    
    # Create embeddings for the summarized contents
    embeddings = embed_text(summarized_contents, model, hf_model_name, hf_api_key)
    index = create_faiss_index(embeddings)
    
    # Embed the search results
    search_texts = [result['snippet'] for result in search_results]
    search_embeddings = embed_text(search_texts, model, hf_model_name, hf_api_key)
    
    # Add search results to the index
    index.add(search_embeddings)
    
    # Embed the user query and search the index for relevant documents
    query_embedding = embed_text([query], model, hf_model_name, hf_api_key)[0].reshape(1, -1)
    relevant_indices = search_faiss_index(index, query_embedding)
    
    # Add relevant documents to the prompt
    relevant_docs = [summarized_contents[i] for i in relevant_indices[0] if i < len(summarized_contents)]
    relevant_search_results = [search_results[i - len(summarized_contents)] for i in relevant_indices[0] if i >= len(summarized_contents)]
    
    combined_prompt = f"User query: {query}\n\nRelevant document contents:\n" + "\n".join(relevant_docs)
    combined_prompt += "\n\nRelevant search results:\n" + "\n".join([f"Title: {result['title']}\nLink: {result['link']}\nSnippet: {result['snippet']}\n" for result in relevant_search_results])
    
    # Generate the response using the combined prompt
    if model == 'openai':
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": combined_prompt}, {"role": "user", "content": query}]
        )
        final_response = response.choices[0].message['content']
    elif model == 'huggingface' and hf_model_name:
        hf_pipeline = pipeline('text-generation', model=hf_model_name, use_auth_token=hf_api_key)
        completion = hf_pipeline(combined_prompt, max_new_tokens=500)
        final_response = completion[0]['generated_text']
    else:
        raise ValueError("Invalid model or missing Hugging Face model name.")

    return jsonify({"response": final_response}), 200

@app.route('/process', methods=['POST'])
def process_request():
    data = request.json
    model = data.get('model')
    user_input = data.get('userInput')
    global_system_prompt = data.get('globalSystemPrompt')
    global_user_prompt = data.get('globalUserPrompt')
    local_system_prompt = data.get('localSystemPrompt')
    local_user_prompt = data.get('localUserPrompt')
    variables = data.get('variables', {})
    tools = data.get('tools', {})
    selected_tool = data.get('selectedTool')
    web_search_query = data.get('webSearchQuery')
    hf_model_name = data.get('hfModelName') if model == 'huggingface' else None
    hf_api_key = data.get('hfApiKey') if model == 'huggingface' else None

    # Process uploaded files
    uploaded_files = data.get('uploadedFiles', [])
    file_contents = []
    for filename in uploaded_files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file_contents.append(extract_text_from_file(file_path))

    # Summarize large document contents
    summarized_contents = [summarize_text(content, model, hf_model_name, hf_api_key) for content in file_contents]

    # Create embeddings for the summarized contents
    embeddings = embed_text(summarized_contents, model, hf_model_name, hf_api_key)
    index = create_faiss_index(embeddings)

    # Perform web search if the selected tool is web search
    search_results = []
    if selected_tool == 'webSearch' and web_search_query:
        search_results = perform_web_search(web_search_query)
        search_texts = [result['snippet'] for result in search_results]
        search_embeddings = embed_text(search_texts, model, hf_model_name, hf_api_key)
        index.add(search_embeddings)

    # Embed the user input and search the index for relevant documents
    user_input_embedding = embed_text([user_input], model, hf_model_name, hf_api_key)[0].reshape(1, -1)
    relevant_indices = search_faiss_index(index, user_input_embedding)

    # Add relevant documents to the prompt
    relevant_docs = [summarized_contents[i] for i in relevant_indices[0] if i < len(summarized_contents)]
    relevant_search_results = [search_results[i - len(summarized_contents)] for i in relevant_indices[0] if i >= len(summarized_contents)]

    combined_prompt = f"{global_system_prompt}\n{global_user_prompt}\n{local_system_prompt}\n{local_user_prompt}\n"
    combined_prompt += "Relevant document contents:\n" + "\n".join(relevant_docs)
    combined_prompt += "\n\nRelevant search results:\n" + "\n".join([f"Title: {result['title']}\nLink: {result['link']}\nSnippet: {result['snippet']}\n" for result in relevant_search_results])
    combined_prompt += f"\nUser input: {user_input}"

    # Replace variables in the prompt
    for key, value in variables.items():
        combined_prompt = combined_prompt.replace(f"{{{{{key}}}}}", str(value))

    # Add available tools to the prompt
    combined_prompt += "\nAvailable tools:\n"
    for tool_name, tool_code in tools.items():
        combined_prompt += f"{tool_name}: {tool_code}\n"

    if model == 'openai':
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": combined_prompt}, {"role": "user", "content": user_input}]
        )
        final_response = completion.choices[0].message['content']
    elif model == 'huggingface' and hf_model_name:
        hf_pipeline = pipeline('text-generation', model=hf_model_name, use_auth_token=hf_api_key)
        completion = hf_pipeline(combined_prompt, max_new_tokens=500)
        final_response = completion[0]['generated_text']
    else:
        return jsonify({"error": "Invalid model selected"}), 400

    return jsonify({"response": final_response}), 200

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
