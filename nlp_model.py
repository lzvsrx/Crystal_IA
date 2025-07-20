import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import os
from dotenv import load_dotenv
import requests
import json
import pandas as pd
import logging

# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# --- Configuração da API SerperDev ---
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not SERPER_API_KEY:
    logging.warning("Chave SERPER_API_KEY não encontrada. A pesquisa na internet não funcionará.")

# --- Configuração do SpaCy ---
try:
    nlp = spacy.load("pt_core_news_sm")
    logging.info("Modelo SpaCy (pt_core_news_sm) carregado.")
except OSError:
    logging.warning("Modelo 'pt_core_news_sm' do SpaCy não encontrado. Baixando...")
    try:
        spacy.cli.download("pt_core_news_sm")
        nlp = spacy.load("pt_core_news_sm")
        logging.info("Modelo SpaCy (pt_core_news_sm) baixado e carregado com sucesso.")
    except Exception as e:
        logging.error(f"Erro ao baixar ou carregar o modelo SpaCy: {e}. O PLN pode ser limitado.")
        def nlp_fallback(text):
            return text.lower()
        nlp = nlp_fallback

# Variáveis globais para o vetorizador TF-IDF e os vetores de conhecimento
vectorizer = None
knowledge_vectors = None
knowledge_data = None # Para armazenar os dados do DB (DataFrame)

# Limiar de similaridade para o banco de dados interno
DB_SIMILARITY_THRESHOLD = 0.70

# --- Funções de Processamento de Texto ---
def preprocess_text(text):
    """
    Normaliza o texto: minúsculas, remove pontuação, lematiza e remove stopwords.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    if hasattr(nlp, 'pipe'):
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_space and token.text.strip()]
        return " ".join(tokens)
    else:
        return text

def train_nlp_model(df_knowledge):
    """
    Treina o vetorizador TF-IDF com os padrões de perguntas e palavras-chave do banco de dados.
    """
    global vectorizer, knowledge_vectors, knowledge_data
    
    texts_to_train = []
    for index, row in df_knowledge.iterrows():
        combined_text = preprocess_text(row['question_pattern'])
        if pd.notna(row['keywords']):
            combined_text += " " + preprocess_text(row['keywords'])
        texts_to_train.append(combined_text)

    if not texts_to_train:
        logging.warning("Nenhum texto para treinar o modelo PLN. O banco de dados pode estar vazio.")
        return

    vectorizer = TfidfVectorizer()
    knowledge_vectors = vectorizer.fit_transform(texts_to_train)
    knowledge_data = df_knowledge
    logging.info("Modelo PLN treinado com o conhecimento da Crystal.")

def get_db_answer(user_query):
    """
    Encontra a resposta mais similar no banco de dados.
    Retorna a resposta e a similaridade, ou (None, score) se não houver similaridade suficiente.
    """
    if vectorizer is None or knowledge_vectors is None or knowledge_data is None:
        logging.warning("Modelo PLN não treinado ou dados de conhecimento ausentes para busca no DB.")
        return None, 0.0

    processed_user_query = preprocess_text(user_query)
    
    if not processed_user_query or not vectorizer.vocabulary_:
        logging.debug("Consulta do usuário processada vazia ou vocabulário do vetorizador vazio.")
        return None, 0.0

    try:
        user_query_vector = vectorizer.transform([processed_user_query])
    except ValueError:
        logging.debug("Erro ao transformar a query do usuário. Pode não haver palavras no vocabulário.")
        return None, 0.0

    similarities = cosine_similarity(user_query_vector, knowledge_vectors).flatten()
    most_similar_idx = np.argmax(similarities)
    max_similarity = similarities[most_similar_idx]

    if max_similarity >= DB_SIMILARITY_THRESHOLD:
        logging.info(f"Resposta do DB encontrada com similaridade: {max_similarity:.2f}")
        return knowledge_data.iloc[most_similar_idx]['answer'], max_similarity
    else:
        logging.info(f"Nenhuma resposta do DB com similaridade >= {DB_SIMILARITY_THRESHOLD:.2f} (max: {max_similarity:.2f}).")
        return None, max_similarity

# --- Funções de Pesquisa na Internet (SerperDev) ---
def search_serper(query):
    """Realiza uma pesquisa na internet usando a API do SerperDev."""
    if not SERPER_API_KEY:
        logging.error("Chave SERPER_API_KEY não configurada ou inválida.")
        return "Desculpe, a funcionalidade de busca na internet não está disponível (chave SerperDev ausente)."

    url = "https://serper.dev/search"
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    payload = json.dumps({
        "q": query,
        "gl": "br", # Localização global (Brasil)
        "hl": "pt"  # Idioma da interface (português)
    })
    
    try:
        logging.info(f"Fazendo requisição SerperDev para: '{query}'")
        response = requests.post(url, headers=headers, data=payload, timeout=15)
        response.raise_for_status()
        results = response.json()
        
        snippets = []
        if 'organic' in results:
            for item in results['organic']:
                if 'snippet' in item and item['snippet'].strip():
                    snippets.append(item['snippet'])
        
        if not snippets:
            logging.info(f"SerperDev retornou resultados vazios para: '{query}'")
            return ""

        logging.info(f"SerperDev retornou {len(snippets)} snippets para '{query}'.")
        return "\n".join(snippets[:3])
        
    except requests.exceptions.Timeout:
        logging.error(f"Erro de Timeout ao conectar ao SerperDev para '{query}'.")
        return "Desculpe, a busca demorou muito e não consegui obter resultados."
    except requests.exceptions.ConnectionError:
        logging.error(f"Erro de Conexão ao SerperDev para '{query}'. Verifique sua conexão à internet.")
        return "Desculpe, não consegui me conectar aos servidores de busca agora. Pode ser um problema de conexão."
    except requests.exceptions.HTTPError as e:
        logging.error(f"Erro HTTP do SerperDev para '{query}': {e}. Resposta: {e.response.text if e.response else 'N/A'}")
        if e.response.status_code == 429:
            return "Ops! Parece que estou recebendo muitas requisições para a busca. Tente novamente em um momento!"
        return f"Desculpe, ocorreu um erro com o serviço de busca (Código: {e.response.status_code if e.response else 'N/A'})."
    except json.JSONDecodeError:
        logging.error(f"Erro ao decodificar a resposta JSON do SerperDev para '{query}'. Resposta inválida.")
        return "Desculpe, recebi uma resposta inválida da busca. Os cristais estão confusos!"
    except Exception as e:
        logging.error(f"Ocorreu um erro inesperado na busca para '{query}': {e}")
        return "Algo deu errado ao tentar buscar na internet. Tente novamente!"

# --- Função Principal de Resposta da Crystal ---
def get_crystal_response(user_query, chat_history):
    """
    Decide como a Crystal deve responder:
    1. Usando o banco de dados interno (respostas objetivas e rápidas).
    2. Sempre tentando pesquisar na internet via SerperDev se não encontrar no DB.
    """
    logging.info(f"Processando query do usuário: '{user_query}'")

    # 1. Tentar encontrar a resposta no banco de dados interno
    db_answer, db_similarity = get_db_answer(user_query)

    if db_answer:
        logging.info(f"Retornando resposta do DB. Similaridade: {db_similarity:.2f}")
        return db_answer, "db_internal"

    # 2. Se não encontrou no DB, use a pesquisa SerperDev diretamente
    logging.info(f"Nenhuma resposta no DB. Tentando pesquisar na web via SerperDev para: '{user_query}'")
    search_results_text = search_serper(user_query)

    if "Desculpe," in search_results_text or "Erro:" in search_results_text or "Ops!" in search_results_text:
        logging.error(f"SerperDev retornou erro ou mensagem de problema: {search_results_text}")
        return search_results_text, "serper_search_error"
    elif not search_results_text.strip():
        logging.warning(f"Busca SerperDev retornou resultados vazios ou sem conteúdo útil para: '{user_query}'.")
        # Fallback message when search yields no useful results
        return "Ah, os cristais da web estão um pouco embaçados agora! Não encontrei informações claras sobre isso. Tente me perguntar de outra forma ou sobre outro tópico!", "serper_no_results"
    else:
        # Construct a response based on search results
        response = (
            f"Encontrei algumas informações interessantes na vastidão da web sobre '{user_query}':\n\n"
            f"{search_results_text}\n\n"
            "Espero que isso ilumine seu caminho! Se precisar de mais detalhes, é só perguntar!"
        )
        return response, "serper_search_success"
