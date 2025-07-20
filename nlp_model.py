import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import os
from dotenv import load_dotenv
import google.generativeai as genai
import requests
import json
import pandas as pd
import logging # Importa o módulo de logging

# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# --- Configuração das APIs ---
GOOGLE_API_KEY = os.gentev(code.txt)
SERPER_API_KEY = os.gentev(code2)

gemini_model = None
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-pro')
        logging.info("API do Google Gemini configurada.")
    except Exception as e:
        logging.error(f"Erro ao configurar a API do Google Gemini: {e}. A geração de texto com LLM não funcionará.")
else:
    logging.warning("Chave GOOGLE_API_KEY não encontrada. A geração de texto com LLM não funcionará.")

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
        # Fallback se o download falhar, pode ser necessário carregar um modelo mais básico ou lidar com isso
        # Neste caso, vamos apenas retornar uma função vazia para preprocess_text
        def nlp_fallback(text):
            return text.lower()
        nlp = nlp_fallback

# Variáveis globais para o vetorizador TF-IDF e os vetores de conhecimento
vectorizer = None
knowledge_vectors = None
knowledge_data = None # Para armazenar os dados do DB (DataFrame)

# Limiar de similaridade para o banco de dados interno
# Ajuste este valor (entre 0 e 1). Maior valor = mais rigoroso para usar o DB.
DB_SIMILARITY_THRESHOLD = 0.70 # Aumentado ligeiramente para priorizar respostas mais exatas do DB

# --- Funções de Processamento de Texto ---
def preprocess_text(text):
    """
    Normaliza o texto: minúsculas, remove pontuação, lematiza e remove stopwords.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    if hasattr(nlp, 'pipe'): # Verifica se o nlp é um objeto SpaCy válido
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_space and token.text.strip()]
        return " ".join(tokens)
    else: # Fallback para o caso de o SpaCy não ter carregado
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
        response = requests.post(url, headers=headers, data=payload, timeout=15) # Aumenta timeout
        response.raise_for_status() # Levanta um erro para códigos de status HTTP ruins (4xx ou 5xx)
        results = response.json()
        
        snippets = []
        if 'organic' in results:
            for item in results['organic']:
                if 'snippet' in item and item['snippet'].strip(): # Garante que o snippet não é vazio
                    snippets.append(item['snippet'])
        
        if not snippets:
            logging.info(f"SerperDev retornou resultados vazios para: '{query}'")
            return "" # Retorna string vazia se não houver snippets úteis

        logging.info(f"SerperDev retornou {len(snippets)} snippets para '{query}'.")
        return "\n".join(snippets[:3]) # Retorna os 3 primeiros snippets como uma string
        
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

# --- Funções de Geração de Texto com LLM (Gemini) ---
def generate_response_with_llm(prompt_parts):
    """
    Gera uma resposta usando o modelo Gemini.
    prompt_parts é uma lista de strings (contexto, pergunta, etc.)
    """
    if not gemini_model:
        return "Desculpe, a funcionalidade de IA avançada não está disponível (chave Gemini ausente ou inválida)."
    
    try:
        logging.info(f"Fazendo requisição Gemini. Prompt inicial: '{prompt_parts[0][:100]}...'")
        response = gemini_model.generate_content(prompt_parts)
        
        if response and response.text:
            logging.info("Resposta do Gemini gerada com sucesso.")
            return response.text
        else:
            logging.warning(f"Gemini retornou uma resposta vazia ou sem texto: {response}")
            return "Desculpe, minha inteligência avançada está um pouco nebulosa agora e não conseguiu gerar uma resposta clara."
    except genai.types.BlockedPromptException as e:
        logging.error(f"Erro: O prompt foi bloqueado por segurança. Detalhes: {e}")
        return "Sinto muito, mas não consigo responder a essa pergunta devido às minhas diretrizes de segurança. Minha programação não me permite abordar esse tópico."
    except Exception as e:
        logging.error(f"Erro ao gerar conteúdo com Gemini: {e}")
        return "Desculpe, tive um problema cósmico ao processar sua solicitação com minha inteligência avançada. Tente de novo!"

# --- Função Principal de Resposta da Crystal ---
def get_crystal_response(user_query, chat_history):
    """
    Decide como a Crystal deve responder:
    1. Usando o banco de dados interno (respostas objetivas e rápidas).
    2. Usando o LLM para classificar a intenção e/ou gerar respostas.
       - Se precisar, pesquisando na internet.
       - Gerando uma resposta direta.
    """
    logging.info(f"Processando query do usuário: '{user_query}'")

    # 1. Tentar encontrar a resposta no banco de dados interno
    db_answer, db_similarity = get_db_answer(user_query)

    if db_answer:
        logging.info(f"Retornando resposta do DB. Similaridade: {db_similarity:.2f}")
        return db_answer, "db_internal"

    # 2. Se não encontrou no DB, usar LLM para decidir se precisa de busca na web ou gerar direto
    
    # Gerenciamento de Contexto: Pega as últimas X mensagens (excluindo a atual do usuário)
    # Ajuste este número para controlar o tamanho do contexto enviado (e o custo da API).
    CONTEXT_MESSAGE_LIMIT = 8 # Total de mensagens no histórico para o LLM

    # Prepara o histórico para o prompt do LLM. O Streamlit gerencia 'messages'
    # de forma que a última é sempre a pergunta atual do usuário.
    # Queremos as mensagens ANTES da atual para o contexto.
    recent_history_for_llm = []
    # Itera sobre o histórico para pegar as mensagens anteriores e formatá-las
    # O slice [-(CONTEXT_MESSAGE_LIMIT+1):-1] pega as últimas CONTEXT_MESSAGE_LIMIT
    # mensagens ANTES da última (que é a query atual do usuário)
    if len(chat_history) > 1: # Pelo menos uma mensagem anterior da Crystal/usuário
        for msg in chat_history[-(CONTEXT_MESSAGE_LIMIT + 1):-1]:
            if 'role' in msg and 'content' in msg:
                recent_history_for_llm.append(f"{'Usuário' if msg['role'] == 'user' else 'Crystal'}: {msg['content']}")
    
    formatted_history = "\n".join(recent_history_for_llm)

    # Melhoria no Prompting: Mais exemplos, persona clara, instruções de saída.
    llm_decision_prompt = (
        "Você é a Crystal, uma assistente pessoal com um toque divertido e cósmico, focada em fornecer soluções e respostas precisas.\n"
        "Analise a última pergunta do usuário e o histórico da conversa. Siga estas instruções rigorosamente:\n"
        "1. Se a pergunta for um **fato específico** que exige busca externa na internet (ex: 'Quem é o atual primeiro-ministro do Canadá?', 'Últimas notícias sobre IA', 'Qual a população da China?', 'Temperatura em Londres?', 'O que é um buraco negro?', 'Cotação do dólar hoje', 'receita de bolo de chocolate'), responda **EXATAMENTE** com 'BUSCAR_WEB: [sua_query_de_busca]'. Formule a busca de forma clara, concisa e focada na informação necessária. Não inclua conversas ou saudação nesta parte.\n"
        "2. Se a pergunta for **opinativa, criativa, requer uma explicação mais geral, ou não exige uma busca factual externa** (ex: 'Me conte uma história', 'Qual sua cor favorita?', 'Me ajude a pensar em nomes de pets', 'Como posso ser mais feliz?', 'Me diga algo interessante', 'Por que o céu é azul?', 'Qual a importância da água?'), responda **diretamente ao usuário**, mantendo seu tom divertido, prestativo e cósmico. Seja criativa, mas vá direto ao ponto e não exceda 80 palavras.\n"
        "3. Se a pergunta for uma **saudação simples, agradecimento ou despedida** (ex: 'Oi', 'Tudo bem?', 'Obrigado', 'Adeus'), responda diretamente de forma breve e amigável, no máximo 20 palavras.\n\n"
        "--- Histórico da Conversa ---\n" + formatted_history +
        f"\nUsuário: {user_query}\n"
        "Crystal (Sua resposta no formato especificado ou BUSCAR_WEB: [query]):"
    )

    llm_decision_response = generate_response_with_llm([llm_decision_prompt])
    
    if llm_decision_response and llm_decision_response.startswith("BUSCAR_WEB:"):
        search_query = llm_decision_response.replace("BUSCAR_WEB:", "").strip()
        logging.info(f"Crystal decidiu pesquisar na web: '{search_query}'")
        
        search_results_text = search_serper(search_query)

        if "Desculpe," in search_results_text or "Erro:" in search_results_text or "Ops!" in search_results_text:
            # Caso a busca falhe (erro na API ou conexão) ou retorne mensagem de erro
            logging.error(f"SerperDev retornou erro ou mensagem de problema: {search_results_text}")
            return search_results_text, "llm_search_error"
        elif not search_results_text.strip():
            # Caso a busca retorne vazio ou irrelevante
            logging.warning(f"Busca SerperDev retornou resultados vazios ou sem conteúdo útil para: '{search_query}'.")
            fallback_prompt = (
                f"O usuário perguntou sobre '{user_query}'. Tentei pesquisar na internet por '{search_query}', mas não encontrei informações claras no momento. "
                "Responda ao usuário de forma divertida e útil, explicando que a pesquisa foi um pouco nebulosa ou que você precisa de mais detalhes para iluminar o caminho. "
                "Sugira que a pergunta seja mais específica ou que tente novamente. Use até 80 palavras.\n"
                f"Histórico recente: {formatted_history}\n"
                f"Pergunta original: {user_query}\n"
                "Sua resposta:"
            )
            return generate_response_with_llm([fallback_prompt]), "llm_fallback_search"
        else:
            # Use o LLM para resumir os resultados da busca e formular uma resposta amigável
            summarize_prompt = (
                f"O usuário perguntou sobre '{user_query}'. Encontrei as seguintes informações atualizadas na internet:\n\n"
                f"{search_results_text}\n\n"
                "Agora, aja como a Crystal, sua assistente pessoal cósmica e divertida. Analise essas informações cuidadosamente e forneça uma resposta clara, objetiva e amigável ao usuário. Use um tom entusiasmado e, se possível, um toque de humor ou uma analogia cósmica.\n"
                "Seja direto e conciso, com um máximo de 100 palavras. Adapte a resposta para ser útil e focada na pergunta. Se a informação não for conclusiva, mencione isso com seu toque Crystal.\n"
                "Sua resposta como Crystal:"
            )
            return generate_response_with_llm([summarize_prompt]), "llm_search"
    else:
        logging.info("LLM decidiu responder diretamente sem pesquisa na web.")
        return llm_decision_response, "llm_direct"
