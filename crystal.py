import streamlit as st
import random
import time
import os
import logging # Importa o módulo de logging para usar no Streamlit

# Configuração de Logging para o Streamlit (opcional, mas bom para debug)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Importa as funções do banco de dados e do modelo de PLN/LLM
import database as db
import nlp_model as nlp

# --- Configurações da Crystal ---
CRYSTAL_NAME = "Crystal"
CRYSTAL_IMAGE_FILENAME = "crystal_avatar.png"
DEFAULT_CRYSTAL_IMAGE_URL = "https://cdn-icons-png.flaticon.com/512/3208/3208753.png"

FUN_THEMES = [
    "Aventuras Cósmicas",
    "Jornadas no Tempo",
    "Explorações Místicas",
    "Descobertas Científicas Divertidas"
]

# --- Inicialização do Banco de Dados e Modelo PLN ---
@st.cache_resource # Cacheia a execução para evitar recarregar o DB e o modelo toda vez
def initialize_crystal_brain():
    logging.info("Iniciando cérebro da Crystal (cache_resource)...")
    db.create_table() # Garante que a tabela do DB existe
    db.populate_initial_data() # Popula com dados iniciais se estiver vazio
    
    knowledge_df = db.get_all_knowledge()
    if not knowledge_df.empty:
        nlp.train_nlp_model(knowledge_df) # Treina o modelo PLN com os dados do DB
        logging.info("Cérebro da Crystal inicializado e modelo PLN treinado!")
    else:
        logging.warning("Banco de dados de conhecimento vazio. Modelo PLN não treinado.")
    return True # Retorna True para indicar sucesso na inicialização

# Garante que o cérebro da Crystal seja inicializado ao carregar o app
initialize_crystal_brain()

# --- Configuração da Página Streamlit ---
st.set_page_config(
    page_title=f"{CRYSTAL_NAME} - Sua Assistente Interativa",
    page_icon="✨",
    layout="centered"
)

# --- Caminho da Imagem da Crystal ---
crystal_avatar_path = CRYSTAL_IMAGE_FILENAME if os.path.exists(CRYSTAL_IMAGE_FILENAME) else DEFAULT_CRYSTAL_IMAGE_URL

# --- Título e Descrição ---
st.title(f"✨ Conheça {CRYSTAL_NAME}, Sua Amiga Cósmica! ✨")
st.markdown(f"**Tema do dia:** *{random.choice(FUN_THEMES)}*")
st.write("Olá! Sou a Crystal, sua assistente pessoal divertida e cheia de soluções. Pergunte-me qualquer coisa!")

st.image(crystal_avatar_path, width=150)
st.markdown("---")

# --- Inicialização do Histórico do Chat e Contadores de API ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_calls_count" not in st.session_state:
    st.session_state.api_calls_count = {'gemini': 0, 'serper': 0}

# --- Botão Limpar Chat (Experiência do Usuário) ---
# Colocado na sidebar ou em uma coluna para não atrapalhar o chat principal
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("Limpar Chat 🧹", key="clear_chat_button"):
        st.session_state.messages = []
        st.session_state.api_calls_count = {'gemini': 0, 'serper': 0} # Zera contadores também
        st.experimental_rerun() # Reinicia o app para limpar a tela
with col2:
    st.markdown(f"**API Calls:** Gemini: `{st.session_state.api_calls_count['gemini']}` | Serper: `{st.session_state.api_calls_count['serper']}`")


# --- Exibir Histórico do Chat ---
for message in st.session_state.messages:
    avatar_to_display = crystal_avatar_path if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar_to_display):
        st.markdown(message["content"])

# --- Entrada do Usuário ---
user_input = st.chat_input("Pergunte algo à Crystal...")

if user_input:
    # Adicionar mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": user_input, "avatar": "👤"})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Simular "pensamento" da IA
    with st.chat_message("assistant", avatar=crystal_avatar_path):
        with st.spinner("Crystal está consultando os cristais do conhecimento e explorando a vastidão da web..."):
            time.sleep(1.0) # Pequeno atraso para o spinner aparecer

            # Chama a função principal que decide a resposta da Crystal
            response, response_type = nlp.get_crystal_response(user_input, st.session_state.messages)
            
            # Atualiza contadores de API e exibe mensagem de origem
            if "llm" in response_type: # Se a resposta envolveu o LLM
                st.session_state.api_calls_count['gemini'] += 1
            if "search" in response_type: # Se a resposta envolveu busca (mesmo que com erro)
                st.session_state.api_calls_count['serper'] += 1

            if response_type == "db_internal":
                st.info("💡 Resposta precisa do banco de dados interno da Crystal!")
            elif response_type == "llm_search":
                st.info("🌐 Crystal pesquisou na web para você e sintetizou!")
            elif response_type == "llm_direct":
                st.info("🧠 Resposta gerada pela inteligência cósmica da Crystal!")
            elif response_type == "llm_fallback_search":
                st.warning("⚠️ Crystal tentou pesquisar, mas os resultados foram nebulosos. Ela te deu uma resposta alternativa!")
            elif response_type == "llm_search_error":
                 st.error("❌ Ops! A Crystal encontrou um problema ao tentar buscar na web. Tente novamente mais tarde.")

            st.markdown(response)
            # Adicionar resposta da Crystal ao histórico
            st.session_state.messages.append({"role": "assistant", "content": response, "avatar": crystal_avatar_path})

st.markdown("---")
st.caption("✨ Criado com amor e um pouco de magia cósmica ✨")