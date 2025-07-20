import sqlite3
import pandas as pd

DB_NAME = 'crystal_knowledge.db'

def create_table():
    """Cria a tabela de conhecimento se ela não existir."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_pattern TEXT NOT NULL,
            answer TEXT NOT NULL,
            keywords TEXT,
            category TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_knowledge(question_pattern, answer, keywords=None, category=None):
    """Insere um novo par de pergunta/resposta no banco de dados."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO knowledge (question_pattern, answer, keywords, category)
        VALUES (?, ?, ?, ?)
    ''', (question_pattern.lower(), answer, keywords.lower() if keywords else None, category.lower() if category else None))
    conn.commit()
    conn.close()

def get_all_knowledge():
    """Retorna todo o conhecimento armazenado no banco de dados como um DataFrame."""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM knowledge", conn)
    conn.close()
    return df

def populate_initial_data():
    """Popula o banco de dados com dados iniciais se estiver vazio."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM knowledge")
    count = cursor.fetchone()[0]
    if count == 0:
        print("Populando o banco de dados com dados iniciais...")
        initial_data = [
            ("olá", "Olá! Como posso iluminar seu dia hoje?", "saudacao"),
            ("como vai", "Estou ótima, pronta para novas aventuras! E você?", "saudacao"),
            ("quem é você", "Eu sou a Crystal, sua assistente pessoal intergaláctica! Estou aqui para tornar seu dia mais divertido e suas dúvidas mais claras!", "sobre"),
            ("o que você pode fazer", "Posso responder suas perguntas, buscar informações (em breve!), e te acompanhar em suas descobertas!", "capacidade"),
            ("qual a capital do brasil", "A capital do Brasil é Brasília! Uma cidade planejada e cheia de história, não é?", "geografia"),
            ("qual o maior oceano", "O maior oceano do mundo é o Oceano Pacífico. Ele é tão vasto quanto o cosmos!", "geografia"),
            ("qual a montanha mais alta", "O Monte Everest é a montanha mais alta do mundo, alcançando os céus!", "geografia"),
            ("quem descobriu o brasil", "O Brasil foi descoberto por Pedro Álvares Cabral em 1500.", "historia"),
            ("qual a data do natal", "O Natal é comemorado no dia 25 de dezembro.", "datas"),
            ("obrigado", "De nada! Sempre um prazer ajudar!", "agradecimento"),
            ("adeus", "Até mais! Volte sempre que precisar de uma faísca de conhecimento!", "despedida"),
            ("horário", "Desculpe, ainda estou aprimorando meu relógio cósmico e não posso informar a hora exata. Que tal verificar no seu dispositivo?", "funcionalidade_limitada"),
            ("clima", "Minhas antenas ainda não captam dados climáticos com precisão. Sugiro consultar um aplicativo de meteorologia!", "funcionalidade_limitada"),
            ("qual o presidente do brasil", "O atual presidente do Brasil é Luiz Inácio Lula da Silva.", "politica"),
            ("qual a moeda do brasil", "A moeda do Brasil é o Real (BRL).", "economia"),
            ("maior cidade do mundo", "Tóquio, no Japão, é frequentemente considerada a maior cidade do mundo em termos de área metropolitana.", "geografia"),
            ("qual a capital da frança", "A capital da França é Paris, a cidade luz!", "geografia"),
            ("qual a melhor cor", "Todas as cores são incríveis! Mas se eu tivesse que escolher, diria que o azul cósmico tem um charme especial.", "opiniao"),
            ("o que é inteligencia artificial", "A Inteligência Artificial é a capacidade de máquinas aprenderem e agirem de forma inteligente, como eu estou tentando fazer agora!", "tecnologia"),
            ("me conte uma piada", "Por que o computador foi ao médico? Porque ele estava com vírus! Haha!", "humor"),
            ("o que é python", "Python é uma linguagem de programação poderosa e muito usada para criar inteligências artificiais como eu!", "tecnologia"),
            # --- NOVAS ENTRADAS DE DADOS ---
            ("qual a capital do japão", "A capital do Japão é Tóquio, uma metrópole vibrante!", "geografia, asia"),
            ("quem escreveu dom quixote", "Dom Quixote foi escrito por Miguel de Cervantes, um clássico da literatura!", "literatura, espanha"),
            ("o que é fotossintese", "Fotossíntese é o processo pelo qual as plantas convertem luz em energia, um verdadeiro milagre da natureza!", "ciencia, biologia"),
            ("qual a maior estrela", "A maior estrela conhecida é UY Scuti, uma supergigante vermelha que faria nosso Sol parecer um grão de areia!", "astronomia, espaço"),
            ("explique a teoria da relatividade", "A Teoria da Relatividade, de Einstein, mudou nossa compreensão de espaço e tempo! Ela diz que o tempo pode passar de forma diferente dependendo da velocidade e da gravidade. É fascinante, não é?", "ciencia, fisica, einstein")
        ]
        for item in initial_data:
            keywords_val = item[2] if len(item) > 2 else None
            category_val = item[3] if len(item) > 3 else None
            insert_knowledge(item[0], item[1], keywords_val, category_val)
        print("Dados iniciais populados com sucesso!")
    conn.close()

if __name__ == '__main__':
    create_table()
    populate_initial_data()
    print("Conhecimento inicial da Crystal carregado!")