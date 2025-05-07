import streamlit as st
from dotenv import load_dotenv
import os
import chromadb
import mysql.connector
from sentence_transformers import SentenceTransformer
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import requests

# ✅ Set Streamlit page configuration
st.set_page_config(page_title="I can Retrieve any SQL query")

# ✅ Load environment variables
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
if not HUGGINGFACE_API_KEY:
    st.error("HUGGINGFACE_API_KEY not found in environment variables.")
    st.stop()

# ✅ Load sentence transformer model
from sentence_transformers import SentenceTransformer

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name, device='cpu')

# ✅ Initialize ChromaDB client and collection
import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings())

try:
    collection = client.get_or_create_collection(name="inventory_sql_queries")
except Exception as e:
    st.error(f"Failed to initialize collection: {e}")
    st.stop()

# ✅ Inventory questions and cleaned SQL queries
inventory_data = [
    # Existing queries
    ("How many apples are in stock?", "SELECT COUNT(*) FROM products WHERE name = 'Apple';"),
    ("How many bananas do we have?", "SELECT COUNT(*) FROM products WHERE name = 'Banana';"),
    ("How many oranges are in stock?", "SELECT COUNT(*) FROM products WHERE name = 'Orange';"),
    ("List all chocolates in stock", "SELECT * FROM products WHERE name = 'Chocolate';"),
    ("List all items in inventory", "SELECT * FROM products;"),
    ("Show all items with quantity less than 5", "SELECT * FROM products WHERE quantity < 5;"),
    ("What is the price of apples?", "SELECT price FROM products WHERE name = 'Apple';"),
    ("Get the quantity of sugar", "SELECT quantity FROM products WHERE name = 'Sugar';"),
    ("What is the total number of items?", "SELECT COUNT(*) FROM products;"),
    
    # New queries
    ("List all products expiring in the next 7 days", "SELECT * FROM products WHERE expiry_date BETWEEN CURDATE() AND CURDATE() + INTERVAL 7 DAY;"),
    ("List all products with stock less than 10", "SELECT * FROM products WHERE quantity < 10;"),
    ("Which product has the highest stock?", "SELECT name, quantity FROM products ORDER BY quantity DESC LIMIT 1;"),
    ("List all products sorted by expiry date", "SELECT * FROM products ORDER BY expiry_date ASC;"),
    ("How many products are there in total?", "SELECT COUNT(*) FROM products;"),
    ("How much milk is in stock?", "SELECT quantity FROM products WHERE name = 'Milk';"),
    ("List products with no expiry date", "SELECT * FROM products WHERE expiry_date IS NULL;"),
    ("List all products with descriptions and prices", "SELECT name, description, price FROM products;"),
    ("List all new arrivals", "SELECT * FROM products ORDER BY created_at DESC LIMIT 5;"),
    ("Which product has the highest price?", "SELECT name, price FROM products ORDER BY price DESC LIMIT 1;"),
    ("Find products that are fresh", "SELECT * FROM products WHERE description LIKE '%fresh%';"),
    ("List all expired products", "SELECT * FROM products WHERE expiry_date < CURDATE();"),
    ("List products added after a specific date", "SELECT * FROM products WHERE created_at > '2025-01-01';"),
    ("Check availability of apple", "SELECT quantity FROM products WHERE name = 'Apple';"),
    ("List products with no description", "SELECT * FROM products WHERE description IS NULL OR description = '';"),
    ("What is the average price of products?", "SELECT AVG(price) FROM products;"),
    ("List products expiring after July 1, 2025", "SELECT * FROM products WHERE expiry_date > '2025-07-01';"),
    ("List out of stock products", "SELECT * FROM products WHERE quantity = 0;")
]

for i, (question, sql) in enumerate(inventory_data):
    embedding = model.encode([question])
    collection.add(
        documents=[question],
        embeddings=embedding.tolist(),
        metadatas=[{'sql': sql}],
        ids=[str(i)]
    )

# ✅ Function to execute SQL query directly on the MySQL database
def execute_sql_query(sql_query):
    try:
        # Establish connection to MySQL
        conn = mysql.connector.connect(
            host="localhost",
            user="root",  # Use your MySQL username
            password="root",  # Use your MySQL password
            database="supermarket_db"  # Use your database name
        )
        cursor = conn.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
        conn.close()
        return result
    except mysql.connector.Error as err:
        st.error(f"Error executing query: {err}")
        return None

# ✅ Function to retrieve closest SQL from ChromaDB
def get_closest_sql_query(user_question):
    embedding = model.encode([user_question])
    results = collection.query(
        query_embeddings=embedding.tolist(),
        n_results=1
    )
    if results and len(results['documents']) > 0:
        document = results['documents'][0][0]
        sql_query = results['metadatas'][0][0]['sql']
        return document, sql_query
    return None, None

# ✅ Generate SQL using LangChain if not found
def generate_sql_with_llm(user_question):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt_template = """
    You are an AI assistant. I will ask you a question about inventory, and you will respond with the appropriate SQL query that would retrieve the answer from a database.

    Question: {user_question}
    SQL Query:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["user_question"])
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    return llm_chain.run(user_question)

# ✅ Streamlit interface
st.title("Gemini App to Retrieve SQL Data")
user_input = st.text_input("Ask your question about the inventory:")

if st.button("Submit") and user_input:
    with st.spinner("Retrieving relevant SQL and executing..."):
        context, sql_query = get_closest_sql_query(user_input)

        if sql_query:
            st.write(f"Generated SQL query from ChromaDB: `{sql_query}`")
            # Execute the SQL query directly on the MySQL database
            result = execute_sql_query(sql_query)
        else:
            st.info("No relevant SQL found in ChromaDB. Generating with LLM...")
            sql_query = generate_sql_with_llm(user_input)
            st.write(f"Generated SQL query from LLM: `{sql_query}`")
            # Execute the SQL query directly on the MySQL database
            result = execute_sql_query(sql_query)

        # ✅ Display results
        if result:
            st.success("Query executed successfully!")
            if len(result) == 1 and len(result[0]) == 1:
                st.write(f"Count: {result[0][0]}")
            else:
                st.write(result)
        else:
            st.error("Error executing the SQL query.")
