import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ------------------------
# 1. Book PDF Files (hard-coded)
# ------------------------
# Replace this with the path to your PDF file(s)
PDF_PATHS = [
    r"C:\Users\domma\Downloads\The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf",  # note trailing comma
    # r"C:\Users\domma\Downloads\another_medical_text.pdf",
]


# ------------------------
# 2. Medical Relevance Checker (Simple)
# ------------------------
MEDICAL_KEYWORDS = [
    "symptom", "disease", "treatment", "fever", "pain", "diabetes", "bp", "pressure",
    "hypertension", "cough", "infection", "medicine", "medication", "health", "wellness",
    "diagnosis", "surgery", "fracture", "prescription", "antibiotic", "doctor",
    "vomiting", "blood", "injury", "injuries", "asthma", "cancer", "tumor", "thermometer",
    "temperature", "allergy", "allergic", "rash", "swelling", "emergency", "hospital",
    "clinic", "dose", "doses", "child", "kid", "infant", "adult", "elderly",
    # Expand this list as needed for full coverage
]

def is_medical_query(query: str) -> bool:
    """Returns True if the query appears to be about a medical topic."""
    # Very simple: adapt with regular expressions or LLM classification for more robustness
    q_lower = query.lower()
    for keyword in MEDICAL_KEYWORDS:
        if keyword in q_lower:
            return True
    return False

# ------------------------
# 3. OpenAI/AzureOpenAI Config
# ------------------------
DEPLOYMENT_NAME = "gpt-4o"
MODEL_NAME = "gpt-4o"
OPENAI_API_BASE = "https://ue2coai06gaoa0a.openai.azure.com/"
OPENAI_API_KEY = "API_KEY"
OPENAI_API_VERSION = "2024-02-15-preview"

# ------------------------
# 4. Streamlit App Header
# ------------------------
st.header("Hello User! I am your AI-based Medibuddy")

# ------------------------
# 5. PDF Book Loading & Vector Store Initialization (Code-based)
# ------------------------
@st.cache_resource(show_spinner="Loading Knowledge and preparing search...")
def load_medical_books(pdf_paths):
    all_chunks = []
    for book_path in pdf_paths:
        pdf_reader = PdfReader(book_path)
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                # Attach metadata for traceability
                all_chunks.append({
                    "text": page_text,
                    "metadata": {"page": i + 1, "source_file": book_path}
                })

    # Chunk and split for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    split_chunks = []
    for chunk in all_chunks:
        splits = text_splitter.split_text(chunk["text"])
        for split_text in splits:
            split_chunks.append({
                "text": split_text,
                "metadata": chunk["metadata"]
            })
    texts = [c["text"] for c in split_chunks]
    metadatas = [c["metadata"] for c in split_chunks]
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vector_store

VECTOR_STORE = load_medical_books(PDF_PATHS)  # Coded-in documents only

# ------------------------
# 6. Conversation History State
# ------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------------
# 7. PromptTemplate for Concise Medical Answers
# ------------------------
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a trusted medical AI assistant and must answer ONLY questions directly related to medical science, healthcare, symptoms, or treatments using the book context below.
If the user's question is not medical, reply strictly: "Sorry, I can only answer medical-related questions."
When you do answer, be concise and cite the source (page number or section) from the context.

Context from the book:
{context}

Question:
{question}

Response (short answer, then 'Source:' and specific page reference): 
"""
)

# ------------------------
# 8. User Question Input and Chat Handling
# ------------------------
for i in range(len(st.session_state.chat_history)//2):
    user_msg = st.session_state.chat_history[2*i]
    assistant_msg = st.session_state.chat_history[2*i + 1]
    st.markdown(f"**Question {i+1}:** {user_msg['content']}")
    st.markdown(f"**Answer {i+1}:** {assistant_msg['content']}")

next_question_idx = len(st.session_state.chat_history)//2

user_question = st.text_input(
    f"Ask your medical question {next_question_idx + 1}:",
    key=f"question_{next_question_idx}"
)

if user_question:
    if not is_medical_query(user_question):
        response = "Sorry, I can only answer medical-related questions."
    else:
        matches = VECTOR_STORE.similarity_search(user_question)
        if matches:
            llm = AzureChatOpenAI(
                api_key=OPENAI_API_KEY,
                openai_api_version=OPENAI_API_VERSION,
                openai_api_base=OPENAI_API_BASE,
                deployment_name=DEPLOYMENT_NAME,
                openai_api_type="azure",
                temperature=0.3,
                max_tokens=4096
            )
            chain = load_qa_chain(
                llm,
                chain_type="stuff",
                prompt=prompt_template
            )
            response = chain.run(input_documents=matches, question=user_question)
        else:
            response = "Sorry, the medical information you're seeking is not found in the provided book."

    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.write(response)
