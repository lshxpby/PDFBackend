from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import fitz
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.llms.huggingface_hub import HuggingFaceHub

app = FastAPI()

conversation = None

def extract_text_from_pdf(pdf_content):
    """
    Extracts text from a PDF file.
    """
    text = ""
    with fitz.open("pdf", pdf_content) as doc:
        for page in doc:
            text += page.get_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=400,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-ul2", model_kwargs={"temperature":0.7, "max_length":1024}, huggingfacehub_api_token="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")
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global conversation 
    pdf_content = await file.read()

    text = extract_text_from_pdf(pdf_content)
    print(text)
    text_chunks = get_text_chunks(text)
    vectorstore = get_vectorstore(text_chunks)
    conversation = get_conversation_chain(vectorstore)  

def handle_input(question):
    response = conversation({'question': question})
    return response

@app.post("/ask-question/")
async def ask_question(request_data: dict):
    question = request_data.get("question", "")
    response = handle_input(question)
    return {"response": response}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

def main():
    load_dotenv()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
