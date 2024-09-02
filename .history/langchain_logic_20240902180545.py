import os
import shutil
import logging
import torch
from queue import Queue
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, JSONLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.schema.output import LLMResult
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

MY_TEMPLATE = """
You are a helpful, respectful, and honest assistant dedicated to providing informative and accurate response based on provided context((delimited by <ctx></ctx>)) only. You don't derive
answer outside context, while answering your answer should be precise, accurate, clear and should not be verbose and only contain answer.

If formatting, such as bullet points, numbered lists, tables, or code blocks, is necessary for a comprehensive response, please apply the appropriate formatting.

<ctx>
CONTEXT:
{context}
</ctx>

QUESTION:
{question}

ANSWER
"""

# Queues for handling requests and responses
prompt_queue = Queue()
sse_event_queue = Queue()

# Variables for storing model and chain instances
qa_chain = None
llm = None
response_thread = None

USE_OPENAI = True

logging.basicConfig(filename="FlaskGPT.log", level=logging.INFO, filemode="w")

class StreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        global sse_event_queue
        sse_event_queue.put({'type': 'token', 'content': token.replace('\n', '<br>')})
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        global sse_event_queue
        sse_event_queue.put({'type': 'start'})

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        global sse_event_queue
        sse_event_queue.put({'type': 'end'})

    def on_llm_error(self, error: BaseException, **kwargs) -> None:
        global sse_event_queue
        sse_event_queue.put({'type': 'error', 'content': str(error)})

def create_vectordb():
    texts = []

    try:
        db_path = os.path.join(os.getcwd(), 'db')

        if os.path.exists(db_path)):
            shutil.rmtree(db_path)
        loader_dir = DirectoryLoader('data', glob='*.json', loader_cls=JSONLoader, loader_kwargs={ 'jq_schema': '.', 'text_content': False })
        documents = loader_dir.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
        texts += text_splitter.split_documents(documents)
        model_kwargs = { 'device': 'cuda' if torch.cuda.is_available() else 'cpu' }
        encode_kwargs = { 'normalize_embeddings': True }
        embedding = HuggingFaceBgeEmbeddings(model_name='thenlper/gte-base', model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
        vectordb = Chroma.from_documents(documents=texts,
                                         embedding=embedding,
                                         persist_directory=db_path)
        logging.info(f"The '{db_path}' folder has been created.")
    except Exception as e:
        logging.error(f"An error occurred while processing file: {e}")

def init_local_llm(model_name):
    global llm

    try:
        model_dir = os.path.join(os.getcwd(), 'models')
        model_path = os.path.join(model_dir, model_name)

        llm = LlamaCpp(model_path=model_path, temperature=0, n_ctx=2048, n_gpu_layers=8,
                        n_batch=100, streaming=True, callbacks=[])
        logging.info(f"Local LLM: {model_name} has been loaded")
    except Exception as e:
        logging.error(f"Failed to load local model '{model_name}': {e}")

def init_openai_llm():
    global llm
    try:
        llm = ChatOpenAI(streaming=True, temperature=0.0, callbacks=[])
    except Exception as e:
        logging.error("OpenAI failed to initialize: {e}.")

def init_llm():
    global llm, qa_chain

    try:
        db_path = os.path.join(os.getcwd(), 'db')

        model_kwargs = { 'device': 'cuda' if torch.cuda.is_available() else 'cpu' }
        encode_kwargs = { 'normalize_embeddings': True }
        embedding = HuggingFaceBgeEmbeddings(model_name='thenlper/gte-base', model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
        vectordb = Chroma(persist_directory=db_path, embedding_function=embedding)
        retriever = vectordb.as_retriever(search_kwargs={ 'k': 10 })
        prompt_template = PromptTemplate.from_template(MY_TEMPLATE)
        chain_type_kwargs = { 'verbose': True, 'prompt': prompt_template }
        qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                            chain_type='stuff',
                                            retriever=retriever,
                                            return_source_documents=False, verbose=True,
                                            chain_type_kwargs=chain_type_kwargs)
        logging.info(f"LLM initialized")
    except Exception as e:
        logging.error(f"LLM failed to initialize : {e}")

# Initialize VectorDB and LLM
create_vectordb()
if USE_OPENAI:
    init_openai_llm()
else:
    init_local_llm('orca_mini_v3_7b.Q4_K_M.gguf')
init_llm()
