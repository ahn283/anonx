from langchain_community.chat_models import ChatOllama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import streamlit as st 

# models
llm = ChatOllama(model='llama3.1')
embedder = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')
vectorstore_path = './vectorstore'
vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embedder)
retriever = vectorstore.as_retriever(kwargs={'k':3})

# documents formatting
def format_docs(docs):
    result = ''
    for doc in docs:
        result += '\n\n'.join([doc.page_content])
    return result

# interface
# title for the chatbot
st.title("Llama with RAG")

# check if chat history exists
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
# show chat history
for content in st.session_state.chat_history:
    with st.chat_message(content['role']):
        st.markdown(content['message'])
        
if question := st.chat_input("Input your message."):
    with st.chat_message("user"):
        st.markdown(question)
        st.session_state.chat_history.append(
            {'role': 'user', 'message': question}
        )
    
    # RAG retrieval
    retrieved = retriever.invoke(question)
    docs = format_docs(retrieved)
     
    # prompt engineering
    prompt = f"""You are a helpful assistant chabot. Answer only in English and based on the documents provided.

    ### documents:
    {docs}

    ### question:
    {question}
    """
    
    # get response from LLM model
    response = llm.invoke(prompt)
    
    with st.chat_message('assistant'):
        st.markdown(response.content)
        
        st.session_state.chat_history.append({
            'role':'assistant', 'message':response.content
        })