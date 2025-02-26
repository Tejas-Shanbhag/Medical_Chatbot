import streamlit as st
from utils import *
from dotenv import load_dotenv

from pinecone import Pinecone 



st.set_page_config(layout='wide')

st.markdown("""
    <style>
    .stForm {
        background-color: #ADD8E6;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Medical Chatbot 💉🚑</h1>", unsafe_allow_html=True)
st.write(" ")
st.write(" ")

fig1,fig2 = st.columns(2)


with fig1:
        st.image('media/doctor.jpg',use_container_width=True)

with fig2:

    with st.container(height=500):
        with st.chat_message("assistant"):
            st.markdown("Hi I am an AI Chatbot!!!")
                

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    with st.container():
        human_question =  st.chat_input("What is up?")
        if human_question:

            embeddings = get_huggingface_embeddings()
            docstore = get_docstore(collection_name='medical-chatbot',embedding_name=embeddings,api_key=PINECONE_API_KEY)
            retriever = docstore.as_retriever(search_type='similarity',search_kwargs={"k":5})

            llm = load_groq_model(groq_api_key=GROQ_API_KEY)
            prompt = get_prompt_template()
            question_answer_chain = create_stuff_documents_chain(llm,prompt)
            rag_chain = create_retrieval_chain(retriever,question_answer_chain)
            rag_output = rag_chain.invoke({"input" :str(human_question)})
            final_answer = rag_output['answer']

            st.session_state.messages.append({"role": "user", "content": human_question})

            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            st.rerun()


