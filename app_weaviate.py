import os
import streamlit as st
from dotenv import load_dotenv

import weaviate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_URL   = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

def main():
    st.title("Weaviate + LangChain Demo")

    # Weaviateクライアント (v3 REST想定)
    client = weaviate.Client(
        url=WEAVIATE_URL,
        additional_headers={
            "Authorization": f"Bearer {WEAVIATE_API_KEY}"
        }
    )

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # VectorStoreとしてWeaviateを使用
    vectorstore = Weaviate(
        client=client,
        index_name="Document",  # Weaviate Class
        text_key="chunkText",   # テキストを保持しているプロパティ
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k":3})

    # RetrievalQAチェーンを生成
    chat_llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_llm,
        chain_type="stuff",
        retriever=retriever
    )

    query = st.text_input("質問を入力してください:")
    if st.button("送信"):
        if query.strip():
            with st.spinner("回答を生成中..."):
                answer = qa_chain.run(query)
            st.write("### 回答")
            st.write(answer)

if __name__ == "__main__":
    main()
