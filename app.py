import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

################################
# Pinecone v6 方式
################################
from pinecone import Pinecone, ServerlessSpec

################################
# langchain関連
################################
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import OpenAI
# langchain-pinecone 新方式
from langchain_pinecone import PineconeVectorStore

from langchain.chains import ConversationalRetrievalChain

INDEX_NAME = "concur-index"
NAMESPACE  = "demo-html"

def main():
    st.title("Concur Helper - RAG Chatbot")

    # 1. Pinecone インスタンスを作成（init() は使わない）
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
    pinecone_env     = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")

    # Pinecone() コンストラクタで認証
    pc = Pinecone(
        api_key=pinecone_api_key,
        # environment=pinecone_env,   # ※regionは ServerlessSpec などで指定も可能
        # project_name="あなたのプロジェクト名" など
    )

    # インデックスオブジェクトを取得
    # すでにIndexがあるなら create_index 等は不要
    my_index = pc.Index(INDEX_NAME)

    # 2. Embeddings
    openai_key = os.getenv("OPENAI_API_KEY", "")
    embeddings = OpenAIEmbeddings(api_key=openai_key)

    # 3. VectorStore
    #   PineconeVectorStore(index=...) に APIキー等を渡すやり方は v6 で廃止され、
    #   代わりに上記 Pineconeインスタンス→Index を使う
    docsearch = PineconeVectorStore(
        embedding = embeddings,
        index     = my_index,       # pinecone.Index オブジェクト
        namespace = NAMESPACE
    )

    # 4. LLM
    llm = OpenAI(api_key=openai_key, temperature=0)

    # 5. ConversationalRetrievalChain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=docsearch.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    # 6. 会話ループ
    if "history" not in st.session_state:
        st.session_state["history"] = []

    query = st.text_input("質問を入力してください:")

    if query:
        result = qa_chain({
            "question": query,
            "chat_history": st.session_state["history"]
        })

        answer = result["answer"]
        st.write("## 回答")
        st.write(answer)

        if "source_documents" in result:
            st.write("### 参照チャンク:")
            for doc in result["source_documents"]:
                st.write(f"- {doc.metadata}")

        st.session_state["history"].append((query, answer))

if __name__ == "__main__":
    main()
