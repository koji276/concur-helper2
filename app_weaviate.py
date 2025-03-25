import os
import streamlit as st
from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
WEAVIATE_URL   = os.getenv("WEAVIATE_URL", "")     # v3 REST endpoint (e.g. "https://xxx.weaviate.cloud")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")

def main():
    st.title("Weaviate (v3) + LangChain Demo")

    # 1) Weaviate v3 クライアントを初期化
    #    AuthApiKey を使ってAPIキーを設定
    auth_config = AuthApiKey(api_key=WEAVIATE_API_KEY)
    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=auth_config,
        additional_headers={
            # 必要に応じて追加
            "X-OpenAI-Api-Key": OPENAI_API_KEY
        }
        # skip_ssl_verify=True, などもオプションで指定可
    )
    # この時点で v3 の REST endpoint にアクセス

    # 2) LangChain の Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # 3) Weaviate VectorStore (v3 style)
    #    index_name : Weaviateのクラス名 (例: "Document")
    #    text_key   : テキストが格納されているプロパティ (例: "chunkText")
    vectorstore = Weaviate(
        client=client,
        index_name="Document",
        text_key="chunkText",
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k":3})

    # 4) ConversationalRetrievalChain を生成
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4", temperature=0)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # Streamlit UI
    if "history" not in st.session_state:
        st.session_state["history"] = []

    user_query = st.text_input("質問を入力してください:")
    if st.button("送信"):
        if user_query.strip():
            with st.spinner("回答を生成中..."):
                # ConversationalRetrievalChainにhistoryを渡す
                result = chain({"question": user_query, "chat_history": st.session_state["history"]})

            answer = result["answer"]
            source_docs = result.get("source_documents", [])

            # historyを更新
            st.session_state["history"].append((user_query, answer))

            st.write("### 回答")
            st.write(answer)

            # 参照文書情報を表示
            if source_docs:
                st.write("### 参照元ドキュメント")
                for i, doc in enumerate(source_docs, start=1):
                    st.write(f"**Doc {i}:**", doc.metadata)

if __name__ == "__main__":
    main()

