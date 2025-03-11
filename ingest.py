import os
import pinecone
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

INDEX_NAME = "concur-index"
NAMESPACE  = "demo-html"
DOC_FILENAME = "Exp_SG_Account_Codes-jp.txt"
DOCS_FOLDER  = "./docs"

def main():
    openai_key   = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

    pinecone.init(api_key=pinecone_key, environment=pinecone_env)
    existing_indexes = pinecone.list_indexes()
    if INDEX_NAME not in existing_indexes:
        pinecone.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine"
        )

    file_path = f"{DOCS_FOLDER}/{DOC_FILENAME}"
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    docs = []
    for i, chunk_text in enumerate(chunks):
        docs.append(Document(
            page_content=chunk_text,
            metadata={"filename": DOC_FILENAME, "chunk_index": i}
        ))

    embeddings = OpenAIEmbeddings(api_key=openai_key)
    my_index = pinecone.Index(INDEX_NAME)
    PineconeVectorStore.from_documents(
        documents = docs,
        embedding = embeddings,
        index     = my_index,
        namespace = NAMESPACE
    )

    print("Ingestion completed!")

if __name__ == "__main__":
    main()
