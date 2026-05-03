import os
import streamlit as st
import json

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.chains import ConversationalRetrievalChain
from langchain_community.memory import ConversationBufferMemory 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_training_data(file_path):
    abs_path = os.path.join(BASE_DIR, file_path)
    with open(abs_path, "r") as file:
        dataset = json.load(file)
    return dataset


def convert_to_documents(dataset):
    document_list = []
    for item in dataset:
        question_text = item["question"]
        answer_text = item["answer"]
        category_label = item["category"]
        combined_text = f"Question: {question_text}\nAnswer: {answer_text}"
        document = Document(
            page_content=combined_text,
            metadata={"category": category_label}
        )
        document_list.append(document)
    return document_list


def load_sop_document(file_path):
    abs_path = os.path.join(BASE_DIR, file_path)
    with open(abs_path, "r") as file:
        sop_text = file.read()
    sop_document = Document(
        page_content=sop_text,
        metadata={"source": "SOP"}
    )
    return sop_document


def split_documents_into_chunks(document_list):
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = text_splitter.split_documents(document_list)
    return split_docs


def create_vector_database(split_documents):
    embedding_model = OllamaEmbeddings(model="llama3:latest")
    vector_database = Chroma.from_documents(
        documents=split_documents,
        embedding=embedding_model,
        persist_directory=os.path.join(BASE_DIR, "chroma_db")
    )
    return vector_database


def build_conversational_qa_system(vector_database, memory):
    language_model = Ollama(model="llama3")
    retriever_object = vector_database.as_retriever(
        search_kwargs={"k": 3}
    )
    qa_system = ConversationalRetrievalChain.from_llm(
        llm=language_model,
        retriever=retriever_object,
        memory=memory,
        return_source_documents=True,
        verbose=False
    )
    return qa_system


def get_relevance_scores(vector_db, query, k=3):
    results = vector_db.similarity_search_with_relevance_scores(query, k=k)
    scored_docs = []
    for doc, score in results:
        relevance_percent = round(score * 100, 1)
        scored_docs.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "relevance": relevance_percent
        })
    return scored_docs


def display_confidence_ui(overall_confidence, scored_docs):
    st.markdown("#### Confidence Score")
    col1, col2 = st.columns([1, 3])

    with col1:
        st.metric(
            label="Confidence",
            value=f"{overall_confidence}%"
        )

    with col2:
        st.progress(overall_confidence / 100)

    with st.expander("Source Relevance Breakdown"):
        for i, doc in enumerate(scored_docs):
            relevance = doc["relevance"]
            st.markdown(f"**Source {i+1}** — Relevance: {relevance}%")
            st.progress(relevance / 100)
            st.caption(doc["content"][:300] + "...")
            st.markdown("---")


def main():
    st.set_page_config(page_title="AI Quality Chatbot", layout="wide")
    st.title("AI-Powered Quality FAQ Chatbot")

    st.sidebar.title("Settings")
    show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)
    st.sidebar.markdown("---")

    if st.sidebar.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        st.rerun()

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    @st.cache_resource
    def initialize_vector_db():
        training_data = load_training_data("data/train_dataset.json")
        documents_from_dataset = convert_to_documents(training_data)
        sop_document = load_sop_document("data/sop.txt")
        all_documents = documents_from_dataset + [sop_document]
        split_docs = split_documents_into_chunks(all_documents)
        vector_db = create_vector_database(split_docs)
        return vector_db

    vector_db = initialize_vector_db()
    qa_system = build_conversational_qa_system(vector_db, st.session_state.memory)

    st.subheader("Ask a Question")
    user_query = st.text_input("Type your question here:")

    if user_query:
        with st.spinner("Thinking..."):
            result = qa_system({"question": user_query})
            scored_docs = get_relevance_scores(vector_db, user_query, k=3)

        answer_text = result["answer"]
        source_documents = result["source_documents"]

        if scored_docs:
            overall_confidence = round(
                sum(d["relevance"] for d in scored_docs) / len(scored_docs), 1
            )
        else:
            overall_confidence = 0.0

        st.session_state.chat_history.append({
            "question": user_query,
            "answer": answer_text,
            "sources": source_documents,
            "scored_docs": scored_docs,
            "confidence": overall_confidence
        })

    st.markdown("---")
    st.subheader("Conversation History")

    if not st.session_state.chat_history:
        st.info("No conversation yet. Ask a question above to get started.")

    for entry in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {entry['question']}")
        st.markdown(f"**Bot:** {entry['answer']}")

        if show_confidence:
            display_confidence_ui(entry["confidence"], entry["scored_docs"])

        st.markdown("---")


if __name__ == "__main__":
    main()