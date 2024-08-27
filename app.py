import streamlit as st
import os
import glob
from dotenv import load_dotenv
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured.staging.base import dict_to_elements
from unstructured.chunking.title import chunk_by_title
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
from typing import List, Tuple
import re
from rank_bm25 import BM25Okapi

# Load environment variables
load_dotenv()

# Get unstructured API key
unstructured_api_key = os.getenv("UNSTRUCTURED_API_KEY")
if not unstructured_api_key:
    raise ValueError("UNSTRUCTURED_API_KEY environment variable not found")

class EnhancedRetriever:
    def __init__(self, vectorstore, documents):
        self.vectorstore = vectorstore
        self.documents = documents
        self.bm25 = self._create_bm25_index()

    def _create_bm25_index(self):
        tokenized_docs = [self._tokenize(doc.page_content) for doc in self.documents]
        return BM25Okapi(tokenized_docs)

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r'\b\d+(?:\.\d+)*(?:\s+[A-Za-z]+(?:\s+[A-Za-z]+)*)?\b|\w+', text.lower())
        return tokens

    def hybrid_search(self, query: str, k: int = 4) -> List[Tuple[float, Document]]:
        vector_results = self.vectorstore.similarity_search_with_score(query, k=k)
        keyword_results = self.keyword_search(query, k=k)
        
        combined_results = {}
        query_keywords = set(query.lower().split())
        
        for doc, score in vector_results:
            combined_results[doc.page_content] = {'doc': doc, 'vector_score': score, 'keyword_score': 0, 'exact_match': False}
            doc_words = set(doc.page_content.lower().split())
            if query_keywords.issubset(doc_words):
                combined_results[doc.page_content]['exact_match'] = True
        
        for score, doc in keyword_results:
            if doc.page_content in combined_results:
                combined_results[doc.page_content]['keyword_score'] = score
            else:
                combined_results[doc.page_content] = {'doc': doc, 'vector_score': 0, 'keyword_score': score, 'exact_match': False}
            
            doc_words = set(doc.page_content.lower().split())
            if query_keywords.issubset(doc_words):
                combined_results[doc.page_content]['exact_match'] = True
        
        final_results = []
        for content, scores in combined_results.items():
            normalized_vector_score = 1 / (1 + scores['vector_score'])
            normalized_keyword_score = scores['keyword_score']
            exact_match_bonus = 2 if scores['exact_match'] else 0
            combined_score = (normalized_vector_score + normalized_keyword_score + exact_match_bonus) / 3
            final_results.append((combined_score, scores['doc']))
        
        return sorted(final_results, key=lambda x: x[0], reverse=True)[:k]

    def keyword_search(self, query: str, k: int = 4) -> List[Tuple[float, Document]]:
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        scored_docs = [(score, self.documents[i]) for i, score in enumerate(bm25_scores)]
        return sorted(scored_docs, key=lambda x: x[0], reverse=True)[:k]

def process_pdfs_and_cache(input_folder, output_folder, strategy):
    # Initialize the UnstructuredClient
    s = UnstructuredClient(api_key_auth=unstructured_api_key, server_url='https://redhorse-d652ahtg.api.unstructuredapp.io')

    os.makedirs(output_folder, exist_ok=True)
    folder_name = os.path.basename(os.path.normpath(input_folder))
    cache_file_path = os.path.join(output_folder, f'{folder_name}_combined_content.json')

    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'r', encoding='utf-8') as f:
            combined_content = json.load(f)
    else:
        combined_content = []
        for filename in glob.glob(os.path.join(input_folder, "*.pdf")):
            with open(filename, "rb") as file:
                req = shared.PartitionParameters(
                    files=shared.Files(
                        content=file.read(),
                        file_name=filename,
                    ),
                    strategy=strategy,
                )
                res = s.general.partition(req)
                combined_content.extend(res.elements)

        with open(cache_file_path, 'w', encoding='utf-8') as f:
            json.dump(combined_content, f)

    return combined_content

def process_data(combined_content):
    pdf_elements = dict_to_elements(combined_content)
    elements = chunk_by_title(pdf_elements, combine_text_under_n_chars=4000, max_characters=8000, new_after_n_chars=7000, overlap=1000)
    documents = []
    for element in elements:
        metadata = element.metadata.to_dict()
        metadata.pop("languages", None)
        metadata["source"] = metadata["filename"]
        documents.append(Document(page_content=element.text, metadata=metadata))
    return documents

@st.cache_resource
def initialize_retriever(folder_path):
    strategy = "auto"
    combined_content = process_pdfs_and_cache(folder_path, "./cache", strategy)
    documents = process_data(combined_content)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return EnhancedRetriever(vectorstore, documents)

def organize_documents(docs):
    organized_text = ""
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get('source', 'unknown source')
        page_number = doc.metadata.get('page_number', 'unknown page number')
        organized_text += f"Document {i}:\nSource: {source}\nPage number: {page_number}\nContent: {doc.page_content}\n\n"
    return organized_text

def create_llm(model_name: str, streaming: bool = False):
    return ChatOpenAI(model_name=model_name, temperature=0, streaming=streaming)

def generate_answer(query: str, relevant_data: str, llm: ChatOpenAI):
    prompt = PromptTemplate(
        template="""
        Based on the following relevant data, please answer the user's query.
        Provide a comprehensive and accurate answer, using the information given.
        If the information is not sufficient to answer the query fully, state so clearly.
        Sometime, the full section got cut off or break down to separated parts. Carefully look at the relevant data and connect them if they belong to each other. Pay attention to the sections number to figure it out.

        The answer should have 3 parts:
        1. An easy to understand answer/explanation with a concise, short example.
        2. The exact wording and format from the original document, ensuring the full section is included (for citing purposes).
        3. If the content refers to any other section or clause(s), state it out to the user. ex: "This section also references: ...."

        Relevant data:
        {relevant_data}
        
        User query: {query}
        
        Answer in a nice markdown format:
        """,
        input_variables=["query", "relevant_data"],
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query, "relevant_data": relevant_data})

def extract_search_keywords(query: str, llm: ChatOpenAI) -> str:
    prompt = PromptTemplate(
        template="""
        Extract the most relevant search keywords or phrases from this query for searching through Federal Acquisition Regulation (FAR) documents.
        Focus on specific terms, section numbers, or phrases that are likely to yield the most relevant results.
        Return your answer as a comma-separated string of keywords.

        User query: {query}
        """,
        input_variables=["query"],
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"query": query})
    return result.strip()

def rag_query_enhanced(user_query: str, enhanced_retriever: EnhancedRetriever, model_name: str = "gpt-4", use_streaming: bool = False, k: int = 4):
    keyword_llm = create_llm(model_name, streaming=False)
    search_keywords = extract_search_keywords(user_query, keyword_llm)
    retrieved_docs = enhanced_retriever.hybrid_search(search_keywords, k=k)
    organized_text = organize_documents([doc for _, doc in retrieved_docs])
    answer_llm = create_llm(model_name, use_streaming)
    answer = generate_answer(user_query, organized_text, answer_llm)
    return answer

def get_cache_folders():
    cache_dir = "./cache"
    return [f for f in os.listdir(cache_dir) if f.endswith('_combined_content.json')]

# Streamlit app
def main():
    st.title("FAR Query Assistant")
    st.write("Ask questions about the Federal Acquisition Regulation (FAR)")

    # Get available folders
    folder_options = get_cache_folders()
    
    if not folder_options:
        st.error("No cached files found in the ./cache directory.")
        return

    # Folder selection with session state
    if 'selected_folder' not in st.session_state:
        st.session_state.selected_folder = None
    selected_folder = st.selectbox("Select file to chat with:", 
                                   folder_options, 
                                   key='folder_selector',
                                   index=folder_options.index(st.session_state.selected_folder) if st.session_state.selected_folder in folder_options else 0)
    
    # Update session state
    st.session_state.selected_folder = selected_folder

    # Display file size information
    cache_file_path = os.path.join("./cache", selected_folder)

    # Initialize the retriever with the selected cache file
    with st.spinner("Initializing system... This may take a while for files large, ~10s for 1000 pages."):
        retriever = initialize_retriever_from_cache(cache_file_path)

    # Query input
    query = st.text_input("Enter your query:  (ex: Tell me about the 1.102-2 Performance standards)")

    # Use session state to store query history
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

    # Model selection
    model_name = st.selectbox("Select model:", ["gpt-4o", "gpt-4o-mini"])

    # Number of documents to retrieve
    k = st.slider("Number of documents to retrieve:", min_value=1, max_value=10, value=4)

    # Submit button
    if st.button("Submit"):
        if query:
            with st.spinner("Processing query..."):
                answer = rag_query_enhanced(query, retriever, model_name=model_name, k=k)
                st.markdown("### Answer:")
                st.markdown(answer)
            
            # Add query to history
            st.session_state.query_history.append(query)

    # Display query history
    if st.session_state.query_history:
        st.subheader("Query History")
        for i, past_query in enumerate(reversed(st.session_state.query_history), 1):
            st.text(f"{i}. {past_query}")

    # Clear history button
    if st.button("Clear History"):
        st.session_state.query_history = []
        st.experimental_rerun()

@st.cache_resource
def initialize_retriever_from_cache(cache_file_path):
    with open(cache_file_path, 'r', encoding='utf-8') as f:
        combined_content = json.load(f)
    documents = process_data(combined_content)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return EnhancedRetriever(vectorstore, documents)

if __name__ == "__main__":
    main()

