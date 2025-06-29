# Import standard and third-party modules
from typing import Dict, List

# LangChain utilities for text splitting, vector store, and web page loading
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import SeleniumURLLoader

# Embeddings and LLM models specifically from the langchain-ollama integration
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM

# LangChain core components for chaining, prompt handling, and output parsing
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# Define the LLM model name used for answering questions
model_name = "llama3.2" 

# List of URLs to scrape content from
documents = [
    "https://beebom.com/what-is-nft-explained/",
    "https://beebom.com/how-delete-servers-discord/",
    "https://beebom.com/how-list-groups-linux",
    "https://beebom.com/how-open-port-linux",
    "https://beebom.com/linux-vs-windows/",
]

# Function to scrape content from the provided URLs using a headless browser
def scrape_docs(urls: List[str]) -> List[Dict]:
    try:
        # Use Selenium-based loader to fetch web pages
        loader = SeleniumURLLoader(urls=urls)
        raw_docs = loader.load()
        
        # Print info about each document fetched
        print(f"\nSuccessfully loaded {len(raw_docs)} documents")
        for doc in raw_docs:
            print(f"\nSource: {doc.metadata.get('source', 'No source')}")
            print(f"Content length: {len(doc.page_content)} characters")
        
        return raw_docs
    except Exception as e:
        # If Selenium or another dependency is missing, show an error
        print(f"Error during document loading: {str(e)}")
        return []

# Function to split each document into manageable chunks for embedding
def split_documents(pages_content: List[Dict]) -> tuple:
    # Use recursive splitter to keep semantic meaning intact while splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_texts, all_metadatas = [], []

    # Iterate through each document and split into chunks
    for document in pages_content:
        text = document.page_content
        source = document.metadata.get("source", "")
        chunks = text_splitter.split_text(text)

        # Store each chunk with its source metadata
        for chunk in chunks:
            all_texts.append(chunk)
            all_metadatas.append({"source": source})

    print(f"Created {len(all_texts)} chunks of text")
    return all_texts, all_metadatas

# Create a Chroma vector store using OllamaEmbeddings
def create_vector_store(texts: List[str], metadatas: List[Dict]):
    # Generate embeddings for the text chunks using Ollama's embedding model
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Create the Chroma vector store with texts and metadata
    db = Chroma.from_texts(texts=texts, metadatas=metadatas, embedding=embeddings)
    return db

# Setup the full LangChain retriever + LLM QA pipeline
def setup_qa_chain(db):
    # Initialize the Ollama LLM with the specified model
    llm = OllamaLLM(model=model_name, temperature=0)

    # Convert the vector DB into a retriever interface
    retriever = db.as_retriever()

    # Define a custom prompt that ensures polite and helpful tone
    prompt = ChatPromptTemplate.from_template(
        """
        Please provide a polite and helpful response to the following question, utilizing the provided context. 
        Ensure that the tone remains professional, courteous, and empathetic, and tailor your response.

        ### Context:
        {context}

        ### Question:
        {question}

        ### Polite Response:
        In your response, consider including:
        - Acknowledge the user's query and express gratitude for the opportunity.
        - Provide a clear and concise answer that directly addresses the query.
        - Use positive language and maintain a supportive tone throughout.
        - If applicable, include relevant information or resources that could help resolve query.
        - Conclude by inviting any follow-up questions or providing encouragement to ask more questions.
        """
    )

    # Build the chain:
    # 1. Fetch context using retriever
    # 2. Format the prompt with context and question
    # 3. Generate response using LLM
    # 4. Parse and clean the response to plain string
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

# Handle a user query using the LLM + retriever chain
def process_query(chain_and_retriever, query: str):
    try:
        chain, retriever = chain_and_retriever

        # Get the response from the chain
        response = chain.invoke(query)

        # Also fetch the documents used for context
        docs = retriever.invoke(query)

        # Extract source URLs from metadata
        sources_str = ", ".join([doc.metadata.get("source", "") for doc in docs])

        return {"answer": response, "sources": sources_str}
    except Exception as e:
        # In case of any error, provide a fallback message
        print(f"Error processing query: {str(e)}")
        return {
            "answer": "I apologize, but I encountered an error while processing your query!",
            "sources": "",
        }

# The main function orchestrates the full chatbot setup and input loop
def main():
    print("Scraping documents...")
    pages_content = scrape_docs(documents)

    print("Splitting documents...")
    all_texts, all_metadatas = split_documents(pages_content)

    print("Creating vector store...")
    db = create_vector_store(all_texts, all_metadatas)

    print("Setting up QA chain...")
    qa_chain = setup_qa_chain(db)

    print("\nReady for questions! (Type 'quit' to exit)")
    
    # Command-line loop for user to ask multiple questions
    while True:
        query = input("\nEnter your question: ").strip()
        if not query:
            continue
        if query.lower() == 'quit':
            break

        # Process the question and display the result
        result = process_query(qa_chain, query)
        print("\nResponse: ")
        print(result["answer"])
        if result["sources"]:
            print("\nSources: ")
            for source in result["sources"].split(","):
                print("- "+ source.strip())

# Entry point for script execution
if __name__ == "__main__":
    main()
