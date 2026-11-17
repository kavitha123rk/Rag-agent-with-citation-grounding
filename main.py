
import ollama
import chromadb

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# STEP 1: Load the embedding model
print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# STEP 2: Read the PDF
print("\nReading PDF...")

pdf = PdfReader("documents/sample.pdf")

print("Total Pages:", len(pdf.pages))


# STEP 3: Split PDF into chunks and save page numbers

chunk_size = 500

chunks = []
metadatas = []

for page_number, page in enumerate(pdf.pages, start=1):

    page_text = page.extract_text()

    if page_text:

        for i in range(0, len(page_text), chunk_size):

            chunk = page_text[i:i + chunk_size]

            chunks.append(chunk)

            metadatas.append({
                "source": "sample.pdf",
                "page": page_number
            })

print("\nTotal chunks:", len(chunks))
print("First chunk metadata:", metadatas[0])


# STEP 4: Convert chunks into embeddings

print("\nCreating embeddings...")

embeddings = model.encode(chunks)

print("Total embeddings:", len(embeddings))


# STEP 5: Store chunks in ChromaDB

print("\nStoring chunks in ChromaDB...")

client = chromadb.PersistentClient(path="./vectorstore")

collection = client.get_or_create_collection(
    name="college_notes"
)

collection.upsert(
    ids=[str(i) for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)

print("Successfully stored chunks!")
print("Total stored documents:", collection.count())


# STEP 6: Ask a question

question = "What is Artificial Intelligence?"
print("\nQuestion:", question)


# STEP 7: Convert question into an embedding

question_embedding = model.encode(question)


# STEP 8: Retrieve relevant information

results = collection.query(
    query_embeddings=[question_embedding.tolist()],
    n_results=3
)

retrieved_documents = results["documents"][0]
retrieved_metadatas = results["metadatas"][0]
retrieved_distances = results["distances"][0]

print("\nRetrieved information:")

for document, metadata in zip(
    retrieved_documents, retrieved_metadatas
):
    print("\n", document)
    print("Source:", metadata["source"])
    print("Page:", metadata["page"])


# STEP 9: Check whether relevant information was found

max_distance = 1.0

if not retrieved_documents or min(retrieved_distances) > max_distance:

    print("\nAI Answer:")
    print("I couldn't find this information in your PDF.")

else:

    # STEP 10: Generate an answer using Ollama

    pdf_information = "\n\n".join(
        [
            f"Source: {metadata['source']}, "
            f"Page: {metadata['page']}\n{document}"
            for document, metadata in zip(
                retrieved_documents, retrieved_metadatas
            )
        ]
    )

    response = ollama.chat(
        model="qwen2.5:1.5b",
        messages=[
            {
                "role": "user",
                "content": (
                    "You are a PDF question-answering assistant.\n"
                    "Answer only using the provided PDF information.\n"
                    "Do not use outside knowledge.\n"
                    "If the answer is not clearly available, "
                    "reply exactly: "
                    "I couldn't find this information in your PDF.\n"
                    "Include the source filename and page number "
                    "for every answer.\n\n"
                    "PDF Information:\n"
                    + pdf_information
                    + "\n\nQuestion: "
                    + question
                )
            }
        ]
    )

    print("\nAI Answer:")
    print(response["message"]["content"])
