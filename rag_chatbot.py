import os
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

load_dotenv()

# Gemini API
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Gemini API key not found!")
    exit()

client = genai.Client(api_key=api_key)

# Load document
with open("document.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Split document into chunks
chunk_size = 300
overlap = 50

chunks = []

for i in range(0, len(text), chunk_size - overlap):
    chunk = text[i:i + chunk_size]
    if chunk.strip():
        chunks.append(chunk)

print("Document loaded successfully!")
print("Total chunks:", len(chunks))

# Create embeddings
print("Creating embeddings...")

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# Create FAISS vector database
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("Vector database created successfully!")

# Start chatbot
print("\n================================")
print("DOCUMENT INTELLIGENCE CHATBOT")
print("================================")
print("Type 'exit' to stop.\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    # Convert question into embedding
    question_embedding = model.encode([question])

    # Search similar chunks
    distances, indices = index.search(question_embedding, k=3)

    relevant_text = "\n\n".join(
        chunks[i] for i in indices[0]
    )

    # Send retrieved information to Gemini
    prompt = f"""
You are a Document Intelligence Assistant.

Answer the user's question using ONLY the information provided
in the document context below.

If the answer is not present in the document, say:
"I could not find this information in the document."

DOCUMENT CONTEXT:
{relevant_text}

USER QUESTION:
{question}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    print("\nChatbot:", response.text)
    print()