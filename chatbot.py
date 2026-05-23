import datetime
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# 1. Greeting based on time
name = input("Enter your name: ")
presentHour = datetime.datetime.now().hour

if 6 <= presentHour <= 12:
    print("Good morning", name)
elif 13 <= presentHour <= 20:
    print("Good afternoon", name)
elif 20 <= presentHour <= 23:   # fixed: 0–23 range only
    print("Good evening", name)
else:
    print("Good night", name)

print("Hello Dear, welcome to your personal chatbot!")

# 2. Load PDF
pdf_path = r"C:\Users\umeru\Downloads\Gale Encyclopedia of Medicine Vol. 1 (A-B).pdf"
pdf_reader = PyPDF2.PdfReader(open(pdf_path, "rb"))

text = ""
for page in pdf_reader.pages:
    page_text = page.extract_text()
    if page_text:   # avoid NoneType error
        text += page_text

# 3. Split into chunks
chunks = [text[i:i+500] for i in range(0, len(text), 500)]

# 4. Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

# 5. Store in FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)   # embeddings already np.ndarray

# 6. Chatbot response function
def chatbot_response(query):
    query_vec = model.encode([query])   # shape (1, dimension)
    D, I = index.search(query_vec, k=1) # top 1 match
    return chunks[I[0][0]]

# 7. Chat loop
print("Welcome to your chatbot!")
while True:
    userInput = input("Ask a question: ")
    if "bye" in userInput.lower():
        print("Goodbye! Have a nice day.")
        break
    reply = chatbot_response(userInput)
    print("Bot response:", reply)