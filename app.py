from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import fitz  # PyMuPDF
import os
import re
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
nltk.download('punkt_tab')


app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploaded_pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.mount("/uploaded_pdfs", StaticFiles(directory="uploaded_pdfs"), name="uploaded_pdfs")


model = SentenceTransformer("all-MiniLM-L6-v2")

# Make sure NLTK punkt tokenizer is downloaded
nltk.download('punkt')

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze/")
async def analyze_pdfs(
    idea: str = Form(...),
    files: list[UploadFile] = File(...)
):
    results = []

    idea_embedding = model.encode(idea, convert_to_tensor=True)

    for file in files:
        file_location = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())

        doc = fitz.open(file_location)
        full_text = ""
        page_texts = []
        for page in doc:
            text = page.get_text()
            full_text += text + "\n"
            page_texts.append(text)

        # Use nltk to split into sentences
        sentences = sent_tokenize(full_text)
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        similarities = util.cos_sim(idea_embedding, sentence_embeddings)[0]
        top_indices = similarities.topk(5).indices.tolist()
        top_sentences = [sentences[i].strip() for i in top_indices]
        score = float(similarities.mean().item())

        # Find sentence positions in text for frontend highlighting
        highlight_positions = []
        for sentence in top_sentences:
            for match in re.finditer(re.escape(sentence), full_text):
                highlight_positions.append({
                    "start": match.start(),
                    "end": match.end(),
                    "sentence": sentence
                })

        results.append({
            "file": file.filename,
            "path": f"/{file_location}",
            "highlights": highlight_positions,
            "score": round(score * 100, 2)
        })

    return JSONResponse(content={"results": results})
