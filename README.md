# Semantic-Research-Aid

This project is a web-based research assistant that allows users to upload PDFs and search for relevant ideas or concepts. It highlights key sentences semantically related to a user's input and renders both the PDF and results side by side.

---

## Features

- Upload multiple research papers in PDF format
- Input a concept/idea to search for
- Uses sentence-transformers for semantic similarity
- Highlights the top relevant sentences dynamically

---

## how to run
```
git clone https://github.com/hperumalla-oops/Semantic-Research-Aid.git
cd Semantic-Research-Aid
pip install -r requirements.txt
uvicorn app:app --reload
```



