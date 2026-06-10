# RAG-based-Medical-Chatbot

# How to run?
### STEPS:

Clone the repository

```bash
git clone https://github.com/priyansh-iitd26/RAG-based-Medical-Chatbot.git
```
### STEP 1: Create a conda environment after opening the repository

```bash
conda create -n venv python=3.10 -y
```

```bash
conda activate venv
```

### STEP 2: Install the requirements
```bash
pip install -r requirements.txt
```

### Create a `.env` file in the root directory and add your Pinecone & openai credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Streamlit
- OpenRouter API
- Pinecone
- Hugging Face SentenceTransformer

