# MedPrompt: A python implementation of medprompt for Ollama models.
---
## 1. Project Overview
This project is a Python-based implementation of [MedPrompt](https://arxiv.org/pdf/2311.16452) for efficient use with Ollama models.

## 2. Getting Started
Follow these steps to set up MedPrompt locally for development and testing.

### 2.1 Prerequisites
- `Python`: 3.9 or higher
- `pip`: Python package manager
- `Ollama`: Download and install the Ollama platform (for Mac or other compatible platforms)

### 2.2 Installation (Mac)
1. **Intall `Ollama` Platform:**
    - step 1: Visit `https://ollama.com/`
    - step 2: Download and install Ollama for Mac (or other supported platforms).
2. **Clone the Repository:**
    ```bash
    git clone https://github.com/fangyizhang-x/medprompt.git
    ```
3. **Set Up a Virtual Environment `.venv`:**
    ```bash
    cd medprompt
    python3 -m venv .venv
    source .venv/bin/activate
    ```
4. **Install dependencies in the Virtual Environment:**
    ```bash
    pip install -r requirements.txt
    ```

## 3. Running the Code
After setting up the environment and installing dependencies, follow these steps to initialize the required models and execute MedPrompt.
1. **Pull the Necessary Models:**
Before running MedPrompt, use the ollama pull command to load the following models. Each model serves a specific purpose for embedding and text processing:
    ```bash
    ollama pull llama3.2:1b # Main model for generating embeddings
    ollama pull all-minilm:22m # Lightweight model for smaller embeddings
    ollama pull all-minilm:33m # Medium-sized model for general embeddings
    ollama pull nomic-embed-text # Embedding model for text-specific processing
    ```
2. **Run the Preprocessing and Evaluation Scripts:**
With the models pulled, you can now preprocess and evaluate data using specified configuration files.
    ```bash
    python preprocessing.py configs/default.yaml # Preprocess data for MedPrompt with default settings
    python evaluate.py configs/default.yaml # Evaluat MedPrompt performance with default settings
    ```

## 4. Folder Structure
1. **Configs**
    - `default.yaml`: Default configuration file which uses all-minilm:22m for embedding.
    - `minilm33m.yaml`: Configuration specific to the minilm33m model for medprompt usage.
    - `nomic-embed-text.yaml`: Configuration file for the nomic-embed-text embedding model.

2. **data**
Directory to store raw data of MedQA.

3. **medprompt_minilm22m**
Results for the case using `all-minilm:22m` for embedding.

4. **medprompt_minilm33m**
Results for the case using `all-minilm:33m` for embedding.

5. **medprompt_nomic**
Results for the case using `nomic-embed-text` for embedding.