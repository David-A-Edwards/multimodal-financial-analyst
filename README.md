# Multimodal Financial Analyst 📊

A sophisticated RAG system capable of parsing and reasoning over complex financial documents (PDFs) containing charts, infographics, and tables.

## 🚀 Capabilities

* **Dual-Store Architecture:** Separates text and images into distinct vector stores for high-fidelity retrieval.
* **Visual Reasoning:** Uses **Gemini 3.0 Pro** to "see" charts and graphs, allowing it to extract data that standard text parsers miss (e.g., "Market Size" bubbles).
* **Advanced Parsing:** Integrates **LlamaParse Premium** to accurately reconstruct document layout.
* **Cognitive Analysis:** Implements a 5-step reasoning protocol (Spatial Locking, Semantic Disambiguation) to prevent hallucinations on complex pages.

## 🛠️ Tech Stack

* **Framework:** LlamaIndex
* **Vision Model:** Gemini 3.0 Pro Preview / Gemini 1.5 Pro
* **Parsing:** LlamaParse (Multimodal Mode)
* **Vector DB:** Qdrant (In-Memory)
* **Frontend:** Streamlit

## 💻 How to Run

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/YOUR-USERNAME/multimodal-financial-analyst.git](https://github.com/YOUR-USERNAME/multimodal-financial-analyst.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Setup Secrets:**
    Create a `.env` file with:
    ```ini
    LLAMA_CLOUD_API_KEY=llx-...
    GOOGLE_API_KEY=AIza...
    ```
4.  **Run:**
    ```bash
    streamlit run app.py
    ```
