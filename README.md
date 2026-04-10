# ⚡ Sovereign Analyst

**Command-grade portfolio intelligence for fundamental research.**

Sovereign Analyst is a local-first investment research cockpit designed for fundamental analysts. It bridges the gap between raw SEC filings and portfolio performance, using LLMs to perform "Surgical Delta" analysis on risk disclosures and providing a "Truth Layer" to audit AI-generated claims against original source text.

---

## 🚀 Features

### 🔬 The Truth Layer
*   **Surgical Delta Analysis**: Automatically compares the "Risk Factors" (Item 1A) of the latest 10-K/Q against previous filings. It identifies **Added**, **Removed**, and **Softened** language to reveal shifting corporate risks that traditional screens miss.
*   **Source-Trace Audit**: Every AI-generated risk claim is backed by a "Source-Trace." The system queries a local **ChromaDB** vector store to find the exact paragraph in the SEC filing that supports the claim, providing a confidence score and cosine distance for full auditability.

### 📈 Portfolio Intelligence
*   **ACB Accounting Engine**: A rigorous Average Cost Basis (ACB) engine that handles buys, sells, dividends, and splits in chronological order to ensure precise P&L tracking.
*   **Sovereign Signal**: A unified visualization combining price action (OHLCV) with AI-generated sentiment overlays and analyst notes.
*   **Investment Memo Export**: Generate professional PDF investment memos that combine current holdings, performance metrics, and the latest Surgical Delta verdict.

### ⚙️ Local-First Data Pipeline
*   **EDGAR Sync**: Integrated SEC downloader that fetches 10-K and 10-Q filings directly from the SEC EDGAR database.
*   **Automated Preprocessing**: Cleans and sections raw filings, extracting high-signal sections for AI analysis.
*   **SQLite + ChromaDB**: All transaction data and vector embeddings are stored locally, ensuring your research and trade history remain private.

---

## 🛠️ Getting Started

### Prerequisites
*   **Python 3.14+**
*   **Gemini API Key** (For Surgical Delta and AI Note generation)
*   **SEC User Agent** (Required by SEC EDGAR for filing downloads)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/sovereign-analyst.git
    cd sovereign-analyst
    ```

2.  **Set up the virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables**:
    Create a `.env` file in the root directory:
    ```env
    GEMINI_API_KEY=your_api_key_here
    USER_AGENT="Your Name your@email.com"
    ```

### Running the Cockpit

Launch the Streamlit dashboard:
```bash
streamlit run app.py
