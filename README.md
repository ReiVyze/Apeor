# 💬 OracleE Sentiment Dashboard

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-red.svg)](https://streamlit.io/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/docs/transformers/index)
[![MongoDB](https://img.shields.io/badge/Database-MongoDB-green.svg)](https://www.mongodb.com/)

An advanced, interactive sentiment analysis platform designed for E-Consultation modules. **OracleE** provides real-time insights into citizen feedback using state-of-the-art NLP models (BERT/RoBERTa), enabling policymakers to make data-driven decisions.

---

## 🌟 Key Features

- **Dual Perspectives**:
  - **👤 Citizen Portal**: Submit feedback on policy clauses with real-time relevance checks.
  - **💼 Policy Dashboard**: Comprehensive analytics for officials with sentiment trends and word frequencies.
- **AI-Powered Analysis**:
  - Multiple model support: **RoBERTa (Hugging Face)**, **Local DistilBERT**, and **Vader (Lexicon)**.
  - Aspect-based sentiment mapping.
- **Dynamic Visualizations**:
  - Interactive sentiment trend charts (Daily shifts).
  - Top feedback themes & Word clouds.
  - Filterable live streams of feedback.
- **Data Integrity**:
  - Robust MongoDB integration for persistent storage.
  - Relevance filtering to ensure feedback matches the selected policy clause.
- **Export Capabilities**: Download filtered analysis reports in CSV format.

---

## 📁 Project Structure

```text
.
├── .streamlit/          # Streamlit configuration
├── data/                # Sample datasets and CSVs
├── model/               # Local DistilBERT model files
├── static/              # CSS and static assets
├── templates/           # HTML templates for custom UI components
├── tests/               # Unit and integration tests
├── app_streamlit.py     # Main application entry point
├── database.py          # MongoDB handler and connection logic
├── model_inference.py   # Sentiment analysis engine and model logic
├── requirements.txt     # Project dependencies
└── .env.example         # Environment variable template
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- MongoDB (Local or Atlas)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/oracle-e-sentiment.git
   cd oracle-e-sentiment
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your MongoDB connection string
   ```

5. **Run the application**:
   ```bash
   streamlit run app_streamlit.py
   ```

---

## 🧪 Testing

The project uses `pytest` for testing. To run tests:
```bash
pytest tests/
```

---

## 🛠️ Built With

- **[Streamlit](https://streamlit.io/)** - For the interactive web dashboard.
- **[Hugging Face Transformers](https://huggingface.co/)** - Deep learning models for NLP.
- **[Plotly](https://plotly.com/)** - Interactive data visualizations.
- **[MongoDB](https://www.mongodb.com/)** - Scalable NoSQL database.
- **[Vader Sentiment](https://github.com/cjhutto/vaderSentiment)** - Lexicon-based sentiment analysis.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

Developed with ❤️ for better governance.
