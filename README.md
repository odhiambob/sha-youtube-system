# SHA YouTube Sentiment System

The **SHA YouTube Sentiment System** is an intelligent application designed to analyze public sentiment and media framing around Kenya’s **Social Health Authority (SHA)** policy.  
It uses fine-tuned multilingual transformer models (**AfriBERTa** and **XLM-RoBERTa**) to classify YouTube comments in **English** and **Kiswahili**, with integrated visual dashboards built in **Streamlit**.

---

## 🧠 Core Features
- **YouTube Data Retrieval** using the YouTube Data API  
- **Multilingual Sentiment Analysis** via fine-tuned AfriBERTa/XLM-R models  
- **Automatic Translation** for Kiswahili → English comments  
- **Media Framing Extraction** from titles and transcripts  
- **Interactive Dashboard** for visualization and comment exploration  
- **Local CSV Storage** for reviewer feedback and retraining data  
- **Firebase Integration (optional)** for secure cloud storage  

---

## 🗂 Project Structure
```
app.py                 → Streamlit dashboard interface  
prepare_labels.py      → Prepares and logs new training labels  
labels_log.csv         → Local log (ignored on GitHub)  
new_training_data.csv  → Local dataset (ignored on GitHub)  
Data.ipynb             → YouTube data acquisition, cleaning, and export  
Sentiment.ipynb        → Model fine-tuning, evaluation, and scoring  
.streamlit/secrets.toml → Private API keys (ignored on GitHub)
```

---

## 🚀 Run Locally
1. Clone the repository  
```bash
git clone https://github.com/odhiambob/sha-youtube-system.git
cd sha-youtube-system
```

2. Create and activate a virtual environment  
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies  
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app  
```bash
streamlit run app.py
```

---

## 📓 Jupyter Notebooks
### Data.ipynb  
This notebook handles **YouTube data collection and preprocessing**. It connects to the YouTube Data API, retrieves video metadata and comments, performs **language detection**, and uses **deep_translator** to translate Kiswahili comments into English. Cleaned and standardized data are exported as CSV files for sentiment modeling.

### Sentiment.ipynb  
This notebook performs **model training and evaluation** using fine-tuned **AfriBERTa** and **XLM-RoBERTa** models. It measures **accuracy**, **precision**, **recall**, and **macro-F1** scores, with the fine-tuned AfriBERTa achieving approximately **0.90 validation accuracy** and **0.43 macro-F1**. It also generates predictions for full comment datasets, producing monthly sentiment summaries for dashboard visualization.

---

## 🧩 System Architecture
The intelligent system comprises five main modules:
1. **User Interface Module** – Built with Streamlit for link or date-range analysis.  
2. **Data Acquisition Module** – Fetches video metadata and comments via the YouTube Data API.  
3. **Sentiment Classification Module** – Uses the fine-tuned AfriBERTa model located at `C:/Users/HP/Desktop/usiu/afriberta_ft_ckpt/checkpoint-90`.  
4. **Media Framing Module** – Applies rule-based logic to detect thematic frames such as Governance, Economic Burden, and Citizen Welfare.  
5. **Local CSV Storage Module** – Logs reviewer feedback and new labeled samples in `labels_log.csv` and `new_training_data.csv`.  

---

## 🛡 Security Note
Sensitive files such as `.streamlit/secrets.toml`, fine-tuned model checkpoints, and local CSV datasets are excluded through `.gitignore` for privacy and security.  
The system can run fully offline when the AfriBERTa model is stored locally. Firebase integration is optional for multi-user storage.

---

## 📊 Research Context
This project supports a master’s-level research study titled  
**“Media Framing and Sentiment Analysis of SHA-related YouTube Content in Kenya (2024–2025)”**,  
which examines how framing by mainstream media outlets influences public sentiment regarding Kenya’s Social Health Authority (SHA) policy. The project combines computational linguistics, media analysis, and AI-based modeling to generate interpretable insights for policymakers and researchers.

---

## 👤 Author
**Odhiambo B.**  
[GitHub Repository](https://github.com/odhiambob/sha-youtube-system)  
[GitHub Profile](https://github.com/odhiambob)  

© 2025 Odhiambo B. | All rights reserved.  
This repository accompanies the academic project *“Media Framing and Sentiment Analysis of SHA-related YouTube Content in Kenya (2024–2025).”*

