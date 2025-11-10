# SHA YouTube Sentiment System

The **SHA YouTube Sentiment System** is an intelligent application designed to analyze public sentiment and media framing around Kenya’s **Social Health Authority (SHA)** policy.  
It uses fine-tuned multilingual transformer models (AfriBERTa and XLM-RoBERTa) to classify YouTube comments in **English** and **Kiswahili**, with integrated visual dashboards built in **Streamlit**.

---

## 🧠 Core Features
- **YouTube Data Retrieval** using the YouTube Data API  
- **Multilingual Sentiment Analysis** via fine-tuned AfriBERTa/XLM-R models  
- **Automatic Translation** for Kiswahili → English comments  
- **Media Framing Extraction** from titles and transcripts  
- **Interactive Dashboard** for visualization and comment exploration  
- **Firebase Integration** for secure data storage  

---

## 🗂 Project Structure
```
app.py                 → Streamlit dashboard interface  
prepare_labels.py      → Prepares and logs new training labels  
labels_log.csv         → Local log (ignored on GitHub)  
new_training_data.csv  → Local dataset (ignored on GitHub)  
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

## 🛡 Security Note
Sensitive files such as `.streamlit/secrets.toml`, model checkpoints, and local CSV data are excluded through `.gitignore` for privacy and security.

---

## 📊 Research Context
This project supports a master’s-level study titled  
**“Media Framing and Sentiment Analysis of SHA-related YouTube Content in Kenya (2024 – 2025)”**,  
investigating how mainstream media framing influences public perception of the SHA policy.

---

## 👤 Author
**Odhiambo B.**  
[GitHub Profile](https://github.com/odhiambob)

