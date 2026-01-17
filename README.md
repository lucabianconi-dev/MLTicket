# MLTicket
Repository that contain the project MLTichek for Pegaso PW by Bianconi Luca 0312301652
# Triage automatico dei ticket con Machine Learning (Prototipo minimale)

## Obiettivo
Classificare ticket testuali in:
- Categoria: Amministrazione / Tecnico / Commerciale
- Priorità: bassa / media / alta
e fornire:
- Valutazione semplice (train/test split 80/20, accuracy, F1 macro, confusion matrix)
- Dashboard per predizione singola e batch + 5 parole più influenti

## Setup
```bash
pip install -r requirements.txt
mkdir -p data models
python src/generate_dataset.py
python src/train_eval.py
streamlit run src/app.py


