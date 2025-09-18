## Installation (poetry)

Clone the repository
```bash
cd Hackathon
```

Install dependencies with Poetry
```bash
poetry install
```


Run the app with
```bash
poetry run streamlit run Home.py
```
Then open the URL shown in your terminal



## Project Structure

Hackathon/
├─ pyproject.toml          # Poetry configuration with dependencies
├─ Home.py                 # Landing page
├─ pages/
│  └─ 1_Chatbot.py         # Main chatbot page
└─ Prediction_model/
   ├─ clean_data.py
   ├─ model.py
   ├─ evaluate_model.py
   ├─ helpers.py
   └─ services/
      ├─ data_service.py
      └─ model_service.py

