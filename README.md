# NLP Customer Intelligence Engine

End-to-end NLP pipeline: customer reviews + support tickets -> sentiment, topics, churn prediction.

Built with: PyTorch, HuggingFace Transformers, BERTopic, XGBoost, Streamlit

## Quick Start

```
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
python src/synthetic_generator.py
python src/data_pipeline.py
python src/sentiment_model.py --train
streamlit run app/streamlit_app.py
```

## Author

Muhammad Fariz Ibrahim - @mufibra23
