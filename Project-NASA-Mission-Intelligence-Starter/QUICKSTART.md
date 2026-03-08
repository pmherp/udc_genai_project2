# Quickstart Guide

Welcome to the NASA Mission Intelligence project from Udacity! This guide will help you get started quickly.

## Prerequisites

- Python 3.12
- Virtual environment (recommended)

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/pmherp/udc_genai_project2.git
   cd PROJECT-NASA-MISSION-INTELLIGENCE-STARTER
   ```

2. Run the setup script:
   ```bash
   ./setup.sh
   ```

   This will create a virtual environment and install all dependencies.

3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

4. Set OpenAI API key in the `.env` file:
   ```
   OPENAI_API_KEY=
   ```

5. Run the embedding pipeline to process the mission transcripts:
   ```bash
   python embedding_pipeline.py \
     --openai-key "$OPENAI_API_KEY" \
     --data-path data_text/ \
     --chroma-dir ./chroma_db_openai \
     --collection-name nasa_space_missions_text
   ```   

6. Run batch RAG evaluation with the included dataset:
   ```bash
   python ragas_evaluator.py \
     --dataset-path test_questions.json \
     --openai-key "$OPENAI_API_KEY" \
     --chroma-dir ./chroma_db_openai \
     --collection-name nasa_space_missions_text
   ```

## Running the Application

To start the application, run:
```bash
streamlit run chat.py
```

## Running Tests

Run the component smoke test suite:

```bash
python tests/test_component_smoke.py
```

Alternative via unittest module:

```bash
python -m unittest -v tests/test_component_smoke.py
```

---

Enjoy exploring NASA mission data with GenAI!