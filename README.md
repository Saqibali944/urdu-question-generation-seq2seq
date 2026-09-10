# urdu-question-generation-seq2seq
A Sequence-to-Sequence neural network built from scratch in PyTorch to automatically generate Urdu questions from text, acting like an AI teacher creating quizzes.
Imagine handing an Urdu textbook to an AI and asking it to generate practice questions for students. That is exactly what this project does.

Given an Urdu sentence and a highlighted answer (e.g., “The Indus is about 3,180 km long”), our model acts like a teacher and generates the exact question that span answers (“How long is the Indus?”).

To deeply understand the mechanics of natural language generation, we built this entirely from scratch. There are no pre-trained Transformers, no HuggingFace pipelines, and no off-the-shelf LLMs here. Everything from the subword tokenizer to the Bidirectional LSTM and Bahdanau Attention mechanism was engineered and trained from the ground up using PyTorch primitives.

Under the hood:

Tokenizer: A custom 8,000-word SentencePiece model trained to understand complex Urdu morphology.

Encoder: 2-layer Bidirectional LSTM.

Decoder: 2-layer LSTM with Bahdanau (additive) Attention.

Inference: Includes both Greedy decoding and Beam Search to find the most natural-sounding Urdu questions.

UI: A lightweight Streamlit web app for real-time inference.
