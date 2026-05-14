# Human vs AI-Generated Text Detection using Hybrid Feature Fusion

## Project Overview
This project presents a Hybrid NLP-based AI Text Detection System designed to distinguish between human-written and AI-generated text. The model combines the semantic understanding capability of a Transformer model (DistilRoBERTa) with handcrafted stylometric and linguistic features such as entropy, lexical richness, punctuation density, burstiness, and average sentence length.

The project is inspired by Hybrid Feature Fusion techniques used in Fake News Detection, where deep contextual embeddings and manually engineered features are combined to improve classification performance.

The system is implemented using PyTorch, Hugging Face Transformers, and scikit-learn, with support for efficient large-scale data handling through dataset streaming.

---

# Features
- Hybrid Transformer + Stylometric Feature Fusion Architecture
- DistilRoBERTa-based semantic text embeddings
- Handcrafted linguistic feature extraction
- Streaming dataset loading for memory efficiency
- Mixed Precision Training using AMP
- Early stopping and learning rate scheduling
- Binary classification (Human vs AI-generated)
- Evaluation using Precision, Recall, F1-score, and ROC-AUC
- Real-time inference with AI probability score

---

# Model Architecture

The model consists of two major components:

### 1. Transformer Backbone
- Pretrained DistilRoBERTa
- Extracts contextual semantic embeddings from text

### 2. Stylometric Feature Extractor
Extracted handcrafted features include:
- Lexical Richness (Type-Token Ratio)
- Entropy
- Burstiness
- Punctuation Density
- Average Sentence Length
- Text Length

### 3. Feature Fusion Layer
The transformer embedding and stylometric feature vector are concatenated and passed into a Multi-Layer Perceptron (MLP) classifier for final prediction.

---

# Dataset

Dataset Used:
- Human vs AI Generated Text Dataset
- Source: Hugging Face

Dataset Link:
https://huggingface.co/datasets/dmitva/human_ai_generated_text

### Dataset Handling
The dataset contains over 1 million rows. To prevent RAM overflow:
- Streaming mode from Hugging Face Datasets is used
- Controlled sampling is implemented

---

#  Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core Programming |
| PyTorch | Deep Learning Framework |
| Transformers | Pretrained NLP Models |
| Hugging Face Datasets | Streaming Dataset Loader |
| scikit-learn | Metrics & Preprocessing |
| NLTK | Text Processing |
| NumPy | Numerical Operations |
| Google Colab | GPU Training Environment |

---

#  Installation

## Install Dependencies

```bash
pip install transformers datasets torch scikit-learn nltk pandas numpy
```

---

#  Running the Project

## 1. Load Dataset
Dataset is loaded in streaming mode for efficiency.

## 2. Extract Features
Stylometric features are extracted from text samples.

## 3. Train Model
The hybrid model is trained using:
- Mixed Precision Training
- AdamW Optimizer
- Learning Rate Scheduler
- Early Stopping

## 4. Evaluate
Performance is evaluated using:
- Precision
- Recall
- F1-score
- ROC-AUC

## 5. Inference
Input any text to get:
- Probability of AI generation
- Human/AI prediction label

---

# Evaluation Metrics

The model is evaluated using:
- Precision
- Recall
- F1-score
- ROC-AUC

Example Results:

| Metric | Score |
|--------|-------|
| Precision | 0.99 |
| Recall | 1.00 |
| F1-score | 0.99 |
| ROC-AUC | 1.00 |

---

# Example Usage

```python
text = "Artificial intelligence has transformed modern industries."

probability = predict_ai_probability(text)

print(probability)
```

Example Output:

```text
0.84
```

---

# Key Contributions
- Developed a hybrid AI-text detection framework
- Combined semantic and stylometric analysis
- Implemented scalable dataset streaming
- Optimized training using AMP and scheduler tuning
- Added inference calibration using threshold tuning

---

# Limitations
- Well-structured human-written text may occasionally be classified as AI-generated
- Performance may vary depending on writing style and AI model sophistication
- Limited training subset used for efficient experimentation

---

#  Future Enhancements
- Deploy using Streamlit or Flask
- Train on larger balanced datasets
- Use DeBERTa-v3 for improved contextual understanding
- Add explainability using SHAP/LIME
- Multi-class AI model source detection

---

# Author
Developed as a Final Year B.Tech Project in Machine Learning and NLP.

---

#  License
This project is intended for educational and research purposes.
