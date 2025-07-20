# ShadowFox Internship Final Task Report

## Project Title: 
**AI-Driven NLP with BERT using Hugging Face**

**Intern Name:** Parimala Dharshini M 
**Date:** 20 July 2025  
**Tools Used:** Google Colab, Hugging Face Transformers, PyTorch, Matplotlib

---

## 1. Objective

The objective of this project is to explore and evaluate the performance of the pre-trained BERT model on various NLP tasks, including:

- Masked Word Prediction
- Sentence Classification

The model is implemented and tested using the Hugging Face Transformers library in a Google Colab environment.

---

## 2. Research Questions

1. How effectively can BERT predict masked words based on surrounding context?
2. How accurate is BERT in understanding sentiment in simple and complex sentences?
3. Does BERT struggle with tasks that involve multiple masked tokens or reasoning?
4. What are the limitations of BERT when applied to domain-specific or ambiguous language?
5. How does BERT perform before and after fine-tuning for classification tasks?
6. Can BERT generalize well to unseen examples or does it rely heavily on dataset bias?

---

## 3. Implementation

### 🔹 Masked Word Prediction
- Used the `bert-base-uncased` model to fill masked tokens in context-rich sentences.
- Tokenization and inference performed using Hugging Face's `transformers` pipeline.

### 🔹 Sentence Classification
- Performed binary sentiment classification using the GLUE SST-2 dataset.
- The BERT model was fine-tuned using the Trainer API on a small subset of the dataset.
- Evaluation was done on validation data with increasing accuracy over epochs.

---

## 4. Exploration and Analysis

### ✅ Masked Sentences Tested:
- The **[MASK]** barks at night. → `dog` (✅)
- The sun rises in the **[MASK]**. → `sky` (❌ expected: east)
- BERT is a **[MASK]** model developed by Google. → `business` (❌ expected: language)
- I want to eat **[MASK]** for dinner. → `it` (❌ expected: pizza)
- Mahatma Gandhi was born in **[MASK]**. → `india` (✅)

### ✅ Multiple Mask Prediction:
- The **[MASK]** wrote a **[MASK]** about artificial intelligence.  
  → `two`, `book`

### ✅ Sentiment Classification Results:
- Positive and negative sentences were accurately classified.
- Accuracy improved after fine-tuning the model.

---

## 5. Visualization of Results

### 🔸 Word Cloud of BERT’s Predicted Words
![Word Cloud](35eb5e3d-422a-4058-9b4a-76a4c1089788.png)

### 🔸 Sentiment Classification Accuracy (per Epoch)
![Line Graph](ea6328db-c365-47a8-8690-7d9851aec23d.png)

### 🔸 Accuracy of Masked Word Predictions
![Bar Chart](eb8d11fe-bb84-489b-8b56-8a2e6c77b932.png)

---

## 6. Evaluation and Alignment

### 📊 Performance Evaluation:
- Strong performance on factual and contextual predictions.
- Struggled with ambiguity and logical reasoning.

### 🎯 NLP Goal Alignment:
- Demonstrated strong understanding in masked language modeling.
- Successfully implemented sentiment classification — a real-world NLP use case.

### ⚖️ Ethical Considerations:
- BERT may reflect societal and linguistic biases.
- Domain-specific tasks may require custom fine-tuning.
- No built-in logic or reasoning capability.

### 🌍 Real-World Applications:
- Chatbots for customer support
- Review sentiment analysis
- Educational question-answering bots

---

## 7. Conclusion

BERT demonstrates strong contextual understanding in both text completion and classification when applied to general-purpose NLP tasks. This project offered hands-on experience in implementing BERT using Hugging Face, and provided insights into its effectiveness, limitations, and ethical use in real-world AI systems.

---

## ✅ Goal Statement

To implement a Language Model using Hugging Face (specifically BERT), apply it to NLP tasks such as masked word prediction and sentiment classification, analyze its capabilities and limitations, and evaluate its alignment with real-world applications and ethical considerations.

---

### 📝 Submitted as part of **ShadowFox Internship Final Task (Task 3)**
