
# Detecting Adverse Drug Reaction Signals from WebMD Reviews using-NLP-machine-learning-and-deep-learning

**NLP, Machine Learning, Deep Learning & Interactive Dashboard**

## Overview

Adverse Drug Reactions (ADRs) are an important patient-safety concern. Patients frequently describe medication side effects and treatment experiences in online reviews, but manually analyzing thousands of free-text reviews is difficult and time-consuming.

This project develops an end-to-end **Natural Language Processing (NLP), Machine Learning, and Deep Learning** framework to analyze WebMD patient drug reviews and identify adverse drug reaction signals.

The system extracts ADR-related information from patient reviews, compares reported reactions with official side effects, and classifies each review into one of three categories:

* **Known ADR** – The reported ADR matches an official side effect of the medication.
* **Potential ADR Signal** – The patient reports an ADR that is not found among the listed official side effects.
* **No ADR Mention** – No adverse drug reaction is detected in the review.

The final stage of the project will include an **interactive dashboard** for exploring ADR patterns, model results, and potential drug-safety signals.

---

## Project Objective

The objective of this project is to:

* Analyze large volumes of WebMD patient drug reviews.
* Clean and preprocess unstructured patient review text.
* Extract patient-reported adverse drug reactions using NLP.
* Handle negated ADR statements.
* Compare extracted ADRs with official medication side effects.
* Identify potential new ADR signals.
* Classify reviews into three ADR categories.
* Build and compare machine learning and deep learning models.
* Explain model predictions using SHAP.
* Develop an interactive dashboard for exploring ADR patterns and potential safety signals.

---

## Dataset

The project uses the **WebMD Patient Drug Reviews Dataset** obtained from Kaggle.

### Original Dataset

* **362,806 patient reviews**
* **12 original variables**

The dataset contains information about:

| Variable        | Description                                          |
| --------------- | ---------------------------------------------------- |
| `Age`           | Patient age group                                    |
| `Condition`     | Medical condition for which the medication was taken |
| `Date`          | Date of the patient review                           |
| `Drug`          | Medication name                                      |
| `DrugId`        | Unique drug identifier                               |
| `EaseofUse`     | Patient rating for ease of medication use            |
| `Effectiveness` | Patient rating of medication effectiveness           |
| `Reviews`       | Patient-written medication review                    |
| `Satisfaction`  | Overall patient satisfaction rating                  |
| `Sex`           | Patient sex                                          |
| `Sides`         | Official side effects listed for the medication      |
| `UsefulCount`   | Number of users who found the review useful          |

After data cleaning and preprocessing, the final dataset contained:

**320,094 patient reviews**

---

## Data Cleaning and Preprocessing

Several data-quality issues were identified before modeling, including missing values, duplicate records, inconsistent text formatting, and invalid rating values.

The preprocessing pipeline included:

* Removing duplicate records.
* Removing missing and empty patient reviews.
* Handling missing categorical values.
* Standardizing column names.
* Standardizing categorical values.
* Standardizing medication names.
* Removing invalid rating values.
* Converting dates to datetime format.
* Extracting year and month features.

---

## NLP Text Preprocessing

Patient review text was processed using Natural Language Processing techniques before ADR extraction and model development.

The NLP pipeline included:

* Converting text to lowercase.
* Expanding contractions.
* Removing punctuation.
* Removing numbers.
* Removing unnecessary stop words.
* Preserving important negation words.
* Lemmatizing words.

Negation words such as:

`no`, `not`, `never`, and `without`

were intentionally preserved because they can completely change the meaning of an ADR statement.

For example:

**"The medication caused nausea."**

and

**"The medication did not cause nausea."**

should not be interpreted in the same way.

---

## ADR Extraction

A custom **spaCy-based NLP pipeline** was developed to identify adverse drug reactions mentioned in patient reviews.

The ADR extraction process included:

1. Processing cleaned patient review text.
2. Detecting possible ADR mentions.
3. Detecting negated ADR statements.
4. Standardizing official side-effect terminology.
5. Mapping synonyms to canonical ADR terms.
6. Applying fuzzy matching for minor spelling and terminology differences.
7. Comparing extracted ADRs with official medication side effects.

### ADR Classification Logic

```text
Patient Review
      ↓
Text Cleaning & NLP Preprocessing
      ↓
ADR Extraction
      ↓
Negation Detection
      ↓
ADR Standardization
      ↓
Compare with Official Side Effects
      ↓
ADR Category
```

---

## Target Variable

A new target variable called **ADR Category** was created.

Each review was assigned to one of three classes:

| ADR Category         |   Count | Percentage |
| -------------------- | ------: | ---------: |
| No ADR Mention       | 152,731 |     47.71% |
| Potential ADR Signal | 113,503 |     35.46% |
| Known ADR            |  53,860 |     16.83% |

This transformed the problem into a **multiclass text-classification task**.

---

## Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the cleaned dataset and identify patterns related to ADR classification.

The analysis included:

* ADR category distribution.
* Engineered feature distributions.
* Review length analysis.
* Sentiment analysis.
* Satisfaction analysis.
* Drug frequency analysis.
* Bivariate analysis.
* Correlation analysis.
* Examination of potential outliers.

---

## Feature Engineering

Several additional variables were engineered from the review text and original dataset.

Features included:

* Review length
* Character count
* Sentence count
* Sentiment score
* Rating gap
* Low satisfaction flag
* Log-transformed UsefulCount
* ADR overlap count
* ADR novel count
* Average word length
* Unique word ratio
* Negation count
* Drug frequency

These engineered variables were combined with the processed review text to provide additional information for ADR classification.

---

## Modeling Preparation

The cleaned review text was converted into numerical features using **TF-IDF vectorization**.

TF-IDF was limited to:

**5,000 text features**

Numerical features were standardized and combined with the TF-IDF representation.

The processed dataset was divided into:

* **64% Training**
* **16% Validation**
* **20% Testing**

Stratified sampling was used to preserve the ADR class distribution across the datasets.

---

## Machine Learning Models

Three linear classifiers were evaluated using Stochastic Gradient Descent (SGD):

1. **SGD – Log Loss**
2. **SGD – Hinge Loss**
3. **SGD – Perceptron**

Models were evaluated using:

* Stratified 5-fold cross-validation
* Bootstrapping
* Hyperparameter tuning

### Initial Linear Model Results

| Model          | CV Macro F1 | Bootstrap Macro F1 |
| -------------- | ----------: | -----------------: |
| SGD Log Loss   |       0.679 |              0.678 |
| SGD Hinge Loss |   **0.707** |          **0.709** |
| SGD Perceptron |       0.684 |              0.708 |

The **SGD Hinge Loss classifier** showed the strongest and most stable performance among the initial linear models.

---

## Hyperparameter Tuning

The SGD Hinge classifier was further optimized using:

* L1 regularization
* L2 regularization
* Multiple alpha values

The best configuration was:

```text
Loss: Hinge
Regularization: L1
Alpha: 0.00001
```

The tuned model achieved:

**Cross-Validation Macro F1 = 0.763**

---

## Deep Learning Model

A **Deep Neural Network (DNN)** was developed to capture more complex relationships between the text and engineered features.

### Initial Architecture

```text
Input Features
      ↓
Dense Layer – 128 neurons + ReLU
      ↓
Dense Layer – 64 neurons + ReLU
      ↓
Dense Layer – 32 neurons + ReLU
      ↓
Output Layer – 3 neurons + Softmax
```

The model used:

* Adam optimizer
* Softmax output
* Early stopping
* Validation monitoring

The initial DNN showed signs of overfitting.

To improve generalization, **Dropout regularization (0.3)** was added after the first and second hidden layers.

---

## Model Comparison

| Model                |  Macro F1 | Macro Precision | Macro Recall | Accuracy |
| -------------------- | --------: | --------------: | -----------: | -------: |
| Tuned SGD Hinge      |     0.758 |            0.77 |         0.75 |     0.81 |
| **DNN with Dropout** | **0.816** |        **0.82** |     **0.81** | **0.85** |

The **DNN with Dropout** achieved the strongest overall performance and was selected as the final model.

---

## Final Model Performance

The final DNN was evaluated on a completely held-out test set containing:

**64,019 patient reviews**

### Test Results

| Metric          |     Score |
| --------------- | --------: |
| Macro F1        | **0.816** |
| Macro Precision |  **0.82** |
| Macro Recall    |  **0.81** |
| Accuracy        |  **0.85** |

The model performed particularly well at identifying **No ADR Mention** reviews.

The largest classification challenge occurred between **Known ADR** and **Potential ADR Signal**, likely because these two categories can contain similar symptom-related language.

---

## Model Explainability with SHAP

**SHAP (Shapley Additive Explanations)** was used to better understand the final model's predictions.

Important engineered features included:

* Review character count
* Review length
* Sentence count
* Drug frequency
* Average word length
* Unique word ratio
* Sentiment score
* Negation count
* Low satisfaction flag
* Log UsefulCount
* Rating gap

Important TF-IDF terms included words related to symptoms and patient experiences such as:

`depression`, `headache`, `tired`, `anxiety`, `dizzy`, `migraine`, `nausea`, `dizziness`, `constipation`, `pain`, `muscle`, and `panic`.

This analysis helps explain which characteristics of patient reviews influence ADR predictions.

---

## Interactive ADR Dashboard

An interactive **Streamlit dashboard** was developed to demonstrate the ADR detection pipeline on new patient reviews.

The dashboard allows users to:

- Select a medication from the WebMD dataset.
- Enter a new patient review.
- Process the review using the NLP pipeline.
- Extract adverse drug reactions (ADRs) mentioned in the review.
- Predict the review category using the trained deep learning model:
  - **Known ADR**
  - **Potential ADR Signal**
  - **No ADR Mention**
- Display the model's prediction confidence.
- Compare detected ADRs with the medication's official side effects.
- Identify whether a detected ADR matches a known side effect or represents a potential ADR signal.

### Dashboard Workflow

```text
Patient Review + Drug
        ↓
Text Preprocessing
        ↓
ADR Extraction
        ↓
Feature Engineering
        ↓
Trained DNN Model
        ↓
ADR Classification
        ↓
Known ADR / Potential ADR Signal / No ADR Mention
        ↓
Comparison with Official Drug Side Effects
        ↓
Interactive Dashboard Results
```

### Running the Dashboard Locally
## How to Run the Project

### 1. Clone the repository

```bash
git clone <repository-url>
cd webmd-adverse-drug-reaction-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Required Files

The dashboard requires the following files:

- `dropout_dnn.keras` — trained deep learning model
- `transformation.pkl` — fitted TF-IDF and feature transformation pipeline
- `label_encoder.pkl` — ADR category label encoder
- `webmd_final.csv` — processed WebMD dataset used by the dashboard
- 
### 4. Launch the Dashboard:

```bash
streamlit run app.py
```

### Future Monitoring Concept

The current dashboard analyzes individual patient reviews. A future version could extend the system to monitor incoming reviews over time.

```text
New Patient Reviews
        ↓
ADR Detection
        ↓
Potential ADR Signals
        ↓
Aggregate by Drug and ADR
        ↓
Repeated Unexpected ADR Patterns
        ↓
Potential Safety Signal for Further Investigation
```

If the same unexpected ADR is repeatedly reported for a particular medication, the system could flag the pattern for further investigation.

> **Important:** This dashboard is a research and educational prototype. A predicted potential ADR signal does not establish that a medication caused the reported reaction and should not be interpreted as medical advice or a confirmed safety alert.
---

## Technology Stack

### Data Analysis

* Python
* Pandas
* NumPy

### NLP

* spaCy
* NLTK
* RapidFuzz
* TF-IDF

### Machine Learning

* Scikit-learn
* SGDClassifier

### Deep Learning

* TensorFlow
* Keras

### Explainable AI

* SHAP

### Visualization

* Matplotlib
* Seaborn
* WordCloud

### Dashboard

* Interactive dashboard 

### Development Environment

* Google Colab
* GitHub

---

## Project Structure

```text
webmd-adverse-drug-reaction-detection/
│
├── app.py
│   └── Streamlit dashboard application
│
├── master_project_file.ipynb
│   └── Data preprocessing, NLP, EDA, feature engineering,
│       model training, and evaluation
│
├── dropout_dnn.keras
│   └── Trained deep learning model
│
├── transformation.pkl
│   └── Fitted TF-IDF and feature transformation pipeline
│
├── label_encoder.pkl
│   └── ADR category label encoder
│
├── webmd_final.csv
│   └── Processed dataset used by the dashboard
│
├── requirements.txt
│   └── Python dependencies required to reproduce the project
│
├── README.md
│   └── Project documentation
│
├── .gitignore
│   └── Files and folders excluded from Git tracking
│
└── data/
    └── webmd.csv
        └── Original WebMD patient review dataset
```
> The original dataset may not be included in the repository because of file size and data-source restrictions.

---


## Limitations

This project has several important limitations:

* Patient reviews are self-reported.
* Reviews may contain spelling errors, informal language, or incomplete information.
* NLP-based ADR extraction may not identify every possible reaction.
* Fuzzy matching may occasionally create incorrect ADR matches.
* Symptoms may be related to an underlying medical condition rather than the medication.
* A Potential ADR Signal does not prove causation.
* Online patient reviews may not represent the overall patient population.

Therefore, this project should be considered a **drug-safety signal detection and research tool**, not a clinical diagnostic system.

---

## Future Work

Future improvements include:

* Develop and deploy the interactive ADR dashboard.
* Integrate new patient reviews for continuous monitoring.
* Track potential ADR signals over time.
* Develop alert mechanisms for repeated unexpected ADR reports.
* Explore **BioBERT** and other biomedical transformer models.
* Include additional healthcare and pharmacovigilance datasets.
* Monitor the model for data drift.
* Periodically retrain the model using newer patient reviews.
* Improve ADR entity extraction and contextual understanding.

---

## Conclusion

This project demonstrates an end-to-end NLP and deep learning approach for detecting adverse drug reaction signals from WebMD patient reviews.

The framework processes unstructured patient feedback, extracts ADR-related information, creates three clinically meaningful ADR categories, and predicts those categories using machine learning and deep learning.

Among the evaluated models, the **Deep Neural Network with dropout achieved the best performance with a Macro F1-score of 0.816 and an accuracy of 85%**.

By combining **NLP, feature engineering, machine learning, deep learning, explainable AI, and an interactive dashboard**, this project provides a foundation for exploring how patient-generated data could support post-market drug-safety surveillance and pharmacovigilance.

---
## Project Team

This project was developed as part of **ADS-504: Machine Learning and Deep Learning for Data Science** at the **Shiley-Marcos School of Engineering, University of San Diego**.

### Team Members

* **Sheshma Jaganathan**
* **Aishwarya Kuduvalli**
* **Sushma Kafle**

The project was completed collaboratively, including data preprocessing, NLP-based ADR extraction, exploratory data analysis, feature engineering, machine learning, deep learning, model evaluation, and interpretation.
