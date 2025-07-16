# 📧 Email Spam Detection using Machine Learning

This project builds a spam classifier for emails using Machine Learning techniques. It applies Natural Language Processing (NLP) for text preprocessing and uses algorithms like **Naive Bayes** and **Logistic Regression** to classify emails as **spam** or **ham** (not spam).

---

## 📂 Project Structure

```
email_spam_detection_ml/
│
├── data/
│   └── spam.csv                 # Dataset file
│
├── main.py                      # Main script for training and evaluation
├── images/                      # Output visualizations (confusion matrix, bar plot, etc.)
│
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview (this file)
```

---

## 📊 Dataset

The dataset used is typically the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), containing:

- `label`: "ham" or "spam"
- `message`: the text content of the email/SMS

---

## 🛠️ Installation

1. **Clone the Repository**

```bash
git clone https://github.com/Ayushgowda27/email_spam_detection_ml.git
cd email_spam_detection_ml
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
python main.py
```

This will:

- Load the dataset
- Preprocess the text (lowercasing, stopword removal, stemming, vectorization using CountVectorizer)
- Train models (Naive Bayes, Logistic Regression)
- Evaluate using accuracy and confusion matrix
- Display visualizations (bar graph, scatter plot)

---

## 📊 Output Samples

- **Confusion Matrix**

![Confusion Matrix](images/confusion_matrix.png)

- **Bar Graph (Spam vs Ham Count)**

![Bar Graph](images/bar_graph.png)

- **Scatter Plot (Sample Distribution)**

![Scatter Plot](images/scatter_plot.png)

---

## 🧠 Models Used

- **Multinomial Naive Bayes**
- **Logistic Regression**

---

## 🧪 Requirements

Put the following in `requirements.txt`:

```
pandas
numpy
matplotlib
seaborn
sklearn
nltk
```

---

## 📬 Contact

Have questions? Open an [issue](https://github.com/Ayushgowda27/email_spam_detection_ml/issues) or reach out via GitHub.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for more information.
