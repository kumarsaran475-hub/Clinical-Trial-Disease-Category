import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import joblib
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.multiclass import OneVsRestClassifier
nltk.download('punkt')
nltk.download('punkt_tab') 
nltk.download('stopwords')
nltk.download('wordnet')

clinical=pd.read_csv("C:/Users/SARAN K/Downloads/clinical_trials_raw_patient2trial_conditions.csv",encoding='latin1')

print(clinical.head())
print(clinical.tail())
print(clinical.info())
print(clinical.dtypes)
print(clinical.describe)
print(clinical.columns)
print(clinical.shape)
print(clinical.isnull().sum())
print(clinical.duplicated().sum())

clinical['official_title'] = clinical['official_title'].fillna(clinical['official_title'].mode()[0])
clinical['conditions'] = clinical['conditions'].fillna(clinical['conditions'].mode()[0])
clinical['interventions'] = clinical['interventions'].fillna('Not specified')
clinical['phase'] = clinical['phase'].fillna('Unknown')
clinical['sex'] = clinical['sex'].fillna(clinical['sex'].mode()[0])
clinical['eligibility_criteria'] = clinical['eligibility_criteria'].fillna('Not specified')
clinical['healthy_volunteers'] = clinical['healthy_volunteers'].fillna(clinical['healthy_volunteers'].mode()[0])

def extract_age(value):
    if pd.isna(value):
        return None
   
    match = re.search(r'\d+', str(value))
    if match:
        return float(match.group())
    else:
        return None  

clinical['minimum_age'] = clinical['minimum_age'].apply(extract_age)
clinical['maximum_age'] = clinical['maximum_age'].apply(extract_age)

clinical['minimum_age'] = clinical['minimum_age'].fillna(clinical['minimum_age'].median())
clinical['maximum_age'] = clinical['maximum_age'].fillna(clinical['maximum_age'].median())

print(clinical.dtypes)
print(clinical.isnull().sum())

def clean_text(text):
    text = text.lower()                            
    text = re.sub(r'[^a-zA-Z\s]', '', text)           
    return text

clinical['clean_summary'] = clinical['brief_summary'].apply(clean_text)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

clinical["processed_summary"] = clinical["brief_summary"].apply(preprocess)

vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1,2))
X = vectorizer.fit_transform(clinical["processed_summary"])

encoder = LabelEncoder()
y = encoder.fit_transform(clinical["source_condition_query"])

print("Feature matrix shape:", X.shape)
print("Number of classes:", len(encoder.classes_))

plt.figure(figsize=(10,6))
sns.countplot(y=clinical["source_condition_query"], 
              order=clinical["source_condition_query"].value_counts().index)
plt.title("Distribution of Disease Categories")
plt.xlabel("Count")
plt.ylabel("Disease Category")
plt.show()

all_words = " ".join(clinical["processed_summary"]).split()
filtered_words = [w for w in all_words if w not in stop_words]
word_freq = Counter(filtered_words).most_common(20)

words, counts = zip(*word_freq)
plt.figure(figsize=(10,6))
sns.barplot(x=list(words), y=list(counts), palette="magma")
plt.xticks(rotation=45)
plt.title("Top 20 Keywords in Clinical Summaries")
plt.show()

def top_terms_for_category(category, n=10):
    idx = (clinical['source_condition_query'] == category).values
    X_cat = X[idx]
    mean_tfidf = np.asarray(X_cat.mean(axis=0)).ravel()
    top_indices = mean_tfidf.argsort()[-n:][::-1]
    return [vectorizer.get_feature_names_out()[i] for i in top_indices]


print(top_terms_for_category("breast cancer"))
print(top_terms_for_category("sickle cell anemia"))

plt.figure(figsize=(10,6))
sns.countplot(x="phase", hue="source_condition_query", data=clinical, palette="Set2")
plt.title("Trial Phases by Disease Category")
plt.xticks(rotation=45)
plt.show()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

param_grid_svm = {
    "C": [0.01, 0.1, 1, 10],
    "loss": ["hinge", "squared_hinge"]
}

grid_svm = GridSearchCV(
    LinearSVC(class_weight="balanced", max_iter=5000),
    param_grid_svm,
    cv=3,
    n_jobs=-1,
    verbose=2
)

grid_svm.fit(X_train, y_train)

print("✅ Best SVM Params:", grid_svm.best_params_)
print("✅ Best SVM CV Accuracy:", grid_svm.best_score_)
print("✅ Tuned SVM Test Accuracy:", grid_svm.score(X_test, y_test))

best_model = grid_svm.best_estimator_

joblib.dump(best_model, "svm_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(encoder, "label_encoder.pkl")
print("💾 Model, vectorizer, and encoder saved.")

y_pred = best_model.predict(X_test)

y_test_str = encoder.inverse_transform(y_test)
y_pred_str = encoder.inverse_transform(y_pred)

print("\nClassification Report (SVM):\n")
print(classification_report(y_test_str, y_pred_str))

cm = confusion_matrix(y_test_str, y_pred_str, labels=encoder.classes_)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("Confusion Matrix - Tuned SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

y_bin = label_binarize(y_test, classes=np.arange(len(encoder.classes_)))
n_classes = y_bin.shape[1]

ovr_svm = OneVsRestClassifier(best_model)
y_score = ovr_svm.fit(X_train, y_train).decision_function(X_test)

fpr, tpr, roc_auc = {}, {}, {}
plt.figure(figsize=(10,8))
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f"{encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0,1], [0,1], "k--")
plt.title("Multi-class ROC Curve (SVM)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

print("Macro AUC:", roc_auc_score(y_bin, y_score, average="macro"))
print("Micro AUC:", roc_auc_score(y_bin, y_score, average="micro"))