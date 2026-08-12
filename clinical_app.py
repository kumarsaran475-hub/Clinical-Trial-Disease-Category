import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split


st.set_page_config(page_title="Disease Prediction", layout="wide")
st.title("🩺 Interactive Disease Prediction System")


tfidf = joblib.load("tfidf_vectorizer.pkl")
best_model = joblib.load("svm_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.sidebar.header("🔮 Try Your Own Prediction")
user_input = st.sidebar.text_area("Enter Clinical Trial Summary:", "")

if st.sidebar.button("Predict"):
    if user_input.strip():
        X_input = tfidf.transform([user_input])
        prediction = best_model.predict(X_input)[0]          
        disease = encoder.inverse_transform([prediction])[0] 
        st.sidebar.success(f"Predicted Disease Category: **{disease}**")

df = pd.read_csv(
        "C:/Users/SARAN K/Downloads/clinical_trials_raw_patient2trial_conditions.csv",
        encoding="latin1"
    )
 
X = tfidf.transform(df["brief_summary"])
y = encoder.transform(df["source_condition_query"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

st.subheader("📊 Model Performance")

y_pred = best_model.predict(X_test)

y_test_str = encoder.inverse_transform(y_test)
y_pred_str = encoder.inverse_transform(y_pred)

report = classification_report(
    y_test_str, y_pred_str, target_names=encoder.classes_, output_dict=True
)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{report['accuracy']:.2f}")
col2.metric("Macro F1", f"{report_df.loc['macro avg','f1-score']:.2f}")
col3.metric("Weighted F1", f"{report_df.loc['weighted avg','f1-score']:.2f}")

cm = confusion_matrix(y_test_str, y_pred_str, labels=encoder.classes_)
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=encoder.classes_, yticklabels=encoder.classes_, ax=ax)
plt.title("Confusion Matrix - Tuned SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

class_acc = report_df.loc[encoder.classes_, "recall"]
fig_bar = px.bar(
    x=encoder.classes_, y=class_acc,
    labels={"x":"Disease Category", "y":"Recall"},
    title="Recall per Disease Category"
)
st.plotly_chart(fig_bar)


st.subheader("📈 Multi-class ROC Curve")

y_bin = label_binarize(y_test, classes=np.arange(len(encoder.classes_)))
n_classes = y_bin.shape[1]

ovr_svm = OneVsRestClassifier(best_model)
y_score = ovr_svm.fit(X_train, y_train).decision_function(X_test)

fpr, tpr, roc_auc = {}, {}, {}
fig_roc, ax = plt.subplots(figsize=(10,8))
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    ax.plot(fpr[i], tpr[i], label=f"{encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})")

ax.plot([0,1], [0,1], "k--")
ax.set_title("Multi-class ROC Curve (SVM)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
st.pyplot(fig_roc)

st.write("Macro AUC:", roc_auc_score(y_bin, y_score, average="macro"))
st.write("Micro AUC:", roc_auc_score(y_bin, y_score, average="micro"))



category_counts = df["source_condition_query"].value_counts()
fig_dist = px.bar(
    x=category_counts.index, y=category_counts.values,
    labels={"x":"Disease Category", "y":"Number of Trials"},
    title="Distribution of Disease Categories"
)
st.plotly_chart(fig_dist)
