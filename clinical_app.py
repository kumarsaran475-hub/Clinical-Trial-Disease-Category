import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

st.set_page_config(page_title="Disease Prediction", layout="wide")
st.title("🩺 Disease Prediction System")

tfidf = joblib.load("tfidf_vectorizer.pkl")
best_model = joblib.load("svm_model.pkl")
encoder = joblib.load("label_encoder.pkl")

user_input = st.text_area("Enter Clinical Trial Summary:")

if st.button("Predict"):
    if user_input.strip():
        # Prediction
        X_input = tfidf.transform([user_input])
        prediction = best_model.predict(X_input)[0]
        disease = encoder.inverse_transform([prediction])[0]

        st.success(f"Predicted Disease: **{disease}**")

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
            y_test_str, y_pred_str, output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report).transpose()

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{report['accuracy']:.2f}")
        col2.metric("Macro F1", f"{report_df.loc['macro avg','f1-score']:.2f}")
        col3.metric("Weighted F1", f"{report_df.loc['weighted avg','f1-score']:.2f}")

        st.subheader("📌 Confusion Matrix")
        present_classes = np.unique(y_test_str)
        cm = confusion_matrix(y_test_str, y_pred_str, labels=present_classes)
        fig, ax = plt.subplots(figsize=(7,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=present_classes, yticklabels=present_classes, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.subheader("📈 Recall per Disease Category")
        class_acc = report_df.loc[present_classes, "recall"]
        fig_bar = px.bar(
            x=present_classes, y=class_acc,
            labels={"x":"Disease Category", "y":"Recall"},
            title="Recall per Category", color=class_acc, color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_bar, width="stretch")

        st.subheader("📉 Multi-class ROC Curve")
        present_class_ids = np.unique(y_test)
        y_bin = label_binarize(y_test, classes=present_class_ids)
        n_classes = y_bin.shape[1]

        ovr_svm = OneVsRestClassifier(best_model)
        y_score = ovr_svm.fit(X_train, y_train).decision_function(X_test)

        fig_roc, ax = plt.subplots(figsize=(8,6))
        for i, cls in enumerate(present_class_ids):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{encoder.inverse_transform([cls])[0]} (AUC={roc_auc:.2f})")

        ax.plot([0,1], [0,1], "k--")
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        st.pyplot(fig_roc)

        st.info(f"Macro AUC: {roc_auc_score(y_bin, y_score, average='macro'):.2f} | "
                f"Micro AUC: {roc_auc_score(y_bin, y_score, average='micro'):.2f}")

        st.subheader("📊 Dataset Distribution")
        category_counts = df["source_condition_query"].value_counts()
        fig_dist = px.bar(
            x=category_counts.index, y=category_counts.values,
            labels={"x":"Disease Category", "y":"Trials"},
            title="Disease Category Distribution", color=category_counts.values,
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_dist, width="stretch")
