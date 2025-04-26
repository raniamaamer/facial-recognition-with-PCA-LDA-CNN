import streamlit as st
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# === Paramètres ===
IMAGE_SIZE = (100, 100)
DATA_DIR = "originalimages_part2"

# === Chargement des données ===
def charger_donnees(path):
    images, labels = [], []
    for file in os.listdir(path):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img = cv2.imread(os.path.join(path, file))
            img = cv2.resize(img, IMAGE_SIZE)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            flat = gray.flatten()
            person_id = int(file.split("-")[0])
            images.append(flat)
            labels.append(person_id - 1)
    return np.array(images), np.array(labels)

# === Page Streamlit ===
st.set_page_config(page_title="Reconnaissance Faciale PCA+LDA+CNN", layout="wide", page_icon="🧠")
st.title("🧠 Reconnaissance Faciale avec PCA, LDA et CNN")

# === Chargement et préparation des données ===
X, y = charger_donnees(DATA_DIR)
st.success(f"{len(X)} images chargées - {len(np.unique(y))} classes détectées")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=100)
X_pca = pca.fit_transform(X_scaled)

lda = LDA(n_components=min(len(np.unique(y)) - 1, 30))
X_lda = lda.fit_transform(X_pca, y)

X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, stratify=y, random_state=42)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# === CNN sur vecteurs 1D ===
X_train_cnn = X_train[..., np.newaxis]
X_test_cnn = X_test[..., np.newaxis]

model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    MaxPooling1D(2),
    Dropout(0.3),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_train_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

with st.spinner("Entraînement du CNN..."):
    model.fit(X_train_cnn, y_train_cat, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
    st.success("Modèle entraîné avec succès")

# === Évaluation ===
y_pred = model.predict(X_test_cnn)
y_pred_labels = np.argmax(y_pred, axis=1)

st.subheader("📊 Résultats")
st.text(classification_report(y_test, y_pred_labels))

fig_cm, ax_cm = plt.subplots(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_labels), annot=True, fmt="d", cmap="Blues", ax=ax_cm)
ax_cm.set_title("Matrice de confusion")
st.pyplot(fig_cm)

# === Prédiction multiple ===
st.header("📷 Prédiction sur plusieurs images")
img_files = st.file_uploader("Téléchargez plusieurs images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if img_files:
    for img_file in img_files:
        np_arr = np.frombuffer(img_file.read(), np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img, IMAGE_SIZE)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY).flatten()
        gray_scaled = scaler.transform([gray])
        pca_feat = pca.transform(gray_scaled)
        lda_feat = lda.transform(pca_feat)
        cnn_input = lda_feat[..., np.newaxis]
        pred = model.predict(cnn_input)
        pred_class = np.argmax(pred)
        st.image(img_resized, caption=f"Prédit : ID {pred_class + 1} ({np.max(pred)*100:.2f}% confiance)", width=200)
