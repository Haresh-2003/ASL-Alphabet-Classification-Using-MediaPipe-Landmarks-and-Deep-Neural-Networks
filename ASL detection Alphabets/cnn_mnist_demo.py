import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import pickle


CSV_PATH = r"C:\Users\HARESH PS\Downloads\archive (4)\asl_landmarks_final.csv"
data = pd.read_csv(CSV_PATH)
print(f"📂 Loaded dataset: {data.shape[0]} samples, {data.shape[1]} features")

data['label'] = data['label'].astype(str).str.strip().str.upper()
replacement_map = {
    'SPACE_': 'SPACE', 'SPC': 'SPACE', 'DEL': 'DELETE',
    'DELETE_': 'DELETE', 'NOTHING': 'SPACE', 'NONE': 'SPACE'
}
data['label'] = data['label'].replace(replacement_map)

valid_labels = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N',
    'O','P','Q','R','S','T','U','V','W','X','Y','SPACE','DELETE'
]
data = data[data['label'].isin(valid_labels)]
print(f"✅ Cleaned and filtered: {len(data)} samples remain")

print("🎯 Unique labels:", sorted(data['label'].unique()))

X = data.drop('label', axis=1).values
y_raw = data['label'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)
y = to_categorical(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"🧩 Training on {X_train.shape[0]} samples, validating on {X_val.shape[0]}")

model = Sequential([
    Dense(512, activation='relu', input_shape=(X.shape[1],), kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("🚀 Model ready. Starting training...")

early_stop = EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=100, batch_size=32, callbacks=[early_stop])

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"🎯 Final Validation Accuracy: {val_acc*100:.2f}%")

model.save("asl_landmark_model.keras")
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("✅ Model, scaler, and encoder saved successfully.")
