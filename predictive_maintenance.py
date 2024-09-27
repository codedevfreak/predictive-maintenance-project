# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Membaca data dari file CSV
data = pd.read_csv('C:/Users/lenovo/Documents/Predictive_Maintenance/data_sensor_mesin.csv')

# Menampilkan beberapa baris pertama untuk memverifikasi data
print("Data Mesin:\n", data.head())

# Mengecek missing values (jika ada)
print("\nMissing values:\n", data.isnull().sum())

# Memisahkan fitur (X) dan label (y)
X = data[['Suhu', 'Tekanan', 'Kecepatan_Putaran', 'Vibrasi']]
y = data['Status_Gagal']

# Membagi data menjadi data training dan data testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membangun model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Melakukan prediksi pada data testing
y_pred = model.predict(X_test)

# Menampilkan hasil akurasi
print(f"\nAkurasi model: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Menampilkan laporan klasifikasi
print("\nLaporan Klasifikasi:\n", classification_report(y_test, y_pred))

# Membuat confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Confusion Matrix')
plt.show()

# Melakukan prediksi pada data baru (misalkan data sensor baru dari mesin)
data_baru = [[85.0, 2.1, 1500, 1.6]]  # Contoh data baru (Suhu, Tekanan, Kecepatan_Putaran, Vibrasi)
prediksi_baru = model.predict(data_baru)

if prediksi_baru[0] == 1:
    print("\nMesin berisiko gagal.")
else:
    print("\nMesin dalam kondisi aman.")

    
