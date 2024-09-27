import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Misalkan ini adalah label asli dan prediksi dari model Anda
y_true = [1, 0, 1, 1, 0, 0, 1, 1]
y_pred = [1, 0, 1, 0, 0, 1, 1, 1]

# Membuat confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix menggunakan seaborn
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

# Menyimpan visualisasi ke file gambar PNG
plt.savefig('C:/Users/lenovoD/ocuments/Predictive_Maintenance/confusion_matrix.png')

# Menampilkan visualisasi (opsional)
plt.show()
