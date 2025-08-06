import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ========================
# 📁 مسیر فایل‌ها
cnn_json = "facialemotionmodel.json"
cnn_weights = "facialemotionmodel.weights.h5"
mobilenet_json = "mobilenetv2_finetuned_model.json"
mobilenet_weights = "mobilenetv2_finetuned_model.weights.h5"

# ========================
# 🧠 بررسی وجود فایل‌ها
missing_files = []

for f in [cnn_json, cnn_weights, mobilenet_json, mobilenet_weights]:
    if not os.path.exists(f):
        missing_files.append(f)

if missing_files:
    print("❌ فایل‌های زیر پیدا نشدند:")
    for f in missing_files:
        print(" -", f)
    raise FileNotFoundError("⛔ لطفاً مطمئن شوید فایل‌های بالا در مسیر اسکریپت وجود دارند.")

# ========================
# ✅ لود مدل‌ها
with open(cnn_json, "r") as f:
    cnn_model = model_from_json(f.read())
cnn_model.load_weights(cnn_weights)
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

with open(mobilenet_json, "r") as f:
    mobilenet_model = model_from_json(f.read())
mobilenet_model.load_weights(mobilenet_weights)
mobilenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ========================
# 🧪 Data generator
test_dir = '/Users/mary/Desktop/ML/FER2013/archive/test'
test_gen = ImageDataGenerator(rescale=1./255)

test_gen_cnn = test_gen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

test_gen_mobilenet = test_gen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# ========================
# 📊 Evaluate both models
cnn_loss, cnn_acc = cnn_model.evaluate(test_gen_cnn, verbose=0)
mobilenet_loss, mobilenet_acc = mobilenet_model.evaluate(test_gen_mobilenet, verbose=0)

print(f"✅ CNN Accuracy: {cnn_acc:.4f} | Loss: {cnn_loss:.4f}")
print(f"✅ MobileNetV2 Accuracy: {mobilenet_acc:.4f} | Loss: {mobilenet_loss:.4f}")

plt.bar(['CNN', 'MobileNetV2'], [cnn_acc, mobilenet_acc], color=['skyblue', 'lightgreen'])
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# ========================
# 📈 Confusion Matrix for MobileNet
y_true = test_gen_mobilenet.classes
y_pred = np.argmax(mobilenet_model.predict(test_gen_mobilenet), axis=1)
labels = list(test_gen_mobilenet.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=labels)
plt.figure(figsize=(8, 8))
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("MobileNetV2 - Confusion Matrix")
plt.show()

print("📋 Classification Report (MobileNetV2):")
print(classification_report(y_true, y_pred, target_names=labels))
