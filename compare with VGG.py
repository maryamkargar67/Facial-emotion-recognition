import matplotlib.pyplot as plt

# رسم نمودار دقت
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train - CNN')
plt.plot(history.history['val_accuracy'], label='Val - CNN')
plt.plot(history_finetune.history['accuracy'], label='Train - VGG16')
plt.plot(history_finetune.history['val_accuracy'], label='Val - VGG16')
plt.title('Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# رسم نمودار خطا
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train - CNN')
plt.plot(history.history['val_loss'], label='Val - CNN')
plt.plot(history_finetune.history['loss'], label='Train - VGG16')
plt.plot(history_finetune.history['val_loss'], label='Val - VGG16')
plt.title('Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
