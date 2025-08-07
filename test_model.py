
mobilenet_json = model.to_json()
with open("mobilenetv2_finetuned_model.json", "w") as json_file:
    json_file.write(mobilenet_json)

model.save_weights("mobilenetv2_finetuned_model.weights.h5")

print("✅ MobileNetV2 is saved")
