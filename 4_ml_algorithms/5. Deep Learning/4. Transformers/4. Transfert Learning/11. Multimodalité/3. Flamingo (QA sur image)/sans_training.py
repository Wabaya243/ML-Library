from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

device = "cuda"
model_name = "openflamingo/OpenFlamingo-3B-vitl-mpt1b"

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(model_name).to(device)

image = Image.open("images/car.jpg").convert("RGB")
question = "Quelle est la couleur de la voiture ?"

inputs = processor(images=image, text=question, return_tensors="pt").to(device)
generated_ids = model.generate(**inputs, max_new_tokens=50)
answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(answer)
