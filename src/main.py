from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import glob

# Rozdeleni indexu do kategorii
plastic = [0, 1, 2, 3, 4]
paper = [7, 8, 9]
metal = [5, 6]
glass = [10, 11, 12]
beverage_carton = [13, 14]

np.set_printoptions(suppress=True)

# Nacteni modelu
model = load_model("./model/keras_model.h5", compile=False)

# Nacteni ciselnych kodu materialu k indexum
class_names = open("./model/labels.txt", "r").readlines()

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Nacteni obrazku
image_list = map(Image.open, glob.glob("./test/*.jpg"))

size = (224, 224)

for image in image_list:
    image_name = image.filename
    image = image.convert("RGB")

    # Zmenseni obrazku na 224x224
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalizace obrazku
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Nacteni obrazku do pole
    data[0] = normalized_image_array

    # Predikce modelu
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index][2:]
    confidence_score = prediction[0][index]

    # Urceni kategorie a kontejneru podle vysledneho indexu
    if index in plastic:
        class_category = "Plastic"
        class_container = "Plastic (yellow) container"
    elif index in paper:
        class_category = "Paper"
        class_container = "Paper (blue) container"
    elif index in metal:
        class_category = "Metal"
        class_container = "Metal (gray) container"
    elif index in glass:
        class_category = "Glass"
        class_container = "Glass (green) container"
    elif index in beverage_carton:
        class_category = "Beverage carton"
        class_container = "Plastic (yellow) container"
    else:
        class_category = ""
        class_container = ""

    # Vypsaní vysledku
    print("File:", image_name)
    print("Class:", class_name, end="")
    print("Class category:", class_category)
    print("Class container:", class_container)
    print("Confidence Score:", confidence_score)
