from tensorflow.keras.models import load_model

from backend.capsnet_train import SquashLayer

model = load_model('models/capsnet_model.keras', custom_objects={'SquashLayer': SquashLayer})
print("CapsNet model loaded successfully")
