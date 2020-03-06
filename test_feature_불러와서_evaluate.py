from keras.models import load_model, Model
from keras.models import Sequential
import numpy as np
from PIL import Image
import glob

(X_train, X_test, y_train, y_test) = np.load("./numpy_data/multi_image_data.npy", allow_pickle=True)

model = load_model('./image.h5')
history = model.evaluate(X_test, y_test, batch_size=32)
print('## evaluation loss and_metrics ##')
print(history)
