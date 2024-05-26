import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

data_dict = pickle.load(open('./data.pickle', 'rb'))
data = data_dict['data']
labels = data_dict['labels']

max_features = max(len(sample) for sample in data)

for i in range(len(data)):
    data[i] = data[i] + [0] * (max_features - len(data[i]))

data_rf = np.array(data)
labels_rf = np.array(labels)

x_train_rf, x_test_rf, y_train_rf, y_test_rf = train_test_split(data_rf, labels_rf, test_size=0.2, shuffle=True, stratify=labels_rf)

model_rf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="sqrt")
model_rf.fit(x_train_rf, y_train_rf)

y_predict_rf = model_rf.predict(x_test_rf)
score_rf = accuracy_score(y_predict_rf, y_test_rf)
print('Random Forest Model Accuracy:', score_rf)

x_train_cnn = x_train_rf.reshape(-1, max_features, 1)
x_test_cnn = x_test_rf.reshape(-1, max_features, 1)

label_encoder = LabelEncoder()
y_train_rf_encoded = label_encoder.fit_transform(y_train_rf)
y_test_rf_encoded = label_encoder.transform(y_test_rf)

model_cnn = models.Sequential([
    layers.Conv1D(32, 3, activation='relu', input_shape=(max_features, 1)),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(np.max(y_train_rf_encoded) + 1, activation='softmax')  
])

model_cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_cnn.fit(x_train_cnn, y_train_rf_encoded, epochs=10, batch_size=32, validation_data=(x_test_cnn, y_test_rf_encoded))

test_loss, test_acc = model_cnn.evaluate(x_test_cnn, y_test_rf_encoded)
print('CNN Model Accuracy:', test_acc)
print('Random Forest Model Accuracy:', score_rf)