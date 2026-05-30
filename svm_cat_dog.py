import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Image size
IMG_SIZE = 64
# Limit the number of images loaded per class to speed up training.
# Set to None to load the entire dataset.
MAX_IMAGES = 1000

# Dataset paths
train_cat_path = "training_set/training_set/cats"
train_dog_path = "training_set/training_set/dogs"

test_cat_path = "test_set/test_set/cats"
test_dog_path = "test_set/test_set/dogs"

# Training Data
X_train = []
y_train = []

print("Loading Training Images...")

# Load training cats
count = 0
for image_name in os.listdir(train_cat_path):
    if MAX_IMAGES is not None and count >= MAX_IMAGES:
        break
    try:
        img_path = os.path.join(train_cat_path, image_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X_train.append(img.flatten())
        y_train.append(0)  # Cat
        count += 1
    except:
        pass

# Load training dogs
count = 0
for image_name in os.listdir(train_dog_path):
    if MAX_IMAGES is not None and count >= MAX_IMAGES:
        break
    try:
        img_path = os.path.join(train_dog_path, image_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X_train.append(img.flatten())
        y_train.append(1)  # Dog
        count += 1
    except:
        pass

# Testing Data
X_test = []
y_test = []

print("Loading Testing Images...")

# Load testing cats (limit to 20% of MAX_IMAGES or 200, or load all if MAX_IMAGES is None)
test_limit = int(MAX_IMAGES * 0.2) if MAX_IMAGES is not None else None
count = 0
for image_name in os.listdir(test_cat_path):
    if test_limit is not None and count >= test_limit:
        break
    try:
        img_path = os.path.join(test_cat_path, image_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X_test.append(img.flatten())
        y_test.append(0)
        count += 1
    except:
        pass

# Load testing dogs
count = 0
for image_name in os.listdir(test_dog_path):
    if test_limit is not None and count >= test_limit:
        break
    try:
        img_path = os.path.join(test_dog_path, image_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X_test.append(img.flatten())
        y_test.append(1)
        count += 1
    except:
        pass

# Convert to NumPy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)

y_train = np.array(y_train)
y_test = np.array(y_test)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Dimensionality Reduction using PCA
print("\nApplying PCA for dimensionality reduction...")
n_components = min(100, X_train.shape[0])
pca = PCA(n_components=n_components, whiten=True, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print("PCA components computed:", n_components)

# Train SVM
print("\nTraining SVM Model (RBF Kernel)...")

# Using RBF kernel with PCA-reduced features is much faster and gives better accuracy
svm = SVC(kernel='rbf', C=5.0, gamma='scale')

svm.fit(X_train_pca, y_train)

# Prediction
y_pred = svm.predict(X_test_pca)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model, scaler, and PCA
import pickle
model_data = {
    'scaler': scaler,
    'pca': pca,
    'svm': svm,
    'img_size': IMG_SIZE
}
model_filename = 'svm_cat_dog_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model_data, f)
print(f"\nTrained model and preprocessing pipeline saved to '{model_filename}'")

# Display Predictions
plt.figure(figsize=(10,6))

for i in range(6):

    plt.subplot(2,3,i+1)

    img = X_test[i]

    img = scaler.inverse_transform([img])[0]
    img = img.reshape(64,64,3)

    prediction = svm.predict([X_test_pca[i]])[0]

    label = "Dog" if prediction == 1 else "Cat"

    plt.imshow(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title(label)
    plt.axis("off")

plt.tight_layout()
plt.show()
