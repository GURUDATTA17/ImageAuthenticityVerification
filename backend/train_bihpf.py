import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import local_binary_pattern
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.filters import laplace
from scipy.fftpack import fft2, ifft2, fftshift
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import tensorflow as tf
import warnings

# Suppress the floating-point warning
warnings.filterwarnings('ignore', category=UserWarning)

# Extracting features using BIHPF (Frequency and Spatial Domain Filtering)
def extract_bihpf_features(image, num_patches=8, radius=3, n_points=16, freq_cutoff=0.05):
    # Convert to uint8 to avoid floating-point warnings
    gray = (rgb2gray(image) * 255).astype(np.uint8)
    patches = []
    h, w = gray.shape
    patch_h, patch_w = h // num_patches, w // num_patches

    # Apply frequency domain high-pass filter (Fourier transform)
    fft_image = fftshift(fft2(gray))
    center_h, center_w = h // 2, w // 2
    mask = np.ones_like(fft_image)
    mask[center_h-int(freq_cutoff*h):center_h+int(freq_cutoff*h),
         center_w-int(freq_cutoff*w):center_w+int(freq_cutoff*w)] = 0
    filtered_fft_image = fft_image * mask
    high_freq_image = np.abs(ifft2(filtered_fft_image))

    # Apply spatial high-pass filter (e.g., Laplacian)
    high_freq_image = laplace(high_freq_image)

    # Divide the image into patches and extract LBP histograms
    for i in range(num_patches):
        for j in range(num_patches):
            patch = high_freq_image[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
            lbp = local_binary_pattern(patch, n_points, radius, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            patches.append(hist)

    return np.concatenate(patches)

# Load dataset and extract features
def load_dataset(data_dir, target_size=(224, 224)):
    X, y = [], []
    for label, folder in enumerate(['training_real', 'training_fake']):
        folder_path = os.path.join(data_dir, folder)
        files = os.listdir(folder_path)
        total_files = len(files)
        print(f"Processing {total_files} images in folder '{folder}'...")

        for idx, filename in enumerate(files):
            if idx % 1000 == 0 and idx > 0:
                print(f"Processed {idx}/{total_files} images from '{folder}'...")
            img_path = os.path.join(folder_path, filename)
            img = imread(img_path)
            img = resize(img, target_size, anti_aliasing=True)
            features = extract_bihpf_features(img)
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

# Train the BIHPF model using cross-validation and hyperparameter tuning
def train_bihpf_model(data_dir):
    print("Loading and preprocessing dataset...")
    X, y = load_dataset(data_dir)

    print("Standardizing features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print("Splitting dataset into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Tuning Random Forest Classifier with Cross-Validation...")

    # Random Forest with more hyperparameter tuning and Cross-Validation
    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("Best parameters found:", grid_search.best_params_)
    best_rf = grid_search.best_estimator_

    # Perform Cross-Validation
    print("Performing Cross-Validation...")
    cross_val_accuracy = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {cross_val_accuracy.mean():.4f} (+/- {cross_val_accuracy.std() * 2:.4f})")

    print("Evaluating on validation set...")
    y_pred = best_rf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['Real', 'Fake']))

    print("Saving the best model and scaler...")
    # Save model using pickle (since Keras doesn't directly support scikit-learn models)
    import pickle
    with open('bihpf_model.pkl', 'wb') as f:
        pickle.dump(best_rf, f)

    # Save scaler parameters
    np.savez('bihpf_scaler_params.npz', 
             mean=scaler.mean_, 
             scale=scaler.scale_)
    
    print("Model saved as 'bihpf_model.pkl'")
    print("Scaler parameters saved as 'bihpf_scaler_params.npz'")

    return best_rf, scaler

def load_model_and_scaler():
    import pickle
    
    # Load model
    with open('bihpf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load scaler
    scaler_params = np.load('bihpf_scaler_params.npz')
    scaler = StandardScaler()
    scaler.mean_ = scaler_params['mean']
    scaler.scale_ = scaler_params['scale']
    scaler.n_samples_seen_ = len(scaler.mean_)
    
    return model, scaler

if __name__ == "__main__":
    data_dir = "real_and_fake_face"
    train_bihpf_model(data_dir)