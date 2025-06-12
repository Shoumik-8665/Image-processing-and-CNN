
# 🐶🐱 Cat vs Dog Image Classifier using CNN (TensorFlow/Keras)

A Convolutional Neural Network (CNN)-based binary image classification model trained to distinguish between cats and dogs using TensorFlow and Keras.

---

## 📌 Project Summary

This project demonstrates how to build a deep learning model using CNNs to classify 24,000+ images of cats and dogs. The model was trained using TensorFlow and achieved significant validation accuracy in just a few epochs. The dataset was preprocessed, normalized, and saved using Python's `pickle` module.

---

## 🚀 Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook
- Pickle

---

## 🧠 Key Concepts Learned
- Image preprocessing and normalization
- Working with binary classification datasets
- Building CNN architectures using Keras
- Overcoming vanishing gradient problems with ReLU
- Monitoring training/validation performance
- Saving and loading data using `pickle`

---

## 🖼️ Dataset Description
- 📁 Source: [Kaggle Cats and Dogs Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
- 🖼️ Size: 25,000 images (12,500 cats and 12,500 dogs)
- 💾 Preprocessed into `x.pickle` and `y.pickle`

---

## 🧪 Model Architecture

```
Input Layer: Normalized 2D images (size: 100x100)
↓
Conv2D Layer (64 filters, 3x3) + ReLU
↓
MaxPooling2D (2x2)
↓
Conv2D Layer (64 filters, 3x3) + ReLU
↓
MaxPooling2D (2x2)
↓
Flatten Layer
↓
Dense Layer (64 units, ReLU)
↓
Output Layer (1 unit, Sigmoid for binary classification)
```

---

## 📊 Results

| Epoch | Accuracy | Loss  | Val Accuracy | Val Loss |
|-------|----------|-------|--------------|----------|
| 1     | 60.4%    | 0.6499| 71.3%        | 0.5515   |
| 2     | 76.8%    | 0.4895| 76.9%        | 0.4839   |
| 3     | 80.4%    | 0.4255| 78.6%        | 0.4573   |

✅ The model showed steady improvement over epochs with reduced loss and improved validation accuracy.

---

## 📝 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/cat-vs-dog-classifier.git
   cd cat-vs-dog-classifier
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have `x.pickle` and `y.pickle` in your directory.

4. Run the Jupyter notebook:
   ```bash
   jupyter notebook Image Processing.ipynb
   ```

---

## 💡 Future Improvements

- Add dropout layers to reduce overfitting
- Use data augmentation for better generalization
- Try transfer learning with pretrained models (e.g., VGG16 or ResNet50)
- Implement early stopping and learning rate scheduling

---

## 📌 License

This project is open-source and free to use under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [Kaggle for the dataset](https://www.kaggle.com/c/dogs-vs-cats)
- TensorFlow/Keras for deep learning tools
- Jupyter for interactive development
