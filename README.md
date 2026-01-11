## **Project: Horse Breed Classification**

### **1. Problem Statement**

The objective was to develop a deep learning system capable of distinguishing between **7 different horse breeds**. This is a challenging task because many breeds share similar anatomical structures. The model must identify fine-grained details like coat patterns, muscle density, and facial proportions to achieve high accuracy.

### **2. Dataset Overview**

* **Data Type:** RGB Images.
* **Classes (7):** Akhal-Teke, Appaloosa, Arabian, Friesian, Percheron, Pony, and Shire.
* **Preprocessing:** * Resized to  pixels.
* **Normalization:** Adjusted pixel values using ImageNet means () and standard deviations () to stabilize training.
* **Splitting:** Divided into Training, Validation, and Test sets.



### **3. Model Architectures**

We compared three different approaches to evaluate performance:

1. **SimpleCNN:** A custom-built convolutional neural network used as a baseline to understand the dataset from scratch.
2. **Vision Transformer (ViT-B/16):** An attention-based model using **Transfer Learning**. We used `DEFAULT` weights and froze the backbone.
3. **EfficientNet-B0:** A state-of-the-art CNN that uses compound scaling for efficiency. We also utilized a frozen backbone for this model.

### **4. Key Techniques Implemented**

* **Frozen Backbones:** By freezing pre-trained layers, we utilized feature extraction from ImageNet, drastically reducing training time and preventing overfitting.
* **Custom Classification Heads:** Replaced the original output layers with a `Sequential` block (`Linear` -> `ReLU` -> `Dropout` -> `Linear`) to map features to our specific 7 breeds.
* **Regularization:** Used **Dropout** () in the heads to ensure the model didn't "memorize" specific training samples.
* **Model Checkpointing:** Implemented a logic to save the **Best Model** based on minimum **Validation Loss** rather than just the final epoch.

### **5. Challenges & Solutions**

* **The "Tuple" Error:** Fixed a common PyTorch `TypeError` by correctly unpacking multiple return values (Loss and Accuracy) from the training functions.
* **Device Management:** Implemented an automatic `device` selection (CUDA vs. CPU) to ensure the code is portable across different hardware.
* **Visualization Fix:** Created a **Denormalization** step in the prediction function. This allowed us to view the horse images in their original colors while displaying the model's predictions.

### **6. Results & Insights**

* **Performance:** The Vision Transformer (ViT) and EfficientNet reached higher accuracy much faster than the custom CNN due to their pre-trained knowledge.
* **Validation Behavior:** We monitored Loss and Accuracy curves to ensure convergence. The validation metrics remained stable, indicating healthy generalization.
* **Visual Confirmation:** Final testing included a grid of predictions showing the model's "guesses" vs. the actual labels, with green text for correct matches and red for errors.
