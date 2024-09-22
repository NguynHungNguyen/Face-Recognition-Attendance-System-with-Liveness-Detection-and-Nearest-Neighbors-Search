# Face-Recognition-Attendance-System-with-Liveness-Detection-and-Nearest-Neighbors-Search
---

# Face Recognition Attendance System with Liveness Detection and Nearest Neighbors Search

This repository contains the code for a robust face recognition attendance system, developed as part of the Applied Machine Learning (COS30082) project at Swinburne University of Technology. The system includes an optimized feature for faster recognition through a nearest neighbors search in the database.

## Project Overview

The system employs advanced face verification techniques for managing employee access control, ensuring that only registered individuals can gain entry. Key features include:

- **Face Recognition:** A Siamese Network based on MobileNetV2 generates discriminative face embeddings, enabling accurate identity verification.
- **Liveness Detection:** A pre-trained MiniFASNetV2 model ([MiniFASNetV2](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)) is integrated to prevent spoofing attempts using photos or digital screens.
- **Nearest Neighbors Search:** To optimize response time, the system features a nearest neighbors search, allowing it to quickly identify the closest matches within the face embeddings stored in the database, enhancing performance and scalability.
- **User-Friendly Interface:** A web interface built with Flask provides real-time feedback for liveness, and the nearest neighbors search, offering a smooth user experience for registration and login.

## Dataset

The system was trained and evaluated using a provided face recognition dataset by dataset from (https://www.kaggle.com/c/11-785-fall-20-homework-2-part-2/overview)

## Model Architecture and Training

- **Metric Learning with MobileNetV2:** The core of the system is a Siamese Network that uses a shared MobileNetV2 backbone (pre-trained on ImageNet) for generating face embeddings. The model is trained with triplet loss to minimize the distance between similar embeddings and maximize the distance between dissimilar ones.
- **Spoofing Detection:** A pre-trained MiniFASNetV2 model detects potential spoofing attempts by analyzing captured images.
- **Nearest Neighbors Search:** The system accelerates face recognition by using a nearest neighbors search on the face embeddings, ensuring quick and accurate retrieval from the database.

## Running the System

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/NguynHungNguyen/Face-Recognition-Attendance-System-with-Liveness-Detection-and-Nearest-Neighbors-Search
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Model Training (Optional):**  
   The pre-trained model weights are included, but you can retrain the model using notebook: TrainModel

5. **Run the Application:**
   For non-nearest search
   ```bash
   python Detect.py
   ```
   
   For nearest search
   ```bash
   python DetectFaiss.py
   ```
7. **Access the Interface:**  
   Open a web browser and navigate to `http://127.0.0.1:5000/` to access the attendance system interface.

## Evaluation

The system demonstrated strong performance in face verification with an AUC score of 0.71 on the test set. The addition of the nearest neighbors search further enhances recognition speed, complementing the system’s accurate spoofing capabilities.

## Innovation

This system innovatively integrates nearest neighbors search for rapid database lookups, significantly improving authentication speed and reliability.

## Future Work

- Explore techniques to further reduce the validation loss of the Siamese Network through regularization and hyperparameter tuning.
- Investigate incorporating additional face recognition models or ensembles for improved accuracy.
- Implement more sophisticated spoofing attacks and countermeasures to enhance the system's robustness against evolving threats.

## Acknowledgements

- [Minivision AI](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) for providing the MiniFASNetV2 model.

---
