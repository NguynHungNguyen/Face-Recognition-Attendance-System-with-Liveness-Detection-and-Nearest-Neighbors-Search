import PIL
from torchvision import transforms
from ClassificationNetwork import *
import torch
import sys
import faiss
import numpy as np
from sklearn.cluster import KMeans
import time

# Initialize the model
model = ClassificationNetwork(4000)
model.load_state_dict(torch.load('model-classification.pth', map_location=torch.device('cpu')))

sys.argv[1]
sys.argv[2]

# Load known face embeddings and names (if any)
known_face_encodings = []
known_face_names = []
known_face_clusters = []  # Store cluster labels
with open("ClassiClustDatabase.txt", "r") as f:
    for line in f:
        parts = line.strip().split("|")
        if len(parts) == 3:  # Assuming format is <Name>|<[embeddings]|<ClusterLabel>
            name, embedding_str, cluster_label = parts
            embedding_str = embedding_str.strip('[]')
            embedding = np.array([float(x) for x in embedding_str.split(",")])  # Convert to NumPy array
            if len(embedding) == 1280:  # Check if the embedding has the correct length
                known_face_encodings.append(embedding)
                known_face_names.append(name)
                known_face_clusters.append(int(cluster_label))
            else:
                print(f"Skipping incomplete embedding: {name}")  # Handle incomplete embeddings

# Create Faiss index if there are embeddings
if known_face_encodings:
    # Create a Faiss index for efficient nearest neighbor search
    index = faiss.IndexFlatL2(1280)  # Assuming your embedding dimension is 512
    index.add(np.array(known_face_encodings))  # Add embeddings to Faiss index
else:
    print("No embeddings found in ClassiClustDatabase.txt. Faiss index will be created after the first upload.")

# Function to upload a new image and perform clustering
def upload_image(image_path, name):
    global known_face_encodings, known_face_names, known_face_clusters, index
    start_time = time.time()

    img = PIL.Image.open(image_path).convert("RGB")
    img = img.resize((64, 64))

    img = transforms.ToTensor()(img)
    img = torch.unsqueeze(img, 0)
    img_embed = model(img.float())[0].detach().numpy()

    # Handle first image upload (no Faiss index yet)
    if not known_face_encodings:
        # Assign cluster label 0 to the first image
        cluster_label = 0
        # Write to face_embeddings.txt
        embedding_str = ",".join(str(x) for x in img_embed.tolist())
        with open("ClassiClustDatabase.txt", "a") as f:
            f.write(f"{name}|{embedding_str}|{cluster_label}\n") 

        # Update known_face_encodings, known_face_names, and known_face_clusters
        known_face_encodings.append(img_embed)
        known_face_names.append(name)
        known_face_clusters.append(cluster_label)
        print(known_face_encodings)

        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Processing time: {processing_time:.4f} seconds")

        return None  # Nothing to compare yet

    # Find nearest neighbors using Faiss
    distances, indices = index.search(img_embed.reshape(1, -1), 10)  # Search for 10 nearest neighbors

    # Perform clustering using KMeans
    kmeans = KMeans(n_clusters=5, random_state=0)  # Adjust the number of clusters as needed
    kmeans.fit(np.array(known_face_encodings)[indices[0]])  # Fit KMeans on the nearest neighbors

    # Assign the new image to the appropriate cluster based on its embedding
    cluster_label = kmeans.labels_[0]  # Get the cluster label for the new image

    # Find the closest known image in the cluster
    closest_index = np.argmin(distances[0])
    closest_name = known_face_names[indices[0][closest_index]]

    # Write to face_embeddings.txt with cluster information
    embedding_str = ",".join(str(x) for x in img_embed.tolist())
    with open("ClassiClustDatabase.txt", "a") as f:
        f.write(f"{name}|{embedding_str}|{cluster_label}\n") 

    # Update known_face_encodings, known_face_names, and known_face_clusters
    known_face_encodings.append(img_embed)
    known_face_names.append(name)
    known_face_clusters.append(cluster_label)

    # Update Faiss index
    index.add(img_embed.reshape(1, -1))

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Processing time: {processing_time:.4f} seconds")

    return closest_name

# Example usage
image_path = f"image/{sys.argv[1]}"
name = sys.argv[2]
closest_name = upload_image(image_path, name)
print(f"Closest known image: {closest_name}")