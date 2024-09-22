import PIL
from torchvision import transforms
from ClassificationNetwork import *
import torch
import sys

sys.argv[1]
sys.argv[2]
# Initialize the model
model = ClassificationNetwork(4000)

# Load the state dict previously saved
model.load_state_dict(torch.load('model-classification.pth', map_location=torch.device('cpu')))
#model.to(torch.device('cuda'))

# Load a sample picture and learn how to recognize it.
img1 = PIL.Image.open(f"image/{sys.argv[1]}").convert("RGB")
img1 = img1.resize((64, 64))

img1 = transforms.ToTensor()(img1)
img1 = torch.unsqueeze(img1, 0)
imgs1Embed = model(img1.float())[0]
name = sys.argv[2]

# Convert the tensor to a string representation
embedding_str = ",".join(str(x) for x in imgs1Embed.detach().numpy().tolist())

with open("ClassiDatabase.txt", "a") as f:
    f.write(f"{name}|{embedding_str}\n")