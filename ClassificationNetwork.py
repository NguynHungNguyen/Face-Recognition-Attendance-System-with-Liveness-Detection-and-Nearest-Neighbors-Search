import torch
import torch.nn as nn

class ClassificationNetwork(nn.Module): #Classification
    def __init__(self, numClasses, feat_dim=1280):
        super(ClassificationNetwork, self).__init__()

        # Load the pretrained model
        self.net = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)

        # Remove the last layer
        self.net = nn.Sequential(*list(self.net.children())[:-1])

        # Add your custom classifier
        self.classifier = nn.Linear(feat_dim, numClasses)

    def forward(self, x):
        x = self.net(x)
        output = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1) # flatten
        classification_out = self.classifier(output)
        embedding_out = output
        return embedding_out, classification_out


