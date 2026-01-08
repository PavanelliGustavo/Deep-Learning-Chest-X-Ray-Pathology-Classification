import torch
import torch.nn as nn
import torchvision.models as models

class DenseNet169Modified(nn.Module):
    def __init__(self, num_classes=3):
        super(DenseNet169Modified, self).__init__()
        # Carrega pesos da ImageNet
        self.densenet = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
        
        # Modifica a primeira camada para aceitar 1 canal (Grayscale) em vez de 3
        conv0 = self.densenet.features.conv0
        new_conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Copia a média dos pesos dos canais RGB para o canal único
        with torch.no_grad():
            new_conv0.weight[:] = conv0.weight.mean(dim=1, keepdim=True)
        self.densenet.features.conv0 = new_conv0
        
        # Modifica a última camada (classificador)
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x):
        return self.densenet(x)
    