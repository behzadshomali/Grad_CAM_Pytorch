import torch.nn as nn


class MyVGG(nn.Module):
    def __init__(self, vgg):
        super(MyVGG, self).__init__()

        self.vgg = vgg
        self.cnn_features = self.vgg.features[:35]
        self.remain_features = self.vgg.features[35:]
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classfier = self.vgg.classifier
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.cnn_features(x)
        x.register_hook(self.activations_hook)
        x = self.remain_features(x)
        x = self.avg_pool(x)
        x = x.view((1, -1))
        x = self.classfier(x)

        return x

    def get_cnn_features(self, x):
        return self.cnn_features(x)

    def get_features_gradients(self):
        return self.gradients
