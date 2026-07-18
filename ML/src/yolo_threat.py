'''
This is an adapted version of a pretrained YOLO model for binary classification of threats.
It loads the pretrained yolo5n6 model, modifies the last layer to have a single predictive category, and adds a linear layer to take the predictions from the yolo model and output a single value.
'''
import torch
import torch.nn.functional as F
import torch.nn as nn
from yolo import Segment

class YoloThreat(nn.Module):
    pixel = 128
    base_model = 'yolov5n6'
    upper_layer = 34

    def __init__(self, yolo_model):
        super(YoloThreat, self).__init__()
        self.yolo = yolo_model
        #self.yolo_pred_count = 102000
        self.yolo_pred_count = 1020
        self.fc = nn.Linear(self.yolo_pred_count * 2, 1)
        self.sigmoid = nn.Sigmoid()


    '''
    There are a few things to note in the forward method:
    1. When we are in evaluation mode, we rescale the input to the pixel size of the yolo model. This is to faciliate deployment for real time inference. When in training, we will ensure that the images are preprocessed to the correct size. This is for convenience but also performance.
    2. The Yolo model returns different types of inputs depending on if it is in training or evaluation mode. We will ashew this distinction. To do so, a section of the Yolo source code was adapted to reshape the output tensor to the same shape in both training and evaluation. This is done in the collect method.
    3. The output of the model breaks the image up into a grid of cells. Each cell has a number of predictions. The objectness and class probabilities detmine the probability of an object being present in a cell and the probability of an object belonging to a certain class. Since we are only using one class, there are two vectors for each image, which we concatenate and push through a linear layer to get a single value.
    '''
    def forward(self, x):
        # Just incase the input is not batched, we add a batch dimension
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        # In evaluation mode, we rescale the input
        if not self.training:
            batch_size, _, _, _ = x.shape
            x = F.interpolate(x, size=(self.pixel, self.pixel), mode='bilinear', align_corners=False)

        yolo_out = self.yolo(x)
        if self.training:
            yolo_out = self.collect(yolo_out)
        else:
            yolo_out = yolo_out[-1]

        objectness = yolo_out[:, :, 4]
        class_probs = yolo_out[:, :, 5]
        yolo_out = torch.cat((objectness, class_probs), dim=-1)
        yolo_out = yolo_out.view(-1, self.yolo_pred_count * 2)

        logit = self.fc(yolo_out)
        return logit
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.sigmoid(self.forward(x))
    
    # Adapted from Yolo source code. This reshapes the output tensor so we get the same shape in training and evaluation
    def collect(self, x):
        yolo = self.yolo.model[-1]  # Detect()
        z = []
        for i in range(yolo.nl):
            bs, _, ny, nx, _ = x[i].shape
            y = x[i].view(bs, yolo.na, yolo.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if yolo.dynamic or yolo.grid[i].shape[2:4] != x[i].shape[2:4]:
                        yolo.grid[i], yolo.anchor_grid[i] = yolo._make_grid(nx, ny, i)

            if isinstance(yolo, Segment):  # (boxes + masks)
                xy, wh, conf, mask = x[i].split((2, 2, yolo.nc + 1, yolo.no - yolo.nc - 5), 4)
                xy = (xy.sigmoid() * 2 + yolo.grid[i]) * yolo.stride[i]  # xy
                wh = (wh.sigmoid() * 2) ** 2 * yolo.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
            else:  # Detect (boxes only)
                xy, wh, conf = x[i].sigmoid().split((2, 2, yolo.nc + 1), 4)
                xy = (xy * 2 + yolo.grid[i]) * yolo.stride[i]  # xy
                wh = (wh * 2) ** 2 * yolo.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf), 4)
            z.append(y.view(bs, yolo.na * nx * ny, yolo.no))

        return torch.cat(z, 1)
    
    '''
    Freeze all yolo layers below the specified layer number, and unfreeze the rest
    Enter 0 to unfreeze all layers
    '''
    def freeze_below(self, layer):
         if layer < 0 or layer > 34:
            raise ValueError('Layer number must be between 0 and 34')

         for name, param in self.yolo.named_parameters():
            # See YoloLayer class for more information: This just verfiies that the layer is a yolo layer and makes a comparison based on the layer number
            yolo_layer = YoloThreat.YoloLayer(name, param)
            if yolo_layer:
                if yolo_layer < layer:
                    param.requires_grad = False
                    continue
            param.requires_grad = True


    '''
    This class is a useful abstraction for the different layers in the Yolo model. By creating one of these, it checks the name of the layer and determines if it is a yolo layer.
    This distinguishes the core layers from the prediction projection at the end, which we added.
    It also determines from the name the layer that the paramter is involved in, and allows for it to be compared to integers.

    This construct is used both in the YoloThreat.freeze_below method and in L2 regularization in the fine tuning process.
    '''
    class YoloLayer:
        def __init__(self, name, param):
            self.name_parts = name.split('.')
            self.param = param
            self.is_yolo = self.is_yolo()
            self.layer = self.get_layer()

        def is_yolo(self):
            return self.name_parts[0] == 'yolo'
        
        def get_layer(self):
            if not self.is_yolo:
                return None
            return int(self.name_parts[2])
        
        def __bool__(self):
            return self.is_yolo
        
        def __lt__(self, layer_num):
            return self.layer < layer_num

    '''
    This method loads the pretrained yolo model and adapts it for use in binary classification.
    As a static method, it should be used as the main entry point to instantiate a YoloThreat object.
    '''
    @staticmethod
    def load_new_model():
        yolo_model = torch.hub.load('ultralytics/yolov5', YoloThreat.base_model, pretrained=True)
        yolo_model = yolo_model.model.model

        num_anchors = yolo_model.model[-1].na  
        yolo_model.model[-1].nc = 1 # Number of outputs for binary classification
        num_anchors = 1
        yolo_model.model[-1].no = num_anchors * (yolo_model.model[-1].nc + 5)  

        for i, conv in enumerate(yolo_model.model[-1].m):
            in_channels = conv.in_channels
            out_channels = yolo_model.model[-1].no * yolo_model.model[-1].na  

            yolo_model.model[-1].m[i] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding
            )

            torch.nn.init.normal_(yolo_model.model[-1].m[i].weight, mean=0.0, std=0.01)
            torch.nn.init.constant_(yolo_model.model[-1].m[i].bias, 0.0)

        return YoloThreat(yolo_model)
    
def transform_image(image):
    return F.interpolate(image, size=(YoloThreat.pixel, YoloThreat.pixel), mode='bilinear', align_corners=False)