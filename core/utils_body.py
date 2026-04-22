import json
import torch
from torch import nn
from torch.nn import init
from torchvision import models
from torch.nn import functional as F
import torchvision.transforms as T
import cv2
from PIL import Image
import numpy as np
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_network(network, model_path=None):
    """Load network weights from file"""
    if model_path is None:
        possible_paths = [
            os.path.join(SCRIPT_DIR, 'models', 'net_last.pth'),
            os.path.join(SCRIPT_DIR, '..', 'assets', 'models', 'net_last.pth'),
            os.path.join('models', 'net_last.pth'),
            os.path.join('assets', 'models', 'net_last.pth'),
            'models/net_last.pth',
            'models\\net_last.pth',
        ]
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
    
    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model weights")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    from system.logger_setup import setup_logger
    logger = setup_logger(__name__)
    logger.info(f'Loaded attribute model from {model_path}')
    return network


def classify_full_body(frame, x1, y1, x2, y2, model, device):
    """
    Classify body attributes from a cropped person image.
    """
    default_result = {"gender": "Unknown", "age": "adult", "age_category": "adult"}
    
    try:
        transforms = T.Compose([
            T.Resize(size=(288, 144)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Ensure valid crop coordinates
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        
        if x2 <= x1 or y2 <= y1:
            return default_result
        
        if (x2 - x1) < 20 or (y2 - y1) < 40:
            return default_result

        cropped = frame[y1:y2, x1:x2]
        
        if cropped.size == 0:
            return default_result
        
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        cropped_pil = Image.fromarray(cropped_rgb)

        frame_tensor = transforms(cropped_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model.forward(frame_tensor)

        # Decode predictions - pass raw sigmoid outputs
        Dec = predict_decoder('market')
        preds = Dec.decode(out)
        
        # Extract and normalize age
        age = extract_age_from_predictions(preds)
        preds['age'] = age
        preds['age_category'] = categorize_age(age)
        
        if 'gender' not in preds or preds['gender'] is None:
            preds['gender'] = 'Unknown'
        
        return preds
        
    except Exception as e:
        print(f"Error in classify_full_body: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return default_result


def extract_age_from_predictions(preds):
    """Extract age from attribute predictions"""
    # Check for age-related keys in predictions
    if preds.get('young'):
        return 'child'
    elif preds.get('teenager'):
        return 'teenager'
    elif preds.get('old'):
        return 'senior'
    elif preds.get('adult'):
        return 'adult'
    
    # Check 'age' key directly
    age_val = preds.get('age')
    if age_val in ['young', 'child']:
        return 'child'
    elif age_val == 'teenager':
        return 'teenager'
    elif age_val in ['old', 'senior']:
        return 'senior'
    
    return 'adult'


def categorize_age(age):
    """Convert age to broader categories"""
    if age in ['child', 'young']:
        return 'young'
    elif age == 'teenager':
        return 'teenager'
    elif age == 'adult':
        return 'adult'
    elif age in ['senior', 'old']:
        return 'senior'
    return 'adult'


def classify_body_onnx(frame, x1, y1, x2, y2, onnx_session):
    """Classifies attributes using ONNX session."""
    default_result = {"gender": "Unknown", "age": "adult", "age_category": "adult"}
    
    try:
        transforms = T.Compose([
            T.Resize(size=(288, 144)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        
        if x2 <= x1 or y2 <= y1:
            return default_result

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return default_result
            
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        cropped_pil = Image.fromarray(cropped_rgb)
        
        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name

        frame_tensor = transforms(cropped_pil).unsqueeze(0).numpy()
        out = onnx_session.run([output_name], {input_name: frame_tensor})[0]
        out = torch.from_numpy(out)

        Dec = predict_decoder('market')
        preds = Dec.decode(out)
        
        age = extract_age_from_predictions(preds)
        preds['age'] = age
        preds['age_category'] = categorize_age(age)
        
        if 'gender' not in preds or preds['gender'] is None:
            preds['gender'] = 'Unknown'
        
        return preds
        
    except Exception as e:
        print(f"Error in classify_body_onnx: {e}")
        return default_result


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class Backbone_nFC(nn.Module):
    def __init__(self, class_num, model_name='resnet50_nfc'):
        super(Backbone_nFC, self).__init__()
        self.model_name = model_name
        self.backbone_name = model_name.split('_')[0]
        self.class_num = class_num

        model_ft = getattr(models, self.backbone_name)(pretrained=True)
        if 'resnet' in self.backbone_name:
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.fc = nn.Sequential()
            self.features = model_ft
            self.num_ftrs = 2048
        elif 'densenet' in self.backbone_name:
            model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.fc = nn.Sequential()
            self.features = model_ft.features
            self.num_ftrs = 1024
        else:
            raise NotImplementedError
        
        for c in range(self.class_num):
            self.__setattr__('class_%d' % c, ClassBlock(input_dim=self.num_ftrs, class_num=1, activ='sigmoid'))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        pred_label = [self.__getattr__('class_%d' % c)(x) for c in range(self.class_num)]
        pred_label = torch.cat(pred_label, dim=1)
        return pred_label


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num=1, activ='sigmoid', num_bottleneck=512):
        super(ClassBlock, self).__init__()

        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=0.5)]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        if activ == 'sigmoid':
            classifier += [nn.Sigmoid()]
        elif activ == 'softmax':
            classifier += [nn.Softmax()]
        elif activ == 'none':
            classifier += []
        else:
            raise AssertionError("Unsupported activation: {}".format(activ))
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


class predict_decoder(object):
    """
    Decoder for attribute predictions.
    
    The model outputs sigmoid probabilities for each attribute.
    For binary attributes like gender: 
        - Output close to 0 = first choice (male)
        - Output close to 1 = second choice (female)
    """
    
    _label_cache = None
    _attribute_cache = None

    def __init__(self, dataset):
        self.dataset = dataset
        
        # Load files only once (cached at class level)
        if predict_decoder._label_cache is None:
            label_path = self._find_file('label.json')
            with open(label_path, 'r') as f:
                predict_decoder._label_cache = json.load(f)
                
        if predict_decoder._attribute_cache is None:
            attr_path = self._find_file('attribute.json')
            with open(attr_path, 'r') as f:
                predict_decoder._attribute_cache = json.load(f)
        
        self.label_list = predict_decoder._label_cache.get(dataset, [])
        self.attribute_dict = predict_decoder._attribute_cache.get(dataset, {})
        self.num_label = len(self.label_list)
        
        # Find gender index
        self.gender_idx = None
        for idx, label in enumerate(self.label_list):
            if label == 'gender':
                self.gender_idx = idx
                break

    def _find_file(self, filename):
        """Search for a file in common locations"""
        possible_paths = [
            filename,
            os.path.join(SCRIPT_DIR, filename),
            os.path.join(SCRIPT_DIR, '..', 'assets', filename),
            os.path.join(os.getcwd(), filename),
            os.path.join(os.getcwd(), 'assets', filename),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Could not find {filename}")

    def decode(self, raw_output):
        """
        Decode raw sigmoid outputs to human-readable attributes.
        Works with both torch tensors and numpy arrays.
        """
        pairs = {}

        # Handle both torch tensors and numpy arrays
        if hasattr(raw_output, 'squeeze'):
            raw_output = raw_output.squeeze()
        if hasattr(raw_output, 'numpy') and not isinstance(raw_output, np.ndarray):
            raw_output = raw_output.detach().cpu().numpy()

        threshold = 0.5
        n = min(self.num_label, len(raw_output))

        for idx in range(n):
            label_name = self.label_list[idx]

            if label_name not in self.attribute_dict:
                continue

            attr_description, choices = self.attribute_dict[label_name]
            prob = float(raw_output[idx])

            pred_idx = 1 if prob > threshold else 0

            if label_name == 'gender':
                pairs['gender'] = choices[pred_idx]
                continue

            if label_name in ['young', 'teenager', 'adult', 'old']:
                if prob > threshold:
                    pairs[label_name] = True
                continue

            if pred_idx < len(choices) and choices[pred_idx] is not None:
                pairs[attr_description] = choices[pred_idx]

        if 'gender' not in pairs:
            if self.gender_idx is not None and self.gender_idx < n:
                prob = float(raw_output[self.gender_idx])
                choices = self.attribute_dict['gender'][1]
                pairs['gender'] = choices[1] if prob > 0.5 else choices[0]
            else:
                pairs['gender'] = 'Unknown'

        return pairs

    def decode_raw(self, raw_output) -> "np.ndarray":
        """
        Return the raw sigmoid probabilities as a float32 numpy array.
        Works with both torch tensors and numpy arrays.
        """
        import numpy as np
        if hasattr(raw_output, 'squeeze'):
            raw_output = raw_output.squeeze()
        if hasattr(raw_output, 'numpy') and not isinstance(raw_output, np.ndarray):
            raw_output = raw_output.detach().cpu().numpy()

        n = min(self.num_label, len(raw_output))
        vec = np.zeros(self.num_label, dtype=np.float32)
        for i in range(n):
            vec[i] = float(raw_output[i])
        return vec