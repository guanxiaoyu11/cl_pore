import torch
import torch.nn as nn
from torch.autograd import Variable


class Classifier(nn.Module):
    def __init__(self, num_classes=2):
        
        super().__init__()

        
        self.fc_block = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        
        self.projection_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Linear(16, 8),
            nn.BatchNorm1d(8)
        )

       
        self.classification_head = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.Linear(16, num_classes)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, x1=None):
        
       
        if x1 is not None:
           
            features_x = self.extract_features(x)
            features_x1 = self.extract_features(x1)

            projection_x = self.projection_head(features_x)
            projection_x1 = self.projection_head(features_x1)

            classification_x = self.classification_head(projection_x)
            classification_x1 = self.classification_head(projection_x1)

            return classification_x, classification_x1, features_x, features_x1
        else:
           
            features = self.extract_features(x)
            projection = self.projection_head(features)
            classification = self.classification_head(projection)

            return classification, features, projection

    def extract_features(self, x):
        
       
        batch_size, _, seq_len = x.size()

        if not hasattr(self, 'feature_extractor'):
            
            self.feature_extractor = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),

                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),

                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),

                nn.Conv1d(256, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1)  
            )

        features = self.feature_extractor(x)
        features = features.view(batch_size, -1)

        return features


class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=2):
        
        super().__init__()

       
        self.fc_block = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

       
        self.projection_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Linear(16, 8),
            nn.BatchNorm1d(8)
        )

        
        self.classification_head = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        
        features = self.fc_block(x)  # [batch_size, 64]
        projection = self.projection_head(features)  # [batch_size, 8]
        classification = self.classification_head(projection)  # [batch_size, num_classes]

        return classification, features, projection

