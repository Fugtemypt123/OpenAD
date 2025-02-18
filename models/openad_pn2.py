import torch
import torch.nn as nn
import clip
import torch.nn.functional as F
import numpy as np
from .pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation


class ClassEncoder(nn.Module):
    def __init__(self):
        super(ClassEncoder, self).__init__()
        self.device = torch.device('cuda')
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
    def forward(self, classes):
        tokens = clip.tokenize(classes).to(self.device)
        text_features = self.clip_model.encode_text(tokens).to(self.device).permute(1, 0).float()
        return text_features

cls_encoder = ClassEncoder()

class OpenAD_PN2(nn.Module):
    def __init__(self, args, num_classes, normal_channel=False):
        super(OpenAD_PN2, self).__init__()

        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [
                                             32, 64, 128], 3+additional_channel, [[32, 32, 64],\
                                                [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(
            128, [0.4, 0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)

        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(
            in_channel=134+additional_channel, mlp=[128, 128])
        
        self.conv1 = nn.Conv1d(128, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @classmethod
    def from_checkpoint(cls, CLPP_layers: Tuple, OpenAD_layers: Tuple):
        model = OpenAD_PN2(args=None, num_classes=None, normal_channel=False)
        (model.fp3, model.fp2, model.fp1, model.bn1, model.conv1) = OpenAD_layers
        (model.sa1, model.sa2, model.sa3) = CLPP_layers
        model.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        return model
    
    def forward(self, xyz, affordance):
        # Set Abstraction layers
        xyz = xyz.contiguous()
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_xyz = xyz
            l0_points = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # print(l1_xyz.size(), l1_points.size())
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # print(l2_xyz.size(), l2_points.size())
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # print(l3_xyz.size(), l3_points.size())
        
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        # print(l2_points.size())
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # print(l1_points.size())
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat(
            [l0_xyz, l0_points], 1), l1_points)
        # print(l0_points.size())

        l0_points = self.bn1(self.conv1(l0_points))

        # cosine similarity

        l0_points = l0_points.permute(0, 2, 1).float()
        with torch.no_grad():
            text_features = cls_encoder(affordance)
            # print(text_features.size())
            # print(l0_points.size())
        x = (self.logit_scale * (l0_points @ text_features) / (torch.norm(l0_points, dim=2, keepdim=True)\
            @ torch.norm(text_features, dim=0, keepdim=True))).permute(0, 2, 1)
        
        x = F.log_softmax(x, dim=1)
        return x


class OpenAD_PN2_CLPP(nn.Module):
    def __init__(self, args, num_classes, normal_channel=False):
        super().__init__()
        
        self.cls_encoder = ClassEncoder()
        # Make the text encoder trainable during CLPP training
        # for param in self.cls_encoder.parameters():
        #     param.requires_grad = True
        
        print("Text encoder main trainable during CLPP training")

        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [
                                             32, 64, 128], 3+additional_channel, [[32, 32, 64],\
                                                [64, 64, 128], [64, 96, 128]])
        #for param in self.sa1.parameters():
        #    param.requires_grad = False
        
        self.sa2 = PointNetSetAbstractionMsg(
            128, [0.4, 0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        
        #for param in self.sa2.parameters():
         #   param.requires_grad = False

        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        
        #for param in self.sa3.parameters():
         #   param.requires_grad = False
       
        self.conv2 = nn.Conv1d(1024, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, xyz, text):
        # Set Abstraction layers
        xyz = xyz.contiguous()
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_xyz = xyz
            l0_points = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # print(l1_xyz.size(), l1_points.size())
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # print(l2_xyz.size(), l2_points.size())
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l3_points = self.bn1(self.conv2(l3_points))

        l3_points = l3_points.squeeze(-1)

        # cosine similarity

        l3_points = l3_points.float()
        
        text_features = self.cls_encoder(text) # Use the trainable text encoder
            # print(text_features.size())
            # print(l3_points.size())
            # raise ValueError("With great power comes great responsibility")
        x = (self.logit_scale * (l3_points @ text_features) / (torch.norm(l3_points, dim=1, keepdim=True)\
            @ torch.norm(text_features, dim=0, keepdim=True)))
        # print(f'x: {x}')
        x = F.log_softmax(x, dim=1)
        # print(f'x: {x}')
        return x

    def get_logits(self, xyz, text):
        xyz = xyz.contiguous()
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_xyz = xyz
            l0_points = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # print(l1_xyz.size(), l1_points.size())
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # print(l2_xyz.size(), l2_points.size())
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l3_points = self.bn1(self.conv2(l3_points))

        l3_points = l3_points.squeeze(-1)

        # cosine similarity

        l3_points = l3_points.float()
        
        text_features = self.cls_encoder(text)
            # print(text_features.size())
            # print(l3_points.size())
            # raise ValueError("With great power comes great responsibility")
        x = (self.logit_scale * (l3_points @ text_features) / (torch.norm(l3_points, dim=1, keepdim=True)\
            @ torch.norm(text_features, dim=0, keepdim=True)))

        return x
