import torch
import torch.nn as nn
import torch.nn.functional as F 
import copy 
from monai.networks.nets import ViT
from model_cross_attention import cross_attn_channel
from longclip_main import longclip
import numpy as np

class DeepFusion(nn.Module):
    # This was modified from https://github.com/VahidooX/DeepCCA/blob/master/models.py
    def __init__(self, outdim_size, encoder_name, fusion_method, roi_size, device):
        super(DeepFusion, self).__init__()
        self.fusion_method = fusion_method

        if self.fusion_method != 'cross_attention':
            self.view1_model = ViTAsEncoder(outdim_size, roi_size, device)
            self.view2_model = CLIPTransformerAsEncoder(outdim_size, device)
        else:
            self.view1_model = ViTAsEncoder(outdim_size, roi_size, device)
            self.view2_model = CLIPTransformerAsEncoder(outdim_size, device)
            cross_attn_layers = 2
            cross_attn_heads = 2
            self.mm_cross_attn = torch.nn.Sequential(*[
                cross_attn_channel(encoder_name, dim_m1 = outdim_size, dim_m2 = outdim_size, pffn_dim = None, heads = cross_attn_heads, seq_len = None, dropout = 0.25)
                for _ in range(cross_attn_layers)
            ])

        self.device = device
   
    def forward(self, x1, x2):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        # feature * batch_size
        output1 = self.view1_model(x1)
        output2 = self.view2_model(x2)

        # Do cross attention
        if 'cross_attention' in self.fusion_method:
            #Run through cross_attn channel
            for layer in self.mm_cross_attn:
                output1, output2 = layer(output1, output2)
            
        # avg pooling across tokens [batch, token_length, feature_dim]
        output1 = output1.mean(dim=1)
        output2 = output2.mean(dim=1)

        return output1, output2

class ViTAsEncoder(nn.Module):
    def __init__(self, outdim_size = 32, roi_size = [50, 50, 50], device = None):
        super(ViTAsEncoder, self).__init__()
        self.encoder = ViT(in_channels=1,
            img_size=(np.asarray(roi_size)),
            patch_size=(4, 4, 4),
            proj_type="conv",
            num_layers=2,
            num_heads=2,
            pos_embed_type='learnable',
            classification=False,
            num_classes=2,
            dropout_rate=0.25,
            spatial_dims=3,
            hidden_size=768,
            mlp_dim=3072)
        self.out = nn.Linear(768, outdim_size)
    
    def forward(self, x):
        x, hidden = self.encoder(x)
        x = self.out(x)
        return x
    
class CLIPTransformerAsEncoder(nn.Module):
    def __init__(self, outdim_size, device):
        super(CLIPTransformerAsEncoder, self).__init__()
        self.encoder, _ = longclip.load("/workspace/modeling/lung_data_fusion/nlst_diagnosis/cnn_version/longclip_main/longclip-B.pt", device=device)
        self.encoder = self.encoder.float()
        self.out = nn.Linear(512, outdim_size)
    
    def forward(self, x):
        x = x.squeeze(1)
        x = self.encoder.encode_text(x)
        x = self.out(x)
        return x

class MLPNetAsEncoder(nn.Module):
    def __init__(self, layer_sizes, input_size, outdim_size, dropout_rate, device):
        super(MLPNetAsEncoder, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        layer_sizes = [input_size] + layer_sizes
        self.hidden = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # hidden layers 
        for k in range(len(layer_sizes)-1):
            self.hidden.append(nn.Linear(layer_sizes[k], layer_sizes[k+1]))
            self.batch_norms.append(nn.BatchNorm1d(layer_sizes[k+1], affine=False))

        self.out = nn.Linear(layer_sizes[-1], outdim_size)
    
    def forward(self, x):
        for layer, batch_norm in zip(self.hidden, self.batch_norms):
            x = layer(x)
            # x = batch_norm(x) 
            x = F.leaky_relu(x, negative_slope=0.01) 
            x = self.dropout(x)

        x = self.out(x)
        return x
    
class MLPNetAsPredictionHead(nn.Module):
    def __init__(self, layer_sizes, input_size, outdim_size, dropout_rate, device):
        super(MLPNetAsPredictionHead, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        layer_sizes = [input_size] + layer_sizes
        self.hidden = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # hidden layers 
        for k in range(len(layer_sizes)-1):
            self.hidden.append(nn.Linear(layer_sizes[k], layer_sizes[k+1]))
            self.batch_norms.append(nn.BatchNorm1d(layer_sizes[k+1], affine=False))

        self.out = nn.Linear(layer_sizes[-1], outdim_size)
    
    def forward(self, x):
        for layer, batch_norm in zip(self.hidden, self.batch_norms):
            x = layer(x)
            x = batch_norm(x) 
            x = F.leaky_relu(x, negative_slope=0.01) 
            x = self.dropout(x)

        x = self.out(x)
        return x

class DownstreamPredictionHead(nn.Module):
    def __init__(self, fusion_method, fusion_model, layer_sizes, input_size, outdim_size, mlp_dropout=0.5, device=torch.device('cpu')):
        super(DownstreamPredictionHead, self).__init__()
        self.fusion_method = fusion_method
        self.fusion_model = copy.deepcopy(fusion_model)
        self.mlp = MLPNetAsPredictionHead(layer_sizes, input_size, outdim_size, mlp_dropout, device=device)

    def forward(self, x1, x2):
        encoder_output1, encoder_output2 = self.fusion_model(x1, x2)

        # apply activation 
        encoder_output1 = F.leaky_relu(encoder_output1, negative_slope=0.01) 
        encoder_output2 = F.leaky_relu(encoder_output2, negative_slope=0.01) 

        if self.fusion_method != 'tensor':
            fusion_output = torch.cat((encoder_output1, encoder_output2), dim=-1)

        else:
            mod0 = encoder_output1
            nonfeature_size = mod0.shape[:-1]
            m = torch.cat([mod0], dim=-1)
            fused = torch.einsum('...i,...j->...ij', m, encoder_output2)
            fusion_output = fused.reshape([*nonfeature_size, -1])
        
        mlp_output = self.mlp(fusion_output)

        return encoder_output1, encoder_output2, mlp_output
