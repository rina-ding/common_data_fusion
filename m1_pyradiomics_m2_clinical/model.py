import torch
import torch.nn as nn
import torch.nn.functional as F 
import copy 

class DeepFusion(nn.Module):
    def __init__(self, encoder_type, layer_sizes1, layer_sizes2, input_size1, input_size2, outdim_size, dropout_rate, use_all_singular_values, device=torch.device('cpu')):
        super(DeepFusion, self).__init__()
        if encoder_type == 'mlp':
            self.view1_model = MLPNetAsEncoder(layer_sizes1, input_size1, outdim_size, dropout_rate, device=device)
            self.view2_model = MLPNetAsEncoder(layer_sizes2, input_size2, outdim_size, dropout_rate, device=device)
        elif encoder_type == 'linear':
            self.view1_model = LinearNetAsEncoder(input_size1, outdim_size, device=device)
            self.view2_model = LinearNetAsEncoder(input_size2, outdim_size, device=device)
   
    def forward(self, x1, x2):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        # feature * batch_size
        output1 = self.view1_model(x1)
        output2 = self.view2_model(x2)

        return output1, output2

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

class LinearNetAsEncoder(nn.Module):
    def __init__(self, input_size, outdim_size, device):
        super(LinearNetAsEncoder, self).__init__()
        self.out = nn.Linear(input_size, outdim_size)
    
    def forward(self, x):
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


