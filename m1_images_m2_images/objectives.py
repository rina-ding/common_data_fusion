import torch 
import torch.nn as nn

class cross_entropy_loss(nn.Module):
    def __init__(self):
        super(cross_entropy_loss, self).__init__()
        self.cross_entropy = nn.BCEWithLogitsLoss()

    def loss(self, x1, x2):
        loss = self.cross_entropy(x1, x2)
        return loss
    
class cosine_similarity_loss(nn.Module):
    def __init__(self, dim=1, reduction='mean'):
        super(cosine_similarity_loss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=dim)
        self.reduction = reduction

    def loss(self, x1, x2):
        # Compute cosine similarity between pairs
        cos_sim = self.cosine_similarity(x1, x2)
        
        # To use it as a loss, take 1 - cosine similarity (maximize similarity)
        loss = 1 - cos_sim
        
        # Reduce loss (mean or sum over batch) if specified
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss

class cca_loss():
    # Copied from https://github.com/Michaelvll/DeepCCA/blob/master/objectives.py
    def __init__(self, outdim_size, use_all_singular_values, device):
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.device = device

    def loss(self, H1, H2):
        """

        It is the loss function of CCA as introduced in the original paper. There can be other formulations.

        """

        # Add some valyes to the sigma matrices due to small values and gradient instability 
        r1 = 1e-6
        r2 = 1e-6
        eps = 1e-9

        # print(f"H1 before transpose has size {H1.size()}")

        H1, H2 = H1.t(), H2.t()
        o1 = o2 = H1.size(0)

        m = H1.size(1)
        # print(f"H1 after transpose has size {H1.size()}")
        # print(f"m: {m}")

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)
        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t()) # m-1 is used to correct bias when calculating sample variance 
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2, device=self.device)
        # assert torch.isnan(SigmaHat11).sum().item() == 0 # assertion failed 
        # assert torch.isnan(SigmaHat12).sum().item() == 0
        # assert torch.isnan(SigmaHat22).sum().item() == 0


        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = torch.linalg.eigh(SigmaHat11)
        [D2, V2] = torch.linalg.eigh(SigmaHat22)
        # [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        # [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)
        # [D1, V1] = torch.symeig(SigmaHat11 + r1*torch.eye(SigmaHat11.shape[0]).to(self.device), eigenvectors=True)
        # [D2, V2] = torch.symeig(SigmaHat22 + r2*torch.eye(SigmaHat22.shape[0]).to(self.device), eigenvectors=True)

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)

        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.matmul(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp))
        else:
            # just the top self.outdim_size singular values are used
            trace_TT = torch.matmul(Tval.t(), Tval)
            trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0])*r1).to(self.device)) # regularization for more stability
            # U, V = torch.linalg.eigh(trace_TT)
            #U, V = torch.symeig(trace_TT, eigenvectors=True)
            U = torch.linalg.eigvalsh(trace_TT, UPLO='U')
            U = torch.where(U>eps, U, (torch.ones(U.shape).double()*eps).to(self.device))
            U = U.topk(self.outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))
        return -corr