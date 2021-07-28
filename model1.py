from torch import nn
import math
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl

class RestorationAutoEncoder(nn.Module):
    def __init__(self, num_layers=10, num_features=64):
        super(RestorationAutoEncoder, self).__init__()
        self.num_layers = num_layers

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(3, num_features, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(inplace=True)))
        for i in range(num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))

        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.ConvTranspose2d(num_features, 3, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)

        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + conv_feat
                x = self.relu(x)

        x += residual
        x = self.relu(x)

        return x

class RestorationModel1(pl.LightningModule):
    def __init__(self, lr: float, num_layers: int = 10, num_features:int = 64):
        super(RestorationModel, self).__init__()
        self.save_hyperparameters()
        self.model = RestorationAutoEncoder(num_layers=self.hparams.num_layers, num_features=self.hparams.num_features)
        
    def forward(self, xb):
        "forward method for nn.Module"
        return self.model(xb)
    
    # def training_step(self, batch, batch_idx):
    #     "Computer on single training step"
    #     # unpack batch
    #     cln_ims, nsy_ims = batch 
        
    #     # restore images given noisy images to the model as Input
    #     outputs = self.forward(nsy_ims)
        
    #     # compare the restored images with the clean Images
    #     loss = F.mse_loss(input=outputs, target=cln_ims)
        
    #     self.log_dict(dict(train_loss=loss), prog_bar=True, on_epoch=True, on_step=False)
    #     return loss
    
    # def configure_optimizers(self):
    #     "steps up the optimizer and scheduler for the experiment"
    #     opt = optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=1e-05)
    #     sch = optim.lr_scheduler.MultiStepLR(opt, milestones=[10, 15], gamma=0.1)
    #     return [opt],[sch]