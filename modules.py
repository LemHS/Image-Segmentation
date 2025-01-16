from UNET import *
import matplotlib.pyplot as plt
import lightning as L

class PatchSegmentationModule(L.LightningModule):
    def __init__(self, n_channels: int, n_classes: int, hidden_dim: int = 64, rgb2class: dict = None, learning_rate: float = 0.0005):
        super(PatchSegmentationModule, self).__init__()
        self.save_hyperparameters()

        if self.hparams.rgb2class:
            self.class2rgb = {class_idx: list(rgb) for rgb, class_idx in rgb2class.items()}

        self.unet = ResUNet(n_channels, n_classes, hidden_dim)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, img):
        logits = self.unet(img)
        return logits
    
    def training_step(self, batch, batch_idx):
        img, mask = batch
        logits = self.forward(img)

        loss = self.criterion(logits, mask)
        self.log("train_loss", loss, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        img, mask = batch
        logits = self.forward(img)

        if (batch_idx == 0) & (self.hparams.rgb2class is not None):
            pred = logits.argmax(dim=1).unsqueeze(-1).cpu()
            pred = torch.concat([pred, pred, pred], dim=-1)
            pred = self._change2rgb(pred)

            mask_vis = mask.unsqueeze(-1).cpu()
            mask_vis = torch.concat([mask_vis, mask_vis, mask_vis], dim=-1)
            mask_vis = self._change2rgb(mask_vis)

            img_vis = img.permute(0, 2, 3, 1).cpu()
            
            fig, axes = plt.subplots(nrows=3, 
                                    ncols=pred.shape[0], 
                                    figsize=(pred.shape[0] * 1.75, 6),
                                    sharex=True,
                                    sharey=True)
            
            row_titles = ["Predictions", "Ground Truth", "Input Image"]
        
            for row_idx, row_title in enumerate(row_titles):
                axes[row_idx, 0].set_ylabel(row_title, fontsize=18, labelpad=10)
                axes[row_idx, 0].yaxis.set_label_coords(-0.3, 0.5)
            
            for i, patch in enumerate(pred):
                axes[0][i].imshow(patch)
                axes[0][i].set_xticks([])
                axes[0][i].set_yticks([])

            for i, patch in enumerate(mask_vis):
                axes[1][i].imshow(patch)
                axes[1][i].set_xticks([])
                axes[1][i].set_yticks([])

            for i, patch in enumerate(img_vis):
                axes[2][i].imshow(patch)
                axes[2][i].set_xlabel(i+1, fontsize=16)
                axes[2][i].set_xticks([])
                axes[2][i].set_yticks([])

            plt.tight_layout()
            plt.show()

        loss = self.criterion(logits, mask)
        self.log("val_loss", loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        img, mask = batch
        logits = self.forward(img)

        loss = self.criterion(logits, mask)
        self.log("test_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=2, mode="min"
                ),
                "monitor": "val_loss",
            }
        }
    
    def _change2rgb(self, input):
        for class_idx, rgb in self.class2rgb.items():
            change = (input[:, :, :, 0] == class_idx) & (input[:, :, :, 1] == class_idx) & (input[:, :, :, 2] == class_idx)
            input[change] = torch.tensor(rgb, dtype=torch.long)

        return input