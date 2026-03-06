from model import ClassificationModel
from dataloader import DataLoaderCsv
import lightning as L
from torch.utils.data import TensorDataset, DataLoader
import torch


def main():
    path = "./datasets/test/second_half.csv"
    ckpt_path = "./checkpoints/best-model-epoch=27-val_loss=0.4256.ckpt"

    dataset = DataLoaderCsv(path)

    X, y = dataset.ft_get_data_from_file()

    predict_dataset = TensorDataset(X, y)
    predict_loader = DataLoader(predict_dataset, batch_size=32, shuffle=False)

    model = ClassificationModel.load_from_checkpoint(ckpt_path)

    trainer = L.Trainer()

    results = trainer.test(model, dataloaders=predict_loader)

    print(results)


if __name__ == "__main__":
    main()