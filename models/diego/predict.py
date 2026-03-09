from model import ClassificationModel
from dataloader import DataLoaderCsv
import lightning as L
from torch.utils.data import TensorDataset, DataLoader
import torch


def main():
    path = "./datasets/test/second_half.csv"
    # path_test="../../datasets/titanic/test.csv"
    # path_submission="../../datasets/titanic/gender_submission.csv"
    ckpt_path = "./checkpoints/best-model-epoch=30-val_loss=0.4165.ckpt"

    dataset = DataLoaderCsv(path)
    X, y = dataset.ft_get_data_from_file()

    # dataset = DataLoaderCsv(path_test)
    # X, y = dataset.ft_get_data_from_file_titanic(path_submission)

    predict_dataset = TensorDataset(X, y)
    predict_loader = DataLoader(predict_dataset, batch_size=32, shuffle=False)

    model = ClassificationModel.load_from_checkpoint(ckpt_path)

    trainer = L.Trainer()

    results = trainer.test(model, dataloaders=predict_loader)

    print(results)


if __name__ == "__main__":
    main()