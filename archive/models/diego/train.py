from model import ClassificationModel
from dataloader import DataLoaderCsv
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch.utils.data import TensorDataset, DataLoader
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.tracking import MlflowClient


def main():
    path = "../../datasets/titanic/train.csv"

    dataset = DataLoaderCsv(path)

    X_train, X_val, y_train, y_val = dataset.ft_get_data_base_torch()

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="best-model-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    callbacks = [
        checkpoint_callback,
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    mlf_logger = MLFlowLogger(
        experiment_name="titanic-classification",
        tracking_uri="http://127.0.0.1:5000",
        log_model=True,
    )

    n_features = X_train.shape[1]
    model = ClassificationModel(n_features, 2)

    trainer = L.Trainer(
        callbacks=callbacks,
        max_epochs=100,
        logger=mlf_logger,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("MLflow run_id:", mlf_logger.run_id)
    print("Best checkpoint:", checkpoint_callback.best_model_path)
    print("Last checkpoint:", checkpoint_callback.last_model_path)

    client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

    client.log_artifact(mlf_logger.run_id, "model.py")
    client.log_artifact(mlf_logger.run_id, "dataloader.py")
    client.log_text(mlf_logger.run_id, str(model), "model_architecture.txt")


if __name__ == "__main__":
    main()