import torch
from SEmodels.gcnn import GCNN
from data.imdb_dataset import get_dataloader
from SEutils.trainer import train


def main():
    batch_size = 64
    num_epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, vocab = get_dataloader(batch_size, "train")
    test_loader, _ = get_dataloader(batch_size, "test")

    model = GCNN(vocab_size=len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train(
        model,
        train_loader,
        test_loader,
        optimizer,
        num_epochs,
        device,
    )


if __name__ == "__main__":
    main()
