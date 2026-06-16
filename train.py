import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
# import wandb

from dataset import CustomDataset, load_dataset, analyze_dataset
from model import MyBertModel
from utils import set_seed, compute_class_weights, device


def train_epoch(model, train_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    total_samples = 0
    correct_pred = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch['label'].to(device)

        optimizer.zero_grad()
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        total_samples += label.size(0)

        _, predicted_labels = torch.max(output, dim=1)
        correct_pred += (predicted_labels == label).sum().item()

    avg_train_loss = total_loss / total_samples
    train_accuracy = correct_pred / total_samples
    return avg_train_loss, train_accuracy


def eval_epoch(model, val_loader, loss_fn):
    model.eval()
    total_loss = 0
    total_samples = 0
    correct_pred = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(output, label)

            total_loss += loss.item() * input_ids.size(0)
            total_samples += label.size(0)

            _, predicted_labels = torch.max(output, dim=1)
            correct_pred += (predicted_labels == label).sum().item()

            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predicted_labels.cpu().numpy())

    avg_eval_loss = total_loss / total_samples
    eval_accuracy = correct_pred / total_samples
    eval_f1 = f1_score(all_labels, all_predictions, average='weighted')

    return avg_eval_loss, eval_accuracy, eval_f1, all_labels, all_predictions


def main():
    set_seed(61)

    checkpoint_dir = "checkpoint"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    df = load_dataset()
    #analyze_dataset(df)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=61, stratify=df['Sentiment']
    )

    #print(f"\nSplits: Train={len(train_df)}, Test={len(test_df)}")

    numeric_labels = df['Sentiment'].values
    class_weights = compute_class_weights(numeric_labels)

    batch_size = 32
    max_length = 248
    train_dataset = CustomDataset(train_df, max_length=max_length)
    test_dataset = CustomDataset(test_df, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MyBertModel(num_classes=5).to(device)

    learning_rate = 2e-5
    n_epochs = 5

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_f1 = 0
    best_epoch = 0
    # wandb.init(project="bert-sentiment", name="bert-classification")
    # wandb.config.update({
    #     "learning_rate": learning_rate,
    #     "batch_size": batch_size,
    #     "n_epochs": n_epochs,
    #     "optimizer": "Adam",
    #     "model": "bert-base-uncased"
    # })

    for epoch in range(n_epochs):
        avg_train_loss, train_accuracy = train_epoch(
            model, train_loader, optimizer, loss_fn
        )

        avg_eval_loss, eval_accuracy, eval_f1, all_labels, all_predictions = eval_epoch(
            model, test_loader, loss_fn
        )

        print(f"Epoch {epoch + 1}/{n_epochs}: "
              f"train_loss={avg_train_loss:.4f}, train_acc={train_accuracy:.4f} | "
              f"eval_loss={avg_eval_loss:.4f}, eval_acc={eval_accuracy:.4f}, eval_f1={eval_f1:.4f}")

        if eval_f1 > best_f1:
            best_f1 = eval_f1
            best_epoch = epoch + 1

            for old_model in glob.glob(os.path.join(checkpoint_dir, "best_model_*.pt")):
                os.remove(old_model)

            model_path = os.path.join(checkpoint_dir, f"best_model_epoch{best_epoch}_f1{best_f1:.4f}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Meilleur modèle sauvegardé (Epoch {best_epoch}, F1={best_f1:.4f})")

        # wandb.log({
        #     "epoch": epoch + 1,
        #     "train_loss": avg_train_loss,
        #     "train_accuracy": train_accuracy,
        #     "test_loss": avg_eval_loss,
        #     "test_accuracy": eval_accuracy,
        #     "F1-Score": eval_f1
        #     })


    print(f"\nTest Results (Best Model):")
    print(f"  Loss: {avg_eval_loss:.4f}")
    print(f"  Accuracy: {eval_accuracy:.4f}")
    print(f"  F1-Score: {eval_f1:.4f}")

    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("\nMatrice de confusion sauvegardee: confusion_matrix.png")

    # wandb.log({
    #     "test_loss": avg_test_loss,
    #     "test_accuracy": test_accuracy,
    #     "test_f1": test_f1
    # })
    # wandb.finish()


if __name__ == "__main__":
    main()
