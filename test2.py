import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from model import ApsPool,ResidualBlock
from class_databatch import MyDataSet


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 7, (1, 2), padding=3),
            ApsPool(),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, (1, 2), padding=1),
            ApsPool(),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            ResidualBlock(32, 64, (1, 2), 16),
            ResidualBlock(64, 64, 1, 16),
            ResidualBlock(64, 128, (1, 2)),
            ResidualBlock(128, 128, 1),
            ResidualBlock(128, 256, (1, 2)),
            ResidualBlock(256, 256, 1),
            torch.nn.AdaptiveAvgPool2d(1)
        )
        self.model_lstm = torch.nn.LSTM(input_size=256, hidden_size=512, num_layers=2, batch_first=True)

        self.model_mlp = torch.nn.Sequential()
        self.model_mlp.add_module("linear1", torch.nn.Linear(512, 4 * 512))
        self.model_mlp.add_module("relu1", torch.nn.ReLU())
        self.model_mlp.add_module("linear2", torch.nn.Linear(4 * 512, 55))

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        img_h = x.size(3)
        img_w = x.size(4)

        x = x.view(-1, 1, img_h, img_w)
        x = self.model_cnn(x)
        x = x.view(batch_size, seq_len, -1)

        output, (h, c) = self.model_lstm(x)
        logits = self.model_mlp(h[1])  # hidden state

        #  Softplus  (Evidence >= 0)
        evidence = F.softplus(logits)

        return torch.clamp(evidence, min=1e-6, max=10000)


# ==========================================
# 2. evaluate
# ==========================================
def evaluate(model, dataloader, device, dataset_name="Test Set", uncertainty_threshold=0.5):

    model.eval()
    model.to(device)
    uncertainties_correct = []
    uncertainties_wrong = []


    filtered_correct = 0
    filtered_total = 0
    total_count = 0

    print(f"Evaluating on {dataset_name}...")
    print(f"Uncertainty Threshold: {uncertainty_threshold}")

    with torch.no_grad():
        for x, y, _ in tqdm(dataloader, leave=False):
            x, y = x.to(device), y.to(device)
            y = torch.sub(y, 1).long()

            evidence = model(x)
            alpha = evidence + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            uncertainty = 55.0 / S
            probs = alpha / S
            predictions = torch.argmax(probs, dim=1)

            for i in range(len(y)):
                u_val = uncertainty[i].item()
                is_correct = (predictions[i] == y[i])
                total_count += 1

                if is_correct:
                    uncertainties_correct.append(u_val)
                else:
                    uncertainties_wrong.append(u_val)

                if u_val < uncertainty_threshold:
                    filtered_total += 1
                    if is_correct:
                        filtered_correct += 1


    u_corr = np.array(uncertainties_correct)
    u_wrong = np.array(uncertainties_wrong)

    if len(u_wrong) > 0:
        all_uncertainties = np.concatenate((u_corr, u_wrong))
    else:
        all_uncertainties = u_corr

    avg_u_total = np.mean(all_uncertainties) if len(all_uncertainties) > 0 else 0.0
    raw_acc = len(u_corr) / total_count if total_count > 0 else 0
    coverage = filtered_total / total_count if total_count > 0 else 0
    filtered_acc = filtered_correct / filtered_total if filtered_total > 0 else 0


    print(f"\n{'=' * 50}")
    print(f"Results for [{dataset_name}]")
    print(f"{'=' * 50}")
    print(f"  Avg Uncertainty   : {avg_u_total:.4f}")
    print(f"\n[Raw Statistics (All Samples)]")
    print(f"  Total Samples     : {total_count}")
    print(f"  Raw Accuracy      : {raw_acc * 100:.2f}%")
    print(f"  Avg U (Correct)   : {np.mean(u_corr):.4f}")
    if len(u_wrong) > 0:
        print(f"  Avg U (Wrong)     : {np.mean(u_wrong):.4f}")
    else:
        print(f"  Avg U (Wrong)     : N/A")

    print(f"\n[Filtered Statistics (u < {uncertainty_threshold})]")
    print(f"  Filtered Samples  : {filtered_total} / {total_count}")
    print(f"  Coverage          : {coverage * 100:.2f}%")
    print(f"  Filtered Accuracy : {filtered_acc * 100:.2f}%")
    print(f"{'=' * 50}\n")

    return filtered_acc


# ==========================================
# 3. main
# ==========================================
def main():
    seed = 114
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading Test Data...")

    # 定义 Batch Size
    BATCH_SIZE = 128

    try:
        # Scene 1 (Test Split)
        test = np.memmap("split-last0.3", mode='r', shape=(9900, 17, 1, 256, 128), dtype=np.float32)
        label_test = np.load('0.3label.npy')
        test_dataset = MyDataSet(test, label_test)
        test_loader = DataLoader(test_dataset, BATCH_SIZE)

        # Scene 2
        test2 = np.memmap("scene2.npy", mode='r', shape=(3300, 17, 1, 256, 128), dtype=np.float32)
        label_test2 = np.load('scene2label.npy')
        test2_dataset = MyDataSet(test2, label_test2)
        test2_loader = DataLoader(test2_dataset, BATCH_SIZE)

        # Scene 3
        test3 = np.memmap("scene3.npy", mode='r', shape=(3300, 17, 1, 256, 128), dtype=np.float32)
        label_test3 = np.load('scene3label.npy')
        test3_dataset = MyDataSet(test3, label_test3)
        test3_loader = DataLoader(test3_dataset, BATCH_SIZE)

        # Scene 4
        test4 = np.memmap("scene4.npy", mode='r', shape=(3300, 17, 1, 256, 128), dtype=np.float32)
        label_test4 = np.load('scene4label.npy')
        test4_dataset = MyDataSet(test4, label_test4)
        test4_loader = DataLoader(test4_dataset, BATCH_SIZE)

    except FileNotFoundError as e:
        print(f"\n[Error] Data files not found: {e}")
        print("Please ensure .npy and .memmap files are in the same directory.")
        return

    # ==========================
    # load
    # ==========================
    net = Net()
    model_path = "uncertain_edl_model.pt"

    print(f"Loading model weights from {model_path}...")
    try:
        net.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"\n[Error] Model file '{model_path}' not found.")
        print("Please upload the pre-trained model file.")
        return

    # ==========================
    # test
    # ==========================

    THRESHOLD = 0.5

    evaluate(net, test_loader, device, dataset_name="Test Set (Scene 1)", uncertainty_threshold=THRESHOLD)
    evaluate(net, test2_loader, device, dataset_name="Unseen Scene 2", uncertainty_threshold=THRESHOLD)
    evaluate(net, test3_loader, device, dataset_name="Unseen Scene 3", uncertainty_threshold=THRESHOLD)
    evaluate(net, test4_loader, device, dataset_name="Unseen Scene 4", uncertainty_threshold=THRESHOLD)

    print("Inference finished.")


if __name__ == "__main__":
    main()
