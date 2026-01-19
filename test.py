import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from model import Net, MyDataSet

# =============================================================================
# Configuration
# =============================================================================
# True:  Evaluate on real datasets (Requires local .npy files, reproduces paper results)
# False: Run a quick demo with random data (For code verification/sanity check)
TEST_REAL_SCENARIOS = True

# Path to the best checkpoint for evaluation
BEST_MODEL_PATH = "CL_APS_nonlinear_model_stage_7.pt"

# Device configuration (Auto-detect)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(dataloader, net, device):
    """
    Compute accuracy on the given dataloader.
    """
    net.eval()
    n_correct = 0
    n_total = 0

    with torch.no_grad():
        for batch in dataloader:
            # Unpack data (Handle cases where MyDataSet might return extra info)
            if len(batch) >= 2:
                x, y = batch[0], batch[1]
            else:
                continue

            x, y = x.to(device), y.to(device)

            # Forward pass
            outputs = net(x)

            # Label alignment (Map 1-55 to 0-54)
            y = torch.sub(y, 1)

            # Vectorized accuracy calculation (Faster than loops)
            preds = torch.argmax(outputs, dim=1)
            n_correct += (preds == y).sum().item()
            n_total += y.size(0)

    return n_correct / n_total if n_total > 0 else 0.0


def run_inference():
    print(f"Running inference on: {DEVICE}")

    # 1. Initialize Model
    model = Net().to(DEVICE)

    # 2. Load Weights
    weights_path = BEST_MODEL_PATH 
    if os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print(f"Successfully loaded weights: {weights_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
    else:
        print(f"Warning: Weights file '{weights_path}' not found. Using random initialization.")

    print("-" * 50)

    # 3. Execution Mode
    if TEST_REAL_SCENARIOS:
        print("[Mode] Full Evaluation on Real Data")

        # Configuration for all test scenarios
        # format: (Dataset Name, Data Path, Label Path, Num Samples)
        scenarios = [
            ("Split Test", "split-last0.3", "0.3label.npy", 9900),
            ("Scene 2", "scene2.npy", "scene2label.npy", 3300),
            ("Scene 3", "scene3.npy", "scene3label.npy", 3300),
            ("Scene 4", "scene4.npy", "scene4label.npy", 3300),
        ]

        results = []

        for name, d_path, l_path, n_samples in scenarios:
            if os.path.exists(d_path) and os.path.exists(l_path):
                try:
                    # Using memmap for memory efficiency
                    data_mem = np.memmap(d_path, mode='r', shape=(n_samples, 17, 1, 256, 128), dtype=np.float32)
                    labels = np.load(l_path)

                    dataset = MyDataSet(data_mem, labels)
                    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

                    print(f"Evaluating {name}...", end=" ", flush=True)
                    acc = evaluate(loader, model, DEVICE)
                    results.append(acc)
                    print(f"Done. Acc: {acc:.2%}")

                    # Clean up memory
                    del data_mem, dataset, loader

                except Exception as e:
                    print(f"\nFailed to evaluate {name}: {e}")
            else:
                print(f"Skipping {name}: File not found.")

        # Print final summary line (matching your original output style)
        print("-" * 50)
        print("Final Accuracy Summary (%):")
        print(" ".join([f"{round(r * 100, 2)}" for r in results]))

    else:
        print("[Mode] Demo Run (Random Input)")

        # Create dummy input: [Batch=1, Seq=17, Ch=1, H=256, W=128]
        dummy_input = torch.randn(1, 17, 1, 256, 128).to(DEVICE)
        model.eval()

        with torch.no_grad():
            output = model(dummy_input)
            pred_prob = torch.exp(output)
            pred_class = torch.argmax(pred_prob, dim=1).item()

        print(f"Input Shape: {dummy_input.shape}")
        print(f"Output Shape: {output.shape}")
        print(f"Predicted Class Index: {pred_class}")
        print("\nNote: Set 'TEST_REAL_SCENARIOS = True' to reproduce paper results.")

    print("-" * 50)


if __name__ == "__main__":
    run_inference()

