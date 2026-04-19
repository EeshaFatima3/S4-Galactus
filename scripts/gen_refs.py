import numpy as np
import struct
import os

ref_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "caal project", "task9_deliverables", "test_validation"
)
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs")
os.makedirs(out_dir, exist_ok=True)

weights_bin = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "caal project", "task9_deliverables", "model_weights.bin"
)

class_names = ["smooth_round", "smooth_cigar", "edge_on_disk", "unbarred_spiral"]


def load_hilbert_indices():
    weights = np.fromfile(weights_bin, dtype=np.float32)

    idx_offset = 0
    hilbert_idx = np.round(weights[idx_offset:idx_offset + 4096]).astype(np.int32)
    return hilbert_idx


def compute_hilbert_scan(image, hilbert_idx):
    out = np.zeros(4096, dtype=np.float32)
    for t in range(4096):
        out[t] = image[hilbert_idx[t]]
    return out


def main():
    print("loading hilbert indices from weights...")

    try:
        hilbert_idx = load_hilbert_indices()
        print(f"  hilbert indices range: [{hilbert_idx.min()}, {hilbert_idx.max()}]")
        has_hilbert = True
    except Exception as e:
        print(f"  [!] could not load hilbert indices: {e}")
        print("  will skip hilbert reference generation")
        has_hilbert = False

    print("\ngenerating reference data for all 10 samples...\n")

    summary = []

    for i in range(10):
        print(f"--- sample {i} ---")

        label_path = os.path.join(ref_dir, f"sample{i}_label.txt")
        label = int(open(label_path).read().strip())
        print(f"  true label: {label} ({class_names[label]})")

        input_path = os.path.join(ref_dir, f"sample{i}_input.bin")
        image = np.fromfile(input_path, dtype=np.float32)
        print(f"  input: {len(image)} floats")

        if has_hilbert:
            hilbert_out = compute_hilbert_scan(image, hilbert_idx)
            hilbert_out.tofile(os.path.join(out_dir, f"sample{i}_hilbert_ref.bin"))
            print(f"  hilbert ref: first4={hilbert_out[:4]}")

        input_proj_path = os.path.join(ref_dir, f"sample{i}_input_proj.bin")
        if os.path.exists(input_proj_path):
            input_proj = np.fromfile(input_proj_path, dtype=np.float32)
            input_proj.tofile(os.path.join(out_dir, f"sample{i}_input_proj_ref.bin"))
            print(f"  input_proj ref: {len(input_proj)} floats, first4={input_proj[:4]}")

        gelu2_path = os.path.join(ref_dir, f"sample{i}_gelu2.bin")
        if os.path.exists(gelu2_path):
            gelu2 = np.fromfile(gelu2_path, dtype=np.float32).reshape(4096, 64)
            post_pool = gelu2[4095, :].copy()
            post_pool.tofile(os.path.join(out_dir, f"sample{i}_post_pool_ref.bin"))
            print(f"  post_pool ref: first4={post_pool[:4]}")

        logits_path = os.path.join(ref_dir, f"sample{i}_logits.bin")
        logits = np.fromfile(logits_path, dtype=np.float32)
        logits.tofile(os.path.join(out_dir, f"sample{i}_logits_ref.bin"))
        print(f"  logits ref: {logits}")

        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        probs.tofile(os.path.join(out_dir, f"sample{i}_probs_ref.bin"))
        pred = int(np.argmax(probs))
        print(f"  probs ref: {probs}")
        print(f"  predicted: {pred} ({class_names[pred]})")

        summary.append({
            "sample": i,
            "label": label,
            "predicted": pred,
            "correct": label == pred,
            "logits": logits.tolist(),
            "probs": probs.tolist(),
        })

    print("\n=== summary ===")
    print(f"{'sample':>8} {'label':>6} {'pred':>6} {'correct':>8} {'class':>20}")
    for s in summary:
        cls = class_names[s["label"]]
        ok = "pass" if s["correct"] else "fail"
        print(f"{s['sample']:>8} {s['label']:>6} {s['predicted']:>6} {ok:>8} {cls:>20}")

    correct = sum(1 for s in summary if s["correct"])
    print(f"\naccuracy: {correct}/{len(summary)} ({100*correct/len(summary):.1f}%)")
    print(f"\nreference files saved to: {out_dir}")


if __name__ == "__main__":
    main()
