import numpy as np
import struct
import os

ref_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs")
test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
os.makedirs(test_dir, exist_ok=True)


def float_to_asm(val):
    return f"{float(val):.10f}"


def gen_fc_test(sample_idx):
    post_pool_path = os.path.join(ref_dir, f"sample{sample_idx}_post_pool_ref.bin")
    if not os.path.exists(post_pool_path):
        print(f"  [!] missing {post_pool_path}")
        return

    post_pool = np.fromfile(post_pool_path, dtype=np.float32)

    logits_path = os.path.join(ref_dir, f"sample{sample_idx}_logits_ref.bin")
    logits = np.fromfile(logits_path, dtype=np.float32)

    lines = []
    lines.append(".section .data")
    lines.append(".align 2")
    lines.append("")
    lines.append(f"fc_test_input:")

    for i in range(0, len(post_pool), 8):
        chunk = post_pool[i:i+8]
        vals = ", ".join(float_to_asm(v) for v in chunk)
        lines.append(f"    .float {vals}")

    lines.append("")
    lines.append(".section .text")
    lines.append(".globl _start")
    lines.append("_start:")
    lines.append("    la      t0, fc_test_input")
    lines.append("    la      t1, post_pool")
    lines.append("    li      t2, 0")
    lines.append("    li      t3, 64")
    lines.append(".fc_copy_loop:")
    lines.append("    bge     t2, t3, .fc_copy_done")
    lines.append("    flw     ft0, 0(t0)")
    lines.append("    fsw     ft0, 0(t1)")
    lines.append("    addi    t0, t0, 4")
    lines.append("    addi    t1, t1, 4")
    lines.append("    addi    t2, t2, 1")
    lines.append("    j       .fc_copy_loop")
    lines.append(".fc_copy_done:")
    lines.append("")
    lines.append("    la      a0, post_pool")
    lines.append("    la      a1, logits")
    lines.append("    call    fc_forward")
    lines.append("")
    lines.append("    la      t0, logits")
    lines.append("    flw     fa0, 0(t0)")
    lines.append("    flw     fa1, 4(t0)")
    lines.append("    flw     fa2, 8(t0)")
    lines.append("    flw     fa3, 12(t0)")
    lines.append("")
    lines.append("    unimp")
    lines.append("")

    out_path = os.path.join(test_dir, f"test_sample{sample_idx}_fc.s")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  generated {out_path}")


def gen_softmax_test(sample_idx):
    logits_path = os.path.join(ref_dir, f"sample{sample_idx}_logits_ref.bin")
    if not os.path.exists(logits_path):
        print(f"  [!] missing {logits_path}")
        return

    logits = np.fromfile(logits_path, dtype=np.float32)

    probs_path = os.path.join(ref_dir, f"sample{sample_idx}_probs_ref.bin")
    probs = np.fromfile(probs_path, dtype=np.float32)

    lines = []
    lines.append(".section .data")
    lines.append(".align 2")
    lines.append("")
    lines.append(f"sm_test_input:")
    for v in logits:
        lines.append(f"    .float {float_to_asm(v)}")

    lines.append("")
    lines.append(".section .text")
    lines.append(".globl _start")
    lines.append("_start:")
    lines.append("    la      t0, sm_test_input")
    lines.append("    la      t1, logits")
    lines.append("    flw     ft0, 0(t0)")
    lines.append("    flw     ft1, 4(t0)")
    lines.append("    flw     ft2, 8(t0)")
    lines.append("    flw     ft3, 12(t0)")
    lines.append("    fsw     ft0, 0(t1)")
    lines.append("    fsw     ft1, 4(t1)")
    lines.append("    fsw     ft2, 8(t1)")
    lines.append("    fsw     ft3, 12(t1)")
    lines.append("")
    lines.append("    la      a0, logits")
    lines.append("    la      a1, probs")
    lines.append("    li      a2, 4")
    lines.append("    call    softmax_forward")
    lines.append("")
    lines.append("    la      t0, probs")
    lines.append("    flw     fa0, 0(t0)")
    lines.append("    flw     fa1, 4(t0)")
    lines.append("    flw     fa2, 8(t0)")
    lines.append("    flw     fa3, 12(t0)")
    lines.append("")
    lines.append("    unimp")
    lines.append("")

    out_path = os.path.join(test_dir, f"test_sample{sample_idx}_softmax.s")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  generated {out_path}")



def gen_gelu_test(sample_idx):
    src_dir = os.path.join(ref_dir, "..", "..", "..", "caal project", "task9_deliverables", "test_validation")
    s4d1_path = os.path.join(src_dir, f"sample{sample_idx}_s4d1.bin")
    if not os.path.exists(s4d1_path):
        return

    s4d1 = np.fromfile(s4d1_path, dtype=np.float32)

    lines = []
    lines.append(".section .data")
    lines.append(".align 2")
    lines.append("")
    lines.append(f"gelu_test_input:")
    
    # Just first 8 elements is enough to validate GELU logic
    chunk = s4d1[:8]
    vals = ", ".join(float_to_asm(v) for v in chunk)
    lines.append(f"    .float {vals}")

    lines.append("")
    lines.append(".section .text")
    lines.append(".globl _start")
    lines.append("_start:")
    lines.append("    la      t0, buf_a")
    lines.append("    li      t1, 262144")
    lines.append("    li      t2, 0")
    lines.append(".zero_loop:")
    lines.append("    bge     t2, t1, .zero_done")
    lines.append("    sw      zero, 0(t0)")
    lines.append("    addi    t0, t0, 4")
    lines.append("    addi    t2, t2, 1")
    lines.append("    j       .zero_loop")
    lines.append(".zero_done:")
    lines.append("")
    
    lines.append("    la      t0, gelu_test_input")
    lines.append("    la      t1, buf_a")
    lines.append(f"    li      t2, 0")
    lines.append(f"    li      t3, 8")
    lines.append(".copy_loop:")
    lines.append("    bge     t2, t3, .copy_done")
    lines.append("    flw     ft0, 0(t0)")
    lines.append("    fsw     ft0, 0(t1)")
    lines.append("    addi    t0, t0, 4")
    lines.append("    addi    t1, t1, 4")
    lines.append("    addi    t2, t2, 1")
    lines.append("    j       .copy_loop")
    lines.append(".copy_done:")
    lines.append("")
    lines.append("    la      a0, buf_a")
    lines.append("    call    gelu_inplace")
    lines.append("")
    lines.append("    la      t0, buf_a")
    for i in range(8):
        lines.append(f"    flw     fa{i}, {i*4}(t0)")
    lines.append("")
    lines.append("    unimp")
    lines.append("")

    out_path = os.path.join(test_dir, f"test_sample{sample_idx}_gelu.s")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  generated {out_path}")

def gen_takelast_test(sample_idx):
    src_dir = os.path.join(ref_dir, "..", "..", "..", "caal project", "task9_deliverables", "test_validation")
    gelu2_path = os.path.join(src_dir, f"sample{sample_idx}_gelu2.bin")
    if not os.path.exists(gelu2_path):
        return

    gelu2 = np.fromfile(gelu2_path, dtype=np.float32)
    last_row = gelu2[-64:]

    lines = []
    lines.append(".section .data")
    lines.append(".align 2")
    lines.append("")
    lines.append(f"takelast_test_input:")
    for i in range(0, 64, 8):
        chunk = last_row[i:i+8]
        vals = ", ".join(float_to_asm(v) for v in chunk)
        lines.append(f"    .float {vals}")
        
    lines.append("")
    lines.append(".section .text")
    lines.append(".globl _start")
    lines.append("_start:")
    
    # Store at row 4095
    lines.append("    la      t0, takelast_test_input")
    lines.append("    la      t1, buf_a")
    lines.append("    li      t2, 1044480")
    lines.append("    add     t1, t1, t2")
    lines.append("    li      t2, 0")
    lines.append("    li      t3, 64")
    lines.append(".tl_copy_loop:")
    lines.append("    bge     t2, t3, .tl_copy_done")
    lines.append("    flw     ft0, 0(t0)")
    lines.append("    fsw     ft0, 0(t1)")
    lines.append("    addi    t0, t0, 4")
    lines.append("    addi    t1, t1, 4")
    lines.append("    addi    t2, t2, 1")
    lines.append("    j       .tl_copy_loop")
    lines.append(".tl_copy_done:")
    lines.append("")
    lines.append("    la      a0, buf_a")
    lines.append("    la      a1, post_pool")
    lines.append("    call    take_last")
    lines.append("")
    lines.append("    la      t0, post_pool")
    for i in range(8):
        lines.append(f"    flw     fa{i}, {i*4}(t0)")
    lines.append("")
    lines.append("    unimp")
    lines.append("")

    out_path = os.path.join(test_dir, f"test_sample{sample_idx}_takelast.s")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  generated {out_path}")

def main():
    print("generating per-sample test assembly files...\n")

    for i in range(10):
        print(f"sample {i}:")
        gen_fc_test(i)
        gen_softmax_test(i)
        gen_gelu_test(i)
        gen_takelast_test(i)
        print()

    print("done. test files are in:", test_dir)


if __name__ == "__main__":
    main()
