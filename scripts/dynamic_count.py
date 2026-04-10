import os
import re

pipeline_config = {
    "hilbert scan": {"file": "test_hilbert.txt", "calls": 1, "scale": 1},
    "input proj": {"file": "test_input_proj.txt", "calls": 1, "scale": 1},
    "gelu": {"file": "test_gelu.txt", "calls": 2, "scale": 1},
    "take last": {"file": "test_takelast.txt", "calls": 1, "scale": 1},
    "fully conn": {"file": "test_fc.txt", "calls": 1, "scale": 1},
    "softmax": {"file": "test_softmax.txt", "calls": 1, "scale": 1},
}

phase_b_start = int("800002e2", 16)
phase_b_end = int("80000378", 16)

def categorize_instruction(op):
    op = op[2:] if op.startswith('c.') else op
    
    # f-type (floating point)
    if op.startswith('f'): return "f-type"
    # s-type (store)
    if op in ['sw', 'sh', 'sb', 'swsp']: return "s-type"
    # b-type (branch)
    if op in ['beq', 'bne', 'blt', 'bge', 'bltu', 'bgeu', 'beqz', 'bnez']: return "b-type"
    # u-type (upper immediate)
    if op in ['lui', 'auipc']: return "u-type"
    # j-type (jump)
    if op in ['jal', 'j']: return "j-type"
    # i-type (immediate / load / jalr)
    if op in ['addi', 'lw', 'lh', 'lb', 'lbu', 'lhu', 'jalr', 'jr', 'srai', 'slli', 'srli', 'xori', 'ori', 'andi', 'slti', 'sltiu', 'lwsp', 'li', 'mv', 'addi16sp', 'addi4spn']: return "i-type"
    # r-type (register)
    if op in ['add', 'sub', 'and', 'or', 'xor', 'sll', 'srl', 'sra', 'mul', 'mulh', 'div', 'rem', 'slt', 'sltu']: return "r-type"
    
    if op == "illegal": return "ignore"
    return "i-type"

def parse_logs():
    total_dyn = 0
    fams = ["r-type", "i-type", "s-type", "b-type", "u-type", "j-type", "f-type"]
    total_fams = {f: 0 for f in fams}
    results = {}

    for layer, cfg in pipeline_config.items():
        counts = {f: 0 for f in fams}
        raw = 0
        try:
            with open(cfg["file"], 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 7 and parts[0].startswith('#'):
                        for p in parts[5:]:
                            if re.match(r"^[a-z.]+$", p):
                                cat = categorize_instruction(p)
                                if cat != "ignore":
                                    counts[cat] += 1
                                    raw += 1
                                break
            if raw > 0:
                scale = cfg["scale"] * cfg["calls"]
                results[layer] = {"total": raw * scale}
                for fam in fams:
                    val = counts[fam] * scale
                    results[layer][fam] = val
                    total_fams[fam] += val
                total_dyn += results[layer]["total"]
        except:
            pass

    s4d_a = {f: 0 for f in fams}
    s4d_a["raw"] = 0
    s4d_b = {f: 0 for f in fams}
    s4d_b["raw"] = 0

    try:
        with open("test_s4d_mini.txt", 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 7 and parts[0].startswith('#'):
                    pc = int(parts[2], 16)
                    for p in parts[5:]:
                        if re.match(r"^[a-z.]+$", p):
                            cat = categorize_instruction(p)
                            if cat != "ignore":
                                if phase_b_start <= pc <= phase_b_end:
                                    s4d_b[cat] += 1
                                    s4d_b["raw"] += 1
                                else:
                                    s4d_a[cat] += 1
                                    s4d_a["raw"] += 1
                            break
        
        scale_a = 32 * 2
        scale_b = 1024 * 2
        
        results["s4d phase a"] = {"total": s4d_a["raw"] * scale_a}
        results["s4d phase b"] = {"total": s4d_b["raw"] * scale_b}
        
        for fam in fams:
            va = s4d_a[fam] * scale_a
            vb = s4d_b[fam] * scale_b
            results["s4d phase a"][fam] = va
            results["s4d phase b"][fam] = vb
            total_fams[fam] += (va + vb)
        
        total_dyn += results["s4d phase a"]["total"] + results["s4d phase b"]["total"]
    except:
        pass

    print("dynamic instruction count report")
    print("--------------------------------")
    for layer, data in results.items():
        if layer == "s4d phase a":
            print("note: the s4d layer trace was split using memory addresses. phase a (kernel gen) scales linearly so it was scaled by 32x. phase b (convolution) scales quadratically so it was scaled by 1024x. both are multiplied by 2 for the pipeline calls.")
            print("")
            
        print(layer + ":")
        print(f"total: {data['total']}")
        for fam in fams:
            if data[fam] > 0:
                pct = (data[fam] / data['total']) * 100
                print(f"{fam}: {data[fam]} ({pct:.2f}%)")
        print("")

    print("total inference pass:")
    print(f"total instructions: {total_dyn}")
    for fam in fams:
        if total_fams[fam] > 0:
            pct = (total_fams[fam] / total_dyn) * 100
            print(f"{fam}: {total_fams[fam]} ({pct:.2f}%)")

if __name__ == "__main__":
    parse_logs()
