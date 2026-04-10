import os

families = {
    'r-type': ['add', 'sub', 'and', 'or', 'xor', 'sll', 'srl', 'sra', 'mul', 'mulh', 'div', 'rem', 'slt', 'sltu'],
    'i-type': ['addi', 'lw', 'lh', 'lb', 'lbu', 'lhu', 'jalr', 'jr', 'srai', 'slli', 'srli', 'xori', 'ori', 'andi', 'slti', 'sltiu', 'li', 'mv', 'la', 'ret', 'nop', 'ecall', 'ebreak', 'unimp'],
    's-type': ['sw', 'sh', 'sb'],
    'b-type': ['beq', 'bne', 'blt', 'bge', 'bltu', 'bgeu', 'bgt', 'ble', 'bgez', 'blez', 'bltz', 'bgtz', 'beqz', 'bnez'],
    'u-type': ['lui', 'auipc'],
    'j-type': ['jal', 'j', 'call', 'tail'],
    'f-type': ['fadd.s', 'fsub.s', 'fmul.s', 'fdiv.s', 'fmadd.s', 'fmsub.s', 'fnmadd.s', 'fnmsub.s', 'flw', 'fsw', 'fsqrt.s', 'fcvt.w.s', 'fcvt.s.w', 'fcvt.wu.s', 'fcvt.s.wu', 'fmv.x.w', 'fmv.w.x', 'fmv.s', 'flt.s', 'fle.s', 'feq.s', 'fneg.s', 'fabs.s', 'fsgnj.s', 'fsgnjn.s', 'fsgnjx.s', 'fmax.s', 'fmin.s', 'fclass.s']
}

def categorize(inst):
    inst = inst.lower()
    for fam, lst in families.items():
        if inst in lst:
            return fam
    return 'unknown'

def count_static(filepath):
    counts = {f: 0 for f in families.keys()}
    counts['unknown'] = 0
    total = 0

    with open(filepath, 'r') as file:
        for line in file:
            line = line.split('#')[0].strip()
            if not line:
                continue
            
            if ':' in line:
                line = line.split(':', 1)[1].strip()
                
            if not line or line.startswith('.'):
                continue
                
            inst = line.split()[0]
            fam = categorize(inst)
            counts[fam] += 1
            total += 1
            
    return total, counts

files = ['nn.s', 'math.s', 'main.s']

print("static instruction count report\n")

grand_total = 0
grand_counts = {f: 0 for f in families.keys()}
grand_counts['unknown'] = 0

for f in files:
    if os.path.exists(f):
        total, counts = count_static(f)
        grand_total += total
        print(f"module: {f} (total: {total})")
        for fam, count in counts.items():
            grand_counts[fam] += count
            if count > 0:
                print(f"{fam}: {count} ({(count/total)*100:.1f}%)")
        if counts['unknown'] > 0:
            print(f"unknown: {counts['unknown']} ({(counts['unknown']/total)*100:.1f}%)")
        print("")

print("aggregate total:")
print(f"total instructions: {grand_total}")
for fam, count in grand_counts.items():
    if count > 0:
        print(f"{fam}: {count} ({(count/grand_total)*100:.1f}%)")
if grand_counts['unknown'] > 0:
    print(f"unknown: {grand_counts['unknown']} ({(grand_counts['unknown']/grand_total)*100:.1f}%)")
print("")
