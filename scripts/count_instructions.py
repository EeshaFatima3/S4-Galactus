import sys
import os

def classify(mnem):
    mnem = mnem.lower()

    f_type = ['fadd.s','fsub.s','fmul.s','fdiv.s','fmadd.s','fmsub.s',
              'fsqrt.s','flt.s','fle.s','feq.s','fcvt.w.s','fcvt.s.w',
              'fmv.s','fneg.s','fmv.x.w','fmv.w.x','flw','fsw','fabs.s',
              'fclass.s','fmin.s','fmax.s']
    s_type = ['sw','sh','sb']
    b_type = ['beq','bne','blt','bge','bltu','bgeu','beqz','bnez',
              'bgtz','bltz','bgez','blez']          # branches incl. pseudos
    j_type = ['jal','j','call','ret','jr']
    u_type = ['lui','auipc']
    # li / la / mv are pseudo-instructions: li→I, la→I, mv→R (reg-reg move)
    i_type = ['addi','lw','lh','lb','jalr','srai','slli','srli','la',
              'andi','ori','xori','slti','sltiu','lwu','li']
    r_type = ['add','sub','and','or','xor','sll','srl','sra','mul',
              'div','rem','slt','sltu','mulh','mulhu','divu','remu','mv']
    v_type_prefixes = ('v',)   # all v* mnemonics

    if mnem in f_type:        return 'F'
    if mnem.startswith('v'):  return 'V'
    if mnem in b_type:        return 'B'
    if mnem in j_type:        return 'J'
    if mnem in u_type:        return 'U'
    if mnem in s_type:        return 'S'
    if mnem in i_type:        return 'I'
    if mnem in r_type:        return 'R'
    return 'Other'

def count_file(filepath):
    counts = {'R':0,'I':0,'S':0,'B':0,'U':0,'J':0,'F':0,'V':0,'Other':0}
    unclassified = []
    total = 0

    with open(filepath, 'r') as f:
        for lineno, line in enumerate(f, 1):
            code = line.split('#')[0].strip()
            if not code:
                continue
            if code.startswith('.'):
                continue
            if ':' in code:
                code = code.split(':', 1)[1].strip()
            if not code or code.startswith('.'):
                continue
            tokens = code.split()
            if not tokens:
                continue
            mnem = tokens[0]
            family = classify(mnem)
            if family == 'Other':
                unclassified.append((lineno, mnem, line.strip()))
            counts[family] += 1
            total += 1

    return total, counts, unclassified

# ── formatting helpers ──────────────────────────────────────────────────────

FAM_ORDER = ['R','I','S','B','U','J','F','V','Other']
FAM_LABEL = {'R':'R-type','I':'I-type','S':'S-type','B':'B-type',
             'U':'U-type','J':'J-type','F':'F-type','V':'V-type','Other':'Other'}

def fmt_cell(v, width):
    return str(v).rjust(width)

def print_module_table(filenames, results):
    # results: list of (basename, total, counts)
    col_w = 7       # width of each count column
    mod_w = max(len(r[0]) for r in results) + 2

    # header
    header_families = FAM_ORDER
    header = '| ' + 'Module'.ljust(mod_w-2) + ' | ' + 'Total'.rjust(col_w)
    for fam in header_families:
        header += ' | ' + FAM_LABEL[fam].rjust(col_w)
    header += ' |'

    sep = '|' + '-'*(mod_w) + '|' + ('-'*(col_w+2) + '|') * (1 + len(header_families))

    print()
    print('### Static Instruction Counts (RVV Vector Modules)')
    print()
    print(header)
    print(sep)

    for (name, total, counts) in results:
        row = '| ' + name.ljust(mod_w-2) + ' | ' + fmt_cell(total, col_w)
        for fam in header_families:
            row += ' | ' + fmt_cell(counts[fam], col_w)
        row += ' |'
        print(row)

    print(sep)

def print_aggregate_family_table(all_counts, grand_total):
    print()
    print('### Static Family Breakdown (Aggregate)')
    print()
    col1, col2, col3 = 10, 8, 14
    hdr = '| ' + 'Family'.ljust(col1) + ' | ' + 'Count'.rjust(col2) + ' | ' + 'Percentage'.rjust(col3) + ' |'
    sep = '|' + '-'*(col1+2) + '|' + '-'*(col2+2) + '|' + '-'*(col3+2) + '|'
    print(hdr)
    print(sep)
    for fam in FAM_ORDER:
        c = all_counts[fam]
        pct = f'{c/grand_total*100:.1f}%' if grand_total > 0 else '—'
        row = ('| ' + FAM_LABEL[fam].ljust(col1) +
               ' | ' + str(c).rjust(col2) +
               ' | ' + pct.rjust(col3) + ' |')
        print(row)
    print(sep)
    tot_row = ('| ' + 'TOTAL'.ljust(col1) +
               ' | ' + str(grand_total).rjust(col2) +
               ' | ' + '100.0%'.rjust(col3) + ' |')
    print(tot_row)
    print()

# ── main ────────────────────────────────────────────────────────────────────

files = sys.argv[1:] if len(sys.argv) > 1 else [f for f in os.listdir('.') if f.endswith('.s')]

if not files:
    print('No .s files found.')
    sys.exit(1)

all_totals = {k:0 for k in FAM_ORDER}
grand_total = 0
results = []

print()
print('=' * 66)
print('  TASK 3.1 – STATIC INSTRUCTION COUNT')
print('=' * 66)

unclassified_all = []

for filepath in files:
    total, counts, unclassified = count_file(filepath)
    name = os.path.basename(filepath)
    results.append((name, total, counts))
    for k in all_totals:
        all_totals[k] += counts[k]
    grand_total += total
    if unclassified:
        unclassified_all.append((name, unclassified))

# Print module table (with AGGREGATE as last row)
agg_counts_row = dict(all_totals)
results_with_agg = results + [('AGGREGATE', grand_total, agg_counts_row)]
print_module_table(files, results_with_agg)

# Print aggregate family breakdown
print_aggregate_family_table(all_totals, grand_total)

# Print unclassified (for debugging)
if unclassified_all:
    print('### Unclassified Instructions')
    for name, items in unclassified_all:
        print(f'\n  {name}:')
        for lineno, mnem, raw in items:
            print(f'    Line {lineno:4d}: {mnem!r:20s} — {raw}')
