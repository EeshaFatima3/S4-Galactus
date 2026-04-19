#!/bin/bash

# ================= Configuration =================
GCC_PREFIX="riscv32-unknown-elf"
ABI="-march=rv32gcv -mabi=ilp32f"
LINK="veer/link.ld"
WHISPER_CFG="veer/whisper.json"
BUILD_DIR="build"
OUT_DIRS=("exe" "hex" "dis" "logs" "asm" "obj")
DEFAULT_OPT=""
# =================================================

show_help() {
    echo "Usage: $0 [options] <file> [<file> ...]"
    echo
    echo "Options:"
    echo "  -a             Compile and execute assembly (.s) files"
    echo "  -c             Clean generated files"
    echo "  -e             Execute the last compiled binary"
    echo "  -g [opt_flag]  Compile C (.c) to assembly/object/hex with optional -O2/-O3"
    echo "  -h             Show this help message"
    echo "  -l <file>      Link additional assembly files"
    echo
    echo "Examples:"
    echo "  $0 -a main.s -l conv2d.s"
    echo "  $0 -g -O3 main.c"
}

make_dirs() {
    for dir in "${OUT_DIRS[@]}"; do
        mkdir -p "${BUILD_DIR}/${dir}"
    done
}

get_basename() {
    filename="$1"
    echo "$(basename "$filename" .s | sed 's/\.c$//')"
}

compile_asm() {
    input_files=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -l)
                shift
                input_files+=("$1")
                ;;
            *)
                input_files+=("$1")
                ;;
        esac
        shift
    done

    if [[ ${#input_files[@]} -eq 0 ]]; then
        echo "Error: No files provided to compile."
        exit 1
    fi

    base=$(get_basename "${input_files[0]}")
    exe="${BUILD_DIR}/exe/${base}.exe"
    hex="${BUILD_DIR}/hex/${base}.hex"
    dis="${BUILD_DIR}/dis/${base}.dis"
    data="${BUILD_DIR}/dis/${base}.data"
    linked_asm="${BUILD_DIR}/asm/${base}.s"

    echo "[*] Compiling assembly: ${input_files[*]} ..."
    $GCC_PREFIX-gcc $ABI -lgcc -T"$LINK" -o "$exe" "${input_files[@]}" -nostartfiles -lm

    $GCC_PREFIX-objcopy -O verilog "$exe" "$hex"
    $GCC_PREFIX-objdump -S "$exe" > "$dis"
    $GCC_PREFIX-objdump -s -j .data "$exe" > "$data"
    $GCC_PREFIX-objdump -d -M no-aliases "$exe" > "$linked_asm"

    echo "[+] Output: $exe, $hex, $dis, $data, $linked_asm"
}

compile_c() {
    opt_flag="$1"
    shift
    input_file="$1"

    if [[ ! -f "$input_file" ]]; then
        echo "Error: C file $input_file not found."
        exit 1
    fi

    base=$(get_basename "$input_file")
    asm="${BUILD_DIR}/asm/${base}${opt_flag}.s"
    obj="${BUILD_DIR}/obj/${base}${opt_flag}.o"

    echo "[*] Compiling C file: $input_file with ${opt_flag:-no optimization} ..."
    
    if [[ -n "$opt_flag" ]]; then
        $GCC_PREFIX-gcc $ABI "$opt_flag" -S "$input_file" -o "$asm"
        $GCC_PREFIX-gcc $ABI "$opt_flag" -c "$input_file" -o "$obj"
    else
        $GCC_PREFIX-gcc $ABI -S "$input_file" -o "$asm"
        $GCC_PREFIX-gcc $ABI -c "$input_file" -o "$obj"
    fi

    echo "[+] Output: $asm, $obj"
}

execute() {
    input_file="$1"
    if [[ ! -f "$input_file" ]]; then
        echo "Error: $input_file not found."
        exit 1
    fi

    base=$(get_basename "$input_file")
    hex_file="${BUILD_DIR}/hex/${base}.hex"
    log_file="${BUILD_DIR}/logs/${base}.txt"

    if [[ ! -f "$hex_file" ]]; then
        echo "Error: $hex_file not found. Compile first."
        exit 1
    fi

    echo "[*] Executing with whisper..."
whisper -x "$hex_file" -s 0x80000000 --tohost 0xd0580000 --profileinst "$log_file" --configfile "$WHISPER_CFG"
    echo "[+] Execution log saved to $log_file"
}

clean() {
    echo "[*] Cleaning generated files..."
    rm -rf "$BUILD_DIR"
    echo "[+] Clean complete."
}

# ===================== Main ======================
if [[ $# -eq 0 ]]; then
    show_help
    exit 1
fi

ACTION=""

while getopts "acehg::" opt; do
    case $opt in
        a) ACTION="all" ;;
        c) ACTION="clean" ;;
        e) ACTION="exec" ;;
        g)
            ACTION="compile_c"
            if [[ "$OPTARG" == -O* ]]; then
                OPT_FLAG="$OPTARG"
            else
                OPT_FLAG="$DEFAULT_OPT"
                [[ -n "$OPTARG" ]] && set -- "$OPTARG" "$@"
            fi
            ;;
        h) show_help; exit 0 ;;
        *) show_help; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

make_dirs

case "$ACTION" in
    all)
        [[ $# -lt 1 ]] && { echo "Error: Please provide at least one .s file."; show_help; exit 1; }
        compile_asm "$@"
        execute "$1"
        ;;
    clean)
        clean
        ;;
    exec)
        [[ $# -lt 1 ]] && { echo "Error: Please provide a .s file."; show_help; exit 1; }
        execute "$1"
        ;;
    compile_c)
        [[ $# -lt 1 ]] && { echo "Error: Please provide a .c file."; show_help; exit 1; }
        compile_c "$OPT_FLAG" "$1"
        ;;
    *)
        show_help
        ;;
esac