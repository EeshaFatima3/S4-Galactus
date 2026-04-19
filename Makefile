GCC_PREFIX  := riscv32-unknown-elf
CC          := $(GCC_PREFIX)-gcc
OBJCOPY     := $(GCC_PREFIX)-objcopy
OBJDUMP     := $(GCC_PREFIX)-objdump

ARCH        := -march=rv32imfc -mabi=ilp32f

LINK        := veer/link.ld
WHISPER_CFG := veer/whisper.json
WHISPER     := whisper

BUILD       := build
DIRS        := $(BUILD)/exe $(BUILD)/hex $(BUILD)/dis \
               $(BUILD)/asm $(BUILD)/obj $(BUILD)/logs

MAIN        := main.s
NN_SRCS     := nn.s math.s weights.s

SAMPLES     := sample0.s sample1.s sample2.s sample3.s sample4.s \
               sample5.s sample6.s sample7.s sample8.s sample9.s

BASE        := $(basename $(MAIN))
EXE         := $(BUILD)/exe/$(BASE).exe
HEX         := $(BUILD)/hex/$(BASE).hex
DIS         := $(BUILD)/dis/$(BASE).dis
DATA        := $(BUILD)/dis/$(BASE).data
LINKED_ASM  := $(BUILD)/asm/$(BASE).s
LOG         := $(BUILD)/logs/$(BASE).txt

EXTRA ?=

.PHONY: all
all: run

$(DIRS):
	mkdir -p $@

.PHONY: compile
compile: $(DIRS)
	@echo "[*] Compiling: $(MAIN) $(NN_SRCS) $(EXTRA) ..."
	$(CC) $(ARCH) -lgcc -T$(LINK) -nostdlib \
	    -o $(EXE) $(MAIN) $(NN_SRCS) $(EXTRA)
	$(OBJCOPY) -O verilog $(EXE) $(HEX)
	$(OBJDUMP) -S          $(EXE) > $(DIS)
	$(OBJDUMP) -s -j .data $(EXE) > $(DATA)
	$(OBJDUMP) -d -M no-aliases $(EXE) > $(LINKED_ASM)
	@echo "[+] Output: $(EXE), $(HEX), $(DIS), $(DATA), $(LINKED_ASM)"

.PHONY: exec
exec:
	@if [ ! -f $(HEX) ]; then \
	    echo "Error: $(HEX) not found. Run 'make compile' first."; exit 1; \
	fi
	@echo "[*] Executing with whisper..."
	$(WHISPER) -x $(HEX) -s 0x80000000 --tohost 0xd0580000 \
	    --profileinst $(LOG) --configfile $(WHISPER_CFG)
	@echo "[+] Execution log saved to $(LOG)"

.PHONY: run
run: compile exec

.PHONY: run-all
run-all: $(DIRS)
	@for sample in $(SAMPLES); do \
	    base=$$(basename $$sample .s); \
	    exe=$(BUILD)/exe/$$base.exe; \
	    hex=$(BUILD)/hex/$$base.hex; \
	    log=$(BUILD)/logs/$$base.txt; \
	    dis=$(BUILD)/dis/$$base.dis; \
	    data=$(BUILD)/dis/$$base.data; \
	    linked=$(BUILD)/asm/$$base.s; \
	    echo "[*] Building $$sample ..."; \
	    $(CC) $(ARCH) -lgcc -T$(LINK) -nostdlib \
	        -o $$exe $(MAIN) $(NN_SRCS) $$sample; \
	    $(OBJCOPY) -O verilog $$exe $$hex; \
	    $(OBJDUMP) -S $$exe > $$dis; \
	    $(OBJDUMP) -s -j .data $$exe > $$data; \
	    $(OBJDUMP) -d -M no-aliases $$exe > $$linked; \
	    echo "[*] Running $$sample with whisper..."; \
	    $(WHISPER) -x $$hex -s 0x80000000 --tohost 0xd0580000 \
	        --profileinst $$log --configfile $(WHISPER_CFG); \
	    echo "[+] Done: $$sample -> $$log"; \
	done

.PHONY: bench
bench:
	@echo "========================================"
	@echo " Static Instruction Count"
	@echo "========================================"
	@if [ -f $(DIS) ]; then \
	    count=$$(grep -c '^\s*[0-9a-f]\+:' $(DIS)); \
	    echo "  $(DIS): $$count instructions"; \
	else \
	    echo "  No .dis file found. Run 'make compile' first."; \
	fi
	@echo ""
	@echo "========================================"
	@echo " Dynamic Instruction Counts (from logs)"
	@echo "========================================"
	@for log in $(BUILD)/logs/*.txt; do \
	    if [ -f $$log ]; then \
	        echo "  $$log:"; \
	        head -5 $$log; \
	        echo "  ..."; \
	    fi; \
	done

.PHONY: dis
dis:
	@if [ -f $(DIS) ]; then \
	    cat $(DIS); \
	else \
	    echo "No disassembly found. Run 'make compile' first."; \
	fi

.PHONY: clean
clean:
	@echo "[*] Cleaning generated files..."
	rm -rf $(BUILD)
	@echo "[+] Clean complete."

.PHONY: help
help:
	@echo ""
	@echo "Usage: make [target] [EXTRA=<sample.s>]"
	@echo ""
	@echo "Targets:"
	@echo "  all        Compile and run main with nn/math/weights (default)"
	@echo "  compile    Compile only (optionally pass EXTRA=sampleN.s)"
	@echo "  exec       Execute the last compiled hex through Whisper"
	@echo "  run        compile + exec (optionally pass EXTRA=sampleN.s)"
	@echo "  run-all    Build and run every sample file individually"
	@echo "  bench      Print static and dynamic instruction counts"
	@echo "  dis        Print disassembly to stdout"
	@echo "  clean      Remove all build artifacts"
	@echo "  help       Show this message"
	@echo ""
	@echo "Examples:"
	@echo "  make                        # compile + run default"
	@echo "  make run EXTRA=sample0.s    # compile + run with sample0"
	@echo "  make run-all                # run all 10 samples"
	@echo "  make bench                  # show instruction counts"
	@echo "  make clean                  # remove build/"
	@echo ""
