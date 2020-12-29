MAX_UPDATE=200000 # 300000
WARMUP=10000      # 15000
INTERVAL=100

SEQ_LEN=512
BS=8
FREQ=1
LR=2e-4

# max_positions=512

for step in 200000 ; do
fairseq-validate \
/bigdata/dataset/text/enar-bin/64K/debug \
--path /bigdata/checkpoint_*_$step.pt \
--train-subset train \
--fp16 \
--memory-efficient-fp16 \
--num-workers 4 \
--task multilingual_masked_lm \
--criterion masked_lm \
--sample-break-mode complete \
--tokens-per-sample $SEQ_LEN \
--max-tokens 8192 \
--max-sentences $BS \
--multilang-sampling-alpha 0.3 \
--required-batch-size-multiple 8 \
--empty-cache-freq 100 \
--skip-invalid-size-inputs-valid-test \
--fast-stat-sync \
--seed 1 \
| tee eval-$step.log
done
