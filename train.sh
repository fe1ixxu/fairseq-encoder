MAX_UPDATE=400000 # 200000
WARMUP=10000
DECAY_STEP=200000
INTERVAL=2500 # 625 or 10000

# effective batch size (8 GPUs) = 8*32*8 = 2048
SEQ_LEN=512
BS=8
FREQ=32 # 128 or 8
LR=2e-4
MIN_LR=5e-5

# max_positions=512

ARCH=roberta_large
DATA_PATH=/srv/local2/shijie/data/EnFa-128K-bin
CODENAME=EnFa-large-128K
RESTORE_CKPT=/srv/local2/shijie/pretrained/xlmr.large/model.pt

mkdir -p /srv/local2/shijie/checkpoints/$CODENAME

MKL_THREADING_LAYER=GNU fairseq-train \
$DATA_PATH/shard0:$DATA_PATH/shard1:$DATA_PATH/shard2:$DATA_PATH/shard3:$DATA_PATH/shard4:$DATA_PATH/shard5:$DATA_PATH/shard6:$DATA_PATH/shard7:$DATA_PATH/shard8:$DATA_PATH/shard9 \
--save-dir /srv/local2/shijie/checkpoints/$CODENAME \
--tensorboard-logdir /srv/local2/shijie/tensorboard/$CODENAME \
--restore-file $RESTORE_CKPT \
--train-subset train \
--fp16 \
--memory-efficient-fp16 \
--num-workers 4 \
--task multilingual_masked_lm \
--criterion masked_lm \
--arch $ARCH \
--sample-break-mode complete \
--tokens-per-sample $SEQ_LEN \
--optimizer adam \
--adam-betas "(0.9, 0.98)" \
--adam-eps 1e-6 \
--clip-norm 1.0 \
--lr-scheduler polynomial_decay \
--lr $LR \
--end-learning-rate $MIN_LR \
--warmup-updates $WARMUP \
--total-num-update $DECAY_STEP \
--dropout 0.1 \
--attention-dropout 0.1 \
--weight-decay 0.01 \
--max-tokens 8192 \
--max-sentences $BS \
--update-freq $FREQ \
--max-update $MAX_UPDATE \
--multilang-sampling-alpha 0.3 \
--required-batch-size-multiple 8 \
--empty-cache-freq 100 \
--skip-invalid-size-inputs-valid-test \
--log-format json \
--log-interval 5 \
--fast-stat-sync \
--seed 1 \
--validate-interval $INTERVAL \
--save-interval-updates $INTERVAL \
--no-epoch-checkpoints \
--distributed-world-size 8 \
| tee /srv/local2/shijie/checkpoints/$CODENAME/train.log
