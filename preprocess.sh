for i in $(seq 0 9); do
fairseq-preprocess \
    --only-source \
    --srcdict dict.txt \
    --trainpref ar.train.$i \
    --validpref ar.dev \
    --destdir shard$i/ar \
    --workers 24
fairseq-preprocess \
    --only-source \
    --srcdict dict.txt \
    --trainpref en.train.$i \
    --validpref en.dev \
    --destdir shard$i/en \
    --workers 24
done