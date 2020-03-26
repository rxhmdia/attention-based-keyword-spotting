
python e2e_hotword/train.py \
        --train_data $tfrecord_dir/train/tfrecords.scp \
        --output_dir $dir \
        --shuffle_size 100000 \
        --max_epochs 10 \
        --batch_size 128 \
        --CMVN_json $global_cmvn \
        --learning_rate 0.1 \
       2> $dir/train.log || exit 1
