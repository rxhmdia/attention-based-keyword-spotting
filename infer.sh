       python infer_streaming_split_batch.py \
           --test_data_path $tfrecord_dir/test/tfrecords.scp \
           --model_path $model_dir \
           --output_confidence $result_dir/confidence_score \
           --batch_size 1 \
           --CMVN_json $global_cmvn || exit 1
