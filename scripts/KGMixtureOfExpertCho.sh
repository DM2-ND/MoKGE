########################
#### Run experiment
########################

DATASET=${}
INPUT_DIR=${}
OUTPUT_DIR=${}

option="
    --data_dir ${INPUT_DIR}/${DATASET}
    --model_name_or_path facebook/bart-base  
    --model_type kgmoe
    --output_dir ${OUTPUT_DIR}/KGMixtureOfExpertCho_Output
    --max_source_length 40
    --max_target_length 60 
    --val_max_target_length 60
    --test_max_target_length 60
    --num_train_epochs 30
    --learning_rate 3e-5 
    --mixture_embedding
    --fp16 
    --do_train 
    --do_eval 
    --do_predict 
    --eval_beams 3 
    --per_device_train_batch_size 60
    --per_device_eval_batch_size 60
    --metric_for_best_model distinct_2
    --predict_with_generate 
    --load_best_model_at_end 
    --overwrite_output_dir 
    --evaluate_during_training
"

cmd="python3 main.py ${option}"

echo $cmd
eval $cmd