NUM_NODES=1
WORLD_SIZE=1
MASTER_ADDR=127.0.0.1 # your master node ip
NODE_RANK=0 # 0 for the master node, 1 and 2 for other sub-nodes

TRAIN_TEST_SPLIT=navtrain
config="default_training" # this config uses the entire navtrain dataset for training
experiment_name=train_flow
agent=flow_agent
cache_path=$NAVSIM_EXP_ROOT/flow_agent_navtrain_dataset_caching/
metric_cache_path=$NAVSIM_EXP_ROOT/navtrain_metric_cache/
pdm_result_path=$NAVSIM_EXP_ROOT/pdm_result.csv
last_epoch_pdm_result_path=$NAVSIM_EXP_ROOT/last_epoch_pdm_result.pkl

# training hyper-parameters
lr=0.0002
bs=27 #96GB
max_epochs=2

MASTER_PORT=29500 MASTER_ADDR=${MASTER_ADDR} WORLD_SIZE=${WORLD_SIZE} NODE_RANK=${NODE_RANK} \
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_dense.py \
    --config-name ${config} \
    agent=${agent} \
    experiment_name=${experiment_name} \
    train_test_split=$TRAIN_TEST_SPLIT \
    dataloader.params.batch_size=${bs} \
    ~trainer.params.strategy \
    trainer.params.max_epochs=${max_epochs} \
    trainer.params.precision=32 \
    agent.lr=${lr} \
    use_cache_without_dataset=true \
    force_cache_computation=false \
    cache_path=$cache_path \
    metric_cache_path=$metric_cache_path \
    agent.config.pdm_result_path=$pdm_result_path \
    agent.config.last_epoch_pdm_result_path=$last_epoch_pdm_result_path
    
    #+resume_ckpt_path=/root/ckpt/last.ckpt
    # trainer.params.accelerator=cpu
    # trainer.params.num_nodes=${NUM_NODES} \
    # agent.config.ckpt_path=${experiment_name} \


