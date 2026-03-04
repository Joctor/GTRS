NUM_NODES=1
MASTER_ADDR=127.0.0.1 # your master node ip
NODE_RANK=0 # 0 for the master node, 1 and 2 for other sub-nodes

TRAIN_TEST_SPLIT=navtrain
config="default_training" # this config uses the entire navtrain dataset for training
experiment_name=train_flow
agent=flow_agent
cache_path=$NAVSIM_EXP_ROOT/flow_agent_navtrain_dataset_caching/
pdm_result_path=$NAVSIM_EXP_ROOT/pdm_result.csv

# training hyper-parameters
lr=0.0002
bs=16
max_epochs=3

# MASTER_PORT=29500 MASTER_ADDR=${MASTER_ADDR} WORLD_SIZE=${NUM_NODES} NODE_RANK=${NODE_RANK} \
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
    +pdm_result_path=$pdm_result_path    
    
    # trainer.params.num_nodes=${NUM_NODES} \
    # agent.config.ckpt_path=${experiment_name} \


