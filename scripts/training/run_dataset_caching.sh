# 运行前需要注销flow_agent.py 里的vocab_score

TRAIN_TEST_SPLIT=navtrain
agent=flow_agent
experiment_name=${agent}_${TRAIN_TEST_SPLIT}_dataset_caching
CACHE_PATH=$NAVSIM_EXP_ROOT/$experiment_name

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=$agent \
experiment_name=$experiment_name \
cache_path=$CACHE_PATH \

