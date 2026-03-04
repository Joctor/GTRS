TRAIN_TEST_SPLIT=navtrain
metric_cache_path=$NAVSIM_EXP_ROOT/${TRAIN_TEST_SPLIT}_metric_cache

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
metric_cache_path=$metric_cache_path
