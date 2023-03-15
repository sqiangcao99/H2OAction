REPO_PATH='/xxx/SlowFast'
export PYTHONPATH=$PYTHONPATH:$REPO_PATH
python tools/run_net.py --cfg configs/H2O/SLOWFAST_4x16_R50_TEST.yaml