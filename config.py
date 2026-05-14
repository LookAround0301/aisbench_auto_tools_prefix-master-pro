# 存放数据集的文件夹路径，需可访问
DATASET_PATH = "/mnt/path_to_store_dataset"

# aisbench 工作路径, 为 git clone aisbench 后得到的 benchmark 目录的绝对路径
# 可通过命令 `pip show ais_bench_benchmark | grep location` 查询
# mindie镜像打包的aisbench位置为 /usr/local/lib/python3.11/site-packages
WORK_PATH = "/usr/local/lib/python3.11/site-packages"

# 服务化配置的模型名称
MODEL_NAME = "ds_r1"
# 模型权重路径, 用于读取 tokenizer
MODEL_PATH = "/mnt/weight/DeepSeek-R1_w8a8_mtp"
# 请求目的 IP
HOST_IP = "141.1.1.101"
# 请求目的端口
HOST_PORT = "31015"

# 如果使用稳态测试请将该字段设置为 "stable_stage"
DEFAULT_PERFORMANCE_TEST = "default_perf"