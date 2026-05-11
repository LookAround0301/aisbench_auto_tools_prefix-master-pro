import os, errno
import argparse
import re
import logging
from process_dataset import create_data
from config import *
from save_file import get_data, save_csv, save_log
from gen_multi_prefix_dataset import ensure_dir,create_multi_prefix_dataset,parse_prefix_ratio
logging.getLogger().setLevel(logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_len', type=int, default=3500, help="input token length")
    parser.add_argument("--output_len", type=str, default="1500", help="output token length")
    parser.add_argument("--data_num", type=int, default=8192, help="dataset number")
    parser.add_argument("--concurrency", type=str, default="2048", help="max concurrency")
    parser.add_argument("--request_rate", type=str, default="0", help="request rate")
    parser.add_argument("--test_type", type=str, default="stream", help="text or stream")
    parser.add_argument("--dataset", type=str, default="none", help="dataset path")
    parser.add_argument("--repeat", type=int, default=1, help="number of test repeat times")
    parser.add_argument("--enable_think", action='store_true', default=False, help="enable thinking for ds v3.1")
    parser.add_argument("--test_accuracy", action='store_true', default=False, help="test accuracy")
    parser.add_argument("--npu_num", type=int, default=1, help="npu numbers")
    parser.add_argument("--dataset_type", type=str, default="normal", help="normal or prefix_cache")
    parser.add_argument("--prefix_num", type=int, default=1, help="prefix numbers")
    parser.add_argument("--repeat_rate", type=str, default="0", help="dataset repeat rate")
    parser.add_argument("--prefix_test", action='store_true', default=False, help="test prefix dataset firstly")
    parser.add_argument("--seed", type=int, default=1, help="dataset random seed")
    parser.add_argument("--dp_size", type=int, default=1, help="dp size")
    return parser.parse_args()

def generate_aisbench_command(DEFAULT_PERFORMANCE_TEST):
    ais_bench_cmd = "ais_bench --models vllm_api_chat_temp --datasets gsm8k_gen_0_shot_cot_str_perf --debug --summarizer stable_stage --mode perf --reuse 20260428_235545"   
    # if test_accuracy:
    #     ais_bench_cmd = "ais_bench --models vllm_api_chat_temp --datasets gsm8k_gen_0_shot_cot_str_perf --dump-eval-details"
    # else:
    #     ais_bench_cmd = "ais_bench --models vllm_api_chat_temp --datasets gsm8k_gen_0_shot_cot_str_perf --mode perf --debug"
    return ais_bench_cmd

def save_result(request_rate, npu_num):
    aisbench_log_dir = "aisbench.log"
    filename = "aisbench_result.csv"
    ans, log_dir=get_data(aisbench_log_dir,request_rate,npu_num)
    save_log(aisbench_log_dir, log_dir)
    save_csv(ans, filename)

if __name__ == '__main__':
    args = parse_arguments()
    input_len = args.input_len
    output_len = args.output_len
    data_num = args.data_num
    concurrency = args.concurrency
    request_rate = args.request_rate
    test_type = args.test_type
    dataset_path_input = args.dataset
    test_times = args.repeat
    enable_think = args.enable_think
    test_accuracy = args.test_accuracy
    npu_num = args.npu_num
    prefix_num = args.prefix_num
    repeat_rate = parse_prefix_ratio(args.repeat_rate)
    prefix_test = args.prefix_test
    dataset_type = args.dataset_type
    seed = args.seed
    dp_size = args.dp_size

    logging.info(f"input token length: {input_len}")
    logging.info(f"output token length: {output_len}")
    logging.info(f"number of dataset: {data_num}")
    logging.info(f"concurrency: {concurrency}")
    logging.info(f"request rate: {request_rate}")
    logging.info(f"test type: {test_type}")
    logging.info(f"test_times: {test_times}")
    logging.info(f"v3.1 enable_think: {enable_think}")
    logging.info(f"accuracy test: {test_accuracy}")
    logging.info(f"npu numbers: {npu_num}")
    logging.info(f"prefix numbers: {prefix_num}")
    logging.info(f"dataset repeat rate: {repeat_rate}")
    logging.info(f"test prefix dataset: {prefix_test}")
    logging.info(f"dataset type: {dataset_type}")
    logging.info(f"seed: {seed}")
    logging.info(f"dp_size: {dp_size}")

    # 生成 aisbench 命令
    ais_bench_cmd = generate_aisbench_command(DEFAULT_PERFORMANCE_TEST)
    logging.info(f"test start, use command: {ais_bench_cmd}")
    # 执行命令    
    logging.info(f"[开始] 全量数据集测试")
    os.system(ais_bench_cmd)
    logging.info(f"[完成] 全量数据集测试完成")

    
    # 保存结果
    save_result(request_rate, npu_num)
