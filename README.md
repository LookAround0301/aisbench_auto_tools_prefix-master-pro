原链接参考
https://github.com/rayn-zzz/aisbench_auto_tools_prefix/tree/main

`使用前请先下载最新版本的 `aisbench`，请按以下步骤**彻底卸载**镜像自带的旧版本，并重新从源码安装。`

## 卸载旧版本（彻底清除）

```bash
# 卸载 pip 包
pip uninstall -y ais-bench-benchmark

# 删除残留的安装目录（请根据实际 Python 路径调整）
rm -rf /usr/local/python3.11.10/lib/python3.11/site-packages/ais_bench/
rm -rf /usr/local/python3.11.10/lib/python3.11/site-packages/ais_bench-*.egg-info

# 下载新版本aisbench
git clone https://github.com/AISBench/benchmark.git
pip3 install -e ./ --use-pep517 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install -r requirements/api.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install -r requirements/extra.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 新增变长参数：不输入及使用原来的定长测试

分别为：
输入长度均值、输入长度标准差、输入长度最小值、输入长度最大值

```
length_mean: Optional[int] = None,
length_std: Optional[float] = None,   
length_min: Optional[int] = None,
length_max: Optional[int] = None,
```

变长数据生成网站：
https://www.bchrt.com/tools/normal-distribution-generator/

### 参考变长aisbench样例：

输入8k\~128k不定长，平均32k；输出300

```
python3 aisbench_test.py --input_len 32768 --output_len 300 --data_num 32 --concurrency 8 --request_rate 0 --dataset_type prefix_cache --repeat_rate 90% --prefix_test --dp_size 2 --length_mean 32768 --length_std 49152 --length_min 8192 --length_max 131072
```


## 常见问题

1. 结果被重定向到了aisbench打屏中，**不会存到aisbench.log**，如有需要请自行保存log
2. 请注意修改config.py中的`WORK_PATH`，**不再是镜像usr中的默认路径**
3. 请在aisbench_auto_tools_prefix-master-pro路径下执行指令，以免找不到文件
