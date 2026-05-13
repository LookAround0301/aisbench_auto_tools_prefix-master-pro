使用前请先下载最新版本的aisbench，卸载镜像自带的

pip uninstall ais-bench-benchmark
rm -rf /usr/local/python3.11.10/lib/python3.11/site-packages/ais_bench/
git clone https://github.com/AISBench/benchmark.git
pip3 install -e ./ --use-pep517 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install -r requirements/api.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install -r requirements/extra.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
