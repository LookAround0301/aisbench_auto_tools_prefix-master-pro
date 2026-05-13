# 安装最新版本 AISBench

使用前请先下载最新版本的 `aisbench`，请按以下步骤**彻底卸载**镜像自带的旧版本，并重新从源码安装。

## 卸载旧版本（彻底清除）

```bash
# 卸载 pip 包
pip uninstall -y ais-bench-benchmark

# 删除残留的安装目录（请根据实际 Python 路径调整）
rm -rf /usr/local/python3.11.10/lib/python3.11/site-packages/ais_bench/
rm -rf /usr/local/python3.11.10/lib/python3.11/site-packages/ais_bench-*.egg-info

# 若不确定 Python 路径，可使用以下命令查找
python3 -c "import site; print(site.getsitepackages())"
