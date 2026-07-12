"""
MCP服务器配置模块 - 包含连接股票数据MCP服务器的配置信息

说明:
- 直接用 conda base 的 python 启动 MCP 服务器(该环境已装 yfinance/torch/
  transformers/peft), 不用 uv(uv 会用另一个未装依赖的环境)。
- cwd 指向 a-share-mcp 项目根目录, 使 `from src.xxx` 导入正常。
- env 在系统环境基础上强制 UTF-8, 避免中文日志在管道下的编码错误。
"""
import os

# a-share-mcp(现已改为美股 yfinance 数据源) 服务器项目根目录
_MCP_SERVER_DIR = r"C:\Users\dty23\Desktop\Project\Finance\a-share-mcp-is-just-i-need"
# 用于运行服务器的 Python 解释器(conda base, 已安装所需依赖)
_PYTHON = r"C:\Users\dty23\miniconda3\python.exe"

# 继承系统环境 + 强制 UTF-8(保留 PATH 等, 保证 torch 的 DLL 能被找到)
_ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

SERVER_CONFIGS = {
    "a_share_mcp_v2": {
        "command": _PYTHON,
        "args": ["mcp_server.py"],
        "cwd": _MCP_SERVER_DIR,
        "env": _ENV,
        "transport": "stdio",
    }
}
