"""服务化层：HTTP 接口（FastAPI）与可视化界面（Streamlit）。

两者都是可选依赖，模块内部延迟导入，未安装时不影响 CLI 与单元测试。
"""

__all__ = ["api", "app"]
