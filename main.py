import uvicorn


def main() -> None:
    """本地启动入口。

    说明：
    - 根目录 `main.py` 不承载业务逻辑，只负责启动应用。
    - 真正的 FastAPI 应用定义放在 `app/main.py` 中。
    - 这样分离后，后续测试、脚本和部署入口会更清晰。
    """

    uvicorn.run(
        app="app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
