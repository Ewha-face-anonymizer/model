"""
Convenience entry point to launch the FastAPI server locally.
"""
import uvicorn


def main() -> None:
    uvicorn.run("api.fastapi_app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
