"""Lunar Landing Site Navigator backend."""


def main() -> None:
    import uvicorn

    uvicorn.run("moonions_backend.api:app", host="0.0.0.0", port=8000, reload=False)
