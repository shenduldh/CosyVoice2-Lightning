import uvicorn
import os
import click
from dotenv import load_dotenv


@click.command()
@click.option("--env", default="./.env")
def main(env):
    load_dotenv(env, override=True)
    uvicorn.run(
        "app:app",
        loop="none",
        host=os.getenv("HOST"),
        port=int(os.getenv("PORT")),
        reload_excludes=["logs/*", "test.py"],
        reload_includes=[],
        reload=False,
    )


if __name__ == "__main__":
    main()
