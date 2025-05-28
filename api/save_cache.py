import requests
import click


@click.command()
@click.option("--ip", default="0.0.0.0")
@click.option("--port", default="12244")
@click.option("--cache_dir", default="./")
@click.option("--prompt_ids", default="")
def main(ip, port, cache_dir, prompt_ids: str):
    res = requests.post(
        f"http://{ip}:{port}/cache/save",
        json={
            "cache_dir": cache_dir,
            "prompt_ids": [i.strip() for i in prompt_ids.split(",")],
        },
    )
    print(res.json())


if __name__ == "__main__":
    main()
