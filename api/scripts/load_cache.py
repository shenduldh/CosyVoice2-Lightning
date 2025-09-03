import requests
import click


@click.command()
@click.option("--ip")
@click.option("--port")
@click.option("--cache_path")
@click.option("--prompt_ids", default="")
def main(ip, port, cache_path, prompt_ids: str):
    if prompt_ids == "":
        prompt_ids = []
    else:
        prompt_ids = [i.strip() for i in prompt_ids.split(",")]
        
    res = requests.post(
        f"http://{ip}:{port}/cache/load",
        json={
            "cache_path": cache_path,
            "prompt_ids": prompt_ids,
        },
    )
    print(res.json())


if __name__ == "__main__":
    main()
