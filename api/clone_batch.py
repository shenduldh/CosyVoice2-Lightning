import requests
import click
import os


@click.command()
@click.option("--ip", default="0.0.0.0")
@click.option("--port", default="12244")
@click.option("--speakers_path", default="../assets/speakers")
def main(ip, port, speakers_path):
    for speaker_id in os.listdir(speakers_path):
        audio_path = os.path.join(speakers_path, speaker_id, "audio.mp3")
        transcript_path = os.path.join(speakers_path, speaker_id, "transcript.txt")

        if os.path.exists(audio_path) and os.path.exists(transcript_path):
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()
                res = requests.post(
                    f"http://{ip}:{port}/clone",
                    json={
                        "prompt_text": transcript,
                        "prompt_audio": audio_path,
                        "prompt_id": speaker_id,
                    },
                )
                print(res.json())


if __name__ == "__main__":
    main()
