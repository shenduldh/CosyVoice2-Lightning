import os
import shutil
import sglang


def register_model(forced=True):
    sglang_path = os.path.dirname(sglang.__file__)
    sglang_models_dir = os.path.join(sglang_path, "srt", "models")
    tgt_model_file = os.path.join(sglang_models_dir, "cosyvoice2llm.py")

    if os.path.exists(tgt_model_file) and not forced:
        print(f"The model is already registered. Use `forced=True` to overwrite.")
        return

    src_model_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model.py"
    )

    shutil.copy(src_model_file, tgt_model_file)
    print("Successfully register model!")


if __name__ == "__main__":
    register_model()
