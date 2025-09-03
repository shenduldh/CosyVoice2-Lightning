import os
import shutil
import vllm


def register_model(forced=True):
    vllm_path = os.path.dirname(vllm.__file__)
    vllm_models_dir = os.path.join(vllm_path, "model_executor", "models")

    ##########
    tgt_model_file = os.path.join(vllm_models_dir, "cosyvoice2llm.py")
    if os.path.exists(tgt_model_file) and not forced:
        print(f"The model is already existed. Use `forced=True` to overwrite.")
        return
    src_model_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model.py"
    )
    shutil.copy(src_model_file, tgt_model_file)

    ##########
    registry_path = os.path.join(vllm_models_dir, "registry.py")
    with open(registry_path, "r") as f:
        lines = f.readlines()

    remove_pos = None
    for i, line in enumerate(lines):
        if "CosyVoice2LLM" in line:
            remove_pos = i
            break
    if remove_pos is not None:
        if not forced:
            print("The entry is already existed. Use `forced=True` to overwrite.")
            return
        else:
            lines.pop(remove_pos)

    insert_pos = None
    for i, line in enumerate(lines):
        if line.strip().startswith(("**_FALLBACK_MODEL", "**_TRANSFORMERS_MODELS")):
            insert_pos = i + 1
            break
    if insert_pos is None:
        raise ValueError("Could not find insertion position!")
    lines.insert(
        insert_pos,
        '    "CosyVoice2LLM": ("cosyvoice2llm", "CosyVoice2LLM"),\n',
    )

    with open(registry_path, "w") as f:
        f.writelines(lines)
    print("Successfully register model!")


if __name__ == "__main__":
    register_model()
