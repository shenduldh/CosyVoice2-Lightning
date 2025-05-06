import os
import shutil
import vllm


vllm_path = os.path.dirname(vllm.__file__)
target_dir = os.path.join(vllm_path, "model_executor", "models")
target_file = os.path.join(target_dir, "cosyvoice2.py")

file_path = os.path.abspath(__file__)

model_file = os.path.join(os.path.dirname(file_path), "vllm_registration.py")

if not os.path.exists(model_file):
    raise FileNotFoundError(f"Source file {model_file} not found")

shutil.copy(model_file, target_file)
print(f"Copied {model_file} to {target_file}")

registry_path = os.path.join(target_dir, "registry.py")
new_entry = '    "CosyVoice2Model": ("cosyvoice2", "CosyVoice2VLLMEngine"),  # noqa: E501\n'

with open(registry_path, "r") as f:
    lines = f.readlines()

entry_exists = any("CosyVoice2Model" in line for line in lines)

if not entry_exists:
    insert_pos = None
    for i, line in enumerate(lines):
        if line.strip().startswith("**_FALLBACK_MODEL"):
            insert_pos = i + 1
            break

    if insert_pos is None:
        raise ValueError("Could not find insertion point in registry.py")

    lines.insert(insert_pos, new_entry)

    with open(registry_path, "w") as f:
        f.writelines(lines)
    print("Successfully updated registry.py")
else:
    print("Entry already exists in registry.py, skipping modification")

print("All operations completed successfully!")
