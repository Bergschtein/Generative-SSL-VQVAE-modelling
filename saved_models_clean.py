import os

generated_ids = [
    "6B4P9V",
    "ILB5XV",
    "ZFMIWQ",
    "293Y9A",
    "P8VIGP"
]


def clean_saved_models(dir):
    num_removed = 0
    num_files = 0
    for file_name in os.listdir(dir):
        num_files += 1
        parts = file_name.split("-")
        file_id = parts[-2]
        if file_id not in generated_ids:
            os.remove(os.path.join(dir, file_name))
            num_removed += 1

    print(f"Removed {num_removed} out of {num_files} files in {dir}")


if __name__ == "__main__":
    dir = "saved_models_copy"
    clean_saved_models(dir)