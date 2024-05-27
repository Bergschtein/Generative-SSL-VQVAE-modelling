import os

# DQZWZT vib and bt gauss all datasets. Seed 1
# Y3A7B9 naive all datasets seed 1

generated_ids = [
    "DQZWZT",
    "Y3A7B9"
]

# generated_ids = [
#     "J4D3NQ",
#     "Q1A6TB",
#     "CYVIE9",
#     "2KCBSS",
#     "6AAE13",
#     "XKF8ZB",
#     "SIPCYD",
#     "WO3S0Z",
#     "P6MCKO",
#     "GDAQEN",
#     "WIZQZ6",
#     "Y3A7B9",
#     "OQFZY4",
#     "DQZWZT",
#     "MKA6WK",
#     "75MW58",
#     "N7NDDJ",
#     "T3PZKA",
#     "52933I",
#     "PWECBS",
#     "NAZB6U",
#     "XF3BTF"
# ]


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
    dir = "saved_models"
    clean_saved_models(dir)
