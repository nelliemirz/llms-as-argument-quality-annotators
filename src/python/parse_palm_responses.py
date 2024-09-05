import json
import os
import re

from predict_argument_quality import parse_response


def main():
    in_path = "/mnt/ceph/storage/data-in-progress/data-research/arguana/ratio24-argquality/predictions"

    for file_name in os.listdir(in_path):
        if re.match(r"^palm2.*\.jsonl$", file_name):
            abs_path = os.path.join(in_path, file_name)

            with open(abs_path, "r") as in_file:
                with open(f"data/ratings/{file_name}", "w+") as out_file:
                    for line in in_file:
                        data = json.loads(line)

                        data["rating"] = parse_response(data["response"])

                        out_file.write(json.dumps(data))
                        out_file.write("\n")


if __name__ == '__main__':
    main()
