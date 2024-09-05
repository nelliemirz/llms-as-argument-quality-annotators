import json
import os.path

IN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data', 'human_annotations'))
OUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data', 'predictions'))


def main():
    annotator_sets = ["novice", "expert"]

    for annotator_set in annotator_sets:
        annotations = []
        in_path = os.path.join(IN_PATH, f"{annotator_set}.tsv")
        with open(in_path, "r") as in_file:
            header = in_file.readline().strip().split("\t")
            last_id = None
            annotator_id = 0

            for line in in_file:
                data = line.strip().split("\t")

                if data[0] == last_id:
                    annotator_id += 1
                else:
                    annotator_id = 0

                for dim, rating in zip(header[2:], data[2:]):
                    if dim == "argumentative":
                        continue

                    try:
                        rating = str(int(float(rating)))
                    except ValueError:
                        rating = None

                    if len(annotations) == annotator_id:
                        annotations.append([])
                    annotations[annotator_id].append(
                        {"id": data[0], "dimension": dim, "rating": rating}
                    )

                last_id = data[0]

        for i, annotations in enumerate(annotations):
            out_path = os.path.join(OUT_PATH, f"human-{annotator_set}-{i + 1}.jsonl")
            with open(out_path, "w+") as out_file:
                for annotation in annotations:
                    out_file.write(json.dumps(annotation))
                    out_file.write("\n")


if __name__ == '__main__':
    main()
