import seaborn as sns
import matplotlib.pyplot as plt
from calculate_alpha import get_annotations, QUALITY_DIMENSIONS
import pandas as pd


def main():
    path = "/mnt/ceph/storage/data-in-progress/data-research/arguana/ratio24-argquality/predictions"

    annotators = ["human", "GPT3", "palm2"]
    prompt_types = ["novice", "expert"]
    reasoning = [False, True]

    all_annotations = None
    for annotator in annotators:
        for prompt_type in prompt_types:
            for reason in reasoning:
                if annotator == "human":
                    if reason:
                        continue
                annotations = get_annotations(annotator, prompt_type, reason, 3, path)
                annotations = annotations.melt(id_vars=["id"], value_vars=QUALITY_DIMENSIONS)
                annotations["Annotator"] = f"{annotator}_{prompt_type}"

                if reason:
                    annotations["Annotator"] = f"{annotator}_{prompt_type}_reasoning"

                if all_annotations is None:
                    all_annotations = annotations
                else:
                    all_annotations = pd.concat([all_annotations, annotations])

    all_annotations = all_annotations.fillna(0)
    all_annotations = all_annotations.reset_index()
    plt.figure(figsize=(10, 5))

    plt.rcParams["pdf.fonttype"] = "truetype"
    sns.set_context("paper", font_scale=1.0)
    g = sns.histplot(data=all_annotations, x="Annotator", hue="value", multiple="dodge", discrete=True, shrink=0.8,
                 palette=sns.color_palette("hot"))
    sns.move_legend(g, "upper right", bbox_to_anchor=(1.20, 1.0), title='Quality Rating', labels=["Cannot judge", "Low", "Medium", "High"])
    # plt.xticks(range(4), labels=["Cannot Judge", "Low", "Medium", "High"])
    plt.xticks(rotation=45)
    plt.xlabel("Annotator")
    plt.grid()
    plt.tight_layout()
    plt.savefig("rating-distribution-histogram.pdf")
    plt.show()


if __name__ == '__main__':
    main()
