# Are Large Language Models Reliable Argument Quality Annotators?


The aggregated dataset used in the paper is available at [Zenodo](https://zenodo.org/records/13692561).

The `novice-annotations-unaggregated.zip` contains unaggregated annotations made by 108 novice annotators.

Each of the annotators was assigned ~10 arguments and their respective annotator id can be found in the filename as `group-X-member-Y`.

Each file contains annotations in the following format:
```json
{
  "<argument_id>-<argument_dimension>: "<annotation>",
    ...
  }
```

Where `<argument_id>` is the id of the argument and `<argument_dimension>` is the dimension of the argument. The annotation is a string that can be one of the following: `3` (High), `2` (Medium), `1` (Low), `?` (Cannot judge).
For more details, please refer to the [novice annotation guidelines](https://github.com/nelliemirz/llms-as-argument-quality-annotators/blob/main/data/annotation-guidelines/novice-annotation-guidelines.pdf).

----

**Quality dimensions:** The argument quality dimensions can be found in the `<zenodo_data>/data/dimensions_definitions.jsonl` file, as well as in the [novice annotation guidelines](https://github.com/nelliemirz/llms-as-argument-quality-annotators/blob/main/data/annotation-guidelines/novice-annotation-guidelines.pdf).

**Arguments:** see `data/arguments.jsonl`.

**Aggregated annotations:** see `data/human_annotations/novice.tsv`.

----

Please use the following citation:

```
@InProceedings{mirzakhmedova:2024b,
  author =                   {Nailia Mirzakhmedova and Marcel Gohsen and Chia Hao Chang and Benno Stein},
  booktitle =                {1st International Conference on Recent Advances in Robust Argumentation Machines {(RATIO-24)}},
  doi =                      {10.1007/978-3-031-63536-6_8},
  editor =                   {Philipp Cimiano and Anette Frank and Michael Kohlhase and Benno Stein},
  month =                    jun,
  pages =                    {129--146},
  publisher =                {Springer},
  site =                     {Bielefeld, Germany},
  title =                    {{Are Large Language Models Reliable Argument Quality Annotators?}},
  volume =                   14638,
  year =                     2024
}
```