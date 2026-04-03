import io
import json
import zipfile
from itertools import zip_longest
from pathlib import Path

import datasets
from datasets.features import Sequence

_SPLITS = {
    "train": "./train.zip",
    "validation": "./validation.zip",
    "test": "./test.zip",
}

_DESCRIPTION = """
The DailyDialog dataset as provided in the original form with a bit of preprocessing applied to enable dast prototyping.
The splits are as in the original distribution.
"""

_CITATION = """
@inproceedings{li2017dailydialog,
  title={DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset},
  author={Li, Yanran and Su, Hui and Shen, Xiaoyu and Li, Wenjie and Cao, Ziqiang and Niu, Shuzi},
  booktitle={Proceedings of the Eighth International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  pages={986--995},
  year={2017}
}
"""

_LICENSE = "Like the original DailyDialogue dataset, this dataset is released under the CC BY-NC-SA 4.0."
_HOMEPAGE = "http://yanran.li/dailydialog"


class DailyDialog(datasets.GeneratorBasedBuilder):
    """
    The DailyDialog dataset as provided in the original form with a bit of preprocessing applied to enable dast prototyping.
    The splits are as in the original distribution.
    """

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="full", version=VERSION, description="The full dataset."
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "acts": Sequence(datasets.Value("int8")),
                    "emotions": Sequence(datasets.Value("int8")),
                    "utterances": Sequence(datasets.Value("string")),
                }
            ),
            citation=_CITATION,
            license=_LICENSE,
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        dl_paths = dl_manager.download(_SPLITS)
        return [
            datasets.SplitGenerator(
                name=split_name,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"data_path": split_path},
            )
            for split_name, split_path in dl_paths.items()
        ]

    def _generate_examples(self, data_path: str):
        split_name: str = str(Path(data_path).stem)

        with zipfile.ZipFile(data_path) as zip_file:
            files_list = list(map(str, zip_file.namelist()))

            acts_file = next((f for f in files_list if "act" in f.lower()))
            emotions_file = next((f for f in files_list if "emotion" in f.lower()))
            utterances_file = next(
                (
                    f
                    for f in files_list
                    if "act" not in f.lower()
                    and "emotion" not in f.lower()
                    and "dialogues" in f.lower()
                )
            )

            acts_file = io.TextIOWrapper(
                zip_file.open(acts_file),
                encoding="utf-8",
            )
            emotions_file = io.TextIOWrapper(
                zip_file.open(emotions_file),
                encoding="utf-8",
            )
            utterances_file = io.TextIOWrapper(
                zip_file.open(utterances_file),
                encoding="utf-8",
            )

            sentinel = object()

            misalignments = 0

            for idx, combo in enumerate(
                zip_longest(
                    acts_file, emotions_file, utterances_file, fillvalue=sentinel
                )
            ):
                if sentinel in combo:
                    raise ValueError("Iterables have different lengths")

                acts, emos, utts = combo

                acts = [int(a.strip()) for a in acts.strip().split(" ")]
                emos = [int(a.strip()) for a in emos.strip().split(" ")]
                utts = [
                    a.strip() for a in utts.strip().strip("__eou__").split("__eou__")
                ]

                lens = dict(utts_len=len(utts), acts_len=len(acts), emos_len=len(emos))

                assert len(utts) == len(acts), lens
                assert len(acts) == len(emos), lens

                item = {
                    "id": f"{split_name}_{idx}",
                    "acts": acts,
                    "emotions": emos,
                    "utterances": utts,
                }

                yield item["id"], item
