"""PhoNLP Named Entity Recognition Dataset"""

import datasets
import os

_URL = "./"
_TRAINING_FILE = "train.txt"
_DEV_FILE = "dev.txt"
_TEST_FILE = "test.txt"

logger = datasets.logging.get_logger(__name__)

class NerPhoConfig(datasets.BuilderConfig):
    """BuilderConfig for NER datasets"""

    def __init__(self, **kwargs):
        """BuilderConfig for NER datasets.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(NerPhoConfig, self).__init__(**kwargs)
        #print (dataset_dir)
        #_URL = dataset_dir + "/"

class NerPho(datasets.GeneratorBasedBuilder):
    """NerPho dataset."""

    BUILDER_CONFIGS = [
        NerPhoConfig(name="ner_phoNLP", version=datasets.Version("1.0.0"), description="Ner PhoNLP dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-PER",
                                "I-PER",
                                "B-ORG", 
                                "I-ORG",
                                "B-LOC",
                                "I-LOC",
                                "B-MISC",
                                "I-MISC",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # urls_to_download = {
        #     "train": f"{_URL}{_TRAINING_FILE}",
        #     "dev": f"{_URL}{_DEV_FILE}",
        #     "test": f"{_URL}{_TEST_FILE}",
        # }
        # downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"{_URL}{_TRAINING_FILE}"}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": f"{_URL}{_DEV_FILE}"}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": f"{_URL}{_TEST_FILE}"}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    # ner phoNLP tokens are \t separated
                    splits = line.split("\t")
                    tokens.append(splits[0])
                    ner_tags.append(splits[1])
            # last example - no need because last one is empty
            # yield guid, {
            #     "id": str(guid),
            #     "tokens": tokens,
            #     "ner_tags": ner_tags,
            # }
