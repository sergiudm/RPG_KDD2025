# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json
import os

from genrec.dataset import AbstractDataset
from genrec.models.DiffAR.tokenizer import DiffARTokenizer


class TigerDiffTokenizer(DiffARTokenizer):
    def __init__(self, config: dict, dataset: AbstractDataset):
        super(TigerDiffTokenizer, self).__init__(config, dataset)

    def _tokenized_cache_path(self) -> str:
        cache_config = {
            "cache_version": 1,
            "tokenizer": self.__class__.__name__,
            "metadata": self.config.get("metadata"),
            "sent_emb_model": self.config.get("sent_emb_model"),
            "sent_emb_dim": self.config.get("sent_emb_dim"),
            "sent_emb_pca": self.config.get("sent_emb_pca"),
            "max_item_seq_len": self.config.get("max_item_seq_len"),
            "label_policy": "history_window_last_next_item",
        }
        cache_key = hashlib.md5(
            json.dumps(cache_config, sort_keys=True).encode("utf-8")
        ).hexdigest()[:10]
        return os.path.join(
            self.cache_dir, "processed", f"tokenized_datasets_TigerDiff_{cache_key}"
        )
