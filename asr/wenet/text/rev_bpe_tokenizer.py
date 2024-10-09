import logging
from os import PathLike
from typing import Dict, List, Optional, Union
from wenet.text.char_tokenizer import CharTokenizer
from wenet.text.tokenize_utils import tokenize_by_bpe_model
import torch
# import numpy as np


class RevBpeTokenizer(CharTokenizer):

    def __init__(
        self,
        bpe_model: Union[PathLike, str],
        symbol_table: Union[str, PathLike, Dict],
        non_lang_syms: Optional[Union[str, PathLike, List]] = None,
        split_with_space: bool = False,
        connect_symbol: str = '',
        unk='<unk>',
        full_config: Dict={}
    ) -> None:
        logging.debug(f"{bpe_model=}, {symbol_table=}, {non_lang_syms=}, {split_with_space=}, {connect_symbol=}, {unk=}, {full_config=}")
        super().__init__(symbol_table, non_lang_syms, split_with_space,
                         connect_symbol, unk)
        self.remove_sw = full_config.get('remove_sw', True)
        self.replace_unk_as_unknown = full_config.get('replace_unk_as_unknown', True)
        self.connect_symbol = connect_symbol
        # JPR: TODO: other flags to implement: force_wb

        self._model = bpe_model
        # NOTE(Mddct): multiprocessing.Process() issues
        #              don't build sp here
        self.bpe_model = None

    def _build_sp(self):
        if self.bpe_model is None:
            import sentencepiece as spm
            self.bpe_model = spm.SentencePieceProcessor()
            self.bpe_model.load(self._model)

    # overriding a lot of things here...
    def text2tokens(self, line: str) -> List[str]:
        self._build_sp()
        line = line.strip()

        if self.remove_sw:
            line = line.replace('<sw>', '').replace('  ',' ').strip()

        if self.replace_unk_as_unknown:
            line = line.replace("<unk>", "<unknown>")

        # other things might be required here...
        # like removing trailing dashes, etc.

        tokens = self.bpe_model.encode(line, out_type=str)
        #tokens = torch.tensor([tk for tk in self.bpe_model.encode(line, out_type=int)])
        #print(f"line = {line}, tokens = {tokens}")

        return tokens


        # if self.non_lang_syms_pattern is not None:
        #     parts = self.non_lang_syms_pattern.split(line.upper())
        #     parts = [w for w in parts if len(w.strip()) > 0]
        # else:
        #     parts = [line]

        # tokens = []
        # for part in parts:
        #     if part in self.non_lang_syms:
        #         tokens.append(part)
        #     else:
        #         tokens.extend(tokenize_by_bpe_model(self.bpe_model, part))
        # return tokens

    # from base class
    def tokens2text(self, tokens: List[str]) -> str:
        #self._build_sp()
        #text = super().tokens2text(tokens)
        text = self.connect_symbol.join(tokens)
        #return text
        return text.replace("▁", ' ').strip()
