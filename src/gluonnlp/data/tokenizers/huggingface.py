__all__ = ['HuggingFaceTokenizer', 'HuggingFaceBPETokenizer', 'HuggingFaceWordPieceTokenizer',
           'HuggingFaceByteBPETokenizer']

import os
import json
from pkg_resources import parse_version
from typing import Optional, Union, List, Tuple
from collections import OrderedDict
from uuid import uuid4
from .base import *
from ..vocab import Vocab, load_vocab
from ...utils.lazy_imports import try_import_huggingface_tokenizers


def is_new_version_model_file(model_file_path: str) -> bool:
    """Check whether the model file belongs to the new version of HuggingFace Tokenizers,
    i.e., >= 0.8

    Parameters
    ----------
    model_file_path
        Path to the model file

    Returns
    -------
    is_new_version
        Whether the model file is generated by the new version of huggingface tokenizer.
    """
    with open(model_file_path, 'r', encoding='utf-8') as f:
        try:
            _ = json.load(f)
            return True
        except Exception:
            return False


def hf_encode(model, sentences, output_type: type = str):
    """

    Parameters
    ----------
    model
        Model object in HuggingFace tokenizer
    sentences
        Input sentences
    output_type
        Output type

    Returns
    -------
    ret
    """
    is_multi_sentences = isinstance(sentences, list)
    if not is_multi_sentences:
        sentences = [sentences]
    encode_sentences = model.encode_batch(sentences, add_special_tokens=False)
    if output_type is str:
        ret = [encode_sentence.tokens for encode_sentence in encode_sentences]
    elif output_type is int:
        ret = [encode_sentence.ids for encode_sentence in encode_sentences]
    else:
        raise TokenTypeNotSupportedError(output_type)
    if is_multi_sentences:
        return ret
    else:
        return ret[0]


def hf_encode_with_offsets(model, sentences, output_type: type = str):
    is_multi_sentences = isinstance(sentences, list)
    if not is_multi_sentences:
        sentences = [sentences]
    encode_sentences = model.encode_batch(sentences, add_special_tokens=False)
    if output_type is str:
        ret = [encode_sentence.tokens for encode_sentence in encode_sentences]
        offsets = [encode_sentence.offsets for encode_sentence in encode_sentences]
    elif output_type is int:
        ret = [encode_sentence.ids for encode_sentence in encode_sentences]
        offsets = [encode_sentence.offsets for encode_sentence in encode_sentences]
    else:
        raise TokenTypeNotSupportedError(output_type)
    if is_multi_sentences:
        return ret, offsets
    else:
        return ret[0], offsets[0]


def hf_decode(model, tokens):
    is_multiple_sentences = is_tokens_from_multiple_sentences(tokens)
    if not is_multiple_sentences:
        tokens = [tokens]
    token_type = get_token_type(tokens)
    if token_type is str:
        id_tokens = [[model.token_to_id(token) for token in sentence] for sentence in
                     tokens]
        ret = model.decode_batch(id_tokens)
    elif token_type is int:
        ret = model.decode_batch(tokens)
    else:
        raise TokenTypeNotSupportedError(token_type)
    if is_multiple_sentences:
        return ret
    else:
        return ret[0]


@TOKENIZER_REGISTRY.register('hf_tokenizer')
class HuggingFaceTokenizer(BaseTokenizerWithVocab):
    def __init__(self, model_file: Optional[str] = None,
                 vocab_file: Optional[str] = None):
        tokenizers = try_import_huggingface_tokenizers()
        assert parse_version(tokenizers.__version__) >= parse_version('0.8'), \
            'Only support tokenizers>=0.8. You can upgrade tokenizers via ' \
            '`python3 -m pip install --upgrade tokenizers`.'
        self._model_file = model_file
        self._vocab_file = vocab_file
        self._model = tokenizers.Tokenizer.from_str(model_file)
        hf_vocab = self._model.get_vocab()
        with open(model_file, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
            self._model_info = model_info
        added_tokens = model_info['added_tokens']
        if vocab_file is not None:
            self._vocab = load_vocab(vocab_file)
        else:
            sorted_hf_vocab_kv = sorted(list(hf_vocab.items()), lambda x: x[1])
            for i, ele in enumerate(sorted_hf_vocab_kv):
                assert ele[1] == i
            all_tokens = [ele[1] for ele in sorted_hf_vocab_kv]
            special_tokens = [token['content'] for token in added_tokens]
            special_token_keys = []
            no_valid_name_key_cnt = 0
            for special_token in special_tokens:
                if special_token.startswith('<') and special_token.endswith('>') \
                        and len(special_token) > 2:
                    key = special_token[1:-1] + '_token'
                else:
                    key = 'special{}_token'.format(no_valid_name_key_cnt)
                    no_valid_name_key_cnt += 1
                assert key not in special_token_keys
                special_token_keys.append(key)
            self._vocab = Vocab(all_tokens,
                                **{key: token
                                   for key, token in zip(special_token_keys, special_tokens)})

        # Verify the special tokens
        for added_token in added_tokens:
            assert self._vocab[added_token['content']] == added_token['id']
            assert added_token['content'] in self._vocab.special_tokens
        # Verify all tokens exist
        for token, idx in hf_vocab.items():
            assert self._vocab[token] == idx

    def encode(self, sentences: SentencesType,
               output_type: type = str) -> Union[TokensType, TokenIDsType]:
        return hf_encode(self._model, sentences, output_type)

    def decode(self, tokens: Union[TokensType, TokenIDsType]) -> SentencesType:
        return hf_decode(self._model, tokens)

    def encode_with_offsets(self, sentences: SentencesType,
                            output_type: type = str) -> Tuple[Union[TokensType, TokenIDsType],
                                                              TokenOffsetsType]:
        return hf_encode_with_offsets(self._model, sentences, output_type)

    @property
    def vocab(self) -> Optional[Vocab]:
        return self._vocab

    def set_vocab(self, vocab):
        raise NotImplementedError('Cannot set vocabulary for the HuggingFaceTokenizer.')

    def __repr__(self):
        ret = '{}(\n' \
              '   type = {}\n' \
              '   model_file = {}\n' \
              '   vocab_file = {}\n' \
              '   normalizer = {}\n' \
              '   vocab = {}\n' \
              ')'.format(self.__class__.__name__,
                         self._model_info['model']['type'],
                         self._model_file,
                         self._vocab_file,
                         self._model_info['normalizer'],
                         self._vocab)
        return ret


class LegacyHuggingFaceTokenizer(BaseTokenizerWithVocab):
    def __init__(self):
        self._vocab = None
        self._model = None

    def encode(self, sentences: SentencesType,
               output_type: type = str) -> Union[TokensType, TokenIDsType]:
        return hf_encode(self._model, sentences, output_type)

    def decode(self, tokens: Union[TokensType, TokenIDsType]) -> SentencesType:
        return hf_decode(self._model, tokens)

    def encode_with_offsets(self, sentences: SentencesType,
                            output_type: type = str) -> Tuple[Union[TokensType, TokenIDsType],
                                                              TokenOffsetsType]:
        return hf_encode_with_offsets(self._model, sentences, output_type)

    @property
    def vocab(self) -> Optional[Vocab]:
        return self._vocab

    def set_vocab(self, vocab):
        raise NotImplementedError('Cannot set vocabulary for the HuggingFaceTokenizer.')


@TOKENIZER_REGISTRY.register('hf_bpe')
class HuggingFaceBPETokenizer(LegacyHuggingFaceTokenizer):
    def __init__(self, merges_file: Optional[str] = None,
                 vocab_file: Optional[str] = None,
                 unk_token: Optional[str] = Vocab.UNK_TOKEN,
                 suffix: Optional[str] = '</w>',
                 dropout: Optional[float] = None,
                 lowercase: bool = False):
        """

        Parameters
        ----------
        merges_file
            The merges file saved by HuggingFace
        vocab_file
            Vocabulary file in GluonNLP
        unk_token
            The unknown token
        suffix
            The suffix for sub-tokens. For example, "Sunnyvale" will be "Sunny vale</w>"
        dropout
            Ratio of the BPE-Dropout
        lowercase
            Whether to lowercase the input before tokenizer
        """
        super().__init__()
        self._merges_file = merges_file
        self._vocab_file = vocab_file
        self._unk_token = unk_token
        self._suffix = suffix
        self._dropout = dropout
        self._lowercase = lowercase
        self.__rebuild_tokenizer()
        self._last_subword_id_set = frozenset([self._vocab[ele]
                                               for ele in self._vocab.all_tokens
                                               if ele.endswith(self._suffix)])

    def is_last_subword(self, tokens: Union[str, int, List[str], List[int]]) \
            -> Union[bool, List[bool]]:
        """Whether the token is the last subword token. This can be used for whole-word masking.

        Parameters
        ----------
        tokens
            The input tokens

        Returns
        -------
        ret
            Whether the token is the last subword token in the list of subwords.
        """
        if isinstance(tokens, str):
            return tokens.endswith(self._suffix)
        elif isinstance(tokens, int):
            return tokens in self._last_subword_id_set
        elif isinstance(tokens, list):
            if len(tokens) == 0:
                return []
            if isinstance(tokens[0], str):
                return [ele.endswith(self._suffix) for ele in tokens]
            elif isinstance(tokens[0], int):
                return [ele in self._last_subword_id_set for ele in tokens]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def set_bpe_dropout(self, bpe_dropout: float):
        """Set the BPE Dropout of the tokenizer

        Parameters
        ----------
        bpe_dropout
            The BPE Dropout ratio

        """
        self._dropout = bpe_dropout
        self.__rebuild_tokenizer()

    def set_lowercase(self, lowercase: float):
        """Set the lowercase flag in the tokenizer

        Parameters
        ----------
        lowercase
            Whether to lowercase the input

        """
        self._lowercase = lowercase
        self.__rebuild_tokenizer()

    @property
    def lowercase(self):
        return self._lowercase

    def __rebuild_tokenizer(self):
        tokenizers = try_import_huggingface_tokenizers()
        # Load the merge file from Huggingface tokenizers < 0.8
        try:
            # using Vocab obj file
            self._vocab = load_vocab(self._vocab_file)
            all_tokens = self._vocab.all_tokens
            hf_vocab = OrderedDict()
            for i in range(len(all_tokens)):
                hf_vocab[all_tokens[i]] = i
            temp_hf_vocab_file = str(uuid4()) + '.hf_vocab'
            with open(temp_hf_vocab_file, 'w', encoding='utf-8') as ftv:
                json.dump(hf_vocab, ftv, ensure_ascii=False)
        except TypeError:
            # using hf_bpe vocab file
            with open(self._vocab_file, 'r', encoding='utf-8') as fv:
                hf_vocab = json.load(fv)
            hf_vocab = sorted(list(hf_vocab.items()), key=lambda x: x[1])
            all_tokens = [x[0] for x in hf_vocab]
            # default special tokens corresponding to the default
            # special_tokens setting in CharBPETokenizer.train
            # and the default special_tokens=[unk]
            self._vocab = Vocab(all_tokens, unk_token=self._unk_token)
            temp_hf_vocab_file = None
        except Exception as exp:
            raise exp
        assert self._unk_token == self._vocab.unk_token
        self._model = tokenizers.CharBPETokenizer(
            vocab_file=temp_hf_vocab_file if temp_hf_vocab_file else self._vocab_file,
            merges_file=self._merges_file,
            unk_token=self._unk_token, suffix=self._suffix, dropout=self._dropout,
            lowercase=self._lowercase)
        if temp_hf_vocab_file:
            os.remove(temp_hf_vocab_file)

    @property
    def vocab(self):
        return self._vocab

    def set_vocab(self, vocab):
        raise NotImplementedError('Cannot set vocabulary for HuggingFaceBPETokenizer.')

    def __repr__(self):
        ret = '{}(\n' \
              '   merges_file = {}\n' \
              '   vocab_file = {}\n' \
              '   unk_token = {}, suffix = {}\n' \
              '   dropout = {}, lowercase = {}\n' \
              '   vocab = {}\n' \
              ')'.format(self.__class__.__name__,
                         os.path.realpath(self._merges_file),
                         os.path.realpath(self._vocab_file),
                         self._is_merge_file,
                         self._unk_token, self._suffix,
                         self._dropout, self._lowercase,
                         self._unicode_normalizer,
                         self._vocab)
        return ret

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_model'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.__rebuild_tokenizer()


@TOKENIZER_REGISTRY.register('hf_bytebpe')
class HuggingFaceByteBPETokenizer(LegacyHuggingFaceTokenizer):
    def __init__(self, merges_file: Optional[str] = None, vocab_file: Optional[str] = None,
                 add_prefix_space: bool = False, lowercase: bool = False,
                 dropout: Optional[float] = None,
                 unicode_normalizer: Optional[str] = None,
                 continuing_subword_prefix: Optional[str] = None,
                 end_of_word_suffix: Optional[str] = None, trim_offsets: bool = False):
        super().__init__()
        self._merges_file = merges_file
        self._vocab_file = vocab_file
        self._add_prefix_space = add_prefix_space
        self._lowercase = lowercase
        self._dropout = dropout
        self._unicode_normalizer = unicode_normalizer
        self._continuing_subword_prefix = continuing_subword_prefix
        self._end_of_word_suffix = end_of_word_suffix
        self._trim_offsets = trim_offsets
        self.__rebuild_tokenizer()

    def set_bpe_dropout(self, bpe_dropout: float):
        self._dropout = bpe_dropout
        self.__rebuild_tokenizer()

    def set_lowercase(self, lowercase: float):
        self._lowercase = lowercase
        self.__rebuild_tokenizer()

    @property
    def lowercase(self):
        return self._lowercase

    def __rebuild_tokenizer(self):
        tokenizers = try_import_huggingface_tokenizers()
        # build vocab and temp_hf_vocab_file
        try:
            # using Vocab obj file
            self._vocab = load_vocab(self._vocab_file)
            all_tokens = self._vocab.all_tokens
            hf_vocab = OrderedDict()
            for i in range(len(all_tokens)):
                hf_vocab[all_tokens[i]] = i
            temp_hf_vocab_file = str(uuid4()) + '.hf_vocab'
            with open(temp_hf_vocab_file, 'w', encoding='utf-8') as ftv:
                json.dump(hf_vocab, ftv, ensure_ascii=False)
        except TypeError:
            # using hf_bytebpe vocab file
            with open(self._vocab_file, 'r', encoding='utf-8') as fv:
                hf_vocab = json.load(fv)
            hf_vocab = sorted(list(hf_vocab.items()), key=lambda x: x[1])
            all_tokens = [x[0] for x in hf_vocab]
            # default special tokens corresponding to the default
            # special_tokens setting in ByteBPETokenizer.train
            # and the default special_tokens=[]
            self._vocab = Vocab(all_tokens)
            temp_hf_vocab_file = None
        self._model = tokenizers.ByteLevelBPETokenizer(
            vocab_file=temp_hf_vocab_file if temp_hf_vocab_file else self._vocab_file,
            merges_file=self._merges_file,
            add_prefix_space=self._add_prefix_space, lowercase=self._lowercase,
            dropout=self._dropout, unicode_normalizer=self._unicode_normalizer,
            continuing_subword_prefix=self._continuing_subword_prefix,
            end_of_word_suffix=self._end_of_word_suffix,
            trim_offsets=self._trim_offsets)
        if temp_hf_vocab_file:
            os.remove(temp_hf_vocab_file)

    def __repr__(self):
        ret = '{}(\n' \
              '   merges_file = {}\n' \
              '   vocab_file = {}\n' \
              '   add_prefix_space = {}, lowercase = {}, dropout = {}\n' \
              '   unicode_normalizer = {}, continuing_subword_prefix = {}\n' \
              '   end_of_word_suffix = {}\n' \
              '   trim_offsets = {}\n' \
              '   vocab = {}\n' \
              ')'.format(self.__class__.__name__,
                         os.path.realpath(self._merges_file),
                         os.path.realpath(self._vocab_file),
                         self._add_prefix_space, self._lowercase, self._dropout,
                         self._unicode_normalizer, self._continuing_subword_prefix,
                         self._end_of_word_suffix,
                         self._trim_offsets,
                         self._vocab)
        return ret

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_model'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.__rebuild_tokenizer()


@TOKENIZER_REGISTRY.register('hf_wordpiece')
class HuggingFaceWordPieceTokenizer(LegacyHuggingFaceTokenizer):
    def __init__(self, vocab_file: Optional[str] = None,
                 unk_token: str = Vocab.UNK_TOKEN,
                 sep_token: str = Vocab.SEP_TOKEN,
                 cls_token: str = Vocab.CLS_TOKEN,
                 pad_token: str = Vocab.PAD_TOKEN,
                 mask_token: str = Vocab.MASK_TOKEN,
                 clean_text: bool = True, handle_chinese_chars: bool = True,
                 strip_accents: bool = False, lowercase: bool = False,
                 wordpieces_prefix: str = "##"):
        super().__init__()
        self._vocab_file = vocab_file
        self._unk_token = unk_token
        self._sep_token = sep_token
        self._cls_token = cls_token
        self._pad_token = pad_token
        self._mask_token = mask_token
        self._clean_text = clean_text
        self._handle_chinese_chars = handle_chinese_chars
        self._strip_accents = strip_accents
        self._lowercase = lowercase
        self._wordpieces_prefix = wordpieces_prefix
        self.__rebuild_tokenizer()
        self._first_subword_id_set = frozenset([self._vocab[ele]
                                                for ele in self._vocab.all_tokens
                                                if not ele.startswith(self._wordpieces_prefix) and
                                                not ele in [self._sep_token, self._cls_token]])

    def is_first_subword(self, tokens: Union[str, int, List[str], List[int]]) \
            -> Union[bool, List[bool]]:
        if isinstance(tokens, str):
            return not tokens.startswith(self._wordpieces_prefix)\
                   and (tokens not in self._vocab.special_tokens)
        elif isinstance(tokens, int):
            return tokens in self._first_subword_id_set
        elif isinstance(tokens, list):
            if len(tokens) == 0:
                return []
            if isinstance(tokens[0], str):
                return [not ele.startswith(self._wordpieces_prefix)
                        and (ele not in self._vocab.special_tokens)
                        for ele in tokens]
            elif isinstance(tokens[0], int):
                return [ele in self._first_subword_id_set for ele in tokens]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def set_lowercase(self, lowercase: bool):
        self._lowercase = lowercase
        self.__rebuild_tokenizer()

    @property
    def lowercase(self):
        return self._lowercase

    def __rebuild_tokenizer(self):
        tokenizers = try_import_huggingface_tokenizers()
        # build vocab and temp_hf_vocab_file
        try:
            # using Vocab obj file
            self._vocab = load_vocab(self._vocab_file)
            all_tokens = self._vocab.all_tokens
        except json.JSONDecodeError:
            # using hf_wordpiece vocab file
            all_tokens = []
            with open(self._vocab_file, 'r', encoding='utf-8') as fv:
                for line in fv:
                    all_tokens.append(line.strip())
            # defualt special tokens corresponding to the default
            # special_tokens setting in BertWordPieceTokenizer.train
            # and the default special_tokens=[pad, unk, cls, sep, mask]
            default_special_tokens = {'pad_token': self._pad_token,
                                      'cls_token': self._cls_token,
                                      'sep_token': self._sep_token,
                                      'mask_token': self._mask_token}
            self._vocab = Vocab(all_tokens, unk_token=self._unk_token, **default_special_tokens)
            all_tokens = self._vocab.all_tokens
        # for safety, also use temp file when using wordpiece vocab file
        # for situation that original all_tokens not cotain special tokens
        # (vocab file of BERT do not contain all special tokens)
        temp_hf_vocab_file = str(uuid4()) + '.hf_vocab'
        with open(temp_hf_vocab_file, 'w', encoding='utf-8') as ftv:
            ftv.write('\n'.join(all_tokens))
        self._vocab.mask_token_id = self._vocab.mask_id
        assert [self._unk_token, self._sep_token, self._cls_token, self._pad_token,
                self._mask_token] == \
               [self._vocab.unk_token, self._vocab.sep_token, self._vocab.cls_token,
                self._vocab.pad_token, self._vocab.mask_token]
        self._model = tokenizers.BertWordPieceTokenizer(
            vocab_file=temp_hf_vocab_file if temp_hf_vocab_file else self._vocab_file,
            unk_token=self._unk_token,
            sep_token=self._sep_token,
            cls_token=self._cls_token,
            pad_token=self._pad_token,
            mask_token=self._mask_token,
            clean_text=self._clean_text,
            handle_chinese_chars=self._handle_chinese_chars,
            strip_accents=self._strip_accents, lowercase=self._lowercase,
            wordpieces_prefix=self._wordpieces_prefix)
        os.remove(temp_hf_vocab_file)

    def __repr__(self):
        ret = '{}(\n' \
              '   vocab_file = {}\n' \
              '   unk_token = {}, sep_token = {}, cls_token = {}\n' \
              '   pad_token = {}, mask_token = {}\n' \
              '   clean_text = {}, handle_chinese_chars = {}\n' \
              '   strip_accents = {}, lowercase = {}\n' \
              '   wordpieces_prefix = {}\n' \
              '   vocab = {}\n' \
              ')'.format(self.__class__.__name__,
                         os.path.realpath(self._vocab_file),
                         self._unk_token, self._sep_token, self._cls_token,
                         self._pad_token, self._mask_token,
                         self._clean_text, self._handle_chinese_chars,
                         self._strip_accents, self._lowercase,
                         self._wordpieces_prefix,
                         self._vocab)
        return ret

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_model'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.__rebuild_tokenizer()
