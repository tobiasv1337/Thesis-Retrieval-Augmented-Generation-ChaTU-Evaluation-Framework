import os
import json
import pandas as pd
import logging
import re

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


def translate(utterances, cache_path=None, source="de", target="en"):
    return translate_api(utterances, cache_path=cache_path, source=source, target=target)

def translate_api(utterances, cache_path=None, source="de", target="en"):
    if isinstance(utterances, (str, pd.DataFrame, pd.Series)):
        raise TypeError("utterances must be a sequence of strings")
    utterances = list(utterances)
    if any(not isinstance(utterance, str) for utterance in utterances):
        raise TypeError("utterances must contain only strings")

    from deep_translator import GoogleTranslator
    from deep_translator.exceptions import TranslationNotFound
    translator = GoogleTranslator(source=source, target=target)

    translation_cache = {}

    # Load the existing cache if it exists
    if cache_path is not None:
        cache_file_path = os.path.join(cache_path, "translations.json")
        if os.path.exists(cache_file_path):
            with open(cache_file_path, 'r', encoding='utf8') as f:
                translation_cache = json.load(f)

    # Identify utterances that need translation
    #utterances_to_translate = [str(utterance) for utterance in utterances["utterance"] if utterance not in translation_cache]
    utterances_to_translate = [str(utterance) for utterance in utterances if utterance not in translation_cache]
    # Keep batches small so one provider failure does not discard a long run.
    for start in range(0, len(utterances_to_translate), 50):
        batch = utterances_to_translate[start:start + 50]
        try:
            logging.info("Batch Translation")
            translated_batch = translator.translate_batch(batch)
            if translated_batch is None or len(translated_batch) != len(batch):
                raise ValueError("translation provider returned an incomplete batch")
        except Exception as error:
            logging.warning(f"Batch translation failed: {error}")
            translated_batch = []
            for utterance in batch:
                try:
                    translated_batch.append(translator.translate(utterance))
                except Exception as item_error:
                    logging.warning(
                        f"Translation not found for utterance {utterance}: "
                        f"{item_error}",
                    )
                    translated_batch.append(utterance)

        for utterance, translation in zip(batch, translated_batch):
            if not isinstance(translation, str) or not translation.strip():
                translation = utterance
            translation_cache[utterance] = translation

        if cache_path is not None:
            temporary_path = cache_file_path + ".tmp"
            with open(temporary_path, 'w', encoding='utf8') as f:
                json.dump(translation_cache, f, ensure_ascii=False, indent=4)
            os.replace(temporary_path, cache_file_path)

    # Apply translations to the DataFrame
    utterances = pd.DataFrame(utterances, columns=["utterance"])
    utterances['translated_utterance'] = utterances['utterance'].map(translation_cache)
    return list(utterances['translated_utterance'])


class LocalTranslator:
    def __init__(self, tgt_lang="en", src_lang="mul", max_length=256, batch_size=32):
        self.tgt_lang = tgt_lang
        self.src_lang = src_lang
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer, self.model = self.load_model()
        self.filtered = []

    def load_model(self):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        model_name = f"Helsinki-NLP/opus-mt-{self.src_lang}-{self.tgt_lang}"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model

    def is_suspicious(self, text):
        return len(text) > 100 and len(set(text.lower().split())) < 5

    def looks_like_hallucinated_garbage(self, text):
        return (
            re.search(r"[@#%^*_=]{3,}", text) or
            re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text) or
            re.search(r"(\b\w+\b)( \1){5,}", text.lower())
        )

    def translate_batch(self, texts):
        import torch
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=4,
                return_dict_in_generate=True,
                output_scores=False
            )

        sequences = outputs.sequences
        decoded = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        lengths = (sequences != self.tokenizer.pad_token_id).sum(dim=1)

        results = []
        for i, text in enumerate(decoded):
            if (
                lengths[i] >= self.max_length or
                self.is_suspicious(text) or
                self.looks_like_hallucinated_garbage(text)
            ):
                self.filtered.append((texts[i], text))
                results.append("")
            else:
                results.append(text)
        return results

    def translate(self, texts):
        import psutil
        logging.info("Starting batch translation...")
        results = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            if 1%10 == 0: logging.info(f"Translating batch {i // self.batch_size + 1} "
                         f"({i}+{len(batch)}/{len(texts)} items), memory={psutil.Process().memory_info().rss / 1024**2:.1f} MB")
            results.extend(self.translate_batch(batch))

        logging.info("Translation complete.")
        return results


def translate_locally(texts, trg_lang="de"):
    """options for target language: de, en
        Returns:
                list of translated utterances in english, list of translated utterances in target language
            if target_language is "en":
                None, list of translated utterances in english
    """
    translator_english = LocalTranslator()
    texts_english = translator_english.translate(texts)
    if trg_lang == "en": return None, texts_english
    translator = LocalTranslator(tgt_lang=trg_lang, src_lang="en")
    return texts_english, translator.translate(texts_english)
