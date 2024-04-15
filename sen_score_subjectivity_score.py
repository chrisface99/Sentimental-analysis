import nltk

def calculate_bleu_score(reference: str, candidate: str) -> float:
    """
    Calculate the BLEU score between a reference and a candidate text.

    :param reference: The reference text.
    :param candidate: The candidate text.
    :return: The BLEU score as a float between 0 and 1.
    """
    bleu_score = nltk.translate.bleu_score.sentence_bleu(
        references=[reference],
        hypothesis=candidate,
        smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1
    )
    return bleu_score

original_text = """
... (provided example original text)...
"""

translated_text = """
... (provided example translated text)...
"""

bleu_score = calculate_bleu_score(original_text, translated_text)
print(f"BLEU Score: {bleu_score}")