from transformers import OpenAIGPTTokenizer

# Initialize the (static) tokenizer
tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")


def get_token_count(text: str) -> int:
    """
    Get the length of a text in (GPT) tokens
    :param text: The text to count the tokens
    :return: The number of tokens
    """
    token_count = len(tokenizer.tokenize(text))
    return token_count


def approximate_truncate_to_token_count(text: str, token_count: int) -> str:
    """
    Get text approximately in the token count length
    :param token_count: Length of text in tokens. The returned text will have this size +- 10%
    :param text: The text to count the tokens
    :return: The number of tokens
    """
    # First approximation
    token_count = int(token_count * 0.95)
    char_length = int(token_count * 10)
    # maximum iterations to 10 to avoid endless loop
    for i in range(0, 10):
        truncated_text = text[:char_length]
        current_token_count = get_token_count(truncated_text)
        # Break if variance < 5%
        if token_count * 1.05 > current_token_count > token_count * 0.95:
            break
        # Calculate correction factor; avoid division by zero
        correction_factor = token_count / max(current_token_count, 1)
        char_length = int(char_length * correction_factor)

    return truncated_text