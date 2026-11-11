
import re


def fixed_size_chunking(text, chunk_size=500, overlap=100):
    """
    Split text into chunks while preserving sentence boundaries
    where possible and maintaining overlap between chunks.
    """

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size.")

    if not text.strip():
        return []

    # Split text into sentences.
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_length = len(sentence)

        # Handle sentences larger than chunk_size.
        if sentence_length > chunk_size:

            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            words = sentence.split()

            temp_chunk = []
            temp_length = 0

            for word in words:

                added_length = len(word) + (1 if temp_chunk else 0)

                if temp_chunk and temp_length + added_length > chunk_size:
                    chunks.append(" ".join(temp_chunk))
                    temp_chunk = []
                    temp_length = 0

                if temp_chunk:
                    temp_length += 1

                temp_chunk.append(word)
                temp_length += len(word)

            if temp_chunk:
                chunks.append(" ".join(temp_chunk))

            continue

        added_length = sentence_length + (1 if current_chunk else 0)

        if current_chunk and current_length + added_length > chunk_size:

            chunks.append(" ".join(current_chunk))

            # Preserve previous text as overlap.
            overlap_sentences = []
            overlap_length = 0

            for previous_sentence in reversed(current_chunk):

                needed = len(previous_sentence) + (
                    1 if overlap_sentences else 0
                )

                if overlap_length + needed > overlap:
                    break

                overlap_sentences.insert(0, previous_sentence)
                overlap_length += needed

            current_chunk = overlap_sentences
            current_length = overlap_length

        if current_chunk:
            current_length += 1

        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
