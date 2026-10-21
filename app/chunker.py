"""Decides which part of the text the LLM has written so far is ready to be spoken.

This is the same logic as the earlier LLM + TTS project, with the numbers passed in
(they come from .env) so the function stays small and easy to test.
"""

PUNCTUATION = ".!?;:"


def extract_speakable_chunks(
    buffer,
    force=False,
    first_chunk=False,
    first_chunk_chars=28,
    target_chars=50,
    max_extra_chars=6,
):
    """Return (chunks_ready_to_speak, text_left_over).

    first_chunk=True uses a much smaller size so the first words go out quickly.
    force=True sends everything that is left (used when the LLM has finished).
    """
    chunks = []
    target = first_chunk_chars if first_chunk else target_chars
    max_chars = target + max_extra_chars

    while buffer.strip():
        boundary = None

        # Prefer a sentence/phrase boundary if one is already available.
        for index, char in enumerate(buffer):
            if char not in PUNCTUATION:
                continue

            # Avoid splitting numbers such as 3.14 and times such as 9:30.
            if char in ".:" and index > 0 and buffer[index - 1].isdigit():
                if index == len(buffer) - 1 and not force:
                    break  # "at 9:" may still become "at 9:30" with the next piece of text
                if index < len(buffer) - 1 and buffer[index + 1].isdigit():
                    continue

            boundary = index + 1
            break

        if boundary is not None:
            chunk = buffer[:boundary].strip()
            buffer = buffer[boundary:].lstrip()
            if chunk:
                chunks.append(chunk)
            continue

        if len(buffer) >= target:
            split_at = buffer.rfind(" ", 0, max_chars + 1)
            if split_at > 0:
                chunk = buffer[:split_at].strip()
                buffer = buffer[split_at:].lstrip()
                if chunk:
                    chunks.append(chunk)
                continue

            if len(buffer) <= max_chars and not force:
                break

            split_at = buffer.rfind(" ")
            if split_at > 0:
                chunk = buffer[:split_at].strip()
                buffer = buffer[split_at:].lstrip()
                if chunk:
                    chunks.append(chunk)
                continue

        if force:
            remainder = buffer.strip()
            if remainder:
                chunks.append(remainder)
            buffer = ""

        break

    return chunks, buffer