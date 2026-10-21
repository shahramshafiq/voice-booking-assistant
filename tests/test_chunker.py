"""Tests for cutting the model's text into phrases for the voice."""
from app.chunker import extract_speakable_chunks


def test_times_and_decimals_are_never_cut_in_half():
    chunks, left = extract_speakable_chunks("I have tomorrow at 9:30 AM or 2:30 PM. Which one?", first_chunk=True)
    assert chunks == ["I have tomorrow at 9:30 AM or 2:30 PM.", "Which one?"] and left == ""
    chunks, _ = extract_speakable_chunks("It costs 3.50 today. Okay?", force=True)
    assert chunks == ["It costs 3.50 today.", "Okay?"]


def test_a_number_at_the_end_waits_for_the_next_piece_of_text():
    chunks, left = extract_speakable_chunks("I have tomorrow at 9:")
    assert chunks == [] and left == "I have tomorrow at 9:"  # the "30 AM" is still on its way
    chunks, left = extract_speakable_chunks(left + "30 AM. Shall I")
    assert chunks == ["I have tomorrow at 9:30 AM."] and left == "Shall I"
    chunks, _ = extract_speakable_chunks("Your reference is 7 3 8, 3 4 5.", force=True)
    assert chunks == ["Your reference is 7 3 8, 3 4 5."]  # at the very end everything is sent


def test_ordinary_phrase_endings_still_cut_at_once():
    chunks, left = extract_speakable_chunks("Just to confirm: a Consultation tomorrow at 9 AM. Shall", first_chunk=True)
    assert chunks == ["Just to confirm:", "a Consultation tomorrow at 9 AM."] and left == "Shall"
