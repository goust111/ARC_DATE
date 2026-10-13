from __future__ import annotations

import hashlib
import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PLAY_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
TEMPLATE_FAMILIES = ("gravity_only", "single_exact", "single_long", "double_chain")
MAX_GENERATION_ATTEMPTS = 400
RECENT_OUTPUT_LIMIT = 64

_GEN_COUNTER = 0
_RECENT_OUTPUT_SIGS: deque[str] = deque(maxlen=RECENT_OUTPUT_LIMIT)


def _threshold_from_bar(grid: np.ndarray) -> int:
    bottom = np.asarray(grid[-1], dtype=int)
    width = int(bottom.shape[0])
    length = 0
    while length < width and int(bottom[length]) == GREY:
        length += 1
    assert 1 <= length < width, "expected a nonempty left-aligned grey scale bar"
    assert np.all(bottom[:length] == GREY), "scale bar must be contiguous"
    assert np.all(bottom[length:] == BLACK), "cells after the scale bar must be black"
    return int(length)


def _settle(playfield: np.ndarray) -> np.ndarray:
    play = np.asarray(playfield, dtype=int)
    height, width = map(int, play.shape)
    out = np.zeros_like(play)
    for c in range(width):
        column = play[:, c]
        falling = column[column != BLACK]
        if falling.size:
            out[height - falling.size :, c] = falling
    return out


def _clearable_mask(playfield: np.ndarray, threshold: int) -> np.ndarray:
    play = np.asarray(playfield, dtype=int)
    height, width = map(int, play.shape)
    mask = np.zeros_like(play, dtype=bool)
    for r in range(height):
        c = 0
        while c < width:
            color = int(play[r, c])
            if color == BLACK:
                c += 1
                continue
            c1 = c + 1
            while c1 < width and int(play[r, c1]) == color:
                c1 += 1
            if c1 - c >= threshold:
                mask[r, c:c1] = True
            c = c1
    return mask


def _simulate_playfield(
    playfield: np.ndarray, threshold: int, *, max_rounds: int | None = None
) -> tuple[np.ndarray, dict]:
    current = np.asarray(playfield, dtype=int).copy()
    removed_counts: list[int] = []
    settled_history: list[np.ndarray] = []
    rounds = 0

    while True:
        current = _settle(current)
        settled_history.append(current.copy())
        clear = _clearable_mask(current, threshold)
        if not np.any(clear):
            return current, {
                "rounds": rounds,
                "removed_counts": removed_counts,
                "settled_history": settled_history,
            }

        removed_counts.append(int(np.count_nonzero(clear)))
        current = current.copy()
        current[clear] = BLACK
        rounds += 1

        if max_rounds is not None and rounds >= max_rounds:
            current = _settle(current)
            return current, {
                "rounds": rounds,
                "removed_counts": removed_counts,
                "settled_history": settled_history,
            }


def _simulate_grid(
    grid: np.ndarray, *, threshold: int | None = None, max_rounds: int | None = None
) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    level = _threshold_from_bar(g) if threshold is None else int(threshold)
    final_play, info = _simulate_playfield(g[:-2], level, max_rounds=max_rounds)
    out = g.copy()
    out[:-2] = final_play
    return out, info


def _stacks_to_settled_playfield(stacks: list[list[int]], height: int) -> np.ndarray:
    play = np.zeros((height, len(stacks)), dtype=int)
    for c, stack in enumerate(stacks):
        if stack:
            values = np.asarray(list(reversed(stack)), dtype=int)
            play[height - len(values) :, c] = values
    return play


def _embed_stacks(base_stacks: list[list[int]], total_width: int, left_pad: int) -> list[list[int]]:
    stacks = [[] for _ in range(total_width)]
    for idx, stack in enumerate(base_stacks):
        stacks[left_pad + idx] = list(stack)
    return stacks


def _sample_palette(ground_color: int, count: int) -> list[int]:
    pool = [color for color in PLAY_COLORS if color != int(ground_color)]
    return random.sample(pool, count)


def _render_input(stacks: list[list[int]], play_height: int, ground_color: int, threshold: int) -> np.ndarray:
    width = len(stacks)
    play = np.zeros((play_height, width), dtype=int)
    nonempty_cols = [c for c, stack in enumerate(stacks) if stack]
    assert nonempty_cols, "expected at least one active column"
    lifted_col = random.choice(nonempty_cols)

    for c, stack in enumerate(stacks):
        if not stack:
            continue
        k = len(stack)
        if c == lifted_col:
            rows = sorted(random.sample(range(play_height - 1), k))
        else:
            rows = sorted(random.sample(range(play_height), k))
        for r, color in zip(rows, reversed(stack)):
            play[r, c] = int(color)

    grid = np.zeros((play_height + 2, width), dtype=int)
    grid[:-2] = play
    grid[-2, :] = int(ground_color)
    grid[-1, :threshold] = GREY
    return grid


def _has_floating_pixel(playfield: np.ndarray) -> bool:
    play = np.asarray(playfield, dtype=int)
    return bool(np.any((play[:-1] != BLACK) & (play[1:] == BLACK)))


def _contains_run_of_exact_length(playfield: np.ndarray, length: int) -> bool:
    if length <= 0:
        return False
    play = np.asarray(playfield, dtype=int)
    for row in play:
        c = 0
        while c < row.shape[0]:
            color = int(row[c])
            if color == BLACK:
                c += 1
                continue
            c1 = c + 1
            while c1 < row.shape[0] and int(row[c1]) == color:
                c1 += 1
            if c1 - c == length:
                return True
            c = c1
    return False


def _threshold_changes_output(grid: np.ndarray, correct_output: np.ndarray) -> bool:
    width = int(grid.shape[1])
    level = _threshold_from_bar(grid)
    for alt in (level - 1, level + 1):
        if 1 <= alt < width:
            alt_out, _ = _simulate_grid(grid, threshold=alt)
            if not np.array_equal(alt_out, correct_output):
                return True
    return False


def _grid_sig(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode(), usedforsecurity=False).hexdigest()


def _make_gravity_only_case(threshold: int, ground_color: int) -> tuple[list[list[int]], dict]:
    a, b, c, d, e, f = _sample_palette(ground_color, 6)
    segment = threshold - 1
    base_width = 2 * threshold + 5
    start = 1
    stacks = [[] for _ in range(base_width)]

    top_cycle = (b, c, d)
    for i in range(segment):
        stacks[start + i] = [a]
        if random.random() < 0.8:
            stacks[start + i].append(top_cycle[i % len(top_cycle)])

    stacks[start + segment + 2] = [e]
    stacks[start + segment + 4] = [f, d]

    return stacks, {
        "family": "gravity_only",
        "threshold": threshold,
        "rounds": 0,
        "removed_counts": [],
        "span": segment,
    }


def _make_single_clear_case(threshold: int, ground_color: int, span: int, family: str) -> tuple[list[list[int]], dict]:
    a, b, c, d, e, f = _sample_palette(ground_color, 6)
    decoy_len = max(1, threshold - 1)
    base_width = span + decoy_len + 7
    start = 1
    stacks = [[] for _ in range(base_width)]

    top_cycle = (b, c, d)
    for i in range(span):
        stacks[start + i] = [a, top_cycle[i % len(top_cycle)]]

    decoy_start = start + span + 2
    for i in range(decoy_len):
        stacks[decoy_start + i] = [e]
    stacks[-2] = [f, c]

    return stacks, {
        "family": family,
        "threshold": threshold,
        "rounds": 1,
        "removed_counts": [span],
        "span": span,
    }


def _make_double_chain_case(threshold: int, ground_color: int) -> tuple[list[list[int]], dict]:
    a, b, c, d, e, f = _sample_palette(ground_color, 6)
    decoy_len = max(1, threshold - 1)
    base_width = 2 * threshold + 6
    start = 1
    stacks = [[] for _ in range(base_width)]

    stacks[start] = [a, b]
    for i in range(1, threshold):
        stacks[start + i] = [a, c, b]
    stacks[start + threshold] = [d, c]

    decoy_start = start + threshold + 2
    for i in range(decoy_len):
        stacks[decoy_start + i] = [e]
    stacks[-2] = [f, d]

    return stacks, {
        "family": "double_chain",
        "threshold": threshold,
        "rounds": 2,
        "removed_counts": [2 * threshold, threshold],
        "span": threshold,
    }


def _build_case(family: str) -> tuple[np.ndarray, dict]:
    if family == "gravity_only":
        threshold = random.randint(3, 5)
    else:
        threshold = random.randint(2, 5)

    ground_color = random.choice(PLAY_COLORS)

    if family == "gravity_only":
        base_stacks, meta = _make_gravity_only_case(threshold, ground_color)
    elif family == "single_exact":
        base_stacks, meta = _make_single_clear_case(threshold, ground_color, threshold, family)
    elif family == "single_long":
        span = threshold + random.randint(1, 2)
        base_stacks, meta = _make_single_clear_case(threshold, ground_color, span, family)
    else:
        base_stacks, meta = _make_double_chain_case(threshold, ground_color)

    max_stack = max(len(stack) for stack in base_stacks)
    play_height = random.randint(max(7, max_stack + 4), min(16, max_stack + 9))
    total_width = len(base_stacks) + random.randint(0, 4)
    left_pad = random.randint(0, total_width - len(base_stacks))
    stacks = _embed_stacks(base_stacks, total_width, left_pad)
    settled_play = _stacks_to_settled_playfield(stacks, play_height)
    meta["settled_play"] = settled_play
    meta["ground_color"] = int(ground_color)

    grid = _render_input(stacks, play_height, ground_color, threshold)
    return grid, meta


def _validate_generated_case(grid: np.ndarray, meta: dict) -> np.ndarray | None:
    g = np.asarray(grid, dtype=int)
    threshold = _threshold_from_bar(g)
    ground_color = int(g[-2, 0])
    assert np.all(g[-2] == ground_color), "ground row must be uniform"
    assert ground_color != GREY, "ground color must differ from grey scale bar"
    assert not np.any(g[:-2] == GREY), "grey is reserved for the scale bar"
    assert not np.any(g[:-2] == ground_color), "ground color must not appear in the playfield"
    assert np.array_equal(_settle(g[:-2]), meta["settled_play"]), "rendered input must preserve stack order"
    assert _has_floating_pixel(g[:-2]), "input should contain at least one suspended pixel"

    active_colors = set(map(int, np.unique(g[:-2]))) - {BLACK}
    assert len(active_colors) >= 3, "need multiple play colors to make the task readable"

    expected, info = _simulate_grid(g)
    actual = main(g)
    assert np.array_equal(actual, expected), "main must match the reference simulation"
    assert info["rounds"] == meta["rounds"], "generated case hit an unexpected number of rounds"
    assert info["removed_counts"] == meta["removed_counts"], "generated case removed unexpected runs"
    assert _threshold_changes_output(g, expected), "scale bar length must matter"

    if meta["family"] == "gravity_only":
        assert threshold >= 3
        assert _contains_run_of_exact_length(meta["settled_play"], threshold - 1)
    elif meta["family"] == "single_exact":
        assert meta["span"] == threshold
        one_round, _ = _simulate_grid(g, max_rounds=1)
        assert np.array_equal(one_round, expected)
    elif meta["family"] == "single_long":
        assert meta["span"] > threshold
        one_round, _ = _simulate_grid(g, max_rounds=1)
        assert np.array_equal(one_round, expected)
    else:
        assert meta["family"] == "double_chain"
        one_round, _ = _simulate_grid(g, max_rounds=1)
        assert not np.array_equal(one_round, expected), "chain cases must need a second round"

    out_sig = _grid_sig(expected)
    if out_sig in _RECENT_OUTPUT_SIGS:
        return None
    _RECENT_OUTPUT_SIGS.append(out_sig)
    return g


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let suspended pixels fall and repeatedly remove same-color horizontal runs whose length meets the bottom scale-bar threshold until the scene stabilizes.

    Input structure:
    - The bottom two rows are a special frame: the second-last row is a full-width ground strip, and the last row has a left-aligned grey scale bar with black cells to its right.
    - Above the ground is a black playfield containing scattered colored pixels, possibly floating or stacked within the same column.
    - Playfield colors are distinct from the ground color and the grey scale-bar color.

    Rule:
    - Read the clear threshold L from the grey scale bar length.
    - Repeatedly settle every playfield column downward onto the ground or other pixels, then simultaneously erase every horizontal same-color run of length at least L.
    - Stop when a settle step creates no erasable run, and keep the ground row and scale bar unchanged.

    Key insight:
    - The puzzle is not a one-pass gravity task: the hidden parameter comes from the outer scale bar, and each erase can unlock a new fall that creates the next run.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 2, "expected a 2D ARC grid with a playfield and two control rows"
    out, _ = _simulate_grid(g)
    return out


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    preferred_family = TEMPLATE_FAMILIES[_GEN_COUNTER % len(TEMPLATE_FAMILIES)]
    _GEN_COUNTER += 1

    for attempt in range(MAX_GENERATION_ATTEMPTS):
        family = preferred_family if attempt < MAX_GENERATION_ATTEMPTS // 2 else random.choice(TEMPLATE_FAMILIES)
        try:
            grid, meta = _build_case(family)
            validated = _validate_generated_case(grid, meta)
            if validated is not None:
                return validated
        except AssertionError:
            continue

    raise RuntimeError("failed to generate a valid scale-chain ARC input")
