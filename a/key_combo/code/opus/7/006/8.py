from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import random

import numpy as np


INPUT_SHAPE = (5, 5)
CORE_BBOX = (1, 1, 3, 3)
OUTPUT_SHAPE = (15, 15)
ALLOWED_COLORS = (1, 2, 3, 4, 7)

# Generator-only canonical palette cycle. It makes the otherwise-discarded
# selector color recoverable inside the generated example family, while main()
# still follows the task rule for any valid input.
PALETTE_RING = (2, 3, 4, 7, 1)

CORE_COORDS = [(r, c) for r in range(3) for c in range(3)]
PAIR_CATEGORIES: dict[str, list[tuple[tuple[int, int], tuple[int, int]]]] = defaultdict(list)
for i, a in enumerate(CORE_COORDS):
    for b in CORE_COORDS[i + 1 :]:
        dr = abs(a[0] - b[0])
        dc = abs(a[1] - b[1])
        if dr + dc == 1:
            cat = "orth_adjacent"
        elif dr == 1 and dc == 1:
            cat = "diagonal_adjacent"
        elif (dr, dc) in {(2, 0), (0, 2)}:
            cat = "line_gap_two"
        elif (dr, dc) in {(2, 1), (1, 2)}:
            cat = "knight_like"
        else:
            cat = "opposite_corners"
        PAIR_CATEGORIES[cat].append((a, b))

_CATEGORY_USAGE = {cat: 0 for cat in PAIR_CATEGORIES}
_SEEN_INPUT_SIGS: set[str] = set()
_SEEN_OUTPUT_SIGS: set[str] = set()


def _sig(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(tuple(grid.shape)).encode()).hexdigest()


def _extract_core(grid: np.ndarray) -> np.ndarray:
    return np.asarray(grid[1:4, 1:4], dtype=int)


def _sorted_roles(core: np.ndarray) -> tuple[int, int, int]:
    counts = Counter(int(v) for v in core.flat if int(v) != 0)
    assert len(counts) == 3, "expected exactly three non-black colors in the 3x3 core"
    ranked = sorted(counts.items(), key=lambda item: item[1])
    freqs = [freq for _color, freq in ranked]
    assert freqs == [2, 3, 4], f"expected frequencies 2,3,4, got {freqs}"
    return int(ranked[0][0]), int(ranked[1][0]), int(ranked[2][0])


def _cascade_stamp(core: np.ndarray, selector: int, frame: int, bulk: int) -> np.ndarray:
    stamp = np.full(core.shape, bulk, dtype=int)
    stamp[core == selector] = frame
    return stamp


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    core = _extract_core(g)
    selector, frame, bulk = _sorted_roles(core)
    stamp = _cascade_stamp(core, selector, frame, bulk)
    out = np.zeros(OUTPUT_SHAPE, dtype=int)
    for r, c in np.argwhere(g == selector):
        out[r * 3 : (r + 1) * 3, c * 3 : (c + 1) * 3] = stamp
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the center 3x3 pattern into selector slots after a frequency-ranked color cascade.

    Input structure:
    - The input is a 5x5 grid with black background everywhere except the centered 3x3 area.
    - The centered 3x3 area uses exactly three non-black colors from a small palette.
    - Those three colors occur with distinct frequencies 2, 3, and 4.

    Rule:
    - Count the three non-black colors inside the centered 3x3 area and rank them by frequency:
      least frequent = selector S, middle = frame F, most frequent = bulk B.
    - Create a 15x15 black output interpreted as a 5x5 array of 3x3 block slots.
    - For every input pixel whose color is S, place one stamp in the corresponding output block slot.
    - The stamp is the centered 3x3 pattern recolored by the cascade S->F, F->B, B->B.

    Key insight:
    - The same frequency ranking controls both where stamps appear and how the copied 3x3 pattern is recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == INPUT_SHAPE, f"expected a 5x5 grid, got {g.shape}"
    assert np.all(g[[0, 4], :] == 0) and np.all(g[:, [0, 4]] == 0), "outer border must be black"

    core = _extract_core(g)
    assert np.all(core != 0), "the centered 3x3 area must be fully colored"
    selector, frame, bulk = _sorted_roles(core)

    out = np.zeros(OUTPUT_SHAPE, dtype=int)
    stamp = _cascade_stamp(core, selector, frame, bulk)
    selector_cells = np.argwhere(g == selector)
    assert selector_cells.shape[0] == 2, "selector color must occur exactly twice in the 5x5 input"

    for r, c in selector_cells:
        r0 = int(r) * 3
        c0 = int(c) * 3
        out[r0 : r0 + 3, c0 : c0 + 3] = stamp

    return out


def _pair_category(pair: tuple[tuple[int, int], tuple[int, int]]) -> str:
    a, b = pair
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    if dr + dc == 1:
        return "orth_adjacent"
    if dr == 1 and dc == 1:
        return "diagonal_adjacent"
    if (dr, dc) in {(2, 0), (0, 2)}:
        return "line_gap_two"
    if (dr, dc) in {(2, 1), (1, 2)}:
        return "knight_like"
    return "opposite_corners"


def _choose_selector_pair() -> tuple[tuple[int, int], tuple[int, int]]:
    least_used = min(_CATEGORY_USAGE.values())
    candidate_cats = [cat for cat, used in _CATEGORY_USAGE.items() if used == least_used]
    cat = random.choice(candidate_cats)
    _CATEGORY_USAGE[cat] += 1
    return random.choice(PAIR_CATEGORIES[cat])


def _nearest_selector_distance(pos: tuple[int, int], pair: tuple[tuple[int, int], tuple[int, int]]) -> int:
    return min(abs(pos[0] - s[0]) + abs(pos[1] - s[1]) for s in pair)


def _canonical_frame_cells(
    pair: tuple[tuple[int, int], tuple[int, int]]
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    selector_set = set(pair)
    remaining = [pos for pos in CORE_COORDS if pos not in selector_set]
    remaining.sort(
        key=lambda pos: (
            _nearest_selector_distance(pos, pair),
            pos[0],
            pos[1],
        )
    )
    return tuple(remaining[:3])


def _choose_role_colors() -> tuple[int, int, int]:
    i = random.randrange(len(PALETTE_RING))
    return (
        int(PALETTE_RING[i]),
        int(PALETTE_RING[(i + 1) % len(PALETTE_RING)]),
        int(PALETTE_RING[(i + 2) % len(PALETTE_RING)]),
    )


def _build_input(
    pair: tuple[tuple[int, int], tuple[int, int]],
    selector: int,
    frame: int,
    bulk: int,
) -> np.ndarray:
    grid = np.zeros(INPUT_SHAPE, dtype=int)
    frame_cells = set(_canonical_frame_cells(pair))
    selector_cells = set(pair)

    for r in range(3):
        for c in range(3):
            cell = (r, c)
            if cell in selector_cells:
                value = selector
            elif cell in frame_cells:
                value = frame
            else:
                value = bulk
            grid[r + 1, c + 1] = int(value)

    return grid


def _selector_swap_variant(grid: np.ndarray) -> np.ndarray:
    variant = np.array(grid, copy=True)
    core = _extract_core(variant)
    selector, frame, _bulk = _sorted_roles(core)
    selector_pos = tuple(map(int, np.argwhere(core == selector)[0]))
    frame_pos = tuple(map(int, np.argwhere(core == frame)[0]))
    sr, sc = selector_pos
    fr, fc = frame_pos
    variant[sr + 1, sc + 1], variant[fr + 1, fc + 1] = variant[fr + 1, fc + 1], variant[sr + 1, sc + 1]
    return variant


def _validate_generated_case(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.shape == INPUT_SHAPE
    assert np.all(grid[[0, 4], :] == 0) and np.all(grid[:, [0, 4]] == 0)

    core = _extract_core(grid)
    assert np.all(core != 0)
    assert set(int(v) for v in np.unique(core)).issubset(ALLOWED_COLORS)

    selector, frame, bulk = _sorted_roles(core)
    counts = Counter(int(v) for v in core.flat)
    assert counts[selector] == 2 and counts[frame] == 3 and counts[bulk] == 4

    selector_cells = tuple(sorted(tuple(map(int, pos)) for pos in np.argwhere(core == selector)))
    frame_cells = tuple(sorted(tuple(map(int, pos)) for pos in np.argwhere(core == frame)))
    assert frame_cells == tuple(sorted(_canonical_frame_cells(selector_cells)))

    expected = _expected_output(grid)
    actual = main(grid)
    assert np.array_equal(actual, expected), "main() must follow the stated rule"

    stamp_positions = np.argwhere(grid == selector)
    assert stamp_positions.shape == (2, 2)
    stamp = _cascade_stamp(core, selector, frame, bulk)
    frame_count = int(np.count_nonzero(stamp == frame))
    bulk_count = int(np.count_nonzero(stamp == bulk))
    assert frame_count == 2 and bulk_count == 7

    swapped = _selector_swap_variant(grid)
    assert not np.array_equal(main(swapped), actual), "selector placement must affect the output"


def generate_input() -> np.ndarray:
    for _ in range(500):
        pair = _choose_selector_pair()
        selector, frame, bulk = _choose_role_colors()
        grid = _build_input(pair, selector, frame, bulk)
        _validate_generated_case(grid)

        inp_sig = _sig(grid)
        out_sig = _sig(main(grid))
        if inp_sig in _SEEN_INPUT_SIGS or out_sig in _SEEN_OUTPUT_SIGS:
            continue

        _SEEN_INPUT_SIGS.add(inp_sig)
        _SEEN_OUTPUT_SIGS.add(out_sig)
        return grid

    raise RuntimeError("failed to generate a fresh valid input")
