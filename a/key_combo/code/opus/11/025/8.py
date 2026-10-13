from __future__ import annotations

from itertools import combinations
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
TEAL = 8

MACRO_ROWS = 3
MACRO_COLS = 3
CELL_SIZE = 2
MACRO_SHAPE = (MACRO_ROWS * CELL_SIZE, MACRO_COLS * CELL_SIZE)

GEN_ATTEMPTS = 500
TOP_PAD_RANGE = (0, 2)
BOTTOM_PAD_RANGE = (0, 2)
LEFT_PAD_RANGE = (0, 2)
RIGHT_PAD_RANGE = (0, 3)

MID_ROW_CELLS = tuple((1, c) for c in range(3))
OUTER_ROW_CELLS = tuple((r, c) for r in (0, 2) for c in range(3))
ALL_MACRO_CELLS = tuple((r, c) for r in range(3) for c in range(3))


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(g: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return g[r0 : r1 + 1, c0 : c1 + 1]


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert len(sep_cols) == 1, "expected exactly one full-height grey separator"
    sep = int(sep_cols[0])
    assert sep >= MACRO_SHAPE[1] and sep + 2 < g.shape[1]

    right = g[:, sep + 1 :]
    comps = _components(right != BLACK)
    assert len(comps) == 2, "expected exactly two control sprites on the right"

    upper = lower = None
    upper_box = lower_box = None
    for comp in comps:
        box = _bbox(comp)
        patch = _crop(right, box)
        assert patch.shape == (2, 2), "control sprites must be 2x2"
        assert np.all(patch != BLACK), "control sprites must be fully filled"
        colors = set(map(int, np.unique(patch)))
        if colors == {BLUE, RED}:
            assert upper is None, "duplicate blue/red sprite"
            upper = patch.copy()
            upper_box = box
        elif colors == {GREEN, RED}:
            assert lower is None, "duplicate green/red sprite"
            lower = patch.copy()
            lower_box = box
        else:
            raise AssertionError(f"unexpected control colors: {colors}")

    assert upper is not None and lower is not None
    assert upper_box is not None and lower_box is not None
    assert upper_box[1] == lower_box[1] and upper_box[3] == lower_box[3], "controls must share columns"
    assert upper_box[0] < lower_box[0], "blue/red control must be above green/red control"
    macro_top = int(upper_box[0])
    macro_bottom = int(lower_box[2])
    assert macro_bottom - macro_top + 1 == MACRO_SHAPE[0], "controls must span the 6-row macro field"

    macro = g[macro_top : macro_bottom + 1, sep - MACRO_SHAPE[1] : sep]
    assert macro.shape == MACRO_SHAPE
    return macro, upper, lower


def _macro_selector(macro: np.ndarray) -> np.ndarray:
    selector = np.zeros((MACRO_ROWS, MACRO_COLS), dtype=bool)
    for mr in range(MACRO_ROWS):
        for mc in range(MACRO_COLS):
            block = macro[2 * mr : 2 * mr + 2, 2 * mc : 2 * mc + 2]
            colors = set(map(int, np.unique(block)))
            assert len(colors) == 1, "every macro-cell must be uniform"
            value = next(iter(colors))
            assert value in {BLACK, TEAL}, "macro-cells must be black or teal"
            selector[mr, mc] = value == TEAL
    return selector


def _decode(selector: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    decoded = np.zeros(MACRO_SHAPE, dtype=int)
    for mr in range(MACRO_ROWS):
        sprite = upper if mr % 2 == 1 else lower
        for mc in range(MACRO_COLS):
            if selector[mr, mc]:
                decoded[2 * mr : 2 * mr + 2, 2 * mc : 2 * mc + 2] = sprite
    return decoded


def _counts_and_first_cols(decoded: np.ndarray) -> tuple[dict[int, int], dict[int, int]]:
    counts = {color: int(np.count_nonzero(decoded == color)) for color in (BLUE, RED, GREEN)}
    assert all(counts[color] > 0 for color in counts), "all three colors must appear after decoding"
    first_cols = {
        color: int(np.argwhere(decoded == color)[:, 1].min())
        for color in (BLUE, RED, GREEN)
    }
    return counts, first_cols


def _render_output_from_decoded(decoded: np.ndarray) -> np.ndarray:
    counts, first_cols = _counts_and_first_cols(decoded)
    order = sorted((BLUE, RED, GREEN), key=lambda color: (-counts[color], first_cols[color]))
    height = sum(counts.values())
    out = np.zeros((height, 3), dtype=int)
    for col, color in enumerate(order):
        count = counts[color]
        out[height - count :, col] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the teal macro-grid through parity-selected control sprites, then stack color counts as sorted histogram bars.

    Input structure:
    - A black-background grid is split by one full-height grey column.
    - Immediately left of the separator is a 3x3 macro-grid, where each macro-cell is a uniform 2x2 block of black or teal.
    - On the right are two separated 2x2 control sprites: an upper blue/red sprite and a lower green/red sprite.

    Rule:
    - Expand the 3x3 macro-grid into a 6x6 decoded image: each teal macro-cell becomes the upper control sprite on odd macro rows and the lower control sprite on even macro rows, while black macro-cells become 2x2 black blocks.
    - Count the decoded blue, red, and green pixels, ignoring black.
    - Sort the three colors from most frequent to least frequent; if two counts tie, the color whose decoded bounding box starts farther left comes first.
    - Output a height-by-3 histogram where each sorted color fills one column upward from the bottom for exactly its count.

    Key insight:
    - The teal macro-grid does not itself reveal the final frequencies: row parity decides which control sprite each active macro-cell expands into, so decoding must happen before counting and sorting.
    """
    macro, upper, lower = _parse_input(grid)
    selector = _macro_selector(macro)
    decoded = _decode(selector, upper, lower)
    return _render_output_from_decoded(decoded)


def _sprite_library(other: int) -> dict[int, dict[str, list[np.ndarray]]]:
    return {
        1: {
            "red_left": [
                np.array([[RED, other], [RED, RED]], dtype=int),
                np.array([[RED, RED], [RED, other]], dtype=int),
            ],
            "any": [
                np.array([[RED, other], [RED, RED]], dtype=int),
                np.array([[RED, RED], [RED, other]], dtype=int),
            ],
        },
        2: {
            "other_left": [np.array([[other, RED], [other, RED]], dtype=int)],
            "red_left": [np.array([[RED, other], [RED, other]], dtype=int)],
            "any": [
                np.array([[other, RED], [other, RED]], dtype=int),
                np.array([[RED, other], [RED, other]], dtype=int),
                np.array([[other, other], [RED, RED]], dtype=int),
                np.array([[RED, RED], [other, other]], dtype=int),
                np.array([[other, RED], [RED, other]], dtype=int),
                np.array([[RED, other], [other, RED]], dtype=int),
            ],
        },
        3: {
            "other_left": [
                np.array([[other, RED], [other, other]], dtype=int),
                np.array([[other, other], [other, RED]], dtype=int),
            ],
            "any": [
                np.array([[other, RED], [other, other]], dtype=int),
                np.array([[other, other], [other, RED]], dtype=int),
            ],
        },
    }


def _sample_sprite(other: int, count: int, relation: str | None = None) -> np.ndarray:
    lib = _sprite_library(other)[int(count)]
    key = relation if relation is not None else "any"
    choices = lib[key]
    return random.choice(choices).copy()


def _select_subset(
    cells: tuple[tuple[int, int], ...],
    count: int,
    first_col: int | None = None,
) -> list[tuple[int, int]]:
    candidates = []
    for subset in combinations(cells, count):
        cols = [c for _r, c in subset]
        if first_col is not None and min(cols) != first_col:
            continue
        candidates.append(list(subset))
    assert candidates
    return random.choice(candidates)


def _sample_no_tie_config() -> tuple[int, int, int, int]:
    configs: list[tuple[int, int, int, int]] = []
    for a_mid in range(1, 4):
        for a_outer in range(1, 7):
            if a_mid + a_outer > 7:
                continue
            for blue_per in (1, 2, 3):
                for green_per in (1, 2, 3):
                    blue = a_mid * blue_per
                    green = a_outer * green_per
                    red = a_mid * (4 - blue_per) + a_outer * (4 - green_per)
                    if len({blue, red, green}) == 3:
                        configs.append((a_mid, a_outer, blue_per, green_per))
    return random.choice(configs)


def _build_instance(mode: str) -> tuple[np.ndarray, np.ndarray, dict]:
    top_pad = random.randint(*TOP_PAD_RANGE)
    bottom_pad = random.randint(*BOTTOM_PAD_RANGE)
    left_pad = random.randint(*LEFT_PAD_RANGE)
    right_pad = random.randint(*RIGHT_PAD_RANGE)

    if mode == "bg_tie_blue_first":
        a_mid, a_outer, blue_per, green_per = 2, 2, 2, 2
        mid_first, outer_first = 0, 1
        upper = _sample_sprite(BLUE, blue_per)
        lower = _sample_sprite(GREEN, green_per)
    elif mode == "bg_tie_green_first":
        a_mid, a_outer, blue_per, green_per = 2, 2, 2, 2
        mid_first, outer_first = 1, 0
        upper = _sample_sprite(BLUE, blue_per)
        lower = _sample_sprite(GREEN, green_per)
    elif mode == "br_tie":
        a_mid, a_outer, blue_per, green_per = 1, 2, 3, 3
        mid_first, outer_first = 0, None
        upper = _sample_sprite(BLUE, blue_per, relation="other_left")
        lower = _sample_sprite(GREEN, green_per, relation="other_left")
    elif mode == "gr_tie":
        a_mid, a_outer, blue_per, green_per = 2, 1, 3, 3
        mid_first, outer_first = None, 0
        upper = _sample_sprite(BLUE, blue_per, relation="other_left")
        lower = _sample_sprite(GREEN, green_per, relation="other_left")
    else:
        a_mid, a_outer, blue_per, green_per = _sample_no_tie_config()
        mid_first = random.choice([None, 0, 1]) if a_mid <= 2 else random.choice([None, 0])
        outer_first = random.choice([None, 0, 1, 2])
        upper_relation = None if blue_per != 2 else random.choice([None, "other_left", "red_left"])
        lower_relation = None if green_per != 2 else random.choice([None, "other_left", "red_left"])
        upper = _sample_sprite(BLUE, blue_per, relation=upper_relation)
        lower = _sample_sprite(GREEN, green_per, relation=lower_relation)

    mid_cells = _select_subset(MID_ROW_CELLS, a_mid, first_col=mid_first)
    outer_cells = _select_subset(OUTER_ROW_CELLS, a_outer, first_col=outer_first)
    selector = np.zeros((3, 3), dtype=bool)
    for r, c in mid_cells + outer_cells:
        selector[r, c] = True

    h = top_pad + 6 + bottom_pad
    w = left_pad + 6 + 1 + 2 + right_pad
    grid = np.zeros((h, w), dtype=int)
    sep = left_pad + 6
    grid[:, sep] = GREY

    macro_top = top_pad
    macro_left = left_pad
    for mr, mc in ALL_MACRO_CELLS:
        value = TEAL if selector[mr, mc] else BLACK
        grid[
            macro_top + 2 * mr : macro_top + 2 * mr + 2,
            macro_left + 2 * mc : macro_left + 2 * mc + 2,
        ] = value

    ctrl_left = sep + 1
    grid[top_pad : top_pad + 2, ctrl_left : ctrl_left + 2] = upper
    grid[top_pad + 4 : top_pad + 6, ctrl_left : ctrl_left + 2] = lower

    decoded = _decode(selector, upper, lower)
    expected = _render_output_from_decoded(decoded)
    meta = {
        "mode": mode,
        "selector": selector,
        "upper": upper,
        "lower": lower,
        "decoded": decoded,
    }
    return grid, expected, meta


def _output_order(decoded: np.ndarray) -> list[int]:
    counts, first_cols = _counts_and_first_cols(decoded)
    return sorted((BLUE, RED, GREEN), key=lambda color: (-counts[color], first_cols[color]))


def _validate_instance(grid: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    selector = np.asarray(meta["selector"], dtype=bool)
    upper = np.asarray(meta["upper"], dtype=int)
    lower = np.asarray(meta["lower"], dtype=int)
    decoded = np.asarray(meta["decoded"], dtype=int)
    counts, first_cols = _counts_and_first_cols(decoded)

    active_mid = int(selector[1].sum())
    active_outer = int(selector[0].sum() + selector[2].sum())
    if active_mid == 0 or active_outer == 0 or active_mid + active_outer > 7:
        return False

    decode_upper_all = _decode(selector, upper, upper)
    decode_lower_all = _decode(selector, lower, lower)
    alt_upper_counts = {color: int(np.count_nonzero(decode_upper_all == color)) for color in (BLUE, RED, GREEN)}
    alt_lower_counts = {color: int(np.count_nonzero(decode_lower_all == color)) for color in (BLUE, RED, GREEN)}
    if alt_upper_counts == counts or alt_lower_counts == counts:
        return False

    if len(set(counts.values())) == 3:
        order = _output_order(decoded)
        if order == [BLUE, RED, GREEN] and first_cols[BLUE] == 0 and first_cols[RED] == 0 and first_cols[GREEN] == 0:
            return False
    else:
        tied = [color for color, count in counts.items() if list(counts.values()).count(count) == 2]
        if len(tied) != 2 or first_cols[tied[0]] == first_cols[tied[1]]:
            return False

    if meta["mode"] in {"bg_tie_green_first", "gr_tie"}:
        color_id_order = sorted((BLUE, RED, GREEN), key=lambda color: (-counts[color], color))
        if color_id_order == _output_order(decoded):
            return False

    return True


def generate_input() -> np.ndarray:
    modes = [
        "no_tie",
        "no_tie",
        "bg_tie_blue_first",
        "bg_tie_green_first",
        "br_tie",
        "gr_tie",
    ]
    for _ in range(GEN_ATTEMPTS):
        mode = random.choice(modes)
        try:
            grid, expected, meta = _build_instance(mode)
        except AssertionError:
            continue
        if _validate_instance(grid, expected, meta):
            return grid
    raise RuntimeError("failed to generate a valid input")
