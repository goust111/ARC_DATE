import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import components_mask, get_bg
except Exception:  # pragma: no cover - fallback keeps the file self-contained
    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])

    def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
        mask = np.asarray(mask, dtype=bool)
        h, w = mask.shape
        seen = np.zeros_like(mask, dtype=bool)
        if connectivity == 8:
            steps = [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]
        else:
            steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        comps: list[np.ndarray] = []
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
                    for dr, dc in steps:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                comps.append(np.array(coords, dtype=int))
        return comps


PAYLOAD_SIZE = 6
BG = 0
PAYLOAD_COLORS = (1, 2, 3)
YELLOW = 4
GREY = 5
ORANGE = 7
COMB_COLORS = (YELLOW, ORANGE)
VALID_TOOTH_COUNTS = tuple(range(1, PAYLOAD_SIZE))
HEIGHT_RANGE = (10, 14)
WIDTH_RANGE = (12, 17)
MAX_ATTEMPTS = 400


def _normalize_coords(component) -> np.ndarray:
    return np.asarray(component, dtype=int)


def _payload_output(payload: np.ndarray, vertical_shift: int, horizontal_shift: int) -> np.ndarray:
    rolled = np.roll(payload, vertical_shift, axis=0)
    return np.roll(rolled, horizontal_shift, axis=1)


def _comb_tooth_count(grid: np.ndarray, color: int) -> int:
    mask = grid[:-1] == color
    comps = [_normalize_coords(comp) for comp in components_mask(mask, connectivity=4)]
    assert len(comps) == 1, f"expected exactly one comb of color {color}"
    coords = comps[0]
    rows = coords[:, 0]
    cols = coords[:, 1]

    unique_cols, counts = np.unique(cols, return_counts=True)
    stem_count = int(counts.max())
    stem_candidates = unique_cols[counts == stem_count]
    stem_c = int(stem_candidates.max())

    r0 = int(rows.min())
    r1 = int(rows.max())
    assert stem_count == r1 - r0 + 1, "comb stem must be a contiguous vertical column"
    assert np.all((cols == stem_c) | (cols == stem_c - 1)), "teeth must be single left protrusions"
    assert np.all(cols <= stem_c), "comb teeth must point left"

    teeth = 0
    for r in range(r0, r1 + 1):
        row_cols = sorted(int(c) for c in cols[rows == r])
        assert row_cols[-1] == stem_c, "each occupied row must include the stem"
        assert len(row_cols) in (1, 2), "comb rows must contain stem only or stem plus one tooth"
        if len(row_cols) == 2:
            assert row_cols[0] == stem_c - 1, "each tooth must extend exactly one pixel left"
            teeth += 1
    assert teeth in VALID_TOOTH_COUNTS, "decoded tooth count must fit the payload size"
    return teeth


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode two comb tooth counts and use the bottom color order to assign them to the payload's roll axes.

    Input structure:
    - The upper-left 6x6 block is the payload, drawn with blue, red, and green pixels on black background.
    - Above the bottom band, the grid contains exactly one yellow comb and one orange comb; each is a 1-pixel vertical stem with some rows having a 1-pixel tooth extending left.
    - The bottom row is a full-width grey sideband with exactly one yellow pixel and one orange pixel marking a left-to-right order.

    Rule:
    - Count the yellow comb's teeth and the orange comb's teeth.
    - Scan the bottom sideband from left to right: the earlier color gives the downward roll amount, and the later color gives the rightward roll amount.
    - Apply those two cyclic shifts to the 6x6 payload, then output only the shifted payload.

    Key insight:
    - The combs encode the two magnitudes, but the sideband order decides which magnitude belongs to the vertical axis and which belongs to the horizontal axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    h, w = g.shape
    assert h >= PAYLOAD_SIZE + 2 and w >= PAYLOAD_SIZE + 6, "grid is too small for the described layout"

    payload = g[:PAYLOAD_SIZE, :PAYLOAD_SIZE]
    payload_colors = set(map(int, np.unique(payload)))
    assert payload_colors <= {BG, *PAYLOAD_COLORS}, "payload must use only black/red/blue/green"

    band = g[-1]
    band_non_grey = [(c, int(v)) for c, v in enumerate(band) if int(v) != GREY]
    assert len(band_non_grey) == 2, "bottom sideband must contain exactly two colored markers"
    assert {color for _, color in band_non_grey} == set(COMB_COLORS), "bottom sideband must use yellow and orange"
    assert np.all((band == GREY) | (band == YELLOW) | (band == ORANGE)), "bottom sideband may only use grey/yellow/orange"

    yellow_count = _comb_tooth_count(g, YELLOW)
    orange_count = _comb_tooth_count(g, ORANGE)
    assert yellow_count != orange_count, "the two combs must encode different roll amounts"

    first_color = band_non_grey[0][1]
    if first_color == YELLOW:
        vertical_shift, horizontal_shift = yellow_count, orange_count
    else:
        vertical_shift, horizontal_shift = orange_count, yellow_count

    return _payload_output(payload, vertical_shift, horizontal_shift)


def _choose_distinct_counts(rng: random.Random) -> tuple[int, int]:
    a, b = rng.sample(VALID_TOOTH_COUNTS, 2)
    return int(a), int(b)


def _shape_offsets() -> list[tuple[tuple[int, int], ...]]:
    return [
        ((0, 0),),
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 1)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (0, 2)),
    ]


_PATCH_SHAPES = _shape_offsets()


def _sample_payload(rng: random.Random, count_a: int, count_b: int) -> np.ndarray | None:
    for _ in range(200):
        payload = np.zeros((PAYLOAD_SIZE, PAYLOAD_SIZE), dtype=int)
        n_shapes = rng.randint(5, 8)
        placed = 0
        for _ in range(n_shapes):
            shape = rng.choice(_PATCH_SHAPES)
            color = rng.choice(PAYLOAD_COLORS)
            max_r = max(r for r, _ in shape)
            max_c = max(c for _, c in shape)
            for _ in range(30):
                r0 = rng.randint(0, PAYLOAD_SIZE - max_r - 1)
                c0 = rng.randint(0, PAYLOAD_SIZE - max_c - 1)
                cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
                if any(payload[r, c] != BG for r, c in cells):
                    continue
                for r, c in cells:
                    payload[r, c] = color
                placed += 1
                break

        if placed < 4:
            continue
        if set(map(int, np.unique(payload))) - {BG} != set(PAYLOAD_COLORS):
            continue
        if np.count_nonzero(payload) < 9:
            continue

        occupied_rows = np.where(np.any(payload != BG, axis=1))[0]
        occupied_cols = np.where(np.any(payload != BG, axis=0))[0]
        if len(occupied_rows) < 4 or len(occupied_cols) < 4:
            continue

        variants = {
            ("v", count_a): np.roll(payload, count_a, axis=0),
            ("v", count_b): np.roll(payload, count_b, axis=0),
            ("h", count_a): np.roll(payload, count_a, axis=1),
            ("h", count_b): np.roll(payload, count_b, axis=1),
        }
        if any(np.array_equal(arr, payload) for arr in variants.values()):
            continue

        out_ab = _payload_output(payload, count_a, count_b)
        out_ba = _payload_output(payload, count_b, count_a)
        if np.array_equal(out_ab, payload) or np.array_equal(out_ba, payload):
            continue
        if np.array_equal(out_ab, out_ba):
            continue
        if np.array_equal(out_ab, variants[("v", count_a)]) or np.array_equal(out_ab, variants[("h", count_b)]):
            continue
        if np.array_equal(out_ba, variants[("v", count_b)]) or np.array_equal(out_ba, variants[("h", count_a)]):
            continue
        return payload
    return None


def _place_comb(
    grid: np.ndarray,
    color: int,
    stem_c: int,
    top_r: int,
    stem_h: int,
    tooth_count: int,
    rng: random.Random,
) -> None:
    rows = list(range(top_r, top_r + stem_h))
    tooth_rows = set(rng.sample(rows, tooth_count))
    grid[top_r : top_r + stem_h, stem_c] = color
    for r in tooth_rows:
        grid[r, stem_c - 1] = color


def _band_positions(rng: random.Random, width: int, forbidden: set[int]) -> tuple[int, int] | None:
    choices = [c for c in range(width) if c not in forbidden]
    if len(choices) < 2:
        return None
    pairs = [(a, b) for i, a in enumerate(choices) for b in choices[i + 1 :] if b - a >= 2]
    if not pairs:
        return None
    return rng.choice(pairs)


def _validate_grid(
    grid: np.ndarray,
    payload: np.ndarray,
    yellow_count: int,
    orange_count: int,
    yellow_left: bool,
) -> bool:
    expected = _payload_output(
        payload,
        yellow_count if yellow_left else orange_count,
        orange_count if yellow_left else yellow_count,
    )
    swapped = _payload_output(
        payload,
        orange_count if yellow_left else yellow_count,
        yellow_count if yellow_left else orange_count,
    )
    if np.array_equal(expected, swapped):
        return False
    if np.array_equal(expected, payload):
        return False
    try:
        solved = main(grid)
    except Exception:
        return False
    return np.array_equal(solved, expected)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    yellow_count, orange_count = _choose_distinct_counts(rng)
    payload = _sample_payload(rng, yellow_count, orange_count)
    if payload is None:
        return None

    h = rng.randint(*HEIGHT_RANGE)
    w = rng.randint(*WIDTH_RANGE)
    grid = np.zeros((h, w), dtype=int)
    grid[:PAYLOAD_SIZE, :PAYLOAD_SIZE] = payload
    grid[-1, :] = GREY

    stem_choices = list(range(PAYLOAD_SIZE + 2, w))
    valid_pairs = [(a, b) for a in stem_choices for b in stem_choices if b - a >= 3]
    if not valid_pairs:
        return None
    left_stem, right_stem = rng.choice(valid_pairs)
    if rng.choice([False, True]):
        stem_y, stem_o = left_stem, right_stem
    else:
        stem_y, stem_o = right_stem, left_stem

    for color, stem_c, tooth_count in ((YELLOW, stem_y, yellow_count), (ORANGE, stem_o, orange_count)):
        max_h = min(8, h - 1)
        min_h = max(4, tooth_count + 1)
        if min_h > max_h:
            return None
        stem_h = rng.randint(min_h, max_h)
        top_r = rng.randint(0, h - 1 - stem_h)
        _place_comb(grid, color, stem_c, top_r, stem_h, tooth_count, rng)

    forbidden = set()
    for stem_c in (stem_y, stem_o):
        forbidden.update(range(stem_c - 2, stem_c + 2))
    band_pair = _band_positions(rng, w, forbidden)
    if band_pair is None:
        return None
    left_c, right_c = band_pair
    yellow_left = rng.choice([False, True])
    if yellow_left:
        grid[-1, left_c] = YELLOW
        grid[-1, right_c] = ORANGE
    else:
        grid[-1, left_c] = ORANGE
        grid[-1, right_c] = YELLOW

    if np.any(grid[:PAYLOAD_SIZE, PAYLOAD_SIZE : w] == GREY):
        return None
    if not _validate_grid(grid, payload, yellow_count, orange_count, yellow_left):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
