import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


BG = 0
YELLOW = 4
GRAY = 5
PAIR_COLOR_POOL = (1, 2, 3, 6, 7, 8, 9)

MIN_RADIUS = 3
MAX_RADIUS = 5
MAX_GENERATION_ATTEMPTS = 400
HEIGHT_RANGE = (24, 27)
WIDTH_RANGE = (15, 22)
SEPARATOR_RANGE = (5, 7)
TWO_PAIR_PROBABILITY = 0.7
SINGLE_PAIR_TWO_SEEDS_PROBABILITY = 0.55
TWO_PAIR_TIE_PROBABILITY = 0.45
TWO_PAIR_PREFER_TIE_PROBABILITY = 0.6


def _seed_radius(height: int, width: int, separator_row: int, row: int, col: int) -> int:
    return min(row - (separator_row + 1), height - 1 - row, col, width - 1 - col)


def _boxes_overlap(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
    gap: int = 0,
) -> bool:
    r0a, r1a, c0a, c1a = box1
    r0b, r1b, c0b, c1b = box2
    return not (
        r1a + gap < r0b or r1b + gap < r0a or c1a + gap < c0b or c1b + gap < c0a
    )


def _nearest_pair_index(seed_row: int, pair_rows: Sequence[int]) -> int:
    return min(range(len(pair_rows)), key=lambda idx: (abs(seed_row - pair_rows[idx]), pair_rows[idx]))


def _parse_instance(grid: np.ndarray) -> Tuple[int, List[Dict[str, int]], List[Dict[str, int]], Dict[Tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    height, width = g.shape

    separator_rows = [r for r in range(height) if np.all(g[r, :] == GRAY)]
    assert len(separator_rows) == 1, "expected exactly one full-width separator row"
    separator_row = separator_rows[0]

    pairs: List[Dict[str, int]] = []
    protected: Dict[Tuple[int, int], int] = {(separator_row, c): GRAY for c in range(width)}
    for r in range(height):
        if r == separator_row:
            continue
        left_marker = int(g[r, 0]) == YELLOW
        right_marker = int(g[r, width - 1]) == YELLOW
        assert left_marker == right_marker, "yellow edge markers must appear in opposite-edge pairs"
        if not left_marker:
            continue

        interior = [(c, int(g[r, c])) for c in range(1, width - 1) if int(g[r, c]) != BG]
        assert len(interior) == 2, "each marked row must contain exactly two interior non-background pixels"
        (c0, color_a), (c1, color_b) = interior
        assert c1 == c0 + 1, "the pair-defining pixels must be adjacent"
        assert color_a not in {BG, YELLOW, GRAY}
        assert color_b not in {BG, YELLOW, GRAY}
        assert color_a != color_b

        pairs.append({"row": r, "a": color_a, "b": color_b, "c0": c0, "c1": c1})
        protected[(r, 0)] = YELLOW
        protected[(r, width - 1)] = YELLOW
        protected[(r, c0)] = color_a
        protected[(r, c1)] = color_b

    pairs.sort(key=lambda item: item["row"])
    assert 1 <= len(pairs) <= 2, "expected one or two swap rows"

    pair_rows = {pair["row"] for pair in pairs}
    seeds: List[Dict[str, int]] = []
    for r in range(separator_row + 1, height):
        if r in pair_rows:
            for c in range(width):
                if (r, c) not in protected:
                    assert int(g[r, c]) == BG, "non-protected cells on a marked row must remain background"
            continue
        for c in range(width):
            value = int(g[r, c])
            if value != BG:
                assert value not in {YELLOW, GRAY}, "only seeds may appear below the separator"
                seeds.append({"row": r, "col": c, "color": value})

    assert 1 <= len(seeds) <= 2, "expected one or two seeds"

    seen_positions = set()
    pair_row_values = [pair["row"] for pair in pairs]
    for seed in seeds:
        pos = (seed["row"], seed["col"])
        assert pos not in seen_positions
        seen_positions.add(pos)

        idx = _nearest_pair_index(seed["row"], pair_row_values)
        pair = pairs[idx]
        assert seed["color"] == pair["a"], "a seed must match the A color of its associated pair"
        radius = _seed_radius(height, width, separator_row, seed["row"], seed["col"])
        assert radius >= MIN_RADIUS, "each seed must have room for at least three rings"
        seed["pair_index"] = idx
        seed["radius"] = radius
        seed["box"] = (seed["row"] - radius, seed["row"] + radius, seed["col"] - radius, seed["col"] + radius)

    if len(seeds) == 2:
        assert not _boxes_overlap(seeds[0]["box"], seeds[1]["box"]), "seed expansions must not overlap"

    return separator_row, pairs, seeds, protected


def _spec_solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    separator_row, pairs, seeds, protected = _parse_instance(g)
    out = g.copy()

    for seed in seeds:
        pair = pairs[seed["pair_index"]]
        row = seed["row"]
        col = seed["col"]
        radius = seed["radius"]
        for rr in range(row - radius, row + radius + 1):
            for cc in range(col - radius, col + radius + 1):
                distance = max(abs(rr - row), abs(cc - col))
                out[rr, cc] = pair["a"] if distance % 2 == 0 else pair["b"]

    for (r, c), value in protected.items():
        out[r, c] = value
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each seed into alternating Chebyshev rings using the nearest yellow-marked swap row as its implicit palette.

    Input structure:
    - A black background contains one or two special rows whose left and right edge cells are yellow.
    - Each yellow-marked row has exactly two adjacent interior pixels; the left one is color A and the right one is color B, defining a swap pair A <-> B.
    - A full-width gray separator row blocks painting into the upper band.
    - One or two seed pixels lie below the separator; each seed matches the A color of the swap row whose row index is closest to the seed.

    Rule:
    - Parse every yellow-marked row as an ordered pair A, B.
    - Associate each seed with the closest swap row by row distance, breaking ties toward the smaller row index.
    - Around each seed, paint every cell at Chebyshev distance d with A for even d and B for odd d, stopping at the largest radius that stays below the separator and inside the grid.
    - Keep the separator row and every yellow/pair-definition pixel unchanged.

    Key insight:
    - The pair rows do not show a literal palette strip; they only encode the two alternating colors that must be repeated outward as concentric square rings.
    """
    return _spec_solve(grid)


def _random_pair_columns(rng: random.Random, width: int, count: int) -> List[int]:
    cols = list(range(2, width - 3))
    rng.shuffle(cols)
    selected: List[int] = []
    for col in cols:
        if all(abs(col - other) >= 3 for other in selected):
            selected.append(col)
            if len(selected) == count:
                return selected
    raise AssertionError("failed to choose pair columns")


def _build_pair_specs(
    rng: random.Random,
    height: int,
    separator_row: int,
    pair_count: int,
    want_tie: bool = False,
) -> List[Tuple[int, int, int]]:
    if pair_count == 1:
        row = rng.randint(1, separator_row - 1)
        color_a, color_b = rng.sample(PAIR_COLOR_POOL, 2)
        return [(row, color_a, color_b)]

    upper_row = rng.randint(max(1, separator_row - 2), separator_row - 1)
    min_lower = max(separator_row + 8, 2 * (separator_row + MIN_RADIUS + 1) - upper_row + 1)
    max_lower = height - (MIN_RADIUS + 5)
    assert min_lower <= max_lower, "grid too short for a two-pair layout"
    lower_choices = list(range(min_lower, max_lower + 1))
    if want_tie:
        parity_choices = [row for row in lower_choices if (row - upper_row) % 2 == 0]
        if parity_choices:
            lower_choices = parity_choices
    lower_row = rng.choice(lower_choices)

    shared_a = rng.choice(PAIR_COLOR_POOL)
    b1, b2 = rng.sample([color for color in PAIR_COLOR_POOL if color != shared_a], 2)
    return [(upper_row, shared_a, b1), (lower_row, shared_a, b2)]


def _enumerate_candidates(
    height: int,
    width: int,
    separator_row: int,
    pair_rows: Sequence[int],
) -> Dict[str, List[Dict[str, int]]]:
    pair_rows = sorted(int(row) for row in pair_rows)
    lower_marked_rows = {row for row in pair_rows if row > separator_row}
    buckets: Dict[str, List[Dict[str, int]]] = {"single": [], "upper": [], "lower": [], "tie": []}

    for row in range(separator_row + MIN_RADIUS + 1, height - MIN_RADIUS):
        nearest = _nearest_pair_index(row, pair_rows)
        is_tie = len(pair_rows) == 2 and abs(row - pair_rows[0]) == abs(row - pair_rows[1])
        for col in range(MIN_RADIUS, width - MIN_RADIUS):
            radius = _seed_radius(height, width, separator_row, row, col)
            if not (MIN_RADIUS <= radius <= MAX_RADIUS):
                continue
            box = (row - radius, row + radius, col - radius, col + radius)
            if any(box[0] <= marked_row <= box[1] for marked_row in lower_marked_rows):
                continue

            candidate = {"row": row, "col": col, "radius": radius, "box": box}
            buckets["single"].append(candidate)
            if len(pair_rows) == 2:
                if is_tie:
                    buckets["tie"].append(candidate)
                elif nearest == 0:
                    buckets["upper"].append(candidate)
                else:
                    buckets["lower"].append(candidate)

    return buckets


def _pick_seed_set(
    rng: random.Random,
    candidate_sets: Dict[str, List[Dict[str, int]]],
    modes: Sequence[str],
) -> Optional[List[Dict[str, int]]]:
    pools: List[List[Dict[str, int]]] = []
    for mode in modes:
        pool = candidate_sets[mode][:]
        rng.shuffle(pool)
        pools.append(pool)

    picked: List[Dict[str, int]] = []

    def search(idx: int) -> bool:
        if idx == len(pools):
            return True
        for candidate in pools[idx]:
            if any(
                candidate["row"] == prior["row"] and candidate["col"] == prior["col"]
                for prior in picked
            ):
                continue
            if any(_boxes_overlap(candidate["box"], prior["box"], gap=1) for prior in picked):
                continue
            picked.append(candidate)
            if search(idx + 1):
                return True
            picked.pop()
        return False

    return picked if search(0) else None


def _validate_generated_grid(grid: np.ndarray) -> None:
    separator_row, pairs, seeds, _protected = _parse_instance(grid)
    assert all(pair["a"] != pair["b"] for pair in pairs)
    if len(pairs) == 2:
        used = {seed["pair_index"] for seed in seeds}
        assert used == {0, 1}, "two-pair instances must require both swap rows"
        tie_rows = [
            seed["row"]
            for seed in seeds
            if abs(seed["row"] - pairs[0]["row"]) == abs(seed["row"] - pairs[1]["row"])
        ]
        if tie_rows:
            assert pairs[0]["a"] == pairs[1]["a"], "tie examples must not be solvable from seed color alone"
    assert all(seed["row"] > separator_row for seed in seeds)


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    height = rng.randint(*HEIGHT_RANGE)
    width = rng.randint(*WIDTH_RANGE)
    separator_row = rng.randint(SEPARATOR_RANGE[0], min(SEPARATOR_RANGE[1], height - 12))
    pair_count = 2 if rng.random() < TWO_PAIR_PROBABILITY else 1
    want_tie = pair_count == 2 and rng.random() < TWO_PAIR_PREFER_TIE_PROBABILITY

    try:
        pair_specs = _build_pair_specs(rng, height, separator_row, pair_count, want_tie=want_tie)
    except AssertionError:
        return None

    pair_rows = [row for row, _a, _b in pair_specs]
    candidates = _enumerate_candidates(height, width, separator_row, pair_rows)

    if pair_count == 1:
        seed_modes = ["single"] if rng.random() >= SINGLE_PAIR_TWO_SEEDS_PROBABILITY else ["single", "single"]
        seeds = _pick_seed_set(rng, candidates, seed_modes)
        if seeds is None:
            return None
        labels = list(seed_modes)
    else:
        mode_options: List[Sequence[str]] = [["upper", "lower"]]
        if candidates["tie"]:
            mode_options.append(["tie", "lower"])
        if len(mode_options) == 2:
            tie_first = want_tie or rng.random() < TWO_PAIR_TIE_PROBABILITY
            if tie_first:
                mode_options = [mode_options[1], mode_options[0]]

        seeds = None
        chosen_modes: Sequence[str] | None = None
        for seed_modes in mode_options:
            seeds = _pick_seed_set(rng, candidates, seed_modes)
            if seeds is not None:
                chosen_modes = seed_modes
                break
        if seeds is None or chosen_modes is None:
            return None
        labels = list(chosen_modes)

    grid = np.full((height, width), BG, dtype=int)
    grid[separator_row, :] = GRAY

    pair_cols = _random_pair_columns(rng, width, len(pair_specs))
    for pair_col, (row, color_a, color_b) in zip(pair_cols, pair_specs):
        grid[row, 0] = YELLOW
        grid[row, width - 1] = YELLOW
        grid[row, pair_col] = color_a
        grid[row, pair_col + 1] = color_b

    sorted_specs = sorted(pair_specs, key=lambda item: item[0])
    sorted_rows = [row for row, _a, _b in sorted_specs]
    for seed, label in zip(seeds, labels):
        if label == "single":
            pair_idx = 0
        elif label == "tie":
            pair_idx = 0
            assert len(sorted_rows) == 2
            assert abs(seed["row"] - sorted_rows[0]) == abs(seed["row"] - sorted_rows[1])
        else:
            pair_idx = 0 if label == "upper" else 1
            assert _nearest_pair_index(seed["row"], sorted_rows) == pair_idx
        grid[seed["row"], seed["col"]] = sorted_specs[pair_idx][1]

    try:
        _validate_generated_grid(grid)
        expected = _spec_solve(grid)
        assert np.array_equal(main(grid), expected)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid alternating-ring swap-palette input")
