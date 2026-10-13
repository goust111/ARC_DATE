import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
QUADRANTS = ("tl", "tr", "bl", "br")

MIN_SIDE = 13
MAX_SIDE = 29
MIN_COLORS = 2
MAX_COLORS = 4
MAX_TOTAL_SEEDS = 6
MAX_PER_COLOR = 4
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _shape_cells(local_r: int, local_c: int, length: int, side: int) -> set[tuple[int, int]]:
    cells = {(local_r, local_c)}
    for step in range(1, length + 1):
        if local_c + step < side:
            cells.add((local_r, local_c + step))
        if local_r + step < side:
            cells.add((local_r + step, local_c))
    return cells


def _actual_position(mid: int, quadrant: str, local_r: int, local_c: int) -> tuple[int, int]:
    if quadrant == "tl":
        return mid - 1 - local_r, mid - 1 - local_c
    if quadrant == "tr":
        return mid - 1 - local_r, mid + 1 + local_c
    if quadrant == "bl":
        return mid + 1 + local_r, mid - 1 - local_c
    if quadrant == "br":
        return mid + 1 + local_r, mid + 1 + local_c
    raise ValueError(f"unknown quadrant: {quadrant}")


def _direction_from_quadrant(quadrant: str) -> tuple[int, int]:
    if quadrant == "tl":
        return -1, -1
    if quadrant == "tr":
        return -1, 1
    if quadrant == "bl":
        return 1, -1
    if quadrant == "br":
        return 1, 1
    raise ValueError(f"unknown quadrant: {quadrant}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow an L-shaped branch from each seed, using its quadrant for direction and its color frequency for length.

    Input structure:
    - The grid is square with a Black background.
    - A Grey center row and Grey center column form a cross that splits the canvas into four quadrants.
    - Several non-Grey, non-Black single-pixel colored seeds lie in the quadrants and never on the Grey cross.

    Rule:
    - Count the total number K of seeds of each color across the whole input.
    - For every seed, draw K pixels horizontally away from the center and K pixels vertically away from the center, excluding the seed itself.
    - Top-left seeds grow left and up; top-right seeds grow right and up; bottom-left seeds grow left and down; bottom-right seeds grow right and down.
    - Branch pixels use the seed color, clip at the grid boundary, and never replace the Grey cross.

    Key insight:
    - The branch direction is determined locally by the seed's quadrant, while the branch length comes from a global same-color seed count shared across all quadrants.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"
    side = int(g.shape[0])
    assert side % 2 == 1, "expected an odd side length so the cross has a single center"

    mid = side // 2
    bg = _most_common_color(g)
    cross = int(g[mid, mid])
    assert bg != cross, "background and cross colors must differ"
    assert np.all(g[mid, :] == cross) and np.all(g[:, mid] == cross), "expected a solid center cross"

    seed_mask = (g != bg) & (g != cross)
    counts = Counter(map(int, g[seed_mask].reshape(-1)))

    out = g.copy()
    for r, c in np.argwhere(seed_mask):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        length = int(counts[color])
        assert r != mid and c != mid, "seeds may not lie on the center cross"

        dr = -1 if r < mid else 1
        dc = -1 if c < mid else 1
        for step in range(1, length + 1):
            rr = r + dr * step
            cc = c + dc * step
            if 0 <= rr < side and out[rr, c] != cross:
                out[rr, c] = color
            if 0 <= cc < side and out[r, cc] != cross:
                out[r, cc] = color

    out[mid, :] = cross
    out[:, mid] = cross
    return out


def _sample_color_counts(n_colors: int) -> list[int] | None:
    if n_colors > MAX_TOTAL_SEEDS:
        return None
    total = random.randint(n_colors + 1, MAX_TOTAL_SEEDS)
    counts = [1] * n_colors
    for _ in range(total - n_colors):
        candidates = [idx for idx, count in enumerate(counts) if count < MAX_PER_COLOR]
        if not candidates:
            return None
        counts[random.choice(candidates)] += 1
    random.shuffle(counts)
    return counts


def _seed_specs(colors: list[int], counts: list[int], want_truncation: bool) -> list[dict] | None:
    specs: list[dict] = []
    seed_id = 0
    for color, count in zip(colors, counts):
        for quadrant in random.sample(QUADRANTS, count):
            specs.append(
                {
                    "id": seed_id,
                    "color": int(color),
                    "length": int(count),
                    "quadrant": quadrant,
                    "truncate": False,
                }
            )
            seed_id += 1

    if want_truncation:
        trunc_candidates = [spec for spec in specs if int(spec["length"]) >= 2]
        if not trunc_candidates:
            return None
        random.choice(trunc_candidates)["truncate"] = True

    return specs


def _candidate_positions(length: int, side: int, must_truncate: bool) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for local_r in range(side):
        for local_c in range(side):
            if must_truncate:
                if local_r >= side - 1 or local_c >= side - 1:
                    continue
                if local_r + length < side and local_c + length < side:
                    continue
            else:
                if local_r + length >= side or local_c + length >= side:
                    continue
            positions.append((local_r, local_c))
    random.shuffle(positions)
    return positions


def _place_quadrant(items: list[dict], side: int) -> dict[int, tuple[int, int]] | None:
    if not items:
        return {}

    placements: dict[int, tuple[int, int]] = {}
    occupied: set[tuple[int, int]] = set()
    order = sorted(items, key=lambda item: (bool(item["truncate"]), int(item["length"])), reverse=True)

    def backtrack(index: int) -> bool:
        if index == len(order):
            return True

        item = order[index]
        for local_r, local_c in _candidate_positions(int(item["length"]), side, bool(item["truncate"])):
            cells = _shape_cells(local_r, local_c, int(item["length"]), side)
            if occupied & cells:
                continue
            placements[int(item["id"])] = (local_r, local_c)
            occupied.update(cells)
            if backtrack(index + 1):
                return True
            del placements[int(item["id"])]
            occupied.difference_update(cells)
        return False

    return placements if backtrack(0) else None


def _spec_positions(specs: list[dict], local_positions: dict[int, tuple[int, int]], mid: int) -> dict[int, tuple[int, int]]:
    positions: dict[int, tuple[int, int]] = {}
    for spec in specs:
        local_r, local_c = local_positions[int(spec["id"])]
        positions[int(spec["id"])] = _actual_position(mid, str(spec["quadrant"]), local_r, local_c)
    return positions


def _oracle(side: int, specs: list[dict], positions: dict[int, tuple[int, int]]) -> np.ndarray:
    mid = side // 2
    out = np.full((side, side), BG, dtype=int)
    out[mid, :] = GREY
    out[:, mid] = GREY

    for spec in specs:
        r, c = positions[int(spec["id"])]
        out[r, c] = int(spec["color"])

    for spec in specs:
        r, c = positions[int(spec["id"])]
        dr, dc = _direction_from_quadrant(str(spec["quadrant"]))
        for step in range(1, int(spec["length"]) + 1):
            rr = r + dr * step
            cc = c + dc * step
            if 0 <= rr < side and out[rr, c] != GREY:
                out[rr, c] = int(spec["color"])
            if 0 <= cc < side and out[r, cc] != GREY:
                out[r, cc] = int(spec["color"])

    return out


def _build_input(side: int, specs: list[dict], positions: dict[int, tuple[int, int]]) -> np.ndarray:
    mid = side // 2
    grid = np.full((side, side), BG, dtype=int)
    grid[mid, :] = GREY
    grid[:, mid] = GREY
    for spec in specs:
        r, c = positions[int(spec["id"])]
        grid[r, c] = int(spec["color"])
    return grid


def _is_clipped(spec: dict, positions: dict[int, tuple[int, int]], side: int) -> bool:
    r, c = positions[int(spec["id"])]
    dr, dc = _direction_from_quadrant(str(spec["quadrant"]))
    length = int(spec["length"])
    return not (0 <= r + dr * length < side and 0 <= c + dc * length < side)


def _gen_attempt() -> np.ndarray | None:
    side = random.randrange(MIN_SIDE, MAX_SIDE + 1, 2)
    half = side // 2

    n_colors = random.randint(MIN_COLORS, MAX_COLORS)
    counts = _sample_color_counts(n_colors)
    if counts is None:
        return None

    colors = random.sample(PALETTE, n_colors)
    want_truncation = random.random() < 0.5
    specs = _seed_specs(colors, counts, want_truncation)
    if specs is None:
        return None

    local_positions: dict[int, tuple[int, int]] = {}
    for quadrant in QUADRANTS:
        quadrant_items = [spec for spec in specs if spec["quadrant"] == quadrant]
        placed = _place_quadrant(quadrant_items, half)
        if placed is None:
            return None
        local_positions.update(placed)

    actual_positions = _spec_positions(specs, local_positions, side // 2)
    grid = _build_input(side, specs, actual_positions)
    expected = _oracle(side, specs, actual_positions)

    if want_truncation:
        truncated = [spec for spec in specs if bool(spec["truncate"])]
        if len(truncated) != 1 or not _is_clipped(truncated[0], actual_positions, side):
            return None

    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
