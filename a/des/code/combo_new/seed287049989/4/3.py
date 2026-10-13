from __future__ import annotations

import numpy as np


BG = 0
MAX_ATTEMPTS = 400
PLACEMENT_TRIES = 250
COLOR_POOL = tuple(range(1, 10))
NEIGHBOR8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))


DIR_SPECS: dict[str, dict[str, object]] = {
    "right": {
        "grow": (0, 1),
        "perp": (1, 0),
        "tails": ((-1, -1), (1, -1)),
        "axis": 1,
    },
    "left": {
        "grow": (0, -1),
        "perp": (1, 0),
        "tails": ((-1, 1), (1, 1)),
        "axis": 1,
    },
    "up": {
        "grow": (-1, 0),
        "perp": (0, 1),
        "tails": ((1, -1), (1, 1)),
        "axis": 0,
    },
    "down": {
        "grow": (1, 0),
        "perp": (0, 1),
        "tails": ((-1, -1), (-1, 1)),
        "axis": 0,
    },
}


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return int(a[0] + b[0]), int(a[1] + b[1])


def _scale(v: tuple[int, int], k: int) -> tuple[int, int]:
    return int(v[0] * k), int(v[1] * k)


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]


def _arrow_candidates(cells: set[tuple[int, int]]) -> list[tuple[tuple[int, int], str, set[tuple[int, int]]]]:
    candidates: list[tuple[tuple[int, int], str, set[tuple[int, int]]]] = []
    for head in sorted(cells):
        for direction, spec in DIR_SPECS.items():
            tails = tuple(spec["tails"])
            arrow = {head}
            for tail in tails:
                arrow.add(_add(head, tail))
            if arrow <= cells:
                candidates.append((head, direction, arrow))
    return candidates


def _find_arrow(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], str, set[tuple[int, int]]]:
    candidates = _arrow_candidates(cells)
    assert len(candidates) == 1, f"expected one arrow, got {len(candidates)}"
    return candidates[0]


def _projection_distance(head: tuple[int, int], seed: tuple[int, int], direction: str) -> int:
    axis = int(DIR_SPECS[direction]["axis"])
    return abs(int(seed[axis]) - int(head[axis]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed into a straight ray whose direction comes from its matching arrow and whose length comes from their projected separation.

    Input structure:
    - The background is black.
    - Each non-black color forms one 3-pixel arrow plus one or more isolated seed pixels of the same color.
    - Every arrow is a V-shaped 3-pixel glyph whose head is the forward tip and whose two tail pixels sit one step diagonally behind it, so the arrow points up, down, left, or right.

    Rule:
    - Pair each seed with the unique arrow of the same color.
    - The arrow orientation gives the direction in which that seed grows.
    - The growth amount is the arrow-head-to-seed distance measured only on the arrow axis: use the absolute column difference for left/right arrows and the absolute row difference for up/down arrows.
    - Leave the seed in place and paint exactly that many extra same-color pixels continuing from the seed in the arrow direction; keep the arrow unchanged.

    Key insight:
    The arrow encodes two independent pieces of information: its shape gives a direction, while its projected offset from the same-color seed supplies the hidden numeric length.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    for color in colors:
        cells = {(int(r), int(c)) for r, c in np.argwhere(g == color)}
        head, direction, arrow = _find_arrow(cells)
        seeds = sorted(cells - arrow)
        assert seeds, "expected at least one seed for each arrow color"

        dr, dc = tuple(DIR_SPECS[direction]["grow"])
        for seed in seeds:
            length = _projection_distance(head, seed, direction)
            for step in range(1, length + 1):
                rr = int(seed[0] + dr * step)
                cc = int(seed[1] + dc * step)
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = color

    return out


def _relative_layout(direction: str, length: int, side: int, orth: int) -> dict[str, object]:
    spec = DIR_SPECS[direction]
    grow = tuple(spec["grow"])
    perp = tuple(spec["perp"])

    arrow = {(0, 0)}
    for tail in tuple(spec["tails"]):
        arrow.add((int(tail[0]), int(tail[1])))

    seed = _add(_scale(perp, orth), _scale(grow, side * length))
    growth = [_add(seed, _scale(grow, step)) for step in range(1, length + 1)]
    reserve = set(arrow)
    reserve.add(seed)
    reserve.update(growth)
    return {"arrow": arrow, "seed": seed, "growth": growth, "reserve": reserve}


def _translate_cells(
    cells: set[tuple[int, int]] | list[tuple[int, int]],
    origin: tuple[int, int],
) -> list[tuple[int, int]]:
    return [(_add(origin, cell)) for cell in cells]


def _seed_is_isolated(seed: tuple[int, int], arrow: list[tuple[int, int]]) -> bool:
    for cell in arrow:
        if max(abs(seed[0] - cell[0]), abs(seed[1] - cell[1])) <= 1:
            return False
    return True


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr, dc in NEIGHBOR8:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h and 0 <= cc < w:
                blocked[rr, cc] = True


def _direction_sequence(count: int, rng: np.random.Generator) -> list[str]:
    base = list(DIR_SPECS)
    rng.shuffle(base)
    chosen = base[: min(count, 4)]
    while len(chosen) < count:
        chosen.append(base[int(rng.integers(0, 4))])
    rng.shuffle(chosen)
    return chosen


def _mode_sequence(count: int, rng: np.random.Generator) -> list[str]:
    modes = ["ahead_axis", "ahead_offset"]
    if count >= 3:
        modes.append("behind_offset")
    all_modes = ("ahead_axis", "ahead_offset", "behind_offset")
    while len(modes) < count:
        modes.append(all_modes[int(rng.integers(0, len(all_modes)))])
    rng.shuffle(modes)
    return modes


def _sample_params(mode: str, rng: np.random.Generator) -> tuple[int, int, int]:
    length = int(rng.integers(2, 8))
    if mode == "ahead_axis":
        return length, 1, 0
    if mode == "ahead_offset":
        orth = int(rng.choice(np.array([-3, -2, -1, 1, 2, 3], dtype=int)))
        return length, 1, orth
    orth = int(rng.choice(np.array([-3, 3], dtype=int)))
    return length, -1, orth


def _place_one(
    shape: tuple[int, int],
    blocked: np.ndarray,
    color: int,
    direction: str,
    mode: str,
    rng: np.random.Generator,
) -> dict[str, object] | None:
    h, w = shape
    for _ in range(PLACEMENT_TRIES):
        length, side, orth = _sample_params(mode, rng)
        rel = _relative_layout(direction, length, side, orth)
        reserve = list(rel["reserve"])
        min_r = min(r for r, _ in reserve)
        max_r = max(r for r, _ in reserve)
        min_c = min(c for _, c in reserve)
        max_c = max(c for _, c in reserve)

        row_lo = -min_r
        row_hi = h - 1 - max_r
        col_lo = -min_c
        col_hi = w - 1 - max_c
        if row_lo > row_hi or col_lo > col_hi:
            continue

        head = (
            int(rng.integers(row_lo, row_hi + 1)),
            int(rng.integers(col_lo, col_hi + 1)),
        )
        arrow_cells = _translate_cells(set(rel["arrow"]), head)
        seed = _add(head, tuple(rel["seed"]))
        reserve_cells = _translate_cells(set(rel["reserve"]), head)
        growth_cells = _translate_cells(list(rel["growth"]), head)

        if not _seed_is_isolated(seed, arrow_cells):
            continue
        if any(blocked[r, c] for r, c in reserve_cells):
            continue
        if not all(_in_bounds(shape, cell) for cell in reserve_cells):
            continue

        return {
            "color": color,
            "direction": direction,
            "head": head,
            "arrow": arrow_cells,
            "seed": seed,
            "growth": growth_cells,
            "length": length,
            "mode": mode,
        }
    return None


def _expected_from_objects(grid: np.ndarray, objects: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for obj in objects:
        color = int(obj["color"])
        for r, c in obj["growth"]:
            out[r, c] = color
    return out


def _build_scene(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    count = int(rng.integers(2, 6))
    min_dim = 12 + 2 * count
    h = int(rng.integers(min_dim, 31))
    w = int(rng.integers(min_dim, 31))

    colors = [int(v) for v in rng.choice(np.array(COLOR_POOL, dtype=int), size=count, replace=False)]
    directions = _direction_sequence(count, rng)
    modes = _mode_sequence(count, rng)

    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []
    for color, direction, mode in zip(colors, directions, modes):
        obj = _place_one((h, w), blocked, color, direction, mode, rng)
        if obj is None:
            return None
        for r, c in obj["arrow"]:
            grid[r, c] = color
        sr, sc = tuple(obj["seed"])
        grid[sr, sc] = color
        _mark_blocked(blocked, obj["arrow"] + [tuple(obj["seed"])] + obj["growth"])
        objects.append(obj)

    expected = _expected_from_objects(grid, objects)
    if np.array_equal(grid, expected):
        return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    # Ensure every color remained uniquely decodable as one arrow plus one seed.
    for obj in objects:
        color = int(obj["color"])
        if int(np.sum(grid == color)) != 4:
            return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        built = _build_scene(rng)
        if built is None:
            continue
        grid, _expected = built
        return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
