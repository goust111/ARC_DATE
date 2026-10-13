from __future__ import annotations

from collections import Counter

import numpy as np


BLACK = 0
GRAY = 5
DOT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
GEN_ATTEMPTS = 200


def _detect_bar(grid: np.ndarray) -> str:
    h, w = grid.shape
    candidates: list[str] = []
    if np.all(grid[0, :] == GRAY):
        candidates.append("top")
    if np.all(grid[h - 1, :] == GRAY):
        candidates.append("bottom")
    if np.all(grid[:, 0] == GRAY):
        candidates.append("left")
    if np.all(grid[:, w - 1] == GRAY):
        candidates.append("right")
    assert len(candidates) == 1, "expected exactly one full gray border bar"
    return candidates[0]


def _distance_to_bar(side: str, r: int, c: int, shape: tuple[int, int]) -> int:
    h, w = shape
    if side == "top":
        return int(r)
    if side == "bottom":
        return int(h - 1 - r)
    if side == "left":
        return int(c)
    assert side == "right"
    return int(w - 1 - c)


def _step_toward_bar(side: str, r: int, c: int, steps: int) -> tuple[int, int]:
    if side == "top":
        return int(r - steps), int(c)
    if side == "bottom":
        return int(r + steps), int(c)
    if side == "left":
        return int(r), int(c - steps)
    assert side == "right"
    return int(r), int(c + steps)


def _bar_cell(side: str, r: int, c: int, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    if side == "top":
        return 0, int(c)
    if side == "bottom":
        return int(h - 1), int(c)
    if side == "left":
        return int(r), 0
    assert side == "right"
    return int(r), int(w - 1)


def _parse_dots(grid: np.ndarray) -> tuple[str, list[tuple[int, int, int]], Counter[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30

    side = _detect_bar(g)
    dots: list[tuple[int, int, int]] = []
    for r, c in np.argwhere((g != BLACK) & (g != GRAY)):
        dots.append((int(r), int(c), int(g[r, c])))

    assert dots, "expected at least one colored dot"
    counts: Counter[int] = Counter(color for _r, _c, color in dots)
    assert all(1 <= cnt <= 4 for cnt in counts.values()), "dot color frequencies must be in 1..4"

    survivors: set[tuple[int, int]] = set()
    notches: set[tuple[int, int]] = set()
    for r, c, color in dots:
        dist = _distance_to_bar(side, r, c, g.shape)
        steps = int(counts[color])
        assert dist >= steps + 1, "dots must not cross or land on the bar"
        nr, nc = _step_toward_bar(side, r, c, steps)
        remaining = _distance_to_bar(side, nr, nc, g.shape)
        if remaining == 1:
            notch = _bar_cell(side, nr, nc, g.shape)
            assert notch not in notches, "absorbed dots must not share a notch"
            notches.add(notch)
        else:
            assert remaining >= 2, "non-absorbed dots must stop before the bar"
            assert (nr, nc) not in survivors, "moved dots must not overlap"
            survivors.add((nr, nc))

    return side, dots, counts


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        side, dots, counts = _parse_dots(grid)
    except AssertionError:
        return False

    absorbed = 0
    survivors = 0
    by_color: dict[int, set[str]] = {}
    for r, c, color in dots:
        steps = int(counts[color])
        nr, nc = _step_toward_bar(side, r, c, steps)
        remaining = _distance_to_bar(side, nr, nc, grid.shape)
        label = "absorb" if remaining == 1 else "survive"
        by_color.setdefault(color, set()).add(label)
        if label == "absorb":
            absorbed += 1
        else:
            survivors += 1

    has_mixed_color = any(len(outcomes) == 2 for outcomes in by_color.values())
    return absorbed >= 1 and survivors >= 1 and has_mixed_color and len(set(counts.values())) >= 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored dot toward the edge bar by its global color frequency and absorb only the ones that end flush to the bar.

    Input structure:
    - The background is black.
    - Exactly one full gray bar occupies an entire border row or border column.
    - The remaining foreground cells are isolated single-pixel colored dots, and each dot color appears between one and four times.

    Rule:
    - Count how many dots of each color appear in the whole input.
    - Every dot moves straight toward the gray bar, perpendicular to it, by exactly that many cells.
    - If the moved dot ends one cell away from the bar, remove the dot and recolor the directly facing bar cell with the dot's color.
    - Otherwise keep the dot at its moved location, and leave every other bar cell gray.

    Key insight:
    - The hidden displacement is not geometric size or color id; it is the global frequency of the dot's own color, so equal-color dots can split into different outcomes depending on their starting distance.
    """
    g = np.asarray(grid, dtype=int)
    side, dots, counts = _parse_dots(g)

    out = g.copy()
    out[(out != BLACK) & (out != GRAY)] = BLACK

    for r, c, color in dots:
        steps = int(counts[color])
        nr, nc = _step_toward_bar(side, r, c, steps)
        remaining = _distance_to_bar(side, nr, nc, g.shape)
        if remaining == 1:
            br, bc = _bar_cell(side, nr, nc, g.shape)
            out[br, bc] = int(color)
        else:
            out[nr, nc] = int(color)

    return out


def _build_grid(
    side: str,
    dot_specs: list[tuple[int, int, bool, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    total = len(dot_specs)
    max_dist = max(dist for _color, _count, _absorb, dist in dot_specs)
    cross = int(total + rng.integers(3, 8))
    depth = int(max_dist + rng.integers(1, 4))

    if side in {"top", "bottom"}:
        h = depth + 1
        w = cross
        coords = [(int(c),) for c in rng.choice(np.arange(1, w - 1), size=total, replace=False)]
        grid = np.full((h, w), BLACK, dtype=int)
        if side == "top":
            grid[0, :] = GRAY
        else:
            grid[h - 1, :] = GRAY
        for (color, _count, _absorb, dist), (c,) in zip(dot_specs, coords):
            r = int(dist) if side == "top" else int(h - 1 - dist)
            grid[r, c] = int(color)
    else:
        h = cross
        w = depth + 1
        coords = [(int(r),) for r in rng.choice(np.arange(1, h - 1), size=total, replace=False)]
        grid = np.full((h, w), BLACK, dtype=int)
        if side == "left":
            grid[:, 0] = GRAY
        else:
            grid[:, w - 1] = GRAY
        for (color, _count, _absorb, dist), (r,) in zip(dot_specs, coords):
            c = int(dist) if side == "left" else int(w - 1 - dist)
            grid[r, c] = int(color)

    assert max(grid.shape) <= 30
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(80):
        n_colors = int(rng.integers(2, 5))
        colors = [int(v) for v in rng.choice(np.array(DOT_COLORS), size=n_colors, replace=False)]
        focus = int(rng.integers(0, n_colors))

        counts = [int(rng.integers(1, 5)) for _ in range(n_colors)]
        counts[focus] = int(rng.integers(2, 5))
        if len(set(counts)) < 2 or sum(counts) > 10:
            continue

        dot_specs: list[tuple[int, int, bool, int]] = []
        for idx, (color, count) in enumerate(zip(colors, counts)):
            if idx == focus:
                absorb_count = int(rng.integers(1, count))
            else:
                absorb_count = int(rng.integers(0, count + 1))

            outcomes = [True] * absorb_count + [False] * (count - absorb_count)
            rng.shuffle(outcomes)
            for absorb in outcomes:
                extra = 0 if absorb else int(rng.integers(1, 6))
                dist = int(count + 1 + extra)
                dot_specs.append((int(color), int(count), bool(absorb), dist))

        rng.shuffle(dot_specs)
        side = str(rng.choice(np.array(SIDES)))
        grid = _build_grid(side, dot_specs, rng)
        if _semantic_ok(grid) and not np.array_equal(main(grid), grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
