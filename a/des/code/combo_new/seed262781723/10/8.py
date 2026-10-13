from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 400

DIRECTIONS = ("up", "down", "left", "right")
HORIZONTAL_DIRS = ("left", "right")
VERTICAL_DIRS = ("up", "down")

STEPS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# Arrow cells are defined relative to the front-center tip cell.
ARROW_FROM_TIP = {
    "right": ((0, 0), (0, -1), (0, -2), (-1, 0), (1, 0)),
    "left": ((0, 0), (0, 1), (0, 2), (-1, 0), (1, 0)),
    "up": ((0, 0), (1, 0), (2, 0), (0, -1), (0, 1)),
    "down": ((0, 0), (-1, 0), (-2, 0), (0, -1), (0, 1)),
}

ARROW_PATTERNS = {
    "right": frozenset({(1, 0), (1, 1), (1, 2), (0, 2), (2, 2)}),
    "left": frozenset({(1, 0), (1, 1), (1, 2), (0, 0), (2, 0)}),
    "up": frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
    "down": frozenset({(0, 1), (1, 1), (2, 1), (2, 0), (2, 2)}),
}

TIP_OFFSETS = {
    "right": (1, 2),
    "left": (1, 0),
    "up": (0, 1),
    "down": (2, 1),
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)

    return comps


def _touches_edge(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in coords)


def _decode_arrow(coords: list[tuple[int, int]]) -> dict[str, object] | None:
    if len(coords) != 5:
        return None

    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 != 2 or c1 - c0 != 2:
        return None

    rel = frozenset((r - r0, c - c0) for r, c in coords)
    for direction, pattern in ARROW_PATTERNS.items():
        if rel == pattern:
            tr, tc = TIP_OFFSETS[direction]
            return {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "direction": direction,
                "tip": (int(r0 + tr), int(c0 + tc)),
                "cells": {(int(r), int(c)) for r, c in coords},
            }
    return None


def _project_color(g: np.ndarray, direction: str, r: int, c: int) -> int:
    if direction == "right":
        return int(g[r, 0])
    if direction == "left":
        return int(g[r, -1])
    if direction == "down":
        return int(g[0, c])
    if direction == "up":
        return int(g[-1, c])
    raise AssertionError(direction)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each grey arrow forward using the opposite edge stripe as its color source.

    Input structure:
    - The background is black.
    - Every grid edge carries a one-cell-thick color stripe band along the border.
    - Inside the border there are one or more grey 3x3 arrow components pointing up, down, left, or right.

    Rule:
    - Identify every internal grey arrow and determine its direction from its 5-pixel shape.
    - The stripe on the edge behind the arrow is that arrow's source: left-facing uses the right edge, right-facing uses the left edge, up-facing uses the bottom edge, and down-facing uses the top edge.
    - Starting one cell in front of the arrow tip, trace a ray in the pointing direction until the opposite border, inclusive.
    - Recolor each ray cell from the source stripe at the same row for horizontal rays or the same column for vertical rays.
    - Keep all arrow pixels grey, and when rays overlap, apply arrows in top-to-bottom then left-to-right order so later arrows overwrite earlier colors.

    Key insight:
    - The arrow encodes both where the beam travels and which opposite border stripe supplies the beam's per-row or per-column colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)

    arrows: list[dict[str, object]] = []
    for coords in _components(g == GREY):
        if _touches_edge(coords, h, w):
            continue
        arrow = _decode_arrow(coords)
        assert arrow is not None, "every internal grey component must be a valid arrow"
        arrows.append(arrow)

    assert arrows, "expected at least one internal arrow"
    arrows.sort(key=lambda arrow: (arrow["bbox"][0], arrow["bbox"][1]))
    arrow_cells = {cell for arrow in arrows for cell in arrow["cells"]}

    out = g.copy()
    for arrow in arrows:
        direction = str(arrow["direction"])
        dr, dc = STEPS[direction]
        r, c = map(int, arrow["tip"])
        r += dr
        c += dc
        while 0 <= r < h and 0 <= c < w:
            if (r, c) not in arrow_cells:
                out[r, c] = _project_color(g, direction, r, c)
            r += dr
            c += dc

    return out


def _arrow_cells_from_tip(tip: tuple[int, int], direction: str) -> set[tuple[int, int]]:
    r, c = tip
    return {
        (int(r + dr), int(c + dc))
        for dr, dc in ARROW_FROM_TIP[direction]
    }


def _ray_cells_from_tip(
    tip: tuple[int, int], direction: str, h: int, w: int
) -> list[tuple[int, int]]:
    dr, dc = STEPS[direction]
    r, c = tip
    r += dr
    c += dc
    out: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        out.append((int(r), int(c)))
        r += dr
        c += dc
    return out


def _candidate_tips(h: int, w: int, direction: str) -> list[tuple[int, int]]:
    tips: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            cells = _arrow_cells_from_tip((r, c), direction)
            if not all(1 <= rr < h - 1 and 1 <= cc < w - 1 for rr, cc in cells):
                continue
            if len(_ray_cells_from_tip((r, c), direction, h, w)) < 2:
                continue
            tips.append((int(r), int(c)))
    return tips


def _expand(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((int(nr), int(nc)))
    return out


def _make_stripe(length: int, palette: tuple[int, ...]) -> np.ndarray:
    colors = random.sample(palette, k=random.randint(2, min(4, len(palette))))
    stripe = np.empty(length, dtype=int)
    i = 0
    prev = None
    while i < length:
        choices = [color for color in colors if color != prev] or colors
        color = random.choice(choices)
        run = random.randint(1, min(4, length - i))
        stripe[i : i + run] = int(color)
        prev = color
        i += run

    if len(set(map(int, stripe.tolist()))) == 1:
        idx = random.randrange(length)
        stripe[idx] = random.choice([color for color in colors if color != int(stripe[idx])])
    return stripe


def _make_edge_stripes(h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    palette = tuple(random.sample(PALETTE, k=random.randint(3, 5)))
    for _ in range(20):
        top = _make_stripe(w, palette)
        bottom = _make_stripe(w, palette)
        left = _make_stripe(h, palette)
        right = _make_stripe(h, palette)
        if not (np.array_equal(top, bottom) and np.array_equal(left, right)):
            break

    top[0] = left[0]
    top[-1] = right[0]
    bottom[0] = left[-1]
    bottom[-1] = right[-1]
    return top, bottom, left, right


def _sample_directions(count: int, want_cross: bool) -> list[str]:
    directions: list[str] = []
    if want_cross:
        directions.append(random.choice(HORIZONTAL_DIRS))
        directions.append(random.choice(VERTICAL_DIRS))

    target_unique = min(count, 4)
    while len(directions) < target_unique:
        choices = [direction for direction in DIRECTIONS if direction not in directions]
        directions.append(random.choice(choices))

    while len(directions) < count:
        directions.append(random.choice(DIRECTIONS))

    random.shuffle(directions)
    return directions


def _cross_tip_choices(
    candidates: list[tuple[int, int]], direction: str, cross_r: int, cross_c: int
) -> list[tuple[int, int]]:
    if direction == "right":
        return [tip for tip in candidates if tip[0] == cross_r and tip[1] < cross_c]
    if direction == "left":
        return [tip for tip in candidates if tip[0] == cross_r and tip[1] > cross_c]
    if direction == "down":
        return [tip for tip in candidates if tip[1] == cross_c and tip[0] < cross_r]
    if direction == "up":
        return [tip for tip in candidates if tip[1] == cross_c and tip[0] > cross_r]
    raise AssertionError(direction)


def _record(direction: str, tip: tuple[int, int]) -> dict[str, object]:
    return {
        "direction": direction,
        "tip": (int(tip[0]), int(tip[1])),
        "cells": _arrow_cells_from_tip(tip, direction),
    }


def _try_place_cross_pair(
    h: int,
    w: int,
    hdir: str,
    vdir: str,
    candidates: dict[str, list[tuple[int, int]]],
    blocked: set[tuple[int, int]],
) -> list[dict[str, object]] | None:
    rows = list(range(3, h - 3))
    cols = list(range(3, w - 3))
    if not rows or not cols:
        return None

    for _ in range(120):
        cross_r = random.choice(rows)
        cross_c = random.choice(cols)
        h_tips = _cross_tip_choices(candidates[hdir], hdir, cross_r, cross_c)
        v_tips = _cross_tip_choices(candidates[vdir], vdir, cross_r, cross_c)
        random.shuffle(h_tips)
        random.shuffle(v_tips)
        if not h_tips or not v_tips:
            continue

        for h_tip in h_tips:
            h_cells = _arrow_cells_from_tip(h_tip, hdir)
            if h_cells & blocked:
                continue
            h_blocked = blocked | _expand(h_cells, h, w)

            for v_tip in v_tips:
                v_cells = _arrow_cells_from_tip(v_tip, vdir)
                if v_cells & h_blocked:
                    continue
                return [_record(hdir, h_tip), _record(vdir, v_tip)]

    return None


def _place_random_arrow(
    direction: str,
    h: int,
    w: int,
    candidates: dict[str, list[tuple[int, int]]],
    blocked: set[tuple[int, int]],
) -> dict[str, object] | None:
    tips = list(candidates[direction])
    random.shuffle(tips)
    for tip in tips:
        cells = _arrow_cells_from_tip(tip, direction)
        if not (cells & blocked):
            return _record(direction, tip)
    return None


def _render_grid(
    h: int,
    w: int,
    top: np.ndarray,
    bottom: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    arrows: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    grid[0, :] = top
    grid[-1, :] = bottom
    grid[:, 0] = left
    grid[:, -1] = right
    for arrow in arrows:
        for r, c in arrow["cells"]:
            grid[r, c] = GREY
    return grid


def _attempt() -> np.ndarray | None:
    h = random.randint(10, 20)
    w = random.randint(10, 20)
    capacity = min(6, ((h - 2) * (w - 2)) // 24)
    if capacity < 2:
        return None

    want_cross = random.random() < 0.6
    arrow_count = random.randint(2, capacity)
    directions = _sample_directions(arrow_count, want_cross)
    candidates = {direction: _candidate_tips(h, w, direction) for direction in DIRECTIONS}
    if any(not candidates[direction] for direction in directions):
        return None

    arrows: list[dict[str, object]] = []
    blocked: set[tuple[int, int]] = set()

    if want_cross:
        hdir = next(direction for direction in directions if direction in HORIZONTAL_DIRS)
        vdir = next(direction for direction in directions if direction in VERTICAL_DIRS)
        pair = _try_place_cross_pair(h, w, hdir, vdir, candidates, blocked)
        if pair is None:
            return None
        for arrow in pair:
            arrows.append(arrow)
            blocked |= _expand(arrow["cells"], h, w)
        directions.remove(hdir)
        directions.remove(vdir)

    for direction in directions:
        arrow = _place_random_arrow(direction, h, w, candidates, blocked)
        if arrow is None:
            return None
        arrows.append(arrow)
        blocked |= _expand(arrow["cells"], h, w)

    top, bottom, left, right = _make_edge_stripes(h, w)
    grid = _render_grid(h, w, top, bottom, left, right, arrows)

    if len({direction for direction in [arrow["direction"] for arrow in arrows]}) < 2:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid projector-arrow puzzle")
