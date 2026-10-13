from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BODY_COLOR = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

MIN_TOWERS = 2
MAX_TOWERS = 6
BODY_HEIGHT_MIN = 2
BODY_HEIGHT_MAX = 4
GRID_H_MIN = 12
GRID_H_MAX = 24
GRID_W_MIN = 12
GRID_W_MAX = 24
MAX_GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.reshape(-1)))
    bg, bg_count = counts.most_common(1)[0]
    assert sum(1 for count in counts.values() if count == bg_count) == 1
    return int(bg)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _find_control_row(grid: np.ndarray, bg: int) -> tuple[int, list[int]]:
    h, _ = grid.shape
    for r in range(h - 1, -1, -1):
        cols = np.flatnonzero(grid[r] != bg)
        if len(cols) < 2:
            continue
        colors = [int(grid[r, c]) for c in cols]
        if len(colors) != len(set(colors)):
            continue
        if bool((grid[r + 1 :] != bg).any()):
            continue
        return int(r), colors
    raise AssertionError("control row not found")


def _parse_tower(grid: np.ndarray, pts: list[tuple[int, int]], bg: int) -> dict[str, int] | None:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if c1 - c0 + 1 != 3 or r1 == r0:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if int((patch != bg).sum()) != len(pts):
        return None
    if not (int(patch[0, 1]) != bg and int(patch[0, 0]) == bg and int(patch[0, 2]) == bg):
        return None

    marker_color = int(patch[0, 1])
    body = patch[1:]
    if body.size == 0:
        return None
    body_color = int(body[0, 0])
    if body_color in (bg, marker_color):
        return None
    if not bool((body == body_color).all()):
        return None

    return {
        "marker_color": marker_color,
        "body_color": body_color,
        "body_height": int(body.shape[0]),
    }


def _extract_scene(grid: np.ndarray) -> tuple[np.ndarray, int, list[int], dict[int, dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    control_row, control_colors = _find_control_row(g, bg)

    body_area = g.copy()
    body_area[control_row, :] = bg
    assert not bool((body_area[control_row + 1 :] != bg).any())

    towers: dict[int, dict[str, int]] = {}
    for comp in _components(body_area != bg):
        tower = _parse_tower(body_area, comp, bg)
        assert tower is not None, "unexpected non-background component"
        marker_color = int(tower["marker_color"])
        assert marker_color not in towers, "tower marker colors must be unique"
        towers[marker_color] = tower

    assert len(control_colors) == len(towers) >= 2
    assert set(control_colors) == set(towers)
    return g, bg, control_colors, towers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort towers by the bottom control-band order, then re-encode each tower's body
    height with its 1-based position in that order.
    """
    _, bg, control_colors, towers = _extract_scene(grid)

    tower_count = len(control_colors)
    out = np.full((tower_count + 1, 3 * tower_count), bg, dtype=int)

    for rank, marker_color in enumerate(control_colors, start=1):
        tower = towers[int(marker_color)]
        c0 = 3 * (rank - 1)
        out[tower_count + 1 - rank :, c0 : c0 + 3] = int(tower["body_color"])
        out[tower_count - rank, c0 + 1] = int(marker_color)

    return out


def _choose_canvas(rng: random.Random, body_heights: list[int], tower_count: int) -> tuple[int, int] | None:
    body_cells = 3 * sum(body_heights)
    non_bg_cells = body_cells + 2 * tower_count
    min_area = max(GRID_H_MIN * GRID_W_MIN, 2 * body_cells + 2 * tower_count + 7)
    min_height = BODY_HEIGHT_MAX + 4

    for _ in range(80):
        h = rng.randint(max(GRID_H_MIN, min_height), GRID_H_MAX)
        w = rng.randint(GRID_W_MIN, GRID_W_MAX)
        if h * w < min_area:
            continue
        if h * w - non_bg_cells <= body_cells:
            continue
        return int(h), int(w)
    return None


def _place_towers(rng: random.Random, h: int, w: int, body_heights: list[int]) -> list[tuple[int, int]] | None:
    occ = np.zeros((h, w), dtype=bool)
    occ[h - 1, :] = True
    placements: list[tuple[int, int] | None] = [None] * len(body_heights)

    order = list(range(len(body_heights)))
    order.sort(key=lambda idx: body_heights[idx], reverse=True)
    if rng.random() < 0.5:
        rng.shuffle(order)

    for idx in order:
        body_h = int(body_heights[idx])
        candidates: list[tuple[int, int]] = []
        max_top = h - body_h - 3
        for top_r in range(max_top + 1):
            for left_c in range(w - 2):
                bottom_r = top_r + body_h
                pr0 = max(0, top_r - 1)
                pr1 = min(h - 1, bottom_r + 1)
                pc0 = max(0, left_c - 1)
                pc1 = min(w - 1, left_c + 3)
                if bool(occ[pr0 : pr1 + 1, pc0 : pc1 + 1].any()):
                    continue
                candidates.append((int(top_r), int(left_c)))
        if not candidates:
            return None

        top_r, left_c = rng.choice(candidates)
        bottom_r = top_r + body_h
        pr0 = max(0, top_r - 1)
        pr1 = min(h - 1, bottom_r + 1)
        pc0 = max(0, left_c - 1)
        pc1 = min(w - 1, left_c + 3)
        occ[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
        placements[idx] = (int(top_r), int(left_c))

    assert all(p is not None for p in placements)
    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _build_input_grid(
    rng: random.Random,
    h: int,
    w: int,
    bg: int,
    control_colors: list[int],
    body_heights: list[int],
    placements: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)

    control_cols_pool = list(range(1, w - 1)) if w >= len(control_colors) + 2 else list(range(w))
    control_cols = sorted(rng.sample(control_cols_pool, len(control_colors)))
    for color, col in zip(control_colors, control_cols):
        grid[h - 1, int(col)] = int(color)

    for color, body_h, (top_r, left_c) in zip(control_colors, body_heights, placements):
        grid[top_r, left_c + 1] = int(color)
        grid[top_r + 1 : top_r + body_h + 1, left_c : left_c + 3] = BODY_COLOR

    return grid


def _is_interesting(control_colors: list[int], body_heights: list[int], placements: list[tuple[int, int]]) -> bool:
    left_to_right_colors = [color for _, color in sorted(zip(placements, control_colors), key=lambda item: (item[0][1], item[0][0]))]
    order_changes = left_to_right_colors != control_colors
    height_changes = any(body_h != rank for rank, body_h in enumerate(body_heights, start=1))
    return order_changes or height_changes


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    tower_count = rng.randint(MIN_TOWERS, MAX_TOWERS)
    bg = rng.choice([color for color in range(10) if color != BODY_COLOR])

    marker_pool = [color for color in range(10) if color not in (bg, BODY_COLOR)]
    control_colors = rng.sample(marker_pool, tower_count)
    body_heights = [rng.randint(BODY_HEIGHT_MIN, BODY_HEIGHT_MAX) for _ in range(tower_count)]

    canvas = _choose_canvas(rng, body_heights, tower_count)
    if canvas is None:
        return None
    h, w = canvas

    placements = _place_towers(rng, h, w, body_heights)
    if placements is None:
        return None

    if not _is_interesting(control_colors, body_heights, placements):
        return None

    grid = _build_input_grid(rng, h, w, bg, control_colors, body_heights, placements)

    try:
        _, parsed_bg, parsed_control_colors, towers = _extract_scene(grid)
    except AssertionError:
        return None

    if parsed_bg != bg or parsed_control_colors != control_colors:
        return None
    if any(int(towers[color]["body_height"]) != body_heights[idx] for idx, color in enumerate(control_colors)):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
