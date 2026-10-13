from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIRS:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _find_legend(g: np.ndarray) -> tuple[int, dict[int, int]]:
    plus = np.array(
        [
            [False, True, False],
            [True, True, True],
            [False, True, False],
        ],
        dtype=bool,
    )
    candidates: list[tuple[int, int, int, dict[int, int]]] = []
    h, w = g.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            sub = g[r - 1 : r + 2, c - 1 : c + 2]
            mask = (sub != BG) & (sub != FRAME)
            if not np.array_equal(mask, plus):
                continue
            center = int(g[r, c])
            color_to_dir = {
                int(g[r - 1, c]): 0,
                int(g[r, c + 1]): 1,
                int(g[r + 1, c]): 2,
                int(g[r, c - 1]): 3,
            }
            colors = [center] + list(color_to_dir)
            if len(set(colors)) != 5:
                continue
            candidates.append((r, c, center, color_to_dir))
    assert candidates, "expected a bottom-right legend cross"
    _, _, center_color, color_to_dir = max(candidates, key=lambda item: (item[0], item[1]))
    return center_color, color_to_dir


def _find_frames(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for pts in _components(g == FRAME):
        r0, c0, r1, c1 = _bbox(pts)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 3:
            continue
        if len(pts) != 2 * (height + width) - 4:
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        if not (
            np.all(sub[0, :] == FRAME)
            and np.all(sub[-1, :] == FRAME)
            and np.all(sub[:, 0] == FRAME)
            and np.all(sub[:, -1] == FRAME)
        ):
            continue
        frames.append((r0, c0, r1, c1))
    frames.sort()
    return frames


def _frame_seed(
    g: np.ndarray, bbox: tuple[int, int, int, int]
) -> tuple[int, tuple[int, int], tuple[int, int], int]:
    r0, c0, r1, c1 = bbox
    h, w = g.shape
    found: list[tuple[int, tuple[int, int], tuple[int, int], int]] = []

    if r0 > 0:
        cells = [(r0 - 1, c) for c in range(c0 + 1, c1)]
        seeds = [(r, c) for r, c in cells if int(g[r, c]) not in (BG, FRAME)]
        assert len(seeds) <= 1, "expected at most one seed on each frame side"
        if seeds:
            r, c = seeds[0]
            found.append((0, (r, c), (r0 + 1, c), int(g[r, c])))

    if c1 + 1 < w:
        cells = [(r, c1 + 1) for r in range(r0 + 1, r1)]
        seeds = [(r, c) for r, c in cells if int(g[r, c]) not in (BG, FRAME)]
        assert len(seeds) <= 1, "expected at most one seed on each frame side"
        if seeds:
            r, c = seeds[0]
            found.append((1, (r, c), (r, c1 - 1), int(g[r, c])))

    if r1 + 1 < h:
        cells = [(r1 + 1, c) for c in range(c0 + 1, c1)]
        seeds = [(r, c) for r, c in cells if int(g[r, c]) not in (BG, FRAME)]
        assert len(seeds) <= 1, "expected at most one seed on each frame side"
        if seeds:
            r, c = seeds[0]
            found.append((2, (r, c), (r1 - 1, c), int(g[r, c])))

    if c0 > 0:
        cells = [(r, c0 - 1) for r in range(r0 + 1, r1)]
        seeds = [(r, c) for r, c in cells if int(g[r, c]) not in (BG, FRAME)]
        assert len(seeds) <= 1, "expected at most one seed on each frame side"
        if seeds:
            r, c = seeds[0]
            found.append((3, (r, c), (r, c0 + 1), int(g[r, c])))

    assert len(found) == 1, "expected exactly one seed per frame"
    return found[0]


def _growth_direction(physical_side: int, seed_color: int, center_color: int, color_to_dir: dict[int, int]) -> int:
    inward = (physical_side + 2) % 4
    if seed_color == center_color:
        return inward
    ring_dir = color_to_dir[seed_color]
    turns = (physical_side - ring_dir) % 4
    return (inward + turns) % 4


def _paint_growth(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    start_cell: tuple[int, int],
    growth_dir: int,
    color: int,
) -> None:
    r0, c0, r1, c1 = bbox
    inner_r0 = r0 + 1
    inner_c0 = c0 + 1
    inner_r1 = r1 - 1
    inner_c1 = c1 - 1
    sr, sc = start_cell

    if growth_dir in (0, 2):
        step = -1 if growth_dir == 0 else 1
        row = sr
        while inner_r0 <= row <= inner_r1:
            out[row, inner_c0 : inner_c1 + 1] = color
            row += step
    else:
        step = 1 if growth_dir == 1 else -1
        col = sc
        while inner_c0 <= col <= inner_c1:
            out[inner_r0 : inner_r1 + 1, col] = color
            col += step


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the seed color's position in a bottom-right color-ring legend to rotate the seed's
    physical orientation into the true growth direction, then fill each framed region by
    sweeping rows or columns from the seed-adjacent interior cell.
    """
    g = np.asarray(grid, dtype=int)
    center_color, color_to_dir = _find_legend(g)
    frames = _find_frames(g)
    assert frames, "expected at least one grey rectangular frame"

    out = g.copy()
    for bbox in frames:
        side, _seed_pos, start_cell, seed_color = _frame_seed(g, bbox)
        growth_dir = _growth_direction(side, seed_color, center_color, color_to_dir)
        _paint_growth(out, bbox, start_cell, growth_dir, seed_color)
    return out


_GEN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
_GEN_RETRIES = 400


def _reserve(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _paint_frame(g: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    g[r0, c0 : c1 + 1] = FRAME
    g[r1, c0 : c1 + 1] = FRAME
    g[r0 : r1 + 1, c0] = FRAME
    g[r0 : r1 + 1, c1] = FRAME


def _legend_positions(center: tuple[int, int]) -> dict[int, tuple[int, int]]:
    r, c = center
    return {
        0: (r - 1, c),
        1: (r, c + 1),
        2: (r + 1, c),
        3: (r, c - 1),
    }


def _seed_color_for(side: int, growth: int, center_color: int, arm_by_dir: dict[int, int]) -> int:
    inward = (side + 2) % 4
    if growth == inward and random.random() < 0.35:
        return center_color
    outer_dir = (2 * side + 2 - growth) % 4
    return arm_by_dir[outer_dir]


def _place_seed(
    g: np.ndarray, bbox: tuple[int, int, int, int], side: int, color: int
) -> None:
    r0, c0, r1, c1 = bbox
    if side == 0:
        col = random.randint(c0 + 1, c1 - 1)
        g[r0 - 1, col] = color
    elif side == 1:
        row = random.randint(r0 + 1, r1 - 1)
        g[row, c1 + 1] = color
    elif side == 2:
        col = random.randint(c0 + 1, c1 - 1)
        g[r1 + 1, col] = color
    else:
        row = random.randint(r0 + 1, r1 - 1)
        g[row, c0 - 1] = color


def _pattern_configs(n_regions: int) -> list[tuple[int, int]]:
    sides = random.sample(range(4), 4)
    configs = [
        (sides[0], (sides[0] + 2) % 4),
        (sides[1], sides[1]),
        (sides[2], random.choice([(sides[2] + 1) % 4, (sides[2] + 3) % 4])),
    ]
    if n_regions == 4:
        configs.append((sides[3], random.randrange(4)))
    random.shuffle(configs)
    return configs


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 30)
    w = random.randint(20, 30)
    max_regions = 4 if min(h, w) >= 25 else 3
    n_regions = random.randint(3, max_regions)
    g = np.full((h, w), BG, dtype=int)

    colors = random.sample(_GEN_COLORS, 5)
    center_color = colors[0]
    arm_colors = colors[1:]
    random.shuffle(arm_colors)
    arm_by_dir = {d: arm_colors[d] for d in range(4)}

    legend_center = (random.choice([h - 3, h - 2]), random.choice([w - 3, w - 2]))
    lr, lc = legend_center
    for d, (r, c) in _legend_positions(legend_center).items():
        g[r, c] = arm_by_dir[d]
    g[lr, lc] = center_color

    occupied = np.zeros((h, w), dtype=bool)
    _reserve(occupied, max(0, lr - 2), max(0, lc - 2), min(h - 1, lr + 2), min(w - 1, lc + 2))

    max_dim = 8 if n_regions == 4 else 9
    for side, growth in _pattern_configs(n_regions):
        placed = False
        for _ in range(250):
            outer_h = random.randint(5, max_dim)
            outer_w = random.randint(5, max_dim)
            if h - outer_h - 5 < 2 or w - outer_w - 5 < 2:
                continue
            r0 = random.randint(2, h - outer_h - 3)
            c0 = random.randint(2, w - outer_w - 3)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1

            rr0 = max(0, r0 - 2)
            cc0 = max(0, c0 - 2)
            rr1 = min(h - 1, r1 + 2)
            cc1 = min(w - 1, c1 + 2)
            if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue

            _paint_frame(g, r0, c0, r1, c1)
            seed_color = _seed_color_for(side, growth, center_color, arm_by_dir)
            _place_seed(g, (r0, c0, r1, c1), side, seed_color)
            _reserve(occupied, rr0, cc0, rr1, cc1)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(g)
    except Exception:
        return None

    changed = (out != g) & (g == BG)
    if not np.any(changed):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
