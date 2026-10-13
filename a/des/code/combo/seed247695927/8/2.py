from __future__ import annotations

import random
from collections import Counter
from typing import Iterable

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = DIR4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
MAX_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR8 if connectivity == 8 else DIR4
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    comps = _components(mask, connectivity=8)
    if not comps:
        raise AssertionError("expected foreground")
    return max(comps, key=len)


def _rotate_color_layer(grid: np.ndarray, color: int, clockwise_turns: int, bg: int) -> np.ndarray:
    turns = int(clockwise_turns) % 4
    if turns == 0:
        return grid
    layer = (grid == int(color)).astype(int)
    rotated = np.rot90(layer, -turns)
    out = grid.copy()
    out[out == int(color)] = int(bg)
    out[rotated == 1] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    board_coords = _largest_component(g != bg)
    board_mask = np.zeros_like(g, dtype=bool)
    board_mask[board_coords[:, 0], board_coords[:, 1]] = True
    r0, c0, r1, c1 = _bbox(board_coords)
    crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    if crop.shape[0] != crop.shape[1]:
        raise AssertionError("expected square board crop")

    out = crop.copy()
    colors = [int(x) for x in np.unique(crop) if int(x) != int(bg)]
    for color in colors:
        cue_size = int(np.count_nonzero((g == int(color)) & (~board_mask)))
        out = _rotate_color_layer(out, int(color), cue_size % 4, int(bg))
    return out


def _rotationally_asymmetric(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _orbit(anchor: tuple[int, int], n: int) -> set[tuple[int, int]]:
    r, c = map(int, anchor)
    pts = set()
    for _ in range(4):
        pts.add((r, c))
        r, c = n - 1 - c, r
    return pts


def _distance_to_edge(n: int) -> np.ndarray:
    rr, cc = np.indices((n, n))
    return np.minimum.reduce([rr, cc, n - 1 - rr, n - 1 - cc])


def _make_band_layer(
    n: int,
    allowed: np.ndarray,
    forced: Iterable[tuple[int, int]],
    density_range: tuple[float, float],
) -> np.ndarray | None:
    forced = [tuple(map(int, p)) for p in forced]
    if not forced:
        return None
    allowed = np.array(allowed, dtype=bool)
    if any(not allowed[r, c] for r, c in forced):
        return None
    lo_frac, hi_frac = density_range
    area = int(allowed.sum())
    lo = max(len(forced), int(np.ceil(area * lo_frac)))
    hi = max(lo, int(np.floor(area * hi_frac)))
    anchor = forced[0]
    blocked = _orbit(anchor, n) - {anchor}
    for r, c in blocked:
        allowed[r, c] = False
    for r, c in forced:
        allowed[r, c] = True
    candidates = [(int(r), int(c)) for r, c in np.argwhere(allowed) if (int(r), int(c)) not in forced]
    for _ in range(80):
        mask = np.zeros((n, n), dtype=bool)
        for r, c in forced:
            mask[r, c] = True
        target = random.randint(lo, hi)
        random.shuffle(candidates)
        for r, c in candidates[: max(0, target - len(forced))]:
            mask[r, c] = True
        if _rotationally_asymmetric(mask):
            return mask
    return None


def _build_board(n: int, colors: list[int], bg: int) -> np.ndarray | None:
    dist = _distance_to_edge(n)
    outer = dist <= 1
    middle = (dist >= 2) & (dist <= 3)
    inner = dist >= 4

    masks = [
        _make_band_layer(n, outer, [(1, 1)], (0.45, 0.85)),
        _make_band_layer(n, middle, [(2, 2), (3, 3)], (0.28, 0.72)),
        _make_band_layer(n, inner, [(4, 4)], (0.20, 0.65)),
    ]
    if any(mask is None for mask in masks):
        return None

    board = np.full((n, n), int(bg), dtype=int)
    for color, mask in zip(colors, masks):
        board[mask] = int(color)

    if len(_components(board != int(bg), connectivity=8)) != 1:
        return None
    return board


def _random_polyomino(area: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < area:
        r, c = random.choice(tuple(cells))
        dr, dc = random.choice(DIR4)
        cells.add((r + dr, c + dc))
    pts = np.array(sorted(cells), dtype=int)
    pts[:, 0] -= pts[:, 0].min()
    pts[:, 1] -= pts[:, 1].min()
    h = int(pts[:, 0].max()) + 1
    w = int(pts[:, 1].max()) + 1
    shape = np.zeros((h, w), dtype=bool)
    shape[pts[:, 0], pts[:, 1]] = True
    return shape


def _chebyshev_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    dr = max(0, max(a[0] - b[2] - 1, b[0] - a[2] - 1))
    dc = max(0, max(a[1] - b[3] - 1, b[1] - a[3] - 1))
    return max(dr, dc)


def _place_shape(
    grid: np.ndarray,
    shape: np.ndarray,
    color: int,
    board_bbox: tuple[int, int, int, int],
    other_bboxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    h, w = shape.shape
    gh, gw = grid.shape
    candidates = []
    for r0 in range(gh - h + 1):
        for c0 in range(gw - w + 1):
            bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            if _chebyshev_gap(bbox, board_bbox) < 2:
                continue
            if any(_chebyshev_gap(bbox, prev) < 2 for prev in other_bboxes):
                continue
            sub = grid[r0 : r0 + h, c0 : c0 + w]
            if np.any(shape & (sub != grid[0, 0])):
                continue
            candidates.append((r0, c0))
    if not candidates:
        return None
    r0, c0 = random.choice(candidates)
    grid[r0 : r0 + h, c0 : c0 + w][shape] = int(color)
    return (r0, c0, r0 + h - 1, c0 + w - 1)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 30)
    w = random.randint(20, 30)
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    colors = random.sample(palette, 3)

    n = random.randint(11, min(17, h - 4, w - 4))
    board = _build_board(n, colors, bg)
    if board is None:
        return None

    grid = np.full((h, w), int(bg), dtype=int)
    board_r = random.randint(1, h - n - 1)
    board_c = random.randint(1, w - n - 1)
    grid[board_r : board_r + n, board_c : board_c + n] = board
    board_bbox = (board_r, board_c, board_r + n - 1, board_c + n - 1)

    cue_rotations = [random.choice([None, 0, 1, 2, 3]) for _ in colors]
    cue_areas = {0: 4, 1: 1, 2: 2, 3: 3}
    cue_bboxes: list[tuple[int, int, int, int]] = []
    for color, rot in zip(colors, cue_rotations):
        if rot is None:
            continue
        shape = _random_polyomino(cue_areas[int(rot)])
        bbox = _place_shape(grid, shape, int(color), board_bbox, cue_bboxes)
        if bbox is None:
            return None
        cue_bboxes.append(bbox)

    fg_comps = _components(grid != int(bg), connectivity=8)
    if not fg_comps:
        return None
    sizes = sorted((len(comp) for comp in fg_comps), reverse=True)
    if sizes[0] <= sum(sizes[1:]):
        return None
    if sizes[0] != int(np.count_nonzero(board != int(bg))):
        return None

    for color in colors:
        outside = int(np.count_nonzero((grid == int(color)) & ~((np.indices((h, w))[0] >= board_r) & (np.indices((h, w))[0] < board_r + n) & (np.indices((h, w))[1] >= board_c) & (np.indices((h, w))[1] < board_c + n))))
        expected = 0
        for c2, rot in zip(colors, cue_rotations):
            if c2 == color and rot is not None:
                expected = cue_areas[int(rot)]
        if outside != expected:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
