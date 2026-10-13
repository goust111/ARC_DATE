from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 5
GEN_ATTEMPTS = 500
PALETTE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _find_separator_row(grid: np.ndarray, bg: int) -> int:
    rows = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if np.all(row == row[0]) and int(row[0]) != bg:
            rows.append(r)
    assert len(rows) == 1, "expected exactly one full-width separator row"
    return int(rows[0])


def _parse_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    sep_r = _find_separator_row(g, bg)
    assert sep_r > 0 and sep_r < g.shape[0] - 1

    code_r = sep_r - 1
    code_non_bg = np.flatnonzero(g[code_r] != bg)
    assert len(code_non_bg) > 0 and int(code_non_bg[0]) == 0
    code = g[code_r, : int(code_non_bg[-1]) + 1].astype(int)
    assert int(code[0]) != bg and int(code[0]) != WALL

    lower = g[sep_r + 1 :]
    assert np.any(lower == WALL), "expected at least one grey wall"
    seed_candidates = np.argwhere((lower != bg) & (lower != WALL))
    assert len(seed_candidates) == 1, "expected exactly one seed pixel"
    seed_r = int(seed_candidates[0, 0]) + sep_r + 1
    seed_c = int(seed_candidates[0, 1])
    assert int(g[seed_r, seed_c]) == int(code[0])

    allowed = {bg, WALL, int(code[0])}
    assert set(map(int, np.unique(lower))).issubset(allowed)
    assert int(np.sum(lower == int(code[0]))) == 1
    return g, bg, sep_r, code, (seed_r, seed_c)


def _distance_map(grid: np.ndarray, sep_r: int, seed: tuple[int, int]) -> np.ndarray:
    h, w = grid.shape
    dist = np.full((h, w), -1, dtype=int)
    q = deque([seed])
    dist[seed] = 0
    while q:
        r, c = q.popleft()
        base = int(dist[r, c]) + 1
        for nr, nc in _neighbors8(r, c):
            if nr <= sep_r or nr >= h or nc < 0 or nc >= w:
                continue
            if int(grid[nr, nc]) == WALL or dist[nr, nc] != -1:
                continue
            dist[nr, nc] = base
            q.append((nr, nc))
    return dist


def _paint_with_palette(grid: np.ndarray, sep_r: int, code: np.ndarray, dist: np.ndarray) -> np.ndarray:
    out = grid.copy()
    lower_mask = np.zeros(grid.shape, dtype=bool)
    lower_mask[sep_r + 1 :, :] = True
    reachable = (dist >= 0) & lower_mask
    far = reachable & (dist >= len(code))
    out[far] = BG
    for d, color in enumerate(code):
        out[reachable & (dist == d)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Color each reachable background layer by geodesic distance from the seed, wrapping around grey walls with the top palette.

    Input structure:
    - The top row contains a left-aligned palette whose span ends at the last non-background cell, and the next row is a full-width separator.
    - Below the separator there is exactly one non-wall seed pixel and one or more grey wall components on a black background.
    - Background entries inside the palette span denote blank distance layers.

    Rule:
    - Keep the palette row, separator row, and all grey walls unchanged.
    - In the lower region, compute the shortest 8-connected path length from the seed to every non-wall cell, forbidding paths from passing through grey walls.
    - A reachable cell at distance d takes palette[d] if d is within the palette span; if that palette entry is background it stays blank, and reachable cells beyond the palette span also stay background.
    - Any lower-region cell unreachable from the seed remains background.

    Key insight:
    - The palette indexes wavefront time, not geometric radius: walls delay or block the wave, so visible rings must follow obstacle-aware geodesic distance rather than plain Chebyshev distance.
    """
    g, _bg, sep_r, code, seed = _parse_scene(grid)
    dist = _distance_map(g, sep_r, seed)
    return _paint_with_palette(g, sep_r, code, dist)


def _compact_palette(code: np.ndarray) -> np.ndarray:
    return np.array([int(color) for color in code if int(color) != BG], dtype=int)


def _chebyshev_fill(grid: np.ndarray, sep_r: int, code: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    out = grid.copy()
    sr, sc = seed
    h, w = grid.shape
    for r in range(sep_r + 1, h):
        for c in range(w):
            if int(grid[r, c]) == WALL:
                continue
            d = max(abs(r - sr), abs(c - sc))
            out[r, c] = int(code[d]) if d < len(code) else BG
    return out


def _count_shortest_paths(dist: np.ndarray, sep_r: int, seed: tuple[int, int]) -> int:
    del seed
    coords = np.argwhere(dist >= 0)
    if len(coords) == 0:
        return 0
    order = sorted(((int(dist[r, c]), int(r), int(c)) for r, c in coords))
    ways = np.zeros(dist.shape, dtype=int)
    ways[coords[0][0], coords[0][1]] = 1
    for d, r, c in order:
        if d == 0:
            ways[r, c] = 1
            continue
        total = 0
        for nr, nc in _neighbors8(r, c):
            if nr <= sep_r or nr >= dist.shape[0] or nc < 0 or nc >= dist.shape[1]:
                continue
            if dist[nr, nc] == d - 1:
                total = min(2, total + int(ways[nr, nc]))
        ways[r, c] = total
    return int(np.sum((ways >= 2) & (dist >= 2)))


def _place_rect_frame(canvas: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    canvas[r0, c0 : c0 + w] = WALL
    canvas[r0 + h - 1, c0 : c0 + w] = WALL
    canvas[r0 : r0 + h, c0] = WALL
    canvas[r0 : r0 + h, c0 + w - 1] = WALL


def _extra_wall_patch(rng: random.Random, h: int, w: int) -> np.ndarray:
    patch_h = rng.randint(2, 5)
    patch_w = rng.randint(2, 6)
    patch = np.zeros((patch_h, patch_w), dtype=int)
    style = rng.choice(["hline", "vline", "elbow", "block"])
    if style == "hline":
        rr = rng.randrange(patch_h)
        patch[rr, :] = WALL
    elif style == "vline":
        cc = rng.randrange(patch_w)
        patch[:, cc] = WALL
    elif style == "elbow":
        rr = rng.randrange(1, patch_h)
        cc = rng.randrange(1, patch_w)
        patch[rr, : cc + 1] = WALL
        patch[: rr + 1, cc] = WALL
    else:
        patch[:, :] = WALL
        if patch_h >= 4 and patch_w >= 4:
            patch[1:-1, 1:-1] = 0
    return patch


def _can_paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, protected: set[tuple[int, int]]) -> bool:
    h, w = canvas.shape
    ph, pw = patch.shape
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        return False
    for pr in range(ph):
        for pc in range(pw):
            if int(patch[pr, pc]) != WALL:
                continue
            rr = r0 + pr
            cc = c0 + pc
            if int(canvas[rr, cc]) == WALL or (rr, cc) in protected:
                return False
    return True


def _paste_walls(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch == WALL
    canvas[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = WALL


def _sample_palette(rng: random.Random, reachable_max: int) -> tuple[int, int, np.ndarray]:
    max_len = min(10, reachable_max)
    if max_len < 6:
        raise ValueError("reachable area is too small for a useful palette")
    length = rng.randint(6, max_len)
    separator = rng.choice(PALETTE_COLORS)
    usable = [color for color in PALETTE_COLORS if color != separator]
    code = np.empty(length, dtype=int)
    code[0] = rng.choice(usable)
    blank_slots = [rng.randint(1, length - 3)]
    if length >= 8 and rng.random() < 0.45:
        candidates = [i for i in range(1, length - 2) if i not in blank_slots]
        blank_slots.append(rng.choice(candidates))
    blank_slots = sorted(set(blank_slots))
    for idx in range(1, length):
        code[idx] = rng.choice(usable)
    for idx in blank_slots:
        code[idx] = BG
    code[-1] = rng.choice(usable)
    return separator, blank_slots[0], code


def _semantic_checks(inp: np.ndarray, code: np.ndarray, blank_idx: int) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False

    g, _bg, sep_r, parsed_code, seed = _parse_scene(inp)
    if not np.array_equal(code, parsed_code):
        return False
    dist = _distance_map(g, sep_r, seed)
    expected = _paint_with_palette(g, sep_r, code, dist)
    if not np.array_equal(out, expected):
        return False

    reachable = dist >= 0
    if not np.any(reachable & (dist >= len(code))):
        return False
    if not np.any(reachable & (dist == blank_idx)):
        return False

    sr, sc = seed
    rows, cols = np.indices(g.shape)
    cheb = np.maximum(np.abs(rows - sr), np.abs(cols - sc))
    detour_mask = reachable & (dist < len(code)) & (dist > cheb) & (g != WALL)
    if not np.any(detour_mask):
        return False

    lower = np.zeros(g.shape, dtype=bool)
    lower[sep_r + 1 :, :] = True
    unreachable = lower & (g != WALL) & (dist < 0)
    if not np.any(unreachable):
        return False
    if int(np.min(cheb[unreachable])) > 4:
        return False

    if _count_shortest_paths(dist, sep_r, seed) == 0:
        return False

    cheb_out = _chebyshev_fill(g, sep_r, code, seed)
    if np.array_equal(out, cheb_out):
        return False

    compact = _compact_palette(code)
    compact_out = _paint_with_palette(g, sep_r, compact, dist)
    if np.array_equal(out, compact_out):
        return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(16, 26)
    w = rng.randint(16, 26)
    main_h = h - 2
    canvas = np.zeros((main_h, w), dtype=int)

    seed_r = rng.randint(5, main_h - 6)
    seed_c = rng.randint(2, max(3, w // 3))
    protected = {(seed_r + dr, seed_c + dc) for dr in range(-1, 2) for dc in range(-1, 2)}

    wall_w = rng.randint(1, 2)
    bar_c = rng.randint(seed_c + 4, w - wall_w - 4)
    top_gap = rng.randint(1, 4)
    bottom_gap = rng.randint(1, 4)
    canvas[top_gap : main_h - bottom_gap, bar_c : bar_c + wall_w] = WALL

    frame_h = rng.randint(4, 6)
    frame_w = rng.randint(4, 6)
    frame_r0_lo = max(1, seed_r - frame_h + 1)
    frame_r0_hi = min(main_h - frame_h - 1, seed_r + 1)
    if frame_r0_lo > frame_r0_hi:
        return None
    frame_r0 = rng.randint(frame_r0_lo, frame_r0_hi)
    frame_c0_lo = max(1, seed_c + 1)
    frame_c0_hi = min(bar_c - frame_w - 1, seed_c + 3)
    if frame_c0_lo > frame_c0_hi:
        return None
    frame_c0 = rng.randint(frame_c0_lo, frame_c0_hi)
    _place_rect_frame(canvas, frame_r0, frame_c0, frame_h, frame_w)

    extra_count = rng.randint(1, 3)
    for _ in range(extra_count):
        patch = _extra_wall_patch(rng, main_h, w)
        placed = False
        for _ in range(30):
            r0 = rng.randint(1, main_h - patch.shape[0] - 1)
            c0 = rng.randint(1, w - patch.shape[1] - 1)
            if not _can_paste(canvas, patch, r0, c0, protected):
                continue
            if c0 <= bar_c + wall_w and c0 + patch.shape[1] >= bar_c:
                continue
            _paste_walls(canvas, patch, r0, c0)
            placed = True
            break
        if not placed:
            return None

    if int(canvas[seed_r, seed_c]) != BG:
        return None

    dist_main = _distance_map(np.vstack([np.zeros((2, w), dtype=int), canvas]), 1, (seed_r + 2, seed_c))[2:, :]
    reachable_max = int(dist_main.max())
    if reachable_max < 8:
        return None

    separator, blank_idx, code = _sample_palette(rng, reachable_max)
    canvas[seed_r, seed_c] = int(code[0])

    inp = np.zeros((h, w), dtype=int)
    inp[0, : len(code)] = code
    inp[1, :] = separator
    inp[2:, :] = canvas

    if np.sum(inp == BG) <= np.sum(inp != BG):
        return None
    if not _semantic_checks(inp, code, blank_idx):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
