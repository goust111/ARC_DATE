from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_HELPER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_HELPER_DIR) not in sys.path:
    sys.path.append(str(_HELPER_DIR))

from arc_utils import Color, crop_bbox, find_components, paste


GEN_TRIES = 500
MOTIF_COLORS = tuple(c for c in Color.NOT_BLACK if c != Color.RED)


def _placement(pivot_r: int, pivot_c: int, distance: int, patch_shape: tuple[int, int], step: int) -> tuple[int, int]:
    h, w = map(int, patch_shape)
    phase = int(step) % 4
    if phase == 0:  # east: left edge is `distance` to the right of the pivot
        return int(pivot_r), int(pivot_c + distance)
    if phase == 1:  # south: top edge is `distance` below the pivot
        return int(pivot_r + distance), int(pivot_c)
    if phase == 2:  # west: right edge is `distance` to the left of the pivot
        return int(pivot_r), int(pivot_c - distance - w + 1)
    return int(pivot_r - distance - h + 1), int(pivot_c)  # north


def _fits(shape: tuple[int, int], r0: int, c0: int, patch_shape: tuple[int, int]) -> bool:
    h, w = map(int, patch_shape)
    H, W = map(int, shape)
    return 0 <= int(r0) and 0 <= int(c0) and int(r0) + h <= H and int(c0) + w <= W


def _trace_spiral(shape: tuple[int, int], pivot_r: int, pivot_c: int, seed: np.ndarray, d0: int, delta: int):
    occupied = np.zeros(shape, dtype=bool)
    occupied[int(pivot_r), int(pivot_c)] = True
    placements = []
    for step in range(64):
        patch = np.rot90(seed, -step)
        distance = int(d0 + step * delta)
        r0, c0 = _placement(pivot_r, pivot_c, distance, patch.shape, step)
        if not _fits(shape, r0, c0, patch.shape):
            return placements
        mask = patch != int(Color.BLACK)
        window = occupied[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        if np.any(window & mask):
            return None
        window |= mask
        placements.append((int(step), int(r0), int(c0), patch.copy()))
    return None


def _extract_rule(grid: np.ndarray) -> tuple[tuple[int, int], np.ndarray, int, int]:
    red_pixels = np.argwhere(grid == int(Color.RED))
    assert red_pixels.shape == (1, 2), "expected exactly one red pivot pixel"
    pivot_r, pivot_c = map(int, red_pixels[0])

    comps = find_components(grid, bg=int(Color.BLACK), connectivity=4, monochromatic=False, return_format="dict")
    motif_comps = []
    for comp in comps:
        coords = comp["coords"]
        if coords.shape[0] == 1 and tuple(map(int, coords[0])) == (pivot_r, pivot_c):
            continue
        motif_comps.append(comp)
    assert len(motif_comps) == 2, "expected exactly two motif copies in the input"

    for east in motif_comps:
        er0, ec0, _er1, _ec1 = map(int, east["bbox"])
        if er0 != pivot_r or ec0 <= pivot_c:
            continue
        east_patch = crop_bbox(grid, east["bbox"])
        for south in motif_comps:
            if south is east:
                continue
            sr0, sc0, _sr1, _sc1 = map(int, south["bbox"])
            if sc0 != pivot_c or sr0 <= pivot_r:
                continue
            south_patch = crop_bbox(grid, south["bbox"])
            if np.array_equal(np.rot90(east_patch, -1), south_patch):
                d0 = int(ec0 - pivot_c)
                delta = int(sr0 - pivot_r - d0)
                assert delta > 0, "radius increment must be positive"
                return (pivot_r, pivot_c), east_patch, d0, delta
    raise AssertionError("failed to identify the east/south seed copies")


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Continue the rotating outward spiral implied by the two seed motif copies.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"
    (pivot_r, pivot_c), seed, d0, delta = _extract_rule(grid)

    out = grid.copy()
    step = 2
    while True:
        patch = np.rot90(seed, -step)
        distance = int(d0 + step * delta)
        r0, c0 = _placement(pivot_r, pivot_c, distance, patch.shape, step)
        if not _fits(out.shape, r0, c0, patch.shape):
            return out
        paste(out, patch, r0, c0, transparent=int(Color.BLACK))
        step += 1


def _random_connected_patch() -> np.ndarray | None:
    h = random.randint(2, 4)
    w = random.randint(2, 4)
    target_area = random.randint(3, min(h * w, 7))

    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < target_area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    hh = r1 - r0 + 1
    ww = c1 - c0 + 1
    if hh < 2 or ww < 2:
        return None

    patch = np.full((hh, ww), int(Color.BLACK), dtype=int)
    n_colors = random.randint(2, min(4, len(cells)))
    palette = random.sample(MOTIF_COLORS, n_colors)
    shifted = [(r - r0, c - c0) for r, c in cells]
    random.shuffle(shifted)
    for idx, (r, c) in enumerate(shifted):
        patch[r, c] = int(palette[idx % n_colors] if idx < n_colors else random.choice(palette))

    if np.array_equal(patch, np.rot90(patch, -1)):
        return None
    return patch


def _gen_attempt() -> np.ndarray | None:
    seed = _random_connected_patch()
    if seed is None:
        return None

    H = random.randint(14, 30)
    W = random.randint(14, 30)
    pivot_r = random.randint(1, H - 2)
    pivot_c = random.randint(1, W - 2)

    base_min = max(seed.shape)
    d0 = random.randint(base_min, base_min + 5)
    delta = random.randint(1, 4)

    placements = _trace_spiral((H, W), pivot_r, pivot_c, seed, d0, delta)
    if placements is None or len(placements) < 3:
        return None

    inp = np.full((H, W), int(Color.BLACK), dtype=int)
    inp[pivot_r, pivot_c] = int(Color.RED)
    for _step, r0, c0, patch in placements[:2]:
        paste(inp, patch, r0, c0, transparent=int(Color.BLACK))

    out = main(inp)
    if out.ndim != 2 or out.shape[0] > 30 or out.shape[1] > 30:
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
