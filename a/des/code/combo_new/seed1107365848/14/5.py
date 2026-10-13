from __future__ import annotations

import random

import numpy as np


BLACK = 0
ORANGE = 7
FRAME_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)

GRID_HEIGHT_RANGE = (12, 30)
GRID_WIDTH_RANGE = (12, 30)
FRAME_COUNT_BOUNDS = (1, 4)
SEED_SIZE_RANGE = (2, 4)
REPEAT_SIZE_RANGE = (2, 3)
PLACEMENT_TRIES = 200
GENERATION_TRIES = 500


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))

    if set(coords) != expected:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _extract_seed(interior: np.ndarray) -> np.ndarray:
    orange = interior == ORANGE
    rows, cols = np.where(orange)
    assert rows.size > 0, "each frame interior must contain an orange seed"

    seed_h = int(rows.max()) + 1
    seed_w = int(cols.max()) + 1
    seed = orange[:seed_h, :seed_w].copy()

    assert interior.shape[0] % seed_h == 0, "seed height must divide the interior height"
    assert interior.shape[1] % seed_w == 0, "seed width must divide the interior width"
    return seed


def _tile_seed(seed: np.ndarray, shape: tuple[int, int], color: int) -> np.ndarray:
    reps = (shape[0] // seed.shape[0], shape[1] // seed.shape[1])
    tiled = np.tile(seed, reps)
    return np.where(tiled, int(color), BLACK)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect each frame's upper-left seed pattern and tile it across the frame interior in the frame color.

    Input structure:
    - The background is Black.
    - One or more non-overlapping one-pixel-thick rectangular frames appear in various colors.
    - Inside each frame, the upper-left corner of the black interior contains a small Orange/Black seed pattern.
    - The seed pattern's height and width each divide that frame interior's height and width.

    Rule:
    - Find every monochrome component that is exactly a hollow rectangular frame.
    - For each frame, look inside its interior and take the top-left-anchored minimal rectangle that contains all Orange seed pixels.
    - Periodically tile that binary seed over the whole interior, replacing Orange positions with the frame color and keeping Black positions Black.
    - Leave the frame border itself unchanged.

    Key insight:
    - The Orange pixels are not the final color; they define a reusable binary motif whose geometry must be extracted first and then repainted using the surrounding frame color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frame_mask = (g != BLACK) & (g != ORANGE)
    for comp in _components(frame_mask):
        color = int(g[comp[0]])
        if any(int(g[r, c]) != color for r, c in comp):
            continue
        bbox = _frame_bbox(comp)
        if bbox is None:
            continue

        r0, c0, r1, c1 = bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0:
            continue
        seed = _extract_seed(interior)
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_seed(seed, interior.shape, color)

    return out


def _sample_seed_shape() -> tuple[int, int]:
    return (
        random.randint(SEED_SIZE_RANGE[0], SEED_SIZE_RANGE[1]),
        random.randint(SEED_SIZE_RANGE[0], SEED_SIZE_RANGE[1]),
    )


def _sample_repeats() -> tuple[int, int]:
    mode = random.choice(("both", "vertical", "horizontal"))
    if mode == "both":
        return (
            random.randint(REPEAT_SIZE_RANGE[0], REPEAT_SIZE_RANGE[1]),
            random.randint(REPEAT_SIZE_RANGE[0], REPEAT_SIZE_RANGE[1]),
        )
    if mode == "vertical":
        return random.randint(REPEAT_SIZE_RANGE[0], REPEAT_SIZE_RANGE[1]), 1
    return 1, random.randint(REPEAT_SIZE_RANGE[0], REPEAT_SIZE_RANGE[1])


def _random_seed(seed_h: int, seed_w: int) -> np.ndarray:
    while True:
        seed = np.random.randint(0, 2, size=(seed_h, seed_w), dtype=int).astype(bool)
        if not seed.any() or seed.all():
            continue
        if not seed[0, :].any() or not seed[-1, :].any():
            continue
        if not seed[:, 0].any() or not seed[:, -1].any():
            continue
        return seed


def _build_frame_patch(color: int, seed: np.ndarray, rep_h: int, rep_w: int) -> np.ndarray:
    seed_h, seed_w = seed.shape
    inner_h = seed_h * rep_h
    inner_w = seed_w * rep_w
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    patch[1 : 1 + seed_h, 1 : 1 + seed_w][seed] = ORANGE
    return patch


def _max_frame_count(height: int, width: int) -> int:
    area = int(height * width)
    if area < 220:
        return 2
    if area < 420:
        return 3
    return FRAME_COUNT_BOUNDS[1]


def _place_patch(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if ph > h - 2 or pw > w - 2:
        return False

    for _ in range(PLACEMENT_TRIES):
        r0 = random.randint(1, h - ph - 1)
        c0 = random.randint(1, w - pw - 1)
        rs = r0 - 1
        cs = c0 - 1
        re = r0 + ph + 1
        ce = c0 + pw + 1
        if blocked[rs:re, cs:ce].any():
            continue
        grid[r0 : r0 + ph, c0 : c0 + pw] = np.where(patch != BLACK, patch, grid[r0 : r0 + ph, c0 : c0 + pw])
        blocked[rs:re, cs:ce] = True
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1])
    width = random.randint(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1])
    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)

    frame_count = random.randint(FRAME_COUNT_BOUNDS[0], _max_frame_count(height, width))
    colors = random.sample(FRAME_COLORS, frame_count)

    patches: list[np.ndarray] = []
    for color in colors:
        seed_h, seed_w = _sample_seed_shape()
        rep_h, rep_w = _sample_repeats()
        seed = _random_seed(seed_h, seed_w)
        patches.append(_build_frame_patch(color, seed, rep_h, rep_w))

    patches.sort(key=lambda patch: patch.shape[0] * patch.shape[1], reverse=True)
    for patch in patches:
        if not _place_patch(grid, blocked, patch):
            return None

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
