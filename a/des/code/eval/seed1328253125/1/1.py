import random
from collections import deque

import numpy as np


BG = 0
DIR_TO_CODE = {
    "up": 2,     # red
    "right": 1,  # blue
    "down": 3,   # green
    "left": 4,   # yellow
}
CODE_TO_DIR = {value: key for key, value in DIR_TO_CODE.items()}
GAP_BY_DIR = {
    "up": (0, 2),
    "right": (2, 4),
    "down": (4, 2),
    "left": (2, 0),
}
ALL_DIRS = ("up", "right", "down", "left")
SPRITE_SIZE = 5
INTERIOR_SLICE = slice(1, 4)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _sprite_pattern(direction: str) -> set[tuple[int, int]]:
    border = {
        (r, c)
        for r in range(SPRITE_SIZE)
        for c in range(SPRITE_SIZE)
        if r in (0, SPRITE_SIZE - 1) or c in (0, SPRITE_SIZE - 1)
    }
    border.remove(GAP_BY_DIR[direction])
    return border


SPRITE_PATTERNS = {direction: _sprite_pattern(direction) for direction in ALL_DIRS}


def _find_palette(grid: np.ndarray) -> tuple[dict[str, int], np.ndarray]:
    h, w = grid.shape
    palette_candidates: list[tuple[dict[str, int], tuple[int, int, int, int]]] = []
    for comp in _components(grid != BG):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if not (r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1):
            continue
        if c1 - c0 + 1 != 2:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(sub == BG):
            continue
        if len(comp) != int(sub.size):
            continue

        mapping: dict[str, int] = {}
        ok = True
        for code, fill in sub.tolist():
            if code not in CODE_TO_DIR:
                ok = False
                break
            direction = CODE_TO_DIR[int(code)]
            if direction in mapping:
                ok = False
                break
            mapping[direction] = int(fill)
        if ok:
            palette_candidates.append((mapping, (r0, c0, r1, c1)))

    assert len(palette_candidates) == 1, "expected exactly one edge palette"
    mapping, (r0, c0, r1, c1) = palette_candidates[0]
    palette_mask = np.zeros_like(grid, dtype=bool)
    palette_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mapping, palette_mask


def _find_sprites(grid: np.ndarray, palette_mask: np.ndarray) -> list[tuple[int, int, str]]:
    sprites: list[tuple[int, int, str]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        for comp in _components((grid == color) & ~palette_mask):
            if len(comp) != 15:
                continue
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if (r1 - r0 + 1, c1 - c0 + 1) != (SPRITE_SIZE, SPRITE_SIZE):
                continue
            local = {(r - r0, c - c0) for r, c in comp}
            for direction, pattern in SPRITE_PATTERNS.items():
                if local == pattern:
                    sprites.append((r0, c0, direction))
                    break
    assert sprites, "expected at least one gapped frame sprite"
    return sprites


def _paint_stripes(out: np.ndarray, r0: int, c0: int, direction: str, fill: int) -> None:
    if direction in ("up", "down"):
        out[r0 + 1, c0 + 1 : c0 + 4] = fill
        out[r0 + 3, c0 + 1 : c0 + 4] = fill
    else:
        out[r0 + 1 : r0 + 4, c0 + 1] = fill
        out[r0 + 1 : r0 + 4, c0 + 3] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the gap direction of each 5x5 frame sprite to choose a fill color from the
    edge palette, then paint stripes inside the sprite perpendicular to the gap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    mapping, palette_mask = _find_palette(g)
    sprites = _find_sprites(g, palette_mask)

    out = g.copy()
    for r0, c0, direction in sprites:
        assert direction in mapping, "missing palette entry for sprite direction"
        _paint_stripes(out, r0, c0, direction, mapping[direction])
    return out


GRID_H_RANGE = (18, 27)
GRID_W_RANGE = (18, 27)
SPRITE_COUNT_CHOICES = (4, 5)
FILL_POOL = (5, 6, 7, 8, 9)
BORDER_POOL = (5, 6, 7, 8, 9)
PALETTE_RETRIES = 400


def _sprite_patch(color: int, direction: str) -> np.ndarray:
    patch = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    gr, gc = GAP_BY_DIR[direction]
    patch[gr, gc] = BG
    return patch


def _make_output(inp: np.ndarray, sprite_specs: list[tuple[int, int, str]], mapping: dict[str, int]) -> np.ndarray:
    out = inp.copy()
    for r0, c0, direction in sprite_specs:
        _paint_stripes(out, r0, c0, direction, mapping[direction])
    return out


def _mark_occ(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    oh, ow = occ.shape
    occ[max(0, r0 - 1) : min(oh, r0 + h + 1), max(0, c0 - 1) : min(ow, c0 + w + 1)] = True


def _place_palette(grid: np.ndarray, occ: np.ndarray, palette: np.ndarray, side: str, rng: random.Random) -> bool:
    h, w = grid.shape
    ph, pw = palette.shape
    if side == "top":
        r0 = 0
        c0 = rng.randint(0, w - pw)
    elif side == "bottom":
        r0 = h - ph
        c0 = rng.randint(0, w - pw)
    elif side == "left":
        r0 = rng.randint(0, h - ph)
        c0 = 0
    elif side == "right":
        r0 = rng.randint(0, h - ph)
        c0 = w - pw
    else:
        raise ValueError(side)
    if occ[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)].any():
        return False
    grid[r0 : r0 + ph, c0 : c0 + pw] = palette
    _mark_occ(occ, r0, c0, ph, pw)
    return True


def _sprite_positions(h: int, w: int, occ: np.ndarray) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - SPRITE_SIZE):
        for c0 in range(1, w - SPRITE_SIZE):
            if occ[r0 - 1 : r0 + SPRITE_SIZE + 1, c0 - 1 : c0 + SPRITE_SIZE + 1].any():
                continue
            positions.append((r0, c0))
    return positions


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    n_sprites = rng.choice(SPRITE_COUNT_CHOICES)
    if n_sprites == 5 and min(h, w) < 19:
        return None

    fills = rng.sample(list(FILL_POOL), 4)
    fill_by_dir = {direction: fill for direction, fill in zip(ALL_DIRS, fills)}
    border_colors = rng.sample(list(BORDER_POOL), n_sprites)

    sprite_dirs = list(ALL_DIRS)
    if n_sprites == 5:
        sprite_dirs.append(rng.choice(ALL_DIRS))
    rng.shuffle(sprite_dirs)

    palette_rows = list(ALL_DIRS)
    rng.shuffle(palette_rows)
    palette = np.array([[DIR_TO_CODE[direction], fill_by_dir[direction]] for direction in palette_rows], dtype=int)

    grid = np.zeros((h, w), dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    if not _place_palette(grid, occ, palette, rng.choice(("top", "bottom", "left", "right")), rng):
        return None

    sprite_specs: list[tuple[int, int, str]] = []
    for direction, color in zip(sprite_dirs, border_colors):
        positions = _sprite_positions(h, w, occ)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        grid[r0 : r0 + SPRITE_SIZE, c0 : c0 + SPRITE_SIZE] = _sprite_patch(color, direction)
        _mark_occ(occ, r0, c0, SPRITE_SIZE, SPRITE_SIZE)
        sprite_specs.append((r0, c0, direction))

    expected = _make_output(grid, sprite_specs, fill_by_dir)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(PALETTE_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
