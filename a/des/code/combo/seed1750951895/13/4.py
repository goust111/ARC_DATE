import random
from collections import Counter

import numpy as np


BLACK = 0
BG = 5
PALETTE = tuple(c for c in range(10) if c != BG)
DIRECTIONS = ("up", "down", "left", "right")


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _infer_arrow(coords: np.ndarray) -> tuple[str, int]:
    assert len(coords) == 6
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    h, w = r1 - r0 + 1, c1 - c0 + 1
    assert (h, w) in {(4, 3), (3, 4)}

    row_counts = Counter(map(int, rows))
    col_counts = Counter(map(int, cols))

    if h == 4:
        shaft_cols = [c for c, n in col_counts.items() if n == 4]
        bar_rows = [r for r, n in row_counts.items() if n == 3]
        assert len(shaft_cols) == 1 and len(bar_rows) == 1
        shaft_col = int(shaft_cols[0])
        bar_row = int(bar_rows[0])
        assert bar_row in (r0, r1)
        return ("up" if bar_row == r0 else "down"), shaft_col

    shaft_rows = [r for r, n in row_counts.items() if n == 4]
    bar_cols = [c for c, n in col_counts.items() if n == 3]
    assert len(shaft_rows) == 1 and len(bar_cols) == 1
    shaft_row = int(shaft_rows[0])
    bar_col = int(bar_cols[0])
    assert bar_col in (c0, c1)
    return ("left" if bar_col == c0 else "right"), shaft_row


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color frame pixels indicated by each asymmetric T-shaped arrow sprite.

    Input structure:
    - Background is grey.
    - The outer boundary is a one-cell-thick colored frame with grey corners.
    - Each interior sprite is a single-color 6-pixel arrow: a length-4 line plus
      a length-3 crossbar attached at one end.

    Rule:
    - Infer the arrow direction from which end carries the crossbar.
    - Paint the matching frame side at the sprite's row/column using the sprite color.
    - Paint the opposite frame side at the same row/column using black.
    - Preserve everything else.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 6 and w >= 6

    bg = int(g[0, 0])
    assert bg == int(g[0, -1]) == int(g[-1, 0]) == int(g[-1, -1])

    out = g.copy()
    inner = g[1:-1, 1:-1]
    comps = _components(inner != bg)

    for comp in comps:
        full = comp + 1
        color = int(g[full[0, 0], full[0, 1]])
        assert np.all(g[full[:, 0], full[:, 1]] == color)
        direction, pos = _infer_arrow(full)
        if direction == "up":
            out[0, pos] = color
            out[-1, pos] = BLACK
        elif direction == "down":
            out[-1, pos] = color
            out[0, pos] = BLACK
        elif direction == "left":
            out[pos, 0] = color
            out[pos, -1] = BLACK
        else:
            out[pos, -1] = color
            out[pos, 0] = BLACK

    return out


def _sprite_cells(direction: str, top: int, left: int) -> tuple[list[tuple[int, int]], int]:
    if direction == "up":
        cells = [(top + r, left + 1) for r in range(4)] + [(top, left + c) for c in range(3)]
        return cells, left + 1
    if direction == "down":
        cells = [(top + r, left + 1) for r in range(4)] + [(top + 3, left + c) for c in range(3)]
        return cells, left + 1
    if direction == "left":
        cells = [(top + 1, left + c) for c in range(4)] + [(top + r, left) for r in range(3)]
        return cells, top + 1
    cells = [(top + 1, left + c) for c in range(4)] + [(top + r, left + 3) for r in range(3)]
    return cells, top + 1


def _can_place(blocked: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = blocked.shape
    r0 = max(1, top - 1)
    c0 = max(1, left - 1)
    r1 = min(h - 2, top + height)
    c1 = min(w - 2, left + width)
    return not blocked[r0 : r1 + 1, c0 : c1 + 1].any()


def _mark_blocked(blocked: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    h, w = blocked.shape
    r0 = max(1, top - 1)
    c0 = max(1, left - 1)
    r1 = min(h - 2, top + height)
    c1 = min(w - 2, left + width)
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(10, 18)
    w = random.randint(10, 18)
    grid = np.full((h, w), BG, dtype=int)

    frame_colors = random.sample([c for c in PALETTE if c != BLACK], 4)
    grid[0, 1:-1] = frame_colors[0]
    grid[-1, 1:-1] = frame_colors[1]
    grid[1:-1, 0] = frame_colors[2]
    grid[1:-1, -1] = frame_colors[3]

    max_vertical = max(1, w - 4)
    max_horizontal = max(1, h - 4)
    max_sprites = min(6, max_vertical + max_horizontal, ((h - 2) * (w - 2)) // 12)
    if max_sprites < 2:
        return None
    count = random.randint(2, max_sprites)

    dirs = random.sample(DIRECTIONS, k=min(len(DIRECTIONS), count))
    while len(dirs) < count:
        dirs.append(random.choice(DIRECTIONS))
    random.shuffle(dirs)

    blocked = np.zeros((h, w), dtype=bool)
    used_vertical: set[int] = set()
    used_horizontal: set[int] = set()
    sprite_colors = [random.choice([c for c in PALETTE if c not in frame_colors]) for _ in range(count)]

    for direction, color in zip(dirs, sprite_colors):
        placed = False
        if direction in ("up", "down"):
            cols = list(range(2, w - 2))
            random.shuffle(cols)
            rows = list(range(1, h - 4))
            random.shuffle(rows)
            for shaft_col in cols:
                if shaft_col in used_vertical:
                    continue
                left = shaft_col - 1
                for top in rows:
                    if _can_place(blocked, top, left, 4, 3):
                        for r, c in _sprite_cells(direction, top, left)[0]:
                            grid[r, c] = color
                        _mark_blocked(blocked, top, left, 4, 3)
                        used_vertical.add(shaft_col)
                        placed = True
                        break
                if placed:
                    break
        else:
            rows = list(range(2, h - 2))
            random.shuffle(rows)
            cols = list(range(1, w - 4))
            random.shuffle(cols)
            for shaft_row in rows:
                if shaft_row in used_horizontal:
                    continue
                top = shaft_row - 1
                for left in cols:
                    if _can_place(blocked, top, left, 3, 4):
                        for r, c in _sprite_cells(direction, top, left)[0]:
                            grid[r, c] = color
                        _mark_blocked(blocked, top, left, 3, 4)
                        used_horizontal.add(shaft_row)
                        placed = True
                        break
                if placed:
                    break
        if not placed:
            return None

    comps = _components(grid[1:-1, 1:-1] != BG)
    if len(comps) != count:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
