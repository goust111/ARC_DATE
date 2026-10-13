from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
MAROON = 9
LEGEND_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.array(pts, dtype=int))
    return comps


def _legend_colors(grid: np.ndarray) -> list[int]:
    assert grid.shape[0] >= 2, "expected at least two rows for the legend"
    colors: list[int] = []
    for c in range(grid.shape[1]):
        top = int(grid[-2, c])
        bot = int(grid[-1, c])
        if top == bot and top != BG:
            colors.append(top)
    assert colors, "expected at least one legend color"
    return colors


def _interior_regions(grid: np.ndarray) -> list[np.ndarray]:
    work = grid[:-2, :]
    passable = work != GREY
    regions: list[np.ndarray] = []
    h, w = map(int, work.shape)
    for comp in _components(passable):
        touches_edge = np.any(
            (comp[:, 0] == 0)
            | (comp[:, 1] == 0)
            | (comp[:, 0] == h - 1)
            | (comp[:, 1] == w - 1)
        )
        if not touches_edge:
            regions.append(comp)
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect each enclosed shape interior, count its maroon defects, and fill the
    entire interior using the nth legend color from the bottom palette.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3, "expected a 2D grid with a bottom legend"

    legend = _legend_colors(g)
    out = g.copy()
    for region in _interior_regions(g):
        defect_count = int(np.sum(g[region[:, 0], region[:, 1]] == MAROON))
        assert 1 <= defect_count <= len(legend), "defect count must index the legend"
        out[region[:, 0], region[:, 1]] = int(legend[defect_count - 1])
    return out


def _boundary_from_filled(mask: np.ndarray) -> np.ndarray:
    h, w = map(int, mask.shape)
    boundary = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                    boundary[r, c] = True
                    break
    return boundary


def _rectangle_mask() -> np.ndarray:
    h = random.randint(4, 8)
    w = random.randint(4, 8)
    return np.ones((h, w), dtype=bool)


def _l_mask() -> np.ndarray:
    h = random.randint(7, 10)
    w = random.randint(7, 10)
    t = 3
    orient = random.choice(("tl", "tr", "bl", "br"))
    mask = np.zeros((h, w), dtype=bool)
    if orient == "tl":
        mask[:, :t] = True
        mask[:t, :] = True
    elif orient == "tr":
        mask[:, -t:] = True
        mask[:t, :] = True
    elif orient == "bl":
        mask[:, :t] = True
        mask[-t:, :] = True
    else:
        mask[:, -t:] = True
        mask[-t:, :] = True
    return mask


def _t_mask() -> np.ndarray:
    orient = random.choice(("up", "down", "left", "right"))
    if orient in ("up", "down"):
        h = random.randint(7, 10)
        w = random.randint(7, 11)
        if w % 2 == 0:
            w += 1
        bar = 3
        stem = 3
        c0 = w // 2 - stem // 2
        mask = np.zeros((h, w), dtype=bool)
        mask[:, c0 : c0 + stem] = True
        if orient == "up":
            mask[:bar, :] = True
        else:
            mask[-bar:, :] = True
        return mask

    h = random.randint(7, 11)
    w = random.randint(7, 10)
    if h % 2 == 0:
        h += 1
    bar = 3
    stem = 3
    r0 = h // 2 - stem // 2
    mask = np.zeros((h, w), dtype=bool)
    mask[r0 : r0 + stem, :] = True
    if orient == "left":
        mask[:, :bar] = True
    else:
        mask[:, -bar:] = True
    return mask


def _sample_shape_mask() -> np.ndarray:
    builders = (_rectangle_mask, _l_mask, _t_mask)
    for _ in range(100):
        mask = random.choice(builders)()
        boundary = _boundary_from_filled(mask)
        interior = mask & (~boundary)
        if int(interior.sum()) >= 5:
            return mask
    raise RuntimeError("failed to build a shape with enough interior area")


def _shape_sprite(defect_count: int) -> tuple[np.ndarray, np.ndarray]:
    filled = _sample_shape_mask()
    boundary = _boundary_from_filled(filled)
    interior = filled & (~boundary)
    cells = [tuple(map(int, pt)) for pt in np.argwhere(interior)]
    maroon_cells = set(random.sample(cells, defect_count))

    sprite = np.full(filled.shape, BG, dtype=int)
    sprite[boundary] = GREY
    for r, c in cells:
        if (r, c) in maroon_cells:
            sprite[r, c] = MAROON
    return sprite, interior


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def _legend_patch(colors: list[int], width: int) -> tuple[np.ndarray, int]:
    patch = np.full((2, width), BG, dtype=int)
    start = random.randint(1, width - (2 * len(colors) - 1) - 1)
    for i, color in enumerate(colors):
        c = start + 2 * i
        patch[:, c] = int(color)
    return patch, start


def _attempt_generate() -> tuple[np.ndarray, np.ndarray] | None:
    n_shapes = random.randint(2, 4)
    defect_counts = [random.randint(1, 5) for _ in range(n_shapes)]
    legend_colors = random.sample(LEGEND_COLOR_POOL, 5)

    sprites: list[tuple[np.ndarray, np.ndarray, int]] = []
    max_h = 0
    max_w = 0
    for count in defect_counts:
        sprite, interior = _shape_sprite(count)
        sprites.append((sprite, interior, count))
        max_h = max(max_h, int(sprite.shape[0]))
        max_w = max(max_w, int(sprite.shape[1]))

    min_h = max(16, n_shapes * 4 + max_h + 3)
    min_w = max(18, n_shapes * 5 + max_w + 4, 2 * len(legend_colors) + 3)
    h = random.randint(min_h, min_h + 8)
    w = random.randint(min_w, min_w + 10)
    main_h = h - 2
    if main_h - 1 < max_h:
        return None

    grid = np.full((h, w), BG, dtype=int)
    legend, _ = _legend_patch(legend_colors, w)
    grid[-2:, :] = legend

    expected = grid.copy()
    placed_boxes: list[tuple[int, int, int, int]] = []

    order = list(range(n_shapes))
    random.shuffle(order)
    for idx in order:
        sprite, interior, count = sprites[idx]
        sh, sw = map(int, sprite.shape)
        placed = False
        for _ in range(200):
            r0 = random.randint(1, main_h - sh - 2)
            c0 = random.randint(1, w - sw - 2)
            box = (r0 - 1, c0 - 1, r0 + sh, c0 + sw)
            if any(_boxes_overlap(box, other) for other in placed_boxes):
                continue
            placed_boxes.append(box)
            placed = True

            patch = grid[r0 : r0 + sh, c0 : c0 + sw]
            assert np.all(patch == BG)
            patch[sprite != BG] = sprite[sprite != BG]

            fill_color = legend_colors[count - 1]
            exp_patch = expected[r0 : r0 + sh, c0 : c0 + sw]
            exp_patch[sprite != BG] = sprite[sprite != BG]
            exp_patch[interior] = fill_color
            break
        if not placed:
            return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(300):
        pair = _attempt_generate()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate a valid input")
