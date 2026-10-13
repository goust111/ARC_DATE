from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = DIRS4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def label_mask(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    dirs = DIRS4 if connectivity == 4 else DIRS8
    current = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            current += 1
            stack = [(r, c)]
            labels[r, c] = current
            while stack:
                rr, cc = stack.pop()
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = current
                        stack.append((nr, nc))
    return labels, current


def largest_enclosed_component_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = mask.shape
    labels, count = label_mask(mask, connectivity=4)
    best = None
    best_area = -1
    for idx in range(1, count + 1):
        pts = np.argwhere(labels == idx)
        if pts.size == 0:
            continue
        rs = pts[:, 0]
        cs = pts[:, 1]
        if (rs == 0).any() or (rs == h - 1).any() or (cs == 0).any() or (cs == w - 1).any():
            continue
        area = int(len(pts))
        if area > best_area:
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            best = (int(r0), int(c0), int(r1), int(c1))
            best_area = area
    return best


def _detect_sprites(grid: np.ndarray, bg: int, frame_left: int) -> list[dict]:
    h, w = grid.shape
    divider_mask = np.zeros((h, w), dtype=bool)
    for c in range(frame_left):
        col = grid[:, c]
        non_bg = col[col != bg]
        if len(non_bg) >= h - 1 and len(set(map(int, non_bg))) == 1:
            divider_mask[:, c] = True

    left_mask = np.arange(w)[None, :] < frame_left
    sprites = []
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        mask = (grid == color) & left_mask & (~divider_mask)
        if not mask.any():
            continue
        labels, count = label_mask(mask, connectivity=8)
        for idx in range(1, count + 1):
            pts = np.argwhere(labels == idx)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            local = labels[r0 : r1 + 1, c0 : c1 + 1] == idx
            sprites.append(
                {
                    "color": int(color),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "mask": local,
                }
            )
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored sprite from the left strip into the framed area.

    For a color C, every interior row whose left/right frame cells are both C is a
    target row, and every interior column whose top/bottom frame cells are both C is
    a target column. The sprite of color C is erased from the left and stamped,
    centered, at every row/column intersection. Higher source sprites are stamped
    later, so they cover lower ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)

    inner = largest_enclosed_component_bbox(g == bg)
    assert inner is not None
    ir0, ic0, ir1, ic1 = inner
    fr0, fc0, fr1, fc1 = ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1
    assert 0 <= fr0 < fr1 < g.shape[0]
    assert 0 <= fc0 < fc1 < g.shape[1]

    sprites = _detect_sprites(g, bg, fc0)
    move_colors = {sprite["color"] for sprite in sprites}

    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)
    for r in range(ir0, ir1 + 1):
        left = int(g[r, fc0])
        right = int(g[r, fc1])
        if left == right and left != bg and left in move_colors:
            row_marks[left].append(r)
    for c in range(ic0, ic1 + 1):
        top = int(g[fr0, c])
        bottom = int(g[fr1, c])
        if top == bottom and top != bg and top in move_colors:
            col_marks[top].append(c)

    targets = {
        color: [(r, c) for r in row_marks[color] for c in col_marks[color]]
        for color in sorted(set(row_marks) & set(col_marks))
    }

    out = g.copy()
    movers = [sprite for sprite in sprites if sprite["color"] in targets]
    for sprite in movers:
        for rr, cc in np.argwhere(sprite["mask"]):
            out[sprite["r0"] + int(rr), sprite["c0"] + int(cc)] = bg

    movers.sort(key=lambda item: (((item["r0"] + item["r1"]) / 2.0), item["c0"]))
    for sprite in movers:
        color = sprite["color"]
        center_r = (sprite["r0"] + sprite["r1"]) // 2
        center_c = (sprite["c0"] + sprite["c1"]) // 2
        pixels = [(sprite["r0"] + int(rr), sprite["c0"] + int(cc)) for rr, cc in np.argwhere(sprite["mask"])]
        for tr, tc in targets[color]:
            dr = tr - center_r
            dc = tc - center_c
            for r, c in pixels:
                nr, nc = r + dr, c + dc
                if ir0 <= nr <= ir1 and ic0 <= nc <= ic1:
                    out[nr, nc] = color

    return out


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    if not candidates:
        return None
    order = candidates[:]
    random.shuffle(order)
    picked: list[int] = []
    for value in order:
        if all(abs(value - prev) >= min_spacing for prev in picked):
            picked.append(value)
            if len(picked) == n:
                return sorted(picked)
    return sorted(picked) if picked else None


def _make_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    sh = random.randint(2, max(2, min(5, max_h)))
    sw = random.randint(2, max(2, min(5, max_w)))
    if sh > max_h or sw > max_w:
        return None

    target_cells = random.randint(max(3, (sh * sw) // 3), min(sh * sw, max(5, (2 * sh * sw) // 3)))
    cells = {(sh // 2, sw // 2)}
    frontier = [(sh // 2, sw // 2)]
    while frontier and len(cells) < target_cells:
        r, c = random.choice(frontier)
        dr, dc = random.choice(DIRS8)
        nr, nc = r + dr, c + dc
        if 0 <= nr < sh and 0 <= nc < sw and (nr, nc) not in cells:
            cells.add((nr, nc))
            frontier.append((nr, nc))
        if len(frontier) > sh * sw * 2:
            frontier = frontier[-sh * sw :]

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in cells:
        sprite[r - r0, c - c0] = color
    return sprite


def _stamp(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    mask = patch != 0
    view = grid[top : top + h, left : left + w]
    view[mask] = patch[mask]


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(list(range(10)))
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)

    interior_h = random.randint(8, 12)
    interior_w = random.randint(8, 12)
    frame_h = interior_h + 2
    frame_w = interior_w + 2

    top_margin = random.randint(0, 2)
    bottom_margin = random.randint(0, 2)
    sprite_w = random.randint(4, 6)
    divider_w = random.randint(1, 2)

    h = top_margin + frame_h + bottom_margin
    w = sprite_w + divider_w + frame_w
    if not (10 <= h <= 18 and 14 <= w <= 22):
        return None

    frame_top = top_margin
    frame_left = sprite_w + divider_w
    frame_bottom = frame_top + frame_h - 1
    frame_right = frame_left + frame_w - 1
    ir0, ic0 = frame_top + 1, frame_left + 1
    ir1, ic1 = frame_bottom - 1, frame_right - 1

    if ir1 <= ir0 or ic1 <= ic0:
        return None

    divider_color = palette.pop()
    frame_color = palette.pop()

    mover_count = random.randint(2, 4)
    distractor_count = random.randint(0, 2)
    if len(palette) < mover_count + distractor_count:
        return None
    mover_colors = palette[:mover_count]
    distractor_colors = palette[mover_count : mover_count + distractor_count]

    sprites: list[np.ndarray] = []
    for color in mover_colors:
        sprite = _make_sprite(color, max_h=5, max_w=sprite_w)
        if sprite is None:
            return None
        sprites.append(sprite)

    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()
    for color, sprite in zip(mover_colors, sprites):
        sh, sw = sprite.shape
        row_lo = ir0 + sh // 2
        row_hi = ir1 - (sh - 1 - sh // 2)
        col_lo = ic0 + sw // 2
        col_hi = ic1 - (sw - 1 - sw // 2)
        if row_lo > row_hi or col_lo > col_hi:
            return None

        row_candidates = [r for r in range(row_lo, row_hi + 1) if r not in occupied_rows]
        col_candidates = [c for c in range(col_lo, col_hi + 1) if c not in occupied_cols]
        row_count = random.randint(1, 2)
        col_count = random.randint(1, 2)
        row_marks = _pick_spaced(row_candidates, row_count, max(2, sh))
        col_marks = _pick_spaced(col_candidates, col_count, max(2, sw))
        if not row_marks or len(row_marks) != row_count:
            return None
        if not col_marks or len(col_marks) != col_count:
            return None
        mover_targets[color] = (row_marks, col_marks)
        occupied_rows.update(row_marks)
        occupied_cols.update(col_marks)

    grid = np.full((h, w), bg, dtype=int)
    grid[:, sprite_w:frame_left] = divider_color

    grid[frame_top, frame_left : frame_right + 1] = frame_color
    grid[frame_bottom, frame_left : frame_right + 1] = frame_color
    grid[frame_top : frame_bottom + 1, frame_left] = frame_color
    grid[frame_top : frame_bottom + 1, frame_right] = frame_color

    for color, (rows, cols) in mover_targets.items():
        for r in rows:
            grid[r, frame_left] = color
            grid[r, frame_right] = color
        for c in cols:
            grid[frame_top, c] = color
            grid[frame_bottom, c] = color

    items: list[tuple[int, np.ndarray]] = list(zip(mover_colors, sprites))
    for color in distractor_colors:
        sprite = _make_sprite(color, max_h=4, max_w=sprite_w)
        if sprite is None:
            return None
        items.append((color, sprite))
    random.shuffle(items)

    cur_row = 0
    for _, sprite in items:
        sh, sw = sprite.shape
        if cur_row + sh > h:
            return None
        left = random.randint(0, sprite_w - sw)
        _stamp(grid, sprite, cur_row, left)
        cur_row += sh
        if cur_row < h and random.random() < 0.35:
            cur_row += 1
        if cur_row > h:
            return None

    if get_bg(grid) != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
