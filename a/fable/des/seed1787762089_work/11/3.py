import random
from collections import deque

import numpy as np


BG = 8
GREY = 5
LINE_COLOR_POOL = (1, 2, 3, 4, 6)

GRID_MIN = 18
GRID_MAX = 26
LINE_COUNT_CHOICES = (3, 4)
VALID_SPRITE_COUNT_CHOICES = (2, 3)
GEN_RETRIES = 400


def _neighbors(r: int, c: int, h: int, w: int, connectivity: int):
    if connectivity == 4:
        steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        steps = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )
    for dr, dc in steps:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
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
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, h, w, connectivity):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(points: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(points[:, 0].min()),
        int(points[:, 1].min()),
        int(points[:, 0].max()),
        int(points[:, 1].max()),
    )


def _hole_count(sprite_mask: np.ndarray) -> int:
    bg_mask = ~sprite_mask
    count = 0
    for pts in _components(bg_mask, connectivity=4):
        if np.any(pts[:, 0] == 0) or np.any(pts[:, 1] == 0):
            continue
        if np.any(pts[:, 0] == sprite_mask.shape[0] - 1):
            continue
        if np.any(pts[:, 1] == sprite_mask.shape[1] - 1):
            continue
        count += 1
    return count


def _ordered_line_colors(grid: np.ndarray) -> list[int]:
    mask = (grid != BG) & (grid != GREY)
    comps = _components(mask, connectivity=8)
    ordered = []
    for pts in comps:
        color = int(grid[pts[0, 0], pts[0, 1]])
        ordered.append((int(pts[:, 0].min()), color))
    ordered.sort()
    return [color for _row, color in ordered]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort colored line segments by top row, use grey-sprite hole counts as indices,
    then replace each grey sprite with its bbox outline in the indexed color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    palette = _ordered_line_colors(g)

    for pts in _components(g == GREY, connectivity=8):
        r0, c0, r1, c1 = _bbox(pts)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            mask[r - r0, c - c0] = True
            out[r, c] = BG
        holes = _hole_count(mask)
        if 1 <= holes <= len(palette):
            color = int(palette[holes - 1])
            out[r0 : r1 + 1, c0 : c1 + 1] = BG
            out[r0, c0 : c1 + 1] = color
            out[r1, c0 : c1 + 1] = color
            out[r0 : r1 + 1, c0] = color
            out[r0 : r1 + 1, c1] = color

    return out


def _is_single_component(mask: np.ndarray, connectivity: int = 8) -> bool:
    return len(_components(mask, connectivity=connectivity)) == 1


def _make_holey_sprite(num_holes: int) -> np.ndarray | None:
    for _ in range(80):
        vertical = random.random() < 0.5
        chamber_sizes = [random.randint(1, 3) for _ in range(num_holes)]
        if vertical:
            h = random.randint(4, 8)
            w = 2 + sum(chamber_sizes) + max(0, num_holes - 1)
            if w > 12:
                continue
            mask = np.zeros((h, w), dtype=bool)
            mask[0, :] = True
            mask[-1, :] = True
            mask[:, 0] = True
            mask[:, -1] = True
            chambers: list[tuple[int, int]] = []
            col = 1
            for i, width in enumerate(chamber_sizes):
                chambers.append((col, col + width - 1))
                col += width
                if i < num_holes - 1:
                    mask[1:-1, col] = True
                    col += 1
            for left, right in chambers:
                if h >= 5 and random.random() < 0.7:
                    col = random.randint(left, right)
                    depth = random.randint(1, h - 4)
                    if random.random() < 0.5:
                        mask[1 : 1 + depth, col] = True
                    else:
                        mask[h - 1 - depth : h - 1, col] = True
        else:
            w = random.randint(4, 8)
            h = 2 + sum(chamber_sizes) + max(0, num_holes - 1)
            if h > 12:
                continue
            mask = np.zeros((h, w), dtype=bool)
            mask[0, :] = True
            mask[-1, :] = True
            mask[:, 0] = True
            mask[:, -1] = True
            chambers = []
            row = 1
            for i, height in enumerate(chamber_sizes):
                chambers.append((row, row + height - 1))
                row += height
                if i < num_holes - 1:
                    mask[row, 1:-1] = True
                    row += 1
            for top, bottom in chambers:
                if w >= 5 and random.random() < 0.7:
                    row = random.randint(top, bottom)
                    depth = random.randint(1, w - 4)
                    if random.random() < 0.5:
                        mask[row, 1 : 1 + depth] = True
                    else:
                        mask[row, w - 1 - depth : w - 1] = True
        if mask.shape[0] < 4 or mask.shape[1] < 4:
            continue
        if not _is_single_component(mask, connectivity=8):
            continue
        if _hole_count(mask) == num_holes:
            return mask
    return None


def _make_zero_hole_sprite() -> np.ndarray | None:
    for _ in range(80):
        h = random.randint(4, 8)
        w = random.randint(4, 8)
        mask = np.ones((h, w), dtype=bool)
        notch_count = random.randint(0, 4)
        for _ in range(notch_count):
            side = random.choice(("top", "bottom", "left", "right"))
            if side in ("top", "bottom"):
                depth = random.randint(1, max(1, min(2, h - 2)))
                width = random.randint(1, max(1, w - 2))
                start = random.randint(0, w - width)
                if width == w:
                    continue
                rows = slice(0, depth) if side == "top" else slice(h - depth, h)
                mask[rows, start : start + width] = False
            else:
                depth = random.randint(1, max(1, min(2, w - 2)))
                height = random.randint(1, max(1, h - 2))
                start = random.randint(0, h - height)
                if height == h:
                    continue
                cols = slice(0, depth) if side == "left" else slice(w - depth, w)
                mask[start : start + height, cols] = False
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if not _is_single_component(mask, connectivity=8):
            continue
        if _hole_count(mask) == 0:
            return mask
    return None


def _place_sprite(
    grid: np.ndarray, blocked: np.ndarray, mask: np.ndarray, color: int
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    mh, mw = mask.shape
    if mh > h - 2 or mw > w - 2:
        return None
    for _ in range(160):
        r0 = random.randint(1, h - mh - 1)
        c0 = random.randint(1, w - mw - 1)
        rr0 = r0 - 1
        cc0 = c0 - 1
        rr1 = r0 + mh + 1
        cc1 = c0 + mw + 1
        if blocked[rr0:rr1, cc0:cc1].any():
            continue
        grid[r0 : r0 + mh, c0 : c0 + mw][mask] = color
        blocked[rr0:rr1, cc0:cc1] = True
        return r0, c0, r0 + mh - 1, c0 + mw - 1
    return None


def _line_points(kind: str, row: int, col: int, length: int) -> list[tuple[int, int]]:
    if kind == "h":
        return [(row, col + i) for i in range(length)]
    if kind == "v":
        return [(row + i, col) for i in range(length)]
    if kind == "d1":
        return [(row + i, col + i) for i in range(length)]
    return [(row + i, col - i) for i in range(length)]


def _place_line(grid: np.ndarray, blocked: np.ndarray, color: int, used_top_rows: set[int]) -> bool:
    h, w = grid.shape
    choices = ["h", "v", "d1", "d2"]
    for _ in range(240):
        kind = random.choice(choices)
        length = random.randint(3, 7)
        if kind == "h":
            max_row = h - 2
            max_col = w - length - 1
            min_col = 1
        elif kind == "v":
            max_row = h - length - 1
            max_col = w - 2
            min_col = 1
        elif kind == "d1":
            max_row = h - length - 1
            max_col = w - length - 1
            min_col = 1
        else:
            max_row = h - length - 1
            max_col = w - 2
            min_col = length
        if max_row < 1 or max_col < min_col:
            continue
        row = random.randint(1, max_row)
        if row in used_top_rows:
            continue
        col = random.randint(min_col, max_col)
        pts = _line_points(kind, row, col, length)
        if any(blocked[r, c] for r, c in pts):
            continue
        for r, c in pts:
            grid[r, c] = color
        for r, c in pts:
            for nr in range(max(0, r - 1), min(h, r + 2)):
                for nc in range(max(0, c - 1), min(w, c + 2)):
                    blocked[nr, nc] = True
        used_top_rows.add(row)
        return True
    return False


def _validate_generated_grid(
    grid: np.ndarray, expected_lines: int, expected_holes: list[int]
) -> bool:
    line_comps = _components((grid != BG) & (grid != GREY), connectivity=8)
    if len(line_comps) != expected_lines:
        return False
    top_rows = [int(pts[:, 0].min()) for pts in line_comps]
    if len(set(top_rows)) != expected_lines:
        return False

    grey_comps = _components(grid == GREY, connectivity=8)
    if len(grey_comps) != len(expected_holes):
        return False
    holes_found = []
    for pts in grey_comps:
        r0, c0, r1, c1 = _bbox(pts)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            mask[r - r0, c - c0] = True
        holes_found.append(_hole_count(mask))
    if sorted(holes_found) != sorted(expected_holes):
        return False
    return np.any(main(grid) != grid)


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(LINE_COUNT_CHOICES)
    valid_sprite_count = random.choice(VALID_SPRITE_COUNT_CHOICES)
    include_distractor = random.random() < 0.5

    valid_holes = random.sample(list(range(1, k + 1)), valid_sprite_count)
    all_holes = list(valid_holes)
    sprite_masks: list[np.ndarray] = []

    for holes in valid_holes:
        mask = _make_holey_sprite(holes)
        if mask is None:
            return None
        sprite_masks.append(mask)

    if include_distractor:
        distractor_high = random.random() < 0.5
        distractor_holes = k + 1 if distractor_high else 0
        mask = _make_holey_sprite(distractor_holes) if distractor_high else _make_zero_hole_sprite()
        if mask is None:
            return None
        all_holes.append(distractor_holes)
        sprite_masks.append(mask)

    max_side = max(max(mask.shape) for mask in sprite_masks)
    lower = max(GRID_MIN, min(GRID_MAX - 2, 14 + k + len(sprite_masks) + max_side // 2))
    h = random.randint(lower, GRID_MAX)
    w = random.randint(lower, GRID_MAX)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    order = sorted(range(len(sprite_masks)), key=lambda i: sprite_masks[i].size, reverse=True)
    for idx in order:
        if _place_sprite(grid, blocked, sprite_masks[idx], GREY) is None:
            return None

    line_colors = random.sample(list(LINE_COLOR_POOL), k)
    used_top_rows: set[int] = set()
    for color in line_colors:
        if not _place_line(grid, blocked, color, used_top_rows):
            return None

    if not _validate_generated_grid(grid, k, all_holes):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
