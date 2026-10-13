import math
import random
from collections import deque

import numpy as np


FRAME_COLOR = 5
GRID_MIN = 22
GRID_MAX = 30
PERIOD_MIN = 3
PERIOD_MAX = 6
ANCHOR_COLOR_MIN = 4
ANCHOR_COLOR_MAX = 6
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _flood_from_boundary(blocked: np.ndarray) -> np.ndarray:
    h, w = blocked.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not blocked[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not blocked[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if blocked[nr, nc] or seen[nr, nc]:
                continue
            seen[nr, nc] = True
            q.append((nr, nc))
    return seen


def _enclosed_region(border: np.ndarray) -> np.ndarray:
    outside = _flood_from_boundary(border)
    return (~border) & (~outside)


def _outline_color(grid: np.ndarray, bg: int) -> int:
    colors, counts = np.unique(grid[grid != bg], return_counts=True)
    assert len(colors) >= 2, "expected outline and anchors"
    idx = int(np.argmax(counts))
    outline = int(colors[idx])
    if len(colors) > 1:
        sorted_counts = sorted((int(c) for c in counts), reverse=True)
        assert sorted_counts[0] > sorted_counts[1], "outline color must be unique"
    return outline


def _infer_periods(anchor_coords: dict[int, np.ndarray]) -> tuple[int, int]:
    dy = 0
    dx = 0
    for coords in anchor_coords.values():
        n = int(coords.shape[0])
        assert 2 <= n <= 4, "each anchor color must appear 2-4 times"
        for i in range(n):
            for j in range(i + 1, n):
                dr = abs(int(coords[i, 0]) - int(coords[j, 0]))
                dc = abs(int(coords[i, 1]) - int(coords[j, 1]))
                if dr:
                    dy = math.gcd(dy, dr)
                if dc:
                    dx = math.gcd(dx, dc)
    assert dy > 0 and dx > 0, "failed to infer both periods"
    return dy, dx


def _tile_grid(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    dy, dx = tile.shape
    rr = np.arange(h)[:, None] % dy
    cc = np.arange(w)[None, :] % dx
    return tile[rr, cc]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the hidden periodic tile from colored single-pixel anchors and
    fill only the interior of the grey mask outline with that tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    outline_color = _outline_color(g, bg)

    border = g == outline_color
    inside = _enclosed_region(border)
    assert inside.any(), "mask must enclose a region"

    anchor_colors = [int(c) for c in np.unique(g) if c not in {bg, outline_color}]
    anchor_coords = {color: np.argwhere(g == int(color)) for color in anchor_colors}
    dy, dx = _infer_periods(anchor_coords)

    tile = np.full((dy, dx), bg, dtype=int)
    for color, coords in anchor_coords.items():
        residues = {(int(r) % dy, int(c) % dx) for r, c in coords}
        assert len(residues) == 1, "same-color anchors must share one tile residue"
        rr, cc = next(iter(residues))
        assert tile[rr, cc] in {bg, color}, "tile residue collision"
        tile[rr, cc] = color

    full = _tile_grid(tile, g.shape)
    out = np.full_like(g, bg)
    out[inside] = full[inside]
    return out


def _proper_divisors(n: int) -> list[int]:
    return [d for d in range(1, n) if n % d == 0]


def _expand8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = min(h, h - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(w, w - dc)
            dst_r0 = max(0, dr)
            dst_r1 = min(h, h + dr)
            dst_c0 = max(0, dc)
            dst_c1 = min(w, w + dc)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _draw_loop(vertices: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    border = np.zeros(shape, dtype=bool)
    for i, (r0, c0) in enumerate(vertices):
        r1, c1 = vertices[(i + 1) % len(vertices)]
        if r0 == r1:
            lo, hi = sorted((c0, c1))
            border[r0, lo : hi + 1] = True
        else:
            lo, hi = sorted((r0, r1))
            border[lo : hi + 1, c0] = True
    return border


def _rectangle_patch(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    inside = _enclosed_region(border)
    return border, inside


def _top_notch_border(height: int, width: int) -> np.ndarray:
    notch_left = random.randint(2, width - 5)
    notch_right = random.randint(notch_left + 2, width - 3)
    notch_depth = random.randint(2, height - 4)
    vertices = [
        (0, 0),
        (0, notch_left),
        (notch_depth, notch_left),
        (notch_depth, notch_right),
        (0, notch_right),
        (0, width - 1),
        (height - 1, width - 1),
        (height - 1, 0),
    ]
    return _draw_loop(vertices, (height, width))


def _notched_patch(height: int, width: int, side: str) -> tuple[np.ndarray, np.ndarray]:
    if side in {"top", "bottom"}:
        base = _top_notch_border(height, width)
        border = np.flipud(base) if side == "bottom" else base
    else:
        base = _top_notch_border(width, height)
        transposed = base.T
        border = np.fliplr(transposed) if side == "right" else transposed
    inside = _enclosed_region(border)
    return border, inside


def _sample_mask_patch(
    grid_h: int, grid_w: int, dy: int, dx: int
) -> tuple[np.ndarray, np.ndarray] | None:
    min_h = max(2 * dy + 3, 9)
    min_w = max(2 * dx + 3, 9)
    max_h = grid_h - 6
    max_w = grid_w - 6
    if min_h > max_h or min_w > max_w:
        return None

    kind = random.choice(["rect", "notch"])
    height = random.randint(min_h, max_h)
    width = random.randint(min_w, max_w)
    if kind == "rect":
        border, inside = _rectangle_patch(height, width)
    else:
        side = random.choice(["top", "bottom", "left", "right"])
        border, inside = _notched_patch(height, width, side)

    rows = np.where(inside.any(axis=1))[0]
    cols = np.where(inside.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    if rows[-1] - rows[0] + 1 < 2 * dy or cols[-1] - cols[0] + 1 < 2 * dx:
        return None
    return border, inside


def _candidate_blocks(
    valid: np.ndarray, dy: int, dx: int, residue: tuple[int, int]
) -> list[tuple[tuple[int, int], ...]]:
    h, w = valid.shape
    rr, cc = residue
    blocks: list[tuple[tuple[int, int], ...]] = []
    for r in range(rr, h - dy, dy):
        for c in range(cc, w - dx, dx):
            cells = ((r, c), (r, c + dx), (r + dy, c), (r + dy, c + dx))
            if all(valid[pr, pc] for pr, pc in cells):
                blocks.append(cells)
    random.shuffle(blocks)
    return blocks


def _selection_blocks_period(
    residues: list[tuple[int, int]], dy: int, dx: int
) -> bool:
    row_groups: dict[int, list[int]] = {}
    col_groups: dict[int, list[int]] = {}
    for r, c in residues:
        row_groups.setdefault(int(r), []).append(int(c))
        col_groups.setdefault(int(c), []).append(int(r))

    for d in _proper_divisors(dx):
        ok = False
        for cols in row_groups.values():
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    if abs(cols[i] - cols[j]) % d == 0:
                        ok = True
                        break
                if ok:
                    break
            if ok:
                break
        if not ok:
            return False

    for d in _proper_divisors(dy):
        ok = False
        for rows in col_groups.values():
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    if abs(rows[i] - rows[j]) % d == 0:
                        ok = True
                        break
                if ok:
                    break
            if ok:
                break
        if not ok:
            return False

    return True


def _choose_residues(
    available: list[tuple[int, int]], dy: int, dx: int
) -> list[tuple[int, int]] | None:
    if len(available) < ANCHOR_COLOR_MIN:
        return None
    hi = min(ANCHOR_COLOR_MAX, len(available))
    for _ in range(200):
        count = random.randint(ANCHOR_COLOR_MIN, hi)
        residues = random.sample(available, count)
        if len({r for r, _ in residues}) < 2:
            continue
        if len({c for _, c in residues}) < 2:
            continue
        if _selection_blocks_period(residues, dy, dx):
            return residues
    return None


_PATTERNS = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
    (0, 1, 2),
    (0, 1, 3),
    (0, 2, 3),
    (1, 2, 3),
    (0, 1, 2, 3),
)


def _mark_buffered(mask: np.ndarray, points: list[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in points:
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        mask[r0:r1, c0:c1] = True


def _place_anchors(
    grid: np.ndarray,
    colors: list[int],
    residues: list[tuple[int, int]],
    candidates: dict[tuple[int, int], list[tuple[tuple[int, int], ...]]],
    dy: int,
    dx: int,
) -> dict[int, np.ndarray] | None:
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)
    order = list(range(len(colors)))
    random.shuffle(order)
    order.sort(key=lambda idx: len(candidates[residues[idx]]))

    anchor_coords: dict[int, np.ndarray] = {}
    for idx in order:
        color = int(colors[idx])
        residue = residues[idx]
        blocks = candidates[residue][:]
        random.shuffle(blocks)
        placed = None
        for block in blocks:
            pattern = random.choice(_PATTERNS)
            points = [block[k] for k in pattern]
            if any(occupied[r, c] for r, c in points):
                continue
            placed = points
            break
        if placed is None:
            return None
        for r, c in placed:
            grid[r, c] = color
        _mark_buffered(occupied, placed)
        anchor_coords[color] = np.array(placed, dtype=int)

    got_dy, got_dx = _infer_periods(anchor_coords)
    if got_dy != dy or got_dx != dx:
        return None
    return anchor_coords


def _tile_from_residues(
    dy: int, dx: int, bg: int, colors: list[int], residues: list[tuple[int, int]]
) -> np.ndarray:
    tile = np.full((dy, dx), bg, dtype=int)
    for color, (r, c) in zip(colors, residues):
        tile[int(r), int(c)] = int(color)
    return tile


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c != FRAME_COLOR])
    dy = random.randint(PERIOD_MIN, PERIOD_MAX)
    dx = random.randint(PERIOD_MIN, PERIOD_MAX)
    grid_h = random.randint(GRID_MIN, GRID_MAX)
    grid_w = random.randint(GRID_MIN, GRID_MAX)

    mask_patch = _sample_mask_patch(grid_h, grid_w, dy, dx)
    if mask_patch is None:
        return None
    border_patch, inside_patch = mask_patch
    ph, pw = border_patch.shape
    top = random.randint(2, grid_h - ph - 2)
    left = random.randint(2, grid_w - pw - 2)

    border = np.zeros((grid_h, grid_w), dtype=bool)
    inside = np.zeros((grid_h, grid_w), dtype=bool)
    border[top : top + ph, left : left + pw] = border_patch
    inside[top : top + ph, left : left + pw] = inside_patch

    valid = ~_expand8(border | inside)
    valid[0, :] = False
    valid[-1, :] = False
    valid[:, 0] = False
    valid[:, -1] = False

    available_residues = []
    residue_candidates: dict[tuple[int, int], list[tuple[tuple[int, int], ...]]] = {}
    for rr in range(dy):
        for cc in range(dx):
            blocks = _candidate_blocks(valid, dy, dx, (rr, cc))
            if blocks:
                available_residues.append((rr, cc))
                residue_candidates[(rr, cc)] = blocks

    residues = _choose_residues(available_residues, dy, dx)
    if residues is None:
        return None

    color_pool = [c for c in range(10) if c not in {bg, FRAME_COLOR}]
    if len(color_pool) < len(residues):
        return None
    colors = random.sample(color_pool, len(residues))
    tile = _tile_from_residues(dy, dx, bg, colors, residues)

    inp = np.full((grid_h, grid_w), bg, dtype=int)
    inp[border] = FRAME_COLOR
    anchors = _place_anchors(inp, colors, residues, residue_candidates, dy, dx)
    if anchors is None:
        return None

    out = np.full((grid_h, grid_w), bg, dtype=int)
    full = _tile_grid(tile, (grid_h, grid_w))
    out[inside] = full[inside]

    if _most_common_color(inp) != bg:
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
