from __future__ import annotations

import numpy as np


BG_COLOR = 3
FENCE_COLOR = 9
ANIMAL_COLOR = 1
FEED_COLOR = 7


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
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_fences(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    fences: list[tuple[int, int, int, int]] = []
    for pts in _components(grid == FENCE_COLOR):
        r0, c0, r1, c1 = _bbox(pts)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        view = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if pts.shape[0] != int(border.sum()):
            continue
        if not np.all(view[border] == FENCE_COLOR):
            continue
        if np.any(view[1:-1, 1:-1] == FENCE_COLOR):
            continue
        fences.append((r0, c0, r1, c1))
    fences.sort()
    return fences


def _feed_orientation(inside: np.ndarray) -> str:
    pts = np.argwhere(inside == FEED_COLOR)
    if pts.size == 0:
        raise ValueError("each fence must contain feed pixels")
    row_span = int(pts[:, 0].max() - pts[:, 0].min())
    col_span = int(pts[:, 1].max() - pts[:, 1].min())
    if row_span <= 1 and col_span > row_span:
        return "horizontal"
    if col_span <= 1 and row_span > col_span:
        return "vertical"
    if col_span > row_span:
        return "horizontal"
    if row_span > col_span:
        return "vertical"
    row_dev = float(np.abs(pts[:, 0] - np.median(pts[:, 0])).sum())
    col_dev = float(np.abs(pts[:, 1] - np.median(pts[:, 1])).sum())
    return "horizontal" if row_dev <= col_dev else "vertical"


def _rotate_inside(inside: np.ndarray, clockwise: bool) -> np.ndarray:
    h, w = inside.shape
    out = np.full((h, w), BG_COLOR, dtype=int)
    center_r2 = h - 1
    center_c2 = w - 1
    for r, c in np.argwhere(inside != BG_COLOR):
        pixel_r2 = 2 * int(r)
        pixel_c2 = 2 * int(c)
        if clockwise:
            new_r2 = center_r2 + (pixel_c2 - center_c2)
            new_c2 = center_c2 - (pixel_r2 - center_r2)
        else:
            new_r2 = center_r2 - (pixel_c2 - center_c2)
            new_c2 = center_c2 + (pixel_r2 - center_r2)
        if (new_r2 & 1) or (new_c2 & 1):
            continue
        nr = new_r2 // 2
        nc = new_c2 // 2
        if 0 <= nr < h and 0 <= nc < w:
            out[nr, nc] = int(inside[r, c])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each fenced pasture according to the feed-line direction and clip overflow."""
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for r0, c0, r1, c1 in _find_fences(g):
        inside = g[r0 + 1 : r1, c0 + 1 : c1]
        clockwise = _feed_orientation(inside) == "horizontal"
        out[r0 + 1 : r1, c0 + 1 : c1] = _rotate_inside(inside, clockwise)
    return out


# --------------- generator ---------------

_RNG = np.random.default_rng()

FENCE_COUNT_MIN = 2
FENCE_COUNT_MAX_EXCLUSIVE = 5
GRID_SIDE_MAX_EXCLUSIVE = 31
MAX_GENERATION_ATTEMPTS = 400
LOCAL_ATTEMPTS = 200

SQUARE_INTERIORS = [(4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]
RECT_INTERIORS = [(4, 6), (6, 4), (4, 8), (8, 4), (5, 7), (7, 5), (6, 8), (8, 6)]

ANIMAL_SHAPES = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
]


def _safe_window(h: int, w: int) -> tuple[int, int, int]:
    side = min(h, w)
    return (h - side) // 2, (w - side) // 2, side


def _transformed_shape(rng: np.random.Generator) -> np.ndarray:
    shape = ANIMAL_SHAPES[int(rng.integers(len(ANIMAL_SHAPES)))].copy()
    if rng.random() < 0.5:
        shape = np.fliplr(shape)
    shape = np.rot90(shape, int(rng.integers(4)))
    return shape.astype(bool)


def _sample_feed_positions(
    h: int,
    w: int,
    orientation: str,
    clip_flag: bool,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    safe_r0, safe_c0, safe_side = _safe_window(h, w)
    safe_r1 = safe_r0 + safe_side - 1
    safe_c1 = safe_c0 + safe_side - 1
    count = int(rng.integers(2, 4))

    if orientation == "horizontal":
        if count > w:
            return None
        if clip_flag and w > h:
            start_c = 0 if rng.random() < 0.5 else w - count
        elif safe_side >= count:
            lo = safe_c0 if not clip_flag else 0
            hi = (safe_c1 - count + 1) if not clip_flag else (w - count)
            if hi < lo:
                return None
            start_c = int(rng.integers(lo, hi + 1))
        else:
            return None
        cols = [start_c + i for i in range(count)]

        if clip_flag and h > w:
            edge = 0 if rng.random() < 0.5 else h - 1
            rows = [edge] * count
            if count >= 2:
                step = 1 if edge == 0 else -1
                rows[int(rng.integers(count))] = edge + step
        else:
            lo = safe_r0 if not clip_flag else 0
            hi = safe_r1 if not clip_flag else h - 1
            if hi < lo:
                return None
            base_row = int(rng.integers(lo, hi + 1))
            rows = [base_row] * count
            if count == 3 and base_row > lo and base_row < hi and rng.random() < 0.5:
                rows[int(rng.integers(count))] += -1 if rng.random() < 0.5 else 1
        pts = list(zip(rows, cols))
    else:
        if count > h:
            return None
        if clip_flag and h > w:
            start_r = 0 if rng.random() < 0.5 else h - count
        elif safe_side >= count:
            lo = safe_r0 if not clip_flag else 0
            hi = (safe_r1 - count + 1) if not clip_flag else (h - count)
            if hi < lo:
                return None
            start_r = int(rng.integers(lo, hi + 1))
        else:
            return None
        rows = [start_r + i for i in range(count)]

        if clip_flag and w > h:
            edge = 0 if rng.random() < 0.5 else w - 1
            cols = [edge] * count
            if count >= 2:
                step = 1 if edge == 0 else -1
                cols[int(rng.integers(count))] = edge + step
        else:
            lo = safe_c0 if not clip_flag else 0
            hi = safe_c1 if not clip_flag else w - 1
            if hi < lo:
                return None
            base_col = int(rng.integers(lo, hi + 1))
            cols = [base_col] * count
            if count == 3 and base_col > lo and base_col < hi and rng.random() < 0.5:
                cols[int(rng.integers(count))] += -1 if rng.random() < 0.5 else 1
        pts = list(zip(rows, cols))

    if len(set(pts)) != len(pts):
        return None
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    if orientation == "horizontal":
        if max(rows) - min(rows) > 1 or max(cols) - min(cols) < 1:
            return None
    else:
        if max(cols) - min(cols) > 1 or max(rows) - min(rows) < 1:
            return None
    return pts


def _place_animal(
    inside: np.ndarray,
    clip_flag: bool,
    rng: np.random.Generator,
) -> bool:
    h, w = inside.shape
    safe_r0, safe_c0, safe_side = _safe_window(h, w)
    safe_r1 = safe_r0 + safe_side - 1
    safe_c1 = safe_c0 + safe_side - 1
    for _ in range(LOCAL_ATTEMPTS):
        shape = _transformed_shape(rng)
        sh, sw = shape.shape
        use_safe = (not clip_flag) or rng.random() < 0.75
        if use_safe:
            row_lo = safe_r0
            row_hi = safe_r1 - sh + 1
            col_lo = safe_c0
            col_hi = safe_c1 - sw + 1
        else:
            row_lo = 0
            row_hi = h - sh
            col_lo = 0
            col_hi = w - sw
        if row_hi < row_lo or col_hi < col_lo:
            continue
        top = int(rng.integers(row_lo, row_hi + 1))
        left = int(rng.integers(col_lo, col_hi + 1))
        coords = np.argwhere(shape)
        cells = [(top + int(r), left + int(c)) for r, c in coords]
        if any(inside[r, c] != BG_COLOR for r, c in cells):
            continue
        for r, c in cells:
            inside[r, c] = ANIMAL_COLOR
        return True
    return False


def _build_pasture(
    h: int,
    w: int,
    orientation: str,
    clip_flag: bool,
    rng: np.random.Generator,
) -> np.ndarray | None:
    clockwise = orientation == "horizontal"
    for _ in range(LOCAL_ATTEMPTS):
        inside = np.full((h, w), BG_COLOR, dtype=int)
        feed_pts = _sample_feed_positions(h, w, orientation, clip_flag, rng)
        if feed_pts is None:
            continue
        for r, c in feed_pts:
            inside[r, c] = FEED_COLOR
        if not _place_animal(inside, clip_flag, rng):
            continue
        if _feed_orientation(inside) != orientation:
            continue

        rotated = _rotate_inside(inside, clockwise)
        opposite = _rotate_inside(inside, not clockwise)
        clipped = int(np.count_nonzero(rotated != BG_COLOR)) < int(np.count_nonzero(inside != BG_COLOR))
        if clipped != clip_flag:
            continue
        if np.array_equal(rotated, opposite):
            continue
        if np.array_equal(rotated, inside):
            continue
        if np.count_nonzero(rotated == ANIMAL_COLOR) == 0:
            continue
        if np.count_nonzero(rotated == FEED_COLOR) == 0:
            continue
        return inside
    return None


def _place_fences(
    specs: list[tuple[int, int, np.ndarray]],
    grid_h: int,
    grid_w: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    order = sorted(range(len(specs)), key=lambda i: (specs[i][0] + 2) * (specs[i][1] + 2), reverse=True)
    positions: list[tuple[int, int] | None] = [None] * len(specs)
    for idx in order:
        h, w, _inside = specs[idx]
        fh = h + 2
        fw = w + 2
        placed = False
        for _ in range(LOCAL_ATTEMPTS):
            top = int(rng.integers(0, grid_h - fh + 1))
            left = int(rng.integers(0, grid_w - fw + 1))
            r0 = max(0, top - 1)
            c0 = max(0, left - 1)
            r1 = min(grid_h, top + fh + 1)
            c1 = min(grid_w, left + fw + 1)
            if occupied[r0:r1, c0:c1].any():
                continue
            occupied[r0:r1, c0:c1] = True
            positions[idx] = (top, left)
            placed = True
            break
        if not placed:
            return None
    return [pos for pos in positions if pos is not None]


def _draw_fence(grid: np.ndarray, top: int, left: int, inside: np.ndarray) -> None:
    h, w = inside.shape
    bottom = top + h + 1
    right = left + w + 1
    grid[top : bottom + 1, left] = FENCE_COLOR
    grid[top : bottom + 1, right] = FENCE_COLOR
    grid[top, left : right + 1] = FENCE_COLOR
    grid[bottom, left : right + 1] = FENCE_COLOR
    grid[top + 1 : bottom, left + 1 : right] = inside


def _gen_attempt() -> np.ndarray | None:
    rng = _RNG
    fence_count = int(rng.integers(FENCE_COUNT_MIN, FENCE_COUNT_MAX_EXCLUSIVE))

    orientations = ["horizontal", "vertical"] + [
        "horizontal" if rng.random() < 0.5 else "vertical"
        for _ in range(fence_count - 2)
    ]
    rng.shuffle(orientations)

    clip_flags = [True, False] + [bool(rng.integers(2)) for _ in range(fence_count - 2)]
    rng.shuffle(clip_flags)

    specs: list[tuple[int, int, np.ndarray]] = []
    for orientation, clip_flag in zip(orientations, clip_flags):
        dim_pool = RECT_INTERIORS if clip_flag else (RECT_INTERIORS + SQUARE_INTERIORS)
        h, w = dim_pool[int(rng.integers(len(dim_pool)))]
        inside = _build_pasture(h, w, orientation, clip_flag, rng)
        if inside is None:
            return None
        specs.append((h, w, inside))

    min_side = 16 + 3 * fence_count
    grid_h = int(rng.integers(min_side, GRID_SIDE_MAX_EXCLUSIVE))
    grid_w = int(rng.integers(min_side, GRID_SIDE_MAX_EXCLUSIVE))
    positions = _place_fences(specs, grid_h, grid_w, rng)
    if positions is None:
        return None

    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    for (h, w, inside), (top, left) in zip(specs, positions):
        _draw_fence(grid, top, left, inside)

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if not np.array_equal(grid == FENCE_COLOR, out == FENCE_COLOR):
        return None

    seen_dirs: set[str] = set()
    saw_clip = False
    saw_no_clip = False
    for r0, c0, r1, c1 in _find_fences(grid):
        before = grid[r0 + 1 : r1, c0 + 1 : c1]
        after = out[r0 + 1 : r1, c0 + 1 : c1]
        seen_dirs.add(_feed_orientation(before))
        clipped = int(np.count_nonzero(after != BG_COLOR)) < int(np.count_nonzero(before != BG_COLOR))
        saw_clip |= clipped
        saw_no_clip |= not clipped
    if seen_dirs != {"horizontal", "vertical"}:
        return None
    if not saw_clip or not saw_no_clip:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
