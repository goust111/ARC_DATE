from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
PALETTE = (1, 3, 4, 5, 6, 7, 8, 9)
GEN_RETRIES = 400


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _component_bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_rectangular_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _component_bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    expected = np.zeros_like(local)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(local, expected))


def _find_red_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == RED):
        if _is_rectangular_frame(comp):
            frames.append(_component_bbox(comp))
    return frames


def _largest_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    frames = _find_red_frames(grid)
    assert frames, "expected at least one red rectangular frame"
    frames.sort(key=_bbox_area, reverse=True)
    assert len(frames) == 1 or _bbox_area(frames[0]) > _bbox_area(frames[1]), "expected a unique largest frame"
    return frames[0]


def _propagate_crop(crop: np.ndarray) -> np.ndarray:
    out = crop.copy()
    h, w = crop.shape
    assert h >= 3 and w >= 3
    marker_row = h - 2
    marked_cols = [c for c in range(w) if int(crop[marker_row, c]) not in (BG, RED)]
    assert len(marked_cols) >= 1
    for c in marked_cols:
        current: int | None = None
        for r in range(marker_row, -1, -1):
            value = int(crop[r, c])
            if value == RED:
                current = None
            elif value != BG:
                current = value
            elif current is not None:
                out[r, c] = current
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop to the largest red frame, then vertically fill only the bottom-marked columns while treating red frame pixels as propagation barriers.

    Input structure:
    - The grid uses black as background and contains 2-3 complete one-pixel-thick red rectangular frames.
    - The unique largest red frame determines a crop window; colored non-red pixels may appear both inside and outside that frame.
    - In the largest frame, the row just above the bottom border contains one or more colored pixels that mark the columns to process.

    Rule:
    - Find every complete red rectangular frame and select the unique one with the largest bounding-box area.
    - Crop the grid to that frame's bounding box, keeping the red border.
    - In the cropped grid, for each marked column, scan upward from the bottom-inner marker row; remember the latest non-black non-red color and fill black cells with it.
    - When the scan hits a red pixel, stop the current color segment; propagation above that red barrier restarts only after a new colored pixel appears higher in the same column.
    - Leave non-marked columns unchanged.

    Key insight:
    - The same red frame pixels play two roles in sequence: first they choose the crop, then inside that crop they become hard vertical barriers that split each marked column into independently filled segments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bbox = _largest_frame_bbox(g)
    return _propagate_crop(_crop(g, bbox))


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = RED
    grid[r1, c0 : c1 + 1] = RED
    grid[r0 : r1 + 1, c0] = RED
    grid[r0 : r1 + 1, c1] = RED


def _disjoint(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _pick_color(exclude: set[int] | None = None) -> int:
    banned = set() if exclude is None else set(exclude)
    choices = [color for color in PALETTE if color not in banned]
    return int(random.choice(choices))


def _choose_inner_frame(
    outer: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    lr0, lc0, lr1, lc1 = outer
    lh = lr1 - lr0 + 1
    lw = lc1 - lc0 + 1
    if lh < 9 or lw < 9:
        return None

    top_margin = random.randint(3, max(3, min(4, lh - 6)))
    bottom_margin = random.randint(3, max(3, min(4, lh - top_margin - 3)))
    left_margin = random.randint(2, max(2, min(4, lw - 6)))
    right_margin = random.randint(2, max(2, min(4, lw - left_margin - 3)))

    inner_h = lh - top_margin - bottom_margin
    inner_w = lw - left_margin - right_margin
    if inner_h < 4 or inner_w < 4:
        return None

    return (
        lr0 + top_margin,
        lc0 + left_margin,
        lr1 - bottom_margin,
        lc1 - right_margin,
    )


def _random_outside_frame(
    shape: tuple[int, int],
    forbidden: list[tuple[int, int, int, int]],
    largest_area: int,
) -> tuple[int, int, int, int] | None:
    h, w = shape
    max_area = largest_area * 2 // 3
    for _ in range(120):
        fh = random.randint(4, min(8, h - 2))
        fw = random.randint(4, min(8, w - 2))
        if fh * fw > max_area:
            continue
        r0 = random.randint(1, h - fh - 1)
        c0 = random.randint(1, w - fw - 1)
        box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        if all(_disjoint(box, other, pad=1) for other in forbidden):
            return box
    return None


def _place_seed(
    grid: np.ndarray,
    r: int,
    c: int,
    exclude: set[int] | None = None,
    preferred: int | None = None,
) -> int:
    if preferred is not None and preferred not in (BG, RED):
        color = int(preferred)
    else:
        color = _pick_color(exclude)
    assert grid[r, c] == BG
    grid[r, c] = color
    return color


def _decorate_barrier_column(
    grid: np.ndarray,
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
    c: int,
) -> None:
    lr0, _lc0, lr1, _lc1 = outer
    ir0, _ic0, ir1, _ic1 = inner
    bottom = lr1 - 1

    base = _place_seed(grid, bottom, c)

    if ir1 + 1 < bottom:
        lower_r = random.randint(ir1 + 1, bottom - 1)
        lower_color = _pick_color({base})
        _place_seed(grid, lower_r, c, preferred=lower_color)

    if lr0 + 1 <= ir0 - 2:
        upper_r = random.randint(lr0 + 1, ir0 - 2)
        upper_color = _pick_color({base})
        _place_seed(grid, upper_r, c, preferred=upper_color)
        if upper_r - (lr0 + 1) >= 2 and random.random() < 0.65:
            top_r = random.randint(lr0 + 1, upper_r - 1)
            top_color = _pick_color({base, upper_color})
            if grid[top_r, c] == BG:
                _place_seed(grid, top_r, c, preferred=top_color)


def _decorate_clear_column(
    grid: np.ndarray,
    outer: tuple[int, int, int, int],
    c: int,
    force_two_colors: bool,
) -> None:
    lr0, _lc0, lr1, _lc1 = outer
    bottom = lr1 - 1

    base = _place_seed(grid, bottom, c)
    high = random.randint(lr0 + 1, bottom - 2)
    second = _pick_color({base}) if force_two_colors or random.random() < 0.8 else base
    _place_seed(grid, high, c, preferred=second)

    if high - (lr0 + 1) >= 2 and random.random() < 0.55:
        top_r = random.randint(lr0 + 1, high - 1)
        third = _pick_color({base, second})
        if grid[top_r, c] == BG:
            _place_seed(grid, top_r, c, preferred=third)


def _scatter_inside_noise(
    grid: np.ndarray,
    outer: tuple[int, int, int, int],
    marked_cols: set[int],
) -> None:
    lr0, lc0, lr1, lc1 = outer
    marker_row = lr1 - 1
    cells = [
        (r, c)
        for r in range(lr0 + 1, lr1)
        for c in range(lc0 + 1, lc1)
        if grid[r, c] == BG and r != marker_row and c not in marked_cols
    ]
    random.shuffle(cells)
    for r, c in cells[: random.randint(2, min(6, len(cells))) if cells else 0]:
        grid[r, c] = _pick_color()


def _scatter_outside_noise(
    grid: np.ndarray,
    outer: tuple[int, int, int, int],
    extra_frame: tuple[int, int, int, int] | None,
) -> None:
    lr0, lc0, lr1, lc1 = outer
    cells = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            inside_outer = lr0 <= r <= lr1 and lc0 <= c <= lc1
            if inside_outer or grid[r, c] != BG:
                continue
            cells.append((r, c))
    random.shuffle(cells)
    count = min(len(cells), random.randint(4, 8))
    for r, c in cells[:count]:
        grid[r, c] = _pick_color()

    if extra_frame is not None and random.random() < 0.7:
        er0, ec0, er1, ec1 = extra_frame
        interior = [
            (r, c)
            for r in range(er0 + 1, er1)
            for c in range(ec0 + 1, ec1)
            if grid[r, c] == BG
        ]
        random.shuffle(interior)
        for r, c in interior[: random.randint(0, min(2, len(interior))) if interior else 0]:
            grid[r, c] = _pick_color()


def _ignore_barriers(crop: np.ndarray) -> np.ndarray:
    out = crop.copy()
    h, w = crop.shape
    marker_row = h - 2
    marked_cols = [c for c in range(w) if int(crop[marker_row, c]) not in (BG, RED)]
    for c in marked_cols:
        current: int | None = None
        for r in range(marker_row, -1, -1):
            value = int(crop[r, c])
            if value not in (BG, RED):
                current = value
            elif value == BG and current is not None:
                out[r, c] = current
    return out


def _semantic_checks(grid: np.ndarray) -> bool:
    frames = _find_red_frames(grid)
    if len(frames) not in (2, 3):
        return False
    frames = sorted(frames, key=_bbox_area, reverse=True)
    if len(frames) > 1 and _bbox_area(frames[0]) * 2 < _bbox_area(frames[1]) * 3:
        return False

    outer = frames[0]
    crop = _crop(grid, outer)
    out = main(grid)
    if out.shape != crop.shape:
        return False

    marker_row = crop.shape[0] - 2
    marked_cols = [c for c in range(crop.shape[1]) if int(crop[marker_row, c]) not in (BG, RED)]
    if len(marked_cols) < 2:
        return False

    if not np.any(grid != BG) or not np.any(grid == RED):
        return False

    outside_mask = np.ones_like(grid, dtype=bool)
    r0, c0, r1, c1 = outer
    outside_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    if not np.any((grid != BG) & (grid != RED) & outside_mask):
        return False

    if not any(len(set(int(v) for v in crop[:, c] if v not in (BG, RED))) >= 2 for c in marked_cols):
        return False

    if not any(np.any(crop[:-1, c] == RED) for c in marked_cols):
        return False

    if np.array_equal(out, crop):
        return False

    if np.array_equal(out, _ignore_barriers(crop)):
        return False

    changed_cols = [c for c in marked_cols if np.any(out[:, c] != crop[:, c])]
    if len(changed_cols) < 2:
        return False

    unmarked_cols = [c for c in range(crop.shape[1]) if c not in marked_cols]
    if unmarked_cols and not all(np.array_equal(out[:, c], crop[:, c]) for c in unmarked_cols):
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(17, 23)
    w = random.randint(17, 23)
    grid = np.zeros((h, w), dtype=int)

    lh = random.randint(10, min(15, h - 3))
    lw = random.randint(10, min(15, w - 3))
    lr0 = random.randint(1, h - lh - 1)
    lc0 = random.randint(1, w - lw - 1)
    outer = (lr0, lc0, lr0 + lh - 1, lc0 + lw - 1)

    inner = _choose_inner_frame(outer)
    if inner is None:
        return None

    _draw_frame(grid, outer)
    _draw_frame(grid, inner)

    extra_frame = None
    if random.random() < 0.6:
        extra_frame = _random_outside_frame(grid.shape, [outer], _bbox_area(outer))
        if extra_frame is None:
            return None
        _draw_frame(grid, extra_frame)

    _lr0, lc0, lr1, lc1 = outer
    ir0, ic0, ir1, ic1 = inner

    barrier_cols = [ic0, ic1]
    clear_cols = [c for c in range(lc0 + 1, lc1) if c < ic0 or c > ic1]
    if not clear_cols:
        return None

    marked_cols = {random.choice(barrier_cols), random.choice(clear_cols)}
    remainder = [c for c in range(lc0 + 1, lc1) if c not in marked_cols]
    extra_markers = random.randint(0, min(2, len(remainder)))
    if extra_markers:
        marked_cols.update(random.sample(remainder, extra_markers))

    barrier_col = next(c for c in marked_cols if c in barrier_cols)
    clear_col = next(c for c in marked_cols if c in clear_cols)
    _decorate_barrier_column(grid, outer, inner, barrier_col)
    _decorate_clear_column(grid, outer, clear_col, force_two_colors=True)

    for c in sorted(marked_cols - {barrier_col, clear_col}):
        if ic0 <= c <= ic1:
            _decorate_barrier_column(grid, outer, inner, c)
        else:
            _decorate_clear_column(grid, outer, c, force_two_colors=False)

    if grid[lr1 - 1, barrier_col] in (BG, RED) or grid[lr1 - 1, clear_col] in (BG, RED):
        return None

    _scatter_inside_noise(grid, outer, marked_cols)
    _scatter_outside_noise(grid, outer, extra_frame)

    crop = _crop(grid, outer)
    if len([c for c in range(crop.shape[1]) if crop[-2, c] not in (BG, RED)]) < 2:
        return None

    return grid if _semantic_checks(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
