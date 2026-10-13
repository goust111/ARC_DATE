from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
BORDER_COLOR = 5
SIDES = ("top", "bottom", "left", "right")
FILL_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

RECT_COUNT_CHOICES = (2, 3, 4)
GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 30
RECT_OUTER_H_RANGE = (4, 9)
RECT_OUTER_W_RANGE = (4, 10)
PLACEMENT_GAP = 2
NOISE_COUNT_MAX_EXTRA = 2
MAX_GENERATION_ATTEMPTS = 300


def _neighbors(r: int, c: int, h: int, w: int, conn8: bool) -> list[tuple[int, int]]:
    steps4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    steps = steps4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)] if conn8 else steps4
    out: list[tuple[int, int]] = []
    for dr, dc in steps:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _components(mask: np.ndarray, conn8: bool) -> list[np.ndarray]:
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
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, h, w, conn8):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comp = np.zeros_like(mask, dtype=bool)
            rr, cc = zip(*cells)
            comp[np.array(rr), np.array(cc)] = True
            comps.append(comp)
    return comps


def _dilate8(mask: np.ndarray, steps: int = 1) -> np.ndarray:
    out = mask.copy()
    for _ in range(steps):
        h, w = out.shape
        padded = np.pad(out, 1, constant_values=False)
        grown = np.zeros_like(out, dtype=bool)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                grown |= padded[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
        out = grown
    return out


def _parse_horizontal_legend(band: np.ndarray) -> tuple[list[int], np.ndarray] | None:
    if band.shape[0] != 2:
        return None
    w = band.shape[1]
    mask = np.zeros_like(band, dtype=bool)
    colors: list[int] = []
    col_nonzero = np.any(band != BACKGROUND, axis=0)
    c = 0
    while c < w:
        if not col_nonzero[c]:
            c += 1
            continue
        start = c
        while c < w and col_nonzero[c]:
            c += 1
        if c - start != 2:
            return None
        block = band[:, start:c]
        color = int(block[0, 0])
        if color == BACKGROUND or not np.all(block == color):
            return None
        colors.append(color)
        mask[:, start:c] = True
    if len(colors) < 2:
        return None
    if np.any((band != BACKGROUND) & (~mask)):
        return None
    return colors, mask


def _parse_vertical_legend(band: np.ndarray) -> tuple[list[int], np.ndarray] | None:
    if band.shape[1] != 2:
        return None
    h = band.shape[0]
    mask = np.zeros_like(band, dtype=bool)
    colors: list[int] = []
    row_nonzero = np.any(band != BACKGROUND, axis=1)
    r = 0
    while r < h:
        if not row_nonzero[r]:
            r += 1
            continue
        start = r
        while r < h and row_nonzero[r]:
            r += 1
        if r - start != 2:
            return None
        block = band[start:r, :]
        color = int(block[0, 0])
        if color == BACKGROUND or not np.all(block == color):
            return None
        colors.append(color)
        mask[start:r, :] = True
    if len(colors) < 2:
        return None
    if np.any((band != BACKGROUND) & (~mask)):
        return None
    return colors, mask


def _find_legend(grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[str, list[int], np.ndarray]] = []

    for side in ("top", "bottom"):
        band = g[:2, :] if side == "top" else g[h - 2 :, :]
        parsed = _parse_horizontal_legend(band)
        if parsed is None:
            continue
        colors, band_mask = parsed
        full_mask = np.zeros_like(g, dtype=bool)
        if side == "top":
            full_mask[:2, :] = band_mask
        else:
            full_mask[h - 2 :, :] = band_mask
        candidates.append((side, colors, full_mask))

    for side in ("left", "right"):
        band = g[:, :2] if side == "left" else g[:, w - 2 :]
        parsed = _parse_vertical_legend(band)
        if parsed is None:
            continue
        colors, band_mask = parsed
        full_mask = np.zeros_like(g, dtype=bool)
        if side == "left":
            full_mask[:, :2] = band_mask
        else:
            full_mask[:, w - 2 :] = band_mask
        candidates.append((side, colors, full_mask))

    assert candidates, "legend strip not found"
    candidates.sort(key=lambda item: len(item[1]), reverse=True)
    _side, colors, mask = candidates[0]
    return mask, [int(color) for color in colors]


def _border_color(grid: np.ndarray, legend_mask: np.ndarray) -> int:
    values = grid[~legend_mask]
    values = values[values != BACKGROUND]
    assert values.size > 0, "no border pixels found"
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[np.argmax(counts)])


def _interior_area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    return int((r1 - r0 - 1) * (c1 - c0 - 1))


def _extract_rectangles(border_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    dilated = _dilate8(border_mask, steps=1)
    found: list[tuple[int, int, int, int, int]] = []

    for comp in _components(dilated, conn8=True):
        original = border_mask & comp
        if int(original.sum()) < 6:
            continue

        row_hits = original.sum(axis=1)
        col_hits = original.sum(axis=0)
        rows = np.where(row_hits >= 2)[0]
        cols = np.where(col_hits >= 2)[0]
        if len(rows) < 2 or len(cols) < 2:
            continue

        r0, r1 = int(rows[0]), int(rows[-1])
        c0, c1 = int(cols[0]), int(cols[-1])
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue

        local = original[r0 : r1 + 1, c0 : c1 + 1]
        perimeter = np.zeros_like(local, dtype=bool)
        perimeter[0, :] = True
        perimeter[-1, :] = True
        perimeter[:, 0] = True
        perimeter[:, -1] = True

        side_counts = (
            int(original[r0, c0 : c1 + 1].sum()),
            int(original[r1, c0 : c1 + 1].sum()),
            int(original[r0 : r1 + 1, c0].sum()),
            int(original[r0 : r1 + 1, c1].sum()),
        )
        if min(side_counts) < 2:
            continue

        perimeter_hits = int(local[perimeter].sum())
        inner_hits = int(local[~perimeter].sum())
        perimeter_len = int(2 * local.shape[0] + 2 * local.shape[1] - 4)
        if perimeter_hits < max(8, perimeter_len // 5):
            continue
        if perimeter_hits < inner_hits + 4:
            continue

        found.append((r0, c0, r1, c1, perimeter_hits))

    found.sort(key=lambda item: item[4], reverse=True)
    rects: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for r0, c0, r1, c1, _score in found:
        rect = (r0, c0, r1, c1)
        if rect not in seen:
            seen.add(rect)
            rects.append(rect)
    return rects


def _render_output(
    grid: np.ndarray,
    legend_mask: np.ndarray,
    border_color: int,
    rects: list[tuple[int, int, int, int]],
    legend_colors: list[int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[(out == int(border_color)) & (~legend_mask)] = BACKGROUND

    ordered = sorted(rects, key=lambda rect: (_interior_area(rect), rect[0], rect[1]))
    assert ordered, "no rectangles detected"
    assert len(ordered) <= len(legend_colors), "not enough legend colors for rectangles"

    for fill_color, (r0, c0, r1, c1) in zip(legend_colors, ordered):
        out[r0, c0 : c1 + 1] = int(border_color)
        out[r1, c0 : c1 + 1] = int(border_color)
        out[r0 : r1 + 1, c0] = int(border_color)
        out[r0 : r1 + 1, c1] = int(border_color)
        out[r0 + 1 : r1, c0 + 1 : c1] = int(fill_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair damaged grey rectangle outlines, remove same-color noise, and fill
    each rectangle using the edge legend ordered by increasing interior area.
    """
    g = np.asarray(grid, dtype=int)
    legend_mask, legend_colors = _find_legend(g)
    border_color = _border_color(g, legend_mask)
    border_mask = (g == int(border_color)) & (~legend_mask)
    rects = _extract_rectangles(border_mask)
    return _render_output(g, legend_mask, border_color, rects, legend_colors)


def _sample_rect_sizes(count: int) -> list[tuple[int, int]] | None:
    candidates: list[tuple[int, int, int]] = []
    for h in range(RECT_OUTER_H_RANGE[0], RECT_OUTER_H_RANGE[1] + 1):
        for w in range(RECT_OUTER_W_RANGE[0], RECT_OUTER_W_RANGE[1] + 1):
            candidates.append((h, w, (h - 2) * (w - 2)))
    random.shuffle(candidates)

    picked: list[tuple[int, int]] = []
    used_areas: set[int] = set()
    for h, w, area in candidates:
        if area in used_areas:
            continue
        picked.append((h, w))
        used_areas.add(area)
        if len(picked) == count:
            return picked
    return None


def _draw_legend(grid: np.ndarray, side: str, colors: list[int]) -> np.ndarray:
    h, w = grid.shape
    span = 3 * len(colors) - 1
    mask = np.zeros_like(grid, dtype=bool)

    if side in ("top", "bottom"):
        start = random.randint(0, w - span)
        r0 = 0 if side == "top" else h - 2
        c = start
        for color in colors:
            grid[r0 : r0 + 2, c : c + 2] = int(color)
            mask[r0 : r0 + 2, c : c + 2] = True
            c += 3
    else:
        start = random.randint(0, h - span)
        c0 = 0 if side == "left" else w - 2
        r = start
        for color in colors:
            grid[r : r + 2, c0 : c0 + 2] = int(color)
            mask[r : r + 2, c0 : c0 + 2] = True
            r += 3
    return mask


def _place_rectangles(
    shape: tuple[int, int],
    blocked: np.ndarray,
    sizes: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]] | None:
    h, w = shape
    work = blocked.copy()
    rects: list[tuple[int, int, int, int]] = []
    for rh, rw in sorted(sizes, key=lambda item: item[0] * item[1], reverse=True):
        placed = False
        for _ in range(200):
            if h - rh - 1 < 1 or w - rw - 1 < 1:
                return None
            r0 = random.randint(1, h - rh - 1)
            c0 = random.randint(1, w - rw - 1)
            r1 = r0 + rh - 1
            c1 = c0 + rw - 1
            rr0 = max(0, r0 - PLACEMENT_GAP)
            rr1 = min(h, r1 + PLACEMENT_GAP + 1)
            cc0 = max(0, c0 - PLACEMENT_GAP)
            cc1 = min(w, c1 + PLACEMENT_GAP + 1)
            if work[rr0:rr1, cc0:cc1].any():
                continue
            work[rr0:rr1, cc0:cc1] = True
            rects.append((r0, c0, r1, c1))
            placed = True
            break
        if not placed:
            return None
    return rects


def _side_cells(rect: tuple[int, int, int, int]) -> tuple[list[tuple[int, int]], ...]:
    r0, c0, r1, c1 = rect
    top = [(r0, c) for c in range(c0, c1 + 1)]
    bottom = [(r1, c) for c in range(c0, c1 + 1)]
    left = [(r, c0) for r in range(r0, r1 + 1)]
    right = [(r, c1) for r in range(r0, r1 + 1)]
    return top, bottom, left, right


def _sample_side(cells: list[tuple[int, int]], gap_prob: float) -> list[tuple[int, int]]:
    if len(cells) <= 2:
        return cells[:]

    internal = cells[1:-1]
    keep = [random.random() > gap_prob for _ in internal]
    if not any(keep):
        keep[random.randrange(len(keep))] = True

    run = 0
    for idx, flag in enumerate(keep):
        if flag:
            run = 0
            continue
        run += 1
        if run > 2:
            keep[idx] = True
            run = 0

    picked = [cells[0], cells[-1]]
    picked.extend(cell for cell, flag in zip(internal, keep) if flag)
    return picked


def _draw_damaged_rectangle(grid: np.ndarray, rect: tuple[int, int, int, int], force_gap: bool) -> None:
    top, bottom, left, right = _side_cells(rect)
    full_points = set(top) | set(bottom) | set(left) | set(right)

    for _ in range(30):
        gap_prob = random.uniform(0.20, 0.45)
        points = set(_sample_side(top, gap_prob))
        points |= set(_sample_side(bottom, gap_prob))
        points |= set(_sample_side(left, gap_prob))
        points |= set(_sample_side(right, gap_prob))
        if not force_gap or len(points) < len(full_points):
            rr, cc = zip(*points)
            grid[np.array(rr), np.array(cc)] = BORDER_COLOR
            return

    rr, cc = zip(*full_points)
    grid[np.array(rr), np.array(cc)] = BORDER_COLOR


def _add_noise(grid: np.ndarray, legend_mask: np.ndarray) -> None:
    border_mask = grid == BORDER_COLOR
    blocked = _dilate8(border_mask | legend_mask, steps=1)
    blocked[:2, :] = True
    blocked[-2:, :] = True
    blocked[:, :2] = True
    blocked[:, -2:] = True
    used_rows = set(np.where(border_mask.any(axis=1))[0].tolist())
    used_cols = set(np.where(border_mask.any(axis=0))[0].tolist())

    candidates = [
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if not blocked[r, c]
    ]
    random.shuffle(candidates)

    need = random.randint(1, min(len(candidates), 4 + NOISE_COUNT_MAX_EXTRA))
    chosen: list[tuple[int, int]] = []
    rows = set(used_rows)
    cols = set(used_cols)
    for r, c in candidates:
        if r in rows or c in cols:
            continue
        chosen.append((r, c))
        rows.add(r)
        cols.add(c)
        if len(chosen) == need:
            break

    for r, c in chosen:
        grid[r, c] = BORDER_COLOR


def _expected_output(
    grid: np.ndarray,
    legend_mask: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    legend_colors: list[int],
) -> np.ndarray:
    return _render_output(grid, legend_mask, BORDER_COLOR, rects, legend_colors)


def _gen_attempt() -> np.ndarray | None:
    rect_count = random.choice(RECT_COUNT_CHOICES)
    side = random.choice(SIDES)
    legend_span = 3 * rect_count - 1

    h_min = max(GRID_SIDE_MIN, legend_span + 2) if side in ("left", "right") else GRID_SIDE_MIN
    w_min = max(GRID_SIDE_MIN, legend_span + 2) if side in ("top", "bottom") else GRID_SIDE_MIN
    h = random.randint(h_min, GRID_SIDE_MAX)
    w = random.randint(w_min, GRID_SIDE_MAX)

    sizes = _sample_rect_sizes(rect_count)
    if sizes is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    colors = random.sample(FILL_COLOR_POOL, rect_count)
    legend_mask = _draw_legend(grid, side, colors)

    blocked = _dilate8(legend_mask, steps=PLACEMENT_GAP)
    blocked[:2, :] = True
    blocked[-2:, :] = True
    blocked[:, :2] = True
    blocked[:, -2:] = True

    rects = _place_rectangles((h, w), blocked, sizes)
    if rects is None:
        return None

    damaged_any = False
    for idx, rect in enumerate(rects):
        force_gap = idx == 0
        before = int((grid == BORDER_COLOR).sum())
        _draw_damaged_rectangle(grid, rect, force_gap=force_gap)
        after = int((grid == BORDER_COLOR).sum())
        perimeter_len = 2 * (rect[2] - rect[0] + 1) + 2 * (rect[3] - rect[1] + 1) - 4
        if after - before < perimeter_len:
            damaged_any = True
    if not damaged_any:
        return None

    _add_noise(grid, legend_mask)

    expected = _expected_output(grid, legend_mask, rects, colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
