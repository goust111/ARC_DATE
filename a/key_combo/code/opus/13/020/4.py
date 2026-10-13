import random
from collections import Counter, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
TEAL = 8
CYCLE = (RED, TEAL, GREEN)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(comp)
    return parts


def _scan_order(coords: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], str]:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height >= width:
        return sorted(coords), "row"
    return sorted(coords, key=lambda rc: (rc[1], rc[0])), "col"


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    rects: list[dict] = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3 or len(comp) != h * w:
            raise AssertionError("grey objects must be filled rectangles with interior")
        rects.append({"bbox": (r0, c0, r1, c1), "h": h, "w": w})
    if not rects:
        raise AssertionError("expected grey rectangles")
    rects.sort(key=lambda rect: rect["bbox"])
    return rects


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected 2D grid")

    blue_coords = [(int(r), int(c)) for r, c in np.argwhere(g == BLUE)]
    if len(blue_coords) < 1:
        raise AssertionError("expected blue single pixels")

    ordered, scan_mode = _scan_order(blue_coords)
    out = g.copy()
    marker_colors: dict[tuple[int, int], int] = {}
    for idx, (r, c) in enumerate(ordered):
        color = CYCLE[idx % len(CYCLE)]
        out[r, c] = color
        marker_colors[(r, c)] = color

    rect_infos = []
    for rect in _extract_rectangles(g):
        r0, c0, r1, c1 = rect["bbox"]
        counts = Counter()
        for rr in range(r0, r1 + 1):
            for cc in (c0, c1):
                for nr, nc in _neighbors4(rr, cc):
                    if (nr, nc) in marker_colors:
                        counts[marker_colors[(nr, nc)]] += 1
        for cc in range(c0 + 1, c1):
            for rr in (r0, r1):
                for nr, nc in _neighbors4(rr, cc):
                    if (nr, nc) in marker_colors:
                        counts[marker_colors[(nr, nc)]] += 1

        winner = None
        if counts:
            best = max(counts.values())
            winners = [color for color, cnt in counts.items() if cnt == best]
            if len(winners) != 1:
                raise AssertionError("adjacency majority must be unique for generated inputs")
            winner = winners[0]
            out[r0 + 1 : r1, c0 + 1 : c1] = winner

        rect_infos.append({"bbox": rect["bbox"], "counts": dict(counts), "winner": winner})

    info = {
        "ordered": ordered,
        "scan_mode": scan_mode,
        "marker_colors": marker_colors,
        "rects": rect_infos,
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor scattered blue pixels by global scan order, then fill each grey rectangle interior with the majority adjacent recolored pixel color.

    Input structure:
    - The grid has black background, 2-4 solid grey axis-aligned rectangles, and 6-12 isolated blue single pixels outside the rectangles.
    - The blue pixels define one global foreground bbox whose longer axis determines the scan direction.
    - Grey rectangles are fully filled and at least 3x3, so each has a one-cell border and a non-empty interior.

    Rule:
    - Compute the bbox of all blue pixels.
    - If the bbox height is at least its width, order blue pixels row-major; otherwise order them column-major.
    - Recolor the ordered blue pixels cyclically as red, teal, green, repeating in that order.
    - For each grey rectangle, count recolored pixels that are 4-neighbor adjacent to its border cells, choose the unique majority color, and repaint only the rectangle interior to that color.
    - Keep the grey border unchanged, keep recolored pixels in place, and leave a rectangle interior grey only if it has no adjacent recolored pixel.

    Key insight:
    - A single global ordering decision on the scattered pixels controls all local rectangle fills, because those fills vote using the colors produced by the scan-order recoloring step.
    """
    return _solve(grid)[0]


def _rect_cells(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _ring_cells(rect: tuple[int, int, int, int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        if r0 - 1 >= 0:
            cells.add((r0 - 1, c))
        if r1 + 1 < shape[0]:
            cells.add((r1 + 1, c))
    for r in range(r0, r1 + 1):
        if c0 - 1 >= 0:
            cells.add((r, c0 - 1))
        if c1 + 1 < shape[1]:
            cells.add((r, c1 + 1))
    return cells


def _non_touching_rect(existing: list[tuple[int, int, int, int]], rect: tuple[int, int, int, int], gap: int = 2) -> bool:
    r0, c0, r1, c1 = rect
    for er0, ec0, er1, ec1 in existing:
        if not (r1 + gap < er0 or er1 + gap < r0 or c1 + gap < ec0 or ec1 + gap < c0):
            return False
    return True


def _marker_ok(cell: tuple[int, int], markers: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    r, c = cell
    if not _in_bounds(shape, r, c) or cell in markers:
        return False
    return all((nr, nc) not in markers for nr, nc in _neighbors4(r, c))


def _make_row_rectangles(rng: random.Random, target: str) -> tuple[int, int, list[tuple[int, int, int, int]], tuple[int, int]]:
    n_rect = rng.randint(2, 4)
    hs = [rng.randint(3, 6) for _ in range(n_rect - 1)] + [rng.randint(3, 6)]
    ws = [rng.randint(3, 6) for _ in range(n_rect - 1)] + [rng.randint(4, 6)]
    gaps = [rng.randint(2, 4) for _ in range(n_rect - 1)]
    total_h = sum(hs) + sum(gaps)
    h = total_h + rng.randint(5, 8)
    if target == "tie":
        w = max(h + rng.randint(-1, 3), max(ws) + rng.randint(8, 12))
    else:
        w = max(max(ws) + rng.randint(8, 12), 14)
    if h > 28 or w > 28:
        raise ValueError("grid too large")

    band_w = min(max(ws) + rng.randint(1, 3), w - 4)
    band_left = rng.randint(2, w - band_w - 2)
    cur_r = rng.randint(1, 2)
    rects = []
    for i in range(n_rect):
        hi, wi = hs[i], ws[i]
        span = band_w - wi
        c0 = band_left + (0 if span <= 0 else rng.randint(0, span))
        r0 = cur_r + rng.randint(0, 1)
        r1 = r0 + hi - 1
        c1 = c0 + wi - 1
        rect = (r0, c0, r1, c1)
        if r0 < 1 or c0 < 1 or r1 > h - 2 or c1 > w - 2:
            raise ValueError("rectangle out of usable area")
        if not _non_touching_rect(rects, rect):
            raise ValueError("rectangles too close")
        rects.append(rect)
        if i < n_rect - 1:
            cur_r = r1 + gaps[i]
    if rects[-1][2] > h - 3:
        raise ValueError("not enough bottom room")
    return h, w, rects, (band_left, band_left + band_w - 1)


def _make_col_rectangles(rng: random.Random) -> tuple[int, int, list[tuple[int, int, int, int]], tuple[int, int]]:
    n_rect = rng.randint(2, 4)
    hs = [rng.randint(3, 6) for _ in range(n_rect - 1)] + [rng.randint(4, 6)]
    ws = [rng.randint(3, 6) for _ in range(n_rect)]
    gaps = [rng.randint(2, 4) for _ in range(n_rect - 1)]
    total_w = sum(ws) + sum(gaps)
    w = total_w + rng.randint(5, 8)
    h = max(max(hs) + rng.randint(8, 12), 14)
    if h > 28 or w > 28:
        raise ValueError("grid too large")

    band_h = min(max(hs) + rng.randint(1, 3), h - 4)
    band_top = rng.randint(2, h - band_h - 2)
    cur_c = rng.randint(1, 2)
    rects = []
    for i in range(n_rect):
        hi, wi = hs[i], ws[i]
        span = band_h - hi
        r0 = band_top + (0 if span <= 0 else rng.randint(0, span))
        c0 = cur_c + rng.randint(0, 1)
        r1 = r0 + hi - 1
        c1 = c0 + wi - 1
        rect = (r0, c0, r1, c1)
        if r0 < 1 or c0 < 1 or r1 > h - 2 or c1 > w - 2:
            raise ValueError("rectangle out of usable area")
        if not _non_touching_rect(rects, rect):
            raise ValueError("rectangles too close")
        rects.append(rect)
        if i < n_rect - 1:
            cur_c = c1 + gaps[i]
    if rects[-1][3] > w - 3:
        raise ValueError("not enough right room")
    return h, w, rects, (band_top, band_top + band_h - 1)


def _row_focus_markers(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, _c1 = rect
    return [(r0 - 1, c0), (r0 - 1, c0 + 2), (r1 + 1, c0 + 1), (r1 + 1, c0 + 3)]


def _col_focus_markers(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, _r1, c1 = rect
    return [(r0, c0 - 1), (r0 + 2, c0 - 1), (r0 + 1, c1 + 1), (r0 + 3, c1 + 1)]


def _pick_non_focus_row_marker(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    mid = (r0 + r1) // 2
    if r1 - r0 >= 2:
        mid = r0 + 1
    return (mid, c0 - 1) if (mid + c0) % 2 == 0 else (mid, c1 + 1)


def _pick_non_focus_col_marker(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    mid = (c0 + c1) // 2
    if c1 - c0 >= 2:
        mid = c0 + 1
    return (r0 - 1, mid) if (mid + r0) % 2 == 0 else (r1 + 1, mid)


def _background_pool(
    shape: tuple[int, int],
    rects: list[tuple[int, int, int, int]],
    band: tuple[int, int],
    mode: str,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    rect_cells = set().union(*(_rect_cells(rect) for rect in rects))
    ring_cells = set().union(*(_ring_cells(rect, shape) for rect in rects))
    pool = []
    if mode == "row":
        left, right = band
        for r in range(shape[0]):
            for c in range(max(0, left - 1), min(shape[1], right + 2)):
                cell = (r, c)
                if cell in rect_cells or cell in ring_cells or cell in forbidden:
                    continue
                pool.append(cell)
    else:
        top, bottom = band
        for r in range(max(0, top - 1), min(shape[0], bottom + 2)):
            for c in range(shape[1]):
                cell = (r, c)
                if cell in rect_cells or cell in ring_cells or cell in forbidden:
                    continue
                pool.append(cell)
    return pool


def _tie_adjust_markers(
    rng: random.Random,
    shape: tuple[int, int],
    rects: list[tuple[int, int, int, int]],
    markers: list[tuple[int, int]],
    band: tuple[int, int],
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = _bbox(markers)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if width > height:
        return None
    if width == height:
        return markers

    need = height - width
    left_room = c0
    right_room = shape[1] - 1 - c1
    left_expand = min(left_room, need // 2)
    right_expand = min(right_room, need - left_expand)
    left_expand = need - right_expand
    if left_expand > left_room:
        return None

    rect_cells = set().union(*(_rect_cells(rect) for rect in rects))
    ring_cells = set().union(*(_ring_cells(rect, shape) for rect in rects))
    used = set(markers)
    rows = list(range(r0, r1 + 1))
    rng.shuffle(rows)
    for row in rows:
        additions = []
        if left_expand > 0:
            additions.append((row, c0 - left_expand))
        if right_expand > 0:
            additions.append((row, c1 + right_expand))
        if any(cell in rect_cells or cell in ring_cells for cell in additions):
            continue
        if all(_marker_ok(cell, used, shape) for cell in additions):
            return markers + additions

    left, right = band
    for row in range(r0, r1 + 1):
        additions = []
        if left_expand > 0:
            additions.append((row, max(0, left - left_expand)))
        if right_expand > 0:
            additions.append((row, min(shape[1] - 1, right + right_expand)))
        if len({cell[1] for cell in additions}) != len(additions):
            continue
        if any(cell in rect_cells or cell in ring_cells for cell in additions):
            continue
        if all(_marker_ok(cell, used, shape) for cell in additions):
            trial = markers + additions
            tr0, tc0, tr1, tc1 = _bbox(trial)
            if tr1 - tr0 == tc1 - tc0:
                return trial
    return None


def _build_attempt(rng: random.Random, target: str) -> np.ndarray | None:
    try:
        if target == "col":
            h, w, rects, band = _make_col_rectangles(rng)
            mode = "col"
        else:
            h, w, rects, band = _make_row_rectangles(rng, target)
            mode = "row"
    except ValueError:
        return None

    markers: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    focus = rects[-1]

    for rect in rects[:-1]:
        cell = _pick_non_focus_col_marker(rect) if mode == "col" else _pick_non_focus_row_marker(rect)
        if not _marker_ok(cell, used, (h, w)):
            return None
        markers.append(cell)
        used.add(cell)

    focus_cells = _col_focus_markers(focus) if mode == "col" else _row_focus_markers(focus)
    for cell in focus_cells:
        if not _marker_ok(cell, used, (h, w)):
            return None
        markers.append(cell)
        used.add(cell)

    forbidden = set(markers)
    pool = _background_pool((h, w), rects, band, mode, forbidden)
    rng.shuffle(pool)
    extra_target = rng.randint(1, min(4, 12 - len(markers)))
    for cell in pool:
        if extra_target == 0:
            break
        if not _marker_ok(cell, used, (h, w)):
            continue
        if mode == "row" and cell[0] >= focus[0]:
            continue
        if mode == "col" and cell[1] >= focus[1]:
            continue
        markers.append(cell)
        used.add(cell)
        extra_target -= 1

    if target == "tie":
        adjusted = _tie_adjust_markers(rng, (h, w), rects, markers, band)
        if adjusted is None:
            return None
        markers = adjusted

    if not (6 <= len(markers) <= 12):
        return None

    r0, c0, r1, c1 = _bbox(markers)
    bbox_h = r1 - r0 + 1
    bbox_w = c1 - c0 + 1
    if target == "row" and bbox_h < bbox_w:
        return None
    if target == "col" and bbox_w <= bbox_h:
        return None
    if target == "tie" and bbox_h != bbox_w:
        return None

    grid = np.zeros((h, w), dtype=int)
    for rect in rects:
        rr0, cc0, rr1, cc1 = rect
        grid[rr0 : rr1 + 1, cc0 : cc1 + 1] = GREY
    for r, c in markers:
        if grid[r, c] != BG:
            return None
        grid[r, c] = BLUE

    try:
        out, info = _solve(grid)
    except AssertionError:
        return None

    if info["scan_mode"] != ("col" if target == "col" else "row"):
        return None

    winners = [rect["winner"] for rect in info["rects"]]
    if any(winner is None for winner in winners):
        return None
    if len(set(winners)) < 2:
        return None

    rich_rects = [rect for rect in info["rects"] if sum(rect["counts"].values()) >= 4]
    if not rich_rects:
        return None

    unique_cols = set(info["marker_colors"].values())
    if unique_cols != {RED, TEAL, GREEN}:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    branches = ["row", "col", "tie"]
    weights = [4, 4, 2]
    for _ in range(600):
        target = rng.choices(branches, weights=weights, k=1)[0]
        grid = _build_attempt(rng, target)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
