import random
from typing import Iterable

import numpy as np


BG = 8
COLOR_POOL = [0, 1, 2, 3, 4, 5, 6, 7, 9]
GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
COLOR_COUNT_RANGE = (3, 5)
PIXEL_COUNT_CHOICES = [2, 3, 4, 5, 6]
GENERATION_ATTEMPTS = 500
MODES = ("hub", "cover", "chain")

Rect = tuple[int, int, int, int]
Cell = tuple[int, int]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    pairs = {int(v): int(c) for v, c in zip(values.tolist(), counts.tolist())}
    if BG in pairs and pairs[BG] == max(pairs.values()):
        return BG
    return int(values[int(np.argmax(counts))])


def _first_seen_colors(grid: np.ndarray, bg: int) -> list[int]:
    colors: list[int] = []
    seen: set[int] = set()
    for color in map(int, grid.reshape(-1)):
        if color == bg or color in seen:
            continue
        seen.add(color)
        colors.append(color)
    return colors


def _analyze_grid(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for color in _first_seen_colors(grid, bg):
        pts = np.argwhere(grid == color)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        region = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        anchor = min((int(r), int(c)) for r, c in pts.tolist())
        records.append(
            {
                "color": int(color),
                "own_count": int(len(pts)),
                "bar_height": int(np.count_nonzero(region != bg)),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "anchor": anchor,
            }
        )

    records.sort(
        key=lambda rec: (
            -int(rec["own_count"]),
            int(rec["anchor"][0]),
            int(rec["anchor"][1]),
            int(rec["bbox"][0]),
            int(rec["bbox"][1]),
            int(rec["bbox"][2]),
            int(rec["bbox"][3]),
        )
    )
    return records


def _render_histogram(bg: int, records: list[dict[str, object]]) -> np.ndarray:
    height = max(int(rec["bar_height"]) for rec in records)
    width = 2 * len(records) - 1
    out = np.full((height, width), bg, dtype=int)
    for i, rec in enumerate(records):
        bar_h = int(rec["bar_height"])
        out[height - bar_h :, 2 * i] = int(rec["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group marker pixels by color, use each color cloud's bbox as a selection,
    count every non-background pixel inside that selection, and redraw the
    counts as a bottom-aligned histogram ordered by cloud size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    records = _analyze_grid(g, bg)
    assert records, "expected at least one non-background color"
    return _render_histogram(bg, records)


def _rect_area(rect: Rect) -> int:
    return (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1)


def _rect_cells(rect: Rect) -> list[Cell]:
    r0, c0, r1, c1 = rect
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _intersection_cells(a: Rect, b: Rect) -> list[Cell]:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return []
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _place_cloud(
    rect: Rect,
    count: int,
    occupied: set[Cell],
    preferred: Iterable[Cell],
    preferred_cap: int = 1,
) -> list[Cell] | None:
    if _rect_area(rect) < count:
        return None

    available = [cell for cell in _rect_cells(rect) if cell not in occupied]
    if len(available) < count:
        return None

    r0, c0, r1, c1 = rect
    corner_pairs = [
        [(r0, c0), (r1, c1)],
        [(r0, c1), (r1, c0)],
    ]
    valid_pairs = [pair for pair in corner_pairs if pair[0] in available and pair[1] in available]
    if not valid_pairs:
        return None

    chosen = list(random.choice(valid_pairs))
    remaining_slots = count - 2
    if remaining_slots == 0:
        return chosen

    pool = [cell for cell in available if cell not in chosen]
    priority = [cell for cell in preferred if cell in pool]
    extras: list[Cell] = []

    take_priority = min(remaining_slots, len(priority), preferred_cap)
    if take_priority > 0:
        extras.extend(random.sample(priority, take_priority))

    rest = [cell for cell in pool if cell not in extras]
    need = remaining_slots - len(extras)
    if len(rest) < need:
        return None
    if need > 0:
        extras.extend(random.sample(rest, need))

    chosen.extend(extras)
    return chosen


def _make_hub_rects(height: int, width: int, counts: list[int]) -> tuple[list[Rect], list[list[Cell]]] | None:
    k = len(counts)
    max_center_h = min(4, height - 6)
    max_center_w = min(4, width - 6)
    if max_center_h < 2 or max_center_w < 2:
        return None

    center_h = random.randint(2, max_center_h)
    center_w = random.randint(2, max_center_w)
    center_r0 = random.randint(2, height - center_h - 3)
    center_c0 = random.randint(2, width - center_w - 3)
    center_r1 = center_r0 + center_h - 1
    center_c1 = center_c0 + center_w - 1
    center_cells = _rect_cells((center_r0, center_c0, center_r1, center_c1))

    rects: list[Rect] = []
    for count in counts:
        rect = None
        for _ in range(40):
            up = random.randint(0, min(5, center_r0))
            down = random.randint(0, min(5, height - 1 - center_r1))
            left = random.randint(0, min(5, center_c0))
            right = random.randint(0, min(5, width - 1 - center_c1))
            candidate = (
                center_r0 - up,
                center_c0 - left,
                center_r1 + down,
                center_c1 + right,
            )
            if candidate in rects:
                continue
            if candidate[0] == center_r0 and candidate[1] == center_c0 and candidate[2] == center_r1 and candidate[3] == center_c1:
                continue
            if _rect_area(candidate) < count:
                continue
            rect = candidate
            break
        if rect is None:
            return None
        rects.append(rect)

    return rects, [center_cells[:] for _ in range(k)]


def _make_cover_rects(height: int, width: int, counts: list[int]) -> tuple[list[Rect], list[list[Cell]]] | None:
    k = len(counts)
    if height < 10 or width < 10:
        return None

    master_idx = random.randrange(k)
    master_h = random.randint(6, min(10, height - 2))
    master_w = random.randint(6, min(10, width - 2))
    master_r0 = random.randint(1, height - master_h - 1)
    master_c0 = random.randint(1, width - master_w - 1)
    master = (master_r0, master_c0, master_r0 + master_h - 1, master_c0 + master_w - 1)

    rects: list[Rect] = [(0, 0, 0, 0) for _ in range(k)]
    rects[master_idx] = master

    for i, count in enumerate(counts):
        if i == master_idx:
            continue
        rect = None
        for _ in range(60):
            h = random.randint(3, min(7, height - 1))
            w = random.randint(3, min(7, width - 1))
            overlap_h = random.randint(2, min(h - 1, master_h))
            overlap_w = random.randint(2, min(w - 1, master_w))
            r_lo = max(0, master_r0 - h + overlap_h)
            r_hi = min(height - h, master[2] - overlap_h + 1)
            c_lo = max(0, master_c0 - w + overlap_w)
            c_hi = min(width - w, master[3] - overlap_w + 1)
            if r_lo > r_hi or c_lo > c_hi:
                continue
            r0 = random.randint(r_lo, r_hi)
            c0 = random.randint(c_lo, c_hi)
            candidate = (r0, c0, r0 + h - 1, c0 + w - 1)
            if candidate in rects:
                continue
            if _rect_area(candidate) < count:
                continue
            rect = candidate
            break
        if rect is None:
            return None
        rects[i] = rect

    preferred: list[list[Cell]] = []
    for i in range(k):
        if i == master_idx:
            preferred.append([])
        else:
            preferred.append(_intersection_cells(rects[i], master))
    return rects, preferred


def _make_chain_rects(height: int, width: int, counts: list[int]) -> tuple[list[Rect], list[list[Cell]]] | None:
    k = len(counts)
    for _ in range(80):
        h0 = random.randint(4, 7)
        w0 = random.randint(4, 7)
        if h0 >= height or w0 >= width:
            continue
        r0 = random.randint(0, height - h0)
        c0 = random.randint(0, width - w0)
        rects = [(r0, c0, r0 + h0 - 1, c0 + w0 - 1)]
        ok = True

        for i in range(1, k):
            prev = rects[-1]
            prev_h = prev[2] - prev[0] + 1
            prev_w = prev[3] - prev[1] + 1
            rect = None
            for _ in range(50):
                h = random.randint(3, 7)
                w = random.randint(3, 7)
                if prev_h < 3 or prev_w < 3:
                    break
                top_lo = prev[0] + 1
                top_hi = prev[2] - 1
                left_lo = prev[1] + 1
                left_hi = prev[3] - 1
                if top_lo > top_hi or left_lo > left_hi:
                    break
                nr0 = random.randint(top_lo, top_hi)
                nc0 = random.randint(left_lo, left_hi)
                candidate = (nr0, nc0, nr0 + h - 1, nc0 + w - 1)
                if candidate[2] >= height or candidate[3] >= width:
                    continue
                if candidate in rects:
                    continue
                if candidate[2] <= prev[2] and candidate[3] <= prev[3]:
                    continue
                if _rect_area(candidate) < counts[i]:
                    continue
                rect = candidate
                break
            if rect is None:
                ok = False
                break
            rects.append(rect)

        if not ok:
            continue

        preferred: list[list[Cell]] = []
        for i in range(k):
            cells: list[Cell] = []
            if i > 0:
                cells.extend(_intersection_cells(rects[i], rects[i - 1]))
            if i + 1 < k:
                cells.extend(_intersection_cells(rects[i], rects[i + 1]))
            preferred.append(list(dict.fromkeys(cells)))
        return rects, preferred
    return None


def _make_scene_rects(mode: str, height: int, width: int, counts: list[int]) -> tuple[list[Rect], list[list[Cell]]] | None:
    if mode == "hub":
        return _make_hub_rects(height, width, counts)
    if mode == "cover":
        return _make_cover_rects(height, width, counts)
    return _make_chain_rects(height, width, counts)


def _validate_scene(
    grid: np.ndarray,
    colors: list[int],
    counts: list[int],
    rects: list[Rect],
) -> bool:
    records = _analyze_grid(grid, BG)
    if len(records) != len(colors):
        return False

    bbox_by_color = {int(rec["color"]): rec["bbox"] for rec in records}
    count_by_color = {int(rec["color"]): int(rec["own_count"]) for rec in records}
    intended = {color: rect for color, rect in zip(colors, rects)}

    for color, count in zip(colors, counts):
        if count_by_color.get(color) != count:
            return False
        if bbox_by_color.get(color) != intended[color]:
            return False

    overlap_gain = sum(int(rec["bar_height"]) - int(rec["own_count"]) for rec in records)
    if overlap_gain < 2:
        return False

    expected = _render_histogram(BG, records)
    return bool(np.array_equal(main(grid), expected))


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(MODES)
    if mode == "chain":
        height = random.randint(15, GRID_H_RANGE[1])
        width = random.randint(15, GRID_W_RANGE[1])
    else:
        height = random.randint(*GRID_H_RANGE)
        width = random.randint(*GRID_W_RANGE)

    k = random.randint(*COLOR_COUNT_RANGE)
    colors = random.sample(COLOR_POOL, k)
    counts = random.sample(PIXEL_COUNT_CHOICES, k)
    rect_data = _make_scene_rects(mode, height, width, counts)
    if rect_data is None:
        return None
    rects, preferred_by_index = rect_data

    grid = np.full((height, width), BG, dtype=int)
    occupied: set[Cell] = set()
    cells_by_index: list[list[Cell] | None] = [None] * k
    placement_order = sorted(range(k), key=lambda i: (counts[i], _rect_area(rects[i])))

    for i in placement_order:
        cells = _place_cloud(rects[i], counts[i], occupied, preferred_by_index[i], preferred_cap=1)
        if cells is None:
            return None
        cells_by_index[i] = cells
        for r, c in cells:
            grid[r, c] = colors[i]
            occupied.add((r, c))

    if any(cells is None for cells in cells_by_index):
        return None
    if not _validate_scene(grid, colors, counts, rects):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
