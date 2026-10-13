from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ALL_COLORS = tuple(range(10))
GEN_RETRIES = 500
PLACEMENT_TRIES = 300
SEGMENT_TRIES = 200


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _segment_signature(points: list[tuple[int, int]]) -> tuple[str, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r0 == r1 and c1 > c0:
        return "h", int(c1 - c0 + 1)
    if c0 == c1 and r1 > r0:
        return "v", int(r1 - r0 + 1)
    raise ValueError("component is not a valid 1-pixel-wide straight segment")


def _extract_objects(grid: np.ndarray) -> tuple[int, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    objects: list[dict[str, int]] = []
    colors = [int(v) for v in np.unique(g) if int(v) != bg]

    for color in colors:
        comps = _components(g == int(color))
        if len(comps) != 4:
            raise ValueError("each color must have exactly four fragments")

        horiz: list[int] = []
        vert: list[int] = []
        pts: list[tuple[int, int]] = []
        for comp in comps:
            orient, length = _segment_signature(comp)
            if orient == "h":
                horiz.append(length)
            else:
                vert.append(length)
            pts.extend(comp)

        if len(horiz) != 2 or len(vert) != 2:
            raise ValueError("each color must have two horizontal and two vertical segments")
        if len(set(horiz)) != 1 or len(set(vert)) != 1:
            raise ValueError("same-orientation segment lengths must match")

        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        rect_h = int(vert[0])
        rect_w = int(horiz[0])
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)

        num_r = r0 + r1 - rect_h + 1
        num_c = c0 + c1 - rect_w + 1
        if num_r % 2 != 0 or num_c % 2 != 0:
            raise ValueError("fragment bbox center does not align to an integer rectangle placement")

        top = int(num_r // 2)
        left = int(num_c // 2)
        bottom = top + rect_h
        right = left + rect_w
        if top < 0 or left < 0 or bottom > g.shape[0] or right > g.shape[1]:
            raise ValueError("reconstructed rectangle would leave the grid")

        objects.append(
            {
                "color": int(color),
                "top": top,
                "left": left,
                "h": rect_h,
                "w": rect_w,
                "area": int(rect_h * rect_w),
            }
        )

    return bg, objects


def _draw_order_key(obj: dict[str, int]) -> tuple[int, int, int, int, int]:
    return (-int(obj["area"]), int(obj["top"]), int(obj["left"]), int(obj["h"]), int(obj["w"]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble each color's scattered line fragments into a centered filled rectangle.

    Input structure:
    - The background is the most common color.
    - Every non-background color appears as exactly four monochrome 1-pixel-wide straight segments.
    - For each color, there are two equal-length horizontal segments and two equal-length vertical segments.
    - The four segments of one color are scattered away from their final rectangle position.

    Rule:
    - Group fragments by color and infer one rectangle per color.
    - The horizontal segment length is the rectangle width, and the vertical segment length is the rectangle height.
    - Compute the minimal bounding box of all pixels of that color and place the rectangle so its center matches that bounding-box center.
    - Fill the whole rectangle with that color.
    - Clear every original fragment pixel unless it lies inside a reconstructed rectangle, and resolve overlaps by drawing larger-area rectangles first and smaller-area rectangles later.

    Key insight:
    - The fragments are not independent marks: each color encodes the four sides of one rectangle, while the fragment cloud's overall center determines where that rectangle must be rebuilt.
    """
    g = np.asarray(grid, dtype=int)
    bg, objects = _extract_objects(g)

    out = np.full_like(g, bg)
    for obj in sorted(objects, key=_draw_order_key):
        top = int(obj["top"])
        left = int(obj["left"])
        h = int(obj["h"])
        w = int(obj["w"])
        out[top : top + h, left : left + w] = int(obj["color"])
    return out


def _overlap_area(a: dict[str, int], b: dict[str, int]) -> int:
    dh = max(0, min(int(a["top"] + a["h"]), int(b["top"] + b["h"])) - max(int(a["top"]), int(b["top"])))
    dw = max(0, min(int(a["left"] + a["w"]), int(b["left"] + b["w"])) - max(int(a["left"]), int(b["left"])))
    return int(dh * dw)


def _too_close(a: dict[str, int], b: dict[str, int], gap: int = 1) -> bool:
    return not (
        int(a["top"] + a["h"] + gap) <= int(b["top"])
        or int(b["top"] + b["h"] + gap) <= int(a["top"])
        or int(a["left"] + a["w"] + gap) <= int(b["left"])
        or int(b["left"] + b["w"] + gap) <= int(a["left"])
    )


def _sample_dimensions(count: int, grid_h: int, grid_w: int) -> list[tuple[int, int]] | None:
    side_max = min(8, grid_h - 2, grid_w - 2)
    if side_max < 2:
        return None

    by_area: dict[int, list[tuple[int, int]]] = {}
    for h in range(2, side_max + 1):
        for w in range(2, side_max + 1):
            by_area.setdefault(h * w, []).append((h, w))
    areas = [area for area in by_area if area <= max(12, (grid_h * grid_w) // 5)]
    if len(areas) < count:
        return None

    chosen_areas = random.sample(areas, count)
    dims = [random.choice(by_area[area]) for area in chosen_areas]
    random.shuffle(dims)
    return dims


def _random_rect(h: int, w: int, grid_h: int, grid_w: int) -> dict[str, int] | None:
    if h + 2 > grid_h or w + 2 > grid_w:
        return None
    top = random.randint(1, grid_h - h - 1)
    left = random.randint(1, grid_w - w - 1)
    return {"top": top, "left": left, "h": h, "w": w, "area": int(h * w)}


def _place_overlap_pair(
    large_hw: tuple[int, int], small_hw: tuple[int, int], grid_h: int, grid_w: int
) -> tuple[dict[str, int], dict[str, int]] | None:
    large = _random_rect(large_hw[0], large_hw[1], grid_h, grid_w)
    if large is None:
        return None
    lh, lw = large_hw
    sh, sw = small_hw
    large_area = lh * lw
    small_area = sh * sw

    min_top = max(1, int(large["top"]) - sh + 1)
    max_top = min(grid_h - sh - 1, int(large["top"] + large["h"] - 1))
    min_left = max(1, int(large["left"]) - sw + 1)
    max_left = min(grid_w - sw - 1, int(large["left"] + large["w"] - 1))
    if min_top > max_top or min_left > max_left:
        return None

    for _ in range(PLACEMENT_TRIES):
        top = random.randint(min_top, max_top)
        left = random.randint(min_left, max_left)
        small = {"top": top, "left": left, "h": sh, "w": sw, "area": int(small_area)}
        overlap = _overlap_area(large, small)
        if overlap <= 0:
            continue
        if overlap >= small_area:
            continue
        if overlap >= max(large_area // 2, 1):
            continue
        if small_area - overlap < max(2, min(sh, sw)):
            continue
        if large_area - overlap < max(2, min(lh, lw)):
            continue
        return large, small
    return None


def _place_rectangles(
    dims: list[tuple[int, int]], grid_h: int, grid_w: int, want_overlap: bool
) -> list[dict[str, int]] | None:
    count = len(dims)
    rects: list[dict[str, int] | None] = [None] * count
    order = sorted(range(count), key=lambda idx: dims[idx][0] * dims[idx][1], reverse=True)

    overlap_pair: tuple[int, int] | None = None
    if want_overlap and count >= 2:
        overlap_pair = tuple(random.sample(order, 2))
        a, b = overlap_pair
        if dims[a][0] * dims[a][1] < dims[b][0] * dims[b][1]:
            a, b = b, a
        pair = _place_overlap_pair(dims[a], dims[b], grid_h, grid_w)
        if pair is None:
            return None
        rects[a], rects[b] = pair
        overlap_pair = (a, b)

    for idx in order:
        if rects[idx] is not None:
            continue
        h, w = dims[idx]
        placed = None
        for _ in range(PLACEMENT_TRIES):
            cand = _random_rect(h, w, grid_h, grid_w)
            if cand is None:
                break
            ok = True
            for j, other in enumerate(rects):
                if other is None:
                    continue
                allowed_pair = overlap_pair is not None and {idx, j} == set(overlap_pair)
                if allowed_pair:
                    continue
                if _too_close(cand, other, gap=1):
                    ok = False
                    break
            if ok:
                placed = cand
                break
        if placed is None:
            return None
        rects[idx] = placed

    return [rect for rect in rects if rect is not None]


def _cells_clear(grid: np.ndarray, bg: int, cells: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cell_set:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != bg:
            return False
    for r, c in cell_set:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cell_set and int(grid[nr, nc]) != bg:
                return False
    return True


def _scatter_one_color(
    grid: np.ndarray, bg: int, color: int, rect: dict[str, int]
) -> bool:
    top = int(rect["top"])
    left = int(rect["left"])
    h = int(rect["h"])
    w = int(rect["w"])
    grid_h, grid_w = grid.shape

    max_m = min(3, top, grid_h - (top + h))
    max_n = min(3, left, grid_w - (left + w))
    if max_m < 1 or max_n < 1:
        return False

    pairs = [(m, n) for m in range(1, max_m + 1) for n in range(1, max_n + 1)]
    random.shuffle(pairs)

    for m, n in pairs:
        r0 = top - m
        r1 = top + h - 1 + m
        c0 = left - n
        c1 = left + w - 1 + n
        top_starts = list(range(c0 + 1, c1 - w + 1))
        side_starts = list(range(r0 + 1, r1 - h + 1))
        if not top_starts or not side_starts:
            continue
        for _ in range(SEGMENT_TRIES):
            ht0 = random.choice(top_starts)
            ht1 = random.choice(top_starts)
            vt0 = random.choice(side_starts)
            vt1 = random.choice(side_starts)

            cells = []
            cells.extend((r0, c) for c in range(ht0, ht0 + w))
            cells.extend((r1, c) for c in range(ht1, ht1 + w))
            cells.extend((r, c0) for r in range(vt0, vt0 + h))
            cells.extend((r, c1) for r in range(vt1, vt1 + h))

            if len(set(cells)) != 2 * (h + w):
                continue
            if not _cells_clear(grid, bg, cells):
                continue

            probe = np.full_like(grid, bg)
            for r, c in cells:
                probe[r, c] = color
            comps = _components(probe == color)
            if len(comps) != 4:
                continue

            for r, c in cells:
                grid[r, c] = color
            return True
    return False


def _render_expected(shape: tuple[int, int], bg: int, objects: list[dict[str, int]]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for obj in sorted(objects, key=_draw_order_key):
        top = int(obj["top"])
        left = int(obj["left"])
        h = int(obj["h"])
        w = int(obj["w"])
        out[top : top + h, left : left + w] = int(obj["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    count = random.choice((2, 3, 4))
    grid_lo = 18 + 2 * max(0, count - 2)
    grid_h = random.randint(grid_lo, 30)
    grid_w = random.randint(grid_lo, 30)
    dims = _sample_dimensions(count, grid_h, grid_w)
    if dims is None:
        return None

    want_overlap = count >= 2 and random.random() < 0.5
    rects = _place_rectangles(dims, grid_h, grid_w, want_overlap)
    if rects is None or len(rects) != count:
        return None

    bg = random.choice(ALL_COLORS)
    colors = random.sample([c for c in ALL_COLORS if c != bg], count)
    objects: list[dict[str, int]] = []
    for color, dim, rect in zip(colors, dims, rects, strict=False):
        objects.append(
            {
                "color": int(color),
                "top": int(rect["top"]),
                "left": int(rect["left"]),
                "h": int(dim[0]),
                "w": int(dim[1]),
                "area": int(dim[0] * dim[1]),
            }
        )

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    place_order = sorted(objects, key=lambda obj: (-int(obj["area"]), int(obj["top"]), int(obj["left"])))
    for obj in place_order:
        if not _scatter_one_color(grid, bg, int(obj["color"]), obj):
            return None

    if _get_bg(grid) != bg:
        return None

    expected = _render_expected(grid.shape, bg, objects)
    try:
        predicted = main(grid)
    except Exception:
        return None
    if not np.array_equal(predicted, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid input")
