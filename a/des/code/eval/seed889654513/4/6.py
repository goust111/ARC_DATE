from __future__ import annotations

import random
from functools import lru_cache

import numpy as np


BLACK = 0
COLOR_POOL = tuple(range(1, 10))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _legend_colors(grid: np.ndarray) -> list[int]:
    return [int(v) for v in grid[:, -1] if int(v) != BLACK]


def _segment_from_coords(color: int, coords: list[tuple[int, int]]) -> dict[str, int]:
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    if len(rows) == 1:
        row = rows[0]
        c0 = cols[0]
        length = len(cols)
        assert cols == list(range(c0, c0 + length))
        return {"color": color, "orientation": 0, "r": row, "c": c0, "length": length}
    if len(cols) == 1:
        col = cols[0]
        r0 = rows[0]
        length = len(rows)
        assert rows == list(range(r0, r0 + length))
        return {"color": color, "orientation": 1, "r": r0, "c": col, "length": length}
    raise AssertionError("component is not a straight segment")


def _extract_segments(body: np.ndarray) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(body) if int(v) != BLACK):
        comps = _components(body == color)
        segments = [_segment_from_coords(color, comp) for comp in comps]
        groups.append({"color": color, "segments": segments})
    return groups


def _partition_masks(lengths: list[int], target: int) -> list[tuple[int, int]]:
    n = len(lengths)
    full = (1 << n) - 1
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for mask in range(1, 1 << n):
        if (mask & 1) == 0:
            continue
        total = 0
        for i, length in enumerate(lengths):
            if (mask >> i) & 1:
                total += length
        if total != target:
            continue
        other = full ^ mask
        key = (min(mask, other), max(mask, other))
        if key in seen:
            continue
        seen.add(key)
        out.append((mask, other))
    return out


def _group_from_mask(segments: list[dict[str, int]], mask: int) -> list[dict[str, int]]:
    return [segments[i] for i in range(len(segments)) if (mask >> i) & 1]


def _prefix_offsets(lengths: list[int]) -> list[int]:
    offsets: list[int] = []
    cur = 0
    for length in lengths:
        offsets.append(cur)
        cur += length
    return offsets


def _horizontal_layouts(
    segments: list[dict[str, int]],
    width: int,
    height: int,
    body_h: int,
    body_w: int,
) -> list[dict[str, object]]:
    layouts: list[dict[str, object]] = []
    lengths = [int(seg["length"]) for seg in segments]
    for mask_a, mask_b in _partition_masks(lengths, width):
        groups = (_group_from_mask(segments, mask_a), _group_from_mask(segments, mask_b))
        for top_idx, bottom_idx in ((0, 1), (1, 0)):
            top = sorted(groups[top_idx], key=lambda seg: (int(seg["c"]), int(seg["r"]), int(seg["length"])))
            bottom = sorted(groups[bottom_idx], key=lambda seg: (int(seg["c"]), int(seg["r"]), int(seg["length"])))
            top_offsets = _prefix_offsets([int(seg["length"]) for seg in top])
            bottom_offsets = _prefix_offsets([int(seg["length"]) for seg in bottom])
            row_costs = []
            for r0 in range(body_h - height + 1):
                bottom_row = r0 + height - 1
                cost = 0
                for seg in top:
                    cost += abs(int(seg["r"]) - r0)
                for seg in bottom:
                    cost += abs(int(seg["r"]) - bottom_row)
                row_costs.append(cost)
            col_costs = []
            for c0 in range(body_w - width + 1):
                cost = 0
                for seg, offset in zip(top, top_offsets):
                    cost += abs(int(seg["c"]) - (c0 + offset))
                for seg, offset in zip(bottom, bottom_offsets):
                    cost += abs(int(seg["c"]) - (c0 + offset))
                col_costs.append(cost)
            layouts.append(
                {
                    "top": top,
                    "bottom": bottom,
                    "row_costs": row_costs,
                    "col_costs": col_costs,
                }
            )
    return layouts


def _vertical_layouts(
    segments: list[dict[str, int]],
    side_height: int,
    full_height: int,
    width: int,
    body_h: int,
    body_w: int,
) -> list[dict[str, object]]:
    layouts: list[dict[str, object]] = []
    lengths = [int(seg["length"]) for seg in segments]
    for mask_a, mask_b in _partition_masks(lengths, side_height):
        groups = (_group_from_mask(segments, mask_a), _group_from_mask(segments, mask_b))
        for left_idx, right_idx in ((0, 1), (1, 0)):
            left = sorted(groups[left_idx], key=lambda seg: (int(seg["r"]), int(seg["c"]), int(seg["length"])))
            right = sorted(groups[right_idx], key=lambda seg: (int(seg["r"]), int(seg["c"]), int(seg["length"])))
            left_offsets = _prefix_offsets([int(seg["length"]) for seg in left])
            right_offsets = _prefix_offsets([int(seg["length"]) for seg in right])
            row_costs = []
            for r0 in range(body_h - full_height + 1):
                base = r0 + 1
                cost = 0
                for seg, offset in zip(left, left_offsets):
                    cost += abs(int(seg["r"]) - (base + offset))
                for seg, offset in zip(right, right_offsets):
                    cost += abs(int(seg["r"]) - (base + offset))
                row_costs.append(cost)
            col_costs = []
            for c0 in range(body_w - width + 1):
                right_col = c0 + width - 1
                cost = 0
                for seg in left:
                    cost += abs(int(seg["c"]) - c0)
                for seg in right:
                    cost += abs(int(seg["c"]) - right_col)
                col_costs.append(cost)
            layouts.append(
                {
                    "left": left,
                    "right": right,
                    "row_costs": row_costs,
                    "col_costs": col_costs,
                }
            )
    return layouts


def _best_rectangle(
    segments: list[dict[str, int]],
    body_h: int,
    body_w: int,
) -> dict[str, int]:
    horiz = [seg for seg in segments if int(seg["orientation"]) == 0]
    vert = [seg for seg in segments if int(seg["orientation"]) == 1]
    total_h = sum(int(seg["length"]) for seg in horiz)
    total_v = sum(int(seg["length"]) for seg in vert)
    assert horiz and vert and total_h % 2 == 0 and total_v % 2 == 0
    width = total_h // 2
    side_height = total_v // 2
    height = side_height + 2
    assert width >= 2 and side_height >= 1
    assert width <= body_w and height <= body_h

    h_layouts = _horizontal_layouts(horiz, width, height, body_h, body_w)
    v_layouts = _vertical_layouts(vert, side_height, height, width, body_h, body_w)
    assert h_layouts and v_layouts

    best: tuple[int, int, int, int, int] | None = None
    for hi, h_layout in enumerate(h_layouts):
        h_row_costs = h_layout["row_costs"]
        h_col_costs = h_layout["col_costs"]
        for vi, v_layout in enumerate(v_layouts):
            row_costs = [int(a) + int(b) for a, b in zip(h_row_costs, v_layout["row_costs"])]
            col_costs = [int(a) + int(b) for a, b in zip(h_col_costs, v_layout["col_costs"])]
            r_cost = min(row_costs)
            c_cost = min(col_costs)
            r0 = row_costs.index(r_cost)
            c0 = col_costs.index(c_cost)
            cand = (r_cost + c_cost, r0, c0, hi, vi)
            if best is None or cand < best:
                best = cand
    assert best is not None
    _cost, r0, c0, _hi, _vi = best
    return {"r0": r0, "c0": c0, "height": height, "width": width}


def _draw_outline(grid: np.ndarray, r0: int, c0: int, height: int, width: int, color: int) -> None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    if height > 2:
        grid[r0 + 1 : r1, c0] = color
        grid[r0 + 1 : r1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble same-color line fragments into rectangles, then fill them by area rank.

    The rightmost column is a vertical legend. Each non-black pixel in that column
    is a fill color; the top legend color goes to the smallest reconstructed rectangle,
    the next legend color to the second smallest, and so on. The output drops the
    legend column and shows only the rebuilt rectangles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 2

    legend = _legend_colors(g)
    body = g[:, :-1]
    out = np.zeros_like(body)

    rects: list[dict[str, int]] = []
    for group in _extract_segments(body):
        rect = _best_rectangle(group["segments"], body.shape[0], body.shape[1])
        color = int(group["color"])
        rect["color"] = color
        rect["area"] = max(0, (rect["height"] - 2) * (rect["width"] - 2))
        rects.append(rect)

    assert len(legend) >= len(rects)
    rects.sort(key=lambda rect: (int(rect["area"]), int(rect["r0"]), int(rect["c0"]), int(rect["height"]), int(rect["width"])))

    for rank, rect in enumerate(rects):
        r0 = int(rect["r0"])
        c0 = int(rect["c0"])
        height = int(rect["height"])
        width = int(rect["width"])
        outline = int(rect["color"])
        fill = int(legend[rank])
        _draw_outline(out, r0, c0, height, width, outline)
        if height > 2 and width > 2:
            out[r0 + 1 : r0 + height - 1, c0 + 1 : c0 + width - 1] = fill

    return out


# --------------- generator ---------------

BODY_H_RANGE = (18, 27)
BODY_W_RANGE = (18, 27)
RECT_COUNT_RANGE = (2, 4)
HALO_RANGE = (3, 5)
MAX_GENERATION_ATTEMPTS = 400
RECT_PLACE_ATTEMPTS = 200
SEGMENT_PLACE_ATTEMPTS = 300


@lru_cache(maxsize=None)
def _compositions(total: int, max_parts: int = 3) -> tuple[tuple[int, ...], ...]:
    if total == 0:
        return ((),)
    out: list[tuple[int, ...]] = []
    for first in range(2, 6):
        if first > total:
            break
        for tail in _compositions(total - first, max_parts=max_parts):
            if len(tail) + 1 > max_parts:
                continue
            out.append((first,) + tail)
    return tuple(out)


def _feasible_dimensions() -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for height in range(4, 11):
        for width in range(4, 12):
            if _compositions(width) and _compositions(height - 2):
                dims.append((height, width))
    return dims


FEASIBLE_DIMS = _feasible_dimensions()


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _choose_rectangles(count: int) -> list[dict[str, int]] | None:
    pool = FEASIBLE_DIMS[:]
    random.shuffle(pool)
    rects: list[dict[str, int]] = []
    used_areas: set[int] = set()
    for height, width in pool:
        area = (height - 2) * (width - 2)
        if area in used_areas:
            continue
        rects.append({"height": height, "width": width, "area": area, "halo": random.randint(*HALO_RANGE)})
        used_areas.add(area)
        if len(rects) == count:
            return rects
    return None


def _place_rectangles(body_h: int, body_w: int, rects: list[dict[str, int]]) -> bool:
    taken: list[tuple[int, int, int, int]] = []
    rects.sort(key=lambda rect: (2 * int(rect["halo"]) + int(rect["height"])) * (2 * int(rect["halo"]) + int(rect["width"])), reverse=True)
    for rect in rects:
        height = int(rect["height"])
        width = int(rect["width"])
        halo = int(rect["halo"])
        if height + 2 * halo > body_h or width + 2 * halo > body_w:
            return False
        placed = False
        for _ in range(RECT_PLACE_ATTEMPTS):
            r0 = random.randint(halo, body_h - height - halo)
            c0 = random.randint(halo, body_w - width - halo)
            expanded = (r0 - halo, c0 - halo, r0 + height - 1 + halo, c0 + width - 1 + halo)
            if any(_boxes_overlap(expanded, other) for other in taken):
                continue
            rect["r0"] = r0
            rect["c0"] = c0
            rect["zone_r0"] = expanded[0]
            rect["zone_c0"] = expanded[1]
            rect["zone_r1"] = expanded[2]
            rect["zone_c1"] = expanded[3]
            taken.append(expanded)
            placed = True
            break
        if not placed:
            return False
    return True


def _build_segments(rect: dict[str, int], color: int) -> list[dict[str, int]]:
    r0 = int(rect["r0"])
    c0 = int(rect["c0"])
    height = int(rect["height"])
    width = int(rect["width"])
    r1 = r0 + height - 1
    c1 = c0 + width - 1

    segments: list[dict[str, int]] = []

    top_parts = random.choice(_compositions(width))
    bottom_parts = random.choice(_compositions(width))
    left_parts = random.choice(_compositions(height - 2))
    right_parts = random.choice(_compositions(height - 2))

    cur = c0
    for length in top_parts:
        segments.append({"color": color, "orientation": 0, "r": r0, "c": cur, "length": int(length)})
        cur += int(length)

    cur = c0
    for length in bottom_parts:
        segments.append({"color": color, "orientation": 0, "r": r1, "c": cur, "length": int(length)})
        cur += int(length)

    cur = r0 + 1
    for length in left_parts:
        segments.append({"color": color, "orientation": 1, "r": cur, "c": c0, "length": int(length)})
        cur += int(length)

    cur = r0 + 1
    for length in right_parts:
        segments.append({"color": color, "orientation": 1, "r": cur, "c": c1, "length": int(length)})
        cur += int(length)

    random.shuffle(segments)
    return segments


def _candidate_positions(seg: dict[str, int], rect: dict[str, int], body_h: int, body_w: int) -> list[tuple[int, int]]:
    length = int(seg["length"])
    positions: list[tuple[int, int]] = []
    if int(seg["orientation"]) == 0:
        row_lo = int(rect["zone_r0"])
        row_hi = int(rect["zone_r1"])
        col_lo = int(rect["zone_c0"])
        col_hi = int(rect["zone_c1"]) - length + 1
    else:
        row_lo = int(rect["zone_r0"])
        row_hi = int(rect["zone_r1"]) - length + 1
        col_lo = int(rect["zone_c0"])
        col_hi = int(rect["zone_c1"])
    row_lo = max(0, row_lo)
    col_lo = max(0, col_lo)
    row_hi = min(body_h - (1 if int(seg["orientation"]) == 0 else length), row_hi)
    col_hi = min(body_w - (length if int(seg["orientation"]) == 0 else 1), col_hi)
    for r in range(row_lo, row_hi + 1):
        for c in range(col_lo, col_hi + 1):
            positions.append((r, c))
    positions.sort(key=lambda pos: (abs(pos[0] - int(seg["r"])) + abs(pos[1] - int(seg["c"])), random.random()))
    return positions


def _can_place_segment(occupied: np.ndarray, orientation: int, r0: int, c0: int, length: int) -> bool:
    h, w = occupied.shape
    rr1 = r0 + (1 if orientation == 0 else length) - 1
    cc1 = c0 + (length if orientation == 0 else 1) - 1
    if rr1 >= h or cc1 >= w:
        return False
    r_lo = max(0, r0 - 1)
    c_lo = max(0, c0 - 1)
    r_hi = min(h - 1, rr1 + 1)
    c_hi = min(w - 1, cc1 + 1)
    return not bool(occupied[r_lo : r_hi + 1, c_lo : c_hi + 1].any())


def _paint_segment(grid: np.ndarray, occupied: np.ndarray, seg: dict[str, int], r0: int, c0: int) -> None:
    color = int(seg["color"])
    length = int(seg["length"])
    if int(seg["orientation"]) == 0:
        grid[r0, c0 : c0 + length] = color
        occupied[r0, c0 : c0 + length] = True
    else:
        grid[r0 : r0 + length, c0] = color
        occupied[r0 : r0 + length, c0] = True


def _scatter_segments(
    body: np.ndarray,
    occupied: np.ndarray,
    rect: dict[str, int],
    segments: list[dict[str, int]],
) -> bool:
    order = sorted(range(len(segments)), key=lambda i: int(segments[i]["length"]), reverse=True)
    placements: list[tuple[int, int] | None] = [None] * len(segments)
    moved = False
    for idx in order:
        seg = segments[idx]
        candidates = _candidate_positions(seg, rect, body.shape[0], body.shape[1])
        if not candidates:
            return False
        chosen = None
        for r0, c0 in candidates[:SEGMENT_PLACE_ATTEMPTS]:
            if not _can_place_segment(occupied, int(seg["orientation"]), r0, c0, int(seg["length"])):
                continue
            chosen = (r0, c0)
            break
        if chosen is None:
            return False
        placements[idx] = chosen
        if chosen != (int(seg["r"]), int(seg["c"])):
            moved = True
        _paint_segment(body, occupied, seg, chosen[0], chosen[1])
    return moved


def _expected_output(body_h: int, body_w: int, rects: list[dict[str, int]], legend_colors: list[int]) -> np.ndarray:
    out = np.zeros((body_h, body_w), dtype=int)
    ordered = sorted(rects, key=lambda rect: int(rect["area"]))
    for rank, rect in enumerate(ordered):
        outline = int(rect["outline"])
        fill = int(legend_colors[rank])
        r0 = int(rect["r0"])
        c0 = int(rect["c0"])
        height = int(rect["height"])
        width = int(rect["width"])
        _draw_outline(out, r0, c0, height, width, outline)
        out[r0 + 1 : r0 + height - 1, c0 + 1 : c0 + width - 1] = fill
    return out


def _gen_attempt() -> np.ndarray | None:
    body_h = random.randint(*BODY_H_RANGE)
    body_w = random.randint(*BODY_W_RANGE)
    rect_count = random.randint(*RECT_COUNT_RANGE)

    rects = _choose_rectangles(rect_count)
    if rects is None:
        return None
    if not _place_rectangles(body_h, body_w, rects):
        return None

    if 2 * rect_count - 1 > body_h:
        return None

    colors = random.sample(COLOR_POOL, 2 * rect_count)
    outline_colors = colors[:rect_count]
    legend_colors = colors[rect_count:]
    random.shuffle(legend_colors)

    for rect, outline in zip(rects, outline_colors):
        rect["outline"] = outline

    expected = _expected_output(body_h, body_w, rects, legend_colors)

    body = np.zeros((body_h, body_w), dtype=int)
    occupied = np.zeros((body_h, body_w), dtype=bool)
    moved_count = 0
    for rect in rects:
        segments = _build_segments(rect, int(rect["outline"]))
        if not _scatter_segments(body, occupied, rect, segments):
            return None
        moved_count += 1
    if moved_count == 0:
        return None

    full = np.zeros((body_h, body_w + 1), dtype=int)
    full[:, :body_w] = body

    legend_start = random.randint(0, body_h - (2 * rect_count - 1))
    for i, color in enumerate(legend_colors):
        full[legend_start + 2 * i, body_w] = int(color)

    if np.array_equal(body, expected):
        return None
    if not np.array_equal(main(full), expected):
        return None
    return full


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 6")
