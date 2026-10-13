import random

import numpy as np


BG = 0
GREY = 5
LINE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 400


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
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_rect_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    hh = r1 - r0 + 1
    ww = c1 - c0 + 1
    if hh < 3 or ww < 3:
        return False
    expected = 2 * (hh + ww) - 4
    if len(coords) != expected:
        return False

    pts = {tuple(map(int, cell)) for cell in coords}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in pts):
                return False
    return True


def _legend_pairs(grid: np.ndarray, bbox0: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox0
    pairs: list[tuple[int, int]] = []
    for r in range(r0 + 1, r1):
        cells = [(c, int(grid[r, c])) for c in range(c0 + 1, c1) if int(grid[r, c]) not in (BG, GREY)]
        if len(cells) != 2:
            continue
        cells.sort()
        if cells[1][0] == cells[0][0] + 1:
            pairs.append((cells[0][1], cells[1][1]))
    return pairs


def _find_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[tuple[tuple[int, int, int, int], list[tuple[int, int]]]] = []
    for comp in _components(grid == GREY):
        if not _is_rect_frame(comp):
            continue
        box = _bbox(comp)
        r0, c0, r1, c1 = box
        touches_edge = r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1
        if not touches_edge:
            continue
        pairs = _legend_pairs(grid, box)
        if len(pairs) >= 2:
            candidates.append((box, pairs))
    assert len(candidates) == 1, "expected exactly one grey legend frame"
    return candidates[0]


def _priority_order(pairs: list[tuple[int, int]]) -> list[int]:
    colors = sorted({int(a) for a, _ in pairs} | {int(b) for _, b in pairs})
    outgoing: dict[int, list[int]] = {color: [] for color in colors}
    indegree: dict[int, int] = {color: 0 for color in colors}

    for high, low in pairs:
        high = int(high)
        low = int(low)
        if low not in outgoing[high]:
            outgoing[high].append(low)
            indegree[low] += 1

    heads = [color for color in colors if indegree[color] == 0]
    assert len(heads) == 1, "priority legend must define a single chain head"

    order: list[int] = []
    current = heads[0]
    seen: set[int] = set()
    while True:
        assert current not in seen, "priority chain cannot contain a cycle"
        seen.add(current)
        order.append(current)
        nxt = outgoing[current]
        if not nxt:
            break
        assert len(nxt) == 1, "each legend color may directly dominate only one lower color"
        current = nxt[0]

    assert len(order) == len(colors), "priority legend must cover a full chain"
    return order


def _majority_color(values: list[int]) -> int:
    assert values, "majority_color needs at least one sample"
    arr = np.asarray(values, dtype=int)
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _extract_segments(
    grid: np.ndarray, legend_mask: np.ndarray
) -> tuple[list[dict], list[dict], np.ndarray, np.ndarray]:
    h, w = grid.shape
    occupied = (grid != BG) & (~legend_mask)

    horiz_raw: list[tuple[int, int, int]] = []
    for r in range(h):
        c = 0
        while c < w:
            if not occupied[r, c]:
                c += 1
                continue
            c0 = c
            while c + 1 < w and occupied[r, c + 1]:
                c += 1
            c1 = c
            if c1 - c0 + 1 >= 2:
                horiz_raw.append((r, c0, c1))
            c += 1

    vert_raw: list[tuple[int, int, int]] = []
    for c in range(w):
        r = 0
        while r < h:
            if not occupied[r, c]:
                r += 1
                continue
            r0 = r
            while r + 1 < h and occupied[r + 1, c]:
                r += 1
            r1 = r
            if r1 - r0 + 1 >= 2:
                vert_raw.append((c, r0, r1))
            r += 1

    h_id = np.full((h, w), -1, dtype=int)
    v_id = np.full((h, w), -1, dtype=int)
    for idx, (r, c0, c1) in enumerate(horiz_raw):
        h_id[r, c0 : c1 + 1] = idx
    for idx, (c, r0, r1) in enumerate(vert_raw):
        v_id[r0 : r1 + 1, c] = idx

    horiz: list[dict] = []
    for idx, (r, c0, c1) in enumerate(horiz_raw):
        samples = [int(grid[r, c]) for c in range(c0, c1 + 1) if v_id[r, c] == -1]
        if not samples:
            samples = [int(grid[r, c]) for c in range(c0, c1 + 1)]
        color = _majority_color(samples)
        horiz.append(
            {
                "color": color,
                "cells": [(r, c) for c in range(c0, c1 + 1)],
            }
        )

    vert: list[dict] = []
    for idx, (c, r0, r1) in enumerate(vert_raw):
        samples = [int(grid[r, c]) for r in range(r0, r1 + 1) if h_id[r, c] == -1]
        if not samples:
            samples = [int(grid[r, c]) for r in range(r0, r1 + 1)]
        color = _majority_color(samples)
        vert.append(
            {
                "color": color,
                "cells": [(r, c) for r in range(r0, r1 + 1)],
            }
        )

    return horiz, vert, h_id, v_id


def _segment_repaints(segments: list[dict], out: np.ndarray, rank: dict[int, int]) -> dict[tuple[int, int], int]:
    repaints: dict[tuple[int, int], int] = {}

    for seg in segments:
        base_color = int(seg["color"])
        cells = seg["cells"]
        keep = [int(out[r, c]) == base_color for r, c in cells]

        pieces: list[tuple[int, int]] = []
        i = 0
        while i < len(cells):
            if not keep[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(cells) and keep[j + 1]:
                j += 1
            pieces.append((i, j))
            i = j + 1

        if len(pieces) <= 1:
            continue

        lengths = [end - start + 1 for start, end in pieces]
        best_len = min(lengths)
        if lengths.count(best_len) != 1:
            continue

        piece_index = lengths.index(best_len)
        start, end = pieces[piece_index]
        neighbors: list[int] = []
        if start - 1 >= 0 and int(out[cells[start - 1]]) != base_color:
            neighbors.append(int(out[cells[start - 1]]))
        if end + 1 < len(cells) and int(out[cells[end + 1]]) != base_color:
            neighbors.append(int(out[cells[end + 1]]))
        if not neighbors or len(set(neighbors)) != 1:
            continue

        new_color = neighbors[0]
        if new_color not in rank or rank[new_color] >= rank[base_color]:
            continue

        for idx in range(start, end + 1):
            repaints[cells[idx]] = new_color

    return repaints


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve pipe crossings from a grey legend and absorb the shorter severed branch into the stronger color.

    Input structure:
    - The background is black.
    - Away from the legend there are several one-pixel-wide horizontal and vertical
      colored line segments; different colors may cross on a single cell.
    - A small grey rectangular frame sits on the grid edge and encloses several
      two-cell horizontal color pairs encoding direct priority relations.

    Rule:
    - Read each legend pair as "left color overrides right color" and combine all
      pairs into one complete priority chain.
    - At every crossing between a horizontal and a vertical segment, recolor the
      shared cell to whichever segment color is higher in that chain.
    - If that change splits a lower-priority segment into multiple same-color pieces,
      recolor the unique shortest remaining piece to the higher color that cuts it.
      Keep the grey legend unchanged.

    Key insight:
    - The crossing colors cannot be trusted locally; the true override order is only
      encoded by the framed legend, and the shortest disconnected branch is then
      swallowed by the stronger network.
    """
    g = np.asarray(grid, dtype=int)
    legend_box, pairs = _find_legend(g)
    order = _priority_order(pairs)
    rank = {color: idx for idx, color in enumerate(order)}

    legend_mask = np.zeros_like(g, dtype=bool)
    r0, c0, r1, c1 = legend_box
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    horiz, vert, h_id, v_id = _extract_segments(g, legend_mask)
    out = g.copy()

    h, w = g.shape
    for r in range(h):
        for c in range(w):
            hid = int(h_id[r, c])
            vid = int(v_id[r, c])
            if hid < 0 or vid < 0:
                continue
            h_color = int(horiz[hid]["color"])
            v_color = int(vert[vid]["color"])
            if h_color == v_color:
                out[r, c] = h_color
            else:
                out[r, c] = h_color if rank[h_color] < rank[v_color] else v_color

    repaints = {}
    repaints.update(_segment_repaints(horiz, out, rank))
    repaints.update(_segment_repaints(vert, out, rank))
    for (r, c), color in repaints.items():
        out[r, c] = color

    return out


def _legend_bbox(height: int, width: int, pair_count: int) -> tuple[int, int, int, int]:
    legend_h = pair_count + 2
    legend_w = 4
    side = random.choice(("top", "bottom", "left", "right"))

    if side == "top":
        r0 = 0
        c0 = random.randint(0, width - legend_w)
    elif side == "bottom":
        r0 = height - legend_h
        c0 = random.randint(0, width - legend_w)
    elif side == "left":
        r0 = random.randint(0, height - legend_h)
        c0 = 0
    else:
        r0 = random.randint(0, height - legend_h)
        c0 = width - legend_w

    return r0, c0, r0 + legend_h - 1, c0 + legend_w - 1


def _draw_legend(grid: np.ndarray, bbox0: tuple[int, int, int, int], pairs: list[tuple[int, int]]) -> None:
    r0, c0, r1, c1 = bbox0
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY

    rows = list(range(r0 + 1, r1))
    pair_rows = pairs[:]
    random.shuffle(pair_rows)
    for row, (high, low) in zip(rows, pair_rows, strict=False):
        grid[row, c0 + 1] = int(high)
        grid[row, c0 + 2] = int(low)


def _cross_item(low_color: int, high_color: int) -> dict:
    low_horizontal = bool(random.getrandbits(1))

    if low_horizontal:
        low_len = random.randint(6, 9)
        high_len = random.randint(5, 8)
        short_len = random.randint(1, min(3, low_len - 3))
        cross_col = short_len if bool(random.getrandbits(1)) else low_len - 1 - short_len
        cross_row = random.randint(1, high_len - 2)

        inp: dict[tuple[int, int], int] = {}
        out: dict[tuple[int, int], int] = {}
        for c in range(low_len):
            inp[(cross_row, c)] = low_color
            out[(cross_row, c)] = low_color
        for r in range(high_len):
            inp[(r, cross_col)] = high_color
            out[(r, cross_col)] = high_color

        inp[(cross_row, cross_col)] = low_color if bool(random.getrandbits(1)) else high_color
        out[(cross_row, cross_col)] = high_color
        if cross_col < low_len - 1 - cross_col:
            for c in range(cross_col):
                out[(cross_row, c)] = high_color
        else:
            for c in range(cross_col + 1, low_len):
                out[(cross_row, c)] = high_color

        return {
            "shape": (high_len, low_len),
            "input_pixels": inp,
            "output_pixels": out,
        }

    low_len = random.randint(6, 9)
    high_len = random.randint(5, 8)
    short_len = random.randint(1, min(3, low_len - 3))
    cross_row = short_len if bool(random.getrandbits(1)) else low_len - 1 - short_len
    cross_col = random.randint(1, high_len - 2)

    inp = {}
    out = {}
    for r in range(low_len):
        inp[(r, cross_col)] = low_color
        out[(r, cross_col)] = low_color
    for c in range(high_len):
        inp[(cross_row, c)] = high_color
        out[(cross_row, c)] = high_color

    inp[(cross_row, cross_col)] = low_color if bool(random.getrandbits(1)) else high_color
    out[(cross_row, cross_col)] = high_color
    if cross_row < low_len - 1 - cross_row:
        for r in range(cross_row):
            out[(r, cross_col)] = high_color
    else:
        for r in range(cross_row + 1, low_len):
            out[(r, cross_col)] = high_color

    return {
        "shape": (low_len, high_len),
        "input_pixels": inp,
        "output_pixels": out,
    }


def _standalone_item(color: int) -> dict:
    horizontal = bool(random.getrandbits(1))
    length = random.randint(4, 8)
    pixels: dict[tuple[int, int], int] = {}
    if horizontal:
        for c in range(length):
            pixels[(0, c)] = color
        shape = (1, length)
    else:
        for r in range(length):
            pixels[(r, 0)] = color
        shape = (length, 1)
    return {
        "shape": shape,
        "input_pixels": pixels,
        "output_pixels": pixels.copy(),
    }


def _mark_rect(blocked: np.ndarray, bbox0: tuple[int, int, int, int], pad: int) -> None:
    h, w = blocked.shape
    r0, c0, r1, c1 = bbox0
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_items(
    height: int, width: int, legend_box: tuple[int, int, int, int], items: list[dict]
) -> list[tuple[dict, int, int]] | None:
    blocked = np.zeros((height, width), dtype=bool)
    _mark_rect(blocked, legend_box, pad=1)

    placements: list[tuple[dict, int, int]] = []
    items_sorted = sorted(items, key=lambda item: item["shape"][0] * item["shape"][1], reverse=True)
    for item in items_sorted:
        ph, pw = item["shape"]
        positions: list[tuple[int, int]] = []
        for r0 in range(height - ph + 1):
            for c0 in range(width - pw + 1):
                if not np.any(blocked[r0 : r0 + ph, c0 : c0 + pw]):
                    positions.append((r0, c0))
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        placements.append((item, r0, c0))
        _mark_rect(blocked, (r0, c0, r0 + ph - 1, c0 + pw - 1), pad=1)
    return placements


def _paint_pixels(grid: np.ndarray, pixels: dict[tuple[int, int], int], top: int, left: int) -> None:
    for (r, c), color in pixels.items():
        grid[top + r, left + c] = int(color)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(22, 30)
    width = random.randint(22, 30)
    color_count = random.randint(3, 4)
    priority = random.sample(LINE_COLORS, color_count)
    pairs = [(priority[i], priority[i + 1]) for i in range(color_count - 1)]

    legend_box = _legend_bbox(height, width, len(pairs))
    items: list[dict] = []

    for idx in range(1, color_count):
        low = priority[idx]
        if idx == 1:
            high = priority[0]
        else:
            high = priority[random.randint(0, idx - 1)]
        items.append(_cross_item(low, high))

    for _ in range(random.randint(0, 2)):
        items.append(_standalone_item(random.choice(priority)))

    placements = _place_items(height, width, legend_box, items)
    if placements is None:
        return None

    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)
    _draw_legend(inp, legend_box, pairs)
    _draw_legend(out, legend_box, pairs)

    for item, top, left in placements:
        _paint_pixels(inp, item["input_pixels"], top, left)
        _paint_pixels(out, item["output_pixels"], top, left)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid crossing-priority puzzle")
