from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _same_color_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "size": len(coords),
                }
            )
    return comps


def _is_horizontal_domino(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return comp["size"] == 2 and r0 == r1 and c1 == c0 + 1


def _is_rectangular_ring(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3 or comp["size"] != 2 * h + 2 * w - 4:
        return False
    pts = set(comp["coords"])
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            on_border = rr in (r0, r1) or cc in (c0, c1)
            if on_border != ((rr, cc) in pts):
                return False
    return True


def _contains_bbox(outer: dict, inner: dict) -> bool:
    r0, c0, r1, c1 = outer["bbox"]
    rr0, cc0, rr1, cc1 = inner["bbox"]
    return r0 < rr0 and c0 < cc0 and r1 > rr1 and c1 > cc1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each nested grey frame's innermost cavity with the legend color at the matching depth rank.

    Input structure:
    - Background is black.
    - On the left side, a vertical legend is made of several separate 1x2 horizontal color blocks, ordered from top to bottom.
    - On the right side, several one-pixel-thick grey rectangular frame components appear; each outer frame may contain smaller grey frames separated by black gaps.

    Rule:
    - Read the legend from top to bottom so the first block gives depth 1, the second gives depth 2, and so on.
    - For each outermost grey frame, count how many nested grey frames belong to its containment chain, including the outer frame itself.
    - Find the innermost frame in that chain and fill only its interior cavity with the legend color at the same depth rank.

    Key insight:
    - The needed fill color is not tied to the frame's size or position; it comes from the frame stack depth, which must be converted into a top-to-bottom legend lookup.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    comps = _same_color_components(g, bg)

    ring_by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in comps:
        if _is_rectangular_ring(comp):
            ring_by_color[int(comp["color"])].append(comp)
    assert ring_by_color, "expected at least one rectangular ring color"

    frame_color = max(
        ring_by_color,
        key=lambda color: (len(ring_by_color[color]), sum(comp["size"] for comp in ring_by_color[color])),
    )
    rings = ring_by_color[frame_color]

    palette_blocks = [
        comp for comp in comps if int(comp["color"]) != frame_color and _is_horizontal_domino(comp)
    ]
    assert palette_blocks, "expected at least one legend block"
    legend_col = min(comp["bbox"][1] for comp in palette_blocks)
    legend = [comp for comp in palette_blocks if comp["bbox"][1] == legend_col]
    legend.sort(key=lambda comp: comp["bbox"][0])
    legend_colors = [int(comp["color"]) for comp in legend]

    children: list[list[int]] = [[] for _ in rings]
    parents: list[list[int]] = [[] for _ in rings]
    for i, outer in enumerate(rings):
        for j, inner in enumerate(rings):
            if i != j and _contains_bbox(outer, inner):
                parents[j].append(i)

    for i, outer in enumerate(rings):
        for j, inner in enumerate(rings):
            if i == j or not _contains_bbox(outer, inner):
                continue
            if any(k not in (i, j) and _contains_bbox(outer, rings[k]) and _contains_bbox(rings[k], inner) for k in range(len(rings))):
                continue
            children[i].append(j)

    depth_cache: dict[int, int] = {}
    leaf_cache: dict[int, int] = {}

    def chain_depth(idx: int) -> int:
        if idx not in depth_cache:
            depth_cache[idx] = 1 + max((chain_depth(child) for child in children[idx]), default=0)
        return depth_cache[idx]

    def deepest_leaf(idx: int) -> int:
        if idx not in leaf_cache:
            if not children[idx]:
                leaf_cache[idx] = idx
            else:
                best_child = max(children[idx], key=chain_depth)
                leaf_cache[idx] = deepest_leaf(best_child)
        return leaf_cache[idx]

    outers = [idx for idx in range(len(rings)) if not parents[idx]]
    out = g.copy()
    for idx in outers:
        depth = chain_depth(idx)
        assert 1 <= depth <= len(legend_colors), "legend must cover every nesting depth"
        leaf = rings[deepest_leaf(idx)]
        r0, c0, r1, c1 = leaf["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = legend_colors[depth - 1]

    return out


BG = 0
FRAME_COLOR = 5
PALETTE_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
OBJECT_COUNT_CHOICES = [2, 3]
DEPTH_CAP = {2: 5, 3: 4}
MAX_LEGEND_EXTRA = 2
MAX_GENERATION_ATTEMPTS = 400


def _outer_size(depth: int, cavity: int) -> int:
    return cavity + 2 + 4 * (depth - 1)


def _sample_object(depth: int, rng: random.Random) -> dict:
    cavity_max = {1: 4, 2: 4, 3: 3, 4: 2, 5: 2}[depth]
    cavity_h = rng.randint(1, cavity_max)
    cavity_w = rng.randint(1, cavity_max)
    return {
        "depth": depth,
        "cavity_h": cavity_h,
        "cavity_w": cavity_w,
        "outer_h": _outer_size(depth, cavity_h),
        "outer_w": _outer_size(depth, cavity_w),
    }


def _random_partition(total: int, parts: int, rng: random.Random) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    points = [0] + cuts + [total]
    return [points[i + 1] - points[i] for i in range(parts)]


def _sample_layout(objects: list[dict], rng: random.Random) -> tuple[list[tuple[int, int]], int, int]:
    n = len(objects)
    n_rows = rng.randint(1, min(n, 3))
    counts = _random_partition(n, n_rows, rng)
    order = list(range(n))
    rng.shuffle(order)

    rows: list[dict] = []
    cursor = 0
    for count in counts:
        row_ids = order[cursor : cursor + count]
        cursor += count
        gaps = [rng.randint(2, 4) for _ in range(max(0, count - 1))]
        row_width = sum(objects[idx]["outer_w"] for idx in row_ids) + sum(gaps)
        row_height = max(objects[idx]["outer_h"] for idx in row_ids)
        rows.append(
            {
                "ids": row_ids,
                "gaps": gaps,
                "width": row_width,
                "height": row_height,
            }
        )

    v_gaps = [rng.randint(2, 4) for _ in range(max(0, n_rows - 1))]
    layout_h = sum(row["height"] for row in rows) + sum(v_gaps)
    layout_w = max(row["width"] for row in rows)

    positions: list[tuple[int, int] | None] = [None] * n
    row_top = 0
    for row_index, row in enumerate(rows):
        row_left = rng.randint(0, layout_w - row["width"]) if row["width"] < layout_w else 0
        x = row_left
        for k, obj_idx in enumerate(row["ids"]):
            obj = objects[obj_idx]
            y_slack = row["height"] - obj["outer_h"]
            y = row_top + (rng.randint(0, y_slack) if y_slack else 0)
            positions[obj_idx] = (y, x)
            x += obj["outer_w"]
            if k < len(row["gaps"]):
                x += row["gaps"][k]
        if row_index < len(v_gaps):
            row_top += row["height"] + v_gaps[row_index]

    return [pos for pos in positions if pos is not None], layout_h, layout_w


def _legend_offsets(length: int, rng: random.Random) -> list[int]:
    offsets = [0]
    for _ in range(1, length):
        offsets.append(offsets[-1] + 1 + rng.randint(0, 1))
    return offsets


def _draw_ring(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    grid[top, left : left + width] = color
    grid[top + height - 1, left : left + width] = color
    grid[top : top + height, left] = color
    grid[top : top + height, left + width - 1] = color


def _draw_object(grid: np.ndarray, top: int, left: int, obj: dict, color: int) -> None:
    depth = obj["depth"]
    cavity_h = obj["cavity_h"]
    cavity_w = obj["cavity_w"]
    for level in range(depth):
        ring_h = _outer_size(depth - level, cavity_h)
        ring_w = _outer_size(depth - level, cavity_w)
        offset = 2 * level
        _draw_ring(grid, top + offset, left + offset, ring_h, ring_w, color)


def _innermost_slice(top: int, left: int, obj: dict) -> tuple[slice, slice]:
    inner_top = top + 2 * (obj["depth"] - 1) + 1
    inner_left = left + 2 * (obj["depth"] - 1) + 1
    return (
        slice(inner_top, inner_top + obj["cavity_h"]),
        slice(inner_left, inner_left + obj["cavity_w"]),
    )


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_objects = rng.choice(OBJECT_COUNT_CHOICES)
    depth_cap = DEPTH_CAP[num_objects]
    depths = [1] + rng.sample(list(range(2, depth_cap + 1)), num_objects - 1)
    rng.shuffle(depths)

    objects = [_sample_object(depth, rng) for depth in depths]
    positions, layout_h, layout_w = _sample_layout(objects, rng)

    legend_len = max(depths) + rng.randint(0, min(MAX_LEGEND_EXTRA, 7 - max(depths)))
    legend_colors = rng.sample(PALETTE_POOL, legend_len)
    legend_rows = _legend_offsets(legend_len, rng)
    legend_span = legend_rows[-1] + 1

    legend_x = rng.randint(0, 1)
    legend_gap = rng.randint(2, 4)
    min_h = max(layout_h, legend_span)
    min_w = legend_x + 2 + legend_gap + layout_w
    if min_h > 30 or min_w > 30:
        return None

    h = rng.randint(min_h, min(30, min_h + 8))
    w = rng.randint(min_w, min(30, min_w + 8))
    legend_y = rng.randint(0, h - legend_span)
    layout_y = rng.randint(0, h - layout_h)
    layout_x = rng.randint(legend_x + 2 + legend_gap, w - layout_w)

    inp = np.full((h, w), BG, dtype=int)
    for row_offset, color in zip(legend_rows, legend_colors):
        rr = legend_y + row_offset
        inp[rr, legend_x : legend_x + 2] = color

    expected = inp.copy()
    for obj, (rel_top, rel_left) in zip(objects, positions):
        top = layout_y + rel_top
        left = layout_x + rel_left
        _draw_object(inp, top, left, obj, FRAME_COLOR)
        _draw_object(expected, top, left, obj, FRAME_COLOR)
        rs, cs = _innermost_slice(top, left, obj)
        expected[rs, cs] = legend_colors[obj["depth"] - 1]

    values, counts = np.unique(inp, return_counts=True)
    if int(values[int(np.argmax(counts))]) != BG:
        return None

    try:
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame ARC input")
