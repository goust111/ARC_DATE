import itertools
import random
from collections import Counter, deque

import numpy as np


GEN_RETRIES = 400
PALETTE_EDGES = ("top", "bottom", "left", "right")
SLOT_GAP = 1
BASE_SHAPES = (
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=bool),
)


def _neighbors4(r: int, c: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yield r + dr, c + dc


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    neighbor_fn = _neighbors8 if connectivity == 8 else _neighbors4
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in neighbor_fn(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_edge(points: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in points)


def _boundary_from_filled(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                    out[r, c] = True
                    break
    return out


def _find_local_holes(outline_mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [
        pts
        for pts in _components_mask(~outline_mask, connectivity=4)
        if not _touches_edge(pts, outline_mask.shape[0], outline_mask.shape[1])
    ]


def _find_palette(
    grid: np.ndarray, bg: int
) -> tuple[str, list[int], list[tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[tuple[int, str, list[int], list[tuple[int, int]]]] = []

    def scan_line(edge: str, line: np.ndarray, inward: np.ndarray | None):
        nonlocal candidates
        n = len(line)
        start = None
        for i in range(n + 1):
            is_non_bg = i < n and int(line[i]) != bg
            if is_non_bg and start is None:
                start = i
            if not is_non_bg and start is not None:
                end = i
                colors = [int(v) for v in line[start:end]]
                distinct = len(colors) >= 2 and len(set(colors)) == len(colors)
                inward_clear = inward is None or all(int(inward[j]) == bg for j in range(start, end))
                if distinct and inward_clear:
                    if edge == "top":
                        coords = [(0, c) for c in range(start, end)]
                    elif edge == "bottom":
                        coords = [(h - 1, c) for c in range(start, end)]
                    elif edge == "left":
                        coords = [(r, 0) for r in range(start, end)]
                    else:
                        coords = [(r, w - 1) for r in range(start, end)]
                    candidates.append((len(colors), edge, colors, coords))
                start = None

    scan_line("top", grid[0], grid[1] if h > 1 else None)
    scan_line("bottom", grid[h - 1], grid[h - 2] if h > 1 else None)
    scan_line("left", grid[:, 0], grid[:, 1] if w > 1 else None)
    scan_line("right", grid[:, w - 1], grid[:, w - 2] if w > 1 else None)

    assert candidates, "expected a distinct-color palette strip on one edge"
    best_len = max(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == best_len]
    best.sort(key=lambda item: PALETTE_EDGES.index(item[1]))
    _, edge, colors, coords = best[0]
    return edge, colors, coords


def _shape_holes(
    grid: np.ndarray, bg: int, outline_color: int
) -> list[dict[str, object]]:
    h, w = grid.shape
    comp_id = np.full((h, w), -1, dtype=int)
    outline_seen = np.zeros((h, w), dtype=bool)
    comp_index = 0
    for r in range(h):
        for c in range(w):
            if grid[r, c] != outline_color or outline_seen[r, c]:
                continue
            q = deque([(r, c)])
            outline_seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                comp_id[cr, cc] = comp_index
                for nr, nc in _neighbors8(cr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not outline_seen[nr, nc]
                        and grid[nr, nc] == outline_color
                    ):
                        outline_seen[nr, nc] = True
                        q.append((nr, nc))
            comp_index += 1

    seen_bg = np.zeros((h, w), dtype=bool)
    holes: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != bg or seen_bg[r, c]:
                continue
            q = deque([(r, c)])
            seen_bg[r, c] = True
            cells: list[tuple[int, int]] = []
            touches_edge = False
            adjacent_ids: set[int] = set()
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
                    touches_edge = True
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w:
                        if grid[nr, nc] == bg and not seen_bg[nr, nc]:
                            seen_bg[nr, nc] = True
                            q.append((nr, nc))
                        elif grid[nr, nc] == outline_color:
                            adjacent_ids.add(int(comp_id[nr, nc]))
                for nr, nc in _neighbors8(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == outline_color:
                        adjacent_ids.add(int(comp_id[nr, nc]))
            if not touches_edge and len(adjacent_ids) == 1:
                holes.append(
                    {
                        "cells": cells,
                        "area": len(cells),
                        "bbox": _bbox(cells),
                        "component": next(iter(adjacent_ids)),
                    }
                )
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow scaled outline with a palette color chosen by area rank, then erase the palette strip.

    Input structure:
    - The most common color is the background.
    - One grid edge contains a contiguous 1×K or K×1 palette strip of K distinct colors, with one color per pixel.
    - Elsewhere are exactly K hollow outlines in a single border color; they are the same shape under rotation/reflection and integer scaling, but have different sizes.
    - Each outline encloses one background-colored interior region.

    Rule:
    - Read the palette in order along the edge strip.
    - Ignore the palette while locating enclosed background regions inside the hollow outlines.
    - Sort those enclosed regions by area from smallest to largest, and fill them with the palette colors in strip order.
    - Keep the outline pixels unchanged and replace the palette strip with background.

    Key insight:
    - The edge strip is metadata: it provides an ordered list of fill colors, while the actual assignment comes from ranking the outlines by the size of their enclosed interiors.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    _edge, palette, palette_coords = _find_palette(g, bg)

    work = g.copy()
    for r, c in palette_coords:
        work[r, c] = bg

    remaining_colors = [int(c) for c in np.unique(work) if int(c) != bg]
    assert remaining_colors, "expected outline objects after removing palette"
    outline_color = max(remaining_colors, key=lambda c: int((work == c).sum()))

    holes = _shape_holes(work, bg, outline_color)
    assert len(holes) == len(palette), "palette length must match hollow-shape count"
    assert len({int(hole["component"]) for hole in holes}) == len(holes), "each outline must enclose exactly one hole"

    holes.sort(key=lambda hole: (int(hole["area"]), tuple(int(x) for x in hole["bbox"])))
    out = work.copy()
    for hole, color in zip(holes, palette):
        for r, c in hole["cells"]:
            out[r, c] = color
    return out


def _transform_mask(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.array(mask, copy=True)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return np.rot90(out, rng.randrange(4))


def _build_scaled_shape(base_mask: np.ndarray, scale: int) -> dict[str, object] | None:
    filled = np.kron(base_mask, np.ones((scale, scale), dtype=bool))
    outline = _boundary_from_filled(filled)
    holes = _find_local_holes(outline)
    if len(holes) != 1:
        return None
    hole_mask = np.zeros_like(outline, dtype=bool)
    for r, c in holes[0]:
        hole_mask[r, c] = True
    return {
        "outline": outline,
        "interior": hole_mask,
        "area": len(holes[0]),
        "h": int(outline.shape[0]),
        "w": int(outline.shape[1]),
        "scale": scale,
    }


def _reserved_margins(edge: str) -> tuple[int, int, int, int]:
    top = 2 if edge == "top" else 1
    bottom = 2 if edge == "bottom" else 1
    left = 2 if edge == "left" else 1
    right = 2 if edge == "right" else 1
    return top, bottom, left, right


def _slot_assignments(count: int, rows: int, cols: int):
    slots = list(range(rows * cols))
    if count == rows * cols:
        for perm in itertools.permutations(slots):
            yield {slot: idx for idx, slot in enumerate(perm)}
        return
    for used in itertools.combinations(slots, count):
        for perm in itertools.permutations(used):
            yield {slot: idx for idx, slot in enumerate(perm)}


def _plan_layout(
    shapes: list[dict[str, object]], edge: str, rng: random.Random
) -> dict[str, object] | None:
    count = len(shapes)
    layout_options = [(1, 2), (2, 1)] if count == 2 else [(2, 2)]
    candidates: list[dict[str, object]] = []
    for rows, cols in layout_options:
        for mapping in _slot_assignments(count, rows, cols):
            row_heights = [0] * rows
            col_widths = [0] * cols
            for slot, shape_idx in mapping.items():
                rr, cc = divmod(slot, cols)
                row_heights[rr] = max(row_heights[rr], int(shapes[shape_idx]["h"]))
                col_widths[cc] = max(col_widths[cc], int(shapes[shape_idx]["w"]))
            content_h = sum(row_heights) + SLOT_GAP * (rows - 1)
            content_w = sum(col_widths) + SLOT_GAP * (cols - 1)
            top, bottom, left, right = _reserved_margins(edge)
            min_h = top + content_h + bottom
            min_w = left + content_w + right
            if min_h <= 30 and min_w <= 30:
                candidates.append(
                    {
                        "rows": rows,
                        "cols": cols,
                        "mapping": mapping,
                        "row_heights": row_heights,
                        "col_widths": col_widths,
                        "content_h": content_h,
                        "content_w": content_w,
                        "min_h": min_h,
                        "min_w": min_w,
                    }
                )
    if not candidates:
        return None
    return rng.choice(candidates)


def _pick_scales(
    family: np.ndarray, edge: str, count: int, rng: random.Random
) -> list[int] | None:
    is_triangle = np.array_equal(family, BASE_SHAPES[-1])
    if is_triangle:
        if count == 4 and edge in ("left", "right"):
            return None
        pool = [2, 3, 4, 5]
    else:
        pool = [1, 2, 3, 4] if count == 4 and edge in ("left", "right") else [1, 2, 3, 4, 5]
    if len(pool) < count:
        return None
    return sorted(rng.sample(pool, count))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.randrange(10)
    edge = rng.choice(PALETTE_EDGES)
    count = rng.choice((2, 3, 4))
    family = _transform_mask(rng.choice(BASE_SHAPES), rng)
    scales = _pick_scales(family, edge, count, rng)
    if scales is None:
        return None

    shapes: list[dict[str, object]] = []
    for scale in scales:
        shape = _build_scaled_shape(family, scale)
        if shape is None:
            return None
        shapes.append(shape)

    if len({int(shape["area"]) for shape in shapes}) != count:
        return None

    layout = _plan_layout(shapes, edge, rng)
    if layout is None:
        return None

    h = rng.randint(int(layout["min_h"]), 30)
    w = rng.randint(int(layout["min_w"]), 30)
    grid = np.full((h, w), bg, dtype=int)

    colors = [c for c in range(10) if c != bg]
    rng.shuffle(colors)
    outline_color = colors[0]
    palette = colors[1 : 1 + count]
    if len(palette) != count:
        return None

    top, bottom, left, right = _reserved_margins(edge)
    content_h = int(layout["content_h"])
    content_w = int(layout["content_w"])
    content_r0 = rng.randint(top, h - bottom - content_h)
    content_c0 = rng.randint(left, w - right - content_w)

    row_starts = []
    cur = content_r0
    for row_h in layout["row_heights"]:
        row_starts.append(cur)
        cur += int(row_h) + SLOT_GAP
    col_starts = []
    cur = content_c0
    for col_w in layout["col_widths"]:
        col_starts.append(cur)
        cur += int(col_w) + SLOT_GAP

    interiors: list[dict[str, object]] = []
    for slot, shape_idx in layout["mapping"].items():
        rr, cc = divmod(slot, int(layout["cols"]))
        shape = shapes[shape_idx]
        slot_r0 = row_starts[rr]
        slot_c0 = col_starts[cc]
        slot_h = int(layout["row_heights"][rr])
        slot_w = int(layout["col_widths"][cc])
        shape_h = int(shape["h"])
        shape_w = int(shape["w"])
        jitter_r = rng.randint(0, slot_h - shape_h)
        jitter_c = rng.randint(0, slot_w - shape_w)
        r0 = slot_r0 + jitter_r
        c0 = slot_c0 + jitter_c

        outline_mask = shape["outline"]
        interior_mask = shape["interior"]
        for lr, lc in np.argwhere(outline_mask):
            grid[r0 + int(lr), c0 + int(lc)] = outline_color

        abs_cells = [(r0 + int(lr), c0 + int(lc)) for lr, lc in np.argwhere(interior_mask)]
        interiors.append({"cells": abs_cells, "area": int(shape["area"])})

    if edge == "top":
        start = rng.randint(0, w - count)
        palette_coords = [(0, start + i) for i in range(count)]
    elif edge == "bottom":
        start = rng.randint(0, w - count)
        palette_coords = [(h - 1, start + i) for i in range(count)]
    elif edge == "left":
        start = rng.randint(0, h - count)
        palette_coords = [(start + i, 0) for i in range(count)]
    else:
        start = rng.randint(0, h - count)
        palette_coords = [(start + i, w - 1) for i in range(count)]

    for (r, c), color in zip(palette_coords, palette):
        grid[r, c] = color

    expected = np.full_like(grid, bg)
    expected[grid == outline_color] = outline_color
    for info, color in zip(sorted(interiors, key=lambda item: int(item["area"])), palette):
        for r, c in info["cells"]:
            expected[r, c] = color

    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
