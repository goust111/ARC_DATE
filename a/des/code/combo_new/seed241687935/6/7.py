from __future__ import annotations

import random

import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLORS = tuple(range(1, 10))

GEN_RETRIES = 400
SHAPE_RETRIES = 200
ARRANGE_RETRIES = 100
OBJECT_COUNT_RANGE = (2, 5)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or visited[r0, c0]:
                continue
            stack = [(r0, c0)]
            visited[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    return len(_mask_components(mask)) == 1


def _has_hole(mask: np.ndarray) -> bool:
    bg_mask = ~mask
    for comp in _mask_components(bg_mask):
        if not any(
            r == 0 or c == 0 or r == mask.shape[0] - 1 or c == mask.shape[1] - 1
            for r, c in comp
        ):
            return True
    return False


def _boundary_pixel_count_mask(mask: np.ndarray) -> int:
    h, w = mask.shape
    total = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                    total += 1
                    break
    return total


def _make_bar() -> np.ndarray:
    length = random.randint(3, 8)
    if random.random() < 0.5:
        return np.ones((1, length), dtype=bool)
    return np.ones((length, 1), dtype=bool)


def _make_rectangle() -> np.ndarray:
    h = random.randint(2, 6)
    w = random.randint(2, 6)
    return np.ones((h, w), dtype=bool)


def _make_corner_cut() -> np.ndarray:
    h = random.randint(3, 6)
    w = random.randint(3, 6)
    mask = np.ones((h, w), dtype=bool)
    cut_h = random.randint(1, h - 1)
    cut_w = random.randint(1, w - 1)
    corner = random.choice(("tl", "tr", "bl", "br"))
    if corner == "tl":
        mask[:cut_h, :cut_w] = False
    elif corner == "tr":
        mask[:cut_h, w - cut_w :] = False
    elif corner == "bl":
        mask[h - cut_h :, :cut_w] = False
    else:
        mask[h - cut_h :, w - cut_w :] = False
    return _crop_mask(mask)


def _make_edge_notch() -> np.ndarray:
    h = random.randint(4, 6)
    w = random.randint(4, 6)
    mask = np.ones((h, w), dtype=bool)
    side = random.choice(("top", "bottom", "left", "right"))
    if side in ("top", "bottom"):
        notch_w = random.randint(1, w - 2)
        start = random.randint(1, w - notch_w - 1)
        depth = random.randint(1, h - 1)
        rows = slice(0, depth) if side == "top" else slice(h - depth, h)
        mask[rows, start : start + notch_w] = False
    else:
        notch_h = random.randint(1, h - 2)
        start = random.randint(1, h - notch_h - 1)
        depth = random.randint(1, w - 1)
        cols = slice(0, depth) if side == "left" else slice(w - depth, w)
        mask[start : start + notch_h, cols] = False
    return _crop_mask(mask)


def _make_overlap() -> np.ndarray:
    canvas_h = random.randint(3, 6)
    canvas_w = random.randint(3, 6)
    for _ in range(40):
        h1 = random.randint(1, canvas_h)
        w1 = random.randint(1, canvas_w)
        r1 = random.randint(0, canvas_h - h1)
        c1 = random.randint(0, canvas_w - w1)

        h2 = random.randint(1, canvas_h)
        w2 = random.randint(1, canvas_w)
        r2 = random.randint(0, canvas_h - h2)
        c2 = random.randint(0, canvas_w - w2)

        if r1 + h1 <= r2 or r2 + h2 <= r1 or c1 + w1 <= c2 or c2 + w2 <= c1:
            continue

        mask = np.zeros((canvas_h, canvas_w), dtype=bool)
        mask[r1 : r1 + h1, c1 : c1 + w1] = True
        mask[r2 : r2 + h2, c2 : c2 + w2] = True
        return _crop_mask(mask)
    return np.ones((3, 3), dtype=bool)


def _random_shape(force_interior: bool) -> np.ndarray:
    families = [
        ("rectangle", _make_rectangle, 4),
        ("bar", _make_bar, 2),
        ("corner_cut", _make_corner_cut, 3),
        ("edge_notch", _make_edge_notch, 3),
        ("overlap", _make_overlap, 4),
    ]
    if force_interior:
        families = [item for item in families if item[0] != "bar"]

    names = [item[0] for item in families]
    weights = [item[2] for item in families]
    makers = {item[0]: item[1] for item in families}

    for _ in range(SHAPE_RETRIES):
        kind = random.choices(names, weights=weights, k=1)[0]
        mask = _crop_mask(np.asarray(makers[kind](), dtype=bool))
        area = int(mask.sum())
        boundary = _boundary_pixel_count_mask(mask)
        density = area / float(mask.size)

        if area < 3 or boundary > 20:
            continue
        if not _is_connected(mask) or _has_hole(mask):
            continue
        if kind != "bar" and density < 0.45:
            continue
        if force_interior and area <= boundary:
            continue
        return mask

    raise RuntimeError("failed to sample a valid shape")


def _shape_record(mask: np.ndarray, color: int) -> dict[str, int | np.ndarray]:
    return {
        "mask": mask,
        "color": int(color),
        "h": int(mask.shape[0]),
        "w": int(mask.shape[1]),
        "area": int(mask.sum()),
        "boundary": int(_boundary_pixel_count_mask(mask)),
    }


def _sample_objects() -> list[dict[str, int | np.ndarray]]:
    for _ in range(SHAPE_RETRIES):
        count = random.randint(*OBJECT_COUNT_RANGE)
        colors = random.sample(COLORS, count)
        interior_idx = random.randrange(count)
        objects = []
        for i, color in enumerate(colors):
            mask = _random_shape(force_interior=i == interior_idx)
            objects.append(_shape_record(mask, color))

        boundaries = [int(obj["boundary"]) for obj in objects]
        if len(set(boundaries)) < 2:
            continue
        if not any(int(obj["area"]) > int(obj["boundary"]) for obj in objects):
            continue
        return objects
    raise RuntimeError("failed to sample object set")


def _same_color_components(grid: np.ndarray) -> list[dict[str, int | list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, int | list[tuple[int, int]]]] = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(g[r0, c0])
            if color == BG or visited[r0, c0]:
                continue
            stack = [(r0, c0)]
            visited[r0, c0] = True
            pts: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                pts.append((r, c))
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            rows = [r for r, _ in pts]
            cols = [c for _, c in pts]
            comps.append(
                {
                    "color": color,
                    "points": pts,
                    "top": min(rows),
                    "left": min(cols),
                    "area": len(pts),
                }
            )
    return comps


def _object_records(grid: np.ndarray) -> list[dict[str, int | list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    records = _same_color_components(g)
    for rec in records:
        boundary = 0
        for r, c in rec["points"]:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or int(g[nr, nc]) == BG:
                    boundary += 1
                    break
        rec["boundary"] = boundary
    records.sort(key=lambda rec: (int(rec["top"]), int(rec["left"])))
    return records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn each filled foreground object into a left-aligned bar whose length equals its boundary-pixel count.

    Input structure:
    - Black (0) is the background color.
    - The grid contains several non-touching monochrome 4-connected foreground objects, each using a different color.
    - Each object is a compact filled shape with no holes, and its number of boundary pixels is at most 20.

    Rule:
    - For each foreground object, count P = the number of its pixels that have at least one 4-neighbor in the background.
    - Sort the objects from top to bottom, then left to right by comparing their bounding-box top row and then left column.
    - Create an output grid with one row per object and width equal to the largest P among them.
    - In row i, draw a contiguous run of P pixels in that object's color starting at the left edge; fill the remaining cells with black.

    Key insight:
    - The output length is not the object's area or bbox size: it is the count of boundary pixels exposed to the background.
    """
    g = np.asarray(grid, dtype=int)
    records = _object_records(g)
    if not records:
        return np.zeros((0, 0), dtype=int)

    width = max(int(rec["boundary"]) for rec in records)
    out = np.zeros((len(records), width), dtype=int)
    for r, rec in enumerate(records):
        out[r, : int(rec["boundary"])] = int(rec["color"])
    return out


def _arrange_objects(objects: list[dict[str, int | np.ndarray]]) -> np.ndarray:
    count = len(objects)
    for _ in range(ARRANGE_RETRIES):
        band_count = random.randint(1, min(3, count))
        order = list(range(count))
        random.shuffle(order)

        bands: list[list[int]] = [[] for _ in range(band_count)]
        for i in range(band_count):
            bands[i].append(order[i])
        for idx in order[band_count:]:
            bands[random.randrange(band_count)].append(idx)
        for band in bands:
            random.shuffle(band)

        band_modes = []
        band_heights = []
        band_widths = []
        band_gaps: list[list[int]] = []
        for band in bands:
            max_h = max(int(objects[idx]["h"]) for idx in band)
            mode = random.choice(("aligned", "jittered"))
            extra = 1 if mode == "jittered" and max_h < 6 and random.random() < 0.8 else 0
            gaps = [random.randint(2, 4) for _ in range(len(band) - 1)]
            width = sum(int(objects[idx]["w"]) for idx in band) + sum(gaps)
            band_modes.append(mode)
            band_heights.append(max_h + extra)
            band_widths.append(width)
            band_gaps.append(gaps)

        row_gaps = [random.randint(2, 4) for _ in range(band_count - 1)]
        core_h = sum(band_heights) + sum(row_gaps)
        core_w = max(band_widths)
        if core_h + 2 > 30 or core_w + 2 > 30:
            continue

        grid_h = random.randint(core_h + 2, min(30, core_h + 6))
        grid_w = random.randint(core_w + 2, min(30, core_w + 8))
        top_pad = random.randint(1, grid_h - core_h - 1)

        grid = np.zeros((grid_h, grid_w), dtype=int)
        current_row = top_pad
        for band_idx, band in enumerate(bands):
            band_left = random.randint(1, grid_w - band_widths[band_idx] - 1)
            x = band_left
            for obj_pos, obj_idx in enumerate(band):
                obj = objects[obj_idx]
                mask = obj["mask"]
                y = current_row
                if band_modes[band_idx] == "jittered":
                    y += random.randint(0, band_heights[band_idx] - int(obj["h"]))
                sub = grid[y : y + int(obj["h"]), x : x + int(obj["w"])]
                sub[np.asarray(mask, dtype=bool)] = int(obj["color"])
                if obj_pos < len(band_gaps[band_idx]):
                    x += int(obj["w"]) + band_gaps[band_idx][obj_pos]
            current_row += band_heights[band_idx]
            if band_idx < len(row_gaps):
                current_row += row_gaps[band_idx]

        return grid

    raise RuntimeError("failed to arrange objects")


def _objects_are_isolated(grid: np.ndarray) -> bool:
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) == BG:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        v = int(grid[nr, nc])
                        if v != BG and v != int(grid[r, c]):
                            return False
    return True


def _valid_generated_grid(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    records = _object_records(grid)
    if not records or len(records) != len(set(int(rec["color"]) for rec in records)):
        return False
    if not _objects_are_isolated(grid):
        return False
    if any(int(rec["boundary"]) > 20 for rec in records):
        return False
    if len({int(rec["boundary"]) for rec in records}) < 2:
        return False
    if not any(int(rec["area"]) > int(rec["boundary"]) for rec in records):
        return False

    out = main(grid)
    expected = np.zeros_like(out)
    for r, rec in enumerate(records):
        expected[r, : int(rec["boundary"])] = int(rec["color"])
    return np.array_equal(out, expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            objects = _sample_objects()
            grid = _arrange_objects(objects)
        except RuntimeError:
            continue
        if _valid_generated_grid(grid):
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
