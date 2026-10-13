from __future__ import annotations

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = tuple(range(10))
GEN_ATTEMPTS = 400
MIN_OBJECTS = 2
MAX_OBJECTS = 4


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue

        stack = [(r, c)]
        seen[r, c] = True
        pts: list[tuple[int, int]] = []

        while stack:
            rr, cc = stack.pop()
            pts.append((rr, cc))
            for dr, dc in DIR4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))

        comps.append(np.array(pts, dtype=int))

    return comps


def _run_lengths(values: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for value in values:
        value = int(value)
        if not runs or runs[-1][0] != value:
            runs.append((value, 1))
        else:
            prev_value, prev_len = runs[-1]
            runs[-1] = (prev_value, prev_len + 1)
    return runs


def _analyze_row_split(subgrid: np.ndarray, mask: np.ndarray) -> dict[str, int] | None:
    row_template = mask[0]
    if not np.all(mask == row_template[None, :]):
        return None
    if not row_template.all():
        return None

    row_colors: list[int] = []
    for r in range(subgrid.shape[0]):
        colors = np.unique(subgrid[r])
        if len(colors) != 1:
            return None
        row_colors.append(int(colors[0]))

    runs = _run_lengths(row_colors)
    if len(runs) != 2:
        return None

    (color_a, size_a), (color_b, size_b) = runs
    if size_a == size_b:
        return None

    if size_a < size_b:
        return {"k": int(size_a), "narrow": int(color_a), "wide": int(color_b)}
    return {"k": int(size_b), "narrow": int(color_b), "wide": int(color_a)}


def _analyze_col_split(subgrid: np.ndarray, mask: np.ndarray) -> dict[str, int] | None:
    col_template = mask[:, 0]
    if not np.all(mask == col_template[:, None]):
        return None
    if not col_template.all():
        return None

    col_colors: list[int] = []
    for c in range(subgrid.shape[1]):
        colors = np.unique(subgrid[:, c])
        if len(colors) != 1:
            return None
        col_colors.append(int(colors[0]))

    runs = _run_lengths(col_colors)
    if len(runs) != 2:
        return None

    (color_a, size_a), (color_b, size_b) = runs
    if size_a == size_b:
        return None

    if size_a < size_b:
        return {"k": int(size_a), "narrow": int(color_a), "wide": int(color_b)}
    return {"k": int(size_b), "narrow": int(color_b), "wide": int(color_a)}


def _analyze_component(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict[str, int | str | tuple[int, int, int, int]]:
    colors = np.unique(grid[coords[:, 0], coords[:, 1]])
    assert bg not in colors, "components must be non-background"
    assert len(colors) == 2, "each object must use exactly two colors"

    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    subgrid = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)

    mask = np.zeros(subgrid.shape, dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True

    row_case = _analyze_row_split(subgrid, mask)
    col_case = _analyze_col_split(subgrid, mask)
    assert (row_case is None) != (col_case is None), "split direction must be unique"

    if row_case is not None:
        return {
            "expand_axis": "vertical",
            "k": int(row_case["k"]),
            "narrow": int(row_case["narrow"]),
            "wide": int(row_case["wide"]),
            "bbox": (r0, c0, r1, c1),
        }

    assert col_case is not None
    return {
        "expand_axis": "horizontal",
        "k": int(col_case["k"]),
        "narrow": int(col_case["narrow"]),
        "wide": int(col_case["wide"]),
        "bbox": (r0, c0, r1, c1),
    }


def _paint_expansion(out: np.ndarray, info: dict[str, int | str | tuple[int, int, int, int]]) -> None:
    r0, c0, r1, c1 = info["bbox"]  # type: ignore[misc]
    k = int(info["k"])
    narrow = int(info["narrow"])
    wide = int(info["wide"])

    if info["expand_axis"] == "horizontal":
        stripe_idx = 0
        left = c0 - k
        while left >= 0:
            color = narrow if stripe_idx % 2 == 0 else wide
            out[r0 : r1 + 1, left : left + k] = color
            stripe_idx += 1
            left -= k

        stripe_idx = 0
        right = c1 + 1
        while right + k <= out.shape[1]:
            color = narrow if stripe_idx % 2 == 0 else wide
            out[r0 : r1 + 1, right : right + k] = color
            stripe_idx += 1
            right += k
        return

    stripe_idx = 0
    top = r0 - k
    while top >= 0:
        color = narrow if stripe_idx % 2 == 0 else wide
        out[top : top + k, c0 : c1 + 1] = color
        stripe_idx += 1
        top -= k

    stripe_idx = 0
    bottom = r1 + 1
    while bottom + k <= out.shape[0]:
        color = narrow if stripe_idx % 2 == 0 else wide
        out[bottom : bottom + k, c0 : c1 + 1] = color
        stripe_idx += 1
        bottom += k


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each split-color object into alternating stripe bands along the axis perpendicular to its internal partition.

    Input structure:
    - The background is the dominant color.
    - Several disconnected foreground objects are scattered across the grid.
    - Each object uses exactly two non-background colors and is split into two rectangular color regions by one straight horizontal or vertical boundary.
    - Objects are spaced so their stripe expansions do not overlap.

    Rule:
    - Inspect each foreground object independently and detect whether its two colors are divided top/bottom or left/right.
    - Measure the thinner side of that division; its row-count or column-count is k.
    - Keep the original object unchanged, then extend stripes from both outward ends along the perpendicular axis.
    - Each added stripe has width k and alternates narrow-side color, wide-side color, narrow-side color, ... on both outward sides until fewer than k rows or columns remain.

    Key insight:
    - The partition orientation chooses the expansion axis, while the thinner partition thickness determines the repeating stripe width and starting color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    comps = _components(g != bg)
    assert comps, "expected at least one foreground object"

    out = g.copy()
    for comp in comps:
        info = _analyze_component(g, comp, bg)
        _paint_expansion(out, info)
    return out


def _rand_composition(total: int, parts: int, rng: np.random.Generator) -> np.ndarray:
    if parts == 1:
        return np.array([total], dtype=int)
    if total == 0:
        return np.zeros(parts, dtype=int)
    return rng.multinomial(total, np.full(parts, 1.0 / parts))


def _pack_intervals(total: int, sizes: list[int], rng: np.random.Generator) -> list[int]:
    assert sizes
    slack = total - sum(sizes) - (len(sizes) + 1)
    assert slack >= 0
    extras = _rand_composition(int(slack), len(sizes) + 1, rng)

    starts: list[int] = []
    cursor = 1 + int(extras[0])
    for idx, size in enumerate(sizes):
        starts.append(cursor)
        cursor += int(size) + 1 + int(extras[idx + 1])
    return starts


def _choose_two_colors(bg: int, rng: np.random.Generator) -> tuple[int, int]:
    colors = [color for color in PALETTE if color != bg]
    chosen = rng.choice(colors, size=2, replace=False)
    return int(chosen[0]), int(chosen[1])


def _build_vertical_split_object(bg: int, rng: np.random.Generator) -> dict[str, int]:
    height = int(rng.integers(2, 6))
    k = int(rng.integers(1, 4))
    wide_max = min(6, 9 - k)
    wide = int(rng.integers(k + 1, wide_max + 1))
    narrow_on_left = bool(rng.integers(0, 2))
    color_a, color_b = _choose_two_colors(bg, rng)

    if narrow_on_left:
        left_width, right_width = k, wide
        left_color, right_color = color_a, color_b
    else:
        left_width, right_width = wide, k
        left_color, right_color = color_b, color_a

    return {
        "height": height,
        "width": left_width + right_width,
        "k": k,
        "left_width": left_width,
        "right_width": right_width,
        "left_color": left_color,
        "right_color": right_color,
        "mode": 0,
    }


def _build_horizontal_split_object(bg: int, rng: np.random.Generator) -> dict[str, int]:
    width = int(rng.integers(2, 6))
    k = int(rng.integers(1, 4))
    wide_max = min(6, 9 - k)
    wide = int(rng.integers(k + 1, wide_max + 1))
    narrow_on_top = bool(rng.integers(0, 2))
    color_a, color_b = _choose_two_colors(bg, rng)

    if narrow_on_top:
        top_height, bottom_height = k, wide
        top_color, bottom_color = color_a, color_b
    else:
        top_height, bottom_height = wide, k
        top_color, bottom_color = color_b, color_a

    return {
        "height": top_height + bottom_height,
        "width": width,
        "k": k,
        "top_height": top_height,
        "bottom_height": bottom_height,
        "top_color": top_color,
        "bottom_color": bottom_color,
        "mode": 1,
    }


def _footprint_cells(obj: dict[str, int], top: int, left: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    cells: set[tuple[int, int]] = set()
    obj_h = int(obj["height"])
    obj_w = int(obj["width"])
    k = int(obj["k"])

    if obj["mode"] == 0:
        r0, c0, r1, c1 = top, left, top + obj_h - 1, left + obj_w - 1
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                cells.add((rr, cc))
        stripe = 0
        cc = c0 - k
        while cc >= 0:
            for rr in range(r0, r1 + 1):
                for col in range(cc, cc + k):
                    cells.add((rr, col))
            stripe += 1
            cc -= k
        cc = c1 + 1
        while cc + k <= w:
            for rr in range(r0, r1 + 1):
                for col in range(cc, cc + k):
                    cells.add((rr, col))
            stripe += 1
            cc += k
        return cells

    r0, c0, r1, c1 = top, left, top + obj_h - 1, left + obj_w - 1
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            cells.add((rr, cc))
    rr = r0 - k
    while rr >= 0:
        for row in range(rr, rr + k):
            for cc in range(c0, c1 + 1):
                cells.add((row, cc))
        rr -= k
    rr = r1 + 1
    while rr + k <= h:
        for row in range(rr, rr + k):
            for cc in range(c0, c1 + 1):
                cells.add((row, cc))
        rr += k
    return cells


def _paint_input_object(grid: np.ndarray, obj: dict[str, int], top: int, left: int) -> None:
    if obj["mode"] == 0:
        split = left + int(obj["left_width"])
        grid[top : top + int(obj["height"]), left:split] = int(obj["left_color"])
        grid[top : top + int(obj["height"]), split : left + int(obj["width"])] = int(obj["right_color"])
        return

    split = top + int(obj["top_height"])
    grid[top:split, left : left + int(obj["width"])] = int(obj["top_color"])
    grid[split : top + int(obj["height"]), left : left + int(obj["width"])] = int(obj["bottom_color"])


def _valid_generated_grid(grid: np.ndarray) -> bool:
    try:
        bg = _get_bg(grid)
        comps = _components(grid != bg)
        if not (MIN_OBJECTS <= len(comps) <= MAX_OBJECTS):
            return False
        occupied: set[tuple[int, int]] = set()
        for comp in comps:
            info = _analyze_component(grid, comp, bg)
            cells = _footprint_cells(
                {
                    "height": int(info["bbox"][2] - info["bbox"][0] + 1),  # type: ignore[index]
                    "width": int(info["bbox"][3] - info["bbox"][1] + 1),   # type: ignore[index]
                    "k": int(info["k"]),
                    "mode": 0 if info["expand_axis"] == "horizontal" else 1,
                },
                int(info["bbox"][0]),  # type: ignore[index]
                int(info["bbox"][1]),  # type: ignore[index]
                grid.shape,
            )
            if occupied & cells:
                return False
            occupied |= cells
        out = main(grid)
        return np.count_nonzero(out != bg) > np.count_nonzero(grid != bg)
    except AssertionError:
        return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    bg = int(rng.integers(0, 10))
    mode = int(rng.integers(0, 2))  # 0: vertical split -> horizontal expansion, 1: horizontal split -> vertical expansion
    object_count = int(rng.integers(MIN_OBJECTS, MAX_OBJECTS + 1))

    if mode == 0:
        objects = [_build_vertical_split_object(bg, rng) for _ in range(object_count)]
        heights = [int(obj["height"]) for obj in objects]
        min_h = sum(heights) + object_count + 1
        min_w = max(int(obj["width"]) + 2 * int(obj["k"]) for obj in objects)
        if min_h > 30 or min_w > 30:
            return None

        h = int(rng.integers(max(10, min_h), 31))
        w = int(rng.integers(max(12, min_w), 31))
        row_starts = _pack_intervals(h, heights, rng)

        placements: list[tuple[dict[str, int], int, int]] = []
        for obj, top in zip(objects, row_starts):
            left_min = int(obj["k"])
            left_max = w - int(obj["width"]) - int(obj["k"])
            if left_min > left_max:
                return None
            left = int(rng.integers(left_min, left_max + 1))
            placements.append((obj, int(top), left))
    else:
        objects = [_build_horizontal_split_object(bg, rng) for _ in range(object_count)]
        widths = [int(obj["width"]) for obj in objects]
        min_w = sum(widths) + object_count + 1
        min_h = max(int(obj["height"]) + 2 * int(obj["k"]) for obj in objects)
        if min_h > 30 or min_w > 30:
            return None

        h = int(rng.integers(max(12, min_h), 31))
        w = int(rng.integers(max(10, min_w), 31))
        col_starts = _pack_intervals(w, widths, rng)

        placements = []
        for obj, left in zip(objects, col_starts):
            top_min = int(obj["k"])
            top_max = h - int(obj["height"]) - int(obj["k"])
            if top_min > top_max:
                return None
            top = int(rng.integers(top_min, top_max + 1))
            placements.append((obj, top, int(left)))

    grid = np.full((h, w), bg, dtype=int)
    seen: set[tuple[int, int]] = set()
    for obj, top, left in placements:
        footprint = _footprint_cells(obj, top, left, grid.shape)
        if seen & footprint:
            return None
        seen |= footprint
        _paint_input_object(grid, obj, top, left)

    return grid if _valid_generated_grid(grid) else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stripe-expansion ARC input")
