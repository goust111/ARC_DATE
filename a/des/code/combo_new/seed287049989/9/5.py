from __future__ import annotations

from collections import Counter

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
BG_COLOR = 0
SHADOW_COLOR = 5
MAX_ATTEMPTS = 400

SHAPE_TEMPLATES = [
    ("###", "###", "##."),
    (".##", "###", "##."),
    ("####", "####", "###.", "##.."),
    (".###", "####", "###.", "##.."),
    ("####", "###.", "####", "..##"),
    ("#####", "#####", "####.", "###..", "..##."),
    (".####", "#####", "####.", "###..", ".##.."),
    ("####.", "#####", ".####", ".###.", "..##."),
]

VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each colored object into a translated monochrome outline whose direction comes from its adjacent grey strip.

    Input structure:
    - The background is black.
    - Several differently colored solid objects are scattered in the grid, each occupying a small irregular 3x3 to 5x5 footprint.
    - Every object has exactly one adjacent grey 1-cell-wide strip of length 1 to 3 on one of its four outer sides.

    Rule:
    - For each grey strip, identify the uniquely adjacent colored object and use the strip's side as the projection direction.
    - Translate that object by its own span in the projection axis: width for left/right projections and height for up/down projections.
    - Draw only the translated object's outline in the object's color; strip length 1 gives a 1-pixel outline, while lengths 2 and 3 thicken the outline by expanding it outward.
    - Keep the original object and its grey strip unchanged while adding the new outline.

    Key insight:
    - The grey strip encodes two independent parameters at once: its side gives the projection direction, but its length controls outline thickness, while the travel distance comes from the object's own size rather than the strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    bg = _background_color(g)
    shadow = _find_shadow_color(g, bg)
    object_by_color = _object_components(g, bg, shadow)
    shadow_components = _components_mask(g == shadow, connectivity=4)
    assert shadow_components, "expected at least one shadow strip"
    assert len(object_by_color) == len(shadow_components), "expected one strip per object"

    out = g.copy()
    paired_colors = set()
    for comp in shadow_components:
        obj_color, direction = _shadow_target(g, comp, bg, shadow)
        assert obj_color in object_by_color, "shadow strip must point at an object"
        assert obj_color not in paired_colors, "each object should have only one strip"
        paired_colors.add(obj_color)

        obj_cells = object_by_color[obj_color]
        outline = _projected_outline(obj_cells, direction, len(comp))
        for r, c in outline:
            if 0 <= r < h and 0 <= c < w and out[r, c] == bg:
                out[r, c] = obj_color

    assert paired_colors == set(object_by_color), "every object must be paired with a strip"
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    object_count = int(rng.choice([3, 4]))
    directions = ["up", "down", "left", "right"]
    rng.shuffle(directions)
    directions = directions[:object_count]

    thicknesses = [1, 2, 3]
    if object_count == 4:
        thicknesses.append(int(rng.integers(1, 4)))
    rng.shuffle(thicknesses)

    colors = list(rng.choice(np.array(PALETTE, dtype=int), size=object_count, replace=False))

    specs = []
    for direction, thickness, color in zip(directions, thicknesses, colors):
        spec = _random_object_spec(rng, direction, int(thickness), int(color))
        if spec is None:
            return None
        specs.append(spec)

    min_size = 22 if object_count == 3 else 25
    h = int(rng.integers(min_size, 31))
    w = int(rng.integers(min_size, 31))

    specs.sort(key=lambda spec: (-_cell_span_area(spec["footprint"]), int(spec["color"])))
    grid = np.full((h, w), BG_COLOR, dtype=int)
    expected = np.full((h, w), BG_COLOR, dtype=int)
    reserved = set()

    for spec in specs:
        anchor = _place_spec(spec, grid.shape, reserved, rng)
        if anchor is None:
            return None

        for r, c in _shift_cells(spec["object_cells"], anchor):
            grid[r, c] = int(spec["color"])
            expected[r, c] = int(spec["color"])
        for r, c in _shift_cells(spec["shadow_cells"], anchor):
            grid[r, c] = SHADOW_COLOR
            expected[r, c] = SHADOW_COLOR
        for r, c in _shift_cells(spec["projection_cells"], anchor):
            if expected[r, c] == BG_COLOR:
                expected[r, c] = int(spec["color"])

        reserved |= _halo(_shift_cells(spec["footprint"], anchor), radius=1)

    try:
        solved = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(solved, expected):
        return None
    return grid


def _random_object_spec(
    rng: np.random.Generator, direction: str, thickness: int, color: int
) -> dict[str, object] | None:
    template_order = list(range(len(SHAPE_TEMPLATES)))
    rng.shuffle(template_order)

    for idx in template_order:
        base = _mask_from_template(SHAPE_TEMPLATES[idx])
        transforms = [(rot, flip) for rot in range(4) for flip in (False, True)]
        rng.shuffle(transforms)
        for rot, flip in transforms:
            mask = _transform_mask(base, rot, flip)
            candidates = _attachment_candidates(mask, direction, thickness)
            if not candidates:
                continue
            rng.shuffle(candidates)
            for candidate in candidates:
                candidate["color"] = color
                return candidate
    return None


def _attachment_candidates(mask: np.ndarray, direction: str, thickness: int) -> list[dict[str, object]]:
    object_cells = _mask_to_cells(mask)
    shadow_lists = _shadow_candidates_from_profiles(mask, direction, thickness)

    candidates: list[dict[str, object]] = []
    for shadow_cells in shadow_lists:
        shadow_set = set(shadow_cells)
        if _local_shadow_direction(object_cells, shadow_set) != direction:
            continue
        projection_cells = _projected_outline(object_cells, direction, thickness)
        visible_projection = projection_cells - shadow_set - object_cells
        footprint = set(object_cells) | shadow_set | projection_cells
        if len(visible_projection) < 6:
            continue

        candidates.append(
            {
                "direction": direction,
                "thickness": thickness,
                "object_cells": set(object_cells),
                "shadow_cells": shadow_set,
                "projection_cells": projection_cells,
                "footprint": footprint,
                "min_r": min(r for r, _ in footprint),
                "max_r": max(r for r, _ in footprint),
                "min_c": min(c for _, c in footprint),
                "max_c": max(c for _, c in footprint),
            }
        )
    return candidates


def _place_spec(
    spec: dict[str, object],
    shape: tuple[int, int],
    reserved: set[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    h, w = shape
    row_min = -int(spec["min_r"])
    row_max = h - 1 - int(spec["max_r"])
    col_min = -int(spec["min_c"])
    col_max = w - 1 - int(spec["max_c"])
    if row_min > row_max or col_min > col_max:
        return None

    preferred_rows = _preferred_axis_values(row_min, row_max, h, str(spec["direction"]), axis="row")
    preferred_cols = _preferred_axis_values(col_min, col_max, w, str(spec["direction"]), axis="col")
    anchors = [(r, c) for r in preferred_rows for c in preferred_cols]

    fallback_rows = list(range(row_min, row_max + 1))
    fallback_cols = list(range(col_min, col_max + 1))
    fallback = [(r, c) for r in fallback_rows for c in fallback_cols if (r, c) not in anchors]

    rng.shuffle(anchors)
    rng.shuffle(fallback)

    for anchor in anchors + fallback:
        footprint = _shift_cells(spec["footprint"], anchor)
        if any(cell in reserved for cell in footprint):
            continue
        return anchor
    return None


def _preferred_axis_values(
    lo: int, hi: int, size: int, direction: str, axis: str
) -> list[int]:
    values = list(range(lo, hi + 1))
    if not values:
        return []

    if axis == "col" and direction == "right":
        cutoff = max(lo, size // 3)
        subset = [v for v in values if v <= cutoff]
    elif axis == "col" and direction == "left":
        cutoff = min(hi, (2 * size) // 3)
        subset = [v for v in values if v >= cutoff]
    elif axis == "row" and direction == "down":
        cutoff = max(lo, size // 3)
        subset = [v for v in values if v <= cutoff]
    elif axis == "row" and direction == "up":
        cutoff = min(hi, (2 * size) // 3)
        subset = [v for v in values if v >= cutoff]
    else:
        subset = values
    return subset if subset else values


def _background_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.flat)
    return int(counts.most_common(1)[0][0])


def _find_shadow_color(grid: np.ndarray, bg: int) -> int:
    candidates = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        comps = _components_mask(grid == color, connectivity=4)
        if comps and all(_is_short_line(comp) for comp in comps):
            total = sum(len(comp) for comp in comps)
            candidates.append((len(comps), -total, color))
    assert candidates, "expected a shadow-strip color"
    candidates.sort(reverse=True)
    return int(candidates[0][2])


def _object_components(grid: np.ndarray, bg: int, shadow: int) -> dict[int, set[tuple[int, int]]]:
    objects: dict[int, set[tuple[int, int]]] = {}
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (bg, shadow)):
        comps = _components_mask(grid == color, connectivity=4)
        assert len(comps) == 1, "each object color must form one component"
        objects[color] = set(map(tuple, comps[0]))
    return objects


def _shadow_target(
    grid: np.ndarray,
    comp: np.ndarray,
    bg: int,
    shadow: int,
) -> tuple[int, str]:
    neighbors: dict[int, set[tuple[int, int]]] = {}
    h, w = grid.shape
    for r, c in comp:
        for dr, dc in DIR4:
            rr = int(r + dr)
            cc = int(c + dc)
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            color = int(grid[rr, cc])
            if color in (bg, shadow):
                continue
            neighbors.setdefault(color, set()).add((dr, dc))

    assert len(neighbors) == 1, "each strip must touch exactly one object color"
    obj_color = next(iter(neighbors))
    adj_dirs = neighbors[obj_color]
    assert len(adj_dirs) == 1, "strip-object adjacency must indicate a unique side"
    dr, dc = next(iter(adj_dirs))
    proj = (-dr, -dc)
    direction = next(name for name, vec in VECTORS.items() if vec == proj)
    return obj_color, direction


def _projected_outline(
    object_cells: set[tuple[int, int]], direction: str, thickness: int
) -> set[tuple[int, int]]:
    dr, dc = VECTORS[direction]
    rows = [r for r, _ in object_cells]
    cols = [c for _, c in object_cells]
    distance = (max(rows) - min(rows) + 1) if dr else (max(cols) - min(cols) + 1)
    shifted = {(r + dr * distance, c + dc * distance) for r, c in object_cells}
    return _thick_outline(shifted, thickness)


def _thick_outline(cells: set[tuple[int, int]], thickness: int) -> set[tuple[int, int]]:
    core = _erode4(cells)
    expanded = _dilate_square(cells, thickness - 1)
    return expanded - core


def _erode4(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    result = set()
    for r, c in cells:
        if all((r + dr, c + dc) in cells for dr, dc in DIR4):
            result.add((r, c))
    return result


def _dilate_square(cells: set[tuple[int, int]], radius: int) -> set[tuple[int, int]]:
    if radius <= 0:
        return set(cells)
    out = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                out.add((r + dr, c + dc))
    return out


def _halo(cells: set[tuple[int, int]], radius: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                out.add((r + dr, c + dc))
    return out


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    dirs = DIR4 if connectivity == 4 else [
        (dr, dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    ]
    components = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _is_short_line(comp: np.ndarray) -> bool:
    if not (1 <= len(comp) <= 3):
        return False
    rows = sorted(int(r) for r in comp[:, 0])
    cols = sorted(int(c) for c in comp[:, 1])
    same_row = rows[0] == rows[-1] and cols == list(range(cols[0], cols[0] + len(cols)))
    same_col = cols[0] == cols[-1] and rows == list(range(rows[0], rows[0] + len(rows)))
    return same_row or same_col


def _mask_from_template(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


def _transform_mask(mask: np.ndarray, rotations: int, flip: bool) -> np.ndarray:
    out = np.rot90(mask, rotations)
    if flip:
        out = np.fliplr(out)
    return out.copy()


def _mask_to_cells(mask: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(mask)}


def _edge_segments(edge: np.ndarray, length: int) -> list[int]:
    starts = []
    run_start = None
    for idx, value in enumerate(edge.tolist()):
        if value and run_start is None:
            run_start = idx
        if not value and run_start is not None:
            run_len = idx - run_start
            for start in range(run_start, idx - length + 1):
                if run_len >= length:
                    starts.append(start)
            run_start = None
    if run_start is not None:
        run_len = len(edge) - run_start
        for start in range(run_start, len(edge) - length + 1):
            if run_len >= length:
                starts.append(start)
    return starts


def _shadow_candidates_from_profiles(
    mask: np.ndarray, direction: str, length: int
) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    result: list[list[tuple[int, int]]] = []

    if direction in {"right", "left"}:
        profile = []
        for row in range(h):
            cols = np.flatnonzero(mask[row])
            if cols.size == 0:
                profile.append(None)
            elif direction == "right":
                profile.append(int(cols.max()))
            else:
                profile.append(int(cols.min()))

        for start, value in _constant_runs(profile, length):
            col = int(value + 1) if direction == "right" else int(value - 1)
            result.append([(row, col) for row in range(start, start + length)])
    else:
        profile = []
        for col in range(w):
            rows = np.flatnonzero(mask[:, col])
            if rows.size == 0:
                profile.append(None)
            elif direction == "down":
                profile.append(int(rows.max()))
            else:
                profile.append(int(rows.min()))

        for start, value in _constant_runs(profile, length):
            row = int(value + 1) if direction == "down" else int(value - 1)
            result.append([(row, col) for col in range(start, start + length)])

    return result


def _constant_runs(values: list[int | None], length: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = 0
    while start < len(values):
        value = values[start]
        if value is None:
            start += 1
            continue
        end = start + 1
        while end < len(values) and values[end] == value:
            end += 1
        if end - start >= length:
            for pos in range(start, end - length + 1):
                runs.append((pos, int(value)))
        start = end
    return runs


def _local_shadow_direction(
    object_cells: set[tuple[int, int]], shadow_cells: set[tuple[int, int]]
) -> str | None:
    adj_dirs = set()
    for r, c in shadow_cells:
        for dr, dc in DIR4:
            if (r + dr, c + dc) in object_cells:
                adj_dirs.add((dr, dc))
    if len(adj_dirs) != 1:
        return None
    proj = tuple(-v for v in next(iter(adj_dirs)))
    for name, vec in VECTORS.items():
        if vec == proj:
            return name
    return None


def _shift_cells(cells: set[tuple[int, int]], anchor: tuple[int, int]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(r + ar, c + ac) for r, c in cells}


def _cell_span_area(cells: set[tuple[int, int]]) -> int:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
