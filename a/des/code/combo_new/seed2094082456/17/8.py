from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


BG = 0
BASELINE_COLOR = 5
OBJECT_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    neighbors = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    parts: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(comp)
    return parts


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _apply_random_symmetry(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(cells)
    if random.random() < 0.5:
        max_r = max(r for r, _ in out)
        out = {(max_r - r, c) for r, c in out}
    if random.random() < 0.5:
        max_c = max(c for _, c in out)
        out = {(r, max_c - c) for r, c in out}
    if random.random() < 0.35:
        out = {(c, r) for r, c in out}
    return _normalize(out)


def _grow_polyomino(area: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))
    return _apply_random_symmetry(_normalize(cells))


def _touches_8(shape: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in shape:
        for dr, dc in NEIGHBORS8:
            if (r + dr, c + dc) in occupied:
                return True
    return False


def _place_touching(shape: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> set[tuple[int, int]] | None:
    if not occupied:
        return set(shape)

    frontier = set()
    for r, c in occupied:
        for dr, dc in NEIGHBORS8:
            frontier.add((r + dr, c + dc))

    shape_cells = list(shape)
    candidate_offsets = set()
    for tr, tc in frontier:
        for sr, sc in shape_cells:
            candidate_offsets.add((tr - sr, tc - sc))

    offsets = list(candidate_offsets)
    random.shuffle(offsets)
    for off_r, off_c in offsets:
        shifted = {(r + off_r, c + off_c) for r, c in shape}
        if shifted & occupied:
            continue
        if not _touches_8(shifted, occupied):
            continue
        return shifted
    return None


def _unique_area_choices(layer_count: int, max_total_area: int) -> list[tuple[int, ...]]:
    choices = []
    upper = min(7, max_total_area)
    min_total = max(5, layer_count + 1)
    for combo in itertools.combinations(range(1, upper + 1), layer_count):
        total = sum(combo)
        if min_total <= total <= max_total_area:
            choices.append(combo)
    return choices


def _make_object(max_total_area: int) -> dict[str, object] | None:
    max_layers = 0
    for layer_count in range(2, 5):
        if layer_count * (layer_count + 1) // 2 <= max_total_area:
            max_layers = layer_count
    if max_layers < 2:
        return None

    layer_count = random.randint(2, max_layers)
    area_choices = _unique_area_choices(layer_count, max_total_area)
    if not area_choices:
        return None

    for _ in range(80):
        areas = list(random.choice(area_choices))
        random.shuffle(areas)
        colors = random.sample(OBJECT_COLOR_POOL, layer_count)

        layers: dict[int, set[tuple[int, int]]] = {}
        occupied: set[tuple[int, int]] = set()
        ok = True
        for color, area in zip(colors, areas):
            placed = None
            for _ in range(80):
                blob = _grow_polyomino(area)
                placed = _place_touching(blob, occupied)
                if placed is not None:
                    layers[color] = placed
                    occupied |= placed
                    break
            if placed is None:
                ok = False
                break
        if not ok:
            continue

        union = _normalize(occupied)
        min_r, min_c, max_r, max_c = _bbox(union)
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        if height > 8 or width > 9:
            continue

        norm_layers: dict[int, set[tuple[int, int]]] = {}
        for color, cells in layers.items():
            norm_layers[color] = _normalize(cells)

        used_cols = sorted({c for cells in norm_layers.values() for _, c in cells})
        anchor_color = random.choice(colors)
        color_cols = sorted({c for _, c in norm_layers[anchor_color]})
        anchor_col = random.choice(color_cols or used_cols)

        return {
            "layers": norm_layers,
            "width": width,
            "height": height,
            "anchor_col": anchor_col,
            "anchor_color": anchor_color,
            "total_area": sum(len(cells) for cells in norm_layers.values()),
        }
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse each multicolor composite object into a one-column stack of area-encoded color bars.

    Input structure:
    - Black background fills the grid above a colored baseline on the bottom row.
    - Several 8-connected multicolor composite objects lie above the baseline.
    - Each object uses 2 to 4 non-black colors, and every same-color layer is 4-connected.
    - The bottom row is mostly one baseline color, with one non-baseline anchor pixel beneath each object.

    Rule:
    - Keep the output grid the same size as the input, with all non-baseline cells reset to black.
    - For each composite object, split its pixels by color and count the area of each monochrome layer.
    - Match the object to the unique anchor whose baseline column lies under the object's horizontal span.
    - In that anchor column, draw one-cell-wide vertical bars stacked upward from the baseline, ordering layers from largest area to smallest and giving each bar height equal to its layer area.
    - Preserve the entire baseline row unchanged.

    Key insight:
    - The objects are not moved as shapes; they are analyzed into colored area counts and re-encoded as vertical histograms positioned by the baseline anchors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 2, "expected room for a baseline row"

    baseline_row = height - 1
    baseline_vals = [int(v) for v in g[baseline_row]]
    baseline_color = Counter(baseline_vals).most_common(1)[0][0]
    anchors = [c for c in range(width) if int(g[baseline_row, c]) != baseline_color]
    assert anchors, "expected at least one anchor on the baseline"

    above = g[:baseline_row, :]
    components = _components(above != BG, connectivity=8)

    out = np.full_like(g, BG)
    out[baseline_row, :] = g[baseline_row, :]

    for comp in components:
        cells = {(int(r), int(c)) for r, c in comp}
        min_c = min(c for _, c in cells)
        max_c = max(c for _, c in cells)
        matched = [c for c in anchors if min_c <= c <= max_c]
        assert len(matched) == 1, "each object must have exactly one baseline anchor under its span"
        anchor_col = matched[0]

        by_color: dict[int, list[tuple[int, int]]] = {}
        for r, c in cells:
            color = int(g[r, c])
            by_color.setdefault(color, []).append((r, c))
        assert 2 <= len(by_color) <= 4, "each composite object must use 2 to 4 colors"

        layers = []
        for color, color_cells in by_color.items():
            mask = np.zeros((baseline_row, width), dtype=bool)
            for r, c in color_cells:
                mask[r, c] = True
            assert len(_components(mask, connectivity=4)) == 1, "each color layer must be 4-connected"
            rows = [r for r, _ in color_cells]
            cols = [c for _, c in color_cells]
            layers.append((len(color_cells), min(rows), min(cols), color))

        layers.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
        cursor = baseline_row - 1
        for area, _row_key, _col_key, color in layers:
            top = cursor - area + 1
            assert top >= 0, "stack must fit above the baseline"
            out[top : cursor + 1, anchor_col] = color
            cursor = top - 1

    return out


def _expected_from_spec(height: int, width: int, placed_objects: list[dict[str, object]], baseline: np.ndarray) -> np.ndarray:
    out = np.full((height, width), BG, dtype=int)
    out[-1, :] = baseline
    for obj in placed_objects:
        anchor_col = int(obj["anchor_col"])
        areas = []
        for color, cells in obj["layers"].items():
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            areas.append((len(cells), min(rows), min(cols), int(color)))
        areas.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
        cursor = height - 2
        for area, _row_key, _col_key, color in areas:
            top = cursor - area + 1
            if top < 0:
                return np.empty((0, 0), dtype=int)
            out[top : cursor + 1, anchor_col] = color
            cursor = top - 1
    return out


def _validate_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    height, width = grid.shape
    if height > 30 or width > 30 or height < 6 or width < 8:
        return False
    if not np.any(grid[-1] != grid[-1, 0]):
        return False

    baseline_color = Counter(map(int, grid[-1])).most_common(1)[0][0]
    anchors = [c for c in range(width) if int(grid[-1, c]) != baseline_color]
    if not anchors:
        return False

    comps = _components(grid[:-1] != BG, connectivity=8)
    if len(comps) != len(anchors):
        return False

    for comp in comps:
        cells = {(int(r), int(c)) for r, c in comp}
        min_c = min(c for _, c in cells)
        max_c = max(c for _, c in cells)
        matched = [c for c in anchors if min_c <= c <= max_c]
        if len(matched) != 1:
            return False
        if len(cells) > height - 1:
            return False

        by_color: dict[int, list[tuple[int, int]]] = {}
        for r, c in cells:
            color = int(grid[r, c])
            if color in (BG, baseline_color):
                return False
            by_color.setdefault(color, []).append((r, c))
        if not (2 <= len(by_color) <= 4):
            return False
        areas = [len(color_cells) for color_cells in by_color.values()]
        if len(set(areas)) != len(areas):
            return False
        for color_cells in by_color.values():
            mask = np.zeros((height - 1, width), dtype=bool)
            for r, c in color_cells:
                mask[r, c] = True
            if len(_components(mask, connectivity=4)) != 1:
                return False
    return True


def _split_slack(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    pieces = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        pieces.append(cut - prev - 1)
        prev = cut
    return pieces


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(10, 18)
    max_object_area = min(height - 2, 13)

    object_count = random.randint(1, 3)
    specs = []
    for _ in range(object_count):
        spec = _make_object(max_object_area)
        if spec is None:
            return None
        specs.append(spec)

    min_width = 2 + sum(int(spec["width"]) for spec in specs) + 2 * (object_count - 1)
    if min_width > 30:
        return None
    width = random.randint(max(10, min_width), min(30, min_width + 8))

    slack = width - min_width
    extras = _split_slack(slack, object_count + 1)
    gaps = [1 + extras[0]]
    for i in range(1, object_count):
        gaps.append(2 + extras[i])
    gaps.append(1 + extras[-1])

    baseline = np.full(width, BASELINE_COLOR, dtype=int)
    grid = np.full((height, width), BG, dtype=int)
    grid[-1, :] = baseline

    placed_specs: list[dict[str, object]] = []
    current_c = gaps[0]
    for i, spec in enumerate(specs):
        obj_h = int(spec["height"])
        obj_w = int(spec["width"])
        max_r0 = height - 2 - obj_h
        if max_r0 < 0:
            return None
        r0 = random.randint(0, max_r0)
        c0 = current_c

        placed_layers: dict[int, set[tuple[int, int]]] = {}
        for color, cells in spec["layers"].items():
            translated = {(r + r0, c + c0) for r, c in cells}
            placed_layers[int(color)] = translated
            for r, c in translated:
                grid[r, c] = int(color)

        anchor_col = c0 + int(spec["anchor_col"])
        anchor_color = int(spec["anchor_color"])
        baseline[anchor_col] = anchor_color
        grid[-1, anchor_col] = anchor_color

        placed_specs.append(
            {
                "layers": placed_layers,
                "anchor_col": anchor_col,
            }
        )
        current_c += obj_w + gaps[i + 1]

    expected = _expected_from_spec(height, width, placed_specs, baseline)
    if expected.size == 0:
        return None
    if not _validate_input(grid):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
