import random
from collections import deque

import numpy as np


BLACK = 0
GRAY = 5
NON_WALL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
QUADRANTS = ("tl", "tr", "bl", "br")
LAYOUTS = ((1, 2), (2, 1), (2, 2), (1, 3), (3, 1))
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _unique_colors(grid: np.ndarray, exclude: tuple[int, ...] = ()) -> list[int]:
    return [int(color) for color in np.unique(grid) if int(color) not in exclude]


def _compartment_boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    boxes = []
    for comp in _components(grid != GRAY):
        box = _bbox(comp)
        r0, c0, r1, c1 = box
        assert len(comp) == (r1 - r0 + 1) * (c1 - c0 + 1), "compartment interior must be rectangular"
        boxes.append(box)
    return sorted(boxes)


def _is_wall_marker_cell(r: int, c: int, h: int, w: int) -> bool:
    on_horizontal = r in (0, h - 1)
    on_vertical = c in (0, w - 1)
    return on_horizontal ^ on_vertical


def _half_interval(length: int, side: str) -> tuple[int, int]:
    if side in ("top", "left"):
        start, end = 1, length // 2 - 1
    else:
        start, end = length - length // 2, length - 2
    if start > end:
        raise AssertionError("compartment too small for a corner-quadrant pattern")
    return start, end


def _pattern_quadrant(pattern_coords: list[tuple[int, int]], h: int, w: int) -> str:
    rows = [r for r, _ in pattern_coords]
    cols = [c for _, c in pattern_coords]
    if max(rows) < h / 2:
        vertical = "t"
    elif min(rows) > (h - 1) / 2:
        vertical = "b"
    else:
        raise AssertionError("pattern must stay in one vertical half")
    if max(cols) < w / 2:
        horizontal = "l"
    elif min(cols) > (w - 1) / 2:
        horizontal = "r"
    else:
        raise AssertionError("pattern must stay in one horizontal half")
    return vertical + horizontal


def _mirror_coords(
    coords: list[tuple[int, int]],
    h: int,
    w: int,
    axis: str,
) -> list[tuple[int, int]]:
    if axis == "horizontal":
        return [(h - 1 - r, c) for r, c in coords]
    if axis == "vertical":
        return [(r, w - 1 - c) for r, c in coords]
    raise ValueError(f"unknown axis {axis}")


def _parse_compartment(subgrid: np.ndarray) -> dict:
    h, w = subgrid.shape
    colors = _unique_colors(subgrid, exclude=(BLACK, GRAY))
    assert len(colors) == 2, "each compartment must contain one pattern color and one marker color"

    marker_candidates = []
    for color in colors:
        coords = [(int(r), int(c)) for r, c in np.argwhere(subgrid == color)]
        comps = _components(subgrid == color)
        if len(coords) >= 2 and all(len(comp) == 1 for comp in comps) and all(
            _is_wall_marker_cell(r, c, h, w) for r, c in coords
        ):
            marker_candidates.append(color)
    assert len(marker_candidates) == 1, "marker color must be uniquely identifiable"

    marker_color = int(marker_candidates[0])
    pattern_color = int([color for color in colors if color != marker_color][0])
    pattern_comps = _components(subgrid == pattern_color)
    assert len(pattern_comps) == 1 and len(pattern_comps[0]) >= 2, "pattern must be one multi-pixel object"

    pattern_coords = sorted(pattern_comps[0])
    marker_coords = [(int(r), int(c)) for r, c in np.argwhere(subgrid == marker_color)]

    horizontal_votes = 0
    vertical_votes = 0
    for r, c in marker_coords:
        if r in (0, h - 1) and c not in (0, w - 1):
            horizontal_votes += 1
        elif c in (0, w - 1) and r not in (0, h - 1):
            vertical_votes += 1
        else:
            raise AssertionError("marker pixels may not occupy compartment corners")
    assert horizontal_votes > 0 and vertical_votes > 0 and horizontal_votes != vertical_votes

    axis = "horizontal" if horizontal_votes > vertical_votes else "vertical"
    quadrant = _pattern_quadrant(pattern_coords, h, w)
    mirror_targets = _mirror_coords(pattern_coords, h, w, axis)
    wrong_axis = "vertical" if axis == "horizontal" else "horizontal"
    wrong_targets = _mirror_coords(pattern_coords, h, w, wrong_axis)

    for r, c in mirror_targets:
        assert subgrid[r, c] == BLACK, "mirror side must be empty before painting"

    return {
        "marker_color": marker_color,
        "pattern_color": pattern_color,
        "pattern_coords": pattern_coords,
        "marker_coords": marker_coords,
        "quadrant": quadrant,
        "axis": axis,
        "horizontal_votes": horizontal_votes,
        "vertical_votes": vertical_votes,
        "mirror_targets": mirror_targets,
        "wrong_targets": wrong_targets,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each corner-quadrant pattern across the axis chosen by the compartment's wall-marker majority.

    Input structure:
    - Gray wall lines partition the black background into rectangular compartments.
    - Each compartment contains one monochrome connected pattern placed entirely in one corner quadrant.
    - The same compartment also contains several one-pixel wall markers of another color, with some markers
      attached to the top/bottom walls and some attached to the left/right walls.

    Rule:
    - Process each compartment independently.
    - Count the marker pixels on horizontal walls versus vertical walls; the strict majority chooses the axis.
    - If horizontal-wall markers are the majority, reflect the pattern across the compartment's horizontal
      midline; otherwise reflect it across the vertical midline.
    - Paint only the reflected copy in the marker color, while leaving the original pattern, marker pixels,
      and gray walls unchanged.

    Key insight:
    - The marker color supplies the repaint color, but the important signal is which class of walls hosts
      more markers, because that vote determines whether the completion is a top-bottom or left-right mirror.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for r0, c0, r1, c1 in _compartment_boxes(g):
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        info = _parse_compartment(sub)
        for rr, cc in info["mirror_targets"]:
            out[r0 + rr, c0 + cc] = info["marker_color"]

    return out


def _sample_sizes(count: int, lo: int, hi: int, max_total: int) -> list[int] | None:
    for _ in range(120):
        sizes = [random.randint(lo, hi) for _ in range(count)]
        if sum(sizes) + count + 1 <= max_total:
            return sizes
    return None


def _boundary_indices(sizes: list[int]) -> list[int]:
    boundaries = [0]
    acc = 0
    for size in sizes:
        acc += size + 1
        boundaries.append(acc)
    return boundaries


def _random_connected_shape(h: int, w: int) -> list[tuple[int, int]]:
    area_hi = min(h * w, 6)
    area_lo = 3 if area_hi >= 3 else 2
    for _ in range(100):
        target = random.randint(area_lo, area_hi)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if len(cells) >= 2 and max(rows) > min(rows) and max(cols) > min(cols):
            return sorted(cells)
    return [(0, 0), (0, min(1, w - 1)), (min(1, h - 1), 0)]


def _place_pattern(h: int, w: int, quadrant: str) -> list[tuple[int, int]] | None:
    row_side = "top" if quadrant[0] == "t" else "bottom"
    col_side = "left" if quadrant[1] == "l" else "right"
    r_lo, r_hi = _half_interval(h, row_side)
    c_lo, c_hi = _half_interval(w, col_side)
    span_h = r_hi - r_lo + 1
    span_w = c_hi - c_lo + 1
    if span_h < 2 or span_w < 2:
        return None

    shape_h = random.randint(2, min(4, span_h))
    shape_w = random.randint(2, min(4, span_w))
    anchor_r = random.randint(r_lo, r_hi - shape_h + 1)
    anchor_c = random.randint(c_lo, c_hi - shape_w + 1)
    shape = _random_connected_shape(shape_h, shape_w)
    return sorted((anchor_r + r, anchor_c + c) for r, c in shape)


def _nonadjacent_sample(slots: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    for _ in range(200):
        chosen = random.sample(slots, count)
        ok = True
        for i, (r1, c1) in enumerate(chosen):
            for r2, c2 in chosen[:i]:
                if abs(r1 - r2) + abs(c1 - c2) == 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return chosen
    return None


def _place_markers(h: int, w: int, axis: str) -> tuple[list[tuple[int, int]], int, int] | None:
    horizontal_slots = [(0, c) for c in range(1, w - 1)] + [(h - 1, c) for c in range(1, w - 1)]
    vertical_slots = [(r, 0) for r in range(1, h - 1)] + [(r, w - 1) for r in range(1, h - 1)]
    max_h_count = 2 * ((w - 1) // 2)
    max_v_count = 2 * ((h - 1) // 2)

    for _ in range(100):
        if axis == "horizontal":
            h_count = random.randint(3, min(4, max_h_count))
            v_count = random.randint(1, min(2, max_v_count, h_count - 1))
        else:
            v_count = random.randint(3, min(4, max_v_count))
            h_count = random.randint(1, min(2, max_h_count, v_count - 1))
        h_markers = _nonadjacent_sample(horizontal_slots, h_count)
        v_markers = _nonadjacent_sample(vertical_slots, v_count)
        if h_markers is not None and v_markers is not None:
            return h_markers + v_markers, h_count, v_count
    return None


def _render_from_specs(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        r0, c0, r1, c1 = spec["box"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        for gr, gc in spec["pattern_global"]:
            lr = gr - r0
            lc = gc - c0
            if spec["axis"] == "horizontal":
                tr, tc = r0 + (h - 1 - lr), gc
            else:
                tr, tc = gr, c0 + (w - 1 - lc)
            assert grid[tr, tc] == BLACK
            out[tr, tc] = spec["marker_color"]
    return out


def _targets_from_spec(spec: dict, axis: str | None = None) -> set[tuple[int, int]]:
    axis = spec["axis"] if axis is None else axis
    r0, c0, r1, c1 = spec["box"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    out = set()
    for gr, gc in spec["pattern_global"]:
        lr = gr - r0
        lc = gc - c0
        if axis == "horizontal":
            out.add((r0 + (h - 1 - lr), gc))
        else:
            out.add((gr, c0 + (w - 1 - lc)))
    return out


def _validate_case(grid: np.ndarray, specs: list[dict]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    boxes = _compartment_boxes(grid)
    spec_by_box = {spec["box"]: spec for spec in specs}
    if sorted(spec_by_box) != boxes:
        return False

    seen_axes = set()
    seen_quadrant_axis = set()
    for box in boxes:
        spec = spec_by_box[box]
        r0, c0, r1, c1 = box
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        info = _parse_compartment(sub)

        parsed_pattern = {(r0 + r, c0 + c) for r, c in info["pattern_coords"]}
        parsed_markers = {(r0 + r, c0 + c) for r, c in info["marker_coords"]}
        parsed_targets = {(r0 + r, c0 + c) for r, c in info["mirror_targets"]}

        if parsed_pattern != set(spec["pattern_global"]):
            return False
        if parsed_markers != set(spec["marker_global"]):
            return False
        if parsed_targets != _targets_from_spec(spec):
            return False
        if info["marker_color"] != spec["marker_color"] or info["pattern_color"] != spec["pattern_color"]:
            return False
        if info["axis"] != spec["axis"] or info["quadrant"] != spec["quadrant"]:
            return False
        if _targets_from_spec(spec, "horizontal") == _targets_from_spec(spec, "vertical"):
            return False
        seen_axes.add(spec["axis"])
        seen_quadrant_axis.add((spec["quadrant"], spec["axis"]))

    if seen_axes != {"horizontal", "vertical"}:
        return False

    shared_quadrant = {quadrant for quadrant, _ in seen_quadrant_axis}
    if not any((quadrant, "horizontal") in seen_quadrant_axis and (quadrant, "vertical") in seen_quadrant_axis for quadrant in shared_quadrant):
        return False

    expected = _render_from_specs(grid, specs)
    if np.array_equal(expected, grid):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    rows, cols = random.choice(LAYOUTS)
    row_sizes = _sample_sizes(rows, 6, 9, 30)
    col_sizes = _sample_sizes(cols, 6, 9, 30)
    if row_sizes is None or col_sizes is None:
        return None

    h = sum(row_sizes) + rows + 1
    w = sum(col_sizes) + cols + 1
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    row_boundaries = _boundary_indices(row_sizes)
    col_boundaries = _boundary_indices(col_sizes)
    grid[row_boundaries, :] = GRAY
    grid[:, col_boundaries] = GRAY

    boxes = []
    for i in range(rows):
        for j in range(cols):
            boxes.append(
                (
                    row_boundaries[i] + 1,
                    col_boundaries[j] + 1,
                    row_boundaries[i + 1] - 1,
                    col_boundaries[j + 1] - 1,
                )
            )

    compartment_count = len(boxes)
    marker_colors = random.sample(NON_WALL_COLORS, compartment_count)

    axes = ["horizontal", "vertical"]
    while len(axes) < compartment_count:
        axes.append(random.choice(["horizontal", "vertical"]))
    random.shuffle(axes)

    quadrants = [random.choice(QUADRANTS) for _ in range(compartment_count)]
    h_index = next(i for i, axis in enumerate(axes) if axis == "horizontal")
    v_index = next(i for i, axis in enumerate(axes) if axis == "vertical")
    shared_quadrant = random.choice(QUADRANTS)
    quadrants[h_index] = shared_quadrant
    quadrants[v_index] = shared_quadrant

    specs = []
    for idx, box in enumerate(boxes):
        r0, c0, r1, c1 = box
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        quadrant = quadrants[idx]
        axis = axes[idx]

        pattern_local = _place_pattern(box_h, box_w, quadrant)
        marker_result = _place_markers(box_h, box_w, axis)
        if pattern_local is None or marker_result is None:
            return None
        marker_local, h_votes, v_votes = marker_result

        marker_color = marker_colors[idx]
        pattern_color_choices = [color for color in NON_WALL_COLORS if color != marker_color]
        pattern_color = random.choice(pattern_color_choices)

        pattern_global = [(r0 + r, c0 + c) for r, c in pattern_local]
        marker_global = [(r0 + r, c0 + c) for r, c in marker_local]

        for gr, gc in pattern_global:
            grid[gr, gc] = pattern_color
        for gr, gc in marker_global:
            grid[gr, gc] = marker_color

        specs.append(
            {
                "box": box,
                "axis": axis,
                "quadrant": quadrant,
                "marker_color": marker_color,
                "pattern_color": pattern_color,
                "pattern_global": pattern_global,
                "marker_global": marker_global,
                "horizontal_votes": h_votes,
                "vertical_votes": v_votes,
            }
        )

    if not _validate_case(grid, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
