import itertools
import random
import sys
from pathlib import Path

import numpy as np

UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, in_bounds, is_contiguous


BG = int(Color.BLACK)
CRACK = int(Color.MAROON)
BODY_TO_COPY = {
    int(Color.BLUE): int(Color.RED),
    int(Color.GREEN): int(Color.YELLOW),
    int(Color.ORANGE): int(Color.PINK),
}
BODY_COLORS = tuple(BODY_TO_COPY)

GRID_ROWS = (16, 22)
GRID_COLS = (16, 22)
OBJECT_COUNT_CHOICES = (2, 3)
GEN_MAX_ATTEMPTS = 500
SHAPE_ATTEMPTS = 200
SCENE_ATTEMPTS = 200


def _neighbor_cells(cell, height, width):
    r, c = cell
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < height and 0 <= cc < width:
            yield rr, cc


def _grow_connected_mask(height, width, area):
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier = set(_neighbor_cells(start, height, width))
    while len(cells) < area and frontier:
        choices = list(frontier)
        weights = []
        for cell in choices:
            degree = sum(nei in cells for nei in _neighbor_cells(cell, height, width))
            weights.append((degree + 1) ** 2)
        cell = random.choices(choices, weights=weights, k=1)[0]
        frontier.remove(cell)
        cells.add(cell)
        for nei in _neighbor_cells(cell, height, width):
            if nei not in cells:
                frontier.add(nei)
    if len(cells) != area:
        return None
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_sparse_support(length):
    valid = []
    for size in range(2, length):
        for combo in itertools.combinations(range(length), size):
            if combo[-1] - combo[0] + 1 > len(combo):
                valid.append(combo)
    if not valid:
        return None
    return random.choice(valid)


def _make_body_mask(body_h, body_w, near_side):
    if near_side in {"up", "down"}:
        required_near = body_h - 1 if near_side == "up" else 0
        required_far = 0 if near_side == "up" else body_h - 1
    else:
        required_near = body_w - 1 if near_side == "left" else 0
        required_far = 0 if near_side == "left" else body_w - 1

    max_area = min(10, body_h * body_w)
    min_area = 5
    if min_area > max_area:
        return None

    for _ in range(SHAPE_ATTEMPTS):
        area = random.randint(min_area, max_area)
        mask = _grow_connected_mask(body_h, body_w, area)
        if mask is None:
            continue
        if not is_contiguous(mask.astype(int), background=0, connectivity=4):
            continue
        if near_side in {"left", "right"}:
            if not np.any(mask[:, required_near]) or not np.any(mask[:, required_far]):
                continue
        else:
            if not np.any(mask[required_near, :]) or not np.any(mask[required_far, :]):
                continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        return mask
    return None


def _make_object_spec(body_color, orientation):
    for _ in range(SHAPE_ATTEMPTS):
        if orientation == "vertical":
            height = random.randint(3, 5)
            width = random.randint(height + 2, min(7, height + 4))
            body_width = width - 1
            body_side = random.choice(["left", "right"])
            mask = _make_body_mask(height, body_width, body_side)
            if mask is None:
                continue

            line_col = body_width if body_side == "left" else 0
            marker_gap = random.randint(1, 3)
            marker_col = line_col + marker_gap + 1 if body_side == "left" else line_col - marker_gap - 1
            support = _make_sparse_support(height)
            if support is None:
                continue

            source_cells = {}
            body_cells = []
            for r, c in np.argwhere(mask):
                rr = int(r)
                cc = int(c) if body_side == "left" else int(c) + 1
                source_cells[(rr, cc)] = int(body_color)
                body_cells.append((rr, cc))
            for r in range(height):
                source_cells[(r, line_col)] = CRACK
            for r in support:
                source_cells[(int(r), marker_col)] = CRACK

            target_cells = []
            for r, c in body_cells:
                dist = abs(c - line_col)
                target_col = marker_col + dist if marker_col > line_col else marker_col - dist
                target_cells.append((r, target_col))
        else:
            width = random.randint(3, 5)
            height = random.randint(width + 2, min(7, width + 4))
            body_height = height - 1
            body_side = random.choice(["up", "down"])
            mask = _make_body_mask(body_height, width, body_side)
            if mask is None:
                continue

            line_row = body_height if body_side == "up" else 0
            marker_gap = random.randint(1, 3)
            marker_row = line_row + marker_gap + 1 if body_side == "up" else line_row - marker_gap - 1
            support = _make_sparse_support(width)
            if support is None:
                continue

            source_cells = {}
            body_cells = []
            for r, c in np.argwhere(mask):
                rr = int(r) if body_side == "up" else int(r) + 1
                cc = int(c)
                source_cells[(rr, cc)] = int(body_color)
                body_cells.append((rr, cc))
            for c in range(width):
                source_cells[(line_row, c)] = CRACK
            for c in support:
                source_cells[(marker_row, int(c))] = CRACK

            target_cells = []
            for r, c in body_cells:
                dist = abs(r - line_row)
                target_row = marker_row + dist if marker_row > line_row else marker_row - dist
                target_cells.append((target_row, c))

        coords = list(source_cells) + target_cells
        min_r = min(r for r, _ in coords)
        max_r = max(r for r, _ in coords)
        min_c = min(c for _, c in coords)
        max_c = max(c for _, c in coords)
        if max_r - min_r + 1 > 12 or max_c - min_c + 1 > 12:
            continue
        return {
            "body_color": int(body_color),
            "copy_color": int(BODY_TO_COPY[body_color]),
            "orientation": orientation,
            "source_cells": source_cells,
            "target_cells": tuple(target_cells),
            "min_r": int(min_r),
            "max_r": int(max_r),
            "min_c": int(min_c),
            "max_c": int(max_c),
            "line_index": int(line_col if orientation == "vertical" else line_row),
            "marker_index": int(marker_col if orientation == "vertical" else marker_row),
            "body_side": body_side,
        }
    return None


def _buffered(coords, shape):
    height, width = shape
    out = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    out.add((rr, cc))
    return out


def _absolute_cells(spec, anchor_r, anchor_c):
    source = {}
    for (r, c), color in spec["source_cells"].items():
        source[(anchor_r + r, anchor_c + c)] = int(color)
    targets = {(anchor_r + r, anchor_c + c) for r, c in spec["target_cells"]}
    return source, targets


def _place_specs(shape, specs):
    height, width = shape
    placed = []
    reserved_cells = set()
    reserved_buffer = set()

    order = sorted(
        specs,
        key=lambda spec: (spec["max_r"] - spec["min_r"], spec["max_c"] - spec["min_c"]),
        reverse=True,
    )
    for spec in order:
        r_lo = -spec["min_r"]
        r_hi = height - 1 - spec["max_r"]
        c_lo = -spec["min_c"]
        c_hi = width - 1 - spec["max_c"]
        candidates = []
        for anchor_r in range(r_lo, r_hi + 1):
            for anchor_c in range(c_lo, c_hi + 1):
                source, targets = _absolute_cells(spec, anchor_r, anchor_c)
                coords = set(source) | targets
                if coords & reserved_buffer:
                    continue
                candidates.append((anchor_r, anchor_c, source, targets))
        if not candidates:
            return None
        anchor_r, anchor_c, source, targets = random.choice(candidates)
        coords = set(source) | targets
        placed.append(
            {
                **spec,
                "anchor_r": int(anchor_r),
                "anchor_c": int(anchor_c),
                "source_abs": source,
                "target_abs": tuple(sorted(targets)),
            }
        )
        reserved_cells |= coords
        reserved_buffer |= _buffered(coords, shape)
    return placed


def _build_expected(shape, placed_specs):
    inp = np.full(shape, BG, dtype=int)
    out = np.full(shape, BG, dtype=int)
    for spec in placed_specs:
        for (r, c), color in spec["source_abs"].items():
            inp[r, c] = int(color)
            if color != CRACK:
                out[r, c] = int(color)
        for r, c in spec["target_abs"]:
            out[r, c] = int(spec["copy_color"])
    return inp, out


def _find_sparse_marker_col(grid, r0, r1, search_cols):
    span = r1 - r0 + 1
    candidates = []
    for c in search_cols:
        if not (0 <= c < grid.shape[1]):
            continue
        rows = np.flatnonzero(grid[:, c] == CRACK)
        rows = rows[(r0 <= rows) & (rows <= r1)]
        if len(rows) < 2 or len(rows) >= span:
            continue
        if rows[-1] - rows[0] + 1 == len(rows):
            continue
        candidates.append((int(c), tuple(int(r) for r in rows)))
    assert len(candidates) == 1, "expected exactly one sparse marker column"
    return candidates[0]


def _find_sparse_marker_row(grid, c0, c1, search_rows):
    span = c1 - c0 + 1
    candidates = []
    for r in search_rows:
        if not (0 <= r < grid.shape[0]):
            continue
        cols = np.flatnonzero(grid[r, :] == CRACK)
        cols = cols[(c0 <= cols) & (cols <= c1)]
        if len(cols) < 2 or len(cols) >= span:
            continue
        if cols[-1] - cols[0] + 1 == len(cols):
            continue
        candidates.append((int(r), tuple(int(c) for c in cols)))
    assert len(candidates) == 1, "expected exactly one sparse marker row"
    return candidates[0]


def _extract_objects(grid):
    components = find_components(
        grid,
        bg=BG,
        colors=[*BODY_COLORS, CRACK],
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    objects = []
    for component in components:
        colors = {int(grid[r, c]) for r, c in component["coords"]}
        body_colors = [color for color in colors if color in BODY_COLORS]
        if colors == {CRACK}:
            continue
        assert len(body_colors) == 1 and colors == {body_colors[0], CRACK}

        body_color = int(body_colors[0])
        r0, c0, r1, c1 = map(int, component["bbox"])
        patch = crop_bbox(grid, component["bbox"])
        body_coords = [tuple(map(int, rc)) for rc in component["coords"] if int(grid[tuple(rc)]) == body_color]
        h, w = patch.shape
        assert h != w, "object bbox must be non-square"

        if h < w:
            local_cols = [c for c in range(w) if np.all(patch[:, c] == CRACK)]
            assert len(local_cols) == 1
            local_line = int(local_cols[0])
            assert local_line in {0, w - 1}
            body_local = np.argwhere(patch == body_color)
            if np.all(body_local[:, 1] < local_line):
                body_side = "left"
                marker_col, support = _find_sparse_marker_col(grid, r0, r1, range(c1 + 2, c1 + 5))
            else:
                assert np.all(body_local[:, 1] > local_line)
                body_side = "right"
                marker_col, support = _find_sparse_marker_col(grid, r0, r1, range(c0 - 2, c0 - 5, -1))
            line_col = c0 + local_line
            target_coords = []
            for r, c in body_coords:
                dist = abs(c - line_col)
                target_col = marker_col + dist if marker_col > line_col else marker_col - dist
                assert in_bounds(grid.shape, r, target_col)
                target_coords.append((r, int(target_col)))
            orientation = "vertical"
            line_index = int(line_col)
            marker_index = int(marker_col)
        else:
            local_rows = [r for r in range(h) if np.all(patch[r, :] == CRACK)]
            assert len(local_rows) == 1
            local_line = int(local_rows[0])
            assert local_line in {0, h - 1}
            body_local = np.argwhere(patch == body_color)
            if np.all(body_local[:, 0] < local_line):
                body_side = "up"
                marker_row, support = _find_sparse_marker_row(grid, c0, c1, range(r1 + 2, r1 + 5))
            else:
                assert np.all(body_local[:, 0] > local_line)
                body_side = "down"
                marker_row, support = _find_sparse_marker_row(grid, c0, c1, range(r0 - 2, r0 - 5, -1))
            line_row = r0 + local_line
            target_coords = []
            for r, c in body_coords:
                dist = abs(r - line_row)
                target_row = marker_row + dist if marker_row > line_row else marker_row - dist
                assert in_bounds(grid.shape, target_row, c)
                target_coords.append((int(target_row), c))
            orientation = "horizontal"
            line_index = int(line_row)
            marker_index = int(marker_row)

        objects.append(
            {
                "body_color": body_color,
                "copy_color": int(BODY_TO_COPY[body_color]),
                "body_coords": tuple(sorted(body_coords)),
                "target_coords": tuple(sorted(target_coords)),
                "orientation": orientation,
                "line_index": line_index,
                "marker_index": marker_index,
                "body_side": body_side,
                "support": support,
            }
        )

    colors = [obj["body_color"] for obj in objects]
    assert len(objects) in {2, 3}
    assert len(colors) == len(set(colors))
    return sorted(objects, key=lambda obj: obj["body_color"])


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Reflect each one-sided fractured object across its paired sparse guide line and recolor the copy.

    Input structure:
    - A black background contains 2 or 3 disconnected foreground objects, each using one body color from blue, green, or orange plus maroon fracture pixels.
    - Inside each object, a full maroon row or column lies on the bounding-box short axis and spans that short side completely, with all body pixels on only one side of it.
    - Outside each object, a sparse maroon row or column runs parallel to the fracture line at distance 1 to 3 and aligns to the same bbox span.

    Rule:
    - For each object, detect whether its internal maroon fracture line is horizontal or vertical, then find the unique external sparse maroon line parallel to it on the empty side.
    - For every body pixel, measure its distance to the fracture line and place a reflected copy on the far side of the sparse guide line at the same distance.
    - Recolor copied pixels by blue->red, green->yellow, and orange->pink; keep the original body pixels unchanged; erase every maroon pixel.

    Key insight:
    - The internal fracture line serves two roles at once: its orientation selects the reflection direction, and together with the external sparse guide it forms the paired parallel axes that locate the mirrored copy.
    """
    grid = np.asarray(input_grid, dtype=int)
    out = grid.copy()
    out[out == CRACK] = BG

    for obj in _extract_objects(grid):
        for r, c in obj["target_coords"]:
            out[r, c] = int(obj["copy_color"])
    return out


def _choose_orientations(count):
    if count == 2:
        orientations = ["vertical", "horizontal"]
    else:
        orientations = ["vertical", "horizontal", random.choice(["vertical", "horizontal"])]
    random.shuffle(orientations)
    return orientations


def _generate_scene():
    rows = random.randint(*GRID_ROWS)
    cols = random.randint(*GRID_COLS)
    count = random.choice(OBJECT_COUNT_CHOICES)
    body_colors = random.sample(BODY_COLORS, count)
    orientations = _choose_orientations(count)

    specs = []
    for body_color, orientation in zip(body_colors, orientations):
        spec = _make_object_spec(body_color, orientation)
        if spec is None:
            return None
        specs.append(spec)

    placed = _place_specs((rows, cols), specs)
    if placed is None:
        return None

    inp, expected = _build_expected((rows, cols), placed)
    return inp, expected, placed


def _valid_scene(inp, expected, placed):
    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False

    parsed = _extract_objects(inp)
    by_color = {obj["body_color"]: obj for obj in parsed}
    if len(by_color) != len(placed):
        return False

    for spec in placed:
        obj = by_color.get(spec["body_color"])
        if obj is None:
            return False
        if obj["orientation"] != spec["orientation"]:
            return False
        if obj["line_index"] != spec["anchor_c"] + spec["line_index"] and spec["orientation"] == "vertical":
            return False
        if obj["line_index"] != spec["anchor_r"] + spec["line_index"] and spec["orientation"] == "horizontal":
            return False
        if obj["marker_index"] != spec["anchor_c"] + spec["marker_index"] and spec["orientation"] == "vertical":
            return False
        if obj["marker_index"] != spec["anchor_r"] + spec["marker_index"] and spec["orientation"] == "horizontal":
            return False
        if tuple(sorted(obj["target_coords"])) != tuple(sorted(spec["target_abs"])):
            return False

    has_vertical = any(obj["orientation"] == "vertical" for obj in parsed)
    has_horizontal = any(obj["orientation"] == "horizontal" for obj in parsed)
    if not (has_vertical and has_horizontal):
        return False
    if np.array_equal(inp, expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        for _ in range(SCENE_ATTEMPTS):
            scene = _generate_scene()
            if scene is None:
                continue
            inp, expected, placed = scene
            if _valid_scene(inp, expected, placed):
                return inp
    raise RuntimeError("failed to generate a valid task instance")
