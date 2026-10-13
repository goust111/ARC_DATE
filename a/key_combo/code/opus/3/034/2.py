from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 0
SOURCE_COLOR = 3
RECOLOR_BY_BENDS = {1: 1, 2: 6, 3: 2}
MARKER_COLORS = (4, 5, 7, 8, 9)

GRID_ROWS_RANGE = (18, 25)
GRID_COLS_RANGE = (18, 25)
OBJECT_COUNT_RANGE = (3, 5)
PLACEMENT_GAP_RANGE = (2, 6)
GENERATION_RETRIES = 400
OBJECT_PLACEMENT_RETRIES = 500


def _normalize(coords):
    coords = tuple(sorted((int(r), int(c)) for r, c in coords))
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_dims(coords):
    r0, c0, r1, c1 = _bbox(coords)
    return r1 - r0 + 1, c1 - c0 + 1


def _coords_to_mask(coords):
    h, w = _shape_dims(coords)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in _normalize(coords):
        mask[r, c] = True
    return mask


def _mask_to_coords(mask):
    return tuple(sorted((int(r), int(c)) for r, c in np.argwhere(mask)))


def _dihedral_variants(coords):
    mask = _coords_to_mask(coords)
    variants = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.add(_normalize(_mask_to_coords(rotated)))
        variants.add(_normalize(_mask_to_coords(np.fliplr(rotated))))
    return sorted(variants)


def _count_bends(coords):
    mask = _coords_to_mask(coords).astype(int)
    bends = 0
    for r in range(mask.shape[0] - 1):
        for c in range(mask.shape[1] - 1):
            if int(mask[r : r + 2, c : c + 2].sum()) == 3:
                bends += 1
    return bends


def _row_anchor_candidates(coords):
    row_counts = Counter(r for r, _ in coords)
    return [(r, c) for r, c in coords if row_counts[r] == 1]


def _col_anchor_candidates(coords):
    col_counts = Counter(c for _, c in coords)
    return [(r, c) for r, c in coords if col_counts[c] == 1]


def _build_shape_library():
    base_shapes = {
        1: [
            ((0, 0), (1, 0), (1, 1)),
            ((0, 0), (1, 0), (2, 0), (2, 1)),
            ((0, 0), (0, 1), (0, 2), (1, 0)),
            ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
        ],
        2: [
            ((0, 1), (0, 2), (1, 0), (1, 1)),
            ((0, 0), (0, 1), (0, 2), (1, 1)),
        ],
        3: [
            ((0, 1), (0, 2), (1, 0), (1, 1), (2, 0)),
            ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
        ],
    }

    library = {1: [], 2: [], 3: []}
    for bends, shapes in base_shapes.items():
        seen = set()
        for shape in shapes:
            for variant in _dihedral_variants(shape):
                if variant in seen:
                    continue
                if _count_bends(variant) != bends:
                    continue
                if not _row_anchor_candidates(variant) and not _col_anchor_candidates(variant):
                    continue
                seen.add(variant)
                library[bends].append(variant)
    return library


SHAPE_LIBRARY = _build_shape_library()


def _find_components_of_color(grid, color):
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == int(color):
                        visited[nr, nc] = True
                        q.append((nr, nc))
            components.append(tuple(sorted(comp)))
    components.sort(key=_bbox)
    return components


def _marker_relation(source_coords, marker):
    mr, mc = marker
    row_matches = [(r, c) for r, c in source_coords if r == mr]
    col_matches = [(r, c) for r, c in source_coords if c == mc]
    if len(row_matches) == 1 and len(col_matches) == 0:
        return "row", row_matches[0]
    if len(col_matches) == 1 and len(row_matches) == 0:
        return "col", col_matches[0]
    return None


def _reflect_coords(source_coords, axis_sum, flip_kind):
    if flip_kind == "horizontal":
        return tuple(sorted((r, int(axis_sum - c)) for r, c in source_coords))
    return tuple(sorted((int(axis_sum - r), c) for r, c in source_coords))


def _analyze_scene(grid):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _find_components_of_color(g, SOURCE_COLOR)
    assert 3 <= len(objects) <= 5

    colors = [int(x) for x in np.unique(g) if int(x) not in (BACKGROUND, SOURCE_COLOR)]
    marker_positions = {}
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (1, 2), "every non-green foreground color must be a singleton marker"
        marker_positions[color] = tuple(map(int, coords[0]))
    assert len(marker_positions) == len(objects)

    object_matches = {}
    marker_matches = {}
    for obj_index, coords in enumerate(objects):
        for marker_color, marker in marker_positions.items():
            relation = _marker_relation(coords, marker)
            if relation is None:
                continue
            assert obj_index not in object_matches, "each source object must match exactly one marker"
            assert marker_color not in marker_matches, "each marker must match exactly one source object"
            object_matches[obj_index] = (marker_color, marker, relation)
            marker_matches[marker_color] = obj_index

    assert len(object_matches) == len(objects)
    assert len(marker_matches) == len(marker_positions)

    info = []
    for obj_index, coords in enumerate(objects):
        marker_color, marker, (alignment, anchor) = object_matches[obj_index]
        bends = _count_bends(_normalize(coords))
        assert bends in RECOLOR_BY_BENDS
        flip_kind = "horizontal" if bends % 2 == 1 else "vertical"
        axis_sum = (anchor[1] + marker[1]) if alignment == "row" else (anchor[0] + marker[0])
        copy_coords = _reflect_coords(coords, axis_sum, flip_kind)
        info.append(
            {
                "source": coords,
                "marker_color": int(marker_color),
                "marker": tuple(marker),
                "alignment": alignment,
                "anchor": tuple(anchor),
                "bends": int(bends),
                "flip_kind": flip_kind,
                "copy": copy_coords,
            }
        )

    info.sort(key=lambda item: _bbox(item["source"]))
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each green bent object, then mirror it in the bend-parity direction using the marker-defined axis value.

    Input structure:
    - A black background contains 3 to 5 small green 4-connected objects, each with 1, 2, or 3 L-shaped 2x2 bends.
    - Each green object is paired with exactly one isolated non-black, non-green marker pixel of a unique color.
    - Every marker shares exactly one row or exactly one column with its paired source object, and source objects are separated from one another.

    Rule:
    - Count the number of 2x2 windows inside each green object's bbox that contain exactly three object pixels; this is the bend count.
    - Recolor the original object in place: 1 bend -> blue, 2 bends -> magenta, 3 bends -> red.
    - Find the paired marker and its uniquely aligned source pixel. If the marker is row-aligned, use the source and marker columns to define the axis value; if it is column-aligned, use the source and marker rows.
    - Odd bend counts perform a horizontal flip (left-right reflection), while even bend counts perform a vertical flip (top-bottom reflection). The copied shape is painted in the marker's color.
    - Keep the recolored original objects and keep all marker pixels.

    Key insight:
    - One topological property, the bend count, drives two outputs at once: it chooses the original recolor and also chooses whether the mirrored copy moves horizontally or vertically, while the marker alignment contributes only the axis coordinate.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    scene = _analyze_scene(g)

    for item in scene:
        recolor = RECOLOR_BY_BENDS[item["bends"]]
        for r, c in item["source"]:
            out[r, c] = recolor

    height, width = g.shape
    for item in scene:
        marker_color = item["marker_color"]
        for r, c in item["copy"]:
            assert 0 <= r < height and 0 <= c < width, "mirrored copy leaves the grid"
            assert int(out[r, c]) in (BACKGROUND, marker_color), "mirrored copy conflicts with existing content"
        for r, c in item["copy"]:
            out[r, c] = marker_color

    return out


def _translate(coords, dr, dc):
    return tuple(sorted((r + dr, c + dc) for r, c in coords))


def _chebyshev_too_close(cells_a, cells_b, min_dist):
    for ra, ca in cells_a:
        for rb, cb in cells_b:
            if max(abs(ra - rb), abs(ca - cb)) < min_dist:
                return True
    return False


def _cell_too_close(cell, cells, min_dist):
    return any(max(abs(cell[0] - r), abs(cell[1] - c)) < min_dist for r, c in cells)


def _try_place_object(height, width, bends, alignment, groups):
    variants = [shape for shape in SHAPE_LIBRARY[bends] if (alignment == "row" and _row_anchor_candidates(shape)) or (alignment == "col" and _col_anchor_candidates(shape))]
    if not variants:
        return None

    existing_input = {cell for group in groups for cell in group["source"]} | {group["marker"] for group in groups}
    existing_output = existing_input | {cell for group in groups for cell in group["copy"]}

    for _ in range(OBJECT_PLACEMENT_RETRIES):
        shape = random.choice(variants)
        anchors = _row_anchor_candidates(shape) if alignment == "row" else _col_anchor_candidates(shape)
        local_anchor = random.choice(anchors)
        h, w = _shape_dims(shape)

        sr = random.randint(0, height - h)
        sc = random.randint(0, width - w)
        source = _translate(shape, sr, sc)

        if _chebyshev_too_close(source, existing_output, 2):
            continue

        flip_kind = "horizontal" if bends % 2 == 1 else "vertical"
        gap = random.randint(*PLACEMENT_GAP_RANGE)
        side = random.choice((-1, 1))
        if flip_kind == "horizontal":
            tc = sc + side * (w + gap)
            tr = sr
            if not (0 <= tc <= width - w):
                continue
            axis_sum = sc + tc + w - 1
        else:
            tr = sr + side * (h + gap)
            tc = sc
            if not (0 <= tr <= height - h):
                continue
            axis_sum = sr + tr + h - 1

        if alignment == "row":
            marker = (sr + local_anchor[0], axis_sum - (sc + local_anchor[1]))
        else:
            marker = (axis_sum - (sr + local_anchor[0]), sc + local_anchor[1])

        mr, mc = marker
        if not (0 <= mr < height and 0 <= mc < width):
            continue

        relation = _marker_relation(source, marker)
        if relation != (alignment, (sr + local_anchor[0], sc + local_anchor[1])):
            continue

        if _cell_too_close(marker, source, 2):
            continue
        if _cell_too_close(marker, existing_output, 2):
            continue

        copy_coords = _reflect_coords(source, axis_sum, flip_kind)
        if any(not (0 <= r < height and 0 <= c < width) for r, c in copy_coords):
            continue

        natural_case = (flip_kind == "horizontal" and alignment == "row") or (flip_kind == "vertical" and alignment == "col")
        if natural_case != (marker in set(copy_coords)):
            continue

        copy_without_self_marker = [cell for cell in copy_coords if cell != marker]
        if _chebyshev_too_close(copy_without_self_marker, source, 2):
            continue
        if _chebyshev_too_close(copy_without_self_marker, existing_output, 2):
            continue
        if marker not in copy_coords and _cell_too_close(marker, copy_without_self_marker, 2):
            continue

        if any(_marker_relation(group["source"], marker) is not None for group in groups):
            continue
        if any(_marker_relation(source, group["marker"]) is not None for group in groups):
            continue

        return {
            "source": source,
            "marker": marker,
            "bends": bends,
            "alignment": alignment,
            "copy": copy_coords,
        }

    return None


def _alignment_plan(bend_counts):
    alignments = [random.choice(("row", "col")) for _ in bend_counts]
    if len(set(alignments)) == 1:
        alignments[random.randrange(len(alignments))] = "row" if alignments[0] == "col" else "col"

    def is_mismatch(bends, alignment):
        return (bends % 2 == 1 and alignment == "col") or (bends % 2 == 0 and alignment == "row")

    if not any(is_mismatch(b, a) for b, a in zip(bend_counts, alignments)):
        idx = random.randrange(len(bend_counts))
        alignments[idx] = "col" if bend_counts[idx] % 2 == 1 else "row"

    if "row" not in alignments:
        idx = random.randrange(len(alignments))
        alignments[idx] = "row"
    if "col" not in alignments:
        idx = random.randrange(len(alignments))
        alignments[idx] = "col"

    if not any(is_mismatch(b, a) for b, a in zip(bend_counts, alignments)):
        for idx, bends in enumerate(bend_counts):
            alignments[idx] = "col" if bends % 2 == 1 else "row"
            break

    return alignments


def _instance_is_valid(grid):
    scene = _analyze_scene(grid)
    bends = sorted(item["bends"] for item in scene)
    if bends[:3] != [1, 2, 3]:
        return False
    alignments = {item["alignment"] for item in scene}
    if alignments != {"row", "col"}:
        return False
    if not any(
        (item["flip_kind"] == "horizontal" and item["alignment"] == "col")
        or (item["flip_kind"] == "vertical" and item["alignment"] == "row")
        for item in scene
    ):
        return False
    out = main(grid)
    if out.shape != grid.shape:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _generate_attempt():
    height = random.randint(*GRID_ROWS_RANGE)
    width = random.randint(*GRID_COLS_RANGE)
    object_count = random.randint(*OBJECT_COUNT_RANGE)

    bend_counts = [1, 2, 3]
    while len(bend_counts) < object_count:
        bend_counts.append(random.choice((1, 2, 3)))
    random.shuffle(bend_counts)
    alignments = _alignment_plan(bend_counts)

    groups = []
    for bends, alignment in zip(bend_counts, alignments):
        group = _try_place_object(height, width, bends, alignment, groups)
        if group is None:
            return None
        groups.append(group)

    random.shuffle(groups)
    marker_colors = random.sample(list(MARKER_COLORS), len(groups))
    grid = np.zeros((height, width), dtype=int)
    for group, marker_color in zip(groups, marker_colors):
        for r, c in group["source"]:
            grid[r, c] = SOURCE_COLOR
        mr, mc = group["marker"]
        grid[mr, mc] = marker_color

    if not _instance_is_valid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bend-mirror ARC input")
