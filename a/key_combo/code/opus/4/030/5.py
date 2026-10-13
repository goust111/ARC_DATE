from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

TOOLBOX_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(TOOLBOX_DIR) not in sys.path:
    sys.path.append(str(TOOLBOX_DIR))

from arc_utils import Color, components_mask, draw_line, in_bounds, neighbors4, paste


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SMALL_MAX = 6
MEDIUM_MAX = 15
COMPACT_DELTAS = (2, 3)
STRETCH_DELTAS = (2, 7)
GEN_MAX_ATTEMPTS = 400

TOPOLOGIES = [
    {
        "name": "vertical_split",
        "segments": [
            ((0, 0), (0, 1)),
            ((0, 1), (0, 2)),
            ((1, 0), (1, 1)),
            ((1, 1), (1, 2)),
            ((0, 0), (1, 0)),
            ((0, 1), (1, 1)),
            ((0, 2), (1, 2)),
        ],
        "regions": [(0, 1, 0, 1), (0, 1, 1, 2)],
        "equal_pair": (0, 1),
        "compact_equal_axis": "cols",
    },
    {
        "name": "horizontal_split",
        "segments": [
            ((0, 0), (0, 1)),
            ((1, 0), (1, 1)),
            ((2, 0), (2, 1)),
            ((0, 0), (1, 0)),
            ((1, 0), (2, 0)),
            ((0, 1), (1, 1)),
            ((1, 1), (2, 1)),
        ],
        "regions": [(0, 1, 0, 1), (1, 2, 0, 1)],
        "equal_pair": (0, 1),
        "compact_equal_axis": "rows",
    },
    {
        "name": "left_branch",
        "segments": [
            ((0, 0), (0, 1)),
            ((0, 1), (0, 2)),
            ((2, 0), (2, 1)),
            ((2, 1), (2, 2)),
            ((0, 0), (1, 0)),
            ((1, 0), (2, 0)),
            ((0, 1), (1, 1)),
            ((1, 1), (2, 1)),
            ((0, 2), (2, 2)),
            ((1, 0), (1, 1)),
        ],
        "regions": [(0, 1, 0, 1), (1, 2, 0, 1), (0, 2, 1, 2)],
        "equal_pair": (0, 1),
        "compact_equal_axis": "rows",
    },
    {
        "name": "right_branch",
        "segments": [
            ((0, 0), (0, 1)),
            ((0, 1), (0, 2)),
            ((2, 0), (2, 1)),
            ((2, 1), (2, 2)),
            ((0, 0), (2, 0)),
            ((0, 1), (1, 1)),
            ((1, 1), (2, 1)),
            ((0, 2), (1, 2)),
            ((1, 2), (2, 2)),
            ((1, 1), (1, 2)),
        ],
        "regions": [(0, 2, 0, 1), (0, 1, 1, 2), (1, 2, 1, 2)],
        "equal_pair": (1, 2),
        "compact_equal_axis": "rows",
    },
    {
        "name": "top_branch",
        "segments": [
            ((0, 0), (0, 1)),
            ((0, 1), (0, 2)),
            ((1, 0), (1, 1)),
            ((1, 1), (1, 2)),
            ((2, 0), (2, 2)),
            ((0, 0), (1, 0)),
            ((1, 0), (2, 0)),
            ((0, 1), (1, 1)),
            ((0, 2), (1, 2)),
            ((1, 2), (2, 2)),
        ],
        "regions": [(0, 1, 0, 1), (0, 1, 1, 2), (1, 2, 0, 2)],
        "equal_pair": (0, 1),
        "compact_equal_axis": "cols",
    },
    {
        "name": "bottom_branch",
        "segments": [
            ((0, 0), (0, 2)),
            ((1, 0), (1, 1)),
            ((1, 1), (1, 2)),
            ((2, 0), (2, 1)),
            ((2, 1), (2, 2)),
            ((0, 0), (1, 0)),
            ((1, 0), (2, 0)),
            ((1, 1), (2, 1)),
            ((0, 2), (1, 2)),
            ((1, 2), (2, 2)),
        ],
        "regions": [(0, 1, 0, 2), (1, 2, 0, 1), (1, 2, 1, 2)],
        "equal_pair": (1, 2),
        "compact_equal_axis": "cols",
    },
]


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()) == 0
        or int(cols.min()) == 0
        or int(rows.max()) == shape[0] - 1
        or int(cols.max()) == shape[1] - 1
    )


def _fill_color(area: int) -> int:
    if area <= SMALL_MAX:
        return Color.BLUE
    if area <= MEDIUM_MAX:
        return Color.GREEN
    return Color.ORANGE


def _area_bin(area: int) -> int:
    if area <= SMALL_MAX:
        return 0
    if area <= MEDIUM_MAX:
        return 1
    return 2


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sep_cols = np.where(np.all(grid == Color.GREY, axis=0))[0]
    assert sep_cols.size > 0, "expected at least one full grey separator column"
    assert np.array_equal(sep_cols, np.arange(sep_cols[0], sep_cols[-1] + 1))
    left = grid[:, : int(sep_cols[0])]
    right = grid[:, int(sep_cols[-1]) + 1 :]
    assert left.size > 0 and right.size > 0
    return left, right


def _is_vertex(mask: np.ndarray, r: int, c: int) -> bool:
    nbrs = [
        (nr, nc)
        for nr, nc in neighbors4(r, c)
        if in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc])
    ]
    if len(nbrs) != 2:
        return True
    (r1, c1), (r2, c2) = nbrs
    return r1 != r2 and c1 != c2


def _extract_vertices(panel: np.ndarray) -> list[tuple[int, int]]:
    mask = panel == Color.RED
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a red prototype"
    vertices = [
        (int(r), int(c))
        for r, c in coords
        if _is_vertex(mask, int(r), int(c))
    ]
    assert len(vertices) >= 4, "expected multiple graph vertices"
    return sorted(vertices)


def _extract_segments(panel: np.ndarray, vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    mask = panel == Color.RED
    vertex_set = set(vertices)
    segments: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for vr, vc in vertices:
        for dr, dc in DIRS:
            nr, nc = vr + dr, vc + dc
            if not in_bounds(mask.shape, nr, nc) or not bool(mask[nr, nc]):
                continue
            while True:
                if (nr, nc) in vertex_set:
                    a = (vr, vc)
                    b = (nr, nc)
                    segments.add(tuple(sorted((a, b))))
                    break
                nr += dr
                nc += dc
                assert in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]), "broken prototype segment"

    assert segments, "expected at least one segment"
    return sorted(segments)


def _mapped_vertices(
    prototype_vertices: list[tuple[int, int]],
    anchor_panel: np.ndarray,
) -> tuple[dict[tuple[int, int], tuple[int, int]], list[tuple[int, int]]]:
    anchors = [tuple(map(int, coord)) for coord in np.argwhere(anchor_panel == Color.YELLOW)]
    assert len(anchors) == len(prototype_vertices), "anchor count must match prototype vertices"

    proto_rows = sorted({r for r, _ in prototype_vertices})
    proto_cols = sorted({c for _, c in prototype_vertices})
    anchor_rows = sorted({r for r, _ in anchors})
    anchor_cols = sorted({c for _, c in anchors})
    assert len(proto_rows) == len(anchor_rows)
    assert len(proto_cols) == len(anchor_cols)

    row_map = dict(zip(proto_rows, anchor_rows))
    col_map = dict(zip(proto_cols, anchor_cols))
    mapped = {
        vertex: (row_map[vertex[0]], col_map[vertex[1]])
        for vertex in prototype_vertices
    }
    assert set(mapped.values()) == set(anchors), "sorted row/col mapping must recover the anchor set"
    return mapped, anchors


def _reconstruct_output(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    vertices = _extract_vertices(left)
    segments = _extract_segments(left, vertices)
    mapped, anchors = _mapped_vertices(vertices, right)

    out = right.copy()
    for start, end in segments:
        r0, c0 = mapped[start]
        r1, c1 = mapped[end]
        assert r0 == r1 or c0 == c1
        draw_line(out, r0, c0, end_r=r1, end_c=c1, color=Color.RED)

    for r, c in anchors:
        out[r, c] = Color.YELLOW

    filled_any = False
    for coords in components_mask(out == Color.BLACK, connectivity=4):
        if _touches_border(coords, out.shape):
            continue
        area = int(coords.shape[0])
        out[coords[:, 0], coords[:, 1]] = _fill_color(area)
        filled_any = True

    assert filled_any, "expected at least one enclosed region after stretching"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Stretch the compact red wireframe to the yellow anchor scaffold and fill the emergent enclosed regions by area.

    Input structure:
    - A full grey column block separates a left prototype panel from a right anchor panel.
    - The left panel is black except for a one-pixel-wide red rectilinear wireframe whose vertices are endpoints, corners, and junctions.
    - The right panel is black except for yellow singleton anchor pixels arranged as the stretched row/column levels of those prototype vertices.

    Rule:
    - Detect the red prototype vertices and the straight red segments between adjacent vertices.
    - Map sorted prototype row levels to sorted anchor row levels, and sorted prototype column levels to sorted anchor column levels.
    - Redraw every prototype segment as a red straight segment between the mapped anchors on the right panel.
    - Treat the resulting non-black wireframe as walls, keep the yellow anchor pixels visible, and fill each enclosed 4-connected black region by area: 1-6 -> blue, 7-15 -> green, 16+ -> orange.

    Key insight:
    - The prototype only fixes which regions will exist; the anchor spacing changes their actual stretched areas, so the fill colors can only be determined after rebuilding the large wireframe.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    left, right = _split_panels(g)
    return _reconstruct_output(left, right)


def _levels_from_deltas(deltas: list[int], offset: int) -> list[int]:
    levels = [int(offset)]
    cur = int(offset)
    for delta in deltas:
        cur += int(delta)
        levels.append(cur)
    return levels


def _region_area(row_deltas: list[int], col_deltas: list[int], region: tuple[int, int, int, int]) -> int:
    ra, rb, ca, cb = region
    height = sum(row_deltas[ra:rb]) - 1
    width = sum(col_deltas[ca:cb]) - 1
    return int(height * width)


def _spec_vertices(spec: dict) -> list[tuple[int, int]]:
    vertices = {
        tuple(map(int, start))
        for start, _ in spec["segments"]
    } | {
        tuple(map(int, end))
        for _, end in spec["segments"]
    }
    return sorted(vertices)


def _render_wire(
    shape: tuple[int, int],
    row_levels: list[int],
    col_levels: list[int],
    segments: list[tuple[tuple[int, int], tuple[int, int]]],
    color: int,
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for (ra, ca), (rb, cb) in segments:
        draw_line(
            grid,
            row_levels[ra],
            col_levels[ca],
            end_r=row_levels[rb],
            end_c=col_levels[cb],
            color=color,
        )
    return grid


def _sample_compact_deltas(spec: dict) -> tuple[list[int], list[int]]:
    row_count = max(r for segment in spec["segments"] for r, _ in segment) + 1
    col_count = max(c for segment in spec["segments"] for _, c in segment) + 1
    row_deltas = [random.choice(COMPACT_DELTAS) for _ in range(row_count - 1)]
    col_deltas = [random.choice(COMPACT_DELTAS) for _ in range(col_count - 1)]
    if spec["compact_equal_axis"] == "rows":
        value = random.choice(COMPACT_DELTAS)
        row_deltas = [value] * (row_count - 1)
    else:
        value = random.choice(COMPACT_DELTAS)
        col_deltas = [value] * (col_count - 1)
    return row_deltas, col_deltas


def _sample_stretch_deltas(spec: dict) -> tuple[list[int], list[int]] | None:
    row_count = max(r for segment in spec["segments"] for r, _ in segment) + 1
    col_count = max(c for segment in spec["segments"] for _, c in segment) + 1
    eq0, eq1 = spec["equal_pair"]

    for _ in range(300):
        row_deltas = [random.randint(*STRETCH_DELTAS) for _ in range(row_count - 1)]
        col_deltas = [random.randint(*STRETCH_DELTAS) for _ in range(col_count - 1)]
        areas = [_region_area(row_deltas, col_deltas, region) for region in spec["regions"]]
        bins = [_area_bin(area) for area in areas]
        if bins[eq0] == bins[eq1]:
            continue
        if len(spec["regions"]) == 2 and len(set(bins)) == 2:
            return row_deltas, col_deltas
        if len(spec["regions"]) == 3 and set(bins) == {0, 1, 2}:
            return row_deltas, col_deltas
    return None


def _build_input_case() -> np.ndarray | None:
    spec = random.choice(TOPOLOGIES)
    compact_row_deltas, compact_col_deltas = _sample_compact_deltas(spec)
    stretch = _sample_stretch_deltas(spec)
    if stretch is None:
        return None
    stretch_row_deltas, stretch_col_deltas = stretch

    compact_areas = [_region_area(compact_row_deltas, compact_col_deltas, region) for region in spec["regions"]]
    stretch_areas = [_region_area(stretch_row_deltas, stretch_col_deltas, region) for region in spec["regions"]]
    eq0, eq1 = spec["equal_pair"]
    if compact_areas[eq0] != compact_areas[eq1]:
        return None
    if max(compact_areas) > 4:
        return None

    vertices = _spec_vertices(spec)
    right_top = random.randint(1, 2)
    right_left = random.randint(1, 2)
    right_bottom = random.randint(1, 2)
    right_right = random.randint(1, 2)
    right_rows = _levels_from_deltas(stretch_row_deltas, right_top)
    right_cols = _levels_from_deltas(stretch_col_deltas, right_left)
    right_h = right_rows[-1] + 1 + right_bottom
    right_w = right_cols[-1] + 1 + right_right
    if max(right_h, right_w) > 18:
        return None

    compact_span_h = sum(compact_row_deltas) + 1
    compact_span_w = sum(compact_col_deltas) + 1
    full_h = right_h
    if compact_span_h + 2 > full_h:
        return None
    left_w = compact_span_w + random.randint(2, 4)
    if left_w > 11:
        return None
    separator_w = random.randint(1, 3)
    if left_w + separator_w + right_w > 30:
        return None

    left = np.zeros((full_h, left_w), dtype=int)
    top_room = full_h - compact_span_h - 1
    left_top = random.randint(1, top_room)
    left_left = random.randint(1, left_w - compact_span_w - 1)
    compact_rows = _levels_from_deltas(compact_row_deltas, left_top)
    compact_cols = _levels_from_deltas(compact_col_deltas, left_left)
    left_patch = _render_wire(left.shape, compact_rows, compact_cols, spec["segments"], Color.RED)
    paste(left, left_patch, 0, 0, transparent=Color.BLACK)

    right = np.zeros((full_h, right_w), dtype=int)
    for vr, vc in vertices:
        right[right_rows[vr], right_cols[vc]] = Color.YELLOW

    grid = np.zeros((full_h, left_w + separator_w + right_w), dtype=int)
    paste(grid, left, 0, 0)
    grid[:, left_w : left_w + separator_w] = Color.GREY
    paste(grid, right, 0, left_w + separator_w)

    out = main(grid)
    if out.shape != right.shape:
        return None
    fill_counts = {
        Color.BLUE: int(np.count_nonzero(out == Color.BLUE)),
        Color.GREEN: int(np.count_nonzero(out == Color.GREEN)),
        Color.ORANGE: int(np.count_nonzero(out == Color.ORANGE)),
    }
    expected_fill_counts = {
        Color.BLUE: sum(area for area in stretch_areas if _fill_color(area) == Color.BLUE),
        Color.GREEN: sum(area for area in stretch_areas if _fill_color(area) == Color.GREEN),
        Color.ORANGE: sum(area for area in stretch_areas if _fill_color(area) == Color.ORANGE),
    }
    if fill_counts != expected_fill_counts:
        return None
    if len(spec["regions"]) == 3 and set(_area_bin(area) for area in stretch_areas) != {0, 1, 2}:
        return None
    if len(spec["regions"]) == 2 and len(set(_area_bin(area) for area in stretch_areas)) != 2:
        return None
    if _area_bin(stretch_areas[eq0]) == _area_bin(stretch_areas[eq1]):
        return None
    if not np.array_equal(out, _reconstruct_output(left, right)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _build_input_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stretch-and-fill ARC input")
