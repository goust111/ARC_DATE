from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    utils_path = str(_ARC_UTILS_DIR)
    if utils_path not in sys.path:
        sys.path.append(utils_path)

try:
    from arc_utils import Color, bbox, draw_line
except Exception:  # pragma: no cover - fallback only if shared helpers are unavailable.
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        ORANGE = 7

    def bbox(mask: np.ndarray):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def draw_line(
        grid: np.ndarray,
        r: int,
        c: int,
        *,
        end_r: int | None = None,
        end_c: int | None = None,
        color: int,
        **_kwargs,
    ):
        rr = int(r)
        cc = int(c)
        er = int(end_r)
        ec = int(end_c)
        dr = 0 if er == rr else (1 if er > rr else -1)
        dc = 0 if ec == cc else (1 if ec > cc else -1)
        steps = max(abs(er - rr), abs(ec - cc)) + 1
        for i in range(steps):
            grid[rr + i * dr, cc + i * dc] = int(color)
        return er, ec


BG = int(Color.BLACK)
BLUE = int(Color.BLUE)
RED = int(Color.RED)
GREEN = int(Color.GREEN)
YELLOW = int(Color.YELLOW)
GREY = int(Color.GREY)
ORANGE = int(Color.ORANGE)
CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))
LEFT_PANEL_W = 8
SAMPLE_ROW = 8
SAMPLE_COL = 2
SAMPLE_CYCLE = (BLUE, GREEN, ORANGE)
MIN_STRETCHED_DELTA = 4
MAX_GEN_TRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in CARDINALS:
        rr = int(r + dr)
        cc = int(c + dc)
        if _in_bounds(mask.shape, rr, cc) and bool(mask[rr, cc]):
            out.append((rr, cc))
    return out


def _full_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 1, "expected exactly one full grey separator column"
    return int(cols[0])


def _sample_cycle_from_left(left: np.ndarray) -> tuple[int, ...]:
    sample_mask = (left != BG) & (left != GREY) & (left != RED)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(sample_mask)]
    assert len(coords) == 3, "expected exactly three sample pixels"
    rows = {r for r, _c in coords}
    cols = {c for _r, c in coords}
    if len(rows) == 1:
        coords.sort(key=lambda rc: rc[1])
    else:
        assert len(cols) == 1, "sample must be a straight 3-pixel line"
        coords.sort(key=lambda rc: rc[0])
    return tuple(int(left[r, c]) for r, c in coords)


def _prototype_vertices(red_mask: np.ndarray) -> list[tuple[int, int]]:
    pts = [tuple(map(int, rc)) for rc in np.argwhere(red_mask)]
    assert pts, "prototype must contain red pixels"
    vertices: list[tuple[int, int]] = []
    for r, c in pts:
        neigh = _neighbors4(red_mask, r, c)
        degree = len(neigh)
        if degree != 2:
            vertices.append((r, c))
            continue
        (r1, c1), (r2, c2) = neigh
        if r1 == r2 or c1 == c2:
            continue
        vertices.append((r, c))
    vertices = sorted(set(vertices))
    assert len(vertices) >= 2, "prototype must expose at least two vertices"
    return vertices


def _prototype_edges(red_mask: np.ndarray, vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    vertex_set = set(vertices)
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for vr, vc in vertices:
        for dr, dc in CARDINALS:
            nr = int(vr + dr)
            nc = int(vc + dc)
            if not _in_bounds(red_mask.shape, nr, nc) or not bool(red_mask[nr, nc]):
                continue
            prev = (vr, vc)
            cur = (nr, nc)
            while True:
                if cur in vertex_set:
                    if cur != (vr, vc):
                        edge = tuple(sorted(((vr, vc), cur)))
                        edges.add(edge)
                    break
                nxts = [pt for pt in _neighbors4(red_mask, cur[0], cur[1]) if pt != prev]
                assert len(nxts) == 1, "prototype lines must be simple between visible vertices"
                prev, cur = cur, nxts[0]
    out = sorted(edges)
    assert out, "prototype must contain at least one segment"
    return out


def _parse_task(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sep = _full_separator_col(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]

    red_mask = left == RED
    assert bbox(red_mask) is not None, "missing prototype"
    vertices = _prototype_vertices(red_mask)
    edges = _prototype_edges(red_mask, vertices)

    anchors = [tuple(map(int, rc)) for rc in np.argwhere(right == YELLOW)]
    assert len(anchors) == len(vertices), "anchor count must match prototype vertex count"
    cycle = _sample_cycle_from_left(left)

    proto_rows = sorted({r for r, _c in vertices})
    proto_cols = sorted({c for _r, c in vertices})
    anchor_rows = sorted({r for r, _c in anchors})
    anchor_cols = sorted({c for _r, c in anchors})
    assert len(proto_rows) == len(anchor_rows), "row-coordinate cardinalities must match"
    assert len(proto_cols) == len(anchor_cols), "col-coordinate cardinalities must match"

    row_map = dict(zip(proto_rows, anchor_rows))
    col_map = dict(zip(proto_cols, anchor_cols))
    mapped_vertices = {(r, c): (row_map[r], col_map[c]) for r, c in vertices}
    assert set(mapped_vertices.values()) == set(anchors), "anchors must equal the stretched prototype vertices"

    return right.shape, cycle, mapped_vertices, edges


def _render_output(
    shape: tuple[int, int],
    cycle: tuple[int, ...],
    mapped_vertices: dict[tuple[int, int], tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for r, c in mapped_vertices.values():
        out[r, c] = YELLOW

    for a, b in edges:
        r0, c0 = mapped_vertices[a]
        r1, c1 = mapped_vertices[b]
        draw_line(out, r0, c0, end_r=r1, end_c=c1, color=RED)
        if r0 == r1:
            lo, hi = sorted((c0, c1))
            for i, c in enumerate(range(lo + 1, hi)):
                out[r0, c] = int(cycle[i % len(cycle)])
        else:
            assert c0 == c1, "prototype edges must stay axis-aligned after stretching"
            lo, hi = sorted((r0, r1))
            for i, r in enumerate(range(lo + 1, hi)):
                out[r, c0] = int(cycle[i % len(cycle)])

    for r, c in mapped_vertices.values():
        out[r, c] = YELLOW
    return out


def _shortcut_edges_from_anchors(anchors: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    by_row: dict[int, list[int]] = defaultdict(list)
    by_col: dict[int, list[int]] = defaultdict(list)
    for r, c in anchors:
        by_row[int(r)].append(int(c))
        by_col[int(c)].append(int(r))

    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for r, cols in by_row.items():
        cols = sorted(cols)
        for c0, c1 in zip(cols, cols[1:]):
            edges.add(tuple(sorted(((int(r), int(c0)), (int(r), int(c1))))))
    for c, rows in by_col.items():
        rows = sorted(rows)
        for r0, r1 in zip(rows, rows[1:]):
            edges.add(tuple(sorted(((int(r0), int(c)), (int(r1), int(c))))))
    return sorted(edges)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch the compact prototype onto the sparse anchors and decorate each stretched segment with a 3-color cycle.

    Input structure:
    - A full grey separator column splits the grid into a left prototype panel and a right anchor panel.
    - The left panel contains one connected red prototype made of horizontal/vertical segments and one 3-pixel sample line.
    - The right panel contains sparse yellow anchor pixels that correspond to the prototype's visible vertices under independent row/column stretching.

    Rule:
    - Detect the prototype's visible vertices: endpoints, corners, and junctions of the red path.
    - Sort prototype vertex rows and anchor rows, and map them one-to-one; do the same independently for columns.
    - For every prototype segment between adjacent visible vertices, draw the stretched segment between the mapped anchors on the right panel.
    - Keep anchor pixels yellow, and recolor each horizontal or vertical segment interior with the sample's 3-color cycle in left-to-right or top-to-bottom order.

    Key insight:
    - The right-panel graph is not determined by anchor geometry alone; the prototype specifies which aligned anchors are connected, while the independent row/column stretch determines each segment length and therefore the visible truncation of the repeating decoration.
    """
    shape, cycle, mapped_vertices, edges = _parse_task(grid)
    return _render_output(shape, cycle, mapped_vertices, edges)


def _transform_point(pt: tuple[int, int], variant: int, size: int = 5) -> tuple[int, int]:
    r, c = map(int, pt)
    last = int(size - 1)
    if variant == 0:
        return r, c
    if variant == 1:
        return r, last - c
    if variant == 2:
        return last - r, c
    if variant == 3:
        return last - r, last - c
    if variant == 4:
        return c, r
    if variant == 5:
        return last - c, r
    if variant == 6:
        return c, last - r
    if variant == 7:
        return last - c, last - r
    raise ValueError("unknown transform variant")


_BASE_SHAPES = (
    ((0, 0), (0, 4), (4, 4), (4, 0)),
    ((0, 0), (0, 4), (2, 4), (2, 0), (4, 0)),
    ((0, 0), (0, 4), (4, 4), (4, 0), (2, 0)),
)


def _choose_template() -> list[tuple[int, int]]:
    base = random.choice(_BASE_SHAPES)
    variant = random.randint(0, 7)
    return [_transform_point(pt, variant) for pt in base]


def _pick_spaced_positions(count: int, size: int, min_gap: int = MIN_STRETCHED_DELTA) -> list[int]:
    assert size >= 3
    choices = list(range(1, size - 1))
    for _ in range(300):
        picked = sorted(random.sample(choices, count))
        if all(b - a >= min_gap for a, b in zip(picked, picked[1:])):
            return picked
    raise ValueError("failed to sample spaced positions")


def _render_prototype(vertices: list[tuple[int, int]], h: int, w: int) -> np.ndarray:
    proto = np.full((h, w), BG, dtype=int)
    for (r0, c0), (r1, c1) in zip(vertices, vertices[1:]):
        draw_line(proto, r0, c0, end_r=r1, end_c=c1, color=RED)
    return proto


def _mapped_vertices_from_template(
    template_vertices: list[tuple[int, int]],
    anchor_rows: list[int],
    anchor_cols: list[int],
    row_offset: int,
    col_offset: int,
) -> tuple[dict[tuple[int, int], tuple[int, int]], list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    proto_vertices = [(int(r + row_offset), int(c + col_offset)) for r, c in template_vertices]
    proto_rows = sorted({r for r, _c in proto_vertices})
    proto_cols = sorted({c for _r, c in proto_vertices})
    row_map = dict(zip(proto_rows, anchor_rows))
    col_map = dict(zip(proto_cols, anchor_cols))
    mapped = {(r, c): (row_map[r], col_map[c]) for r, c in proto_vertices}
    edges = [tuple(sorted((proto_vertices[i], proto_vertices[i + 1]))) for i in range(len(proto_vertices) - 1)]
    return mapped, proto_vertices, edges


def _segment_lengths(mapped: dict[tuple[int, int], tuple[int, int]], edges) -> list[int]:
    lengths: list[int] = []
    for a, b in edges:
        r0, c0 = mapped[a]
        r1, c1 = mapped[b]
        lengths.append(abs(r1 - r0) + abs(c1 - c0) - 1)
    return lengths


def _build_example() -> tuple[np.ndarray, np.ndarray]:
    template = _choose_template()
    proto_h = max(r for r, _c in template) + 1
    proto_w = max(c for _r, c in template) + 1
    row_offset, col_offset = 1, 1

    row_count = len({r for r, _c in template})
    col_count = len({c for _r, c in template})
    min_h = max(SAMPLE_ROW + 2, row_count * MIN_STRETCHED_DELTA + 3)
    min_right_w = col_count * MIN_STRETCHED_DELTA + 3
    grid_h = random.randint(min_h, min(26, min_h + 6))
    right_w = random.randint(min_right_w, min(20, min_right_w + 6))

    anchor_rows = _pick_spaced_positions(row_count, grid_h)
    anchor_cols = _pick_spaced_positions(col_count, right_w)
    mapped, proto_vertices, edges = _mapped_vertices_from_template(
        template, anchor_rows, anchor_cols, row_offset=row_offset, col_offset=col_offset
    )

    widths = _segment_lengths(mapped, edges)
    if min(widths) < 3:
        raise ValueError("segment interior too short")

    right = np.full((grid_h, right_w), BG, dtype=int)
    for r, c in mapped.values():
        right[r, c] = YELLOW

    intended = _render_output(right.shape, SAMPLE_CYCLE, mapped, edges)
    shortcut = _render_output(
        right.shape,
        SAMPLE_CYCLE,
        {anchor: anchor for anchor in sorted(mapped.values())},
        _shortcut_edges_from_anchors(sorted(mapped.values())),
    )
    assert not np.array_equal(shortcut, intended), "prototype must matter beyond anchor alignment alone"

    assert len(set(widths)) >= 2 or any(length % 3 for length in widths), "need visible cycle truncation differences"
    assert {BLUE, GREEN, ORANGE}.issubset(set(intended.reshape(-1))), "all sample colors must appear in the output"

    left = np.full((grid_h, LEFT_PANEL_W), BG, dtype=int)
    proto = _render_prototype(template, proto_h, proto_w)
    left[row_offset : row_offset + proto_h, col_offset : col_offset + proto_w] = np.maximum(
        left[row_offset : row_offset + proto_h, col_offset : col_offset + proto_w], proto
    )
    left[SAMPLE_ROW, SAMPLE_COL : SAMPLE_COL + 3] = np.array(SAMPLE_CYCLE, dtype=int)

    grid = np.full((grid_h, LEFT_PANEL_W + 1 + right_w), BG, dtype=int)
    grid[:, LEFT_PANEL_W] = GREY
    grid[:, :LEFT_PANEL_W] = left
    grid[:, LEFT_PANEL_W + 1 :] = right

    out = main(grid)
    assert np.array_equal(out, intended), "solver must reproduce the intended output"
    return grid, intended


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_TRIES):
        try:
            grid, _out = _build_example()
            return grid
        except (AssertionError, ValueError):
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
