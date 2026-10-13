from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BG = 0
PATH = 5
OUT_SINGLE = 2
OUT_DOUBLE = 1
OUT_BLOCK = 3

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
KINDS = ("none", "single", "double", "block")


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _infer_path_color(g: np.ndarray, bg: int) -> int:
    counts = Counter(int(v) for v in g.ravel() if int(v) != bg)
    assert counts, "expected non-background colors"
    best_count = max(counts.values())
    cands = [color for color, count in counts.items() if count == best_count]
    if len(cands) == 1:
        return int(cands[0])

    def score(color: int) -> int:
        mask = g == int(color)
        pts = np.argwhere(mask)
        total = 0
        for r, c in pts:
            deg = 0
            for nr, nc in _neighbors4(int(r), int(c), g.shape[0], g.shape[1]):
                deg += int(mask[nr, nc])
            total += deg
        return total

    return int(max(cands, key=score))


def _classify_marker(coords: list[tuple[int, int]]) -> str:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    cells = set(coords)
    if len(coords) == 1:
        return "single"
    if len(coords) == 2 and h * w == 2 and (h == 1 or w == 1):
        return "double"
    if len(coords) == 4 and h == 2 and w == 2:
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if (rr, cc) not in cells:
                    raise AssertionError("2x2 marker must be solid")
        return "block"
    raise AssertionError("unexpected marker shape")


def _kind_to_color(kind: str) -> int:
    if kind == "single":
        return OUT_SINGLE
    if kind == "double":
        return OUT_DOUBLE
    if kind == "block":
        return OUT_BLOCK
    return PATH


def _extract_segments(path_mask: np.ndarray) -> tuple[list[dict], set[tuple[int, int]]]:
    h, w = path_mask.shape
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(path_mask):
        coord = (int(r), int(c))
        neighbors[coord] = [p for p in _neighbors4(int(r), int(c), h, w) if bool(path_mask[p])]

    nodes: set[tuple[int, int]] = set()
    crosspoints: set[tuple[int, int]] = set()
    for coord, nbrs in neighbors.items():
        dirs = {(rr - coord[0], cc - coord[1]) for rr, cc in nbrs}
        has_h = (0, -1) in dirs or (0, 1) in dirs
        has_v = (-1, 0) in dirs or (1, 0) in dirs
        if len(nbrs) == 4:
            crosspoints.add(coord)
        if len(nbrs) != 2 or (has_h and has_v):
            nodes.add(coord)

    assert nodes, "expected segment endpoints or junctions"

    def edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        return (a, b) if a <= b else (b, a)

    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    segments: list[dict] = []

    for node in sorted(nodes):
        for nbr in neighbors[node]:
            key = edge_key(node, nbr)
            if key in visited_edges:
                continue
            coords = [node]
            prev = node
            cur = nbr
            while True:
                visited_edges.add(edge_key(prev, cur))
                coords.append(cur)
                if cur in nodes:
                    break
                nxts = [p for p in neighbors[cur] if p != prev]
                assert len(nxts) == 1, "non-node path pixels must continue straight"
                prev, cur = cur, nxts[0]
            rows = {r for r, _ in coords}
            cols = {c for _, c in coords}
            if len(rows) == 1:
                orient = "h"
            else:
                assert len(cols) == 1, "segments must be axis-aligned"
                orient = "v"
            segments.append({"coords": coords, "orient": orient})

    return segments, crosspoints


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor path segments by splitting a grey orthogonal network at its crossings and reading adjacent marker shapes.

    Input structure:
    - The grid has a dominant black background plus one thin orthogonal path color.
    - The path network is one pixel wide and is made of horizontal and vertical runs that may cross.
    - Non-path foreground pixels are compact markers placed 4-neighbor adjacent to exactly one path segment.
    - Marker shapes are single pixels, 2-cell dominoes, or solid 2x2 blocks.

    Rule:
    - Identify the path color and find the crossing pixels where the path continues in all four cardinal directions.
    - Split the network into maximal straight segments between endpoints, bends, and crossings.
    - For each segment, inspect its adjacent marker shape: singleton -> recolor that segment red, domino -> blue, 2x2 block -> green.
    - Segments with no adjacent marker stay grey; marker pixels themselves stay unchanged.
    - Crossing pixels take the color of the horizontal segment passing through them.

    Key insight:
    The color cue comes from marker geometry, not marker color, so the solver must first decompose the path into straight segments and only then use local shape adjacency to recolor them.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(Counter(int(v) for v in g.ravel()).most_common(1)[0][0])
    path_color = _infer_path_color(g, bg)

    path_mask = g == int(path_color)
    segments, crosspoints = _extract_segments(path_mask)

    segment_paint: list[list[tuple[int, int]]] = []
    coord_to_segments: dict[tuple[int, int], list[int]] = defaultdict(list)
    for sid, seg in enumerate(segments):
        paint_coords = [coord for coord in seg["coords"] if coord not in crosspoints]
        assert paint_coords, "every segment should have paintable pixels"
        segment_paint.append(paint_coords)
        for coord in paint_coords:
            coord_to_segments[coord].append(int(sid))

    marker_mask = (g != int(bg)) & (g != int(path_color))
    marker_components = _components(marker_mask)
    segment_kind: dict[int, str] = {}
    for comp in marker_components:
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "markers must be monochrome"
        kind = _classify_marker(comp)
        touched: set[int] = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c, h, w):
                for sid in coord_to_segments.get((nr, nc), []):
                    touched.add(int(sid))
        assert len(touched) <= 1, "each marker must identify at most one segment"
        if touched:
            sid = next(iter(touched))
            assert sid not in segment_kind, "each segment may have at most one marker"
            segment_kind[int(sid)] = kind

    out = g.copy()
    segment_color: dict[int, int] = {}
    for sid, seg in enumerate(segments):
        color = _kind_to_color(segment_kind.get(int(sid), "none"))
        segment_color[int(sid)] = int(color)
        for r, c in segment_paint[sid]:
            out[r, c] = int(color)

    horizontal_cross_colors: dict[tuple[int, int], set[int]] = defaultdict(set)
    for sid, seg in enumerate(segments):
        if seg["orient"] != "h":
            continue
        for endpoint in (seg["coords"][0], seg["coords"][-1]):
            if endpoint in crosspoints:
                horizontal_cross_colors[endpoint].add(int(segment_color[int(sid)]))

    for r, c in crosspoints:
        colors = horizontal_cross_colors.get((r, c), set())
        assert colors, "crossing must belong to a horizontal segment"
        assert len(colors) == 1, "crossing color must be horizontally unambiguous"
        out[r, c] = int(next(iter(colors)))

    return out


def _make_axis_positions(line_count: int) -> tuple[int, list[int], int]:
    top = random.randint(1, 3)
    intervals = [random.randint(5, 6) for _ in range(line_count + 1)]
    pos = top
    lines: list[int] = []
    for gap in intervals[:-1]:
        pos += gap
        lines.append(int(pos))
    bottom = int(pos + intervals[-1])
    return int(top), lines, int(bottom)


def _horizontal_candidates(
    kind: str,
    r: int,
    c0: int,
    c1: int,
    cross_cols: set[int],
) -> list[list[tuple[int, int]]]:
    usable = [c for c in range(c0, c1 + 1) if c not in cross_cols]
    candidates: list[list[tuple[int, int]]] = []
    for side in (-1, 1):
        near_r = r + side
        far_r = r + 2 * side
        if kind == "single":
            for c in usable:
                candidates.append([(near_r, c)])
        elif kind == "double":
            for c in usable:
                candidates.append([(near_r, c), (far_r, c)])
            for c in range(c0, c1):
                if c in cross_cols or (c + 1) in cross_cols:
                    continue
                candidates.append([(near_r, c), (near_r, c + 1)])
        elif kind == "block":
            for c in range(c0, c1):
                if c in cross_cols or (c + 1) in cross_cols:
                    continue
                candidates.append([(near_r, c), (near_r, c + 1), (far_r, c), (far_r, c + 1)])
    random.shuffle(candidates)
    return candidates


def _vertical_candidates(
    kind: str,
    c: int,
    r0: int,
    r1: int,
    cross_rows: set[int],
) -> list[list[tuple[int, int]]]:
    usable = [r for r in range(r0, r1 + 1) if r not in cross_rows]
    candidates: list[list[tuple[int, int]]] = []
    for side in (-1, 1):
        near_c = c + side
        far_c = c + 2 * side
        if kind == "single":
            for r in usable:
                candidates.append([(r, near_c)])
        elif kind == "double":
            for r in usable:
                candidates.append([(r, near_c), (r, far_c)])
            for r in range(r0, r1):
                if r in cross_rows or (r + 1) in cross_rows:
                    continue
                candidates.append([(r, near_c), (r + 1, near_c)])
        elif kind == "block":
            for r in range(r0, r1):
                if r in cross_rows or (r + 1) in cross_rows:
                    continue
                candidates.append([(r, near_c), (r + 1, near_c), (r, far_c), (r + 1, far_c)])
    random.shuffle(candidates)
    return candidates


def _marker_cells_ok(
    cells: list[tuple[int, int]],
    grid: np.ndarray,
    path_mask: np.ndarray,
    marker_mask: np.ndarray,
    segment_cells: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    touched = False
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BG or bool(marker_mask[r, c]):
            return False
        for nr, nc in _neighbors4(r, c, h, w):
            if bool(marker_mask[nr, nc]) and (nr, nc) not in cells:
                return False
        adj_paths = {(nr, nc) for nr, nc in _neighbors4(r, c, h, w) if bool(path_mask[nr, nc])}
        if adj_paths - segment_cells:
            return False
        if adj_paths & segment_cells:
            touched = True
    return touched


def _expected_output(
    grid: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int,
    rows: list[int],
    cols: list[int],
    row_types: list[str],
    vertical_types: dict[tuple[int, int], str],
) -> np.ndarray:
    out = grid.copy()
    out[(grid == PATH)] = PATH
    for row_index, r in enumerate(rows):
        out[r, left : right + 1] = int(_kind_to_color(row_types[row_index]))
    row_set = set(rows)
    boundaries = [top] + rows + [bottom]
    for col_index, c in enumerate(cols):
        for interval_index in range(len(boundaries) - 1):
            kind = vertical_types[(col_index, interval_index)]
            color = _kind_to_color(kind)
            r0 = boundaries[interval_index]
            r1 = boundaries[interval_index + 1]
            for rr in range(r0, r1 + 1):
                if rr in row_set:
                    continue
                out[rr, c] = int(color)
    return out


def generate_input() -> np.ndarray:
    for _ in range(500):
        nh = random.randint(2, 3)
        nv = random.randint(2, 3)

        top, rows, bottom = _make_axis_positions(nh)
        left, cols, right = _make_axis_positions(nv)
        h = bottom + random.randint(1, 3) + 1
        w = right + random.randint(1, 3) + 1
        if h > 30 or w > 30:
            continue

        grid = np.full((h, w), BG, dtype=int)
        for r in rows:
            grid[r, left : right + 1] = PATH
        for c in cols:
            grid[top : bottom + 1, c] = PATH

        path_mask = grid == PATH
        cross_cols = set(cols)
        cross_rows = set(rows)
        row_types = [random.choice(KINDS) for _ in rows]
        if all(kind == "none" for kind in row_types):
            row_types[random.randrange(len(row_types))] = random.choice(["single", "double", "block"])
        if all(kind != "none" for kind in row_types):
            row_types[random.randrange(len(row_types))] = "none"

        vertical_types: dict[tuple[int, int], str] = {}
        for col_index in range(len(cols)):
            for interval_index in range(len(rows) + 1):
                vertical_types[(col_index, interval_index)] = random.choice(KINDS)

        all_group_types = list(row_types) + list(vertical_types.values())
        missing = [kind for kind in KINDS if kind not in all_group_types]
        vertical_keys = list(vertical_types.keys())
        random.shuffle(vertical_keys)
        for key, kind in zip(vertical_keys, missing):
            vertical_types[key] = kind
        if "none" not in list(row_types) + list(vertical_types.values()):
            vertical_types[random.choice(vertical_keys)] = "none"

        marker_colors = random.sample([1, 2, 3], 3)
        shape_color = {
            "single": marker_colors[0],
            "double": marker_colors[1],
            "block": marker_colors[2],
        }

        marker_mask = np.zeros_like(grid, dtype=bool)
        placed = True
        boundaries_h = [left] + cols + [right]
        boundaries_v = [top] + rows + [bottom]

        horizontal_segments: list[tuple[int, str, int, int]] = []
        for row_index, r in enumerate(rows):
            kind = row_types[row_index]
            for interval_index in range(len(boundaries_h) - 1):
                horizontal_segments.append((r, kind, boundaries_h[interval_index], boundaries_h[interval_index + 1]))

        vertical_segments: list[tuple[int, str, int, int]] = []
        for col_index, c in enumerate(cols):
            for interval_index in range(len(boundaries_v) - 1):
                kind = vertical_types[(col_index, interval_index)]
                vertical_segments.append((c, kind, boundaries_v[interval_index], boundaries_v[interval_index + 1]))

        random.shuffle(horizontal_segments)
        random.shuffle(vertical_segments)

        for r, kind, c0, c1 in horizontal_segments:
            if kind == "none":
                continue
            segment_cells = {(r, c) for c in range(c0, c1 + 1) if c not in cross_cols}
            candidates = _horizontal_candidates(kind, r, c0, c1, cross_cols)
            choice = None
            for cells in candidates:
                if _marker_cells_ok(cells, grid, path_mask, marker_mask, segment_cells):
                    choice = cells
                    break
            if choice is None:
                placed = False
                break
            for rr, cc in choice:
                grid[rr, cc] = int(shape_color[kind])
                marker_mask[rr, cc] = True
        if not placed:
            continue

        for c, kind, r0, r1 in vertical_segments:
            if kind == "none":
                continue
            segment_cells = {(r, c) for r in range(r0, r1 + 1) if r not in cross_rows}
            candidates = _vertical_candidates(kind, c, r0, r1, cross_rows)
            choice = None
            for cells in candidates:
                if _marker_cells_ok(cells, grid, path_mask, marker_mask, segment_cells):
                    choice = cells
                    break
            if choice is None:
                placed = False
                break
            for rr, cc in choice:
                grid[rr, cc] = int(shape_color[kind])
                marker_mask[rr, cc] = True
        if not placed:
            continue

        counts = Counter(int(v) for v in grid.ravel())
        non_bg_counts = {color: count for color, count in counts.items() if color != BG}
        if PATH not in non_bg_counts:
            continue
        if non_bg_counts[PATH] != max(non_bg_counts.values()):
            continue
        if list(non_bg_counts.values()).count(non_bg_counts[PATH]) != 1:
            continue

        expected = _expected_output(grid, top, bottom, left, right, rows, cols, row_types, vertical_types)
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue

        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
