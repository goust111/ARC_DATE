from __future__ import annotations

import bisect
from collections import Counter, defaultdict

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _rotate_cw_offsets(offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(dc, -dr) for dr, dc in offsets]


def _rotate_offsets(offsets: list[tuple[int, int]], turns: int) -> list[tuple[int, int]]:
    turns %= 4
    result = list(offsets)
    for _ in range(turns):
        result = _rotate_cw_offsets(result)
    return result


def _is_connected(coords: list[tuple[int, int]]) -> bool:
    if not coords:
        return False
    coord_set = set(coords)
    stack = [coords[0]]
    seen = {coords[0]}
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in coord_set and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _find_wall_structure(
    grid: np.ndarray, bg: int
) -> tuple[int, list[int], list[int], dict[int, int], dict[int, int]]:
    h, w = grid.shape
    row_candidates: list[tuple[int, int]] = []
    col_candidates: list[tuple[int, int]] = []

    for r in range(h):
        counts = Counter(int(v) for v in grid[r])
        if counts.get(bg, 0) == 1 and len(counts) == 2:
            color = next(color for color in counts if color != bg)
            if counts[color] == w - 1:
                row_candidates.append((r, color))
    for c in range(w):
        counts = Counter(int(v) for v in grid[:, c])
        if counts.get(bg, 0) == 1 and len(counts) == 2:
            color = next(color for color in counts if color != bg)
            if counts[color] == h - 1:
                col_candidates.append((c, color))

    colors = [color for _idx, color in row_candidates + col_candidates]
    assert colors, "failed to find separator lines"
    wall = Counter(colors).most_common(1)[0][0]
    sep_rows = sorted(r for r, color in row_candidates if color == wall)
    sep_cols = sorted(c for c, color in col_candidates if color == wall)
    assert sep_rows and sep_cols, "expected both horizontal and vertical separators"

    row_regions: dict[int, int] = {}
    start = 0
    for i, sep in enumerate(sep_rows + [h]):
        for r in range(start, sep):
            row_regions[r] = i
        start = sep + 1
    col_regions: dict[int, int] = {}
    start = 0
    for j, sep in enumerate(sep_cols + [w]):
        for c in range(start, sep):
            col_regions[c] = j
        start = sep + 1
    return int(wall), sep_rows, sep_cols, row_regions, col_regions


def _region_bounds(separators: list[int], size: int) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        bounds.append((start, sep - 1))
        start = sep + 1
    bounds.append((start, size - 1))
    return bounds


def _extract_region_pattern(
    grid: np.ndarray,
    bounds: tuple[int, int, int, int],
    bg: int,
    wall: int,
) -> dict[str, object] | None:
    r0, r1, c0, c1 = bounds
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = (patch != bg) & (patch != wall)
    coords_local = np.argwhere(mask)
    if coords_local.size == 0:
        return None
    colors = np.unique(patch[mask])
    assert len(colors) == 1, "each region should contain a monochrome pattern"
    coords = [(int(r0 + r), int(c0 + c)) for r, c in coords_local]
    assert len(coords) <= 6, "pattern must stay small"
    assert _is_connected(coords), "pattern must be 4-connected"
    return {"color": int(colors[0]), "coords": coords}


def _anchor_pixel(coords: list[tuple[int, int]], gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = gap
    return min(
        coords,
        key=lambda rc: (
            abs(rc[0] - gr) + abs(rc[1] - gc),
            abs(rc[1] - gc),
            abs(rc[0] - gr),
            rc[0],
            rc[1],
        ),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each small pattern through its unique wall gap into the adjacent empty compartment, rotating it clockwise.

    Input structure:
    - The background is the dominant color.
    - One-pixel separator rows and separator columns in a single wall color split the grid into rectangular regions.
    - Every separator line has exactly one one-cell background gap, and the gap connects the two regions on either side of that line segment.
    - Some regions contain one small monochrome 4-connected pattern; the remaining regions are empty.

    Rule:
    - For each non-empty region, find the unique gap whose opposite side is an empty neighboring region.
    - Remove the pattern from its source region.
    - Use the pattern pixel nearest to the gap as the leading anchor, rotate the full pattern 90 degrees clockwise, and place the rotated anchor immediately across the gap in the destination region.
    - Keep all separator pixels and all gap cells unchanged.

    Key insight:
    - The gap does two jobs at once: it identifies which neighboring region receives the pattern, and it fixes the anchor that determines the clockwise quarter-turn during the transfer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _get_bg(g)
    wall, sep_rows, sep_cols, row_region_map, col_region_map = _find_wall_structure(g, bg)

    row_bounds = _region_bounds(sep_rows, h)
    col_bounds = _region_bounds(sep_cols, w)
    regions = {
        (ri, ci): (rb[0], rb[1], cb[0], cb[1])
        for ri, rb in enumerate(row_bounds)
        for ci, cb in enumerate(col_bounds)
    }
    patterns = {
        region: _extract_region_pattern(g, bounds, bg, wall)
        for region, bounds in regions.items()
    }

    edges: list[dict[str, object]] = []
    incidents: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)

    for wall_index, r in enumerate(sep_rows):
        gap_cols = np.flatnonzero(g[r] == bg)
        assert len(gap_cols) == 1, "horizontal separator must have exactly one gap"
        gap_c = int(gap_cols[0])
        region_col = bisect.bisect_left(sep_cols, gap_c)
        top = (wall_index, region_col)
        bottom = (wall_index + 1, region_col)
        edge = {"kind": "h", "gap": (r, gap_c), "a": top, "b": bottom}
        edges.append(edge)
        incidents[top].append(edge)
        incidents[bottom].append(edge)

    for wall_index, c in enumerate(sep_cols):
        gap_rows = np.flatnonzero(g[:, c] == bg)
        assert len(gap_rows) == 1, "vertical separator must have exactly one gap"
        gap_r = int(gap_rows[0])
        region_row = bisect.bisect_left(sep_rows, gap_r)
        left = (region_row, wall_index)
        right = (region_row, wall_index + 1)
        edge = {"kind": "v", "gap": (gap_r, c), "a": left, "b": right}
        edges.append(edge)
        incidents[left].append(edge)
        incidents[right].append(edge)

    moves: list[tuple[list[tuple[int, int]], list[tuple[int, int]], int]] = []
    for region, pattern in patterns.items():
        if pattern is None:
            continue
        candidate_edges = []
        for edge in incidents[region]:
            other = edge["b"] if edge["a"] == region else edge["a"]
            if patterns[other] is None:
                candidate_edges.append(edge)
        assert len(candidate_edges) == 1, "each source region must have one empty gap-neighbor"
        edge = candidate_edges[0]
        gap_r, gap_c = edge["gap"]
        anchor = _anchor_pixel(pattern["coords"], (gap_r, gap_c))
        offsets = [(r - anchor[0], c - anchor[1]) for r, c in pattern["coords"]]
        rotated = _rotate_cw_offsets(offsets)

        if edge["kind"] == "h":
            if edge["a"] == region:
                dest_anchor = (gap_r + 1, gap_c)
                target_region = edge["b"]
            else:
                dest_anchor = (gap_r - 1, gap_c)
                target_region = edge["a"]
        else:
            if edge["a"] == region:
                dest_anchor = (gap_r, gap_c + 1)
                target_region = edge["b"]
            else:
                dest_anchor = (gap_r, gap_c - 1)
                target_region = edge["a"]

        dst_coords = [(dest_anchor[0] + dr, dest_anchor[1] + dc) for dr, dc in rotated]
        tr0, tr1, tc0, tc1 = regions[target_region]
        for rr, cc in dst_coords:
            assert tr0 <= rr <= tr1 and tc0 <= cc <= tc1, "rotated pattern must stay inside destination region"
            assert row_region_map[rr] == target_region[0] and col_region_map[cc] == target_region[1]
        moves.append((pattern["coords"], dst_coords, int(pattern["color"])))

    out = g.copy()
    for src_coords, _dst_coords, _color in moves:
        for r, c in src_coords:
            out[r, c] = bg
    for _src_coords, dst_coords, color in moves:
        for r, c in dst_coords:
            assert out[r, c] == bg, "destination cells must start empty"
            out[r, c] = color
    return out


_RNG = np.random.default_rng()
_GEN_RETRIES = 300
_REGION_SIZE_MIN = 4
_REGION_SIZE_MAX = 8
_WALL_COUNT_MIN = 1
_WALL_COUNT_MAX = 3

_CANONICAL_SHAPES = [
    [(0, 0), (-1, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (-1, 0), (0, 1)],
    [(0, 0), (-1, 0), (-2, 0)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (-1, 0), (-1, 1)],
    [(0, 0), (-1, 0), (0, 1), (-1, 1)],
    [(0, 0), (-1, 0), (-2, 0), (-2, 1)],
    [(0, 0), (0, 1), (-1, 1)],
    [(0, 0), (-1, 0), (-2, 0), (-1, 1)],
    [(0, 0), (0, 1), (0, 2), (-1, 2)],
    [(0, 0), (-1, 0), (-2, 0), (-2, 1), (-2, 2)],
    [(0, 0), (0, 1), (0, 2), (-1, 2), (-2, 2)],
    [(0, 0), (-1, 0), (-1, 1), (0, 1), (-2, 1)],
    [(0, 0), (-1, 0), (-2, 0), (-1, 1), (-1, 2)],
    [(0, 0), (-1, 0), (-1, 1), (0, 1), (0, 2), (-1, 2)],
    [(0, 0)],
]


def _orient_shape(shape: list[tuple[int, int]], source_side: str) -> list[tuple[int, int]]:
    turns = {"above": 0, "right": 1, "below": 2, "left": 3}[source_side]
    return _rotate_offsets(shape, turns)


def _compute_bounds(lengths: np.ndarray) -> list[tuple[int, int]]:
    bounds = []
    start = 0
    for idx, length in enumerate(lengths):
        bounds.append((start, start + int(length) - 1))
        start += int(length)
        if idx != len(lengths) - 1:
            start += 1
    return bounds


def _source_side(region: tuple[int, int], edge: dict[str, object]) -> str:
    if edge["kind"] == "h":
        return "above" if edge["a"] == region else "below"
    return "left" if edge["a"] == region else "right"


def _other_region(region: tuple[int, int], edge: dict[str, object]) -> tuple[int, int]:
    return edge["b"] if edge["a"] == region else edge["a"]


def _choose_sources(
    candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not candidates:
        return []
    target = int(_RNG.integers(1, min(4, len(candidates)) + 1))
    by_side: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in candidates:
        by_side[item["side"]].append(item)
    chosen: list[dict[str, object]] = []
    used_regions: set[tuple[int, int]] = set()
    used_dests: set[tuple[int, int]] = set()

    side_order = [side for side in _RNG.permutation(["above", "below", "left", "right"]).tolist()]
    for side in side_order:
        items = by_side.get(side, [])
        if not items:
            continue
        for idx in _RNG.permutation(len(items)):
            item = items[int(idx)]
            region = item["region"]
            dest = item["dest"]
            if region in used_regions or dest in used_regions or dest in used_dests:
                continue
            chosen.append(item)
            used_regions.add(region)
            used_dests.add(dest)
            break
        if len(chosen) >= target:
            return chosen

    all_items = [candidates[int(i)] for i in _RNG.permutation(len(candidates))]
    for item in all_items:
        region = item["region"]
        dest = item["dest"]
        if region in used_regions or dest in used_regions or dest in used_dests:
            continue
        chosen.append(item)
        used_regions.add(region)
        used_dests.add(dest)
        if len(chosen) >= target:
            break
    return chosen


def _placement_for_source(
    source_bounds: tuple[int, int, int, int],
    dest_bounds: tuple[int, int, int, int],
    gap_line: int,
    span: tuple[int, int],
    side: str,
) -> tuple[list[tuple[int, int]], int] | None:
    sr0, sr1, sc0, sc1 = source_bounds
    dr0, dr1, dc0, dc1 = dest_bounds
    shape_order = [int(i) for i in _RNG.permutation(len(_CANONICAL_SHAPES))]
    for shape_idx in shape_order:
        oriented = _orient_shape(_CANONICAL_SHAPES[shape_idx], side)
        rotated = _rotate_cw_offsets(oriented)
        if side in ("above", "below"):
            gap_cols = [int(v) for v in _RNG.permutation(np.arange(span[0], span[1] + 1))]
            depth_limit = sr1 - sr0 + 1
            depths = [int(v) for v in _RNG.permutation(np.arange(1, depth_limit + 1))]
            for gap_c in gap_cols:
                for depth in depths:
                    anchor_r = gap_line - depth if side == "above" else gap_line + depth
                    anchor = (anchor_r, gap_c)
                    src_coords = [(anchor[0] + r, anchor[1] + c) for r, c in oriented]
                    dst_anchor = (gap_line + 1, gap_c) if side == "above" else (gap_line - 1, gap_c)
                    dst_coords = [(dst_anchor[0] + r, dst_anchor[1] + c) for r, c in rotated]
                    if all(sr0 <= r <= sr1 and sc0 <= c <= sc1 for r, c in src_coords) and all(
                        dr0 <= r <= dr1 and dc0 <= c <= dc1 for r, c in dst_coords
                    ):
                        return src_coords, gap_c
        else:
            gap_rows = [int(v) for v in _RNG.permutation(np.arange(span[0], span[1] + 1))]
            depth_limit = sc1 - sc0 + 1
            depths = [int(v) for v in _RNG.permutation(np.arange(1, depth_limit + 1))]
            for gap_r in gap_rows:
                for depth in depths:
                    anchor_c = gap_line - depth if side == "left" else gap_line + depth
                    anchor = (gap_r, anchor_c)
                    src_coords = [(anchor[0] + r, anchor[1] + c) for r, c in oriented]
                    dst_anchor = (gap_r, gap_line + 1) if side == "left" else (gap_r, gap_line - 1)
                    dst_coords = [(dst_anchor[0] + r, dst_anchor[1] + c) for r, c in rotated]
                    if all(sr0 <= r <= sr1 and sc0 <= c <= sc1 for r, c in src_coords) and all(
                        dr0 <= r <= dr1 and dc0 <= c <= dc1 for r, c in dst_coords
                    ):
                        return src_coords, gap_r
    return None


def _gen_attempt() -> np.ndarray | None:
    h_walls = int(_RNG.integers(_WALL_COUNT_MIN, _WALL_COUNT_MAX + 1))
    v_walls = int(_RNG.integers(_WALL_COUNT_MIN, _WALL_COUNT_MAX + 1))
    row_sizes = _RNG.integers(_REGION_SIZE_MIN, _REGION_SIZE_MAX + 1, size=h_walls + 1)
    col_sizes = _RNG.integers(_REGION_SIZE_MIN, _REGION_SIZE_MAX + 1, size=v_walls + 1)
    height = int(row_sizes.sum() + h_walls)
    width = int(col_sizes.sum() + v_walls)
    if max(height, width) > 30:
        return None

    colors = _RNG.permutation(10)
    bg, wall = int(colors[0]), int(colors[1])
    pattern_palette = [int(c) for c in colors[2:]]

    row_bounds = _compute_bounds(row_sizes)
    col_bounds = _compute_bounds(col_sizes)
    sep_rows = [b[1] + 1 for b in row_bounds[:-1]]
    sep_cols = [b[1] + 1 for b in col_bounds[:-1]]

    regions = {
        (ri, ci): (rb[0], rb[1], cb[0], cb[1])
        for ri, rb in enumerate(row_bounds)
        for ci, cb in enumerate(col_bounds)
    }

    h_gap_segments = [int(_RNG.integers(0, len(col_bounds))) for _ in range(h_walls)]
    v_gap_segments = [int(_RNG.integers(0, len(row_bounds))) for _ in range(v_walls)]

    incidents: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    edges: list[dict[str, object]] = []
    for i, seg_j in enumerate(h_gap_segments):
        a = (i, seg_j)
        b = (i + 1, seg_j)
        edge = {"kind": "h", "line": i, "a": a, "b": b, "span": col_bounds[seg_j]}
        edges.append(edge)
        incidents[a].append(edge)
        incidents[b].append(edge)
    for j, seg_i in enumerate(v_gap_segments):
        a = (seg_i, j)
        b = (seg_i, j + 1)
        edge = {"kind": "v", "line": j, "a": a, "b": b, "span": row_bounds[seg_i]}
        edges.append(edge)
        incidents[a].append(edge)
        incidents[b].append(edge)

    candidate_sources = []
    for region, edge_list in incidents.items():
        if len(edge_list) != 1:
            continue
        edge = edge_list[0]
        candidate_sources.append(
            {
                "region": region,
                "dest": _other_region(region, edge),
                "edge": edge,
                "side": _source_side(region, edge),
            }
        )
    chosen_sources = _choose_sources(candidate_sources)
    if not chosen_sources:
        return None

    grid = np.full((height, width), bg, dtype=int)
    for r in sep_rows:
        grid[r, :] = wall
    for c in sep_cols:
        grid[:, c] = wall

    horizontal_gaps: dict[int, int] = {}
    vertical_gaps: dict[int, int] = {}
    occupied_regions = {item["region"] for item in chosen_sources}
    occupied_dests = {item["dest"] for item in chosen_sources}
    if occupied_regions & occupied_dests:
        return None

    if len(pattern_palette) < len(chosen_sources):
        return None
    pattern_colors = pattern_palette[: len(chosen_sources)]

    for item, color in zip(chosen_sources, pattern_colors):
        source = item["region"]
        dest = item["dest"]
        edge = item["edge"]
        side = item["side"]
        placement = _placement_for_source(
            regions[source],
            regions[dest],
            sep_rows[edge["line"]] if edge["kind"] == "h" else sep_cols[edge["line"]],
            edge["span"],
            side,
        )
        if placement is None:
            return None
        src_coords, gap_coord = placement
        if edge["kind"] == "h":
            horizontal_gaps[edge["line"]] = gap_coord
        else:
            vertical_gaps[edge["line"]] = gap_coord
        for r, c in src_coords:
            grid[r, c] = color

    for i, seg_j in enumerate(h_gap_segments):
        if i not in horizontal_gaps:
            c0, c1 = col_bounds[seg_j]
            horizontal_gaps[i] = int(_RNG.integers(c0, c1 + 1))
        grid[sep_rows[i], horizontal_gaps[i]] = bg
    for j, seg_i in enumerate(v_gap_segments):
        if j not in vertical_gaps:
            r0, r1 = row_bounds[seg_i]
            vertical_gaps[j] = int(_RNG.integers(r0, r1 + 1))
        grid[vertical_gaps[j], sep_cols[j]] = bg

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid separator-push puzzle")
