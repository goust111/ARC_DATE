# Combo variant of 80a900e0 (index 27)
# combo_spec: D11/3+D12/0
# Name: 端点标记裁剪
#
# Design analysis for the fused variant
# ------------------------------------
# fused_rule_summary:
# - Keep the original input encoding: checkerboard background, 8-pixel hollow diamonds,
#   and short diagonal outer segments attached to selected diamond edges.
# - Change the output effect from "paint whole rays" to "place a fixed terminal marker"
#   at the boundary cell reached by each endpoint ray.
# - After placing those markers on the full canvas, return only the tight crop covering
#   every non-checkerboard pixel.
#
# a) concretization and behavior trace:
# - Local example:
#   - Suppose a segment lies on a diamond's top-right side. In the original task its
#     outward direction is (-1, +1), and each endpoint paints a full diagonal corridor
#     until the canvas boundary.
#   - In this fused variant, the same two endpoint rays are inferred, but the solver
#     only keeps their final boundary cells and colors those cells with a fixed marker
#     color 9.
#   - Then, instead of returning the whole HxW board, the solver computes the smallest
#     rectangle containing diamonds, segments, and these new marker pixels.
# - Difference from the original:
#   - transform change: sparse boundary markers replace long colored corridors
#   - grid-shape change: exterior checkerboard-only margins are removed
# - Difference from the two single variants:
#   - D11 alone would leave the full board around the sparse markers.
#   - D12 alone would crop a board whose rays are still thick colored corridors.
#   - Here the crop is driven by the sparse marker geometry, so marker placement and
#     output shape interact directly.
#
# b) complexity calibration:
# - The marker color is fixed to 9 and reserved from the object palette. This keeps the
#   new terminal semantics visually learnable without introducing a large color mapping.
# - The generator uses a single omitted boundary side per sample. Every diamond then
#   uses the two opposite-facing outer edges, which makes the crop change easy to see:
#   one side is trimmed away, while marker endpoints still often pin the crop to other
#   canvas edges.
# - Full hidden ray corridors are still reserved during placement. This avoids ambiguous
#   samples where one object's hidden ray would hit another object before the boundary.
#
# c) implicit-assumption audit for main():
# - `_top_k_colors(g, 2)` assumes the checkerboard colors remain the two most frequent.
#   This still holds because markers appear only in the output and generator keeps the
#   background-dominance safety check on the full canvas.
# - Diamond detection by same-color 8-connected components is unchanged; the input still
#   contains only diamonds and short segments as non-background objects.
# - Outer segments are still found by taking non-background neighbors of each diamond.
#   The fused rule does not add any input-side decoration, so this assumption survives.
# - The original fill loop assumed every background cell on the ray should become the
#   segment color. That is no longer valid. It is replaced by a terminal search that
#   verifies the hidden path stays background and returns only the final boundary cell.
# - Tight cropping introduces a new assumption: there is at least one non-background
#   pixel in the solved full canvas. Generator guarantees this because each sample has
#   at least one diamond and two outer segments.
#
# d) placement legality:
# - Generator remains output-first, but "output-first" now means building the full
#   canvas with diamonds, visible segments, and terminal markers only.
# - Each candidate placement also reserves the entire hidden ray corridors inside
#   `all_cells`, so no two objects, markers, or hypothetical ray paths overlap.
# - Each ray must have positive length; otherwise its terminal marker would coincide with
#   the segment endpoint and the endpoint-marking rule would be visually degenerate.
#
# e) visual observability:
# - Across samples, color 9 appears only on boundary cells, never in the input. That
#   makes the "ray endpoint marker" behavior visible from a few examples.
# - The crop is also visible because the generator enforces at least one omitted side,
#   so the output is strictly smaller than the original canvas.
# - Because markers can sit on canvas edges and are included in the crop, a solver must
#   infer both where the hidden rays end and how those endpoints affect the returned box.
#
# f) visual neatness:
# - Sparse markers are cleaner than long ray corridors, but could become confusing if
#   many objects crowded the same boundary. Hidden-path reservation prevents that.
# - Restricting each sample to one global omitted side keeps the cropped silhouette
#   regular and avoids clutter from every side being simultaneously active.
#
# g) change inferability:
# - The only new color semantics is "9 means ray terminal marker". Diamonds and segments
#   keep their usual color roles, so there is no large unexplained recoloring system.
# - Crop boundaries are explainable from visible structure: they are the min/max rows and
#   columns of all non-background pixels after marker placement, not arbitrary slicing.
#
# post-change solver assumptions and generator guarantees:
# - Each adjacent component around a diamond is a clipped diagonal segment of length >= 2.
# - Every chosen endpoint has a background-only diagonal path to a boundary cell.
# - Marker color 9 never appears in the input palette for diamonds or segments.
# - The solved full output always omits at least one canvas side, so tight crop differs
#   from the full canvas and the D12 component is not visually silent.

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors8


MARKER_COLOR = 9
GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
MIN_SEGMENT_LEN = 2
MAX_GEN_RETRIES = 300
COLOR_RANGE = tuple(range(2, 9))

_DIAMOND8_OFFSETS = {
    (0, 2),
    (1, 1),
    (1, 3),
    (2, 0),
    (2, 4),
    (3, 1),
    (3, 3),
    (4, 2),
}

_DIAMOND_COORDS = (
    (0, 2),
    (1, 1),
    (1, 3),
    (2, 0),
    (2, 4),
    (3, 1),
    (3, 3),
    (4, 2),
)

_EDGE_INFO = {
    "TR": {"segment": ((-1, 3), (0, 4), (1, 5)), "outward": (-1, 1), "slope": "diff"},
    "TL": {"segment": ((-1, 1), (0, 0), (1, -1)), "outward": (-1, -1), "slope": "sum"},
    "BL": {"segment": ((3, -1), (4, 0), (5, 1)), "outward": (1, -1), "slope": "diff"},
    "BR": {"segment": ((3, 5), (4, 4), (5, 3)), "outward": (1, 1), "slope": "sum"},
}

_OMITTED_SIDE_TO_EDGES = {
    "top": ("BL", "BR"),
    "bottom": ("TL", "TR"),
    "left": ("TR", "BR"),
    "right": ("TL", "BL"),
}


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    return bbox_coords(np.asarray(pts))


def _top_k_colors(g: np.ndarray, k: int) -> list[int]:
    return [int(c) for c, _ in Counter(map(int, g.ravel())).most_common(k)]


def _components8_by_color(g: np.ndarray, colors: set[int]) -> list[tuple[int, list[tuple[int, int]]]]:
    return [(int(col), pts) for col in colors for pts in _cc8(g == col)]


def _is_diamond8(comp: list[tuple[int, int]]) -> bool:
    if len(comp) != 8:
        return False
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    if (r1 - r0) != 4 or (c1 - c0) != 4:
        return False
    norm = {(r - r0, c - c0) for r, c in comp}
    return norm == _DIAMOND8_OFFSETS


def _outer_line_info(pts: list[tuple[int, int]]):
    if len(pts) < 2:
        return None
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    if all((r - c) == (rs[0] - cs[0]) for r, c in pts):
        return ("rc_diff", sorted(pts, key=lambda p: (p[0], p[1])))
    if all((r + c) == (rs[0] + cs[0]) for r, c in pts):
        return ("rc_sum", sorted(pts, key=lambda p: (p[0], -p[1])))
    return None


def _terminal_boundary_cell(
    g: np.ndarray,
    bg_colors: set[int],
    start: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[int, int] | None:
    """Return the boundary terminal if the whole hidden corridor stays background."""
    h, w = g.shape
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    while 0 <= r < h and 0 <= c < w:
        if int(g[r, c]) not in bg_colors:
            return None
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return (r, c)
        r, c = nr, nc
    return None


def _tight_crop_non_bg(grid: np.ndarray, bg_colors: set[int]) -> np.ndarray:
    mask = ~np.isin(grid, list(bg_colors))
    if not np.any(mask):
        return grid.copy()
    r0, c0, r1, c1 = bbox_coords(np.argwhere(mask))
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _solve_full(grid: np.ndarray) -> tuple[np.ndarray, set[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    comps = _components8_by_color(g, non_bg)
    diamonds: list[tuple[int, list[tuple[int, int]]]] = [(col, pts) for col, pts in comps if _is_diamond8(pts)]

    out = g.copy()

    def bbox_key(item):
        _, pts = item
        return _bbox(pts)

    for _, dpts in sorted(diamonds, key=bbox_key):
        dset = set(dpts)
        adj: list[tuple[int, int]] = []
        for r, c in dpts:
            for rr, cc in neighbors8(r, c):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dset:
                    if int(g[rr, cc]) not in bg_colors:
                        adj.append((rr, cc))
        if not adj:
            continue

        adj_mask = np.zeros((h, w), dtype=bool)
        for r, c in adj:
            adj_mask[r, c] = True
        outer_comps = _cc8(adj_mask)

        for oc in outer_comps:
            info = _outer_line_info(oc)
            if info is None:
                continue
            slope_key, endpoints = info

            if slope_key == "rc_diff":
                normals = [(-1, 1), (1, -1)]
            else:
                normals = [(-1, -1), (1, 1)]

            best_n = None
            best_score = -1
            for n in normals:
                score = sum(((r + n[0], c + n[1]) in dset) for r, c in oc)
                if score > best_score:
                    best_score = score
                    best_n = n
            if best_n is None or best_score <= 0:
                continue
            outward = (-best_n[0], -best_n[1])

            for start in (endpoints[0], endpoints[-1]):
                terminal = _terminal_boundary_cell(g, bg_colors, start, outward)
                if terminal is not None:
                    out[terminal] = MARKER_COLOR

    return out, bg_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct boundary endpoint markers from the hidden outward rays, then crop away
    checkerboard-only margins from the completed board.

    Input structure:
    - A full checkerboard canvas using the two most common colors as background.
    - One to three hollow 8-pixel diamonds in distinct non-background colors.
    - Short diagonal outer segments adjacent to selected diamond edges; these are the
      only ray cues present in the input.

    Rule:
    - For each outer segment, infer the outward diagonal direction by checking which
      candidate normal points back toward the nearby diamond.
    - From each segment endpoint, follow the hidden diagonal ray until the last
      in-bounds background cell on the canvas boundary, and place marker color 9 there.
    - After all markers are added, return the tight bounding box of every non-background
      pixel rather than the original full canvas.

    Key insight:
    - The rays are still determined exactly as in the base task, but their visible effect
      is reduced to boundary terminals; those sparse terminals then participate in the
      crop, so output size depends on the inferred ray endpoints.
    """
    full_out, bg_colors = _solve_full(grid)
    return _tight_crop_non_bg(full_out, bg_colors)


def _checkerboard(h: int, w: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    return ((rows + cols + 1) % 2).astype(int)


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _sort_segment(cells: list[tuple[int, int]], slope: str) -> list[tuple[int, int]]:
    if slope == "diff":
        return sorted(cells, key=lambda p: (p[0], p[1]))
    return sorted(cells, key=lambda p: (p[0], -p[1]))


def _ray_cells(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _placement_keeps_side_open(h: int, w: int, r0: int, c0: int, omitted_side: str) -> bool:
    if omitted_side == "top":
        return r0 >= 1
    if omitted_side == "bottom":
        return r0 <= h - 6
    if omitted_side == "left":
        return c0 >= 1
    return c0 <= w - 6


def _candidate_placement(
    h: int,
    w: int,
    r0: int,
    c0: int,
    edges: tuple[str, ...],
) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]
    all_cells = set(diamond)
    edge_parts = []

    for edge in edges:
        info = _EDGE_INFO[edge]
        segment = [
            (r0 + dr, c0 + dc)
            for dr, dc in info["segment"]
            if 0 <= r0 + dr < h and 0 <= c0 + dc < w
        ]
        if len(segment) < MIN_SEGMENT_LEN:
            return None
        segment = _sort_segment(segment, info["slope"])

        ray_a = _ray_cells((h, w), segment[0], info["outward"])
        ray_b = _ray_cells((h, w), segment[-1], info["outward"])
        if not ray_a or not ray_b:
            return None

        terminals = [ray_a[-1], ray_b[-1]]
        hidden_cells = ray_a + ray_b
        part_cells = segment + hidden_cells
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None

        all_cells.update(part_cells)
        edge_parts.append(
            {
                "segment": segment,
                "terminals": terminals,
                "weight": len(segment),
            }
        )

    return {"diamond": diamond, "edges": edge_parts, "all_cells": frozenset(all_cells)}


def _search_placements(
    h: int,
    w: int,
    diamond_count: int,
    edges: tuple[str, ...],
    omitted_side: str,
) -> list[dict[str, object]] | None:
    positions = [
        (r, c)
        for r in range(h - 4)
        for c in range(w - 4)
        if (r + c) % 2 == 0 and _placement_keeps_side_open(h, w, r, c, omitted_side)
    ]
    random.shuffle(positions)

    candidates = [
        placement
        for r0, c0 in positions
        if (placement := _candidate_placement(h, w, r0, c0, edges)) is not None
    ]
    if len(candidates) < diamond_count:
        return None

    occupied: set[tuple[int, int]] = set()
    placed: list[dict[str, object]] = []

    def backtrack(i: int) -> bool:
        if i == diamond_count:
            return True
        for placement in candidates:
            cells = placement["all_cells"]
            if occupied.isdisjoint(cells):
                occupied.update(cells)
                placed.append(placement)
                if backtrack(i + 1):
                    return True
                placed.pop()
                occupied.difference_update(cells)
        return False

    if not backtrack(0):
        return None
    return placed


def _assign_segment_colors(placements: list[dict[str, object]], diamond_colors: list[int]) -> list[list[int]]:
    diamond_set = set(diamond_colors)
    palette = [c for c in COLOR_RANGE if c not in diamond_set]
    counts = Counter({c: 0 for c in palette})
    for c in diamond_colors:
        counts[c] += len(_DIAMOND_COORDS)

    weighted_edges = sorted(
        (
            (edge_part["weight"], random.random(), placement_i, edge_i)
            for placement_i, placement in enumerate(placements)
            for edge_i, edge_part in enumerate(placement["edges"])
        ),
        reverse=True,
    )
    segment_colors: list[list[int]] = [[0] * len(placement["edges"]) for placement in placements]
    for _, _, placement_i, edge_i in weighted_edges:
        chosen = min(palette, key=lambda c: (counts[c], random.random()))
        segment_colors[placement_i][edge_i] = chosen
        counts[chosen] += placements[placement_i]["edges"][edge_i]["weight"]
    return segment_colors


def _bg_counts_safe(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    max_non_bg = max((count for color, count in counts.items() if color not in (0, 1)), default=0)
    return counts[0] > max_non_bg and counts[1] > max_non_bg


def _render_full_output(
    h: int,
    w: int,
    placements: list[dict[str, object]],
    diamond_colors: list[int],
    segment_colors: list[list[int]],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    out = _checkerboard(h, w)
    marker_coords: list[tuple[int, int]] = []
    for placement_i, placement in enumerate(placements):
        _paint_cells(out, placement["diamond"], diamond_colors[placement_i])
        for edge_i, edge_part in enumerate(placement["edges"]):
            _paint_cells(out, edge_part["segment"], segment_colors[placement_i][edge_i])
            _paint_cells(out, edge_part["terminals"], MARKER_COLOR)
            marker_coords.extend(edge_part["terminals"])
    return out, marker_coords


def _gen_attempt(return_meta: bool = False):
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    diamond_count = random.randint(*DIAMOND_COUNT_RANGE)
    omitted_side = random.choice(tuple(_OMITTED_SIDE_TO_EDGES))
    edges = _OMITTED_SIDE_TO_EDGES[omitted_side]

    placements = _search_placements(h, w, diamond_count, edges, omitted_side)
    if placements is None:
        return None

    diamond_colors = random.sample(list(COLOR_RANGE), len(placements))
    segment_colors = _assign_segment_colors(placements, diamond_colors)
    full_out, marker_coords = _render_full_output(h, w, placements, diamond_colors, segment_colors)

    if not _bg_counts_safe(full_out):
        return None

    inp = full_out.copy()
    if marker_coords:
        rs, cs = zip(*marker_coords)
        rs = np.asarray(rs)
        cs = np.asarray(cs)
        inp[rs, cs] = (rs + cs + 1) % 2

    expected = _tight_crop_non_bg(full_out, {0, 1})
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    if expected.shape == full_out.shape:
        return None

    input_non_bg_mask = ~np.isin(inp, [0, 1])
    in_r0, in_c0, in_r1, in_c1 = bbox_coords(np.argwhere(input_non_bg_mask))
    out_r0, out_c0, out_r1, out_c1 = bbox_coords(np.argwhere(~np.isin(full_out, [0, 1])))
    marker_expands_bbox = (out_r0 < in_r0) or (out_c0 < in_c0) or (out_r1 > in_r1) or (out_c1 > in_c1)
    if not marker_expands_bbox:
        return None

    if return_meta:
        return {
            "input": inp,
            "full_output": full_out,
            "cropped_output": expected,
            "omitted_side": omitted_side,
            "marker_coords": marker_coords,
            "input_bbox": (in_r0, in_c0, in_r1, in_c1),
            "output_bbox": (out_r0, out_c0, out_r1, out_c1),
        }
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
