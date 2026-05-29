# Variant of 80a900e0
# Dimension: D12 — 输出网格形状 (grid_shape)
# Change: 输出与输入同尺寸（same）... → 裁剪到有效区域
#
# Design analysis for the D12 variant
# - Reference-task check:
#   - The prompt names source task e87109e9 for the tight-crop idea.
#   - Its solver file is not present in the allowed workspace, so the implementation below
#     follows the variant description directly and keeps the original 80a900e0 mechanics intact.
# - Concrete behavior:
#   - Input is still a full checkerboard canvas with 1-3 hollow diamonds plus short diagonal
#     outer segments.
#   - Solver still reconstructs the missing outward rays on the full input canvas first.
#   - New behavior happens only after reconstruction: instead of returning the full canvas,
#     return the smallest axis-aligned rectangle containing every non-checkerboard pixel.
#   - "Non-background" means "not one of the two checkerboard colors", exactly as in the
#     original task. The crop therefore keeps every diamond, segment, and generated ray pixel.
# - Small trace:
#   - Original: if a bottom-facing pair of outer segments fires rays to the bottom boundary,
#     the completed output still occupies only the lower part of the canvas, but the solver
#     returns the whole HxW board.
#   - Variant: the same rays are reconstructed on the HxW board, then rows/cols containing
#     only checkerboard are stripped away. The returned board may be shorter or narrower.
#   - Difference: reasoning now includes "where does the meaningful drawing end?" in addition
#     to "where do the rays go?".
# - Complexity calibration:
#   - The crop must be visually observable; otherwise many samples would look identical to the
#     original task. To make that reliable, the generator chooses a global omitted boundary
#     side (top/bottom/left/right) and places every diamond so that no non-background pixel
#     touches that side in the completed output.
#   - This is a deliberate adaptation tied to D12. The ray rule, object shapes, colors, and
#     checkerboard semantics remain unchanged.
# - Implicit-assumption review of the original solver:
#   - `_top_k_colors(g, 2)`: assumes the checkerboard colors remain the two most frequent.
#     Still valid; generator preserves the original frequency safety check on the full input.
#   - Diamond detection via 8-connected same-color components: unchanged. Cropping happens
#     only after the full output is computed, so input-side geometry is identical.
#   - Adjacent non-bg pixels around each diamond are taken from the input grid, not from the
#     evolving output. Still valid because the variant does not alter the input encoding.
#   - Rays stop on the first non-background hit or on the full-grid boundary. This must still
#     happen on the original canvas, not on the cropped canvas; otherwise the crop would feed
#     back into the ray geometry and change the original task semantics.
#   - Final return path in the original solver implicitly assumed full-canvas output. That is
#     the single assumption intentionally broken by D12; we replace it with a crop over all
#     non-background pixels after reconstruction.
# - Placement legality:
#   - Placements are still output-first and reject any overlap between diamonds, segments, and
#     the full rays they imply.
#   - The omitted-side constraint is enforced on diamond top-left positions, ensuring that the
#     final non-background bbox is strictly smaller than the canvas on at least one side.
#   - Generator validation still calls `main(inp)` and compares it against the expected cropped
#     output, so solver assumptions are exercised directly.
# - Visual observability / cleanliness:
#   - Every sample visibly shows the original ray-completion pattern plus an outer checkerboard
#     margin removed on one side. This is easy to spot across 2-4 examples.
#   - Colors do not encode any hidden ordering; they only distinguish diamonds and segments as
#     in the original task.
# - New solver assumptions introduced by D12:
#   - The crop helper assumes there is at least one non-background pixel. Generator guarantees
#     at least one diamond, so this always holds.
#   - The crop helper assumes the same two checkerboard colors remain background after solving.
#     Generator keeps the original bg-dominance condition, so the assumption holds.

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors8


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _bbox(pts: list[tuple[int, int]]):
    return bbox_coords(np.asarray(pts))


def _top_k_colors(g: np.ndarray, k: int) -> list[int]:
    return [int(c) for c, _ in Counter(map(int, g.ravel())).most_common(k)]


def _components8_by_color(g: np.ndarray, colors: set[int]) -> list[tuple[int, list[tuple[int, int]]]]:
    return [(int(col), pts) for col in colors for pts in _cc8(g == col)]


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
        adj = []
        for r, c in dpts:
            for rr, cc in neighbors8(r, c):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dset:
                    v = int(g[rr, cc])
                    if v not in bg_colors:
                        adj.append((rr, cc))
        if not adj:
            continue

        adj_mask = np.zeros((h, w), dtype=bool)
        for r, c in adj:
            adj_mask[r, c] = True
        outer_comps = _cc8(adj_mask)

        for oc in outer_comps:
            ocol = int(g[oc[0][0], oc[0][1]])
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
                sc = sum(((r + n[0], c + n[1]) in dset) for r, c in oc)
                if sc > best_score:
                    best_score = sc
                    best_n = n
            if best_n is None or best_score <= 0:
                continue
            outward = (-best_n[0], -best_n[1])

            for pr, pc in (endpoints[0], endpoints[-1]):
                rr, cc = pr + outward[0], pc + outward[1]
                while 0 <= rr < h and 0 <= cc < w:
                    if int(out[rr, cc]) in bg_colors:
                        out[rr, cc] = ocol
                    else:
                        break
                    rr += outward[0]
                    cc += outward[1]

    return out, bg_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Original ray reconstruction on the full checkerboard, followed by tight cropping
    to the minimal rectangle containing all non-checkerboard pixels.
    """
    out, bg_colors = _solve_full(grid)
    return _tight_crop_non_bg(out, bg_colors)


# --------------- generator ---------------

GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 300
MIN_SEGMENT_LEN = 2

_DIAMOND_COORDS = (
    (0, 2),
    (1, 1), (1, 3),
    (2, 0), (2, 4),
    (3, 1), (3, 3),
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
    h: int, w: int, r0: int, c0: int, edges: tuple[str, ...]
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
        rays = _ray_cells((h, w), segment[0], info["outward"]) + _ray_cells((h, w), segment[-1], info["outward"])
        part_cells = segment + rays
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None
        all_cells.update(part_cells)
        edge_parts.append({"segment": segment, "rays": rays, "weight": len(part_cells)})
    return {"diamond": diamond, "edges": edge_parts, "all_cells": frozenset(all_cells)}


def _search_placements(h: int, w: int, diamond_count: int, edges: tuple[str, ...], omitted_side: str):
    positions = [
        (r, c)
        for r in range(h - 4)
        for c in range(w - 4)
        if (r + c) % 2 == 0 and _placement_keeps_side_open(h, w, r, c, omitted_side)
    ]
    random.shuffle(positions)
    candidates = [
        p for r0, c0 in positions
        if (p := _candidate_placement(h, w, r0, c0, edges)) is not None
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
        ((ep["weight"], random.random(), pi, ei) for pi, pl in enumerate(placements) for ei, ep in enumerate(pl["edges"])),
        reverse=True,
    )
    segment_colors: list[list[int]] = [[0] * len(pl["edges"]) for pl in placements]
    for _, _, pi, ei in weighted_edges:
        chosen = min(palette, key=lambda c: (counts[c], random.random()))
        segment_colors[pi][ei] = chosen
        counts[chosen] += placements[pi]["edges"][ei]["weight"]
    return segment_colors


def _bg_counts_safe(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    max_non_bg = max((counts[c] for c in COLOR_RANGE), default=0)
    return counts[0] > max_non_bg and counts[1] > max_non_bg


def _render_full_output(h: int, w: int, placements: list[dict[str, object]], diamond_colors: list[int], segment_colors: list[list[int]]):
    out = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    for pi, placement in enumerate(placements):
        _paint_cells(out, placement["diamond"], diamond_colors[pi])
        for ei, edge_part in enumerate(placement["edges"]):
            color = segment_colors[pi][ei]
            _paint_cells(out, edge_part["segment"], color)
            _paint_cells(out, edge_part["rays"], color)
            ray_coords.extend(edge_part["rays"])
    return out, ray_coords


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
    full_out, ray_coords = _render_full_output(h, w, placements, diamond_colors, segment_colors)

    if not _bg_counts_safe(full_out):
        return None

    inp = full_out.copy()
    if ray_coords:
        rs, cs = zip(*ray_coords)
        rs, cs = np.array(rs), np.array(cs)
        inp[rs, cs] = (rs + cs + 1) % 2

    expected = _tight_crop_non_bg(full_out, {0, 1})
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    if expected.shape == full_out.shape:
        return None

    if return_meta:
        return {
            "input": inp,
            "full_output": full_out,
            "cropped_output": expected,
            "omitted_side": omitted_side,
        }
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
