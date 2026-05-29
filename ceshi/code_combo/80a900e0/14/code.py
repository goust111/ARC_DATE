# Combo variant of 80a900e0 (index 14)
# combo_spec: D12/0+D18/2
# Name: 补全并裁剪
#
# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - Base scene stays the same: a full checkerboard canvas, 1-3 hollow 5x5 diamonds, and
#      short outer diagonal segments on selected diamond edges.
#    - D18/2 changes the input encoding: each endpoint ray is only partly present in the input.
#      The kept part is always a contiguous prefix attached to the segment endpoint; the removed
#      suffix is restored to checkerboard.
#    - D12/0 changes the output shape: after reconstructing the complete full-canvas picture,
#      return only the smallest rectangle containing every non-checkerboard pixel.
#    - Small trace:
#      1. Full output is built with a diamond whose active edges are BL and BR, so all rays point
#         downward and the top boundary stays empty.
#      2. Input is derived by trimming each ray to a prefix, e.g. one ray keeps 2 cells, another
#         keeps 0, another keeps 5, another keeps 1.
#      3. Solver detects the fixed outer segments from the diamond anchor, follows any same-color
#         prefix already present, extends each ray to the canvas boundary, then crops away the
#         untouched checkerboard margin at the top.
#      4. Relative to the original task, the input no longer has clean isolated 3-pixel segments,
#         and the output is no longer the full canvas.
# b) Complexity calibration:
#    - The crop must be visually obvious, so the generator chooses one global omitted boundary side
#      and restricts every diamond to the opposite-facing edge pair. This guarantees at least one
#      checkerboard-only margin in the completed full output.
#    - The partial-ray effect must also be visible, so each sample is required to contain at least
#      two genuinely partial rays and at least one shown ray cell plus one hidden ray cell overall.
#    - Freedom is controlled: omitted side has only 4 choices and each ray trim is a single prefix
#      length. That is enough variety to avoid repetition without making the rule hard to infer.
# c) Implicit-assumption review of the original solver:
#    - Original adjacent-component parsing assumed rays were absent in the input. That fails here
#      because segment + visible ray prefix can form bent 8-connected shapes, so the combo solver
#      must read candidate segments from fixed edge templates instead of connected components.
#    - Original painting assumed the first step after an endpoint was background. That no longer
#      holds; the combo solver must step through same-color prefix cells before filling bg suffixes.
#    - Original return path assumed full-canvas output. The combo intentionally breaks that and
#      adds a final crop over all non-background pixels after full reconstruction.
#    - The checkerboard colors still need to be the two most frequent colors for bg inference via
#      `_top_k_colors`; generator keeps the original bg-dominance safety check on the full output.
# d) Placement legality:
#    - Placements are validated on the completed full output, not on the trimmed input. This keeps
#      diamonds, segments, and both endpoint rays pairwise non-overlapping and within bounds.
#    - Input trimming only removes ray suffix cells back to checkerboard, so it cannot create new
#      collisions or destroy the segment template cells that the solver depends on.
# e) Visual observability:
#    - Across examples, a human sees both effects directly: some rays are already partially drawn in
#      the input, and the output frame is tighter than the input canvas.
#    - The omitted-side choice is readable from geometry alone because all completed non-background
#      pixels avoid one boundary; no hidden ordering by color value or index is used.
# f) Visual cleanliness:
#    - No extra markers are introduced. The scene still contains only checkerboard, diamonds,
#      segments, and rays, with the original non-overlap rule preventing cluttered crossings.
# g) New solver assumptions and generator guarantees:
#    - Solver assumes real segments still fully occupy their fixed template cells and are monochrome
#      non-background; generator never trims segment cells.
#    - Solver assumes each visible ray, if present, is a contiguous outward prefix in the segment
#      color; generator only removes suffixes and never punches holes into a ray.
#    - Solver assumes at least one non-background pixel exists for cropping; generator always places
#      at least one diamond, so the cropped bbox is well-defined.

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


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


# ── Tunable parameters ──────────────────────────────────────
GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
MIN_SEGMENT_LEN = 2
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 600

# ── Geometry constants ──────────────────────────────────────
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
_EDGE_NAMES = tuple(_EDGE_INFO)

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


def _diamond_anchor(dpts: list[tuple[int, int]]) -> tuple[int, int]:
    return min(r for r, _ in dpts), min(c for _, c in dpts)


def _visible_segment(anchor: tuple[int, int], shape: tuple[int, int], edge: str) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0 = anchor
    info = _EDGE_INFO[edge]
    cells = [
        (r0 + dr, c0 + dc)
        for dr, dc in info["segment"]
        if 0 <= r0 + dr < h and 0 <= c0 + dc < w
    ]
    if len(cells) < MIN_SEGMENT_LEN:
        return []
    return _sort_segment(cells, info["slope"])


def _iter_diamonds(g: np.ndarray, bg_colors: set[int]) -> list[tuple[int, list[tuple[int, int]]]]:
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    diamonds = [(col, pts) for col, pts in _components8_by_color(g, non_bg) if _is_diamond8(pts)]
    return sorted(diamonds, key=lambda item: _diamond_anchor(item[1]))


def _detect_edge_segment(
    g: np.ndarray,
    bg_colors: set[int],
    anchor: tuple[int, int],
    edge: str,
) -> tuple[int, list[tuple[int, int]], tuple[int, int]] | None:
    segment = _visible_segment(anchor, g.shape, edge)
    if not segment:
        return None
    vals = [int(g[r, c]) for r, c in segment]
    if any(v in bg_colors for v in vals):
        return None
    if len(set(vals)) != 1:
        return None
    return vals[0], segment, _EDGE_INFO[edge]["outward"]


def _extend_ray(
    out: np.ndarray,
    bg_colors: set[int],
    color: int,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> None:
    h, w = out.shape
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    while 0 <= r < h and 0 <= c < w:
        v = int(out[r, c])
        if v == color:
            r += dr
            c += dc
            continue
        if v in bg_colors:
            out[r, c] = color
            r += dr
            c += dc
            continue
        break


def _tight_crop_non_bg(grid: np.ndarray, bg_colors: set[int]) -> np.ndarray:
    mask = ~np.isin(grid, list(bg_colors))
    if not np.any(mask):
        return grid.copy()
    r0, c0, r1, c1 = bbox_coords(np.argwhere(mask))
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _solve_full(grid: np.ndarray) -> tuple[np.ndarray, set[int]]:
    g = np.asarray(grid, dtype=int)
    bg_colors = set(_top_k_colors(g, 2))
    out = g.copy()

    for _, dpts in _iter_diamonds(g, bg_colors):
        anchor = _diamond_anchor(dpts)
        for edge in _EDGE_NAMES:
            found = _detect_edge_segment(g, bg_colors, anchor, edge)
            if found is None:
                continue
            color, segment, outward = found
            for endpoint in (segment[0], segment[-1]):
                _extend_ray(out, bg_colors, color, endpoint, outward)

    return out, bg_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct clipped outward rays around checkerboard diamonds, then return only the active area.

    Input structure:
    - A full checkerboard canvas whose two most frequent colors form the background.
    - One to three hollow 8-pixel diamonds in non-background colors.
    - Short outer diagonal segments beside some diamond edges, with same-color ray prefixes that may
      already extend outward partway from one or both segment endpoints.

    Rule:
    - Read each possible outer segment from the diamond's fixed edge template rather than from a
      connected component, because visible ray prefixes can be attached to the segment.
    - For every detected segment endpoint, continue along the outward diagonal, passing through any
      existing same-color prefix and filling the remaining background suffix until the grid boundary.
    - After the full reconstruction is complete on the original canvas, crop to the minimal rectangle
      containing every non-background pixel.

    Key insight:
    - The stable information is the diamond geometry and the fixed placement of its adjacent outer
      segments. Once the segment and outward direction are recovered from that template, the visible
      prefix reveals which diagonal to continue, and the final crop is just the bbox of all colored cells.
    """
    out, bg_colors = _solve_full(grid)
    return _tight_crop_non_bg(out, bg_colors)


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
        segment = _visible_segment((r0, c0), (h, w), edge)
        if not segment:
            return None
        ray_a = _ray_cells((h, w), segment[0], info["outward"])
        ray_b = _ray_cells((h, w), segment[-1], info["outward"])
        part_cells = segment + ray_a + ray_b
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None
        all_cells.update(part_cells)
        edge_parts.append(
            {
                "segment": segment,
                "rays": (ray_a, ray_b),
                "weight": len(part_cells),
            }
        )
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


def _choose_keep_len(full_len: int) -> int:
    if full_len == 0:
        return 0
    if full_len == 1:
        return random.randint(0, 1)
    roll = random.random()
    if roll < 0.7:
        return random.randint(1, full_len - 1)
    if roll < 0.85:
        return 0
    return full_len


def _render_full_output(
    h: int,
    w: int,
    placements: list[dict[str, object]],
    diamond_colors: list[int],
    segment_colors: list[list[int]],
):
    out = _checkerboard(h, w)
    ray_groups: list[list[tuple[int, int]]] = []
    for pi, placement in enumerate(placements):
        _paint_cells(out, placement["diamond"], diamond_colors[pi])
        for ei, edge_part in enumerate(placement["edges"]):
            color = segment_colors[pi][ei]
            _paint_cells(out, edge_part["segment"], color)
            for ray in edge_part["rays"]:
                _paint_cells(out, ray, color)
                ray_groups.append(ray)
    return out, ray_groups


def _crop_margins(full_grid: np.ndarray, bg_colors: set[int]) -> tuple[int, int, int, int]:
    mask = ~np.isin(full_grid, list(bg_colors))
    r0, c0, r1, c1 = bbox_coords(np.argwhere(mask))
    h, w = full_grid.shape
    return r0, h - 1 - r1, c0, w - 1 - c1


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
    full_out, ray_groups = _render_full_output(h, w, placements, diamond_colors, segment_colors)

    if not _bg_counts_safe(full_out):
        return None

    long_rays = [ray for ray in ray_groups if len(ray) >= 2]
    if len(long_rays) < 2 or max((len(ray) for ray in ray_groups), default=0) < 4:
        return None

    inp = full_out.copy()
    keep_lengths: list[int] = []
    partial_ray_count = 0
    shown_ray_cells = 0
    hidden_ray_cells = 0
    for ray in ray_groups:
        keep = _choose_keep_len(len(ray))
        keep_lengths.append(keep)
        if 0 < keep < len(ray):
            partial_ray_count += 1
        shown_ray_cells += keep
        hidden_ray_cells += len(ray) - keep
        for r, c in ray[keep:]:
            inp[r, c] = (r + c + 1) % 2

    if partial_ray_count < 2 or shown_ray_cells == 0 or hidden_ray_cells == 0:
        return None

    cropped_out = _tight_crop_non_bg(full_out, {0, 1})
    if cropped_out.shape == full_out.shape:
        return None

    actual = main(inp)
    if not np.array_equal(actual, cropped_out):
        return None

    if return_meta:
        return {
            "input": inp,
            "full_output": full_out,
            "cropped_output": cropped_out,
            "omitted_side": omitted_side,
            "partial_ray_count": partial_ray_count,
            "keep_lengths": keep_lengths,
            "crop_margins": _crop_margins(full_out, {0, 1}),
        }
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
