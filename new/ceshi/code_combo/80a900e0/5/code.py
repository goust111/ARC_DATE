# Combo variant of 80a900e0 (index 5)
# combo_spec: D3/0+D11/0
# Name: 穿透覆盖射线

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors8


# Design analysis
# a) Concrete trace:
#    Put diamond A at (r,c) and diamond B at (r+gap,c+gap). A always owns a BR segment and
#    B always owns a TL segment, so their ray families lie on the same two r-c diagonals.
#    In the legacy task, A's BR ray stops as soon as it touches B's non-background pixels.
#    In this combo, the same ray recolors B's diamond/segment pixels and keeps going to the
#    boundary. B is processed later (top-left order), so B's TL rays then repaint the shared
#    diagonal cells again, making overwrite priority visible.
# b) Complexity calibration:
#    The generator does not allow arbitrary clutter. It always creates one deliberate A/B
#    interaction pair on the same main diagonal, then adds only 0-1 extra edges per diamond.
#    That keeps the "laser" behavior obvious while still giving repeated evidence.
# c) Implicit assumptions reviewed:
#    - `main()` still assumes the two most common input colors are the checkerboard colors.
#      The generator keeps input non-background counts small enough for that to hold.
#    - `main()` assumes the only non-background pixels adjacent to a diamond belong to its
#      own outer segments. The generator validates this local neighborhood constraint.
#    - Output overwrite order now matters, so `main()` sorts diamonds and segments by bbox to
#      make the priority spatially stable and generator-visible.
# d) Placement legality:
#    Input objects never overlap or leave the grid. Only rays may overlap in the output.
# e/f/g) Observability and tidiness:
#    Every generated sample must have all three signals:
#    - non-background pixels recolored by a ray,
#    - background cells painted after a ray has already pierced an obstacle,
#    - cells hit by multiple rays.
#    This rejects degenerate samples that would look like the original task.


GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 300

MANDATORY_GAP_RANGE = (7, 16)
EDGE_COUNT_CHOICES = (2, 2, 3)

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

_DIAMOND_COORDS = tuple(sorted(_DIAMOND8_OFFSETS))

_EDGE_INFO = {
    "TR": {"segment": ((-1, 3), (0, 4), (1, 5)), "outward": (-1, 1), "slope": "rc_diff"},
    "TL": {"segment": ((-1, 1), (0, 0), (1, -1)), "outward": (-1, -1), "slope": "rc_sum"},
    "BL": {"segment": ((3, -1), (4, 0), (5, 1)), "outward": (1, -1), "slope": "rc_diff"},
    "BR": {"segment": ((3, 5), (4, 4), (5, 3)), "outward": (1, 1), "slope": "rc_sum"},
}
_EDGE_NAMES = tuple(_EDGE_INFO)


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    return bbox_coords(np.asarray(pts, dtype=int))


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
    return {(r - r0, c - c0) for r, c in comp} == _DIAMOND8_OFFSETS


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


def _checkerboard(h: int, w: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    return ((rows + cols + 1) % 2).astype(int)


def _diamond_cells(top_left: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = top_left
    return [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]


def _segment_cells(top_left: tuple[int, int], edge_name: str) -> list[tuple[int, int]]:
    r0, c0 = top_left
    return [(r0 + dr, c0 + dc) for dr, dc in _EDGE_INFO[edge_name]["segment"]]


def _edge_sort_key(top_left: tuple[int, int], edge_name: str) -> tuple[int, int, int, int]:
    return _bbox(_segment_cells(top_left, edge_name))


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


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _ordered_rays_from_layout(
    shape: tuple[int, int], placements: list[dict[str, object]]
) -> list[dict[str, object]]:
    rays: list[dict[str, object]] = []
    for placement in sorted(placements, key=lambda pl: pl["top_left"]):
        top_left = placement["top_left"]
        for edge in sorted(placement["edges"], key=lambda item: _edge_sort_key(top_left, item["name"])):
            edge_name = edge["name"]
            info = _EDGE_INFO[edge_name]
            segment = _outer_line_info(_segment_cells(top_left, edge_name))[1]
            outward = info["outward"]
            color = int(edge["color"])
            rays.append({"color": color, "cells": _ray_cells(shape, segment[0], outward)})
            rays.append({"color": color, "cells": _ray_cells(shape, segment[-1], outward)})
    return rays


def _paint_ordered_rays(
    base: np.ndarray,
    bg_colors: set[int],
    rays: list[dict[str, object]],
    *,
    stop_on_obstacle: bool,
    overwrite_non_bg: bool,
) -> np.ndarray:
    out = np.array(base, dtype=int, copy=True)
    for ray in rays:
        color = int(ray["color"])
        for r, c in ray["cells"]:
            cell = int(out[r, c])
            if cell in bg_colors or overwrite_non_bg:
                out[r, c] = color
            if stop_on_obstacle and cell not in bg_colors:
                break
    return out


def _build_input_from_layout(h: int, w: int, placements: list[dict[str, object]]) -> np.ndarray | None:
    inp = _checkerboard(h, w)
    occupied: set[tuple[int, int]] = set()

    for placement in placements:
        diamond = _diamond_cells(placement["top_left"])
        if any(not (0 <= r < h and 0 <= c < w) for r, c in diamond):
            return None
        if not occupied.isdisjoint(diamond):
            return None
        occupied.update(diamond)
        _paint_cells(inp, diamond, int(placement["diamond_color"]))

    for placement in placements:
        for edge in placement["edges"]:
            segment = _segment_cells(placement["top_left"], edge["name"])
            if any(not (0 <= r < h and 0 <= c < w) for r, c in segment):
                return None
            if not occupied.isdisjoint(segment):
                return None
            occupied.update(segment)
            _paint_cells(inp, segment, int(edge["color"]))

    bg_colors = {0, 1}
    for placement in placements:
        dset = set(_diamond_cells(placement["top_left"]))
        own_segment_cells = {
            cell
            for edge in placement["edges"]
            for cell in _segment_cells(placement["top_left"], edge["name"])
        }
        for r, c in dset:
            for rr, cc in neighbors8(r, c):
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if (rr, cc) in dset or (rr, cc) in own_segment_cells:
                    continue
                if int(inp[rr, cc]) not in bg_colors:
                    return None

    return inp


def _fusion_metrics(inp: np.ndarray, placements: list[dict[str, object]]) -> dict[str, int]:
    rays = _ordered_rays_from_layout(inp.shape, placements)
    fusion = _paint_ordered_rays(inp, {0, 1}, rays, stop_on_obstacle=False, overwrite_non_bg=True)
    legacy = _paint_ordered_rays(inp, {0, 1}, rays, stop_on_obstacle=True, overwrite_non_bg=False)

    hit_counts: Counter[tuple[int, int]] = Counter()
    post_obstacle_bg = 0
    for ray in rays:
        seen_non_bg = False
        for r, c in ray["cells"]:
            hit_counts[(r, c)] += 1
            if int(inp[r, c]) in (0, 1):
                if seen_non_bg:
                    post_obstacle_bg += 1
            else:
                seen_non_bg = True

    recolored_non_bg = int(np.count_nonzero((inp != fusion) & (inp != 0) & (inp != 1)))
    legacy_delta = int(np.count_nonzero(fusion != legacy))
    multi_hit_cells = sum(v > 1 for v in hit_counts.values())
    return {
        "recolored_non_bg": recolored_non_bg,
        "post_obstacle_bg": post_obstacle_bg,
        "multi_hit_cells": int(multi_hit_cells),
        "legacy_delta": legacy_delta,
    }


def _solve(
    grid: np.ndarray,
    *,
    stop_on_obstacle: bool,
    overwrite_non_bg: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    diamonds = [(col, pts) for col, pts in _components8_by_color(g, non_bg) if _is_diamond8(pts)]

    rays: list[dict[str, object]] = []
    for _col, dpts in sorted(diamonds, key=lambda item: _bbox(item[1])):
        dset = set(dpts)
        adj = []
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
        outer_comps = sorted(_cc8(adj_mask), key=_bbox)

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
            color = int(g[oc[0][0], oc[0][1]])
            rays.append({"color": color, "cells": _ray_cells((h, w), endpoints[0], outward)})
            rays.append({"color": color, "cells": _ray_cells((h, w), endpoints[-1], outward)})

    return _paint_ordered_rays(
        g,
        bg_colors,
        rays,
        stop_on_obstacle=stop_on_obstacle,
        overwrite_non_bg=overwrite_non_bg,
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire diagonal "laser" rays from short outer border segments around hollow diamonds on a checkerboard.

    Input structure:
    - The background is a 0/1 checkerboard, with those two colors dominating the input.
    - Hollow 8-pixel diamonds mark the inner object, and short diagonal segments sit one step farther out.
    - Each outer segment has its own color and is attached to one side of one diamond.

    Rule:
    - For every outer segment, find the diagonal direction perpendicular to the segment and pointing away from its diamond.
    - Emit one ray from each segment endpoint in that outward direction.
    - Unlike the original task, each ray ignores obstacles completely: it runs all the way to the grid boundary and repaints every traversed cell, even if that cell already belonged to another object or an earlier ray.

    Key insight:
    - The only stopping condition is the border of the grid, so diamonds, segments, and previous rays are all just paintable targets.
    - Because later rays overwrite earlier ones, the stable processing order of diamonds and their attached segments determines the final visible color where rays overlap.
    """
    return _solve(grid, stop_on_obstacle=False, overwrite_non_bg=True)


def _make_placement(
    top_left: tuple[int, int],
    diamond_color: int,
    edge_names: list[str],
    edge_colors: list[int],
) -> dict[str, object]:
    ordered_names = sorted(edge_names, key=lambda name: _edge_sort_key(top_left, name))
    return {
        "top_left": top_left,
        "diamond_color": int(diamond_color),
        "edges": [{"name": name, "color": int(color)} for name, color in zip(ordered_names, edge_colors)],
    }


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    gap_max = min(min(h, w) - 7, MANDATORY_GAP_RANGE[1])
    gap_min = MANDATORY_GAP_RANGE[0]
    if gap_max < gap_min:
        return None
    gap = random.randint(gap_min, gap_max)

    a_r_max = h - 6 - gap
    a_c_max = w - 6 - gap
    if a_r_max < 1 or a_c_max < 1:
        return None

    a_top_left = (random.randint(1, a_r_max), random.randint(1, a_c_max))
    b_top_left = (a_top_left[0] + gap, a_top_left[1] + gap)

    a_edges = {"BR"}
    while len(a_edges) < random.choice(EDGE_COUNT_CHOICES):
        a_edges.add(random.choice([name for name in _EDGE_NAMES if name != "BR"]))

    b_edges = {"TL"}
    while len(b_edges) < random.choice(EDGE_COUNT_CHOICES):
        b_edges.add(random.choice([name for name in _EDGE_NAMES if name != "TL"]))

    diamond_colors = random.sample(list(COLOR_RANGE), 2)
    remaining = [c for c in COLOR_RANGE if c not in diamond_colors]
    total_edges = len(a_edges) + len(b_edges)
    if total_edges > len(remaining):
        return None
    segment_colors = random.sample(remaining, total_edges)

    a_color_count = len(a_edges)
    placements = [
        _make_placement(a_top_left, diamond_colors[0], list(a_edges), segment_colors[:a_color_count]),
        _make_placement(b_top_left, diamond_colors[1], list(b_edges), segment_colors[a_color_count:]),
    ]

    inp = _build_input_from_layout(h, w, placements)
    if inp is None:
        return None

    metrics = _fusion_metrics(inp, placements)
    if metrics["recolored_non_bg"] <= 0:
        return None
    if metrics["post_obstacle_bg"] <= 0:
        return None
    if metrics["multi_hit_cells"] <= 0:
        return None
    if metrics["legacy_delta"] < 8:
        return None

    expected = _paint_ordered_rays(
        inp,
        {0, 1},
        _ordered_rays_from_layout(inp.shape, placements),
        stop_on_obstacle=False,
        overwrite_non_bg=True,
    )
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
