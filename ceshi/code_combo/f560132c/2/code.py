# Combo variant of f560132c (index 2)
# combo_spec: D1/1+D13/0
# Name: 面积排序匹配+简单矩形精灵

import math
import random
from itertools import permutations, product

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# spec_variant:
#   task: Keep the original "scatter 4 pieces + one embedded 2x2 legend palette" presentation,
#     but replace irregular polyominoes with plain rectangles and replace shape-based color matching
#     with visible area ranking.
#   preserved:
#     - 4 disconnected sprites on a black background, exactly one of them is the multicolor legend.
#     - Output is still the tight assembled 2x2 composite rectangle.
#     - Legend colors are still read from a 2x2 block in row-major order.
#   fused_changes:
#     - D13/0: every piece is a filled rectangle; the assembled object is a rectangle split by one
#       horizontal and one vertical cut, so the four destination regions are also rectangles.
#     - D1/1: after the rectangle is reassembled, colors are assigned by piece area rank rather than
#       by where a piece lands. Largest, second, third, smallest map to legend cells
#       top-left, top-right, bottom-left, bottom-right.
#   concrete_trace:
#     - Example assembled cuts: top height 5, bottom height 3, left width 6, right width 4.
#     - The four quadrant rectangles are therefore 5x6, 5x4, 3x6, 3x4 with areas 30, 20, 18, 12.
#     - Suppose the legend palette seen in the input is [[8, 2], [4, 7]].
#     - Geometry step: the legend piece is fixed to the assembled top-left corner; the other three
#       rectangles are placed by width/height compatibility, not by jagged contour matching.
#     - Color step: the 30-cell region gets 8, the 20-cell region gets 2, the 18-cell region gets 4,
#       and the 12-cell region gets 7, even if those regions land in different quadrants than this
#       descending area order.
#     - This is the key difference from the original task: in the old rule the top-right region would
#       always get the top-right legend color, but here top-right may instead get the color for the
#       third-largest piece if its area rank is third.
#   complexity_calibration:
#     - Plain rectangles are much easier to compare than irregular polyominoes, so the generator must
#       avoid near-ties. It enforces 4 distinct areas, a minimum adjacent rank gap, and a minimum
#       total spread to keep the size-order clue visually usable.
#     - Simple rectangles can also create assembly ambiguity if widths/heights repeat too much.
#       The generator therefore accepts only samples for which the rectangle-layout search finds
#       exactly one valid 2x2 arrangement with the legend anchored at the assembled top-left corner.
#     - To stop the combo from collapsing back to the original "color by quadrant" behavior, samples
#       reject the identity case where area-rank order already equals row-major quadrant order.
#   main_assumption_audit:
#     - Legend detection by "most distinct colors" still works because only the legend contains the
#       embedded 2x2 palette.
#     - The old exact-cover search over arbitrary polyomino masks is no longer appropriate: every
#       component now must fill its bounding box completely, and each piece has only rectangle
#       orientations (h,w) or (w,h).
#     - The old centroid-to-quadrant recolor rule is invalid under D1/1. main() must instead compute
#       strict area ranks from visible rectangle areas.
#     - The original code assumed shape uniqueness can recover destination identity. That assumption is
#       broken here because rectangles can be confused by contour alone. The replacement assumption is:
#       given the legend rectangle at top-left with its observed orientation preserved, one
#       permutation/orientation of the remaining three rectangles uniquely closes the 2x2 cut layout.
#   placement_legality:
#     - All four sprites are scattered with random_free_location using border/padding, so they do not
#       overlap or touch unexpectedly.
#     - The legend palette is always placed strictly inside the legend rectangle with a one-cell body
#       border, so the legend remains a single connected rectangular component.
#     - Every generated scene is validated by main(input) before acceptance.
#   visual_observability:
#     - Ranking depends only on occupied-cell counts, which are directly visible to humans.
#     - The rectangles make the size comparison easier, while the legend still provides the 2x2 color
#       lookup structure needed to infer the rank->color mapping from a few examples.
#     - Because rank order is forced to differ from quadrant order, the output colors cannot be
#       explained by "read color from final position"; the viewer must notice the size ordering.
#   cleanliness:
#     - Each component is a clean filled rectangle; there are no internal holes, staircase edges, or
#       overlapping annotations.
#     - The output is a single neat 2x2 block with four rectangular regions, so the scene remains
#       readable despite the extra rank-based logic.
#   fusion_metric:
#     - The generator tracks whether the descending area order of the four pieces differs from the
#       row-major quadrant order [TL, TR, BL, BR]. This metric must be true for accepted samples;
#       otherwise the combo would visually degenerate to the original quadrant-color behavior.
#   new_main_assumptions_checked_against_generator:
#     - main() assumes exactly 4 components and k=2; generate_input() always constructs 4 pieces.
#     - main() assumes each component is a filled rectangle; generate_input() creates only rectangles.
#     - main() assumes areas are strictly ordered; generate_input() rejects ties and tiny gaps.
#     - main() assumes the legend keeps its observed orientation and that a unique rectangle layout
#       exists around that anchor; generate_input() preserves the legend orientation and explicitly
#       tests that uniqueness before scattering a sample.


K = 2
NUM_PIECES = K * K
NUM_COLORS = 8
SIDE_CHOICES = [8, 10, 12]
AREA_FRAC_LO = 0.08
AREA_FRAC_HI = 0.43
AREA_ABS_LO = 11
AREA_ABS_HI = 62
INPUT_SIZE_LO = 20
INPUT_SIZE_HI = 26
DENSITY_LO = 0.15
DENSITY_HI = 0.25
SCATTER_BORDER = 1
SCATTER_PADDING = 1
GEN_RETRIES = 400
AREA_RANK_GAP = 3
AREA_SPREAD_MIN = 8
LEGEND_MARGIN = 1


def _extract_legend_map(g: np.ndarray, pts: np.ndarray, k: int) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    hh, ww = sub.shape
    assert k > 0 and hh >= k and ww >= k, "legend component must fit k×k map"
    best = None
    best_rc = None
    for rr in range(hh - k + 1):
        for cc in range(ww - k + 1):
            window = sub[rr : rr + k, cc : cc + k]
            distinct = int(len(set(map(int, window.ravel()))))
            key = (distinct, -rr, -cc)
            if best is None or key > best:
                best = key
                best_rc = (rr, cc)
    assert best_rc is not None, "legend component must contain a valid k×k color map window"
    rr, cc = best_rc
    return np.asarray(sub[rr : rr + k, cc : cc + k], dtype=int).copy()


def _row_major_cells(k: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(k) for c in range(k)]


def _rect_orientations(h: int, w: int) -> list[tuple[int, int]]:
    if h == w:
        return [(int(h), int(w))]
    return [(int(h), int(w)), (int(w), int(h))]


def _piece_records_from_grid(g: np.ndarray) -> tuple[list[dict], int, np.ndarray, int]:
    bg = get_bg(g)
    comps = components_mask(g != bg, connectivity=4)
    assert len(comps) == NUM_PIECES, "combo expects exactly 4 components"
    k = int(round(len(comps) ** 0.5))
    assert k == K and k * k == len(comps), "combo keeps the original 2×2 legend structure"

    comp_colors = []
    for pts in comps:
        cols = set(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg)
        comp_colors.append(cols)

    max_distinct = max(len(cols) for cols in comp_colors)
    legend_candidates = [i for i, cols in enumerate(comp_colors) if len(cols) == max_distinct]
    assert len(legend_candidates) == 1, "legend component must be uniquely identified by distinct colors"
    legend_idx = legend_candidates[0]
    legend_map = _extract_legend_map(g, comps[legend_idx], k)

    pieces: list[dict] = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        area = int(len(pts))
        assert area == h * w, "every combo piece must be a filled rectangle"
        pieces.append(
            {
                "dims": (h, w),
                "oris": _rect_orientations(h, w),
                "area": area,
            }
        )
    return pieces, legend_idx, legend_map, bg


def _find_layout(
    pieces: list[dict],
    legend_idx: int,
) -> dict[int, tuple[int, int, int, int]] | None:
    """
    Return a unique 2x2 rectangle layout anchored by the legend at top-left.

    The result maps piece index -> (row0, col0, h, w).
    """
    other_pis = [pi for pi in range(len(pieces)) if pi != legend_idx]
    solutions: list[dict[int, tuple[int, int, int, int]]] = []

    for lh, lw in [pieces[legend_idx]["dims"]]:
        for tr_pi, bl_pi, br_pi in permutations(other_pis):
            for (trh, trw), (blh, blw), (brh, brw) in product(
                pieces[tr_pi]["oris"],
                pieces[bl_pi]["oris"],
                pieces[br_pi]["oris"],
            ):
                if trh != lh:
                    continue
                if blw != lw:
                    continue
                if brh != blh or brw != trw:
                    continue
                layout = {
                    legend_idx: (0, 0, int(lh), int(lw)),
                    tr_pi: (0, int(lw), int(trh), int(trw)),
                    bl_pi: (int(lh), 0, int(blh), int(blw)),
                    br_pi: (int(lh), int(lw), int(brh), int(brw)),
                }
                sig = tuple(sorted((pi, *vals) for pi, vals in layout.items()))
                if all(tuple(sorted((pi, *v) for pi, v in sol.items())) != sig for sol in solutions):
                    solutions.append(layout)

    if len(solutions) != 1:
        return None
    return solutions[0]


def _rank_color_map(pieces: list[dict], legend_map: np.ndarray, k: int) -> dict[int, int]:
    areas = [int(p["area"]) for p in pieces]
    assert len(set(areas)) == len(areas), "piece areas must be strictly ordered for size-based recoloring"
    ranked_pis = sorted(range(len(pieces)), key=lambda pi: (-int(pieces[pi]["area"]), int(pi)))
    rank_cells = _row_major_cells(k)
    return {
        pi: int(legend_map[rank_cells[rank][0], rank_cells[rank][1]])
        for rank, pi in enumerate(ranked_pis)
    }


def _render_layout(
    layout: dict[int, tuple[int, int, int, int]],
    piece_colors: dict[int, int],
) -> np.ndarray:
    H = max(r0 + h for r0, _c0, h, _w in layout.values())
    W = max(c0 + w for _r0, c0, _h, w in layout.values())
    out = np.zeros((int(H), int(W)), dtype=int)
    for pi, (r0, c0, h, w) in layout.items():
        out[int(r0) : int(r0 + h), int(c0) : int(c0 + w)] = int(piece_colors[pi])
    return out


def _solve_combo(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    pieces, legend_idx, legend_map, _bg = _piece_records_from_grid(g)
    layout = _find_layout(pieces, legend_idx)
    assert layout is not None, "rectangle layout must be uniquely recoverable from legend anchor and side lengths"
    piece_colors = _rank_color_map(pieces, legend_map, K)
    out = _render_layout(layout, piece_colors)

    pos_order = []
    pos_to_name = {
        (0, 0): "TL",
        (0, 1): "TR",
        (1, 0): "BL",
        (1, 1): "BR",
    }
    ordered_positions = sorted(layout.items(), key=lambda item: (item[1][0], item[1][1]))
    for pi, _vals in ordered_positions:
        pos_order.append(int(pi))

    area_order = sorted(range(len(pieces)), key=lambda pi: (-int(pieces[pi]["area"]), int(pi)))
    info = {
        "legend_idx": int(legend_idx),
        "areas": [int(p["area"]) for p in pieces],
        "area_order": [int(pi) for pi in area_order],
        "position_order": [int(pi) for pi in pos_order],
        "rank_matches_position": bool(area_order == pos_order),
        "layout_named": {
            pos_to_name[(r0 > 0, c0 > 0)]: {"piece": int(pi), "dims": (int(h), int(w))}
            for pi, (r0, c0, h, w) in layout.items()
        },
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble four scattered rectangles into one 2×2 rectangle, then recolor them by size rank using the legend sprite.

    Input structure:
    - The background is the most frequent color.
    - Exactly four disconnected non-background components are present; each one is a filled rectangle.
    - One rectangle is the legend piece because it contains an interior 2×2 multicolor palette, while the other three are monochrome.

    Rule:
    - Reconstruct the unique 2×2 rectangle layout with the legend piece anchored at the assembled top-left corner; the legend keeps its observed orientation, while the other rectangles may be turned.
    - Read the legend palette in row-major order: top-left, top-right, bottom-left, bottom-right.
    - Ignore final position when assigning colors: rank the four pieces by area from largest to smallest, and color them with those four legend cells in that order.

    Key insight: Once the pieces become plain rectangles, contour matching no longer identifies which color each region should get. The decisive clue is visible size order, while the legend rectangle's fixed frame plus side-length compatibility only serves to recover the clean 2×2 assembly.
    """
    return _solve_combo(grid)[0]


def _area_bounds(side: int) -> tuple[int, int]:
    total = side * side
    return max(AREA_ABS_LO, math.ceil(AREA_FRAC_LO * total)), min(AREA_ABS_HI, math.floor(AREA_FRAC_HI * total))


def _candidate_cuts(side: int) -> list[tuple[int, int, int, int]]:
    min_area, max_area = _area_bounds(side)
    opts: list[tuple[int, int, int, int]] = []
    for top_h in range(3, side - 2):
        bot_h = side - top_h
        for left_w in range(4, side - 3):
            right_w = side - left_w
            dims = [
                (top_h, left_w),
                (top_h, right_w),
                (bot_h, left_w),
                (bot_h, right_w),
            ]
            areas = [h * w for h, w in dims]
            if any(a < min_area or a > max_area for a in areas):
                continue
            if len(set(areas)) != NUM_PIECES:
                continue
            ordered = sorted(areas, reverse=True)
            if ordered[0] - ordered[-1] < AREA_SPREAD_MIN:
                continue
            if any(ordered[i] - ordered[i + 1] < AREA_RANK_GAP for i in range(NUM_PIECES - 1)):
                continue
            if top_h < K + 2 * LEGEND_MARGIN or left_w < K + 2 * LEGEND_MARGIN:
                continue
            opts.append((int(top_h), int(bot_h), int(left_w), int(right_w)))
    return opts


def _random_orientation_patch(patch: np.ndarray) -> np.ndarray:
    opts = []
    seen = set()
    for rot_k in range(4):
        arr = np.rot90(patch, rot_k)
        key = (arr.shape, tuple(map(int, arr.ravel())))
        if key not in seen:
            seen.add(key)
            opts.append(arr.copy())
    return random.choice(opts)


def _legend_spots(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    spots = []
    for r in range(LEGEND_MARGIN, h - K - LEGEND_MARGIN + 1):
        for c in range(LEGEND_MARGIN, w - K - LEGEND_MARGIN + 1):
            spots.append((int(r), int(c)))
    return spots


def _build_patches(
    dims_by_position: list[tuple[int, int]],
    body_colors: list[int],
    legend_map: np.ndarray,
) -> list[np.ndarray]:
    patches: list[np.ndarray] = []
    for pi, (h, w) in enumerate(dims_by_position):
        patch = np.full((int(h), int(w)), int(body_colors[pi]), dtype=int)
        if pi == 0:
            spots = _legend_spots((int(h), int(w)))
            assert spots, "legend rectangle must support an interior 2×2 palette"
            rr, cc = random.choice(spots)
            patch[rr : rr + K, cc : cc + K] = legend_map
        patches.append(patch)
    return patches


def _scatter_patches(
    patches: list[np.ndarray],
    side: int,
) -> np.ndarray | None:
    total = side * side
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    if nlo > nhi:
        return None
    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    for pi in random.sample(range(NUM_PIECES), NUM_PIECES):
        oriented = patches[pi].copy() if pi == 0 else _random_orientation_patch(patches[pi])
        try:
            box = np.ones(oriented.shape, dtype=int)
            r0, c0 = random_free_location(
                occupied,
                box,
                transparent=0,
                border=SCATTER_BORDER,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        occupied[r0 : r0 + oriented.shape[0], c0 : c0 + oriented.shape[1]] = 1
        paste(grid, oriented, r0, c0, transparent=0)
    return grid


def _expected_output(
    dims_by_position: list[tuple[int, int]],
    legend_map: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    areas = [h * w for h, w in dims_by_position]
    ranked_pis = sorted(range(NUM_PIECES), key=lambda pi: (-areas[pi], pi))
    rank_cells = _row_major_cells(K)
    piece_colors = {
        pi: int(legend_map[rank_cells[rank][0], rank_cells[rank][1]])
        for rank, pi in enumerate(ranked_pis)
    }

    top_h, left_w = dims_by_position[0]
    _top_h2, right_w = dims_by_position[1]
    bottom_h, _left_w2 = dims_by_position[2]
    out = np.zeros((top_h + bottom_h, left_w + right_w), dtype=int)
    placements = {
        0: (0, 0, *dims_by_position[0]),
        1: (0, left_w, *dims_by_position[1]),
        2: (top_h, 0, *dims_by_position[2]),
        3: (top_h, left_w, *dims_by_position[3]),
    }
    for pi, (r0, c0, h, w) in placements.items():
        out[r0 : r0 + h, c0 : c0 + w] = int(piece_colors[pi])
    return out, ranked_pis


def _gen_attempt(return_info: bool = False):
    side = random.choice(SIDE_CHOICES)
    cuts = _candidate_cuts(side)
    if not cuts:
        return None
    top_h, bot_h, left_w, right_w = random.choice(cuts)

    dims_by_position = [
        (top_h, left_w),
        (top_h, right_w),
        (bot_h, left_w),
        (bot_h, right_w),
    ]
    areas = [h * w for h, w in dims_by_position]
    ranked_pis = sorted(range(NUM_PIECES), key=lambda pi: (-areas[pi], pi))
    if ranked_pis == list(range(NUM_PIECES)):
        return None

    colors = random.sample(range(1, 10), NUM_COLORS)
    body_colors = colors[:NUM_PIECES]
    palette_colors = colors[NUM_PIECES:]
    random.shuffle(palette_colors)
    legend_map = np.array(palette_colors, dtype=int).reshape(K, K)

    expected, ranked_pis = _expected_output(dims_by_position, legend_map)
    patches = _build_patches(dims_by_position, body_colors, legend_map)
    grid = _scatter_patches(patches, side)
    if grid is None:
        return None

    solved, info = _solve_combo(grid)
    if not np.array_equal(solved, expected):
        return None
    if info["rank_matches_position"]:
        return None

    if return_info:
        info.update(
            {
                "side": int(side),
                "cuts": (int(top_h), int(bot_h), int(left_w), int(right_w)),
                "expected_rank_order": [int(pi) for pi in ranked_pis],
            }
        )
        return grid, info
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt(return_info=False)
        if result is not None:
            return result
    raise RuntimeError("failed to generate f560132c combo input")
