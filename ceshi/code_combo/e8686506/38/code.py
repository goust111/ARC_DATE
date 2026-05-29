# Combo variant of e8686506 (index 38)
# combo_spec: D1/4+D8/2
# Name: 面积排序决定位置+碎片面积均等

import random

import numpy as np

from arc_utils import components_mask, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and flow impact
# - Original e8686506 used geometric assembly: one large frame-like 8-connected fragment
#   (or two halves) already sat in output-relative position, so the solver first locked the
#   rectangle bbox, then fit the smaller interior pieces by edge compatibility plus LR color
#   symmetry.
# - D1/4 removes that geometric matching rule: output position is no longer discovered by
#   "which edge fits which hole", but by a global ordering over fragments.
# - D8/2 removes the dominant-size clue: every fragment has equal area, so area ranking
#   collapses and the solver needs a stable secondary rule.
# - Fused behavior here:
#   1. every non-background fragment is a 1xW monochrome bar, and all bars have the same area;
#   2. the bars are ranked by (-area, input top row, input left col), which in practice means
#      top-to-bottom order because the areas are equal and the generator uses distinct rows;
#   3. the ranked bars are written into a dense HxW rectangle on rows ordered center-outward.
# - Small trace on a 5-piece example:
#   old flow:
#     find frame -> deduce 5x5 bbox -> place irregular fillers into the only cells they fit
#   new fused flow:
#     collect five equal 1x5 bars -> see every area is 5 -> tie-break by input order
#     (topmost, second, ...) -> map that ranked list to output rows [center, one above,
#     one below, two above, two below].
#
# b) Complexity calibration
# - The transformation is deliberately simpler than the original jigsaw, but not degenerate:
#   the input-to-output change is still non-identity and the output position depends on a
#   nontrivial center-out permutation rather than plain sorting top-to-bottom.
# - Restricting to four valid dimensions keeps the solver reliable:
#   (5,5), (6,5), (6,6), (7,5). In all four cases, the common fragment area equals the output
#   width and the piece count equals the output height.
# - Using 5-7 bars keeps the palette small enough for human inference while still showing a
#   visible ranked redistribution.
#
# c) Implicit assumption audit against the original main()
# - `_build_requirements(...)` and all corner/bar/vertical-line anchor heuristics are invalid:
#   equal-area bars provide no frame markers.
# - The 8-connected frame detection logic is invalid: no input component spans the output bbox.
# - The propagation-only tiler is unnecessary and would stall, because many row placements are
#   legal unless one first infers the ordering rule.
# - New assumptions introduced by this fused variant:
#   1. every intended fragment is a single 4-connected monochrome bar of height 1;
#   2. all fragment areas are equal (or differ by at most 1, though the generator uses exact
#      equality);
#   3. the common area equals the output width;
#   4. the number of fragments equals the output height;
#   5. the output row assignment order is center-outward.
# - The generator constructs samples to satisfy all five assumptions explicitly.
#
# d) Placement legality
# - The generator builds the output first, then cuts each row into one standalone bar.
# - Input bars are pasted onto distinct rows with at least one blank row between them, so they
#   never overlap and never become 8-neighbor-adjacent.
# - Oracle acceptance still requires `main(input) == intended_output`.
#
# e) Visual observability
# - The key observable change from the original task is that there is no large frame fragment
#   anywhere in the input; instead, all pieces are visibly the same size.
# - Across 2-4 examples, a human can compare the top-to-bottom order of input bars against the
#   center-out order of output rows and infer the hidden tie-break without needing invisible
#   information such as numeric color ids.
#
# f) Visual cleanliness
# - Every object is a flat bar; no overlapping markers, no nested cues, no micro-objects.
# - Distinct rows plus random columns make the input look scattered but still easy to parse.
# - The output is a solid portrait rectangle with one color per row, so the reordering effect is
#   immediately legible.
#
# g) Inferability of the changed rule
# - Color does not drive the ordering rule. Colors are only payloads carried by the bars.
# - The structural cue is consistent across every sample:
#   equal-size bars + no frame + output rows populated from center outward by input order.
# - This avoids the non-observable failure mode where a solver would need to know that
#   "maroon < teal" numerically; the secondary key is purely spatial.
#
# New-solver assumption audit
# - `_extract_pieces(...)` assumes each non-bg component is one fragment. The generator uses one
#   paste per row-bar and row spacing >= 2, so components never merge.
# - `_infer_dims(...)` assumes the common bar area, the fragment count, and the closest-square
#   factor pair all agree. The dimension pool is chosen precisely so this identity holds.
# - The main loop that fills `out[row, :]` assumes each ranked fragment becomes one full output
#   row. The generator's output-first construction enforces this by definition.


_VALID_DIMS = [(5, 5), (6, 5), (6, 6), (7, 5)]
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_GEN_MAX_RETRIES = 300


def _center_out_rows(height: int) -> list[int]:
    rows = []
    left = (height - 1) // 2
    right = height // 2
    while left >= 0 or right < height:
        if left == right:
            rows.append(left)
        else:
            if left >= 0:
                rows.append(left)
            if right < height:
                rows.append(right)
        left -= 1
        right += 1
    return rows


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            out.append((n // w, w))
    return out


def _infer_dims(total_area: int) -> tuple[int, int]:
    dims = [(h, w) for h, w in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist"
    best_gap = min(abs(h - w) for h, w in dims)
    best = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best) == 1, "closest-to-square portrait factor pair must be unique"
    return best[0]


def _extract_pieces(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    pieces = []
    scan_idx = 0
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=4):
            arr = np.asarray(pts, dtype=int)
            rs = arr[:, 0]
            cs = arr[:, 1]
            top = int(rs.min())
            left = int(cs.min())
            height = int(rs.max() - top + 1)
            width = int(cs.max() - left + 1)
            pieces.append(
                {
                    "color": int(col),
                    "area": int(len(arr)),
                    "top": top,
                    "left": left,
                    "height": height,
                    "width": width,
                    "idx": scan_idx,
                }
            )
            scan_idx += 1
    return bg, pieces


def _rank_pieces(pieces: list[dict]) -> list[dict]:
    return sorted(
        pieces,
        key=lambda p: (-int(p["area"]), int(p["top"]), int(p["left"]), int(p["idx"])),
    )


def _build_output_from_ranked(pieces: list[dict], H: int, W: int) -> np.ndarray:
    order = _center_out_rows(H)
    out = np.full((H, W), -1, dtype=int)
    for piece, row in zip(pieces, order):
        out[row, :] = int(piece["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a dense left-right symmetric rectangle by ranking equal-size fragments.

    Input structure:
    - The background is the most common color.
    - Every non-background object is a separated monochrome bar, and all bars have the same
      or nearly the same area; there is no frame fragment left in place.

    Rule:
    - Extract every non-background 4-connected component and compare their areas.
    - Use descending area as the primary sort key; because the generated bars are equal-size,
      ties are resolved by input position from top to bottom, then left to right.
    - Infer the portrait output dimensions from the total area and write the ranked colors into
      full-width rows ordered from the vertical center outward.

    Key insight:
    - The task no longer asks "which shape fits which hole"; the fragments act as ranked tokens.
      Once equal area removes the size cue, the hidden but visible secondary rule is the input
      reading order, and that order controls which color gets the center rows first.
    """
    g = np.asarray(grid, dtype=int)
    _bg, pieces = _extract_pieces(g)
    assert pieces, "grid must contain non-background fragments"

    areas = [int(p["area"]) for p in pieces]
    assert max(areas) - min(areas) <= 1, "variant expects equal-area or near-equal fragments"
    assert all(int(p["height"]) == 1 for p in pieces), "fragments must be single-row bars"

    total_area = sum(areas)
    H, W = _infer_dims(total_area)
    assert H == len(pieces), "piece count must determine output height in this variant"
    assert max(areas) == W, "common fragment area must equal output width"
    assert all(int(p["width"]) == int(p["area"]) for p in pieces), "bars must be solid rows"

    ranked = _rank_pieces(pieces)
    out = _build_output_from_ranked(ranked, H, W)
    assert np.array_equal(out, out[:, ::-1]), "row bars must yield LR symmetry"
    return out.astype(int)


def _choose_colors(bg: int, count: int) -> list[int]:
    colors = [c for c in range(10) if c != bg]
    chosen = random.sample(colors, count)
    random.shuffle(chosen)
    return chosen


def _make_output() -> tuple[np.ndarray, int, list[int]]:
    H, W = random.choice(_VALID_DIMS)
    bg = random.randint(0, 9)
    ranked_colors = _choose_colors(bg, H)
    out = _build_output_from_ranked(
        [{"color": col, "area": W, "top": i, "left": 0, "idx": i} for i, col in enumerate(ranked_colors)],
        H,
        W,
    )
    return out.astype(int), bg, ranked_colors


def _scatter_ranked_bars(ranked_colors: list[int], width: int, bg: int) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    assert cols >= width

    canvas = np.full((rows, cols), bg, dtype=int)
    candidate_rows = list(range(0, rows, 2))
    assert len(candidate_rows) >= len(ranked_colors), "canvas must have enough spaced rows"
    chosen_rows = sorted(random.sample(candidate_rows, len(ranked_colors)))

    for row, col in zip(chosen_rows, ranked_colors):
        patch = np.full((1, width), int(col), dtype=int)
        left = random.randint(0, cols - width)
        paste(canvas, patch, int(row), int(left))
    return canvas


def _piece_areas(grid: np.ndarray) -> list[int]:
    _bg, pieces = _extract_pieces(grid)
    return sorted(int(p["area"]) for p in pieces)


def _gen_attempt() -> np.ndarray | None:
    out, bg, ranked_colors = _make_output()
    if not np.array_equal(out, out[:, ::-1]):
        return None

    try:
        inp = _scatter_ranked_bars(ranked_colors, out.shape[1], bg)
        got = main(inp)
    except Exception:
        return None

    if _piece_areas(inp).count(out.shape[1]) != out.shape[0]:
        return None
    if np.array_equal(got, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 combo input")
