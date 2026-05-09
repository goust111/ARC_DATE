# Variant of a6f40cea
# Dimension: D3 — 源对象类型 (src)
# Change: 外部开放矩形框架（L形：2条完整边+2条截断stub；或U形：3条完整边），通过触碰闭合框边来标记端点位置... → 对角像素对暗示端点

from __future__ import annotations

from collections import Counter
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


# Design analysis for the D3 variant
# ----------------------------------
# a) Concrete behavior:
#    - Keep the original closed frame, dominant fill color, cropped output, and internal Manhattan-L completion.
#    - Replace each external open rectangle with a diagonal marker pair tied to one frame corner:
#        * one pixel sits in the outer corner-adjacent cell just outside the frame,
#        * one pixel sits at the intended elbow cell inside the frame.
#      These two pixels are opposite corners of an axis-aligned rectangle that crosses the frame corner.
#      The other two rectangle corners lie on the frame ring and become the two projected path endpoints.
#    - A small TL example:
#        frame corner ring anchor = (r0-1, c0-1)
#        visible pair = outer pixel at (r0-1, c0-1), elbow pixel at (r0+1+r, c0+1+c)
#        inferred missing corners = (r0-1, c0+1+c) and (r0+1+r, c0-1)
#        crop+clamp maps them to output endpoints (0, c) and (r, 0)
#        output path is therefore the L from top edge to left edge via elbow (r, c).
#      Old logic used long open rectangles to hit those two ring cells directly; new logic infers the same
#      endpoints from the rectangle implied by a diagonal pair.
# b) Complexity calibration:
#    - Limit source objects to 2-3 diagonal-pair components, one per frame corner at most.
#    - Keep the elbow in a disjoint corner quadrant of the interior, so pairing is spatially obvious and solver-side
#      matching does not depend on hidden ordering or color ids.
#    - Preserve two appearance classes:
#        * same-color pair -> solid path
#        * two-color pair -> alternating path
#      This keeps the task visually rich enough that the D3 change is noticeable.
# c) Implicit assumptions from the original main() that no longer hold:
#    - Old `comp_mask` + connected-components logic assumed each source object was a 4-connected outside trace.
#      Diagonal pairs are disconnected, so component labeling by shape must be removed.
#    - Old endpoint extraction read colors from ring-touch cells. Those cells no longer carry the marker colors.
#      In the new rule, the outer-corner pixel and inner elbow pixel provide the color information instead.
#    - Old deferred aggregation for monochrome U-pairs relied on multiple same-color touch points on opposite sides.
#      Diagonal pairs always encode one corner-to-corner L, so the U/stripe branch is intentionally dropped.
# d) Placement legality:
#    - The outer marker uses one of four fixed cells `(r0-1,c0-1)`, `(r0-1,c1+1)`, `(r1+1,c0-1)`, `(r1+1,c1+1)`.
#      Therefore only one component can occupy each corner.
#    - The elbow marker is chosen from a disjoint interior corner window:
#        TL top-left, TR top-right, BL bottom-left, BR bottom-right.
#      This guarantees one valid elbow candidate per occupied corner and prevents ambiguity in `main()`.
#    - Distractors stay outside the frame and away from the frame neighborhood, so they never enter the elbow windows.
# e) Visual observability:
#    - The transformation is visible in 2-3 examples because the input no longer has long external frames at all:
#      instead each completed path is hinted by a sparse diagonal rectangle cue spanning across a frame corner.
#    - The inner elbow pixel remains visible in the input, which helps humans see where the finished L should bend.
# f) Visual cleanliness:
#    - No overlap between completed output paths.
#    - No source-object overlap because corners are unique and distractors avoid the frame neighborhood.
#    - Only one inside-frame cue per occupied corner window, so the scene remains readable.
# g) Color inferability:
#    - Same-color pair means the entire inferred L is solid.
#    - Two-color pair means the inferred L alternates between those two colors, with the inside elbow pixel's color
#      preserved at the elbow in the output. This gives a direct visible anchor for alternation phase.
# h) New solver assumptions that the generator must enforce:
#    - If a corner-adjacent outer marker exists, there is exactly one elbow marker in the matching interior window.
#    - Elbows are strictly inside the frame (never on the boundary), so the completed shape is always an actual L.
#    - Distractors do not occupy the four outer anchor cells or the four interior elbow windows.


def _find_framed_rect_uniform_fill(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
) -> tuple[int, int, int, int, int, int] | None:
    """
    Detect the main 1px-thick rectangular frame with a dominant interior fill.

    Returns (r0, c0, r1, c1, border_color, fill_color) for the largest such rectangle, else None.
    """
    if bg is None:
        bg = int(get_bg(g))
    else:
        bg = int(bg)

    best = None  # (area, r0, c0, r1, c1, bc, fill)
    for bc in sorted(set(map(int, g.flatten())) - {int(bg)}):
        lab, n = label(g == int(bc), structure=structure)
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if int(len(pts)) < 8:
                continue

            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < 3 or w < 3:
                continue

            per = int(2 * (h + w) - 4)
            if int(len(pts)) != per:
                continue
            if not (g[r0, c0 : c1 + 1] == int(bc)).all():
                continue
            if not (g[r1, c0 : c1 + 1] == int(bc)).all():
                continue
            if not (g[r0 : r1 + 1, c0] == int(bc)).all():
                continue
            if not (g[r0 : r1 + 1, c1] == int(bc)).all():
                continue

            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size == 0:
                continue
            fill = int(Counter(inner.flatten()).most_common(1)[0][0])
            if fill == int(bc):
                continue
            noise = int(np.count_nonzero(inner != int(fill)))
            if noise > 4:
                continue

            area = int(h * w)
            cand = (area, r0, c0, r1, c1, int(bc), int(fill))
            if best is None or cand[0] > best[0]:
                best = cand

    if best is None:
        return None
    _area, r0, c0, r1, c1, bc, fill = best
    return int(r0), int(c0), int(r1), int(c1), int(bc), int(fill)


def _line(a: int, b: int) -> range:
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _vpath(r0: int, r1: int, c: int) -> list[tuple[int, int]]:
    return [(r, c) for r in _line(r0, r1)]


def _hpath(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    return [(r, c) for c in _line(c0, c1)]


def _join(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in parts:
        for p in part:
            if not out or out[-1] != p:
                out.append(p)
    return out


def _corner_anchor(kind: str, r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    if kind == "tl":
        return int(r0 - 1), int(c0 - 1)
    if kind == "tr":
        return int(r0 - 1), int(c1 + 1)
    if kind == "bl":
        return int(r1 + 1), int(c0 - 1)
    if kind == "br":
        return int(r1 + 1), int(c1 + 1)
    raise ValueError(kind)


def _corner_windows(outH: int, outW: int) -> dict[str, tuple[list[int], list[int]]]:
    top_rows = list(range(1, (outH - 1) // 2 + 1))
    bottom_rows = list(range((outH - 1) // 2 + 1, outH - 1))
    left_cols = list(range(1, (outW - 1) // 2 + 1))
    right_cols = list(range((outW - 1) // 2 + 1, outW - 1))
    return {
        "tl": (top_rows, left_cols),
        "tr": (top_rows, right_cols),
        "bl": (bottom_rows, left_cols),
        "br": (bottom_rows, right_cols),
    }


def _corner_path(kind: str, er: int, ec: int, outH: int, outW: int) -> tuple[list[tuple[int, int]], int]:
    if kind == "tl":
        first = _vpath(0, er, ec)
        second = _hpath(er, ec, 0)
    elif kind == "tr":
        first = _vpath(0, er, ec)
        second = _hpath(er, ec, outW - 1)
    elif kind == "bl":
        first = _vpath(outH - 1, er, ec)
        second = _hpath(er, ec, 0)
    elif kind == "br":
        first = _vpath(outH - 1, er, ec)
        second = _hpath(er, ec, outW - 1)
    else:
        raise ValueError(kind)
    return _join(first, second), len(first) - 1


def _paint_alternating(out: np.ndarray, path: list[tuple[int, int]], a: int, b: int, anchor_idx: int, anchor_color: int) -> None:
    start = int(anchor_color if anchor_idx % 2 == 0 else (b if int(anchor_color) == int(a) else a))
    cur = int(start)
    for rr, cc in path:
        out[int(rr), int(cc)] = int(cur)
        cur = int(b) if cur == int(a) else int(a)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct corner-crossing L paths inside a framed rectangle from diagonal marker pairs.

    Input structure:
    - A background grid contains one 1px rectangular frame whose interior is mostly one fill color,
      with at most a few colored elbow markers inside.
    - Around some frame corners there is one colored pixel just outside the corner, plus one colored pixel
      inside the matching corner quadrant of the frame.

    Rule:
    - Output only the frame interior, initially filled with the frame's fill color.
    - Each outer/inner diagonal pair defines a rectangle across that frame corner.
    - The rectangle's two unseen corners lie on the frame ring; after cropping they become the L-path endpoints.
    - Same-color pairs yield a solid L, while two-color pairs yield an alternating L whose elbow keeps the inner marker color.

    Key insight: The input no longer marks endpoints by touching the frame; instead the solver must infer the two
    missing corners of a diagonal rectangle that spans from the outer corner cue to the visible elbow cue.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    outH = int(r1 - r0 - 1)
    outW = int(c1 - c0 - 1)
    out = np.full((outH, outW), int(fill), dtype=int)

    marker_mask = (g != int(bg)) & (g != int(bc)) & (g != int(fill))
    windows = _corner_windows(outH, outW)

    for kind in ("tl", "tr", "bl", "br"):
        ar, ac = _corner_anchor(kind, r0, c0, r1, c1)
        if not marker_mask[ar, ac]:
            continue

        row_choices, col_choices = windows[kind]
        candidates: list[tuple[int, int]] = []
        for er in row_choices:
            for ec in col_choices:
                rr = r0 + 1 + er
                cc = c0 + 1 + ec
                if marker_mask[rr, cc]:
                    candidates.append((int(er), int(ec)))

        assert len(candidates) == 1
        er, ec = candidates[0]
        inner_r = int(r0 + 1 + er)
        inner_c = int(c0 + 1 + ec)
        outer_color = int(g[ar, ac])
        inner_color = int(g[inner_r, inner_c])
        path, elbow_idx = _corner_path(kind, er, ec, outH, outW)

        if outer_color == inner_color:
            for rr, cc in path:
                out[int(rr), int(cc)] = int(inner_color)
        else:
            _paint_alternating(out, path, int(outer_color), int(inner_color), elbow_idx, int(inner_color))

    return out


# --------------- generator (a6 D3 variant) ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (4, 13)
_INTERIOR_W_RANGE = (7, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 6
_N_COMPONENTS_RANGE = (2, 3)
_DUAL_COLOR_PROB = 0.45
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_CORNER_RETRIES = 50
_GEN_RETRIES = 300


def _can_place_outside(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    H, W = grid.shape
    s = set(coords)
    for r, c in s:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != int(bg):
            return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in s and 0 <= nr < H and 0 <= nc < W and comp_mask[nr, nc]:
                return False
    return True


def _paint_expected_component(expected: np.ndarray, path: list[tuple[int, int]], elbow_idx: int, outer_color: int, inner_color: int) -> None:
    if int(outer_color) == int(inner_color):
        for rr, cc in path:
            expected[int(rr), int(cc)] = int(inner_color)
    else:
        _paint_alternating(expected, path, int(outer_color), int(inner_color), int(elbow_idx), int(inner_color))


def _add_distractor(grid: np.ndarray, comp_mask: np.ndarray, bg: int, frame: tuple[int, int, int, int], color: int) -> None:
    H, W = grid.shape
    r0, c0, r1, c1 = frame
    gap = _DISTRACTOR_CLEARANCE
    sz_lo, sz_hi = _DISTRACTOR_SIZE_RANGE
    for _ in range(_DISTRACTOR_TRIES):
        size = random.randint(sz_lo, sz_hi)
        r = random.randint(1, H - 2)
        c = random.randint(1, W - 2)
        coords = [(r, c)]
        used = {(r, c)}
        for _ in range(size - 1):
            base = random.choice(coords)
            opts = [
                (nr, nc)
                for nr, nc in neighbors4(*base)
                if 1 <= nr < H - 1 and 1 <= nc < W - 1 and (nr, nc) not in used
            ]
            if not opts:
                break
            p = random.choice(opts)
            used.add(p)
            coords.append(p)
        if any(r0 - gap <= rr <= r1 + gap and c0 - gap <= cc <= c1 + gap for rr, cc in coords):
            continue
        if _can_place_outside(grid, coords, bg, comp_mask):
            rs, cs = zip(*coords)
            grid[rs, cs] = int(color)
            comp_mask[rs, cs] = True
            return


def _take_colors(palette: list[int], idx: int, dual: bool) -> tuple[tuple[int, int], int]:
    if dual:
        return (int(palette[idx]), int(palette[idx + 1])), idx + 2
    col = int(palette[idx])
    return (col, col), idx + 1


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_BG_CHOICES)
    bc = _BORDER_COLOR
    fill = bg if random.random() < _FILL_EQ_BG_PROB else random.choice([x for x in range(10) if x not in (bg, bc)])
    outH = random.randint(*_INTERIOR_H_RANGE)
    outW = random.randint(*_INTERIOR_W_RANGE)
    rows = random.randint(max(_GRID_SIZE_RANGE[0], outH + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    cols = random.randint(max(_GRID_SIZE_RANGE[0], outW + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - outH - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - outW - _FRAME_MARGIN_FAR)
    r1, c1 = r0 + outH + 1, c0 + outW + 1

    grid = np.full((rows, cols), bg, dtype=int)
    grid[r0, c0 : c1 + 1] = bc
    grid[r1, c0 : c1 + 1] = bc
    grid[r0 : r1 + 1, c0] = bc
    grid[r0 : r1 + 1, c1] = bc
    grid[r0 + 1 : r1, c0 + 1 : c1] = fill

    expected = np.full((outH, outW), fill, dtype=int)
    used_inner = np.zeros((outH, outW), dtype=bool)
    outside_mask = np.zeros_like(grid, dtype=bool)
    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    windows = _corner_windows(outH, outW)
    n_components = random.randint(*_N_COMPONENTS_RANGE)
    corners = random.sample(["tl", "tr", "bl", "br"], k=n_components)

    max_needed = 2 * n_components
    if len(palette) < max_needed:
        return None
    color_i = 0

    for kind in corners:
        row_choices, col_choices = windows[kind]
        placed = False
        for _ in range(_CORNER_RETRIES):
            er = random.choice(row_choices)
            ec = random.choice(col_choices)
            path, elbow_idx = _corner_path(kind, int(er), int(ec), outH, outW)
            prs, pcs = zip(*path)
            if used_inner[prs, pcs].any():
                continue

            outer_r, outer_c = _corner_anchor(kind, r0, c0, r1, c1)
            inner_r = r0 + 1 + int(er)
            inner_c = c0 + 1 + int(ec)
            if int(grid[outer_r, outer_c]) != int(bg):
                continue
            if int(grid[inner_r, inner_c]) != int(fill):
                continue

            colors, color_i2 = _take_colors(palette, color_i, random.random() < _DUAL_COLOR_PROB)
            outer_color, inner_color = colors

            grid[outer_r, outer_c] = int(outer_color)
            grid[inner_r, inner_c] = int(inner_color)
            outside_mask[outer_r, outer_c] = True
            used_inner[prs, pcs] = True
            _paint_expected_component(expected, path, elbow_idx, int(outer_color), int(inner_color))
            color_i = color_i2
            placed = True
            break
        if not placed:
            return None

    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        color = random.choice([x for x in range(10) if x not in (bg, bc, fill)])
        _add_distractor(grid, outside_mask, bg, (r0, c0, r1, c1), int(color))

    counts = Counter(grid.flatten().tolist())
    if int(counts.most_common(1)[0][0]) != int(bg):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("failed to generate a valid a6f40cea D3 input")
