# Combo variant of a6f40cea (index 37)
# combo_spec: D4/0+D11/2+D20/0
# Name: T形跨组件颜色配对

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


"""
Design analysis for combo variant D4/0 + D11/2 + D20/0

a) Concrete behavior and trace
   - Original task: each outside source was solved locally. A component touched the frame in two places,
     and the solver completed that same component inside the frame.
   - D11/2 changes the endpoint cue from "pick one farthest pair" to "same touch color means match".
   - D20/0 changes the source from a two-touch L-source to a three-touch T-source.
   - Combined rule used here: there are two opposite-corner T-sources. Each source contributes three ring
     touches with three distinct colors. The same three colors reappear once each on the other source, so
     every useful color forms one cross-component pair.
   - Small trace:
       TL source touches = top@(0,6)=A, top@(0,3)=B, left@(2,0)=C
       BR source touches = right@(5,8)=A, bottom@(8,1)=C, bottom@(8,4)=B
     Original logic would try to solve each component independently and would fail to find a meaningful
     local pair. The combo logic groups by color globally and draws A:(0,6)->(5,8), B:(0,3)->(8,4),
     C:(2,0)->(8,1).

b) Complexity calibration
   - Exactly two main T-sources are generated. This is enough to make the cross-component rule visible
     while keeping the grid sparse and easy to parse.
   - Each useful color appears exactly twice on the ring, once per component. That removes many-to-many
     ambiguity and gives repeated evidence for the color-pairing rule within a single sample.
   - The geometry is restricted so the three induced interior paths are pairwise disjoint. This avoids
     color-overwrite artifacts that would make the rule visually noisy.

c) Hidden-assumption review of the original main()
   - The original per-component loop assumed one component contains the two endpoints of one path. That
     is false now: the endpoints of each path live on different components.
   - The original two-color branch treated component colors as an alternation palette for one path. That
     is also false now: colors are labels for three separate cross-component pairings.
   - The original deferred monochrome aggregation merged same-color touches after local decoding. That
     would blur the new rule, so the combo solver instead decodes ring touches globally by color first.

d) Placement legality
   - The generator places one T-source near the top-left corner of the frame and one near the bottom-right.
   - Each source is a connected outside tree with a neutral connector skeleton and three colored terminal
     touches, so the component stays 4-connected while keeping the color cue isolated at the endpoints.
   - Every candidate sample is checked for bounds, background occupancy, 4-neighbor clearance from other
     components, non-overlapping expected interior paths, and solver/output equality.

e) Visual observability
   - A human sees two outside T-shaped objects, each with three colored terminals.
   - The same three colors appear on both objects, but not repeated within either object.
   - In the output, those repeated cross-object colors become the only paths. This makes the "pair across
     components by color" rule directly visible from a few examples without hidden ordering.

f) Visual cleanliness
   - The two useful components stay in opposite corners, so the outside sources do not crowd one another.
   - The three interior paths are chosen to be disjoint by column ordering constraints, so the crop does
     not contain crossing colors or ambiguous overlaps.
   - Distractors, when present, use unused colors and stay away from the frame ring.

g) Change inferability
   - Colors serve a single purpose: equal ring-touch colors match across components.
   - The generator never uses numeric color order as a cue; colors are sampled from a shuffled palette.
   - Because each sample contains three independent same-color cross-component pairings, the rule gets
     repeated support instead of relying on one isolated coincidence.

h) Generator-to-solver contract
   - There are exactly two main ring-touching components, and each has exactly three ring touches.
   - Each useful color appears on exactly two ring touches total, with the two touches belonging to
     different components.
   - The chosen touch coordinates satisfy `cb < cv < ct`, which keeps the three resulting paths disjoint
     under the solver's standard inside-elbow Manhattan construction.

i) New implicit assumptions introduced by the combo
   - The solver assumes dominant ring-touch color is locally unambiguous. The generator guarantees each
     ring touch is a single monochrome terminal.
   - The solver assumes a useful color pair spans two different components. The generator enforces that by
     giving every source three distinct touch colors and reusing those colors only on the opposite source.
   - The solver paints one path per useful color. The generator therefore ensures no useful color appears
     more than twice on the ring.
"""


def _find_framed_rect_uniform_fill(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
) -> tuple[int, int, int, int, int, int] | None:
    """
    Detect the main 1px-thick rectangular frame with uniform interior fill.

    Returns (r0, c0, r1, c1, border_color, fill_color) for the largest such rectangle, else None.
    """
    if bg is None:
        bg = int(get_bg(g))
    else:
        bg = int(bg)

    best = None
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
            if not (inner == int(fill)).all():
                continue

            area = int(h * w)
            cand = (area, r0, c0, r1, c1, int(bc), int(fill))
            if best is None or cand[0] > best[0]:
                best = cand

    if best is None:
        return None
    _area, r0, c0, r1, c1, bc, fill = best
    return int(r0), int(c0), int(r1), int(c1), int(bc), int(fill)


def _frame_ring_cells(H: int, W: int, r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    ring = set()
    for cc in range(c0, c1 + 1):
        for rr in (r0 - 1, r1 + 1):
            if 0 <= rr < H:
                ring.add((int(rr), int(cc)))
    for rr in range(r0, r1 + 1):
        for cc in (c0 - 1, c1 + 1):
            if 0 <= cc < W:
                ring.add((int(rr), int(cc)))
    return ring


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


def _build_manhattan(
    p1: tuple[int, int], p2: tuple[int, int], cornerA: bool
) -> list[tuple[int, int]]:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb:
        step = 1 if cb >= ca else -1
        return [(int(ra), int(c)) for c in range(int(ca), int(cb) + step, step)]
    if ca == cb:
        step = 1 if rb >= ra else -1
        return [(int(r), int(ca)) for r in range(int(ra), int(rb) + step, step)]
    if cornerA:
        step = 1 if rb >= ra else -1
        path = [(int(r), int(ca)) for r in range(int(ra), int(rb) + step, step)]
        step = 1 if cb >= ca else -1
        path += [(int(rb), int(c)) for c in range(int(ca) + step, int(cb) + step, step)]
        return path
    step = 1 if cb >= ca else -1
    path = [(int(ra), int(c)) for c in range(int(ca), int(cb) + step, step)]
    step = 1 if rb >= ra else -1
    path += [(int(r), int(cb)) for r in range(int(ra) + step, int(rb) + step, step)]
    return path


def _inside_corner(
    p1: tuple[int, int], p2: tuple[int, int], cornerA: bool, outH: int, outW: int
) -> bool:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb or ca == cb:
        return True
    corner = (rb, ca) if cornerA else (ra, cb)
    rr, cc = corner
    return (0 < rr < outH - 1) and (0 < cc < outW - 1)


def _chosen_path(
    p1: tuple[int, int], p2: tuple[int, int], outH: int, outW: int
) -> list[tuple[int, int]]:
    pathA = _build_manhattan(p1, p2, True)
    pathB = _build_manhattan(p1, p2, False)
    if _inside_corner(p1, p2, False, outH, outW) and not _inside_corner(p1, p2, True, outH, outW):
        return pathB
    return pathA


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover cross-component color routes from three-touch T-sources around one framed rectangle.

    Input structure:
    - A background grid contains one complete 1px rectangular frame with a uniform interior fill.
    - Outside the frame are 4-connected T-shaped source components, each touching the ring in three places.
    - Useful touch colors repeat across different components rather than repeating within one component.

    Rule:
    - Output only the frame interior, initialized to the frame's fill color.
    - Project each ring-touch cell inward to the corresponding interior-edge coordinate.
    - Collect all ring touches globally by their visible touch color, not by source component.
    - Whenever one color appears on exactly two touches belonging to different components, connect those
      two projected coordinates with a solid Manhattan path, preferring the elbow that lies strictly inside.

    Key insight:
    - The T-shapes are only sources of terminals; they are not solved locally. Repeated touch colors are
      the pairing signal, and the matching happens across components, so one T-source can feed several
      different interior paths at once.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    outH = int(r1 - r0 - 1)
    outW = int(c1 - c0 - 1)
    out = np.full((outH, outW), int(fill), dtype=int)
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    def map_to_out(rr: int, cc: int) -> tuple[int, int]:
        r = max(0, min(outH - 1, int(rr - r_int0)))
        c = max(0, min(outW - 1, int(cc - c_int0)))
        return int(r), int(c)

    def paint_checked(path: list[tuple[int, int]], color: int) -> None:
        for rr, cc in path:
            prev = int(out[rr, cc])
            if prev != int(fill):
                assert prev == int(color)
            out[rr, cc] = int(color)

    comp_mask = (g != int(bg)) & (g != int(fill)) & (g != int(bc))
    clab, _cn = label(comp_mask, structure=STRUCT4)
    ring = _frame_ring_cells(H, W, r0, c0, r1, c1)

    coord_hist: dict[tuple[int, int, int], Counter] = defaultdict(Counter)
    for rr, cc in ring:
        cid = int(clab[rr, cc])
        if cid == 0:
            continue
        mr, mc = map_to_out(rr, cc)
        coord_hist[(int(cid), int(mr), int(mc))][int(g[rr, cc])] += 1

    color_hits: dict[int, list[tuple[tuple[int, int], int]]] = defaultdict(list)
    for (cid, mr, mc), ctr in coord_hist.items():
        dom = int(ctr.most_common(1)[0][0])
        color_hits[dom].append(((int(mr), int(mc)), int(cid)))

    for color, hits in color_hits.items():
        uniq: list[tuple[tuple[int, int], int]] = []
        seen = set()
        for coord, cid in hits:
            key = (int(coord[0]), int(coord[1]), int(cid))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(((int(coord[0]), int(coord[1])), int(cid)))
        if len(uniq) != 2:
            continue
        (p1, cid1), (p2, cid2) = uniq
        if int(cid1) == int(cid2):
            continue
        paint_checked(_chosen_path(p1, p2, outH, outW), int(color))

    return out


# --------------- generator (a6 combo D4/0 + D11/2 + D20/0) ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (4, 13)
_INTERIOR_W_RANGE = (7, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 6
_DISTRACTOR_COUNT_RANGE = (0, 1)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 500
_T_HUB_OFFSET = 3


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], color: int) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    grid[rs, cs] = int(color)


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
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


def _place_component(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    path: list[tuple[int, int]],
    color: int,
    bg: int,
) -> bool:
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, bg, comp_mask):
        return False
    _paint_path(grid, path, int(color))
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _place_colored_component(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    colored_cells: dict[tuple[int, int], int],
    bg: int,
) -> bool:
    coords = list(colored_cells.keys())
    if not coords:
        return False
    if not _can_place(grid, coords, bg, comp_mask):
        return False
    for (r, c), col in colored_cells.items():
        grid[int(r), int(c)] = int(col)
    rs, cs = zip(*coords)
    comp_mask[rs, cs] = True
    return True


def _merge_colored_parts(parts: list[tuple[list[tuple[int, int]], int]]) -> dict[tuple[int, int], int] | None:
    merged: dict[tuple[int, int], int] = {}
    for path, col in parts:
        for p in path:
            if p in merged and int(merged[p]) != int(col):
                return None
            merged[p] = int(col)
    return merged


def _rand_closed(lo: int, hi: int) -> int:
    assert lo <= hi
    return random.randint(int(lo), int(hi))


def _tl_component_specs(
    H: int,
    W: int,
    r0: int,
    c0: int,
    left_row: int,
    vertical_col: int,
    top_right_col: int,
    col_left_bottom: int,
    col_vertical: int,
    col_top_right: int,
    connector_color: int,
) -> tuple[dict[tuple[int, int], int], dict[str, tuple[int, int]]]:
    hub_r = r0 - _T_HUB_OFFSET
    hub_c = c0 - _T_HUB_OFFSET
    top_row = r0 - 2
    left_col = c0 - 2

    top_vertical = (r0 - 1, c0 + 1 + vertical_col)
    top_right = (r0 - 1, c0 + 1 + top_right_col)
    left_touch = (r0 + 1 + left_row, c0 - 1)

    colored: dict[tuple[int, int], int] = {}
    connector_paths = [
        _join(_vpath(hub_r, top_row, hub_c), _hpath(top_row, hub_c, top_vertical[1])),
        _join(_vpath(hub_r, top_row, hub_c), _hpath(top_row, hub_c, top_right[1])),
        _join(_hpath(hub_r, hub_c, left_col), _vpath(hub_r, left_touch[0], left_col)),
    ]
    for path in connector_paths:
        for cell in path:
            colored[cell] = int(connector_color)
    colored[top_vertical] = int(col_vertical)
    colored[top_right] = int(col_top_right)
    colored[left_touch] = int(col_left_bottom)
    return colored, {
        "left_bottom": (int(left_row), 0),
        "vertical": (0, int(vertical_col)),
        "top_right": (0, int(top_right_col)),
    }


def _br_component_specs(
    H: int,
    W: int,
    r1: int,
    c1: int,
    right_row: int,
    bottom_left_col: int,
    vertical_col: int,
    col_left_bottom: int,
    col_vertical: int,
    col_top_right: int,
    connector_color: int,
) -> tuple[dict[tuple[int, int], int], dict[str, tuple[int, int]]]:
    r0 = int(r1 - H - 1)
    c0 = int(c1 - W - 1)
    hub_r = r1 + _T_HUB_OFFSET
    hub_c = c1 + _T_HUB_OFFSET
    bottom_row = r1 + 2
    right_col = c1 + 2

    bottom_left = (r1 + 1, c0 + 1 + bottom_left_col)
    bottom_vertical = (r1 + 1, c0 + 1 + vertical_col)
    right_touch = (r0 + 1 + right_row, c1 + 1)

    colored: dict[tuple[int, int], int] = {}
    connector_paths = [
        _join(_vpath(hub_r, bottom_row, hub_c), _hpath(bottom_row, hub_c, bottom_vertical[1])),
        _join(_vpath(hub_r, bottom_row, hub_c), _hpath(bottom_row, hub_c, bottom_left[1])),
        _join(_hpath(hub_r, hub_c, right_col), _vpath(hub_r, right_touch[0], right_col)),
    ]
    for path in connector_paths:
        for cell in path:
            colored[cell] = int(connector_color)
    colored[bottom_vertical] = int(col_vertical)
    colored[bottom_left] = int(col_left_bottom)
    colored[right_touch] = int(col_top_right)
    return colored, {
        "left_bottom": (H - 1, int(bottom_left_col)),
        "vertical": (H - 1, int(vertical_col)),
        "top_right": (int(right_row), W - 1),
    }


def _add_distractor(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    bg: int,
    frame: tuple[int, int, int, int],
    color: int,
) -> None:
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
        if _place_component(grid, comp_mask, coords, int(color), bg):
            return


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_BG_CHOICES)
    bc = _BORDER_COLOR
    fill = bg if random.random() < _FILL_EQ_BG_PROB else random.choice([x for x in range(10) if x not in (bg, bc)])
    H = random.randint(*_INTERIOR_H_RANGE)
    W = random.randint(*_INTERIOR_W_RANGE)
    rows = random.randint(max(_GRID_SIZE_RANGE[0], H + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    cols = random.randint(max(_GRID_SIZE_RANGE[0], W + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - H - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - W - _FRAME_MARGIN_FAR)
    r1, c1 = r0 + H + 1, c0 + W + 1

    grid = np.full((rows, cols), bg, dtype=int)
    grid[r0, c0 : c1 + 1] = bc
    grid[r1, c0 : c1 + 1] = bc
    grid[r0 : r1 + 1, c0] = bc
    grid[r0 : r1 + 1, c1] = bc
    grid[r0 + 1 : r1, c0 + 1 : c1] = fill

    expected = np.full((H, W), fill, dtype=int)
    comp_mask = np.zeros_like(grid, dtype=bool)

    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    if len(palette) < 5:
        return None
    col_left_bottom, col_vertical, col_top_right = map(int, palette[:3])
    col_tl_connector, col_br_connector = map(int, palette[3:5])

    left_row = _rand_closed(1, max(1, H // 2))
    right_row = _rand_closed(max(1, H // 2), H - 2)

    bottom_left_col, vertical_col, top_right_col = sorted(random.sample(range(1, W - 1), 3))

    tl_colored, tl_pts = _tl_component_specs(
        H,
        W,
        r0,
        c0,
        left_row,
        vertical_col,
        top_right_col,
        col_left_bottom,
        col_vertical,
        col_top_right,
        col_tl_connector,
    )
    br_colored, br_pts = _br_component_specs(
        H,
        W,
        r1,
        c1,
        right_row,
        bottom_left_col,
        vertical_col,
        col_left_bottom,
        col_vertical,
        col_top_right,
        col_br_connector,
    )

    if not _place_colored_component(grid, comp_mask, tl_colored, bg):
        return None
    if not _place_colored_component(grid, comp_mask, br_colored, bg):
        return None

    paths = {
        int(col_left_bottom): _chosen_path(tl_pts["left_bottom"], br_pts["left_bottom"], H, W),
        int(col_vertical): _chosen_path(tl_pts["vertical"], br_pts["vertical"], H, W),
        int(col_top_right): _chosen_path(tl_pts["top_right"], br_pts["top_right"], H, W),
    }

    used_inner: dict[tuple[int, int], int] = {}
    for color, path in paths.items():
        for p in path:
            if p in used_inner and int(used_inner[p]) != int(color):
                return None
            used_inner[p] = int(color)

    for color, path in paths.items():
        _paint_path(expected, path, int(color))

    for extra_color in palette[5 : 5 + random.randint(*_DISTRACTOR_COUNT_RANGE)]:
        _add_distractor(grid, comp_mask, bg, (r0, c0, r1, c1), int(extra_color))

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a6f40cea combo input")
