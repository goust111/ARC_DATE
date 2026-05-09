# Combo variant of a6f40cea (index 7)
# combo_spec: D3/1+D11/2
# Name: 边缘像素颜色配对

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
   - Original task: an outside open rectangle touches the frame ring in two places, and the solver
     projects those touch coordinates inward to recover one interior path.
   - D3/1 removes the large outside geometry and keeps only the essential endpoint code: single pixels
     on the outer border of the whole grid. A top or bottom border pixel projects to the top/bottom
     edge of the framed interior by column; a left or right border pixel projects by row.
   - D11/2 replaces "pick the farthest touch pair from one source object" with "pair by color".
   - In this combo, the important nontrivial case is one corner with four control pixels:
       top(ca)=A, left(ra)=A, top(cb)=B, left(rb)=B.
     The output contains two monochrome L paths, one for color A and one for color B. A distance-only
     rule would collapse this bundle incorrectly, but color pairing recovers both paths.

b) Complexity calibration
   - The generator always includes exactly one paired-corner bundle using two repeated colors. This
     makes the combo behavior visible in every sample instead of appearing only rarely.
   - Remaining structure is limited to at most one extra color group: either one simple monochrome L
     path or one monochrome U-style stripe family. That keeps the puzzle legible.
   - Alternating two-color paths are intentionally removed. In this combo, colors must carry pairing
     information directly, so reusing them simultaneously as an alternation palette would blur the cue.

c) Hidden-assumption review of the original main()
   - Original code assumes meaningful evidence comes from 4-connected outside components and their ring
     touches. That assumption fails completely once the source objects shrink to isolated edge pixels.
   - Original code also assumes one component may need farthest-pair selection. In the combo, geometry
     no longer identifies pairs; color identity does.
   - The only original idea that still survives is the interior routing policy: once two projected
     endpoints are known, prefer the Manhattan elbow whose corner lies strictly inside the frame.

d) Placement legality
   - Control pixels are placed only on the outermost border of the whole grid and never share a cell.
   - Each signal color is reserved for one path family globally, so the solver can group by color
     without ambiguity.
   - Paired-corner paths use monotone ordered row/column choices, which guarantees the two L paths in
     that corner are nested rather than crossing.
   - Distractors stay away from both the frame neighborhood and the outer grid border so they cannot be
     mistaken for controls.

e) Visual observability
   - A human sees a framed rectangle plus a few isolated border dots. Two colors repeat exactly twice in
     one corner-aligned bundle, and the output contains two matching same-color L paths in that corner.
   - Because the repeated colors are the only repeated labels in the control bundle, the pairing rule is
     visible directly in the pixels rather than depending on numeric color ids or hidden ordering.

f) Visual cleanliness
   - Signals live on the global border, the frame sits in the middle, and distractors stay off both.
     The three layers do not visually interfere.
   - The mandatory paired bundle occupies one corner of the interior, while any optional extra structure
     uses a different side pattern or another corner, reducing clutter.

g) New solver assumptions introduced here
   - Every meaningful non-frame color appears either exactly twice (one L/segment) or exactly four times
     on opposite sides (a stripe family). The generator enforces this.
   - No two different path families share a color. The generator enforces unique color ownership.
   - If a four-point color group appears, it is an opposite-side stripe family rather than an ambiguous
     many-to-many corner bundle. The generator enforces that by reserving the corner bundle for two
     distinct colors with two points each.
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


def _line(a: int, b: int) -> range:
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _vpath(r0: int, r1: int, c: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r in _line(r0, r1)]


def _hpath(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for c in _line(c0, c1)]


def _join(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in parts:
        for p in part:
            if not out or out[-1] != p:
                out.append(p)
    return out


def _build_manhattan(
    p1: tuple[int, int],
    p2: tuple[int, int],
    corner_a: bool,
) -> list[tuple[int, int]]:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb:
        return [(int(ra), int(c)) for c in _line(ca, cb)]
    if ca == cb:
        return [(int(r), int(ca)) for r in _line(ra, rb)]
    if corner_a:
        return _join(_vpath(ra, rb, ca), _hpath(rb, ca, cb))
    return _join(_hpath(ra, ca, cb), _vpath(ra, rb, cb))


def _inside_corner(
    p1: tuple[int, int],
    p2: tuple[int, int],
    corner_a: bool,
    out_h: int,
    out_w: int,
) -> bool:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb or ca == cb:
        return True
    rr, cc = ((rb, ca) if corner_a else (ra, cb))
    return (0 < int(rr) < out_h - 1) and (0 < int(cc) < out_w - 1)


def _choose_path(
    p1: tuple[int, int],
    p2: tuple[int, int],
    out_h: int,
    out_w: int,
) -> list[tuple[int, int]]:
    path_a = _build_manhattan(p1, p2, True)
    path_b = _build_manhattan(p1, p2, False)
    if _inside_corner(p1, p2, False, out_h, out_w) and not _inside_corner(p1, p2, True, out_h, out_w):
        return path_b
    return path_a


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], color: int) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    grid[rs, cs] = int(color)


def _collect_edge_controls(
    g: np.ndarray,
    bg: int,
    bc: int,
    fill: int,
    out_h: int,
    out_w: int,
    r0: int,
    c0: int,
) -> dict[int, list[tuple[str, tuple[int, int]]]]:
    H, W = g.shape

    def clamp_row(rr: int) -> int:
        return max(0, min(out_h - 1, int(rr - (r0 + 1))))

    def clamp_col(cc: int) -> int:
        return max(0, min(out_w - 1, int(cc - (c0 + 1))))

    by_color: dict[int, list[tuple[str, tuple[int, int]]]] = defaultdict(list)

    for cc in range(W):
        val = int(g[0, cc])
        if val not in (int(bg), int(bc), int(fill)):
            by_color[val].append(("top", (0, clamp_col(cc))))
        val = int(g[H - 1, cc])
        if val not in (int(bg), int(bc), int(fill)):
            by_color[val].append(("bottom", (out_h - 1, clamp_col(cc))))

    for rr in range(1, H - 1):
        val = int(g[rr, 0])
        if val not in (int(bg), int(bc), int(fill)):
            by_color[val].append(("left", (clamp_row(rr), 0)))
        val = int(g[rr, W - 1])
        if val not in (int(bg), int(bc), int(fill)):
            by_color[val].append(("right", (clamp_row(rr), out_w - 1)))

    return by_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct interior routes from color-paired edge pixels around one framed rectangle.

    Input structure:
    - A uniform background contains one 1px closed rectangular frame with a single fill color.
    - The only path clues are isolated colored pixels on the outer border of the whole grid.
    - Each clue pixel projects inward onto one side of the framed interior by sharing its row or column.

    Rule:
    - Output only the frame interior, initialized to the fill color.
    - Group projected border clues by color; equal colors belong to the same path family.
    - A two-point color group becomes one solid Manhattan path between its projected endpoints.
    - A four-point color group on opposite sides becomes one or more full-span solid stripes.

    Key insight: the missing outside geometry has been compressed into edge coordinates, and color is now
    the pairing signal itself, so repeated colors tell you which projected endpoints must connect.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    out_h = int(r1 - r0 - 1)
    out_w = int(c1 - c0 - 1)
    out = np.full((out_h, out_w), int(fill), dtype=int)

    by_color = _collect_edge_controls(g, int(bg), int(bc), int(fill), out_h, out_w, r0, c0)

    for col, records in by_color.items():
        uniq = sorted({(int(pt[0]), int(pt[1])) for _side, pt in records})
        if not uniq:
            continue

        sides = {side for side, _pt in records}

        if sides <= {"top", "bottom"} and "top" in sides and "bottom" in sides:
            for cc in sorted({c for _r, c in uniq}):
                rs = [r for r, c in uniq if c == cc]
                for rr in range(min(rs), max(rs) + 1):
                    out[int(rr), int(cc)] = int(col)
            continue

        if sides <= {"left", "right"} and "left" in sides and "right" in sides:
            for rr in sorted({r for r, _c in uniq}):
                cs = [c for r, c in uniq if r == rr]
                for cc in range(min(cs), max(cs) + 1):
                    out[int(rr), int(cc)] = int(col)
            continue

        if len(uniq) == 1:
            rr, cc = uniq[0]
            out[int(rr), int(cc)] = int(col)
            continue

        if len(uniq) == 2:
            p1, p2 = uniq
            for rr, cc in _choose_path(p1, p2, out_h, out_w):
                out[int(rr), int(cc)] = int(col)

    return out


_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (4, 13)
_INTERIOR_W_RANGE = (7, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 6
_EXTRA_MODE_CHOICES = ["none", "single_l", "u"]
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 400
_PAIR_SPEC_TRIES = 80
_L_SPEC_TRIES = 80
_U_SPEC_TRIES = 80


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    H, W = grid.shape
    pts = set((int(r), int(c)) for r, c in coords)
    for r, c in pts:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != int(bg):
            return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in pts and 0 <= nr < H and 0 <= nc < W and bool(comp_mask[nr, nc]):
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


def _place_controls(
    grid: np.ndarray,
    edge_mask: np.ndarray,
    controls: list[tuple[tuple[int, int], int]],
    bg: int,
) -> bool:
    pts = [(int(r), int(c)) for (r, c), _col in controls]
    if len(set(pts)) != len(pts):
        return False
    for (r, c), _col in controls:
        r, c = int(r), int(c)
        if int(grid[r, c]) != int(bg):
            return False
        if bool(edge_mask[r, c]):
            return False
    for (r, c), col in controls:
        r, c = int(r), int(c)
        grid[r, c] = int(col)
        edge_mask[r, c] = True
    return True


def _edge_control_for_side(
    side: str,
    idx: int,
    rows: int,
    cols: int,
    r0: int,
    c0: int,
) -> tuple[int, int]:
    if side == "top":
        return 0, int(c0 + 1 + idx)
    if side == "bottom":
        return rows - 1, int(c0 + 1 + idx)
    if side == "left":
        return int(r0 + 1 + idx), 0
    return int(r0 + 1 + idx), cols - 1


def _choose_pair(values: list[int], increasing: bool) -> tuple[int, int] | None:
    if len(values) < 2:
        return None
    a, b = sorted(random.sample(values, 2))
    return (int(a), int(b)) if increasing else (int(b), int(a))


def _sample_paired_corner(
    kind: str,
    H: int,
    W: int,
    rows: int,
    cols: int,
    r0: int,
    c0: int,
) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]]] | None:
    row_inc = kind[0] == "t"
    col_inc = kind[1] == "l"
    side_a = "top" if kind[0] == "t" else "bottom"
    side_b = "left" if kind[1] == "l" else "right"
    edge_r = 0 if kind[0] == "t" else H - 1
    edge_c = 0 if kind[1] == "l" else W - 1

    row_vals = list(range(1, H - 1))
    col_vals = list(range(1, W - 1))

    for _ in range(_PAIR_SPEC_TRIES):
        rows_pair = _choose_pair(row_vals, row_inc)
        cols_pair = _choose_pair(col_vals, col_inc)
        if rows_pair is None or cols_pair is None:
            return None
        ra, rb = rows_pair
        ca, cb = cols_pair

        inner_a = _join(_vpath(edge_r, ra, ca), _hpath(ra, ca, edge_c))
        inner_b = _join(_vpath(edge_r, rb, cb), _hpath(rb, cb, edge_c))
        all_inner = inner_a + inner_b
        if len(set(all_inner)) != len(all_inner):
            continue

        ctrls_a = [
            _edge_control_for_side(side_a, ca, rows, cols, r0, c0),
            _edge_control_for_side(side_b, ra, rows, cols, r0, c0),
        ]
        ctrls_b = [
            _edge_control_for_side(side_a, cb, rows, cols, r0, c0),
            _edge_control_for_side(side_b, rb, rows, cols, r0, c0),
        ]
        return [inner_a, inner_b], [ctrls_a, ctrls_b]

    return None


def _sample_single_l(
    kind: str,
    H: int,
    W: int,
    rows: int,
    cols: int,
    r0: int,
    c0: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"
    side_a = "top" if is_top else "bottom"
    side_b = "left" if is_left else "right"
    edge_r = 0 if is_top else H - 1
    edge_c = 0 if is_left else W - 1

    for _ in range(_L_SPEC_TRIES):
        if is_top:
            r = random.randint(1, max(1, H // 2))
        else:
            r = random.randint(max(1, H // 2), H - 2)
        if is_left:
            c = random.randint(1, max(1, W // 2))
        else:
            c = random.randint(max(1, W // 2), W - 2)

        inner = _join(_vpath(edge_r, r, c), _hpath(r, c, edge_c))
        ctrls = [
            _edge_control_for_side(side_a, c, rows, cols, r0, c0),
            _edge_control_for_side(side_b, r, rows, cols, r0, c0),
        ]
        return inner, ctrls

    return None


def _sample_u_specs(
    orientation: str,
    H: int,
    W: int,
    rows: int,
    cols: int,
    r0: int,
    c0: int,
    used_inner: np.ndarray,
    edge_mask: np.ndarray,
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]]] | None:
    if orientation == "vertical":
        feasible: list[tuple[list[list[tuple[int, int]]], list[tuple[int, int]]]] = []
        candidates = [int(c) for c in range(W) if not bool(used_inner[:, c].any())]
        pairs = [(int(a), int(b)) for i, a in enumerate(candidates) for b in candidates[i + 1 :] if int(b) - int(a) >= 2]
        for c_a, c_b in pairs:
            ctrls = [
                _edge_control_for_side("top", c_a, rows, cols, r0, c0),
                _edge_control_for_side("bottom", c_a, rows, cols, r0, c0),
                _edge_control_for_side("top", c_b, rows, cols, r0, c0),
                _edge_control_for_side("bottom", c_b, rows, cols, r0, c0),
            ]
            if any(bool(edge_mask[int(rr), int(cc)]) for rr, cc in ctrls):
                continue
            inner = [[(int(r), int(c_a)) for r in range(H)], [(int(r), int(c_b)) for r in range(H)]]
            feasible.append((inner, ctrls))
        if not feasible:
            return None
        return random.choice(feasible)

    feasible = []
    candidates = [int(r) for r in range(H) if not bool(used_inner[r, :].any())]
    pairs = [(int(a), int(b)) for i, a in enumerate(candidates) for b in candidates[i + 1 :] if int(b) - int(a) >= 2]
    for r_a, r_b in pairs:
        ctrls = [
            _edge_control_for_side("left", r_a, rows, cols, r0, c0),
            _edge_control_for_side("right", r_a, rows, cols, r0, c0),
            _edge_control_for_side("left", r_b, rows, cols, r0, c0),
            _edge_control_for_side("right", r_b, rows, cols, r0, c0),
        ]
        if any(bool(edge_mask[int(rr), int(cc)]) for rr, cc in ctrls):
            continue
        inner = [[(int(r_a), int(c)) for c in range(W)], [(int(r_b), int(c)) for c in range(W)]]
        feasible.append((inner, ctrls))
    if not feasible:
        return None
    return random.choice(feasible)


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
    size_lo, size_hi = _DISTRACTOR_SIZE_RANGE

    for _ in range(_DISTRACTOR_TRIES):
        size = random.randint(size_lo, size_hi)
        r = random.randint(1, H - 2)
        c = random.randint(1, W - 2)
        coords = [(int(r), int(c))]
        used = {(int(r), int(c))}
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
            used.add((int(p[0]), int(p[1])))
            coords.append((int(p[0]), int(p[1])))
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

    grid = np.full((rows, cols), int(bg), dtype=int)
    grid[r0, c0 : c1 + 1] = int(bc)
    grid[r1, c0 : c1 + 1] = int(bc)
    grid[r0 : r1 + 1, c0] = int(bc)
    grid[r0 : r1 + 1, c1] = int(bc)
    grid[r0 + 1 : r1, c0 + 1 : c1] = int(fill)

    expected = np.full((H, W), int(fill), dtype=int)
    used_inner = np.zeros((H, W), dtype=bool)
    comp_mask = np.zeros_like(grid, dtype=bool)
    edge_mask = np.zeros_like(grid, dtype=bool)

    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    if len(palette) < 3:
        return None

    color_i = 0
    extra_mode = random.choice(_EXTRA_MODE_CHOICES)

    paired_kind = random.choice(["tl", "tr", "bl", "br"])
    paired = _sample_paired_corner(paired_kind, H, W, rows, cols, r0, c0)
    if paired is None:
        return None
    inner_paths, control_pairs = paired
    all_inner = [p for path in inner_paths for p in path]
    if len(set(all_inner)) != len(all_inner):
        return None
    irs, ics = zip(*all_inner)
    if used_inner[irs, ics].any():
        return None

    col_a = int(palette[color_i])
    col_b = int(palette[color_i + 1])
    color_i += 2
    controls = [(control_pairs[0][0], col_a), (control_pairs[0][1], col_a), (control_pairs[1][0], col_b), (control_pairs[1][1], col_b)]
    if not _place_controls(grid, edge_mask, controls, int(bg)):
        return None
    _paint_path(expected, inner_paths[0], int(col_a))
    _paint_path(expected, inner_paths[1], int(col_b))
    used_inner[irs, ics] = True

    if extra_mode != "none":
        extra_color = int(palette[color_i])
        color_i += 1
        if extra_mode == "u":
            orientations = ["vertical", "horizontal"]
            random.shuffle(orientations)
            if H < 5 and "horizontal" in orientations:
                orientations = [o for o in orientations if o != "horizontal"] + ["horizontal"]
            specs = None
            for orientation in orientations:
                specs = _sample_u_specs(orientation, H, W, rows, cols, r0, c0, used_inner, edge_mask)
                if specs is not None:
                    break
            if specs is None:
                return None
            inner_lines, ctrl_pts = specs
            flat_inner = [p for line in inner_lines for p in line]
            irs, ics = zip(*flat_inner)
            if used_inner[irs, ics].any():
                return None
            if not _place_controls(grid, edge_mask, [(pt, extra_color) for pt in ctrl_pts], int(bg)):
                return None
            expected[irs, ics] = int(extra_color)
            used_inner[irs, ics] = True
        else:
            extra_kind_choices = [k for k in ["tl", "tr", "bl", "br"] if k != paired_kind]
            random.shuffle(extra_kind_choices)
            placed = False
            for extra_kind in extra_kind_choices:
                specs = _sample_single_l(extra_kind, H, W, rows, cols, r0, c0)
                if specs is None:
                    continue
                inner, ctrl_pts = specs
                irs, ics = zip(*inner)
                if used_inner[irs, ics].any():
                    continue
                if not _place_controls(grid, edge_mask, [(pt, extra_color) for pt in ctrl_pts], int(bg)):
                    continue
                _paint_path(expected, inner, int(extra_color))
                used_inner[irs, ics] = True
                placed = True
                break
            if not placed:
                return None

    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        if color_i >= len(palette):
            break
        _add_distractor(grid, comp_mask, int(bg), (r0, c0, r1, c1), int(palette[color_i]))
        color_i += 1

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
