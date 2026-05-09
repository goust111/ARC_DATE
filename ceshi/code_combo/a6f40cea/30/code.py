# Combo variant of a6f40cea (index 30)
# combo_spec: D3/1+D11/2+D16/0
# Name: 像素颜色配对镜像投影

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
   - Original task:
     one outside connected trace touches the frame ring, the solver clamps those touch cells directly
     onto the nearest interior edge, then chooses one path per component.
   - Fused task:
     the outside trace geometry is removed entirely. Only single control pixels remain on the outer
     border of the whole grid. The solver must:
       1. find those border pixels,
       2. group them by equal color,
       3. mirror-project each control to the opposite interior edge of the framed rectangle,
       4. connect the projected endpoints.
   - Small trace:
     suppose a red control sits on the bottom border under interior column 2, and another red control
     sits on the right border aligned with interior row 3. Direct projection would yield endpoints
     (bottom, col 2) and (row 3, right). The fused rule mirrors each side first, so the endpoints become
     (top, col 2) and (row 3, left), producing the flipped L path instead.
   - A same-color 4-pixel group, e.g. yellow on top+bottom at two columns, projects to two full stripes.
     That is the visible "pair by color" case: one color can encode multiple matched endpoint pairs.

b) Complexity calibration
   - If every color appeared only twice, D11/2 would be too weak: color would merely identify one path.
     Therefore every generated sample contains one 4-control same-color stripe group.
   - If every group were stripes, D16/0 would be invisible because mirroring top<->bottom or left<->right
     does not change a full-span stripe. Therefore every sample also contains at least one mirrored L group.
   - The generator uses only monochrome color groups. That avoids a conflict where color would otherwise
     need to mean both "pair label" and "alternation phase".

c) Hidden-assumption audit of the original main()
   - Original assumption: the source object is a 4-connected component near the frame.
     Broken by D3/1, so the new solver must scan the outer grid border directly.
   - Original assumption: projection is direct clamping through the frame border.
     Broken by D16/0, so the mapping must send top->bottom, bottom->top, left->right, right->left.
   - Original assumption: one component usually collapses to one chosen endpoint pair.
     Broken by D11/2, so the new solver groups by color first and lets one color yield multiple stripes.
   - New combo assumption: for any one color group, the controls form either
     one adjacent-side pair (an L/segment) or one opposite-side stripe family.
     The generator enforces exactly those legible cases.

d) Placement legality
   - Control pixels are restricted to the global outer border and tracked with `edge_mask`.
   - Inner paths are built output-first and tracked with `used_inner`, preventing overlaps.
   - Stripe columns / rows are separated by at least 2 cells to stay visually distinct.
   - Distractors remain off the border and away from the frame so they cannot be mistaken for controls.

e) Visual observability
   - Humans can see the entire control alphabet at a glance: only isolated border pixels carry meaning.
   - Same-color repetition is directly visible and is supported by repeated structure in the output:
     one color may create two stripes, not just one path.
   - Mirroring is also visible because an exterior bottom/right cue can generate an interior top/left L.
     The path appears in the corner opposite to the control sides.

f) Visual cleanliness
   - Controls live on the outermost border, the frame lives in the middle, and distractors stay elsewhere.
     Those three layers do not visually compete.
   - The generator limits itself to 2-3 color groups, with at most two L groups plus one stripe group.
     That keeps the interior readable.

g) Color-rule inferability
   - Color has exactly one role in the fused rule: equal colors belong together.
   - No ordering by color id, no special treatment for black, and no hidden palette semantics are used.
   - Because at least one sample feature uses four same-color controls, the solver cannot plausibly fall
     back to a distance-only rule and still match the outputs.

h) Generator-to-solver contract
   - Every color group has either 2 controls on adjacent sides or 4 controls on opposite sides.
   - L groups always map to a unique elbow-inside Manhattan path.
   - Stripe groups always project to exactly two full-span rows or columns.
   - At least one L group is present in every sample, so mirror projection changes the output relative to
     direct projection on the same input.

i) New implicit assumptions introduced here
   - `main()` assumes border controls do not share cells and do not reuse a color in two incompatible
     layouts. `generate_input()` guarantees that via one-color-per-group sampling.
   - `main()` paints each color group independently and would allow overwrites on malformed inputs.
     The generator prevents this with `used_inner`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


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


def _build_manhattan(
    p1: tuple[int, int],
    p2: tuple[int, int],
    corner_a: bool,
) -> list[tuple[int, int]]:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb:
        step = 1 if cb >= ca else -1
        return [(int(ra), int(c)) for c in range(int(ca), int(cb) + step, step)]
    if ca == cb:
        step = 1 if rb >= ra else -1
        return [(int(r), int(ca)) for r in range(int(ra), int(rb) + step, step)]
    if corner_a:
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
    p1: tuple[int, int],
    p2: tuple[int, int],
    corner_a: bool,
    out_h: int,
    out_w: int,
) -> bool:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb or ca == cb:
        return True
    corner = (rb, ca) if corner_a else (ra, cb)
    rr, cc = corner
    return (0 < rr < out_h - 1) and (0 < cc < out_w - 1)


def _choose_path(
    p1: tuple[int, int],
    p2: tuple[int, int],
    out_h: int,
    out_w: int,
) -> list[tuple[int, int]]:
    path_a = _build_manhattan(p1, p2, True)
    path_b = _build_manhattan(p1, p2, False)
    use_b = _inside_corner(p1, p2, False, out_h, out_w) and not _inside_corner(p1, p2, True, out_h, out_w)
    return path_b if use_b else path_a


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], color: int) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    grid[rs, cs] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover mirrored interior paths from color-paired edge pixels around one framed rectangle.

    Input structure:
    - A background grid contains one largest 1px rectangular frame with a uniform interior fill.
    - Colored single control pixels sit only on the outer border of the whole grid.
    - Equal-colored controls belong to the same source group and may mark either one L path or two stripes.

    Rule:
    - Output only the frame interior, initialized with the fill color.
    - Mirror-project each control onto the opposite interior edge of the frame:
      top->bottom, bottom->top, left->right, right->left.
    - Group the projected endpoints by control color.
    - A two-control adjacent-side group becomes one Manhattan path; a four-control opposite-side group
      becomes full-span stripes at the matched rows or columns.

    Key insight: color no longer decorates geometry; it is the pairing label, and the coordinate cue must
    be read through an opposite-side mirror before any path can be drawn.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    out_h = int(r1 - r0 - 1)
    out_w = int(c1 - c0 - 1)
    out = np.full((out_h, out_w), int(fill), dtype=int)

    def clamp_row(rr: int) -> int:
        return max(0, min(out_h - 1, int(rr - (r0 + 1))))

    def clamp_col(cc: int) -> int:
        return max(0, min(out_w - 1, int(cc - (c0 + 1))))

    def project_control(rr: int, cc: int, side: str) -> tuple[int, int]:
        if side == "top":
            return out_h - 1, clamp_col(cc)
        if side == "bottom":
            return 0, clamp_col(cc)
        if side == "left":
            return clamp_row(rr), out_w - 1
        assert side == "right"
        return clamp_row(rr), 0

    by_color: dict[int, list[tuple[str, tuple[int, int]]]] = defaultdict(list)
    for cc in range(W):
        top_val = int(g[0, cc])
        if top_val not in (int(bg), int(bc), int(fill)):
            by_color[top_val].append(("top", project_control(0, cc, "top")))
        bot_val = int(g[H - 1, cc])
        if bot_val not in (int(bg), int(bc), int(fill)):
            by_color[bot_val].append(("bottom", project_control(H - 1, cc, "bottom")))
    for rr in range(1, H - 1):
        left_val = int(g[rr, 0])
        if left_val not in (int(bg), int(bc), int(fill)):
            by_color[left_val].append(("left", project_control(rr, 0, "left")))
        right_val = int(g[rr, W - 1])
        if right_val not in (int(bg), int(bc), int(fill)):
            by_color[right_val].append(("right", project_control(rr, W - 1, "right")))

    for col, recs in sorted(by_color.items()):
        sides = {side for side, _pt in recs}
        uniq = sorted({(int(pt[0]), int(pt[1])) for _side, pt in recs})
        if not uniq:
            continue

        if sides <= {"top", "bottom"} and ("top" in sides) and ("bottom" in sides):
            cols = sorted({c for _r, c in uniq})
            for cc in cols:
                rows = sorted({r for r, c in uniq if c == cc})
                if len(rows) >= 2:
                    for rr in range(min(rows), max(rows) + 1):
                        out[int(rr), int(cc)] = int(col)
                else:
                    out[int(rows[0]), int(cc)] = int(col)
            continue

        if sides <= {"left", "right"} and ("left" in sides) and ("right" in sides):
            rows = sorted({r for r, _c in uniq})
            for rr in rows:
                cols = sorted({c for r, c in uniq if r == rr})
                if len(cols) >= 2:
                    for cc in range(min(cols), max(cols) + 1):
                        out[int(rr), int(cc)] = int(col)
                else:
                    out[int(rr), int(cols[0])] = int(col)
            continue

        if len(uniq) == 1:
            rr, cc = uniq[0]
            out[int(rr), int(cc)] = int(col)
            continue

        if len(uniq) == 2:
            p1, p2 = uniq
        else:
            best = None
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    d = abs(uniq[i][0] - uniq[j][0]) + abs(uniq[i][1] - uniq[j][1])
                    if best is None or d > best[0]:
                        best = (d, uniq[i], uniq[j])
            assert best is not None
            _dist, p1, p2 = best

        for rr, cc in _choose_path(p1, p2, out_h, out_w):
            out[int(rr), int(cc)] = int(col)

    return out


# --------------- generator (combo variant) ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (4, 13)
_INTERIOR_W_RANGE = (7, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 6
_N_GROUPS_RANGE = (2, 3)
_STRIPE_GAP_MIN = 2
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 400
_L_SPEC_TRIES = 120
_STRIPE_SPEC_TRIES = 120
_L_KIND_CYCLE: list[str] = []
_STRIPE_ORIENTATION_CYCLE: list[str] = []
_N_GROUP_CYCLE: list[int] = []


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


def _draw_balanced_choice(cache_name: str, options: list[int] | list[str]) -> int | str:
    cache = globals()[cache_name]
    if not cache:
        cache = list(options)
        random.shuffle(cache)
        globals()[cache_name] = cache
    return globals()[cache_name].pop()


def _can_place_blob(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    h, w = grid.shape
    s = set(coords)
    for r, c in s:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != int(bg):
            return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in s and 0 <= nr < h and 0 <= nc < w and comp_mask[nr, nc]:
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
    if not _can_place_blob(grid, path, bg, comp_mask):
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
    pts = [pt for pt, _col in controls]
    if len(set(pts)) != len(pts):
        return False
    for (r, c), _col in controls:
        if int(grid[r, c]) != int(bg):
            return False
        if bool(edge_mask[r, c]):
            return False
    for (r, c), col in controls:
        grid[r, c] = int(col)
        edge_mask[r, c] = True
    return True


def _sample_l_specs(
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
    for _ in range(_L_SPEC_TRIES):
        if is_top:
            r = random.randint(1, max(1, H // 2))
        else:
            r = random.randint(max(1, H // 2), H - 2)
        if is_left:
            c = random.randint(1, max(1, W // 2))
        else:
            c = random.randint(max(1, W // 2), W - 2)

        edge_r = 0 if is_top else H - 1
        edge_c = 0 if is_left else W - 1
        inner = _join(_vpath(edge_r, r, c), _hpath(r, c, edge_c))

        if kind == "tl":
            controls = [(rows - 1, c0 + 1 + c), (r0 + 1 + r, cols - 1)]
        elif kind == "tr":
            controls = [(rows - 1, c0 + 1 + c), (r0 + 1 + r, 0)]
        elif kind == "bl":
            controls = [(0, c0 + 1 + c), (r0 + 1 + r, cols - 1)]
        else:
            controls = [(0, c0 + 1 + c), (r0 + 1 + r, 0)]
        return inner, controls
    return None


def _sample_stripe_specs(
    orientation: str,
    H: int,
    W: int,
    rows: int,
    cols: int,
    r0: int,
    c0: int,
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]]] | None:
    for _ in range(_STRIPE_SPEC_TRIES):
        if orientation == "vertical":
            if W < 4:
                return None
            c_a, c_b = sorted(random.sample(range(1, W - 1), 2))
            if c_b - c_a < _STRIPE_GAP_MIN:
                continue
            inner = [[(r, c_a) for r in range(H)], [(r, c_b) for r in range(H)]]
            controls = [
                (0, c0 + 1 + c_a),
                (rows - 1, c0 + 1 + c_a),
                (0, c0 + 1 + c_b),
                (rows - 1, c0 + 1 + c_b),
            ]
            return inner, controls

        if H < 4:
            return None
        r_a, r_b = sorted(random.sample(range(1, H - 1), 2))
        if r_b - r_a < _STRIPE_GAP_MIN:
            continue
        inner = [[(r_a, c) for c in range(W)], [(r_b, c) for c in range(W)]]
        controls = [
            (r0 + 1 + r_a, 0),
            (r0 + 1 + r_a, cols - 1),
            (r0 + 1 + r_b, 0),
            (r0 + 1 + r_b, cols - 1),
        ]
        return inner, controls
    return None


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
        r, c = random.randint(1, H - 2), random.randint(1, W - 2)
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


def _gen_attempt(
    *,
    forced_n_groups: int | None = None,
    forced_l_kind: str | None = None,
    forced_stripe_orientation: str | None = None,
) -> np.ndarray | None:
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
    used_inner = np.zeros((H, W), dtype=bool)
    edge_mask = np.zeros_like(grid, dtype=bool)
    comp_mask = np.zeros_like(grid, dtype=bool)
    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    if len(palette) < 3:
        return None
    color_i = 0

    n_groups = int(forced_n_groups) if forced_n_groups is not None else random.randint(*_N_GROUPS_RANGE)

    mirrored_l_kind = str(forced_l_kind) if forced_l_kind is not None else random.choice(["tl", "tr", "bl", "br"])
    l_specs = _sample_l_specs(mirrored_l_kind, H, W, rows, cols, r0, c0)
    if l_specs is None:
        return None
    inner_l, ctrl_l = l_specs
    irs, ics = zip(*inner_l)
    if used_inner[irs, ics].any():
        return None
    l_color = int(palette[color_i])
    color_i += 1
    if not _place_controls(grid, edge_mask, [(pt, l_color) for pt in ctrl_l], bg):
        return None
    _paint_path(expected, inner_l, l_color)
    used_inner[irs, ics] = True

    stripe_orientation = (
        str(forced_stripe_orientation)
        if forced_stripe_orientation is not None
        else random.choice(["vertical", "horizontal"])
    )
    if stripe_orientation == "horizontal" and H < 4:
        stripe_orientation = "vertical"
    stripe_specs = _sample_stripe_specs(stripe_orientation, H, W, rows, cols, r0, c0)
    if stripe_specs is None:
        return None
    stripe_lines, ctrl_s = stripe_specs
    all_stripe = [p for line in stripe_lines for p in line]
    irs, ics = zip(*all_stripe)
    if used_inner[irs, ics].any():
        return None
    stripe_color = int(palette[color_i])
    color_i += 1
    if not _place_controls(grid, edge_mask, [(pt, stripe_color) for pt in ctrl_s], bg):
        return None
    for line in stripe_lines:
        _paint_path(expected, line, stripe_color)
    used_inner[irs, ics] = True

    if n_groups == 3:
        remaining_kinds = [k for k in ["tl", "tr", "bl", "br"] if k != mirrored_l_kind]
        random.shuffle(remaining_kinds)
        placed_extra = False
        for kind in remaining_kinds:
            if color_i >= len(palette):
                return None
            extra_specs = _sample_l_specs(kind, H, W, rows, cols, r0, c0)
            if extra_specs is None:
                continue
            inner_e, ctrl_e = extra_specs
            irs, ics = zip(*inner_e)
            if used_inner[irs, ics].any():
                continue
            extra_color = int(palette[color_i])
            if not _place_controls(grid, edge_mask, [(pt, extra_color) for pt in ctrl_e], bg):
                continue
            color_i += 1
            _paint_path(expected, inner_e, extra_color)
            used_inner[irs, ics] = True
            placed_extra = True
            break
        if not placed_extra:
            return None

    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        if color_i >= len(palette):
            break
        _add_distractor(grid, comp_mask, bg, (r0, c0, r1, c1), int(palette[color_i]))
        color_i += 1

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    forced_n_groups = int(_draw_balanced_choice("_N_GROUP_CYCLE", [2, 3]))
    forced_l_kind = str(_draw_balanced_choice("_L_KIND_CYCLE", ["tl", "tr", "bl", "br"]))
    forced_stripe_orientation = str(
        _draw_balanced_choice("_STRIPE_ORIENTATION_CYCLE", ["vertical", "horizontal"])
    )
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(
            forced_n_groups=forced_n_groups,
            forced_l_kind=forced_l_kind,
            forced_stripe_orientation=forced_stripe_orientation,
        )
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a6f40cea combo input")
