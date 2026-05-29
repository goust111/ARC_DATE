# Combo variant of a6f40cea (index 11)
# combo_spec: D1/3+D4/3
# Name: 直线贯穿射线

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

    Returns (r0, c0, r1, c1, border_color, fill_color) for the largest such rectangle.
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
                out.append((int(p[0]), int(p[1])))
    return out


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = int(colors[0])
        return
    a, b = map(int, colors)
    vals = [a if i % 2 == 0 else b for i in range(len(path))]
    grid[rs, cs] = vals


def _paint_alt_with_start(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    palette: tuple[int, int],
    start_color: int,
) -> None:
    if not path:
        return
    a, b = map(int, palette)
    cur = int(start_color)
    for rr, cc in path:
        grid[int(rr), int(cc)] = int(cur)
        cur = int(b) if cur == int(a) else int(a)


def _ray_from_touch(
    side: str,
    coord: tuple[int, int],
    out_h: int,
    out_w: int,
) -> list[tuple[int, int]]:
    r, c = map(int, coord)
    if side == "top":
        return [(rr, c) for rr in range(0, out_h)]
    if side == "bottom":
        return [(rr, c) for rr in range(out_h - 1, -1, -1)]
    if side == "left":
        return [(r, cc) for cc in range(0, out_w)]
    if side == "right":
        return [(r, cc) for cc in range(out_w - 1, -1, -1)]
    raise ValueError(f"unknown side: {side}")


# Design analysis for the fused variant:
# a) Behavior trace.
#    Original a6f40cea reads two frame-adjacent touchpoints from one outside component and
#    fills the missing part between them inside the frame, usually as an L. Here that fails:
#    D1/3 removes elbows, and D4/3 removes the "connect the two touches" destination. A top-side
#    U with touches at columns 2 and 5 now means "emit two vertical rays from row 0 down to row H-1",
#    not "draw a horizontal segment between (0,2) and (0,5)". The same logic rotates for left/right.
# b) Complexity calibration.
#    To keep the rule visible, every ring-touching component uses two same-side touches, so each
#    component yields a pair of parallel bars. This preserves repetition and prevents the task from
#    collapsing into isolated single-pixel hints. Components may still be mono or 2-color alternating.
# c) Broken assumptions from the original solver.
#    - Farthest-pair Manhattan reconstruction is no longer valid.
#    - "Different sides imply one interior elbow" is no longer valid.
#    - Deferred aggregation for opposite-side monochrome traces is no longer needed.
#    The new solver instead assumes each useful component contributes >=2 touches all on one frame side.
# d) Placement legality.
#    The generator chooses one global orientation per sample, samples well-spaced stripe positions,
#    pairs adjacent positions into non-overlapping U-shapes, and validates placement with the same
#    4-neighbor separation checks used in the source task.
# e) Visual observability.
#    Humans can see that every useful outside shape is a U hanging from one side of the closed frame,
#    and the output keeps only the inside while converting each touch into a full-span parallel bar.
#    This is visibly different from the original L-completion behavior in 2-3 examples.
# f) Visual cleanliness.
#    All generated bars are parallel within a sample, so stripes do not cross. That avoids mixed-color
#    intersections and keeps the cropped output readable.
# g) Color inference.
#    Monochrome components create solid bars. Two-color components alternate along the outside U and
#    the inside ray; the color seen at each touchpoint is reused as the phase anchor for that ray.
# Assumption audit.
#    - main() assumes ring-touch coords for one component share one boundary row or one boundary col.
#      generate_input() guarantees this by constructing only single-side U-shapes.
#    - main() assumes at most two colors per useful component. The palette sampler enforces that.
#    - main() assumes rays do not need conflict resolution. The generator guarantees one orientation
#      per sample and unique stripe indices, so projected rays are parallel and disjoint.
#    - Training visibility matters, so the generator deliberately includes both monochrome and
#      alternating components and uses both opposite sides for the chosen orientation.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project same-side frame touchpoints inward as full-span straight rays inside the cropped frame.

    Input structure:
    - Background, one 1px closed rectangle with uniform fill, and colored components outside it.
    - Each useful outside component is an open U-shaped frame whose two stubs touch the same side of the closed frame.

    Rule:
    - Output only the interior of the closed frame, initialized to the fill color.
    - For each component, map its frame-adjacent touchpoints onto the interior boundary.
    - Every touchpoint emits one straight ray perpendicular to that side until the opposite interior edge.
    - Monochrome components paint solid bars; two-color components alternate along each ray, starting from the touchpoint color.

    Key insight:
    - The two touches no longer define two ends of one path. They only reveal which side is active and where separate parallel rays should start.
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
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    def map_to_out(rr: int, cc: int) -> tuple[int, int]:
        mr = max(0, min(out_h - 1, int(rr - r_int0)))
        mc = max(0, min(out_w - 1, int(cc - c_int0)))
        return int(mr), int(mc)

    comp_mask = (g != int(bg)) & (g != int(fill)) & (g != int(bc))
    clab, cn = label(comp_mask, structure=STRUCT4)

    ring: set[tuple[int, int]] = set()
    for cc in range(c0, c1 + 1):
        for rr in (r0 - 1, r1 + 1):
            if 0 <= rr < H:
                ring.add((int(rr), int(cc)))
    for rr in range(r0, r1 + 1):
        for cc in (c0 - 1, c1 + 1):
            if 0 <= cc < W:
                ring.add((int(rr), int(cc)))

    comp_hits: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for rr, cc in ring:
        cid = int(clab[rr, cc])
        if cid == 0:
            continue
        mr, mc = map_to_out(rr, cc)
        comp_hits[cid].append((int(mr), int(mc), int(g[rr, cc])))

    comp_cols: dict[int, list[int]] = {}
    for cid in range(1, int(cn) + 1):
        comp_cols[cid] = sorted(set(map(int, g[clab == int(cid)].flatten())))

    for cid, hits in comp_hits.items():
        if len(hits) < 2:
            continue
        cols = comp_cols.get(cid, [])
        if not cols or len(cols) > 2:
            continue

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for rr, cc, col in hits:
            hist[(int(rr), int(cc))][int(col)] += 1

        coords = sorted(hist.keys())
        if len(coords) < 2:
            continue

        rows = {int(r) for r, _c in coords}
        cols_set = {int(c) for _r, c in coords}

        side = None
        if len(rows) == 1:
            row = next(iter(rows))
            if row == 0:
                side = "top"
            elif row == out_h - 1:
                side = "bottom"
            coords = sorted(coords, key=lambda p: (p[1], p[0]))
        elif len(cols_set) == 1:
            col = next(iter(cols_set))
            if col == 0:
                side = "left"
            elif col == out_w - 1:
                side = "right"
            coords = sorted(coords, key=lambda p: (p[0], p[1]))
        if side is None:
            continue

        if len(cols) == 1:
            color = int(cols[0])
            for coord in coords:
                for rr, cc in _ray_from_touch(side, coord, out_h, out_w):
                    out[int(rr), int(cc)] = int(color)
            continue

        palette = (int(cols[0]), int(cols[1]))
        for coord in coords:
            start_color = int(hist[coord].most_common(1)[0][0])
            _paint_alt_with_start(
                out,
                _ray_from_touch(side, coord, out_h, out_w),
                palette,
                start_color,
            )

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
_OUTER_OFFSET_RANGE = (2, 3)
_N_COMPONENTS_RANGE = (2, 3)
_DUAL_COLOR_PROB = 0.5
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 300


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    H, W = grid.shape
    s = set(coords)
    for r, c in s:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != int(bg):
            return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in s and 0 <= nr < H and 0 <= nc < W and bool(comp_mask[nr, nc]):
                return False
    return True


def _place_component(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    bg: int,
) -> bool:
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, bg, comp_mask):
        return False
    _paint_path(grid, path, colors)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _sample_spaced_positions(length: int, count: int, min_gap: int = 2) -> list[int] | None:
    vals = list(range(length))
    for _ in range(80):
        picks = sorted(random.sample(vals, count))
        if all(picks[i + 1] - picks[i] >= min_gap for i in range(len(picks) - 1)):
            return picks
    return None


def _vertical_u_specs(
    side: str,
    cols_pair: tuple[int, int],
    r0: int,
    c0: int,
    r1: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    lo, hi = _OUTER_OFFSET_RANGE
    c_a = int(c0 + 1 + cols_pair[0])
    c_b = int(c0 + 1 + cols_pair[1])
    if side == "top":
        ring_a = (int(r0 - 1), int(c_a))
        ring_b = (int(r0 - 1), int(c_b))
        bend_r = int(r0 - random.randint(lo, hi))
    else:
        ring_a = (int(r1 + 1), int(c_a))
        ring_b = (int(r1 + 1), int(c_b))
        bend_r = int(r1 + random.randint(lo, hi))
    path = _join(
        _vpath(ring_a[0], bend_r, ring_a[1]),
        _hpath(bend_r, ring_a[1], ring_b[1]),
        _vpath(bend_r, ring_b[0], ring_b[1]),
    )
    return path, [ring_a, ring_b]


def _horizontal_u_specs(
    side: str,
    rows_pair: tuple[int, int],
    r0: int,
    c0: int,
    c1: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    lo, hi = _OUTER_OFFSET_RANGE
    r_a = int(r0 + 1 + rows_pair[0])
    r_b = int(r0 + 1 + rows_pair[1])
    if side == "left":
        ring_a = (int(r_a), int(c0 - 1))
        ring_b = (int(r_b), int(c0 - 1))
        bend_c = int(c0 - random.randint(lo, hi))
    else:
        ring_a = (int(r_a), int(c1 + 1))
        ring_b = (int(r_b), int(c1 + 1))
        bend_c = int(c1 + random.randint(lo, hi))
    path = _join(
        _hpath(ring_a[0], ring_a[1], bend_c),
        _vpath(ring_a[0], ring_b[0], bend_c),
        _hpath(ring_b[0], bend_c, ring_b[1]),
    )
    return path, [ring_a, ring_b]


def _inside_ray(side: str, pos: int, H: int, W: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(int(r), int(pos)) for r in range(0, H)]
    if side == "bottom":
        return [(int(r), int(pos)) for r in range(H - 1, -1, -1)]
    if side == "left":
        return [(int(pos), int(c)) for c in range(0, W)]
    if side == "right":
        return [(int(pos), int(c)) for c in range(W - 1, -1, -1)]
    raise ValueError(f"unknown side: {side}")


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
        coords = [(int(r), int(c))]
        used = {(int(r), int(c))}
        for _ in range(size - 1):
            base = random.choice(coords)
            opts = [
                (int(nr), int(nc))
                for nr, nc in neighbors4(*base)
                if 1 <= nr < H - 1 and 1 <= nc < W - 1 and (int(nr), int(nc)) not in used
            ]
            if not opts:
                break
            p = random.choice(opts)
            used.add(p)
            coords.append(p)
        if any(r0 - gap <= rr <= r1 + gap and c0 - gap <= cc <= c1 + gap for rr, cc in coords):
            continue
        if _place_component(grid, comp_mask, coords, (int(color),), bg):
            return


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_BG_CHOICES)
    bc = _BORDER_COLOR
    fill = bg if random.random() < _FILL_EQ_BG_PROB else random.choice(
        [x for x in range(10) if x not in (bg, bc)]
    )
    H = random.randint(*_INTERIOR_H_RANGE)
    W = random.randint(*_INTERIOR_W_RANGE)
    rows = random.randint(max(_GRID_SIZE_RANGE[0], H + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    cols = random.randint(max(_GRID_SIZE_RANGE[0], W + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - H - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - W - _FRAME_MARGIN_FAR)
    r1, c1 = int(r0 + H + 1), int(c0 + W + 1)

    grid = np.full((rows, cols), int(bg), dtype=int)
    grid[r0, c0 : c1 + 1] = int(bc)
    grid[r1, c0 : c1 + 1] = int(bc)
    grid[r0 : r1 + 1, c0] = int(bc)
    grid[r0 : r1 + 1, c1] = int(bc)
    grid[r0 + 1 : r1, c0 + 1 : c1] = int(fill)

    expected = np.full((H, W), int(fill), dtype=int)
    comp_mask = np.zeros_like(grid, dtype=bool)
    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    color_i = 0

    family = random.choice(["vertical", "horizontal"])
    axis_len = W if family == "vertical" else H
    max_components = min(_N_COMPONENTS_RANGE[1], ((axis_len + 1) // 2) // 2)
    if max_components < _N_COMPONENTS_RANGE[0]:
        return None
    n_components = random.randint(_N_COMPONENTS_RANGE[0], max_components)

    positions = _sample_spaced_positions(axis_len, 2 * n_components, min_gap=2)
    if positions is None:
        return None
    pairs = [(int(positions[2 * i]), int(positions[2 * i + 1])) for i in range(n_components)]

    if family == "vertical":
        side_choices = ["top", "bottom"]
    else:
        side_choices = ["left", "right"]
    sides = [side_choices[i % 2] for i in range(n_components)]
    random.shuffle(sides)

    dual_indices = {random.randrange(n_components)}
    mono_indices = {random.randrange(n_components)}
    while mono_indices & dual_indices:
        mono_indices = {random.randrange(n_components)}
    for i in range(n_components):
        if i in dual_indices or i in mono_indices:
            continue
        if random.random() < _DUAL_COLOR_PROB:
            dual_indices.add(i)

    used_mask = np.zeros((H, W), dtype=bool)
    for i, pair in enumerate(pairs):
        need = 2 if i in dual_indices else 1
        if color_i + need > len(palette):
            return None
        if need == 2:
            colors = (int(palette[color_i]), int(palette[color_i + 1]))
            color_i += 2
            if random.random() < 0.5:
                colors = (int(colors[1]), int(colors[0]))
        else:
            colors = (int(palette[color_i]),)
            color_i += 1

        side = sides[i]
        if family == "vertical":
            outer, ring_pts = _vertical_u_specs(side, pair, r0, c0, r1)
            stripe_specs = [(int(pair[0]), ring_pts[0]), (int(pair[1]), ring_pts[1])]
        else:
            outer, ring_pts = _horizontal_u_specs(side, pair, r0, c0, c1)
            stripe_specs = [(int(pair[0]), ring_pts[0]), (int(pair[1]), ring_pts[1])]

        if not _place_component(grid, comp_mask, outer, colors, bg):
            return None

        for pos, ring_pt in stripe_specs:
            inner = _inside_ray(side, pos, H, W)
            irs, ics = zip(*inner)
            if used_mask[irs, ics].any():
                return None
            if len(colors) == 1:
                expected[irs, ics] = int(colors[0])
            else:
                start_color = int(grid[ring_pt])
                _paint_alt_with_start(expected, inner, (int(colors[0]), int(colors[1])), start_color)
            used_mask[irs, ics] = True

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
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a6f40cea combo input")
