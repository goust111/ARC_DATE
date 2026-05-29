# Combo variant of a6f40cea (index 24)
# combo_spec: D11/2+D20/0
# Name: T形源颜色配对

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


"""
Design analysis for combo variant D11/2 + D20/0

a) Concrete behavior and trace
   - Original task: one outside L-source usually gives two ring touches, and the solver either connects
     the only pair or treats two colors as an alternation palette for one path.
   - Combo change: each relevant source is now a three-touch corner component. Two of the touches share
     one color, and the third touch has another color. Only the repeated color produces an interior path.
   - Small top-left trace:
       mapped touches = {(0, 1, A), (2, 0, A), (0, 4, B)}.
       Original-style reasoning would favor the farthest pair (2,0)-(0,4) and could even alternate A/B.
       Combo reasoning groups by color first, keeps only A because it appears twice, and draws the
       shorter L path (0,1)->(2,0). Touch (0,4,B) is a visible decoy terminal.

b) Complexity calibration
   - The T-source cue is only visible if a component truly has three ring touches, so every generated
     main component has exactly three ring contacts.
   - Each source uses exactly two colors with a 2+1 multiplicity pattern. This keeps the rule legible:
     equal colors pair; the singleton is ignored.
   - All path-producing sources live in distinct frame corners. That preserves the original task family
     of sparse corner L-paths instead of turning the puzzle into a dense routing problem.

c) Hidden-assumption review of the original main()
   - The original monochrome branch assumes one component collapses to one endpoint pair. That fails
     because a T-source has three terminals and only two of them are meaningful.
   - The original two-color branch assumes the two colors are the paint palette of one alternating path.
     In this combo, the two colors are not a path palette; they are endpoint labels with counts 2 and 1.
   - The original deferred stripe aggregation is unnecessary here. Once a same-color pair is identified,
     a direct Manhattan connection is enough, including the straight-row and straight-column cases.

d) Placement legality
   - The generator uses a fixed outside hub offset from the frame, so every branch stays outside except
     at its designated ring-touch terminal.
   - A component is painted as one connected colored tree plus a compact hub block; placement is checked
     for in-bounds occupancy, background availability, and 4-neighbor clearance from other components.
   - Interior paths are reserved with `used_inner`, so no two generated outputs overwrite each other.

e) Visual observability
   - In each training example, a human sees corner-adjacent outside components with three terminals.
     Two terminals repeat one color and one terminal carries a different color.
   - The output keeps only the repeated-color pair, so the singleton terminal is visibly inactive.
   - This difference is inferable from a few examples because the decoy touch remains present in input
     but never produces any extra output segment.

f) Visual cleanliness
   - The three-touch sources are restricted to the four corners around the frame, so branches do not pile
     into the same narrow area.
   - Only one interior L path is produced per source, and the ignored singleton touch prevents clutter
     instead of adding a second route.
   - Distractors remain small, sparse, and away from the frame ring.

g) Change inferability
   - Colors have one job only: repeated color means "pair these two terminals". The singleton color has
     no secondary transformation rule.
   - The generator avoids extra recoloring conventions and does not rely on hidden numeric color order.
   - The repeated-vs-singleton contrast is supported multiple times in a single grid because there are
     2-3 T-sources per sample.

h) Generator-to-solver contract
   - Every main source contributes exactly three mapped ring touches and exactly one repeated touch color.
   - The repeated-color pair is deliberately not the Manhattan-farthest pair, so the combo does not
     collapse back to the original farthest-endpoint behavior.
   - The repeated pair always consists of one touch on each of two adjacent frame sides, so the produced
     output remains a corner L-path or a straight degenerate case handled by the Manhattan builder.

i) New implicit assumptions introduced by the combo
   - The solver assumes ring-touch colors are locally unambiguous; the generator guarantees each touch is
     a solid monochrome branch tip.
   - The solver ignores singleton colors. The generator therefore guarantees every meaningful source has a
     `2+1` color histogram rather than `1+1+1` or `3`.
   - The solver treats any same-color group with more than two touches as malformed input and falls back
     to a farthest-pair heuristic, but the generator never depends on that fallback.
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


def _decode_t_pairings(
    pts: list[tuple[int, int, int]],
) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
    for r, c, col in pts:
        hist[(int(r), int(c))][int(col)] += 1

    if len(hist) != 3:
        return []

    per_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for coord, counter in hist.items():
        dom = int(counter.most_common(1)[0][0])
        per_color[dom].append((int(coord[0]), int(coord[1])))

    shape = sorted(len(set(coords)) for coords in per_color.values())
    if shape != [1, 2]:
        return []

    out = []
    for colv, coords in per_color.items():
        uniq = sorted(set(coords))
        if len(uniq) == 2:
            out.append((int(colv), uniq[0], uniq[1]))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the framed interior by decoding three-touch outside T-sources.

    Input structure:
    - A uniform background contains one 1px closed rectangular frame with a uniform interior fill.
    - Outside the frame are colored 4-connected source components touching the ring around the frame.
    - Each relevant source presents three ring touches near one corner, with colors split as one repeated
      pair plus one singleton.

    Rule:
    - Output only the frame interior, initialized with the frame's fill color.
    - Project each ring-touch cell through the border into interior coordinates.
    - For each outside component, group mapped touch coordinates by their dominant touch color.
    - Only colors appearing on exactly two distinct mapped touches create a solid Manhattan path; any
      singleton touch is ignored.

    Key insight: the repeated color inside a three-terminal source is the endpoint-matching signal, so
    the odd-colored terminal is a decoy even though it belongs to the same outside connected component.
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

    def build_manhattan(
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

    def inside_corner(p1: tuple[int, int], p2: tuple[int, int], cornerA: bool) -> bool:
        (ra, ca), (rb, cb) = p1, p2
        if ra == rb or ca == cb:
            return True
        corner = (rb, ca) if cornerA else (ra, cb)
        rr, cc = corner
        return (0 < rr < outH - 1) and (0 < cc < outW - 1)

    comp_mask = (g != int(bg)) & (g != int(fill)) & (g != int(bc))
    clab, _cn = label(comp_mask, structure=STRUCT4)

    ring = _frame_ring_cells(H, W, r0, c0, r1, c1)

    comp_pts: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for rr, cc in ring:
        cid = int(clab[rr, cc])
        if cid == 0:
            continue
        mr, mc = map_to_out(rr, cc)
        comp_pts[cid].append((mr, mc, int(g[rr, cc])))

    for pts in comp_pts.values():
        for colv, p1, p2 in _decode_t_pairings(pts):
            pathA = build_manhattan(p1, p2, True)
            pathB = build_manhattan(p1, p2, False)
            chosen = pathB if (inside_corner(p1, p2, False) and not inside_corner(p1, p2, True)) else pathA
            for rr, cc in chosen:
                out[int(rr), int(cc)] = int(colv)

    return out


# --------------- generator (a6 combo D11/2 + D20/0) ---------------

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
_DISTRACTOR_COUNT_RANGE = (0, 1)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 500
_T_HUB_OFFSET = 3


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
        if grid[r, c] != bg:
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


def _hub_block(r: int, c: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)), (int(r), int(c + 1)), (int(r + 1), int(c)), (int(r + 1), int(c + 1))]


def _rand_closed(lo: int, hi: int) -> int:
    assert lo <= hi
    return random.randint(int(lo), int(hi))


def _t_source_specs(
    kind: str,
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    pair_color: int,
    single_color: int,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int], tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    if kind == "tl":
        r = _rand_closed(1, max(1, H // 2))
        c_pair = _rand_closed(1, max(1, (W // 2) - 1))
        c_extra = _rand_closed(c_pair + 2, W - 2)
        inner = _join(_vpath(0, r, c_pair), _hpath(r, c_pair, 0))
        top_pair = (r0 - 1, c0 + 1 + c_pair)
        left_pair = (r0 + 1 + r, c0 - 1)
        top_extra = (r0 - 1, c0 + 1 + c_extra)
        hub_r, hub_c = r0 - _T_HUB_OFFSET, c0 - _T_HUB_OFFSET
        colored = _merge_colored_parts(
            [
                (_hub_block(hub_r, hub_c), pair_color),
                (_join(_vpath(top_pair[0], hub_r + 1, top_pair[1]), _hpath(hub_r + 1, top_pair[1], hub_c + 1)), pair_color),
                (_join(_hpath(left_pair[0], left_pair[1], hub_c + 1), _vpath(left_pair[0], hub_r + 1, hub_c + 1)), pair_color),
                (_join(_vpath(top_extra[0], hub_r - 1, top_extra[1]), _hpath(hub_r - 1, top_extra[1], hub_c)), single_color),
            ]
        )
        assert colored is not None
        return inner, colored, ((0, c_pair), (r, 0), (0, c_extra))

    if kind == "tr":
        r = _rand_closed(1, max(1, H // 2))
        c_pair = _rand_closed(max(1, W // 2), W - 2)
        c_extra = _rand_closed(1, c_pair - 2)
        inner = _join(_vpath(0, r, c_pair), _hpath(r, c_pair, W - 1))
        top_pair = (r0 - 1, c0 + 1 + c_pair)
        right_pair = (r0 + 1 + r, c1 + 1)
        top_extra = (r0 - 1, c0 + 1 + c_extra)
        hub_r, hub_c = r0 - _T_HUB_OFFSET, c1 + (_T_HUB_OFFSET - 1)
        colored = _merge_colored_parts(
            [
                (_hub_block(hub_r, hub_c), pair_color),
                (_join(_vpath(top_pair[0], hub_r + 1, top_pair[1]), _hpath(hub_r + 1, top_pair[1], hub_c)), pair_color),
                (_join(_hpath(right_pair[0], right_pair[1], hub_c), _vpath(right_pair[0], hub_r + 1, hub_c)), pair_color),
                (_join(_vpath(top_extra[0], hub_r - 1, top_extra[1]), _hpath(hub_r - 1, top_extra[1], hub_c + 1)), single_color),
            ]
        )
        assert colored is not None
        return inner, colored, ((0, c_pair), (r, W - 1), (0, c_extra))

    if kind == "bl":
        r = _rand_closed(max(1, H // 2), H - 2)
        c_pair = _rand_closed(1, max(1, (W // 2) - 1))
        c_extra = _rand_closed(c_pair + 2, W - 2)
        inner = _join(_vpath(H - 1, r, c_pair), _hpath(r, c_pair, 0))
        bot_pair = (r1 + 1, c0 + 1 + c_pair)
        left_pair = (r0 + 1 + r, c0 - 1)
        bot_extra = (r1 + 1, c0 + 1 + c_extra)
        hub_r, hub_c = r1 + (_T_HUB_OFFSET - 1), c0 - _T_HUB_OFFSET
        colored = _merge_colored_parts(
            [
                (_hub_block(hub_r, hub_c), pair_color),
                (_join(_vpath(bot_pair[0], hub_r, bot_pair[1]), _hpath(hub_r, bot_pair[1], hub_c + 1)), pair_color),
                (_join(_hpath(left_pair[0], left_pair[1], hub_c + 1), _vpath(left_pair[0], hub_r, hub_c + 1)), pair_color),
                (_join(_vpath(bot_extra[0], hub_r + 2, bot_extra[1]), _hpath(hub_r + 2, bot_extra[1], hub_c)), single_color),
            ]
        )
        assert colored is not None
        return inner, colored, ((H - 1, c_pair), (r, 0), (H - 1, c_extra))

    assert kind == "br"
    r = _rand_closed(max(1, H // 2), H - 2)
    c_pair = _rand_closed(max(1, W // 2), W - 2)
    c_extra = _rand_closed(1, c_pair - 2)
    inner = _join(_vpath(H - 1, r, c_pair), _hpath(r, c_pair, W - 1))
    bot_pair = (r1 + 1, c0 + 1 + c_pair)
    right_pair = (r0 + 1 + r, c1 + 1)
    bot_extra = (r1 + 1, c0 + 1 + c_extra)
    hub_r, hub_c = r1 + (_T_HUB_OFFSET - 1), c1 + (_T_HUB_OFFSET - 1)
    colored = _merge_colored_parts(
        [
            (_hub_block(hub_r, hub_c), pair_color),
            (_join(_vpath(bot_pair[0], hub_r, bot_pair[1]), _hpath(hub_r, bot_pair[1], hub_c)), pair_color),
            (_join(_hpath(right_pair[0], right_pair[1], hub_c), _vpath(right_pair[0], hub_r, hub_c)), pair_color),
            (_join(_vpath(bot_extra[0], hub_r + 2, bot_extra[1]), _hpath(hub_r + 2, bot_extra[1], hub_c + 1)), single_color),
        ]
    )
    assert colored is not None
    return inner, colored, ((H - 1, c_pair), (r, W - 1), (H - 1, c_extra))


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


def _gen_attempt(target_n_components: int | None = None) -> np.ndarray | None:
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
    used_inner = np.zeros((H, W), dtype=bool)

    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    if target_n_components is None:
        n_components = random.randint(*_N_COMPONENTS_RANGE)
    else:
        n_components = int(target_n_components)
    if len(palette) < 2 * n_components:
        return None

    color_i = 0
    kinds = random.sample(["tl", "tr", "bl", "br"], k=n_components)

    for kind in kinds:
        pair_color = int(palette[color_i])
        single_color = int(palette[color_i + 1])
        color_i += 2

        inner, colored, mapped = _t_source_specs(kind, H, W, r0, c0, r1, c1, pair_color, single_color)
        irs, ics = zip(*inner)
        if used_inner[irs, ics].any():
            return None
        if not _place_colored_component(grid, comp_mask, colored, bg):
            return None
        _paint_path(expected, inner, pair_color)
        used_inner[irs, ics] = True

        pair_a, pair_b, single = mapped
        d_pair = abs(pair_a[0] - pair_b[0]) + abs(pair_a[1] - pair_b[1])
        d_single_a = abs(pair_a[0] - single[0]) + abs(pair_a[1] - single[1])
        d_single_b = abs(pair_b[0] - single[0]) + abs(pair_b[1] - single[1])
        if d_pair >= max(d_single_a, d_single_b):
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
    target_n_components = random.randint(*_N_COMPONENTS_RANGE)
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(target_n_components=target_n_components)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a6f40cea combo input")
