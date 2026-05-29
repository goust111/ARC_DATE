# Variant of a6f40cea
# Dimension: D14 — 处理顺序与多步逻辑 (meta)
# Change: 路径一次性独立完成，组件之间互不影响... → 链式反应处理
#
# Design analysis
# - Concrete behavior:
#   Keep the same framed-rectangle setting, but distinguish one starter component
#   with two ring touches from later chain-link components with only one ring touch.
#   The starter paints the first interior path exactly once. Its end then becomes
#   the start of the next segment, which connects to the nearest remaining link
#   touch. Repeating this yields a visible multi-step chain inside the frame.
# - Why this fits D14:
#   The key change is process order: later segments are not recoverable in
#   isolation from their own outside geometry; they depend on where the previous
#   segment terminated. That is the visible "chain reaction" step.
# - Hidden-assumption audit:
#   The original solver assumed every useful outside component already encodes both
#   endpoints of one finished path. That assumption is broken here for link
#   components, so the new solver treats one-touch stubs as deferred targets rather
#   than full standalone paths.
# - Generator contract:
#   Exactly one starter component has two ring touches. Each later link is a simple
#   monochrome outward stub with one ring touch. The generator keeps nearest-link
#   distances distinct at each chain step so the order is inferable and stable.

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


def _build_manhattan(p1: tuple[int, int], p2: tuple[int, int], corner_a: bool) -> list[tuple[int, int]]:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb:
        return [(int(ra), int(c)) for c in _line(int(ca), int(cb))]
    if ca == cb:
        return [(int(r), int(ca)) for r in _line(int(ra), int(rb))]
    if corner_a:
        return _join(_vpath(ra, rb, ca), _hpath(rb, ca, cb))
    return _join(_hpath(ra, ca, cb), _vpath(ra, rb, cb))


def _inside_corner(p1: tuple[int, int], p2: tuple[int, int], corner_a: bool, out_h: int, out_w: int) -> bool:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb or ca == cb:
        return True
    corner = (rb, ca) if corner_a else (ra, cb)
    rr, cc = corner
    return 0 < rr < out_h - 1 and 0 < cc < out_w - 1


def _choose_inside_path(p1: tuple[int, int], p2: tuple[int, int], out_h: int, out_w: int) -> list[tuple[int, int]]:
    path_a = _build_manhattan(p1, p2, True)
    path_b = _build_manhattan(p1, p2, False)
    if _inside_corner(p1, p2, False, out_h, out_w) and not _inside_corner(p1, p2, True, out_h, out_w):
        return path_b
    return path_a


def _paint_alt(out: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, int], start_color: int) -> None:
    ca, cb = map(int, colors)
    cur = int(start_color)
    for rr, cc in path:
        out[int(rr), int(cc)] = int(cur)
        cur = int(cb) if int(cur) == int(ca) else int(ca)


def _clockwise_ring(r0: int, c0: int, r1: int, c1: int, h: int, w: int) -> list[tuple[int, int]]:
    ring: list[tuple[int, int]] = []
    for cc in range(c0, c1 + 1):
        rr = r0 - 1
        if 0 <= rr < h:
            ring.append((int(rr), int(cc)))
    for rr in range(r0, r1 + 1):
        cc = c1 + 1
        if 0 <= cc < w:
            ring.append((int(rr), int(cc)))
    for cc in range(c1, c0 - 1, -1):
        rr = r1 + 1
        if 0 <= rr < h:
            ring.append((int(rr), int(cc)))
    for rr in range(r1, r0 - 1, -1):
        cc = c0 - 1
        if 0 <= cc < w:
            ring.append((int(rr), int(cc)))
    return ring


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the frame interior by chaining later segments from the previous segment's endpoint.

    Input structure:
    - One closed 1px rectangular frame with a uniform fill.
    - Exactly one starter component outside the frame touches the ring in two places.
    - One or more later link components are simple monochrome stubs that touch the ring once.

    Rule:
    - Output only the frame interior, initialized with the fill color.
    - Reconstruct the starter's full Manhattan path first.
    - Then repeatedly connect the previous segment's endpoint to the nearest remaining link touch.
    - Each link paints a solid path in its own color; the starter keeps its original solid or alternating style.

    Key insight: only the first component supplies both endpoints. Later components only expose targets,
    so their interior paths are triggered step-by-step from the endpoint reached by the previous segment.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
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
    ring = _clockwise_ring(r0, c0, r1, c1, h, w)
    ring_order = {cell: i for i, cell in enumerate(ring)}

    starters: list[dict[str, object]] = []
    links: list[dict[str, object]] = []

    for cid in range(1, int(cn) + 1):
        pts = np.argwhere(clab == int(cid))
        if pts.size == 0:
            continue
        ring_hits = [(int(rr), int(cc)) for rr, cc in pts if (int(rr), int(cc)) in ring_order]
        if not ring_hits:
            continue

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        orders = []
        for rr, cc in ring_hits:
            orders.append(ring_order[(int(rr), int(cc))])
            hist[map_to_out(int(rr), int(cc))][int(g[int(rr), int(cc)])] += 1

        coords = list(hist.keys())
        cols = sorted(set(map(int, g[clab == int(cid)].flatten())))
        first_order = min(orders)
        if len(coords) >= 2:
            best = None
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    d = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
                    if best is None or d > best[0]:
                        best = (d, coords[i], coords[j])
            assert best is not None
            _d, a, b = best
            path = _choose_inside_path(a, b, out_h, out_w)
            # orient starter by ring order: the earliest touched endpoint is the first path cell
            a_order = min(ring_order[(rr, cc)] for rr, cc in ring_hits if map_to_out(rr, cc) == a)
            b_order = min(ring_order[(rr, cc)] for rr, cc in ring_hits if map_to_out(rr, cc) == b)
            if b_order < a_order:
                path = list(reversed(path))
                a, b = b, a
            starters.append(
                {
                    "order": int(first_order),
                    "path": path,
                    "a": a,
                    "b": b,
                    "colors": tuple(cols),
                    "start_color": int(hist[a].most_common(1)[0][0]),
                }
            )
        else:
            coord = coords[0]
            color = int(hist[coord].most_common(1)[0][0])
            links.append({"order": int(first_order), "coord": coord, "color": int(color)})

    assert starters
    starter = min(starters, key=lambda item: (int(item["order"]), -len(item["path"])))
    starter_path = list(starter["path"])
    starter_colors = tuple(starter["colors"])
    if len(starter_colors) == 1:
        for rr, cc in starter_path:
            out[int(rr), int(cc)] = int(starter_colors[0])
    else:
        _paint_alt(out, starter_path, (int(starter_colors[0]), int(starter_colors[1])), int(starter["start_color"]))

    current = starter_path[-1]
    remaining = list(links)
    while remaining:
        remaining.sort(
            key=lambda item: (
                abs(int(item["coord"][0]) - int(current[0])) + abs(int(item["coord"][1]) - int(current[1])),
                int(item["order"]),
            )
        )
        nxt = remaining.pop(0)
        path = _choose_inside_path(current, tuple(nxt["coord"]), out_h, out_w)
        for rr, cc in path:
            out[int(rr), int(cc)] = int(nxt["color"])
        current = path[-1]

    return out


_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (6, 12)
_INTERIOR_W_RANGE = (8, 14)
_GRID_SIZE_RANGE = (22, 32)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 6
_OUTER_OFFSET_RANGE = (2, 3)
_STARTER_DUAL_PROB = 0.4
_LINK_COUNT_RANGE = (1, 2)
_LINK_LEN_RANGE = (2, 4)
_DISTRACTOR_COUNT_RANGE = (0, 1)
_DISTRACTOR_SIZE_RANGE = (1, 3)
_DISTRACTOR_TRIES = 30
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 600


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    hh, ww = grid.shape
    cells = set(coords)
    for r, c in cells:
        if not (0 <= r < hh and 0 <= c < ww):
            return False
        if int(grid[int(r), int(c)]) != int(bg):
            return False
        for nr, nc in neighbors4(int(r), int(c)):
            if (int(nr), int(nc)) not in cells and 0 <= nr < hh and 0 <= nc < ww and comp_mask[int(nr), int(nc)]:
                return False
    return True


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = int(colors[0])
        return
    a, b = map(int, colors)
    grid[rs, cs] = [a if i % 2 == 0 else b for i in range(len(path))]


def _place_component(grid: np.ndarray, comp_mask: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...], bg: int) -> bool:
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, bg, comp_mask):
        return False
    _paint_path(grid, path, colors)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _starter_specs(kind: str, hh: int, ww: int, r0: int, c0: int, r1: int, c1: int):
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"
    row = random.randint(1, hh - 2)
    col = random.randint(1, ww - 2)
    ring_a = (r0 - 1 if is_top else r1 + 1, c0 + 1 + col)
    ring_b = (r0 + 1 + row, c0 - 1 if is_left else c1 + 1)
    rr = (r0 - random.randint(*_OUTER_OFFSET_RANGE)) if is_top else (r1 + random.randint(*_OUTER_OFFSET_RANGE))
    cc = (c0 - random.randint(*_OUTER_OFFSET_RANGE)) if is_left else (c1 + random.randint(*_OUTER_OFFSET_RANGE))
    outer = _join(_vpath(ring_a[0], rr, ring_a[1]), _hpath(rr, ring_a[1], cc), _vpath(rr, ring_b[0], cc), _hpath(ring_b[0], cc, ring_b[1]))
    p1 = (0 if is_top else hh - 1, int(col))
    p2 = (int(row), 0 if is_left else ww - 1)
    inner = _choose_inside_path(p1, p2, hh, ww)
    return inner, outer, ring_a, ring_b


def _link_stub(side: str, pos: int, length: int, r0: int, c0: int, r1: int, c1: int) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    if side == "top":
        ring_touch = (int(r0 - 1), int(c0 + 1 + pos))
        outer = _vpath(int(r0 - 1), int(r0 - length), int(c0 + 1 + pos))
    elif side == "bottom":
        ring_touch = (int(r1 + 1), int(c0 + 1 + pos))
        outer = _vpath(int(r1 + 1), int(r1 + length), int(c0 + 1 + pos))
    elif side == "left":
        ring_touch = (int(r0 + 1 + pos), int(c0 - 1))
        outer = _hpath(int(r0 + 1 + pos), int(c0 - 1), int(c0 - length))
    else:
        ring_touch = (int(r0 + 1 + pos), int(c1 + 1))
        outer = _hpath(int(r0 + 1 + pos), int(c1 + 1), int(c1 + length))
    return ring_touch, outer


def _add_distractor(grid: np.ndarray, comp_mask: np.ndarray, bg: int, frame: tuple[int, int, int, int], color: int) -> None:
    hh, ww = grid.shape
    r0, c0, r1, c1 = frame
    gap = _DISTRACTOR_CLEARANCE
    sz_lo, sz_hi = _DISTRACTOR_SIZE_RANGE
    for _ in range(_DISTRACTOR_TRIES):
        size = random.randint(sz_lo, sz_hi)
        r = random.randint(1, hh - 2)
        c = random.randint(1, ww - 2)
        coords = [(int(r), int(c))]
        used = {(int(r), int(c))}
        for _step in range(size - 1):
            base = random.choice(coords)
            opts = [(int(nr), int(nc)) for nr, nc in neighbors4(*base) if 1 <= nr < hh - 1 and 1 <= nc < ww - 1 and (int(nr), int(nc)) not in used]
            if not opts:
                break
            pick = random.choice(opts)
            used.add(pick)
            coords.append(pick)
        if any(r0 - gap <= rr <= r1 + gap and c0 - gap <= cc <= c1 + gap for rr, cc in coords):
            continue
        if _place_component(grid, comp_mask, coords, (int(color),), bg):
            return


def _gen_attempt() -> np.ndarray | None:
    bg = int(random.choice(_BG_CHOICES))
    bc = int(_BORDER_COLOR)
    fill_choices = [x for x in range(10) if x not in (bg, bc)]
    fill = int(bg if random.random() < _FILL_EQ_BG_PROB else random.choice(fill_choices))
    hh = random.randint(*_INTERIOR_H_RANGE)
    ww = random.randint(*_INTERIOR_W_RANGE)
    rows = random.randint(max(_GRID_SIZE_RANGE[0], hh + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    cols = random.randint(max(_GRID_SIZE_RANGE[0], ww + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - hh - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - ww - _FRAME_MARGIN_FAR)
    r1, c1 = int(r0 + hh + 1), int(c0 + ww + 1)

    grid = np.full((rows, cols), int(bg), dtype=int)
    grid[r0, c0 : c1 + 1] = int(bc)
    grid[r1, c0 : c1 + 1] = int(bc)
    grid[r0 : r1 + 1, c0] = int(bc)
    grid[r0 : r1 + 1, c1] = int(bc)
    grid[r0 + 1 : r1, c0 + 1 : c1] = int(fill)

    comp_mask = np.zeros_like(grid, dtype=bool)
    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    if len(palette) < 3:
        return None

    starter_kind = random.choice(["tl", "tr", "bl", "br"])
    starter_inner, starter_outer, starter_ring_a, starter_ring_b = _starter_specs(starter_kind, hh, ww, r0, c0, r1, c1)
    starter_colors = (int(palette[0]), int(palette[1])) if random.random() < _STARTER_DUAL_PROB else (int(palette[0]),)
    color_i = 2 if len(starter_colors) == 2 else 1
    if not _place_component(grid, comp_mask, starter_outer, starter_colors, int(bg)):
        return None

    # Recreate the starter end as main() will orient it: earliest ring-touch first.
    starter_a = (max(0, min(hh - 1, starter_ring_a[0] - (r0 + 1))), max(0, min(ww - 1, starter_ring_a[1] - (c0 + 1))))
    starter_b = (max(0, min(hh - 1, starter_ring_b[0] - (r0 + 1))), max(0, min(ww - 1, starter_ring_b[1] - (c0 + 1))))
    starter_path = list(starter_inner)
    ring = _clockwise_ring(r0, c0, r1, c1, rows, cols)
    ring_order = {cell: i for i, cell in enumerate(ring)}
    if ring_order[(int(starter_ring_b[0]), int(starter_ring_b[1]))] < ring_order[(int(starter_ring_a[0]), int(starter_ring_a[1]))]:
        starter_path = list(reversed(starter_path))
    current = starter_path[-1]

    link_count = random.randint(*_LINK_COUNT_RANGE)
    used_targets: list[tuple[int, int]] = []
    links_info: list[tuple[tuple[int, int], int]] = []
    for j in range(link_count):
        if color_i >= len(palette):
            return None
        color = int(palette[color_i])
        color_i += 1
        candidates = []
        for side in ("top", "bottom", "left", "right"):
            if side in ("top", "bottom"):
                for pos in range(1, ww - 1):
                    coord = (0 if side == "top" else hh - 1, int(pos))
                    if coord not in used_targets:
                        candidates.append((side, pos, coord))
            else:
                for pos in range(1, hh - 1):
                    coord = (int(pos), 0 if side == "left" else ww - 1)
                    if coord not in used_targets:
                        candidates.append((side, pos, coord))
        random.shuffle(candidates)
        placed = False
        for side, pos, target in candidates:
            if abs(int(target[0]) - int(current[0])) + abs(int(target[1]) - int(current[1])) < 3:
                continue
            stub_len = random.randint(*_LINK_LEN_RANGE)
            ring_touch, outer = _link_stub(side, int(pos), int(stub_len), r0, c0, r1, c1)
            if not _place_component(grid, comp_mask, outer, (int(color),), int(bg)):
                continue
            # maintain unique nearest target at each step
            d_here = abs(int(target[0]) - int(current[0])) + abs(int(target[1]) - int(current[1]))
            if any(abs(int(prev[0]) - int(current[0])) + abs(int(prev[1]) - int(current[1])) == d_here for prev in used_targets):
                rs, cs = zip(*outer)
                grid[rs, cs] = int(bg)
                comp_mask[rs, cs] = False
                continue
            used_targets.append(tuple(target))
            links_info.append((tuple(target), int(color)))
            current = tuple(target)
            placed = True
            break
        if not placed:
            return None

    for extra_color in palette[color_i : color_i + random.randint(*_DISTRACTOR_COUNT_RANGE)]:
        _add_distractor(grid, comp_mask, int(bg), (r0, c0, r1, c1), int(extra_color))

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != (hh, ww):
        return None
    # Non-degeneracy: at least starter plus one link must contribute visible non-fill pixels.
    if len(links_info) < 1:
        return None
    if int(np.sum(out != int(fill))) < len(starter_path) + 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a6f40cea D14 chain input")
