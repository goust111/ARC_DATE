# Variant of a6f40cea
# Dimension: D2 — 轨迹着色模式 (trajectory)
# Change: 单色组件用纯色填充路径；双色组件用两色交替填充，相位由端点颜色锚定... → 周期性图案铺设

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4

# Design analysis
# - 具体化: 仍然保留“主闭框 + 外部开放矩形框 + 将触点映射到框内”的主体结构。变化只放在轨迹着色:
#   外部每个开放框组件不再是纯色或双色交替，而是沿组件路径重复一个长度为 p 的 tile；输出中的对应内部路径
#   复制同一个 tile。为了让 p 有稳定、可见的几何来源，生成器强制同一组件两端从 ring 端点到第一次拐弯的直段等长，
#   因而 p = 这个可见直段的长度。小例子: 原题外部路径颜色可能是 6,7,6,7,6...，内部也 6/7 交替；现在外部可能是
#   6,2,8,6,2,8,...，而且这个 3 格周期长度由“端点到第一次拐弯共有 3 格”编码，内部路径按 6,2,8 周期继续铺设。
# - 复杂度校准: p 取 3-5，tile 至少含 2 种颜色，通常 3 种；这比原题 1/2 色明显更可观察，同时不会引入过多自由度。
#   生成器只保留 L 型组件。原题里同色 U 对需要跨组件再做一次配对聚合，那是旧着色规则的副产物；若继续保留，会把
#   D2 之外的第二套匹配逻辑重新引入，反而冲淡“从单个外部组件恢复周期 tile”这个核心变化。
# - 隐式假设审查: 原 solver 对 len(cols)==1/2 分支强依赖旧着色模式，在新题会直接失效；deferred 聚合也依赖
#   “同色直条需要跨组件合并”的旧生成方式。新 solver 改为先把 ring-touch 组件恢复成“有序简单路径”，再从外部路径
#   读取颜色序列与周期长度，避免依赖颜色个数。它新增的关键假设是: ring-touch 组件是无分叉简单路径，且 ring 上正好
#   有两个端点。生成器会保证这一点。
# - 放置合法性: 继续沿用 output-first。先确定框内目标路径，再反推外部开放框路径。组件放置前检查不重叠、不越界、
#   不与已有组件 4 邻接。U 对的两个组件仍然分开放置，避免外部结构粘连成单个分叉组件。
# - 视觉可观察性: 训练样例里不同组件会出现不同外凸深度，于是 ring 到第一次拐弯的长度不同，对应不同周期长度。
#   人眼既能看到外部路径本身是重复图案，也能看到“更深的外凸 → 更长的周期 tile”。
# - 可视化整洁度: 仍只在框外摆放细 1px 路径，框内只画对应 L/直线路径，避免多层叠加。生成器禁止内部路径重叠。
# - 颜色变化可推断性: 颜色来源只依赖外部组件本身，不依赖颜色 id 排序或不可见元数据。tile 直接可见，输出只是把它
#   搬运到框内路径上并周期重复。


def _find_framed_rect_uniform_fill(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
) -> tuple[int, int, int, int, int, int] | None:
    """Detect the largest 1px rectangle border whose interior is uniformly filled."""
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
    return [(r, c) for r in _line(r0, r1)]


def _hpath(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    return [(r, c) for c in _line(c0, c1)]


def _join(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in parts:
        for p in part:
            if not out or out[-1] != p:
                out.append((int(p[0]), int(p[1])))
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


def _ordered_simple_path(cells: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    """Return the cells ordered from one endpoint to the other for a simple 4-connected path."""
    pts = {(int(r), int(c)) for r, c in cells}
    if not pts:
        return None
    if len(pts) == 1:
        return [next(iter(pts))]

    neigh: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for p in pts:
        neigh[p] = [q for q in neighbors4(*p) if q in pts]
    if any(len(v) not in (1, 2) for v in neigh.values()):
        return None
    ends = [p for p, v in neigh.items() if len(v) == 1]
    if len(ends) != 2:
        return None

    start = min(ends)
    out = [start]
    prev = None
    cur = start
    while True:
        nxts = [q for q in neigh[cur] if q != prev]
        if not nxts:
            break
        if len(nxts) != 1:
            return None
        nxt = nxts[0]
        out.append(nxt)
        prev, cur = cur, nxt
        if len(out) > len(pts):
            return None

    if len(out) != len(pts):
        return None
    return out


def _run_length_to_first_turn(path: list[tuple[int, int]]) -> int:
    """Number of cells in the first straight run, inclusive of the endpoint cell."""
    if len(path) <= 1:
        return int(len(path))
    dirs = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dirs.append((int(r1 - r0), int(c1 - c0)))
    first = dirs[0]
    edges = 1
    while edges < len(dirs) and dirs[edges] == first:
        edges += 1
    return int(edges + 1)


def _paint_periodic(
    grid: np.ndarray, path: list[tuple[int, int]], tile: tuple[int, ...]
) -> None:
    p = len(tile)
    assert p >= 1
    for i, (r, c) in enumerate(path):
        grid[int(r), int(c)] = int(tile[i % p])


def _choose_inner_path(
    p1: tuple[int, int], p2: tuple[int, int], outH: int, outW: int
) -> list[tuple[int, int]]:
    pathA = _build_manhattan(p1, p2, True)
    pathB = _build_manhattan(p1, p2, False)
    return pathB if (_inside_corner(p1, p2, False, outH, outW) and not _inside_corner(p1, p2, True, outH, outW)) else pathA


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct framed Manhattan paths by copying periodic color motifs from outer open rectangles.

    Input structure:
    - A uniform-background grid contains one closed 1px rectangular frame with a single fill color.
    - Outside the frame are thin open-rectangle components that touch the frame ring at exactly two endpoints.
    - Each outside component is itself colored by a repeating tile along its path.

    Rule:
    - Output only the frame interior, initially filled with the frame's fill color.
    - Project each ring-touch endpoint inward to the corresponding interior boundary cell.
    - Connect the two projected endpoints with the same Manhattan path geometry as in the base task.
    - Recover the component's period from the length of its initial straight run, take that many colors as the tile, and repeat the tile along the interior path.

    Key insight: The outer component is not just a hint of endpoint colors; it already exposes the full periodic motif, and the motif length is encoded by the component's visible protrusion depth before its first turn.
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

    comp_mask = (g != int(bg)) & (g != int(fill)) & (g != int(bc))
    clab, cn = label(comp_mask, structure=STRUCT4)

    ring = set()
    for cc in range(c0, c1 + 1):
        for rr in (r0 - 1, r1 + 1):
            if 0 <= rr < H:
                ring.add((int(rr), int(cc)))
    for rr in range(r0, r1 + 1):
        for cc in (c0 - 1, c1 + 1):
            if 0 <= cc < W:
                ring.add((int(rr), int(cc)))

    ring_touching: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for rr, cc in ring:
        cid = int(clab[rr, cc])
        if cid > 0:
            ring_touching[cid].append((int(rr), int(cc)))

    for cid in range(1, int(cn) + 1):
        if cid not in ring_touching:
            continue
        comp_cells = [tuple(map(int, p)) for p in np.argwhere(clab == int(cid))]
        ordered = _ordered_simple_path(comp_cells)
        assert ordered is not None
        assert ordered[0] in ring and ordered[-1] in ring

        candidates = [ordered, list(reversed(ordered))]
        painted = False
        for seq in candidates:
            start_cell = seq[0]
            end_cell = seq[-1]
            p1 = map_to_out(*start_cell)
            p2 = map_to_out(*end_cell)
            inner_path = _choose_inner_path(p1, p2, outH, outW)

            period = _run_length_to_first_turn(seq)
            assert period >= 2
            tile = tuple(int(g[r, c]) for r, c in seq[:period])
            end_expected = int(tile[(len(inner_path) - 1) % period])
            if end_expected != int(g[end_cell]):
                continue
            _paint_periodic(out, inner_path, tile)
            painted = True
            break

        assert painted

    return out


# --------------- generator (a6 periodic tiling) ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (4, 13)
_INTERIOR_W_RANGE = (7, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 12
_FRAME_MARGIN = 5
_FRAME_MARGIN_FAR = 7
_OUTER_DEPTH_RANGE = (3, 5)
_N_COMPONENTS_RANGE = (2, 3)
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 400


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
    tile: tuple[int, ...],
    bg: int,
) -> bool:
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, bg, comp_mask):
        return False
    _paint_periodic(grid, path, tile)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _inner_tile_is_valid(tile: tuple[int, ...]) -> bool:
    if len(tile) < 2:
        return False
    if len(set(tile)) < 2:
        return False
    if tile[0] == tile[-1]:
        return False
    for p in range(1, len(tile)):
        if len(tile) % p != 0:
            continue
        if all(tile[i] == tile[i % p] for i in range(len(tile))):
            return False
    return True


def _make_tile(period: int, palette: list[int]) -> tuple[int, ...]:
    ncols = min(len(palette), max(2, min(period, 4)))
    for _ in range(200):
        k = random.randint(2, ncols)
        chosen = random.sample(palette, k)
        vals = [random.choice(chosen) for _ in range(period)]
        for col in chosen:
            vals[random.randrange(period)] = col
        tile = tuple(int(v) for v in vals)
        if _inner_tile_is_valid(tile):
            return tile
    base = random.sample(palette, min(period, len(palette)))
    while len(base) < period:
        base.append(random.choice(base))
    tile = tuple(base[:period])
    if tile[0] == tile[-1]:
        repl = next(col for col in palette if col != tile[0])
        tile = tile[:-1] + (int(repl),)
    return tile


def _l_specs(kind: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    """Build one L-type interior path and its ordered outer open-rectangle path."""
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"

    r = random.randint(1, max(1, H // 2)) if is_top else random.randint(max(1, H // 2), H - 2)
    c = random.randint(1, max(1, W // 2)) if is_left else random.randint(max(1, W // 2), W - 2)

    depth = random.randint(*_OUTER_DEPTH_RANGE)
    R = (r0 - depth) if is_top else (r1 + depth)
    C = (c0 - depth) if is_left else (c1 + depth)

    ring_a = (r0 - 1 if is_top else r1 + 1, c0 + 1 + c)
    ring_b = (r0 + 1 + r, c0 - 1 if is_left else c1 + 1)

    edge_r = 0 if is_top else H - 1
    edge_c = 0 if is_left else W - 1
    inner = _join(_vpath(edge_r, r, c), _hpath(r, c, edge_c))
    outer = _join(
        _vpath(ring_a[0], R, ring_a[1]),
        _hpath(R, ring_a[1], C),
        _vpath(R, ring_b[0], C),
        _hpath(ring_b[0], C, ring_b[1]),
    )
    period = depth
    assert period == _run_length_to_first_turn(outer)
    return inner, outer, period


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
        if _place_component(grid, comp_mask, coords, (int(color),), bg):
            # Distractors are non-ring simple paths/blobs; actual colors do not matter because solver ignores them.
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
    used_inner = np.zeros((H, W), dtype=bool)
    palette = [x for x in range(10) if x not in (bg, bc, fill)]

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    remaining = n_components

    corners = random.sample(["tl", "tr", "bl", "br"], k=remaining)
    for kind in corners:
        inner, outer, period = _l_specs(kind, H, W, r0, c0, r1, c1)
        irs, ics = zip(*inner)
        if used_inner[irs, ics].any():
            return None
        tile = _make_tile(period, palette)
        if not _place_component(grid, comp_mask, outer, tile, bg):
            return None
        _paint_periodic(expected, inner, tile)
        used_inner[irs, ics] = True

    # Distractors stay tiny and away from the frame; they should not touch the ring.
    distractor_palette = [x for x in range(10) if x not in (bg, bc, fill)]
    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        _add_distractor(grid, comp_mask, bg, (r0, c0, r1, c1), random.choice(distractor_palette))

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
    raise RuntimeError("failed to generate a valid a6f40cea periodic-tiling input")
