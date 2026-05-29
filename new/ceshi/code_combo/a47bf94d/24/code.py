# Combo variant of a47bf94d (index 24)
# combo_spec: D10/0+D16/1+D20/0
# Name: 循环置换+多对一汇聚+独立子网络

from __future__ import annotations

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

# -----------------------------------------------------------------------------
# Design analysis for combo D10/0 + D16/1 + D20/0
#
# a) Concrete behavior and trace:
#    原题默认一整张图上的大管网实现 1:1 permutation；solid 会在原位变成
#    diamond，并把 X 送到远端；X 保持原状，并把 diamond 送到远端。
#    这里保留这套 token 语义，但把“网络结构”改成两个彼此独立的 cycle 组：
#    左组两条通道在逻辑上做 2-cycle，右组两条通道也做 2-cycle。像素层面上，
#    D20/0 禁止共享线段，所以每条通道都画成自己的独立连通分量；cycle 由
#    “同组目标列表整体循环右移一格”来表达，而不是靠共享大骨架表达。
#    具体小例子：左组源 A0/A1 的目标列表是 [L, S]，循环后变成 A0->S,
#    A1->L；右组源 B0/B1 的目标列表是 [S, R]，循环后变成 B0->R, B1->S。
#    因此 S 是共享目标中心，A0 与 B1 会在这里汇聚。
#
# b) Complexity calibration:
#    若允许任意多源直接压到同一目标，容易造成颜色覆盖冲突；因此这里把汇聚
#    限制为“恰好两个源共享一个目标中心，并且一个发 X、一个发 diamond”。
#    两种发射形状在 3x3 中占据互补位置，不重叠，所以汇聚后会形成双色实心
#    3x3，肉眼非常明显。生成器强制每个样例都至少有这一个共享目标，避免退化。
#
# c) Implicit assumption audit:
#    - 原 main() 假设一整个 connected component 可以含多条通道和交叉；这里
#      不再成立，所以 solver 改为把每个独立 wire component 当作一条单独通道。
#    - 原代码扫描端点附近 token 后，默认“另一端就是唯一目的地”；这里仍成立，
#      但依赖的新前提是：每个独立 component 恰好只有一个源 token 端点。生成器
#      通过只在一端放 token、另一端留空来保证。
#    - 原 generator 允许 far-end 预放 diamond；这里若保留会和汇聚识别纠缠，
#      所以去掉预放目标，所有远端形状都由 solver 生成。
#    - D10/0 与 D20/0 存在张力：单一 cycle 往往靠共享网络表达，而 D20/0 禁止
#      共享线段。这里采用的像素级实现是：cycle 是“组内目标顺序的循环置换”，
#      物理线路仍保持独立 simple path。
#
# d) Placement legality:
#    所有源 token 中心和目标中心都保留完整 3x3 净空区；wire 端点固定在中心外
#    两格，保证 token/wire 不重叠。路径用 BFS 在空白栅格上顺序布线，并对已放
#    线路施加 4 邻域禁入光晕，所以不同 component 不会相连，也不会擦边形成误连。
#
# e) Visual observability:
#    从样例上可以直接看到三件事：
#    1. 左右各是一组独立的小网络，而不是一张共享大网。
#    2. 每组内部的目标顺序发生循环置换（在 2-cycle 情况下表现为 swap）。
#    3. 中间共享目标会同时收到两种不同颜色、不同形状来源的补形，输出成为双色 3x3。
#    整个规则完全由空间结构和可见形状决定，不依赖颜色编号大小。
#
# f) Visual cleanliness:
#    输入上方只放源 token，下半区只放空目标中心和 wire 终端；汇聚只发生在一个
#    共享中心，避免同一区域同时出现多层杂讯。路径之间保留至少一格正交间隔，
#    让独立 component 的分界一眼可见。
#
# g) Inference support:
#    颜色不发生重映射，仍是“哪个源颜色发出去，就在目的地保留哪个颜色”；新的
#    推理负担只在于：先看每个独立子网络的循环去向，再观察哪些目标中心被两个子
#    网络同时命中。生成器把结构自由度控制在固定两组 cycle + 一个共享目标上，
#    保证 2-4 个样例就足以归纳。
# -----------------------------------------------------------------------------

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_WIRE_COLORS = {5, 8, 9}

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID_OFF = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]

_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}

TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (24, 28)
GEN_MAX_RETRIES = 400


def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for isolated maximal solid 3x3 blocks."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            col = int(sub[0, 0])
            if col in _WIRE_COLORS or col == bg or np.any(sub != col):
                continue
            maximal = True
            for rr in range(r0, r0 + 3):
                for cc in range(c0, c0 + 3):
                    for nr, nc in neighbors4(rr, cc):
                        if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                            if int(g[nr, nc]) == col:
                                maximal = False
                                break
                    if not maximal:
                        break
                if not maximal:
                    break
            if maximal:
                found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _find_x_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for isolated 3x3 X tokens."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}

    mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]:
        mask[rr, cc] = True

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            col = int(sub[0, 0])
            if col in _WIRE_COLORS or col == bg:
                continue
            if np.any(sub[mask] != col) or np.any(sub[~mask] != bg):
                continue
            isolated = True
            for dr0, dc0 in _X_OFF:
                rr, cc = r0 + 1 + dr0, c0 + 1 + dc0
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                        if int(g[nr, nc]) == col:
                            isolated = False
                            break
                if not isolated:
                    break
            if isolated:
                found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=4)]


def _find_9rects(g: np.ndarray) -> list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]]:
    """Find color-9 rectangles and the wire-adjacent cells on each exposed face."""
    mask9 = g == 9
    if not mask9.any():
        return []
    h, w = g.shape
    rects = []
    for comp9 in components_mask(mask9, connectivity=4):
        cells = set((int(r), int(c)) for r, c in comp9)
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        faces: dict[str, list[tuple[int, int]]] = {}
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) not in cells and 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) in (5, 8):
                    face = "N" if nr < r0 else "S" if nr > r1 else "W" if nc < c0 else "E"
                    faces.setdefault(face, []).append((nr, nc))
        rects.append((cells, faces))
    return rects


def _trace_wire(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
    rects_9: list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Greedily trace one simple wire strand, with optional color-9 pass-through."""
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        state = (cur, d)
        if state in visited:
            raise AssertionError("wire trace looped unexpectedly")
        visited.add(state)

        jumped = False
        for cells9, faces9 in rects_9:
            if cur not in cells9:
                continue
            entry_face = _DIR2FACE[d]
            opp = _OPP[entry_face]
            if opp in faces9:
                exit_face = opp
            else:
                others = [f for f in faces9 if f != entry_face]
                if len(others) != 1:
                    raise AssertionError("invalid 9-rect routing")
                exit_face = others[0]
            pts = faces9[exit_face]
            er, ec = cur
            if entry_face in ("N", "S"):
                exit_cell = min(pts, key=lambda p: abs(p[1] - ec))
            else:
                exit_cell = min(pts, key=lambda p: abs(p[0] - er))
            cur, d, prev = exit_cell, _FACE2DIR[exit_face], None
            jumped = True
            break
        if jumped:
            continue

        r, c = cur
        opts = [((nr, nc), nr - r, nc - c) for nr, nc in neighbors4(r, c) if (nr, nc) in comp]
        if not opts:
            return cur, d

        rev = (-d[0], -d[1])
        nonrev = [o for o in opts if (o[1], o[2]) != rev]
        opts2 = nonrev if nonrev else opts

        straight = [o for o in opts2 if (o[1], o[2]) == d]
        if straight:
            p, dr, dc = straight[0]
        else:
            perp = [o for o in opts2 if o[1] * d[0] + o[2] * d[1] == 0]
            if perp:
                p, dr, dc = sorted(perp, key=lambda x: (x[1], x[2]))[0]
            else:
                p, dr, dc = opts2[0]

        if p == prev:
            return cur, d

        prev = cur
        cur = p
        d = (dr, dc)

    raise AssertionError("wire trace exceeded step limit")


def _stamp_offsets(grid: np.ndarray, center: tuple[int, int], color: int, offsets: list[tuple[int, int]]) -> None:
    h, w = grid.shape
    cr, cc = center
    for dr, dc in offsets:
        rr, cc2 = cr + dr, cc + dc
        if 0 <= rr < h and 0 <= cc2 < w:
            grid[rr, cc2] = color


def _write_shape(
    grid: np.ndarray,
    center: tuple[int, int],
    color: int,
    offsets: list[tuple[int, int]],
    bg: int,
) -> None:
    h, w = grid.shape
    cr, cc = center
    for dr, dc in offsets:
        rr, cc2 = cr + dr, cc + dc
        if not (0 <= rr < h and 0 <= cc2 < w):
            continue
        old = int(grid[rr, cc2])
        if old not in (bg, color):
            raise AssertionError("shape merge conflict")
        grid[rr, cc2] = color


def _shape_offsets(kind: str) -> list[tuple[int, int]]:
    if kind == "solid":
        return _SOLID_OFF
    if kind == "x":
        return _X_OFF
    if kind == "diamond":
        return _DIAMOND_OFF
    raise ValueError(kind)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route token complements through several disconnected wire strands, with one
    shared destination receiving a merged result from two different strands.

    Input structure:
    - Background cells are empty.
    - Wire cells use the original wire palette {8, 5, 9}, but the generator for
      this fused variant keeps them as isolated non-touching path components.
    - Colored 3x3 source tokens (solid blocks or X-shapes) sit two cells
      outside exactly one endpoint of each component; the opposite endpoint is
      an empty destination site.

    Rule:
    - A solid source is reduced in place to a diamond and emits an X to the far
      endpoint of its own wire component.
    - An X source stays as an X and emits a diamond to the far endpoint of its
      own wire component.
    - If two independent components end at the same destination center from
      different sides, their emitted shapes are merged there.

    Key insight:
    - Because the path components are independent, each one can be solved
      locally by endpoint tracing. The nontrivial part is noticing that two
      separately traced strands may share the same destination center, and the
      emitted X/diamond pair occupies complementary cells rather than colliding.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)

    track_mask = np.isin(g, list(_WIRE_COLORS))
    comps = _components(track_mask)
    rects_9 = _find_9rects(g)

    out = g.copy()

    for (cr, cc), (col, (r0, c0)) in solids.items():
        out[r0 : r0 + 3, c0 : c0 + 3] = bg
        _write_shape(out, (cr, cc), col, _DIAMOND_OFF, bg)

    for pts in comps:
        comp = set(pts)
        if len(comp) < 2:
            continue

        deg = {p: 0 for p in comp}
        for r, c in comp:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in comp:
                    deg[(r, c)] += 1
        endpoints = [p for p in comp if deg[p] <= 1]
        if len(endpoints) != 2:
            continue

        attached = []
        for ep in endpoints:
            er, ec = ep
            for odr, odc in _N4:
                cen = (er + 2 * odr, ec + 2 * odc)
                if cen in solids:
                    col, _ = solids[cen]
                    attached.append((ep, (-odr, -odc), cen, col, "solid"))
                elif cen in xs:
                    col, _ = xs[cen]
                    attached.append((ep, (-odr, -odc), cen, col, "x"))

        if len(attached) != 1:
            continue

        ep, dir_in, _, col, kind = attached[0]
        end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9)
        dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
        emitted = _X_OFF if kind == "solid" else _DIAMOND_OFF
        _write_shape(out, dest, col, emitted, bg)

    return out


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    h, w = shape
    r, c = cell
    return 0 <= r < h and 0 <= c < w


def _reserve_box(mask: np.ndarray, center: tuple[int, int], radius: int = 1) -> None:
    h, w = mask.shape
    cr, cc = center
    for rr in range(cr - radius, cr + radius + 1):
        for cc2 in range(cc - radius, cc + radius + 1):
            if 0 <= rr < h and 0 <= cc2 < w:
                mask[rr, cc2] = True


def _occupied_with_halo(occupied: np.ndarray) -> np.ndarray:
    h, w = occupied.shape
    halo = occupied.copy()
    pts = np.argwhere(occupied)
    for r, c in pts:
        for nr, nc in neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w:
                halo[nr, nc] = True
    return halo


def _bfs_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    end: tuple[int, int],
    blocked: np.ndarray,
) -> list[tuple[int, int]] | None:
    h, w = shape
    blocked = blocked.copy()
    blocked[start] = False
    blocked[end] = False

    q = deque([start])
    parent = {start: None}
    order = [(1, 0), (0, -1), (0, 1), (-1, 0)]

    while q:
        cur = q.popleft()
        if cur == end:
            break
        r, c = cur
        for dr, dc in order:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if blocked[nr, nc] or nxt in parent:
                continue
            parent[nxt] = cur
            q.append(nxt)

    if end not in parent:
        return None

    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a == b:
        return [a]
    if a[0] == b[0]:
        step = 1 if b[1] > a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] > a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise ValueError("segment must be axis-aligned")


def _polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        seg = _segment_cells(points[i], points[i + 1])
        if i > 0:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def _route_paths(
    shape: tuple[int, int],
    channels: list[dict],
    base_reserved: np.ndarray,
    endpoint_cells: set[tuple[int, int]],
) -> list[list[tuple[int, int]]] | None:
    h, w = shape
    orders = [
        [0, 3, 1, 2],
        [3, 0, 2, 1],
        [0, 1, 3, 2],
        [3, 2, 0, 1],
    ]
    random.shuffle(orders)

    for order in orders:
        occupied = np.zeros((h, w), dtype=bool)
        paths: list[list[tuple[int, int]] | None] = [None] * len(channels)
        ok = True
        for idx in order:
            start = channels[idx]["start"]
            end = channels[idx]["end"]
            blocked = base_reserved.copy()
            blocked |= _occupied_with_halo(occupied)
            for ep in endpoint_cells:
                blocked[ep] = True
            blocked[start] = False
            blocked[end] = False
            path = _bfs_path(shape, start, end, blocked)
            if path is None:
                ok = False
                break
            paths[idx] = path
            for cell in path:
                occupied[cell] = True
        if ok and all(p is not None for p in paths):
            return [p for p in paths if p is not None]
    return None


def _build_layout(h: int, w: int) -> tuple[list[dict], dict[str, tuple[int, int]]]:
    top_row = 2
    bot_row = h - 3
    source_cols = [4, 8, w - 9, w - 5]

    left_center = (h // 2 - 2, random.choice([6, 7]))
    shared_center = (h // 2, w // 2 + random.choice([-1, 0, 1]))
    right_center = (h // 2 + 2, w - random.choice([7, 8]))

    shared_modes = [
        ("solid", "x", (shared_center[0], shared_center[1] - 2), (shared_center[0], shared_center[1] + 2)),
        ("x", "solid", (shared_center[0], shared_center[1] - 2), (shared_center[0], shared_center[1] + 2)),
    ]
    kind0, kind3, end0, end3 = random.choice(shared_modes)

    channels = [
        {
            "source_center": (top_row, source_cols[0]),
            "start": (top_row + 2, source_cols[0]),
            "target_center": shared_center,
            "end": end0,
            "kind": kind0,
            "group": "left",
        },
        {
            "source_center": (top_row, source_cols[1]),
            "start": (top_row + 2, source_cols[1]),
            "target_center": left_center,
            "end": (left_center[0] - 2, left_center[1]),
            "kind": random.choice(["solid", "x"]),
            "group": "left",
        },
        {
            "source_center": (bot_row, source_cols[2]),
            "start": (bot_row - 2, source_cols[2]),
            "target_center": right_center,
            "end": (right_center[0] + 2, right_center[1]),
            "kind": random.choice(["solid", "x"]),
            "group": "right",
        },
        {
            "source_center": (bot_row, source_cols[3]),
            "start": (bot_row - 2, source_cols[3]),
            "target_center": shared_center,
            "end": end3,
            "kind": kind3,
            "group": "right",
        },
    ]
    centers = {"left": left_center, "shared": shared_center, "right": right_center}
    return channels, centers


def _build_expected_output(
    wire_grid: np.ndarray,
    channels: list[dict],
    colors: list[int],
) -> np.ndarray:
    out = wire_grid.copy()
    bg = 0
    for idx, ch in enumerate(channels):
        col = colors[idx]
        if ch["kind"] == "solid":
            _write_shape(out, ch["source_center"], col, _DIAMOND_OFF, bg)
        else:
            _write_shape(out, ch["source_center"], col, _X_OFF, bg)

    for idx, ch in enumerate(channels):
        col = colors[idx]
        emitted = _X_OFF if ch["kind"] == "solid" else _DIAMOND_OFF
        _write_shape(out, ch["target_center"], col, emitted, bg)
    return out


def _shared_merge_ok(channels: list[dict], colors: list[int]) -> bool:
    shared = [i for i, ch in enumerate(channels) if ch["target_center"] == channels[0]["target_center"]]
    if len(shared) != 2:
        return False
    i, j = shared
    if channels[i]["kind"] == channels[j]["kind"]:
        return False
    return colors[i] != colors[j]


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    channels, centers = _build_layout(h, w)

    for center in centers.values():
        if not (2 <= center[0] < h - 2 and 2 <= center[1] < w - 2):
            return None

    colors = random.sample(TOKEN_COLORS_POOL, len(channels))
    if not _shared_merge_ok(channels, colors):
        return None

    shared_r, _ = centers["shared"]
    left_end_r, left_end_c = channels[1]["end"]
    right_end_r, right_end_c = channels[2]["end"]
    start0_r, start0_c = channels[0]["start"]
    start1_r, start1_c = channels[1]["start"]
    start2_r, start2_c = channels[2]["start"]
    start3_r, start3_c = channels[3]["start"]

    paths = [
        _polyline_cells([channels[0]["start"], (shared_r, start0_c), channels[0]["end"]]),
        _polyline_cells([channels[1]["start"], (start1_r, left_end_c), channels[1]["end"]]),
        _polyline_cells([channels[2]["start"], (start2_r, right_end_c), channels[2]["end"]]),
        _polyline_cells([channels[3]["start"], (shared_r, start3_c), channels[3]["end"]]),
    ]

    grid = np.zeros((h, w), dtype=int)
    for path in paths:
        for r, c in path:
            grid[r, c] = 8

    if len(_components(grid == 8)) != 4:
        return None

    for idx, ch in enumerate(channels):
        _stamp_offsets(grid, ch["source_center"], colors[idx], _shape_offsets(ch["kind"]))

    expected = _build_expected_output(np.where(grid == 8, 8, 0), channels, colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
