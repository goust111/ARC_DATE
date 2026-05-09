# Combo variant of 4c416de3 (index 17)
# combo_spec: D7/1+D9/1
# Name: 嵌套框架+链式传播

import random

import numpy as np

from arc_utils import draw_line, get_bg


# Design analysis
# a) 具体化与行为追踪
#    - 本融合版只保留一组两层同心嵌套的完整 0 方框：outer 与 inner，层间固定留 3 格。
#      输入里只有 outer 的一个角点是完整 source；另外两个 outer 角点各有一个 marker，
#      且它们在 inner 的同角位上也各有一个 marker。剩余第四种角位只放一个未配对 decoy marker。
#    - 行为追踪：
#      原题：所有 marker 角点都被同一个规范图案直接补全，彼此独立。
#      D7 单变体：只有相邻嵌套层同角位之间才会复制，孤立 marker 不动。
#      本融合：先在 outer 层做一段最近邻链，走完 outer 的两个 active marker 后，
#      再继续钻入 inner 层，并只处理“outer 里已参与链条的那两个同角位”。
#      同时沿用 D9 的可见链痕迹：每次落点时，图案的 outward tail 继承上一步颜色，
#      corner/inward head 使用当前 marker 自己的颜色。
#    - 具体小例子：若 outer TL 是 source，outer 的 TR 与 BR 为 active marker，
#      inner 的 TR 与 BR 也各有 marker，BL 只有一个 decoy marker。
#      则输出顺序是 TL(源) -> TR(outer) -> BR(outer) -> BR(inner) -> TR(inner)。
#      与原题不同，inner 不会在一开始就补全；与纯嵌套复制不同，补全后的图案还是双色，
#      记录了链式传播的先后顺序。
# b) 复杂度校准
#    - 若 inner 只做“同角位单步复制”，视觉上会太像 D7；若去掉双色 tail/head，顺序信息又会退化。
#      因此这里强制两段都使用链式双色着色，使融合行为在输出里肉眼可见。
#    - 自由度刻意收敛到少量档位：固定 2 层、固定 2 个 active 角位、固定 1 个 decoy 角位。
#      这样样例之间会重复出现“outer 两跳 + inner 两跳 + 一个不动 marker”的结构支撑。
# c) 隐式假设审查
#    - 原题假设“所有 marker 都是 target”已失效；这里必须先恢复 nested frames，再用
#      outer/inner 同角位是否成对来筛出 active 角位，未配对 marker 是 decoy。
#    - 原题的 target 着色是单色；这里改成 outward tail 用上一步颜色、其余位置用本角 marker 颜色，
#      所以 `main()` 必须把规范图案拆成 outward/head 两部分。
#    - 原题处理顺序无关；这里顺序决定颜色，因此 outer phase 与 inner phase 都要逐步选最近邻。
#    - 为避免 nested 同角位的邻域相互混入，图案模板去掉 (-1,-1)，并固定 gap=3、k_max<=2。
#      generator 显式保证这些前提。
# d) 放置合法性
#    - 生成器先选 inner size，再按固定 gap 向外扩成 outer，所以两层天然不重叠且严格嵌套。
#    - 图案只取 `(-1,0),(0,-1),(0,0),(1,1),(2,2)` 的子集，所有着色都落在 frame 周围 2 格内，
#      不会跨层或跨角冲突。
#    - output-first 构造后仍用 `main(input)` 回验，防止隐藏冲突或越界。
# e) 视觉可观察性
#    - 人类可以直接看到两点：
#      1. 先补全 outer 的 marker，再补全 inner 的对应 marker；
#      2. 补全后的图案不是纯本色，而是带着前一步留下来的 outward tail。
#    - decoy marker 提供反例，说明“有 marker 并不够”，必须既满足嵌套配对，也满足阶段顺序。
# f) 可视化整洁度
#    - 只用一组两层框，区域集中且层级清楚；active 与 decoy 只占四个标准角位，不会杂乱堆叠。
#    - 每个角位最多只有一层 source/marker 语义，且颜色只承担链路记录，不参与任何隐藏排序。
# g) 变化的可推断性
#    - 每个链节点都有结构提示：其 own color 只出现在 head/inward 部分，而 tail 明显继承前一步颜色。
#    - inner 的两个 active 角位与 outer 的 active 角位一一对应，给出了“为何第二阶段只处理它们”的
#      可见线索。颜色变化不依赖数值大小，只依赖局部位置与链顺序。
#
# New main() assumptions checked against generator:
# - 恰好 2 层完整同心方框，固定 gap 为 3。
# - 恰好 1 个 outer source，且图案总包含 (0,0)，因此 source 角本身有颜色。
# - 恰好 2 个 active kinds：它们同时出现在 outer 和 inner；另有 1 个未配对 decoy marker。
# - outer phase 的第一跳距离唯一：active kinds 总是“一个相邻角 + 一个对角”。
# - inner phase 从最后一个 outer 节点开始，因此其同角位 inner 节点会成为唯一最近目标。


CORNER_KINDS = ("TL", "TR", "BL", "BR")
_CORNER_OFFSETS = {"TL": (0, 0), "TR": (0, 1), "BL": (1, 0), "BR": (1, 1)}
OPPOSITE_KIND = {"TL": "BR", "TR": "BL", "BL": "TR", "BR": "TL"}
ADJACENT_KINDS = {
    "TL": ("TR", "BL"),
    "TR": ("TL", "BR"),
    "BL": ("TL", "BR"),
    "BR": ("TR", "BL"),
}

_NEIGH_R = 3
NEST_GAP = 3

H_RANGE = (19, 24)
W_RANGE = (19, 24)
BORDER = 1
INNER_SIZE_RANGE = (5, 8)
MAX_ATTEMPTS = 500

# Omit (-1,-1) so adjacent nested corners never leak into each other's local scan.
PATTERN_TEMPLATES = (
    [(-1, 0), (0, -1), (0, 0), (1, 1)],
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)],
)


def _rot(dr: int, dc: int, kind: str) -> tuple[int, int]:
    """Map actual offset <-> canonical TL-oriented offset; self-inverse."""
    if kind == "TL":
        return dr, dc
    if kind == "TR":
        return dr, -dc
    if kind == "BL":
        return -dr, dc
    if kind == "BR":
        return -dr, -dc
    raise ValueError(kind)


def _is_outward(off: tuple[int, int]) -> bool:
    return off[0] < 0 or off[1] < 0


def _corner_coord(frame: dict, kind: str) -> tuple[int, int]:
    dr, dc = _CORNER_OFFSETS[kind]
    span = frame["size"] - 1
    return frame["r"] + dr * span, frame["c"] + dc * span


def _colored_around(
    g: np.ndarray, r: int, c: int, kind: str, bg: int
) -> list[tuple[tuple[int, int], int]]:
    H, W = g.shape
    items: list[tuple[tuple[int, int], int]] = []
    for dr in range(-_NEIGH_R, _NEIGH_R + 1):
        for dc in range(-_NEIGH_R, _NEIGH_R + 1):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            v = int(g[rr, cc])
            if v == bg or v == 0:
                continue
            items.append((_rot(dr, dc, kind), v))
    return items


def _scan_frames(g: np.ndarray, bg: int) -> list[dict]:
    """Detect the best 2-layer concentric stack of complete square 0-frames."""
    H, W = g.shape
    frames: list[dict] = []
    max_size = min(H, W)
    for size in range(5, max_size + 1):
        span = size - 1
        for r in range(0, H - size + 1):
            for c in range(0, W - size + 1):
                r2, c2 = r + span, c + span
                corners = (int(g[r, c]), int(g[r, c2]), int(g[r2, c]), int(g[r2, c2]))
                if any(v == bg for v in corners):
                    continue
                if not (
                    np.all(g[r, c + 1 : c2] == 0)
                    and np.all(g[r2, c + 1 : c2] == 0)
                    and np.all(g[r + 1 : r2, c] == 0)
                    and np.all(g[r + 1 : r2, c2] == 0)
                ):
                    continue
                frames.append({"r": r, "c": c, "size": size})
    if not frames:
        return []

    center_groups: dict[tuple[int, int], list[dict]] = {}
    for frame in frames:
        key = (2 * frame["r"] + frame["size"] - 1, 2 * frame["c"] + frame["size"] - 1)
        center_groups.setdefault(key, []).append(frame)

    best_pair: list[dict] = []
    best_score = (-1, -1)
    for group in center_groups.values():
        dedup = {(f["r"], f["c"], f["size"]): f for f in group}
        ordered = sorted(dedup.values(), key=lambda f: f["size"], reverse=True)
        by_size = {f["size"]: f for f in ordered}
        for outer in ordered:
            inner_size = outer["size"] - 2 * NEST_GAP
            if inner_size not in by_size:
                continue
            pair = [outer, by_size[inner_size]]
            score = (outer["size"], -pair[1]["size"])
            if score > best_score:
                best_score = score
                best_pair = pair
    return best_pair


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _next_unique_nearest(
    curr_corner: tuple[int, int], remaining: list[tuple[int, int, str, int]]
) -> tuple[int, int, str, int]:
    scored = [(_manhattan(curr_corner, t[:2]), idx, t) for idx, t in enumerate(remaining)]
    scored.sort(key=lambda x: x[0])
    assert scored, "no remaining targets"
    best_d = scored[0][0]
    assert len(scored) == 1 or scored[1][0] != best_d, (
        f"nearest-target tie from {curr_corner}: {[t for d, _, t in scored if d == best_d]}"
    )
    return scored[0][2]


def _draw_frame(grid: np.ndarray, r: int, c: int, size: int) -> None:
    r2, c2 = r + size - 1, c + size - 1
    draw_line(grid, r, c, end_r=r, end_c=c2, color=0)
    draw_line(grid, r, c, end_r=r2, end_c=c, color=0)
    draw_line(grid, r2, c, end_r=r2, end_c=c2, color=0)
    draw_line(grid, r, c2, end_r=r2, end_c=c2, color=0)


def _paint_pattern(
    grid: np.ndarray, r: int, c: int, kind: str, pattern: list[tuple[int, int]], color: int
) -> None:
    for dr, dc in pattern:
        rr, cc = _rot(dr, dc, kind)
        grid[r + rr, c + cc] = color


def _apply_chain(
    out: np.ndarray,
    source_corner: tuple[int, int],
    source_color: int,
    outer_targets: list[tuple[int, int, str, int]],
    inner_targets: list[tuple[int, int, str, int]],
    canon: frozenset[tuple[int, int]],
    outward_canon: frozenset[tuple[int, int]],
    bg: int,
) -> tuple[list[str], list[str]]:
    curr_corner = source_corner
    curr_color = int(source_color)
    outer_order: list[str] = []
    inner_order: list[str] = []

    remaining_outer = list(outer_targets)
    while remaining_outer:
        target = _next_unique_nearest(curr_corner, remaining_outer)
        r, c, kind, own_color = target
        for cdr, cdc in canon:
            adr, adc = _rot(cdr, cdc, kind)
            rr, cc = r + adr, c + adc
            paint = curr_color if (cdr, cdc) in outward_canon else own_color
            prev = int(out[rr, cc])
            assert prev == bg or prev == 0 or prev == paint, (
                f"paint conflict at ({rr},{cc}) while landing on outer {target}: {prev} vs {paint}"
            )
            out[rr, cc] = paint
        curr_corner = (r, c)
        curr_color = int(own_color)
        outer_order.append(kind)
        remaining_outer.remove(target)

    remaining_inner = list(inner_targets)
    while remaining_inner:
        target = _next_unique_nearest(curr_corner, remaining_inner)
        r, c, kind, own_color = target
        for cdr, cdc in canon:
            adr, adc = _rot(cdr, cdc, kind)
            rr, cc = r + adr, c + adc
            paint = curr_color if (cdr, cdc) in outward_canon else own_color
            prev = int(out[rr, cc])
            assert prev == bg or prev == 0 or prev == paint, (
                f"paint conflict at ({rr},{cc}) while landing on inner {target}: {prev} vs {paint}"
            )
            out[rr, cc] = paint
        curr_corner = (r, c)
        curr_color = int(own_color)
        inner_order.append(kind)
        remaining_inner.remove(target)

    return outer_order, inner_order


def _choose_pattern() -> tuple[list[tuple[int, int]], int]:
    pattern = list(random.choice(PATTERN_TEMPLATES))
    k_max = max(dr for dr, dc in pattern if dr == dc and dr > 0)
    return pattern, k_max


def _build_nested_frames(H: int, W: int) -> list[dict] | None:
    max_inner = min(H, W) - 2 * BORDER - 2 * NEST_GAP
    lo = INNER_SIZE_RANGE[0]
    hi = min(INNER_SIZE_RANGE[1], max_inner)
    if hi < lo:
        return None
    inner_size = random.randint(lo, hi)
    outer_size = inner_size + 2 * NEST_GAP
    r0 = random.randint(BORDER, H - outer_size - BORDER)
    c0 = random.randint(BORDER, W - outer_size - BORDER)
    return [
        {"r": r0, "c": c0, "size": outer_size},
        {"r": r0 + NEST_GAP, "c": c0 + NEST_GAP, "size": inner_size},
    ]


def _allocate_roles() -> tuple[str, tuple[str, str], tuple[int, str]]:
    source_kind = random.choice(CORNER_KINDS)
    adj_a, adj_b = ADJACENT_KINDS[source_kind]
    chosen_adj = random.choice((adj_a, adj_b))
    decoy_kind = adj_b if chosen_adj == adj_a else adj_a
    active_kinds = (chosen_adj, OPPOSITE_KIND[source_kind])
    decoy_level = random.choice((0, 1))
    return source_kind, active_kinds, (decoy_level, decoy_kind)


def _role_colors(
    bg: int,
    source_kind: str,
    active_kinds: tuple[str, str],
    decoy_info: tuple[int, str],
) -> dict[tuple[int, str], tuple[str, int]]:
    palette = [c for c in range(1, 10) if c != bg]
    needed = 1 + 2 + 2 + 1
    colors = random.sample(palette, needed)
    out: dict[tuple[int, str], tuple[str, int]] = {}
    out[(0, source_kind)] = ("source", colors[0])
    out[(0, active_kinds[0])] = ("marker", colors[1])
    out[(0, active_kinds[1])] = ("marker", colors[2])
    out[(1, active_kinds[0])] = ("marker", colors[3])
    out[(1, active_kinds[1])] = ("marker", colors[4])
    decoy_level, decoy_kind = decoy_info
    out[(decoy_level, decoy_kind)] = ("marker", colors[5])
    return out


def _render_markers_only(
    base: np.ndarray,
    frames: list[dict],
    roles: dict[tuple[int, str], tuple[str, int]],
    pattern: list[tuple[int, int]],
    marker_canon: tuple[int, int],
) -> np.ndarray:
    grid = base.copy()
    for (level, kind), (role, color) in roles.items():
        r, c = _corner_coord(frames[level], kind)
        if role == "source":
            _paint_pattern(grid, r, c, kind, pattern, color)
        else:
            dr, dc = _rot(*marker_canon, kind)
            grid[r + dr, c + dc] = color
    return grid


def _derive_input(
    output: np.ndarray,
    base: np.ndarray,
    frames: list[dict],
    roles: dict[tuple[int, str], tuple[str, int]],
    pattern: list[tuple[int, int]],
    marker_canon: tuple[int, int],
) -> np.ndarray:
    inp = output.copy()
    for (level, kind), (role, _color) in roles.items():
        if role != "marker":
            continue
        r, c = _corner_coord(frames[level], kind)
        mdr, mdc = _rot(*marker_canon, kind)
        keep = (r + mdr, c + mdc)
        for dr, dc in pattern:
            rr, cc = _rot(dr, dc, kind)
            rr += r
            cc += c
            if (rr, cc) != keep:
                inp[rr, cc] = base[rr, cc]
    return inp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate one corner glyph across a nested two-stage chain.

    Input structure:
    - The grid contains exactly one pair of concentric square 0-frames: an outer frame and an inner frame.
    - Exactly one outer corner is the initial source and already shows the full shared glyph in one color.
    - Two other corner kinds appear as paired markers on both layers; one extra marker may appear only on a
      single layer as a decoy.

    Rule:
    - Recover the canonical TL-oriented glyph from the unique outer source and split it into an outward tail
      part and a corner/inward head part.
    - First, among the paired markers on the outer frame, repeatedly land on the unique nearest remaining
      target. Then continue the same process on the paired markers of the inner frame.
    - Each landing paints the rotated glyph with a chained color rule: outward tail cells inherit the previous
      step's color, while head/inward cells use the current marker's own color.

    Key insight:
    - A marker is productive only if its corner kind is paired across the nested layers, and the process is
      staged: outer-layer hops happen before inner-layer hops. The two-color landed glyphs make that order
      visible, because every hop carries the previous color forward into the new tail.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    frames = _scan_frames(g, bg)
    assert len(frames) == 2, f"expected exactly 2 nested frames, found {len(frames)}"
    outer, inner = frames
    center_key = (2 * outer["r"] + outer["size"] - 1, 2 * outer["c"] + outer["size"] - 1)
    inner_key = (2 * inner["r"] + inner["size"] - 1, 2 * inner["c"] + inner["size"] - 1)
    assert center_key == inner_key, "frames are not concentric"
    assert inner["r"] - outer["r"] == NEST_GAP and inner["c"] - outer["c"] == NEST_GAP, "bad nesting gap"
    assert outer["size"] - inner["size"] == 2 * NEST_GAP, "bad nested sizes"

    per_corner: dict[tuple[int, str], list[tuple[tuple[int, int], int]]] = {}
    for level, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            per_corner[(level, kind)] = _colored_around(g, r, c, kind, bg)

    sources: list[tuple[str, int]] = []
    for kind in CORNER_KINDS:
        r, c = _corner_coord(outer, kind)
        corner_color = int(g[r, c])
        if corner_color == bg or corner_color == 0:
            continue
        items = per_corner[(0, kind)]
        outward = [(off, color) for off, color in items if _is_outward(off) and color == corner_color]
        if len(outward) >= 2:
            sources.append((kind, corner_color))
    assert len(sources) == 1, f"expected exactly one outer source, got {len(sources)}"
    source_kind, source_color = sources[0]

    source_items = per_corner[(0, source_kind)]
    canon = frozenset(off for off, color in source_items if color == source_color)
    assert len(canon) >= 4, "source pattern too small"
    for dr, dc in canon:
        assert (dc, dr) in canon, f"pattern not diagonally symmetric: ({dr},{dc}) has no mirror"
    outward_canon = frozenset(off for off in canon if _is_outward(off))
    head_canon = frozenset(off for off in canon if not _is_outward(off))
    assert len(outward_canon) >= 2, "pattern needs outward tail cells"
    diag_ks = [dr for dr, dc in head_canon if dr == dc and dr > 0]
    assert diag_ks, "pattern has no inward diagonal marker anchor"
    marker_k = max(diag_ks)
    marker_canon = (marker_k, marker_k)

    marker_colors: dict[tuple[int, str], int] = {}
    for level, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            if level == 0 and kind == source_kind:
                continue
            r, c = _corner_coord(frame, kind)
            mdr, mdc = _rot(*marker_canon, kind)
            mr, mc = r + mdr, c + mdc
            v = int(g[mr, mc])
            if v == bg or v == 0:
                continue
            for off, color in per_corner[(level, kind)]:
                if off in canon:
                    assert color == v, (
                        f"marker corner {(level, kind)} has conflicting color {color} at {off}; expected {v}"
                    )
            marker_colors[(level, kind)] = v

    outer_kinds = {kind for (level, kind) in marker_colors if level == 0}
    inner_kinds = {kind for (level, kind) in marker_colors if level == 1}
    active_kinds = outer_kinds & inner_kinds
    assert len(active_kinds) == 2, f"expected exactly 2 paired active kinds, got {sorted(active_kinds)}"
    assert source_kind not in active_kinds, "source kind cannot also be active"

    outer_targets = []
    inner_targets = []
    for kind in active_kinds:
        r, c = _corner_coord(outer, kind)
        outer_targets.append((r, c, kind, marker_colors[(0, kind)]))
        r, c = _corner_coord(inner, kind)
        inner_targets.append((r, c, kind, marker_colors[(1, kind)]))

    out = g.copy()
    source_corner = _corner_coord(outer, source_kind)
    outer_order, inner_order = _apply_chain(
        out, source_corner, source_color, outer_targets, inner_targets, canon, outward_canon, bg
    )
    assert len(outer_order) == 2 and len(inner_order) == 2, "chain must have two outer and two inner hops"
    return out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    bg = random.randint(1, 9)
    pattern, k_max = _choose_pattern()
    marker_canon = (k_max, k_max)

    frames = _build_nested_frames(H, W)
    if frames is None:
        return None

    base = np.full((H, W), bg, dtype=int)
    for frame in frames:
        _draw_frame(base, frame["r"], frame["c"], frame["size"])

    source_kind, active_kinds, decoy_info = _allocate_roles()
    roles = _role_colors(bg, source_kind, active_kinds, decoy_info)

    marker_grid = _render_markers_only(base, frames, roles, pattern, marker_canon)
    output = marker_grid.copy()

    source_corner = _corner_coord(frames[0], source_kind)
    source_color = roles[(0, source_kind)][1]
    canon = frozenset(pattern)
    outward_canon = frozenset(off for off in canon if _is_outward(off))

    outer_targets = []
    inner_targets = []
    for kind in active_kinds:
        r, c = _corner_coord(frames[0], kind)
        outer_targets.append((r, c, kind, roles[(0, kind)][1]))
        r, c = _corner_coord(frames[1], kind)
        inner_targets.append((r, c, kind, roles[(1, kind)][1]))

    outer_order, inner_order = _apply_chain(
        output, source_corner, source_color, outer_targets, inner_targets, canon, outward_canon, bg
    )

    # Guard the intended fusion behavior explicitly.
    assert outer_order[0] == active_kinds[0], "outer phase must start at the chosen adjacent kind"
    assert inner_order[0] == outer_order[-1], "inner phase must begin at the last outer kind's nested partner"

    inp = _derive_input(output, base, frames, roles, pattern, marker_canon)

    # Variant-specific guarantees:
    # - exactly one decoy marker remains single in the output;
    # - each inner active node is bi-colored (tail color differs from own marker color).
    decoy_level, decoy_kind = decoy_info
    dr, dc = _rot(*marker_canon, decoy_kind)
    rr, cc = _corner_coord(frames[decoy_level], decoy_kind)
    decoy_cell = (rr + dr, cc + dc)
    if int(output[decoy_cell]) != roles[(decoy_level, decoy_kind)][1]:
        return None

    for kind in active_kinds:
        r, c = _corner_coord(frames[1], kind)
        own = roles[(1, kind)][1]
        tail_colors = set()
        for off in outward_canon:
            dr, dc = _rot(*off, kind)
            tail_colors.add(int(output[r + dr, c + dc]))
        if not tail_colors or tail_colors == {own}:
            return None

    solved = main(inp)
    if not np.array_equal(solved, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
