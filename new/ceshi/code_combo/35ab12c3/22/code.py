# Combo variant of 35ab12c3 (index 22)
# combo_spec: D5/1+D16/0
# Name: 不过滤共线偏移+多层嵌套阴影
"""
Design analysis for the combo variant
=====================================

[a] 具体化与行为追踪
- 原题：同色顶点先按角度成环/成折线，输出把主色边连起来；若起始顶点旁边有一个异色单像素，就把整条主形状按该单步偏移平移一次作为单层阴影，而且原 solver 会跳过“阴影偏移与边方向共线”的边。
- 融合后：
  1. 每个 shape 都强制带 2-3 层阴影链。输入里，基色起始顶点 `p0` 旁边是第一层锚点 `p1`，`p1` 旁边是第二层锚点 `p2`，第三层同理。
  2. 第 k 层的整条轮廓画在 `S0 + (offset1+...+offsetk)` 上，而不是所有层都直接相对 `p0`。
  3. 不再过滤“累计偏移与边方向共线”的情况；因此某些阴影边即使沿着主边的延长线，也会被尝试绘制。
- 小例子：主形状是一个 2x2 方框，基色锚点在左上角。若链步长依次为 `(1,1)`、`(-1,1)`，第二层累计偏移是 `(0,2)`，与方框的水平边共线。原始过滤逻辑会把这条第二层水平影边整段跳过；融合后它会被画出，于是方框右侧出现一圈继续向外推开的轮廓，看起来像逐层加粗。

[b] 复杂度校准
- 颜色预算仍只有 `{1,2,3,4,6,7,8}` 七色，因此不能让 2 个 shape 都带 3 层阴影。这里保留 1-2 个 shape，但层数计划只允许 `[2]`、`[3]`、`[2,2]`、`[2,3]`、`[3,2]`。
- 若层数太少或没有共线层，视觉上会退化成普通 D5 多层平移轮廓；因此生成器额外强制每个 shape patch 都要让“不过滤共线”真的改变输出。
- 若层间完全自由重叠，画面会变脏且 solver 的输入清障条件会被锚点破坏；因此允许有限重叠，但要求每一层都至少新增可见像素，并且锚链本身不落在已有轮廓上。

[c] 隐式假设审查
- 原 `carried` 假设所有阴影锚点都直接贴着起始顶点；融合后失效，所以 solver 必须改成沿邻接链递归读取 2-3 层。
- 原按单步偏移做叉积过滤；融合后这条前提被移除。对于多层链，真正对应的“阴影位置”是累计偏移，所以验证退化时要用累计偏移与边方向比较。
- 原 `_clear_in_original(... allow=set())` 仍保留，这意味着输入里的锚点不能落到别层边的内部，否则合法边会被误判阻塞。生成器显式避免锚链节点落在已有轮廓单元上，并用 `main(inp)` 复核。
- 原 `len(pts_rc) < 2` 会跳过阴影色；这个前提仍成立，因为阴影色在输入中依旧只有一个链锚点，不会被单独当 shape。

[d] 放置合法性
- 先在局部 patch 内构造一个 shape + 2/3 层阴影链，验证 solver 可还原，再整体粘贴到大画布。
- patch 内检查：
  1. 每个新锚点只和链前驱 8 邻接，不贴其他顶点/旧锚点。
  2. 每层平移轮廓都至少贡献 2 个新可见像素，避免整层被前层吞没。
  3. 每个锚点不落在已有轮廓单元上，避免 `_clear_in_original` 被锚点阻断。
- 全局仍用 `random_free_location(... padding=2)` 隔开不同对象，避免多对象外层阴影互贴。

[e] 视觉可观察性
- 输入里会出现稳定的“顶点组 + 短彩色链”，输出里出现 2-3 道同形轮廓；这是 D5 的主特征。
- 其中至少一层会让 unfiltered solver 比 filtered solver 多画出一批像素；这批像素沿主边方向继续推进，是 D16 的可见特征。
- 这些规律都由相对位置决定，不依赖颜色编号排序，解题者能从 2-3 组样例看到重复模式。

[f] 可视化整洁度
- 把全局 padding 提高到 2，减少不同对象外轮廓互撞。
- 每个 shape 只允许 2 或 3 层，且 2-shape 情况下总颜色数受控，不制造高密度彩色噪声。
- 允许共线层，但仍要求每层有新增像素，避免完全重叠成不可见的脏块。

[g] 变化的可推断性
- 颜色变化没有隐藏映射：输入链上的第 1/2/3 个单像素，直接对应输出里的第 1/2/3 层轮廓颜色。
- “不过滤共线”也不引入额外的颜色条件；它只改变某些层在几何上是否继续沿主边方向生长。
- 生成器还会拒绝那些虽然带链式阴影、但 filtered / unfiltered 输出完全相同的退化样本，保证融合行为真实出现。

新增 solver 假设与 generator 保证
-----------------------------------
- solver 假设：起始顶点附近至多存在一条简单阴影链。generator 通过局部锚点邻接检查保证这一点。
- solver 假设：累计偏移后的阴影边，如果要成功绘制，其内部路径在输入中不能被别的锚点挡住。generator 通过“锚点不得落在已有轮廓单元上”与 `main(inp)` 复核来保证。
- solver 假设：至少有一个 base 顶点靠近第一层锚点，因此能稳定选到链起点。generator 只把链挂在 `min(vertices)` 对应的起始顶点上。
"""

import math

import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg, paste, random_free_location

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _aligned_8(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
    if dr == 0 and dc != 0:
        return True
    if dc == 0 and dr != 0:
        return True
    if dr != 0 and abs(dr) == abs(dc):
        return True
    return False


def _step_dir(r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    dr = r1 - r0
    dc = c1 - c0
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    return sr, sc


def _segment_cells(r0: int, c0: int, r1: int, c1: int):
    sr, sc = _step_dir(r0, c0, r1, c1)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    for k in range(steps + 1):
        yield r0 + sr * k, c0 + sc * k


def _clear_in_original(g: np.ndarray, r0: int, c0: int, r1: int, c1: int, bg: int, allow: set[int]) -> bool:
    first = True
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if first:
            first = False
            continue
        if rr == r1 and cc == c1:
            break
        v = int(g[rr, cc])
        if v != bg and v not in allow:
            return False
    return True


def _draw_line_bg_only(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, col: int, bg: int) -> None:
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if out[rr, cc] == bg:
            out[rr, cc] = col


def _angle_cycle(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points_rc) / len(points_rc)
    cc = sum(c for _, c in points_rc) / len(points_rc)

    def key(p: tuple[int, int]):
        r, c = p
        ang = math.atan2(r - cr, c - cc)
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return (ang, rad, r, c)

    return sorted(points_rc, key=key)


def _extract_shadow_chain(
    g: np.ndarray,
    start_rc: tuple[int, int],
    base_col: int,
    bg: int,
    max_layers: int = 3,
) -> list[tuple[int, int, int]]:
    """Read a simple 8-neighbor chain of shadow-anchor pixels."""
    h, w = g.shape
    chain: list[tuple[int, int, int]] = []
    used_colors = {base_col}
    used_positions = {start_rc}
    prev: tuple[int, int] | None = None
    cur = start_rc

    for _ in range(max_layers):
        candidates: list[tuple[int, int, int, int, int]] = []
        for drn, dcn in DIR8:
            rr = cur[0] + drn
            cc = cur[1] + dcn
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if prev is not None and (rr, cc) == prev:
                continue
            if (rr, cc) in used_positions:
                continue
            v = int(g[rr, cc])
            if v == bg or v in used_colors:
                continue
            candidates.append((rr, cc, drn, dcn, v))
        if not candidates:
            break
        rr, cc, drn, dcn, v = min(candidates, key=lambda x: (x[0], x[1], x[4]))
        chain.append((drn, dcn, v))
        used_colors.add(v)
        used_positions.add((rr, cc))
        prev, cur = cur, (rr, cc)

    return chain


def _render(grid: np.ndarray, *, filter_collinear: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        cycle = _angle_cycle(pts_rc)
        if len(cycle) < 2:
            continue

        seed_idxs = []
        for i, (r, c) in enumerate(cycle):
            for drn, dcn in DIR8:
                rr = r + drn
                cc = c + dcn
                if 0 <= rr < h and 0 <= cc < w:
                    v = int(g[rr, cc])
                    if v != bg and v != col:
                        seed_idxs.append(i)
                        break

        if seed_idxs:
            start = min(seed_idxs, key=lambda i: (cycle[i][0], cycle[i][1]))
        else:
            start = min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))

        cycle = cycle[start:] + cycle[:start]

        if len(cycle) == 2:
            edges = [(cycle[0], cycle[1])]
        else:
            edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

        drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for (r0, c0), (r1, c1) in edges:
            if not _aligned_8(r0, c0, r1, c1):
                continue
            if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
                continue
            drawable_edges.append(((r0, c0), (r1, c1)))

        for (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_bg_only(out, r0, c0, r1, c1, col, bg)

        chain = _extract_shadow_chain(g, cycle[0], col, bg, max_layers=3)
        cum_r = 0
        cum_c = 0
        for drn, dcn, ncol in chain:
            cum_r += drn
            cum_c += dcn
            for (r0, c0), (r1, c1) in drawable_edges:
                if filter_collinear:
                    drs, dcs = _step_dir(r0, c0, r1, c1)
                    if drs * cum_c - dcs * cum_r == 0:
                        continue
                rr0 = r0 + cum_r
                cc0 = c0 + cum_c
                rr1 = r1 + cum_r
                cc1 = c1 + cum_c
                if not (0 <= rr0 < h and 0 <= cc0 < w and 0 <= rr1 < h and 0 <= cc1 < w):
                    continue
                if not _aligned_8(rr0, cc0, rr1, cc1):
                    continue
                if not _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow=set()):
                    continue
                _draw_line_bg_only(out, rr0, cc0, rr1, cc1, ncol, bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color vertex set into a line/polygon, then add 2-3 nested shadows.

    Input structure:
    Several colors appear either as shape vertices (2+ cells of one color) or as single
    shadow-anchor pixels that form a short 8-neighbor chain off one chosen start vertex.

    Rule:
    Draw the base shape along H/V/45-degree edges. Then follow the anchor chain in order;
    after each step, copy the full base shape at the cumulative offset using that layer's
    color. Unlike the original task, edges are not filtered out when the cumulative shadow
    offset is collinear with the edge direction.

    Key insight:
    The extra single pixels are not separate objects; they encode a recursive offset chain.
    Because collinear copies are now allowed, some layers visibly continue along the same
    line as the base edge, creating a stepped thickening effect.
    """
    return _render(np.asarray(grid, dtype=int), filter_collinear=False)


def _render_filtered(grid: np.ndarray) -> np.ndarray:
    return _render(np.asarray(grid, dtype=int), filter_collinear=True)


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_PLANS = [
    (2, [2, 2]),
    (2, [2, 3]),
    (2, [3, 2]),
    (1, [2]),
    (1, [3]),
]
SHAPE_PLAN_WEIGHTS = np.array([0.25, 0.20, 0.15, 0.20, 0.20], dtype=float)
DISTRACTOR_PROB = 0.35
PLACEMENT_PADDING = 2
PLACEMENT_PAD_CONNECTIVITY = 8

LINE_VERTICES_RANGE = (2, 8)
LINE_TOTAL_LENGTH = 13
LINE_MIN_GAP = 2
TRIANGLE_SIDE_RANGE = (2, 8)
RECT_H_RANGE = (2, 9)
RECT_W_RANGE = (2, 9)
DIAMOND_ARM_RANGE = (2, 6)
PENTAGON_ROOF_RANGE = (2, 5)
PENTAGON_WALL_RANGE = (2, 7)
HEXAGON_SLANT_RANGE = (2, 5)
HEXAGON_WALL_RANGE = (2, 6)
SHAPE_WEIGHTS = np.array([0.28, 0.18, 0.18, 0.14, 0.12, 0.10], dtype=float)

CHAIN_RETRIES = 120
SHAPE_PLACEMENT_RETRIES = 60
GENERATE_RETRIES = 500

def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    return (0 if dr == 0 else (1 if dr > 0 else -1), 0 if dc == 0 else (1 if dc > 0 else -1))


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    drs, dcs = _step(a, b)
    steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    return [(a[0] + drs * k, a[1] + dcs * k) for k in range(steps + 1)]


def _transform_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(coords, dtype=int)
    if np.random.rand() < 0.5:
        arr = arr[:, ::-1]
    if np.random.rand() < 0.5:
        arr[:, 0] *= -1
    if np.random.rand() < 0.5:
        arr[:, 1] *= -1
    arr -= arr.min(axis=0)
    return [tuple(map(int, p)) for p in arr]


def _build_line() -> list[tuple[int, int]]:
    n_vertices = int(np.random.randint(*LINE_VERTICES_RANGE))
    remaining = LINE_TOTAL_LENGTH
    pos = [0]
    for i in range(n_vertices - 1):
        steps_left = n_vertices - 2 - i
        gap_hi = remaining - LINE_MIN_GAP * steps_left
        gap = int(np.random.randint(LINE_MIN_GAP, gap_hi + 1))
        pos.append(pos[-1] + gap)
        remaining -= gap
    return _transform_coords([(0, x) for x in pos])


def _build_triangle() -> list[tuple[int, int]]:
    side = int(np.random.randint(*TRIANGLE_SIDE_RANGE))
    return _transform_coords([(0, 0), (0, side), (side, side)])


def _build_rectangle() -> list[tuple[int, int]]:
    h = int(np.random.randint(*RECT_H_RANGE))
    w = int(np.random.randint(*RECT_W_RANGE))
    return _transform_coords([(0, 0), (0, w), (h, w), (h, 0)])


def _build_diamond() -> list[tuple[int, int]]:
    arm = int(np.random.randint(*DIAMOND_ARM_RANGE))
    return _transform_coords([(0, arm), (arm, 2 * arm), (2 * arm, arm), (arm, 0)])


def _build_pentagon() -> list[tuple[int, int]]:
    roof = int(np.random.randint(*PENTAGON_ROOF_RANGE))
    wall = int(np.random.randint(*PENTAGON_WALL_RANGE))
    h = roof + wall
    return _transform_coords([(0, roof), (roof, 0), (h, 0), (h, 2 * roof), (roof, 2 * roof)])


def _build_hexagon() -> list[tuple[int, int]]:
    slant = int(np.random.randint(*HEXAGON_SLANT_RANGE))
    wall = int(np.random.randint(*HEXAGON_WALL_RANGE))
    mid = slant + wall
    bot = 2 * slant + wall
    return _transform_coords([(0, slant), (slant, 0), (mid, 0), (bot, slant), (mid, 2 * slant), (slant, 2 * slant)])


_SHAPE_BUILDERS = [_build_line, _build_triangle, _build_rectangle, _build_diamond, _build_pentagon, _build_hexagon]


def _make_vertices() -> list[tuple[int, int]]:
    weights = SHAPE_WEIGHTS / SHAPE_WEIGHTS.sum()
    idx = int(np.random.choice(len(_SHAPE_BUILDERS), p=weights))
    return _SHAPE_BUILDERS[idx]()


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(vertices) == 2:
        return [(vertices[0], vertices[1])]
    return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _edge_dirs(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [_step(a, b) for a, b in _shape_edges(vertices)]


def _translate_cells(cells: set[tuple[int, int]], offset: tuple[int, int]) -> set[tuple[int, int]]:
    dr, dc = offset
    return {(r + dr, c + dc) for r, c in cells}


def _collinear_with_any(offset: tuple[int, int], directions: list[tuple[int, int]]) -> bool:
    return any(ds_r * offset[1] - ds_c * offset[0] == 0 for ds_r, ds_c in directions)


def _is_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1


def _collinear_bonus_pixels(inp: np.ndarray) -> int:
    base = main(inp)
    filtered = _render_filtered(inp)
    return int(np.count_nonzero(base != filtered))


def _choose_shadow_steps(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
    layer_count: int,
) -> list[tuple[int, int]] | None:
    """Choose 2-3 chain steps that keep anchors unambiguous and permit collinear bonus."""
    anchor = min(vertices)
    directions = _edge_dirs(vertices)

    for _ in range(CHAIN_RETRIES):
        used_dirs: set[tuple[int, int]] = set()
        existing_points = set(vertices)
        occupied = set(shape_cells)
        steps: list[tuple[int, int]] = []
        prev_pos = anchor
        cum = (0, 0)
        saw_collinear_layer = False
        ok = True

        for _layer in range(layer_count):
            candidates: list[tuple[tuple[int, int], tuple[int, int], bool]] = []
            for off in DIR8:
                if off in used_dirs:
                    continue
                new_pos = (prev_pos[0] + off[0], prev_pos[1] + off[1])
                if new_pos in existing_points or new_pos in shape_cells or new_pos in occupied:
                    continue
                if any(_is_adjacent(new_pos, pt) for pt in existing_points if pt != prev_pos):
                    continue

                new_cum = (cum[0] + off[0], cum[1] + off[1])
                shifted = _translate_cells(shape_cells, new_cum)
                visible = shifted - occupied
                if len(visible) < 2:
                    continue

                is_collinear = _collinear_with_any(new_cum, directions)
                candidates.append((off, new_cum, is_collinear))

            if not candidates:
                ok = False
                break

            if not saw_collinear_layer:
                preferred = [cand for cand in candidates if cand[2]]
                if preferred and np.random.rand() < 0.8:
                    off, new_cum, is_collinear = preferred[int(np.random.randint(len(preferred)))]
                else:
                    off, new_cum, is_collinear = candidates[int(np.random.randint(len(candidates)))]
            else:
                off, new_cum, is_collinear = candidates[int(np.random.randint(len(candidates)))]

            new_pos = (prev_pos[0] + off[0], prev_pos[1] + off[1])
            shifted = _translate_cells(shape_cells, new_cum)
            occupied |= shifted
            existing_points.add(new_pos)
            used_dirs.add(off)
            steps.append(off)
            prev_pos = new_pos
            cum = new_cum
            saw_collinear_layer = saw_collinear_layer or is_collinear

        if ok and saw_collinear_layer:
            return steps

    return None


def _make_shape_patch(shape_color: int, shadow_colors: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _make_vertices()
    shape_cells = _shape_cells(vertices)
    steps = _choose_shadow_steps(vertices, shape_cells, len(shadow_colors))
    if steps is None:
        return None

    anchor = min(vertices)
    anchor_positions: list[tuple[int, int]] = []
    all_cells = set(shape_cells)
    cum = (0, 0)
    pos = anchor
    for off in steps:
        cum = (cum[0] + off[0], cum[1] + off[1])
        pos = (pos[0] + off[0], pos[1] + off[1])
        anchor_positions.append(pos)
        all_cells |= _translate_cells(shape_cells, cum)

    r0, c0, r1, c1 = bbox_coords(np.array(sorted(all_cells), dtype=int))
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)

    for r, c in vertices:
        inp[r - r0, c - c0] = int(shape_color)
    for (r, c), col in zip(anchor_positions, shadow_colors):
        rr = r - r0
        cc = c - c0
        if not (0 <= rr < h and 0 <= cc < w) or inp[rr, cc] != BG:
            return None
        inp[rr, cc] = int(col)

    out = main(inp)
    filtered = _render_filtered(inp)

    if len(_extract_shadow_chain(inp, (anchor[0] - r0, anchor[1] - c0), int(shape_color), BG, max_layers=3)) != len(shadow_colors):
        return None
    if len(colors_present(inp, exclude=[BG])) != 1 + len(shadow_colors):
        return None
    for col in shadow_colors:
        if int(np.count_nonzero(out == int(col))) <= 1:
            return None
    if int(np.count_nonzero(out == int(shape_color))) <= len(vertices):
        return None
    if np.array_equal(out, filtered):
        return None
    if _collinear_bonus_pixels(inp) < 2:
        return None

    return inp, out


def _make_distractor_patch(color: int) -> tuple[np.ndarray, np.ndarray]:
    # Keep distractors to a single pixel so they read as obvious noise rather than
    # a second candidate same-color vertex set.
    patch = np.full((1, 1), color, dtype=int)
    return patch.copy(), patch.copy()


def _sample_canvas_size() -> tuple[int, int]:
    H = int(np.random.randint(*CANVAS_H_RANGE))
    W = int(np.clip(H + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return H, W


def _sample_shape_plan() -> tuple[list[int], bool]:
    weights = SHAPE_PLAN_WEIGHTS / SHAPE_PLAN_WEIGHTS.sum()
    idx = int(np.random.choice(len(SHAPE_PLANS), p=weights))
    shape_count, layer_counts = SHAPE_PLANS[idx]
    has_distractor = shape_count == 1 and bool(np.random.rand() < DISTRACTOR_PROB)
    return list(layer_counts), has_distractor


def _sample_palette(layer_counts: list[int], has_distractor: bool) -> tuple[list[tuple[int, list[int]]], int | None] | None:
    total = len(layer_counts) + sum(layer_counts) + int(has_distractor)
    if total > len(COLORS):
        return None
    palette = list(np.random.permutation(COLORS)[:total])
    ptr = 0
    plans: list[tuple[int, list[int]]] = []
    for count in layer_counts:
        base = int(palette[ptr])
        ptr += 1
        shadows = [int(v) for v in palette[ptr:ptr + count]]
        ptr += count
        plans.append((base, shadows))
    distractor = int(palette[ptr]) if has_distractor else None
    return plans, distractor


def _try_place(inp: np.ndarray, out: np.ndarray, patch_in: np.ndarray, patch_out: np.ndarray) -> bool:
    try:
        loc = random_free_location(
            out,
            patch_out,
            transparent=BG,
            padding=PLACEMENT_PADDING,
            padding_connectivity=PLACEMENT_PAD_CONNECTIVITY,
        )
    except ValueError:
        return False
    paste(out, patch_out, loc[0], loc[1], transparent=BG)
    paste(inp, patch_in, loc[0], loc[1], transparent=BG)
    return True


def _gen_attempt() -> np.ndarray | None:
    H, W = _sample_canvas_size()
    out = np.full((H, W), BG, dtype=int)
    inp = np.full((H, W), BG, dtype=int)

    layer_counts, has_distractor = _sample_shape_plan()
    palette = _sample_palette(layer_counts, has_distractor)
    if palette is None:
        return None
    shape_plans, distractor_color = palette
    total_colors = len(layer_counts) + sum(layer_counts) + int(has_distractor)

    for shape_color, shadow_colors in shape_plans:
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), shadow_colors)
            if patch is None:
                continue
            if _try_place(inp, out, patch[0], patch[1]):
                placed = True
                break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        if not _try_place(inp, out, *_make_distractor_patch(distractor_color)):
            return None

    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None
    if _collinear_bonus_pixels(inp) < 2:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
