# Variant of 35ab12c3
# Dimension: D18 — 起始顶点选择策略 (meta)
# Change: 有阴影锚点的顶点优先作为起始点（seed_idxs），若有多个则取坐标最小的；无阴影时取坐标最小的顶点。起始顶点决定了极... → 所有顶点都可以是阴影锚定点
#
# Design analysis
# a) Concrete behavior:
#    原题里一个 shape 只从单一起始顶点“携带”阴影邻居，所以整条轮廓的平行阴影偏移是全局统一的。
#    这个变体改成：极角排序仍然给出一个稳定的顶点环，但阴影不再绑定到全局 start。每条边
#    (v_i -> v_{i+1}) 只继承 v_i 自己在输入中 8 邻域里的异色像素，因此同一 shape 的不同边
#    可以有不同偏移。例子：矩形四角里，左上角邻居在 (-1,0)，右上角邻居在 (0,1)，则上边复制
#    成上移线，右边复制成右移线；原题则会把某一个角的偏移复制到四条边。
# b) Complexity calibration:
#    若仍然只给每个 shape 一个阴影锚点，输入可完全退化成原题。为让 D18 可见，本生成器要求每个
#    有阴影的 shape 至少 2 个锚点，且至少出现 2 种不同偏移。shadow 颜色仍保持“每个 shape 一种”，
#    避免引入额外颜色映射维度。
# c) Hidden assumptions audit for main():
#    - 原代码 `seed_idxs/start/cycle rotation` 假设“一个 start 决定所有阴影”。变体下不成立，只能保留
#      一个与阴影无关的稳定旋转：取坐标最小顶点作 cycle 起点。
#    - 原代码 `carried` 只从 `cycle[0]` 收集邻居。变体下必须改成“按顶点收集”，并把边与其起点索引绑定。
#    - 原代码 phase 2 对所有边复用同一个 `carried`。变体下若继续这样做，会把某个顶点的偏移错误地扩散到
#      其他边，因此必须改成 `carried_by_vertex[idx]`。
#    - `_clear_in_original(... allow=set())` 仍可保留，但生成器必须保证其他锚点不会落在某条阴影边的内部，
#      否则新的多锚点输入会让合法边被误判为阻塞。
# d) Placement legality:
#    patch 的 bbox 现在要覆盖：主 shape 全部边、每个锚点像素、每条边对应的平行阴影边。全局放置仍使用
#    `random_free_location` 基于输出 patch 做不重叠布置，因此不同对象的最终输出不会相撞。
# e) Visual observability:
#    人类可见的关键线索是：同一 shadow 颜色在多个顶点出现，而输出里该颜色不是整圈统一平移，而是只沿局部边
#    复制，并且不同边的偏移方向可不同。生成器强制至少一个 shadowed shape 满足这个条件。
# f) Visual cleanliness:
#    为避免 shadow 色自己被 solver 当成新 shape 连线，同一 shadow 颜色的多个锚点被约束为两两不成
#    水平/垂直/45° 对齐；同时锚点不得落在其他阴影边已占的单元上，减少视觉打架。
# g) New solver assumptions and generator guarantees:
#    - 假设真正的 shadow anchor 8 邻接于唯一一个主顶点；生成器显式检查这一点。
#    - 假设 shadowed shape 至少有一个可作为边起点的锚点；生成器对 shadowed shape 只接受非共线、
#      至少三顶点的形状，并为其选 2-3 个起点锚点。

import math

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, draw_line, get_bg, paste, random_free_location

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


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _step_dir(r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    dr = r1 - r0
    dc = c1 - c0
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    return sr, sc


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return _step_dir(a[0], a[1], b[0], b[1])


def _segment_cells(r0: int, c0: int, r1: int, c1: int):
    sr, sc = _step_dir(r0, c0, r1, c1)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    for k in range(steps + 1):
        yield r0 + sr * k, c0 + sc * k


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    return list(_segment_cells(a[0], a[1], b[0], b[1]))


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


def _rotate_cycle_to_min(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not points_rc:
        return []
    start = min(range(len(points_rc)), key=lambda i: (points_rc[i][0], points_rc[i][1]))
    return points_rc[start:] + points_rc[:start]


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(vertices) == 2:
        return [(vertices[0], vertices[1])]
    return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _all_collinear(points: list[tuple[int, int]]) -> bool:
    if len(points) < 3:
        return True
    r0, c0 = points[0]
    r1, c1 = points[1]
    dr = r1 - r0
    dc = c1 - c0
    for r, c in points[2:]:
        if (r - r0) * dc != (c - c0) * dr:
            return False
    return True


def _pairwise_non_aligned(points_rc: list[tuple[int, int]]) -> bool:
    for i in range(len(points_rc)):
        for j in range(i + 1, len(points_rc)):
            if _aligned(points_rc[j][0] - points_rc[i][0], points_rc[j][1] - points_rc[i][1]):
                return False
    return True


def _vertex_neighbors(g: np.ndarray, r: int, c: int, self_color: int, bg: int) -> list[tuple[int, int, int]]:
    h, w = g.shape
    carried: list[tuple[int, int, int]] = []
    for drn, dcn in DIR8:
        rr = r + drn
        cc = c + dcn
        if 0 <= rr < h and 0 <= cc < w:
            v = int(g[rr, cc])
            if v != bg and v != self_color:
                carried.append((drn, dcn, v))
    return carried


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        cycle = _rotate_cycle_to_min(_angle_cycle(pts_rc))
        if len(cycle) < 2:
            continue

        carried_by_vertex = [_vertex_neighbors(g, r, c, col, bg) for r, c in cycle]
        edges = _shape_edges(cycle)

        drawable_edges: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
        for idx, ((r0, c0), (r1, c1)) in enumerate(edges):
            if not _aligned_8(r0, c0, r1, c1):
                continue
            if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
                continue
            drawable_edges.append((idx, (r0, c0), (r1, c1)))

        for _, (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_bg_only(out, r0, c0, r1, c1, col, bg)

        for idx, (r0, c0), (r1, c1) in drawable_edges:
            drs, dcs = _step_dir(r0, c0, r1, c1)
            for drn, dcn, ncol in carried_by_vertex[idx]:
                if drs * dcn - dcs * drn == 0:
                    continue
                rr0 = r0 + drn
                cc0 = c0 + dcn
                rr1 = r1 + drn
                cc1 = c1 + dcn
                if not (0 <= rr0 < h and 0 <= cc0 < w and 0 <= rr1 < h and 0 <= cc1 < w):
                    continue
                if not _aligned_8(rr0, cc0, rr1, cc1):
                    continue
                if not _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow=set()):
                    continue
                _draw_line_bg_only(out, rr0, cc0, rr1, cc1, ncol, bg)

    return out


BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MIN_SHADOW_SHAPES = 1
MAX_SHADOWS_CAP = 2
DISTRACTOR_PROB = 0.5
PLACEMENT_PADDING = 1
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

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50
SHADOW_ASSIGN_RETRIES = 80
SHAPE_PLACEMENT_RETRIES = 40
GENERATE_RETRIES = 300


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


def _normalized_cycle(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return _rotate_cycle_to_min(_angle_cycle(vertices))


def _shadow_offset_candidates(
    cycle: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
    start_idx: int,
    anchor_positions: list[tuple[int, int]],
    used_shadow_cells: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], list[tuple[int, int]]]]:
    edges = _shape_edges(cycle)
    start, end = edges[start_idx]
    drs, dcs = _step(start, end)
    vertex_set = set(cycle)
    anchor_set = set(anchor_positions)
    candidates: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    for dr, dc in DIR8:
        if drs * dc - dcs * dr == 0:
            continue
        pos = (start[0] + dr, start[1] + dc)
        if pos in shape_cells or pos in vertex_set or pos in anchor_set or pos in used_shadow_cells:
            continue
        if sum(1 for vr, vc in cycle if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        if any(_aligned(pos[0] - pr, pos[1] - pc) for pr, pc in anchor_positions):
            continue
        shifted = [(r + dr, c + dc) for r, c in _gen_segment_cells(start, end)]
        if any(cell in vertex_set for cell in shifted[1:]):
            continue
        if any(cell in anchor_set for cell in shifted[1:-1]):
            continue
        candidates.append(((dr, dc), shifted))
    return candidates


def _choose_shadow_specs(cycle: list[tuple[int, int]], shadow_color: int) -> list[tuple[int, tuple[int, int], int]] | None:
    if len(cycle) < 3 or _all_collinear(cycle):
        return None

    max_anchors = min(3, len(cycle))
    for _ in range(SHADOW_ASSIGN_RETRIES):
        k = int(np.random.randint(2, max_anchors + 1))
        idxs = np.random.choice(len(cycle), size=k, replace=False).tolist()
        np.random.shuffle(idxs)
        specs: list[tuple[int, tuple[int, int], int]] = []
        anchor_positions: list[tuple[int, int]] = []
        used_shadow_cells: set[tuple[int, int]] = set()
        shape_cells = _shape_cells(cycle)
        ok = True
        for idx in idxs:
            candidates = _shadow_offset_candidates(cycle, shape_cells, idx, anchor_positions, used_shadow_cells)
            if not candidates:
                ok = False
                break
            used_offsets = {offset for _, offset, _ in specs}
            fresh = [cand for cand in candidates if cand[0] not in used_offsets]
            pool = fresh or candidates
            offset, shifted = pool[int(np.random.randint(len(pool)))]
            specs.append((idx, offset, shadow_color))
            anchor_positions.append((cycle[idx][0] + offset[0], cycle[idx][1] + offset[1]))
            used_shadow_cells.update(shifted)
        if ok and len({offset for _, offset, _ in specs}) >= 2 and _pairwise_non_aligned(anchor_positions):
            return specs
    return None


def _patch_bbox(vertices: list[tuple[int, int]], shadow_specs: list[tuple[int, tuple[int, int], int]]) -> tuple[int, int, int, int]:
    coords = list(vertices) + list(_shape_cells(vertices))
    edges = _shape_edges(vertices)
    for idx, (dr, dc), _ in shadow_specs:
        start, end = edges[idx]
        coords.extend((r + dr, c + dc) for r, c in _gen_segment_cells(start, end))
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _make_shape_patch(shape_color: int, shadow_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    cycle = _normalized_cycle(_make_vertices())
    if shadow_color is not None and (len(cycle) < 3 or _all_collinear(cycle)):
        return None

    shadow_specs: list[tuple[int, tuple[int, int], int]] = []
    if shadow_color is not None:
        chosen = _choose_shadow_specs(cycle, int(shadow_color))
        if chosen is None:
            return None
        shadow_specs = chosen

    r0, c0, r1, c1 = _patch_bbox(cycle, shadow_specs)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)

    for r, c in cycle:
        inp[r - r0, c - c0] = shape_color

    for idx, (dr, dc), scol in shadow_specs:
        ar, ac = cycle[idx]
        rr = ar + dr - r0
        cc = ac + dc - c0
        if inp[rr, cc] != BG:
            return None
        inp[rr, cc] = int(scol)

    out = main(inp)
    if shadow_color is not None:
        if np.count_nonzero(inp == shadow_color) != len(shadow_specs):
            return None
        if np.count_nonzero(out == shadow_color) <= len(shadow_specs):
            return None
    return inp, out


def _sample_distractor_coords(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    coords: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(coords) < count:
        cand = (int(np.random.randint(h)), int(np.random.randint(w)))
        if cand in seen:
            continue
        coords.append(cand)
        seen.add(cand)
    if not _pairwise_non_aligned(coords):
        return None
    return coords


def _make_distractor_patch(color: int) -> tuple[np.ndarray, np.ndarray]:
    count = int(np.random.randint(*DISTRACTOR_COUNT_RANGE))
    if count == 1:
        patch = np.full((1, 1), color, dtype=int)
        return patch.copy(), patch.copy()
    for _ in range(DISTRACTOR_RETRIES):
        h = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        w = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        coords = _sample_distractor_coords(h, w, count)
        if coords is None:
            continue
        patch = np.full((h, w), BG, dtype=int)
        for r, c in coords:
            patch[r, c] = color
        return patch.copy(), patch.copy()
    patch = np.full((1, 1), color, dtype=int)
    return patch.copy(), patch.copy()


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None] | None:
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(COLORS) - shape_count - int(has_distractor))
    min_shadows = min(MIN_SHADOW_SHAPES, max_shadows)
    if max_shadows < min_shadows:
        return None
    shadow_count = int(np.random.randint(min_shadows, max_shadows + 1))
    total = shape_count + shadow_count + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = palette[:shape_count]
    shadow_colors = palette[shape_count:shape_count + shadow_count]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, shadow_colors, distractor_color


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> dict[int, int]:
    if not shadow_colors:
        return {}
    targets = np.random.choice(shape_count, size=len(shadow_colors), replace=False).tolist()
    return {int(idx): int(col) for idx, col in zip(sorted(targets), shadow_colors)}


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


def _sample_canvas_size() -> tuple[int, int]:
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_canvas_size()
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)
    total_colors = shape_count + len(shadow_colors) + int(has_distractor)

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), shadow_map.get(idx))
            if patch is None:
                continue
            if _try_place(inp, out, *patch):
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
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")


def _analyze_shadow_usage(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    stats: list[dict] = []
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        pts = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts) < 2:
            continue
        cycle = _rotate_cycle_to_min(_angle_cycle(pts))
        edges = _shape_edges(cycle)
        drawable = [
            idx
            for idx, ((r0, c0), (r1, c1)) in enumerate(edges)
            if _aligned_8(r0, c0, r1, c1) and _clear_in_original(g, r0, c0, r1, c1, bg, allow={col})
        ]
        if not drawable:
            continue
        per_vertex = [_vertex_neighbors(g, r, c, col, bg) for r, c in cycle]
        anchor_vertices = [idx for idx, carried in enumerate(per_vertex) if carried]
        if not anchor_vertices:
            continue
        offsets = {(dr, dc) for idx in anchor_vertices for dr, dc, _ in per_vertex[idx]}
        stats.append(
            {
                "color": col,
                "vertex_count": len(cycle),
                "anchor_vertices": len(anchor_vertices),
                "distinct_offsets": len(offsets),
            }
        )
    return stats
