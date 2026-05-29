# Variant of 35ab12c3
# Dimension: D13 — 多步处理顺序 (multi_step)
# Change: 三步流程：Step1识别同色像素对 -> Step2画连线 -> Step3画平行阴影线。主线先画、阴影后画（避免阴影阻... → 链式反应——连线触发新连线
#
# Design analysis
# ---------------
# variant_spec:
#   unchanged_core:
#     - 背景色仍为 0，颜色仍取自 {1,2,3,4,6,7,8}
#     - 输入仍由若干同色顶点 + 1 个邻接异色像素（阴影/种子标记）构成
#     - 输出仍画 H/V/45° 线段，且所有阴影都在主线完成后再画，避免覆盖主线
#   d13_change:
#     - 原题对同色点集是“一次性全局连边”；这里改为“局部激活的开放折线”
#     - 每个 shape 恰好有一个被异色邻居标记的端点，称为 seed endpoint
#     - solver 不再先求整条形状，而是从 seed endpoint 出发，只画它唯一可见的下一段
#     - 当该段到达下一个同色顶点时，这个顶点才被激活，再继续画下一段，直到链结束
#   concrete_object:
#     - 一个有效 shape 是 3-5 个同色顶点组成的开放折线
#     - 只有相邻顶点彼此 H/V/45° 对齐；非相邻顶点故意避免共线，防止“提前连到更远顶点”
#     - seed endpoint 旁边放 1 个异色像素；这个像素既是“从哪端开始”的提示，也决定阴影偏移/颜色
#   small_trace:
#     - 输入若有红点 A-B-C-D 四个顶点，A 邻接一个蓝色 seed 像素
#     - 原题式逻辑会把同色点全局排序后一次性尝试连边，顺序与 A->B->C->D 无关
#     - 新逻辑先只允许 A->B；线到达 B 后，B 才能触发 B->C；再到 C 后触发 C->D
#     - 因此 D13 的差异体现在“哪些段存在”依赖前一段是否先完成，而不是一次性已知
#   complexity_guard:
#     - 不做跨颜色级联，避免把原题的“同色连线 + 阴影”改成完全不同的网络题
#     - 但每个 shape 至少 2 段，且每个样例至少一个长 shape(>=3 段)，保证链式传播肉眼可见
#   implicit_assumptions_checked:
#     - 原 solver 依赖“同色点集可整体排序成 cycle”；变体里这不成立，必须改为基于 seed 的 path tracing
#     - 原 solver 默认 start vertex 的邻居只影响阴影，不影响主线选择；变体里它还决定起点
#     - 原 solver 对每个颜色立即画主线后再画该颜色阴影；变体里若仍逐色画阴影，会阻挡后续颜色主线，
#       所以改成“所有 shape 的主线先收集并统一绘制，再统一绘制阴影”
#   placement_legality:
#     - generator 只生成开放折线，禁止自交、禁止非相邻顶点 H/V/diag 共线
#     - seed 阴影像素要求只 8 邻接一个顶点，且偏移不与任一线段共线
#     - patch 放置仍按完整 output patch 做无重叠采样，因此主线/阴影都不会与别组冲突
#   observability:
#     - seed 总在一个端点旁，且输出明显从该端按折线路径展开，和原题的“整体包边/整体连边”可区分
#     - 每个样例有 2-3 个 shape，重复支撑同一规则，不依赖颜色 id 或隐藏排序
#
import numpy as np

from arc_utils import bbox_coords, colors_present, draw_line, get_bg, neighbors8, paste, random_free_location

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


def _build_visibility_graph(g: np.ndarray, pts_rc: list[tuple[int, int]], col: int, bg: int) -> list[set[int]]:
    graph = [set() for _ in pts_rc]
    for i in range(len(pts_rc)):
        r0, c0 = pts_rc[i]
        for j in range(i + 1, len(pts_rc)):
            r1, c1 = pts_rc[j]
            if not _aligned_8(r0, c0, r1, c1):
                continue
            if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
                continue
            graph[i].add(j)
            graph[j].add(i)
    return graph


def _seed_candidates(g: np.ndarray, pts_rc: list[tuple[int, int]], col: int, bg: int) -> list[tuple[int, list[tuple[int, int, int]]]]:
    seeds: list[tuple[int, list[tuple[int, int, int]]]] = []
    h, w = g.shape
    for i, (r, c) in enumerate(pts_rc):
        carried: list[tuple[int, int, int]] = []
        for drn, dcn in DIR8:
            rr = r + drn
            cc = c + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    carried.append((drn, dcn, v))
        if carried:
            seeds.append((i, carried))
    return seeds


def _trace_chain(
    pts_rc: list[tuple[int, int]],
    graph: list[set[int]],
    seed_idx: int,
) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], set[int]]:
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    visited = {int(seed_idx)}
    used_edges: set[tuple[int, int]] = set()
    prev = -1
    curr = int(seed_idx)

    while True:
        candidates = [nxt for nxt in graph[curr] if nxt != prev and (min(curr, nxt), max(curr, nxt)) not in used_edges]
        fresh = [nxt for nxt in candidates if nxt not in visited]
        if len(fresh) != 1:
            break
        nxt = int(fresh[0])
        edges.append((pts_rc[curr], pts_rc[nxt]))
        used_edges.add((min(curr, nxt), max(curr, nxt)))
        prev, curr = curr, nxt
        visited.add(curr)

    return edges, visited


def _analyze_shape(g: np.ndarray, col: int, bg: int) -> dict | None:
    pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
    if len(pts_rc) < 2:
        return None

    graph = _build_visibility_graph(g, pts_rc, col, bg)
    degrees = [len(nbs) for nbs in graph]
    if any(d == 0 or d > 2 for d in degrees):
        return None

    endpoints = [i for i, d in enumerate(degrees) if d == 1]
    if len(pts_rc) > 2 and len(endpoints) != 2:
        return None

    seeds = _seed_candidates(g, pts_rc, col, bg)
    if len(seeds) != 1:
        return None
    seed_idx, carried = seeds[0]
    if degrees[seed_idx] != 1:
        return None

    edges, visited = _trace_chain(pts_rc, graph, seed_idx)
    if len(edges) != len(pts_rc) - 1:
        return None
    if len(visited) != len(pts_rc):
        return None

    return {
        "color": int(col),
        "carried": carried,
        "edges": edges,
    }


def _collect_shape_infos(g: np.ndarray, bg: int) -> list[dict]:
    infos: list[dict] = []
    for col in colors_present(g, exclude={bg}):
        info = _analyze_shape(g, int(col), int(bg))
        if info is not None:
            infos.append(info)
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    infos = _collect_shape_infos(g, bg)

    # Phase 1: trace and draw all activated main-chain segments for every shape.
    for info in infos:
        for (r0, c0), (r1, c1) in info["edges"]:
            _draw_line_bg_only(out, r0, c0, r1, c1, int(info["color"]), bg)

    # Phase 2: only after all main lines exist, draw the carried parallel shadows.
    for info in infos:
        for (r0, c0), (r1, c1) in info["edges"]:
            drs, dcs = _step_dir(r0, c0, r1, c1)
            for drn, dcn, ncol in info["carried"]:
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
                _draw_line_bg_only(out, rr0, cc0, rr1, cc1, int(ncol), bg)

    return out


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 4)         # 2-3 shapes, all chain-reactive
DISTRACTOR_PROB = 0.35
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

CHAIN_SEGMENT_COUNT_RANGE = (2, 5)  # 2-4 segments => 3-5 vertices
CHAIN_SEG_LEN_RANGE = (2, 6)        # 2-5 cells
CHAIN_BUILD_RETRIES = 80
LONG_CHAIN_MIN_SEGMENTS = 3

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

SHAPE_PLACEMENT_RETRIES = 60
GENERATE_RETRIES = 400


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return _step_dir(a[0], a[1], b[0], b[1])


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    return list(_segment_cells(a[0], a[1], b[0], b[1]))


def _chain_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [(vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)]


def _chain_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _chain_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _normalize_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(coords, dtype=int)
    arr -= arr.min(axis=0)
    return [tuple(map(int, p)) for p in arr]


def _valid_chain_vertices(vertices: list[tuple[int, int]]) -> bool:
    if len(vertices) < 3:
        return False
    if len(set(vertices)) != len(vertices):
        return False

    seen: set[tuple[int, int]] = set()
    for i, (a, b) in enumerate(_chain_edges(vertices)):
        cells = _gen_segment_cells(a, b)
        for cell in cells:
            if cell in seen and cell != a:
                return False
        seen.update(cells)
        if i > 0 and _step(vertices[i - 1], vertices[i]) == _step(vertices[i], vertices[i + 1]):
            return False

    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if j == i + 1:
                continue
            if _aligned_8(vertices[i][0], vertices[i][1], vertices[j][0], vertices[j][1]):
                return False
    return True


def _build_chain_vertices(min_segments: int = 2) -> list[tuple[int, int]] | None:
    max_segments = CHAIN_SEGMENT_COUNT_RANGE[1] - 1
    min_segments = max(min_segments, CHAIN_SEGMENT_COUNT_RANGE[0])
    if min_segments > max_segments:
        return None

    for _ in range(CHAIN_BUILD_RETRIES):
        seg_count = int(np.random.randint(min_segments, max_segments + 1))
        vertices = [(0, 0)]
        dirs: list[tuple[int, int]] = []
        used_cells = {(0, 0)}
        ok = True

        for _seg_idx in range(seg_count):
            last = vertices[-1]
            dir_candidates = DIR8.copy()
            np.random.shuffle(dir_candidates)
            chosen = None
            for drs, dcs in dir_candidates:
                if dirs and ((drs, dcs) == dirs[-1] or (drs, dcs) == (-dirs[-1][0], -dirs[-1][1])):
                    continue
                lengths = list(range(CHAIN_SEG_LEN_RANGE[0], CHAIN_SEG_LEN_RANGE[1]))
                np.random.shuffle(lengths)
                for seg_len in lengths:
                    nxt = (last[0] + drs * seg_len, last[1] + dcs * seg_len)
                    if nxt in vertices:
                        continue
                    cells = _gen_segment_cells(last, nxt)
                    if any(cell in used_cells and cell != last for cell in cells):
                        continue
                    vertices.append(nxt)
                    dirs.append((drs, dcs))
                    used_cells.update(cells[1:])
                    chosen = nxt
                    break
                if chosen is not None:
                    break
            if chosen is None:
                ok = False
                break

        if not ok:
            continue
        vertices = _normalize_coords(vertices)
        if _valid_chain_vertices(vertices):
            return vertices

    return None


def _choose_shadow(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    anchor = vertices[0]
    directions = [_step(a, b) for a, b in _chain_edges(vertices)]
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIR8:
        if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
            continue
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in shape_cells:
            continue
        if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        candidates.append((dr, dc))
    if not candidates:
        return None
    return anchor, candidates[int(np.random.randint(len(candidates)))]


def _patch_bbox(
    shape_cells: set[tuple[int, int]],
    vertices: list[tuple[int, int]],
    shadow: tuple[tuple[int, int], tuple[int, int]] | None,
) -> tuple[int, int, int, int]:
    coords = list(shape_cells) + vertices
    if shadow is not None:
        (ar, ac), (dr, dc) = shadow
        coords.append((ar + dr, ac + dc))
        coords.extend((r + dr, c + dc) for r, c in shape_cells)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _paint_shadow(out: np.ndarray, shape_color: int, shadow_color: int, offset: tuple[int, int]) -> None:
    dr, dc = offset
    h, w = out.shape
    src = out == shape_color
    shifted = np.zeros_like(src)
    r_src = slice(max(0, -dr), h - max(0, dr))
    c_src = slice(max(0, -dc), w - max(0, dc))
    r_dst = slice(max(0, dr), h - max(0, -dr))
    c_dst = slice(max(0, dc), w - max(0, -dc))
    shifted[r_dst, c_dst] = src[r_src, c_src]
    mask = shifted & (out == BG)
    out[mask] = shadow_color


def _make_shape_patch(shape_color: int, shadow_color: int, min_segments: int) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _build_chain_vertices(min_segments=min_segments)
    if vertices is None:
        return None
    cells = _chain_cells(vertices)

    shadow = _choose_shadow(vertices, cells)
    if shadow is None:
        return None

    r0, c0, r1, c1 = _patch_bbox(cells, vertices, shadow)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    for a, b in _chain_edges(shifted_vertices):
        draw_line(out, a[0], a[1], end_r=b[0], end_c=b[1], color=shape_color)
    for r, c in shifted_vertices:
        inp[r, c] = shape_color

    (ar, ac), offset = shadow
    anchor = (ar - r0, ac - c0)
    seed_pos = (anchor[0] + offset[0], anchor[1] + offset[1])
    if not (0 <= seed_pos[0] < h and 0 <= seed_pos[1] < w):
        return None
    if inp[seed_pos] != BG:
        return None
    inp[seed_pos] = int(shadow_color)

    _paint_shadow(out, shape_color, int(shadow_color), offset)

    nb_hits = sum(
        1
        for rr, cc in neighbors8(anchor[0], anchor[1])
        if 0 <= rr < h and 0 <= cc < w and int(inp[rr, cc]) == int(shadow_color)
    )
    if nb_hits != 1:
        return None

    if not np.array_equal(main(inp), out):
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
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if _aligned(coords[j][0] - coords[i][0], coords[j][1] - coords[i][1]):
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


def _sample_canvas_size() -> tuple[int, int]:
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


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
    h, w = _sample_canvas_size()
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)

    total_needed = shape_count * 2 + int(has_distractor)
    if total_needed > len(COLORS):
        return None
    palette = list(np.random.permutation(COLORS)[:total_needed])
    shape_colors = [int(v) for v in palette[:shape_count]]
    shadow_colors = [int(v) for v in palette[shape_count : 2 * shape_count]]
    distractor_color = int(palette[-1]) if has_distractor else None

    long_shape_idx = int(np.random.randint(shape_count))
    for idx, (shape_color, shadow_color) in enumerate(zip(shape_colors, shadow_colors)):
        placed = False
        min_segments = LONG_CHAIN_MIN_SEGMENTS if idx == long_shape_idx else CHAIN_SEGMENT_COUNT_RANGE[0]
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(shape_color, shadow_color, min_segments=min_segments)
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
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
