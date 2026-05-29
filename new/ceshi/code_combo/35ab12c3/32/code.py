# Combo variant of 35ab12c3 (index 32)
# combo_spec: D1/1+D9/2
# Name: 形状匹配配对+干扰项删除
#
# Design analysis
# a) Concrete behavior:
#    原题按“同色单像素顶点”成组；这里改成先做两层分类：
#    1) 识别哪些组件是合法 marker，哪些只是要删除的干扰项；
#    2) 对合法 marker 再按局部模式分组，而不是按颜色分组。
#    我把可见模式收紧为两类，正好对应题意里的“十字形 vs 单点”：
#    - plus: 5 像素十字，中心是顶点
#    - dot: 1 像素孤立点，像素本身是顶点
#    同一模式的中心点按极角排序后连线。颜色只决定输出线的颜色，不决定谁和谁配对。
#    具体小例子：若输入里有 3 个红色 plus 顶点、3 个红色 dot 顶点、再加一小块蓝色 L 形噪声，
#    原始 solver 会把全部红色像素混成一个颜色组并保留蓝色噪声；融合后 solver 只会
#    “plus 连 plus、dot 连 dot”，而蓝色噪声会在输出中被擦掉。
# b) Complexity calibration:
#    只保留 plus/dot 两类 marker，避免模式空间太大；同时强制两类 shape 共用同一种主色，
#    保证“不是按颜色连”的现象在少量样例里稳定可见。干扰项限制为 2-3 格连通非 marker 小块，
#    避免与 dot marker 或 shadow 锚点混淆。
# c) Hidden assumptions review:
#    - 原 `colors = ...` 默认“颜色就是组 id”，融合后失效，必须先做组件识别。
#    - 原 `_clear_in_original(... allow={col})` 默认同色像素都属于同一 shape；融合后若 plus 和
#      dot 共色，这会错误穿过另一种 marker，因此改成 `allow_coords`。
#    - 原 `out = g.copy()` 默认所有非背景像素都保留；D9/2 失效，必须改为从空白画布重建。
#    - dot marker 与单像素 shadow 锚点外观冲突，所以 solver 新增假设：
#      dot 必须是 8 邻域内没有别的非背景像素的“孤立单点”；generator 相应地不给 dot 组挂 shadow，
#      并且不生成 1 像素 distractor。
# d) Placement legality:
#    plus footprint 彼此不能重叠或 4 连通；dot 顶点彼此不能 8 邻接，否则就不再是“孤立单点”。
#    所有 patch 用输出 footprint 做 >=1 格 padding 放置；distractor 因为输出为空，需要对
#    `input OR output` 的占用并集做放置，避免挡住有效线段。
# e) Visual observability:
#    训练样例中 plus 组与 dot 组始终复用同色，因此解题者能直接看到：
#    “同色对象不会全部连在一起，只有同局部形状才会互连”。同时小色块在输出里消失，
#    两条规则会一起出现而不是分别隐身。
# f) Visual cleanliness:
#    marker 只用最规整的十字和单点；干扰项只用小 domino/L 形连通块；不同 patch 保持间距，
#    避免同一区域叠太多层，保证 plus、dot、shadow、distractor 的角色能肉眼拆开。
# g) Color inferability:
#    不引入颜色映射。主色仍然就是主线颜色；shadow 颜色仍由锚点像素给出；变化来自
#    “按模式分组”和“删去非 marker 噪声”，不是来自隐藏的颜色重编码。
# h) Variant metric used after testing:
#    统计生成样本中：
#    - `main(inp)` 与原始按颜色 solver 的输出差异率；
#    - 输入里被擦除的像素数（应主要来自 distractor）；
#    这两个指标同时高，才说明融合行为真的被触发。
# i) New solver assumptions audited:
#    - 只识别 plus 与孤立 dot 两种 marker；
#    - 每种模式至多一组，且该组内部主色一致；
#    - shadow 只挂在 plus 组上，锚点是对角邻居；
#    - distractor 是 2-3 格连通非 marker 小块。
#    生成器逐项保证这些前提成立。

from __future__ import annotations

import math
from collections import deque

import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg, get_bg as _bg_color, paste, random_free_location

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

PATTERN_MASKS = {
    "plus": {(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)},
    "dot": {(0, 0)},
}
PATTERN_IDS = ["plus", "dot"]


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


def _clear_in_original(
    g: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    bg: int,
    allow_coords: set[tuple[int, int]],
) -> bool:
    first = True
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if first:
            first = False
            continue
        if rr == r1 and cc == c1:
            break
        if (rr, cc) in allow_coords:
            continue
        if int(g[rr, cc]) != bg:
            return False
    return True


def _clear_in_original_colors(
    g: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    bg: int,
    allow_colors: set[int],
) -> bool:
    first = True
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if first:
            first = False
            continue
        if rr == r1 and cc == c1:
            break
        v = int(g[rr, cc])
        if v != bg and v not in allow_colors:
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


def _same_color_components(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            col = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells: set[tuple[int, int]] = set()
            while q:
                rr, cc = q.popleft()
                cells.add((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == col:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": col, "cells": cells})
    return comps


def _recognize_marker(g: np.ndarray, cells: set[tuple[int, int]], bg: int) -> tuple[str, tuple[int, int]] | None:
    if len(cells) == 1:
        r, c = next(iter(cells))
        for dr, dc in DIR8:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and int(g[nr, nc]) != bg:
                return None
        return "dot", (r, c)

    if len(cells) != 5:
        return None
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    if sum(rs) % 5 != 0 or sum(cs) % 5 != 0:
        return None
    center = (sum(rs) // 5, sum(cs) // 5)
    if center not in cells:
        return None
    rel = {(r - center[0], c - center[1]) for r, c in cells}
    if rel == PATTERN_MASKS["plus"]:
        return "plus", center
    return None


def _detect_markers(g: np.ndarray, bg: int) -> list[dict]:
    markers: list[dict] = []
    for comp in _same_color_components(g, bg):
        found = _recognize_marker(g, comp["cells"], bg)
        if found is None:
            continue
        pid, center = found
        markers.append(
            {
                "pattern": pid,
                "center": center,
                "color": int(comp["color"]),
                "cells": set(comp["cells"]),
            }
        )
    return markers


def _extra_neighbors(
    g: np.ndarray,
    center: tuple[int, int],
    marker_cells: set[tuple[int, int]],
    bg: int,
) -> list[tuple[int, int, int]]:
    h, w = g.shape
    r0, c0 = center
    carried: list[tuple[int, int, int]] = []
    for dr, dc in DIR8:
        rr = r0 + dr
        cc = c0 + dc
        if not (0 <= rr < h and 0 <= cc < w):
            continue
        if (rr, cc) in marker_cells:
            continue
        v = int(g[rr, cc])
        if v != bg:
            carried.append((dr, dc, v))
    return carried


def _pattern_program(
    g: np.ndarray,
    markers: list[dict],
    pattern_id: str,
    bg: int,
    h: int,
    w: int,
) -> dict | None:
    group = [m for m in markers if m["pattern"] == pattern_id]
    if len(group) < 2:
        return None

    group_colors = {int(m["color"]) for m in group}
    if len(group_colors) != 1:
        return None
    col = next(iter(group_colors))

    cycle = _angle_cycle([m["center"] for m in group])
    center_to_cells = {m["center"]: set(m["cells"]) for m in group}
    allow_coords = set().union(*(m["cells"] for m in group))

    seed_idxs = []
    for i, center in enumerate(cycle):
        if _extra_neighbors(g, center, center_to_cells[center], bg):
            seed_idxs.append(i)
    if seed_idxs:
        start = min(seed_idxs, key=lambda i: (cycle[i][0], cycle[i][1]))
    else:
        start = min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))
    cycle = cycle[start:] + cycle[:start]

    carried = _extra_neighbors(g, cycle[0], center_to_cells[cycle[0]], bg)
    if len(cycle) == 2:
        edges = [(cycle[0], cycle[1])]
    else:
        edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

    drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (r0, c0), (r1, c1) in edges:
        if not _aligned_8(r0, c0, r1, c1):
            continue
        if not _clear_in_original(g, r0, c0, r1, c1, bg, allow_coords):
            continue
        drawable_edges.append(((r0, c0), (r1, c1)))

    if not drawable_edges:
        return None

    return {
        "color": int(col),
        "marker_cells": allow_coords,
        "edges": drawable_edges,
        "carried": carried,
    }


def _legacy_color_main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        cycle = _angle_cycle(pts_rc)
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

        carried: list[tuple[int, int, int]] = []
        sr0, sc0 = cycle[0]
        for drn, dcn in DIR8:
            rr = sr0 + drn
            cc = sc0 + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    carried.append((drn, dcn, v))

        if len(cycle) == 2:
            edges = [(cycle[0], cycle[1])]
        else:
            edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

        drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for (r0, c0), (r1, c1) in edges:
            if not _aligned_8(r0, c0, r1, c1):
                continue
            if not _clear_in_original_colors(g, r0, c0, r1, c1, bg, allow_colors={col}):
                continue
            drawable_edges.append(((r0, c0), (r1, c1)))

        for (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_bg_only(out, r0, c0, r1, c1, int(col), bg)

        for (r0, c0), (r1, c1) in drawable_edges:
            drs, dcs = _step_dir(r0, c0, r1, c1)
            for drn, dcn, ncol in carried:
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
                if not _clear_in_original_colors(g, rr0, cc0, rr1, cc1, bg, allow_colors=set()):
                    continue
                _draw_line_bg_only(out, rr0, cc0, rr1, cc1, ncol, bg)

    return out


# Overview:
# Rebuild the output from recognized marker groups only, so same-pattern markers
# connect together while non-marker clutter disappears.
# Input structure:
# The input contains exactly two real shape groups that reuse one main color:
# a plus-marker group and an isolated-dot group. Only the plus group may have a
# one-pixel shadow anchor of another color. A small extra connected blob is a distractor.
# Rule:
# Detect same-color connected components, keep only components that are either a 5-cell
# plus or an isolated 1-cell dot, group them by local pattern, then connect each pattern's
# centers by H/V/45-degree edges in angle order. Any extra neighbor next to the chosen
# start center seeds the usual parallel shadow copy. Everything else is erased.
# Key insight:
# Color no longer defines the pairing class; local geometry does. The output must therefore
# be reconstructed from valid marker groups instead of copied from the input.
def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    out = np.full_like(g, bg)

    markers = _detect_markers(g, bg)
    for pattern_id in PATTERN_IDS:
        prog = _pattern_program(g, markers, pattern_id, bg, h, w)
        if prog is None:
            continue

        col = int(prog["color"])
        for r, c in prog["marker_cells"]:
            out[r, c] = col

        for (r0, c0), (r1, c1) in prog["edges"]:
            _draw_line_bg_only(out, r0, c0, r1, c1, col, bg)

        for (r0, c0), (r1, c1) in prog["edges"]:
            drs, dcs = _step_dir(r0, c0, r1, c1)
            for drn, dcn, ncol in prog["carried"]:
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
                if not _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow_coords=set()):
                    continue
                _draw_line_bg_only(out, rr0, cc0, rr1, cc1, int(ncol), bg)

    return out


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_PATTERNS = ["plus", "dot"]
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8
DISTRACTOR_PROB = 1.0
PLUS_SHADOW_PROB = 0.75

LINE_VERTICES_RANGE = (2, 6)
LINE_TOTAL_LENGTH = 16
LINE_MIN_GAP = 4
TRIANGLE_SIDE_RANGE = (4, 8)
RECT_H_RANGE = (4, 8)
RECT_W_RANGE = (4, 8)
DIAMOND_ARM_RANGE = (3, 6)
PENTAGON_ROOF_RANGE = (3, 5)
PENTAGON_WALL_RANGE = (3, 6)
HEXAGON_SLANT_RANGE = (2, 4)
HEXAGON_WALL_RANGE = (3, 5)

SHAPE_WEIGHTS = np.array([0.30, 0.20, 0.18, 0.14, 0.10, 0.08], dtype=float)

DISTRACTOR_SHAPES = [
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1), (1, 0)],
    [(0, 0), (1, 0), (1, 1)],
]

SHAPE_PLACEMENT_RETRIES = 80
GENERATE_RETRIES = 500


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


def _shape_line_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_segment_cells(a[0], a[1], b[0], b[1]))
    return cells


def _pattern_cells(center: tuple[int, int], pattern_id: str) -> set[tuple[int, int]]:
    return {(center[0] + dr, center[1] + dc) for dr, dc in PATTERN_MASKS[pattern_id]}


def _footprints_well_formed(vertices: list[tuple[int, int]], pattern_id: str) -> bool:
    if pattern_id == "dot":
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if max(abs(vertices[i][0] - vertices[j][0]), abs(vertices[i][1] - vertices[j][1])) <= 1:
                    return False
        return True

    footprints = [_pattern_cells(v, pattern_id) for v in vertices]
    for i in range(len(footprints)):
        for j in range(i + 1, len(footprints)):
            if footprints[i] & footprints[j]:
                return False
            for r0, c0 in footprints[i]:
                for r1, c1 in footprints[j]:
                    if abs(r0 - r1) + abs(c0 - c1) == 1:
                        return False
    return True


def _choose_shadow(
    vertices: list[tuple[int, int]],
    marker_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    anchor = min(vertices)
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIAG_DIRS:
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in marker_cells:
            continue
        if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        candidates.append((dr, dc))
    if not candidates:
        return None
    return anchor, candidates[int(np.random.randint(len(candidates)))]


def _shape_bbox(
    vertices: list[tuple[int, int]],
    pattern_id: str,
    shadow: tuple[tuple[int, int], tuple[int, int]] | None,
) -> tuple[int, int, int, int]:
    marker_cells = set().union(*[_pattern_cells(v, pattern_id) for v in vertices])
    line_cells = _shape_line_cells(vertices)
    coords = list(marker_cells | line_cells)
    if shadow is not None:
        (ar, ac), (dr, dc) = shadow
        coords.append((ar + dr, ac + dc))
        coords.extend((r + dr, c + dc) for r, c in line_cells)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _make_shape_patch(shape_color: int, pattern_id: str, shadow_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _make_vertices()
    if not _footprints_well_formed(vertices, pattern_id):
        return None

    marker_cells = set().union(*[_pattern_cells(v, pattern_id) for v in vertices])
    shadow = None
    if pattern_id == "plus" and shadow_color is not None:
        shadow = _choose_shadow(vertices, marker_cells)
        if shadow is None:
            return None
    elif pattern_id == "dot" and shadow_color is not None:
        return None

    r0, c0, r1, c1 = _shape_bbox(vertices, pattern_id, shadow)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    for vr, vc in shifted_vertices:
        for rr, cc in _pattern_cells((vr, vc), pattern_id):
            inp[rr, cc] = int(shape_color)

    if shadow is not None:
        (ar, ac), offset = shadow
        sa = (ar - r0 + offset[0], ac - c0 + offset[1])
        if inp[sa] != BG:
            return None
        inp[sa] = int(shadow_color)

    out = main(inp)
    if np.array_equal(out, inp):
        return None
    if int(np.count_nonzero(out != inp)) < 4:
        return None
    return inp, out


def _make_distractor_patch(color: int) -> tuple[np.ndarray, np.ndarray]:
    coords = _transform_coords(DISTRACTOR_SHAPES[int(np.random.randint(len(DISTRACTOR_SHAPES)))])
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)
    for r, c in coords:
        inp[r - r0, c - c0] = int(color)
    out = np.full((h, w), BG, dtype=int)
    return inp, out


def _sample_palette(has_distractor: bool) -> tuple[int, int | None, int | None]:
    pool = list(np.random.permutation(COLORS))
    shape_color = int(pool[0])
    idx = 1
    shadow_color = None
    if np.random.rand() < PLUS_SHADOW_PROB:
        shadow_color = int(pool[idx])
        idx += 1
    distractor_color = int(pool[idx]) if has_distractor else None
    return shape_color, shadow_color, distractor_color


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


def _try_place_distractor(inp: np.ndarray, out: np.ndarray, patch_in: np.ndarray, patch_out: np.ndarray) -> bool:
    occupied = np.where((inp != BG) | (out != BG), 1, BG)
    try:
        loc = random_free_location(
            occupied,
            patch_in,
            transparent=BG,
            padding=PLACEMENT_PADDING,
            padding_connectivity=PLACEMENT_PAD_CONNECTIVITY,
        )
    except ValueError:
        return False
    paste(inp, patch_in, loc[0], loc[1], transparent=BG)
    paste(out, patch_out, loc[0], loc[1], transparent=BG)
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
    shape_color, shadow_color, distractor_color = _sample_palette(has_distractor)
    patterns = list(SHAPE_PATTERNS)
    np.random.shuffle(patterns)

    shadow_map = {"plus": shadow_color, "dot": None}

    for pattern_id in patterns:
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(shape_color, pattern_id, shadow_map[pattern_id])
            if patch is None:
                continue
            if _try_place(inp, out, *patch):
                placed = True
                break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        if not _try_place_distractor(inp, out, *_make_distractor_patch(distractor_color)):
            return None

    expected_colors = 1 + int(shadow_color is not None) + int(has_distractor)
    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != expected_colors:
        return None
    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_legacy_color_main(inp), out):
        return None
    if int(np.count_nonzero((inp != BG) & (out == BG))) < 2:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
