# Combo variant of 13e47133 (index 2)
# combo_spec: D1/0+D5/0
# Name: 螺旋填充+标记锚点保留
#
# Design analysis
# ---------------
# a) 具体化与行为追踪:
#    - 保留原题的墙分区结构：整行/整列墙把画布切成 3..6 个矩形区域，每区大部分是同一种底色。
#    - D1 让 marker 不再编码“边界深度 -> 调色板位置”，而是编码“某个角出发、某个旋向的矩形螺旋入口”。
#      输入里 active 区域的 marker 放在该螺旋前缀的连续几个格子上；前两个 marker 强制同色，形成可见的入口种子。
#    - D5 再要求这些 marker 在输出中原样保留，因此输出不是纯覆盖式螺旋条纹，而是“螺旋条纹 + 若干未被覆盖的锚点断口”。
#    - 小例子：若某区的螺旋前缀颜色是 [A, A, B]，起点位于 path index 0，则解出的循环序列是 [A, B]。
#      原题会按同心框重涂，既看不到路径顺序，也会覆盖 marker；纯 D1 会输出 ...A,B,A,B... 的整条螺旋，
#      第二个 A 会被改写成 B；本融合版在同一路径上跳过 marker 坐标，所以该格仍保持 A，形成锚点保留。
# b) 复杂度校准:
#    - 螺旋自由度只保留 8 种候选（4 角 x 2 旋向），并在每个样例中重复出现，便于从少量样例归纳。
#    - cue 长度限制在 2..4 个格子，既足够区分方向，又不会让输入太花。
#    - 至少两个区域是 active，且允许 inactive 区域保留纯底色，帮助解题者隔离“有 cue 才触发”的规律。
# c) 隐式假设审查:
#    - 原 main() 依赖 _boundary_distance 和 “marker 深度排序”；D1 破坏了这点，所以融合版必须完全改用螺旋路径索引。
#    - 原 main() 默认每个格子最终都会被重涂；D5 破坏了这点，因此螺旋填色循环必须显式跳过 marker 坐标。
#    - 原 main() 只要求 marker 稀疏且每深度至多一个；新逻辑则额外要求 marker 是某条候选螺旋的短连续前缀。
#    - 区域检测仍可沿用非墙 4 连通分量，但螺旋生成要求区域是实心矩形；生成器继续用递归整行/整列切分来保证。
# d) 放置合法性:
#    - 墙由整行/整列切割得到，不重叠且不会产生洞，故每个自由区域都是实心矩形。
#    - marker 只落在其所属矩形螺旋前缀内，天然不越界、不碰墙；写入前仍检查目标格必须还是底色。
#    - 生成器在每个 active 区域上回跑解码器，确认起点和旋向唯一可解。
# e) 视觉可观察性:
#    - 方向信息完全来自几何位置和连续前缀顺序，不依赖颜色编号、排序技巧或不可见元数据。
#    - 保留的 marker 会在螺旋条纹上形成明显断口；若去掉 D5，这些位置会被条纹覆盖，因此融合差异肉眼可见。
#    - 生成器拒绝输出与“原题同心框版”或“纯螺旋覆盖版”一致的样本，避免退化。
# f) 可视化整洁度:
#    - 每区最多 3 种非底色 cue 颜色，marker 总数很少，且都集中在某个角附近，画面仍以大块区域和清晰路径为主。
#    - 墙、底色、cue、输出螺旋各自承担不同角色，没有多层复杂叠压。
# g) 变化的可推断性:
#    - 颜色变化不是任意映射，而是“输入前缀颜色继续沿螺旋重复”；marker 原色保留本身就是结构提示。
#    - 若 cue 从 path index 1 开始，则 base 会补到序列开头，这个偏移由入口前空出的一个底色格可见地提示。
# 9) 新增假设审查:
#    - solver 假设 active 区域的 marker 恰好来自某条候选矩形螺旋的连续短前缀，且前两个 marker 同色。
#    - solver 还假设应按坐标保护 marker，而不是按颜色保护；生成器通过稀疏单像素 cue 和可见断口检查来保证。

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "solvers"))

from arc_utils import components_mask

DIR4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
CORNER_SPECS = [
    ("tl", True, (0, 0), 1),
    ("tl", False, (0, 0), 2),
    ("tr", True, (0, 1), 2),
    ("tr", False, (0, 1), 3),
    ("br", True, (1, 1), 3),
    ("br", False, (1, 1), 0),
    ("bl", True, (1, 0), 0),
    ("bl", False, (1, 0), 1),
]


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _boundary_distance(component_mask: np.ndarray) -> np.ndarray:
    """Original task helper kept for anti-regression checks against frame fill."""
    h, w = component_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if not component_mask[r, c]:
                continue
            is_boundary = False
            for dr, dc in DIR8:
                rr, cc = r + dr, c + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w or not component_mask[rr, cc]:
                    is_boundary = True
                    break
            if is_boundary:
                dist[r, c] = 0
                q.append((r, c))

    qi = 0
    while qi < len(q):
        r, c = q[qi]
        qi += 1
        d = dist[r, c]
        for dr, dc in DIR8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and component_mask[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = d + 1
                q.append((rr, cc))
    return dist


def _spiral_path_rect(rect, corner: str, clockwise: bool) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    for name, cw, (row_from, col_from), init_dir in CORNER_SPECS:
        if name == corner and cw == clockwise:
            sr = 0 if row_from == 0 else h - 1
            sc = 0 if col_from == 0 else w - 1
            break
    else:
        raise ValueError("unknown spiral spec")

    turn = 1 if clockwise else -1
    visited = np.zeros((h, w), dtype=bool)
    path: list[tuple[int, int]] = []
    r, c, d = sr, sc, init_dir

    for _ in range(h * w):
        visited[r, c] = True
        path.append((r0 + r, c0 + c))
        nr, nc = r + DIR4[d][0], c + DIR4[d][1]
        if not (0 <= nr < h and 0 <= nc < w and not visited[nr, nc]):
            d = (d + turn) % 4
            nr, nc = r + DIR4[d][0], c + DIR4[d][1]
        r, c = nr, nc
    return path


def _infer_region_pattern(g: np.ndarray, pts: list[tuple[int, int]], base: int):
    rows = [int(r) for r, _ in pts]
    cols = [int(c) for _, c in pts]
    rect = (min(rows), min(cols), max(rows), max(cols))
    area = (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1)
    if area != len(pts):
        raise ValueError("combo variant expects rectangular regions")

    marker_cells = {(int(r), int(c)): int(g[r, c]) for r, c in pts if int(g[r, c]) != base}
    if not marker_cells:
        return rect, None, None, [int(base)], {}

    candidates = []
    for corner, clockwise, *_ in CORNER_SPECS:
        path = _spiral_path_rect(rect, corner, clockwise)
        idx = {cell: i for i, cell in enumerate(path)}
        try:
            positions = sorted(idx[cell] for cell in marker_cells)
        except KeyError:
            continue
        if positions[0] not in (0, 1):
            continue
        if positions != list(range(positions[0], positions[0] + len(positions))):
            continue

        colors = [int(g[path[i]]) for i in positions]
        if len(colors) < 2:
            continue
        if colors[0] != colors[1]:
            continue
        tail = colors[2:]
        if len(set(tail)) != len(tail):
            continue
        if colors[0] in tail:
            continue

        pattern = ([int(base)] if positions[0] == 1 else []) + [colors[0]] + tail
        if len(pattern) < 2:
            continue

        candidates.append(
            (
                len(colors),
                -positions[0],
                corner,
                clockwise,
                path,
                pattern,
                idx,
            )
        )

    if not candidates:
        raise ValueError("could not infer spiral seed")

    candidates.sort(reverse=True)
    best = candidates[0]
    if len(candidates) > 1 and candidates[1][:2] == best[:2]:
        raise ValueError("ambiguous spiral seed")

    _, _, corner, clockwise, path, pattern, idx = best
    return rect, (corner, clockwise), path, pattern, idx


def _pure_spiral_output(grid: np.ndarray) -> np.ndarray:
    """D1-only behavior: fill the spiral and overwrite cue cells."""
    g = np.asarray(grid, dtype=int)
    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    free = g != wall
    for pts in _cc4(free):
        vals = np.array([g[r, c] for r, c in pts], dtype=int)
        base = _mode(vals)
        _rect, _spec, path, pattern, _idx = _infer_region_pattern(g, pts, base)
        if path is None:
            for r, c in pts:
                out[r, c] = base
            continue
        m = len(pattern)
        for i, (r, c) in enumerate(path):
            out[r, c] = pattern[i % m]
    return out


def _frame_fill_output(grid: np.ndarray) -> np.ndarray:
    """Original task behavior for anti-regression checks."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    free = g != wall
    for pts in _cc4(free):
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[int(r), int(c)] = True

        base = _mode(g[comp_mask])
        dist = _boundary_distance(comp_mask)
        dist_to_color: dict[int, int] = {}
        for r, c in pts:
            r = int(r)
            c = int(c)
            v = int(g[r, c])
            if v == base:
                continue
            d = int(dist[r, c])
            if d >= 0 and d not in dist_to_color:
                dist_to_color[d] = v

        if dist_to_color:
            ds = sorted(dist_to_color)
            palette = ([] if ds[0] == 0 else [int(base)]) + [int(dist_to_color[d]) for d in ds]
        else:
            palette = [int(base)]

        for r, c in pts:
            out[int(r), int(c)] = palette[int(dist[int(r), int(c)]) % len(palette)]
    return out


def _count_anchor_breaks(grid: np.ndarray) -> int:
    """How many cue cells would be overwritten by pure spiral fill but are preserved here."""
    g = np.asarray(grid, dtype=int)
    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)
    total = 0

    for pts in _cc4(g != wall):
        vals = np.array([g[r, c] for r, c in pts], dtype=int)
        base = _mode(vals)
        _rect, _spec, path, pattern, idx = _infer_region_pattern(g, pts, base)
        if path is None:
            continue
        m = len(pattern)
        for r, c in pts:
            r = int(r)
            c = int(c)
            v = int(g[r, c])
            if v == base:
                continue
            if pattern[idx[(r, c)] % m] != v:
                total += 1
    return total


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend wall-separated region cues into spiral stripe fills while preserving the cue pixels as anchors.

    Input structure:
    - The wall color partitions the grid into rectangular non-wall regions.
    - Each region has one dominant base color, and some regions contain a short run of non-base marker pixels on the entrance of a corner-started spiral.
    - In an active region, the first two markers share a color, which identifies the spiral entrance and helps disambiguate direction.

    Rule:
    - For each region, test the eight rectangular spirals (4 corners x 2 turn directions) and find the one whose marked cells form a short consecutive prefix starting at path index 0 or 1.
    - Read those marker colors in spiral order to recover the repeating sequence, prepending the base color when the cue starts at index 1.
    - Repaint the region along that spiral sequence, but skip every original marker coordinate so those source pixels remain unchanged in the output.

    Key insight: The cue now serves two roles at once: it is both the ordered prefix that reveals which spiral to continue and a set of protected anchor cells where the spiral pattern must not overwrite the input color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    free = g != wall
    for pts in _cc4(free):
        vals = np.array([g[r, c] for r, c in pts], dtype=int)
        base = _mode(vals)
        _rect, _spec, path, pattern, _idx = _infer_region_pattern(g, pts, base)
        if path is None:
            for r, c in pts:
                out[int(r), int(c)] = base
            continue

        marker_cells = {(int(r), int(c)) for r, c in pts if int(g[r, c]) != base}
        m = len(pattern)
        for i, (r, c) in enumerate(path):
            if (r, c) in marker_cells:
                continue
            out[r, c] = pattern[i % m]

    return out


H_MIN, H_MAX = 10, 30
W_MIN, W_MAX = 13, 30

N_SPLITS_MIN, N_SPLITS_MAX = 2, 5
MIN_REGION_DIM = 3
SPLITTABLE_LEN = 2 * MIN_REGION_DIM + 1

N_REGIONS_MIN, N_REGIONS_MAX = 3, 6

ACTIVE_REGION_MIN = 2
MARKER_STYLE_WEIGHTS = [
    ("inactive", 2),
    ("offset1_len1", 2),
    ("offset1_len2", 2),
    ("offset1_len3", 1),
    ("offset0_len2", 3),
    ("offset0_len3", 2),
]

DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9
ANCHOR_BREAKS_MIN = 2
GEN_MAX_ATTEMPTS = 400


def _place_walls(H: int, W: int, rng: random.Random):
    wall_mask = np.zeros((H, W), dtype=int)
    rects = [(0, 0, H - 1, W - 1)]
    n_splits = rng.randint(N_SPLITS_MIN, N_SPLITS_MAX)

    for _ in range(n_splits):
        candidates = []
        for i, (r0, c0, r1, c1) in enumerate(rects):
            can_h = (r1 - r0 + 1) >= SPLITTABLE_LEN
            can_v = (c1 - c0 + 1) >= SPLITTABLE_LEN
            if can_h or can_v:
                candidates.append((i, can_h, can_v))
        if not candidates:
            break

        i, can_h, can_v = rng.choice(candidates)
        r0, c0, r1, c1 = rects.pop(i)
        axes = ("h",) * can_h + ("v",) * can_v
        axis = rng.choice(axes)

        if axis == "h":
            rr = rng.randint(r0 + MIN_REGION_DIM, r1 - MIN_REGION_DIM)
            wall_mask[rr, c0 : c1 + 1] = 1
            rects.append((r0, c0, rr - 1, c1))
            rects.append((rr + 1, c0, r1, c1))
        else:
            cc = rng.randint(c0 + MIN_REGION_DIM, c1 - MIN_REGION_DIM)
            wall_mask[r0 : r1 + 1, cc] = 1
            rects.append((r0, c0, r1, cc - 1))
            rects.append((r0, cc + 1, r1, c1))

    return wall_mask, rects


def _pick_palette(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    return bg, wall, colors


def _cue_spec(rng: random.Random):
    labels, weights = zip(*MARKER_STYLE_WEIGHTS)
    choice = rng.choices(labels, weights=weights)[0]
    if choice == "inactive":
        return None
    if choice.startswith("offset1"):
        return 1, int(choice[-1])
    return 0, int(choice[-1])


def _stamp_region_cues(grid, rect, bg: int, marker_pool: list[int], rng: random.Random):
    spec = _cue_spec(rng)
    if spec is None:
        return {
            "active": False,
            "used_colors": set(),
            "corner": None,
            "clockwise": None,
            "start_idx": None,
            "pattern_len": 1,
        }

    start_idx, nonbase_len = spec
    corner, clockwise, *_ = rng.choice(CORNER_SPECS)
    path = _spiral_path_rect(rect, corner, clockwise)

    cue_colors = rng.sample(marker_pool, nonbase_len)
    cue_cells = [cue_colors[0], cue_colors[0], *cue_colors[1:]]
    cells = path[start_idx : start_idx + len(cue_cells)]
    if len(cells) != len(cue_cells):
        return None
    for (r, c), color in zip(cells, cue_cells):
        if grid[r, c] != bg:
            return None
        grid[r, c] = color

    return {
        "active": True,
        "used_colors": set(cue_colors),
        "corner": corner,
        "clockwise": clockwise,
        "start_idx": start_idx,
        "pattern_len": nonbase_len + start_idx,
    }


def _stamp_all_cues(grid, rects, bg: int, marker_pool: list[int], rng: random.Random):
    infos = []
    used_colors = set()
    for rect in rects:
        info = _stamp_region_cues(grid, rect, bg, marker_pool, rng)
        if info is None:
            return None, None
        infos.append(info)
        used_colors |= info["used_colors"]
    return infos, used_colors


def _validate_color_order(grid, bg: int, wall: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _candidate_signature(g: np.ndarray, rect, base: int):
    r0, c0, r1, c1 = rect
    pts = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    _rect, spec, _path, pattern, _idx = _infer_region_pattern(g, pts, base)
    return spec, pattern


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)

    bg, wall, marker_pool = _pick_palette(rng)
    wall_mask, rects = _place_walls(H, W, rng)
    if not (N_REGIONS_MIN <= len(rects) <= N_REGIONS_MAX):
        return None

    grid = np.full((H, W), bg, dtype=int)
    grid[wall_mask == 1] = wall

    infos, used_colors = _stamp_all_cues(grid, rects, bg, marker_pool, rng)
    if infos is None:
        return None

    if sum(info["active"] for info in infos) < ACTIVE_REGION_MIN:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used_colors) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    for rect, info in zip(rects, infos):
        if not info["active"]:
            continue
        try:
            spec, pattern = _candidate_signature(grid, rect, bg)
        except Exception:
            return None
        if spec != (info["corner"], info["clockwise"]):
            return None
        if len(pattern) != info["pattern_len"]:
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    if _count_anchor_breaks(grid) < ANCHOR_BREAKS_MIN:
        return None
    if np.array_equal(out, _pure_spiral_output(grid)):
        return None
    if np.array_equal(out, _frame_fill_output(grid)):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
