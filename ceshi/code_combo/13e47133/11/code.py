# Combo variant of 13e47133 (index 11)
# combo_spec: D1/0+D4/0+D5/0
# Name: 螺旋填充+深度限制+标记保留
#
# Design analysis
# ---------------
# a) 具体化与行为追踪:
#    - 原题: 每个非墙区域读取 marker 的边界深度顺序得到 palette，再按
#      palette[boundary_depth % m] 把整个区域刷成同心框。
#    - 融合后: 仍然保留“墙分割矩形区域”的骨架，但 marker 不再表示深度层本身，
#      而是放在某个角出发的螺旋入口前缀上，用来编码起点与转向。solver 先恢复
#      这条螺旋，再按螺旋路径顺序循环颜色。
#    - D4 融入后，区域并不会整块刷满。若该区有 N 个 marker，则只保留边界深度
#      <= N 的螺旋层；更深处回到底色，形成留白内核。
#    - D5 融入后，marker 坐标最后不再被螺旋颜色覆盖，而是保持输入原色，于是
#      外层短螺旋上会出现几个被保留的“锚点”断口。
#    - 小例子: 7x9 矩形区里，top-left 顺时针螺旋前缀在 path[1:3] 上放两个同色
#      A，则 pattern=[base,A]，marker 数 N=2。原题会得到整区同心框；融合后只给
#      深度 0,1,2 的外层三圈按螺旋次序着色，path[1] 仍是 A，path[2] 若本该变成
#      base 也会被保留成 A，中心更深处保持底色。
#
# b) 复杂度校准:
#    - 为了让 D1 可观察，不能采用“每整圈一个纯色”的字面螺旋，否则在矩形区上会
#      与同心框过于接近；这里沿用单变体里已验证的做法，直接用螺旋路径序号作为
#      循环索引。
#    - 自由度只保留 4 个角 x 2 个方向，以及少量 cue 样式(2 或 3 个 marker)。
#      N 也只落在 2/3 两档，足够让人类从少量样例中归纳。
#
# c) 隐式假设审查:
#    - 原 main() 依赖“marker 按边界深度排序即可解码 palette”，在 D1 下失效，
#      因此必须改为“先解码螺旋候选，再读 cue 顺序”。
#    - 原 main() 默认无条件覆盖所有区域格子，在 D4+D5 下同时失效: 深层格子应回
#      到 base，marker 格子应跳过覆盖。
#    - 新 solver 额外假设区域是实心矩形，且 marker 落在某一条合法螺旋的起始短
#      前缀上；generator 会严格按该假设构造并做唯一性验证。
#
# d) 放置合法性:
#    - 墙仍由整行/整列切分组成，因此自由区域仍是不重叠矩形。
#    - 只有当矩形最大内缩深度大于 marker 数时，才允许把它设为 active 区，这样
#      D4 的留白一定存在。
#    - marker 只写在所选螺旋前缀的背景格上，不会越界或撞墙。
#
# e) 视觉可观察性:
#    - 输入里可见的是“角落短 cue”；输出里可见的是对应方向的外层短螺旋、被保留
#      的 marker 断点，以及未填充的内部留白。三种效应都直接来自空间结构，而非
#      颜色编号。
#
# f) 可视化整洁度:
#    - 每个区域仍只有底色、少量单像素 marker 和单像素宽螺旋纹三层信息，没有额
#      外叠加对象；内核留白让输出更容易与原题区分。
#
# g) 变化的可推断性:
#    - 颜色变化只由“cue 在螺旋上的顺序”决定，深度限制只由 marker 数决定，marker
#      保留只由 marker 坐标决定。三者都有清晰可见的支撑线索。
#
# 9) 新增假设审查:
#    - solver 假设 active 区至少有 2 个 marker，且前两个 marker 同色，以唯一锚定
#      螺旋入口；generator 只生成这类 cue 并验证唯一解码。
#    - solver 假设所有 marker 都落在被填充的外层 N 圈之内；generator 只把 marker
#      放在螺旋入口前缀，因此这一点天然满足。

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


def _ring_depth(rect, r: int, c: int) -> int:
    r0, c0, r1, c1 = rect
    return min(r - r0, c - c0, r1 - r, c1 - c)


def _palette_from_depths(base: int, dist_to_color: dict[int, int]) -> list[int]:
    if not dist_to_color:
        return [int(base)]
    ds = sorted(dist_to_color)
    palette = [int(base)] if ds[0] > 0 else []
    palette.extend(int(dist_to_color[d]) for d in ds)
    return palette


def _infer_region_pattern(g: np.ndarray, pts: list[tuple[int, int]], base: int):
    rows = [int(r) for r, _ in pts]
    cols = [int(c) for _, c in pts]
    rect = (min(rows), min(cols), max(rows), max(cols))
    area = (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1)
    if area != len(pts):
        raise ValueError("combo solver expects rectangular regions")

    marker_cells = {(int(r), int(c)): int(g[r, c]) for r, c in pts if int(g[r, c]) != base}
    if not marker_cells:
        return rect, None, None, [int(base)], marker_cells

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
        if len(colors) < 2 or colors[0] != colors[1]:
            continue

        tail = colors[2:]
        if len(set(tail)) != len(tail) or colors[0] in tail:
            continue

        pattern = ([int(base)] if positions[0] == 1 else []) + [int(colors[0])] + [int(v) for v in tail]
        if len(pattern) < 2:
            continue

        candidates.append((len(colors), -positions[0], corner, clockwise, path, pattern))

    if not candidates:
        raise ValueError("could not infer spiral cue")

    candidates.sort(reverse=True)
    best = candidates[0]
    if len(candidates) > 1 and candidates[1][:2] == best[:2]:
        raise ValueError("ambiguous spiral cue")

    _, _, corner, clockwise, path, pattern = best
    return rect, (corner, clockwise), path, pattern, marker_cells


def _frame_transform(grid: np.ndarray, *, capped: bool, keep_markers: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    for pts in _cc4(g != wall):
        comp_mask = np.zeros((h, w), dtype=bool)
        marker_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[int(r), int(c)] = True

        base = _mode(g[comp_mask])
        dist = _boundary_distance(comp_mask)
        dist_to_color: dict[int, int] = {}
        marker_count = 0
        for r, c in pts:
            r = int(r)
            c = int(c)
            v = int(g[r, c])
            if v == base:
                continue
            marker_count += 1
            marker_mask[r, c] = True
            d = int(dist[r, c])
            if d >= 0 and d not in dist_to_color:
                dist_to_color[d] = v

        palette = _palette_from_depths(base, dist_to_color)
        m = len(palette)
        for r, c in pts:
            r = int(r)
            c = int(c)
            if keep_markers and marker_mask[r, c]:
                continue
            d = int(dist[r, c])
            if capped and d > marker_count:
                out[r, c] = base
            else:
                out[r, c] = palette[d % m]

    return out


def _spiral_transform(grid: np.ndarray, *, capped: bool, keep_markers: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    for pts in _cc4(g != wall):
        vals = np.array([g[int(r), int(c)] for r, c in pts], dtype=int)
        base = _mode(vals)
        rect, _spec, path, pattern, marker_cells = _infer_region_pattern(g, pts, base)
        if path is None:
            for r, c in pts:
                out[int(r), int(c)] = base
            continue

        marker_set = set(marker_cells)
        fill_depth = len(marker_cells)
        m = len(pattern)
        for i, (r, c) in enumerate(path):
            if keep_markers and (r, c) in marker_set:
                out[r, c] = int(g[r, c])
                continue
            if capped and _ring_depth(rect, r, c) > fill_depth:
                out[r, c] = base
            else:
                out[r, c] = pattern[i % m]

    return out


def _variant_deltas(grid: np.ndarray) -> dict[str, int]:
    combo = _spiral_transform(grid, capped=True, keep_markers=True)
    frame_combo = _frame_transform(grid, capped=True, keep_markers=True)
    uncapped_spiral = _spiral_transform(grid, capped=False, keep_markers=True)
    overwrite_spiral = _spiral_transform(grid, capped=True, keep_markers=False)
    return {
        "d1_path_delta": int(np.count_nonzero(combo != frame_combo)),
        "d4_cap_delta": int(np.count_nonzero(combo != uncapped_spiral)),
        "d5_keep_delta": int(np.count_nonzero(combo != overwrite_spiral)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend short spiral cues inside wall-split rectangles, but stop after a marker-count depth cap and preserve the cue pixels.

    Input structure:
    - One wall color partitions the canvas into rectangular non-wall regions.
    - Each active region has one dominant base color plus 2 or 3 cue markers sitting on the first cells of a corner-started rectangular spiral.
    - The first two cue markers share a color, which exposes the spiral entrance; inactive regions have no markers.

    Rule:
    - For each active region, test the eight rectangular spirals (4 corners x 2 turn directions) and keep the unique one whose first marked cells match the cue.
    - Read the cue colors in spiral order to build the repeating sequence; if the cue starts at path index 1, prepend the base color.
    - Let N be the number of marker pixels in that region. Color only the cells whose inward ring depth is at most N, using sequence[path_index % sequence_length] along the spiral path.
    - Leave all marker pixels unchanged, and reset any deeper cells to the base color.

    Key insight:
    - The markers no longer encode boundary-depth palette slots directly. They jointly specify three things at once: which spiral to follow, how the spiral colors repeat, and how many outer spiral layers survive before the untouched base-colored core begins.
    """
    return _spiral_transform(grid, capped=True, keep_markers=True)


H_MIN, H_MAX = 10, 30
W_MIN, W_MAX = 13, 30

N_SPLITS_MIN, N_SPLITS_MAX = 2, 5
MIN_REGION_DIM = 3
SPLITTABLE_LEN = 2 * MIN_REGION_DIM + 1

N_REGIONS_MIN, N_REGIONS_MAX = 3, 6
ACTIVE_REGION_MIN = 2

DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 8
GEN_MAX_ATTEMPTS = 500

MARKER_STYLE_WEIGHTS = [
    ("inactive", 5),
    ("offset1_len1", 3),  # 2 markers -> fill depth 2
    ("offset1_len2", 2),  # 3 markers -> fill depth 3
    ("offset0_len2", 3),  # 3 markers -> fill depth 3
]


def _max_cheb_depth(h: int, w: int) -> int:
    return (min(h, w) - 1) // 2


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


def _cue_spec_from_label(label: str):
    if label == "inactive":
        return None
    if label == "offset1_len1":
        return 1, 1
    if label == "offset1_len2":
        return 1, 2
    if label == "offset0_len2":
        return 0, 2
    raise ValueError(f"unknown cue label: {label}")


def _marker_count_for_spec(spec) -> int:
    if spec is None:
        return 0
    _start_idx, nonbase_len = spec
    return nonbase_len + 1


def _allowed_labels_for_rect(rect) -> list[str]:
    r0, c0, r1, c1 = rect
    max_d = _max_cheb_depth(r1 - r0 + 1, c1 - c0 + 1)
    labels = ["inactive"]
    if max_d > 2:
        labels.append("offset1_len1")
    if max_d > 3:
        labels.extend(["offset1_len2", "offset0_len2"])
    return labels


def _sample_label(allowed: list[str], rng: random.Random) -> str:
    labels = [label for label, _weight in MARKER_STYLE_WEIGHTS if label in allowed]
    weights = [weight for label, weight in MARKER_STYLE_WEIGHTS if label in allowed]
    return rng.choices(labels, weights=weights)[0]


def _stamp_region_cues(grid, rect, bg: int, marker_pool: list[int], used_colors: set[int], rng: random.Random, label: str):
    spec = _cue_spec_from_label(label)
    if spec is None:
        return {
            "active": False,
            "used_colors": set(),
            "corner": None,
            "clockwise": None,
            "start_idx": None,
            "pattern_len": 1,
            "marker_count": 0,
        }

    start_idx, nonbase_len = spec
    corner, clockwise, *_ = rng.choice(CORNER_SPECS)
    path = _spiral_path_rect(rect, corner, clockwise)

    available = [c for c in marker_pool if c not in used_colors]
    source = available if len(available) >= nonbase_len else marker_pool
    cue_colors = rng.sample(source, nonbase_len)
    cue_cells = [cue_colors[0], cue_colors[0], *cue_colors[1:]]
    cells = path[start_idx : start_idx + len(cue_cells)]
    if len(cells) != len(cue_cells):
        return None

    for (r, c), color in zip(cells, cue_cells):
        if grid[r, c] != bg:
            return None
        grid[r, c] = int(color)

    return {
        "active": True,
        "used_colors": set(int(v) for v in cue_colors),
        "corner": corner,
        "clockwise": clockwise,
        "start_idx": start_idx,
        "pattern_len": nonbase_len + start_idx,
        "marker_count": len(cue_cells),
    }


def _assign_region_labels(rects, rng: random.Random):
    allowed = [_allowed_labels_for_rect(rect) for rect in rects]
    count2 = [i for i, labels in enumerate(allowed) if "offset1_len1" in labels]
    count3 = [i for i, labels in enumerate(allowed) if "offset1_len2" in labels or "offset0_len2" in labels]

    if len(count2) < 2 or not count3:
        return None

    labels = ["inactive"] * len(rects)

    idx3 = rng.choice(count3)
    labels[idx3] = rng.choice([label for label in allowed[idx3] if label in ("offset1_len2", "offset0_len2")])

    remaining2 = [i for i in count2 if i != idx3]
    idx2 = rng.choice(remaining2)
    labels[idx2] = "offset1_len1"

    extra_candidates = [i for i, opts in enumerate(allowed) if i not in (idx2, idx3) and len(opts) > 1]
    rng.shuffle(extra_candidates)
    for i in extra_candidates:
        if rng.random() < 0.35:
            labels[i] = _sample_label([label for label in allowed[i] if label != "inactive"], rng)

    return labels


def _stamp_all_cues(grid, rects, bg: int, marker_pool: list[int], rng: random.Random):
    labels = _assign_region_labels(rects, rng)
    if labels is None:
        return None, None

    infos = []
    used_colors: set[int] = set()
    for rect, label in zip(rects, labels):
        info = _stamp_region_cues(grid, rect, bg, marker_pool, used_colors, rng, label)
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
    _rect, spec, _path, pattern, marker_cells = _infer_region_pattern(g, pts, base)
    return spec, pattern, len(marker_cells)


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

    active_infos = [info for info in infos if info["active"]]
    if len(active_infos) < ACTIVE_REGION_MIN:
        return None
    marker_counts = sorted(info["marker_count"] for info in active_infos)
    if 2 not in marker_counts or 3 not in marker_counts:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used_colors) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    for rect, info in zip(rects, infos):
        if not info["active"]:
            continue
        try:
            spec, pattern, marker_count = _candidate_signature(grid, rect, bg)
        except Exception:
            return None
        if spec != (info["corner"], info["clockwise"]):
            return None
        if len(pattern) != info["pattern_len"]:
            return None
        if marker_count != info["marker_count"]:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, out):
        return None

    deltas = _variant_deltas(grid)
    if deltas["d1_path_delta"] < 6:
        return None
    if deltas["d4_cap_delta"] < 4:
        return None
    if deltas["d5_keep_delta"] < 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
