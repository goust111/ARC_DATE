# Combo variant of 8e5c0c38 (index 10)
# combo_spec: D2/0+D5/0
# Name: 标记水平轴修复
#
# Design analysis for this combo variant
# -------------------------------------
# a) 具体化与行为追踪
# - 原题对每个颜色层搜索竖直对称轴；这里改成每个精灵围绕某条水平轴修复，且这条轴不再靠
#   搜索推断，而是由外部红色标记直接指出。
# - 具体像素形态：每个 payload 精灵是近水平对称的彩色连通块；红色标记是一个单像素或短水
#   平线段，放在该精灵左侧或右侧，并且与真正的水平轴处在同一行。输入里还会有同色噪声像
#   素，它们相对标记所指的水平轴没有上下镜像伙伴，因此输出要删除这些像素，并同时擦掉标记。
# - 小例子：若某精灵真正轴行为 r=9，标记也落在第 9 行；精灵在 (7,10) 与 (11,10) 成对，
#   但额外噪声在 (6,9) 没有 (12,9) 伙伴，则新规则删除它。若再加一对围绕错误假轴 r=8 的
#   噪声 (7,8)/(9,8)，盲目搜索水平轴的 solver 可能被误导；本融合版本仍必须服从红标记给
#   出的第 9 行，因此这对噪声也会被删掉。
#
# b) 复杂度校准
# - 保留 22x22 画布、3 个互异颜色精灵、外接噪声和 output-first 生成，避免退化成单对象小题。
# - 为了让“水平轴”与“外部标记”都足够显眼，精灵限制为奇数高度且至少 5 行，这样轴总是落
#   在真实的一整行上，标记也能用单行结构稳定表达。
# - 标记样式只在两三个档位中变化：长度 1/2/3，左右两侧放置。这样既有重复证据，又不会引入
#   太多编码自由度。
#
# c) 隐式假设审查
# - 原 main() 中“所有非背景色都是 payload”不再成立；红色 2 现在是元信息，必须从输出中删除。
# - 原 main() 的“按颜色独立搜索最佳轴”也不再成立；正确分组应是“marker 组件 -> 邻近 sprite
#   组件”的配对，然后对该组件使用 marker 指定的行。
# - 原 `cc = k - c` 与列域枚举都假设左右镜像；融合后必须统一改成 `rr = k - r` 和行域配对。
# - D5 的显式标记在原参考里默认竖直线段并编码列轴；与 D2 融合后若仍保留竖线，就会出现
#   “标记视觉方向”和“真正对称方向”不一致的问题，因此这里同步改成水平线段/单点并读取行号。
# - 新 main() 还隐含：每个 payload 精灵在输入里保持单个 4 连通组件，并且存在唯一近邻 marker。
#   generator 会显式保证这两点。
#
# d) 放置合法性
# - 三个 slot 彼此分离；sprite 放在 slot 内部，marker 与 sprite 作为一个横向 bundle 一起放置，
#   因此不会越界。
# - marker 与 sprite 之间保留 1 格空隙，并把 marker/gap 单元都加入 blocked 集，噪声采样不
#   会占用这些位置，也不会把 marker 与 payload 连成一个组件。
# - 噪声只从 clean sprite 的 4 邻域采样，因此仍附着于本精灵，不会跨 slot 连到别的对象。
#
# e) 视觉可观察性
# - 人类能直接看到三条重复证据：红色短横标记只出现在输入、输出会消失；payload 是上下近对称；
#   被删除的是相对标记所在行缺乏镜像伙伴的像素，而不是相对某一列缺乏镜像伙伴的像素。
# - 标记被放在左右两侧而不是上下两侧，是为了避免与“水平轴修复”混淆成“给出一条横向分割线”；
#   它更像外部标签，而不是对象本体的一部分。
#
# f) 可视化整洁度
# - 标记使用固定红色 2，payload 颜色从其余颜色中抽样，角色区分清楚。
# - 标记外置且与 sprite 保持空隙，因此不会覆盖 payload 像素，也不会形成难辨认的混色组件。
# - 左右两侧 marker 长度很短，且三对象 slot 相隔较远，不会出现多个标记纠缠在同一区域。
#
# g) 变化的可推断性
# - 没有颜色映射；唯一“会消失的颜色”是固定 marker 色。
# - 关键结构线索来自空间关系而不是颜色编号大小：marker 与某个 sprite 同行且最近，因此人类
#   可以从少量样例推断“它指的是这一行的对称轴”。
#
# h) 量化退化检查思路
# - 仅检查 `main(inp) == out` 不够，因为若输入恰好也能被“水平搜索轴”或“竖直搜索轴”解释，
#   组合中的 D5 或 D2 就形同虚设。
# - 生成器因此额外拒绝两类退化样本：
#   1. 去掉 marker、改回水平搜索轴的 payload-only ablation 仍得到同一输出；
#   2. 去掉 marker、改回原题竖直搜索轴的 payload-only ablation 仍得到同一输出。
# - 这迫使样本同时依赖“水平轴方向”与“显式 marker 指轴”两个控制变量。
#
# i) main() 新假设与 generator 对应保证
# - 假设：每个 marker 组件恰好位于单一行。保证：generator 只生成 1x1 或 1xL 的水平标记。
# - 假设：marker 行一定落在对应 sprite bbox 的行跨度内。保证：marker 总与 sprite 中心行对齐。
# - 假设：最近且同行跨度覆盖 marker 的 payload 组件就是正确目标。保证：slot 分离、gap 保留、
#   每个 sprite 只有一个 marker，且别的 sprite 不会横向贴近到造成歧义。

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ARC_UTILS_DIR = ROOT / "solvers"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import get_bg, label_mask, paste


MARK = 2
GRID_SIZE = 22


def _bbox_gap(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> int:
    r0a, c0a, r1a, c1a = map(int, b1)
    r0b, c0b, r1b, c1b = map(int, b2)
    if r1a < r0b:
        dr = int(r0b - r1a - 1)
    elif r1b < r0a:
        dr = int(r0a - r1b - 1)
    else:
        dr = 0
    if c1a < c0b:
        dc = int(c0b - c1a - 1)
    elif c1b < c0a:
        dc = int(c0a - c1b - 1)
    else:
        dc = 0
    return int(dr + dc)


def _best_axis_and_score(coords: list[tuple[int, int]], span: int, *, horizontal: bool) -> tuple[int, int]:
    pts = {(int(r), int(c)) for r, c in coords}
    best_score = -1
    best_k = 0
    for k in range(0, 2 * (span - 1) + 1):
        score = 0
        for r, c in pts:
            if horizontal:
                rr = int(k) - int(r)
                if 0 <= rr < span and (rr, int(c)) in pts:
                    score += 1
            else:
                cc = int(k) - int(c)
                if 0 <= cc < span and (int(r), cc) in pts:
                    score += 1
        if score > best_score:
            best_score = int(score)
            best_k = int(k)
    return int(best_k), int(best_score)


def _search_denoise_grid(grid: np.ndarray, *, horizontal: bool, exclude_mark: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    out = np.full_like(g, int(bg))
    excluded = {bg}
    if exclude_mark:
        excluded.add(int(MARK))

    for color in sorted(set(map(int, g.flatten())) - excluded):
        coords = [(int(r), int(c)) for r, c in np.argwhere(g == int(color))]
        if not coords:
            continue
        best_k, _ = _best_axis_and_score(coords, H if horizontal else W, horizontal=horizontal)
        for r, c in coords:
            if horizontal:
                rr = int(best_k) - int(r)
                keep = 0 <= rr < H and int(g[rr, int(c)]) == int(color)
            else:
                cc = int(best_k) - int(c)
                keep = 0 <= cc < W and int(g[int(r), cc]) == int(color)
            if keep:
                out[int(r), int(c)] = int(color)
    return out


def _horizontal_search_kept_coords(coords: list[tuple[int, int]], height: int) -> set[tuple[int, int]]:
    best_k, _ = _best_axis_and_score(coords, height, horizontal=True)
    pts = {(int(r), int(c)) for r, c in coords}
    kept: set[tuple[int, int]] = set()
    for r, c in pts:
        rr = int(best_k) - int(r)
        if 0 <= rr < height and (rr, int(c)) in pts:
            kept.add((int(r), int(c)))
    return kept


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each sprite by using an explicitly marked horizontal reflection axis.

    Input structure:
    - One background color fills most cells. Red (2) is a special marker color.
    - Three non-red sprites appear as nearly top/bottom-symmetric components, and each one has a nearby red single pixel or short horizontal segment on the same row as its true symmetry axis.
    - Extra same-color pixels may cling to a sprite even though they do not have a reflected partner across the marked row.

    Rule:
    - Remove the red markers from the output.
    - Pair each red marker component with the nearby non-red component whose bounding box spans that marker row.
    - Use the marker row as the sprite's horizontal axis, and delete every sprite pixel whose same-color reflection across that row is absent.

    Key insight:
    - The puzzle is no longer about discovering the axis from the noisy shape; the axis is given externally, so the job is to associate marker to sprite and enforce top/bottom pairing around that marked row.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, _W = g.shape

    out = g.copy()
    out[g == int(MARK)] = int(bg)

    obj_lab, obj_n = label_mask((g != int(bg)) & (g != int(MARK)), connectivity=4)
    marker_lab, marker_n = label_mask(g == int(MARK), connectivity=4)

    obj_comps: list[dict[str, object]] = []
    for lbl in range(1, int(obj_n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(obj_lab == int(lbl))]
        if not coords:
            continue
        colors = {int(g[r, c]) for r, c in coords}
        if len(colors) != 1:
            continue
        rs = [r for r, _c in coords]
        cs = [c for _r, c in coords]
        obj_comps.append(
            {
                "id": int(lbl),
                "coords": coords,
                "color": int(next(iter(colors))),
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )

    marker_comps: list[dict[str, object]] = []
    for lbl in range(1, int(marker_n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(marker_lab == int(lbl))]
        if not coords:
            continue
        rows = {r for r, _c in coords}
        if len(rows) != 1:
            continue
        cols = [c for _r, c in coords]
        axis_r = int(next(iter(rows)))
        marker_comps.append(
            {
                "coords": coords,
                "axis_r": int(axis_r),
                "bbox": (int(axis_r), int(min(cols)), int(axis_r), int(max(cols))),
            }
        )

    used_obj_ids: set[int] = set()
    for marker in sorted(marker_comps, key=lambda rec: (int(rec["bbox"][0]), int(rec["bbox"][1]))):
        axis_r = int(marker["axis_r"])
        candidates = [
            comp
            for comp in obj_comps
            if comp["id"] not in used_obj_ids and int(comp["bbox"][0]) <= axis_r <= int(comp["bbox"][2])
        ]
        if not candidates:
            continue

        comp = min(
            candidates,
            key=lambda rec: (
                _bbox_gap(marker["bbox"], rec["bbox"]),
                abs(axis_r - ((int(rec["bbox"][0]) + int(rec["bbox"][2])) // 2)),
            ),
        )
        used_obj_ids.add(int(comp["id"]))

        k = 2 * int(axis_r)
        color = int(comp["color"])
        for r, c in comp["coords"]:
            rr = int(k - r)
            if not (0 <= rr < H and int(g[rr, int(c)]) == int(color)):
                out[int(r), int(c)] = int(bg)

    return out


# --------------- generator (explicit marked horizontal-axis variant) ---------------

# ── CFG: layout ──
NUM_SPRITES = 3
LAYOUTS = [
    [(0, 0, 8, 9), (0, 12, 8, 21), (13, 3, 21, 18)],
    [(0, 1, 9, 10), (0, 11, 9, 20), (13, 4, 21, 17)],
]

# ── CFG: clean sprite geometry ──
PIXEL_COUNT_MIN = 6
PIXEL_COUNT_MAX = 46
HEIGHT_MIN = 5
HEIGHT_MAX = 9
WIDTH_MIN = 3
WIDTH_MAX = 12
ODD_SIDE_H_RANGE = range(2, 5)  # heights 5, 7, 9
DENSITY_LO = 0.35
DENSITY_HI = 0.89

# ── CFG: noise + anti-degeneracy ──
NOISE_FRAC_LO = 0.11
NOISE_FRAC_HI = 0.33
NOISE_MAX = 12
FALSE_AXIS_OFFSETS = (-4, -2, 2, 4)
NOISE_SELECTION_RETRIES = 160

# ── CFG: marker placement ──
MARKER_GAP = 1
MARKER_LENGTHS = (1, 2, 3)

# ── CFG: retry limits ──
SPRITE_RETRIES = 220
GEN_RETRIES = 500

# ── CFG: augmentation ──
FLIP_PROB = 0.5


def _choose_shape_spec(max_h: int, max_w: int) -> tuple[int, int, int, int]:
    options: list[tuple[int, int, int, int]] = []
    h_max = min(HEIGHT_MAX, max_h)
    w_max = min(WIDTH_MAX, max_w)
    for target_count in range(PIXEL_COUNT_MIN, PIXEL_COUNT_MAX + 1):
        for w in range(WIDTH_MIN, w_max + 1):
            for side_h in ODD_SIDE_H_RANGE:
                h = 2 * side_h + 1
                if h > h_max:
                    continue
                area = h * w
                lo = max(PIXEL_COUNT_MIN, math.ceil(DENSITY_LO * area))
                hi = min(PIXEL_COUNT_MAX, math.floor(DENSITY_HI * area))
                if not (lo <= target_count <= hi):
                    continue
                for center_count in range(1, w + 1):
                    if (target_count - center_count) % 2 != 0:
                        continue
                    top_count = (target_count - center_count) // 2
                    if w <= top_count <= side_h * w:
                        options.append((h, w, target_count, center_count))
    if not options:
        raise ValueError("no feasible clean sprite spec")
    target_count = random.choice(sorted({rec[2] for rec in options}))
    target_options = [rec for rec in options if rec[2] == target_count]
    weights = [1.0 / (rec[0] * rec[1]) for rec in target_options]
    return random.choices(target_options, weights=weights, k=1)[0]


def _top_cells(side_h: int, width: int, top_count: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    deep_anchor_col = random.randrange(width)
    for c in range(width):
        if c == 0:
            r = 0
        elif side_h > 1 and c == deep_anchor_col:
            r = side_h - 1
        else:
            r = random.randrange(side_h)
        cells.add((int(r), int(c)))

    remaining = top_count - len(cells)
    if remaining < 0:
        raise ValueError("top_count too small")
    pool = [(r, c) for r in range(side_h) for c in range(width) if (r, c) not in cells]
    cells.update(random.sample(pool, remaining))
    return cells


def _center_cols(width: int, center_count: int) -> np.ndarray:
    cols = np.zeros(width, dtype=bool)
    cols[random.sample(range(width), center_count)] = True
    return cols


def _crop_to_bbox(sprite: np.ndarray, color: int) -> np.ndarray:
    coords = np.argwhere(sprite == int(color))
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return sprite[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _build_clean_sprite(color: int, max_h: int, max_w: int) -> np.ndarray:
    for _ in range(SPRITE_RETRIES):
        h, w, target_count, center_count = _choose_shape_spec(max_h, max_w)
        side_h = h // 2
        top_count = (target_count - center_count) // 2
        top = _top_cells(side_h, w, top_count)
        centers = _center_cols(w, center_count)

        sprite = np.zeros((h, w), dtype=int)
        for r, c in top:
            mirror_r = h - 1 - r
            sprite[int(r), int(c)] = int(color)
            sprite[int(mirror_r), int(c)] = int(color)
        sprite[side_h, centers] = int(color)
        sprite = _crop_to_bbox(sprite, color)

        sh, sw = sprite.shape
        if sh > max_h or sw > max_w or sh % 2 == 0:
            continue
        count = int(np.sum(sprite == int(color)))
        area = int(sh * sw)
        if not (PIXEL_COUNT_MIN <= count <= PIXEL_COUNT_MAX):
            continue
        density = count / area
        if not (DENSITY_LO <= density <= DENSITY_HI):
            continue
        _lab, n = label_mask(sprite == int(color), connectivity=4)
        if int(n) != 1:
            continue
        coords = [(int(r), int(c)) for r, c in np.argwhere(sprite == int(color))]
        _vk, vscore = _best_axis_and_score(coords, sw, horizontal=False)
        if int(vscore) == len(coords):
            continue
        return sprite
    raise ValueError("failed to build clean sprite")


def _choose_marker_styles() -> tuple[list[int], list[str]]:
    lengths = [1, random.choice([2, 3]), random.choice(MARKER_LENGTHS)]
    random.shuffle(lengths)
    sides = ["left", "right", random.choice(["left", "right"])]
    random.shuffle(sides)
    return lengths, sides


def _slot_place_sprites(
    bg: int, sprite_specs: list[dict[str, object]], slots: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, list[dict[str, object]], np.ndarray] | None:
    out = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    placed: list[dict[str, object]] = []

    for spec, (r0s, c0s, r1s, c1s) in zip(sprite_specs, slots):
        spec = dict(spec)
        sprite = np.array(spec["sprite"], dtype=int)
        h, w = sprite.shape
        marker_len = int(spec["marker_len"])
        side = str(spec["marker_side"])
        slot_h = int(r1s - r0s + 1)
        slot_w = int(c1s - c0s + 1)
        bundle_w = int(w + MARKER_GAP + marker_len)
        if h > slot_h or bundle_w > slot_w:
            return None

        r0 = random.randint(int(r0s), int(r1s - h + 1))
        left = random.randint(int(c0s), int(c1s - bundle_w + 1))
        if side == "left":
            marker_c0 = int(left)
            sprite_c0 = int(left + marker_len + MARKER_GAP)
            gap_cells = [(int(r0 + h // 2), int(sprite_c0 - 1))]
        else:
            sprite_c0 = int(left)
            marker_c0 = int(sprite_c0 + w + MARKER_GAP)
            gap_cells = [(int(r0 + h // 2), int(sprite_c0 + w))]

        axis_r = int(r0 + h // 2)
        marker_cells = [(int(axis_r), int(marker_c0 + dc)) for dc in range(marker_len)]

        paste(out, sprite, int(r0), int(sprite_c0), transparent=0)
        for rr, cc in marker_cells + gap_cells:
            blocked[int(rr), int(cc)] = True

        spec["r0"] = int(r0)
        spec["c0"] = int(sprite_c0)
        spec["axis_r"] = int(axis_r)
        spec["true_k"] = int(2 * axis_r)
        spec["marker_cells"] = marker_cells
        spec["gap_cells"] = gap_cells
        placed.append(spec)

    return out, placed, blocked


def _noise_candidates(clean_mask: np.ndarray, true_k: int, blocked: np.ndarray) -> list[tuple[int, int]]:
    H, _W = clean_mask.shape
    adj = np.zeros_like(clean_mask)
    adj[1:, :] |= clean_mask[:-1, :]
    adj[:-1, :] |= clean_mask[1:, :]
    adj[:, 1:] |= clean_mask[:, :-1]
    adj[:, :-1] |= clean_mask[:, 1:]
    adj &= ~clean_mask
    adj &= ~blocked

    rows = np.arange(H)
    mirror_rows = int(true_k) - rows
    on_axis = mirror_rows == rows
    adj[on_axis, :] = False

    valid_mirror = (mirror_rows >= 0) & (mirror_rows < H) & ~on_axis
    for r in np.where(valid_mirror)[0]:
        adj[int(r), :] &= ~clean_mask[int(mirror_rows[int(r)]), :]

    rows_arr, cols_arr = np.where(adj)
    return list(zip(rows_arr.tolist(), cols_arr.tolist()))


def _false_axis_pairs(
    clean_mask: np.ndarray,
    true_k: int,
    blocked: np.ndarray,
) -> dict[int, list[tuple[tuple[int, int], tuple[int, int]]]]:
    candidates = set(_noise_candidates(clean_mask, true_k, blocked))
    pair_map: dict[int, list[tuple[tuple[int, int], tuple[int, int]]]] = {}
    for delta in FALSE_AXIS_OFFSETS:
        fake_k = int(true_k + delta)
        pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        for r, c in candidates:
            rr = int(fake_k - r)
            mate = (int(rr), int(c))
            if rr == r or mate not in candidates:
                continue
            pair = tuple(sorted(((int(r), int(c)), mate)))
            pairs.add(pair)
        if pairs:
            pair_map[int(fake_k)] = sorted(pairs)
    return pair_map


def _add_noise(inp: np.ndarray, out: np.ndarray, spec: dict[str, object], blocked: np.ndarray) -> bool:
    color = int(spec["color"])
    true_k = int(spec["true_k"])
    clean_mask = out == int(color)
    clean_coords = [(int(r), int(c)) for r, c in np.argwhere(clean_mask)]
    clean_set = set(clean_coords)
    clean_count = len(clean_coords)

    lo = max(2, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if lo > hi:
        return False

    candidates = _noise_candidates(clean_mask, true_k, blocked)
    pair_map = _false_axis_pairs(clean_mask, true_k, blocked)
    if len(candidates) < lo or not pair_map:
        return False

    for _ in range(NOISE_SELECTION_RETRIES):
        noise_target = random.randint(lo, hi)
        usable_axes = [fake_k for fake_k, pairs in pair_map.items() if pairs and noise_target >= 2]
        if not usable_axes:
            return False
        fake_k = random.choice(sorted(usable_axes, key=lambda k: (-len(pair_map[k]), abs(k - true_k))))
        pairs = pair_map[fake_k][:]
        random.shuffle(pairs)

        chosen: list[tuple[int, int]] = []
        forbidden: set[tuple[int, int]] = set()
        max_pairs = min(len(pairs), noise_target // 2)
        if max_pairs <= 0:
            continue
        pair_goal = max_pairs if max_pairs == 1 else random.randint(max(1, max_pairs - 1), max_pairs)

        for a, b in pairs:
            if a in forbidden or b in forbidden:
                continue
            for cell in (a, b):
                chosen.append(cell)
                forbidden.add(cell)
                mirror = (int(true_k - cell[0]), int(cell[1]))
                if 0 <= mirror[0] < GRID_SIZE:
                    forbidden.add(mirror)
            if len(chosen) >= 2 * pair_goal:
                break

        if len(chosen) < 2:
            continue

        singles = candidates[:]
        random.shuffle(singles)
        for cell in singles:
            if len(chosen) >= noise_target:
                break
            if cell in forbidden:
                continue
            chosen.append(cell)
            forbidden.add(cell)
            mirror = (int(true_k - cell[0]), int(cell[1]))
            if 0 <= mirror[0] < GRID_SIZE:
                forbidden.add(mirror)

        if len(chosen) != noise_target:
            continue

        trial_coords = clean_coords + chosen
        if _horizontal_search_kept_coords(trial_coords, GRID_SIZE) == clean_set:
            continue

        for r, c in chosen:
            inp[int(r), int(c)] = int(color)
        spec["noise_cells"] = chosen
        spec["fake_k"] = int(fake_k)
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    palette = [c for c in range(10) if c != MARK]
    bg, *obj_colors = random.sample(palette, NUM_SPRITES + 1)
    slots = random.choice(LAYOUTS)
    marker_lengths, marker_sides = _choose_marker_styles()

    sprite_specs: list[dict[str, object]] = []
    try:
        for color, slot, marker_len, marker_side in zip(obj_colors, slots, marker_lengths, marker_sides):
            slot_h = int(slot[2] - slot[0] + 1)
            slot_w = int(slot[3] - slot[1] + 1)
            max_h = int(slot_h)
            max_w = int(slot_w - MARKER_GAP - marker_len)
            sprite = _build_clean_sprite(int(color), max_h, max_w)
            sprite_specs.append(
                {
                    "color": int(color),
                    "sprite": sprite,
                    "marker_len": int(marker_len),
                    "marker_side": str(marker_side),
                }
            )
    except ValueError:
        return None

    placed = _slot_place_sprites(int(bg), sprite_specs, slots)
    if placed is None:
        return None
    out, placed_specs, blocked = placed

    inp = out.copy()
    for spec in placed_specs:
        if not _add_noise(inp, out, spec, blocked):
            return None
        for r, c in spec["marker_cells"]:
            inp[int(r), int(c)] = int(MARK)

    if random.random() < FLIP_PROB:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_search_denoise_grid(inp, horizontal=True, exclude_mark=True), out):
        return None
    if np.array_equal(_search_denoise_grid(inp, horizontal=False, exclude_mark=True), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
