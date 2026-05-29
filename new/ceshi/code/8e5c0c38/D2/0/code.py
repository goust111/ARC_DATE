# Variant of 8e5c0c38
# Dimension: D2 — 对称轴方向 (transform)
# Change: 仅搜索垂直对称轴（遍历所有可能的垂直轴位置，选择得分最高的轴）... → 水平对称轴修复

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

for parent in Path(__file__).resolve().parents:
    if (parent / "arc_utils.py").exists():
        sys.path.insert(0, str(parent))
        break

from arc_utils import colors_present, get_bg, paste

# Design analysis for the D2 variant
# a) 具体化与行为追踪
# - 每个非背景色仍然代表一个独立精灵，但“正确形态”从左右镜像改成上下镜像。
# - 输入里每个精灵都接近某条水平轴 r = k/2 对称；噪声像素是同色单点，贴着精灵边缘，
#   且它在该水平轴另一侧的镜像位置不是同色，因此输出应删除它。
# - 小例子：若某色在行 2 与行 6 上有对应像素，则当 k=8 时二者互为镜像；若额外在行 1
#   放入一个同色点，而行 7 同列不是该色，那么原题的纵轴逻辑会检查左右配对，可能保留
#   这个点；本变体必须检查上下配对，并删除它。
# - 受影响环节是整条管线：solver 的候选轴枚举、评分与删除判据都要从列坐标改成行坐标；
#   generator 也必须把“干净图形的镜像复制方向”和“噪声镜像失配方向”一起改掉。
#
# b) 复杂度校准
# - 保留原题的三色独立对象、22x22 画布、像素数/密度/噪声比例与轴分差约束，避免把题做成
#   单对象或过于稀疏的平凡镜像。
# - 仍允许整数轴与半整数轴两类水平轴：奇数高精灵有中心行，偶数高精灵无中心行。
# - 为了让水平对称在视觉上明确，生成器要求上半部分覆盖所有列，从而让对象横向展开，
#   不会退化成窄竖线导致“横轴改动”肉眼不明显。
#
# c) 隐式假设审查
# - 原 main() 的 `cc = k - c` 假设“镜像对象沿列变化”；变体中必须改成 `rr = k - r`。
# - 原 `for k in range(0, 2*(W-1)+1)` 依赖候选轴定义在列域；现在必须改为高 H 的行域。
# - 原删除分支 `out[rr, c] = bg` 用当前行、镜像列判定是否保留；若只改评分不改删除，solver
#   会按错误方向删点，因此删除阶段也要同步改成检查镜像行。
# - 原放置时 `true_k = axis_k_local + 2*c0` 假设平移作用在列坐标；水平轴的全局换算应为
#   `axis_k_local + 2*r0`。若漏改，noise 约束和二次评分都会围绕错误的轴。
# - 原噪声候选里“在轴上”的定义是 `mirror_c == c`，且排除 clean_mask[:, mirror_c]；这些都
#   隐含了左右镜像，必须整体替换为按行镜像。
#
# d) 放置合法性
# - 继续使用固定的三个 slot。它们本来就相互分离，能保证 clean sprite 不重叠、不过界。
# - 噪声仍只从 clean sprite 的 4 邻域里挑选，slot 之间的留白足以避免不同对象颜色互相覆盖。
# - solver 逐色处理整张网格，因此 generator 只需保证同色不会分裂成多处无关对象；三色独立
#   slot 满足这一点。
#
# e) 视觉可观察性
# - 输入输出对里最直观的差异是：输出会清除“上方或下方多出来”的同色点，而不是左右多出的点。
# - 保留随机上下翻转与左右翻转。左右翻转不会改变“水平轴”这个核心规律；上下翻转会移动轴位置，
#   有助于样例覆盖不同轴高度，但人眼仍能看出是上下镜像修复。
#
# f) 可视化整洁度
# - 每个精灵独占一个区域，颜色互异，噪声贴近本体且数量受限，因此不会出现大面积混色覆盖。
# - 输出仍是原位置上的干净精灵，不引入额外标记或重排，视觉负担与原题一致。
#
# g) 颜色变化可推断性
# - 本变体不做颜色映射，只利用“同色像素是否具备上下镜像伙伴”这一可见结构条件。
#
# h) 量化退化检查思路
# - 若生成器真的完成了 D2 变更，则在一批样本上，新 solver 与原始纵轴 solver 对同一输入应当
#   频繁产生不同输出；否则就可能退化成同时满足两种对称的特例。测试阶段将统计这一差异率。
#
# i) main() 新假设与 generator 对应保证
# - main() 现在假设：对每种颜色，最佳解释来自单一水平轴，且错误像素主要体现为“缺少上下镜像”。
# - generator 用 output-first 构造严格水平对称的干净精灵，再只添加没有上下镜像伙伴的噪声，
#   并要求最佳轴比分次佳轴至少高 2，从而保证这些假设成立。


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each color layer by restoring horizontal mirror symmetry.

    Input structure:
    - One background color fills most cells. Three other colors each form a nearly symmetric sprite around some horizontal axis.
    - A few same-color pixels sit above or below the true shape without a reflected partner.

    Rule:
    - For each non-background color, search every possible horizontal axis independently and keep the axis with the highest mirror-match score.
    - Delete any pixel whose same-color reflection across that chosen horizontal axis is absent.

    Key insight:
    - The symmetry cue is local to each color layer, and the only evidence that matters is top/bottom pairing rather than whole-grid arrangement.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    out = g.copy()
    colors = colors_present(g, exclude={bg})
    assert colors, "grid must contain at least one non-background color"

    # Search all possible horizontal axes: k in [0, 2*(H-1)] corresponds to axis at r = k/2.
    for col in colors:
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue

        best_score = -1
        best_k = 0
        for k in range(0, 2 * (H - 1) + 1):
            score = 0
            for r, c in pts:
                rr = int(k) - int(r)
                if 0 <= rr < H and int(g[rr, int(c)]) == int(col):
                    score += 1
            if score > best_score:
                best_score = int(score)
                best_k = int(k)

        # Delete unpaired pixels under the chosen horizontal axis.
        k = int(best_k)
        for r, c in pts:
            rr = int(k) - int(r)
            if not (0 <= rr < H and int(g[rr, int(c)]) == int(col)):
                out[int(r), int(c)] = int(bg)

    return out


# --------------- generator (codex-att3, D2 horizontal-axis variant) ---------------

# ── CFG: canvas & layout ──
GRID_SIZE = 22
NUM_OBJECTS = 4
LAYOUTS = [
    [(0, 0, 7, 7), (0, 14, 7, 21), (14, 5, 21, 16)],
    [(0, 0, 8, 5), (0, 16, 8, 21), (15, 5, 21, 16)],
]

# ── CFG: clean sprite geometry ──
PIXEL_COUNT_MIN = 6
PIXEL_COUNT_MAX = 46
HEIGHT_MIN = 2
HEIGHT_MAX = 9
WIDTH_MIN = 3
WIDTH_MAX = 12
ODD_SIDE_H_RANGE = range(1, 5)   # h = 2*side_h + 1 -> 3,5,7,9
EVEN_SIDE_H_RANGE = range(1, 5)  # h = 2*side_h     -> 2,4,6,8
DENSITY_LO = 0.35
DENSITY_HI = 0.89

# ── CFG: noise ──
NOISE_FRAC_LO = 0.11
NOISE_FRAC_HI = 0.33
NOISE_MAX = 12

# ── CFG: axis gap thresholds ──
SPRITE_AXIS_GAP_MIN = 3
FINAL_AXIS_GAP_MIN = 2

# ── CFG: retry limits ──
SPRITE_RETRIES = 120
GEN_RETRIES = 300

# ── CFG: augmentation ──
FLIP_PROB = 0.5


def _axis_scores(coords: list[tuple[int, int]], height: int = GRID_SIZE) -> dict[int, int]:
    """Return {axis_k: symmetric_pair_count} for every candidate horizontal axis."""
    pts = {(int(r), int(c)) for r, c in coords}
    scores: dict[int, int] = {}
    for k in range(2 * (height - 1) + 1):
        score = 0
        for r, c in pts:
            rr = k - r
            if 0 <= rr < height and (rr, c) in pts:
                score += 1
        scores[k] = score
    return scores


def _second_best_score(coords: list[tuple[int, int]], true_k: int) -> int:
    """Return the highest horizontal-axis score excluding the true axis."""
    scores = _axis_scores(coords)
    return max(s for k, s in scores.items() if k != true_k)


def _feasible_shape_options(max_h: int, max_w: int) -> list[tuple[int, int, int, int]]:
    """Enumerate all (h, w, target_count, center_count) tuples for horizontal symmetry.

    For odd-height sprites (integer axis), center_count >= 1 counts center-row pixels.
    For even-height sprites (half-integer axis), center_count == 0.
    """
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
                if not (math.ceil(DENSITY_LO * area) <= target_count <= math.floor(DENSITY_HI * area)):
                    continue
                for center_count in range(1, w + 1):
                    if (target_count - center_count) % 2 != 0:
                        continue
                    top_count = (target_count - center_count) // 2
                    if w <= top_count <= side_h * w:
                        options.append((h, w, target_count, center_count))
            if target_count % 2 != 0:
                continue
            for side_h in EVEN_SIDE_H_RANGE:
                h = 2 * side_h
                if h > h_max:
                    continue
                area = h * w
                if not (math.ceil(DENSITY_LO * area) <= target_count <= math.floor(DENSITY_HI * area)):
                    continue
                top_count = target_count // 2
                if w <= top_count <= side_h * w:
                    options.append((h, w, target_count, 0))
    return options


def _choose_shape_spec(max_h: int, max_w: int) -> tuple[int, int, int, int]:
    """Pick a random (h, w, target_count, center_count), weighting toward denser fills."""
    options = _feasible_shape_options(max_h, max_w)
    if not options:
        raise ValueError("no feasible clean sprite spec")
    counts = sorted({tc for _, _, tc, _ in options})
    target_count = random.choice(counts)
    target_options = [opt for opt in options if opt[2] == target_count]
    weights = [1.0 / (h * w) for h, w, _, _ in target_options]
    return random.choices(target_options, weights=weights, k=1)[0]


def _top_cells(side_h: int, w: int, top_count: int) -> set[tuple[int, int]]:
    """Choose top-half cells from a (side_h x w) grid, ensuring every column is represented."""
    cells: set[tuple[int, int]] = set()
    inner_anchor_col = random.randrange(w)
    for c in range(w):
        if c == 0:
            r = 0
        elif side_h > 1 and c == inner_anchor_col:
            r = side_h - 1
        else:
            r = random.randrange(side_h)
        cells.add((r, c))

    remaining = top_count - len(cells)
    if remaining < 0:
        raise ValueError("top_count too small")

    pool = [(r, c) for r in range(side_h) for c in range(w) if (r, c) not in cells]
    cells.update(random.sample(pool, remaining))
    return cells


def _center_cols(w: int, center_count: int) -> np.ndarray:
    """Return a boolean mask of which columns carry a center-row pixel."""
    cols = np.zeros(w, dtype=bool)
    if center_count > 0:
        cols[random.sample(range(w), center_count)] = True
    return cols


def _build_clean_sprite(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, int]:
    """Build a horizontally symmetric sprite and return (sprite_array, local_axis_k)."""
    for _ in range(SPRITE_RETRIES):
        h, w, target_count, center_count = _choose_shape_spec(max_h, max_w)
        side_h = h // 2
        top_count = (target_count - center_count) // 2
        top = _top_cells(side_h, w, top_count)
        centers = _center_cols(w, center_count)

        sprite = np.zeros((h, w), dtype=int)
        for r, c in top:
            mirror_r = h - 1 - r
            sprite[r, c] = color
            sprite[mirror_r, c] = color
        if center_count > 0:
            sprite[side_h, centers] = color

        coords = [tuple(x) for x in np.argwhere(sprite == color)]
        true_k = h - 1
        if len(coords) != target_count:
            continue
        if _second_best_score(coords, true_k) > target_count - SPRITE_AXIS_GAP_MIN:
            continue
        return sprite, true_k
    raise ValueError("failed to build clean sprite")


def _slot_place_sprites(
    bg: int, sprite_specs: list[dict], slots: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, list[dict]] | None:
    """Place sprites into slot regions on a fresh grid. Returns (grid, placed_specs) or None."""
    out = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    placed: list[dict] = []

    for spec, (r0s, c0s, r1s, c1s) in zip(sprite_specs, slots):
        spec = dict(spec)
        sprite = spec["sprite"]
        h, w = sprite.shape
        if h > r1s - r0s + 1 or w > c1s - c0s + 1:
            return None
        r0 = random.randint(r0s, r1s - h + 1)
        c0 = random.randint(c0s, c1s - w + 1)
        paste(out, sprite, r0, c0, transparent=0)
        spec["r0"] = r0
        spec["c0"] = c0
        spec["true_k"] = spec["axis_k_local"] + 2 * r0
        placed.append(spec)
    return out, placed


def _noise_candidates(clean_mask: np.ndarray, true_k: int) -> list[tuple[int, int]]:
    """Find 4-adjacent cells whose vertical mirror across the horizontal axis is absent."""
    H, W = clean_mask.shape
    adj = np.zeros_like(clean_mask)
    adj[1:, :] |= clean_mask[:-1, :]
    adj[:-1, :] |= clean_mask[1:, :]
    adj[:, 1:] |= clean_mask[:, :-1]
    adj[:, :-1] |= clean_mask[:, 1:]
    adj &= ~clean_mask

    rows = np.arange(H)
    mirror_rows = true_k - rows

    on_axis = mirror_rows == rows
    adj[on_axis, :] = False

    valid_mirror = (mirror_rows >= 0) & (mirror_rows < H) & ~on_axis
    for r in np.where(valid_mirror)[0]:
        adj[r, :] &= ~clean_mask[mirror_rows[r], :]

    rows_arr, cols_arr = np.where(adj)
    return list(zip(rows_arr.tolist(), cols_arr.tolist()))


def _add_noise(inp: np.ndarray, out: np.ndarray, spec: dict) -> bool:
    """Add asymmetric noise pixels to one sprite in inp. Returns False on constraint failure."""
    color = spec["color"]
    true_k = spec["true_k"]
    clean_coords = [tuple(x) for x in np.argwhere(out == color)]
    clean_mask = out == color
    clean_count = len(clean_coords)

    lo = max(1, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if lo > hi:
        return False
    noise_target = random.randint(lo, hi)
    candidates = _noise_candidates(clean_mask, true_k)
    if len(candidates) < noise_target:
        return False

    chosen: list[tuple[int, int]] = []
    forbidden: set[tuple[int, int]] = set()
    current_coords = list(clean_coords)
    for _ in range(noise_target):
        viable = [cell for cell in candidates if cell not in forbidden and cell not in chosen]
        if not viable:
            return False

        scored = []
        for cell in viable:
            alt = _second_best_score(current_coords + [cell], true_k)
            scored.append((alt, random.random(), cell))
        scored.sort(key=lambda t: (t[0], t[1]))
        best_alt = scored[0][0]
        pool = [cell for alt, _, cell in scored if alt <= best_alt + 1]
        cell = random.choice(pool)
        chosen.append(cell)
        current_coords.append(cell)
        r, c = cell
        forbidden.add(cell)
        mirror = (true_k - r, c)
        if 0 <= mirror[0] < GRID_SIZE:
            forbidden.add(mirror)

    if clean_count - _second_best_score(current_coords, true_k) < FINAL_AXIS_GAP_MIN:
        return False

    for r, c in chosen:
        inp[r, c] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: build clean sprites, place them, add noise, validate via solver."""
    bg, *obj_colors = random.sample(range(10), NUM_OBJECTS)
    slots = random.choice(LAYOUTS)

    sprite_specs = []
    try:
        for color, slot in zip(obj_colors, slots):
            max_h = slot[2] - slot[0] + 1
            max_w = slot[3] - slot[1] + 1
            sprite, axis_k_local = _build_clean_sprite(color, max_h, max_w)
            sprite_specs.append({"color": color, "sprite": sprite, "axis_k_local": axis_k_local})
    except ValueError:
        return None

    result = _slot_place_sprites(bg, sprite_specs, slots)
    if result is None:
        return None
    out, placed_specs = result

    inp = out.copy()
    for spec in placed_specs:
        if not _add_noise(inp, out, spec):
            return None

    if random.random() < FLIP_PROB:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
