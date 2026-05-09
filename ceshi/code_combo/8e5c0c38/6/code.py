# Combo variant of 8e5c0c38 (index 6)
# combo_spec: D1/4+D2/0
# Name: 水平轴点对称清理

from __future__ import annotations

import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

for parent in Path(__file__).resolve().parents:
    if (parent / "arc_utils.py").exists():
        sys.path.insert(0, str(parent))
        break

from arc_utils import colors_present, get_bg, paste

# Design analysis for the combo variant
# a) 具体化与行为追踪
# - 输入仍是 22x22、3 个异色精灵分槽摆放，但“正确形态”不再是单轴镜像，而是每个颜色各自
#   围绕某个中心做 180 度旋转不变。
# - 与 D1/4 的纯点对称版不同，这里把 D2/0 的“水平轴搜索”融合进中心推断：solver 先枚举
#   候选中心行和 `kr = r1 + r2`，再在该行下选择最匹配的中心列 `kc = c1 + c2`。
# - 例子：若同色点有 (1,2) 与 (5,6)，它们支持中心和为 (6,8)；原题只会检查同一行上的
#   左右配对 `(r, k-c)`，D2 单变体只会检查同一列上的上下配对 `(k-r, c)`；本融合版必须让
#   这两个点作为一对对角旋转伙伴一起保留。
# - 因此被删掉的噪声不再是“缺少左右镜像”或“缺少上下镜像”的点，而是“缺少 180 度旋转
#   伙伴”的点。差异体现在行和列都要同时变。
#
# b) 复杂度校准
# - 保留原题的多对象、固定槽位、颜色独立处理、局部稀疏噪声，这样样例仍有足够结构支撑，
#   但不会因为对象互相干扰而让中心匹配失真。
# - 为了让“不是简单水平镜像”肉眼可见，clean sprite 会拒绝过于接近水平轴镜像或垂直轴镜像
#   的形状，尽量保留对角配对、偏心块状等更像点对称的视觉证据。
# - 同时又不能让自由度过大，所以仍限定单个颜色只对应一个精灵、一个真实中心、一个噪声族。
#
# c) 隐式假设审查
# - 原题 `best_k` 只需一个标量轴；融合后必须先选 `kr` 再选 `kc`，否则无法体现 D2/0 的
#   “先按水平轴方向搜索”的约束。
# - 原题/单轴版删除时只改一维坐标；现在删除条件必须重算 `(rr, cc) = (kr-r, kc-c)`。
# - 原 generator 的左右镜像构造和 D2 的上下镜像构造都不够，需要改成 180 度轨道对。
# - 若只保证“最佳完整中心”唯一，但不保证“最佳中心行”唯一，row-first solver 可能先选错行；
#   因此生成器必须额外控制中心行分差。
# - 若 clean shape 太接近水平镜像，解题者会误以为规则只是 D2；因此生成器要反退化。
#
# d) 放置合法性
# - 继续沿用原题的 3 个固定 slot，先在局部 bbox 里生成 clean sprite，再整体平移放入 slot。
# - slot 之间留白较大，4 邻域噪声只在本对象附近采样，避免跨色覆盖、越界或对象重叠。
# - 全局中心和通过 `(local_kr + 2*r0, local_kc + 2*c0)` 计算，保证平移后 solver 与 generator
#   使用同一组坐标系。
#
# e) 视觉可观察性
# - 人类从样例里应看到：输出保留的是“绕某个中心旋转 180 度可对上”的像素对，而不是单纯的
#   左右或上下镜像。斜对角双点、偏移矩形块等都能提供这种证据。
# - D2 的影响不表现为最终输出变成水平镜像，而表现为 solver 的中心行由水平轴式搜索稳定定位；
#   生成器用明显的上下跨度和 row-gap 约束来支撑这一内部推理。
#
# f) 可视化整洁度
# - 三个对象分区明确，颜色不混，噪声只贴着本体边缘，删噪后输出仍是原位的干净精灵。
# - 不引入额外标记、排序或颜色映射，观察者只需要盯住每个单色对象的局部几何关系。
#
# g) 变化的可推断性
# - 颜色本身不携带规则；任何非背景色都按同一套“row-first 点对称清理”处理。
# - 因为不存在颜色映射，样例数量少时也不会把归纳负担转移到不可见的颜色编号上。
#
# h) 量化退化检查思路
# - 生成时拒绝 `main(inp)` 与原始纵轴 solver 或 D2 水平轴 solver 过于接近的样本，确保它
#   不是偶然也能被单轴镜像解释的特例。
# - 另外保留“最佳中心行比分次佳高”的数值约束，验证 D2 融入后的控制变量确实在起作用。
#
# i) main() 新假设与 generator 对应保证
# - main() 假设：每种颜色只有一个对象，且存在一个真实中心行 `kr/2` 先于列中心被稳定辨认。
# - generator 通过单对象分槽、row-gap / center-gap 阈值、以及不成对噪声采样来保证这些前提。


def _rotation_score_rows(coords: list[tuple[int, int]]) -> dict[int, Counter[int]]:
    """Map each center-row sum kr to a Counter of center-column sums kc."""
    row_scores: dict[int, Counter[int]] = defaultdict(Counter)
    pts = [(int(r), int(c)) for r, c in coords]
    for r1, c1 in pts:
        for r2, c2 in pts:
            row_scores[r1 + r2][c1 + c2] += 1
    return row_scores


def _rotation_score_map(coords: list[tuple[int, int]]) -> Counter[tuple[int, int]]:
    """Score every 180-degree center `(kr, kc)` using ordered point pairs."""
    scores: Counter[tuple[int, int]] = Counter()
    for kr, cols in _rotation_score_rows(coords).items():
        for kc, score in cols.items():
            scores[(kr, kc)] = int(score)
    return scores


def _best_center_by_horizontal_search(coords: list[tuple[int, int]]) -> tuple[int, int]:
    """Choose a rotation center by row-first search: best `kr`, then best `kc` under that row."""
    row_scores = _rotation_score_rows(coords)
    best_row_score = max(max(cols.values()) for cols in row_scores.values())
    best_kr = min(kr for kr, cols in row_scores.items() if max(cols.values()) == best_row_score)
    col_scores = row_scores[best_kr]
    best_col_score = max(col_scores.values())
    best_kc = min(kc for kc, score in col_scores.items() if score == best_col_score)
    return int(best_kr), int(best_kc)


def _second_best_rotation_score(
    coords: list[tuple[int, int]],
    true_center: tuple[int, int],
) -> int:
    """Return the highest full-center score excluding the true center."""
    scores = _rotation_score_map(coords)
    scores.pop((int(true_center[0]), int(true_center[1])), None)
    return max(scores.values(), default=0)


def _second_best_row_score(coords: list[tuple[int, int]], true_kr: int) -> int:
    """Return the highest row-first score excluding the true center row."""
    row_scores = _rotation_score_rows(coords)
    best_per_row = {kr: max(cols.values()) for kr, cols in row_scores.items()}
    return max((score for kr, score in best_per_row.items() if kr != int(true_kr)), default=0)


def _axis_scores_vertical(coords: list[tuple[int, int]], width: int) -> dict[int, int]:
    """Original-task style vertical-axis scores, used only for anti-degeneration checks."""
    pts = {(int(r), int(c)) for r, c in coords}
    scores: dict[int, int] = {}
    for k in range(2 * (width - 1) + 1):
        score = 0
        for r, c in pts:
            cc = k - c
            if 0 <= cc < width and (r, cc) in pts:
                score += 1
        scores[k] = score
    return scores


def _axis_scores_horizontal(coords: list[tuple[int, int]], height: int) -> dict[int, int]:
    """D2-style horizontal-axis scores, used only for anti-degeneration checks."""
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clean each color layer by enforcing point symmetry with a row-first center search.

    Input structure:
    - One background color fills most cells. Each non-background color forms one noisy sprite that is close to 180-degree rotational symmetry.
    - The true center is color-specific, and extra same-color pixels appear near the sprite without a valid rotated counterpart.

    Rule:
    - For each non-background color, first search for the best center row as if locating a hidden horizontal axis, then choose the best center column under that row.
    - Delete any pixel whose 180-degree partner around the chosen center is absent.

    Key insight:
    - The task is still independent per color, but the hidden parameter is no longer a mirror axis. You must infer a rotation center, with the row coordinate found first and the column coordinate resolved second.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    out = g.copy()
    colors = colors_present(g, exclude={bg})
    assert colors, "grid must contain at least one non-background color"

    for col in colors:
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue

        coords = [tuple(x) for x in pts]
        kr, kc = _best_center_by_horizontal_search(coords)
        for r, c in pts:
            rr = int(kr) - int(r)
            cc = int(kc) - int(c)
            if not (0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(col)):
                out[int(r), int(c)] = int(bg)

    return out


# --------------- generator ---------------

# Canvas and layout
GRID_SIZE = 22
NUM_OBJECTS = 4
LAYOUTS = [
    [(0, 0, 7, 7), (0, 14, 7, 21), (14, 5, 21, 16)],
    [(0, 0, 8, 5), (0, 16, 8, 21), (15, 5, 21, 16)],
]

# Clean sprite geometry
PIXEL_COUNT_MIN = 6
PIXEL_COUNT_MAX = 46
HEIGHT_MIN = 2
HEIGHT_MAX = 9
WIDTH_MIN = 3
WIDTH_MAX = 12
DENSITY_LO = 0.35
DENSITY_HI = 0.89

# Noise
NOISE_FRAC_LO = 0.11
NOISE_FRAC_HI = 0.33
NOISE_MAX = 12

# Uniqueness and anti-degeneration
SPRITE_ROW_GAP_MIN = 3
SPRITE_CENTER_GAP_MIN = 3
FINAL_ROW_GAP_MIN = 2
FINAL_CENTER_GAP_MIN = 2
CLEAN_VERTICAL_GAP_MIN = 2
CLEAN_HORIZONTAL_GAP_MIN = 2
MIN_VARIANT_DIFF = 3

# Retry limits
SPRITE_RETRIES = 220
GEN_RETRIES = 450

# Augmentation
FLIP_PROB = 0.5


def _feasible_shape_options(max_h: int, max_w: int) -> list[tuple[int, int, int, int]]:
    """Enumerate feasible `(h, w, target_count, center_count)` tuples for point-symmetric sprites."""
    options: list[tuple[int, int, int, int]] = []
    h_max = min(HEIGHT_MAX, max_h)
    w_max = min(WIDTH_MAX, max_w)
    for target_count in range(PIXEL_COUNT_MIN, PIXEL_COUNT_MAX + 1):
        for h in range(HEIGHT_MIN, h_max + 1):
            for w in range(WIDTH_MIN, w_max + 1):
                area = h * w
                if not (math.ceil(DENSITY_LO * area) <= target_count <= math.floor(DENSITY_HI * area)):
                    continue

                center_options = [0]
                if h % 2 == 1 and w % 2 == 1:
                    center_options = [0, 1]

                for center_count in center_options:
                    remaining = target_count - center_count
                    if remaining < 0 or remaining % 2 != 0:
                        continue
                    pair_count = remaining // 2
                    max_pairs = (area - center_count) // 2
                    if 1 <= pair_count <= max_pairs:
                        options.append((h, w, target_count, center_count))
    return options


def _choose_shape_spec(max_h: int, max_w: int) -> tuple[int, int, int, int]:
    """Pick one feasible sprite spec, biased toward denser boxes."""
    options = _feasible_shape_options(max_h, max_w)
    if not options:
        raise ValueError("no feasible clean sprite spec")
    counts = sorted({target_count for _, _, target_count, _ in options})
    target_count = random.choice(counts)
    target_options = [opt for opt in options if opt[2] == target_count]
    weights = [1.0 / (h * w) for h, w, _, _ in target_options]
    return random.choices(target_options, weights=weights, k=1)[0]


def _rotation_orbits(h: int, w: int) -> list[tuple[tuple[int, int], ...]]:
    """List all 180-degree orbits inside an `h x w` bbox."""
    orbits: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            rr, cc = h - 1 - r, w - 1 - c
            if (r, c) == (rr, cc):
                orbit = ((r, c),)
            else:
                orbit = ((r, c), (rr, cc))
            for cell in orbit:
                seen.add(cell)
            orbits.append(orbit)
    return orbits


def _touch_count(orbit: tuple[tuple[int, int], ...], occupied: set[tuple[int, int]]) -> int:
    """Count 4-neighbor contacts between an orbit and the current occupied set."""
    total = 0
    for r, c in orbit:
        total += int((r - 1, c) in occupied)
        total += int((r + 1, c) in occupied)
        total += int((r, c - 1) in occupied)
        total += int((r, c + 1) in occupied)
    return total


def _build_clean_sprite(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a clean point-symmetric sprite with a distinctive center row."""
    for _ in range(SPRITE_RETRIES):
        h, w, target_count, center_count = _choose_shape_spec(max_h, max_w)
        pair_count = (target_count - center_count) // 2
        orbits = _rotation_orbits(h, w)
        pair_indices = [i for i, orbit in enumerate(orbits) if len(orbit) == 2]
        if len(pair_indices) < pair_count:
            continue

        top_candidates = [i for i in pair_indices if any(r == 0 for r, _ in orbits[i])]
        left_candidates = [i for i in pair_indices if any(c == 0 for _, c in orbits[i])]
        if not top_candidates or not left_candidates:
            continue

        selected: set[int] = {random.choice(top_candidates), random.choice(left_candidates)}
        if len(selected) > pair_count:
            continue

        occupied: set[tuple[int, int]] = set()
        for idx in selected:
            occupied.update(orbits[idx])

        while len(selected) < pair_count:
            remaining = [i for i in pair_indices if i not in selected]
            if not remaining:
                break
            frontier = [i for i in remaining if _touch_count(orbits[i], occupied) > 0]
            pool = frontier if frontier else remaining
            weights = [1 + _touch_count(orbits[i], occupied) for i in pool]
            idx = random.choices(pool, weights=weights, k=1)[0]
            selected.add(idx)
            occupied.update(orbits[idx])

        if len(selected) != pair_count:
            continue

        sprite = np.zeros((h, w), dtype=int)
        for idx in selected:
            for r, c in orbits[idx]:
                sprite[r, c] = color
        if center_count == 1:
            sprite[h // 2, w // 2] = color

        coords = [tuple(x) for x in np.argwhere(sprite == color)]
        if len(coords) != target_count:
            continue

        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        if min(rows) != 0 or max(rows) != h - 1 or min(cols) != 0 or max(cols) != w - 1:
            continue

        true_center = (h - 1, w - 1)
        true_kr = true_center[0]
        if _second_best_row_score(coords, true_kr) > target_count - SPRITE_ROW_GAP_MIN:
            continue
        if _second_best_rotation_score(coords, true_center) > target_count - SPRITE_CENTER_GAP_MIN:
            continue
        if max(_axis_scores_vertical(coords, w).values(), default=0) > target_count - CLEAN_VERTICAL_GAP_MIN:
            continue
        if max(_axis_scores_horizontal(coords, h).values(), default=0) > target_count - CLEAN_HORIZONTAL_GAP_MIN:
            continue
        return sprite, true_center
    raise ValueError("failed to build clean sprite")


def _slot_place_sprites(
    bg: int,
    sprite_specs: list[dict],
    slots: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, list[dict]] | None:
    """Place clean sprites into fixed non-overlapping slots."""
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
        local_kr, local_kc = spec["center_local"]
        spec["true_center"] = (local_kr + 2 * r0, local_kc + 2 * c0)
        placed.append(spec)
    return out, placed


def _noise_candidates(clean_mask: np.ndarray, true_center: tuple[int, int]) -> list[tuple[int, int]]:
    """Return adjacent cells whose rotated partner is absent from the clean sprite."""
    H, W = clean_mask.shape
    adj = np.zeros_like(clean_mask)
    adj[1:, :] |= clean_mask[:-1, :]
    adj[:-1, :] |= clean_mask[1:, :]
    adj[:, 1:] |= clean_mask[:, :-1]
    adj[:, :-1] |= clean_mask[:, 1:]
    adj &= ~clean_mask

    kr, kc = int(true_center[0]), int(true_center[1])
    rows, cols = np.where(adj)
    candidates: list[tuple[int, int]] = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        rr = kr - r
        cc = kc - c
        if rr == r and cc == c:
            continue
        if 0 <= rr < H and 0 <= cc < W and clean_mask[rr, cc]:
            continue
        candidates.append((r, c))
    return candidates


def _add_noise(inp: np.ndarray, out: np.ndarray, spec: dict) -> bool:
    """Add unmatched rotational noise while preserving row-first center identifiability."""
    color = int(spec["color"])
    true_center = tuple(spec["true_center"])
    clean_coords = [tuple(x) for x in np.argwhere(out == color)]
    clean_mask = out == color
    clean_count = len(clean_coords)

    lo = max(1, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if lo > hi:
        return False

    noise_target = random.randint(lo, hi)
    candidates = _noise_candidates(clean_mask, true_center)
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
            trial = current_coords + [cell]
            row_alt = _second_best_row_score(trial, true_center[0])
            center_alt = _second_best_rotation_score(trial, true_center)
            scored.append((row_alt, center_alt, random.random(), cell))
        scored.sort(key=lambda item: (item[0], item[1], item[2]))
        best_row_alt, best_center_alt = scored[0][0], scored[0][1]
        pool = [
            cell
            for row_alt, center_alt, _, cell in scored
            if row_alt <= best_row_alt + 1 and center_alt <= best_center_alt + 1
        ]
        cell = random.choice(pool)
        chosen.append(cell)
        current_coords.append(cell)
        forbidden.add(cell)
        rr = int(true_center[0]) - cell[0]
        cc = int(true_center[1]) - cell[1]
        if 0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE:
            forbidden.add((rr, cc))

    if clean_count - _second_best_row_score(current_coords, true_center[0]) < FINAL_ROW_GAP_MIN:
        return False
    if clean_count - _second_best_rotation_score(current_coords, true_center) < FINAL_CENTER_GAP_MIN:
        return False

    for r, c in chosen:
        inp[r, c] = color
    return True


def _vertical_baseline(grid: np.ndarray) -> np.ndarray:
    """Original-task solver, used only for generator rejection tests."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _, W = g.shape
    out = g.copy()

    for col in colors_present(g, exclude={bg}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        best_score = -1
        best_k = 0
        for k in range(2 * (W - 1) + 1):
            score = 0
            for r, c in pts:
                cc = int(k) - int(c)
                if 0 <= cc < W and int(g[int(r), cc]) == int(col):
                    score += 1
            if score > best_score:
                best_score = int(score)
                best_k = int(k)

        for r, c in pts:
            cc = int(best_k) - int(c)
            if not (0 <= cc < W and int(g[int(r), cc]) == int(col)):
                out[int(r), int(c)] = bg
    return out


def _horizontal_baseline(grid: np.ndarray) -> np.ndarray:
    """D2-only solver, used only for generator rejection tests."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, _ = g.shape
    out = g.copy()

    for col in colors_present(g, exclude={bg}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        best_score = -1
        best_k = 0
        for k in range(2 * (H - 1) + 1):
            score = 0
            for r, c in pts:
                rr = int(k) - int(r)
                if 0 <= rr < H and int(g[rr, int(c)]) == int(col):
                    score += 1
            if score > best_score:
                best_score = int(score)
                best_k = int(k)

        for r, c in pts:
            rr = int(best_k) - int(r)
            if not (0 <= rr < H and int(g[rr, int(c)]) == int(col)):
                out[int(r), int(c)] = bg
    return out


def _gen_attempt() -> np.ndarray | None:
    """Single output-first generation attempt."""
    bg, *obj_colors = random.sample(range(10), NUM_OBJECTS)
    slots = random.choice(LAYOUTS)

    sprite_specs = []
    try:
        for color, slot in zip(obj_colors, slots):
            max_h = slot[2] - slot[0] + 1
            max_w = slot[3] - slot[1] + 1
            sprite, center_local = _build_clean_sprite(color, max_h, max_w)
            sprite_specs.append({"color": color, "sprite": sprite, "center_local": center_local})
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

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None

    vertical = _vertical_baseline(inp)
    horizontal = _horizontal_baseline(inp)
    if np.array_equal(vertical, out):
        return None
    if np.array_equal(horizontal, out):
        return None
    if int(np.count_nonzero(vertical != out)) < MIN_VARIANT_DIFF:
        return None
    if int(np.count_nonzero(horizontal != out)) < MIN_VARIANT_DIFF:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
