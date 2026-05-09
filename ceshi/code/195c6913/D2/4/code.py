# Variant of 195c6913
# Dimension: D2 — 碰撞转弯规则：严格交替左右转 (path)
# Change: 碰撞时第1/3/5次左转(CCW)，第2/4/6次右转(CW)，严格交替；若计划转向方向不是path则停止（无后备方案）... → 碰撞时有后备方案

from __future__ import annotations

# Design analysis for the D2 fallback-turn variant
# ------------------------------------------------
# variant_spec:
#   concrete_change:
#     objects:
#       - 左上角 2x2 单色模板块仍定义循环颜色序列。
#       - 左边界 col 0 上的孤立标记像素仍从路径带内部发射粒子。
#       - 右下背景区的 2x2 终止块仍提供碰撞角标颜色。
#     path_behavior:
#       - 粒子依旧沿 path_color 区域直行，进入路径格时按模板循环推进颜色索引。
#       - 碰到前方非路径格时，仍按碰撞次数奇偶决定“首选转向”：
#         第 1/3/5... 次首选左转(CCW)，第 2/4/6... 次首选右转(CW)。
#       - 新规则只改这一点：如果首选 90 度方向不是路径，允许尝试另一个 90 度方向；
#         只有左右两个方向都不是路径时才停止。
#   behavior_trace:
#     example:
#       - 设粒子正向右移动，首次碰撞时按旧规则首选左转(向上)。
#       - 若上方是背景、下方是路径：
#         old: 在碰撞格留下终止色后直接停下。
#         new: 先检查向上失败，再检查向下成功，于是继续存活并沿下方通道前进。
#       - 可见差异不是抽象参数，而是输出里会多出一段延长的彩色轨迹和后续角标。
#   complexity_calibration:
#     - 仍然只保留“交替左右转”这一条主规则，不引入更多分支。
#     - 备用方案只在首选方向失败时触发，取值只有“尝试另一侧/两侧都失败”两种，
#       复杂度与原题接近，但样例间能稳定观察到更长轨迹。
#   implicit_assumptions_review:
#     - 路径色仍靠“g[0,0] 之外的最高频颜色”识别，因此生成器必须继续保证 path_color
#       是非背景颜色中出现次数最多的颜色。
#     - 模板检测仍依赖左上区域的 2x2 单色块，生成器必须继续在 rows 1-3 留出背景窗口。
#     - 原 solver 的 visited 状态只记录碰撞奇偶，而不是完整碰撞次数。
#       这在变体里仍成立，因为未来的首选方向只由奇偶决定；无论实际走的是首选还是备用，
#       只要成功拐弯，碰撞计数都加一即可。
#     - 原题里“首选失败就停”让很多路径天然不重叠；变体里粒子更长寿，因此生成器必须再次用
#       新 solver 校验轨迹互不重叠，不能复用旧分布假设。
#   placement_legality:
#     - 模板块、终止块、标记像素的放置规则维持原样，避免破坏识别逻辑。
#     - 生成器新增 oracle：同一输入上，fallback solver 必须至少发生一次备用拐弯，
#       且输出必须与 strict solver 不同，确保样本真的体现变体。
#   observability:
#     - 人类能直接看到：有些粒子在某个死角没有停住，而是绕向另一条 90 度支路继续前进，
#       这会让输出轨迹显著长于原题版本。
#   cleanliness:
#     - 不叠加新标记，不改变模板或终止块外观；视觉差异只来自已有轨迹延长，画面仍干净。
#   color_inference:
#     - 颜色循环本身不变，仍由模板块从左到右显式给出；新增规则不依赖颜色 id 排序或隐藏编码。

from collections import Counter
import random
import numpy as np

from arc_utils import label_mask


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: col-0 markers sit inside a vertical path segment."""
    H, _ = g.shape
    for r, c, _ in markers:
        if c != 0:
            continue
        if r > 0:
            assert int(g[r - 1, c]) == path_color, (
                f"marker at ({r},{c}) up-neighbor ({r-1},{c})={int(g[r-1,c])} "
                f"must be path_color {path_color}"
            )
        if r < H - 1:
            assert int(g[r + 1, c]) == path_color, (
                f"marker at ({r},{c}) down-neighbor ({r+1},{c})={int(g[r+1,c])} "
                f"must be path_color {path_color}"
            )


def _parse_scene(
    grid: np.ndarray,
) -> tuple[
    np.ndarray,
    int,
    int,
    list[int],
    list[tuple[int, int]],
    int | None,
    tuple[int, int] | None,
    list[tuple[int, int, int]],
]:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    template_bg = int(g[0, 0])
    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max([(cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg)])[1]
    )
    _assert_path_single_component(g, path_color)

    pattern: list[int] = []
    pat_cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r in range(min(6, H - 1)):
        for c in range(min(25, W - 1)):
            if (int(r), int(c)) in seen:
                continue
            v = int(g[int(r), int(c)])
            if v == int(template_bg) or v == int(path_color):
                continue
            if (
                int(g[int(r + 1), int(c)]) == v
                and int(g[int(r), int(c + 1)]) == v
                and int(g[int(r + 1), int(c + 1)]) == v
            ):
                pattern.append(int(v))
                for rr, cc in [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]:
                    seen.add((int(rr), int(cc)))
                    pat_cells.append((int(rr), int(cc)))

    end_color = None
    end_pos = None
    pat_set = set(map(int, pattern))
    for r in range(H - 1):
        for c in range(W - 1):
            v = int(g[int(r), int(c)])
            if v == int(template_bg) or v == int(path_color) or v in pat_set:
                continue
            if (
                int(g[int(r + 1), int(c)]) == v
                and int(g[int(r), int(c + 1)]) == v
                and int(g[int(r + 1), int(c + 1)]) == v
            ):
                end_color = int(v)
                end_pos = (int(r), int(c))
                break
        if end_pos is not None:
            break

    markers: list[tuple[int, int, int]] = []
    for r in range(H):
        for c in range(W):
            v = int(g[int(r), int(c)])
            if v not in pat_set:
                continue
            same = 0
            has_path = False
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    if int(g[rr, cc]) == v:
                        same += 1
                    if int(g[rr, cc]) == int(path_color):
                        has_path = True
            if same == 0 and has_path:
                markers.append((int(r), int(c), int(v)))

    _assert_marker_embedded_in_path(g, markers, int(path_color))
    return g, template_bg, path_color, pattern, pat_cells, end_color, end_pos, markers


def _simulate_particles(
    g: np.ndarray,
    template_bg: int,
    path_color: int,
    pattern: list[int],
    pat_cells: list[tuple[int, int]],
    end_color: int | None,
    end_pos: tuple[int, int] | None,
    markers: list[tuple[int, int, int]],
    *,
    allow_fallback: bool,
) -> tuple[np.ndarray, dict[str, int], list[set[tuple[int, int]]]]:
    out = g.copy()
    for r, c in pat_cells:
        out[int(r), int(c)] = int(template_bg)
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    dir_order = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U
    metrics = {"fallback_turns": 0, "total_turns": 0, "painted_steps": 0}
    per_marker_cells: list[set[tuple[int, int]]] = []

    for sr, sc, sv in markers:
        init = None
        for dr, dc in dir_order:
            rr = int(sr + dr)
            cc = int(sc + dc)
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and int(g[rr, cc]) == int(path_color):
                init = (int(dr), int(dc))
                break
        if init is None or not pattern:
            per_marker_cells.append(set())
            continue

        r, c = int(sr), int(sc)
        dr, dc = init
        idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
        hits = 0
        visited: set[tuple[int, int, int, int, int, int]] = set()
        my_cells: set[tuple[int, int]] = set()

        for _ in range(g.shape[0] * g.shape[1] * 4):
            state = (int(r), int(c), int(dr), int(dc), int(idx), int(hits & 1))
            if state in visited:
                break
            visited.add(state)

            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
                break

            if int(g[nr, nc]) == int(path_color):
                idx = int((idx + 1) % len(pattern))
                out[nr, nc] = int(pattern[idx])
                my_cells.add((int(nr), int(nc)))
                metrics["painted_steps"] += 1
                r, c = nr, nc
                continue

            if int(g[nr, nc]) == int(template_bg) and end_color is not None:
                out[nr, nc] = int(end_color)

            left = (-int(dc), int(dr))
            right = (int(dc), -int(dr))
            primary = left if (int(hits) % 2 == 0) else right
            secondary = right if (int(hits) % 2 == 0) else left
            candidates = [primary]
            if allow_fallback:
                candidates.append(secondary)

            turned = False
            for attempt_idx, (tdr, tdc) in enumerate(candidates):
                rr = int(r + tdr)
                cc = int(c + tdc)
                if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and int(g[rr, cc]) == int(path_color):
                    if attempt_idx == 1:
                        metrics["fallback_turns"] += 1
                    dr, dc = int(tdr), int(tdc)
                    hits += 1
                    metrics["total_turns"] += 1
                    turned = True
                    break
            if not turned:
                break

        per_marker_cells.append(my_cells)

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"

    return out, metrics, per_marker_cells


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    parsed = _parse_scene(grid)
    strict_out, strict_metrics, _ = _simulate_particles(*parsed, allow_fallback=False)
    variant_out, variant_metrics, _ = _simulate_particles(*parsed, allow_fallback=True)
    return {
        "fallback_turns": int(variant_metrics["fallback_turns"]),
        "strict_steps": int(strict_metrics["painted_steps"]),
        "variant_steps": int(variant_metrics["painted_steps"]),
        "output_diff_cells": int(np.sum(strict_out != variant_out)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace cycle-colored particles across a diagonal path, using a preferred turn and a backup turn at collisions.

    Input structure:
    - The upper-left background pocket contains a row of 2x2 monochrome blocks that spell out a cyclic color order.
    - One or two isolated pixels in those cycle colors sit on the left edge, embedded inside the path band as launch markers.
    - A separate 2x2 block of another color sits in the opposite background region and supplies the collision-marker color.

    Rule:
    - Each marker launches a particle that moves straight through path cells and advances one step through the template cycle every time it paints a path cell.
    - When the next cell is not path, that collided cell is marked with the end color if it is background.
    - Collision turns still alternate by hit count: 1st/3rd/5th prefer CCW, 2nd/4th/6th prefer CW; if that preferred turn is blocked, try the other 90-degree direction as a backup.
    - The particle stops only when neither 90-degree option leads back into the path.

    Key insight:
    - The visible novelty is not a new object type but the extra survival rule: failed preferred turns no longer kill the particle immediately, so some trajectories bend into a backup corridor and continue much farther.
    """
    parsed = _parse_scene(grid)
    out, _, _ = _simulate_particles(*parsed, allow_fallback=True)
    return out


def _build_band(H: int, W: int, lobe_rows: list[int] | None = None) -> np.ndarray:
    """Return a boolean mask for a thick meandering anti-diagonal band."""
    bot_center = random.randint(0, W // 4)
    centers = np.linspace(W - 1, bot_center, H).astype(float)
    half_w = np.array(
        [0.5 * (3 + (1.0 - abs(2.0 * r / max(1, H - 1) - 1.0)) * (W - 6)) for r in range(H)]
    )

    c_left = np.zeros(H, dtype=int)
    c_right = np.zeros(H, dtype=int)

    cc_jit = np.convolve(
        np.array([random.randint(-3, 3) for _ in range(H)], dtype=float),
        np.ones(3) / 3,
        mode="same",
    )
    w_jit = np.convolve(
        np.array([random.randint(-3, 3) for _ in range(H)], dtype=float),
        np.ones(3) / 3,
        mode="same",
    )

    for r in range(H):
        c_left_param = centers[r] - half_w[r]
        if c_left_param <= 0:
            c_left[r] = 0
            c_right[r] = int(round(centers[r] + cc_jit[r] + half_w[r] + w_jit[r] / 2))
        else:
            w = max(2, min(W, int(round(2 * half_w[r] + w_jit[r]))))
            cc = centers[r] + cc_jit[r]
            c_left[r] = int(round(cc - w / 2))
            c_right[r] = c_left[r] + w - 1

    if lobe_rows:
        lmin = min(lobe_rows) - 1
        lmax = H - 1
        for r in range(H):
            d_outside = max(0, lmin - r, r - lmax)
            c_left[r] = min(c_left[r], 2 * d_outside)

    for r in range(1, min(4, H)):
        c_left[r] = max(c_left[r], 12)
        if c_right[r] < c_left[r]:
            c_right[r] = c_left[r]

    c_right[0] = W - 1
    c_left[0] = max(c_left[0], W - 6)
    c_left[H - 1] = 0
    c_right[H - 1] = max(5, min(W - 1, c_right[H - 1]))

    for _ in range(2):
        for r in range(1, H - 1):
            if c_left[r] > c_left[r - 1] and c_left[r] > c_left[r + 1]:
                c_left[r] = max(c_left[r - 1], c_left[r + 1])

    for r in range(1, H):
        c_left[r] = min(c_left[r], c_left[r - 1] + 2)
        c_right[r] = min(c_right[r], c_right[r - 1] + 2)
        if c_left[r] > c_right[r - 1]:
            c_left[r] = c_right[r - 1]
        if c_right[r] < c_left[r - 1]:
            c_right[r] = c_left[r - 1]
        if c_right[r] < c_left[r]:
            c_right[r] = c_left[r]

    c_left = np.clip(c_left, 0, W - 1)
    c_right = np.clip(c_right, 0, W - 1)
    c_right[0] = W - 1
    c_left[H - 1] = 0

    mask = np.zeros((H, W), dtype=bool)
    for r in range(H):
        mask[r, c_left[r] : c_right[r] + 1] = True
    return mask


def _carve_fjord(mask: np.ndarray, r1: int, r2: int) -> bool:
    """Carve a background fjord from col 0 without disconnecting the band."""
    H, W = mask.shape
    sep_h = random.randint(2, 3)
    sr_min = r1 + 2
    sr_max = r2 - 1 - sep_h
    if sr_max < sr_min:
        return False
    sr = random.randint(sr_min, sr_max)

    cur_top = sr
    made_carve = False
    for k in range(W - 1):
        trial = mask.copy()
        for dr in range(sep_h):
            rr = cur_top + dr
            if 0 <= rr < H:
                trial[rr, k] = False
        _, n_trial = label_mask(trial, connectivity=4)
        if n_trial != 1:
            break
        np.copyto(mask, trial)
        made_carve = True
        if random.random() < 0.3 and cur_top > 0:
            cur_top -= 1
    return made_carve


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)
    n_markers = 2 if random.random() < 0.8 else 1

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    n_blocks = random.randint(2, 4)
    cycle_pool = palette[3:]
    cycle_colors = [random.choice(cycle_pool) for _ in range(n_blocks)]

    if n_markers == 1:
        r1 = random.randint(H // 2, H - 4)
        markers = [r1]
    else:
        r1 = random.randint(H // 3, H // 2)
        r2 = random.randint(max(r1 + 5, 2 * H // 3), H - 4)
        markers = [r1, r2]

    mask = _build_band(H, W, lobe_rows=markers)
    if mask[0, 0]:
        return None
    _, n_path = label_mask(mask, connectivity=4)
    if n_path != 1:
        return None
    if not mask[0, :].any() or not mask[:, W - 1].any():
        return None
    if not mask[H - 1, :].any() or not mask[:, 0].any():
        return None
    for r in markers:
        if not (mask[r, 0] and mask[r - 1, 0] and mask[r + 1, 0]):
            return None

    if n_markers == 2:
        trial = mask.copy()
        if not _carve_fjord(trial, r1, r2):
            return None
        if random.random() < 0.25:
            K = random.randint(2, 4)
            pre_carve_cells = mask & ~trial
            for c in range(min(K, W)):
                trial[:, c] |= pre_carve_cells[:, c]
        _, nc = label_mask(trial, connectivity=4)
        if nc != 1:
            return None
        if not trial[0, :].any() or not trial[:, W - 1].any():
            return None
        if not trial[H - 1, :].any() or not trial[:, 0].any():
            return None
        for r in markers:
            if not (trial[r, 0] and trial[r - 1, 0] and trial[r + 1, 0]):
                return None
        mask = trial

    _, n_bg = label_mask(~mask, connectivity=4)
    if n_bg < 2:
        return None

    template_row = random.choice([1, 2])
    block_cols = [1, 4, 7, 10][:n_blocks]
    if block_cols[-1] + 1 >= W:
        return None
    for bc in block_cols:
        for dr in range(2):
            for dc in range(2):
                rr, cc = template_row + dr, bc + dc
                if rr >= H or cc >= W or mask[rr, cc]:
                    return None

    g = np.full((H, W), bg, dtype=int)
    g[mask] = path_color

    for i, bc in enumerate(block_cols):
        g[template_row : template_row + 2, bc : bc + 2] = cycle_colors[i]

    end_candidates = []
    for r in range(H // 2, H - 1):
        for c in range(W // 2, W - 1):
            if not (g[r : r + 2, c : c + 2] == bg).all():
                continue
            r0, r1_ = max(0, r - 1), min(H, r + 3)
            c0, c1_ = max(0, c - 1), min(W, c + 3)
            if (g[r0:r1_, c0:c1_] == path_color).any():
                continue
            end_candidates.append((r, c))
    if not end_candidates:
        return None
    er, ec = random.choice(end_candidates)
    g[er : er + 2, ec : ec + 2] = end_color

    for r in markers:
        g[r, 0] = cycle_colors[0]

    _, n_path_after = label_mask(g == path_color, connectivity=4)
    if n_path_after != 1:
        return None
    if g[0, 0] != bg:
        return None

    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted([(c, col) for col, c in cnt.items() if col != bg], reverse=True)
    if not non_bg or non_bg[0][1] != path_color:
        return None

    try:
        metrics = _variant_metrics(g)
        _ = main(g)
    except Exception:
        return None

    if metrics["fallback_turns"] < 1:
        return None
    if metrics["variant_steps"] <= metrics["strict_steps"]:
        return None
    if metrics["output_diff_cells"] <= 0:
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(300):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 300 attempts")
