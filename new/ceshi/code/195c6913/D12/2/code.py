# Variant of 195c6913
# Dimension: D12 — 颜色循环起始位 (trajectory)
# Change: 每个标记的循环起始位由其颜色在模板循环中的索引决定（标记色=循环色[0]，所以从索引0开始）。继续沿整条轨迹累积循环索引 → 碰撞后循环重置为0

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask

# Design analysis
# ---------------
# spec_variant:
#   objects:
#     - 输入仍然由左上模板 2x2 色块、对角粗路径、col 0 上的起点像素、右下终止 2x2 色块组成。
#     - 标记仍然全部使用模板循环的第一个颜色；因此首段直行仍从循环索引 0 起步。
#   core_change:
#     - 原任务中，粒子转弯后保留碰撞前的 idx，下一段直行会继续当前相位。
#     - 本变体中，只要碰撞后成功按既定方向转弯，就把 idx 重置为 0。
#     - 因为涂色逻辑是“先 idx=(idx+1)%k 再上色”，所以每一段新直行的首格都会再次涂成 pattern[1]，
#       从而让各段直行共享同一局部起始相位。
#   behavior_trace:
#     - 例：pattern=[A,B,C]，marker=A。首段 4 格时，两版都涂 B,C,A,B。
#     - 若随后发生一次成功左转：
#       原版保留 idx=1，下一段首格会涂 C，再涂 A,B,...
#       变体重置 idx=0，下一段首格会涂 B，再涂 C,A,...
#     - 可观察差异不在路径形状，而在“每次转弯后的颜色条纹是否重新从同一颜色开始”。
#   complexity_calibration:
#     - 若模板只有 2 色或允许重复色，reset 后的相位差异容易被偶然抵消，人眼也更难确认规律。
#     - 因此 generator 收紧为 3-4 个互异循环色，且拒绝新旧输出完全相同的样本。
#   implicit_assumptions_review:
#     - 原 solver 里真正依赖旧属性的是碰撞成功转弯后的 `idx` 保持不变；其余对象检测、转向奇偶、路径单连通、
#       标记嵌入等都不依赖旧相位规则，可保持不变。
#     - visited 状态必须继续包含 idx；因为 reset 后同一位置与方向在不同 idx 下仍可能导向不同后续上色。
#   placement_legality:
#     - 模板区仍固定在左上留白区域；终止块仍放在右下背景区并与路径隔开至少 1 格。
#     - 标记仍覆盖 col 0 的路径单元，且上下邻居必须仍为 path，保证 solver 的嵌入断言成立。
#     - n=2 时保留 fjord / interior-island 分隔器，保证两条轨迹不重叠。
#   visual_observability:
#     - 核心视觉线索是“每次拐弯后颜色序列重新从同一模板位置开始”，所以样本必须真的发生转弯，
#       且转弯后的直行段不能太短。
#     - generator 通过对照旧规则，要求 `main_new(input)` 与 legacy 输出至少在 2 个格子上不同，
#       避免退化成原题不可区分样本。
#   cleanliness:
#     - 不引入新对象，也不让同一区域叠加额外标记；差异只体现在已有轨迹的颜色节奏上，视觉上仍整洁。


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: each col-0 marker keeps path cells directly above/below."""
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


def _parse_instance(
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
        max(
            [(cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg)]
        )[1]
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
    return g, H, W, pattern, pat_cells, end_color, end_pos, markers


def _solve(
    grid: np.ndarray, *, reset_on_turn: bool, return_stats: bool = False
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    g, H, W, pattern, pat_cells, end_color, end_pos, markers = _parse_instance(grid)
    template_bg = int(g[0, 0])

    out = g.copy()
    for r, c in pat_cells:
        out[int(r), int(c)] = int(template_bg)
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max(
            [(cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg)]
        )[1]
    )
    dir_order = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U

    per_marker_cells: list[set[tuple[int, int]]] = []
    total_turns = 0
    markers_with_turn = 0
    for sr, sc, sv in markers:
        init = None
        for dr, dc in dir_order:
            rr = int(sr + dr)
            cc = int(sc + dc)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
                init = (int(dr), int(dc))
                break
        if init is None or not pattern:
            per_marker_cells.append(set())
            continue

        r, c = int(sr), int(sc)
        dr, dc = init
        idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
        hits = 0
        my_turns = 0
        visited: set[tuple[int, int, int, int, int, int]] = set()
        my_cells: set[tuple[int, int]] = set()

        for _ in range(H * W * 4):
            state = (int(r), int(c), int(dr), int(dc), int(idx), int(hits & 1))
            if state in visited:
                break
            visited.add(state)

            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if int(g[nr, nc]) == int(path_color):
                idx = int((idx + 1) % len(pattern))
                out[nr, nc] = int(pattern[idx])
                my_cells.add((int(nr), int(nc)))
                r, c = nr, nc
                continue

            if int(g[nr, nc]) == int(template_bg) and end_color is not None:
                out[nr, nc] = int(end_color)

            left = (-int(dc), int(dr))
            right = (int(dc), -int(dr))
            tdr, tdc = left if (int(hits) % 2 == 0) else right
            rr = int(r + tdr)
            cc = int(c + tdc)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
                dr, dc = int(tdr), int(tdc)
                hits += 1
                my_turns += 1
                if reset_on_turn:
                    idx = 0
                continue
            break

        total_turns += my_turns
        if my_turns > 0:
            markers_with_turn += 1
        per_marker_cells.append(my_cells)

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, (
                f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"
            )

    if not return_stats:
        return out
    return out, {
        "markers": len(markers),
        "turns": total_turns,
        "markers_with_turn": markers_with_turn,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace particles through a path while each successful turn restarts the color phase.

    Input structure:
    - A left-top row of 2x2 monochrome blocks provides the repeating color cycle.
    - One or two isolated cycle-colored pixels on col 0 act as launch markers inside the path.
    - A separate 2x2 block of another color marks the collision color used at dead ends.

    Rule:
    - Each marker moves along the path in the first available direction and paints visited path cells by cycling template colors.
    - Colliding with a non-path cell paints that collision cell with the end color, then attempts the scheduled 90-degree turn.
    - Turn direction still alternates strictly CCW, CW, CCW, CW, and if the scheduled turn is not path the particle stops.
    - After every successful turn, the cycle index resets to 0, so every new straight segment restarts the same paint phase.

    Key insight:
    - The path geometry and alternating turns decide where segments begin; the variant's visible twist is that segment colors re-synchronize after each turn instead of carrying phase across corners.
    """
    return _solve(grid, reset_on_turn=True)


def _build_band(H: int, W: int, lobe_rows: list[int] | None = None) -> np.ndarray:
    """Return a boolean mask for a thick meandering anti-diagonal band."""
    bot_center = random.randint(0, W // 4)
    centers = np.linspace(W - 1, bot_center, H).astype(float)
    half_w = np.array(
        [
            0.5 * (3 + (1.0 - abs(2.0 * r / max(1, H - 1) - 1.0)) * (W - 6))
            for r in range(H)
        ]
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
            c_right[r] = int(
                round(centers[r] + cc_jit[r] + half_w[r] + w_jit[r] / 2)
            )
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
    """Carve a bg fjord from col 0 without disconnecting the band."""
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


def _count_changed_cells(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(np.array(a, dtype=int) != np.array(b, dtype=int)))


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)
    n_markers = 2 if random.random() < 0.7 else 1

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    n_blocks = random.randint(3, 4)
    cycle_pool = palette[3:]
    cycle_colors = random.sample(cycle_pool, n_blocks)

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
        new_out, new_stats = _solve(g, reset_on_turn=True, return_stats=True)
        old_out = _solve(g, reset_on_turn=False)
    except Exception:
        return None

    changed = _count_changed_cells(new_out, old_out)
    if new_stats["turns"] < 1:
        return None
    if changed < 2:
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 400 attempts")
