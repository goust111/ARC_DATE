# Combo variant of 195c6913 (index 5)
# combo_spec: D1/2+D2/4
# Name: 走廊跟随+碰撞有后备

from __future__ import annotations

# Design analysis for combo D1/2 + D2/4
# --------------------------------------
# combo_spec:
#   concrete_change:
#     - 保留原题三类显式对象：左上模板 2x2 色块、col 0 启动标记、远处 2x2 终止块。
#     - D1/2 把原先“厚路径带”改成 1 格为主、局部可扩到 2 格的窄走廊。
#       这样粒子不再在大带状区域里长距离直冲后才拐弯，而是被连续墙面逼着频繁转向。
#     - D2/4 保留“碰撞奇偶决定首选左右转”的节奏，但若首选方向被墙堵住，就尝试另一侧；
#       只有左右都走不通时才停。
#   behavior_trace:
#     - 设标记从左边沿向右进入走廊。第一段水平走到头时，前方是墙，走廊只向下开口。
#     - 原题 strict 规则：第一次碰撞首选左转(朝上)；上方不是 path，于是立刻停住。
#     - 本融合规则：同样先尝试朝上；失败后再试朝下，成功进入下一段竖走廊。
#     - 之后粒子沿蛇形走廊继续：有些拐角恰好命中首选方向，有些拐角只能靠后备方向续命。
#       输出因此出现更长的循环涂色轨迹，以及多个终止色“撞墙角标”。
#   complexity_calibration:
#     - 走廊仍只有一种显式材质(path_color)，没有额外控制符号；人类只需读懂“模板给颜色循环，
#       标记沿走廊前进，撞墙后按左右优先级转弯”这一条主线。
#     - 生成器固定把第一拐角做成“只能向下”，确保 fallback 可见；同时再加入 2-4 个后续拐角，
#       让样例不退化成一次偶然补救。
#     - 局部 2 格加宽只用于让“窄走廊”肉眼可见，不改变主通路的拓扑。
#   implicit_assumptions_review:
#     - 原 solver 依赖“g[0,0] 是模板背景、path_color 是最大非背景颜色、模板块在左上窗口内”；
#       新生成器继续显式满足这些条件。
#     - 原 solver 的 visited 只记录碰撞奇偶而不是精确碰撞次数；这里仍成立，因为转向首选只由奇偶决定。
#     - 原题厚路径常靠几何边界自然防止回环；窄走廊若随意加宽会引入旁支和环路。
#       因此生成器只在长直段内部加宽，并避开端点，保证粒子实际追踪的仍是单条主骨架。
#     - 终止色角标仍画在“前方那个非 path 墙格”上，所以走廊必须与模板区、终止块保持间隔，
#       否则会把显式对象污染成角标；生成器为它们保留独立背景口袋。
#   placement_legality:
#     - 模板仍放在 rows 1-3、cols {1,4,7,10} 的 2x2 背景窗口内。
#     - 标记固定在较低的 col 0 竖直走廊里，且上下邻居都保留 path，满足原题识别器。
#     - 终止块固定在右下背景口袋，与走廊至少隔两格，既不重叠也不被角标覆盖。
#     - 生成后用融合 solver 当 oracle，并与 strict-no-fallback 版本对比，拒绝退化样本。
#   observability:
#     - 人眼能直接看到：输入里是一条蛇形窄走廊；输出里彩色轨迹沿走廊折返前进，
#       而不是在第一处死角停下。strict 对照会短很多，因此 fallback 的存在可从少量样例推断。
#   cleanliness:
#     - 画面中只有一条主走廊、一个标记、一行模板和一个终止块，没有多层重叠对象。
#       局部加宽只发生在直道内部，不与角标、模板、终止块混杂。
#   color_inference:
#     - 颜色循环仍由模板块从左到右公开给出；走廊形状决定运动，颜色 id 大小完全不起语义作用.
#   new_solver_assumptions:
#     - 新 main() 假设 path 更接近单通道拓扑：前方通则直走，前方堵则只在左右 90 度中选。
#       生成器因此避免四向十字路口和可回头岔路，只产生“前进或左右拐”的窄走廊局面。

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
                for rr, cc in ((r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)):
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
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
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
    metrics = {
        "fallback_turns": 0,
        "total_turns": 0,
        "painted_steps": 0,
        "collision_marks": 0,
    }
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

        for _ in range(g.shape[0] * g.shape[1] * 8):
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
                metrics["collision_marks"] += 1

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
    combo_out, combo_metrics, _ = _simulate_particles(*parsed, allow_fallback=True)
    return {
        "fallback_turns": int(combo_metrics["fallback_turns"]),
        "combo_turns": int(combo_metrics["total_turns"]),
        "combo_steps": int(combo_metrics["painted_steps"]),
        "combo_collision_marks": int(combo_metrics["collision_marks"]),
        "strict_steps": int(strict_metrics["painted_steps"]),
        "output_diff_cells": int(np.sum(strict_out != combo_out)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor particles as they thread through a narrow corridor, using alternating turn priority with a backup turn.

    Input structure:
    - The upper-left background pocket holds a row of 2x2 monochrome blocks whose left-to-right colors define a cycle.
    - A single isolated pixel on the left edge matches the first cycle color and sits inside a vertical path segment.
    - The path itself is a thin 1-2 cell corridor, and a separate 2x2 block of another color sits in a distant background pocket.

    Rule:
    - The marker launches a particle that advances through path cells, repainting each visited path cell by stepping forward through the template cycle.
    - Because the path is a narrow corridor, the particle keeps moving straight until the corridor wall blocks its next cell; that blocked wall cell is marked with the end color if it is background.
    - At each such collision, turn priority alternates by hit count: the 1st, 3rd, 5th, ... collisions prefer CCW; the 2nd, 4th, 6th, ... prefer CW.
    - If the preferred 90-degree turn is blocked, try the other 90-degree direction as a backup; stop only when neither side leads back into the corridor.

    Key insight:
    - The template still only controls color order; the new challenge is topological. The path is now a forced corridor, so the important reasoning is which bend survives each wall hit, especially when the scheduled alternating turn fails and the backup side keeps the particle alive.
    """
    parsed = _parse_scene(grid)
    out, _, _ = _simulate_particles(*parsed, allow_fallback=True)
    return out


def _draw_segment(mask: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    """Rasterize an axis-aligned segment inclusive of both endpoints."""
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        lo, hi = sorted((int(c0), int(c1)))
        mask[int(r0), lo : hi + 1] = True
    else:
        lo, hi = sorted((int(r0), int(r1)))
        mask[lo : hi + 1, int(c0)] = True


def _corridor_is_clean(mask: np.ndarray) -> bool:
    """Reject local rooms so the path reads as a narrow 1-2 cell corridor."""
    H, W = mask.shape

    for r in range(H - 2):
        for c in range(W - 2):
            if bool(mask[r : r + 3, c : c + 3].all()):
                return False

    for r in range(H):
        for c in range(W):
            if not bool(mask[r, c]):
                continue
            deg = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and bool(mask[rr, cc]):
                    deg += 1
            if deg >= 4 and c > 1:
                return False

    return True


def _build_corridor(H: int, W: int) -> tuple[np.ndarray, int] | None:
    """Build a single narrow serpentine corridor that guarantees an early fallback turn."""
    start_row_hi = min(H - 12, 9)
    if start_row_hi < 6:
        return None
    start_row = random.randint(6, start_row_hi)
    left_limit = random.randint(2, 4)
    right_limit = W - random.randint(6, 8)
    if right_limit - left_limit < 8:
        return None

    desired_runs = random.randint(3, 5)
    row = start_row
    going_right = True
    points: list[tuple[int, int]] = [(int(start_row), 0)]
    horizontal_runs = 0

    while True:
        target_col = int(right_limit if going_right else left_limit)
        points.append((int(row), int(target_col)))
        horizontal_runs += 1
        if horizontal_runs >= desired_runs:
            break
        step = random.randint(3, 4)
        if row + step > H - 6:
            break
        row += step
        points.append((int(row), int(target_col)))
        going_right = not going_right

    if horizontal_runs < 3:
        return None

    mask = np.zeros((H, W), dtype=bool)
    stem_top = max(0, start_row - 1)
    stem_bot = min(H - 1, start_row + 1)
    mask[stem_top : stem_bot + 1, 0] = True
    mask[stem_top : stem_bot + 1, 1] = True
    for a, b in zip(points[:-1], points[1:]):
        _draw_segment(mask, a, b)

    # At most one gentle width-2 bulge keeps the corridor visibly narrow instead of turning it into a room.
    if random.random() < 0.6:
        candidates: list[tuple[int, int, int]] = []
        for a, b in zip(points[:-1], points[1:]):
            if a[0] != b[0]:
                continue
            r = int(a[0])
            c0, c1 = sorted((int(a[1]), int(b[1])))
            if c1 - c0 < 8:
                continue
            candidates.append((r, c0, c1))

        random.shuffle(candidates)
        for r, c0, c1 in candidates:
            sides = [-1, 1]
            random.shuffle(sides)
            for side in sides:
                rr = int(r + side)
                lo = int(c0 + 2)
                hi = int(c1 - 2)
                if not (0 <= rr < H and lo <= hi):
                    continue
                if bool(mask[rr, lo : hi + 1].any()):
                    continue

                trial = mask.copy()
                trial[rr, lo : hi + 1] = True
                if _corridor_is_clean(trial):
                    mask = trial
                    candidates = []
                    break
            if not candidates:
                break

    if not _corridor_is_clean(mask):
        return None

    return mask, int(start_row)


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    n_blocks = random.randint(2, 4)
    cycle_colors = random.sample(palette[3:], n_blocks)

    built = _build_corridor(H, W)
    if built is None:
        return None
    mask, marker_row = built

    if mask[0, 0]:
        return None
    _, n_path = label_mask(mask, connectivity=4)
    if n_path != 1:
        return None
    if not (mask[marker_row - 1, 0] and mask[marker_row, 0] and mask[marker_row + 1, 0]):
        return None

    template_row = random.choice([1, 2])
    block_cols = [1, 4, 7, 10][:n_blocks]
    if block_cols[-1] + 1 >= W:
        return None
    for bc in block_cols:
        for dr in range(2):
            for dc in range(2):
                rr = int(template_row + dr)
                cc = int(bc + dc)
                if rr >= H or cc >= W or bool(mask[rr, cc]):
                    return None

    er, ec = H - 3, W - 3
    end_ring = mask[max(0, er - 1) : min(H, er + 3), max(0, ec - 1) : min(W, ec + 3)]
    if bool(end_ring.any()):
        return None

    g = np.full((H, W), bg, dtype=int)
    g[mask] = path_color

    for i, bc in enumerate(block_cols):
        g[template_row : template_row + 2, bc : bc + 2] = cycle_colors[i]

    g[er : er + 2, ec : ec + 2] = end_color
    g[marker_row, 0] = cycle_colors[0]

    _, n_path_after = label_mask(g == path_color, connectivity=4)
    if n_path_after != 1:
        return None
    if g[0, 0] != bg:
        return None

    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted([(c, col) for col, c in cnt.items() if int(col) != int(bg)], reverse=True)
    if not non_bg or non_bg[0][1] != path_color:
        return None

    try:
        metrics = _variant_metrics(g)
        _ = main(g)
    except Exception:
        return None

    if metrics["fallback_turns"] < 1:
        return None
    if metrics["combo_turns"] < 4:
        return None
    if metrics["combo_collision_marks"] < 3:
        return None
    if metrics["combo_steps"] <= metrics["strict_steps"] + 8:
        return None
    if metrics["output_diff_cells"] <= 10:
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
