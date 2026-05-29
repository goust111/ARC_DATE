# Variant of db695cfb
# Dimension: D2 — 延伸路径方向规则 (path)
# Change: 垂直对角延伸（perpendicular_extension）——从品红像素出发，沿原对角线旋转90度的两个方向延伸至网格边界 → 同向延伸（平行于主对角线）

import random
import sys

import numpy as np

_SOLVER_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if _SOLVER_UTILS_DIR not in sys.path:
    sys.path.append(_SOLVER_UTILS_DIR)

from arc_utils import get_bg


# Design analysis for the D2 variant
# ----------------------------------
# concrete_behavior:
#   original:
#     - 蓝色端点确定一条 45 度主对角线。
#     - 品红触发像素如果落在这条主对角线上，输出里会从该像素沿“另一种斜率”的两侧打出对角射线。
#   variant:
#     - 主对角线的补全方式保持不变。
#     - 变化只发生在触发后的方向选择：品红像素改为沿主对角线本身的两个反向延伸。
#     - 因为触发点本来就在主对角线上，所以平行延伸会先沿已有蓝线穿过端点，再继续把端点外侧的背景补成品红，得到“蓝色中段 + 品红两端延长”的同斜率结构。
#   trace_example:
#     - 若端点在 (1,1) 与 (4,4)，触发点在 (2,2)，原规则会从 (2,2) 朝 (3,1)/(1,3) 方向发射。
#     - 新规则则朝 (3,3)/(1,1) 方向延伸；(1,1)-(4,4) 这段因为已是主对角线，只在背景处着色，所以真正新增的是端点之外同斜率的部分，例如 (0,0)、(5,5)、(6,6)。
# complexity_calibration:
#   - 若没有触发点，D2 完全不可见，因此生成器必须保证至少一个触发真正产生可见延伸。
#   - 多段主对角线、两种斜率、0-2 个干扰蓝点和 0-4 个离线品红点仍然保留，避免任务被过度简化。
#   - 触发方向仍只有两档（slash/backslash），不会引入新的条件分支爆炸。
# implicit_assumptions_review:
#   - 选 line_color 的“最长同色对角配对”逻辑不依赖 D2，仍然成立。
#   - 原 `draw_segment()` 里记录的是垂直方向；这里必须改成与段方向一致的 `(dr, dc)` / `(-dr, -dc)`。
#   - 平行射线会与当前主对角线共线，旧题里“射线与段天然分离”的前提不再成立；因此 solver 必须继续依赖 `if out[r, c] == bg` 才能避免覆盖蓝线和已有触发像素。
#   - generator 里若直接把整条平行射线都当作“新占位”，会把与当前 path 重合的合法部分误判为冲突；因此合法性检查只对 `ray_cells - path_cells` 做额外碰撞约束。
# placement_legality:
#   - 所有主对角线路径仍然互不重叠，且不穿过既有输出轨迹。
#   - 触发的平行延伸只允许落在背景区域，不允许撞上更早放置的端点、触发器或既有输出-only 轨迹。
#   - 干扰像素仍然只能放在不会被主线或射线覆盖的位置。
# visual_observability:
#   - 生成器拒绝“新规则输出与原垂直规则输出完全相同”的样本，确保 D2 差异可见。
#   - 另外要求至少有一个触发点带来 `path` 之外的平行延伸，否则会退化成只有主线补全。
# cleanliness:
#   - 通过禁止平行延伸撞到既有对象/轨迹，减少一团交叉覆盖导致的阅读噪声。
# color_inference:
#   - 颜色职责不变：蓝色始终是端点/主线，品红始终是触发点/延伸线，没有新增颜色映射负担。


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete same-color diagonal endpoint pairs, then continue through trigger pixels along the same diagonal direction.

    Input structure:
    - The background is the most frequent color. One non-background color marks diagonal endpoint pairs, while differently-colored trigger pixels may lie on those diagonals.

    Rule:
    - Fill each diagonal segment connecting the extreme same-color endpoints.
    - Whenever that filled segment passes through a trigger pixel, extend diagonal rays from the trigger in the segment's own direction (both opposite ways), continuing to the grid boundary without overwriting existing non-background cells.

    Key insight:
    - Trigger pixels no longer turn the beam by 90 degrees; they preserve the diagonal orientation and push color outward along the same slope.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    bg = get_bg(grid)

    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    assert non_bg, "grid must contain at least one non-background pixel"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

    def diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
        r0, c0 = p
        r1, c1 = q
        dr = 1 if r1 > r0 else -1
        dc = 1 if c1 > c0 else -1
        steps = abs(r1 - r0)
        return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]

    best = None  # (dist, color)
    for col, pts in by_color.items():
        if len(pts) < 2:
            continue
        for i in range(len(pts)):
            r0, c0 = pts[i]
            for j in range(i + 1, len(pts)):
                r1, c1 = pts[j]
                dr = r1 - r0
                dc = c1 - c0
                if dr == 0 or abs(dr) != abs(dc):
                    continue
                dist = abs(dr)
                cand = (dist, col)
                if best is None or cand > best:
                    best = cand

    assert best is not None, "at least one diagonal-aligned same-color pair must exist"

    _, line_color = best
    pts = by_color[line_color]

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        parallel_dirs = [(dr, dc), (-dr, -dc)]
        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color:
                key = (r, c, v)
                triggers.setdefault(key, set()).update(parallel_dirs)

    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = tcol
                r += ddr
                c += ddc

    return out


# --------------- generator (c) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_SEGMENT_COUNT = (1, 4)
_SEG_LEN_RANGE = (3, 16)
_TRIGGERS_PER_SEG = (0, 2)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_DIAG_TRIGGERS = (0, 4)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 80
_DISTRACTOR_RETRIES = 50
_GEN_RETRIES = 400


def _compute_ray_cells(
    triggers: list[tuple[int, int]],
    ray_dirs: list[tuple[int, int]],
    H: int,
    W: int,
) -> set[tuple[int, int]]:
    """Compute all cells touched by rays from trigger positions."""
    cells: set[tuple[int, int]] = set()
    for tr, tc in triggers:
        for ddr, ddc in ray_dirs:
            r, c = tr + ddr, tc + ddc
            while 0 <= r < H and 0 <= c < W:
                cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _build_expected(inp: np.ndarray, bg: int, segments) -> np.ndarray:
    """Build the expected output grid using same-slope trigger extensions."""
    H, W = inp.shape
    expected = inp.copy()
    for _, _, trigger_positions, path, ray_dirs in segments:
        for r, c in path:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
        for tr, tc in trigger_positions:
            for ddr, ddc in ray_dirs:
                r, c = tr + ddr, tc + ddc
                while 0 <= r < H and 0 <= c < W:
                    if expected[r, c] == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc
    return expected


def _build_perpendicular_expected(inp: np.ndarray, bg: int, segments) -> np.ndarray:
    """Reference builder for the original rule, used only to reject degenerate samples."""
    H, W = inp.shape
    expected = inp.copy()
    for _, _, trigger_positions, path, _ in segments:
        dr = 1 if path[-1][0] > path[0][0] else -1
        dc = 1 if path[-1][1] > path[0][1] else -1
        perp_dirs = [(dr, -dc), (-dr, dc)]
        for r, c in path:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
        for tr, tc in trigger_positions:
            for ddr, ddc in perp_dirs:
                r, c = tr + ddr, tc + ddc
                while 0 <= r < H and 0 <= c < W:
                    if expected[r, c] == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc
    return expected


def _try_place_segment(
    slope_type: str,
    seg_len: int,
    H: int,
    W: int,
    occupied: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
):
    """Try to place one diagonal segment, returning (path, dc) or None."""
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if H - 1 - seg_len < 0 or W - 1 - seg_len < 0:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(0, W - 1 - seg_len)
            dc = 1
        else:
            if H - 1 - seg_len < 0 or seg_len > W - 1:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(seg_len, W - 1)
            dc = -1

        path = [(r0 + k, c0 + k * dc) for k in range(seg_len + 1)]
        if any(rc in occupied or rc in drawn for rc in path):
            continue
        return path, dc
    return None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    n_segments = random.randint(*_SEGMENT_COUNT)
    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied: set[tuple[int, int]] = set()
    drawn: set[tuple[int, int]] = set()
    segments = []
    has_trigger = False
    has_visible_parallel_extension = False

    for _ in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))

        result = _try_place_segment(slope_type, seg_len, H, W, occupied, drawn)
        if result is None:
            continue
        path, dc = result

        path_interior = path[1:-1]
        n_triggers = random.randint(_TRIGGERS_PER_SEG[0], min(_TRIGGERS_PER_SEG[1], len(path_interior)))
        trigger_positions = random.sample(path_interior, n_triggers) if n_triggers > 0 and path_interior else []

        ray_dirs = [(1, dc), (-1, -dc)]
        ray_cells = _compute_ray_cells(trigger_positions, ray_dirs, H, W)
        path_cells = set(path)
        extra_ray_cells = ray_cells - path_cells

        if any(rc in occupied or rc in drawn for rc in extra_ray_cells):
            continue

        ep0, ep1 = path[0], path[-1]
        segments.append((ep0, ep1, trigger_positions, path, ray_dirs))
        occupied.update([ep0, ep1])
        occupied.update(trigger_positions)
        drawn.update(set(path_interior) - set(trigger_positions))
        drawn.update(ray_cells)

        if trigger_positions:
            has_trigger = True
        if extra_ray_cells:
            has_visible_parallel_extension = True

    if not segments or not has_trigger or not has_visible_parallel_extension:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for ep0, ep1, trigger_positions, _, _ in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for tp in trigger_positions:
            inp[tp] = _TRIGGER_COLOR

    all_line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            if any(abs(r - lr) == abs(c - lc) and r != lr for lr, lc in all_line_positions):
                continue
            inp[r, c] = _LINE_COLOR
            occupied.add((r, c))
            all_line_positions.append((r, c))
            break

    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            break

    expected = _build_expected(inp, bg, segments)
    if np.array_equal(expected, _build_perpendicular_expected(inp, bg, segments)):
        return None

    try:
        result = main(inp)
    except Exception:
        return None

    if np.array_equal(result, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
