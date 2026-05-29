# Variant of db695cfb
# Dimension: D1 — 主路径类型 (path)
# Change: 对角线连接（diagonal_line）——在两个蓝色像素之间绘制严格45度对角线，
#         改为最短路径连接（含拐弯）——在同一对角线上的两个蓝色端点之间绘制固定规则的 L 形最短路。
#
# Design analysis
# - concrete behavior:
#   输入里仍然只有一批蓝色端点和若干品红触发点，端点成对地落在同一条 45° 对角线上。
#   原题会直接补齐那条对角线；本变体改为先找到较上方的端点，再沿它所在的行水平走到另一端点的列，
#   最后沿该列竖直走到较下方的端点。品红像素只有在这条 L 形主路径上时才触发。
# - trace of the changed step:
#   例：端点 (2,2) 与 (5,5)。原题主路径是 (2,2)(3,3)(4,4)(5,5)；
#   现在主路径是 (2,2)(2,3)(2,4)(2,5)(3,5)(4,5)(5,5)。
#   若触发点在 (2,4)，原题完全不会理它；新规则会把它视为路径上的触发点并发射对角射线。
# - complexity calibration:
#   仍保留“端点必须共对角线”这一可见结构，让 solver 的配对复杂度与原题同级，不引入多对端点的自由配对。
#   真正变化只在“连接形状”，且 L 形路径的两个候选方向中固定为“先水平后竖直”，肉眼可从多组样例归纳。
# - hidden assumptions in the original solver:
#   `diag_path()` 依赖 |dr|==|dc| 才能逐步同时改行列；变体下主路径不再是单一斜率，必须改成两段式路径。
#   `draw_segment()` 原来把整段路径的局部方向视为常量；现在路径局部方向变了，但射线规则若也跟着局部方向变，
#   就会同时改动主路径与射线两个维度。因此这里保留原题的“由端点对的隐含对角方向决定射线方向”。
#   `groups_a/groups_b` 的分组假设仍成立，但前提变成“每条可用对角线只对应一对蓝色端点”；生成器显式保证这一点。
# - placement legality:
#   生成器按输出优先放置完整 L 路径；每条路径、其触发点射线、以及额外干扰像素都不得碰撞已有输入像素或既有输出轨迹。
#   同时禁止新增蓝色干扰点与任一蓝点共对角线，避免 solver 误配对。
# - observability / cleanliness:
#   端点仍共对角线，但输出补出的不是对角线而是明显的“上边横臂 + 右/左边竖臂”，和原题肉眼可区分。
#   触发点被限制在折线路径内部，且其射线不与已有路径/射线重叠，从而保持画面整洁。
# - color inference:
#   颜色职能完全沿用原题：蓝色负责主路径，品红负责触发与射线。没有基于颜色 id 的排序或隐式映射。
# - new solver assumptions checked against generator:
#   solver 现在假设每个真实蓝色配对仍可由同对角线分组唯一识别，且端点行列都不同；生成器只放置长度 >= 3 的对角端点对。
#   solver 还假设触发点如果被主路径经过，就不会与别的非背景结构重叠；生成器通过 `occupied/drawn` 检查保证这一点。

import random

import numpy as np

from arc_utils import get_bg


def _ordered_endpoints(p: tuple[int, int], q: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return endpoints ordered by row (then by col)."""
    return (p, q) if p < q else (q, p)


def _l_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    """
    Shortest L path used by the variant:
    sort endpoints by row, go horizontally on the upper row, then vertically.
    """
    (r0, c0), (r1, c1) = _ordered_endpoints(p, q)
    assert r0 != r1 and c0 != c1

    hstep = 1 if c1 > c0 else -1
    path = [(r0, c) for c in range(c0, c1 + hstep, hstep)]
    path.extend((r, c1) for r in range(r0 + 1, r1 + 1))
    return path


def _perp_dirs(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    """
    Keep the original ray orientation logic:
    the endpoint pair still implies one of the two diagonal families.
    """
    (r0, c0), (r1, c1) = _ordered_endpoints(p, q)
    dc = 1 if c1 > c0 else -1
    return [(1, -dc), (-1, dc)]


def _ray_cells_from_trigger(
    trigger: tuple[int, int], dirs: list[tuple[int, int]], H: int, W: int
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    tr, tc = trigger
    for ddr, ddc in dirs:
        r, c = tr + ddr, tc + ddc
        while 0 <= r < H and 0 <= c < W:
            cells.add((r, c))
            r += ddr
            c += ddc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color diagonal endpoint pairs with a bent shortest path, then emit diagonal rays from differently-colored pixels touched by that path.

    Input structure:
    - bg is the most frequent color. One non-bg color forms endpoint pairs that still lie on shared diagonals; another color can appear on the would-be completed path as trigger pixels.

    Rule:
    - For each detected endpoint pair, sort the two endpoints by row and draw the shortest L path that goes horizontally across the upper row, then vertically to the lower endpoint.
    - Whenever that path passes through a non-line-color pixel, keep it and emit the same two diagonal rays the original diagonal family would have produced, filling only background cells.

    Key insight:
    - The blue pair still tells you which two points belong together, but the completion shape is no longer the straight diagonal; the bend location is fixed by a "top row first" convention.
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
                cand = (abs(dr), col)
                if best is None or cand > best:
                    best = cand

    assert best is not None, "at least one diagonal-aligned same-color pair must exist"

    _, line_color = best
    pts = by_color[line_color]

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    groups_a: dict[int, list[tuple[int, int]]] = {}  # r - c constant
    groups_b: dict[int, list[tuple[int, int]]] = {}  # r + c constant
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = _l_path(p, q)
        perp_dirs = _perp_dirs(p, q)
        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color:
                triggers.setdefault((r, c, v), set()).update(perp_dirs)

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


# --------------- generator (variant) ---------------

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
_OFF_PATH_TRIGGERS = (0, 4)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 100
_DISTRACTOR_RETRIES = 50
_GEN_RETRIES = 400


def _compute_ray_cells(triggers, perp_dirs, H, W):
    """Compute all cells touched by diagonal rays from trigger positions."""
    cells = set()
    for tr in triggers:
        cells.update(_ray_cells_from_trigger(tr, perp_dirs, H, W))
    return cells


def _build_expected(inp, bg, segments):
    """Build the expected output by drawing L paths and then emitting rays."""
    H, W = inp.shape
    expected = inp.copy()
    for _, _, trigger_positions, path, perp_dirs in segments:
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
    used_backslash: set[int],
    used_slash: set[int],
):
    """Try to place one endpoint pair and return its path info."""
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if H - 1 - seg_len < 0 or W - 1 - seg_len < 0:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(0, W - 1 - seg_len)
            ep0 = (r0, c0)
            ep1 = (r0 + seg_len, c0 + seg_len)
            diag_id = r0 - c0
            if diag_id in used_backslash:
                continue
        else:
            if H - 1 - seg_len < 0 or seg_len > W - 1:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(seg_len, W - 1)
            ep0 = (r0, c0)
            ep1 = (r0 + seg_len, c0 - seg_len)
            diag_id = r0 + c0
            if diag_id in used_slash:
                continue

        path = _l_path(ep0, ep1)
        if any(rc in occupied or rc in drawn for rc in path):
            continue

        perp_dirs = _perp_dirs(ep0, ep1)
        path_set = set(path)
        candidate_info = []
        for cell in path[1:-1]:
            ray_cells = _ray_cells_from_trigger(cell, perp_dirs, H, W)
            if ray_cells & (occupied | drawn):
                continue
            if ray_cells & (path_set - {cell}):
                continue
            candidate_info.append((cell, ray_cells))

        return ep0, ep1, path, perp_dirs, diag_id, candidate_info
    return None


def _choose_triggers(
    candidate_info: list[tuple[tuple[int, int], set[tuple[int, int]]]]
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    """Pick 0-2 non-conflicting trigger positions on the L path."""
    if not candidate_info:
        return [], set()

    random.shuffle(candidate_info)
    max_triggers = min(_TRIGGERS_PER_SEG[1], len(candidate_info))
    target = random.randint(_TRIGGERS_PER_SEG[0], max_triggers)

    chosen_positions: list[tuple[int, int]] = []
    chosen_rays: set[tuple[int, int]] = set()
    for cell, ray_cells in candidate_info:
        if len(chosen_positions) >= target:
            break
        if cell in chosen_rays:
            continue
        if ray_cells & chosen_rays:
            continue
        chosen_positions.append(cell)
        chosen_rays.update(ray_cells)

    return chosen_positions, chosen_rays


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    n_segments = random.randint(*_SEGMENT_COUNT)
    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied: set[tuple[int, int]] = set()  # input pixels
    drawn: set[tuple[int, int]] = set()     # output-only cells
    used_backslash: set[int] = set()
    used_slash: set[int] = set()
    segments = []

    for _ in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))

        result = _try_place_segment(
            slope_type, seg_len, H, W, occupied, drawn, used_backslash, used_slash
        )
        if result is None:
            continue

        ep0, ep1, path, perp_dirs, diag_id, candidate_info = result
        trigger_positions, ray_cells = _choose_triggers(candidate_info)

        segments.append((ep0, ep1, trigger_positions, path, perp_dirs))
        occupied.update([ep0, ep1])
        occupied.update(trigger_positions)
        drawn.update(set(path) - {ep0, ep1} - set(trigger_positions))
        drawn.update(ray_cells)

        if slope_type == "backslash":
            used_backslash.add(diag_id)
        else:
            used_slash.add(diag_id)

    if not segments:
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

    for _ in range(random.randint(*_OFF_PATH_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            break

    try:
        result = main(inp)
    except Exception:
        return None

    expected = _build_expected(inp, bg, segments)
    if np.array_equal(result, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
