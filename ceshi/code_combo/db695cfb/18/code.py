# Combo variant of db695cfb (index 18)
# combo_spec: D1/1+D7/2
# Name: L形折线路径+障碍阻断
#
# Design analysis
# - concrete behavior:
#   原题会把同对角线的一对蓝端点直接补成整条 45 度对角线；D1/1 把这一步改成固定方向的 L 形最短路。
#   这里保留该 L 路径，规则固定为：先取较上方端点，先沿其所在行水平走到另一端点的列，再沿该列竖直到较下方端点。
#   D7/2 叠加到这条有序 L 路径上：当遍历首次碰到路径上的品红像素时，蓝色主路径在该格停止，不再继续补后半段；
#   同时该品红仍向与原端点对隐含对角线垂直的两条对角方向发射射线。
#   例：端点 (2,2) 与 (6,6)，品红在 (4,6)。原题会补对角线 (3,3)(4,4)(5,5)；
#   D1 单变体会补 L 路径 (2,2)(2,3)(2,4)(2,5)(2,6)(3,6)(4,6)(5,6)(6,6) 并继续越过品红；
#   本融合版则只补到 (3,6) 为止，(5,6) 保持背景、(6,6) 仍是原始蓝端点，于是既能看见 L 形拐角，也能看见被品红截断的空尾巴。
# - complexity calibration:
#   如果把“阻断”弱化成“只是不覆盖品红、但继续补后段”，输出会退化成 D1 单变体，因此这里必须采用“首个障碍即停止”的强阻断。
#   同时把真实障碍限制在 L 路径的竖直腿内部，而不是横腿或拐角，可稳定露出“先横后竖”的选择规则，避免 2-3 个样例不足以看出折线方向。
#   仍允许 1-4 条真实段、两种对角家族和少量干扰点，保持和原题接近的配对难度。
# - hidden assumptions in the original solver:
#   原 `diag_path()` 假设主路径是单一斜率；融合后必须换成 `_l_path()`。
#   原 `draw_segment()` 默认整条路径都可继续扫描且触发器透明；这里改为沿 L 路径顺序扫描，遇到首个非线色像素就记录射线并 `break`。
#   射线方向仍由蓝端点对所属的原始对角家族决定，而不是由 L 路径的局部水平/竖直方向决定；否则会同时改变另一维规则。
#   分组逻辑 `groups_a/groups_b` 仍能用，但前提是每条真实蓝端点对在所有蓝点中都可唯一识别；生成器通过禁止额外蓝点与真实端点共享任意对角线来保证。
# - placement legality:
#   生成器按 output-first 放置。每条真实段先确定端点，再得到完整 L 路径、可见障碍候选与对应射线。
#   路径、射线、端点、触发器、干扰点都不得重叠；真实段还会保留整条 L 走廊为 `reserved`，这样不会有干扰点落进“被阻断后的空白尾段”。
#   障碍射线不得碰到已有结构，也不得穿回自己的 L 路径。
# - observability / cleanliness:
#   真实障碍强制落在竖直腿中段，保证输出里能同时看到完整横臂、部分竖臂、障碍处的两条对角射线，以及障碍后的空白尾段。
#   这让人类可从少量样例同时归纳 D1 的折线路径和 D7 的阻断效应；若障碍只放在横腿前部，则拐角信息会被遮蔽，因此不采用。
#   干扰品红点放在任何真实路径/射线之外，且避免与真实障碍共享对角线，减少把干扰误认成有效分裂器的风险。
# - color inference:
#   颜色职能不变：蓝色仍标记待连接端点，品红仍是触发/分裂对象。新增的只是“品红会让蓝色主路径提前终止”。
# - new solver assumptions checked against generator:
#   solver 假设真实障碍至多 1 个且位于竖直腿内部；生成器只生成这样的真实障碍。
#   solver 假设若一条真实段被障碍截断，则障碍后至少留有 1 个背景尾格再到下端点；生成器从 `path[corner+1:-2]` 中选障碍位置来保证可见空尾。
#   solver 假设所有真实蓝端点对在对角分组中无歧义，且射线无需覆盖非背景像素；生成器分别用 `_shares_diagonal` 与 `reserved` 检查来保证。

import random

import numpy as np

from arc_utils import get_bg


def _ordered_endpoints(p: tuple[int, int], q: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    """Order endpoints by row, then by column."""
    return (p, q) if p < q else (q, p)


def _l_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    """Top-row-first shortest L path between diagonal endpoints."""
    (r0, c0), (r1, c1) = _ordered_endpoints(p, q)
    assert r0 != r1 and c0 != c1

    hstep = 1 if c1 > c0 else -1
    path = [(r0, c) for c in range(c0, c1 + hstep, hstep)]
    path.extend((r, c1) for r in range(r0 + 1, r1 + 1))
    return path


def _perp_dirs(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    """Perpendicular diagonal ray directions implied by the original endpoint diagonal."""
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


def _shares_diagonal(cell: tuple[int, int], positions: list[tuple[int, int]] | set[tuple[int, int]]) -> bool:
    r, c = cell
    for rr, cc in positions:
        if r != rr and abs(r - rr) == abs(c - cc):
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each blue diagonal pair with a fixed L route, but let the first touched obstacle cut that route short and branch rays.

    Input structure:
    - The background is the most frequent color. One non-background color forms endpoint pairs that still lie on shared 45-degree diagonals, and magenta-like obstacle pixels may sit on the L route implied by those endpoints.

    Rule:
    - For every detected endpoint pair, sort the endpoints from top to bottom and draw the shortest L path that goes horizontally across the upper row first, then vertically down the destination column.
    - While tracing that ordered L path, stop at the first non-line-color pixel instead of crossing it; keep that obstacle in place and emit the two perpendicular diagonal rays associated with the pair's original diagonal family, filling only background cells.

    Key insight:
    - The diagonal pair still tells you which two blue points belong together, but completion now uses a fixed bent route, and magenta is no longer transparent on that route: it marks both the split point for rays and the cut point of the blue path.
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
                break

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


# --------------- generator (combo) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_SEGMENT_COUNT = (1, 4)
_SEG_LEN_RANGE = (3, 16)
_TRIGGERS_PER_SEG = (0, 1)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_PATH_TRIGGERS = (0, 4)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 120
_DISTRACTOR_RETRIES = 80
_GEN_RETRIES = 500


def _compute_line_cells(path: list[tuple[int, int]], trigger_positions: list[tuple[int, int]]) -> set[tuple[int, int]]:
    trigger_set = set(trigger_positions)
    cells: set[tuple[int, int]] = set()
    for r, c in path:
        if (r, c) in trigger_set:
            break
        cells.add((r, c))
    return cells


def _build_expected(inp: np.ndarray, bg: int, segments) -> np.ndarray:
    H, W = inp.shape
    expected = inp.copy()
    for _, _, trigger_positions, path, perp_dirs in segments:
        for r, c in _compute_line_cells(path, trigger_positions):
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
    reserved: set[tuple[int, int]],
    used_backslash: set[int],
    used_slash: set[int],
    line_positions: list[tuple[int, int]],
    trigger_positions_all: list[tuple[int, int]],
):
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

        if _shares_diagonal(ep0, line_positions) or _shares_diagonal(ep1, line_positions):
            continue

        path = _l_path(ep0, ep1)
        path_set = set(path)
        if any(rc in occupied or rc in reserved for rc in path):
            continue

        perp_dirs = _perp_dirs(ep0, ep1)
        (_, c_top), (_, c_bot) = _ordered_endpoints(ep0, ep1)
        corner_idx = abs(c_bot - c_top)
        candidate_info = []
        for cell in path[corner_idx + 1 : -2]:
            if _shares_diagonal(cell, trigger_positions_all):
                continue
            ray_cells = _ray_cells_from_trigger(cell, perp_dirs, H, W)
            if ray_cells & (occupied | reserved):
                continue
            if ray_cells & (path_set - {cell}):
                continue
            candidate_info.append((cell, ray_cells))

        return ep0, ep1, path, perp_dirs, diag_id, candidate_info
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
    reserved: set[tuple[int, int]] = set()
    used_backslash: set[int] = set()
    used_slash: set[int] = set()
    line_positions: list[tuple[int, int]] = []
    trigger_positions_all: list[tuple[int, int]] = []
    segments = []
    has_visible_block = False

    for _ in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))
        result = _try_place_segment(
            slope_type,
            seg_len,
            H,
            W,
            occupied,
            reserved,
            used_backslash,
            used_slash,
            line_positions,
            trigger_positions_all,
        )
        if result is None:
            continue

        ep0, ep1, path, perp_dirs, diag_id, candidate_info = result
        trigger_positions: list[tuple[int, int]] = []
        ray_cells: set[tuple[int, int]] = set()
        if candidate_info and (not has_visible_block or random.random() < 0.55):
            trigger, ray_cells = random.choice(candidate_info)
            trigger_positions = [trigger]

        segments.append((ep0, ep1, trigger_positions, path, perp_dirs))
        occupied.update([ep0, ep1])
        occupied.update(trigger_positions)
        reserved.update(path)
        reserved.update(ray_cells)
        line_positions.extend([ep0, ep1])
        trigger_positions_all.extend(trigger_positions)

        if slope_type == "backslash":
            used_backslash.add(diag_id)
        else:
            used_slash.add(diag_id)
        if trigger_positions:
            has_visible_block = True

    if not segments or not has_visible_block:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for ep0, ep1, trigger_positions, _, _ in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for tp in trigger_positions:
            inp[tp] = _TRIGGER_COLOR

    all_line_positions = list(line_positions)
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            if _shares_diagonal((r, c), all_line_positions):
                continue
            inp[r, c] = _LINE_COLOR
            occupied.add((r, c))
            all_line_positions.append((r, c))
            break

    all_trigger_positions = list(trigger_positions_all)
    for _ in range(random.randint(*_OFF_PATH_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            if _shares_diagonal((r, c), all_trigger_positions):
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            all_trigger_positions.append((r, c))
            break

    if np.count_nonzero(inp != bg) > 18:
        return None

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
