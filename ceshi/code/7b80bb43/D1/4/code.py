# Variant of 7b80bb43
# Dimension: D1 — 间隙填充变换方式 (transform)
# Change: 删除对角提示像素后，在提示相邻处填补水平/垂直直线上的3像素间隙，恢复断裂的线段
#   → 提示链方向决定间隙填充方向；对角链向下左延伸时只填左半部分，向下右延伸时只填右半部分。
#
# Design analysis
# ===============
# variant_spec:
#   input_structure:
#     - 背景色 bg + 单一前景色 fg。
#     - 输出基底仍然是由水平/竖直线段组成的网络，含 2-3 个 T 形连接。
#     - 输入里仍有 2-3 条 45 度对角提示链，以及 0-2 条纯噪声对角链。
#   core_change:
#     - 原题: 删除提示链后，把提示旁边的 3 像素断口整段补满。
#     - 变体: 只在水平断口上做“半恢复”。
#       * 若提示链整体是 down-left（从上到下往左走），只补 gap 的左 2 格，保留最右 1 格为空。
#       * 若提示链整体是 down-right（从上到下往右走），只补 gap 的右 2 格，保留最左 1 格为空。
#   concretized_example:
#     - clean line:      .. F F F F F F F ..
#     - input (down-right hint):
#         row r:         .. F . . . F F F ..
#         rows r+1/r+2:      F
#                            . F
#       output 应变成:   .. F . F F F F F ..
#       差异点: 旧题会得到 .. F F F F F F F ..，变体只补右边两格。
#   scope_adjustment:
#     - “左半/右半”只在水平 gap 上有清晰、可见、无歧义的像素解释。
#     - 因此 generator 仍保留竖直线段作为结构背景，但真实需要解码的 gap 只放在水平线段上。
#       这是为了让 D1 的变化在 2-4 个样例中肉眼可见，而不是引入一套额外的竖直“类比规则”。
#   complexity_calibration:
#     - 每个样例仍放 2-3 个真实 gap，且尽量同时包含 left/right 两种提示方向，避免规则退化成单一情况。
#     - 保留 0-2 条 distractor，对 solver 仍有“删噪声 + 局部修剪”的需求。
#   hidden_assumptions_review:
#     - 原 solver 的关键旧假设是:
#       * 只要某条短水平/竖直 gap 靠近任意 diag，就可以整段填满。
#       * gap 的完整恢复不需要区分是哪一条提示链触发。
#     - 这些在变体中都失效，因为:
#       * 同样是 3 像素 gap，down-left 和 down-right 的目标输出不同。
#       * 无条件整段赋值会覆盖掉本应保留的 1 格空洞。
#     - 新 solver 改为“按 diag 组件逐个恢复”: 先找每条提示链唯一对应的根像素，再根据链斜率决定补哪两格。
#   placement_legality:
#     - gap 仍固定为 3 格，链长仍固定为 2。
#     - gap 不能靠近 T 连接点或线段端点；链不能越界、不能与前景正交接触、不能互相重叠。
#     - 多个 gap 允许在同一水平线段上，但其 gap 区域和链邻域都会被额外保留，避免视觉粘连。
#   observability:
#     - target output 故意保留 1 格空洞，使“只补左半 / 右半”直接可见。
#     - 每个样例优先同时包含左右两种方向，解题者可以在单个样例内部对照。
#   cleanliness:
#     - 输出仍是轴对齐线段网络；只有被提示的位置留下 1 格规则性缺口。
#     - distractor 继续远离主结构，避免和真实 gap 混淆。
#   new_solver_assumptions_to_guarantee:
#     - 每条真实提示链恰好有 1 个对角相邻的线段根像素。
#     - 真实提示链只服务于水平 gap。
#     - 根像素同一行的另一侧在 gap 外仍有前景，且 3 格 gap 完整为空。
#     - generator 明确构造这些条件，并用 oracle `main(input) == target_output` 反查。

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, draw_line, find_components, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove diagonal staircase noise, then perform direction-sensitive partial gap filling.

    Real hint chains only sit beside horizontal 3-cell gaps:
    - down-left chain  -> fill the left  two cells of the gap
    - down-right chain -> fill the right two cells of the gap

    Distractor chains are still deleted and may trigger local leaf pruning near their old sites.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    colors = {int(x) for x in np.unique(g)}
    colors.discard(int(bg))
    assert colors
    fg = int(next(iter(colors)))

    mask = g == fg
    H, W = g.shape

    def deg4(m: np.ndarray, r: int, c: int) -> int:
        d = 0
        if r > 0 and bool(m[r - 1, c]):
            d += 1
        if r + 1 < H and bool(m[r + 1, c]):
            d += 1
        if c > 0 and bool(m[r, c - 1]):
            d += 1
        if c + 1 < W and bool(m[r, c + 1]):
            d += 1
        return d

    def has_diag(m: np.ndarray, r: int, c: int) -> bool:
        if r > 0 and c > 0 and bool(m[r - 1, c - 1]):
            return True
        if r > 0 and c + 1 < W and bool(m[r - 1, c + 1]):
            return True
        if r + 1 < H and c > 0 and bool(m[r + 1, c - 1]):
            return True
        if r + 1 < H and c + 1 < W and bool(m[r + 1, c + 1]):
            return True
        return False

    def diag_component_slope(coords: np.ndarray) -> int:
        pts = sorted((int(r), int(c)) for r, c in coords)
        return 1 if pts[-1][1] > pts[0][1] else -1

    diag = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if bool(mask[r, c]) and deg4(mask, r, c) == 0 and has_diag(mask, r, c):
                diag[r, c] = True

    near = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if not bool(diag[r, c]):
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        near[rr, cc] = True

    base = mask & (~diag)
    out = np.full((H, W), bg, dtype=int)
    out[base] = fg

    for coords in components_mask(diag, connectivity=8):
        if coords.size == 0:
            continue
        roots: set[tuple[int, int]] = set()
        for r, c in coords:
            r = int(r)
            c = int(c)
            for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and bool(base[rr, cc]):
                    roots.add((rr, cc))
        assert len(coords) + len(roots) <= 3, f"diagonal chain too large ({len(coords)} diag + {len(roots)} root)"
        if len(roots) != 1:
            continue

        root_r, root_c = next(iter(roots))
        slope = diag_component_slope(coords)
        if slope > 0:
            gap_cols = [root_c + 1, root_c + 2, root_c + 3]
            fill_cols = [root_c + 2, root_c + 3]
            far_col = root_c + 4
        else:
            gap_cols = [root_c - 3, root_c - 2, root_c - 1]
            fill_cols = [root_c - 3, root_c - 2]
            far_col = root_c - 4

        if not all(0 <= cc < W for cc in gap_cols + [far_col]):
            continue
        if not bool(base[root_r, far_col]):
            continue
        if any(bool(base[root_r, cc]) for cc in gap_cols):
            continue
        for cc in fill_cols:
            out[root_r, cc] = fg

    m = out == fg
    changed = True
    while changed:
        changed = False
        to_clear: list[tuple[int, int]] = []
        for r in range(H):
            for c in range(W):
                if not bool(m[r, c]) or not bool(near[r, c]):
                    continue
                if deg4(m, r, c) != 1:
                    continue
                if not has_diag(m, r, c):
                    continue
                to_clear.append((r, c))
        if to_clear:
            changed = True
            for r, c in to_clear:
                m[r, c] = False

    out2 = np.full((H, W), bg, dtype=int)
    out2[m] = fg
    return out2


# --------------- generator ---------------

# Grid dimensions
GRID_H_RANGE = (16, 29)
GRID_W_RANGE = (17, 30)
N_COLORS = 10

# Segment network topology (h + v = segments; junctions = T-junction count)
VALID_CONFIGS = [
    {"segments": 6, "h": 2, "v": 4, "junctions": 2},
    {"segments": 6, "h": 3, "v": 3, "junctions": 2},
    {"segments": 6, "h": 3, "v": 3, "junctions": 3},
    {"segments": 7, "h": 2, "v": 5, "junctions": 2},
    {"segments": 7, "h": 3, "v": 4, "junctions": 2},
    {"segments": 7, "h": 4, "v": 3, "junctions": 2},
    {"segments": 7, "h": 3, "v": 4, "junctions": 3},
    {"segments": 7, "h": 4, "v": 3, "junctions": 3},
    {"segments": 8, "h": 3, "v": 5, "junctions": 3},
    {"segments": 8, "h": 4, "v": 4, "junctions": 3},
    {"segments": 8, "h": 5, "v": 3, "junctions": 3},
]
T_SLOTS = ("LT", "RT", "LB", "RB")

# Layout spacing
EDGE_MARGIN = 3
ROW_COL_GAP = 2
T_BRANCH_OFFSET_MIN = 3
T_BRANCH_OFFSET_MAX = 9
H_SEG_END_MARGIN = 4
H_SEG_MAX_END = 25

# Simple H-segment sizing
SIMPLE_H_LEN_MIN = 7
SIMPLE_H_LEN_MAX = 11
SIMPLE_V_MARGIN_NEAR = 7
SIMPLE_V_MARGIN_FAR = 3

# Gap / chain placement
GAP_SIZE = 3
CHAIN_LENGTH = 2
GAP_BLOCKED_LO = 2
GAP_BLOCKED_HI = 4
GAP_SEG_INSET = 2
TARGET_GAP_COUNTS = (2, 3)

# Distractor chains
DISTRACTOR_LENGTHS = (3, 2)
DISTRACTOR_MARGIN = 1

# Retry budget
GEN_RETRIES = 300


def _pick_distinct(rng: random.Random, values: list[int], k: int, gap: int = ROW_COL_GAP) -> list[int]:
    vals = values[:]
    rng.shuffle(vals)
    picked: list[int] = []
    for v in vals:
        if all(abs(v - p) >= gap for p in picked):
            picked.append(v)
            if len(picked) == k:
                return sorted(picked)
    return []


def _chain_coords(row: int, gap_start: int, anchor: str) -> list[tuple[int, int]]:
    if anchor == "left":
        return [(row + i, gap_start - 1 + i) for i in range(1, CHAIN_LENGTH + 1)]
    return [(row + i, gap_start + GAP_SIZE - i) for i in range(1, CHAIN_LENGTH + 1)]


def _cells_are_color(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    h, w = grid.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w) or grid[r, c] != color:
            return False
    return True


def _validate_chain(grid: np.ndarray, coords: list[tuple[int, int]], fg: int,
                    occupied: set[tuple[int, int]]) -> bool:
    h, w = grid.shape
    coord_set = set(coords)
    if any(cell in occupied for cell in coords):
        return False
    for r, c in coords:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in coord_set:
                return False
            if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == fg:
                return False
        has_diag = False
        for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
            if (rr, cc) in coord_set or (0 <= rr < h and 0 <= cc < w and grid[rr, cc] == fg):
                has_diag = True
                break
        if not has_diag:
            return False
    return True


def _place_t_component(out: np.ndarray, fg: int, slot: str, row: int,
                       branch_col: int, end_col: int) -> tuple[dict, dict]:
    H, W = out.shape
    h_c0 = 0 if slot[0] == "L" else end_col
    h_c1 = end_col if slot[0] == "L" else W - 1
    v_r0 = 0 if slot[1] == "T" else row
    v_r1 = row if slot[1] == "T" else H - 1
    draw_line(out, row, h_c0, end_r=row, end_c=h_c1, color=fg)
    draw_line(out, v_r0, branch_col, end_r=v_r1, end_c=branch_col, color=fg)
    h_seg = {"row": row, "c0": h_c0, "c1": h_c1, "blocked": {branch_col}}
    v_seg = {"col": branch_col, "r0": v_r0, "r1": v_r1, "blocked": {row}}
    return h_seg, v_seg


def _place_simple_h(out: np.ndarray, fg: int, side: str, row: int, length: int) -> dict:
    W = out.shape[1]
    c0 = 0 if side == "left" else W - length
    c1 = length - 1 if side == "left" else W - 1
    draw_line(out, row, c0, end_r=row, end_c=c1, color=fg)
    return {"row": row, "c0": c0, "c1": c1, "blocked": set()}


def _place_simple_v(out: np.ndarray, fg: int, side: str, col: int, length: int) -> dict:
    H = out.shape[0]
    r0 = 0 if side == "top" else H - length
    r1 = length - 1 if side == "top" else H - 1
    draw_line(out, r0, col, end_r=r1, end_c=col, color=fg)
    return {"col": col, "r0": r0, "r1": r1, "blocked": set()}


def _h_gap_options(seg: dict, out: np.ndarray, fg: int, bg: int,
                   occupied: set[tuple[int, int]]) -> list[tuple[int, str]]:
    row, c0, c1 = seg["row"], seg["c0"], seg["c1"]
    blocked = seg["blocked"]
    options: list[tuple[int, str]] = []
    for g in range(c0 + GAP_SEG_INSET, c1 - GAP_SIZE - 1):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        temp = out.copy()
        temp[row, g:g + GAP_SIZE] = bg
        for anchor in ("left", "right"):
            coords = _chain_coords(row, g, anchor)
            if not _cells_are_color(temp, coords, bg):
                continue
            if not _validate_chain(temp, coords, fg, occupied):
                continue
            options.append((g, anchor))
    return options


def _reserve_neighborhood(cells: set[tuple[int, int]], coords) -> None:
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cells.add((r + dr, c + dc))


def _distractor_options(inp: np.ndarray, bg: int,
                        forbidden: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    h, w = inp.shape
    options: list[list[tuple[int, int]]] = []
    for length in DISTRACTOR_LENGTHS:
        for r in range(DISTRACTOR_MARGIN, h - length):
            for c in range(DISTRACTOR_MARGIN, w - DISTRACTOR_MARGIN):
                for direction in (-1, 1):
                    coords = [(r + i, c + direction * i) for i in range(length)]
                    if not _cells_are_color(inp, coords, bg):
                        continue
                    if any(cell in forbidden for cell in coords):
                        continue
                    options.append(coords)
    return options


def _output_hole_col(gap_start: int, anchor: str) -> int:
    return gap_start if anchor == "left" else gap_start + GAP_SIZE - 1


def _gap_forbidden_band(row: int, gap_start: int) -> set[tuple[int, int]]:
    return {(row, c) for c in range(gap_start - 1, gap_start + GAP_SIZE + 1)}


def _build_output(rng: random.Random, h: int, w: int, bg: int, fg: int,
                  cfg: dict) -> tuple[np.ndarray, list[dict], list[dict]] | None:
    out = np.full((h, w), bg, dtype=int)
    h_segments: list[dict] = []
    v_segments: list[dict] = []

    h_simple = cfg["h"] - cfg["junctions"]
    v_simple = cfg["v"] - cfg["junctions"]
    t_slots = rng.sample(list(T_SLOTS), cfg["junctions"])

    top_slots = [s for s in t_slots if s[1] == "T"]
    bottom_slots = [s for s in t_slots if s[1] == "B"]
    top_pool = list(range(EDGE_MARGIN, max(EDGE_MARGIN + 1, h // 2 - 2)))
    bottom_pool = list(range(max(h // 2 + 1, EDGE_MARGIN + 2), h - EDGE_MARGIN))
    top_rows = _pick_distinct(rng, top_pool, len(top_slots))
    bottom_rows = _pick_distinct(rng, bottom_pool, len(bottom_slots))
    if len(top_rows) != len(top_slots) or len(bottom_rows) != len(bottom_slots):
        return None
    row_by_slot = dict(zip(top_slots, top_rows)) | dict(zip(bottom_slots, bottom_rows))

    branch_cols = _pick_distinct(rng, list(range(EDGE_MARGIN, w - EDGE_MARGIN)), len(t_slots))
    if len(branch_cols) != len(t_slots):
        return None
    rng.shuffle(branch_cols)
    branch_by_slot = dict(zip(t_slots, branch_cols))

    for slot in t_slots:
        branch_col = branch_by_slot[slot]
        row = row_by_slot[slot]
        if slot[0] == "L":
            min_end = max(SIMPLE_H_LEN_MIN + 1, branch_col + T_BRANCH_OFFSET_MIN)
            max_end = min(w - H_SEG_END_MARGIN, branch_col + T_BRANCH_OFFSET_MAX, H_SEG_MAX_END)
        else:
            min_end = max(2, branch_col - T_BRANCH_OFFSET_MAX, w - H_SEG_MAX_END - 1)
            max_end = min(w - SIMPLE_H_LEN_MIN - 1, branch_col - T_BRANCH_OFFSET_MIN)
        if min_end > max_end:
            return None
        end_col = rng.randint(min_end, max_end)
        h_seg, v_seg = _place_t_component(out, fg, slot, row, branch_col, end_col)
        h_segments.append(h_seg)
        v_segments.append(v_seg)

    center_start = max(top_rows + [1]) + 2
    center_end = min(bottom_rows + [h - 2]) - 2
    if h_simple > 0 and center_start > center_end:
        return None

    simple_h_rows = _pick_distinct(rng, list(range(center_start, center_end + 1)), h_simple) if h_simple else []
    if len(simple_h_rows) != h_simple:
        return None
    if h_simple == 0:
        h_sides: list[str] = []
    elif h_simple == 1:
        h_sides = [rng.choice(["left", "right"])]
    else:
        h_sides = ["left", "right"]
        rng.shuffle(h_sides)

    simple_v_cols: list[int] = []
    used_branch_cols = set(branch_by_slot.values())
    if v_simple:
        has_left = "left" in h_sides
        has_right = "right" in h_sides
        if has_left and has_right:
            col_pool = list(range(SIMPLE_V_MARGIN_NEAR, w - SIMPLE_V_MARGIN_NEAR))
        elif has_left:
            col_pool = list(range(SIMPLE_V_MARGIN_NEAR, w - SIMPLE_V_MARGIN_FAR))
        elif has_right:
            col_pool = list(range(2, w - SIMPLE_V_MARGIN_NEAR))
        else:
            col_pool = list(range(2, w - 2))
        col_pool = [c for c in col_pool if c not in used_branch_cols]
        simple_v_cols = _pick_distinct(rng, col_pool, v_simple)
        if len(simple_v_cols) != v_simple:
            return None

    for row, side in zip(simple_h_rows, h_sides):
        if side == "left":
            max_len = (min(simple_v_cols) if simple_v_cols else w - EDGE_MARGIN) - 1
        else:
            max_len = w - (max(simple_v_cols) if simple_v_cols else 2) - 2
        max_len = min(SIMPLE_H_LEN_MAX, max_len)
        if max_len < SIMPLE_H_LEN_MIN:
            return None
        h_segments.append(_place_simple_h(out, fg, side, row, rng.randint(SIMPLE_H_LEN_MIN, max_len)))

    top_v = min(2, v_simple)
    top_v_cols = simple_v_cols[:top_v]
    bottom_v_cols = simple_v_cols[top_v:]
    top_limit = min(min(top_rows) - 2 if top_rows else center_start - 2, center_start - 2)
    bottom_start = max(max(bottom_rows) + 2 if bottom_rows else center_end + 2, center_end + 2)
    for col in top_v_cols:
        if top_limit < 2:
            return None
        v_segments.append(_place_simple_v(out, fg, "top", col, rng.randint(2, top_limit)))
    for col in bottom_v_cols:
        max_len = h - bottom_start
        if max_len < 2:
            return None
        v_segments.append(_place_simple_v(out, fg, "bottom", col, rng.randint(2, max_len)))

    comps = find_components(out, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    if len(comps) != cfg["segments"] - cfg["junctions"]:
        return None
    return out, h_segments, v_segments


def _choose_h_gaps(rng: random.Random, clean: np.ndarray, h_segments: list[dict],
                   fg: int, bg: int, gap_count: int) -> list[tuple[int, int, str]] | None:
    occupied: set[tuple[int, int]] = set()
    forbidden_gap_band: set[tuple[int, int]] = set()
    chosen: list[tuple[int, int, str]] = []

    required_anchors = ["left", "right"] if gap_count >= 2 else [rng.choice(["left", "right"])]
    for anchor_needed in required_anchors:
        candidates: list[tuple[int, int, str, list[tuple[int, int]]]] = []
        for seg in h_segments:
            row = int(seg["row"])
            for gap_start, anchor in _h_gap_options(seg, clean, fg, bg, occupied):
                if anchor != anchor_needed:
                    continue
                band = _gap_forbidden_band(row, gap_start)
                if any(cell in forbidden_gap_band for cell in band):
                    continue
                candidates.append((row, int(gap_start), anchor, _chain_coords(row, int(gap_start), anchor)))
        if not candidates:
            return None
        row, gap_start, anchor, chain = rng.choice(candidates)
        chosen.append((row, gap_start, anchor))
        occupied.update(chain)
        forbidden_gap_band.update(_gap_forbidden_band(row, gap_start))

    while len(chosen) < gap_count:
        candidates: list[tuple[int, int, str, list[tuple[int, int]]]] = []
        for seg in h_segments:
            row = int(seg["row"])
            for gap_start, anchor in _h_gap_options(seg, clean, fg, bg, occupied):
                band = _gap_forbidden_band(row, gap_start)
                if any(cell in forbidden_gap_band for cell in band):
                    continue
                candidates.append((row, int(gap_start), anchor, _chain_coords(row, int(gap_start), anchor)))
        if not candidates:
            return None
        row, gap_start, anchor, chain = rng.choice(candidates)
        chosen.append((row, gap_start, anchor))
        occupied.update(chain)
        forbidden_gap_band.update(_gap_forbidden_band(row, gap_start))

    return chosen


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.randrange(N_COLORS)
    fg = rng.choice([c for c in range(N_COLORS) if c != bg])
    cfg = rng.choice(VALID_CONFIGS)
    gap_count = rng.choice(TARGET_GAP_COUNTS)
    distractor_count = rng.randint(0, 2)

    built = _build_output(rng, h, w, bg, fg, cfg)
    if built is None:
        return None
    clean, h_segments, _v_segments = built

    chosen = _choose_h_gaps(rng, clean, h_segments, fg, bg, gap_count)
    if chosen is None:
        return None

    out = clean.copy()
    inp = clean.copy()
    occupied: set[tuple[int, int]] = set()
    for row, gap_start, anchor in chosen:
        out[row, _output_hole_col(gap_start, anchor)] = bg
        inp[row, gap_start:gap_start + GAP_SIZE] = bg
        chain = _chain_coords(row, gap_start, anchor)
        for r, c in chain:
            inp[r, c] = fg
        occupied.update(chain)

    forbidden: set[tuple[int, int]] = set()
    _reserve_neighborhood(forbidden, list(map(tuple, np.argwhere(inp == fg))))
    _reserve_neighborhood(forbidden, list(occupied))
    distractors = _distractor_options(inp, bg, forbidden)
    rng.shuffle(distractors)
    placed = 0
    for coords in distractors:
        if placed >= distractor_count:
            break
        if any(cell in forbidden for cell in coords):
            continue
        for r, c in coords:
            inp[r, c] = fg
        _reserve_neighborhood(forbidden, coords)
        placed += 1

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 7b80bb43 D1 variant")
