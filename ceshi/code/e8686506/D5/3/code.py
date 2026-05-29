# Variant of e8686506
# Dimension: D5 — 同色不相邻约束 (meta)
# Change: 拼合后同色碎片不可4-连通相邻（相同颜色的像素之间不能共享边）... → 特定颜色对不可相邻

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location

"""
design_analysis:
  source_task: none
  concrete_variant:
    summary: >
      保留原题“把散落碎片拼成严格左右镜像矩形”的主干，不再使用“同色碎片彼此不能4邻接”
      作为解题约束，而是改成“某一对特定颜色不能4邻接”。这对颜色由输入左上角固定
      的 2x2 双色图例显式给出：左列是一种颜色，右列是另一种颜色，二者构成禁邻颜色对。
    input_visuals:
      - 左上角保留一个 2x2 图例块，外侧有 1 格背景护城河，避免与散落碎片混淆。
      - 其余非背景像素仍是原题那种散落拼图块；frame 仍以整体相对位置保留在输入中。
      - 输出仍是稠密矩形，严格左右镜像。
    output_visuals:
      - 图例中的两种颜色会在输出内部反复出现。
      - 它们都可以分别接触其他颜色，但二者彼此绝不共享边。
      - 其他不同颜色之间允许正常接触，因此肉眼可见“禁止的是某一对颜色，而不是所有同色/异色规则”。
  behavior_trace:
    original_local_step:
      - 原 solver 在 can_place() 里，只要某个候选单元格旁边已有同色，就拒绝该放置。
      - 这个判断依赖旧设定：同色的不同碎片在最终输出中绝不能贴边。
    variant_local_step:
      - 新 solver 先读图例得到禁邻对 (A,B)，然后在 can_place() 里检查的是：
        当前碎片颜色 col 与已落子邻居颜色 nb 若构成 {A,B}，则拒绝。
      - 同色相邻不再被特殊处理；只有图例指定的颜色对被禁止。
    concrete_example:
      - 若图例是红/绿，则红块可以贴黄块、蓝块，绿块也可以贴黄块、蓝块；
        但任何红-绿边接触都非法。
      - 旧逻辑会错误地忽略红-绿冲突；新逻辑会把这些候选位置删掉，从而继续纯传播装配。
  complexity_calibration:
    - 禁邻对只保留 1 对，避免出现多对颜色关系导致训练样例不足以归纳。
    - 两个禁邻颜色都要求在输出中至少出现两个 4 连通分量，避免图例变成一次性装饰。
    - 输出高度下限抬到 7，使内部块数足够多，禁邻约束在视觉和求解上都能真正参与。
  implicit_assumption_audit:
    - 原 main() 里 color_piece_count -> disallow_touch 的推导完全绑定“同色不能邻接”，此处必须移除。
    - total_area、frame 检测、镜像检测都只依赖真实拼图碎片，因此必须先把图例从输入中剔除，
      否则面积、因子分解、frame bbox 都会被污染。
    - 原 _build_requirements() 与 frame 锁定逻辑不依赖同色禁令，保留即可。
    - 新 main() 新增假设：左上角一定存在 2x2 双色图例且周围一圈背景隔离。generator 必须保证。
  placement_legality:
    - generator 固定把图例放在输入左上角，并预留 3x3 空角，避免 frame 或散件越界/重叠到图例。
    - 其余碎片仍通过 random_free_location() 在保留区外散布，保持不重叠且具 1 格 Chebyshev 间距。
    - 输出内部块先在左半边/中轴分块，再镜像到右半边，天然满足左右对称。
  observability:
    - 人类会看到：每个样例左上角都有一个稳定的双色图例；这两种颜色在输出内部多次出现，
      且彼此永不贴边，但会各自贴着第三种颜色。
    - 这样训练样例能支撑“图例定义了一对禁邻颜色”的规则，而不是依赖颜色 id 排序。
  cleanliness:
    - 图例与拼图本体物理分区，避免叠加混乱。
    - frame、散件、图例三者职责清楚：frame 锁定外轮廓，图例只编码颜色关系，散件负责拼装。
  color_inference:
    - 图例直接复用真实输出颜色，而不是额外映射到隐藏 id。
    - 颜色总数仍控制在 4-6；禁邻颜色之外至少保留一种第三方填充色，给出“允许接触”的对照样本。
"""


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            h = n // w
            out.append((h, w))
    return out


def _norm_pair(a: int, b: int) -> tuple[int, int]:
    a = int(a)
    b = int(b)
    return (a, b) if a <= b else (b, a)


def _read_forbidden_pair(grid: np.ndarray, bg: int) -> tuple[tuple[int, int], np.ndarray]:
    """
    Read the fixed 2x2 legend in the top-left corner and remove it from the grid.

    Legend layout (with a 1-cell moat of background):
      A B .
      A B .
      . . .
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    assert H >= 3 and W >= 3, "legend requires at least a 3x3 input corner"
    a0, a1 = int(g[0, 0]), int(g[1, 0])
    b0, b1 = int(g[0, 1]), int(g[1, 1])
    assert a0 == a1 and b0 == b1, "legend columns must be monochrome vertical dominoes"
    assert a0 != bg and b0 != bg and a0 != b0, "legend must contain two distinct non-background colors"
    moat = [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
    for r, c in moat:
        assert int(g[r, c]) == bg, "legend moat must stay background"
    cleaned = g.copy()
    cleaned[0:2, 0:2] = bg
    return _norm_pair(a0, b0), cleaned


def _build_requirements(
    H: int,
    W: int,
    comps_by_color: dict[int, list[list[tuple[int, int]]]],
) -> np.ndarray:
    """
    Build per-cell required color constraints (req[r,c] == -1 means unconstrained).

    Heuristics supported by training examples:
      - If there is a color with exactly 4 singleton components, those are the four corners.
      - If there are full horizontal bars of length W-2 in the input, use them as the top/bottom edges
        between the corners (ordered by their vertical position in the input).
      - If there is a color with exactly two full vertical line components (width=1) in the input,
        place them on the left/right edges centered vertically.
    """
    req = np.full((H, W), -1, dtype=int)

    corner_colors = [
        col
        for col, comps in comps_by_color.items()
        if len(comps) == 4 and all(len(p) == 1 for p in comps)
    ]
    assert len(corner_colors) <= 1, "corner singleton color must be unique when present"
    corner_color = corner_colors[0] if corner_colors else None
    if corner_color is not None:
        req[0, 0] = corner_color
        req[0, W - 1] = corner_color
        req[H - 1, 0] = corner_color
        req[H - 1, W - 1] = corner_color

    bar_colors: list[tuple[int, int]] = []
    if W >= 3 and corner_color is not None:
        for col, comps in comps_by_color.items():
            for pts in comps:
                r0, c0, r1, c1 = bbox_coords(pts)
                hh, ww = r1 - r0 + 1, c1 - c0 + 1
                if hh == 1 and ww == (W - 2) and len(pts) == ww:
                    bar_colors.append((r0, col))
                    break
        if bar_colors:
            assert len(bar_colors) <= 2, "horizontal edge bar colors must be top/bottom only"
            bar_colors.sort()
            top_col = bar_colors[0][1]
            bot_col = bar_colors[-1][1]
            req[0, 1 : W - 1] = top_col
            req[H - 1, 1 : W - 1] = bot_col

    vertical_line_candidates = []
    for col, comps in comps_by_color.items():
        if len(comps) != 2:
            continue
        sigs = [_piece_sig(pts) for pts in comps]
        (h0, w0, coords0), (h1, w1, coords1) = sigs[0], sigs[1]
        if w0 == 1 and w1 == 1 and h0 == h1 and coords0 == coords1 and h0 > 1:
            vertical_line_candidates.append((col, h0))
    assert len(vertical_line_candidates) <= 1, "paired vertical edge-line color must be unique when present"
    if vertical_line_candidates:
        col, L = vertical_line_candidates[0]
        top = (H - L) // 2
        assert 0 <= top and top + L <= H, "paired vertical edge lines must fit in the output height"
        req[top : top + L, 0] = col
        req[top : top + L, W - 1] = col

    return req


def _assemble(
    types: list[dict],
    H: int,
    W: int,
    req: np.ndarray,
    forbidden_touch_pairs: set[tuple[int, int]] | None = None,
) -> np.ndarray:
    board = np.full((H, W), -1, dtype=int)
    counts = [int(t["cnt"]) for t in types]
    forbidden_touch_pairs = forbidden_touch_pairs or set()

    def mirror_ok(r: int, c: int, col: int) -> bool:
        mc = W - 1 - c
        v = board[r, mc]
        return (v == -1) or (v == col)

    def can_place(ti: int, r0: int, c0: int) -> bool:
        t = types[ti]
        col = int(t["col"])
        is_singleton = len(t["coords"]) == 1
        for dr, dc in t["coords"]:
            r, c = r0 + dr, c0 + dc
            if board[r, c] != -1:
                return False
            if req[r, c] != -1 and int(req[r, c]) != col:
                return False
            if not mirror_ok(r, c, col):
                return False
            if is_singleton and not (r == 0 or r == H - 1 or c == 0 or c == W - 1):
                return False
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < H and 0 <= cc < W and board[rr, cc] != -1:
                    if _norm_pair(col, int(board[rr, cc])) in forbidden_touch_pairs:
                        return False
        return True

    def place(ti: int, r0: int, c0: int) -> None:
        t = types[ti]
        col = int(t["col"])
        for dr, dc in t["coords"]:
            r, c = r0 + dr, c0 + dc
            board[r, c] = col

    def placements_for(ti: int) -> list[tuple[int, int]]:
        t = types[ti]
        return [
            (r0, c0)
            for r0 in range(H - int(t["h"]) + 1)
            for c0 in range(W - int(t["w"]) + 1)
            if can_place(ti, r0, c0)
        ]

    changed = True
    while changed:
        changed = False
        for ti in range(len(types)):
            if counts[ti] <= 0:
                continue
            places = placements_for(ti)
            assert len(places) > 0, "assembly dead end: a piece has no valid placement"
            if len(places) == 1:
                counts[ti] -= 1
                place(ti, places[0][0], places[0][1])
                changed = True
            elif len(places) == counts[ti]:
                for r0, c0 in places:
                    counts[ti] -= 1
                    place(ti, r0, c0)
                changed = True
    assert all(c == 0 for c in counts), "assembly not purely propagation-solvable"
    assert np.array_equal(board, board[:, ::-1]), "assembled board must be LR-symmetric"
    return board


def _solve_cleaned_grid(
    g: np.ndarray,
    bg: int,
    forbidden_touch_pairs: set[tuple[int, int]] | None,
    same_color_rule: bool = False,
) -> np.ndarray:
    comps_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    pieces = []
    color_piece_count: dict[int, int] = {}

    for col in sorted(set(map(int, np.unique(g)))):
        if col == bg:
            continue
        comps = components_mask(g == col, connectivity=4)
        comps_by_color[col] = comps
        color_piece_count[col] = len(comps)
        for pts in comps:
            _, _, coords = _piece_sig(pts)
            pieces.append((col, coords))

    ctr = Counter(pieces)
    types = []
    for (col, coords), cnt in sorted(ctr.items(), key=lambda kv: (-len(kv[0][1]), kv[0][0], kv[0][1])):
        arr = np.array(coords, dtype=int)
        h = int(arr[:, 0].max() + 1)
        w = int(arr[:, 1].max() + 1)
        types.append({"col": int(col), "coords": coords, "h": h, "w": w, "cnt": int(cnt)})

    total_area = sum(int(t["cnt"]) * len(t["coords"]) for t in types)
    assert total_area > 0, "total_area must be positive: grid must contain non-background pieces"

    dims = [(h, w) for (h, w) in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist for the total piece area"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]

    comps8_all = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), pts))

    def _is_lr_symmetric(parts: list[tuple[int, np.ndarray]], h: int, w: int) -> bool:
        all_pts = np.concatenate([p for _, p in parts])
        r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
        bmp = np.zeros((h, w), dtype=bool)
        for _, pts in parts:
            for r, c in pts:
                bmp[int(r) - r0, int(c) - c0] = True
        return np.array_equal(bmp, bmp[:, ::-1])

    frame_parts: list[tuple[int, np.ndarray]] = []
    for i, (c1, p1) in enumerate(comps8_all):
        r0, c0, r1, c1b = bbox_coords(p1)
        if r1 - r0 + 1 == H and c1b - c0 + 1 == W and _is_lr_symmetric([(c1, p1)], H, W):
            frame_parts = [(c1, p1)]
            break
        for j in range(i + 1, len(comps8_all)):
            c2, p2 = comps8_all[j]
            union = np.concatenate([p1, p2])
            ur0, uc0, ur1, uc1 = bbox_coords(union)
            if ur1 - ur0 + 1 == H and uc1 - uc0 + 1 == W and _is_lr_symmetric([(c1, p1), (c2, p2)], H, W):
                frame_parts = [(c1, p1), (c2, p2)]
                break
        if frame_parts:
            break
    assert frame_parts, "input must contain 1-2 LR-symmetric pieces (8-conn) whose union bbox equals output dims"

    req = _build_requirements(H, W, comps_by_color)

    all_frame_pts = np.concatenate([p for _, p in frame_parts])
    fr0, fc0 = int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())
    for col, pts in frame_parts:
        for r, c in pts:
            or_, oc = int(r) - fr0, int(c) - fc0
            if 0 <= or_ < H and 0 <= oc < W:
                req[or_, oc] = col

    all_forbidden = set(forbidden_touch_pairs or set())
    if same_color_rule:
        for col, n in color_piece_count.items():
            if n > 1:
                all_forbidden.add(_norm_pair(col, col))

    sol = _assemble(types, H, W, req, all_forbidden)
    return sol.astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered fragments into a left-right symmetric rectangle, guided by a color-pair legend.

    Input structure:
    - The most common color is background; all other pixels are puzzle material except for a fixed 2x2 legend in the top-left corner.
    - That legend contains two vertical dominoes of different colors and is isolated by background cells.
    - Optional frame hints still appear as in the original family: corner singletons, edge bars, or paired vertical edge lines.

    Rule:
    - Read the top-left legend to obtain the single forbidden color pair, then remove the legend from consideration.
    - Collect the remaining non-background connected components as pieces and infer the output rectangle dimensions from total area.
    - Pack the pieces into a dense rectangle so the result is exactly left-right mirrored and the legend's two colors never 4-touch each other.

    Key insight:
    - The frame pieces lock much of the rectangle, while the legend replaces the old same-color rule with a visible pair-specific adjacency ban.
    - That extra pairwise ban removes the remaining color-swap ambiguities, so the puzzle still resolves by pure propagation.
    """
    raw = np.asarray(grid, dtype=int)
    bg = get_bg(raw)
    forbidden_pair, g = _read_forbidden_pair(raw, bg)
    return _solve_cleaned_grid(g, bg, {forbidden_pair})


# --------------- generator (o5x) ---------------

# === CFG: Output dimensions ===
_AREA_MIN = 40
_AREA_MAX = 45
_WIDTH_REQUIRED = 5
_HEIGHT_MAX = 9

# === CFG: Color and palette ===
_N_COLORS_MIN = 4
_N_COLORS_MAX = 6
_TWO_FRAME_MIN_H = 8
_TWO_FRAME_MIN_COLORS = 5
_TWO_FRAME_PROB = 0.25
_CENTER_CAP_PROB = 0.5

# === CFG: Piece constraints ===
_MAX_PIECE_AREA = 6
_MAX_CHUNK_SIZE = 3
_PIECE_COUNT_MIN = 10
_PIECE_COUNT_MAX = 18
_MIN_FORBID_COMPONENTS = 2
_MIN_NONPAIR_TOUCHES = 4

# === CFG: Canvas / scatter ===
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1
_LEGEND_BLOCK = 3

# === CFG: Retry ===
_GEN_MAX_RETRIES = 800


def _valid_dims() -> list[tuple[int, int]]:
    results = []
    for area in range(_AREA_MIN, _AREA_MAX + 1):
        dims = [(area // w, w) for w in range(1, area + 1) if area % w == 0 and area // w >= w]
        best_gap = min(abs(h - w) for h, w in dims)
        best = [(h, w) for h, w in dims if abs(h - w) == best_gap]
        if len(best) == 1:
            h, w = best[0]
            if w == _WIDTH_REQUIRED and h <= _HEIGHT_MAX:
                results.append((h, w))
    return results


_VALID_DIMS = _valid_dims()
_DIM_WEIGHTS = [h * w for h, w in _VALID_DIMS]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _zigzag_frame(H: int) -> tuple[set[tuple[int, int]], int, int]:
    cells = set()
    start_c = random.randint(0, 1)
    c = start_c
    for r in range(H):
        cells.add((r, c))
        if r == H - 1:
            break
        move = random.choice(["down", "horiz", "diag"])
        if move == "horiz":
            cells.add((r, 1 - c))
            c = 1 - c
        elif move == "diag":
            c = 1 - c
    return cells, start_c, c


def _split_connected_chunk(cells: set[tuple[int, int]], limit: int) -> list[list[tuple[int, int]]]:
    remaining = set(cells)
    chunks: list[list[tuple[int, int]]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        chunk = [start]
        frontier = [start]
        fi = 0
        while fi < len(frontier) and len(chunk) < limit:
            r, c = frontier[fi]
            fi += 1
            nbrs = [(rr, cc) for rr, cc in _neighbors4(r, c) if (rr, cc) in remaining]
            random.shuffle(nbrs)
            for nb in nbrs:
                if len(chunk) >= limit:
                    break
                remaining.remove(nb)
                chunk.append(nb)
                frontier.append(nb)
        chunks.append(sorted(chunk))
    return chunks


def _canonical_chunks(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = grid.shape
    half = W // 2
    left_holes = components_mask(grid[:, : half + 1] == -1, connectivity=4)
    chunks: list[list[tuple[int, int]]] = []
    for pts in left_holes:
        cells = {(int(r), int(c)) for r, c in pts}
        chunks.extend(_split_connected_chunk(cells, _MAX_CHUNK_SIZE))
    return chunks


def _symmetrize_chunk(chunk: list[tuple[int, int]], W: int) -> set[tuple[int, int]]:
    cells = set()
    for r, c in chunk:
        cells.add((int(r), int(c)))
        cells.add((int(r), int(W - 1 - c)))
    return cells


def _chunk_adjacency(full_chunks: list[set[tuple[int, int]]]) -> list[set[int]]:
    where: dict[tuple[int, int], int] = {}
    for i, cells in enumerate(full_chunks):
        for cell in cells:
            where[cell] = i
    adj = [set() for _ in full_chunks]
    for i, cells in enumerate(full_chunks):
        for r, c in cells:
            for nb in _neighbors4(r, c):
                j = where.get(nb)
                if j is not None and j != i:
                    adj[i].add(j)
                    adj[j].add(i)
    return adj


def _assign_chunk_colors(
    chunks: list[list[tuple[int, int]]],
    W: int,
    forbidden_pair: tuple[int, int],
    fill_colors: list[int],
) -> list[int] | None:
    if len(chunks) < 5:
        return None
    a, b = forbidden_pair
    extras = [c for c in fill_colors if c not in forbidden_pair]
    if not extras:
        return None

    full_chunks = [_symmetrize_chunk(chunk, W) for chunk in chunks]
    adjacency = _chunk_adjacency(full_chunks)
    order = sorted(range(len(chunks)), key=lambda i: (-len(adjacency[i]), -len(full_chunks[i]), i))
    assigned = [-1] * len(chunks)
    counts: Counter[int] = Counter()

    def feasible(i: int, col: int) -> bool:
        for j in adjacency[i]:
            other = assigned[j]
            if other == -1:
                continue
            if other == col:
                return False
            if _norm_pair(other, col) == forbidden_pair:
                return False
        return True

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return counts[a] >= 2 and counts[b] >= 2 and any(counts[x] > 0 for x in extras)
        remaining = len(order) - pos
        if counts[a] + remaining < 2 or counts[b] + remaining < 2:
            return False
        if not any(counts[x] > 0 for x in extras) and remaining < 1:
            return False

        i = order[pos]
        cands = [col for col in fill_colors if feasible(i, col)]
        if not cands:
            return False

        extra_used = any(counts[x] > 0 for x in extras)
        random.shuffle(cands)
        cands.sort(
            key=lambda col: (
                0 if (col == a and counts[a] < 2) or (col == b and counts[b] < 2) else 1,
                0 if (col in extras and not extra_used) else 1,
                0 if col in forbidden_pair else 1,
                counts[col],
            )
        )
        for col in cands:
            assigned[i] = col
            counts[col] += 1
            if backtrack(pos + 1):
                return True
            counts[col] -= 1
            assigned[i] = -1
        return False

    return assigned if backtrack(0) else None


def _pair_touch_count(grid: np.ndarray, pair: tuple[int, int]) -> int:
    a, b = pair
    total = 0
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if c + 1 < W and _norm_pair(int(grid[r, c]), int(grid[r, c + 1])) == pair:
                total += 1
            if r + 1 < H and _norm_pair(int(grid[r, c]), int(grid[r + 1, c])) == pair:
                total += 1
    return total


def _other_touch_count(grid: np.ndarray, forbidden_pair: tuple[int, int]) -> int:
    total = 0
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            v = int(grid[r, c])
            if c + 1 < W:
                u = int(grid[r, c + 1])
                if u != v and _norm_pair(u, v) != forbidden_pair:
                    total += 1
            if r + 1 < H:
                u = int(grid[r + 1, c])
                if u != v and _norm_pair(u, v) != forbidden_pair:
                    total += 1
    return total


def _build_output(H: int, W: int) -> tuple[np.ndarray | None, set[int], tuple[int, int] | None]:
    half = W // 2
    n_colors = random.randint(_N_COLORS_MIN, _N_COLORS_MAX)
    palette = list(range(n_colors))
    random.shuffle(palette)

    use_two_frames = (
        H >= _TWO_FRAME_MIN_H
        and n_colors >= _TWO_FRAME_MIN_COLORS
        and random.random() < _TWO_FRAME_PROB
    )
    ci = 0
    frame1 = palette[ci]
    ci += 1
    frame2 = frame1
    if use_two_frames:
        frame2 = palette[ci]
        ci += 1

    remaining = palette[ci:]
    if len(remaining) < 3:
        return None, set(), None
    forbid_a, forbid_b = remaining[0], remaining[1]
    extra_fill = remaining[2:]
    if not extra_fill:
        return None, set(), None
    fill_colors = [forbid_a, forbid_b, extra_fill[0]]
    forbidden_pair = _norm_pair(forbid_a, forbid_b)

    frame_half, start_c, end_c = _zigzag_frame(H)
    split = H // 2 if frame1 != frame2 else H
    grid = np.full((H, W), -1, dtype=int)

    for r, c in frame_half:
        fc = frame1 if r < split else frame2
        grid[r, c] = fc
        grid[r, W - 1 - c] = fc
    if start_c == 1 and random.random() < _CENTER_CAP_PROB:
        grid[0, half] = frame1
    if end_c == 1 and random.random() < _CENTER_CAP_PROB:
        grid[H - 1, half] = frame2

    chunks = _canonical_chunks(grid)
    chunk_colors = _assign_chunk_colors(chunks, W, forbidden_pair, fill_colors)
    if chunk_colors is None:
        return None, set(), None

    for chunk, col in zip(chunks, chunk_colors):
        for r, c in _symmetrize_chunk(chunk, W):
            grid[r, c] = col

    if np.any(grid < 0):
        return None, set(), None
    if _pair_touch_count(grid, forbidden_pair) != 0:
        return None, set(), None
    if _other_touch_count(grid, forbidden_pair) < _MIN_NONPAIR_TOUCHES:
        return None, set(), None
    if len(components_mask(grid == forbid_a, connectivity=4)) < _MIN_FORBID_COMPONENTS:
        return None, set(), None
    if len(components_mask(grid == forbid_b, connectivity=4)) < _MIN_FORBID_COMPONENTS:
        return None, set(), None

    frame_set = {frame1}
    if frame2 != frame1:
        frame_set.add(frame2)
    return grid, frame_set, forbidden_pair


def _make_output() -> tuple[np.ndarray | None, int, set[int], tuple[int, int] | None]:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    out, abstract_fcs, abstract_pair = _build_output(H, W)
    if out is None or abstract_pair is None:
        return None, 0, set(), None
    bg = random.randint(0, 9)
    abstract = sorted(np.unique(out))
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < len(abstract):
        return None, 0, set(), None
    cmap = {int(a): available[i] for i, a in enumerate(abstract)}
    real_fcs = {cmap[a] for a in abstract_fcs}
    real_pair = _norm_pair(cmap[abstract_pair[0]], cmap[abstract_pair[1]])
    mapped = np.vectorize(lambda x: cmap[int(x)])(out)
    return mapped.astype(int), bg, real_fcs, real_pair


def _extract_non_frame_patches(out: np.ndarray, bg: int, fcs: set[int]) -> list[np.ndarray]:
    patches = []
    for col in sorted(map(int, np.unique(out))):
        if col in fcs:
            continue
        for pts in components_mask(out == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in pts:
                patch[int(r) - r0, int(c) - c0] = col
            patches.append(patch)
    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), random.random()))
    return patches


def _place_legend(canvas: np.ndarray, bg: int, forbidden_pair: tuple[int, int]) -> None:
    canvas[:_LEGEND_BLOCK, :_LEGEND_BLOCK] = bg
    a, b = forbidden_pair
    canvas[0:2, 0] = a
    canvas[0:2, 1] = b


def _scatter(out: np.ndarray, bg: int, fcs: set[int], forbidden_pair: tuple[int, int]) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape

    canvas[:_LEGEND_BLOCK, :_LEGEND_BLOCK] = -1

    min_off_r = _LEGEND_BLOCK
    min_off_c = _LEGEND_BLOCK
    max_r = rows - oh - 1
    max_c = cols - ow - 1
    if max_r < min_off_r or max_c < min_off_c:
        raise ValueError
    off_r = random.randint(min_off_r, max_r)
    off_c = random.randint(min_off_c, max_c)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1

    for patch in _extract_non_frame_patches(out, bg, fcs):
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    frame_mask = np.zeros((oh, ow), dtype=bool)
    for col in fcs:
        frame_mask |= out == col
    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    bbox_region[frame_mask] = out[frame_mask]

    canvas[canvas == -1] = bg
    _place_legend(canvas, bg, forbidden_pair)
    return canvas


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _gen_attempt() -> np.ndarray | None:
    out, bg, fcs, forbidden_pair = _make_output()
    if out is None or forbidden_pair is None:
        return None
    if not np.array_equal(out, out[:, ::-1]):
        return None
    if not (_N_COLORS_MIN <= len(np.unique(out)) <= _N_COLORS_MAX):
        return None
    pc = _piece_count(out)
    if not (_PIECE_COUNT_MIN <= pc <= _PIECE_COUNT_MAX):
        return None
    if any(len(pts) > _MAX_PIECE_AREA for col in np.unique(out) for pts in components_mask(out == col, connectivity=4)):
        return None
    if _pair_touch_count(out, forbidden_pair) != 0:
        return None
    if _other_touch_count(out, forbidden_pair) < _MIN_NONPAIR_TOUCHES:
        return None
    try:
        inp = _scatter(out, bg, fcs, forbidden_pair)
        got = main(inp)
    except Exception:
        return None
    if np.array_equal(got, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 D5 input")
