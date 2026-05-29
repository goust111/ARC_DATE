# Combo variant of e8686506 (index 48)
# combo_spec: D5/3+D13/1
# Name: 背景色变化+特定颜色对禁邻

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location

"""
design_analysis:
  source_task: e8686506
  combo_spec: D5/3 + D13/1
  summary: >
    保留原题“把散落碎片拼成严格左右镜像矩形”的骨架，同时叠加两层新要求：
    其一，训练样例之间背景色会变化，因此 solver 不能写死任何背景值，只能按当前样例统计识别；
    其二，最终拼装时不再是“同色不能贴边”，而是由输入左上角 2x2 双色图例指定的一对颜色
    不能 4 邻接。图例本身要从拼图碎片里剔除，避免污染面积、组件和 frame 检测。
  concrete_visualization:
    input:
      - 每个样例左上角固定有一个 2x2 图例：A B / A B，外侧 1 格护城河全部是该样例自己的背景色。
      - 背景色在不同样例间轮换，例如某个样例是 8，另一个是 3，第三个是 0；图例护城河也随之变。
      - 其余非背景像素仍是散落碎片；frame 仍整体保留相对位置，非 frame 碎片散布在别处。
    output:
      - 仍是稠密矩形，严格左右镜像。
      - 图例给出的两种颜色会在输出里出现多次，但永不共享边；它们各自仍可与第三种颜色贴边。
  behavior_trace:
    original:
      - solver 先把“最常见颜色”当背景，提取所有非背景碎片。
      - can_place() 里若某位置会让两个同色碎片 4 邻接，则拒绝该放置。
    combo:
      - solver 仍先按频数识别当前样例背景，但随后必须先读取并移除左上角图例。
      - can_place() 不再检查“同色相邻”；改为检查当前碎片颜色与已放邻居颜色是否构成图例指定禁邻对。
      - 图例护城河必须等于当前样例背景色，因此背景变化会直接参与前处理；若误判背景，图例和碎片分离都会失败。
    small_example: >
      若某样例背景是 3，左上图例为 6/8，则 6 与 8 不能贴边；同一个程序在另一样例里若背景变成 1、
      图例变成 4/9，就要自动改判成 4 与 9 不能贴边。旧规则只会阻止 6-6 或 8-8 接触，无法表达这种样例间变化。
  complexity_calibration:
    - 只保留 1 对禁邻颜色，避免关系网络过密导致 2-4 个样例难以归纳。
    - 两个禁邻颜色都要求至少出现多个 4 连通分量，确保图例不是一次性装饰。
    - 额外要求存在若干“非禁邻异色接触”，让人能观察到“禁止的是这一对，而不是所有异色边接触”。
    - 背景通过跨调用轮换增强可见性，减少连续训练样例恰好同背景而弱化 D13/1 的风险。
  implicit_assumption_audit:
    - 原 main() 把所有非背景像素都当拼图材料；融合后这不成立，图例必须先删掉。
    - 原 main() 由 color_piece_count 推出 same-color 禁邻；这一推导与新规则冲突，必须彻底移除。
    - 输出面积、因子分解、frame bbox、锚点检测都依赖“只统计真实碎片”，所以图例清理必须发生在这些步骤之前。
    - 图例解析假设左上角 2x2 为两列纯色且外圈 3x3 右下角护城河仍是背景；generator 需要稳定满足。
    - frame 检测仍假设存在 1-2 个 8 连通部件，其 union bbox 恰好等于输出矩形；generator 保持原骨架不变。
  placement_legality:
    - generator 先在左上角预留 3x3 区域给图例，frame bbox 只能放在其右下方，避免重叠。
    - 非 frame patch 仍借助 random_free_location() 放到保留区外，并保留 1 格 Chebyshev 间距。
    - 输出先生成，再抽取组件散布，因此不会越界，也不会出现图例与拼图覆盖。
  observability:
    - 多个样例里，背景色明显改变，但左上图例的“护城河=当前背景”模式保持不变，解题者会自然把它识别为背景而非额外物体。
    - 同时，图例中的两色在输出中反复出现却从不贴边，而它们各自会贴第三色，能支撑“特定颜色对禁邻”的归纳。
  cleanliness:
    - 图例、frame、散件三者分区明确，避免左上角提示与主拼图区混叠。
    - 颜色总数控制在 4-6，避免背景变化与禁邻关系一起引入过多颜色自由度。
  inferability:
    - 背景变化不依赖颜色 id，而依赖“当前样例中占绝对多数且包围图例护城河的颜色”这一可见结构。
    - 禁邻对不靠隐藏排序，而靠固定位置图例显式编码；同一对颜色在输出里有重复出现作为统计支撑。
"""


_BG_CYCLE: list[int] = []


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            out.append((n // w, w))
    return out


def _norm_pair(a: int, b: int) -> tuple[int, int]:
    a = int(a)
    b = int(b)
    return (a, b) if a <= b else (b, a)


def _next_background() -> int:
    global _BG_CYCLE
    if not _BG_CYCLE:
        _BG_CYCLE = list(range(10))
        random.shuffle(_BG_CYCLE)
    return int(_BG_CYCLE.pop())


def _read_forbidden_pair(grid: np.ndarray, bg: int) -> tuple[tuple[int, int], np.ndarray]:
    """
    Read the fixed top-left legend and return (forbidden_pair, cleaned_grid).

    Layout, with the current sample background forming the moat:
      A B .
      A B .
      . . .
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    assert H >= 3 and W >= 3, "legend requires a 3x3 corner"
    a0, a1 = int(g[0, 0]), int(g[1, 0])
    b0, b1 = int(g[0, 1]), int(g[1, 1])
    assert a0 == a1 and b0 == b1, "legend columns must be monochrome dominoes"
    assert a0 != bg and b0 != bg and a0 != b0, "legend must encode two distinct non-background colors"
    for r, c in ((0, 2), (1, 2), (2, 0), (2, 1), (2, 2)):
        assert int(g[r, c]) == bg, "legend moat must match this sample's background"
    cleaned = g.copy()
    cleaned[0:2, 0:2] = bg
    return _norm_pair(a0, b0), cleaned


def _build_requirements(
    H: int,
    W: int,
    comps_by_color: dict[int, list[list[tuple[int, int]]]],
) -> np.ndarray:
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
            req[0, 1 : W - 1] = bar_colors[0][1]
            req[H - 1, 1 : W - 1] = bar_colors[-1][1]

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
        assert 0 <= top and top + L <= H, "paired vertical lines must fit the output height"
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
    forbidden_touch_pairs = set(forbidden_touch_pairs or set())

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
            board[r0 + dr, c0 + dc] = col

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
            assert places, "assembly dead end: a piece has no valid placement"
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
    forbidden_touch_pairs: set[tuple[int, int]],
) -> np.ndarray:
    comps_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    pieces: list[tuple[int, tuple[tuple[int, int], ...]]] = []

    for col in sorted(set(map(int, np.unique(g)))):
        if col == bg:
            continue
        comps = components_mask(g == col, connectivity=4)
        comps_by_color[col] = comps
        for pts in comps:
            _, _, coords = _piece_sig(pts)
            pieces.append((col, coords))

    ctr = Counter(pieces)
    types = []
    for (col, coords), cnt in sorted(ctr.items(), key=lambda kv: (-len(kv[0][1]), kv[0][0], kv[0][1])):
        arr = np.array(coords, dtype=int)
        types.append(
            {
                "col": int(col),
                "coords": coords,
                "h": int(arr[:, 0].max() + 1),
                "w": int(arr[:, 1].max() + 1),
                "cnt": int(cnt),
            }
        )

    total_area = sum(int(t["cnt"]) * len(t["coords"]) for t in types)
    assert total_area > 0, "grid must contain non-background pieces"

    dims = [(h, w) for h, w in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]

    comps8_all = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), pts))

    def is_lr_symmetric(parts: list[tuple[int, np.ndarray]], h: int, w: int) -> bool:
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
        if r1 - r0 + 1 == H and c1b - c0 + 1 == W and is_lr_symmetric([(c1, p1)], H, W):
            frame_parts = [(c1, p1)]
            break
        for j in range(i + 1, len(comps8_all)):
            c2, p2 = comps8_all[j]
            union = np.concatenate([p1, p2])
            ur0, uc0, ur1, uc1 = bbox_coords(union)
            if ur1 - ur0 + 1 == H and uc1 - uc0 + 1 == W and is_lr_symmetric([(c1, p1), (c2, p2)], H, W):
                frame_parts = [(c1, p1), (c2, p2)]
                break
        if frame_parts:
            break
    assert frame_parts, "input must contain 1-2 LR-symmetric frame parts spanning the output bbox"

    req = _build_requirements(H, W, comps_by_color)
    all_frame_pts = np.concatenate([p for _, p in frame_parts])
    fr0, fc0 = int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())
    for col, pts in frame_parts:
        for r, c in pts:
            or_, oc = int(r) - fr0, int(c) - fc0
            if 0 <= or_ < H and 0 <= oc < W:
                req[or_, oc] = col

    return _assemble(types, H, W, req, forbidden_touch_pairs).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered fragments into a left-right symmetric rectangle while honoring a sample-specific forbidden color pair.

    Input structure:
    - The most common color of each sample is the background, and that background may differ from one sample to the next.
    - A fixed 2x2 legend sits in the top-left corner, isolated by a 1-cell moat of the current background; its two colors define the forbidden touching pair.
    - All remaining non-background pixels are puzzle fragments, with the usual optional corner/edge anchors and an LR-symmetric frame still embedded among them.

    Rule:
    - Infer the current sample background, read and remove the top-left legend, then treat the remaining connected components as jigsaw pieces.
    - Recover the output rectangle dimensions from total non-legend area and lock any frame or anchor cells exactly as in the base task.
    - Pack the pieces into a dense rectangle so the result is perfectly left-right mirrored and the two legend colors never become 4-neighbors.

    Key insight:
    - The variable background forces robust preprocessing: the legend moat only makes sense if background is inferred from the sample itself.
    - After that cleanup, the frame fixes much of the rectangle and the legend-specified pairwise ban replaces the old same-color ban, leaving a propagation-only assembly.
    """
    raw = np.asarray(grid, dtype=int)
    bg = get_bg(raw)
    forbidden_pair, cleaned = _read_forbidden_pair(raw, bg)
    return _solve_cleaned_grid(cleaned, bg, {forbidden_pair})


# --------------- generator ---------------

_AREA_MIN = 40
_AREA_MAX = 45
_WIDTH_REQUIRED = 5
_HEIGHT_MAX = 9

_N_COLORS_MIN = 4
_N_COLORS_MAX = 6
_TWO_FRAME_MIN_H = 8
_TWO_FRAME_MIN_COLORS = 5
_TWO_FRAME_PROB = 0.25
_CENTER_CAP_PROB = 0.5

_MAX_PIECE_AREA = 6
_MAX_CHUNK_SIZE = 3
_PIECE_COUNT_MIN = 10
_PIECE_COUNT_MAX = 18
_MIN_FORBID_COMPONENTS = 2
_MIN_NONPAIR_TOUCHES = 4

_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1
_LEGEND_BLOCK = 3

_GEN_MAX_RETRIES = 2500


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
    del H
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
    bg = _next_background()
    abstract = sorted(np.unique(out))
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < len(abstract):
        return None, 0, set(), None
    cmap = {int(a): available[i] for i, a in enumerate(abstract)}
    mapped = np.vectorize(lambda x: cmap[int(x)])(out).astype(int)
    real_fcs = {cmap[a] for a in abstract_fcs}
    real_pair = _norm_pair(cmap[abstract_pair[0]], cmap[abstract_pair[1]])
    return mapped, bg, real_fcs, real_pair


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
        raise ValueError("output bbox does not fit outside legend reserve")
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


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        out, bg, fcs, forbidden_pair = _make_output()
        if out is None or forbidden_pair is None:
            continue
        if not np.array_equal(out, out[:, ::-1]):
            continue
        if not (_N_COLORS_MIN <= len(np.unique(out)) <= _N_COLORS_MAX):
            continue
        pc = _piece_count(out)
        if not (_PIECE_COUNT_MIN <= pc <= _PIECE_COUNT_MAX):
            continue
        if any(
            len(pts) > _MAX_PIECE_AREA
            for col in np.unique(out)
            for pts in components_mask(out == col, connectivity=4)
        ):
            continue
        if _pair_touch_count(out, forbidden_pair) != 0:
            continue
        if _other_touch_count(out, forbidden_pair) < _MIN_NONPAIR_TOUCHES:
            continue
        try:
            inp = _scatter(out, bg, fcs, forbidden_pair)
            got = main(inp)
        except Exception:
            continue
        if np.array_equal(got, out):
            return inp
    raise RuntimeError("failed to generate a valid e8686506 combo input")
