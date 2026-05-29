# Variant of cbebaa4b
# Dimension: D3 — 目标锚点的结构 (dst)
# Change: 中央黄色方块作为初始锚点，已连接的精灵作为后续连接的扩展锚点
#   → 多个独立锚点：网格上有 2-3 个黄色方块作为独立锚点，精灵会分组后分别连接到各自锚点

# Design analysis for the D3 variant
# ----------------------------------
# variant_spec:
#   input:
#     - 黑底 24-26 x 24-26。
#     - 2-3 个黄色锚点组件，每个组件都含一个黄色矩形块和若干红色连接端口。
#     - 其余彩色精灵被打散；每个精灵仍是单个 4 连通组件，且只有平移、无旋转。
#     - 同一组精灵共用一种非黄非红颜色，作为“这批精灵属于同一锚点簇”的可见提示。
#   output:
#     - 所有黄色锚点保持原位不动。
#     - 每个彩色组只会接到某一个黄色锚点簇上，形成 2-3 个彼此分离的连通结构。
#     - 红色像素仍然两两重合配对；每个输出组件内部连通，但不同锚点簇之间不相连。
# behavior_trace:
#   - 原题：先把唯一黄色锚点放到画布上，DFS 只从这一棵树的未配对红端口继续扩张。
#   - 变体：先把所有黄色锚点都放到画布上，DFS 的初始未配对红端口来自多个锚点。
#     后续每一步仍然是“找一个未配对红端口，用某个散件去覆盖它”，但这个端口可能属于任一锚点簇。
#   - 差异：原题只需要决定“下一个 piece 接到当前整棵树哪里”，变体还需要决定“它属于哪一个锚点簇”。
# complexity_calibration:
#   - 若仍保留原来的 4-6 个散件且再增加到 3 个锚点，输入组件总数会过多，散布布局容易失败。
#   - 因此这里把每个锚点簇的散件数控制在 1-3，总散件数控制在 4-5，并把 piece 尺寸略缩小，
#     同时把画布下界提高到 24，保证 2-3 个独立簇在输入/输出里都看得清且能合法摆放。
# implicit_assumptions_checked:
#   - 原版 `anchors = [...] ; assert len(anchors) == 1` 直接依赖“唯一锚点”；这里改成 2-3 个锚点并全部作为初始种子。
#   - 原版 `red_counts` 只用单个锚点初始化；这里要把所有锚点的红端口都计入初始 frontier。
#   - 原版 `_try_place` 允许 `arr == sub` 的非红重叠，这在“每个 piece 颜色唯一”时基本无害；
#     但变体里同一锚点簇会复用颜色，这会制造伪合法重叠。这里把合法重叠收紧为“只允许红压红”。
# placement_legality:
#   - 生成器仍采用 output-first：先在局部坐标里造好每个锚点簇，再把每个簇当作一个大 patch 摆进全局输出。
#   - 输出簇之间要求留出至少 1 格空隙；输入打散时，所有组件之间仍要求 Chebyshev padding=2。
#   - 所有样本最后都通过 `main(input) == output` 验证，避免 generator 造出 solver 无法唯一求解的输入。
# visual_observability:
#   - 输入里能直接看到“黄色锚点不止一个”，输出里能直接看到“生成多个互不接触的装配簇”。
#   - 每个簇内部彩色一致、簇之间颜色不同，减少视觉噪声，让“哪些精灵是一组”在少量样例里可观察。

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        reds = np.argwhere(arr == 2)
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "reds": [(int(r), int(c)) for r, c in reds],
                "shape": (int(arr.shape[0]), int(arr.shape[1])),
                "area": int(np.sum(arr != bg)),
            }
        )
    return pieces


def _try_place(
    canvas: np.ndarray, arr: np.ndarray, top: int, left: int, bg: int
) -> np.ndarray | None:
    h, w = canvas.shape
    ph, pw = arr.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return None

    sub = canvas[top : top + ph, left : left + pw]
    mask = arr != bg
    if not np.any(mask):
        return None

    # Only red-on-red overlap is legal; all non-red overlaps are conflicts.
    conflict = mask & (sub != bg) & ~((arr == 2) & (sub == 2))
    if np.any(conflict):
        return None

    out = canvas.copy()
    out_sub = out[top : top + ph, left : left + pw]
    out_sub[mask] = arr[mask]
    out[top : top + ph, left : left + pw] = out_sub
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered jigsaw sprites onto several fixed yellow anchors by pairing red connector pixels.

    Input structure:
    - bg = black (0). Several colored sprites are scattered on the grid.
    - There are 2-3 anchor sprites, each containing a yellow (4) block.
    - Every anchor and every non-anchor sprite has red (2) connector pixels along its boundary.

    Rule:
    - All yellow anchors stay where they are; they are simultaneous starting seeds.
    - Each remaining sprite is translated, without rotation, so that its red pixels overlap currently unpaired red pixels on one anchor-led assembly.
    - Every red connector must end up paired exactly once, producing several disconnected finished assemblies, one around each yellow anchor.

    Key insight:
    - Red pixels still behave like jigsaw ports, but now the open ports belong to multiple anchor groups at once.
    - Solving requires both connector matching and inferring which scattered sprites belong to which yellow anchor cluster.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    anchors = [p for p in pieces if np.any(p["arr"] == 4)]
    assert 2 <= len(anchors) <= 3, "input must have 2 or 3 yellow (4) anchor pieces"
    rest = [p for p in pieces if not np.any(p["arr"] == 4)]
    assert rest, "input must contain non-anchor pieces"

    canvas = np.full((h, w), bg, dtype=int)
    red_counts: dict[tuple[int, int], int] = {}
    placed = canvas
    for anchor in anchors:
        ar0, ac0, _, _ = anchor["bbox"]
        placed2 = _try_place(placed, anchor["arr"], ar0, ac0, bg)
        assert placed2 is not None, "anchor piece must fit on canvas at its original position"
        placed = placed2
        for rr, cc in anchor["reds"]:
            key = (ar0 + rr, ac0 + cc)
            red_counts[key] = red_counts.get(key, 0) + 1

    total_ports = int(sum(len(p["reds"]) for p in pieces))
    assert total_ports % 2 == 0, "total red connector ports must be even for complete pairing"
    target_unique_red = total_ports // 2

    solutions: list[np.ndarray] = []

    def dfs(cur: np.ndarray, remaining: list[dict], rc: dict[tuple[int, int], int]) -> None:
        if len(solutions) >= 2:
            return

        if not remaining:
            if len(rc) == target_unique_red and all(v == 2 for v in rc.values()):
                solutions.append(cur)
            return

        if len(rc) > target_unique_red:
            return

        unpaired = [k for k, v in rc.items() if v == 1]
        if not unpaired:
            return
        rc1 = {k for k, v in rc.items() if v == 1}

        best_moves = None
        for t in unpaired:
            moves = []
            tr, tc = t
            for idx, p in enumerate(remaining):
                ph, pw = p["shape"]
                for pr, pc in p["reds"]:
                    top = tr - pr
                    left = tc - pc
                    if top < 0 or left < 0 or top + ph > h or left + pw > w:
                        continue
                    bad = False
                    overlap = 0
                    for rr, cc in p["reds"]:
                        gr, gc = top + rr, left + cc
                        prev = rc.get((gr, gc), 0)
                        if prev >= 2:
                            bad = True
                            break
                        if (gr, gc) in rc1:
                            overlap += 1
                    if bad:
                        continue
                    moves.append((idx, top, left, overlap))
            if not moves:
                return
            if best_moves is None or len(moves) < len(best_moves):
                best_moves = moves
                if len(best_moves) == 1:
                    break

        assert best_moves is not None, "there must be at least one feasible connector move"

        dedup: dict[tuple[int, int, int], int] = {}
        for idx, top, left, ov in best_moves:
            key = (idx, top, left)
            dedup[key] = max(dedup.get(key, 0), ov)
        moves2 = [(idx, top, left, ov) for (idx, top, left), ov in dedup.items()]
        moves2.sort(key=lambda x: (-x[3], -remaining[x[0]]["area"], x[1], x[2], x[0]))

        for idx, top, left, _ov in moves2:
            p = remaining[idx]
            next_remaining = remaining[:idx] + remaining[idx + 1 :]

            next_rc = dict(rc)
            for rr, cc in p["reds"]:
                gr, gc = top + rr, left + cc
                next_rc[(gr, gc)] = next_rc.get((gr, gc), 0) + 1

            nxt = _try_place(cur, p["arr"], top, left, bg)
            if nxt is None:
                continue
            dfs(nxt, next_remaining, next_rc)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, red_counts)
    assert len(solutions) == 1, "multi-anchor assembly must have exactly one valid solution"
    return solutions[0]


# --------------- generator (o5x) ---------------

# Canvas
GRID_H_RANGE = (24, 26)
GRID_W_RANGE = (24, 26)
OUTPUT_GROUP_PADDING = 1
OUTPUT_BORDER = 1

# Group structure
ANCHOR_COUNT_CHOICES = [2, 3]
ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
ANCHOR_CAP_CHOICES = [1, 2]
GROUP_SIZE_OPTIONS = {
    2: [(2, 2), (2, 3)],
    3: [(1, 1, 2), (1, 2, 2)],
}
GROUP_MAX_DIM = 10

# Pieces
PIECE_H_RANGE = (4, 5)
PIECE_W_RANGE = (4, 6)
PIECE_DIM_CANDIDATES = 10
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
PIECE_CAP = 2
BAR_SKIP_PROB = 0.35

# Scatter
SCATTER_PADDING = 2

# Generation
GEN_RETRY_LIMIT = 400

# Side geometry lookups
SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}


def _new_node(kind, h, w, color, top=0, left=0, cap=PIECE_CAP):
    return {
        "kind": kind,
        "shape": (int(h), int(w)),
        "color": int(color),
        "top": int(top),
        "left": int(left),
        "cap": int(cap),
        "sides": set(),
        "ports": {},
    }


def _ensure_port(node, side):
    if side in node["ports"]:
        return
    h, w = node["shape"]
    span = h if side in ("left", "right") else w
    node["ports"][side] = random.randint(1, span - 3)


def _port_cells(node, side):
    h, w = node["shape"]
    k = node["ports"][side]
    if side == "top":
        return [(0, k), (0, k + 1)]
    if side == "bottom":
        return [(h - 1, k), (h - 1, k + 1)]
    if side == "left":
        return [(k, 0), (k + 1, 0)]
    return [(k, w - 1), (k + 1, w - 1)]


def _global_port(node, side):
    tr, tc = node["top"], node["left"]
    return [(tr + r, tc + c) for r, c in _port_cells(node, side)]


def _bbox_of(node):
    h, w = node["shape"]
    return (node["top"], node["left"], node["top"] + h - 1, node["left"] + w - 1)


def _rects_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _union_bbox(nodes):
    boxes = [_bbox_of(n) for n in nodes]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _draw_path(arr, start, end, color, order):
    r, c = start
    er, ec = end
    axes = (
        [(r, er, c, True), (c, ec, er, False)]
        if order == 0
        else [(c, ec, r, False), (r, er, ec, True)]
    )
    for a_start, a_end, fixed, is_row in axes:
        step = 1 if a_end >= a_start else -1
        for v in range(a_start, a_end + step, step):
            rr, cc = (v, fixed) if is_row else (fixed, v)
            if arr[rr, cc] != 2:
                arr[rr, cc] = color


def _paint_bar(arr, side, color, used):
    h, w = arr.shape
    is_horizontal = side in ("top", "bottom")
    fixed = 0 if side in ("top", "left") else (h - 1 if side == "bottom" else w - 1)
    span = w if is_horizontal else h
    lo_conflict = "left" if is_horizontal else "top"
    hi_conflict = "right" if is_horizontal else "bottom"
    for i in range(span):
        if (i == 0 and lo_conflict in used) or (i == span - 1 and hi_conflict in used):
            continue
        if is_horizontal:
            arr[fixed, i] = color
        else:
            arr[i, fixed] = color


def _bar_endpoint(shape, side, hub):
    h, w = shape
    hr, hc = hub
    if side in ("top", "bottom"):
        row = 0 if side == "top" else h - 1
        return (row, min(max(hc, 1), w - 2))
    col = 0 if side == "left" else w - 1
    return (min(max(hr, 1), h - 2), col)


def _make_piece(node):
    h, w = node["shape"]
    color = node["color"]
    used = set(node["sides"])
    arr = np.zeros((h, w), dtype=int)
    hub = (random.randint(1, h - 2), random.randint(1, w - 2))

    bar_sides = [s for s in SIDES if s not in used]
    if len(bar_sides) > 1 and random.random() < BAR_SKIP_PROB:
        bar_sides.remove(random.choice(bar_sides))
    for side in bar_sides:
        _paint_bar(arr, side, color, used)

    terminals = []
    for side in used:
        dr, dc = INWARD[side]
        for r, c in _port_cells(node, side):
            arr[r, c] = 2
            rr, cc = r + dr, c + dc
            arr[rr, cc] = color
            terminals.append((rr, cc))

    arr[hub] = color
    order = random.randint(0, 1)
    for t in terminals:
        _draw_path(arr, t, hub, color, order)
    for side in bar_sides:
        _draw_path(arr, hub, _bar_endpoint((h, w), side, hub), color, 1 - order)

    if len(components_mask(arr != 0, connectivity=4)) != 1:
        return None
    if int(np.sum(arr == 2)) != 2 * len(used):
        return None
    return arr


def _make_anchor(node):
    bh, bw = node["block"]
    h, w = node["shape"]
    arr = np.zeros((h, w), dtype=int)
    arr[1 : 1 + bh, 1 : 1 + bw] = 4
    for side in node["sides"]:
        for r, c in _port_cells(node, side):
            arr[r, c] = 2
    return arr


def _candidate_fits(nodes, parent_i, child):
    cb = _bbox_of(child)
    if not _rects_overlap(cb, _bbox_of(nodes[parent_i])):
        return False
    for i, other in enumerate(nodes):
        if i == parent_i:
            continue
        if _rects_overlap(cb, _bbox_of(other)):
            return False
    r0, c0, r1, c1 = _union_bbox(nodes + [child])
    return (r1 - r0 + 1) <= GROUP_MAX_DIM and (c1 - c0 + 1) <= GROUP_MAX_DIM


def _choose_anchor_blocks(anchor_count):
    if anchor_count == 2:
        blocks = list(ANCHOR_BLOCK_CHOICES)
        random.shuffle(blocks)
        return blocks
    blocks = [ANCHOR_BLOCK_CHOICES[0], ANCHOR_BLOCK_CHOICES[1], random.choice(ANCHOR_BLOCK_CHOICES)]
    random.shuffle(blocks)
    return blocks


def _choose_group_sizes(anchor_count):
    sizes = list(random.choice(GROUP_SIZE_OPTIONS[anchor_count]))
    random.shuffle(sizes)
    return sizes


def _grow_group(block, n_pieces, color):
    bh, bw = block
    anchor = _new_node("anchor", bh + 2, bw + 2, 4, cap=random.choice(ANCHOR_CAP_CHOICES))
    anchor["block"] = (bh, bw)
    nodes = [anchor]

    for _ in range(n_pieces):
        placed = False
        dims = [
            (random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE))
            for _ in range(PIECE_DIM_CANDIDATES)
        ]
        random.shuffle(dims)
        for ph, pw in dims:
            parents = [j for j, n in enumerate(nodes) if len(n["sides"]) < n["cap"]]
            random.shuffle(parents)
            for parent_i in parents:
                parent = nodes[parent_i]
                sides = [s for s in SIDES if s not in parent["sides"]]
                random.shuffle(sides)
                for side in sides:
                    child = _new_node("piece", ph, pw, color)
                    incoming = OPPOSITE[side]
                    _ensure_port(parent, side)
                    _ensure_port(child, incoming)
                    pr, pc = _global_port(parent, side)[0]
                    cr, cc = _port_cells(child, incoming)[0]
                    child["top"] = pr - cr
                    child["left"] = pc - cc
                    child["sides"].add(incoming)
                    if not _candidate_fits(nodes, parent_i, child):
                        continue
                    parent["sides"].add(side)
                    nodes.append(child)
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None
    return nodes


def _normalize_nodes(nodes):
    r0, c0, _, _ = _union_bbox(nodes)
    for node in nodes:
        node["top"] -= r0
        node["left"] -= c0


def _render_group(nodes):
    for node in nodes:
        node["arr"] = _make_anchor(node) if node["kind"] == "anchor" else _make_piece(node)
        if node["arr"] is None:
            return None

    _normalize_nodes(nodes)
    _, _, r1, c1 = _union_bbox(nodes)
    patch = np.zeros((r1 + 1, c1 + 1), dtype=int)
    red_counts = {}
    for node in nodes:
        placed = _try_place(patch, node["arr"], node["top"], node["left"], 0)
        if placed is None:
            return None
        patch = placed
        for rr, cc in np.argwhere(node["arr"] == 2):
            key = (node["top"] + int(rr), node["left"] + int(cc))
            red_counts[key] = red_counts.get(key, 0) + 1

    total_ports = sum(int(np.sum(node["arr"] == 2)) for node in nodes)
    if len(red_counts) * 2 != total_ports:
        return None
    if any(v != 2 for v in red_counts.values()):
        return None
    comps = components_mask(patch != 0, connectivity=4)
    if len(comps) != 1:
        return None
    return patch


def _assemble_groups(groups, h, w):
    out = np.zeros((h, w), dtype=int)
    ordered = sorted(groups, key=lambda g: g["patch"].shape[0] * g["patch"].shape[1], reverse=True)
    for group in ordered:
        try:
            gr, gc = random_free_location(
                out,
                group["patch"],
                transparent=0,
                border=OUTPUT_BORDER,
                padding=OUTPUT_GROUP_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(out, group["patch"], gr, gc, transparent=0)
        for node in group["nodes"]:
            node["pos"] = (gr + node["top"], gc + node["left"])
    if len(components_mask(out != 0, connectivity=4)) != len(groups):
        return None
    return out


def _scatter(groups, h, w):
    inp = np.zeros((h, w), dtype=int)
    for group in groups:
        anchor = group["nodes"][0]
        paste(inp, anchor["arr"], *anchor["pos"], transparent=0)

    rest = [node for group in groups for node in group["nodes"][1:]]
    random.shuffle(rest)
    for node in rest:
        try:
            r, c = random_free_location(
                inp,
                node["arr"],
                transparent=0,
                border=1,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, node["arr"], r, c, transparent=0)
    return inp


def _gen_attempt():
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    anchor_count = random.choice(ANCHOR_COUNT_CHOICES)
    group_sizes = _choose_group_sizes(anchor_count)
    group_colors = random.sample(PIECE_COLORS, anchor_count)
    blocks = _choose_anchor_blocks(anchor_count)

    groups = []
    for block, n_pieces, color in zip(blocks, group_sizes, group_colors):
        nodes = _grow_group(block, n_pieces, color)
        if nodes is None:
            return None
        patch = _render_group(nodes)
        if patch is None:
            return None
        groups.append({"nodes": nodes, "patch": patch, "size": n_pieces})

    out = _assemble_groups(groups, h, w)
    if out is None:
        return None
    inp = _scatter(groups, h, w)
    if inp is None:
        return None

    # Core variant checks: multiple anchors survive in input, multiple output components remain.
    if len([g for g in groups]) < 2:
        return None
    if len(components_mask(out != 0, connectivity=4)) != anchor_count:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cbebaa4b D3 input")
