# Combo variant of e8686506 (index 6)
# combo_spec: D4/0+D6/1
# Name: 上下对称+对称位颜色交换

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis
# - concretization:
#   - 原题把碎片拼成一个左右镜像的矩形，并默认镜像两侧同色。这里不能再沿用这个前提。
#   - 融合后先识别“上下镜像”的几何槽位：上半部的每个非 frame 槽位都对应一个下半部镜像槽位。
#   - 输入仍然给出一个成组保留的 frame；非 frame 碎片散落在外部。先用 frame 重建 pre-swap
#     几何，再把每个上/下镜像槽位的颜色互换得到最终输出。
#   - 具体例子：若上方是形状 A、颜色红，下方镜像位是形状 B、颜色黄，那么输出里上方位置保留 A
#     的几何但显示黄，下方位置保留 B 的几何但显示红。
# - behavior_trace:
#   - 原 solver 在 `_assemble()` 里逐格要求镜像位同色；D6 单变体则改成“先重建槽位，再交换镜像
#     槽位颜色”。本融合版沿用后者，但镜像轴从左右换成上下。
#   - 因而 solver 的关键不再是颜色一致性，而是：`frame -> open slots -> oriented signatures ->
#     上下镜像签名颜色互换`。
# - complexity_calibration:
#   - 若任由输出偶然同时左右对称，解题者会看不出 D4 轴变化；所以 generator 显式拒绝 LR 对称的
#     frame 版式。
#   - 若镜像槽位颜色相同，则颜色交换退化为无效操作；所以 generator 强制至少两组上下镜像签名家族
#     使用不同颜色，并统计真正被交换的像素数。
#   - 为了避免任意配对歧义，generator 让每个“定向签名”在输入中只有一种颜色，并让同签名重复出现，
#     这样人类能从少量样例看出“上槽位吃下槽位的颜色”。
# - hidden_assumptions_audit:
#   - 原题依赖的 corners / bars / side-lines 颜色锚，在颜色交换后不再稳定，因此这里不再把颜色锚
#     作为推理核心；主约束改为 frame 派生出的几何槽位。
#   - solver 新增了“输出高度为偶数”的假设，用来避免中心行既属于上半部又属于下半部。generator
#     固定只产生 10x5 输出来保证这一点。
#   - solver 还假设 frame 的开口槽位能用与 generator 相同的 `_slotize()` 规则唯一分块；
#     generator 通过构造 4 个彼此分离的上半部三格槽位来保证这一点。
# - placement_legality:
#   - 散布时先整块占住 frame bbox，再把非 frame patch 全部放到 bbox 外，最后仅恢复 frame 像素；
#     因此不会与 frame 或其他 patch 重叠，也不会越界。
# - observability:
#   - 输入里能直接看到一个上下镜像的 frame 轮廓，且顶部/底部碎片成对出现；输出则保留这套上下
#     配对几何，但成对颜色互换。因为 frame 设计成非 LR 对称，人类不会把轴误读成左右。
# - visual_cleanliness:
#   - frame 单色，非 frame 只用少量重复签名家族，颜色数控制在 4-6，避免视觉噪声过大。
# - color-change inferability:
#   - 颜色变化只依赖可见的上下镜像对应关系，不依赖颜色 id 或数值排序。每个签名家族重复至少两次，
#     给解题者足够证据归纳“上下镜像配对后互换颜色”。
# - solver assumptions checked against generator:
#   - 总前景面积始终是 50，因此最近方形画像因子唯一为 10x5。
#   - 输入必含一个由少量 8 连通碎片组成的 UD-symmetric frame 组，其 union bbox 就是输出尺寸。
#   - 非 frame 槽位全部来自 frame 开口，且每个定向签名在输入里只有一种颜色。
#   - 每个定向签名都有一个上下镜像签名，且二者拷贝数相同。


# === CFG: Output dimensions ===
_AREA_MIN = 50
_AREA_MAX = 50
_WIDTH_REQUIRED = 5
_HEIGHT_REQUIRED = 10

# === CFG: Color and palette ===
_N_COLORS_MIN = 4
_N_COLORS_MAX = 6
_MIN_SWAP_FAMILIES = 2
_MIN_FILL_COLORS_USED = 3

# === CFG: Piece constraints ===
_MAX_PIECE_AREA = 6
_PIECE_COUNT_MIN = 9
_PIECE_COUNT_MAX = 17
_MAX_FRAME_PARTS = 5

# === CFG: Canvas / scatter ===
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1

# === CFG: Retry ===
_GEN_MAX_RETRIES = 400


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _ud_mirror_sig(sig: tuple[int, int, tuple[tuple[int, int], ...]]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    h, w, coords = sig
    return h, w, tuple(sorted((h - 1 - int(r), int(c)) for r, c in coords))


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            h = n // w
            out.append((h, w))
    return out


def _valid_dims() -> list[tuple[int, int]]:
    out = []
    for area in range(_AREA_MIN, _AREA_MAX + 1):
        dims = [(area // w, w) for w in range(1, area + 1) if area % w == 0 and area // w >= w]
        best_gap = min(abs(h - w) for h, w in dims)
        best = [(h, w) for h, w in dims if abs(h - w) == best_gap]
        if len(best) == 1:
            h, w = best[0]
            if w == _WIDTH_REQUIRED and h == _HEIGHT_REQUIRED and h % 2 == 0:
                out.append((h, w))
    return out


_VALID_DIMS = _valid_dims()
_DIM_WEIGHTS = [h * w for h, w in _VALID_DIMS]


def _chunk_component(cells: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    remaining = set(cells)
    chunks: list[list[tuple[int, int]]] = []
    while remaining:
        start = min(remaining)
        queue = [start]
        queued = {start}
        chunk: list[tuple[int, int]] = []
        while queue and len(chunk) < _MAX_PIECE_AREA:
            r, c = queue.pop(0)
            if (r, c) not in remaining:
                continue
            remaining.remove((r, c))
            chunk.append((r, c))
            nbrs = []
            for rr, cc in ((r - 1, c), (r, c - 1), (r, c + 1), (r + 1, c)):
                if (rr, cc) in remaining and (rr, cc) not in queued:
                    nbrs.append((rr, cc))
                    queued.add((rr, cc))
            queue.extend(sorted(nbrs))
        chunks.append(sorted(chunk))
    return chunks


def _slotize(frame_mask: np.ndarray) -> list[dict]:
    H, W = frame_mask.shape
    assert H % 2 == 0, "UD swap variant requires even output height"
    mid = H // 2
    open_mask = ~frame_mask
    slots: list[dict] = []

    top_open = open_mask[:mid, :]
    for comp in components_mask(top_open, connectivity=4):
        top_cells = sorted((int(r), int(c)) for r, c in comp)
        for chunk in _chunk_component(top_cells):
            arr = np.array(chunk, dtype=int)
            slots.append({"cells": chunk, "sig": _piece_sig(arr)})
            mirror = sorted((H - 1 - int(r), int(c)) for r, c in chunk)
            marr = np.array(mirror, dtype=int)
            slots.append({"cells": mirror, "sig": _piece_sig(marr)})
    return slots


def _slot_groups(slots: list[dict]) -> dict[tuple[int, int, tuple[tuple[int, int], ...]], list[list[tuple[int, int]]]]:
    out: dict[tuple[int, int, tuple[tuple[int, int], ...]], list[list[tuple[int, int]]]] = {}
    for slot in slots:
        out.setdefault(slot["sig"], []).append(slot["cells"])
    return out


def _swap_board(pre: np.ndarray, slot_groups: dict) -> np.ndarray:
    out = pre.copy()
    sig_color: dict[tuple[int, int, tuple[tuple[int, int], ...]], int] = {}
    for sig, groups in slot_groups.items():
        r, c = groups[0][0]
        sig_color[sig] = int(pre[r, c])
    for sig, groups in slot_groups.items():
        msig = _ud_mirror_sig(sig)
        assert msig in slot_groups, "every slot signature must have a UD mirror partner"
        target = sig_color[msig]
        for cells in groups:
            for r, c in cells:
                out[r, c] = target
    return out


def _is_ud_symmetric_bitmap(parts: list[tuple[int, np.ndarray]], H: int, W: int) -> bool:
    all_pts = np.concatenate([p for _, p in parts])
    r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
    bmp = np.zeros((H, W), dtype=bool)
    for _, pts in parts:
        for r, c in pts:
            bmp[int(r) - r0, int(c) - c0] = True
    return np.array_equal(bmp, bmp[::-1, :])


def _find_frame_parts(g: np.ndarray, bg: int, H: int, W: int) -> list[tuple[int, np.ndarray]]:
    comps8_all: list[tuple[int, np.ndarray]] = []
    for col in sorted(set(map(int, np.unique(g))) - {int(bg)}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), pts))
    assert comps8_all, "grid must contain non-background 8-connected parts"

    best: list[tuple[int, np.ndarray]] = []
    n = len(comps8_all)
    for mask in range(1, 1 << n):
        if mask.bit_count() > _MAX_FRAME_PARTS:
            continue
        parts = [comps8_all[i] for i in range(n) if (mask >> i) & 1]
        union = np.concatenate([p for _, p in parts])
        r0, c0, r1, c1 = bbox_coords(union)
        if r1 - r0 + 1 != H or c1 - c0 + 1 != W:
            continue
        if not _is_ud_symmetric_bitmap(parts, H, W):
            continue
        if not best:
            best = parts
            continue
        area = sum(len(p) for _, p in parts)
        best_area = sum(len(p) for _, p in best)
        if area > best_area or (area == best_area and len(parts) < len(best)):
            best = parts

    assert best, "input must contain a grouped UD-symmetric frame pattern"
    return best


def _build_pre_from_input(g: np.ndarray, bg: int) -> tuple[np.ndarray, dict]:
    total_area = int((g != int(bg)).sum())
    assert total_area > 0, "grid must contain non-background pieces"
    dims = [(h, w) for h, w in _factors(total_area) if h >= w]
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]
    assert H % 2 == 0, "UD swap variant expects an even output height"

    frame_parts = _find_frame_parts(g, bg, H, W)
    frame_input_mask = np.zeros_like(g, dtype=bool)
    for _, pts in frame_parts:
        for r, c in pts:
            frame_input_mask[int(r), int(c)] = True

    pre = np.full((H, W), -1, dtype=int)
    all_frame_pts = np.concatenate([p for _, p in frame_parts])
    fr0, fc0 = int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())
    for col, pts in frame_parts:
        for r, c in pts:
            pre[int(r) - fr0, int(c) - fc0] = int(col)

    frame_mask = pre != -1
    slots = _slotize(frame_mask)
    slot_groups = _slot_groups(slots)

    input_groups: dict[tuple[int, int, tuple[tuple[int, int], ...]], list[int]] = {}
    seen = 0
    for col in sorted(set(map(int, np.unique(g))) - {int(bg)}):
        for pts in components_mask(g == col, connectivity=4):
            if all(frame_input_mask[int(r), int(c)] for r, c in pts):
                continue
            sig = _piece_sig(pts)
            input_groups.setdefault(sig, []).append(int(col))
            seen += 1

    assert seen == len(slots), "non-frame piece count must match reconstructed slots"

    sig_color: dict[tuple[int, int, tuple[tuple[int, int], ...]], int] = {}
    for sig, groups in slot_groups.items():
        cols = input_groups.get(sig, [])
        assert len(cols) == len(groups), "slot multiplicity must match input multiplicity"
        uniq = set(cols)
        assert len(uniq) == 1, "each oriented signature must have one input color"
        sig_color[sig] = int(next(iter(uniq)))
        for cells in groups:
            for r, c in cells:
                pre[r, c] = sig_color[sig]

    assert np.all(pre >= 0), "reconstructed pre-swap board must be dense"
    meta = {"slot_groups": slot_groups, "frame_mask": frame_mask, "dims": (H, W)}
    return pre, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a dense rectangle from scattered fragments, then swap colors across top-bottom mirror partners.

    Input structure:
    - The background is the most frequent color.
    - A small grouped set of 8-connected frame fragments remains in output-relative positions and reveals a horizontally mirrored rectangle.
    - All other pieces are scattered outside that bbox and keep their original orientation and pre-swap color.

    Rule:
    - Infer the output size from total fragment area and locate the grouped UD-symmetric frame.
    - Use the frame openings to reconstruct the deterministic non-frame slots, then refill each slot by matching scattered piece shape signatures.
    - After the pre-swap rectangle is rebuilt, recolor every non-frame slot with the color of its top-bottom mirrored partner.

    Key insight:
    - Geometry is recovered from the frame rather than from mirrored colors: once the solver identifies the horizontal symmetry axis and rebuilds the pre-swap slot layout, the final step is a color exchange between each upper slot and its lower mirror mate.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pre, meta = _build_pre_from_input(g, bg)
    out = _swap_board(pre, meta["slot_groups"])
    return out.astype(int)


def _frame_template(H: int, W: int) -> tuple[np.ndarray | None, set[int], list[int]]:
    n_colors = random.randint(_N_COLORS_MIN, _N_COLORS_MAX)
    palette = list(range(n_colors))
    random.shuffle(palette)

    frame_color = palette[0]
    fill_pool = [c for c in palette if c != frame_color]
    if len(fill_pool) < _MIN_FILL_COLORS_USED:
        return None, set(), []

    shape_a = [(0, 0), (1, 0), (1, 1)]
    shape_c = [(0, 0), (0, 1), (1, 1)]
    layouts = [
        [shape_a, shape_c, shape_c, shape_a],
        [shape_c, shape_a, shape_a, shape_c],
    ]
    anchors = [(0, 0), (0, W - 2), (H // 2 - 2, 0), (H // 2 - 2, W - 2)]
    layout = random.choice(layouts)

    open_mask = np.zeros((H, W), dtype=bool)
    for (r0, c0), shape in zip(anchors, layout):
        for dr, dc in shape:
            r = r0 + dr
            c = c0 + dc
            open_mask[r, c] = True
            open_mask[H - 1 - r, c] = True

    grid = np.full((H, W), frame_color, dtype=int)
    grid[open_mask] = -1
    frame_colors = {frame_color}
    return grid, frame_colors, fill_pool


def _signature_graph(
    frame_grid: np.ndarray,
    slots: list[dict],
) -> tuple[dict, dict, bool]:
    H, W = frame_grid.shape
    adjacency: dict[tuple[int, int, tuple[tuple[int, int], ...]], set] = {}
    frame_adj: dict[tuple[int, int, tuple[tuple[int, int], ...]], set[int]] = {}
    cell_to_slot: dict[tuple[int, int], int] = {}
    for i, slot in enumerate(slots):
        sig = slot["sig"]
        adjacency.setdefault(sig, set())
        frame_adj.setdefault(sig, set())
        for cell in slot["cells"]:
            cell_to_slot[cell] = i

    for i, slot in enumerate(slots):
        sig = slot["sig"]
        for r, c in slot["cells"]:
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                v = int(frame_grid[rr, cc])
                if v != -1:
                    frame_adj[sig].add(v)
                    continue
                j = cell_to_slot.get((rr, cc))
                if j is None or j == i:
                    continue
                other_sig = slots[j]["sig"]
                if other_sig == sig:
                    return {}, {}, False
                adjacency[sig].add(other_sig)
    return adjacency, frame_adj, True


def _assign_signature_colors(
    slot_groups: dict,
    adjacency: dict,
    frame_adj: dict,
    fill_pool: list[int],
) -> dict | None:
    signatures = list(slot_groups.keys())
    for sig in signatures:
        msig = _ud_mirror_sig(sig)
        if msig not in slot_groups:
            return None
        if len(slot_groups[sig]) != len(slot_groups[msig]):
            return None

    seen: set = set()
    asym_families = []
    for sig in signatures:
        msig = _ud_mirror_sig(sig)
        if sig == msig or sig in seen or msig in seen:
            continue
        seen.add(sig)
        seen.add(msig)
        asym_families.append((sig, msig))
    if len(asym_families) < _MIN_SWAP_FAMILIES:
        return None

    degrees = {sig: len(adjacency.get(sig, set())) + len(frame_adj.get(sig, set())) for sig in signatures}

    def try_once() -> dict | None:
        assignment: dict = {}

        def domain(sig) -> list[int]:
            used = {assignment[n] for n in adjacency.get(sig, set()) if n in assignment}
            usage = {col: list(assignment.values()).count(col) for col in fill_pool}
            opts = []
            for col in fill_pool:
                if col in frame_adj.get(sig, set()):
                    continue
                if col in used:
                    continue
                msig = _ud_mirror_sig(sig)
                if msig != sig and msig in assignment and assignment[msig] == col:
                    continue
                opts.append(col)
            random.shuffle(opts)
            opts.sort(key=lambda col: usage[col])
            return opts

        def choose_unassigned():
            cands = [sig for sig in signatures if sig not in assignment]
            cands.sort(key=lambda sig: (len(domain(sig)), -degrees[sig]))
            return cands[0] if cands else None

        def dfs() -> bool:
            sig = choose_unassigned()
            if sig is None:
                return True
            for col in domain(sig):
                assignment[sig] = int(col)
                if dfs():
                    return True
                assignment.pop(sig)
            return False

        if not dfs():
            return None
        used_colors = {assignment[sig] for sig in signatures}
        if len(used_colors) < _MIN_FILL_COLORS_USED:
            return None
        return assignment

    for _ in range(80):
        out = try_once()
        if out is not None:
            return out
    return None


def _build_pre_and_out() -> tuple[np.ndarray | None, np.ndarray | None, int, set[int], dict]:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    frame_grid, frame_colors, fill_pool = _frame_template(H, W)
    if frame_grid is None:
        return None, None, 0, set(), {}

    frame_mask = frame_grid != -1
    if np.array_equal(frame_mask, frame_mask[:, ::-1]):
        return None, None, 0, set(), {}

    slots = _slotize(frame_mask)
    slot_groups = _slot_groups(slots)
    adjacency, frame_adj, ok = _signature_graph(frame_grid, slots)
    if not ok:
        return None, None, 0, set(), {}

    colors_by_sig = _assign_signature_colors(slot_groups, adjacency, frame_adj, fill_pool)
    if colors_by_sig is None:
        return None, None, 0, set(), {}

    pre_abs = frame_grid.copy()
    for sig, groups in slot_groups.items():
        col = int(colors_by_sig[sig])
        for cells in groups:
            for r, c in cells:
                pre_abs[r, c] = col
    assert np.all(pre_abs >= 0)

    out_abs = _swap_board(pre_abs, slot_groups)
    changed_cells = int(np.sum(pre_abs != out_abs))
    if changed_cells == 0:
        return None, None, 0, set(), {}

    bg = random.randint(0, 9)
    abstract_colors = list(map(int, np.unique(pre_abs)))
    random.shuffle(abstract_colors)
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < len(abstract_colors):
        return None, None, 0, set(), {}
    cmap = {ac: available[i] for i, ac in enumerate(abstract_colors)}
    pre = np.vectorize(lambda x: cmap[int(x)])(pre_abs).astype(int)
    out = np.vectorize(lambda x: cmap[int(x)])(out_abs).astype(int)
    real_frame_colors = {cmap[int(c)] for c in frame_colors}

    swap_families = sum(
        1
        for sig in slot_groups
        if sig < _ud_mirror_sig(sig) and colors_by_sig[sig] != colors_by_sig[_ud_mirror_sig(sig)]
    )
    meta = {
        "swap_families": int(swap_families),
        "changed_cells": changed_cells,
        "frame_lr_symmetric": bool(np.array_equal(frame_mask, frame_mask[:, ::-1])),
        "used_fill_colors": len({int(colors_by_sig[sig]) for sig in colors_by_sig}),
    }
    return pre, out, bg, real_frame_colors, meta


def _extract_non_frame_patches(pre: np.ndarray, bg: int, frame_colors: set[int]) -> list[np.ndarray]:
    patches: list[np.ndarray] = []
    for col in sorted(map(int, np.unique(pre))):
        if col in frame_colors:
            continue
        for pts in components_mask(pre == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in pts:
                patch[int(r) - r0, int(c) - c0] = col
            patches.append(patch)
    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), random.random()))
    return patches


def _scatter(pre: np.ndarray, bg: int, frame_colors: set[int]) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = pre.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("frame bbox does not fit canvas")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1

    for patch in _extract_non_frame_patches(pre, bg, frame_colors):
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    frame_mask = np.isin(pre, list(frame_colors))
    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    bbox_region[frame_mask] = pre[frame_mask]
    return canvas


def _piece_count(grid: np.ndarray) -> int:
    return sum(len(components_mask(grid == col, connectivity=4)) for col in np.unique(grid))


def _gen_attempt() -> tuple[np.ndarray | None, dict]:
    pre, out, bg, frame_colors, meta = _build_pre_and_out()
    if pre is None or out is None:
        return None, {}
    if not (_N_COLORS_MIN <= len(np.unique(out)) <= _N_COLORS_MAX):
        return None, {}
    pc = _piece_count(pre)
    if not (_PIECE_COUNT_MIN <= pc <= _PIECE_COUNT_MAX):
        return None, {}
    if any(
        len(pts) > _MAX_PIECE_AREA
        for col in np.unique(pre)
        if col not in frame_colors
        for pts in components_mask(pre == col, connectivity=4)
    ):
        return None, {}
    if meta.get("swap_families", 0) < _MIN_SWAP_FAMILIES:
        return None, {}
    if meta.get("changed_cells", 0) == 0 or meta.get("frame_lr_symmetric", True):
        return None, {}
    try:
        inp = _scatter(pre, bg, frame_colors)
        got = main(inp)
    except Exception:
        return None, {}
    if np.array_equal(got, out):
        return inp, meta
    return None, {}


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid, _meta = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 combo input")
