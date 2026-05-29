# Variant of e8686506
# Dimension: D1 — 匹配条件：边缘对齐拼合 vs 其他匹配逻辑 (match_cond)
# Change: s.edge.position aligns with d.adjacent_piece.edge — 碎片的边缘位置与... → 形状嵌合匹配

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D1 variant
# ----------------------------------
# spec:
#   invariant_core:
#     - 输入仍然是在 13-15 的大画布上散落若干彩色碎片，背景色为多数色。
#     - 输出仍然是一个严格左右镜像的稠密竖向矩形，尺寸仍由总面积的唯一近方因子决定。
#     - 仍保留一个留在原位的 LR 对称“框架”对象，用它把最终矩形的位置锁定。
#   D1_change:
#     old_match:
#       - 原题 solver 会把每个碎片当作可平移的块，在整个 HxW 矩形里寻找“不冲突且满足镜像/边框约束”的位置。
#       - 唯一性主要来自：矩形边界、镜像、同色不接触、以及少量角/边标记。
#     new_match:
#       - 现在框架不是单纯提供轮廓，而是在自己的边界/内部刻出多个不规则凹槽。
#       - 散落碎片就是这些凹槽的补块；solver 不再扫描“哪里还能放得下”，而是直接提取凹槽轮廓，
#         按形状标准化后一一匹配，把对应碎片嵌回去。
#       - 因为不允许旋转/翻转，左侧 L 形凹槽和右侧镜像 L 形凹槽需要不同朝向的补块，
#         这正是变体里“凹凸互补”可观察的地方。
#   concrete_trace:
#     - 例：框架左上角缺一个 2x2 的 L 形口子 {(0,0),(1,0),(1,1)}，右侧缺其镜像。
#     - 原逻辑会把这个 L 形碎片视为“某处可放的块”，再靠整体约束收敛到正确位置。
#     - 新逻辑先从固定框架中直接读出该凹槽的相对坐标，再找完全同形的散落碎片回填；
#       差异点是：位置不是通过边缘齐平搜索得出，而是由凹槽轮廓直接指定。
#   complexity_calibration:
#     - 如果所有凹槽都做成单像素或小矩形，视觉上会退化成普通填空，D1 不明显。
#     - 因此生成器强制产生多个非矩形凹槽，并要求至少一部分凹槽直接触边，形成明显的“缺口”。
#     - 同时限制凹槽模板到少量小形状，避免自由度过高导致人类难以从 2-3 例中归纳。
#   implicit_assumption_audit:
#     - 原 `_assemble()` 依赖“任意类型块都要枚举所有平移位置”的前提；新变体中该前提不再成立，
#       因为正确位置来自框架凹槽，枚举整个矩形反而会把匹配条件写错。
#     - 原 `_build_requirements()` 假定角点/横条/竖条是决定性锚点；新变体里这些锚点不是必要信息，
#       框架凹槽本身已经足够锁定每个碎片的位置。
#     - 原“同色碎片不可 4 邻接”的限制只是在排布搜索时避免组件合并；新 solver 不再依赖它求解，
#       但生成器仍让不同凹槽之间隔着框架，因此同色补块在输出中也不会互相粘连。
#   placement_legality:
#     - 所有凹槽都先在 HxW 输出框内雕刻，再把补块裁成最小 bbox 后散到大画布，保证不越界、不重叠。
#     - 雕刻时显式检查：新凹槽不能与旧凹槽 4 邻接，且剩余框架必须保持 8 连通与完整 bbox。
#   visual_observability:
#     - 解题者能直接看到：输入里有一个已经对称摆好的大框架，上面缺了若干锯齿/拐角形口子；
#       旁边散落的碎片外形恰好对应这些口子，而输出就是把这些口子全部嵌满。
#     - 颜色不参与排序或映射推理；颜色只是在回填后保留原碎片颜色，真正决定位置的是形状。
#   cleanliness:
#     - 只有两类视觉层次：固定框架、散落补块。没有额外标记覆盖同一区域，读图更直接。
#   new_solver_assumptions_and_generator_guarantees:
#     - `main()` 新假设 1：存在 1 个 LR 对称框架，其 bbox 就是输出 bbox。生成器始终保留该框架在原位。
#     - `main()` 新假设 2：每个缺口都是一个单独 4 连通区域。生成器用 4 邻接间隔阻止缺口合并。
#     - `main()` 新假设 3：同一标准化形状不会对应多种颜色。生成器对同形补块统一配色。


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _coords_to_patch(coords: tuple[tuple[int, int], ...], col: int, bg: int) -> np.ndarray:
    arr = np.array(coords, dtype=int)
    h = int(arr[:, 0].max() + 1)
    w = int(arr[:, 1].max() + 1)
    patch = np.full((h, w), bg, dtype=int)
    for r, c in coords:
        patch[int(r), int(c)] = int(col)
    return patch


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            h = n // w
            out.append((h, w))
    return out


def _find_frame_parts(g: np.ndarray, bg: int, H: int, W: int) -> list[tuple[int, np.ndarray]]:
    comps8_all = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), np.array(pts, dtype=int)))

    def is_lr_symmetric(parts: list[tuple[int, np.ndarray]]) -> bool:
        all_pts = np.concatenate([p for _, p in parts], axis=0)
        r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
        bmp = np.zeros((H, W), dtype=bool)
        for _, pts in parts:
            for r, c in pts:
                bmp[int(r) - r0, int(c) - c0] = True
        return np.array_equal(bmp, bmp[:, ::-1])

    frame_parts: list[tuple[int, np.ndarray]] = []
    for i, (c1, p1) in enumerate(comps8_all):
        r0, c0, r1, c1b = bbox_coords(p1)
        if r1 - r0 + 1 == H and c1b - c0 + 1 == W and is_lr_symmetric([(c1, p1)]):
            frame_parts = [(c1, p1)]
            break
        for j in range(i + 1, len(comps8_all)):
            c2, p2 = comps8_all[j]
            union = np.concatenate([p1, p2], axis=0)
            ur0, uc0, ur1, uc1 = bbox_coords(union)
            if ur1 - ur0 + 1 == H and uc1 - uc0 + 1 == W and is_lr_symmetric([(c1, p1), (c2, p2)]):
                frame_parts = [(c1, p1), (c2, p2)]
                break
        if frame_parts:
            break

    assert frame_parts, "input must contain 1-2 LR-symmetric frame parts whose union bbox equals output dims"
    return frame_parts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a left-right symmetric rectangle by plugging scattered pieces back into a fixed notched frame.

    Input structure:
    - bg is the dominant color.
    - One LR-symmetric frame component (or a 2-color frame split into two components) already sits in output position.
    - The other colored components are scattered plug pieces whose silhouettes match the frame's missing notches.

    Rule:
    - Infer the output rectangle size from the total non-bg area.
    - Detect the anchored symmetric frame and copy it into the output.
    - Read every missing region inside the frame bbox as a shape template, then place the scattered pieces into those holes by exact shape match with no rotation or flip.

    Key insight:
    - The correct location of each piece is determined by the complementary notch profile already carved into the frame, not by globally searching for edge-aligned placements.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    pieces = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=4):
            pts = np.array(pts, dtype=int)
            h, w, coords = _piece_sig(pts)
            pieces.append({"col": int(col), "coords": coords, "h": h, "w": w, "abs_pts": pts})

    total_area = sum(len(p["coords"]) for p in pieces)
    assert total_area > 0, "grid must contain non-background pieces"

    dims = [(h, w) for (h, w) in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]

    frame_parts = _find_frame_parts(g, bg, H, W)
    frame_abs = np.concatenate([pts for _, pts in frame_parts], axis=0)
    fr0, fc0 = int(frame_abs[:, 0].min()), int(frame_abs[:, 1].min())

    board = np.full((H, W), -1, dtype=int)
    frame_cells_abs = set()
    for col, pts in frame_parts:
        for r, c in pts:
            rr, cc = int(r) - fr0, int(c) - fc0
            board[rr, cc] = int(col)
            frame_cells_abs.add((int(r), int(c)))

    scattered = []
    for p in pieces:
        abs_set = set((int(r), int(c)) for r, c in p["abs_pts"])
        if abs_set <= frame_cells_abs:
            continue
        scattered.append(p)

    hole_regions = [np.array(hole, dtype=int) for hole in components_mask(board == -1, connectivity=4)]
    assert hole_regions, "frame must contain at least one missing region"

    by_shape: dict[tuple[tuple[int, int], ...], list[dict]] = defaultdict(list)
    shape_colors: dict[tuple[tuple[int, int], ...], set[int]] = defaultdict(set)
    for p in scattered:
        by_shape[p["coords"]].append(p)
        shape_colors[p["coords"]].add(int(p["col"]))

    for coords, cols in shape_colors.items():
        assert len(cols) == 1, f"shape {coords} must map to a single color"

    needed = Counter()
    for hole in hole_regions:
        _, _, coords = _piece_sig(hole)
        needed[coords] += 1

    available = Counter({coords: len(items) for coords, items in by_shape.items()})
    assert needed == available, "scattered piece shapes must match frame notch shapes exactly"

    for hole in hole_regions:
        r0, c0, _, _ = bbox_coords(hole)
        _, _, coords = _piece_sig(hole)
        piece = by_shape[coords].pop()
        col = int(piece["col"])
        for dr, dc in coords:
            board[int(r0) + int(dr), int(c0) + int(dc)] = col

    assert np.all(board >= 0), "all frame holes must be filled"
    assert np.array_equal(board, board[:, ::-1]), "assembled board must stay LR-symmetric"
    return board.astype(int)


# --------------- generator (D1 shape-match variant) ---------------

_AREA_MIN = 25
_AREA_MAX = 50
_WIDTH_REQUIRED = 5
_HEIGHT_MAX = 10

_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1

_GEN_MAX_RETRIES = 500
_MIN_NON_RECT_HOLES = 3
_MIN_BOUNDARY_HOLES = 2

_SIDE_TEMPLATES = [
    ((0, 0),),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 0)),
]

_CENTER_TEMPLATES = [
    ((0, 0),),
    ((0, 0), (1, 0)),
    ((0, 0), (1, 0), (2, 0)),
]


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


def _mirror_cells(cells: set[tuple[int, int]], W: int) -> set[tuple[int, int]]:
    return {(int(r), int(W - 1 - c)) for r, c in cells}


def _is_rect_shape(coords: tuple[tuple[int, int], ...]) -> bool:
    arr = np.array(coords, dtype=int)
    h = int(arr[:, 0].max() + 1)
    w = int(arr[:, 1].max() + 1)
    return len(coords) == h * w


def _touches_boundary(cells: set[tuple[int, int]], H: int, W: int) -> bool:
    return any(r == 0 or r == H - 1 or c == 0 or c == W - 1 for r, c in cells)


def _can_carve(
    holes_mask: np.ndarray,
    new_cells: set[tuple[int, int]],
    H: int,
    W: int,
) -> bool:
    for r, c in new_cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if holes_mask[r, c]:
            return False
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < H and 0 <= cc < W and holes_mask[rr, cc]:
                return False

    frame_mask = np.ones((H, W), dtype=bool)
    frame_mask[holes_mask] = False
    for r, c in new_cells:
        frame_mask[r, c] = False
    comps = components_mask(frame_mask, connectivity=8)
    return len(comps) == 1


def _candidate_groups(H: int, W: int) -> list[list[set[tuple[int, int]]]]:
    groups: list[list[set[tuple[int, int]]]] = []

    for coords in _SIDE_TEMPLATES:
        arr = np.array(coords, dtype=int)
        th = int(arr[:, 0].max() + 1)
        tw = int(arr[:, 1].max() + 1)
        for r0 in range(H - th + 1):
            for c0 in range(2 - tw + 1):
                left = {(int(r0 + dr), int(c0 + dc)) for dr, dc in coords}
                right = _mirror_cells(left, W)
                if left & right:
                    continue
                groups.append([left, right])

    for coords in _CENTER_TEMPLATES:
        arr = np.array(coords, dtype=int)
        th = int(arr[:, 0].max() + 1)
        for r0 in range(H - th + 1):
            cells = {(int(r0 + dr), 2) for dr, _dc in coords}
            groups.append([cells])

    random.shuffle(groups)
    groups.sort(key=lambda grp: (-sum(len(c) for c in grp), random.random()))
    return groups


def _build_holes(H: int, W: int) -> list[list[set[tuple[int, int]]]] | None:
    groups = _candidate_groups(H, W)
    holes_mask = np.zeros((H, W), dtype=bool)
    chosen: list[list[set[tuple[int, int]]]] = []

    target_pieces = random.randint(7 if H <= 5 else 9, 9 if H <= 5 else 13)
    target_area = random.randint(H + 6, min(H * 2 + 4, H * W - 12))

    piece_count = 0
    area_count = 0
    for grp in groups:
        new_cells = set().union(*grp)
        add_pieces = len(grp)
        add_area = sum(len(c) for c in grp)
        if piece_count >= target_pieces and area_count >= target_area:
            break
        if piece_count + add_pieces > 13:
            continue
        if area_count + add_area > target_area + 4:
            continue
        if not _can_carve(holes_mask, new_cells, H, W):
            continue
        for r, c in new_cells:
            holes_mask[r, c] = True
        chosen.append(grp)
        piece_count += add_pieces
        area_count += add_area

    pieces = [cells for grp in chosen for cells in grp]
    if not (6 <= len(pieces) <= 13):
        return None

    hole_comps = [set((int(r), int(c)) for r, c in comp) for comp in components_mask(holes_mask, connectivity=4)]
    if len(hole_comps) != len(pieces):
        return None

    non_rect = 0
    boundary = 0
    for cells in pieces:
        pts = np.array(sorted(cells), dtype=int)
        _, _, coords = _piece_sig(pts)
        if not _is_rect_shape(coords):
            non_rect += 1
        if _touches_boundary(cells, H, W):
            boundary += 1

    if non_rect < _MIN_NON_RECT_HOLES or boundary < _MIN_BOUNDARY_HOLES:
        return None

    return chosen


def _paint_output(H: int, W: int) -> tuple[np.ndarray, int, int, list[tuple[int, tuple[tuple[int, int], ...]]]] | None:
    chosen = _build_holes(H, W)
    if chosen is None:
        return None

    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)

    frame_color = palette.pop()
    hole_color_count = random.randint(3, 5)
    hole_palette = palette[:hole_color_count]

    out = np.full((H, W), frame_color, dtype=int)
    piece_specs: list[tuple[int, tuple[tuple[int, int], ...]]] = []

    hole_groups: list[list[tuple[tuple[int, int], ...]]] = []
    for grp in chosen:
        sigs = []
        for cells in grp:
            pts = np.array(sorted(cells), dtype=int)
            _, _, coords = _piece_sig(pts)
            sigs.append(coords)
        hole_groups.append(sigs)

    random.shuffle(hole_palette)
    group_colors = []
    for i in range(len(hole_groups)):
        if i < len(hole_palette):
            group_colors.append(hole_palette[i])
        else:
            group_colors.append(random.choice(hole_palette))
    random.shuffle(group_colors)

    shape_color: dict[tuple[tuple[int, int], ...], int] = {}
    for grp_sigs, grp_color in zip(hole_groups, group_colors):
        existing = [shape_color[coords] for coords in grp_sigs if coords in shape_color]
        use_color = existing[0] if existing else grp_color
        for coords in grp_sigs:
            shape_color[coords] = use_color

    for grp_sigs, grp_cells in zip(hole_groups, chosen):
        grp_color = shape_color[grp_sigs[0]]
        for cells, coords in zip(grp_cells, grp_sigs):
            for r, c in cells:
                out[int(r), int(c)] = int(grp_color)
            piece_specs.append((int(grp_color), coords))

    if not np.array_equal(out, out[:, ::-1]):
        return None

    colors_n = len(np.unique(out))
    if not (4 <= colors_n <= 6):
        return None

    return out, bg, frame_color, piece_specs


def _scatter(
    out: np.ndarray,
    bg: int,
    frame_color: int,
    piece_specs: list[tuple[int, tuple[tuple[int, int], ...]]],
) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    H, W = out.shape

    max_r = rows - H - 2
    max_c = cols - W - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("canvas too small for anchored frame")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    frame_mask = out == int(frame_color)

    canvas[off_r : off_r + H, off_c : off_c + W] = -1

    patches = []
    for col, coords in piece_specs:
        patches.append(_coords_to_patch(coords, col, bg))
    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), random.random()))

    for patch in patches:
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    bbox_region = canvas[off_r : off_r + H, off_c : off_c + W]
    bbox_region[:] = bg
    bbox_region[frame_mask] = out[frame_mask]
    return canvas


def _piece_count(grid: np.ndarray, bg: int) -> int:
    return sum(len(components_mask(grid == col, connectivity=4)) for col in np.unique(grid) if int(col) != int(bg))


def _gen_attempt() -> np.ndarray | None:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    built = _paint_output(H, W)
    if built is None:
        return None
    out, bg, frame_color, piece_specs = built

    if _piece_count(out, bg) < 7:
        return None

    try:
        inp = _scatter(out, bg, frame_color, piece_specs)
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
    raise RuntimeError("failed to generate a valid e8686506 D1 input")
