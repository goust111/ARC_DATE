# Variant of e8686506
# Dimension: D4 — 对称约束：输出的对称性类型 (match_cond)
# Change: 输出具有严格的左右（垂直轴）镜像对称性... → 上下（水平轴）镜像对称
#
# Design analysis
# - concretization:
#   - 原题把所有碎片拼成一个稠密矩形，并要求矩形关于垂直轴左右镜像。
#   - 本变体仍然是同一个“散落碎片 -> 拼成矩形”的任务，但矩形改为关于水平轴上下镜像。
#   - 可见结构上，输入里仍有 1-2 个保持相对位置的 frame 碎片，其 union bbox 就是输出矩形；
#     非 frame 碎片仍散落在外部。差别在于 frame 的轮廓现在由“上半带状 zigzag + 向下镜像”
#     生成，而不是“左半带状 zigzag + 向右镜像”。
# - behavior_trace:
#   - 原逻辑在 `_assemble()` 里对每个候选落点检查 `board[r, W-1-c]` 是否同色，
#     因而每个碎片只能被放到一个左右镜像一致的位置。
#   - 变体里同一检查必须改成 `board[H-1-r, c]`，约束的是上下镜像位置。
#   - 对 frame 识别也是同样的轴变更：原先检测 union bitmap 是否等于 `bmp[:, ::-1]`，
#     现在必须检测是否等于 `bmp[::-1, :]`。
# - complexity_calibration:
#   - 只改镜像轴会让少量样本意外同时具备 LR/UD 双对称，从而削弱视觉差异。
#   - 为了让 D4 变化肉眼可见，generator 额外拒绝 LR 对称输出；这不改变核心任务，
#     只是避免退化成“新旧规则在样例上看起来一样”。
#   - frame 双色分界从“上下半区”改为“左右半区”，否则 frame 自己就会违反 UD 对称。
# - hidden_assumptions_audit:
#   - `_build_requirements()` 里的四角 singleton 约束与轴无关，可以保留。
#   - 顶/底横条锚在 UD 版本中若出现两条，就必须同色；原题里允许不同色只是因为 LR 对称
#     不会冲突。这里显式断言同色，generator 通过构造保证这一点。
#   - 左/右竖线锚与新轴不冲突，仍可保留。
#   - 原 generator 的中心 cap 依赖“宽度恒为奇数，中心列存在”；变体若直接照搬会出错，
#     因为高度不总是奇数。这里仅在 `H` 为奇数时才添加中心行 cap。
#   - 另外，生成器只采样偶数高度候选，这样上下镜像后轴附近最多把一个小 chunk 翻倍到面积 6，
#     不会像奇数高度那样通过中心行把镜像块连成更大的组件。
# - placement_legality:
#   - 散布阶段仍先把 frame bbox 整体占住，再把非 frame 碎片放到外部，最后仅恢复 frame
#     像素，所以不会和填充碎片重叠。
#   - 上下镜像填洞时只遍历每个空洞的“规范上半部分”单元，再一次性向下镜像，避免同一对
#     镜像格被不同 chunk 反复覆盖。
# - observability:
#   - 看样例时，人类会看到 frame 在输入中保留为一个横向 zigzag 轮廓，输出则沿水平轴镜像，
#     顶半部与底半部互为镜像，而左右两侧通常不同色/不同孔洞结构，因此不会误读成原题。
# - visual_cleanliness:
#   - 仍保留单色碎片、背景留白和 frame 成组放置的布局，避免新旧元素叠加。
#   - 颜色映射不依赖颜色 id，也不按颜色数值排序决定几何行为。

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


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


def _build_requirements(
    H: int,
    W: int,
    comps_by_color: dict[int, list[list[tuple[int, int]]]],
) -> np.ndarray:
    """
    Build per-cell required color constraints (req[r,c] == -1 means unconstrained).

    Heuristics retained from the seed task:
      - A color with exactly 4 singleton components marks the four corners.
      - Full horizontal bars of length W-2, when present, anchor the top/bottom edges.
      - Two identical vertical line components can anchor the centered left/right edges.
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
            assert top_col == bot_col, "top/bottom edge bars must share a color under UD symmetry"
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
    disallow_touch: set[int] | None = None,
) -> np.ndarray:
    board = np.full((H, W), -1, dtype=int)
    counts = [int(t["cnt"]) for t in types]

    def mirror_ok(r: int, c: int, col: int) -> bool:
        mr = H - 1 - r
        v = board[mr, c]
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
            if disallow_touch and col in disallow_touch:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < H and 0 <= cc < W and board[rr, cc] == col:
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
    assert np.array_equal(board, board[::-1, :]), "assembled board must be UD-symmetric"
    return board


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered fragments into a dense rectangle whose top half mirrors its bottom half.

    Input structure:
    - The most frequent color is background; every other color forms small disconnected fragments.
    - A preserved 8-connected frame fragment group still reveals the output bbox, and optional anchors
      may appear as four corner singletons, matching top/bottom edge bars, or paired side lines.

    Rule:
    - Extract every non-background connected component as a jigsaw piece and infer the rectangle size
      from total area.
    - Pack the pieces into that rectangle so the final picture is an exact horizontal-axis mirror and
      different pieces of the same color never become 4-connected neighbors.

    Key insight:
    - The frame fixes absolute cells, while the no-same-color-touching rule preserves component identity;
      once the symmetry axis is recognized as horizontal rather than vertical, the remaining placements are forced.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

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

    from collections import Counter

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

    def _is_ud_symmetric(parts: list[tuple[int, np.ndarray]], h: int, w: int) -> bool:
        all_pts = np.concatenate([p for _, p in parts])
        r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
        bmp = np.zeros((h, w), dtype=bool)
        for _, pts in parts:
            for r, c in pts:
                bmp[int(r) - r0, int(c) - c0] = True
        return np.array_equal(bmp, bmp[::-1, :])

    frame_parts: list[tuple[int, np.ndarray]] = []
    for i, (c1, p1) in enumerate(comps8_all):
        r0, c0, r1, c1b = bbox_coords(p1)
        if r1 - r0 + 1 == H and c1b - c0 + 1 == W:
            if _is_ud_symmetric([(c1, p1)], H, W):
                frame_parts = [(c1, p1)]
                break
        for j in range(i + 1, len(comps8_all)):
            c2, p2 = comps8_all[j]
            union = np.concatenate([p1, p2])
            ur0, uc0, ur1, uc1 = bbox_coords(union)
            if ur1 - ur0 + 1 == H and uc1 - uc0 + 1 == W:
                if _is_ud_symmetric([(c1, p1), (c2, p2)], H, W):
                    frame_parts = [(c1, p1), (c2, p2)]
                    break
        if frame_parts:
            break
    assert frame_parts, "input must contain 1-2 UD-symmetric pieces (8-conn) whose union bbox equals output dims"

    req = _build_requirements(H, W, comps_by_color)

    all_frame_pts = np.concatenate([p for _, p in frame_parts])
    fr0, fc0 = int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())
    for col, pts in frame_parts:
        for r, c in pts:
            or_, oc = int(r) - fr0, int(c) - fc0
            if 0 <= or_ < H and 0 <= oc < W:
                req[or_, oc] = col

    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    sol = _assemble(types, H, W, req, disallow_touch)
    return sol.astype(int)


# --------------- generator (o5x) ---------------

# === CFG: Output dimensions ===
_AREA_MIN = 25
_AREA_MAX = 50
_WIDTH_REQUIRED = 5
_HEIGHT_MAX = 10

# === CFG: Color and palette ===
_N_COLORS_MIN = 4
_N_COLORS_MAX = 6
_TWO_FRAME_MIN_H = 6
_TWO_FRAME_MIN_COLORS = 4
_TWO_FRAME_PROB = 0.0
_CENTER_CAP_PROB = 0.5

# === CFG: Piece constraints ===
_MAX_PIECE_AREA = 6
_MAX_CHUNK_SIZE = 3
_PIECE_COUNT_MIN = 9
_PIECE_COUNT_MAX = 17

# === CFG: Canvas / scatter ===
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1

# === CFG: Retry ===
_GEN_MAX_RETRIES = 500


def _valid_dims():
    results = []
    for area in range(_AREA_MIN, _AREA_MAX + 1):
        dims = [(area // w, w) for w in range(1, area + 1) if area % w == 0 and area // w >= w]
        best_gap = min(abs(h - w) for h, w in dims)
        best = [(h, w) for h, w in dims if abs(h - w) == best_gap]
        if len(best) == 1:
            h, w = best[0]
            if w == _WIDTH_REQUIRED and h <= _HEIGHT_MAX and h % 2 == 0:
                results.append((h, w))
    return results


_VALID_DIMS = _valid_dims()
_DIM_WEIGHTS = [h * w for h, w in _VALID_DIMS]


def _zigzag_frame(W: int) -> tuple[set[tuple[int, int]], int, int]:
    """Generate a 2xW top-band zigzag path that will later mirror downward."""
    cells = set()
    start_r = random.randint(0, 1)
    r = start_r
    for c in range(W):
        cells.add((r, c))
        if c == W - 1:
            break
        move = random.choice(["right", "vert", "diag"])
        if move == "right":
            pass
        elif move == "vert":
            cells.add((1 - r, c))
            r = 1 - r
        else:
            r = 1 - r
    return cells, start_r, r


def _canonical_ud_cells(hole: np.ndarray, H: int) -> list[tuple[int, int]]:
    return sorted((int(r), int(c)) for r, c in hole if int(r) <= H - 1 - int(r))


def _split_connected_chunks(cells: list[tuple[int, int]], cap: int) -> list[list[tuple[int, int]]]:
    remaining = set(cells)
    chunks: list[list[tuple[int, int]]] = []
    while remaining:
        start = min(remaining)
        stack = [start]
        chunk = []
        while stack and len(chunk) < cap:
            r, c = stack.pop()
            if (r, c) not in remaining:
                continue
            remaining.remove((r, c))
            chunk.append((r, c))
            nbrs = []
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (rr, cc) in remaining:
                    nbrs.append((rr, cc))
            stack.extend(sorted(nbrs, reverse=True))
        chunks.append(sorted(chunk))
    return chunks


def _adjacent_colors(grid: np.ndarray, H: int, W: int, chunk: list[tuple[int, int]]) -> set[int]:
    cells = set(chunk)
    cells |= {(H - 1 - r, c) for r, c in chunk}
    colors = set()
    for r, c in cells:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in cells and grid[rr, cc] != -1:
                colors.add(int(grid[rr, cc]))
    return colors


def _fill_holes(grid: np.ndarray, H: int, W: int, frame1: int, frame2: int, split: int, fill_pool: list[int]) -> None:
    """Fill non-frame cells with mirrored chunks whose local adjacency forbids same-color merging."""
    holes = components_mask(grid == -1, connectivity=4)
    chunk_index = 0
    for hole in holes:
        cells = _canonical_ud_cells(hole, H)
        chunks = _split_connected_chunks(cells, _MAX_CHUNK_SIZE)
        for chunk in chunks:
            fc_set = {frame1 if c < split else frame2 for _, c in chunk}
            avoid = fc_set | _adjacent_colors(grid, H, W, chunk)
            preferred = fill_pool[chunk_index % len(fill_pool)]
            ordered = [preferred] + [c for c in fill_pool if c != preferred]
            opts = [c for c in ordered if c not in avoid]
            col = opts[0] if opts else ordered[0]
            for r, c in chunk:
                grid[r, c] = col
                grid[H - 1 - r, c] = col
            chunk_index += 1


def _build_output(H: int, W: int) -> tuple[np.ndarray | None, set[int]]:
    """Build an abstract UD-symmetric output grid with a horizontal zigzag frame."""
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
    fill_pool = palette[ci:]
    if not fill_pool:
        return None, set()

    frame_band, start_r, end_r = _zigzag_frame(W)
    split = W // 2 if frame1 != frame2 else W

    grid = np.full((H, W), -1, dtype=int)
    for r, c in frame_band:
        fc = frame1 if c < split else frame2
        grid[r, c] = fc
        grid[H - 1 - r, c] = fc

    if H % 2 == 1:
        mid = H // 2
        if start_r == 1 and random.random() < _CENTER_CAP_PROB:
            grid[mid, 0] = frame1 if 0 < split else frame2
        if end_r == 1 and random.random() < _CENTER_CAP_PROB:
            grid[mid, W - 1] = frame1 if (W - 1) < split else frame2

    _fill_holes(grid, H, W, frame1, frame2, split, fill_pool)

    if np.any(grid < 0):
        return None, set()
    frame_set = {frame1}
    if frame2 != frame1:
        frame_set.add(frame2)
    return grid, frame_set


def _make_output() -> tuple[np.ndarray | None, int, set[int]]:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    out, abstract_fcs = _build_output(H, W)
    if out is None:
        return None, 0, set()
    bg = random.randint(0, 9)
    abstract = list(map(int, np.unique(out)))
    random.shuffle(abstract)
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < len(abstract):
        return None, 0, set()
    cmap = {a: available[i] for i, a in enumerate(abstract)}
    real_fcs = {cmap[a] for a in abstract_fcs}
    return np.vectorize(lambda x: cmap[int(x)])(out), bg, real_fcs


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


def _scatter(out: np.ndarray, bg: int, fcs: set[int]) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

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
    return canvas


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _gen_attempt() -> np.ndarray | None:
    out, bg, fcs = _make_output()
    if out is None:
        return None
    if not np.array_equal(out, out[::-1, :]):
        return None
    if np.array_equal(out, out[:, ::-1]):
        return None
    if not (_N_COLORS_MIN <= len(np.unique(out)) <= _N_COLORS_MAX):
        return None
    pc = _piece_count(out)
    if not (_PIECE_COUNT_MIN <= pc <= _PIECE_COUNT_MAX):
        return None
    if any(
        len(pts) > _MAX_PIECE_AREA
        for col in np.unique(out)
        for pts in components_mask(out == col, connectivity=4)
    ):
        return None
    try:
        inp = _scatter(out, bg, fcs)
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
    raise RuntimeError("failed to generate a valid e8686506 input")
