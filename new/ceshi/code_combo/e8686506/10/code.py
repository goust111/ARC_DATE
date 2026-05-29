# Combo variant of e8686506 (index 10)
# combo_spec: D3/1+D10/1
# Name: 填入框架洞中+矩形环框架

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste, random_free_location


# Design analysis
# a) 具体化与行为追踪:
#    - 原题先把全部非背景碎片当作待拼块，用总面积推回一个致密矩形，再在整块矩形里做左右镜像拼装。
#    - 本融合版输入里直接给出一个已经摆正的完整矩形环；它四边闭合、内部留空，其他碎片都散落在环外。
#    - 具体例子: 若环的 bbox 是 8x5，则输入中这块大对象只占 22 个周边格，内部 6x3 是纯背景洞；外部散落的小块需要被放回
#      这个 6x3 空腔，而不是再去猜整个 8x5 的外轮廓。相比原题，推理入口从“矩形尺寸反推”改成“框架定位 -> 洞尺寸读取”。
# b) 复杂度校准:
#    - 如果洞只是一个完全同色的大矩形，视觉上会退化成“把若干条带塞回去”；因此生成器把 3 列内腔切成若干 1/2 行高的带，
#      并在每带中随机选择“整条 3xk”或“左右 1xk + 中间 1xk”的模式，让外部碎片既重复又不至于混乱。
#    - 腔体宽度固定为 3，可保证 solver 的搜索空间小；高度在 5..8 之间变化，能稳定出现 7-12 个组件，视觉上也明显是“填洞”。
# c) 隐式假设审查:
#    - 原 `main()` 的 `_factors(total_area)` 不再成立，因为输出尺寸直接由环 bbox 决定；继续保留会把环也当作普通拼块处理。
#    - 原角点/横条/竖条锚点都服务于外边界约束；融合后边界已经被完整环固定，再读取这些启发式只会制造伪约束，因此全部移除。
#    - 原“1x1 只能落在外边界”同样失效；现在单格碎片允许落在环内腔体中。
#    - 新 solver 引入的新前提是“存在唯一完整矩形环，且其 bbox 内除了环色就是背景”；生成器会严格构造并校验这一点。
# d) 放置合法性:
#    - 生成器先把整块环 bbox 预留出来，再把内部填充碎片全部散落到 bbox 外，最后仅把环周边贴回，因此不会有碎片越界进洞。
#    - 输出内腔只由 1 或 2 行高的带状块组成，每块面积最多 6，符合原题碎片尺度，也避免随机散布时难以安放。
# e) 视觉可观察性:
#    - 输入最显眼的对象总是一块“矩形环带洞”，输出则是同一矩形被内部补满，框架位置不变，因果关系可以直接肉眼跟踪。
#    - 颜色仅作为碎片身份，不依赖颜色编号排序；同一颜色可在多个非相邻带中复用，但位置线索足以支撑观察。
# f) 可视化整洁度:
#    - 框架永远是一块大单色对象，碎片永远在框外，输入层次清晰。
#    - 输出里只有“环边框”和“内腔填充”两层结构，没有多重标记叠加，5 列宽也能保持可读。
# g) 变化的可推断性:
#    - 颜色数量控制在 4-6；相邻带在每一列都禁止使用同色，避免人类看到大块无提示同色合并。
#    - 规则可概括为: “先认出完整矩形环，再把外部碎片拼回环内部”，不需要借助任何不可见元数据。
# h) solver 假设与 generator 对应保证:
#    - 假设唯一完整矩形环存在: 生成器只放一块周边闭合的单色大环，面积也远大于任一小块。
#    - 假设环 bbox 内不存在预填充碎片: 通过先封锁 bbox、后贴回环周边来保证。
#    - 假设外部碎片恰好能铺满内腔并保持左右对称: 输出先构好，再裁块散布，并用 `main()` 反验。
# i) 量化判据:
#    - 对输入检测最大环 bbox，统计其中背景内腔面积 `hole_area` 与 bbox 外碎片面积 `outside_fill_area`。本融合版要求
#      `hole_area == outside_fill_area > 0`，且 bbox 内非背景只出现在完整矩形环周边；这正是“封闭环 + 洞填充”的核心信号。


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _ring_coords(H: int, W: int) -> tuple[tuple[int, int], ...]:
    cells = []
    for r in range(H):
        for c in range(W):
            if r == 0 or r == H - 1 or c == 0 or c == W - 1:
                cells.append((r, c))
    return tuple(cells)


def _is_full_ring(coords: tuple[tuple[int, int], ...], H: int, W: int) -> bool:
    return len(coords) == (2 * H + 2 * W - 4) and coords == _ring_coords(H, W)


def _detect_ring(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int], np.ndarray]:
    ring_candidates: list[tuple[int, int, tuple[int, int, int, int], np.ndarray]] = []
    for col in sorted(set(map(int, np.unique(g))) - {int(bg)}):
        for pts in components_mask(g == int(col), connectivity=4):
            arr = np.array(pts, dtype=int)
            h, w, coords = _piece_sig(arr)
            if h >= 3 and w >= 3 and _is_full_ring(coords, h, w):
                ring_candidates.append((len(arr), int(col), bbox_coords(arr), arr))

    assert ring_candidates, "input must contain a complete rectangular ring"
    max_area = max(area for area, _, _, _ in ring_candidates)
    best = [(col, bbox, pts) for area, col, bbox, pts in ring_candidates if area == max_area]
    assert len(best) == 1, "rectangular ring must be unique"
    return best[0]


def _extract_fill_types(
    g: np.ndarray,
    bg: int,
    ring_bbox: tuple[int, int, int, int],
) -> tuple[list[dict], set[int], int]:
    r0, c0, r1, c1 = ring_bbox
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)

    pieces: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    color_piece_count: dict[int, int] = {}
    total_area = 0
    for col in sorted(set(map(int, np.unique(outside))) - {int(bg)}):
        comps = components_mask(outside == int(col), connectivity=4)
        if comps:
            color_piece_count[int(col)] = len(comps)
        for pts in comps:
            arr = np.array(pts, dtype=int)
            _, _, coords = _piece_sig(arr)
            pieces.append((int(col), coords))
            total_area += len(arr)

    assert pieces, "input must contain fill pieces outside the ring"

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

    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    return types, disallow_touch, int(total_area)


def _solve_cavity(
    types: list[dict],
    H: int,
    W: int,
    ring_col: int,
    disallow_touch: set[int],
) -> np.ndarray:
    board = np.full((H, W), -1, dtype=int)
    ring_cells = _ring_coords(H, W)
    for r, c in ring_cells:
        board[r, c] = int(ring_col)

    counts = [int(t["cnt"]) for t in types]

    def mirror_ok(r: int, c: int, col: int) -> bool:
        mc = W - 1 - c
        v = int(board[r, mc])
        return v == -1 or v == col

    def can_place(ti: int, r0: int, c0: int) -> bool:
        t = types[ti]
        col = int(t["col"])
        for dr, dc in t["coords"]:
            r, c = r0 + dr, c0 + dc
            if not (1 <= r < H - 1 and 1 <= c < W - 1):
                return False
            if board[r, c] != -1:
                return False
            if not mirror_ok(r, c, col):
                return False
            if col in disallow_touch:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < H and 0 <= cc < W and board[rr, cc] == col:
                        return False
        return True

    def place(ti: int, r0: int, c0: int) -> list[tuple[int, int]]:
        col = int(types[ti]["col"])
        changed = []
        for dr, dc in types[ti]["coords"]:
            r, c = r0 + dr, c0 + dc
            board[r, c] = col
            changed.append((r, c))
        return changed

    def unplace(changed: list[tuple[int, int]]) -> None:
        for r, c in changed:
            board[r, c] = -1

    def placements_for(ti: int) -> list[tuple[int, int]]:
        t = types[ti]
        out = []
        for r0 in range(1, H - int(t["h"])):
            for c0 in range(1, W - int(t["w"])):
                if can_place(ti, r0, c0):
                    out.append((r0, c0))
        return out

    def dfs() -> bool:
        if all(c == 0 for c in counts):
            return True

        best_ti = None
        best_places = None
        for ti in range(len(types)):
            if counts[ti] <= 0:
                continue
            places = placements_for(ti)
            if not places:
                return False
            if best_places is None or len(places) < len(best_places):
                best_ti = ti
                best_places = places
                if len(best_places) == 1:
                    break

        assert best_ti is not None and best_places is not None
        for r0, c0 in best_places:
            counts[best_ti] -= 1
            changed = place(best_ti, r0, c0)
            if dfs():
                return True
            unplace(changed)
            counts[best_ti] += 1
        return False

    assert dfs(), "fill pieces cannot tile the ring cavity"
    assert not np.any(board < 0), "ring cavity must be fully filled"
    assert np.array_equal(board, board[:, ::-1]), "assembled result must be LR-symmetric"
    return board


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a left-right symmetric rectangle by filling the hole of a closed rectangular ring.

    Input structure:
    - The background is the most common color.
    - One large non-background component is a complete rectangular ring already sitting in its final pose.
    - Every other non-background component is a detached fill fragment scattered outside that ring's bounding box.

    Rule:
    - Find the unique full rectangular ring and use its bounding box as the output size.
    - Keep that ring fixed, treat the background cells inside it as the empty cavity, and place all outside fragments strictly into that cavity.
    - The completed rectangle must remain exactly left-right symmetric, and repeated-color fragments may not merge by 4-neighbor contact.

    Key insight:
    - The job is no longer “discover the rectangle from all pieces”; the rectangle is already exposed by the closed ring, so the real task is to read the cavity and tile only that interior void with the scattered fragments.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    ring_col, ring_bbox, ring_pts = _detect_ring(g, bg)
    r0, c0, r1, c1 = ring_bbox
    sub = crop_bbox(g, ring_bbox)
    H, W = sub.shape

    frame_mask = sub == int(ring_col)
    hole_mask = sub == int(bg)
    assert np.all(frame_mask | hole_mask), "ring bbox may contain only ring color and background"
    assert _is_full_ring(_piece_sig(ring_pts)[2], H, W), "detected frame must be a full rectangular ring"
    assert hole_mask.any(), "ring must enclose a non-empty cavity"
    assert not hole_mask[0].any() and not hole_mask[-1].any()
    assert not hole_mask[:, 0].any() and not hole_mask[:, -1].any()
    assert np.array_equal(frame_mask, frame_mask[:, ::-1]), "ring must be LR-symmetric"
    assert np.array_equal(hole_mask, hole_mask[:, ::-1]), "cavity must be LR-symmetric"

    types, disallow_touch, fill_area = _extract_fill_types(g, bg, ring_bbox)
    assert fill_area == int(hole_mask.sum()), "outside fill area must match the cavity area"

    sol = _solve_cavity(types, H, W, ring_col, disallow_touch)
    return sol.astype(int)


# --------------- generator (combo) ---------------

# === CFG: Output dimensions ===
_OUT_W = 5
_OUT_H_CHOICES = [7, 8, 9, 10]

# === CFG: Color and palette ===
_N_COLORS_MIN = 4
_N_COLORS_MAX = 6
_MIN_USED_FILL_COLORS = 3

# === CFG: Piece constraints ===
_MAX_PIECE_AREA = 6
_PIECE_COUNT_MIN = 7
_PIECE_COUNT_MAX = 12
_MIN_SPLIT_BANDS = 2

# === CFG: Canvas / scatter ===
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1

# === CFG: Retry ===
_GEN_MAX_RETRIES = 400


def _random_partition(total: int) -> list[int]:
    parts = []
    rem = int(total)
    while rem > 0:
        if rem == 1:
            step = 1
        else:
            step = random.choice([1, 2])
            if step > rem:
                step = 1
        parts.append(step)
        rem -= step
    random.shuffle(parts)
    return parts


def _band_modes(heights: list[int]) -> list[str] | None:
    idxs = list(range(len(heights)))
    random.shuffle(idxs)
    modes = ["bar"] * len(heights)
    split_count = 0
    for i in idxs:
        if random.random() < 0.6:
            modes[i] = "split"
            split_count += 1
    if split_count < _MIN_SPLIT_BANDS:
        for i in idxs:
            if modes[i] == "bar":
                modes[i] = "split"
                split_count += 1
                if split_count >= _MIN_SPLIT_BANDS:
                    break
    piece_count = 1 + sum(3 if mode == "split" else 1 for mode in modes)
    if not (_PIECE_COUNT_MIN <= piece_count <= _PIECE_COUNT_MAX):
        return None
    return modes


def _choose_triplet(
    fill_pool: list[int],
    prev_triplet: tuple[int, int, int] | None,
    mode: str,
) -> tuple[int, int, int] | None:
    candidates = []
    if mode == "bar":
        for col in fill_pool:
            trip = (col, col, col)
            if prev_triplet is not None and any(trip[k] == prev_triplet[k] for k in range(3)):
                continue
            candidates.append(trip)
    else:
        for side in fill_pool:
            for center in fill_pool:
                if center == side:
                    continue
                trip = (side, center, side)
                if prev_triplet is not None and any(trip[k] == prev_triplet[k] for k in range(3)):
                    continue
                candidates.append(trip)
    if not candidates:
        return None
    random.shuffle(candidates)
    return candidates[0]


def _build_output(H: int, W: int) -> np.ndarray | None:
    n_colors = random.randint(_N_COLORS_MIN, _N_COLORS_MAX)
    palette = list(range(n_colors))
    random.shuffle(palette)
    ring_col = palette[0]
    fill_pool = palette[1:]
    if len(fill_pool) < _MIN_USED_FILL_COLORS:
        return None

    inner_h = H - 2
    heights = _random_partition(inner_h)
    modes = _band_modes(heights)
    if modes is None:
        return None

    out = np.full((H, W), ring_col, dtype=int)
    used_fill: set[int] = set()
    prev_triplet = None
    rr = 1
    for h, mode in zip(heights, modes):
        trip = _choose_triplet(fill_pool, prev_triplet, mode)
        if trip is None:
            return None
        for dr in range(h):
            out[rr + dr, 1] = trip[0]
            out[rr + dr, 2] = trip[1]
            out[rr + dr, 3] = trip[2]
        prev_triplet = trip
        used_fill.update(trip)
        rr += h

    if rr != H - 1:
        return None
    if len(used_fill) < _MIN_USED_FILL_COLORS:
        return None
    if not np.array_equal(out, out[:, ::-1]):
        return None
    if not (_N_COLORS_MIN <= len(np.unique(out)) <= _N_COLORS_MAX):
        return None
    return out


def _remap_colors(out: np.ndarray) -> tuple[np.ndarray | None, int]:
    bg = random.randint(0, 9)
    abstract = sorted(map(int, np.unique(out)))
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < len(abstract):
        return None, 0
    cmap = {a: available[i] for i, a in enumerate(abstract)}
    return np.vectorize(lambda x: cmap[int(x)])(out), int(bg)


def _extract_patches(out: np.ndarray, bg: int) -> tuple[np.ndarray, list[np.ndarray]]:
    ring_col = int(out[0, 0])
    ring_patch = np.full(out.shape, bg, dtype=int)
    ring_mask = np.zeros(out.shape, dtype=bool)
    ring_mask[0, :] = True
    ring_mask[-1, :] = True
    ring_mask[:, 0] = True
    ring_mask[:, -1] = True
    ring_patch[ring_mask] = ring_col

    inner = out.copy()
    inner[ring_mask] = bg
    patches = []
    for col in sorted(set(map(int, np.unique(inner))) - {int(bg)}):
        for pts in components_mask(inner == int(col), connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in pts:
                patch[int(r) - r0, int(c) - c0] = int(col)
            patches.append(patch)

    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), p.shape[0], p.shape[1], random.random()))
    return ring_patch, patches


def _scatter(out: np.ndarray, bg: int) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("ring bbox does not fit canvas")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1
    ring_patch, patches = _extract_patches(out, bg)

    for patch in patches:
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    region[:] = bg
    paste(canvas, ring_patch, off_r, off_c, transparent=bg)
    return canvas


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == int(col), connectivity=4)) for col in np.unique(out))


def _is_combo_output(out: np.ndarray) -> bool:
    H, W = out.shape
    if W != _OUT_W or H not in _OUT_H_CHOICES:
        return False
    ring_col = int(out[0, 0])
    if not np.all(out[0, :] == ring_col):
        return False
    if not np.all(out[-1, :] == ring_col):
        return False
    if not np.all(out[:, 0] == ring_col):
        return False
    if not np.all(out[:, -1] == ring_col):
        return False
    return np.all(out[1:-1, 1:-1] != ring_col)


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_OUT_H_CHOICES)
    out = _build_output(H, _OUT_W)
    if out is None:
        return None
    out, bg = _remap_colors(out)
    if out is None:
        return None

    if not _is_combo_output(out):
        return None
    if not np.array_equal(out, out[:, ::-1]):
        return None
    if not (_N_COLORS_MIN <= len(np.unique(out)) <= _N_COLORS_MAX):
        return None
    if not (_PIECE_COUNT_MIN <= _piece_count(out) <= _PIECE_COUNT_MAX):
        return None

    ring_col = int(out[0, 0])
    for col in np.unique(out):
        for pts in components_mask(out == int(col), connectivity=4):
            h, w, coords = _piece_sig(np.array(pts, dtype=int))
            if int(col) == ring_col and h == out.shape[0] and w == out.shape[1] and _is_full_ring(coords, h, w):
                continue
            if len(pts) > _MAX_PIECE_AREA:
                return None

    try:
        inp = _scatter(out, bg)
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
    raise RuntimeError("failed to generate a valid combo e8686506 input")
