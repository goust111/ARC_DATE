# Combo variant of 5961cc34 (index 31)
# combo_spec: D5/2+D9/1+D18/0
# Name: 十字形色块+单轴对称+隐式端口

# Design analysis for the fused variant
# -------------------------------------
# a) 具体化与行为追踪
#    - 输入只保留 4 种功能色：背景 8、蓝色十字块 1、底部红杆 2、黄色帽子 4；绿色显式端口完全移除。
#    - 每个蓝块现在是轴对齐十字：中段是一条整行/整列的横臂或竖臂，两端臂宽不相同，因此整体只沿一条轴对称。
#      例如一个“竖轴对称”的块会呈现为：顶部 1 宽竖臂，中央 3-4 行通栏横臂，底部 3-5 宽竖臂。
#      这时最窄臂唯一出现在顶部，所以发射方向=向上，且粗射线宽度=该顶端边宽度（1 或 2）。
#    - 行为追踪：
#      原题里，起点红线先打到蓝块，再由与蓝块 4 邻接的绿色像素决定下一段粗射线方向。
#      现在 solver 不再读颜色邻接，而是先把每个蓝块裁成局部 mask，判断它是“仅左右对称”还是“仅上下对称”，
#      再从非对称轴两端比较两只臂的末端宽度，唯一更窄的一端就是隐式端口。
#      其余 BFS 传播逻辑保持不变：命中 -> 整块变红 -> 从隐式端口继续发射 -> 继续命中新块。
# b) 复杂度校准
#    - 如果只是删掉绿色而保留原菱形，极值边会成对并列，方向不可判；因此必须把方向信息嵌回轮廓本身。
#    - 这里不放开为任意不规则形，而是限制成少量离散十字 profile：
#      端口宽度只取 1 或 2，发射方向只有四向，横/竖臂厚度与臂长也只在少数档位中变化。
#    - 这样样例里会重复出现“单轴十字 + 唯一最窄臂”的模式，视觉差异明确，但不会退化成高自由度形状分类题。
# c) 隐式假设审查（对应原 main 的关键前提）
#    - 原 Step 2 依赖“存在唯一绿色端口色且与蓝块 4 邻接”；融合后完全失效，必须删掉整段绿色检测。
#    - 原 Step 4 依赖“port_pixels = 紧邻绿色的蓝像素”；融合后改成“port_pixels = 最窄臂末端整条极值边”。
#    - 原 blob 形状默认 LR+UD 双轴对称；融合后只保留单轴对称，所以 solver 不能再把双轴对称当作默认前提。
#    - 新 solver 的额外前提是：
#      * 每个蓝组件都能被识别为单轴对称十字；
#      * 唯一最窄臂只出现在非对称轴的一个端点，且其宽度为 1 或 2；
#      * 组件仍然是单个 8 连通蓝组件。
#      generator 会显式构造并验证这些前提。
# d) 放置合法性
#    - generator 继续使用 output-first：先沿红光路放置级联链，再逆推出输入蓝块。
#    - 候选十字块只有通过 `_infer_cross_port` 校验后才会进入规格池，因此每个块都保证：
#      * 恰有一条对称轴；
#      * 中段存在连续通栏横臂/竖臂，肉眼可见是十字；
#      * 最窄臂唯一且宽度在 {1,2}。
#    - 放置时仍要求：不越界、不与已有块 8 邻接、不压住已有光路，且入射光道只在目标 entry side 处触碰该块。
# e) 视觉可观察性
#    - 从 2-4 组样例中，人类可以直接看到三件事：
#      * 绿色端口消失了；
#      * 蓝块都不再是菱形，而是明显的十字轮廓；
#      * 红光总是从“更细的一只臂”射出，而且每个块只沿一条轴对称。
#    - 这些判断只依赖可见几何，不依赖颜色编号、遍历顺序或任何人类不可见量。
# f) 可视化整洁度
#    - 继续保留稀疏布局和 1 格安全边距；干扰块至多 1 个，且完全避开所有光路。
#    - 方向信息被收进十字轮廓，不再叠加额外标记，因此图面比“蓝块 + 绿端口”更干净。
# g) 变化的可推断性
#    - 输入颜色数从 5 降到 4，没有新增颜色映射负担。
#    - 输出仍保持原题的结构性变化：被命中的块和所有光路都变红，未命中的块整体消失为背景。
#    - 方向变化由“哪条臂最窄”这个稳定视觉线索支撑，不会出现纯靠颜色记忆的不可归纳变化。
# h) 融合行为量化指标
#    - 核心指标 1：`green_pixels == 0`，排除退化回原题显式端口。
#    - 核心指标 2：每个 blob 都通过 `_infer_cross_port`，意味着它同时满足：
#      单轴对称、十字中段连续、唯一最窄臂宽度 in {1,2}。
#    - 统计口径：端口宽度 1/2、对称轴 vertical/horizontal、端口方向四向、chain_len 0-3。
# i) solver/generator 对齐检查
#    - solver 按局部 mask 的行宽/列宽 profile 判定端口；generator 只生成同一判定器可识别的 profile。
#    - solver 依赖“级联链外最多 1 个干扰块”；generator 保持这一上限。
#    - solver 对起点、BFS、终止条件的假设与原题一致；融合只替换 blob 几何推断部分，不引入隐藏排序规则。

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


# --- Colors ---
BG = 8
BLUE = 1
RED = 2
YELLOW = 4

# --- Directions ---
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
OPP = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


def _perp_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    """Coordinate perpendicular to direction d (column for vertical, row for horizontal)."""
    return rc[1] if d[0] else rc[0]


def _along_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    """Coordinate along direction d (row for vertical, column for horizontal)."""
    return rc[0] if d[0] else rc[1]


def _sort_on_ray(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    """Sort coordinates by their perpendicular value relative to direction d."""
    return sorted(coords, key=lambda p: _perp_value(p, d))


def _edge_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    """Return sorted mask-local coordinates on the extreme side facing direction d."""
    pts = np.argwhere(mask)
    axis = 0 if d in (UP, DOWN) else 1
    extreme = pts[:, axis].min() if d in (UP, LEFT) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return _sort_on_ray([(int(r), int(c)) for r, c in pts[keep]], d)


def _is_contiguous(indices: list[int]) -> bool:
    return bool(indices) and indices == list(range(indices[0], indices[-1] + 1))


def _mask_from_row_profile(profile: tuple[int, ...]) -> np.ndarray:
    """Build an UP-oriented base cross mask from row widths centered in a common bbox."""
    h = len(profile)
    w = max(profile)
    mask = np.zeros((h, w), dtype=bool)
    for r, width in enumerate(profile):
        c0 = (w - width) // 2
        mask[r, c0 : c0 + width] = True
    return mask


def _orient_up_mask(mask: np.ndarray, d: tuple[int, int]) -> np.ndarray:
    """Rotate/flip an UP-oriented cross so its hidden port faces direction d."""
    if d == UP:
        return mask.copy()
    if d == DOWN:
        return np.flipud(mask)
    if d == LEFT:
        return mask.T.copy()
    if d == RIGHT:
        return np.fliplr(mask.T).copy()
    raise ValueError(d)


def _infer_cross_port(mask: np.ndarray) -> tuple[tuple[int, int], list[tuple[int, int]], str]:
    """Infer the hidden port from a single-axis symmetric cross mask."""
    mask = np.array(mask, dtype=bool)
    h, w = mask.shape
    vert = bool(np.array_equal(mask, np.fliplr(mask)))
    horiz = bool(np.array_equal(mask, np.flipud(mask)))
    assert vert ^ horiz, "blob must have exactly one symmetry axis"

    if vert:
        row_counts = [int(x) for x in mask.sum(axis=1)]
        mid_rows = [i for i, cnt in enumerate(row_counts) if cnt == w]
        assert _is_contiguous(mid_rows), "cross middle band must be contiguous"
        assert mid_rows[0] > 0 and mid_rows[-1] < h - 1, "cross must have arms on both ends"
        top_counts = row_counts[: mid_rows[0]]
        bottom_counts = row_counts[mid_rows[-1] + 1 :]
        assert top_counts and bottom_counts
        assert len(set(top_counts)) == 1 and len(set(bottom_counts)) == 1, "arm widths must be constant per side"
        top_w = top_counts[0]
        bottom_w = bottom_counts[0]
        side_w = len(mid_rows)
        if top_w < bottom_w and top_w < side_w:
            assert top_w in (1, 2)
            return UP, _edge_offsets(mask, UP), "vertical"
        if bottom_w < top_w and bottom_w < side_w:
            assert bottom_w in (1, 2)
            return DOWN, _edge_offsets(mask, DOWN), "vertical"
        raise AssertionError("vertical cross must have one uniquely narrow end arm")

    col_counts = [int(x) for x in mask.sum(axis=0)]
    mid_cols = [i for i, cnt in enumerate(col_counts) if cnt == h]
    assert _is_contiguous(mid_cols), "cross middle band must be contiguous"
    assert mid_cols[0] > 0 and mid_cols[-1] < w - 1, "cross must have arms on both ends"
    left_counts = col_counts[: mid_cols[0]]
    right_counts = col_counts[mid_cols[-1] + 1 :]
    assert left_counts and right_counts
    assert len(set(left_counts)) == 1 and len(set(right_counts)) == 1, "arm widths must be constant per side"
    left_w = left_counts[0]
    right_w = right_counts[0]
    side_w = len(mid_cols)
    if left_w < right_w and left_w < side_w:
        assert left_w in (1, 2)
        return LEFT, _edge_offsets(mask, LEFT), "horizontal"
    if right_w < left_w and right_w < side_w:
        assert right_w in (1, 2)
        return RIGHT, _edge_offsets(mask, RIGHT), "horizontal"
    raise AssertionError("horizontal cross must have one uniquely narrow end arm")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a red ray network by activating single-axis cross blobs whose narrowest arm is a hidden port.

    Input structure:
    - bg = most common color. Blue blobs are axis-aligned crosses with no green markers; each cross has exactly one symmetry axis.
    - Start marker: a yellow cap sitting above a short red segment that touches the bottom edge.

    Rule:
    - Shoot a thin red ray upward from the start; when it reaches a blue cross, the whole cross turns red.
    - The activated cross then emits a ray from the uniquely narrowest arm tip, with width 1 or 2 depending on that arm.
    - Continue this BFS-style cascade until no new cross is hit; unhit crosses are erased to background.

    Key insight: Direction is encoded geometrically, not by color. The only arm end that is strictly narrower than the other three extremes is the implicit launch port.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect the start marker (red vertical segment with a non-red cap above) ---
    start_candidates: list[tuple[int, int, int, list[tuple[int, int]]]] = []

    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=8):
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

            if c0 != c1 or (r1 - r0 + 1) < 2:
                continue
            if r1 != H - 1:
                continue
            if any(int(g[rr, c0]) != int(col) for rr in range(r0, r1 + 1)):
                continue
            if r0 - 1 < 0:
                continue
            cap = int(g[r0 - 1, c0])
            if cap == bg or cap == int(col):
                continue

            start_candidates.append(
                (
                    int(col),
                    int(cap),
                    c0,
                    [(rr, c0) for rr in range(r0, r1 + 1)],
                )
            )

    assert len(start_candidates) == 1
    red_col, yellow_col, start_col, start_seg = start_candidates[0]

    # --- STEP 2) Identify the blob color (dominant non-bg, non-start color) ---
    counts = Counter(map(int, g.flatten()))
    other_cols = [(int(cnt), int(col)) for col, cnt in counts.items() if int(col) not in (bg, red_col, yellow_col)]
    assert other_cols
    max_count = max(cnt for cnt, _col in other_cols)
    blue_candidates = [int(col) for cnt, col in other_cols if cnt == max_count]
    assert len(blue_candidates) == 1
    blue_col = blue_candidates[0]

    out = np.full_like(g, bg)

    # --- STEP 3) Extract blue blobs and infer each implicit port from its cross geometry ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []

    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = len(blobs)
        for r, c in pts:
            blob_id[r, c] = bi

        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            local[r - r0, c - c0] = True

        port_dir, local_port_pixels, _axis = _infer_cross_port(local)
        global_port_pixels = [(r0 + r, c0 + c) for r, c in local_port_pixels]
        blobs.append(
            {
                "pts": pts,
                "ports": [{"dir": port_dir, "pixels": _sort_on_ray(global_port_pixels, port_dir)}],
            }
        )

    activated: set[int] = set()
    queue: list[int] = []

    def activate_blob_at(r: int, c: int) -> None:
        """Recolor a blob to red when first hit; enqueue it once for BFS propagation."""
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(bi)
        queue.append(bi)
        for pr, pc in blobs[bi]["pts"]:
            out[pr, pc] = red_col

    # --- STEP 4) Draw the start ray upward and activate the first blob it hits ---
    for r, c in start_seg:
        out[r, c] = red_col
    top_r = min(int(r) for r, _c in start_seg)
    assert top_r > 0 and int(g[top_r - 1, start_col]) == yellow_col
    out[top_r - 1, start_col] = red_col
    for r in range(top_r - 1, -1, -1):
        out[r, start_col] = red_col
        if int(g[r, start_col]) == blue_col:
            activate_blob_at(r, start_col)
            break

    def extend_thick(port_pixels: list[tuple[int, int]], d: tuple[int, int]) -> None:
        """Extend a ray from each tip pixel until boundary or the first blob hit in that lane."""
        dr, dc = d
        for pr, pc in port_pixels:
            r, c = pr, pc
            while True:
                r += dr
                c += dc
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[r, c] = red_col
                if int(g[r, c]) == blue_col:
                    activate_blob_at(r, c)
                    break

    # --- STEP 5) BFS propagation: activated blob -> hidden-port ray -> new activations ---
    while queue:
        bi = queue.pop(0)
        for port in blobs[bi]["ports"]:
            extend_thick(port["pixels"], port["dir"])

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"

    return out


# --------------- generator (a2_cascade, single-axis cross variant) ---------------

# Profiles are row-width tuples for UP-pointing, vertically symmetric cross blobs.
# Each has:
# - a constant narrow top arm (width 1 or 2),
# - a contiguous full-width middle band (the horizontal arm),
# - a constant wider bottom arm,
# so the result is a clear cross with exactly one symmetry axis.
UP_PROFILES = (
    (1, 1, 5, 5, 5, 3, 3),
    (1, 1, 7, 7, 7, 3, 3, 3),
    (1, 1, 1, 7, 7, 7, 5, 5),
    (2, 2, 6, 6, 6, 4, 4),
    (2, 2, 8, 8, 8, 8, 4, 4),
    (2, 2, 2, 8, 8, 8, 6, 6),
)

# --- Canvas bounds ---
H_MIN, H_MAX = 15, 27
W_MIN, W_MAX = 17, 27
START_COL_PAD = 2

# --- Start segment ---
START_H_MIN, START_H_MAX = 2, 3

# --- Chain / distractor ---
CHAIN_LENGTHS = (1, 2, 3)
CHAIN_LEN_WEIGHTS = [1, 1, 1]
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1

# --- Placement ---
MIN_GAP_FIRST = 4
MIN_GAP_CHAIN = 3

# --- Retry ---
GEN_RETRIES = 400


def _blob_spec(profile: tuple[int, ...], port_dir: tuple[int, int]) -> dict | None:
    """Build a blob specification dict for a given cross profile and hidden-port direction."""
    mask = _orient_up_mask(_mask_from_row_profile(profile), port_dir)
    inferred_dir, port_pixels, axis = _infer_cross_port(mask)
    if inferred_dir != port_dir:
        return None

    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = BLUE
    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return {
        "size": tuple(int(x) for x in mask.shape),
        "mask": mask,
        "patch": patch,
        "blue": full,
        "full": full,
        "port_dir": port_dir,
        "port_pixels": port_pixels,
        "port_width": len(port_pixels),
        "axis": axis,
    }


def _all_specs(port_dir: tuple[int, int] | None = None) -> list[dict]:
    """Generate all valid cross specs, optionally filtered by hidden-port direction."""
    dirs = [port_dir] if port_dir is not None else list(DIRS)
    out = [s for profile in UP_PROFILES for d in dirs if (s := _blob_spec(profile, d)) is not None]
    random.shuffle(out)
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    """Dilate a boolean mask by 1 pixel in all 8 directions."""
    H, W = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, dr), H + min(0, dr)
            c0, c1 = max(0, dc), W + min(0, dc)
            out[r0:r1, c0:c1] |= mask[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out


def _offset(offsets: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    """Translate local offsets to global coordinates."""
    return [(r0 + r, c0 + c) for r, c in offsets]


def _in_bounds(coords: list[tuple[int, int]], H: int, W: int) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in coords)


def _path_to_entries(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    entries: list[tuple[int, int]],
    include_origin: bool,
    H: int,
    W: int,
) -> list[tuple[int, int]] | None:
    """Trace ray lanes from starts to entry cells; return all traversed cells or None on failure."""
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for (r, c), (er, ec) in zip(starts, entries):
        if include_origin:
            cells.append((r, c))
        guard = H + W + 10
        while (r, c) != (er, ec):
            r, c = r + dr, c + dc
            guard -= 1
            if guard < 0 or not (0 <= r < H and 0 <= c < W):
                return None
            cells.append((r, c))
    return cells


def _path_to_boundary(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    """Trace ray lanes from starts to the grid boundary."""
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for r, c in starts:
        if include_origin:
            cells.append((r, c))
        while True:
            r, c = r + dr, c + dc
            if not (0 <= r < H and 0 <= c < W):
                break
            cells.append((r, c))
    return cells


def _mark_ray(out: np.ndarray, ray_mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    """Paint cells red in the output grid and record them in the ray mask."""
    paint_coords(out, cells, RED)
    paint_coords(ray_mask, cells, True)


def _try_place_along_axis(
    spec: dict,
    window: list[tuple[int, int]],
    d: tuple[int, int],
    lane_vals: list[int],
    starts: list[tuple[int, int]],
    include_origin: bool,
    min_gap: int,
    occ_pad: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    H: int,
    W: int,
) -> list[dict]:
    """Find all valid placements for a blob spec along the ray's travel axis."""
    h, w = spec["mask"].shape
    win_vals = [_perp_value(p, d) for p in window]
    edge_line = _along_value(window[0], d)
    start_line = _along_value(starts[0], d)
    is_vertical = bool(d[0])

    fixed = lane_vals[0] - win_vals[0]
    extent = H - h if is_vertical else W - w
    low = edge_line
    high = extent + edge_line
    if d in (UP, LEFT):
        high = min(high, start_line - min_gap)
    else:
        low = max(low, start_line + min_gap)

    entry_lines = list(range(low, high + 1))
    random.shuffle(entry_lines)
    candidates: list[dict] = []

    for entry_line in entry_lines:
        origin_along = entry_line - edge_line
        r0 = origin_along if is_vertical else fixed
        c0 = fixed if is_vertical else origin_along
        if is_vertical:
            entry_cells = [(entry_line, c0 + c) for _, c in window]
        else:
            entry_cells = [(r0 + r, entry_line) for r, _ in window]
        full = _offset(spec["full"], r0, c0)
        if not _in_bounds(full, H, W):
            continue
        if any(occ_pad[r, c] or ray_mask[r, c] for r, c in full):
            continue
        path = _path_to_entries(starts, d, entry_cells, include_origin, H, W)
        if path is None:
            continue
        entry_set = set(entry_cells)
        if any(blue_mask[r, c] for r, c in path if (r, c) not in entry_set):
            continue
        candidates.append(
            {
                "spec": spec,
                "r0": r0,
                "c0": c0,
                "entry": entry_cells,
                "path": path,
                "blue": _offset(spec["blue"], r0, c0),
                "full": full,
                "ports": _sort_on_ray(_offset(spec["port_pixels"], r0, c0), spec["port_dir"]),
            }
        )

    return candidates


def _candidate_next_blob(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> dict | None:
    """Find a random valid placement for the next blob in the cascade chain."""
    starts = _sort_on_ray(starts, d)
    thickness = len(starts)
    lane_vals = [_perp_value(p, d) for p in starts]
    lane_delta = [v - lane_vals[0] for v in lane_vals]
    entry_side = OPP[d]
    occ_pad = _dilate8(occupied)
    min_gap = MIN_GAP_FIRST if include_origin else MIN_GAP_CHAIN
    spec_candidates: list[list[dict]] = []

    for spec in _all_specs():
        if spec["port_dir"] == entry_side:
            continue
        candidates: list[dict] = []
        edge = _edge_offsets(spec["mask"], entry_side)
        if len(edge) < thickness:
            continue
        for k in range(len(edge) - thickness + 1):
            window = edge[k : k + thickness]
            win_delta = [_perp_value(p, d) - _perp_value(window[0], d) for p in window]
            if win_delta != lane_delta:
                continue
            candidates.extend(
                _try_place_along_axis(
                    spec,
                    window,
                    d,
                    lane_vals,
                    starts,
                    include_origin,
                    min_gap,
                    occ_pad,
                    blue_mask,
                    ray_mask,
                    H,
                    W,
                )
            )
        if candidates:
            spec_candidates.append(candidates)

    if not spec_candidates:
        return None
    return random.choice(random.choice(spec_candidates))


def _place_blob(inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, blob: dict) -> None:
    """Paste a blob's blue patch onto the input grid and update occupancy masks."""
    paste(inp, blob["spec"]["patch"], blob["r0"], blob["c0"], transparent=BG)
    for r, c in blob["full"]:
        occupied[r, c] = True
    for r, c in blob["blue"]:
        blue_mask[r, c] = True


def _place_distractor(
    inp: np.ndarray,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> bool:
    """Try to place a single distractor blob in free space. Return success."""
    H, W = inp.shape
    occ_pad = _dilate8(occupied)
    candidates: list[dict] = []
    for spec in _all_specs():
        h, w = spec["mask"].shape
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                full = _offset(spec["full"], r0, c0)
                if any(occ_pad[r, c] or ray_mask[r, c] for r, c in full):
                    continue
                candidates.append(
                    {
                        "spec": spec,
                        "r0": r0,
                        "c0": c0,
                        "full": full,
                        "blue": _offset(spec["blue"], r0, c0),
                    }
                )
    if not candidates:
        return False
    _place_blob(inp, occupied, blue_mask, random.choice(candidates))
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid fused-variant input grid."""
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    # Place start segment (yellow cap + red stem touching the bottom edge).
    start_h = random.randint(START_H_MIN, START_H_MAX)
    cap_r = H - start_h - 1
    start_col = random.randint(START_COL_PAD, W - START_COL_PAD - 1)
    inp[cap_r, start_col] = YELLOW
    out[cap_r, start_col] = RED
    ray_mask[cap_r, start_col] = True
    occupied[cap_r, start_col] = True
    for r in range(cap_r + 1, H):
        inp[r, start_col] = RED
        out[r, start_col] = RED
        occupied[r, start_col] = True

    # Decide chain length and distractor count.
    chain_len = random.choices(CHAIN_LENGTHS, weights=CHAIN_LEN_WEIGHTS)[0]
    max_dist = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - chain_len)
    num_distractors = random.randint(0, max_dist)

    # Build the cascade chain in the output grid.
    starts = [(cap_r, start_col)]
    direction = UP
    include_origin = True
    for _ in range(chain_len):
        blob = _candidate_next_blob(H, W, starts, direction, include_origin, occupied, blue_mask, ray_mask)
        if blob is None:
            return None
        _mark_ray(out, ray_mask, blob["path"])
        _place_blob(inp, occupied, blue_mask, blob)
        paint_coords(out, blob["blue"], RED)
        starts = blob["ports"]
        direction = blob["spec"]["port_dir"]
        include_origin = False

    # Final ray continues to the boundary.
    final_path = _path_to_boundary(starts, direction, include_origin, H, W)
    if any(blue_mask[r, c] for r, c in final_path):
        return None
    _mark_ray(out, ray_mask, final_path)

    # Place distractors away from the red network.
    for _ in range(num_distractors):
        if not _place_distractor(inp, occupied, blue_mask, ray_mask):
            return None

    # Oracle check: the solver must exactly reconstruct the planned output.
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 combo input")
