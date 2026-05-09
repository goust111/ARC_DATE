# Variant of 5961cc34
# Dimension: D5 — 色块端口结构 (src)
# Change: 每个色块恰好一条边有1-2个相邻绿色像素作为端口，端口方向决定射线发射方向... → 隐式端口

# Design analysis for the D5 variant
# ----------------------------------
# a) 具体化与行为追踪
#    - 输入里不再有绿色端口色。保留背景、蓝色块、底部黄帽+红杆起点。
#    - 每个蓝色块不再是四向完全对称的菱形/八边形，而是“单尖端”的风筝/箭头轮廓：
#      四个极值边里恰好一边最窄，长度为 1 或 2，这一窄边就是隐式发射口。
#    - 输出行为仍是同一条 BFS 级联红色光路：起点向上打到第一个蓝块，蓝块整体变红，
#      然后从其最窄边继续发射等宽红射线，命中新块后继续传播。
#    - 小例子追踪：
#      原题里，一个上尖块若要向右发射，需要在其右侧额外放 1-2 个绿色像素；solver 先找绿像素，
#      再推出 “右发射”。本变体里，这个块本身就做成“右侧最尖”的轮廓，solver 不再查颜色邻接，
#      而是比较四个极值边上的像素数，唯一最短边=右边，因此直接得到 “向右发射”。
#      差异集中在“端口识别”环节；起点检测、命中激活、BFS 传播整体保持原逻辑。
# b) 复杂度校准
#    - 如果单纯删除绿色而保留原对称菱形，则上下或左右会出现并列尖端，方向不可判。
#      因此必须给块加入单尖端的方向性轮廓，但仍只保留少量离散档位：
#      端口宽度只有 1 或 2；方向只有上下左右；基础 profile 只有 7 种。
#    - 这样既能让样例中重复出现、支撑归纳，也不会把任务改成高自由度形状匹配题。
# c) 隐式假设审查（对应原 main 的 D5 相关段）
#    - 原 Step 2 依赖“存在唯一绿色颜色且 4 邻接蓝色”；变体后此假设失效，必须删除整段绿色情况分析。
#    - 原 Step 4 依赖“端口=与蓝块 4 邻接的绿色像素”；变体后必须改成“端口=蓝块四个极值边里唯一最短边”。
#    - 原 extend_thick 从蓝块内部、紧邻绿色的位置起射；变体里没有绿色，所以改成从最窄边自身起射。
#      这在原题里若直接无脑沿用会错，因为会找不到任何 port_pixels。
# d) 放置合法性
#    - 生成器继续 output-first：先沿红光路放置将被激活的块，再反推输入。
#    - 每个方向块都保证：
#      * port_dir 对应的极值边长度恰好为 1 或 2；
#      * 四边中它是严格唯一最短边；
#      * 其他边足够宽，可承接 1 或 2 宽的入射光。
#    - 对象仍需：不越界、不与已有块 8 邻接重叠、不与已画光路冲突，且最终由 solver 复算可完全还原输出。
# e) 视觉可观察性
#    - 人类在 2-4 组样例里能直接看到：绿色消失了，但不同蓝块的“尖端方向”不同，输出红光总是从尖端射出。
#    - 方向由可见几何轮廓决定，不依赖颜色 id、遍历顺序等不可见因素。
# f) 可视化整洁度
#    - 仍保留原题的稀疏布局和 1 格安全间距；光路和块不会层层覆盖到难以辨认。
#    - 方向信息被压缩进块轮廓本身，没有额外叠加标记，视觉上反而比“蓝+绿”更干净。
# g) 颜色变化可推断性
#    - 输入颜色从 5 种降到 4 种：8 背景、1 蓝块、4 黄帽、2 红杆。
#      没有新的颜色映射负担，颜色规则仍是“被打到的块和光路全部变红，其他全部消失到背景”。
# h) 变体量化指标
#    - 核心指标 1：green_pixels == 0，确保不是“原题输入少量退化样本”。
#    - 核心指标 2：每个 blob 都满足 unique_min_edge_count in {1,2}，说明 solver 真的是靠几何尖端解题。
#    - 关键分支分布：统计端口宽度 1/2、方向四向分布、chain_len 0-3 分布，避免生成器长期固定成单一子情况。
# i) 新 solver 的额外假设与生成保障
#    - solver 假设每个蓝块四个极值边中有且仅有一条最短边，且最短长度为 1 或 2；
#      generator 通过 profile 验证严格保证。
#    - solver 假设 blob 仍是单个 8 连通蓝组件；generator 的所有 profile 都满足。
#    - solver 假设级联链外最多 1 个干扰块；generator 保持原上限不变。

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a red ray network by activating blue blobs whose shape encodes the exit edge.

    Input structure:
    - bg = most common color. Blue blobs are one-tip directional kites; there are no green port markers.
    - Start marker: yellow pixel atop a short red segment at grid bottom.

    Rule:
    - Extend a ray upward from start; on hitting a blue blob, recolor it red.
    - Activated blobs emit a ray from their uniquely narrowest outer edge until boundary or another blob.
    - Cascade until no new blobs are reached.

    Key insight: The launch direction and thickness are implicit in each blob's silhouette: the only 1-2 pixel extreme edge is the hidden port.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect the start "port" (red vertical segment with a non-red cap above) ---
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

    # --- STEP 2) Identify blob color ---
    counts = Counter(map(int, g.flatten()))
    other_cols = [(int(cnt), int(col)) for col, cnt in counts.items() if int(col) not in (bg, red_col, yellow_col)]
    assert other_cols
    max_count = max(cnt for cnt, _col in other_cols)
    blue_candidates = [int(col) for cnt, col in other_cols if cnt == max_count]
    assert len(blue_candidates) == 1
    blue_col = blue_candidates[0]

    out = np.full_like(g, bg)

    # --- STEP 3) Extract blue blobs as connected components, and infer each hidden port from shape ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []

    def edge_pixels(pts: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
        if d == UP:
            extreme = min(r for r, _ in pts)
            cells = [(r, c) for r, c in pts if r == extreme]
        elif d == DOWN:
            extreme = max(r for r, _ in pts)
            cells = [(r, c) for r, c in pts if r == extreme]
        elif d == LEFT:
            extreme = min(c for _, c in pts)
            cells = [(r, c) for r, c in pts if c == extreme]
        else:
            extreme = max(c for _, c in pts)
            cells = [(r, c) for r, c in pts if c == extreme]
        return _sort_on_ray(cells, d)

    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = len(blobs)
        for r, c in pts:
            blob_id[r, c] = bi

        edge_map = {d: edge_pixels(pts, d) for d in DIRS}
        edge_counts = {d: len(cells) for d, cells in edge_map.items()}
        min_count = min(edge_counts.values())
        port_dirs = [d for d, cnt in edge_counts.items() if cnt == min_count]
        assert len(port_dirs) == 1 and min_count in (1, 2)

        port_dir = port_dirs[0]
        blobs.append(
            {
                "pts": pts,
                "ports": [{"dir": port_dir, "pixels": edge_map[port_dir]}],
            }
        )

    activated: set[int] = set()
    queue: list[int] = []

    def activate_blob_at(r: int, c: int) -> None:
        """Recolor a blob to red when first hit by a ray; enqueue it to emit its hidden port."""
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(bi)
        queue.append(bi)
        for pr, pc in blobs[bi]["pts"]:
            out[pr, pc] = red_col

    # --- STEP 4) Draw the start ray upward (thin) and stop at the first blob hit ---
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
        """Extend a ray from each inferred tip pixel until boundary or blob hit."""
        ddr, ddc = map(int, d)
        for pr, pc in port_pixels:
            r, c = pr, pc
            while True:
                r += ddr
                c += ddc
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[r, c] = red_col
                if int(g[r, c]) == blue_col:
                    activate_blob_at(r, c)
                    break

    # --- STEP 5) Propagate (BFS): activated blob -> emit hidden-tip ray -> activate more blobs ---
    while queue:
        bi = queue.pop(0)
        for port in blobs[bi]["ports"]:
            extend_thick(port["pixels"], port["dir"])

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"

    return out


# --------------- generator (a2_cascade, implicit-tip variant) ---------------

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

# --- Directional shape profiles ---
# Each tuple is the row-width profile for an UP-pointing blob before rotation.
TIP_PROFILES = (
    (1, 3, 5),
    (1, 3, 5, 5),
    (1, 3, 5, 5, 5),
    (1, 3, 5, 7, 7),
    (1, 3, 5, 7, 7, 7),
    (2, 4, 6, 8, 8, 8),
    (2, 4, 6, 8, 8, 8, 8),
)

# --- Canvas bounds ---
H_MIN, H_MAX = 13, 27
W_MIN, W_MAX = 15, 27
START_COL_PAD = 2

# --- Start segment ---
START_H_MIN, START_H_MAX = 2, 3

# --- Chain / distractor ---
CHAIN_LEN_WEIGHTS = [1, 3, 3, 3]
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1

# --- Placement ---
MIN_GAP_FIRST = 4
MIN_GAP_CHAIN = 3

# --- Retry ---
GEN_RETRIES = 300


def _shape_mask(profile: tuple[int, ...]) -> np.ndarray:
    """Build a centered UP-pointing kite mask from a row-width profile."""
    h = len(profile)
    w = max(profile)
    mask = np.zeros((h, w), dtype=bool)
    for r, ww in enumerate(profile):
        c0 = (w - ww) // 2
        mask[r, c0 : c0 + ww] = True
    return mask


def _rotate_mask(mask: np.ndarray, d: tuple[int, int]) -> np.ndarray:
    """Rotate an UP-pointing base mask so its unique tip faces direction d."""
    if d == UP:
        return mask
    if d == RIGHT:
        return np.rot90(mask, 3)
    if d == DOWN:
        return np.rot90(mask, 2)
    if d == LEFT:
        return np.rot90(mask, 1)
    raise ValueError(d)


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
    """Return sorted mask-local coordinates on the edge facing direction d."""
    pts = np.argwhere(mask)
    axis = 0 if d in (UP, DOWN) else 1
    extreme = pts[:, axis].min() if d in (UP, LEFT) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return _sort_on_ray([(int(r), int(c)) for r, c in pts[keep]], d)


def _unique_tip_dir(mask: np.ndarray) -> tuple[int, int] | None:
    """Return the unique shortest extreme side if it exists and has width 1 or 2."""
    counts = {d: len(_edge_offsets(mask, d)) for d in DIRS}
    min_count = min(counts.values())
    dirs = [d for d, cnt in counts.items() if cnt == min_count]
    if len(dirs) != 1 or min_count not in (1, 2):
        return None
    return dirs[0]


def _blob_spec(profile: tuple[int, ...], port_dir: tuple[int, int]) -> dict | None:
    """Build a blob specification dict for a given profile and implicit tip direction."""
    mask = _rotate_mask(_shape_mask(profile), port_dir)
    inferred = _unique_tip_dir(mask)
    if inferred != port_dir:
        return None

    tip = _edge_offsets(mask, port_dir)
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
        "port_pixels": tip,
        "port_width": len(tip),
    }


def _all_specs(port_dir: tuple[int, int] | None = None) -> list[dict]:
    """Generate all valid blob specs, optionally filtered by port direction, in shuffled order."""
    dirs = [port_dir] if port_dir is not None else list(DIRS)
    out = [s for profile in TIP_PROFILES for d in dirs if (s := _blob_spec(profile, d)) is not None]
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
    """Trace ray lanes from starts to entries; return cell list or None on failure."""
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
    """Trace ray lanes from starts to grid boundary."""
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
    """Paint cells red in the output grid and mark them in the ray mask."""
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
    candidates: list[dict] = []

    for spec in _all_specs():
        if spec["port_dir"] == entry_side:
            continue
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

    return random.choice(candidates) if candidates else None


def _place_blob(inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, blob: dict) -> None:
    """Paste a blob's patch onto the input grid and update tracking masks."""
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
    """Single attempt to generate a valid input grid. Returns None on placement failure."""
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

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

    chain_len = random.choices(range(len(CHAIN_LEN_WEIGHTS)), weights=CHAIN_LEN_WEIGHTS)[0]
    min_dist = 1 if chain_len == 0 else 0
    max_dist = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - chain_len)
    num_distractors = random.randint(min_dist, max_dist)

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

    final_path = _path_to_boundary(starts, direction, include_origin, H, W)
    if any(blue_mask[r, c] for r, c in final_path):
        return None
    _mark_ray(out, ray_mask, final_path)

    for _ in range(num_distractors):
        if not _place_distractor(inp, occupied, blue_mask, ray_mask):
            return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 input")
