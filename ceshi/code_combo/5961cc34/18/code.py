# Combo variant of 5961cc34 (index 18)
# combo_spec: D5/2+D9/3
# Name: 隐式端口+L形色块

# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
#    - 输入仍有背景、蓝色块、底部黄帽+红杆起点，但所有绿色端口都被移除。
#    - 每个蓝色块不再是对称菱形，而是实心 L 形：其 bbox 内恰好缺失一个角落矩形，
#      因而肉眼能看出“短臂”和“长臂”。
#    - 发射口不再由绿色标记给出，而是由 L 形短臂末端那条最窄边隐式决定；
#      这条边的长度严格为 1 或 2，并且是四个极值边里唯一最短的一条。
#    - 小例子追踪：
#      原题里，起点向上打到一个蓝色菱形后，solver 通过菱形某侧邻接的绿色像素推出
#      “向右发射”。本融合版里，同样位置换成一个 L 形块：上来的射线先命中 L 块，
#      整块变红；随后 solver 比较该块上/下/左/右四条极值边的长度，发现右侧短臂末端
#      只有 2 个像素、且其余三侧都更宽，于是改为从右侧发射 2 像素宽红射线。
#      起点检测、首次命中、BFS 级联都保留，变化集中在“对象形状”和“端口识别”。
# b) 复杂度校准
#    - 若只删除绿色而保留对称块，则方向不可判；若 L 形参数完全自由，则样例难以归纳。
#    - 因此生成器只使用少量离散模板：短臂厚度只取 1 或 2，长臂厚度固定大于短臂，
#      每个模板都等于“满矩形减去一个角落矩形”，既明显是 L 形，又稳定产生唯一窄边。
#    - 同一题内仍会重复出现 1/2 宽端口、四个方向和 0-3 级联长度，不会退化成单一子情形。
# c) 隐式假设审查
#    - 原 Step 2 假设存在唯一绿色端口颜色；融合后彻底失效，必须删除绿色检索逻辑。
#    - 原 Step 4 假设端口像素可由“蓝块 4 邻接绿色”获得；融合后需改为“比较四个极值边长度，
#      取唯一最短边”，并要求其长度属于 {1,2}。
#    - 原生成器默认只需避开实际蓝像素；L 形带有缺角，如果射线穿过 bbox 内空角，会让视觉上
#      像是光线贴着甚至钻过对象。融合后把每个 L 块的整个 bbox 作为保留区，避免这种歧义。
# d) 放置合法性
#    - 继续采用 output-first：先沿红色光路依次放置将被命中的 L 块，再补充干扰块，最后反推输入。
#    - 每个候选块都要求：唯一最短极值边长度为 1 或 2；入射侧边界至少容纳当前射线宽度；
#      整个 bbox 不越界、不与已有 bbox 8 邻接、不与任何已画射线路径重叠。
#    - 最终仍用 `main(inp) == out` 做 oracle 校验，保证 solver/generator 一致。
# e) 视觉可观察性
#    - 2-4 组样例里，人类能同时看到两件事：绿色标记消失了；蓝块全变成明显的 L 形。
#    - 输出里红射线总是从 L 形短臂末端出去，而不是从长边或拐角出去，几组样例即可归纳。
#    - 方向由可见几何轮廓决定，不依赖颜色编号、扫描顺序或其他不可视因素。
# f) 可视化整洁度
#    - 通过 bbox 级保留区，避免另一条射线从某个 L 块的缺角穿过，保持每个对象轮廓清晰。
#    - 干扰块仍至多 1 个，整体布局继续保持稀疏，输出中不会出现大面积红蓝叠压难以辨认。
# g) 变化的可推断性
#    - 输入颜色从原题的 5 种降到 4 种：背景 8、蓝块 1、起点黄帽 4、起点红杆 2。
#    - 没有新的颜色映射负担；核心变化全部来自几何：L 形轮廓决定端口，命中后整块和光路变红。
# h) 融合量化指标
#    - 指标 1：`green_pixels == 0`，确保输入不再含显式端口。
#    - 指标 2：每个蓝块 bbox 内都存在非空角落空矩形，证明对象确为 L 形而非菱形/实心矩形。
#    - 指标 3：每个蓝块都满足 `unique_min_edge_count in {1,2}`，说明端口确由短臂末端隐式给出。
# i) 新 solver 假设与生成保障
#    - solver 假设每个蓝色连通块的四个极值边中恰有一条最短，且长度为 1 或 2；
#      生成器只从满足该条件的模板库取样。
#    - solver 假设每个蓝块在自身 bbox 内恰好是“满矩形减去一个角落矩形”的单个 8 连通组件；
#      生成器在建 spec 时显式验证这一点。
#    - solver 假设链外最多 1 个干扰块；生成器沿用原上限并在失败时重试。

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

# --- L-shape templates ---
# (tip_width, short_arm_len, long_arm_len, long_arm_thickness)
L_PARAMS = (
    (1, 2, 4, 2),
    (1, 2, 5, 3),
    (1, 3, 6, 3),
    (2, 2, 5, 3),
    (2, 2, 6, 4),
    (2, 3, 7, 4),
)
HANDEDNESS = ("left", "right")

# --- Canvas bounds ---
H_MIN, H_MAX = 13, 27
W_MIN, W_MAX = 15, 27
START_COL_PAD = 2

# --- Start segment ---
START_H_MIN, START_H_MAX = 2, 3

# --- Chain / distractor ---
CHAIN_LENGTHS = (1, 2, 3)
# Longer chains are harder to place under bbox/ray separation constraints, so sample
# them more often to keep the accepted generator outputs reasonably balanced.
CHAIN_LEN_WEIGHTS = [1, 3, 12]
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1

# --- Placement ---
MIN_GAP_FIRST = 4
MIN_GAP_CHAIN = 3

# --- Retry ---
GEN_RETRIES = 400


def _build_up_l_mask(
    tip_width: int,
    short_arm_len: int,
    long_arm_len: int,
    long_arm_thickness: int,
    handed: str,
) -> np.ndarray:
    """Build an UP-emitting L mask as a full rectangle minus one corner rectangle."""
    h = short_arm_len + long_arm_thickness
    w = long_arm_len
    mask = np.zeros((h, w), dtype=bool)
    if handed == "left":
        c0 = 0
    else:
        c0 = w - tip_width
    mask[:short_arm_len, c0 : c0 + tip_width] = True
    mask[short_arm_len:, :] = True
    return mask


def _rotate_mask(mask: np.ndarray, d: tuple[int, int]) -> np.ndarray:
    """Rotate an UP-emitting base mask so the hidden port faces direction d."""
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
    """Coordinate perpendicular to direction d (col for vertical rays, row for horizontal rays)."""
    return rc[1] if d[0] else rc[0]


def _along_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    """Coordinate along direction d (row for vertical rays, col for horizontal rays)."""
    return rc[0] if d[0] else rc[1]


def _sort_on_ray(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    """Sort coordinates by lane order relative to direction d."""
    return sorted(coords, key=lambda p: _perp_value(p, d))


def _edge_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    """Return sorted mask-local coordinates on the extreme side facing direction d."""
    pts = np.argwhere(mask)
    axis = 0 if d in (UP, DOWN) else 1
    extreme = pts[:, axis].min() if d in (UP, LEFT) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return _sort_on_ray([(int(r), int(c)) for r, c in pts[keep]], d)


def _infer_hidden_port(mask: np.ndarray) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """Infer the unique 1-2 pixel extreme edge that acts as the implicit launch port."""
    edge_map = {d: _edge_offsets(mask, d) for d in DIRS}
    edge_counts = {d: len(cells) for d, cells in edge_map.items()}
    min_count = min(edge_counts.values())
    port_dirs = [d for d, cnt in edge_counts.items() if cnt == min_count]
    assert len(port_dirs) == 1 and min_count in (1, 2)
    port_dir = port_dirs[0]
    return port_dir, edge_map[port_dir]


def _corner_void_info(mask: np.ndarray) -> dict | None:
    """Return info about the missing corner rectangle inside an L-shape bbox."""
    empty = np.argwhere(~mask)
    if empty.size == 0:
        return None
    r0, c0 = empty.min(axis=0)
    r1, c1 = empty.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    if area != int(empty.shape[0]):
        return None
    h, w = mask.shape
    touches_top = r0 == 0
    touches_bottom = r1 == h - 1
    touches_left = c0 == 0
    touches_right = c1 == w - 1
    if touches_top and touches_left:
        corner = "tl"
    elif touches_top and touches_right:
        corner = "tr"
    elif touches_bottom and touches_left:
        corner = "bl"
    elif touches_bottom and touches_right:
        corner = "br"
    else:
        return None
    return {"corner": corner, "area": area, "rect": (r0, c0, r1, c1)}


def _blob_spec(params: tuple[int, int, int, int], port_dir: tuple[int, int], handed: str) -> dict | None:
    """Build a valid L-shape spec, or None if the geometry fails the fusion constraints."""
    tip_width, short_arm_len, long_arm_len, long_arm_thickness = params
    base = _build_up_l_mask(tip_width, short_arm_len, long_arm_len, long_arm_thickness, handed)
    mask = _rotate_mask(base, port_dir)
    inferred_dir, port_pixels = _infer_hidden_port(mask)
    if inferred_dir != port_dir:
        return None
    void = _corner_void_info(mask)
    if void is None:
        return None
    h, w = mask.shape
    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    box = [(r, c) for r in range(h) for c in range(w)]
    patch = np.full((h, w), BG, dtype=int)
    patch[mask] = BLUE
    return {
        "size": (h, w),
        "mask": mask,
        "patch": patch,
        "blue": full,
        "full": full,
        "box": box,
        "port_dir": port_dir,
        "port_pixels": _sort_on_ray(port_pixels, port_dir),
        "port_width": len(port_pixels),
        "void_corner": void["corner"],
        "void_area": void["area"],
        "handed": handed,
        "params": params,
    }


def _all_specs(port_dir: tuple[int, int] | None = None) -> list[dict]:
    """Enumerate all legal L-shape specs in random order."""
    dirs = [port_dir] if port_dir is not None else list(DIRS)
    out = [
        spec
        for params in L_PARAMS
        for handed in HANDEDNESS
        for d in dirs
        if (spec := _blob_spec(params, d, handed)) is not None
    ]
    random.shuffle(out)
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    """Dilate a boolean mask by 1 pixel in all 8 directions."""
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, dr), h + min(0, dr)
            c0, c1 = max(0, dc), w + min(0, dc)
            out[r0:r1, c0:c1] |= mask[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out


def _offset(offsets: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    """Translate local offsets to global coordinates."""
    return [(r0 + r, c0 + c) for r, c in offsets]


def _in_bounds(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in coords)


def _path_to_entries(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    entries: list[tuple[int, int]],
    include_origin: bool,
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    """Trace all ray lanes from starts to target entry cells."""
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for (r, c), (er, ec) in zip(starts, entries):
        if include_origin:
            cells.append((r, c))
        guard = h + w + 10
        while (r, c) != (er, ec):
            r += dr
            c += dc
            guard -= 1
            if guard < 0 or not (0 <= r < h and 0 <= c < w):
                return None
            cells.append((r, c))
    return cells


def _path_to_boundary(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    """Trace all ray lanes from starts until they leave the grid."""
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for r, c in starts:
        if include_origin:
            cells.append((r, c))
        while True:
            r += dr
            c += dc
            if not (0 <= r < h and 0 <= c < w):
                break
            cells.append((r, c))
    return cells


def _mark_ray(out: np.ndarray, ray_mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    """Paint a ray into the output and tracking mask."""
    paint_coords(out, cells, RED)
    paint_coords(ray_mask, cells, True)


def _reserve_box(reserved: np.ndarray, cells: list[tuple[int, int]]) -> None:
    """Mark a blob bbox as unavailable for future blobs/rays."""
    paint_coords(reserved, cells, True)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a red cascade by activating L-shaped blue blobs whose short arm end is a hidden port.

    Input structure:
    - bg = most common color. Blue objects are asymmetric filled L-shapes with no green markers.
    - Start marker: one yellow cap sits directly above a short red segment touching the bottom edge.

    Rule:
    - Shoot a thin ray upward from the start; when it first hits a blue L-shape, recolor that whole blob red.
    - Each activated blob launches a ray from its uniquely narrowest extreme side, with width 1 or 2 from that side.
    - Propagate breadth-first until every reachable blob has fired or the rays leave the grid.

    Key insight: The firing side is encoded by geometry alone: the short arm tip of each L-shape is the only extreme edge with length 1-2, so its direction and ray thickness can be inferred from the silhouette.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
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
            if r1 != h - 1:
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

    # --- STEP 2) Identify the blob color: the dominant non-bg, non-start color is blue ---
    counts = Counter(map(int, g.flatten()))
    other_cols = [(int(cnt), int(col)) for col, cnt in counts.items() if int(col) not in (bg, red_col, yellow_col)]
    assert other_cols
    max_count = max(cnt for cnt, _col in other_cols)
    blue_candidates = [int(col) for cnt, col in other_cols if cnt == max_count]
    assert len(blue_candidates) == 1
    blue_col = blue_candidates[0]

    out = np.full_like(g, bg)

    # --- STEP 3) Extract blue blobs and infer each hidden port from its L-shaped silhouette ---
    blob_id = np.full((h, w), -1, dtype=int)
    blobs: list[dict] = []
    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = len(blobs)
        for r, c in pts:
            blob_id[r, c] = bi

        r0 = min(r for r, _ in pts)
        r1 = max(r for r, _ in pts)
        c0 = min(c for _, c in pts)
        c1 = max(c for _, c in pts)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            mask[r - r0, c - c0] = True

        port_dir, local_port = _infer_hidden_port(mask)
        assert _corner_void_info(mask) is not None

        blobs.append(
            {
                "pts": pts,
                "ports": [{"dir": port_dir, "pixels": [(r0 + r, c0 + c) for r, c in local_port]}],
            }
        )

    activated: set[int] = set()
    queue: list[int] = []

    def activate_blob_at(r: int, c: int) -> None:
        """Recolor a blob to red when first hit; enqueue it to emit from its hidden short-arm tip."""
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(bi)
        queue.append(bi)
        for pr, pc in blobs[bi]["pts"]:
            out[pr, pc] = red_col

    # --- STEP 4) Draw the start ray upward and activate the first blob it touches ---
    for r, c in start_seg:
        out[r, c] = red_col
    top_r = min(int(r) for r, _ in start_seg)
    assert top_r > 0 and int(g[top_r - 1, start_col]) == yellow_col
    out[top_r - 1, start_col] = red_col
    for r in range(top_r - 1, -1, -1):
        out[r, start_col] = red_col
        if int(g[r, start_col]) == blue_col:
            activate_blob_at(r, start_col)
            break

    def extend_thick(port_pixels: list[tuple[int, int]], d: tuple[int, int]) -> None:
        """Extend a thick ray from the inferred short-arm tip until boundary or another blob hit."""
        dr, dc = map(int, d)
        for pr, pc in port_pixels:
            r, c = pr, pc
            while True:
                r += dr
                c += dc
                if not (0 <= r < h and 0 <= c < w):
                    break
                out[r, c] = red_col
                if int(g[r, c]) == blue_col:
                    activate_blob_at(r, c)
                    break

    # --- STEP 5) BFS propagate: every activated L-shape emits from its hidden port ---
    while queue:
        bi = queue.pop(0)
        for port in blobs[bi]["ports"]:
            extend_thick(port["pixels"], port["dir"])

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"
    return out


def _try_place_along_axis(
    spec: dict,
    window: list[tuple[int, int]],
    d: tuple[int, int],
    lane_vals: list[int],
    starts: list[tuple[int, int]],
    include_origin: bool,
    min_gap: int,
    reserved_pad: np.ndarray,
    reserved: np.ndarray,
    ray_mask: np.ndarray,
    h: int,
    w: int,
) -> list[dict]:
    """Find all valid placements for one blob spec along the current ray axis."""
    sh, sw = spec["mask"].shape
    win_vals = [_perp_value(p, d) for p in window]
    edge_line = _along_value(window[0], d)
    start_line = _along_value(starts[0], d)
    is_vertical = bool(d[0])

    fixed = lane_vals[0] - win_vals[0]
    extent = h - sh if is_vertical else w - sw
    low = edge_line
    high = extent + edge_line
    if d in (UP, LEFT):
        high = min(high, start_line - min_gap)
    else:
        low = max(low, start_line + min_gap)

    entry_lines = list(range(low, high + 1))
    random.shuffle(entry_lines)
    candidates: list[dict] = []
    start_set = set(starts)

    for entry_line in entry_lines:
        origin_along = entry_line - edge_line
        r0 = origin_along if is_vertical else fixed
        c0 = fixed if is_vertical else origin_along
        if is_vertical:
            entry_cells = [(entry_line, c0 + c) for _, c in window]
        else:
            entry_cells = [(r0 + r, entry_line) for r, _ in window]

        full = _offset(spec["full"], r0, c0)
        box = _offset(spec["box"], r0, c0)
        if not _in_bounds(box, h, w):
            continue
        if any(reserved_pad[r, c] or ray_mask[r, c] for r, c in box):
            continue

        path = _path_to_entries(starts, d, entry_cells, include_origin, h, w)
        if path is None:
            continue
        entry_set = set(entry_cells)
        if any((reserved[r, c] or ray_mask[r, c]) for r, c in path if (r, c) not in entry_set and (r, c) not in start_set):
            continue

        candidates.append(
            {
                "spec": spec,
                "r0": r0,
                "c0": c0,
                "entry": entry_cells,
                "path": path,
                "blue": full,
                "full": full,
                "box": box,
                "ports": _sort_on_ray(_offset(spec["port_pixels"], r0, c0), spec["port_dir"]),
            }
        )
    return candidates


def _candidate_next_blob(
    h: int,
    w: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    reserved: np.ndarray,
    ray_mask: np.ndarray,
) -> dict | None:
    """Find a random valid placement for the next blob in the cascade chain."""
    starts = _sort_on_ray(starts, d)
    thickness = len(starts)
    lane_vals = [_perp_value(p, d) for p in starts]
    lane_delta = [v - lane_vals[0] for v in lane_vals]
    entry_side = OPP[d]
    reserved_pad = _dilate8(reserved)
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
                    reserved_pad,
                    reserved,
                    ray_mask,
                    h,
                    w,
                )
            )

    return random.choice(candidates) if candidates else None


def _place_blob(inp: np.ndarray, reserved: np.ndarray, blob: dict) -> None:
    """Paste a blob patch into the input grid and reserve its bbox."""
    paste(inp, blob["spec"]["patch"], blob["r0"], blob["c0"], transparent=BG)
    _reserve_box(reserved, blob["box"])


def _place_distractor(inp: np.ndarray, reserved: np.ndarray, ray_mask: np.ndarray) -> bool:
    """Place one off-path distractor L-shape if space exists."""
    h, w = inp.shape
    reserved_pad = _dilate8(reserved)
    candidates: list[dict] = []
    for spec in _all_specs():
        sh, sw = spec["mask"].shape
        for r0 in range(h - sh + 1):
            for c0 in range(w - sw + 1):
                box = _offset(spec["box"], r0, c0)
                if any(reserved_pad[r, c] or ray_mask[r, c] for r, c in box):
                    continue
                candidates.append(
                    {
                        "spec": spec,
                        "r0": r0,
                        "c0": c0,
                        "box": box,
                    }
                )
    if not candidates:
        return False
    _place_blob(inp, reserved, random.choice(candidates))
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single output-first generation attempt."""
    h = random.randint(H_MIN, H_MAX)
    w = random.randint(W_MIN, W_MAX)
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    ray_mask = np.zeros((h, w), dtype=bool)

    # Place the fixed start marker.
    start_h = random.randint(START_H_MIN, START_H_MAX)
    cap_r = h - start_h - 1
    start_col = random.randint(START_COL_PAD, w - START_COL_PAD - 1)
    inp[cap_r, start_col] = YELLOW
    out[cap_r, start_col] = RED
    ray_mask[cap_r, start_col] = True
    reserved[cap_r, start_col] = True
    for r in range(cap_r + 1, h):
        inp[r, start_col] = RED
        out[r, start_col] = RED
        reserved[r, start_col] = True

    chain_len = random.choices(CHAIN_LENGTHS, weights=CHAIN_LEN_WEIGHTS)[0]
    min_dist = 0
    max_dist = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - chain_len)
    num_distractors = random.randint(min_dist, max_dist)

    starts = [(cap_r, start_col)]
    direction = UP
    include_origin = True
    for _ in range(chain_len):
        blob = _candidate_next_blob(h, w, starts, direction, include_origin, reserved, ray_mask)
        if blob is None:
            return None
        _mark_ray(out, ray_mask, blob["path"])
        _place_blob(inp, reserved, blob)
        paint_coords(out, blob["blue"], RED)
        starts = blob["ports"]
        direction = blob["spec"]["port_dir"]
        include_origin = False

    final_path = _path_to_boundary(starts, direction, include_origin, h, w)
    start_set = set(starts)
    if any((reserved[r, c] or ray_mask[r, c]) for r, c in final_path if (r, c) not in start_set):
        return None
    _mark_ray(out, ray_mask, final_path)

    for _ in range(num_distractors):
        if not _place_distractor(inp, reserved, ray_mask):
            return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a valid combo-variant input grid."""
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 combo input")
