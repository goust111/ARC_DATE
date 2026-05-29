# Combo variant of 5961cc34 (index 20)
# combo_spec: D7/4+D19/0
# Name: 围绕激活色块画框+射线分裂

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


# Design analysis for the fused variant
# a) Concrete behavior:
#    原题里，命中一个 blob 之后只沿绿色端口方向继续发射；这里改成：
#    1) 把命中的 blue+green 联合组件整体涂红；
#    2) 以该联合组件的 bbox 为核心，在 bbox 外扩 1 格画一圈红框；
#    3) 从红框四边的外侧同时发射四组射线。例：若 bbox 为 [r0:r1, c0:c1]，
#       则顶部红框在 r0-1，顶侧射线从 r0-2 的 c0-1..c1+1 全部起射并向上。
#    这样与原题的差异是可见的：不再是“单端口单方向链”，而是“框+四向分光”。
# b) Complexity calibration:
#    若允许任意多层分裂树，生成器会很容易产生难以读图的重叠。本实现把激活主干控制为
#    “起点 -> 根 blob -> 可选 1 个分裂子 blob”，但根 blob 的四向射线始终存在，
#    因此视觉差异稳定、明确且不会退化为原题。
# c) Hidden assumptions replaced:
#    原 solver 假设 “blob 只需记录端口像素”。融合后这个前提失效，因为端口不再控制发射。
#    新 solver 改为记录每个 blue+green 联合组件的 full coords 与 bbox，并允许命中 green
#    也触发激活。另一个原假设是“厚射线从 blob 内部端口出发”；现在改为“从外框外侧出发”。
# d) Placement legality:
#    生成器对 blob 使用 8 邻域留白，避免组件粘连；可选子 blob 只放在某一条分裂射线 lane 上，
#    且与已放置 blob 保持间隔。distractor 在求出当前输出后再放置到所有红路径之外。
# e) Visual observability:
#    样例里即使只有 1 个激活 blob，人眼也能看到两条融合特征：
#    “blob 外围新增矩形框”和“从四边而非绿色端口继续发射”。绿色端口仍保留在输入里，
#    其失效本身也是可观察线索。
# f) Visual cleanliness:
#    根 blob 远离边界，保证四边框和四向射线大多可见；子 blob 最多 1 个，避免多树交叉。
#    distractor 强制避开输出红区，防止画面混杂成大片红覆盖。
# g) Inferability:
#    颜色关系没有新增复杂映射：激活相关的全部结构都变红，未命中的 blue/green 全擦成 bg。
#    关键线索来自位置和几何结构，而不是不可见的颜色编号排序。


# --- Colors ---
BG = 8
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

# --- Directions ---
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
OPP = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

# --- Diamond/octagon shapes (h, w) ---
SHAPES = ((3, 4), (4, 4), (5, 4), (5, 6), (6, 6), (8, 8))
_SHAPE_WIDTHS = {
    (3, 4): (2, 4, 2),
    (4, 4): (2, 4, 4, 2),
    (5, 4): (2, 4, 4, 4, 2),
    (5, 6): (2, 4, 6, 4, 2),
    (6, 6): (2, 4, 6, 6, 4, 2),
    (8, 8): (2, 4, 6, 8, 8, 6, 4, 2),
}

# --- Generator bounds ---
H_MIN, H_MAX = 17, 27
W_MIN, W_MAX = 17, 27
START_H_MIN, START_H_MAX = 2, 3
ROOT_MARGIN = 2
MIN_START_GAP = 4
MIN_CHILD_GAP = 4
SECONDARY_PROB = 0.7
DISTRACTOR_PROB = 0.55
MAX_RED_RATIO = 0.78
GEN_RETRIES = 400


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a red split-ray pattern where each activated blob gets an outer frame.

    Input structure:
    - Background is the most common color. Blue bodies with attached green edge markers form the blob components.
    - A yellow cap above a short red stem at the bottom marks the starting launcher.

    Rule:
    - Shoot a thin ray upward from the launcher until it reaches a blob; the struck blue+green component turns red.
    - Around every activated component, draw a 1-cell red rectangle just outside its bbox, then emit rays outward from all four framed sides.
    - Each lane stops at the first blob it meets or at the grid boundary, and newly hit blobs repeat the same framed four-way split.

    Key insight:
    - Green ports no longer select the outgoing direction. The meaningful object is the combined blob component and its bbox, because both the outer frame and the split rays are defined from that rectangle.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    start_candidates: list[tuple[int, int, int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=8):
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            if c0 != c1 or (r1 - r0 + 1) < 2 or r1 != H - 1:
                continue
            if any(int(g[rr, c0]) != int(col) for rr in range(r0, r1 + 1)):
                continue
            if r0 - 1 < 0:
                continue
            cap = int(g[r0 - 1, c0])
            if cap in (bg, int(col)):
                continue
            start_candidates.append(
                (int(col), int(cap), c0, [(rr, c0) for rr in range(r0, r1 + 1)])
            )

    assert len(start_candidates) == 1
    red_col, yellow_col, start_col, start_seg = start_candidates[0]

    counts = Counter(map(int, g.flatten()))
    other_cols = [
        (int(cnt), int(col))
        for col, cnt in counts.items()
        if int(col) not in (bg, red_col, yellow_col)
    ]
    assert other_cols
    max_count = max(cnt for cnt, _col in other_cols)
    blue_candidates = [int(col) for cnt, col in other_cols if cnt == max_count]
    assert len(blue_candidates) == 1
    blue_col = blue_candidates[0]

    green_candidates: list[int] = []
    for col in colors_present(g, exclude={bg, red_col, yellow_col, blue_col}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        touches_blue = False
        for r, c in pts:
            rr, cc = int(r), int(c)
            for dr, dc in DIRS:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W and int(g[nr, nc]) == int(blue_col):
                    touches_blue = True
                    break
            if touches_blue:
                break
        if touches_blue:
            green_candidates.append(int(col))

    assert len(green_candidates) == 1
    green_col = green_candidates[0]

    blob_mask = (g == int(blue_col)) | (g == int(green_col))
    comps = components_mask(blob_mask, connectivity=8)
    assert comps

    out = np.full_like(g, int(bg))
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []

    for coords in comps:
        pts = [(int(r), int(c)) for r, c in coords]
        bi = len(blobs)
        blue_pts = [(r, c) for r, c in pts if int(g[r, c]) == int(blue_col)]
        assert blue_pts
        for r, c in pts:
            assert blob_id[r, c] == -1
            blob_id[r, c] = bi
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        blobs.append(
            {
                "full": pts,
                "bbox": (min(rs), min(cs), max(rs), max(cs)),
            }
        )

    activated: set[int] = set()
    queue: deque[int] = deque()

    def paint_outer_box(bbox: tuple[int, int, int, int]) -> None:
        r0, c0, r1, c1 = bbox
        rr0, cc0, rr1, cc1 = r0 - 1, c0 - 1, r1 + 1, c1 + 1
        for c in range(cc0, cc1 + 1):
            if 0 <= rr0 < H and 0 <= c < W:
                out[rr0, c] = int(red_col)
            if 0 <= rr1 < H and 0 <= c < W:
                out[rr1, c] = int(red_col)
        for r in range(rr0, rr1 + 1):
            if 0 <= r < H and 0 <= cc0 < W:
                out[r, cc0] = int(red_col)
            if 0 <= r < H and 0 <= cc1 < W:
                out[r, cc1] = int(red_col)

    def split_starts(bbox: tuple[int, int, int, int]) -> list[tuple[list[tuple[int, int]], tuple[int, int]]]:
        r0, c0, r1, c1 = bbox
        starts: list[tuple[list[tuple[int, int]], tuple[int, int]]] = []
        top = [(r0 - 2, c) for c in range(c0 - 1, c1 + 2) if 0 <= r0 - 2 < H and 0 <= c < W]
        bottom = [(r1 + 2, c) for c in range(c0 - 1, c1 + 2) if 0 <= r1 + 2 < H and 0 <= c < W]
        left = [(r, c0 - 2) for r in range(r0 - 1, r1 + 2) if 0 <= r < H and 0 <= c0 - 2 < W]
        right = [(r, c1 + 2) for r in range(r0 - 1, r1 + 2) if 0 <= r < H and 0 <= c1 + 2 < W]
        if top:
            starts.append((top, UP))
        if bottom:
            starts.append((bottom, DOWN))
        if left:
            starts.append((left, LEFT))
        if right:
            starts.append((right, RIGHT))
        return starts

    def activate_blob(bi: int) -> None:
        if bi in activated:
            return
        activated.add(int(bi))
        queue.append(int(bi))
        for r, c in blobs[bi]["full"]:
            out[r, c] = int(red_col)
        paint_outer_box(blobs[bi]["bbox"])

    def trace_lanes(starts: list[tuple[int, int]], d: tuple[int, int], source_bi: int | None) -> None:
        dr, dc = map(int, d)
        for sr, sc in starts:
            r, c = int(sr), int(sc)
            while 0 <= r < H and 0 <= c < W:
                out[r, c] = int(red_col)
                bi = int(blob_id[r, c])
                if bi >= 0 and bi != source_bi:
                    activate_blob(int(bi))
                    break
                r += dr
                c += dc

    for r, c in start_seg:
        out[r, c] = int(red_col)
    top_r = min(r for r, _ in start_seg)
    assert top_r > 0 and int(g[top_r - 1, start_col]) == int(yellow_col)
    out[top_r - 1, start_col] = int(red_col)
    for r in range(top_r - 1, -1, -1):
        out[r, start_col] = int(red_col)
        bi = int(blob_id[r, start_col])
        if bi >= 0:
            activate_blob(int(bi))
            break

    while queue:
        bi = int(queue.popleft())
        for starts, d in split_starts(blobs[bi]["bbox"]):
            trace_lanes(starts, d, bi)

    return out


def _shape_mask(h: int, w: int) -> np.ndarray:
    widths = _SHAPE_WIDTHS[(h, w)]
    mask = np.zeros((h, w), dtype=bool)
    for r, ww in enumerate(widths):
        c0 = (w - ww) // 2
        mask[r, c0 : c0 + ww] = True
    return mask


def _offset(offsets: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + r, c0 + c) for r, c in offsets]


def _in_bounds(coords: list[tuple[int, int]], H: int, W: int) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in coords)


def _sort_on_ray(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    return sorted(coords, key=lambda p: p[1] if d[0] else p[0])


def _edge_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    pts = np.argwhere(mask)
    axis = 0 if d in (UP, DOWN) else 1
    extreme = pts[:, axis].min() if d in (UP, LEFT) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return _sort_on_ray([(int(r), int(c)) for r, c in pts[keep]], d)


def _contact_offsets(spec: dict, d: tuple[int, int]) -> list[tuple[int, int]]:
    edge = _edge_offsets(spec["mask"], d)
    blue_edge = [(r, c) for r, c in edge if int(spec["patch"][r, c]) == BLUE]
    return blue_edge or edge


def _blob_spec(size: tuple[int, int], port_dir: tuple[int, int]) -> dict | None:
    h, w = size
    mask = _shape_mask(h, w)
    green = _edge_offsets(mask, port_dir)
    if len(green) not in (1, 2):
        return None
    green_set = set(green)
    patch = np.full((h, w), BG, dtype=int)
    patch[mask] = BLUE
    for r, c in green:
        patch[r, c] = GREEN
    blue = [(int(r), int(c)) for r, c in np.argwhere(mask) if (int(r), int(c)) not in green_set]
    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return {"mask": mask, "patch": patch, "blue": blue, "full": full}


def _all_specs() -> list[dict]:
    specs = [s for size in SHAPES for d in DIRS if (s := _blob_spec(size, d)) is not None]
    random.shuffle(specs)
    return specs


def _dilate8(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, dr), H + min(0, dr)
            c0, c1 = max(0, dc), W + min(0, dc)
            out[r0:r1, c0:c1] |= mask[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out


def _blob_bbox(blob: dict) -> tuple[int, int, int, int]:
    return int(blob["r0"]), int(blob["c0"]), int(blob["r0"] + blob["h"] - 1), int(blob["c0"] + blob["w"] - 1)


def _split_side_starts(bbox: tuple[int, int, int, int], d: tuple[int, int], H: int, W: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if d == UP:
        row = r0 - 2
        return [(row, c) for c in range(c0 - 1, c1 + 2) if 0 <= row < H and 0 <= c < W]
    if d == DOWN:
        row = r1 + 2
        return [(row, c) for c in range(c0 - 1, c1 + 2) if 0 <= row < H and 0 <= c < W]
    if d == LEFT:
        col = c0 - 2
        return [(r, col) for r in range(r0 - 1, r1 + 2) if 0 <= r < H and 0 <= col < W]
    col = c1 + 2
    return [(r, col) for r in range(r0 - 1, r1 + 2) if 0 <= r < H and 0 <= col < W]


def _path_clear(start: tuple[int, int], end: tuple[int, int], d: tuple[int, int], occupied: np.ndarray) -> bool:
    dr, dc = d
    r, c = start
    while (r, c) != end:
        if occupied[r, c]:
            return False
        r += dr
        c += dc
        if not (0 <= r < occupied.shape[0] and 0 <= c < occupied.shape[1]):
            return False
    return not occupied[end[0], end[1]]


def _place_blob(inp: np.ndarray, occupied: np.ndarray, blob: dict) -> None:
    paste(inp, blob["spec"]["patch"], blob["r0"], blob["c0"], transparent=BG)
    for r, c in blob["full"]:
        occupied[r, c] = True


def _candidate_root(H: int, W: int, cap_r: int) -> tuple[dict, int] | None:
    specs = _all_specs()
    random.shuffle(specs)
    for spec in specs:
        h, w = spec["patch"].shape
        if cap_r - h - MIN_START_GAP < ROOT_MARGIN:
            continue
        entry_cols = sorted(
            {
                c
                for r, c in _edge_offsets(spec["mask"], DOWN)
                if int(spec["patch"][r, c]) == BLUE
            }
        )
        if not entry_cols:
            continue
        r0 = random.randint(ROOT_MARGIN, cap_r - h - MIN_START_GAP)
        c0 = random.randint(ROOT_MARGIN, W - w - ROOT_MARGIN - 1)
        full = _offset(spec["full"], r0, c0)
        start_col = c0 + random.choice(entry_cols)
        return (
            {
                "spec": spec,
                "r0": r0,
                "c0": c0,
                "h": h,
                "w": w,
                "full": full,
            },
            start_col,
        )
    return None


def _candidate_child(
    H: int,
    W: int,
    parent: dict,
    d: tuple[int, int],
    occupied: np.ndarray,
) -> dict | None:
    starts = _split_side_starts(_blob_bbox(parent), d, H, W)
    if not starts:
        return None
    occ_pad = _dilate8(occupied)
    random.shuffle(starts)

    for lane_r, lane_c in starts:
        for spec in _all_specs():
            h, w = spec["patch"].shape
            edge = _contact_offsets(spec, OPP[d])
            edge = edge[:]
            random.shuffle(edge)
            for er, ec in edge:
                if d == UP:
                    entries = list(range(1 + er, lane_r - MIN_CHILD_GAP + 1))
                    random.shuffle(entries)
                    for entry_r in entries:
                        r0 = entry_r - er
                        c0 = lane_c - ec
                        full = _offset(spec["full"], r0, c0)
                        if not _in_bounds(full, H, W):
                            continue
                        if any(occ_pad[r, c] for r, c in full):
                            continue
                        if not _path_clear((lane_r, lane_c), (entry_r, lane_c), d, occupied):
                            continue
                        return {"spec": spec, "r0": r0, "c0": c0, "h": h, "w": w, "full": full}
                elif d == DOWN:
                    entries = list(range(lane_r + MIN_CHILD_GAP, H - (h - er)))
                    random.shuffle(entries)
                    for entry_r in entries:
                        r0 = entry_r - er
                        c0 = lane_c - ec
                        full = _offset(spec["full"], r0, c0)
                        if not _in_bounds(full, H, W):
                            continue
                        if any(occ_pad[r, c] for r, c in full):
                            continue
                        if not _path_clear((lane_r, lane_c), (entry_r, lane_c), d, occupied):
                            continue
                        return {"spec": spec, "r0": r0, "c0": c0, "h": h, "w": w, "full": full}
                elif d == LEFT:
                    entries = list(range(1 + ec, lane_c - MIN_CHILD_GAP + 1))
                    random.shuffle(entries)
                    for entry_c in entries:
                        r0 = lane_r - er
                        c0 = entry_c - ec
                        full = _offset(spec["full"], r0, c0)
                        if not _in_bounds(full, H, W):
                            continue
                        if any(occ_pad[r, c] for r, c in full):
                            continue
                        if not _path_clear((lane_r, lane_c), (lane_r, entry_c), d, occupied):
                            continue
                        return {"spec": spec, "r0": r0, "c0": c0, "h": h, "w": w, "full": full}
                else:
                    entries = list(range(lane_c + MIN_CHILD_GAP, W - (w - ec)))
                    random.shuffle(entries)
                    for entry_c in entries:
                        r0 = lane_r - er
                        c0 = entry_c - ec
                        full = _offset(spec["full"], r0, c0)
                        if not _in_bounds(full, H, W):
                            continue
                        if any(occ_pad[r, c] for r, c in full):
                            continue
                        if not _path_clear((lane_r, lane_c), (lane_r, entry_c), d, occupied):
                            continue
                        return {"spec": spec, "r0": r0, "c0": c0, "h": h, "w": w, "full": full}
    return None


def _candidate_distractor(inp: np.ndarray, occupied: np.ndarray, red_mask: np.ndarray) -> dict | None:
    H, W = inp.shape
    occ_pad = _dilate8(occupied)
    candidates: list[dict] = []
    for spec in _all_specs():
        h, w = spec["patch"].shape
        for r0 in range(0, H - h + 1):
            for c0 in range(0, W - w + 1):
                full = _offset(spec["full"], r0, c0)
                if any(occ_pad[r, c] or red_mask[r, c] for r, c in full):
                    continue
                candidates.append({"spec": spec, "r0": r0, "c0": c0, "h": h, "w": w, "full": full})
    return random.choice(candidates) if candidates else None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    start_h = random.randint(START_H_MIN, START_H_MAX)
    cap_r = H - start_h - 1

    root_pack = _candidate_root(H, W, cap_r)
    if root_pack is None:
        return None
    root, start_col = root_pack

    inp[cap_r, start_col] = YELLOW
    for r in range(cap_r + 1, H):
        inp[r, start_col] = RED
        occupied[r, start_col] = True
    occupied[cap_r, start_col] = True

    _place_blob(inp, occupied, root)

    if random.random() < SECONDARY_PROB:
        dirs = list(DIRS)
        random.shuffle(dirs)
        for d in dirs:
            child = _candidate_child(H, W, root, d, occupied)
            if child is None:
                continue
            _place_blob(inp, occupied, child)
            break

    try:
        active_out = main(inp)
    except Exception:
        return None

    if random.random() < DISTRACTOR_PROB:
        distractor = _candidate_distractor(inp, occupied, active_out == RED)
        if distractor is not None:
            _place_blob(inp, occupied, distractor)

    try:
        final_out = main(inp)
    except Exception:
        return None

    red_pixels = int(np.count_nonzero(final_out == RED))
    if red_pixels <= H:
        return None
    if red_pixels / final_out.size > MAX_RED_RATIO:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 combo input")
