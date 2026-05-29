# Combo variant of 6e4f6532 (index 21)
# combo_spec: D3/1+D14/1
# Name: 空心火车+内部标记
#
# Design analysis:
# - 具体化与行为追踪:
#   原题里 train 是实心 8-blob，条纹色 marker 多半在车身外围；solver 通过 coupler 数量选 station，
#   再用 marker 相对 coupler 质心的朝向决定旋转/翻转。
#   本融合版把车身改成空心矩形框：输出态里能看到一个保留背景洞的 8 框架，9 coupler 贴在框架外侧，
#   少数框架壁上的 8 被替换成区域条纹色 marker。散射到输入后，这些异常色仍跟着火车一起移动。
#   因此 solver 仍按“coupler 数量匹配站台 + marker 方向定朝向”工作，但 marker 不再是外伸触角，
#   而是藏在空心框架壁内的颜色异常点；同时 dock 后内部空洞必须保留。
# - 复杂度校准:
#   为避免规则退化回原题，generator 强制每列车都拥有真正的空洞，且 marker 只能落在框架壁上。
#   coupler 计数仍只用 {1,2,4} 三档，保证 station 匹配关系清晰；size=4 改用长度 4 的直线 coupler/station，
#   这样每个 9 都能直接贴着空心框外壁，既可见又不会把部分 9 误生成 station。
# - 隐式假设审查:
#   `main()` 假设 train 是一个包含 8 的连通分量，marker 是该分量内部的非 bg/非 9 条纹色像素；
#   空心框仍满足这一点，因为洞保留为 bg，不会把组件拆开。`station_groups()` 依赖“station 9 周围没有 8”，
#   generator 因而保证所有 coupler 9 都与框架壁 4 邻接。方向评分原先对偶数长度 coupler 会受质心取整影响，
#   这里改成先变换所有 coupler/marker 坐标再取均值，避免细长 coupler 条带引入伪方向。
# - 放置合法性:
#   仍采用 output-first。先在目标 region 内放 docked hollow train，再逆变换散射到输入；
#   放置时检查不碰条纹、不越界、不与其他 train/station 重叠，并对输入态保持至少 1 格间距。
# - 视觉可观察性:
#   人类能直接看到两个核心变化：一是 8 车身不是实心而是带洞框架；二是条纹色不在外侧游离，
#   而是嵌在框架壁上。洞在输入和输出中都保留，可从 2-3 组样例稳定归纳。
# - 可视化整洁度:
#   车身限定为小型矩形框，marker 只占 2-3 格且避开角点，避免多层叠加。coupler 统一贴在单一外侧，
#   视觉上能清楚区分“外侧 9 条带”“框架壁异常色”“内部背景洞”。
# - 变化的可推断性:
#   颜色种类不增加；marker 颜色仍直接来自目标 region 的边界条纹，结构线索来自 marker 相对 coupler 的方向。
#   由于 marker 藏在框架壁上，解题者需要同时利用颜色和空心结构，而不是依赖原题的外围小触角。

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import get_bg, paste


def apply_transform(arr: np.ndarray, flip_ud: bool = False, rot_k: int = 0) -> np.ndarray:
    """Positional-args helper (kept local; only used by this solver)."""
    out = np.flipud(arr) if flip_ud else arr
    return np.rot90(out, int(rot_k))


def transform_coords(coords, shape, flip_ud: bool = False, rot_k: int = 0):
    """Positional-args helper (kept local; only used by this solver)."""
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    out = apply_transform(mask, flip_ud=flip_ud, rot_k=rot_k)
    return set(map(tuple, np.argwhere(out)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-dock hollow train sprites by coupler count, then orient them using stripe-colored anomalies embedded in the frame walls.

    Input structure:
    - A background color plus thick edge/internal divider stripes partition the board into rectangular regions.
    - Color 8 forms each train as an empty-centered frame; color 9 denotes both isolated stations and couplers attached outside the frame.
    - A few frame-wall cells are recolored with divider-stripe colors, so the orientation clues are hidden inside the train body rather than hanging outside it.

    Rule:
    - Station groups are 8-connected 9-clusters with no neighboring 8; trains are connected components that contain color 8.
    - Match each train to the unique station group whose size equals the number of attached couplers.
    - Rotate/flip the whole hollow sprite so each embedded marker ends up on the same side of the coupler centroid as the same-colored boundary stripe of the target region.
    - Erase the train from its original location, translate it so its couplers land exactly on the station cells, and redraw only the non-bg non-9 pixels, preserving the hollow center.

    Key insight: Coupler count still selects the destination, but the decisive orientation clue now comes from stripe-colored defects hidden in the hollow frame walls.
    """
    inp = grid
    grid = inp.copy()
    H, W = grid.shape
    bg = get_bg(grid)

    def find_edge_stripes():
        stripe = np.zeros((H, W), dtype=bool)
        colors = set()
        top_h = 0
        for r in range(H):
            row = grid[r, :]
            if (row == row[0]).all() and row[0] != bg:
                top_h += 1
            else:
                break
        if top_h:
            stripe[:top_h, :] = True
            colors.add(int(grid[0, 0]))

        bot_h = 0
        for r in range(H - 1, -1, -1):
            row = grid[r, :]
            if (row == row[0]).all() and row[0] != bg:
                bot_h += 1
            else:
                break
        if bot_h:
            stripe[H - bot_h :, :] = True
            colors.add(int(grid[H - 1, 0]))

        inner_r0, inner_r1 = top_h, H - bot_h
        left_w = 0
        for c in range(W):
            col = grid[inner_r0:inner_r1, c]
            if col.size == 0:
                break
            if (col == col[0]).all() and col[0] != bg:
                left_w += 1
            else:
                break
        if left_w:
            stripe[inner_r0:inner_r1, :left_w] = True
            colors.add(int(grid[inner_r0, 0]))

        right_w = 0
        for c in range(W - 1, -1, -1):
            col = grid[inner_r0:inner_r1, c]
            if col.size == 0:
                break
            if (col == col[0]).all() and col[0] != bg:
                right_w += 1
            else:
                break
        if right_w:
            stripe[inner_r0:inner_r1, W - right_w :] = True
            colors.add(int(grid[inner_r0, W - 1]))
        return stripe, colors, (top_h, bot_h, left_w, right_w)

    def find_internal_stripes(edge_info):
        top_h, bot_h, left_w, right_w = edge_info
        r0, r1, c0, c1 = top_h, H - bot_h, left_w, W - right_w
        stripe = np.zeros((H, W), dtype=bool)
        colors = set()

        c = c0
        while c < c1:
            col = grid[r0:r1, c]
            if col.size and (col == col[0]).all() and col[0] != bg:
                color = int(col[0])
                c2 = c
                while c2 + 1 < c1:
                    col2 = grid[r0:r1, c2 + 1]
                    if col2.size and (col2 == col2[0]).all() and int(col2[0]) == color:
                        c2 += 1
                    else:
                        break
                stripe[r0:r1, c : c2 + 1] = True
                colors.add(color)
                c = c2 + 1
            else:
                c += 1

        r = r0
        while r < r1:
            row = grid[r, c0:c1]
            if row.size and (row == row[0]).all() and row[0] != bg:
                color = int(row[0])
                r2 = r
                while r2 + 1 < r1:
                    row2 = grid[r2 + 1, c0:c1]
                    if row2.size and (row2 == row2[0]).all() and int(row2[0]) == color:
                        r2 += 1
                    else:
                        break
                stripe[r : r2 + 1, c0:c1] = True
                colors.add(color)
                r = r2 + 1
            else:
                r += 1
        return stripe, colors

    def station_groups():
        sm = np.zeros((H, W), dtype=bool)
        for r in range(H):
            for c in range(W):
                if grid[r, c] != 9:
                    continue
                ok = all(
                    grid[r + dr, c + dc] != 8
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if (dr != 0 or dc != 0) and 0 <= r + dr < H and 0 <= c + dc < W
                )
                if ok:
                    sm[r, c] = True
        struct = np.ones((3, 3), dtype=int)
        lab, n = label(sm, structure=struct)
        groups = []
        for k in range(1, n + 1):
            coords = np.argwhere(lab == k)
            groups.append([(int(r), int(c)) for r, c in coords])
        return groups

    def regions_from_stripes(stripe_mask):
        non = ~stripe_mask
        lab, n = label(non)
        regs = []
        for k in range(1, n + 1):
            coords = np.argwhere(lab == k)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            if (r1 - r0 + 1) >= 5 and (c1 - c0 + 1) >= 5:
                regs.append({"r0": int(r0), "r1": int(r1), "c0": int(c0), "c1": int(c1)})
        return regs

    def boundary_dirs_for_region(reg, stripe_colors):
        dirs = {}
        r0, r1, c0, c1 = reg["r0"], reg["r1"], reg["c0"], reg["c1"]
        if r0 - 1 >= 0:
            mc = Counter(grid[r0 - 1, c0:c1 + 1].tolist()).most_common(1)[0][0]
            if mc in stripe_colors:
                dirs[int(mc)] = 0
        if r1 + 1 < H:
            mc = Counter(grid[r1 + 1, c0:c1 + 1].tolist()).most_common(1)[0][0]
            if mc in stripe_colors:
                dirs[int(mc)] = 2
        if c0 - 1 >= 0:
            mc = Counter(grid[r0:r1 + 1, c0 - 1].tolist()).most_common(1)[0][0]
            if mc in stripe_colors:
                dirs[int(mc)] = 3
        if c1 + 1 < W:
            mc = Counter(grid[r0:r1 + 1, c1 + 1].tolist()).most_common(1)[0][0]
            if mc in stripe_colors:
                dirs[int(mc)] = 1
        return dirs

    def xform_coords(coords, shape, flip_ud, rot_k):
        result = transform_coords(coords, shape, flip_ud, rot_k)
        new_shape = apply_transform(np.zeros(shape), flip_ud, rot_k).shape
        return list(result), new_shape

    def choose_best_op(train, divider_dirs):
        bbox0 = train["bbox"]
        markers_by_color = defaultdict(list)
        for v, r, c in train["marker_pixels"]:
            if v in divider_dirs:
                markers_by_color[v].append((r, c))
        assert markers_by_color, "Each train must expose divider-colored markers"
        assert train["body9"], "Each train must carry at least one coupler"

        best, best_score = (False, 0), -1
        for flip_ud in (False, True):
            for k in range(4):
                t_body9, _ = xform_coords(train["body9"], bbox0.shape, flip_ud, k)
                ref_r = sum(r for r, _ in t_body9) / len(t_body9)
                ref_c = sum(c for _, c in t_body9) / len(t_body9)
                score = 0
                for color, pts in markers_by_color.items():
                    pts_t, _ = xform_coords(pts, bbox0.shape, flip_ud, k)
                    mr = sum(r for r, _ in pts_t) / len(pts_t)
                    mc = sum(c for _, c in pts_t) / len(pts_t)
                    dr, dc = mr - ref_r, mc - ref_c
                    dirc = (0 if dr < 0 else 2) if abs(dr) >= abs(dc) else (3 if dc < 0 else 1)
                    if dirc == divider_dirs[color]:
                        score += 1
                if score > best_score:
                    best_score, best = score, (flip_ud, k)
        return best

    edge_mask, edge_colors, edge_info = find_edge_stripes()
    internal_mask, internal_colors = find_internal_stripes(edge_info)
    stripe_mask = edge_mask | internal_mask
    stripe_colors = edge_colors | internal_colors
    regions = regions_from_stripes(stripe_mask)

    groups = station_groups()
    groups_by_size = defaultdict(list)
    for g in groups:
        groups_by_size[len(g)].append(g)
    station_set = set(pt for g in groups for pt in g)

    comp_mask = (grid != bg) & (~stripe_mask)
    lab, n = label(comp_mask, structure=np.ones((3, 3), dtype=int))
    trains = []
    for k in range(1, n + 1):
        coords = np.argwhere(lab == k)
        vals = [int(grid[r, c]) for r, c in coords]
        if 8 not in vals:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        bbox = grid[r0:r1 + 1, c0:c1 + 1].copy()
        abs_coords = [(int(r), int(c)) for r, c in coords]
        body9, marker_pixels = [], []
        for r, c in abs_coords:
            v = int(grid[r, c])
            if v == 9 and (r, c) not in station_set:
                body9.append((int(r - r0), int(c - c0)))
            if v in stripe_colors:
                marker_pixels.append((v, int(r - r0), int(c - c0)))
        trains.append(
            {
                "abs_coords": abs_coords,
                "bbox": bbox,
                "body9": body9,
                "marker_pixels": marker_pixels,
            }
        )

    out = grid.copy()
    for t in trains:
        for r, c in t["abs_coords"]:
            out[r, c] = bg

    for t in trains:
        n9 = len(t["body9"])
        assert n9 > 0, "Each train must carry at least one coupler"
        candidate_groups = groups_by_size.get(n9)
        assert candidate_groups is not None and len(candidate_groups) == 1, (
            f"Expected one station group for coupler count {n9}"
        )
        g = candidate_groups[0]
        gset = set(g)
        matching_regions = [
            rr
            for rr in regions
            if all(rr["r0"] <= r <= rr["r1"] and rr["c0"] <= c <= rr["c1"] for r, c in g)
        ]
        assert len(matching_regions) == 1, "Each station group must belong to exactly one region"
        reg = matching_regions[0]
        divider_dirs = boundary_dirs_for_region(reg, stripe_colors)
        op = choose_best_op(t, divider_dirs)
        tbbox = apply_transform(t["bbox"], *op)
        t_body9, (H1, W1) = xform_coords(t["body9"], t["bbox"].shape, *op)
        body9_set = set(t_body9)
        matching_offsets = {
            (sr - br, sc - bc)
            for br, bc in body9_set
            for sr, sc in gset
            if all((r + sr - br, c + sc - bc) in gset for r, c in body9_set)
        }
        assert len(matching_offsets) == 1, "Couplers must align to a unique station-group offset"
        off_r, off_c = next(iter(matching_offsets))

        ok = True
        for r in range(H1):
            for c in range(W1):
                v = int(tbbox[r, c])
                if v == bg or v == 9:
                    continue
                rr, cc = r + off_r, c + off_c
                if not (reg["r0"] <= rr <= reg["r1"] and reg["c0"] <= cc <= reg["c1"]):
                    ok = False
                    break
                if stripe_mask[rr, cc] or (out[rr, cc] != bg and out[rr, cc] != 9):
                    ok = False
                    break
            if not ok:
                break
        assert ok, "Docked train must fit entirely within its target region"

        for r in range(H1):
            for c in range(W1):
                v = int(tbbox[r, c])
                if v != bg and v != 9:
                    out[r + off_r, c + off_c] = v
    return out


# --------------- generator ---------------

GRID_H_RANGE = (13, 31)
GRID_W_RANGE = (16, 29)
GRID_MIN_AREA = 300
MIN_REGION_DIM = 10
MIN_REGION_DIRS = 3
INTERNAL_STRIPE_THICK = 2

STRIPE_THICK_RANGE = (2, 4)
INTERNAL_PROB = 0.62

TRAIN_COUNTS = [2, 3]
STATION_SIZES = [1, 2, 4]

BODY_MIN = 8
BODY_MAX = 20
PATCH_H_RANGE = (4, 9)
PATCH_W_RANGE = (3, 8)
FRAME_H_RANGE = (4, 7)
FRAME_W_RANGE = (4, 7)
THREE_MARKER_PROB = 0.65

LAYOUT_RETRIES = 200
DOCK_RETRIES = 30
GENERATE_RETRIES = 300

COLOR_POOL = list(range(8))

OPS = [
    (False, 0),
    (False, 1),
    (False, 2),
    (False, 3),
    (True, 0),
    (True, 1),
    (True, 2),
    (True, 3),
]
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _apply_transform(arr, flip_ud=False, rot_k=0):
    out = np.flipud(arr) if flip_ud else arr
    return np.rot90(out, int(rot_k))


def _inverse_patch(patch, op):
    for inv in OPS:
        cand = _apply_transform(patch, *inv)
        if np.array_equal(_apply_transform(cand, *op), patch):
            return cand
    raise ValueError("no inverse transform")


def _valid_ops_for_patch(patch):
    hlo, hhi = PATCH_H_RANGE
    wlo, whi = PATCH_W_RANGE
    valid = []
    for op in OPS:
        h, w = _inverse_patch(patch, op).shape
        if hlo <= h <= hhi and wlo <= w <= whi:
            valid.append(op)
    return valid


def _bbox(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _dims(coords):
    r0, c0, r1, c1 = _bbox(coords)
    return r1 - r0 + 1, c1 - c0 + 1


def _dilate(mask):
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[:-1, 1:] |= mask[1:, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _assign_region_dirs(regions, edge_col, top, bottom, left, right, H, W, internal):
    for reg in regions:
        dirs = {}
        if top and reg["r0"] == top:
            dirs[edge_col["top"]] = 0
        if bottom and reg["r1"] == H - bottom - 1:
            dirs[edge_col["bottom"]] = 2
        if left and reg["c0"] == left:
            dirs[edge_col["left"]] = 3
        if right and reg["c1"] == W - right - 1:
            dirs[edge_col["right"]] = 1
        if internal:
            kind, pos, col = internal
            if kind == "v":
                dirs[col] = 1 if reg["c1"] == pos - 1 else 3
            else:
                dirs[col] = 2 if reg["r1"] == pos - 1 else 0
        reg["dirs"] = dirs
        rh = reg["r1"] - reg["r0"] + 1
        rw = reg["c1"] - reg["c0"] + 1
        if rh < MIN_REGION_DIM or rw < MIN_REGION_DIM or len(dirs) < MIN_REGION_DIRS:
            return False
    return True


def _layout():
    for _ in range(LAYOUT_RETRIES):
        use_internal = random.random() < INTERNAL_PROB
        if use_internal:
            orient = random.choice(["v", "h"])
            if orient == "v":
                edges = ["left", "right", random.choice(["top", "bottom"])]
            else:
                edges = ["top", "bottom", random.choice(["left", "right"])]
        else:
            orient = None
            edges = random.sample(["top", "bottom", "left", "right"], random.choice([3, 4]))

        thick = {e: random.randint(*STRIPE_THICK_RANGE) for e in edges}
        top, bottom = thick.get("top", 0), thick.get("bottom", 0)
        left, right = thick.get("left", 0), thick.get("right", 0)

        min_inner_for_split = MIN_REGION_DIM * 2 + INTERNAL_STRIPE_THICK
        candidates = []
        for H in range(*GRID_H_RANGE):
            for W in range(*GRID_W_RANGE):
                if H * W < GRID_MIN_AREA:
                    continue
                ih, iw = H - top - bottom, W - left - right
                if ih < MIN_REGION_DIM or iw < MIN_REGION_DIM:
                    continue
                if orient == "v" and iw < min_inner_for_split:
                    continue
                if orient == "h" and ih < min_inner_for_split:
                    continue
                candidates.append((H, W))
        if not candidates:
            continue

        H, W = random.choice(candidates)
        pool = COLOR_POOL[:]
        bg = random.choice(pool)
        pool.remove(bg)
        stripe_cols = random.sample(pool, len(edges) + int(use_internal))
        edge_col = dict(zip(edges, stripe_cols[: len(edges)]))
        internal_col = stripe_cols[-1] if use_internal else None
        grid = np.full((H, W), bg, dtype=int)

        if top:
            grid[:top, :] = edge_col["top"]
        if bottom:
            grid[H - bottom :, :] = edge_col["bottom"]
        if left:
            grid[top : H - bottom, :left] = edge_col["left"]
        if right:
            grid[top : H - bottom, W - right :] = edge_col["right"]

        inner = {"r0": top, "r1": H - bottom - 1, "c0": left, "c1": W - right - 1}
        regions, internal = [], None
        if orient == "v":
            lo, hi = left + MIN_REGION_DIM, W - right - MIN_REGION_DIM - INTERNAL_STRIPE_THICK
            if lo > hi:
                continue
            c = random.randint(lo, hi)
            grid[top : H - bottom, c : c + INTERNAL_STRIPE_THICK] = internal_col
            internal = ("v", c, internal_col)
            regions = [
                {"r0": top, "r1": H - bottom - 1, "c0": left, "c1": c - 1},
                {
                    "r0": top,
                    "r1": H - bottom - 1,
                    "c0": c + INTERNAL_STRIPE_THICK,
                    "c1": W - right - 1,
                },
            ]
        elif orient == "h":
            lo, hi = top + MIN_REGION_DIM, H - bottom - MIN_REGION_DIM - INTERNAL_STRIPE_THICK
            if lo > hi:
                continue
            r = random.randint(lo, hi)
            grid[r : r + INTERNAL_STRIPE_THICK, left : W - right] = internal_col
            internal = ("h", r, internal_col)
            regions = [
                {"r0": top, "r1": r - 1, "c0": left, "c1": W - right - 1},
                {
                    "r0": r + INTERNAL_STRIPE_THICK,
                    "r1": H - bottom - 1,
                    "c0": left,
                    "c1": W - right - 1,
                },
            ]
        else:
            regions = [inner]

        if _assign_region_dirs(regions, edge_col, top, bottom, left, right, H, W, internal):
            return grid, bg, regions
    return None


def _marker_specs(dirs, forbidden_dir):
    items = [(color, direction) for color, direction in dirs.items() if direction != forbidden_dir]
    if len(items) < 2:
        return None
    if len(items) >= 3 and random.random() < THREE_MARKER_PROB:
        return random.sample(items, 3)
    pairs = []
    for i, a in enumerate(items):
        for b in items[i + 1 :]:
            if (a[1] - b[1]) % 4 in (1, 3):
                pairs.append((a, b))
    return list(random.choice(pairs)) if pairs else random.sample(items, 2)


def _frame_cells(outer_h, outer_w):
    return {
        (r, c)
        for r in range(outer_h)
        for c in range(outer_w)
        if r in (0, outer_h - 1) or c in (0, outer_w - 1)
    }


def _coupler_line(size, side, outer_h, outer_w):
    if side in (0, 2):
        if outer_w < size + 1:
            return None
        start_hi = outer_w - size
        start = random.randint(0, start_hi)
        row = -1 if side == 0 else outer_h
        return {(row, start + i) for i in range(size)}
    if outer_h < size + 1:
        return None
    start_hi = outer_h - size
    start = random.randint(0, start_hi)
    col = -1 if side == 3 else outer_w
    return {(start + i, col) for i in range(size)}


def _body_neighbors(cell, body):
    r, c = cell
    return sum((r + dr, c + dc) in body for dr, dc in DIRS4)


def _marker_candidates_for_direction(body, couplers, direction):
    cr = sum(r for r, _ in couplers) / len(couplers)
    cc = sum(c for _, c in couplers) / len(couplers)
    scored = []
    for r, c in body:
        if _body_neighbors((r, c), body) < 2:
            continue
        if r in (min(rr for rr, _ in body), max(rr for rr, _ in body)) and c in (
            min(cc0 for _, cc0 in body),
            max(cc0 for _, cc0 in body),
        ):
            continue
        dr, dc = r - cr, c - cc
        if direction == 0:
            primary, secondary = -dr, abs(dc)
            side_bonus = int(r == min(rr for rr, _ in body))
        elif direction == 2:
            primary, secondary = dr, abs(dc)
            side_bonus = int(r == max(rr for rr, _ in body))
        elif direction == 3:
            primary, secondary = -dc, abs(dr)
            side_bonus = int(c == min(cc0 for _, cc0 in body))
        else:
            primary, secondary = dc, abs(dr)
            side_bonus = int(c == max(cc0 for _, cc0 in body))
        if primary < 1 or primary < secondary:
            continue
        scored.append(((side_bonus, primary, -secondary, random.random()), (r, c)))
    scored.sort(reverse=True)
    return [cell for _, cell in scored]


def _choose_marker_cells(body, couplers, specs):
    candidate_map = {spec: _marker_candidates_for_direction(body, couplers, spec[1]) for spec in specs}
    if any(not cells for cells in candidate_map.values()):
        return None
    used = set()
    markers = []
    for color, direction in sorted(specs, key=lambda spec: len(candidate_map[spec])):
        pool = [cell for cell in candidate_map[(color, direction)] if cell not in used]
        if not pool:
            return None
        choice = random.choice(pool[: min(5, len(pool))])
        used.add(choice)
        markers.append((choice, color, direction))
    return markers


def _normalize_coords(*groups):
    all_cells = [cell for group in groups for cell in group]
    r0, c0, _, _ = _bbox(all_cells)
    shifted = []
    for group in groups:
        shifted.append({(r - r0, c - c0) for r, c in group})
    return shifted


def _has_unique_orientation(patch, dirs):
    body9 = [(int(r), int(c)) for r, c in np.argwhere(patch == 9)]
    markers_by_color = defaultdict(list)
    for color in dirs:
        for r, c in np.argwhere(patch == color):
            markers_by_color[int(color)].append((int(r), int(c)))
    if not body9 or not markers_by_color:
        return False

    best_score = -1
    winners = []
    for op in OPS:
        t_body9 = list(transform_coords(body9, patch.shape, *op))
        ref_r = sum(r for r, _ in t_body9) / len(t_body9)
        ref_c = sum(c for _, c in t_body9) / len(t_body9)
        score = 0
        for color, pts in markers_by_color.items():
            pts_t = list(transform_coords(pts, patch.shape, *op))
            mr = sum(r for r, _ in pts_t) / len(pts_t)
            mc = sum(c for _, c in pts_t) / len(pts_t)
            dr, dc = mr - ref_r, mc - ref_c
            direction = (0 if dr < 0 else 2) if abs(dr) >= abs(dc) else (3 if dc < 0 else 1)
            if direction == dirs[color]:
                score += 1
        if score > best_score:
            best_score = score
            winners = [op]
        elif score == best_score:
            winners.append(op)
    return winners == [(False, 0)]


def _build_docked_patch(bg, dirs, size):
    for _ in range(80):
        side = random.choice([0, 1, 2, 3])
        specs = _marker_specs(dirs, side)
        if specs is None:
            continue

        options = []
        for outer_h in range(*FRAME_H_RANGE):
            for outer_w in range(*FRAME_W_RANGE):
                body_total = 2 * outer_h + 2 * outer_w - 4
                n8 = body_total - len(specs)
                if not (BODY_MIN <= n8 <= BODY_MAX):
                    continue
                couplers = _coupler_line(size, side, outer_h, outer_w)
                if couplers is None:
                    continue
                all_cells = _frame_cells(outer_h, outer_w) | couplers
                ph, pw = _dims(all_cells)
                if PATCH_H_RANGE[0] <= ph <= PATCH_H_RANGE[1] and PATCH_W_RANGE[0] <= pw <= PATCH_W_RANGE[1]:
                    options.append((outer_h, outer_w, couplers))
        if not options:
            continue

        outer_h, outer_w, couplers = random.choice(options)
        body = _frame_cells(outer_h, outer_w)
        body, couplers = _normalize_coords(body, couplers)
        markers = _choose_marker_cells(body, couplers, specs)
        if markers is None:
            continue

        all_cells = body | couplers
        ph, pw = _dims(all_cells)
        patch = np.full((ph, pw), bg, dtype=int)
        for r, c in body:
            patch[r, c] = 8
        for (r, c), color, _ in markers:
            patch[r, c] = color
        rel_couplers = []
        for r, c in couplers:
            patch[r, c] = 9
            rel_couplers.append((r, c))
        if BODY_MIN <= int((patch == 8).sum()) <= BODY_MAX and _has_unique_orientation(patch, dirs):
            return patch, rel_couplers
    return None


def _patch_locations(region, patch, bg, occupied, coupler_rel=None, station_clearance=False, margin=0):
    H, W = occupied.shape
    ph, pw = patch.shape
    coords = [(int(r), int(c)) for r, c in np.argwhere(patch != bg)]
    locs = []
    r_start = max(region["r0"] + margin, 0)
    r_stop = min(region["r1"] - margin - ph + 2, H - ph + 1)
    c_start = max(region["c0"] + margin, 0)
    c_stop = min(region["c1"] - margin - pw + 2, W - pw + 1)
    for rr in range(r_start, r_stop):
        for cc in range(c_start, c_stop):
            ok = True
            for pr, pc in coords:
                if occupied[rr + pr, cc + pc]:
                    ok = False
                    break
            if not ok:
                continue
            if station_clearance and coupler_rel:
                for pr, pc in coupler_rel:
                    ar, ac = rr + pr, cc + pc
                    if not (
                        region["r0"] + 2 <= ar <= region["r1"] - 2
                        and region["c0"] + 2 <= ac <= region["c1"] - 2
                    ):
                        ok = False
                        break
            if ok:
                locs.append((rr, cc))
    return locs


def _place_docked_trains(base, bg, regions):
    output = base.copy()
    occupied = output != bg

    n_trains = random.choice(TRAIN_COUNTS)
    sizes = random.sample(STATION_SIZES, n_trains)
    trains = []
    for size in sizes:
        for _ in range(DOCK_RETRIES):
            reg = random.choice(regions)
            result = _build_docked_patch(bg, reg["dirs"], size)
            if result is None:
                continue
            patch, coupler_rel = result
            locs = _patch_locations(reg, patch, bg, _dilate(occupied), coupler_rel, True, 0)
            if not locs:
                continue
            r0, c0 = random.choice(locs)
            paste(output, patch, r0, c0, transparent=bg)
            occupied |= _dilate((output != bg) & ~occupied) | (output != bg)
            station_abs = [(r0 + r, c0 + c) for r, c in coupler_rel]
            trains.append({"patch": patch, "station": station_abs})
            break
        else:
            return None
    return output, trains


def _scatter_trains(base, bg, regions, trains):
    H, W = base.shape
    input_grid = base.copy()
    station_mask = np.zeros((H, W), dtype=bool)
    for t in trains:
        for r, c in t["station"]:
            input_grid[r, c] = 9
            station_mask[r, c] = True

    valid_ops = []
    for t in trains:
        valid = _valid_ops_for_patch(t["patch"])
        if not valid:
            return None
        valid_ops.append(valid)
    ops = [random.choice(valid) for valid in valid_ops]
    if all(op == (False, 0) for op in ops):
        non_id = [op for op in valid_ops[0] if op != (False, 0)]
        if not non_id:
            return None
        ops[0] = random.choice(non_id)

    forbidden = _dilate(base != bg) | _dilate(station_mask)
    for t, op in zip(trains, ops):
        in_patch = _inverse_patch(t["patch"], op)
        regs = regions[:]
        random.shuffle(regs)
        all_locs = []
        for reg in regs:
            locs = _patch_locations(reg, in_patch, bg, forbidden, None, False, 1)
            all_locs.extend((reg, p) for p in locs)
        if not all_locs:
            return None
        _, (r0, c0) = random.choice(all_locs)
        paste(input_grid, in_patch, r0, c0, transparent=bg)
        ph, pw = in_patch.shape
        patch_mask = np.zeros((H, W), dtype=bool)
        patch_mask[r0 : r0 + ph, c0 : c0 + pw] = in_patch != bg
        forbidden |= _dilate(patch_mask)

    return input_grid


def _gen_attempt():
    layout = _layout()
    if layout is None:
        return None
    base, bg, regions = layout

    placed = _place_docked_trains(base, bg, regions)
    if placed is None:
        return None
    output, trains = placed

    input_grid = _scatter_trains(base, bg, regions, trains)
    if input_grid is None:
        return None

    counts = Counter(map(int, input_grid.flatten()))
    if counts[bg] != max(counts.values()):
        return None
    try:
        if np.array_equal(main(input_grid), output):
            return input_grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 6e4f6532 combo input")
