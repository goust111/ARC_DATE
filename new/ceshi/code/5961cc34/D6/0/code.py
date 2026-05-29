# Variant of 5961cc34
# Dimension: D6 — 色块停止条件与终点 (dst)
# Change: 射线遇到色块时停止并激活该色块；若未遇到色块则延伸到网格边界... → 穿透激活

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste

# ---------------------------------------------------------------------------
# Design analysis for the D6 variant
#
# design:
#   concretization:
#     - 输入外观保持原题：背景色上有若干蓝色菱形/八边形 blob，每个 blob 只有一侧
#       带 1-2 个绿色端口；底部仍是黄色帽子 + 红色竖杆。
#     - 输出仍只保留红色激活网络。但 D6 改的是“ray 的终点”：
#       原题一条 ray 命中第一个 blob 后立刻停止；现在 ray 会穿过 blob 继续前进，
#       因而一根竖直主射线可以连续点亮多个 blob。
#     - 为了让这一变化在 2-4 个样例里肉眼可见，生成器不再主打折线单链，而是稳定构造
#       2-3 个共线 blob 被起始竖线依次扫中的场景；每个 blob 仍会沿自己的绿色侧边发射
#       粗射线到边界。
#   behavior_trace:
#     original:
#       - 起始竖线向上走，第一次碰到蓝色就 break，只激活最下面那个 blob。
#       - 之后只有这个 blob 的端口 ray 会继续传播。
#     variant:
#       - 起始竖线向上整列扫描到边界；碰到第 1 个、第 2 个、第 3 个 blob 都会激活，
#         但不会停止。
#       - 这些 blob 全部入队，各自再从绿色端口向左右发射粗 ray。
#     visible_difference:
#       - 原题里同一列上方的第二个 blob 会保持蓝/绿并在输出消失；本变体里它也会变红，
#         所以“单列多命中”是最直接的可观察证据。
#   complexity_calibration:
#     - 如果仍然像原题那样每条 ray 最多命中一个 blob，则 D6 虽然在代码层改了，样例却
#       可能退化成与原题一致的视觉行为。所以生成器强制大多数样例具有 >=2 个主射线命中。
#     - 另一方面不把每条副射线都做成多重命中树，避免样例太乱；保留原有 blob 形状、
#       颜色体系、最多 1 个 distractor，复杂度仍在可读范围内。
#   implicit_assumption_review:
#     - main() 里原先两处 break（起始细 ray、端口粗 ray）都依赖“第一次命中即终止”。
#       D6 下这两个 break 都必须删除，否则高处 blob 永远不会被激活。
#     - activate_blob_at() 仍然只依赖命中的是蓝色 body 像素；生成器保证主射线穿过每个
#       激活 blob 的蓝色列，而不是只擦到绿色端口。
#     - 原 solver 的 `len(blobs) - len(activated) <= 1` 假设了至多一个 distractor。
#       生成器继续保持这一点，避免引入与 D6 无关的维度变化。
#   placement_legality:
#     - 主射线上的 blob 共享同一 start_col，但它们的 bbox 在纵向上至少隔 1 个背景行，
#       因此不会互相重叠，也不会让横向副射线穿进另一个主链 blob。
#     - 激活 blob 的端口只取 LEFT/RIGHT。这样副射线横向出界，不会和起始竖线形成难以
#       辨认的再穿插结构。
#     - distractor 仍然要求离所有对象至少 1 像素，且完全避开任何红色 ray 轨迹。
#   visual_observability:
#     - 人类可以直接比较：同一条竖直红线是否把多个蓝 blob 都变成红色。
#     - 颜色变化仍然只有“被扫到的 blob + 所有 ray 全部变红”这一条规则，没有新增隐藏
#       的颜色映射或依赖颜色 id 的排序。
#   cleanliness:
#     - 保留原题的高对比对象：蓝/绿 blob 与红 ray 分层清楚。
#     - 限制副射线只向左右发射，避免多个方向同时重叠在主干附近造成画面脏乱。
#   new_solver_assumptions_and_generator_guarantees:
#     - solver 假设每个被主射线命中的 blob 在 start_col 上至少有一个蓝像素。
#       generator 通过把 start_col 对齐到底边蓝像素窗口来保证这一点。
#     - solver 假设 blob 的端口仍可由“蓝色 body 4 邻接绿色像素”恢复。
#       generator 沿用原始 blob 构造，不改变端口编码方式。
# ---------------------------------------------------------------------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint a red relay network by scanning through blue blobs and following green ports.

    Input structure:
    - The background is dominant; large blue diamond/octagon blobs carry a short green bar on exactly one side.
    - A bottom red post topped by one yellow cap marks the launch column for the initial upward ray.

    Rule:
    - Repaint the post and cap red, then sweep a thin ray upward all the way to the top; every blue blob crossed by that ray activates, but the ray does not stop at the first hit.
    - Each activated blob turns red and emits a thick ray through its green side; those rays also keep travelling after crossing blobs, activating every new blob on their line until they leave the grid.

    Key insight: Green bars still encode direction and thickness, but D6 changes each ray from "first-hit only" into a full line scan that can light multiple blobs before reaching the boundary.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect the start marker: a bottom-touching thin red segment with a non-red cap ---
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
            if r1 != int(H - 1):
                continue
            if any(int(g[rr, c0]) != int(col) for rr in range(int(r0), int(r1) + 1)):
                continue
            if int(r0) - 1 < 0:
                continue
            cap = int(g[int(r0) - 1, int(c0)])
            if cap == int(bg) or cap == int(col):
                continue

            start_candidates.append(
                (
                    int(col),
                    int(cap),
                    int(c0),
                    [(int(rr), int(c0)) for rr in range(int(r0), int(r1) + 1)],
                )
            )

    assert len(start_candidates) == 1
    red_col, yellow_col, start_col, start_seg = start_candidates[0]

    # --- STEP 2) Recover the blob colors (blue bodies, green ports) ---
    counts = Counter(map(int, g.flatten()))
    other_cols = [(int(cnt), int(col)) for col, cnt in counts.items() if int(col) not in (bg, red_col, yellow_col)]
    assert other_cols
    max_count = max(cnt for cnt, _col in other_cols)
    blue_candidates = [int(col) for cnt, col in other_cols if cnt == max_count]
    assert len(blue_candidates) == 1
    blue_col = int(blue_candidates[0])

    green_candidates: list[int] = []
    for col in colors_present(g, exclude={bg, red_col, yellow_col, blue_col}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        ok = False
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(blue_col):
                    ok = True
                    break
            if ok:
                break
        if ok:
            green_candidates.append(int(col))

    assert len(green_candidates) == 1
    green_col = int(green_candidates[0])

    out = np.full_like(g, int(bg))

    # --- STEP 3) Extract blobs and map each blue pixel to its component id ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []

    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[int(r), int(c)] = bi
        blobs.append({"pts": pts, "ports": []})

    # --- STEP 4) Recover each blob's green-sided ports ---
    for blob in blobs:
        pts = blob["pts"]
        seen_dirs: set[tuple[int, int]] = set()
        for r, c in pts:
            for ddr, ddc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + ddr)
                cc = int(c + ddc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(green_col):
                    seen_dirs.add((int(ddr), int(ddc)))
        for ddr, ddc in seen_dirs:
            port_pixels = [
                (int(r), int(c))
                for (r, c) in pts
                if 0 <= int(r + ddr) < H and 0 <= int(c + ddc) < W and int(g[int(r + ddr), int(c + ddc)]) == int(green_col)
            ]
            assert port_pixels
            blob["ports"].append({"dir": (int(ddr), int(ddc)), "pixels": port_pixels})

    activated: set[int] = set()
    queue: list[int] = []

    def activate_blob_at(r: int, c: int) -> None:
        """Turn a blob red the first time any ray crosses one of its blue pixels."""
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(int(bi))
        queue.append(int(bi))
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = int(red_col)

    # --- STEP 5) Draw the initial thin ray upward; D6 means it never stops on a hit ---
    for r, c in start_seg:
        out[int(r), int(c)] = int(red_col)
    top_r = min(int(r) for r, _c in start_seg)
    assert int(top_r) > 0 and int(g[int(top_r) - 1, int(start_col)]) == int(yellow_col)
    out[int(top_r) - 1, int(start_col)] = int(red_col)
    for r in range(int(top_r) - 1, -1, -1):
        out[int(r), int(start_col)] = int(red_col)
        if int(g[int(r), int(start_col)]) == int(blue_col):
            activate_blob_at(int(r), int(start_col))

    def extend_thick(port_pixels: list[tuple[int, int]], d: tuple[int, int]) -> None:
        """Extend a thick ray to the boundary; every crossed blob activates, but none blocks the ray."""
        ddr, ddc = map(int, d)
        for pr, pc in port_pixels:
            r, c = int(pr), int(pc)
            while True:
                r += int(ddr)
                c += int(ddc)
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[int(r), int(c)] = int(red_col)
                if int(g[int(r), int(c)]) == int(blue_col):
                    activate_blob_at(int(r), int(c))

    # --- STEP 6) BFS over activated blobs; only the stop condition changed, not the relay structure ---
    while queue:
        bi = int(queue.pop(0))
        for port in blobs[bi]["ports"]:
            extend_thick(port["pixels"], port["dir"])

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"

    return out


# --------------- generator ---------------

BG = 8
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)

SHAPES = ((3, 4), (4, 4), (5, 4), (5, 6), (6, 6), (8, 8))
_SHAPE_WIDTHS = {
    (3, 4): (2, 4, 2),
    (4, 4): (2, 4, 4, 2),
    (5, 4): (2, 4, 4, 4, 2),
    (5, 6): (2, 4, 6, 4, 2),
    (6, 6): (2, 4, 6, 6, 4, 2),
    (8, 8): (2, 4, 6, 8, 8, 6, 4, 2),
}

H_MIN, H_MAX = 13, 27
W_MIN, W_MAX = 15, 27
START_COL_PAD = 2
START_H_MIN, START_H_MAX = 2, 3

START_BG_GAP = 2        # blank rows between yellow cap and the first trunk blob
TRUNK_BG_GAP = 1        # blank rows between consecutive trunk blobs on the main scan line
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1
GEN_RETRIES = 300


def _shape_mask(h: int, w: int) -> np.ndarray:
    widths = _SHAPE_WIDTHS[(h, w)]
    mask = np.zeros((h, w), dtype=bool)
    for r, ww in enumerate(widths):
        c0 = (w - ww) // 2
        mask[r, c0 : c0 + ww] = True
    return mask


def _perp_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    return rc[1] if d[0] else rc[0]


def _sort_on_ray(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    return sorted(coords, key=lambda p: _perp_value(p, d))


def _edge_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    pts = np.argwhere(mask)
    axis = 0 if d in (UP, DOWN) else 1
    extreme = pts[:, axis].min() if d in (UP, LEFT) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return _sort_on_ray([(int(r), int(c)) for r, c in pts[keep]], d)


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

    port_pixels: list[tuple[int, int]] = []
    for gr, gc in green:
        br, bc = gr - port_dir[0], gc - port_dir[1]
        if not (0 <= br < h and 0 <= bc < w):
            return None
        if not mask[br, bc] or (br, bc) in green_set:
            return None
        port_pixels.append((br, bc))

    blue = [(int(r), int(c)) for r, c in np.argwhere(mask) if (int(r), int(c)) not in green_set]
    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return {
        "size": (h, w),
        "mask": mask,
        "patch": patch,
        "blue": blue,
        "full": full,
        "port_dir": port_dir,
        "port_pixels": _sort_on_ray(port_pixels, port_dir),
    }


def _all_specs(port_dir: tuple[int, int] | None = None) -> list[dict]:
    dirs = [port_dir] if port_dir is not None else list(DIRS)
    out = [s for size in SHAPES for d in dirs if (s := _blob_spec(size, d)) is not None]
    random.shuffle(out)
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, dr), H + min(0, dr)
            c0, c1 = max(0, dc), W + min(0, dc)
            out[r0:r1, c0:c1] |= mask[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out


def _offset(offsets: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + r, c0 + c) for r, c in offsets]


def _trace_to_boundary(
    starts: list[tuple[int, int]], d: tuple[int, int], include_origin: bool, H: int, W: int
) -> list[tuple[int, int]]:
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for r, c in starts:
        if include_origin:
            cells.append((r, c))
        while True:
            r += dr
            c += dc
            if not (0 <= r < H and 0 <= c < W):
                break
            cells.append((r, c))
    return cells


def _mark_ray(out: np.ndarray, ray_mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    paint_coords(out, cells, RED)
    paint_coords(ray_mask, cells, True)


def _place_blob(inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, blob: dict) -> None:
    paste(inp, blob["spec"]["patch"], blob["r0"], blob["c0"], transparent=BG)
    for r, c in blob["full"]:
        occupied[r, c] = True
    for r, c in blob["blue"]:
        blue_mask[r, c] = True


def _place_distractor(
    inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, ray_mask: np.ndarray
) -> bool:
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
                candidates.append({
                    "spec": spec,
                    "r0": r0,
                    "c0": c0,
                    "full": full,
                    "blue": _offset(spec["blue"], r0, c0),
                })
    if not candidates:
        return False
    _place_blob(inp, occupied, blue_mask, random.choice(candidates))
    return True


def _trunk_specs() -> list[dict]:
    return [spec for spec in _all_specs() if spec["port_dir"] in (LEFT, RIGHT)]


def _find_trunk_layout(
    H: int, W: int, start_col: int, cap_r: int, active_count: int, occupied: np.ndarray
) -> list[dict] | None:
    specs = _trunk_specs()
    initial_limit = cap_r - START_BG_GAP - 1
    if initial_limit < 0:
        return None

    def dfs(entry_limit: int, remaining: int, occ: np.ndarray) -> list[dict] | None:
        if remaining == 0:
            return []
        occ_pad = _dilate8(occ)
        candidates: list[dict] = []

        for spec in specs:
            h, w = spec["mask"].shape
            for entry_local in _edge_offsets(spec["mask"], DOWN):
                c0 = start_col - int(entry_local[1])
                if not (0 <= c0 <= W - w):
                    continue
                min_entry_row = int(entry_local[0])
                if min_entry_row > entry_limit:
                    continue
                rows = list(range(min_entry_row, entry_limit + 1))
                random.shuffle(rows)
                for entry_row in rows:
                    r0 = int(entry_row - entry_local[0])
                    full = _offset(spec["full"], r0, c0)
                    if any(occ_pad[r, c] for r, c in full):
                        continue
                    candidates.append({
                        "spec": spec,
                        "r0": r0,
                        "c0": c0,
                        "full": full,
                        "blue": _offset(spec["blue"], r0, c0),
                        "ports": _sort_on_ray(_offset(spec["port_pixels"], r0, c0), spec["port_dir"]),
                    })

        random.shuffle(candidates)
        for cand in candidates:
            occ2 = occ.copy()
            for r, c in cand["full"]:
                occ2[r, c] = True
            next_limit = cand["r0"] - TRUNK_BG_GAP - 1
            rest = dfs(next_limit, remaining - 1, occ2)
            if rest is not None:
                return [cand] + rest
        return None

    return dfs(initial_limit, active_count, occupied.copy())


def _active_count_for(cap_r: int) -> int:
    if cap_r >= 12:
        return random.choices([2, 3], weights=[3, 2])[0]
    return 2


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    # Place the bottom start marker.
    start_h = random.randint(START_H_MIN, START_H_MAX)
    cap_r = H - start_h - 1
    start_col = random.randint(START_COL_PAD, W - START_COL_PAD - 1)
    inp[cap_r, start_col] = YELLOW
    out[cap_r, start_col] = RED
    occupied[cap_r, start_col] = True
    for r in range(cap_r + 1, H):
        inp[r, start_col] = RED
        out[r, start_col] = RED
        occupied[r, start_col] = True

    active_count = _active_count_for(cap_r)
    if active_count > MAX_BLOB_COUNT:
        return None
    trunk = _find_trunk_layout(H, W, start_col, cap_r, active_count, occupied)
    if trunk is None:
        return None

    # Place active blobs in the input.
    for blob in trunk:
        _place_blob(inp, occupied, blue_mask, blob)

    # Build the expected output: main vertical scan hits every trunk blob.
    start_path = _trace_to_boundary([(cap_r, start_col)], UP, True, H, W)
    _mark_ray(out, ray_mask, start_path)
    for blob in trunk:
        paint_coords(out, blob["full"], RED)
    for blob in trunk:
        side_path = _trace_to_boundary(blob["ports"], blob["spec"]["port_dir"], False, H, W)
        _mark_ray(out, ray_mask, side_path)

    max_distractors = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - active_count)
    num_distractors = random.randint(0, max_distractors)
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
    raise RuntimeError("failed to generate a valid 5961cc34 D6 input")
