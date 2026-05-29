# Variant of 8b9c3697
# Dimension: D4 — 变换：轨迹涂色 (trajectory)
# Change: 红色色块滑入时，所有经过的中间位置涂黑（continuous_trail），形成从原始位置到最终停靠位置的黑色拖尾
# → 保留源不涂色：已匹配红块保留原位，滑行路径不改色，只在插槽终点额外出现红块

from __future__ import annotations

# Design analysis for the D4 variant
# ==================================
# a) 具体化与行为追踪
# - 原题里，每个梳齿结构会从开口方向“吸入”一个矩形红块；输出中，原红块位置和中间路径都会变黑，
#   最后只有插槽里的红块保留。变体后，匹配与滑动方向完全不变，但输出不再绘制黑色轨迹。
# - 具体表现：若一个 2x3 红块在结构下方 5 格处，原题输出会出现一段 2x3 的黑色竖向拖尾；
#   现在输出改为：原始 2x3 红块仍保留在起点，拖尾经过的背景仍保持背景色，只在结构内部插槽再出现
#   一个 2x3 红块。未匹配的红色干扰块仍按原题消失。
# - 流程上仅影响“build output”阶段：原逻辑先删掉所有红块，再给匹配块画黑拖尾并画终点红块；
#   新逻辑改为：只删除未匹配红块，已匹配红块原地保留，并额外在终点重画一份红块。
#
# b) 复杂度校准
# - 结构形状、旋转、红块尺寸、滑动距离、干扰红块数量全部保持原规格，避免把题简化成静态复制。
# - 仍保留 2-3 个结构、4-8 格滑移、2-8 个干扰红块，这样去掉拖尾后依旧需要基于开口方向和射线可见性
#   做空间匹配，而不是靠局部邻接关系直接看出答案。
#
# c) 隐式假设审查（与 main 直接相关）
# - 红块检测、结构检测、射线匹配、贪心分配都不依赖“轨迹必须变黑”，所以这部分可原样沿用。
# - 原版 `out[out == RED] = bg` 依赖“所有源红块最终都必须消失”；变体下此假设失效，因为匹配红块源位置要保留。
# - 原版逐步 `out[rr, cc] = BLACK` 依赖“路径要显式编码进输出”；变体下这些写入必须全部删除，否则会破坏
#   “路径保持输入原样”的新规则。
# - 新版 `main()` 的额外假设是：最终停靠位置不覆盖其他红块源位置。生成器沿用原始 line-of-sight 与结构间距
#   约束，保证终点位于结构内部 socket，和所有外部红块分离。
#
# d) 放置合法性
# - 生成器仍采用 output-first：先构造“结构 + 终点红块 + 源红块”的目标 patch，再反推出输入 patch。
# - 结构放置、bbox 间距、干扰红块避开所有 socket 射线的逻辑保持原样，因此不会引入越界、重叠或多义匹配。
# - 匹配红块源位置仍离 socket 4-8 格，确保终点和源点始终分离，输出里能同时看见两份红块。
#
# e) 视觉可观察性
# - 变体的核心可见差异是：输出中不存在黑色拖尾，而且匹配成功的源红块不会消失。
# - 因为每个样例至少有 1 个 matched 结构，且 slide >= 4，所以在 2-3 组样例里都能明显看到
#   “输入外部红块还在，同时内部多出一份红块”的规律。
#
# f) 可视化整洁度
# - 去掉黑色轨迹后，输出比原题更干净；为避免“太干净导致看不出移动”，保留多结构、多距离和干扰块，
#   让“哪些红块保留、哪些消失、哪些被复制到内部”仍然具备足够信息量。
#
# g) 颜色变化可推断性
# - 颜色体系不变：背景 1 色，结构 1 色，红块固定红色。变体没有新增颜色映射，也不使用颜色 id 参与任何排序。

from collections import defaultdict
import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock red components into the interior sockets of large comb structures without drawing a trail.

    Input structure:
    - The background is the most common color. Large non-background, non-red comb/bracket shapes sit on the canvas.
    - Several red rectangles lie outside the structures; some align with a structure opening, others are distractors.

    Rule:
    - For each structure, inspect interior socket cells and ray-cast outward to find the red component seen through its opening.
    - Slide that red component into the socket, but keep its source cells and every traversed path cell unchanged.
    - Only the docked destination is newly red; unmatched red distractors disappear.

    Key insight: The task is still solved by matching sockets to visible red components along a straight opening direction, but the movement is inferred from correspondence alone because no black trajectory is drawn.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    RED = int(Color.RED)

    assert RED in set(map(int, np.unique(g)))

    # --- connected components: red blobs ---
    red_comps = components_mask(g == RED, connectivity=4)
    assert red_comps

    red_coords: list[list[tuple[int, int]]] = []
    red_bbox: list[tuple[int, int, int, int]] = []
    red_at: dict[tuple[int, int], int] = {}

    for rid, coords in enumerate(red_comps):
        pts = [(int(r), int(c)) for r, c in coords]
        red_coords.append(pts)
        red_bbox.append(tuple(map(int, bbox_coords(coords))))
        for rc in pts:
            red_at[rc] = int(rid)

    # --- connected components: structures (non-bg, non-red) ---
    struct_mask = (g != bg) & (g != RED)
    struct_comps = components_mask(struct_mask, connectivity=4)
    assert struct_comps, "expected at least one non-bg non-red structure to dock into"

    struct_coords: list[set[tuple[int, int]]] = []
    struct_bbox: list[tuple[int, int, int, int]] = []
    for coords in struct_comps:
        pts = {(int(r), int(c)) for r, c in coords}
        struct_coords.append(pts)
        struct_bbox.append(tuple(map(int, bbox_coords(coords))))

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # For each structure, find its best (red, dir, anchor) triple.
    struct_best: list[tuple[tuple[int, int], int, tuple[int, int], tuple[int, int], int]] = []
    # entries: ((hit_count, -min_dist), rid, dir, anchor, sid)

    for sid, S in enumerate(struct_coords):
        sr0, sc0, sr1, sc1 = map(int, struct_bbox[sid])
        best = None  # (hit_count, -min_dist, rid, dir, anchor)

        for dr, dc in dirs:
            # rid -> [count, min_dist, best_anchor]
            hits: dict[int, list] = defaultdict(lambda: [0, 10**9, None])
            for r, c in S:
                nr, nc = int(r + dr), int(c + dc)
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if int(g[nr, nc]) != bg:
                    continue
                # Socket must lie strictly inside the structure bbox.
                if not (sr0 < nr < sr1 and sc0 < nc < sc1):
                    continue

                rr, cc = int(nr), int(nc)
                while 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == bg:
                    rr += int(dr)
                    cc += int(dc)

                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == RED:
                    rid = int(red_at[(int(rr), int(cc))])
                    rec = hits[rid]
                    rec[0] += 1
                    dist = abs(int(rr) - int(r)) + abs(int(cc) - int(c))
                    if int(dist) < int(rec[1]):
                        rec[1] = int(dist)
                        rec[2] = (int(r), int(c))

            if not hits:
                continue

            rid_best = max(hits.keys(), key=lambda k: (int(hits[k][0]), -int(hits[k][1])))
            cnt, mind, anchor = hits[rid_best]
            cand = (int(cnt), -int(mind), int(rid_best), (int(dr), int(dc)), anchor)
            if best is None or cand > best:
                best = cand

        if best is not None:
            cnt, neg_mind, rid, dirv, anchor = best
            struct_best.append(
                ((int(cnt), int(neg_mind)), int(rid), (int(dirv[0]), int(dirv[1])), anchor, int(sid))
            )

    # Assign each red component to at most one structure (greedy best-first).
    struct_best.sort(reverse=True, key=lambda t: t[0])
    used_red: set[int] = set()
    assignments: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for _score, rid, (dr, dc), anchor, _sid in struct_best:
        if int(rid) in used_red or anchor is None:
            continue
        used_red.add(int(rid))
        assignments.append((int(rid), (int(dr), int(dc)), (int(anchor[0]), int(anchor[1]))))

    # --- build output ---
    out = g.copy()

    # Only unmatched reds disappear. Matched sources stay visible at their original positions.
    for rid, coords in enumerate(red_coords):
        if int(rid) in used_red:
            continue
        for r, c in coords:
            out[int(r), int(c)] = bg

    for rid, (dr, dc), (ar, ac) in assignments:
        coords = red_coords[int(rid)]
        r0, c0, r1, c1 = map(int, red_bbox[int(rid)])
        tr, tc = int(ar + dr), int(ac + dc)  # socket cell

        # Shift so the component face nearest the structure lands on the socket.
        if dr == 0 and dc == 1:  # east; align min col
            shift_r, shift_c = 0, int(tc - c0)
        elif dr == 0 and dc == -1:  # west; align max col
            shift_r, shift_c = 0, int(tc - c1)
        elif dr == 1 and dc == 0:  # south; align min row
            shift_r, shift_c = int(tr - r0), 0
        else:  # north; align max row
            shift_r, shift_c = int(tr - r1), 0

        # Final placement of the red component. Path/source cells are intentionally untouched.
        for r, c in coords:
            rr = int(r + shift_r)
            cc = int(c + shift_c)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = RED

    return out


# --------------- generator (codex-att2, adapted for keep-source trajectory) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (19, 23)          # output grid height
GRID_W_RANGE = (22, 27)          # output grid width
BG_COLORS = (3, 4)               # candidate background colors
STRUCT_COLORS = (1, 4, 8)        # candidate structure colors

# ── CFG: structures ─────────────────────────────────────────────────
STRUCT_WIDTH_TEETH = ((7, 1), (8, 2), (9, 1), (9, 3))  # (back_wall_width, teeth_count)
STRUCT_COUNT_RANGE = (2, 3)      # number of comb/bracket structures
UNMATCHED_PROB = 0.35            # probability of 1 unmatched structure
SLIDE_RANGE = (4, 8)             # red slide distance into socket
STRUCT_PADDING = 4               # min gap between placed structures

# ── CFG: distractors ────────────────────────────────────────────────
DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
TOTAL_RED_MIN = 5                # min total red components (matched + distractors)
TOTAL_RED_MAX = 11               # max total red components
DISTRACTOR_MIN = 2               # absolute min distractors
DISTRACTOR_MAX = 8               # absolute max distractors

# ── CFG: retry ──────────────────────────────────────────────────────
MAX_RETRIES = 300                # outer retry loop for generate_input

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _dilate4(mask: np.ndarray) -> np.ndarray:
    """4-connected dilation of a boolean mask."""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _structure_array(width: int, teeth: int, bg: int, sc: int) -> np.ndarray:
    """Build a 4-row normalized comb structure (opening facing down)."""
    g = np.full((4, width), bg, dtype=int)
    start = (width - teeth) // 2
    g[0, :] = sc
    g[1, 1] = g[1, width - 2] = sc
    g[2:4, 1] = g[2:4, width - 2] = sc
    g[1, start:start + teeth] = sc
    return g


def _matched_patches(width: int, teeth: int, slide: int, rot_k: int,
                     bg: int, sc: int) -> dict:
    """Build input/output/structure patches for a matched structure+red pair."""
    red = int(Color.RED)
    red_h = min(teeth, 2)
    base = _structure_array(width, teeth, bg, sc)
    patch_h = 2 + slide + red_h

    structure = np.full((patch_h, width), bg, dtype=int)
    paste(structure, base, 0, 0, transparent=bg)
    input_patch = structure.copy()
    output_patch = structure.copy()

    start = (width - teeth) // 2
    # Source copy remains in the output.
    output_patch[2 + slide:2 + slide + red_h, start:start + teeth] = red
    # Docked copy appears inside the socket.
    output_patch[2:2 + red_h, start:start + teeth] = red
    input_patch[2 + slide:2 + slide + red_h, start:start + teeth] = red

    return {
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(input_patch, k=rot_k),
        "output": np.rot90(output_patch, k=rot_k),
    }


def _unmatched_patches(width: int, teeth: int, rot_k: int,
                       bg: int, sc: int) -> dict:
    """Build patches for an unmatched structure (no red component)."""
    structure = np.rot90(_structure_array(width, teeth, bg, sc), k=rot_k)
    return {"structure": structure, "input": structure.copy(), "output": structure.copy()}


def _structure_rays(struct_grid: np.ndarray, placed: list[dict], bg: int) -> np.ndarray:
    """Return a boolean mask of cells on line-of-sight from any socket outward."""
    h, w = struct_grid.shape
    forbidden = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        for dr, dc in _DIRS:
            for r, c in item["coords"]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if struct_grid[nr, nc] != bg:
                    continue
                if not (r0 < nr < r1 and c0 < nc < c1):
                    continue
                rr, cc = nr, nc
                while 0 <= rr < h and 0 <= cc < w and struct_grid[rr, cc] == bg:
                    forbidden[rr, cc] = True
                    rr += dr
                    cc += dc
    return forbidden


def _place_patch(canvas: np.ndarray, patch: np.ndarray, bg: int,
                 padding: int) -> tuple[int, int] | None:
    """Try to place patch on canvas; return (r, c) or None."""
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


def _unique_matched_visibility(grid: np.ndarray, bg: int,
                               expected_matched: int) -> bool:
    """Check that exactly expected_matched reds are each visible from exactly one structure."""
    h, w = grid.shape
    red_val = int(Color.RED)
    red_comps = components_mask(grid == red_val, connectivity=4)
    if len(red_comps) != expected_matched:
        return False

    red_at: dict[tuple[int, int], int] = {}
    for rid, coords in enumerate(red_comps):
        for r, c in coords:
            red_at[(int(r), int(c))] = rid

    struct_mask = (grid != bg) & (grid != red_val)
    struct_comps = components_mask(struct_mask, connectivity=4)
    red_to_structs: dict[int, set[int]] = {}
    struct_hits = 0

    for sid, coords in enumerate(struct_comps):
        pts = {(int(r), int(c)) for r, c in coords}
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        visible_rids: set[int] = set()
        for dr, dc in _DIRS:
            for r, c in pts:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if grid[nr, nc] != bg:
                    continue
                if not (r0 < nr < r1 and c0 < nc < c1):
                    continue
                rr, cc = nr, nc
                while 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == bg:
                    rr += dr
                    cc += dc
                if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == red_val:
                    visible_rids.add(red_at[(rr, cc)])
        if len(visible_rids) > 1:
            return False
        if visible_rids:
            struct_hits += 1
            rid = next(iter(visible_rids))
            red_to_structs.setdefault(rid, set()).add(sid)

    if struct_hits != expected_matched:
        return False
    return (
        len(red_to_structs) == expected_matched
        and all(len(sids) == 1 for sids in red_to_structs.values())
    )


def _place_structures(specs: list[dict], h: int, w: int, bg: int,
                      sc: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]] | None:
    """Place all structure patches on fresh canvases. Returns (inp, out, struct_grid, placed) or None."""
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    struct_grid = np.full((h, w), bg, dtype=int)
    placed: list[dict] = []

    for spec in specs:
        loc = _place_patch(out, spec["output"], bg, padding=STRUCT_PADDING)
        if loc is None:
            return None
        r0, c0 = loc
        paste(out, spec["output"], r0, c0, transparent=bg)
        paste(inp, spec["input"], r0, c0, transparent=bg)
        paste(struct_grid, spec["structure"], r0, c0, transparent=bg)

        coords_arr = np.argwhere(spec["structure"] == sc)
        coords = [(r0 + int(rr), c0 + int(cc)) for rr, cc in coords_arr]
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        placed.append({"coords": coords, "bbox": (min(rows), min(cols), max(rows), max(cols))})

    return inp, out, struct_grid, placed


def _place_distractors(inp: np.ndarray, struct_grid: np.ndarray, placed: list[dict],
                       bg: int, sc: int, count: int) -> bool:
    """Scatter distractor red rectangles avoiding line-of-sight. Returns False on failure."""
    h, w = inp.shape
    forbidden = _structure_rays(struct_grid, placed, bg)
    bbox_blocked = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        bbox_blocked[r0:r1 + 1, c0:c1 + 1] = True
    red_mask = inp == int(Color.RED)

    shapes = list(DISTRACTOR_SHAPES)
    for _ in range(count):
        blocked = forbidden | bbox_blocked | (inp != bg) | _dilate4(red_mask)
        canvas = np.full((h, w), bg, dtype=int)
        canvas[blocked] = sc
        random.shuffle(shapes)
        ok = False
        for rh, rw in shapes:
            patch = np.full((rh, rw), int(Color.RED), dtype=int)
            loc = _place_patch(canvas, patch, bg, padding=0)
            if loc is not None:
                paste(inp, patch, loc[0], loc[1], transparent=bg)
                red_mask[loc[0]:loc[0] + rh, loc[1]:loc[1] + rw] = True
                ok = True
                break
        if not ok:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid."""
    bg = random.choice(BG_COLORS)
    sc = random.choice([c for c in STRUCT_COLORS if c not in (bg, int(Color.RED), int(Color.BLACK))])
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_structs = random.randint(*STRUCT_COUNT_RANGE)
    unmatched = 1 if random.random() < UNMATCHED_PROB else 0
    matched = n_structs - unmatched
    if matched <= 0:
        return None

    min_dist = max(DISTRACTOR_MIN, TOTAL_RED_MIN - matched)
    max_dist = min(DISTRACTOR_MAX, TOTAL_RED_MAX - matched)
    if min_dist > max_dist:
        return None
    distractor_count = random.randint(min_dist, max_dist)

    specs = []
    for idx in range(n_structs):
        width, teeth = random.choice(STRUCT_WIDTH_TEETH)
        rot_k = random.randrange(4)
        if idx < matched:
            slide = random.randint(*SLIDE_RANGE)
            specs.append(_matched_patches(width, teeth, slide, rot_k, bg, sc))
        else:
            specs.append(_unmatched_patches(width, teeth, rot_k, bg, sc))
    random.shuffle(specs)
    specs.sort(key=lambda s: np.count_nonzero(s["output"] != bg), reverse=True)

    result = _place_structures(specs, h, w, bg, sc)
    if result is None:
        return None
    inp, out, struct_grid, placed = result

    if not _unique_matched_visibility(inp, bg, matched):
        return None

    if not _place_distractors(inp, struct_grid, placed, bg, sc, distractor_count):
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697")
