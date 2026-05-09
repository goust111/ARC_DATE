from __future__ import annotations

# Variant of 8b9c3697
# Dimension: D6 — 网格形状：输出框架 (grid_shape)
# Change: 输出与输入网格尺寸完全相同（19-23行 x 22-27列），仅改变部分像素值
#   → 分离面板输出：每个结构的局部求解结果先被裁剪成独立面板，再按输入中的行优先顺序横向拼接。
#
# Design analysis
# - a) 具体化与行为追踪:
#   输入仍是原题的大画布：2-3 个梳齿结构散落在背景上，红块从结构开口一侧与内部插槽对齐。
#   原题输出是在原画布上直接把红块滑入、路径涂黑、无关红块抹除。
#   本变体保留“谁匹配谁、如何滑入、哪些格子变黑”的局部逻辑，但最后不再保留整张画布，
#   而是把每个结构的局部结果单独裁成小面板，再按结构在输入中的上到下、左到右顺序拼成一行。
#   例如：原先左上结构和右下结构都在同一输出图里；现在它们会变成两个小图，中间隔着黑色分隔带。
# - b) 复杂度校准:
#   结构数量仍为 2-3，匹配仍靠同样的射线可见性，未匹配结构仍可出现；因此规则没有简化成“只裁一个块”。
#   分隔带固定为 2 列黑色，避免与局部黑轨迹混淆，同时让“输出框架变化”肉眼可见。
# - c) 隐式假设审查:
#   原 solver 末尾默认输出尺寸与输入一致，并直接在整张网格上覆写颜色；这个前提在 D6 中失效。
#   因此保留前半段的匹配与位移计算，只替换最后的“build output”阶段为“build per-structure panels + concat”。
#   原来的红块贪心分配、位移公式、黑轨迹绘制都依赖结构/红块在输入中的几何关系，这些前提仍然成立。
# - d) 放置合法性:
#   生成器仍在单一输入画布中放置结构和红块，保持不重叠、无遮挡、干扰块不落在任一插槽的视线方向上。
#   输出面板并不反向影响输入放置；面板只是从每个局部解中裁出再拼接。
# - e) 视觉可观察性:
#   2-3 个结构会稳定地产生 2-3 个输出面板；分隔带是显眼的整高黑条，和原题“整图改像素”明显不同。
#   面板顺序按输入中的空间顺序排序，不依赖颜色编号等不可见信息。
# - f) 可视化整洁度:
#   每个面板只保留一个结构及其黑轨迹/最终红块，不带任何干扰红块；因此输出比原题整画布更整洁。
# - g) 颜色变化可推断性:
#   颜色角色不变：背景色保持原值，结构保持原值，红块仍表示滑入结果，黑色仍表示路径，另兼作分隔带。
# - 新增 solver 假设与 generator 保障:
#   solver 现在假设“每个面板按结构 bbox 的行优先顺序输出”；generator 用同样的顺序构造期望输出并校验。
#   solver 还假设至少存在一个匹配红块以触发原题逻辑；generator 继续保证 matched >= 1。

from collections import defaultdict
import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location


PANEL_DIVIDER_W = 2
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _crop_to_content(patch: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(patch != int(bg))
    assert coords.size, "panel crop expected at least one non-background cell"
    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _concat_panels(
    panels: list[np.ndarray], bg: int, divider_color: int, divider_w: int = PANEL_DIVIDER_W
) -> np.ndarray:
    assert panels
    if len(panels) == 1:
        return panels[0].copy()

    out_h = max(int(panel.shape[0]) for panel in panels)
    out_w = sum(int(panel.shape[1]) for panel in panels) + int(divider_w) * (len(panels) - 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    c = 0
    for idx, panel in enumerate(panels):
        paste(out, panel, 0, c)
        c += int(panel.shape[1])
        if idx + 1 < len(panels):
            out[:, c : c + divider_w] = int(divider_color)
            c += int(divider_w)
    return out


def _discover_assignments(
    g: np.ndarray,
) -> tuple[
    int,
    list[list[tuple[int, int]]],
    list[tuple[int, int, int, int]],
    list[set[tuple[int, int]]],
    list[tuple[int, int, int, int]],
    dict[int, tuple[int, tuple[int, int], tuple[int, int]]],
]:
    bg = int(get_bg(g))
    H, W = g.shape

    RED = int(Color.RED)
    assert RED in set(map(int, np.unique(g)))

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

    struct_mask = (g != bg) & (g != RED)
    struct_comps = components_mask(struct_mask, connectivity=4)
    assert struct_comps, "expected at least one non-bg non-red structure to dock into"

    struct_coords: list[set[tuple[int, int]]] = []
    struct_bbox: list[tuple[int, int, int, int]] = []
    for coords in struct_comps:
        pts = {(int(r), int(c)) for r, c in coords}
        struct_coords.append(pts)
        struct_bbox.append(tuple(map(int, bbox_coords(coords))))

    struct_best: list[tuple[tuple[int, int], int, tuple[int, int], tuple[int, int], int]] = []
    # entries: ((hit_count, -min_dist), rid, dir, anchor, sid)

    for sid, S in enumerate(struct_coords):
        sr0, sc0, sr1, sc1 = map(int, struct_bbox[sid])
        best = None  # (hit_count, -min_dist, rid, dir, anchor)

        for dr, dc in _DIRS:
            hits: dict[int, list] = defaultdict(lambda: [0, 10**9, None])
            for r, c in S:
                nr, nc = int(r + dr), int(c + dc)
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if int(g[nr, nc]) != bg:
                    continue
                if not (sr0 < nr < sr1 and sc0 < nc < sc1):
                    continue

                rr, cc = int(nr), int(nc)
                while 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == bg:
                    rr += int(dr)
                    cc += int(dc)

                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == RED:
                    rid = int(red_at[(rr, cc)])
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

    struct_best.sort(reverse=True, key=lambda t: t[0])
    used_red: set[int] = set()
    assignments: dict[int, tuple[int, tuple[int, int], tuple[int, int]]] = {}
    for _score, rid, dirv, anchor, sid in struct_best:
        if int(rid) in used_red or anchor is None:
            continue
        used_red.add(int(rid))
        assignments[int(sid)] = (int(rid), (int(dirv[0]), int(dirv[1])), (int(anchor[0]), int(anchor[1])))

    return bg, red_coords, red_bbox, struct_coords, struct_bbox, assignments


def _structure_panel(
    g: np.ndarray,
    bg: int,
    sid: int,
    red_coords: list[list[tuple[int, int]]],
    red_bbox: list[tuple[int, int, int, int]],
    struct_coords: list[set[tuple[int, int]]],
    assignments: dict[int, tuple[int, tuple[int, int], tuple[int, int]]],
) -> np.ndarray:
    RED = int(Color.RED)
    BLACK = int(Color.BLACK)

    cells: dict[tuple[int, int], int] = {}
    for r, c in struct_coords[sid]:
        cells[(int(r), int(c))] = int(g[r, c])

    if sid in assignments:
        rid, (dr, dc), (ar, ac) = assignments[sid]
        coords = red_coords[int(rid)]
        r0, c0, r1, c1 = map(int, red_bbox[int(rid)])
        tr, tc = int(ar + dr), int(ac + dc)

        if dr == 0 and dc == 1:
            shift_r, shift_c = 0, int(tc - c0)
        elif dr == 0 and dc == -1:
            shift_r, shift_c = 0, int(tc - c1)
        elif dr == 1 and dc == 0:
            shift_r, shift_c = int(tr - r0), 0
        else:
            shift_r, shift_c = int(tr - r1), 0

        steps = int(max(abs(int(shift_r)), abs(int(shift_c))))
        step_dr = 0 if shift_r == 0 else (1 if shift_r > 0 else -1)
        step_dc = 0 if shift_c == 0 else (1 if shift_c > 0 else -1)

        for step in range(0, steps):
            off_r = int(step * step_dr)
            off_c = int(step * step_dc)
            for r, c in coords:
                cells[(int(r + off_r), int(c + off_c))] = BLACK

        for r, c in coords:
            cells[(int(r + shift_r), int(c + shift_c))] = RED

    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    panel = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for (r, c), color in cells.items():
        panel[int(r - r0), int(c - c0)] = int(color)
    return panel


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solve each comb structure locally, then rebuild the answer as separated panels.

    Input structure:
    - A full-size background canvas with 2-3 large non-background comb/bracket structures.
    - Red rectangles lie outside the structures; some line up with an opening and some are distractors.

    Rule:
    - Use the same socket detection and ray-cast matching as the base task: a matched red component slides
      into the interior socket and paints its traversed cells black, while unmatched reds disappear.
    - After solving each structure, crop just that structure's local result to a tight bbox.
    - Sort those local crops by the structures' top-left positions in the input and concatenate them
      left-to-right, inserting a full-height black divider band between neighboring panels.

    Key insight: The motion rule is unchanged; the new challenge is to separate the solved scene into one
    panel per structure and reassemble the final output from those local views.
    """
    g = np.array(grid, dtype=int)
    bg, red_coords, red_bbox, struct_coords, struct_bbox, assignments = _discover_assignments(g)

    order = sorted(range(len(struct_coords)), key=lambda sid: (int(struct_bbox[sid][0]), int(struct_bbox[sid][1])))
    panels = [
        _structure_panel(g, bg, sid, red_coords, red_bbox, struct_coords, assignments)
        for sid in order
    ]
    return _concat_panels(panels, bg, int(Color.BLACK))


# --------------- generator (codex-att2, D6 panel-output variant) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (19, 23)          # input grid height
GRID_W_RANGE = (22, 27)          # input grid width
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
    g[1, start : start + teeth] = sc
    return g


def _matched_patches(width: int, teeth: int, slide: int, rot_k: int, bg: int, sc: int) -> dict:
    """Build input patch and cropped output panel for a matched structure+red pair."""
    red, black = int(Color.RED), int(Color.BLACK)
    red_h = min(teeth, 2)
    base = _structure_array(width, teeth, bg, sc)
    patch_h = 2 + slide + red_h

    structure = np.full((patch_h, width), bg, dtype=int)
    paste(structure, base, 0, 0, transparent=bg)
    input_patch = structure.copy()
    output_patch = structure.copy()

    start = (width - teeth) // 2
    for off in range(1, slide + 1):
        output_patch[2 + off : 2 + off + red_h, start : start + teeth] = black
    output_patch[2 : 2 + red_h, start : start + teeth] = red
    input_patch[2 + slide : 2 + slide + red_h, start : start + teeth] = red

    structure = np.rot90(structure, k=rot_k)
    input_patch = np.rot90(input_patch, k=rot_k)
    output_patch = np.rot90(output_patch, k=rot_k)
    return {
        "structure": structure,
        "input": input_patch,
        "panel": _crop_to_content(output_patch, bg),
    }


def _unmatched_patches(width: int, teeth: int, rot_k: int, bg: int, sc: int) -> dict:
    """Build patches for an unmatched structure (no red component)."""
    structure = np.rot90(_structure_array(width, teeth, bg, sc), k=rot_k)
    return {
        "structure": structure,
        "input": structure.copy(),
        "panel": _crop_to_content(structure, bg),
    }


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


def _place_patch(canvas: np.ndarray, patch: np.ndarray, bg: int, padding: int) -> tuple[int, int] | None:
    """Try to place patch on canvas; return (r, c) or None."""
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


def _unique_matched_visibility(grid: np.ndarray, bg: int, expected_matched: int) -> bool:
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
    return len(red_to_structs) == expected_matched and all(len(sids) == 1 for sids in red_to_structs.values())


def _place_structures(
    specs: list[dict], h: int, w: int, bg: int, sc: int
) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    """Place all structure patches on fresh canvases. Returns (inp, struct_grid, placed) or None."""
    inp = np.full((h, w), bg, dtype=int)
    struct_grid = np.full((h, w), bg, dtype=int)
    placed: list[dict] = []

    for spec in specs:
        loc = _place_patch(inp, spec["input"], bg, padding=STRUCT_PADDING)
        if loc is None:
            return None
        r0, c0 = loc
        paste(inp, spec["input"], r0, c0, transparent=bg)
        paste(struct_grid, spec["structure"], r0, c0, transparent=bg)

        coords_arr = np.argwhere(spec["structure"] == sc)
        coords = [(r0 + int(rr), c0 + int(cc)) for rr, cc in coords_arr]
        rows = [r for r, _c in coords]
        cols = [c for _r, c in coords]
        placed.append(
            {
                "coords": coords,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "panel": spec["panel"],
            }
        )

    return inp, struct_grid, placed


def _place_distractors(
    inp: np.ndarray, struct_grid: np.ndarray, placed: list[dict], bg: int, sc: int, count: int
) -> bool:
    """Scatter distractor red rectangles avoiding line-of-sight. Returns False on failure."""
    h, w = inp.shape
    forbidden = _structure_rays(struct_grid, placed, bg)
    bbox_blocked = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        bbox_blocked[r0 : r1 + 1, c0 : c1 + 1] = True
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
                red_mask[loc[0] : loc[0] + rh, loc[1] : loc[1] + rw] = True
                ok = True
                break
        if not ok:
            return False
    return True


def _expected_output(placed: list[dict], bg: int) -> np.ndarray:
    order = sorted(placed, key=lambda item: (int(item["bbox"][0]), int(item["bbox"][1])))
    panels = [item["panel"] for item in order]
    return _concat_panels(panels, bg, int(Color.BLACK))


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
    specs.sort(key=lambda s: np.count_nonzero(s["input"] != bg), reverse=True)

    result = _place_structures(specs, h, w, bg, sc)
    if result is None:
        return None
    inp, struct_grid, placed = result

    if not _unique_matched_visibility(inp, bg, matched):
        return None

    if not _place_distractors(inp, struct_grid, placed, bg, sc, distractor_count):
        return None

    expected = _expected_output(placed, bg)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 D6 panel variant")
