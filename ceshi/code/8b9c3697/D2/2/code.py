# Variant of 8b9c3697
# Dimension: D2 — 源对象：红色色块属性 (src)
# Change: 红色矩形色块（1x1到3x2），分为匹配色块（每个结构最多1个，由齿数决定尺寸）和干扰色块（2-8个散布在网格中）... → 精灵带方向标记

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location


"""
design_analysis:
  variant_core:
    original: |
      结构从内部插槽向外做全局射线，哪个方向先看到红色矩形，就把那个红块沿该方向滑回结构内部。
      方向知识完全来自“结构-红块”的远距离共线关系。
    new: |
      每个红块都升级成一个局部精灵：红色矩形 + 一个单像素方向标记。
      标记固定贴在“滑入方向”的那一侧，因此解题时要先从红块局部读出方向，再判断它能否沿该方向对接某个结构。
      标记本身只是输入提示，不会在输出中保留。

  concrete_sprite_design:
    marker_color: fixed pink(6), distinct from bg/structure/red/black
    red_core: |
      仍然保持原题的矩形红核。匹配块尺寸仍由 teeth 决定，干扰块仍取 1x1 到 3x2 的矩形。
    marker_position_rule: |
      标记贴在红核将要前进的一侧中央；若边长为偶数，则取偏左/偏上的“中点”。
      例如：向上滑入时，标记位于红块上边外侧 1 格；向右滑入时，位于右边外侧 1 格。
    output_effect: |
      所有标记都会消失。匹配精灵只移动红核，轨迹仍然涂黑；干扰精灵整组消失。

  behavior_trace:
    original_local_step:
      scene: 一个 opening-down 的梳齿结构，下方放着 2x2 红块。
      logic: |
        结构从内部 socket 向下做射线，只要先碰到红色，就推断“这个块要向上滑入”。
    variant_local_step:
      scene: 同样布局，但 2x2 红块上边再贴一个粉色单像素。
      logic: |
        新 solver 先读到“粉色像素贴在红块上边”，因此知道它要向上移动；
        然后只检查“沿上方这条通道是否存在可进入的 socket”。
        旧 solver 若仍按原法从结构射线找红色，会先撞到粉色标记而不是红色，从而不能再把方向完全建立在全局射线上。
    visible_difference: |
      输入里会出现“朝结构侧多出一个彩色触角”的红精灵；输出里这个触角消失，保留原题的红块入槽 + 黑色拖痕。

  complexity_calibration:
    why_not_use_color_id_for_direction: |
      方向完全由标记所在边决定，不由颜色编号映射，以免规则退化成不可见的 id 对应关系。
    why_single_marker_is_enough: |
      结构数量仍为 2-3，红精灵总数仍为 5-11，方向只取四个正交档位。
      单像素标记既能肉眼看出差异，又不会把输入变成复杂箭头语法。
    ambiguity_control: |
      结构仍保持原题的 comb/socket 形状，匹配块尺寸仍由 teeth 决定，因此“哪个红核能真正塞进哪个结构”仍有几何约束；
      生成器额外禁止干扰精灵出现在任意 socket 射线上，避免错误匹配。

  line_by_line_assumption_audit_of_original_main:
    red_component_extraction: |
      原代码默认“所有红色像素都属于可移动块”，现在不再成立，因为方向标记不是红色。
      新版仍以红色连通块作为移动主体，但必须额外在其四邻域寻找单像素标记并把它们排除出结构检测。
    structure_mask: |
      原代码把所有 non-bg 且 non-red 的像素都视为结构；变体后这会把方向标记误当结构。
      新版必须先识别并扣除这些单像素标记，再做结构连通域。
    ray_hit_test: |
      原代码把“射线碰到红色”当作同时完成了两件事：找到对象 + 推断方向。
      变体后这条假设失效。新版只允许射线命中标记像素，并要求该标记已在局部被解析成一个方向。
    output_cleanup: |
      原代码只清除红色像素；现在若不额外清掉标记，输出会残留粉色噪声。
    shift_logic: |
      原代码的位移对齐逻辑其实依赖的是“source 在 structure 的哪一侧”，不是“颜色本身”。
      这一点在变体中仍成立，所以保留该部分，只是把方向来源改成标记。

  placement_legality:
    matched_sprites: |
      仍按 output-first 生成：先画好 docked red + black trail，再反推输入中的 source red。
      方向标记放在 source red 朝向结构的一侧，且位置落在未来黑色轨迹上，因此输出不会新增额外图层冲突。
    distractors: |
      干扰精灵使用完整 sprite patch（红核+标记）放置，整体不得越界、重叠，且任一像素都不能落在任何结构 socket 的射线上。
    solver_compatibility: |
      生成后继续用 main(input) == output 做 oracle 验证，保证生成器构造满足 solver 的全部新假设。

  visual_observability:
    human_inference: |
      观察 2-4 个样例时，可以看到所有红块都多了同色单像素触角；
      真正参与滑入的块，其触角总朝着结构开口，且输出中触角消失、红核沿该朝向进入结构。
    cleanliness: |
      标记只有 1 像素，结构之间保留原题间距，干扰精灵仍限制数量和尺寸，避免画面被高密度彩色噪声淹没。

  new_solver_assumptions_to_be_guaranteed_by_generator:
    - 每个红色连通块恰好对应一个正交相邻的单像素标记。
    - 标记不会与结构相连，也不会与其他红块共享。
    - 每个可匹配结构至多有一个 marker-visible 精灵；干扰精灵不会让射线先撞到它们的标记。
    - 匹配精灵的红核尺寸仍与目标 structure 的 cavity 几何兼容。
"""


MARKER_COLOR = int(Color.PINK)
RED = int(Color.RED)
BLACK = int(Color.BLACK)
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _side_center(r0: int, c0: int, r1: int, c1: int, dirv: tuple[int, int]) -> tuple[int, int]:
    dr, dc = dirv
    if dr == -1:
        return int(r0 - 1), int(c0 + (c1 - c0) // 2)
    if dr == 1:
        return int(r1 + 1), int(c0 + (c1 - c0) // 2)
    if dc == -1:
        return int(r0 + (r1 - r0) // 2), int(c0 - 1)
    return int(r0 + (r1 - r0) // 2), int(c1 + 1)


def _marker_dir(marker: tuple[int, int], bbox: tuple[int, int, int, int]) -> tuple[int, int] | None:
    mr, mc = marker
    r0, c0, r1, c1 = bbox
    if mr == r0 - 1 and c0 <= mc <= c1:
        return (-1, 0)
    if mr == r1 + 1 and c0 <= mc <= c1:
        return (1, 0)
    if mc == c0 - 1 and r0 <= mr <= r1:
        return (0, -1)
    if mc == c1 + 1 and r0 <= mr <= r1:
        return (0, 1)
    return None


def _sprite_patch(height: int, width: int, move_dir: tuple[int, int]) -> np.ndarray:
    dr, dc = move_dir
    if dr != 0:
        patch = np.full((height + 1, width), -1, dtype=int)
        if dr == -1:
            patch[0, (width - 1) // 2] = MARKER_COLOR
            patch[1:, :] = RED
        else:
            patch[:height, :] = RED
            patch[height, (width - 1) // 2] = MARKER_COLOR
    else:
        patch = np.full((height, width + 1), -1, dtype=int)
        if dc == -1:
            patch[(height - 1) // 2, 0] = MARKER_COLOR
            patch[:, 1:] = RED
        else:
            patch[:, :width] = RED
            patch[(height - 1) // 2, width] = MARKER_COLOR
    return patch


def _extract_sprites(g: np.ndarray, bg: int) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int, int, int]], dict[tuple[int, int], int], dict[int, tuple[int, int]], dict[int, tuple[int, int]], np.ndarray]:
    red_comps = components_mask(g == RED, connectivity=4)
    assert red_comps, "expected at least one red sprite"

    red_coords: list[list[tuple[int, int]]] = []
    red_bbox: list[tuple[int, int, int, int]] = []
    red_at: dict[tuple[int, int], int] = {}
    for rid, coords in enumerate(red_comps):
        pts = [(int(r), int(c)) for r, c in coords]
        red_coords.append(pts)
        red_bbox.append(tuple(map(int, bbox_coords(coords))))
        for rc in pts:
            red_at[rc] = int(rid)

    marker_of_red: dict[int, tuple[int, int]] = {}
    move_dir_of_red: dict[int, tuple[int, int]] = {}
    marker_mask = np.zeros_like(g, dtype=bool)

    for coords in components_mask(g == MARKER_COLOR, connectivity=4):
        if len(coords) != 1:
            continue
        mr, mc = map(int, coords[0])
        adj_rids = {int(red_at[(rr, cc)]) for rr, cc in ((mr - 1, mc), (mr + 1, mc), (mr, mc - 1), (mr, mc + 1)) if (rr, cc) in red_at}
        if len(adj_rids) != 1:
            continue
        rid = int(next(iter(adj_rids)))
        dirv = _marker_dir((mr, mc), red_bbox[rid])
        if dirv is None:
            continue
        if rid in marker_of_red:
            raise AssertionError("each red component must have exactly one direction marker")
        marker_of_red[rid] = (mr, mc)
        move_dir_of_red[rid] = dirv
        marker_mask[mr, mc] = True

    assert len(marker_of_red) == len(red_coords), "every red component must carry one side marker"
    return red_coords, red_bbox, marker_of_red, move_dir_of_red, red_at, marker_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert direction-marked red sprites into interior sockets of large structures, leaving a black trail.

    Input structure:
    - bg = most common. Large non-bg structures still provide comb-like interior cavities.
    - Every red rectangle has one adjacent non-red marker pixel on one side, encoding the slide direction.

    Rule:
    - Read each red sprite's direction from its attached side marker.
    - For each structure, only consider socket rays whose outward direction is opposite that declared slide direction.
    - Slide the matched red rectangle into the socket, erase the marker, and paint the traversed red positions black.
    - Unmatched direction-marked sprites disappear completely.

    Key insight: The motion direction is no longer inferred from long-distance red visibility; it is carried locally by each sprite's one-pixel side marker.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    assert RED in set(map(int, np.unique(g)))

    red_coords, red_bbox, marker_of_red, move_dir_of_red, _red_at, marker_mask = _extract_sprites(g, bg)

    struct_mask = (g != int(bg)) & (g != RED) & (g != MARKER_COLOR)
    struct_comps = components_mask(struct_mask, connectivity=4)
    assert struct_comps, "expected at least one non-bg structure to dock into"

    struct_coords: list[set[tuple[int, int]]] = []
    struct_bbox: list[tuple[int, int, int, int]] = []
    for coords in struct_comps:
        pts = {(int(r), int(c)) for r, c in coords}
        struct_coords.append(pts)
        struct_bbox.append(tuple(map(int, bbox_coords(coords))))

    marker_to_red = {marker_of_red[rid]: int(rid) for rid in marker_of_red}

    struct_best: list[tuple[tuple[int, int], int, tuple[int, int], tuple[int, int], int]] = []
    # entries: ((hit_count, -min_dist), rid, outward_dir, anchor, sid)
    for sid, S in enumerate(struct_coords):
        sr0, sc0, sr1, sc1 = map(int, struct_bbox[sid])
        best = None

        for out_dr, out_dc in _DIRS:
            wanted_move = (-int(out_dr), -int(out_dc))
            hits: dict[int, list] = defaultdict(lambda: [0, 10**9, None])
            for r, c in S:
                nr, nc = int(r + out_dr), int(c + out_dc)
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if int(g[nr, nc]) != int(bg):
                    continue
                if not (sr0 < nr < sr1 and sc0 < nc < sc1):
                    continue

                rr, cc = int(nr), int(nc)
                while 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(bg):
                    rr += int(out_dr)
                    cc += int(out_dc)

                rid = marker_to_red.get((int(rr), int(cc)))
                if rid is None:
                    continue
                if move_dir_of_red[int(rid)] != wanted_move:
                    continue

                rec = hits[int(rid)]
                rec[0] += 1
                dist = abs(int(rr) - int(r)) + abs(int(cc) - int(c))
                if int(dist) < int(rec[1]):
                    rec[1] = int(dist)
                    rec[2] = (int(r), int(c))

            if not hits:
                continue

            rid_best = max(hits.keys(), key=lambda k: (int(hits[k][0]), -int(hits[k][1])))
            cnt, mind, anchor = hits[int(rid_best)]
            cand = (int(cnt), -int(mind), int(rid_best), (int(out_dr), int(out_dc)), anchor)
            if best is None or cand > best:
                best = cand

        if best is not None:
            cnt, neg_mind, rid, out_dir, anchor = best
            struct_best.append(((int(cnt), int(neg_mind)), int(rid), (int(out_dir[0]), int(out_dir[1])), anchor, int(sid)))

    struct_best.sort(reverse=True, key=lambda t: t[0])
    used_red: set[int] = set()
    assignments: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for _score, rid, out_dir, anchor, _sid in struct_best:
        if int(rid) in used_red or anchor is None:
            continue
        used_red.add(int(rid))
        assignments.append((int(rid), (int(out_dir[0]), int(out_dir[1])), (int(anchor[0]), int(anchor[1]))))

    out = g.copy()
    out[out == RED] = int(bg)
    out[marker_mask] = int(bg)

    for rid, (out_dr, out_dc), (ar, ac) in assignments:
        coords = red_coords[int(rid)]
        r0, c0, r1, c1 = map(int, red_bbox[int(rid)])
        tr, tc = int(ar + out_dr), int(ac + out_dc)

        if out_dr == 0 and out_dc == 1:
            shift_r, shift_c = 0, int(tc - c0)
        elif out_dr == 0 and out_dc == -1:
            shift_r, shift_c = 0, int(tc - c1)
        elif out_dr == 1 and out_dc == 0:
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
                rr = int(r + off_r)
                cc = int(c + off_c)
                if 0 <= rr < H and 0 <= cc < W:
                    out[rr, cc] = int(BLACK)

        for r, c in coords:
            rr = int(r + shift_r)
            cc = int(c + shift_c)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(RED)

    return out


# --------------- generator ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (19, 23)
GRID_W_RANGE = (22, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

# ── CFG: structures ─────────────────────────────────────────────────
STRUCT_WIDTH_TEETH = ((7, 1), (8, 2), (9, 1), (9, 3))
STRUCT_COUNT_RANGE = (2, 3)
UNMATCHED_PROB = 0.35
SLIDE_RANGE = (4, 8)
STRUCT_PADDING = 4

# ── CFG: distractors ────────────────────────────────────────────────
DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
TOTAL_RED_MIN = 5
TOTAL_RED_MAX = 11
DISTRACTOR_MIN = 2
DISTRACTOR_MAX = 8

# ── CFG: retry ──────────────────────────────────────────────────────
MAX_RETRIES = 400


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _structure_array(width: int, teeth: int, bg: int, sc: int) -> np.ndarray:
    g = np.full((4, width), bg, dtype=int)
    start = (width - teeth) // 2
    g[0, :] = sc
    g[1, 1] = g[1, width - 2] = sc
    g[2:4, 1] = g[2:4, width - 2] = sc
    g[1, start:start + teeth] = sc
    return g


def _matched_patches(width: int, teeth: int, slide: int, rot_k: int, bg: int, sc: int) -> dict:
    red_h = min(teeth, 2)
    base = _structure_array(width, teeth, bg, sc)
    patch_h = 2 + slide + red_h

    structure = np.full((patch_h, width), bg, dtype=int)
    paste(structure, base, 0, 0, transparent=bg)
    input_patch = structure.copy()
    output_patch = structure.copy()

    start = (width - teeth) // 2
    src_r0 = 2 + slide
    src_r1 = src_r0 + red_h
    input_patch[src_r0:src_r1, start:start + teeth] = RED
    input_patch[src_r0 - 1, start + (teeth - 1) // 2] = MARKER_COLOR

    for off in range(1, slide + 1):
        output_patch[2 + off:2 + off + red_h, start:start + teeth] = BLACK
    output_patch[2:2 + red_h, start:start + teeth] = RED

    return {
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(input_patch, k=rot_k),
        "output": np.rot90(output_patch, k=rot_k),
    }


def _unmatched_patches(width: int, teeth: int, rot_k: int, bg: int, sc: int) -> dict:
    structure = np.rot90(_structure_array(width, teeth, bg, sc), k=rot_k)
    return {"structure": structure, "input": structure.copy(), "output": structure.copy()}


def _structure_rays(struct_grid: np.ndarray, placed: list[dict], bg: int) -> np.ndarray:
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
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


def _place_structures(specs: list[dict], h: int, w: int, bg: int, sc: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]] | None:
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


def _place_distractors(inp: np.ndarray, struct_grid: np.ndarray, placed: list[dict], bg: int, sc: int, count: int) -> bool:
    h, w = inp.shape
    forbidden = _structure_rays(struct_grid, placed, bg)
    bbox_blocked = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        bbox_blocked[r0:r1 + 1, c0:c1 + 1] = True

    shapes = list(DISTRACTOR_SHAPES)
    for _ in range(count):
        sprite_mask = (inp == RED) | (inp == MARKER_COLOR)
        blocked = forbidden | bbox_blocked | (inp != bg) | _dilate4(sprite_mask)
        canvas = np.full((h, w), bg, dtype=int)
        canvas[blocked] = sc
        random.shuffle(shapes)
        random_dirs = list(_DIRS)
        random.shuffle(random_dirs)
        ok = False
        for rh, rw in shapes:
            for move_dir in random_dirs:
                patch = _sprite_patch(rh, rw, move_dir)
                loc = _place_patch(canvas, patch, bg, padding=0)
                if loc is None:
                    continue
                paste(inp, patch, loc[0], loc[1], transparent=-1)
                ok = True
                break
            if ok:
                break
        if not ok:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_COLORS)
    sc = random.choice([c for c in STRUCT_COLORS if c not in (bg, RED, BLACK, MARKER_COLOR)])
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
    raise RuntimeError("failed to generate a valid input for 8b9c3697 D2 variant")
