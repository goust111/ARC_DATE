# Combo variant of 8b9c3697 (index 20)
# combo_spec: D3/0+D6/2+D8/0
# Name: 嵌套间隔齿内部提取

from __future__ import annotations

import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location

# Design analysis
# ===============
# a) 具体化与行为追踪
# - 原题的每个结构只有 4 层深度，红块只需贴到最外层齿下方；D8/0 把“连续齿 + 矩形红块”
#   改成了“间隔齿 + 带缺口的红块”，而 D6/2 只是在求解结束后裁掉大画布。
# - 这次融合把结构本体也改成双层嵌套：外层仍是一个大支架，但其内部上方再嵌一层更窄的
#   小支架，真正的齿位于小支架处。规范朝向向下时，结构高度固定为 6：
#   第 0 行整行背板；第 1 行是外腔顶缘；第 2 行出现内层间隔齿；第 3-4 行是内腔；
#   第 5 行只有外层侧壁，形成向外开的走廊。
# - 具体例子：若齿型为 `10101`，原 D8 单变体会把红块停在较浅的 `2x5` 区域；融合后红块
#   仍是 `10101/11111`，但它必须继续滑到更深的内层目标位。输出不再保留整张图，只保留
#   每个结构的“内部窗口”并横向打包，因此解题者看到的是内腔被填充后的局部结果。
#
# b) 复杂度校准
# - 间隔齿仍只保留两档：跨度 3 的 `101` 和跨度 5 的 `10101`。这样一方面明显区别于原题
#   的连续矩形齿，另一方面不会把形状自由度放得过大。
# - 嵌套深度固定为双层，而不是再引入 3-4 层随机深度。这样“真正目标在内层”会稳定出现，
#   但识别规则仍然足够集中，可从少量样例归纳。
# - 结构数量仍保留 2-3 个，并保留 0-1 个无匹配结构和若干红色干扰块，避免融合后退化成
#   单对象模板匹配题。
#
# c) 隐式假设审查
# - 原 solver 依赖“只要从 socket 朝外看见红块就匹配”，这在间隔齿与嵌套深度同时存在时
#   已不成立：外层走廊中的矩形干扰红块不应被接受，所以新 solver 必须显式恢复结构模板、
#   读出齿型，并校验红块的精确布尔形状。
# - 原 solver 把最终停靠点锚定在离结构最近的一层空位；融合后该假设错误，因为外层空腔
#   中也有空位，但真正目标是内层齿下方的唯一窗口，因此新代码直接由模板生成目标坐标。
# - D6/2 在单变体中裁的是结构 bbox；这里若继续直接裁 bbox，会把大量外部背景角也带进来，
#   “仅输出结构内部区域”的变化就不够明确。因此新版本改为裁模板内部窗口：
#   规范朝向下的 `[1:, 1:-1]`，旋转回原朝向后再打包。
#
# d) 放置合法性
# - 生成器继续使用 output-first：先造出结构、最终红块和黑色轨迹，再把红块平移回输入位置。
# - 结构宽度只有 9 或 11，高度 6；匹配红块沿开口方向平移 4-8 格。输入画布相对原题略放大，
#   否则 2-3 个嵌套结构连同外部红块很容易挤不下。
# - 干扰红块放置时避开结构 bbox、避开所有合法滑入走廊、并与已有红块保持 4 邻域分离，
#   以免和真实匹配块连成一个连通分量。
#
# e) 视觉可观察性
# - 解题者能直接看到每个结构明显高于原题，且内部还有一层更窄的小框；真正的红块也不是
#   普通矩形，而是在靠齿一侧显出 `101` 或 `10101` 节奏。
# - 输出高度显著缩小，并且只剩下各结构的内部窗口。黑色轨迹只在这些窗口中保留可见片段，
#   使“先滑入，再裁内部”的推理链具备稳定视觉证据。
#
# f) 可视化整洁度
# - 结构都用单一颜色，移动红块只产生黑色轨迹，不额外引入新标记色。
# - 内部裁剪窗口去除了结构外的大片背景，也去掉了背板外沿，输出更紧凑，多个结构横向拼接
#   后仍容易分辨各自的局部形态。
#
# g) 变化的可推断性
# - 颜色规则和原题一致：红块最终仍是红色，路径仍变黑，其余红块消失。
# - 变化完全由几何结构驱动：更深的内层目标、间隔齿轮廓、以及内部裁剪。排序使用结构在输入
#   中的可见行优先顺序，不依赖颜色编号或隐藏元数据。


_PATTERN_SPECS = (
    (9, (1, 0, 1)),
    (11, (1, 0, 1, 0, 1)),
)
_ALLOWED_TOOTH_MASKS = {mask for _width, mask in _PATTERN_SPECS}
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _rotate_dir_ccw(dirv: tuple[int, int], k: int) -> tuple[int, int]:
    dr, dc = map(int, dirv)
    for _ in range(k % 4):
        dr, dc = -dc, dr
    return int(dr), int(dc)


def _red_bool_mask(tooth_mask: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros((2, len(tooth_mask)), dtype=bool)
    mask[0, :] = np.array(tooth_mask, dtype=bool)
    mask[1, :] = True
    return mask


def _mask_to_patch(mask: np.ndarray, fg: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = int(fg)
    return patch


def _structure_array(width: int, tooth_mask: tuple[int, ...], bg: int, sc: int) -> np.ndarray:
    """Build a 6-row nested comb structure with opening facing down."""
    span = len(tooth_mask)
    assert width == span + 6

    g = np.full((6, width), bg, dtype=int)
    tooth_arr = np.array(tooth_mask, dtype=bool)

    g[0, :] = sc
    g[1, 1:-1] = sc

    g[2, 1] = g[2, width - 2] = sc
    g[2, 2] = g[2, width - 3] = sc
    g[2, 3:3 + span][tooth_arr] = sc

    g[3:5, 1] = g[3:5, width - 2] = sc
    g[3:5, 2] = g[3:5, width - 3] = sc

    g[5, 1] = g[5, width - 2] = sc
    return g


def _component_infos(mask: np.ndarray) -> list[dict]:
    infos: list[dict] = []
    for coords in components_mask(mask, connectivity=4):
        pts = [(int(r), int(c)) for r, c in coords]
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            local[int(r - r0), int(c - c0)] = True
        infos.append(
            {
                "coords": pts,
                "bbox": (r0, c0, r1, c1),
                "mask": local,
            }
        )
    return infos


def _expected_row(width: int, row_idx: int, tooth_mask: tuple[int, ...]) -> np.ndarray:
    span = len(tooth_mask)
    row = np.zeros(width, dtype=bool)
    if row_idx == 0:
        row[:] = True
    elif row_idx == 1:
        row[1:-1] = True
    elif row_idx == 2:
        row[1] = row[-2] = True
        row[2] = row[-3] = True
        row[3:3 + span] = np.array(tooth_mask, dtype=bool)
    elif row_idx in (3, 4):
        row[1] = row[-2] = True
        row[2] = row[-3] = True
    elif row_idx == 5:
        row[1] = row[-2] = True
    else:
        raise ValueError("invalid row index")
    return row


def _normalize_structure_patch(local: np.ndarray, bg: int) -> tuple[np.ndarray, int, tuple[int, ...]]:
    """Rotate a structure bbox back to the canonical opening-down template."""
    for rot in range(4):
        cand = np.rot90(local, rot)
        occ = cand != int(bg)
        if cand.shape[0] != 6 or cand.shape[1] not in (9, 11):
            continue

        span = int(cand.shape[1] - 6)
        tooth_mask = tuple(int(v) for v in occ[2, 3:3 + span])
        if tooth_mask not in _ALLOWED_TOOTH_MASKS:
            continue

        ok = True
        for row_idx in range(6):
            if not np.array_equal(occ[row_idx], _expected_row(int(cand.shape[1]), row_idx, tooth_mask)):
                ok = False
                break
        if ok:
            return cand, int(rot), tooth_mask

    raise AssertionError("structure does not match the nested interval-tooth template")


def _structure_infos(grid: np.ndarray, bg: int, red: int) -> list[dict]:
    struct_mask = (grid != int(bg)) & (grid != int(red))
    comps = components_mask(struct_mask, connectivity=4)
    assert comps, "expected at least one non-bg non-red structure"

    infos: list[dict] = []
    for sid, coords in enumerate(comps):
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        local = np.array(grid[r0:r1 + 1, c0:c1 + 1], dtype=int)
        norm, rot_to_norm, tooth_mask = _normalize_structure_patch(local, bg)
        span = len(tooth_mask)

        red_norm_mask = _red_bool_mask(tooth_mask)

        target_norm = np.zeros_like(norm, dtype=bool)
        target_norm[3:5, 3:3 + span] = red_norm_mask

        interior_norm = np.zeros_like(norm, dtype=bool)
        interior_norm[1:, 1:-1] = True

        undo_rot = (4 - rot_to_norm) % 4
        target_local = np.rot90(target_norm, undo_rot)
        expected_mask = np.rot90(red_norm_mask, undo_rot)
        interior_local = np.rot90(interior_norm, undo_rot)

        rel_target = np.argwhere(target_local)
        tr0, tc0, tr1, tc1 = map(int, bbox_coords(rel_target))
        target_coords = [(r0 + int(rr), c0 + int(cc)) for rr, cc in rel_target]

        rel_crop = np.argwhere(interior_local)
        cr0, cc0, cr1, cc1 = map(int, bbox_coords(rel_crop))

        infos.append(
            {
                "sid": int(sid),
                "bbox": (r0, c0, r1, c1),
                "tooth_mask": tooth_mask,
                "shape_size": int(expected_mask.sum()),
                "opening_dir": _rotate_dir_ccw((1, 0), undo_rot),
                "expected_mask": expected_mask,
                "target_bbox": (r0 + tr0, c0 + tc0, r0 + tr1, c0 + tc1),
                "target_coords": target_coords,
                "crop_bbox": (r0 + cr0, c0 + cc0, r0 + cr1, c0 + cc1),
            }
        )
    return infos


def _slide_amount(
    red_bbox: tuple[int, int, int, int],
    target_bbox: tuple[int, int, int, int],
    opening_dir: tuple[int, int],
) -> int | None:
    rr0, rc0, rr1, rc1 = map(int, red_bbox)
    tr0, tc0, tr1, tc1 = map(int, target_bbox)
    dr, dc = map(int, opening_dir)

    if (rr1 - rr0) != (tr1 - tr0) or (rc1 - rc0) != (tc1 - tc0):
        return None

    drow = int(rr0 - tr0)
    dcol = int(rc0 - tc0)
    if dr != 0:
        if dcol != 0 or drow * dr <= 0:
            return None
        return abs(drow)
    if drow != 0 or dcol * dc <= 0:
        return None
    return abs(dcol)


def _path_is_clear(
    grid: np.ndarray,
    bg: int,
    target_coords: list[tuple[int, int]],
    opening_dir: tuple[int, int],
    slide: int,
    allowed_coords: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    dr, dc = map(int, opening_dir)
    for step in range(0, slide):
        for r, c in target_coords:
            rr = int(r + step * dr)
            cc = int(c + step * dc)
            if not (0 <= rr < h and 0 <= cc < w):
                return False
            if int(grid[rr, cc]) != int(bg) and (int(rr), int(cc)) not in allowed_coords:
                return False
    return True


def _find_assignments(grid: np.ndarray, bg: int, red: int) -> tuple[list[dict], list[dict], list[dict]]:
    red_infos = _component_infos(grid == int(red))
    assert red_infos, "expected at least one red component"
    struct_infos = _structure_infos(grid, bg, red)

    candidates: dict[int, list[tuple[int, int]]] = {info["sid"]: [] for info in struct_infos}
    for sinfo in struct_infos:
        expected_mask = sinfo["expected_mask"]
        for rid, rinfo in enumerate(red_infos):
            if rinfo["mask"].shape != expected_mask.shape:
                continue
            if not np.array_equal(rinfo["mask"], expected_mask):
                continue
            slide = _slide_amount(rinfo["bbox"], sinfo["target_bbox"], sinfo["opening_dir"])
            if slide is None or slide <= 0:
                continue
            if not _path_is_clear(grid, bg, sinfo["target_coords"], sinfo["opening_dir"], slide, set(rinfo["coords"])):
                continue
            candidates[sinfo["sid"]].append((int(slide), int(rid)))

    used_red: set[int] = set()
    assignments: list[dict] = []
    order = sorted(
        struct_infos,
        key=lambda info: (int(info["shape_size"]), len(info["tooth_mask"]), -int(info["bbox"][0]), -int(info["bbox"][1])),
        reverse=True,
    )
    for sinfo in order:
        matches = sorted(candidates[sinfo["sid"]], key=lambda item: item[0])
        for slide, rid in matches:
            if rid in used_red:
                continue
            used_red.add(int(rid))
            assignments.append(
                {
                    "sid": int(sinfo["sid"]),
                    "rid": int(rid),
                    "slide": int(slide),
                    "opening_dir": tuple(map(int, sinfo["opening_dir"])),
                    "target_coords": list(sinfo["target_coords"]),
                }
            )
            break
    return assignments, red_infos, struct_infos


def _solve_full_canvas(grid: np.ndarray, bg: int) -> tuple[np.ndarray, list[dict]]:
    red = int(Color.RED)
    black = int(Color.BLACK)

    assignments, _red_infos, struct_infos = _find_assignments(grid, bg, red)
    out = np.array(grid, dtype=int)
    out[out == int(red)] = int(bg)

    h, w = out.shape
    for item in assignments:
        dr, dc = map(int, item["opening_dir"])
        for step in range(1, int(item["slide"]) + 1):
            for r, c in item["target_coords"]:
                rr = int(r + step * dr)
                cc = int(c + step * dc)
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = int(black)
        for r, c in item["target_coords"]:
            out[int(r), int(c)] = int(red)

    return out, struct_infos


def _pack_internal_crops(solved_full: np.ndarray, struct_infos: list[dict], bg: int) -> np.ndarray:
    ordered = sorted(struct_infos, key=lambda info: (int(info["bbox"][0]), int(info["bbox"][1])))
    crops = [
        solved_full[int(info["crop_bbox"][0]) : int(info["crop_bbox"][2]) + 1,
                    int(info["crop_bbox"][1]) : int(info["crop_bbox"][3]) + 1].copy()
        for info in ordered
    ]
    out_h = max(int(crop.shape[0]) for crop in crops)
    out_w = sum(int(crop.shape[1]) for crop in crops)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    c = 0
    for crop in crops:
        h, w = crop.shape
        out[: int(h), int(c) : int(c + w)] = crop
        c += int(w)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock interval-shaped red plugs into the innermost cavity of nested comb structures, then emit only the packed interior windows.

    Input structure:
    - The board contains 2-3 large non-bg, non-red structures plus multiple red blobs outside them.
    - Each structure is a rotated nested comb: a tall outer cavity contains a narrower inner cavity, and the true teeth are spaced apart (`101` or `10101`).

    Rule:
    - Recover each structure's orientation and nested template, then keep only red components whose silhouette matches the interval-tooth pattern and whose position is a straight translation from the inner target window.
    - Slide each matching red plug into the innermost cavity, paint every traversed position black, erase all unmatched reds, and finally crop each solved structure to its internal window before packing those crops left-to-right in row-major structure order.

    Key insight:
    - The correct destination is no longer the first visible cavity cell: you must identify the deeper inner chamber and use the tooth-spacing pattern to select the one red component that can reach that specific nested slot.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    assert int(Color.RED) in set(map(int, np.unique(g)))

    solved_full, struct_infos = _solve_full_canvas(g, bg)
    return _pack_internal_crops(solved_full, struct_infos, bg)


# --------------- generator ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (24, 28)
GRID_W_RANGE = (30, 34)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

# ── CFG: structures ─────────────────────────────────────────────────
STRUCT_PATTERN_CHOICES = _PATTERN_SPECS
STRUCT_COUNT_RANGE = (2, 3)
UNMATCHED_PROB = 0.35
SLIDE_RANGE = (4, 8)
STRUCT_PADDING = 3

# ── CFG: distractors ────────────────────────────────────────────────
DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
TOTAL_RED_MIN = 5
TOTAL_RED_MAX = 11
DISTRACTOR_MIN = 2
DISTRACTOR_MAX = 8

# ── CFG: retry ──────────────────────────────────────────────────────
MAX_RETRIES = 500


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _matched_patches(
    width: int,
    tooth_mask: tuple[int, ...],
    slide: int,
    rot_k: int,
    bg: int,
    sc: int,
) -> dict:
    """Build input/output/structure patches for a matched structure+red pair."""
    red_val = int(Color.RED)
    black_val = int(Color.BLACK)

    base = _structure_array(width, tooth_mask, bg, sc)
    red_mask = _red_bool_mask(tooth_mask)
    red_patch = _mask_to_patch(red_mask, red_val, bg)
    black_patch = _mask_to_patch(red_mask, black_val, bg)

    patch_h = 5 + int(slide)
    structure = np.full((patch_h, width), bg, dtype=int)
    paste(structure, base, 0, 0, transparent=bg)
    input_patch = structure.copy()
    output_patch = structure.copy()

    start_col = 3
    target_row = 3
    for off in range(1, int(slide) + 1):
        paste(output_patch, black_patch, target_row + off, start_col, transparent=bg)
    paste(output_patch, red_patch, target_row, start_col, transparent=bg)
    paste(input_patch, red_patch, target_row + int(slide), start_col, transparent=bg)

    return {
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(input_patch, k=rot_k),
        "output": np.rot90(output_patch, k=rot_k),
    }


def _unmatched_patches(width: int, tooth_mask: tuple[int, ...], rot_k: int, bg: int, sc: int) -> dict:
    structure = np.rot90(_structure_array(width, tooth_mask, bg, sc), k=rot_k)
    return {"structure": structure, "input": structure.copy(), "output": structure.copy()}


def _place_patch(canvas: np.ndarray, patch: np.ndarray, bg: int, padding: int) -> tuple[int, int] | None:
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


def _count_assignments(grid: np.ndarray, bg: int) -> int:
    assignments, _red_infos, _struct_infos = _find_assignments(grid, bg, int(Color.RED))
    return len(assignments)


def _place_structures(
    specs: list[dict],
    h: int,
    w: int,
    bg: int,
    sc: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    struct_grid = np.full((h, w), bg, dtype=int)

    for spec in specs:
        loc = _place_patch(out, spec["output"], bg, padding=STRUCT_PADDING)
        if loc is None:
            return None
        r0, c0 = loc
        paste(out, spec["output"], r0, c0, transparent=bg)
        paste(inp, spec["input"], r0, c0, transparent=bg)
        paste(struct_grid, spec["structure"], r0, c0, transparent=bg)

    return inp, out, struct_grid


def _forbidden_rays(struct_grid: np.ndarray, bg: int) -> np.ndarray:
    infos = _structure_infos(struct_grid, bg, int(Color.RED))
    h, w = struct_grid.shape
    forbidden = np.zeros((h, w), dtype=bool)
    for info in infos:
        dr, dc = map(int, info["opening_dir"])
        for r, c in info["target_coords"]:
            rr = int(r + dr)
            cc = int(c + dc)
            while 0 <= rr < h and 0 <= cc < w and int(struct_grid[rr, cc]) == int(bg):
                forbidden[rr, cc] = True
                rr += dr
                cc += dc
    return forbidden


def _place_distractors(inp: np.ndarray, struct_grid: np.ndarray, bg: int, sc: int, count: int) -> bool:
    forbidden = _forbidden_rays(struct_grid, bg)
    bbox_blocked = np.zeros_like(forbidden)
    for info in _structure_infos(struct_grid, bg, int(Color.RED)):
        r0, c0, r1, c1 = map(int, info["bbox"])
        bbox_blocked[r0:r1 + 1, c0:c1 + 1] = True

    red_mask = inp == int(Color.RED)
    shapes = list(DISTRACTOR_SHAPES)
    for _ in range(count):
        blocked = forbidden | bbox_blocked | (inp != bg) | _dilate4(red_mask)
        canvas = np.full(inp.shape, bg, dtype=int)
        canvas[blocked] = sc
        random.shuffle(shapes)
        ok = False
        for rh, rw in shapes:
            patch = np.full((rh, rw), int(Color.RED), dtype=int)
            loc = _place_patch(canvas, patch, bg, padding=0)
            if loc is None:
                continue
            paste(inp, patch, loc[0], loc[1], transparent=bg)
            red_mask[loc[0]:loc[0] + rh, loc[1]:loc[1] + rw] = True
            ok = True
            break
        if not ok:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
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
        width, tooth_mask = random.choice(STRUCT_PATTERN_CHOICES)
        rot_k = random.randrange(4)
        if idx < matched:
            slide = random.randint(*SLIDE_RANGE)
            specs.append(_matched_patches(width, tooth_mask, slide, rot_k, bg, sc))
        else:
            specs.append(_unmatched_patches(width, tooth_mask, rot_k, bg, sc))

    random.shuffle(specs)
    specs.sort(key=lambda spec: np.count_nonzero(spec["output"] != bg), reverse=True)

    placed = _place_structures(specs, h, w, bg, sc)
    if placed is None:
        return None
    inp, out_full, struct_grid = placed

    if _count_assignments(inp, bg) != matched:
        return None

    if not _place_distractors(inp, struct_grid, bg, sc, distractor_count):
        return None

    struct_infos = _structure_infos(struct_grid, bg, int(Color.RED))
    expected = _pack_internal_crops(out_full, struct_infos, bg)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 combo variant")
