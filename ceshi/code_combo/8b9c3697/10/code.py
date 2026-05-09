# Combo variant of 8b9c3697 (index 10)
# combo_spec: D6/2+D8/0
# Name: 间隔齿内部提取

from __future__ import annotations

import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location

# Design analysis
# ===============
# a) 具体化与行为追踪
# - 保留原题的“大型单色梳齿结构 + 外部红块沿开口滑入 + 黑色轨迹 + 干扰红块消失”主线，
#   但把齿改成可见的间隔模式 `101` 与 `10101`，并把输出改成“只看结构内部”。
# - 规范朝向仍设为“背板在上，开口朝下”。此时 7 宽结构的内部跨度是 3，对应 2 齿模式 `101`；
#   9 宽结构的内部跨度是 5，对应 3 齿模式 `10101`。匹配红块在贴齿的一行复制同样的间隔，
#   另一行填满整个跨度，所以分别是 `101/111` 和 `10101/11111`。
# - 原题一个 9 宽、3 连续齿结构会接收 `2x3` 连续红矩形，输出仍是整张大画布；融合后同位置会接收
#   `10101/11111` 形红块，滑入后只保留该结构的内部 3x7（旋转后可能是 7x3）区域，再与别的结构内部区域横向拼接。
# - 差异因此落在两处：匹配不再只看“有没有一块红色在射线上”，而是要恢复齿间隔并检查红块形状；
#   输出也不再展示结构外部与远处轨迹，只展示每个结构内部的填充结果。
#
# b) 复杂度校准
# - 间隔齿只保留两档：`101` 与 `10101`。这样样例中会重复出现相同类别，解题者能从少量样例稳定归纳。
# - 不生成 1 齿情形，因为它和原题视觉上几乎无差别，会削弱 D8 的可观察性。
# - 仍保留 2-3 个结构、0-1 个未匹配结构、2-8 个红色干扰块、四向旋转和 4-8 格滑行距离，
#   避免融合后退化成“单对象模板题”。
#
# c) 隐式假设审查
# - 原 solver 默认：只要射线看到同一红色连通块，就能把它当合法插块。融合后这个前提失效，
#   因为矩形红块不应插入间隔齿，所以必须先标准化结构，再显式比较红块局部布尔形状。
# - 原 solver 默认：最终停靠点可由“最近 socket + bbox 对齐”恢复。对间隔齿红块这不够，
#   因为同一 bbox 内可能有必须保留为空洞的格子。新版本直接由结构模板构造唯一的目标坐标集合。
# - D6 在这里不能简单裁整块 structure bbox，否则会把背板外框也保留下来，弱化“内部提取”。
#   融合版改为提取结构内部矩形：规范模板中的 rows 1..3 与 cols 1..width-2，旋回原朝向后再打包。
# - 新 solver 假设所有结构都满足同一 4 层深度模板，generator 用固定模板构造并通过 oracle 校验保证这一点。
#
# d) 放置合法性
# - structure patch 继续按 output-first 生成，并用 `random_free_location(..., padding=4)` 保证不同结构彼此分离。
# - 匹配红块始终沿开口方向平移 4-8 格，patch 高度/宽度预留足够空间；旋转后整体再随机落到大画布。
# - 干扰红块仅使用 1x1 到 3x2 的实心矩形，因此不会误触发间隔齿匹配；同时它们避开结构 bbox 与射线区域，
#   保持视觉整洁并减少偶然遮挡。
#
# e) 视觉可观察性
# - 解题者能直接看到输入中的齿是“齿-空-齿”或“齿-空-齿-空-齿”，并且合法红块靠近结构的一边复制同样节奏。
# - 输出高度显著降低，只剩每个结构的内部矩形，并且其中能看到红块进入后留下的黑轨迹或最终填充。
# - packed 顺序使用结构在输入中的行优先位置，而不是颜色值、组件编号等不可视信息。
#
# f) 可视化整洁度
# - 输出去除了结构外部大面积背景、远处干扰红块和外侧轨迹，只留下紧凑的内部区域，视觉上比整板更干净。
# - 内部裁剪矩形保留侧壁、齿、空腔、黑轨迹和最终红块，足以表达规则，又不会出现多层对象在同一区域混杂。
#
# g) 变化的可推断性
# - 颜色逻辑仍是原题的三类：结构色不变，匹配红块变为内部红色，轨迹变黑，其余红块消失。
# - 真正新增的信息全部由形状和裁剪区域提供，不依赖颜色 id、隐藏排序或随机分支，因此可由 2-4 个样例观察归纳。


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_PATTERN_SPECS = (
    (7, (1, 0, 1)),
    (9, (1, 0, 1, 0, 1)),
)
_ALLOWED_TOOTH_MASKS = {mask for _width, mask in _PATTERN_SPECS}
_PATTERN_BY_SPAN = {len(mask): (width, mask) for width, mask in _PATTERN_SPECS}


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
    """Build a normalized 4-row comb structure with the opening facing down."""
    g = np.full((4, width), bg, dtype=int)
    span = len(tooth_mask)
    assert width == span + 4

    g[0, :] = sc
    g[1, 1] = g[1, width - 2] = sc
    g[2:4, 1] = g[2:4, width - 2] = sc
    tooth_arr = np.array(tooth_mask, dtype=bool)
    g[1, 2 : 2 + span][tooth_arr] = sc
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


def _normalize_structure_patch(local: np.ndarray, bg: int) -> tuple[np.ndarray, int, tuple[int, ...]]:
    """Rotate a structure bbox back to the canonical opening-down template."""
    for rot in range(4):
        cand = np.rot90(local, rot)
        occ = cand != int(bg)
        if cand.shape[0] != 4 or cand.shape[1] not in (7, 9):
            continue
        if not np.all(occ[0, :]):
            continue
        if occ[1, 0] or occ[1, -1] or occ[2:, 0].any() or occ[2:, -1].any():
            continue
        if not (occ[1, 1] and occ[1, -2]):
            continue
        if not np.all(occ[2:, 1]) or not np.all(occ[2:, -2]):
            continue
        if occ[2:, 2:-2].any():
            continue
        tooth_mask = tuple(int(v) for v in occ[1, 2:-2])
        if tooth_mask not in _ALLOWED_TOOTH_MASKS:
            continue
        return cand, int(rot), tooth_mask
    raise AssertionError("structure does not match the interval-tooth template")


def _structure_infos(grid: np.ndarray, bg: int, red: int) -> list[dict]:
    struct_mask = (grid != int(bg)) & (grid != int(red))
    comps = components_mask(struct_mask, connectivity=4)
    assert comps, "expected at least one non-bg non-red structure"

    infos: list[dict] = []
    for sid, coords in enumerate(comps):
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        local = np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
        norm, rot_to_norm, tooth_mask = _normalize_structure_patch(local, bg)

        red_norm_mask = _red_bool_mask(tooth_mask)
        dock_norm = np.zeros_like(norm, dtype=bool)
        dock_norm[2:4, 2 : 2 + len(tooth_mask)] = red_norm_mask

        interior_norm = np.zeros_like(norm, dtype=bool)
        interior_norm[1:4, 1:-1] = True

        undo_rot = (4 - rot_to_norm) % 4
        target_local = np.rot90(dock_norm, undo_rot)
        expected_mask = np.rot90(red_norm_mask, undo_rot)
        interior_local = np.rot90(interior_norm, undo_rot)

        target_rel = np.argwhere(target_local)
        tr0, tc0, tr1, tc1 = map(int, bbox_coords(target_rel))
        target_coords = [(r0 + int(rr), c0 + int(cc)) for rr, cc in target_rel]

        interior_rel = np.argwhere(interior_local)
        ir0, ic0, ir1, ic1 = map(int, bbox_coords(interior_rel))
        opening_dir = _rotate_dir_ccw((1, 0), undo_rot)

        infos.append(
            {
                "sid": int(sid),
                "bbox": (r0, c0, r1, c1),
                "tooth_mask": tooth_mask,
                "opening_dir": opening_dir,
                "expected_mask": expected_mask,
                "target_bbox": (r0 + tr0, c0 + tc0, r0 + tr1, c0 + tc1),
                "target_coords": target_coords,
                "shape_size": int(expected_mask.sum()),
                "interior_bbox": (r0 + ir0, c0 + ic0, r0 + ir1, c0 + ic1),
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
        target_bbox = sinfo["target_bbox"]
        opening_dir = sinfo["opening_dir"]
        target_coords = sinfo["target_coords"]
        for rid, rinfo in enumerate(red_infos):
            if rinfo["mask"].shape != expected_mask.shape:
                continue
            if not np.array_equal(rinfo["mask"], expected_mask):
                continue
            slide = _slide_amount(rinfo["bbox"], target_bbox, opening_dir)
            if slide is None or slide <= 0:
                continue
            if not _path_is_clear(grid, bg, target_coords, opening_dir, slide, set(rinfo["coords"])):
                continue
            candidates[sinfo["sid"]].append((int(slide), int(rid)))

    used_red: set[int] = set()
    assignments: list[dict] = []
    order = sorted(
        struct_infos,
        key=lambda info: (int(info["shape_size"]), len(info["tooth_mask"])),
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


def _solve_full_canvas(grid: np.ndarray, bg: int, red: int, black: int) -> tuple[np.ndarray, list[dict]]:
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
    crops = []
    for info in ordered:
        r0, c0, r1, c1 = map(int, info["interior_bbox"])
        crops.append(np.array(solved_full[r0 : r1 + 1, c0 : c1 + 1], dtype=int))

    out_h = max(int(crop.shape[0]) for crop in crops)
    out_w = sum(int(crop.shape[1]) for crop in crops)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    cc = 0
    for crop in crops:
        h, w = crop.shape
        out[: int(h), int(cc) : int(cc + w)] = crop
        cc += int(w)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interval-tooth red plugs into the comb cavities, then emit only each structure's internal view.

    Input structure:
    - The board contains 2-3 large single-color comb/bracket structures and several red components on the background.
    - Each structure uses a spaced tooth pattern (`101` or `10101` in canonical orientation), while distractor reds are ordinary rectangles elsewhere.

    Rule:
    - Recover every structure's orientation from its solid back wall, read its tooth-spacing mask, and match only the red component whose docking face has the same gapped silhouette and lies on a straight opening-direction translation.
    - Slide each matched red inward, paint all traversed cells black, erase unmatched reds, then crop the solved board to each structure's internal rectangle and pack those crops left-to-right in row-major structure order.

    Key insight:
    - The answer is determined jointly by shape and view: interval teeth uniquely specify which red plug can dock, and the final output discards the outer scene so only the internal filled cavities remain.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    red = int(Color.RED)
    black = int(Color.BLACK)

    assert int(red) in set(map(int, np.unique(g)))

    solved_full, struct_infos = _solve_full_canvas(g, bg, red, black)
    return _pack_internal_crops(solved_full, struct_infos, bg)


# --------------- generator ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (19, 23)
GRID_W_RANGE = (22, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

# ── CFG: structures ─────────────────────────────────────────────────
STRUCT_PATTERN_CHOICES = _PATTERN_SPECS
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


def _matched_patches(width: int, tooth_mask: tuple[int, ...], slide: int, rot_k: int, bg: int, sc: int) -> dict:
    """Build input/output/structure patches for a matched structure+red pair."""
    red_val = int(Color.RED)
    black_val = int(Color.BLACK)

    base = _structure_array(width, tooth_mask, bg, sc)
    red_mask = _red_bool_mask(tooth_mask)
    red_patch = _mask_to_patch(red_mask, red_val, bg)
    black_patch = _mask_to_patch(red_mask, black_val, bg)

    patch_h = 2 + slide + red_patch.shape[0]
    structure = np.full((patch_h, width), bg, dtype=int)
    paste(structure, base, 0, 0, transparent=bg)
    input_patch = structure.copy()
    output_patch = structure.copy()

    start = 2
    for off in range(1, slide + 1):
        paste(output_patch, black_patch, 2 + off, start, transparent=bg)
    paste(output_patch, red_patch, 2, start, transparent=bg)
    paste(input_patch, red_patch, 2 + slide, start, transparent=bg)

    return {
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(input_patch, k=rot_k),
        "output": np.rot90(output_patch, k=rot_k),
    }


def _unmatched_patches(width: int, tooth_mask: tuple[int, ...], rot_k: int, bg: int, sc: int) -> dict:
    structure = np.rot90(_structure_array(width, tooth_mask, bg, sc), k=rot_k)
    return {"structure": structure, "input": structure.copy(), "output": structure.copy()}


def _structure_rays(struct_grid: np.ndarray, placed: list[dict], bg: int) -> np.ndarray:
    """Return a conservative mask of cells lying on potential sight-lines from structure sockets."""
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


def _count_assignments(grid: np.ndarray, bg: int) -> int:
    assignments, _red_infos, _struct_infos = _find_assignments(grid, bg, int(Color.RED))
    return len(assignments)


def _place_structures(
    specs: list[dict], h: int, w: int, bg: int, sc: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]] | None:
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

        coords_arr = np.argwhere(spec["structure"] != bg)
        coords = [(r0 + int(rr), c0 + int(cc)) for rr, cc in coords_arr]
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        placed.append({"coords": coords, "bbox": (min(rows), min(cols), max(rows), max(cols))})

    return inp, out, struct_grid, placed


def _place_distractors(
    inp: np.ndarray, struct_grid: np.ndarray, placed: list[dict], bg: int, sc: int, count: int
) -> bool:
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


def _pattern_schedule(
    n_structs: int, matched: int
) -> tuple[list[tuple[int, tuple[int, ...]]], list[tuple[int, tuple[int, ...]]]]:
    """
    Choose structure tooth patterns with balanced coverage.

    Every generated puzzle shows both interval-tooth spans among the matched structures,
    so the fused D8 behavior is visible without relying on cross-example statistics.
    """
    assert 2 <= matched <= n_structs

    matched_specs = [spec for spec in _PATTERN_SPECS]
    while len(matched_specs) < matched:
        matched_specs.append(random.choice(_PATTERN_SPECS))

    counts = {len(mask): 0 for _, mask in _PATTERN_SPECS}
    for _, mask in matched_specs:
        counts[len(mask)] += 1

    unmatched_specs: list[tuple[int, tuple[int, ...]]] = []
    while len(matched_specs) + len(unmatched_specs) < n_structs:
        span = min(counts, key=lambda key: (counts[key], random.random()))
        spec = _PATTERN_BY_SPAN[span]
        unmatched_specs.append(spec)
        counts[span] += 1

    random.shuffle(matched_specs)
    random.shuffle(unmatched_specs)
    return matched_specs, unmatched_specs


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_COLORS)
    sc = random.choice([c for c in STRUCT_COLORS if c not in (bg, int(Color.RED), int(Color.BLACK))])
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_structs = random.randint(*STRUCT_COUNT_RANGE)
    unmatched = 1 if n_structs >= 3 and random.random() < UNMATCHED_PROB else 0
    matched = n_structs - unmatched
    if matched < 2:
        return None

    min_dist = max(DISTRACTOR_MIN, TOTAL_RED_MIN - matched)
    max_dist = min(DISTRACTOR_MAX, TOTAL_RED_MAX - matched)
    if min_dist > max_dist:
        return None
    distractor_count = random.randint(min_dist, max_dist)

    matched_patterns, unmatched_patterns = _pattern_schedule(n_structs, matched)
    specs = []
    for width, tooth_mask in matched_patterns:
        rot_k = random.randrange(4)
        slide = random.randint(*SLIDE_RANGE)
        specs.append(_matched_patches(width, tooth_mask, slide, rot_k, bg, sc))
    for width, tooth_mask in unmatched_patterns:
        rot_k = random.randrange(4)
        specs.append(_unmatched_patches(width, tooth_mask, rot_k, bg, sc))
    random.shuffle(specs)
    specs.sort(key=lambda s: np.count_nonzero(s["output"] != bg), reverse=True)

    result = _place_structures(specs, h, w, bg, sc)
    if result is None:
        return None
    inp, out_full, struct_grid, placed = result

    if _count_assignments(inp, bg) != matched:
        return None

    if not _place_distractors(inp, struct_grid, placed, bg, sc, distractor_count):
        return None

    struct_infos = _structure_infos(inp, bg, int(Color.RED))
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
