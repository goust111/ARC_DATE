# Combo variant of 8b9c3697 (index 9)
# combo_spec: D1/4+D8/0
# Name: 尺寸排序间隔齿匹配

from __future__ import annotations

import random
import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location

# Design analysis
# ===============
# a) 具体化与行为追踪
# - 原题里，结构内部的 socket 朝某个方向发射射线，谁在这条线上就和谁配对；连续齿决定匹配红块是连续矩形。
# - 现在保留“梳齿结构 + 红块直线滑入 + 轨迹涂黑”的外壳，但把两件事同时改掉：
#   1) 齿改为间隔齿，只生成 `101` 和 `10101` 两档，配套红块分别是 `101/111` 与 `10101/11111`。
#   2) 配对不再由单个结构的局部视线决定，而是先恢复所有结构的齿间跨度，再把所有可直线滑入的间隔红块按面积排序，
#      让最小红块进最窄插槽、较大红块进较宽插槽。
# - 具体小例子：若图里有一个 3 跨度齿隙和一个 5 跨度齿隙，则对应可匹配红块面积固定为 5 和 8。
#   输出时面积 5 的红块进入窄 socket，面积 8 的红块进入宽 socket，沿滑行方向经过的格子变黑，其余红色消失。
#
# b) 复杂度校准
# - 只保留两档间隔齿，避免“自由度太多导致样例过乱”；同时又强制每个可解样例同时出现两档尺寸，
#   保证“尺寸排序”与“间隔齿形状”都能被肉眼看到。
# - matched 结构固定为两座：一窄一宽。可额外加入一座 unmatched 结构，但它不参与排序链，避免三方 tie。
#
# c) 原 main() 的隐式假设审查
# - 原代码默认“看见同一块红色即可”，没有恢复结构模板，也不检查红块是否复现齿间隔；融合后这两个前提都失效。
# - 原代码把 assignment 建立在 structure->ray hit 的局部贪心上；D1/4 要求改成全局排序，所以必须先收集全部结构和全部候选红块。
# - 原代码把最终停靠位置靠 bbox 一侧对齐；对间隔红块会丢失空齿位，因此这里必须直接从结构模板构造 target_coords。
#
# d) 放置合法性
# - generator 延续 output-first：先构造“已停靠”的输出 patch，再把红块沿开口方向平移 4-8 格得到输入。
# - 结构之间保留 padding，matched 红块与干扰红块都不得重叠；用 `main(input) == output` 做 oracle。
#
# e) 视觉可观察性
# - 人类能直接看见两条独立线索：齿是隔一格一根；红块贴近结构的一侧也带相同节奏。
# - 同时每个样例都至少有一窄一宽两个可解结构，因此可以从 2-3 组样例归纳出“小块进窄缝，大块进宽缝”。
#
# f) 可视化整洁度
# - 仍然只用原题的三类颜色角色：结构色、红色、黑色轨迹；不引入额外标记色。
# - 干扰红块仍是普通小矩形，并避开结构 bbox 与滑入走廊，避免把结构模板淹没。
#
# g) 变化的可推断性
# - 这里没有颜色映射变化，所有新增推理都来自可见几何：齿间隔模式、红块面积大小、以及滑入后的黑色轨迹。
# - 排序使用的是人类可见的绝对面积与齿隙宽度，不依赖颜色 id、生成顺序或隐藏索引。
#
# h) 新 solver 的额外假设
# - solver 现在假设可匹配红块本身就是完整的间隔插头形状，并且与目标结构之间存在一条纯背景直线平移路径。
# - generator 通过模板生成和 distractor 避让显式保证这些条件成立。


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_PATTERN_SPECS = (
    (7, (1, 0, 1)),
    (9, (1, 0, 1, 0, 1)),
)
_ALLOWED_TOOTH_MASKS = {mask for _width, mask in _PATTERN_SPECS}


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
    """Build a 4-row normalized comb structure (opening facing down)."""
    g = np.full((4, width), bg, dtype=int)
    span = len(tooth_mask)
    assert width == span + 4
    g[0, :] = sc
    g[1, 1] = g[1, width - 2] = sc
    g[2:4, 1] = g[2:4, width - 2] = sc
    tooth_arr = np.array(tooth_mask, dtype=bool)
    g[1, 2:2 + span][tooth_arr] = sc
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
                "area": int(local.sum()),
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
        local = np.array(grid[r0:r1 + 1, c0:c1 + 1], dtype=int)
        _norm, rot_to_norm, tooth_mask = _normalize_structure_patch(local, bg)

        red_norm_mask = _red_bool_mask(tooth_mask)
        dock_norm = np.zeros((4, len(tooth_mask) + 4), dtype=bool)
        dock_norm[2:4, 2:2 + len(tooth_mask)] = red_norm_mask

        undo_rot = (4 - rot_to_norm) % 4
        target_local = np.rot90(dock_norm, undo_rot)
        expected_mask = np.rot90(red_norm_mask, undo_rot)

        rel_coords = np.argwhere(target_local)
        tr0, tc0, tr1, tc1 = map(int, bbox_coords(rel_coords))
        target_coords = [(r0 + int(rr), c0 + int(cc)) for rr, cc in rel_coords]
        opening_dir = _rotate_dir_ccw((1, 0), undo_rot)

        infos.append(
            {
                "sid": int(sid),
                "bbox": (r0, c0, r1, c1),
                "tooth_mask": tooth_mask,
                "span": int(len(tooth_mask)),
                "opening_dir": opening_dir,
                "expected_mask": expected_mask,
                "target_bbox": (r0 + tr0, c0 + tc0, r0 + tr1, c0 + tc1),
                "target_coords": target_coords,
                "shape_size": int(expected_mask.sum()),
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


def _candidate_map(grid: np.ndarray, bg: int, red: int) -> tuple[list[dict], list[dict], dict[tuple[int, int], dict]]:
    red_infos = _component_infos(grid == int(red))
    assert red_infos, "expected at least one red component"
    struct_infos = _structure_infos(grid, bg, red)

    candidates: dict[tuple[int, int], dict] = {}
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
            candidates[(int(sinfo["sid"]), int(rid))] = {
                "sid": int(sinfo["sid"]),
                "rid": int(rid),
                "slide": int(slide),
                "shape_size": int(rinfo["area"]),
                "opening_dir": tuple(map(int, sinfo["opening_dir"])),
                "target_coords": list(sinfo["target_coords"]),
            }
    return red_infos, struct_infos, candidates


def _find_assignments(grid: np.ndarray, bg: int, red: int) -> tuple[list[dict], list[dict], list[dict]]:
    red_infos, struct_infos, candidates = _candidate_map(grid, bg, red)

    candidate_rids = sorted(
        {rid for _sid, rid in candidates},
        key=lambda rid: (
            int(red_infos[rid]["area"]),
            int(red_infos[rid]["mask"].shape[0]),
            int(red_infos[rid]["mask"].shape[1]),
            int(rid),
        ),
    )
    struct_order = sorted(
        struct_infos,
        key=lambda info: (
            int(info["shape_size"]),
            int(info["span"]),
            int(info["sid"]),
        ),
    )

    used_red: set[int] = set()
    assignments: list[dict] = []
    for sinfo in struct_order:
        for rid in candidate_rids:
            if rid in used_red:
                continue
            key = (int(sinfo["sid"]), int(rid))
            if key not in candidates:
                continue
            used_red.add(int(rid))
            assignments.append(dict(candidates[key]))
            break

    return assignments, red_infos, struct_infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort interval-shaped red plugs by size and slide them into interval-tooth sockets, painting black trails behind them.

    Input structure:
    - The background is the most common color. Large single-color comb/bracket structures face one of four directions.
    - Each structure uses spaced teeth (`101` or `10101` in normalized form), and red components sit outside the openings.

    Rule:
    - Recover every structure's opening direction and tooth-gap span from its solid back wall and spaced teeth.
    - Keep only red components that already have a matching interval-plug silhouette and can translate straight into some socket.
    - Rank those candidate red plugs from smaller area to larger area, rank sockets from narrower span to wider span, then dock them in that order; all traversed cells become black and every other red vanishes.

    Key insight:
    - The correct assignment is global rather than local: after recognizing the gapped tooth pattern, solve by comparing plug size against socket width order, not by asking which red blob a single structure sees first.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    red = int(Color.RED)
    black = int(Color.BLACK)

    assert int(red) in set(map(int, np.unique(g)))

    assignments, _red_infos, _struct_infos = _find_assignments(g, bg, red)

    out = g.copy()
    out[out == int(red)] = int(bg)

    h, w = g.shape
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

    return out


# --------------- generator ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (19, 23)
GRID_W_RANGE = (22, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

# ── CFG: structures ─────────────────────────────────────────────────
MATCHED_PATTERN_CHOICES = _PATTERN_SPECS
STRUCT_COUNT_CHOICES = (2, 3)
SLIDE_RANGE = (4, 8)
STRUCT_PADDING = 4

# ── CFG: distractors ────────────────────────────────────────────────
DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
TOTAL_RED_MIN = 5
TOTAL_RED_MAX = 10
DISTRACTOR_MIN = 2
DISTRACTOR_MAX = 7

# ── CFG: retry ──────────────────────────────────────────────────────
MAX_RETRIES = 500


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
        "matched": True,
        "width": int(width),
        "area": int(red_mask.sum()),
    }


def _unmatched_patches(width: int, tooth_mask: tuple[int, ...], rot_k: int, bg: int, sc: int) -> dict:
    structure = np.rot90(_structure_array(width, tooth_mask, bg, sc), k=rot_k)
    return {
        "structure": structure,
        "input": structure.copy(),
        "output": structure.copy(),
        "matched": False,
        "width": int(width),
        "area": int(_red_bool_mask(tooth_mask).sum()),
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
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


def _place_structures(
    specs: list[dict],
    h: int,
    w: int,
    bg: int,
    sc: int,
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

        coords_arr = np.argwhere(spec["structure"] == sc)
        coords = [(r0 + int(rr), c0 + int(cc)) for rr, cc in coords_arr]
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        placed.append(
            {
                "coords": coords,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "matched": bool(spec["matched"]),
                "width": int(spec["width"]),
                "area": int(spec["area"]),
            }
        )

    return inp, out, struct_grid, placed


def _place_distractors(inp: np.ndarray, struct_grid: np.ndarray, placed: list[dict], bg: int, sc: int, count: int) -> bool:
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


def _count_assignments(grid: np.ndarray, bg: int) -> int:
    assignments, _red_infos, _struct_infos = _find_assignments(grid, bg, int(Color.RED))
    return len(assignments)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_COLORS)
    sc = random.choice([c for c in STRUCT_COLORS if c not in (bg, int(Color.RED), int(Color.BLACK))])
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    n_structs = random.choice(STRUCT_COUNT_CHOICES)
    matched = 2
    unmatched = int(n_structs - matched)

    min_dist = max(DISTRACTOR_MIN, TOTAL_RED_MIN - matched)
    max_dist = min(DISTRACTOR_MAX, TOTAL_RED_MAX - matched)
    if min_dist > max_dist:
        return None
    distractor_count = random.randint(min_dist, max_dist)

    specs = []
    matched_specs = list(MATCHED_PATTERN_CHOICES)
    random.shuffle(matched_specs)
    for width, tooth_mask in matched_specs:
        slide = random.randint(*SLIDE_RANGE)
        rot_k = random.randrange(4)
        specs.append(_matched_patches(width, tooth_mask, slide, rot_k, bg, sc))

    for _ in range(unmatched):
        width, tooth_mask = random.choice(MATCHED_PATTERN_CHOICES)
        rot_k = random.randrange(4)
        specs.append(_unmatched_patches(width, tooth_mask, rot_k, bg, sc))

    random.shuffle(specs)
    specs.sort(key=lambda s: np.count_nonzero(s["output"] != bg), reverse=True)

    result = _place_structures(specs, h, w, bg, sc)
    if result is None:
        return None
    inp, out, struct_grid, placed = result

    if _count_assignments(inp, bg) != matched:
        return None

    if not _place_distractors(inp, struct_grid, placed, bg, sc, distractor_count):
        return None

    if _count_assignments(inp, bg) != matched:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 combo variant")
