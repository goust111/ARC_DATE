# Variant of 8b9c3697
# Dimension: D8 — 齿的数量与形状 (meta)
# Change: 每个结构有1-3个连续的齿（teeth），齿决定了插槽宽度和匹配红色的形状（min(teeth,2) x teeth）... → 非连续齿（间隔齿）

from __future__ import annotations

import random
import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location

# Design analysis
# ===============
# a) 具体化与行为追踪
# - 结构仍是单色梳齿/支架，仍然只有一个完整背板、一对侧壁和一个向外开口的腔体。
# - 变体只改齿的排列：不再出现 `111` 这类连续齿，而是使用可见的间隔模式
#   `101`（2 齿，跨度 3）和 `10101`（3 齿，跨度 5）。
# - 与之配对的红块必须在“靠近结构的一侧”复制该间隔模式；为了保持这是一个单一 4 连通组件，
#   红块的另一行/列填满整个跨度，所以规范朝向下的红块形状分别是：
#   `101/111` 和 `10101/11111`。
# - 小例子：原题里 3 齿结构接收一个 `2x3` 的连续红矩形；现在 3 齿结构接收一个 `2x5`
#   的非矩形红块，贴近齿的一行只落在 3 个齿正下方，空位保持背景。
#
# b) 复杂度校准
# - 只保留两档间隔模式，避免自由度过大，同时两档都足够醒目，不会和原题的连续矩形混淆。
# - 1 齿情形与原题视觉上等价，所以不生成；否则 D8 的变化会在部分样例中不可观察。
#
# c) 原 main() 的隐式假设审查
# - 原代码默认“看见同一块红色即可匹配”，没有检查形状；这在连续矩形时代由 generator 保证无害，
#   但现在会把错误的红矩形也当成合法目标，所以必须显式比较红块形状。
# - 原代码用 bbox 边界内的射线命中数挑选最佳方向；这仍可用于“可见性”直觉，但不足以恢复间隔模式。
#   新版本先根据完整背板恢复朝向，再读取齿模式。
# - 原代码把最终停靠位置看成“红块贴住最近 socket 的某一边”；对非矩形红块而言，这样会丢失模式对齐，
#   所以新版本直接从结构模板构造唯一合法的最终停靠形状。
#
# d) 放置合法性
# - 结构仍保持 4 层深度的规范模板，宽度固定为 7 或 9，对应 3 或 5 的内部跨度。
# - 匹配红块沿开口方向平移 4-8 格，路径必须全是背景；generator 仍采用 output-first，
#   放置后用 `main(input) == output` 做 oracle。
#
# e) 视觉可观察性
# - 解题者能直接看到：结构上的齿是“齿-空-齿”或“齿-空-齿-空-齿”，外部红块靠近结构的一边也有同样节奏。
# - 输出中红块进入腔体后，贴齿的一边保留这些空位，因此变体不会退化成“任何同向矩形都能滑入”。
#
# f) 可视化整洁度
# - 红块仍只有两层厚，轨迹仍只涂黑，不引入额外标记色。
# - 结构间距和干扰红块的避让规则沿用原题，避免多条轨迹和多个结构在同一区域堆叠。
#
# g) 颜色变化可推断性
# - 颜色逻辑完全不变：红块移动后仍是红，经过路径仍是黑，其他红色消失。
# - 变体的可推断性完全来自形状和空间关系，不依赖颜色 id、排序或隐藏元数据。


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
        norm, rot_to_norm, tooth_mask = _normalize_structure_patch(local, bg)

        red_norm_mask = _red_bool_mask(tooth_mask)
        dock_norm = np.zeros_like(norm, dtype=bool)
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
                "opening_dir": opening_dir,
                "expected_mask": expected_mask,
                "target_bbox": (r0 + tr0, c0 + tc0, r0 + tr1, c0 + tc1),
                "target_coords": target_coords,
                "shape_size": int(expected_mask.sum()),
            }
        )
    return infos


def _slide_amount(red_bbox: tuple[int, int, int, int],
                  target_bbox: tuple[int, int, int, int],
                  opening_dir: tuple[int, int]) -> int | None:
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


def _path_is_clear(grid: np.ndarray, bg: int,
                   target_coords: list[tuple[int, int]],
                   opening_dir: tuple[int, int],
                   slide: int,
                   allowed_coords: set[tuple[int, int]]) -> bool:
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide only the correctly shaped red plugs into interval-tooth comb sockets, leaving black motion trails.

    Input structure:
    - The background is the most common color. Large single-color comb/bracket structures face one of four directions.
    - Their teeth are spaced apart (`101` or `10101` in normalized form), and red blobs sit outside the openings.

    Rule:
    - Recover each structure's orientation from its solid back wall, then read the tooth-spacing pattern.
    - A red component matches only if its leading edge repeats that spacing pattern and it is a straight translation away from the opening.
    - Matching reds slide into the cavity, all traversed cells turn black, and every unmatched red disappears.

    Key insight:
    - The tooth mask, not just the ray direction, determines the valid plug: the structure and the red piece share the same gapped silhouette on the docking face.
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


def _matched_patches(width: int, tooth_mask: tuple[int, ...], slide: int, rot_k: int,
                     bg: int, sc: int) -> dict:
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


def _unmatched_patches(width: int, tooth_mask: tuple[int, ...], rot_k: int,
                       bg: int, sc: int) -> dict:
    structure = np.rot90(_structure_array(width, tooth_mask, bg, sc), k=rot_k)
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
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


def _count_assignments(grid: np.ndarray, bg: int) -> int:
    assignments, _red_infos, _struct_infos = _find_assignments(grid, bg, int(Color.RED))
    return len(assignments)


def _place_structures(specs: list[dict], h: int, w: int, bg: int,
                      sc: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]] | None:
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
    specs.sort(key=lambda s: np.count_nonzero(s["output"] != bg), reverse=True)

    result = _place_structures(specs, h, w, bg, sc)
    if result is None:
        return None
    inp, out, struct_grid, placed = result

    if _count_assignments(inp, bg) != matched:
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
    raise RuntimeError("failed to generate a valid input for 8b9c3697 interval-tooth variant")
