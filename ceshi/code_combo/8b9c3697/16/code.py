# Combo variant of 8b9c3697 (index 16)
# combo_spec: D1/1+D2/0+D8/0
# Name: 非矩形形状嵌合间隔齿

from __future__ import annotations

import random
import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location

# design_analysis:
#   variant_goal:
#     - 保留“红块沿单一方向滑入结构并留下黑色轨迹”的原始主行为。
#     - 同时叠加三项变化：匹配改成精确二维形状嵌合、红块改成非矩形 polyomino、结构腔体顶部改成间隔齿。
#   concrete_visual:
#     - 每个结构都是单色实心块，某一侧切出一个触边开口腔体；腔体不是矩形，而是带有 `101` 或 `10101`
#       顶缘节奏的异形空腔，因此肉眼能看到“齿-空-齿”或“齿-空-齿-空-齿”的不连续齿。
#     - 真匹配红块与腔体完全同形同向，只在开口法线方向做平移；更靠近结构的位置故意放一个同宽但错形的红色诱饵。
#     - 输出时，真块嵌入腔体，整条平移轨迹涂黑；诱饵和其他红块全部消失。
#   behavior_trace:
#     - 原题：从结构内部 socket 沿行/列打射线，先撞到的红块就是匹配对象，且红块通常是矩形。
#     - 本融合版：先从结构 bbox 内提取唯一触边腔体的完整 2D 形状，再在所有红连通块中寻找“形状完全一致且位于同一平移轴上”的候选。
#     - 差异 1：最近红块不再可靠，因为近端诱饵虽然也在开口轴上，但深度或下缘轮廓不同，不能嵌入。
#     - 差异 2：结构的可停靠空间本身已是不规则形状；只看宽度、只看齿节奏、或只看射线方向都会误配。
#   complexity_calibration:
#     - 形状库只保留少量可区分模板，按 3 列/5 列两种间隔齿族分组；同一网格内结构形状互异，避免多解。
#     - 每个模板都满足“沿开口方向逐行/逐列单调扩张”，保证整块平移时不会穿过结构实体，轨迹可视上始终在腔体延长线上。
#     - 每个已匹配结构都带一个更近的错形 decoy，使 D1 的变化在每个样例里都稳定可观察，而不会退化成“看最近红块”。
#   implicit_assumption_audit:
#     - 原 main 依赖“射线命中的第一个红块就是目标”；这里被近端错形诱饵显式打破，因此必须删掉 ray-hit 评分逻辑。
#     - 原 main 假设红块是矩形，只需用 bbox 的 min/max row/col 对齐；这里红块是非矩形，必须先做归一化形状比较再计算平移。
#     - 原 generator 只需保证某条视线唯一；这里要同时保证：每个结构只有一个触边腔体、真块与腔体同形同向、近端诱饵更近但错形。
#   placement_legality:
#     - 仍采用 output-first：先构造“结构 + 最终红块 + 黑轨迹”的补丁，再反推出输入中的真块和近端诱饵位置。
#     - 结构放置时保留完整结构 bbox 和整条轨迹 footprint，避免别的结构、诱饵或干扰红块侵入腔体/轨迹区域。
#     - 形状模板经过单调性约束，平移轨迹不会覆盖结构实心部分；因此黑轨迹不会把结构色错误涂黑。
#   observability:
#     - 人眼可以同时观察到三个线索：结构口沿是间隔齿、候选红块是非矩形、真正进入的是“远处同形块”而不是“近处错形块”。
#     - 不同样例会重复出现 3 列与 5 列两类齿节奏，避免把变化误读成随机洞形。
#   cleanliness:
#     - 结构之间保留 >=4 格间距；诱饵只放在对应结构的直线路径上；额外干扰红块避开所有结构 bbox 与轨迹 footprint。
#     - 颜色逻辑保持原题简洁：结构单色、匹配块红色、轨迹黑色、其他红块消失。
# solver_assumptions:
#   - 每个结构 bbox 内存在且仅存在一个最大的触边 bg 腔体，该腔体只触及一个 bbox 边。
#   - 真匹配红块与该腔体同形同向，仅沿腔体开口法线平移，不旋转不翻转。
#   - 同一网格内不会出现两个完全同形且都在同一结构合法平移轴上的真候选；generator 用唯一模板和 oracle 保证这一点。
#   - 每个 matched 结构的更近红块必为错形，因此 solver 的“按形状精确筛选”是真正必要的，而不是 generator 偶然多余。


_DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

_SHAPES: dict[str, np.ndarray] = {
    "u3": np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    "tall3": np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    "deep3": np.array([[1, 0, 1], [1, 1, 1], [1, 1, 1]], dtype=bool),
    "u5": np.array([[1, 0, 1, 0, 1], [1, 1, 1, 1, 1]], dtype=bool),
    "tall5": np.array([[1, 0, 1, 0, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]], dtype=bool),
    "hook5": np.array([[1, 0, 1, 0, 1], [1, 1, 1, 0, 1], [1, 1, 1, 1, 1]], dtype=bool),
}

_SHAPES_BY_WIDTH: dict[int, list[str]] = {}
for _name, _shape in _SHAPES.items():
    _SHAPES_BY_WIDTH.setdefault(int(_shape.shape[1]), []).append(_name)

_DISTRACTOR_MASKS = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
)


def _normalize_points(coords: list[tuple[int, int]] | np.ndarray) -> tuple[tuple[tuple[int, int], ...], int, int]:
    pts = [(int(r), int(c)) for r, c in coords]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    rel = sorted((int(r - r0), int(c - c0)) for r, c in pts)
    h = max(r for r, _ in rel) + 1
    w = max(c for _, c in rel) + 1
    return tuple(rel), int(h), int(w)


def _component_infos(grid: np.ndarray) -> list[dict]:
    red = int(Color.RED)
    infos: list[dict] = []
    for rid, coords in enumerate(components_mask(grid == red, connectivity=4)):
        pts = [(int(r), int(c)) for r, c in coords]
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        shape_key, h, w = _normalize_points(pts)
        infos.append(
            {
                "rid": int(rid),
                "coords": pts,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "top_left": (int(r0), int(c0)),
                "shape": shape_key,
                "h": int(h),
                "w": int(w),
                "area": int(len(pts)),
            }
        )
    return infos


def _extract_socket(struct_grid: np.ndarray, bg: int, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    local = np.array(struct_grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    cavities = components_mask(local == int(bg), connectivity=4)
    candidates = []
    for comp in cavities:
        pts = [(int(rr), int(cc)) for rr, cc in comp]
        sides = []
        if any(rr == 0 for rr, _ in pts):
            sides.append("top")
        if any(rr == local.shape[0] - 1 for rr, _ in pts):
            sides.append("bottom")
        if any(cc == 0 for _, cc in pts):
            sides.append("left")
        if any(cc == local.shape[1] - 1 for _, cc in pts):
            sides.append("right")
        if len(sides) != 1:
            continue
        shape_key, h, w = _normalize_points(pts)
        rr0 = min(rr for rr, _ in pts)
        cc0 = min(cc for _, cc in pts)
        candidates.append(
            {
                "side": sides[0],
                "shape": shape_key,
                "h": int(h),
                "w": int(w),
                "area": int(len(pts)),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "top_left": (int(r0 + rr0), int(c0 + cc0)),
                "coords": [(int(r0 + rr), int(c0 + cc)) for rr, cc in pts],
            }
        )
    assert candidates, "expected one boundary socket cavity per structure"
    candidates.sort(key=lambda d: (-int(d["area"]), int(d["top_left"][0]), int(d["top_left"][1])))
    return candidates[0]


def _extract_structures(grid: np.ndarray, bg: int) -> list[dict]:
    red = int(Color.RED)
    struct_mask = (grid != int(bg)) & (grid != red)
    comps = components_mask(struct_mask, connectivity=4)
    assert comps, "expected at least one non-bg non-red structure"
    out: list[dict] = []
    for sid, coords in enumerate(comps):
        socket = _extract_socket(grid, bg, coords)
        socket["sid"] = int(sid)
        out.append(socket)
    return out


def _aligned_distance(red_info: dict, socket: dict) -> int | None:
    rr0, cc0 = map(int, red_info["top_left"])
    sr0, sc0 = map(int, socket["top_left"])
    side = str(socket["side"])
    if side == "bottom":
        if cc0 != sc0 or rr0 <= sr0:
            return None
        return int(rr0 - sr0)
    if side == "top":
        if cc0 != sc0 or rr0 >= sr0:
            return None
        return int(sr0 - rr0)
    if side == "right":
        if rr0 != sr0 or cc0 <= sc0:
            return None
        return int(cc0 - sc0)
    if rr0 != sr0 or cc0 >= sc0:
        return None
    return int(sc0 - cc0)


def _socket_candidates(red_infos: list[dict], socket: dict) -> list[tuple[int, dict]]:
    candidates = []
    for info in red_infos:
        if info["shape"] != socket["shape"]:
            continue
        dist = _aligned_distance(info, socket)
        if dist is None or dist <= 0:
            continue
        candidates.append((int(dist), info))
    candidates.sort(key=lambda t: (int(t[0]), int(t[1]["rid"])))
    return candidates


def _find_assignments(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict], list[dict]]:
    red_infos = _component_infos(grid)
    assert red_infos, "expected at least one red component"
    structures = _extract_structures(grid, bg)

    assignments: list[dict] = []
    used_red: set[int] = set()
    order = sorted(structures, key=lambda sock: (int(sock["area"]), int(sock["h"]), int(sock["w"])), reverse=True)
    for socket in order:
        for dist, info in _socket_candidates(red_infos, socket):
            if int(info["rid"]) in used_red:
                continue
            used_red.add(int(info["rid"]))
            assignments.append({"socket": socket, "red": info, "dist": int(dist)})
            break
    return assignments, red_infos, structures


def _count_axis_decoys(grid: np.ndarray, bg: int) -> int:
    red_infos = _component_infos(grid)
    structures = _extract_structures(grid, bg)
    count = 0
    for socket in structures:
        shape_cands = _socket_candidates(red_infos, socket)
        if not shape_cands:
            continue
        true_dist = int(shape_cands[0][0])
        for info in red_infos:
            if info["shape"] == socket["shape"]:
                continue
            dist = _aligned_distance(info, socket)
            if dist is not None and 0 < int(dist) < true_dist:
                count += 1
                break
    return int(count)


def analyze_fusion_features(grid: np.ndarray) -> dict:
    bg = int(get_bg(grid))
    assignments, _red_infos, structures = _find_assignments(grid, bg)
    nonrect = sum(1 for item in assignments if int(item["red"]["area"]) < int(item["red"]["h"]) * int(item["red"]["w"]))
    return {
        "structures": int(len(structures)),
        "matched": int(len(assignments)),
        "nonrect_matched": int(nonrect),
        "axis_decoys": int(_count_axis_decoys(grid, bg)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide only the exact non-rectangular red plugs into interval-tooth cavities, painting the translation path black.

    Input structure:
    - The background is the dominant color. Non-bg, non-red components are solid single-color blocks with one boundary cavity.
    - Each cavity is an irregular socket whose upper profile has spaced teeth (`101` or `10101` in canonical form).
    - Several red components sit outside the openings; some are exact copies of a socket, while nearer axis-aligned decoys have the wrong shape.

    Rule:
    - Extract each structure's unique boundary cavity as a full 2D shape, including its gapped top profile and deeper irregular outline.
    - A red component matches only if its normalized shape is identical to that cavity and it lies on the same straight translation axis outside the opening.
    - Slide the matching red straight into the cavity, color every intermediate placement black, and erase every non-matching red.

    Key insight: The valid plug is determined by full geometric fit to the interval-tooth cavity, so a closer red on the same axis is ignored unless its whole shape can nest exactly.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    red = int(Color.RED)
    black = int(Color.BLACK)
    h, w = g.shape

    assert red in set(map(int, np.unique(g)))

    assignments, _red_infos, _structures = _find_assignments(g, bg)

    out = g.copy()
    out[out == red] = int(bg)

    for item in assignments:
        socket = item["socket"]
        info = item["red"]
        fr, fc = map(int, socket["top_left"])
        rr0, cc0 = map(int, info["top_left"])
        shift_r = int(fr - rr0)
        shift_c = int(fc - cc0)
        assert shift_r == 0 or shift_c == 0

        step_dr = 0 if shift_r == 0 else (1 if shift_r > 0 else -1)
        step_dc = 0 if shift_c == 0 else (1 if shift_c > 0 else -1)
        steps = int(max(abs(shift_r), abs(shift_c)))

        for step in range(steps):
            off_r = int(step * step_dr)
            off_c = int(step * step_dc)
            for r, c in info["coords"]:
                nr = int(r + off_r)
                nc = int(c + off_c)
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = int(black)

        for r, c in info["coords"]:
            nr = int(r + shift_r)
            nc = int(c + shift_c)
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = int(red)

    return out


# --------------- generator ---------------

GRID_H_RANGE = (20, 25)
GRID_W_RANGE = (24, 30)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

TRUE_SLIDE_RANGE = (7, 9)
DECOY_EXTRA_RANGE = (0, 1)
STRUCT_PADDING = 4

TOTAL_RED_MIN = 5
TOTAL_RED_MAX = 11
DISTRACTOR_MIN = 1
DISTRACTOR_MAX = 6

MAX_RETRIES = 500

_LAYOUT_BRANCHES = (
    {"n_structs": 2, "matched": 2},
    {"n_structs": 2, "matched": 1},
    {"n_structs": 3, "matched": 3},
    {"n_structs": 3, "matched": 2},
)


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _paint_shape(patch: np.ndarray, shape: np.ndarray, top: int, left: int, color: int) -> None:
    sh, sw = map(int, shape.shape)
    for rr in range(sh):
        for cc in range(sw):
            if not bool(shape[rr, cc]):
                continue
            patch[int(top + rr), int(left + cc)] = int(color)


def _mask_to_patch(mask: np.ndarray, fg: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(fg)
    return patch


def _build_structure(shape: np.ndarray, bg: int, sc: int) -> tuple[np.ndarray, tuple[int, int]]:
    sh, sw = map(int, shape.shape)
    structure = np.full((sh + 2, sw + 4), int(sc), dtype=int)
    top = 2
    left = 2
    cavity = structure[top : top + sh, left : left + sw]
    cavity[shape] = int(bg)
    return structure, (int(top), int(left))


def _matched_patches(
    shape_name: str,
    decoy_name: str,
    slide: int,
    decoy_slide: int,
    rot_k: int,
    bg: int,
    sc: int,
) -> dict:
    red = int(Color.RED)
    black = int(Color.BLACK)
    shape = _SHAPES[shape_name]
    decoy = _SHAPES[decoy_name]
    structure, (slot_top, slot_left) = _build_structure(shape, bg, sc)

    base_h, base_w = map(int, structure.shape)
    patch_h = int(base_h + slide)

    struct_patch = np.full((patch_h, base_w), int(bg), dtype=int)
    paste(struct_patch, structure, 0, 0, transparent=bg)
    input_patch = struct_patch.copy()
    output_patch = struct_patch.copy()

    for off in range(1, int(slide) + 1):
        _paint_shape(output_patch, shape, int(slot_top + off), int(slot_left), int(black))
    _paint_shape(output_patch, shape, int(slot_top), int(slot_left), int(red))
    _paint_shape(input_patch, shape, int(slot_top + slide), int(slot_left), int(red))
    _paint_shape(input_patch, decoy, int(slot_top + decoy_slide), int(slot_left), int(red))

    occupancy = np.full_like(struct_patch, int(bg))
    occupancy[:base_h, :] = int(sc)
    occupancy[(input_patch != int(bg)) | (output_patch != int(bg))] = int(sc)

    return {
        "matched": True,
        "shape_name": shape_name,
        "decoy_name": decoy_name,
        "structure": np.rot90(struct_patch, k=rot_k),
        "input": np.rot90(input_patch, k=rot_k),
        "output": np.rot90(output_patch, k=rot_k),
        "occupancy": np.rot90(occupancy, k=rot_k),
    }


def _unmatched_patches(shape_name: str, rot_k: int, bg: int, sc: int) -> dict:
    shape = _SHAPES[shape_name]
    structure, _slot = _build_structure(shape, bg, sc)
    occupancy = np.full_like(structure, int(sc))
    return {
        "matched": False,
        "shape_name": shape_name,
        "decoy_name": None,
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(structure.copy(), k=rot_k),
        "output": np.rot90(structure.copy(), k=rot_k),
        "occupancy": np.rot90(occupancy, k=rot_k),
    }


def _place_patch(canvas: np.ndarray, patch: np.ndarray, bg: int, padding: int) -> tuple[int, int] | None:
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


def _choose_shape_names(n_structs: int) -> list[str]:
    width3 = list(_SHAPES_BY_WIDTH[3])
    width5 = list(_SHAPES_BY_WIDTH[5])
    chosen: list[str] = []
    if n_structs >= 2:
        chosen.append(random.choice(width3))
        chosen.append(random.choice(width5))
    pool = [name for name in _SHAPES if name not in chosen]
    while len(chosen) < n_structs:
        chosen.append(random.choice(pool))
        pool = [name for name in _SHAPES if name not in chosen]
    random.shuffle(chosen)
    return chosen


def _sample_canvas_size(n_structs: int) -> tuple[int, int]:
    if n_structs >= 3:
        h_lo = max(int(GRID_H_RANGE[0]), 22)
        w_lo = max(int(GRID_W_RANGE[0]), 27)
        return random.randint(h_lo, int(GRID_H_RANGE[1])), random.randint(w_lo, int(GRID_W_RANGE[1]))
    return random.randint(*GRID_H_RANGE), random.randint(*GRID_W_RANGE)


def _place_structures(
    specs: list[dict], h: int, w: int, bg: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    inp = np.full((h, w), int(bg), dtype=int)
    out = np.full((h, w), int(bg), dtype=int)
    struct_grid = np.full((h, w), int(bg), dtype=int)
    occ_canvas = np.full((h, w), int(bg), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    for spec in specs:
        loc = _place_patch(occ_canvas, spec["occupancy"], bg, padding=STRUCT_PADDING)
        if loc is None:
            return None
        r0, c0 = map(int, loc)
        ph, pw = map(int, spec["occupancy"].shape)
        paste(occ_canvas, spec["occupancy"], r0, c0, transparent=bg)
        paste(out, spec["output"], r0, c0, transparent=bg)
        paste(inp, spec["input"], r0, c0, transparent=bg)
        paste(struct_grid, spec["structure"], r0, c0, transparent=bg)
        reserved[r0 : r0 + ph, c0 : c0 + pw] |= spec["occupancy"] != int(bg)

    return inp, out, struct_grid, reserved


def _place_distractors(inp: np.ndarray, reserved: np.ndarray, bg: int, sc: int, count: int) -> bool:
    h, w = inp.shape
    red_mask = inp == int(Color.RED)
    blocked_core = reserved | (inp != int(bg))

    for _ in range(count):
        blocked = blocked_core | _dilate4(red_mask)
        canvas = np.full((h, w), int(bg), dtype=int)
        canvas[blocked] = int(sc)
        masks = list(_DISTRACTOR_MASKS)
        random.shuffle(masks)
        placed = False
        for mask in masks:
            patch = _mask_to_patch(mask, int(Color.RED), int(bg))
            loc = _place_patch(canvas, patch, bg, padding=0)
            if loc is None:
                continue
            rr, cc = map(int, loc)
            paste(inp, patch, rr, cc, transparent=bg)
            red_mask[rr : rr + patch.shape[0], cc : cc + patch.shape[1]] |= mask
            blocked_core[rr : rr + patch.shape[0], cc : cc + patch.shape[1]] |= mask
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt(layout: dict) -> np.ndarray | None:
    bg = random.choice(BG_COLORS)
    sc = random.choice([c for c in STRUCT_COLORS if c not in (bg, int(Color.RED), int(Color.BLACK))])
    n_structs = int(layout["n_structs"])
    matched = int(layout["matched"])
    if not (0 < matched <= n_structs):
        return None
    h, w = _sample_canvas_size(n_structs)

    shape_names = _choose_shape_names(n_structs)

    specs = []
    for idx, shape_name in enumerate(shape_names):
        shape = _SHAPES[shape_name]
        rot_k = random.randrange(4)
        if idx < matched:
            family = int(shape.shape[1])
            decoy_choices = [name for name in _SHAPES_BY_WIDTH[family] if name != shape_name]
            if not decoy_choices:
                return None
            decoy_name = random.choice(decoy_choices)
            decoy = _SHAPES[decoy_name]
            slide = random.randint(*TRUE_SLIDE_RANGE)
            decoy_slide = int(shape.shape[0]) + random.randint(*DECOY_EXTRA_RANGE)
            if int(decoy_slide) + int(decoy.shape[0]) > int(slide) - 1:
                return None
            specs.append(_matched_patches(shape_name, decoy_name, slide, decoy_slide, rot_k, bg, sc))
        else:
            specs.append(_unmatched_patches(shape_name, rot_k, bg, sc))

    random.shuffle(specs)
    specs.sort(key=lambda spec: np.count_nonzero(spec["occupancy"] != bg), reverse=True)

    placed = _place_structures(specs, h, w, bg)
    if placed is None:
        return None
    inp, out, _struct_grid, reserved = placed

    metrics = analyze_fusion_features(inp)
    if int(metrics["matched"]) != int(matched):
        return None
    if int(metrics["nonrect_matched"]) != int(matched):
        return None
    if int(metrics["axis_decoys"]) < int(matched):
        return None

    base_reds = len(components_mask(inp == int(Color.RED), connectivity=4))
    min_dist = max(DISTRACTOR_MIN, TOTAL_RED_MIN - base_reds)
    max_dist = min(DISTRACTOR_MAX, TOTAL_RED_MAX - base_reds)
    if min_dist > max_dist:
        return None
    distractors = random.randint(min_dist, max_dist)
    if not _place_distractors(inp, reserved, bg, sc, distractors):
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    branch_order = list(_LAYOUT_BRANCHES)
    random.shuffle(branch_order)
    per_branch = max(1, MAX_RETRIES // len(branch_order))
    for layout in branch_order:
        for _ in range(per_branch):
            grid = _gen_attempt(layout)
            if grid is not None:
                return grid
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(random.choice(_LAYOUT_BRANCHES))
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 combo D1/1+D2/0+D8/0")
