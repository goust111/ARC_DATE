# Variant of 8b9c3697
# Dimension: D1 — 匹配条件：行列对齐射线投射 (match_cond)
# Change: 从结构内部插槽（socket）向四个基本方向发射射线，命中红色色块即确定匹配关系和滑入方向（s.col/row == d... → 形状嵌合匹配

from __future__ import annotations

import random
import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location

# design_analysis:
#   variant_goal:
#     - 保留“红块滑入结构槽口并留下黑色轨迹”的主行为。
#     - 将匹配依据从“一维射线先看到谁”改成“二维槽形与红块形状完全一致”。
#   concrete_visual:
#     - 每个结构都是单色实体条块，某一侧挖出一个朝外开口的异形槽。
#     - 槽形来自少量可见差异明显的多格形：L、T、Z、U、P。
#     - 真正的匹配红块与槽形同形同向，沿开口法线平移进入槽内。
#     - 每个已匹配结构的通道上还会放一个更近的错形红诱饵；旧题的射线规则会先撞到它，新规则必须忽略它。
#   behavior_trace:
#     - 原逻辑：从槽口内部沿开口方向打射线，最近的红块就被认为是目标。
#     - 新逻辑：先从结构 bbox 内提取开口槽的完整 bg 形状，再在全部红连通块里找“归一化形状完全相同”且位于开口外同一平移轴上的候选。
#     - 差异：诱饵红块虽更近，但形状不符，因此不会被吸入；真正同形块即使更远也会被选中。
#   complexity_calibration:
#     - 形状库只保留 5 种，且单个网格里结构形状互不重复，避免匹配歧义。
#     - 每个匹配结构都带一个近端错形诱饵，保证 D1 改动在样例中稳定可见。
#     - 额外干扰红块限制为小矩形，不会意外与异形槽同形。
#   implicit_assumption_audit:
#     - 原 main 依赖“沿某列/某行看到的第一个红块就是匹配对象”；这里不再成立，相关 ray hit 统计全部移除。
#     - 原 main 按 bbox 边界面对齐 min/max row/col；这里必须先验证红块形状与槽形一致，再用槽 bbox 的左上角计算纯平移。
#     - 原 generator 会主动避开所有视线；这里改为反向利用视线，在真块前面放错形诱饵。
#   placement_legality:
#     - 结构放置仍采用 output-first；输出补丁包含结构、最终红块和整条黑轨迹，因此结构之间不会在最终态相撞。
#     - 诱饵放在真块轨迹上但不与真块初始位置重叠，因此输入合法，输出中该位置自然转成黑轨迹。
#     - 额外干扰红块避开输入与输出的所有非背景区域，避免遮挡结构或轨迹。
#   observability:
#     - 2-3 个样例中人眼可直接看到：不同结构缺口形状不同，而进入槽中的红块正是同形那一个，不是最近的那一个。
#   cleanliness:
#     - 结构之间保留 >=4 格间隔；红诱饵只占据滑行通道，不引入额外颜色或覆盖层级。
#   color_reasoning:
#     - 颜色规则不变：结构单色、匹配块最终仍是红色、轨迹变黑、其他红块消失。
# solver_assumptions:
#   - 每个结构 bbox 内只有一个触边槽形连通块，并且只触及一个 bbox 边。
#   - 真匹配红块与槽形同形同向，且仅沿开口法线平移，不旋转不翻转。
#   - 同一网格内结构槽形互异，因此不会出现一个红块同时合法匹配多个结构。
#   - generator 通过构造保证以上条件，并通过 oracle 与诱饵计数再次验证。


_DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

_SHAPES: dict[str, np.ndarray] = {
    "L": np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
    "P": np.array([[1, 1], [1, 1], [1, 0]], dtype=int),
    "T": np.array([[0, 1, 0], [1, 1, 1]], dtype=int),
    "Z": np.array([[0, 1, 1], [1, 1, 0]], dtype=int),
    "U": np.array([[1, 0, 1], [1, 1, 1]], dtype=int),
}

_SHAPES_BY_WIDTH: dict[int, list[str]] = {}
for _name, _shape in _SHAPES.items():
    _SHAPES_BY_WIDTH.setdefault(int(_shape.shape[1]), []).append(_name)


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
    comps = components_mask(grid == red, connectivity=4)
    infos = []
    for rid, coords in enumerate(comps):
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
    local = struct_grid[r0 : r1 + 1, c0 : c1 + 1]
    cavities = components_mask(local == int(bg), connectivity=4)
    cand = []
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
        cand.append(
            {
                "side": sides[0],
                "local_pts": pts,
                "shape": shape_key,
                "h": int(h),
                "w": int(w),
                "top_left": (int(r0 + rr0), int(c0 + cc0)),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "area": int(len(pts)),
            }
        )
    assert cand, "expected a boundary socket cavity in each structure"
    cand.sort(key=lambda d: (-int(d["area"]), int(d["top_left"][0]), int(d["top_left"][1])))
    return cand[0]


def _extract_structures(grid: np.ndarray, bg: int) -> list[dict]:
    red = int(Color.RED)
    struct_mask = (grid != int(bg)) & (grid != red)
    struct_comps = components_mask(struct_mask, connectivity=4)
    assert struct_comps, "expected at least one non-bg non-red structure"
    out = []
    for sid, coords in enumerate(struct_comps):
        sock = _extract_socket(grid, bg, coords)
        sock["sid"] = int(sid)
        sock["coords"] = [(int(r), int(c)) for r, c in coords]
        out.append(sock)
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
    cands = []
    for info in red_infos:
        if info["shape"] != socket["shape"]:
            continue
        dist = _aligned_distance(info, socket)
        if dist is None or dist <= 0:
            continue
        cands.append((int(dist), info))
    cands.sort(key=lambda t: (int(t[0]), int(t[1]["rid"])))
    return cands


def _count_shape_decoys(grid: np.ndarray, bg: int) -> int:
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide matching red polyominoes into shaped side-notches of solid structures, painting the traversed path black.

    Input structure:
    - bg is the dominant color. Non-bg, non-red components are solid single-color blocks with one boundary notch-shaped cavity.
    - Several red components sit outside the blocks; some are exact shape copies of a cavity, others are distractors.

    Rule:
    - For each structure, extract the full cavity shape from the bg cells cut into one bbox edge.
    - Find the red component with the same normalized 2D shape that lies on the cavity's translation axis outside that edge.
    - Slide that matching red straight into the cavity; every intermediate placement becomes black, and all non-matching reds disappear.

    Key insight: Matching is determined by exact cavity-shape agreement, not by whichever red happens to be first on a row/column ray.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    red = int(Color.RED)
    black = int(Color.BLACK)
    h, w = g.shape

    assert red in set(map(int, np.unique(g)))

    red_infos = _component_infos(g)
    assert red_infos, "expected at least one red component"
    structures = _extract_structures(g, bg)

    assignments = []
    used_red: set[int] = set()
    for socket in structures:
        cands = _socket_candidates(red_infos, socket)
        for dist, info in cands:
            if int(info["rid"]) in used_red:
                continue
            used_red.add(int(info["rid"]))
            assignments.append((socket, info, int(dist)))
            break

    out = g.copy()
    out[out == red] = int(bg)

    for socket, info, _dist in assignments:
        fr, fc = map(int, socket["top_left"])
        rr0, cc0 = map(int, info["top_left"])
        shift_r = int(fr - rr0)
        shift_c = int(fc - cc0)
        assert shift_r == 0 or shift_c == 0
        steps = int(max(abs(shift_r), abs(shift_c)))
        step_dr = 0 if shift_r == 0 else (1 if shift_r > 0 else -1)
        step_dc = 0 if shift_c == 0 else (1 if shift_c > 0 else -1)

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


# --------------- generator (D1 shape-match variant) ---------------

GRID_H_RANGE = (19, 23)
GRID_W_RANGE = (22, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

STRUCT_COUNT_RANGE = (2, 3)
UNMATCHED_PROB = 0.30
TRUE_SLIDE_RANGE = (6, 8)
DECOY_EXTRA_RANGE = (0, 1)
STRUCT_PADDING = 4

TOTAL_RED_MIN = 5
TOTAL_RED_MAX = 9
DISTRACTOR_MIN = 1
DISTRACTOR_MAX = 4
DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1))

MAX_RETRIES = 400


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _build_structure(shape_name: str, bg: int, sc: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    shape = _SHAPES[shape_name]
    sh, sw = map(int, shape.shape)
    h = int(sh + 3)
    w = int(sw + 4)
    structure = np.full((h, w), int(sc), dtype=int)
    top = int(h - sh)
    left = int((w - sw) // 2)
    structure[top : top + sh, left : left + sw][shape.astype(bool)] = int(bg)
    return structure, shape.copy(), (int(top), int(left))


def _paint_shape(patch: np.ndarray, shape: np.ndarray, top: int, left: int, color: int) -> None:
    sh, sw = map(int, shape.shape)
    for rr in range(sh):
        for cc in range(sw):
            if int(shape[rr, cc]) != 1:
                continue
            patch[int(top + rr), int(left + cc)] = int(color)


def _matched_patches(shape_name: str, decoy_name: str, slide: int, decoy_slide: int, rot_k: int, bg: int, sc: int) -> dict:
    red = int(Color.RED)
    black = int(Color.BLACK)
    structure, shape, (slot_top, slot_left) = _build_structure(shape_name, bg, sc)
    decoy = _SHAPES[decoy_name]
    sh, _sw = map(int, shape.shape)
    dh, _dw = map(int, decoy.shape)

    base_h, base_w = map(int, structure.shape)
    patch_h = int(base_h + slide)
    patch_w = int(base_w)

    struct_patch = np.full((patch_h, patch_w), int(bg), dtype=int)
    paste(struct_patch, structure, 0, 0, transparent=bg)
    input_patch = struct_patch.copy()
    output_patch = struct_patch.copy()

    for off in range(1, int(slide) + 1):
        _paint_shape(output_patch, shape, int(slot_top + off), int(slot_left), int(black))
    _paint_shape(output_patch, shape, int(slot_top), int(slot_left), int(red))
    _paint_shape(input_patch, shape, int(slot_top + slide), int(slot_left), int(red))

    decoy_top = int(base_h - dh + decoy_slide)
    _paint_shape(input_patch, decoy, int(decoy_top), int(slot_left), int(red))

    return {
        "matched": True,
        "shape_name": shape_name,
        "decoy_name": decoy_name,
        "structure": np.rot90(struct_patch, k=rot_k),
        "input": np.rot90(input_patch, k=rot_k),
        "output": np.rot90(output_patch, k=rot_k),
        "occupancy": np.rot90(output_patch, k=rot_k),
    }


def _unmatched_patches(shape_name: str, rot_k: int, bg: int, sc: int) -> dict:
    structure, _shape, _slot = _build_structure(shape_name, bg, sc)
    return {
        "matched": False,
        "shape_name": shape_name,
        "decoy_name": None,
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(structure.copy(), k=rot_k),
        "output": np.rot90(structure.copy(), k=rot_k),
        "occupancy": np.rot90(structure.copy(), k=rot_k),
    }


def _place_patch(canvas: np.ndarray, patch: np.ndarray, bg: int, padding: int) -> tuple[int, int] | None:
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


def _place_structures(
    specs: list[dict], h: int, w: int, bg: int, sc: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    inp = np.full((h, w), int(bg), dtype=int)
    out = np.full((h, w), int(bg), dtype=int)
    struct_grid = np.full((h, w), int(bg), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    for spec in specs:
        loc = _place_patch(out, spec["occupancy"], bg, padding=STRUCT_PADDING)
        if loc is None:
            return None
        r0, c0 = map(int, loc)
        ph, pw = map(int, spec["occupancy"].shape)
        paste(out, spec["output"], r0, c0, transparent=bg)
        paste(inp, spec["input"], r0, c0, transparent=bg)
        paste(struct_grid, spec["structure"], r0, c0, transparent=bg)
        reserved[r0 : r0 + ph, c0 : c0 + pw] = True

    if np.any((struct_grid != bg) & (inp == int(Color.RED))):
        return None
    return inp, out, struct_grid, reserved


def _place_distractors(inp: np.ndarray, out: np.ndarray, reserved: np.ndarray, bg: int, sc: int, count: int) -> bool:
    h, w = inp.shape
    red_mask = inp == int(Color.RED)
    # Keep red noise fully outside each placed structure patch; otherwise it can fill a cavity
    # and break the solver's socket extraction.
    blocked_core = reserved | (inp != int(bg)) | (out != int(bg))
    for _ in range(count):
        blocked = blocked_core | _dilate4(red_mask)
        canvas = np.full((h, w), int(bg), dtype=int)
        canvas[blocked] = int(sc)
        shapes = list(DISTRACTOR_SHAPES)
        random.shuffle(shapes)
        placed = False
        for rh, rw in shapes:
            patch = np.full((rh, rw), int(Color.RED), dtype=int)
            loc = _place_patch(canvas, patch, bg, padding=0)
            if loc is None:
                continue
            paste(inp, patch, int(loc[0]), int(loc[1]), transparent=bg)
            red_mask[int(loc[0]) : int(loc[0] + rh), int(loc[1]) : int(loc[1] + rw)] = True
            blocked_core[int(loc[0]) : int(loc[0] + rh), int(loc[1]) : int(loc[1] + rw)] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_COLORS)
    sc = random.choice([c for c in STRUCT_COLORS if c not in (bg, int(Color.RED), int(Color.BLACK))])
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_structs = random.randint(*STRUCT_COUNT_RANGE)
    unmatched = 1 if random.random() < UNMATCHED_PROB else 0
    matched = int(n_structs - unmatched)
    if matched <= 0 or n_structs > len(_SHAPES):
        return None

    shape_names = random.sample(list(_SHAPES.keys()), k=n_structs)
    specs = []
    for idx, shape_name in enumerate(shape_names):
        rot_k = random.randrange(4)
        if idx < matched:
            decoy_choices = [nm for nm in _SHAPES_BY_WIDTH[int(_SHAPES[shape_name].shape[1])] if nm != shape_name]
            if not decoy_choices:
                return None
            decoy_name = random.choice(decoy_choices)
            slide = random.randint(*TRUE_SLIDE_RANGE)
            sh = int(_SHAPES[shape_name].shape[0])
            dh = int(_SHAPES[decoy_name].shape[0])
            decoy_slide = int(dh + random.randint(*DECOY_EXTRA_RANGE))
            if int(decoy_slide) + int(sh) > int(slide):
                return None
            specs.append(_matched_patches(shape_name, decoy_name, slide, decoy_slide, rot_k, bg, sc))
        else:
            specs.append(_unmatched_patches(shape_name, rot_k, bg, sc))

    random.shuffle(specs)
    specs.sort(key=lambda s: np.count_nonzero(s["occupancy"] != bg), reverse=True)

    placed = _place_structures(specs, h, w, bg, sc)
    if placed is None:
        return None
    inp, out, _struct_grid, reserved = placed

    base_reds = len(components_mask(inp == int(Color.RED), connectivity=4))
    min_dist = max(DISTRACTOR_MIN, TOTAL_RED_MIN - base_reds)
    max_dist = min(DISTRACTOR_MAX, TOTAL_RED_MAX - base_reds)
    if min_dist > max_dist:
        return None
    distractors = random.randint(min_dist, max_dist)
    if not _place_distractors(inp, out, reserved, bg, sc, distractors):
        return None

    if _count_shape_decoys(inp, bg) < matched:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 D1 shape-match variant")
