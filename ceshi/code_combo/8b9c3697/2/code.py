# Combo variant of 8b9c3697 (index 2)
# combo_spec: D1/1+D2/0
# Name: 形状嵌合非矩形色块

from __future__ import annotations

import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location

# design_analysis:
#   variant_goal:
#     - 保留原题“外部红块直线滑入结构槽口、路径染黑、其余红块消失”的整体叙事。
#     - 同时落实两个改动：红块改成非矩形多格块；匹配条件从射线先后改成“允许旋转后的形状嵌合”。
#   concrete_visual:
#     - 每个结构仍是单色大块，但某一侧被挖出一个触边异形槽，槽形来自少量醒目的 polyomino：L、T、U、P。
#     - 真匹配红块与槽属于同一形状家族，但输入里通常被旋转成另一个朝向；它位于槽口外的平移轴线上。
#     - 同一滑行通道上再放一个更近的错形红诱饵，肉眼能看到“最近的不一定会进入槽里”。
#     - 额外红干扰块也全部使用非矩形小 polyomino，避免退化回“红矩形滑入”。
#   behavior_trace:
#     - 原逻辑：从结构内部空位沿行/列发射射线，先撞到哪个红块就把它吸进来。
#     - 融合后：先提取槽口的完整二维空腔形状，再对每个红连通块尝试四种旋转；只有某个旋转后与槽形完全一致且 top-left 仍在同一滑轴上的红块才可匹配。
#     - 输出时，真匹配块按匹配后的朝向滑入槽内；原始位置和中间通道都变黑；近处错形块与所有其他红块一起被擦除。
#   complexity_calibration:
#     - 结构形状库只保留 4 类，且单个网格里结构家族互不重复，避免一个红块可同时匹配多个槽。
#     - 每个已匹配结构强制带 1 个更近的错形诱饵，保证 D1 的“不能看最近红块”在样例里稳定可见。
#     - 真匹配块强制至少有 1 个需要非零旋转才能嵌合，保证 D2 与 D1 的协同不是表面上的。
#     - 干扰红块使用面积 3-4 的小异形，视觉上仍明显是“非矩形”，但不会和结构槽形同构。
#   implicit_assumption_audit:
#     - 原 main 假设“匹配对象是同列/同行射线先撞到的红块”；这里被近端诱饵明确打破，因此必须移除 ray-hit 计数逻辑。
#     - 原 main 用红块 bbox 的某一侧去贴住矩形 socket；这里不再成立，因为最终贴入的是旋转后的 polyomino，必须先求出匹配旋转，再用旋转后形状的 top-left 对齐槽口 top-left。
#     - 原 generator 依赖“干扰红块不能出现在视线里”；这里反过来主动把错形诱饵放进视线里，并用形状判别来排除它。
#     - 新 main 额外依赖“旋转只围绕红块归一化 bbox 的左上角重表达，不涉及翻折”；generator 用统一的 top-left 平移轴构造来保证这一点。
#   placement_legality:
#     - 继续使用 output-first：每个结构补丁的 occupancy 同时覆盖结构、最终红块、黑轨迹以及输入态真红块/诱饵红块，因此不同补丁不会互相遮挡。
#     - 真红块和诱饵都只放在槽口外的背景通道中，不进入结构实体，也不会越界。
#     - 额外干扰块避开所有已保留区域并与现有红块至少隔一个 4 邻域，避免连成同一连通块。
#   observability:
#     - 观察 2-3 个样例时，人能直接看到：结构缺口是 L/T/U/P 一类异形；真正进入槽里的红块和槽是同形但常常转了向；更近的错形红块没有被选中。
#     - 颜色规则仍简单稳定：结构单色、匹配块最终为红、轨迹为黑、其余红消失，没有额外颜色映射负担。
#   cleanliness:
#     - 结构之间通常保留 2-3 格缓冲，单个补丁的黑轨迹和红块都局限在一个窄通道内，避免多层覆盖导致难以阅读。
# solver_assumptions:
#   - 每个结构 bbox 内只有一个触及单一边界的 bg 槽形连通块。
#   - 真匹配红块与槽形在某个 90 度旋转下完全同构，且滑入仅是沿槽口法线的纯平移。
#   - 同一网格内结构的形状家族互不重复，因此每个槽至多有一个合法真匹配。
#   - generator 通过构造与指标校验保证：所有红块均为非矩形；至少一个匹配需要旋转；每个匹配槽前面都有更近的错形诱饵。


_DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

_MATCH_SHAPES: dict[str, np.ndarray] = {
    "L4": np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
    "T4": np.array([[0, 1, 0], [1, 1, 1]], dtype=int),
    "U5": np.array([[1, 0, 1], [1, 1, 1]], dtype=int),
    "P5": np.array([[1, 1], [1, 1], [1, 0]], dtype=int),
}

_NOISE_SHAPES: dict[str, np.ndarray] = {
    "L3": np.array([[1, 0], [1, 1]], dtype=int),
    "S4": np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    "J4": np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
}

GRID_H_RANGE = (20, 25)
GRID_W_RANGE = (24, 30)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

STRUCT_COUNT_RANGE = (2, 3)
UNMATCHED_PROB = 0.30
TRUE_SLIDE_RANGE = (5, 7)
STRUCT_PADDING = 3

TOTAL_RED_MIN = 5
TOTAL_RED_MAX = 9
DISTRACTOR_MIN = 1
DISTRACTOR_MAX = 3

MAX_RETRIES = 500


def _trim_shape(shape: np.ndarray) -> np.ndarray:
    pts = np.argwhere(shape != 0)
    if pts.size == 0:
        return np.zeros((0, 0), dtype=int)
    r0, c0, r1, c1 = map(int, bbox_coords(pts))
    return np.array(shape[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _rotate_shape(shape: np.ndarray, rot_k: int) -> np.ndarray:
    return _trim_shape(np.rot90(np.array(shape, dtype=int), k=int(rot_k)))


def _shape_signature(shape: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    tight = _trim_shape(shape)
    pts = tuple(sorted((int(r), int(c)) for r, c in np.argwhere(tight == 1)))
    return (int(tight.shape[0]), int(tight.shape[1])), pts


def _canonical_family_signature(shape: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    return min(_shape_signature(_rotate_shape(shape, k)) for k in range(4))


_MATCH_FAMILY_SIG = {name: _canonical_family_signature(shape) for name, shape in _MATCH_SHAPES.items()}


def _normalize_points(coords: list[tuple[int, int]] | np.ndarray) -> tuple[tuple[tuple[int, int], ...], int, int]:
    pts = [(int(r), int(c)) for r, c in coords]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    rel = sorted((int(r - r0), int(c - c0)) for r, c in pts)
    h = max(r for r, _ in rel) + 1
    w = max(c for _, c in rel) + 1
    return tuple(rel), int(h), int(w)


def _mask_from_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    _, h, w = _normalize_points(coords)
    mask = np.zeros((h, w), dtype=int)
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    for r, c in coords:
        mask[int(r - r0), int(c - c0)] = 1
    return mask


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
            if int(shape[rr, cc]) != 1:
                continue
            patch[int(top + rr), int(left + cc)] = int(color)


def _place_patch(canvas: np.ndarray, patch: np.ndarray, bg: int, padding: int) -> tuple[int, int] | None:
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


def _component_infos(grid: np.ndarray) -> list[dict]:
    red = int(Color.RED)
    infos = []
    for rid, coords in enumerate(components_mask(grid == red, connectivity=4)):
        pts = [(int(r), int(c)) for r, c in coords]
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        mask = _mask_from_coords(pts)
        infos.append(
            {
                "rid": int(rid),
                "coords": pts,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "top_left": (int(r0), int(c0)),
                "mask": mask,
                "sig": _shape_signature(mask),
                "area": int(len(pts)),
                "bbox_area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
            }
        )
    return infos


def _extract_socket(struct_grid: np.ndarray, bg: int, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    local = struct_grid[r0 : r1 + 1, c0 : c1 + 1]
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
        rr0 = min(rr for rr, _ in pts)
        cc0 = min(cc for _, cc in pts)
        tight = np.zeros((max(rr for rr, _ in pts) - rr0 + 1, max(cc for _, cc in pts) - cc0 + 1), dtype=int)
        for rr, cc in pts:
            tight[int(rr - rr0), int(cc - cc0)] = 1
        candidates.append(
            {
                "side": sides[0],
                "top_left": (int(r0 + rr0), int(c0 + cc0)),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": tight,
                "sig": _shape_signature(tight),
                "area": int(len(pts)),
            }
        )
    assert candidates, "expected one boundary socket cavity per structure"
    candidates.sort(key=lambda d: (-int(d["area"]), int(d["top_left"][0]), int(d["top_left"][1])))
    return candidates[0]


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


def _rotation_to_match(mask: np.ndarray, target_sig: tuple[tuple[int, int], tuple[tuple[int, int], ...]]) -> tuple[int, np.ndarray] | None:
    for rot_k in range(4):
        rotated = _rotate_shape(mask, rot_k)
        if _shape_signature(rotated) == target_sig:
            return int(rot_k), rotated
    return None


def _socket_candidates(red_infos: list[dict], socket: dict) -> list[tuple[int, int, np.ndarray, dict]]:
    cands = []
    for info in red_infos:
        match = _rotation_to_match(info["mask"], socket["sig"])
        if match is None:
            continue
        dist = _aligned_distance(info, socket)
        if dist is None or dist <= 0:
            continue
        rot_k, rotated = match
        cands.append((int(dist), int(rot_k), rotated, info))
    cands.sort(key=lambda t: (int(t[0]), int(t[3]["rid"])))
    return cands


def combo_behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    bg = int(get_bg(np.array(grid, dtype=int)))
    red_infos = _component_infos(np.array(grid, dtype=int))
    structures = _extract_structures(np.array(grid, dtype=int), bg)

    ambiguous_sockets = 0
    rotation_needed = 0
    axis_decoys = 0
    used_red: set[int] = set()
    assignments = 0

    for socket in structures:
        cands = _socket_candidates(red_infos, socket)
        if len(cands) > 1:
            ambiguous_sockets += 1
        chosen = None
        for cand in cands:
            if int(cand[3]["rid"]) in used_red:
                continue
            chosen = cand
            break
        if chosen is None:
            continue
        dist, rot_k, _rotated, info = chosen
        used_red.add(int(info["rid"]))
        assignments += 1
        if int(rot_k) != 0:
            rotation_needed += 1
        for other in red_infos:
            if int(other["rid"]) == int(info["rid"]):
                continue
            other_dist = _aligned_distance(other, socket)
            if other_dist is not None and 0 < int(other_dist) < int(dist):
                axis_decoys += 1
                break

    nonrect_reds = sum(1 for info in red_infos if int(info["area"]) < int(info["bbox_area"]))
    return {
        "structures": int(len(structures)),
        "red_components": int(len(red_infos)),
        "matched": int(assignments),
        "ambiguous_sockets": int(ambiguous_sockets),
        "rotation_needed": int(rotation_needed),
        "axis_decoys": int(axis_decoys),
        "nonrect_reds": int(nonrect_reds),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide rotatable red polyominoes into matching boundary sockets of large structures, painting the traversed route black.

    Input structure:
    - The dominant background surrounds several large non-bg, non-red structures, each containing one socket-shaped cavity cut from a single border.
    - All red components are non-rectangular polyominoes. Some are true matches for a socket after a 90-degree rotation; others are wrong-shape distractors.

    Rule:
    - Extract each structure's full socket shape as the bg component that touches exactly one side of the structure bbox.
    - Among red components on that socket's straight translation axis, keep the one whose shape can be rotated to exactly match the socket.
    - Erase every red component, then redraw only the matched one in the fitted orientation inside the socket; color its original position and intermediate translated placements black.

    Key insight: The target is chosen by 2D shape compatibility under rotation, not by the first red blob seen along a row or column.
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
        for dist, rot_k, rotated, info in _socket_candidates(red_infos, socket):
            if int(info["rid"]) in used_red:
                continue
            used_red.add(int(info["rid"]))
            assignments.append((socket, info, int(dist), int(rot_k), rotated))
            break

    out = g.copy()
    out[out == red] = int(bg)

    for socket, info, _dist, _rot_k, rotated in assignments:
        fr, fc = map(int, socket["top_left"])
        rr0, cc0 = map(int, info["top_left"])
        shift_r = int(fr - rr0)
        shift_c = int(fc - cc0)
        assert shift_r == 0 or shift_c == 0

        for r, c in info["coords"]:
            out[int(r), int(c)] = int(black)

        steps = int(max(abs(shift_r), abs(shift_c)))
        step_dr = 0 if shift_r == 0 else (1 if shift_r > 0 else -1)
        step_dc = 0 if shift_c == 0 else (1 if shift_c > 0 else -1)
        rh, rw = map(int, rotated.shape)

        for step in range(steps):
            top = int(rr0 + step * step_dr)
            left = int(cc0 + step * step_dc)
            for rr in range(rh):
                for cc in range(rw):
                    if int(rotated[rr, cc]) != 1:
                        continue
                    nr = int(top + rr)
                    nc = int(left + cc)
                    if 0 <= nr < h and 0 <= nc < w:
                        out[nr, nc] = int(black)

        for rr in range(rh):
            for cc in range(rw):
                if int(rotated[rr, cc]) != 1:
                    continue
                nr = int(fr + rr)
                nc = int(fc + cc)
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = int(red)

    return out


def _build_structure(shape: np.ndarray, bg: int, sc: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    shape = _trim_shape(shape)
    sh, sw = map(int, shape.shape)
    h = int(sh + 4)
    w = int(sw + 4)
    structure = np.full((h, w), int(sc), dtype=int)
    top = int(h - sh)
    left = int((w - sw) // 2)
    structure[top : top + sh, left : left + sw][shape.astype(bool)] = int(bg)
    return structure, shape.copy(), (int(top), int(left))


def _matched_patches(
    shape_name: str,
    input_rot_k: int,
    decoy_name: str,
    decoy_rot_k: int,
    slide: int,
    decoy_slide: int,
    patch_rot_k: int,
    bg: int,
    sc: int,
) -> dict:
    red = int(Color.RED)
    black = int(Color.BLACK)

    fitted_shape = _MATCH_SHAPES[shape_name]
    input_shape = _rotate_shape(fitted_shape, input_rot_k)
    decoy_shape = _rotate_shape(_MATCH_SHAPES[decoy_name], decoy_rot_k)
    structure, fitted_shape, (slot_top, slot_left) = _build_structure(fitted_shape, bg, sc)

    base_h, base_w = map(int, structure.shape)
    ih, _iw = map(int, input_shape.shape)
    fh, _fw = map(int, fitted_shape.shape)
    dh, _dw = map(int, decoy_shape.shape)
    patch_h = int(max(base_h, slot_top + slide + max(ih, fh, dh)))
    patch_w = int(base_w)

    struct_patch = np.full((patch_h, patch_w), int(bg), dtype=int)
    paste(struct_patch, structure, 0, 0, transparent=bg)
    input_patch = struct_patch.copy()
    output_patch = struct_patch.copy()

    start_top = int(slot_top + slide)
    _paint_shape(input_patch, input_shape, start_top, int(slot_left), int(red))
    _paint_shape(output_patch, input_shape, start_top, int(slot_left), int(black))

    for off in range(1, int(slide) + 1):
        _paint_shape(output_patch, fitted_shape, int(slot_top + off), int(slot_left), int(black))
    _paint_shape(output_patch, fitted_shape, int(slot_top), int(slot_left), int(red))

    decoy_top = int(slot_top + decoy_slide)
    _paint_shape(input_patch, decoy_shape, decoy_top, int(slot_left), int(red))

    occupancy = np.full_like(output_patch, int(bg))
    union_mask = (struct_patch != int(bg)) | (input_patch != int(bg)) | (output_patch != int(bg))
    occupancy[union_mask] = int(sc)

    return {
        "matched": True,
        "shape_name": shape_name,
        "input_rot_k": int(input_rot_k),
        "decoy_name": decoy_name,
        "structure": np.rot90(struct_patch, k=patch_rot_k),
        "input": np.rot90(input_patch, k=patch_rot_k),
        "output": np.rot90(output_patch, k=patch_rot_k),
        "occupancy": np.rot90(occupancy, k=patch_rot_k),
    }


def _unmatched_patches(shape_name: str, patch_rot_k: int, bg: int, sc: int) -> dict:
    structure, _shape, _slot = _build_structure(_MATCH_SHAPES[shape_name], bg, sc)
    return {
        "matched": False,
        "shape_name": shape_name,
        "input_rot_k": 0,
        "decoy_name": None,
        "structure": np.rot90(structure, k=patch_rot_k),
        "input": np.rot90(structure.copy(), k=patch_rot_k),
        "output": np.rot90(structure.copy(), k=patch_rot_k),
        "occupancy": np.rot90(structure.copy(), k=patch_rot_k),
    }


def _place_structures(specs: list[dict], h: int, w: int, bg: int, padding: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    inp = np.full((h, w), int(bg), dtype=int)
    out = np.full((h, w), int(bg), dtype=int)
    struct_grid = np.full((h, w), int(bg), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    for spec in specs:
        loc = _place_patch(out, spec["occupancy"], bg, padding=padding)
        if loc is None:
            return None
        r0, c0 = map(int, loc)
        ph, pw = map(int, spec["occupancy"].shape)
        paste(out, spec["output"], r0, c0, transparent=bg)
        paste(inp, spec["input"], r0, c0, transparent=bg)
        paste(struct_grid, spec["structure"], r0, c0, transparent=bg)
        reserved[r0 : r0 + ph, c0 : c0 + pw] = True

    if np.any((struct_grid != int(bg)) & (inp == int(Color.RED))):
        return None
    return inp, out, struct_grid, reserved


def _place_distractors(inp: np.ndarray, out: np.ndarray, reserved: np.ndarray, bg: int, sc: int, count: int) -> bool:
    h, w = inp.shape
    red_mask = inp == int(Color.RED)
    blocked_core = reserved | (inp != int(bg)) | (out != int(bg))
    noise_names = list(_NOISE_SHAPES.keys())

    for _ in range(count):
        blocked = blocked_core | _dilate4(red_mask)
        canvas = np.full((h, w), int(bg), dtype=int)
        canvas[blocked] = int(sc)
        placed = False
        random.shuffle(noise_names)
        for name in noise_names:
            rots = list(range(4))
            random.shuffle(rots)
            for rot_k in rots:
                shape = _rotate_shape(_NOISE_SHAPES[name], rot_k)
                patch = np.full(shape.shape, int(bg), dtype=int)
                patch[shape == 1] = int(Color.RED)
                loc = _place_patch(canvas, patch, bg, padding=0)
                if loc is None:
                    continue
                rr, cc = map(int, loc)
                paste(inp, patch, rr, cc, transparent=bg)
                red_mask[rr : rr + shape.shape[0], cc : cc + shape.shape[1]] |= shape.astype(bool)
                blocked_core[rr : rr + shape.shape[0], cc : cc + shape.shape[1]] |= shape.astype(bool)
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _sample_canvas_size(n_structs: int, matched: int) -> tuple[int, int]:
    min_h, max_h = GRID_H_RANGE
    min_w, max_w = GRID_W_RANGE

    # Denser branches need larger canvases, otherwise accepted samples collapse
    # toward the easier 2-structure / partially unmatched cases.
    h_floor = min_h + max(0, int(n_structs) - 2) * 2 + (1 if int(matched) == int(n_structs) else 0)
    w_floor = min_w + max(0, int(n_structs) - 2) * 3 + (1 if int(matched) == int(n_structs) else 0)
    h_floor = min(int(max_h), int(h_floor))
    w_floor = min(int(max_w), int(w_floor))
    return random.randint(int(h_floor), int(max_h)), random.randint(int(w_floor), int(max_w))


def _structure_padding(n_structs: int) -> int:
    return max(2, int(STRUCT_PADDING) - max(0, int(n_structs) - 2))


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_COLORS)
    sc = random.choice([c for c in STRUCT_COLORS if c not in (bg, int(Color.RED), int(Color.BLACK))])
    n_structs = random.randint(*STRUCT_COUNT_RANGE)
    unmatched = 1 if random.random() < UNMATCHED_PROB else 0
    matched = int(n_structs - unmatched)
    if matched <= 0 or n_structs > len(_MATCH_SHAPES):
        return None
    h, w = _sample_canvas_size(n_structs, matched)

    chosen_names = random.sample(list(_MATCH_SHAPES.keys()), k=n_structs)
    specs = []
    matched_specs = 0

    for idx, shape_name in enumerate(chosen_names):
        patch_rot_k = random.randrange(4)
        if idx < matched:
            input_rot_choices = [k for k in (1, 2, 3) if _shape_signature(_rotate_shape(_MATCH_SHAPES[shape_name], k)) != _shape_signature(_MATCH_SHAPES[shape_name])]
            if not input_rot_choices:
                return None
            input_rot_k = random.choice(input_rot_choices)
            decoy_choices = [nm for nm in chosen_names if nm != shape_name]
            if not decoy_choices:
                decoy_choices = [nm for nm in _MATCH_SHAPES if _MATCH_FAMILY_SIG[nm] != _MATCH_FAMILY_SIG[shape_name]]
            decoy_name = random.choice(decoy_choices)
            decoy_rot_k = random.randrange(4)
            slide = random.randint(*TRUE_SLIDE_RANGE)
            sh = int(_MATCH_SHAPES[shape_name].shape[0])
            dh = int(_rotate_shape(_MATCH_SHAPES[decoy_name], decoy_rot_k).shape[0])
            low = int(sh)
            high = int(slide - dh - 1)
            if low > high:
                return None
            decoy_slide = random.randint(low, high)
            specs.append(
                _matched_patches(
                    shape_name=shape_name,
                    input_rot_k=input_rot_k,
                    decoy_name=decoy_name,
                    decoy_rot_k=decoy_rot_k,
                    slide=slide,
                    decoy_slide=decoy_slide,
                    patch_rot_k=patch_rot_k,
                    bg=bg,
                    sc=sc,
                )
            )
            matched_specs += 1
        else:
            specs.append(_unmatched_patches(shape_name, patch_rot_k, bg, sc))

    if matched_specs != matched:
        return None

    random.shuffle(specs)
    specs.sort(key=lambda s: np.count_nonzero(s["occupancy"] != int(bg)), reverse=True)
    placed = _place_structures(specs, h, w, bg, padding=_structure_padding(n_structs))
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

    metrics = combo_behavior_metrics(inp)
    total_reds = len(components_mask(inp == int(Color.RED), connectivity=4))
    if int(metrics["matched"]) != int(matched):
        return None
    if int(metrics["ambiguous_sockets"]) != 0:
        return None
    if int(metrics["rotation_needed"]) < 1:
        return None
    if int(metrics["axis_decoys"]) < int(matched):
        return None
    if int(metrics["nonrect_reds"]) != int(total_reds):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 combo D1/1+D2/0")
