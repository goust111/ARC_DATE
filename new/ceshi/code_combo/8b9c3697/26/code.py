from __future__ import annotations

# Combo variant of 8b9c3697 (index 26)
# combo_spec: D3/0+D6/1
# Name: 嵌套插槽分离面板
#
# Design analysis
# - a) 具体化与行为追踪:
#   原题里每个结构只有一层可停靠空腔，红块沿开口方向直线滑到那一层；融合后，每个结构改成
#   2 层或 3 层的嵌套槽，外层只是通道，真正目标是最内层。比如一个向下开口的 3 层结构，
#   红块会从结构下方进入，先穿过外层空腔，再穿过中层空腔，最后停在最内层横梁下方；输出
#   不再回写到整张画布，而是把每个结构的局部结果裁成单独面板并横向拼接。
# - b) 复杂度校准:
#   若所有结构都同深度，解题者只会看到“更厚的结构”；因此每个样例固定放 2 个结构，并强制
#   一个为 2 层、另一个为 3 层，让“不同深度但都要滑到最内层”的规律在单个样例中可对照。
#   分隔带固定为 2 列黑色，避免与局部轨迹混淆。
# - c) 隐式假设审查:
#   原 solver 的关键隐式前提有两个：其一，“bbox 内沿开口方向遇到的第一层 socket 就是目标”；
#   其二，“输出尺寸等于输入尺寸”。D3/0 直接打破前者，所以新 solver 不能再从任意内壁旁的
#   空白格贪心取锚点，而要先识别结构开口方向与层数，再定位规范化后的最内层目标带。D6/1
#   打破后者，因此结果必须先局部作画，再裁剪拼接。
# - d) 放置合法性:
#   生成器仍采用 output-first：先构造每个结构的局部输入/输出补丁，再放回整张输入画布。
#   结构补丁之间保留 padding；干扰红块既不能与已有物体重叠，也不能落在任一最内层目标沿开口
#   方向的可见射线上，否则会破坏唯一匹配。
# - e) 视觉可观察性:
#   人眼可直接看到两个结构的嵌套深度不同，且输出都只保留各自的局部求解结果，并用黑色竖条
#   分开。面板顺序按输入中结构的上到下、左到右顺序，不依赖颜色编号等不可见信息。
# - f) 可视化整洁度:
#   输出面板只包含单个结构、黑色轨迹和最终红块，没有整张输入中的其他干扰红块，因此比原题
#   更清晰；同时结构之间完全拆开，避免多对象覆盖。
# - g) 变化的可推断性:
#   颜色角色保持稳定：背景色不变，结构色不变，红色仍是滑入后的终态，黑色仍是路径，同时兼作
#   面板分隔符。新增信息完全由空间层级与面板框架表达，而不是额外颜色映射。
# - 新增 solver 假设与 generator 保障:
#   solver 假设所有目标结构都符合同一种“嵌套 U/梳齿槽”规范，且开口边是四条边中结构像素最少
#   的那一边；generator 用固定模板保证这一点。solver 还假设匹配红块的尺寸与最内层目标带一致；
#   generator 始终按该尺寸生成匹配红块，并把干扰块排除在目标射线之外。

import random
from collections import defaultdict

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location


PANEL_DIVIDER_W = 2
_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L


def _crop_to_content(patch: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(patch != int(bg))
    assert coords.size, "expected non-empty patch when cropping panel"
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


def _sign(x: int) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _edge_counts(mask: np.ndarray) -> list[int]:
    return [
        int(np.count_nonzero(mask[0, :])),
        int(np.count_nonzero(mask[:, -1])),
        int(np.count_nonzero(mask[-1, :])),
        int(np.count_nonzero(mask[:, 0])),
    ]


def _target_dims_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _analyze_structure(g: np.ndarray, coords: np.ndarray, sid: int, bg: int) -> dict:
    pts = {(int(r), int(c)) for r, c in coords}
    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in pts:
        mask[int(r - r0), int(c - c0)] = True

    edge_counts = _edge_counts(mask)
    opening_dir = min(range(4), key=lambda d: edge_counts[d])
    assert edge_counts.count(edge_counts[opening_dir]) == 1, "opening side must be unique"

    rot_k = int((opening_dir - 2) % 4)
    mask_rot = np.rot90(mask, k=rot_k)
    rh, rw = map(int, mask_rot.shape)

    layers = int((rh - 2) // 2)
    inner_w = int(rw - 4 * layers + 2)
    assert layers in (2, 3), "expected 2 or 3 nested layers"
    assert rh == 2 * layers + 2, "unexpected normalized structure height"
    assert inner_w in (1, 2, 3), "unexpected innermost slot width"

    target_row = int(layers + 2)
    target_c0 = int(2 * layers - 1)
    target_rot = np.zeros_like(mask_rot, dtype=bool)
    target_rot[target_row, target_c0 : target_c0 + inner_w] = True
    target_local = np.rot90(target_rot, k=(4 - rot_k) % 4)

    target_rel = [(int(r), int(c)) for r, c in np.argwhere(target_local)]
    target_global = [(int(r0 + rr), int(c0 + cc)) for rr, cc in target_rel]
    target_rows = [r for r, _ in target_global]
    target_cols = [c for _, c in target_global]
    target_bbox = (min(target_rows), min(target_cols), max(target_rows), max(target_cols))

    return {
        "sid": int(sid),
        "coords": pts,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "opening_dir": int(opening_dir),
        "layers": int(layers),
        "inner_w": int(inner_w),
        "target_cells": target_global,
        "target_bbox": tuple(map(int, target_bbox)),
    }


def _shape_matches(red_bbox: tuple[int, int, int, int], target_bbox: tuple[int, int, int, int]) -> bool:
    return _target_dims_from_bbox(red_bbox) == _target_dims_from_bbox(target_bbox)


def _discover_scene(
    g: np.ndarray,
) -> tuple[
    int,
    list[list[tuple[int, int]]],
    list[tuple[int, int, int, int]],
    list[dict],
    dict[int, tuple[int, int]],
]:
    bg = int(get_bg(g))
    red = int(Color.RED)
    H, W = g.shape

    assert red in set(map(int, np.unique(g)))

    red_comps = components_mask(g == red, connectivity=4)
    assert red_comps

    red_coords: list[list[tuple[int, int]]] = []
    red_bbox: list[tuple[int, int, int, int]] = []
    red_at: dict[tuple[int, int], int] = {}
    for rid, comp in enumerate(red_comps):
        pts = [(int(r), int(c)) for r, c in comp]
        red_coords.append(pts)
        red_bbox.append(tuple(map(int, bbox_coords(comp))))
        for rc in pts:
            red_at[rc] = int(rid)

    struct_mask = (g != int(bg)) & (g != int(red))
    struct_comps = components_mask(struct_mask, connectivity=4)
    assert struct_comps, "expected at least one non-bg, non-red structure"

    structures = [_analyze_structure(g, coords, sid, bg) for sid, coords in enumerate(struct_comps)]

    candidates: list[tuple[tuple[int, int], int, int]] = []
    for info in structures:
        hits: dict[int, list[int]] = defaultdict(lambda: [0, 10**9])
        dr, dc = _DIRS[int(info["opening_dir"])]
        for r, c in info["target_cells"]:
            rr, cc = int(r + dr), int(c + dc)
            dist = 1
            while 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(bg):
                rr += dr
                cc += dc
                dist += 1
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(red):
                rid = int(red_at[(int(rr), int(cc))])
                if not _shape_matches(red_bbox[rid], info["target_bbox"]):
                    continue
                rec = hits[rid]
                rec[0] += 1
                rec[1] = min(int(rec[1]), int(dist))
        if hits:
            rid_best = max(hits.keys(), key=lambda rid: (int(hits[rid][0]), -int(hits[rid][1])))
            cnt, mind = hits[rid_best]
            candidates.append(((int(cnt), -int(mind)), int(rid_best), int(info["sid"])))

    candidates.sort(reverse=True, key=lambda item: item[0])
    used_red: set[int] = set()
    assignments: dict[int, tuple[int, int]] = {}
    for _score, rid, sid in candidates:
        if int(rid) in used_red:
            continue
        used_red.add(int(rid))
        assignments[int(sid)] = (int(rid), int(sid))

    return bg, red_coords, red_bbox, structures, assignments


def _structure_panel(
    g: np.ndarray,
    bg: int,
    info: dict,
    red_coords: list[list[tuple[int, int]]],
    red_bbox: list[tuple[int, int, int, int]],
    assignments: dict[int, tuple[int, int]],
) -> np.ndarray:
    red = int(Color.RED)
    black = int(Color.BLACK)

    cells: dict[tuple[int, int], int] = {}
    for r, c in info["coords"]:
        cells[(int(r), int(c))] = int(g[r, c])

    sid = int(info["sid"])
    if sid in assignments:
        rid, _ = assignments[sid]
        coords = red_coords[int(rid)]
        src_r0, src_c0, _src_r1, _src_c1 = map(int, red_bbox[int(rid)])
        dst_r0, dst_c0, _dst_r1, _dst_c1 = map(int, info["target_bbox"])

        shift_r = int(dst_r0 - src_r0)
        shift_c = int(dst_c0 - src_c0)
        steps = int(max(abs(shift_r), abs(shift_c)))
        step_dr = _sign(shift_r)
        step_dc = _sign(shift_c)

        for step in range(steps):
            off_r = int(step * step_dr)
            off_c = int(step * step_dc)
            for r, c in coords:
                cells[(int(r + off_r), int(c + off_c))] = int(black)

        for r, c in coords:
            cells[(int(r + shift_r), int(c + shift_c))] = int(red)

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    panel = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for (r, c), color in cells.items():
        panel[int(r - r0), int(c - c0)] = int(color)
    return panel


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each matched red bar into the innermost nested slot, then export one panel per structure.

    Input structure:
    - A large background canvas contains two large non-background nested slot structures, each opening in
      one cardinal direction and containing either 2 or 3 embedded cavity layers.
    - Red rectangles sit outside the openings; only the red bar aligned with a structure's innermost slot
      is relevant, while other red rectangles are distractors.

    Rule:
    - For each structure, identify its opening side and normalize its nested geometry to recover the deepest
      target band inside the smallest cavity.
    - Follow rays from that innermost band outward through the opening to find the matched red component;
      slide that component straight inward, painting every intermediate position black. Unmatched reds vanish.
    - After solving each structure locally, crop the local result tightly and concatenate the panels from
      top-to-bottom / left-to-right structure order, separating adjacent panels with a black divider strip.

    Key insight: The visible outer cavity is not the destination. The red bar must be matched against the
    deepest nested slot, and the final answer is assembled from per-structure solved panels rather than the
    original whole canvas.
    """
    g = np.array(grid, dtype=int)
    bg, red_coords, red_bbox, structures, assignments = _discover_scene(g)

    order = sorted(range(len(structures)), key=lambda sid: (int(structures[sid]["bbox"][0]), int(structures[sid]["bbox"][1])))
    panels = [_structure_panel(g, bg, structures[sid], red_coords, red_bbox, assignments) for sid in order]
    return _concat_panels(panels, bg, int(Color.BLACK))


# --------------- generator (codex fused D3/0 + D6/1 variant) ---------------

# ── CFG: canvas & output framing ────────────────────────────────────
GRID_H_RANGE = (20, 24)         # input grid height
GRID_W_RANGE = (28, 34)         # input grid width
PANEL_COUNT = 2                 # fixed to expose one 2-layer and one 3-layer structure
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

# ── CFG: nested structures ──────────────────────────────────────────
LAYER_PATTERN = (2, 3)          # both depths appear in every example
INNER_WIDTH_RANGE = (1, 3)      # width of the innermost slot
SLIDE_RANGE = (4, 8)
ROTATIONS = (0, 1, 2, 3)
STRUCT_PADDING = 4

# ── CFG: distractors ────────────────────────────────────────────────
DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
DISTRACTOR_COUNT_RANGE = (3, 6)

# ── CFG: retry ──────────────────────────────────────────────────────
MAX_RETRIES = 400


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _canonical_structure(layers: int, inner_w: int, bg: int, sc: int) -> np.ndarray:
    h = int(2 * layers + 2)
    w = int(inner_w + 4 * layers - 2)
    g = np.full((h, w), int(bg), dtype=int)

    for k in range(layers):
        top = int(2 * k)
        if k == 0:
            span_l, span_r = 0, w - 1
        else:
            span_l = int(2 * (k - 1))
            span_r = int(w - 1 - 2 * (k - 1))
        g[top, span_l : span_r + 1] = int(sc)

        left = int(2 * k)
        right = int(w - 1 - 2 * k)
        bottom = int(2 * layers + 1 - k)
        g[top + 1 : bottom + 1, left] = int(sc)
        g[top + 1 : bottom + 1, right] = int(sc)

    return g


def _canonical_target_mask(structure: np.ndarray, layers: int, inner_w: int) -> np.ndarray:
    mask = np.zeros_like(structure, dtype=bool)
    row = int(layers + 2)
    c0 = int(2 * layers - 1)
    mask[row, c0 : c0 + inner_w] = True
    return mask


def _matched_spec(layers: int, inner_w: int, slide: int, rot_k: int, bg: int, sc: int) -> dict:
    red = int(Color.RED)
    black = int(Color.BLACK)

    structure = _canonical_structure(layers, inner_w, bg, sc)
    target_mask = _canonical_target_mask(structure, layers, inner_w)
    target_cells = np.argwhere(target_mask)
    target_row = int(target_cells[0, 0])
    target_c0 = int(target_cells[:, 1].min())

    sh, sw = map(int, structure.shape)
    patch_h = max(int(sh), int(target_row + slide + 1))
    patch = np.full((patch_h, sw), int(bg), dtype=int)
    paste(patch, structure, 0, 0, transparent=bg)

    input_patch = patch.copy()
    output_patch = patch.copy()
    input_patch[target_row + slide, target_c0 : target_c0 + inner_w] = int(red)
    output_patch[target_row + 1 : target_row + slide + 1, target_c0 : target_c0 + inner_w] = int(black)
    output_patch[target_row, target_c0 : target_c0 + inner_w] = int(red)

    target_patch = np.zeros_like(patch, dtype=bool)
    target_patch[:sh, :sw] = target_mask

    structure_rot = np.rot90(patch, k=rot_k)
    input_rot = np.rot90(input_patch, k=rot_k)
    output_rot = np.rot90(output_patch, k=rot_k)
    target_rot = np.rot90(target_patch, k=rot_k)

    return {
        "structure": structure_rot,
        "input": input_rot,
        "panel": _crop_to_content(output_rot, bg),
        "target_mask": target_rot,
        "opening_dir": int((2 - rot_k) % 4),
    }


def _place_patch(canvas: np.ndarray, patch: np.ndarray, bg: int, padding: int) -> tuple[int, int] | None:
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


def _place_structures(specs: list[dict], h: int, w: int, bg: int, sc: int) -> tuple[np.ndarray, list[dict]] | None:
    inp = np.full((h, w), int(bg), dtype=int)
    placed: list[dict] = []

    for spec in specs:
        loc = _place_patch(inp, spec["input"], bg, padding=STRUCT_PADDING)
        if loc is None:
            return None
        r0, c0 = map(int, loc)
        paste(inp, spec["input"], r0, c0, transparent=bg)

        struct_coords_arr = np.argwhere(spec["structure"] == int(sc))
        struct_coords = [(int(r0 + rr), int(c0 + cc)) for rr, cc in struct_coords_arr]
        rows = [r for r, _ in struct_coords]
        cols = [c for _, c in struct_coords]

        target_arr = np.argwhere(spec["target_mask"])
        target_cells = [(int(r0 + rr), int(c0 + cc)) for rr, cc in target_arr]

        placed.append(
            {
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "target_cells": target_cells,
                "panel": spec["panel"],
                "opening_dir": int(spec["opening_dir"]),
            }
        )

    return inp, placed


def _structure_rays(inp: np.ndarray, placed: list[dict], bg: int) -> np.ndarray:
    h, w = inp.shape
    forbidden = np.zeros((h, w), dtype=bool)
    for item in placed:
        dr, dc = _DIRS[int(item["opening_dir"])]
        for r, c in item["target_cells"]:
            rr, cc = int(r + dr), int(c + dc)
            while 0 <= rr < h and 0 <= cc < w and int(inp[rr, cc]) == int(bg):
                forbidden[rr, cc] = True
                rr += dr
                cc += dc
    return forbidden


def _place_distractors(inp: np.ndarray, placed: list[dict], bg: int, sc: int, count: int) -> bool:
    h, w = inp.shape
    forbidden = _structure_rays(inp, placed, bg)
    bbox_blocked = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = map(int, item["bbox"])
        bbox_blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    red_mask = inp == int(Color.RED)
    shapes = list(DISTRACTOR_SHAPES)

    for _ in range(count):
        blocked = forbidden | bbox_blocked | (inp != int(bg)) | _dilate4(red_mask)
        canvas = np.full((h, w), int(bg), dtype=int)
        canvas[blocked] = int(sc)
        random.shuffle(shapes)

        placed_one = False
        for rh, rw in shapes:
            patch = np.full((rh, rw), int(Color.RED), dtype=int)
            loc = _place_patch(canvas, patch, bg, padding=0)
            if loc is None:
                continue
            rr, cc = map(int, loc)
            paste(inp, patch, rr, cc, transparent=bg)
            red_mask[rr : rr + rh, cc : cc + rw] = True
            placed_one = True
            break
        if not placed_one:
            return False
    return True


def _expected_output(placed: list[dict], bg: int) -> np.ndarray:
    order = sorted(placed, key=lambda item: (int(item["bbox"][0]), int(item["bbox"][1])))
    panels = [item["panel"] for item in order]
    return _concat_panels(panels, bg, int(Color.BLACK))


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_COLORS)
    sc = random.choice([c for c in STRUCT_COLORS if c not in (bg, int(Color.RED), int(Color.BLACK))])
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    distractor_count = random.randint(*DISTRACTOR_COUNT_RANGE)

    layer_choices = list(LAYER_PATTERN)
    random.shuffle(layer_choices)

    specs = []
    for layers in layer_choices[:PANEL_COUNT]:
        inner_w = random.randint(*INNER_WIDTH_RANGE)
        slide = random.randint(*SLIDE_RANGE)
        rot_k = random.choice(ROTATIONS)
        specs.append(_matched_spec(layers, inner_w, slide, rot_k, bg, sc))

    random.shuffle(specs)
    specs.sort(key=lambda spec: np.count_nonzero(spec["input"] != int(bg)), reverse=True)

    placed_result = _place_structures(specs, h, w, bg, sc)
    if placed_result is None:
        return None
    inp, placed = placed_result

    if not _place_distractors(inp, placed, bg, sc, distractor_count):
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
    raise RuntimeError("failed to generate a valid input for 8b9c3697 combo variant 26")
