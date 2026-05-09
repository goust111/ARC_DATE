# Combo variant of 898e7135 (index 6)
# combo_spec: D1/0+D7/1
# Name: 颜色复制填充

from __future__ import annotations

"""
Design analysis for the combo variant
=====================================

spec_variant:
  preserved_core:
    - 背景仍为 0，最大前景连通块仍是包围若干背景洞的 frame。
    - 洞对应的外部精灵仍是按 k x k 放大的单色 polyomino，输出仍是裁剪后的放大 frame。
    - k 仍由最小的实心 k x k 方块精灵推断，因此样例里继续保留 unit-slot 类。
  fused_changes:
    - D1/0: 洞与精灵的身份不再靠跨精灵形状匹配，而是靠“洞边墙色 == 精灵颜色”。
    - D7/1: 一个物理精灵可被重复使用，填入多个同色洞。
    - 融合后不只是“颜色先验 + 形状复用”的并列叠加；生成器还强制出现“同形不同色”的洞组，
      让 solver 必须先读颜色，再对同色洞复用对应精灵。

behavior_trace:
  local_example:
    - 设有两个红色 L 洞、一个绿色 L 洞、一个黄色 unit 洞；框外只有红 L、绿 L、黄方块三块精灵。
    - 原题逻辑会逐洞做形状配对，并且每块精灵只能使用一次；第二个红 L 洞会因为红 L 已被消费而失败。
    - 只做 D7/1 而不做颜色匹配时，红/绿两个 L 精灵又会因同形而互相歧义。
    - 融合后：solver 先从洞四邻墙色读出“这个洞要红/绿/黄哪块精灵”，再在该颜色精灵内部尝试 D4 朝向，
      所以红 L 可复制到两个红洞，绿色 L 只填绿色 L 洞。
  visible_difference:
    - 输入中能同时看到“同色墙标记 + 同色外部精灵”以及“洞数量多于精灵数量”。
    - 输出中同一颜色的块会在多个洞位置重复出现，而不同颜色的同形洞会落入不同颜色。

complexity_calibration:
  - 每个样例都至少包含一个重复非 unit 颜色类，保证 1:many 可见，且 `slot_count > piece_count`。
  - 每个样例都至少包含一组“同 canonical shape、不同颜色”的非 unit 洞，保证颜色匹配不可省略。
  - 颜色种类控制在 3-4 个精灵色，加 1 个 frame 基础墙色与 1 个干扰色；人眼能稳定追踪，不会过杂。
  - 仍保留随机旋转/翻转，所以颜色只解决“身份”，姿态恢复仍需理解块级缩放和 D4。

implicit_assumption_audit:
  - 原 main 把 frame 当作单色连通块，并用主色直接作为输出背景；这里不再成立。
    新 solver 从 frame 组件颜色里减去所有 piece 颜色，剩下的专属颜色才是基础墙色。
  - 原 main 的 `used_piece` 假设 1:1，对融合变体失效；同色 piece 现在可以被多个洞重复使用。
  - 原 main 依赖“形状决定身份”；这里改成“颜色决定身份，形状只恢复朝向”，
    因此 generator 必须保证每个洞四邻墙色与某个 piece 颜色唯一相交。
  - 单独的 D7/1 solver 常假设每个 shape class 至多一块物理精灵；融合后也不成立，
    因为这里故意允许两个不同颜色的精灵同形，靠颜色区分。

placement_legality:
  - 洞之间继续保持 8 分离，并额外禁止不同洞共享四邻墙像素，避免一个洞邻接多个标记色。
  - frame、外部 pieces、单像素干扰点都保持不重叠，外部 piece 与 frame/彼此之间至少隔 1 格背景。
  - 生成器继续要求洞布局 tight-crop 后触及四条边，使 solver 的洞裁剪逻辑稳定。

visual_observability:
  - 输入中同形洞会以两种颜色出现，且至少一种颜色会重复 2-3 次；这让“颜色配对 + 复制”一眼可见。
  - 标记色只画在每个洞的四邻墙环上，层次简单，不与洞本体和外部精灵叠色。
  - 干扰物仍是独立单像素颜色，不会与任何 piece/frame 颜色混用。

inference_cleanliness:
  - 不使用颜色值大小、扫描顺序等不可视因素定义映射；颜色仅作为可见类别标签。
  - 若解题者只看 2-3 个样例，也能看到：
      1. 同形但不同色的洞要去拿同形不同色的不同精灵。
      2. 同色洞数量可能大于对应物理精灵数量，因此那块精灵会被复制填多次。
  - 这两个规律在同一组样例里同时出现，避免退化成“只是形状题”或“只是计数题”。

new_solver_assumptions_and_generator_guarantees:
  - solver 假设每种 piece 颜色恰对应一块物理精灵；generator 保证 piece 颜色互异。
  - solver 假设每个洞的相邻墙色与 piece 颜色交集恰为 1；generator 用独占墙环保证这一点。
  - solver 假设 piece 的某个 D4 变体能按 k 下采样回洞形；generator 仍由洞形直接放大再随机 D4 得到 piece。
  - solver 假设存在至少一个 unit 类以推断 k；generator 固定保留一个 unit piece。
"""

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import bbox, bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _canonical_shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    variants: list[tuple[int, int, bytes]] = []
    for kk in range(4):
        rot = np.rot90(mask, kk)
        variants.append((int(rot.shape[0]), int(rot.shape[1]), rot.tobytes()))
        flip = np.fliplr(rot)
        variants.append((int(flip.shape[0]), int(flip.shape[1]), flip.tobytes()))
    return min(variants)


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = map(int, mask.shape)
    if hh % int(k) != 0 or ww % int(k) != 0:
        return None
    out = np.zeros((hh // int(k), ww // int(k)), dtype=bool)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            block = mask[r * int(k) : (r + 1) * int(k), c * int(k) : (c + 1) * int(k)]
            if bool(block.all()):
                out[r, c] = True
            elif bool(block.any()):
                return None
    return out


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    raw: list[tuple[np.ndarray, np.ndarray]] = []
    for kk in range(4):
        rot_patch = np.rot90(patch, kk)
        rot_mask = np.rot90(mask, kk)
        raw.append((rot_patch, rot_mask))
        raw.append((np.fliplr(rot_patch), np.fliplr(rot_mask)))
    uniq: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[int, int, bytes]] = set()
    for patch_v, mask_v in raw:
        key = (int(mask_v.shape[0]), int(mask_v.shape[1]), mask_v.tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((patch_v, mask_v))
    return uniq


def _dihedral_masks(mask: np.ndarray) -> list[np.ndarray]:
    dummy = np.zeros(mask.shape, dtype=int)
    return [mv for _, mv in _dihedral_variants(dummy, mask)]


def _tight_mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, int, int]:
    rr0, cc0, rr1, cc1 = bbox_coords(coords)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
    tight = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
    tight[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
    return tight, int(rr0), int(cc0)


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain foreground components"
    frame_comp = max(comps, key=lambda d: int(d["area"]))

    pieces: list[dict] = []
    for comp in comps:
        if comp is frame_comp:
            continue
        coords = comp["coords"]
        if coords.size == 0 or int(coords.shape[0]) <= 1:
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        mask = np.zeros(patch.shape, dtype=bool)
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append(
            {
                "color": int(g[int(coords[0, 0]), int(coords[0, 1])]),
                "area": int(mask.sum()),
                "mask": mask,
                "patch": patch,
            }
        )
    assert pieces, "at least one non-frame piece must exist"

    piece_colors = {int(p["color"]) for p in pieces}
    assert len(piece_colors) == len(pieces), "piece colors must uniquely identify physical pieces"

    k = None
    for piece in pieces:
        h, w = map(int, piece["mask"].shape)
        if h == w and int(piece["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    k = int(k or 1)

    frame_color_counts = Counter(int(g[int(r), int(c)]) for r, c in frame_comp["coords"])
    base_candidates = [color for color in frame_color_counts if int(color) not in piece_colors]
    assert base_candidates, "frame must contain a dedicated base wall color"
    frame_color = max(base_candidates, key=lambda color: int(frame_color_counts[int(color)]))

    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])
    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == int(bg)
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots: list[dict] = []
    for coords in components_mask(layout, connectivity=4):
        mask, rr0, cc0 = _tight_mask_from_coords(coords)
        adj_colors = set()
        for r, c in coords:
            sr = int(lr0 + r)
            sc = int(lc0 + c)
            for dr, dc in DIR4:
                nr = int(sr + dr)
                nc = int(sc + dc)
                if 0 <= nr < sub.shape[0] and 0 <= nc < sub.shape[1]:
                    val = int(sub[nr, nc])
                    if val != int(bg):
                        adj_colors.add(val)
        slot_piece_colors = adj_colors & piece_colors
        assert len(slot_piece_colors) == 1, "each slot must expose exactly one piece color on its wall ring"
        slots.append(
            {
                "area": int(mask.sum()),
                "r0": int(rr0),
                "c0": int(cc0),
                "mask": mask,
                "color": int(next(iter(slot_piece_colors))),
                "shape_key": _canonical_shape_key(mask),
            }
        )
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"]), int(s["color"])))
    assert slots, "frame must contain at least one slot"

    return {
        "grid": g,
        "bg": int(bg),
        "frame_color": int(frame_color),
        "layout": layout,
        "slots": slots,
        "pieces": pieces,
        "piece_colors": piece_colors,
        "k": int(k),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate the frame layout and fill each hole with the piece indicated by its wall color.

    Input structure:
    - Background is the most common color. The largest foreground component is a multicolor frame enclosing bg holes.
    - Outside the frame, each physical piece is a monochrome k-by-k-block sprite; some hole colors may appear more often than there are physical pieces.

    Rule:
    - Tight-crop the hole layout inside the frame and read each slot's marker color from the wall pixels touching that hole.
    - Match the slot to the unique outside piece of the same color, then rotate/flip that piece until its block-level mask fits the slot.
    - Paste that oriented piece into every slot of its color, allowing one physical piece to be reused multiple times.

    Key insight:
    - Color determines which piece class a hole wants, while geometry only resolves orientation after the color match; repeated hole colors mean copying the same sprite into several locations.
    """
    scene = _parse_scene(np.array(grid, dtype=int))
    layout = scene["layout"]
    slots = scene["slots"]
    pieces = scene["pieces"]
    frame_color = int(scene["frame_color"])
    k = int(scene["k"])

    piece_by_color = {int(piece["color"]): piece for piece in pieces}
    assert len(piece_by_color) == len(pieces), "piece colors must stay unique"

    placements: list[tuple[dict, dict]] = []
    for slot in slots:
        color = int(slot["color"])
        piece = piece_by_color.get(color)
        assert piece is not None, "every slot color must have a matching piece"
        need_area = int(slot["area"]) * int(k) * int(k)
        assert int(piece["area"]) == int(need_area), "piece area must equal the slot area after scaling"

        found = None
        for patch_v, mask_v in _dihedral_variants(piece["patch"], piece["mask"]):
            down = _downsample_blocks(mask_v, k)
            if down is None:
                continue
            if down.shape != slot["mask"].shape:
                continue
            if np.array_equal(down, slot["mask"]):
                found = {"patch": patch_v, "mask": mask_v}
                break
        assert found is not None, "matched piece must admit a D4 orientation that fits the slot"
        placements.append((slot, found))

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)

    for slot, placement in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, placement["patch"], int(top), int(left), mask=placement["mask"], overwrite=True)

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)
REPEAT_MULT_CHOICES = [2, 3]
UNIQUE_NON_UNIT_CHOICES = [0, 1]
SHARED_SIZE_CHOICES = [2, 3, 4, 5, 6]
UNIQUE_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]

HOLE_DENSITY_RANGE = (0.20, 0.50)
PLACE_RETRIES = 80
SHAPE_RETRIES = 50

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 400


def _random_polyomino(target_size: int) -> np.ndarray:
    chosen = {(0, 0)}
    frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
    while len(chosen) < int(target_size):
        nr, nc = random.choice(tuple(frontier))
        frontier.discard((nr, nc))
        chosen.add((int(nr), int(nc)))
        for dr, dc in DIR4:
            nxt = (int(nr + dr), int(nc + dc))
            if nxt not in chosen:
                frontier.add(nxt)
    coords = np.array(sorted(chosen), dtype=int)
    r0, c0 = coords.min(axis=0)
    coords[:, 0] -= int(r0)
    coords[:, 1] -= int(c0)
    r1, c1 = coords.max(axis=0)
    mask = np.zeros((int(r1 + 1), int(c1 + 1)), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _gen_shape_mask(
    target_size: int,
    banned_keys: set[tuple[int, int, bytes]],
    *,
    require_multi_variant: bool,
) -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        mask = _random_polyomino(int(target_size))
        key = _canonical_shape_key(mask)
        if key in banned_keys:
            continue
        if require_multi_variant and len(_dihedral_masks(mask)) < 2:
            continue
        return mask
    return None


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[:-1, 1:] |= mask[1:, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    return out


def _forbid_coords(forbidden: np.ndarray, coords: np.ndarray) -> None:
    cell_mask = np.zeros_like(forbidden)
    cell_mask[coords[:, 0], coords[:, 1]] = True
    forbidden |= _dilate8(cell_mask)


def _wall_neighbors_layout(coords: np.ndarray, h: int, w: int) -> np.ndarray:
    occupied = {tuple(map(int, rc)) for rc in coords.tolist()}
    walls = set()
    for r, c in occupied:
        for dr, dc in DIR4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in occupied:
                walls.add((nr, nc))
    if not walls:
        return np.zeros((0, 2), dtype=int)
    return np.array(sorted(walls), dtype=int)


def _place_mask(
    base_mask: np.ndarray,
    layout_h: int,
    layout_w: int,
    forbidden: np.ndarray,
    claimed_walls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    variants = _dihedral_masks(base_mask)
    random.shuffle(variants)
    for variant in variants:
        pts = np.argwhere(variant)
        vh, vw = map(int, variant.shape)
        positions = [(r0, c0) for r0 in range(layout_h - vh + 1) for c0 in range(layout_w - vw + 1)]
        random.shuffle(positions)
        for r0, c0 in positions[:PLACE_RETRIES]:
            coords = pts + np.array([[int(r0), int(c0)]], dtype=int)
            if bool(forbidden[coords[:, 0], coords[:, 1]].any()):
                continue
            walls = _wall_neighbors_layout(coords, layout_h, layout_w)
            if walls.size and bool(claimed_walls[walls[:, 0], walls[:, 1]].any()):
                continue
            return coords, variant.copy()
    return None


def _gen_layout(layout_h: int, layout_w: int, plans: list[dict]) -> tuple[list[dict], np.ndarray] | None:
    forbidden = np.zeros((layout_h, layout_w), dtype=bool)
    claimed_walls = np.zeros((layout_h, layout_w), dtype=bool)

    ordered = list(plans)
    random.shuffle(ordered)
    ordered.sort(key=lambda item: -int(item["base_mask"].sum()))

    slots: list[dict] = []
    for plan in ordered:
        placed = _place_mask(plan["base_mask"], layout_h, layout_w, forbidden, claimed_walls)
        if placed is None:
            return None
        coords, actual_mask = placed
        walls = _wall_neighbors_layout(coords, layout_h, layout_w)
        if walls.size:
            claimed_walls[walls[:, 0], walls[:, 1]] = True
        _forbid_coords(forbidden, coords)
        slots.append(
            {
                "class_id": str(plan["class_id"]),
                "color": int(plan["color"]),
                "coords": coords,
                "area": int(coords.shape[0]),
                "mask": actual_mask,
                "shape_key": _canonical_shape_key(actual_mask),
            }
        )

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for slot in slots:
        layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True

    density = float(layout_mask.sum()) / float(layout_h * layout_w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (float(lo) <= density <= float(hi)):
        return None
    if not (bool(layout_mask[0].any()) and bool(layout_mask[-1].any()) and bool(layout_mask[:, 0].any()) and bool(layout_mask[:, -1].any())):
        return None

    color_counts = Counter(int(slot["color"]) for slot in slots)
    if max(color_counts.values()) < 2:
        return None

    shape_colors: dict[tuple[int, int, bytes], set[int]] = defaultdict(set)
    for slot in slots:
        shape_colors[slot["shape_key"]].add(int(slot["color"]))
    if max(len(colors) for colors in shape_colors.values()) < 2:
        return None

    return slots, layout_mask


def _make_piece(base_mask: np.ndarray, color: int, k: int, *, apply_dihedral: bool) -> dict:
    if int(base_mask.sum()) == 1:
        patch = np.full((k, k), int(color), dtype=int)
        mask = np.ones((k, k), dtype=bool)
        return {"color": int(color), "in_patch": patch, "in_mask": mask}

    out_mask = np.kron(base_mask, np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, int(color), 0).astype(int)
    if not apply_dihedral:
        return {"color": int(color), "in_patch": out_patch, "in_mask": out_mask}

    patch_v, mask_v = random.choice(_dihedral_variants(out_patch, out_mask))
    return {"color": int(color), "in_patch": patch_v, "in_mask": mask_v}


def _build_expected_output(slots: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    scale = np.ones((k, k), dtype=bool)
    for slot in slots:
        slot_mask = np.kron(slot["mask"], scale)
        slot_patch = np.where(slot_mask, int(slot["color"]), 0).astype(int)
        rr0, cc0 = slot["coords"].min(axis=0)
        top = (int(rr0) + 1) * int(k)
        left = (int(cc0) + 1) * int(k)
        paste(out, slot_patch, int(top), int(left), mask=slot_mask, overwrite=True)
    return out


def _slot_wall_neighbors_frame(slot: dict, layout_mask: np.ndarray, layout_h: int, layout_w: int) -> list[tuple[int, int]]:
    walls = set()
    for r, c in slot["coords"]:
        rr = int(r + 1)
        cc = int(c + 1)
        for dr, dc in DIR4:
            nr = int(rr + dr)
            nc = int(cc + dc)
            if not (0 <= nr < layout_h + 2 and 0 <= nc < layout_w + 2):
                continue
            if 1 <= nr <= layout_h and 1 <= nc <= layout_w and bool(layout_mask[int(nr - 1), int(nc - 1)]):
                continue
            walls.add((nr, nc))
    return sorted(walls)


def _build_input_canvas(
    pieces: list[dict],
    slots: list[dict],
    layout_mask: np.ndarray,
    layout_h: int,
    layout_w: int,
    frame_color: int,
    distractor_color: int,
) -> np.ndarray | None:
    frame_h, frame_w = layout_h + 2, layout_w + 2
    total_piece_area = sum(int(piece["in_mask"].sum()) for piece in pieces)
    needed_cells = frame_h * frame_w + AREA_BUFFER_MULT * total_piece_area + AREA_BUFFER_CONST

    min_h = max(CANVAS_DIM_RANGE[0], frame_h + 2)
    min_w = max(CANVAS_DIM_RANGE[0], frame_w + 2)
    canvas_h = random.randint(min_h, CANVAS_DIM_RANGE[1])
    canvas_w = random.randint(min_w, CANVAS_DIM_RANGE[1])
    while canvas_h * canvas_w < needed_cells and (canvas_h < CANVAS_DIM_RANGE[1] or canvas_w < CANVAS_DIM_RANGE[1]):
        if canvas_h <= canvas_w and canvas_h < CANVAS_DIM_RANGE[1]:
            canvas_h += 1
        elif canvas_w < CANVAS_DIM_RANGE[1]:
            canvas_w += 1
        else:
            break
    if canvas_h * canvas_w < needed_cells:
        return None

    canvas = np.zeros((canvas_h, canvas_w), dtype=int)

    frame_img = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    frame_img[1:-1, 1:-1] = int(frame_color)
    frame_img[1:-1, 1:-1][layout_mask] = 0
    for slot in slots:
        for wr, wc in _slot_wall_neighbors_frame(slot, layout_mask, layout_h, layout_w):
            frame_img[int(wr), int(wc)] = int(slot["color"])

    fr_max = canvas_h - frame_h - 1
    fc_max = canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    for piece in sorted(pieces, key=lambda item: -int(item["in_mask"].sum())):
        try:
            pr, pc = random_free_location(
                canvas,
                piece["in_patch"],
                transparent=0,
                border=0,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, piece["in_patch"], int(pr), int(pc), mask=piece["in_mask"], overwrite=True)

    n_distractors = random.randint(*DISTRACTOR_RANGE)
    placed = 0
    for _ in range(n_distractors * DISTRACTOR_RETRY_MULT):
        if placed >= n_distractors:
            break
        rr = random.randint(0, canvas_h - 1)
        cc = random.randint(0, canvas_w - 1)
        if int(canvas[rr, cc]) != 0:
            continue
        r_lo, r_hi = max(0, rr - 1), min(canvas_h, rr + 2)
        c_lo, c_hi = max(0, cc - 1), min(canvas_w, cc + 2)
        if bool(np.any(canvas[r_lo:r_hi, c_lo:c_hi] != 0)):
            continue
        canvas[rr, cc] = int(distractor_color)
        placed += 1
    if placed < MIN_DISTRACTORS:
        return None

    vals, counts = np.unique(canvas, return_counts=True)
    if int(vals[np.argmax(counts)]) != 0:
        return None

    return canvas


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    repeat_mult = random.choice(REPEAT_MULT_CHOICES)
    unique_non_unit = random.choice(UNIQUE_NON_UNIT_CHOICES)

    slot_count = int(repeat_mult + 1 + unique_non_unit + 1)  # repeated color + twin color + unique + unit
    if not (3 <= slot_count <= 6):
        return None

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = int(palette.pop())
    distractor_color = int(palette.pop())
    piece_color_count = 3 + int(unique_non_unit)
    if len(palette) < piece_color_count:
        return None

    shared_mask = _gen_shape_mask(
        random.choice(SHARED_SIZE_CHOICES),
        banned_keys=set(),
        require_multi_variant=True,
    )
    if shared_mask is None:
        return None
    shared_key = _canonical_shape_key(shared_mask)

    banned = {shared_key}
    unique_masks: list[np.ndarray] = []
    for _ in range(unique_non_unit):
        mask = _gen_shape_mask(
            random.choice(UNIQUE_SIZE_CHOICES),
            banned_keys=banned,
            require_multi_variant=False,
        )
        if mask is None:
            return None
        banned.add(_canonical_shape_key(mask))
        unique_masks.append(mask)

    class_specs = [
        {"class_id": "repeat", "color": int(palette.pop()), "base_mask": shared_mask, "count": int(repeat_mult)},
        {"class_id": "twin", "color": int(palette.pop()), "base_mask": shared_mask, "count": 1},
    ]
    for idx, mask in enumerate(unique_masks):
        class_specs.append({"class_id": f"unique_{idx}", "color": int(palette.pop()), "base_mask": mask, "count": 1})
    class_specs.append({"class_id": "unit", "color": int(palette.pop()), "base_mask": np.ones((1, 1), dtype=bool), "count": 1})

    plans: list[dict] = []
    for spec in class_specs:
        for _ in range(int(spec["count"])):
            plans.append(
                {
                    "class_id": str(spec["class_id"]),
                    "color": int(spec["color"]),
                    "base_mask": spec["base_mask"],
                }
            )

    layout_res = _gen_layout(layout_h, layout_w, plans)
    if layout_res is None:
        return None
    slots, layout_mask = layout_res

    frame_area = int((layout_h + 2) * (layout_w + 2) - layout_mask.sum())
    pieces = []
    for spec in class_specs:
        piece = _make_piece(
            spec["base_mask"],
            int(spec["color"]),
            int(k),
            apply_dihedral=bool(int(spec["base_mask"].sum()) > 1),
        )
        piece["class_id"] = str(spec["class_id"])
        pieces.append(piece)
    if frame_area <= max(int(piece["in_mask"].sum()) for piece in pieces):
        return None

    expected = _build_expected_output(slots, layout_h, layout_w, k, frame_color)
    canvas = _build_input_canvas(pieces, slots, layout_mask, layout_h, layout_w, frame_color, distractor_color)
    if canvas is None:
        return None

    try:
        produced = main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    scene = _parse_scene(np.array(grid, dtype=int))
    slots = scene["slots"]
    pieces = scene["pieces"]

    color_counts = Counter(int(slot["color"]) for slot in slots)
    shape_colors: dict[tuple[int, int, bytes], set[int]] = defaultdict(set)
    for slot in slots:
        shape_colors[slot["shape_key"]].add(int(slot["color"]))

    return {
        "slot_count": int(len(slots)),
        "piece_count": int(len(pieces)),
        "reuse_gap": int(len(slots) - len(pieces)),
        "reused_color_classes": int(sum(count > 1 for count in color_counts.values())),
        "max_color_multiplicity": int(max(color_counts.values())),
        "multi_color_shape_groups": int(sum(len(colors) > 1 for colors in shape_colors.values())),
    }
