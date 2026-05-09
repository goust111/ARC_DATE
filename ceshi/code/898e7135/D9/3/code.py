# Variant of 898e7135
# Dimension: D9 — 洞区域的空间约束 (meta)
# Change: 洞区域 4-连通且彼此 8-隔离（相邻洞之间至少隔 1 像素墙壁，含对角线方向），面积 1-7 像素... → 所有洞面积相同（仅形状不同）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location

# ---------------------------------------------------------------------------
# Design analysis for the D9 variant
#
# spec:
#   preserved_core:
#     - 输入仍然由一个单色框架、框外散落的拼块、以及单像素干扰点构成。
#     - 每个拼块仍然是某个洞形状按 k×k 放大后的单色精灵，并允许旋转/翻转。
#     - 输出仍然是把框内洞布局按 k 放大，并在对应位置填入匹配精灵。
#   changed_dimension:
#     - 原题非单元洞的面积可不同，因此 solver 可以先按面积过滤候选。
#     - 变体要求所有非单元洞面积完全相同；它们只能靠形状区分。
#   concrete_example_trace:
#     - 原题某个样例可以有洞面积 [5, 4, 2, 1]。solver 处理面积为 4 的洞时，
#       先用 area == 4*k*k 过滤，再比较形状。
#     - 本变体中一个样例可能有三个非单元洞，面积都为 4，但形状分别是 L/T/Z。
#       此时 piece 的面积对三者都一样，面积过滤不再提供信息，必须把拼块降采样后
#       直接与洞的布尔形状比较。
#   complexity_calibration:
#     - 每个样例至少含 2 个非单元洞，否则“同面积”在人眼上不明显。
#     - 共同面积只从少量档位里选（3/4/5），并且同一档位内使用多个不同 D4 形状，
#       避免自由度过大，也避免退化成“只有一个可匹配对象”。
#   hidden_assumption_review:
#     - `need_area = slot_area * k * k` 是原 solver 对旧属性的关键依赖；变体中必须删除。
#     - `k` 的推断仍依赖最小实心方块，所以生成器继续保留 1-2 个 unit 洞。
#     - 贪心匹配现在默认“非单元洞在 D4 意义下互不相同”；生成器通过 canonical
#       shape 去重来保证这一点，否则纯形状匹配会产生歧义。
#   placement_legality:
#     - 洞仍然必须 4-连通且彼此 8-隔离，所以布局生成继续在布尔占位图上做 8-邻域膨胀。
#     - 框外拼块仍用 `random_free_location(..., padding=1, padding_connectivity=8)`
#       放置，保证与框和彼此至少隔 1 个背景像素。
#   visual_observability:
#     - 人类在样例中能直接看到：多个非单元洞面积相同，但轮廓不同；外部对应拼块也都有
#       相同像素面积，只能靠形状去对应。
#     - 不使用颜色数值、面积数值的隐含排序；颜色仅用于区分不同拼块。
#   cleanliness:
#     - 仍保留单像素干扰，但颜色统一且与框/拼块不同，不与主体结构叠加。
#   new_solver_assumptions_and_generator_guarantees:
#     - solver 现在假设非单元形状在 D4 下唯一；generator 保证。
#     - solver 假设存在至少一个 unit piece 以恢复 k；generator 保证 1-2 个 unit 洞。
#     - solver 假设每个 piece 是严格的 k×k block-scaling，没有半块；generator 继续用
#       `np.kron` 构造，确保降采样合法。
# ---------------------------------------------------------------------------


def _mask_signature(mask: np.ndarray) -> tuple[int, int, bytes]:
    """Exact signature for a tight-cropped bool mask."""
    tight = np.array(mask, dtype=bool, copy=False)
    return int(tight.shape[0]), int(tight.shape[1]), tight.tobytes()


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    box = bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = map(int, box)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _canonical_shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    """Canonical signature of a bool mask under the dihedral group."""
    best = None
    for kk in range(4):
        rot = np.rot90(mask, int(kk))
        for var in (rot, np.fliplr(rot)):
            tight = _tight_mask(np.array(var, dtype=bool, copy=False))
            sig = _mask_signature(tight)
            if best is None or sig < best:
                best = sig
    assert best is not None
    return best


def _dihedral_masks(mask: np.ndarray) -> list[np.ndarray]:
    """Unique tight-cropped dihedral variants of a bool mask."""
    out: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    for kk in range(4):
        rot = np.rot90(mask, int(kk))
        for var in (rot, np.fliplr(rot)):
            tight = _tight_mask(np.array(var, dtype=bool, copy=False))
            sig = _mask_signature(tight)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(tight.copy())
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate a frame's hole-layout and fill each hole with its matching piece.

    D9 variant twist:
    - Non-unit holes all share the same area.
    - Matching therefore cannot rely on area pre-filtering and must compare
      pure block-level shapes under rotation/flip.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    bgc = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == bg
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one bg hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
        sm = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append(
            {
                "r0": int(rr0),
                "c0": int(cc0),
                "mask": sm,
                "shape_key": _canonical_shape_key(sm),
            }
        )
    slots.sort(key=lambda s: (int(s["r0"]), int(s["c0"])))
    assert slots, "frame must contain at least one slot"

    pieces = []
    for comp in comps:
        if comp is frame_comp:
            continue
        coords = comp["coords"]
        if coords.size == 0 or int(coords.shape[0]) <= 1:
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        pm = np.zeros(tuple(map(int, patch.shape)), dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"mask": pm, "patch": patch})

    assert pieces, "at least one non-frame piece must exist"

    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    k = int(k or 1)

    def downsample_blocks(mask: np.ndarray) -> np.ndarray | None:
        hh, ww = map(int, mask.shape)
        if hh % k != 0 or ww % k != 0:
            return None
        out = np.zeros((hh // k, ww // k), dtype=bool)
        for r in range(out.shape[0]):
            for c in range(out.shape[1]):
                block = mask[r * k : (r + 1) * k, c * k : (c + 1) * k]
                if bool(block.all()):
                    out[r, c] = True
                elif bool(block.any()):
                    return None
        return out

    def dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        base = []
        for kk in range(4):
            rot_p = np.rot90(patch, int(kk))
            rot_m = np.rot90(mask, int(kk))
            base.append((rot_p, rot_m))
            base.append((np.fliplr(rot_p), np.fliplr(rot_m)))
        uniq = []
        seen: set[tuple[int, int, bytes]] = set()
        for vp, vm in base:
            sig = _mask_signature(np.array(vm, dtype=bool, copy=False))
            if sig in seen:
                continue
            seen.add(sig)
            uniq.append((vp, vm))
        return uniq

    candidates = []
    for pi, piece in enumerate(pieces):
        for vp, vm in dihedral_variants(piece["patch"], piece["mask"]):
            um = downsample_blocks(vm)
            if um is None:
                continue
            candidates.append(
                {
                    "pi": int(pi),
                    "um": um,
                    "exact_key": _mask_signature(um),
                    "vp": vp,
                    "vm": vm,
                }
            )

    used_piece: set[int] = set()
    placements: list[tuple[dict, dict]] = []
    for slot in slots:
        found = None
        slot_key = _mask_signature(slot["mask"])
        for cand in candidates:
            if int(cand["pi"]) in used_piece:
                continue
            if cand["exact_key"] != slot_key:
                continue
            found = cand
            break
        assert found is not None, "every slot must match a piece by pure shape"
        used_piece.add(int(found["pi"]))
        placements.append((slot, found))

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(bgc), dtype=int)

    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)

    return out


# --------------- generator ---------------------------------------------------

K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
N_UNITS_MAX = 2
NON_UNIT_AREA_CHOICES = [3, 4, 5]

HOLE_DENSITY_RANGE = (0.20, 0.50)
LAYOUT_PLACE_RETRIES = 40

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 500


def _coords_to_mask(coords: set[tuple[int, int]]) -> np.ndarray:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    out = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    for r, c in coords:
        out[int(r - r0), int(c - c0)] = True
    return out


def _enumerate_free_polyominoes(max_area: int) -> dict[int, list[np.ndarray]]:
    by_area: dict[int, list[np.ndarray]] = {1: [np.ones((1, 1), dtype=bool)]}
    for area in range(2, max_area + 1):
        next_shapes: dict[tuple[int, int, bytes], np.ndarray] = {}
        for shape in by_area[area - 1]:
            coords = {tuple(map(int, rc)) for rc in np.argwhere(shape)}
            for r, c in list(coords):
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (nr, nc) in coords:
                        continue
                    grown = set(coords)
                    grown.add((int(nr), int(nc)))
                    mask = _coords_to_mask(grown)
                    key = _canonical_shape_key(mask)
                    if key not in next_shapes:
                        next_shapes[key] = _tight_mask(mask)
        by_area[area] = list(next_shapes.values())
    return by_area


SHAPES_BY_AREA = _enumerate_free_polyominoes(max(NON_UNIT_AREA_CHOICES))


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


def _place_mask(forbidden: np.ndarray, base_mask: np.ndarray) -> dict | None:
    H, W = map(int, forbidden.shape)
    variants = _dihedral_masks(base_mask)
    random.shuffle(variants)
    placements: list[dict] = []
    for vm in variants:
        h, w = map(int, vm.shape)
        if h > H or w > W:
            continue
        locs = [(r0, c0) for r0 in range(H - h + 1) for c0 in range(W - w + 1)]
        random.shuffle(locs)
        rel = np.argwhere(vm)
        for r0, c0 in locs:
            coords = rel + np.array([[int(r0), int(c0)]], dtype=int)
            if np.any(forbidden[coords[:, 0], coords[:, 1]]):
                continue
            placements.append({"coords": coords, "mask": vm})
            break
    if not placements:
        return None
    return random.choice(placements)


def _crop_slots(slots: list[dict]) -> tuple[list[dict], np.ndarray] | tuple[None, None]:
    if not slots:
        return None, None
    all_coords = np.concatenate([s["coords"] for s in slots], axis=0)
    r0, c0, r1, c1 = bbox_coords(all_coords)
    layout_h = int(r1 - r0 + 1)
    layout_w = int(c1 - c0 + 1)
    if not (LAYOUT_H_RANGE[0] <= layout_h <= LAYOUT_H_RANGE[1]):
        return None, None
    if not (LAYOUT_W_RANGE[0] <= layout_w <= LAYOUT_W_RANGE[1]):
        return None, None

    out_slots = []
    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for slot in slots:
        coords = slot["coords"] - np.array([[int(r0), int(c0)]], dtype=int)
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        sm = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        layout_mask[coords[:, 0], coords[:, 1]] = True
        out_slots.append(
            {
                "coords": coords,
                "area": int(coords.shape[0]),
                "mask": sm,
                "shape_key": _canonical_shape_key(sm),
            }
        )

    density = float(layout_mask.sum()) / float(layout_h * layout_w)
    if not (HOLE_DENSITY_RANGE[0] <= density <= HOLE_DENSITY_RANGE[1]):
        return None, None
    return out_slots, layout_mask


def _gen_layout(stage_h: int, stage_w: int, non_unit_shapes: list[np.ndarray], n_units: int) -> tuple[list[dict], np.ndarray] | tuple[None, None]:
    shape_order = sorted(non_unit_shapes, key=lambda m: (-int(m.shape[0] * m.shape[1]), -int(m.sum())))
    for _ in range(LAYOUT_PLACE_RETRIES):
        forbidden = np.zeros((stage_h, stage_w), dtype=bool)
        slots: list[dict] = []
        ok = True

        for shape in shape_order:
            placed = _place_mask(forbidden, shape)
            if placed is None:
                ok = False
                break
            coords = placed["coords"]
            slots.append(
                {
                    "coords": coords,
                    "area": int(coords.shape[0]),
                    "mask": placed["mask"],
                    "shape_key": _canonical_shape_key(placed["mask"]),
                }
            )
            _forbid_coords(forbidden, coords)
        if not ok:
            continue

        unit_mask = np.ones((1, 1), dtype=bool)
        for _unit_idx in range(n_units):
            placed = _place_mask(forbidden, unit_mask)
            if placed is None:
                ok = False
                break
            coords = placed["coords"]
            slots.append({"coords": coords, "area": 1, "mask": unit_mask, "shape_key": _canonical_shape_key(unit_mask)})
            _forbid_coords(forbidden, coords)
        if not ok:
            continue

        cropped_slots, layout_mask = _crop_slots(slots)
        if cropped_slots is None:
            continue

        non_unit_keys = [s["shape_key"] for s in cropped_slots if int(s["area"]) > 1]
        if len(non_unit_keys) != len(set(non_unit_keys)):
            continue
        non_unit_areas = {int(s["area"]) for s in cropped_slots if int(s["area"]) > 1}
        if len(non_unit_areas) != 1:
            continue
        return cropped_slots, layout_mask

    return None, None


def _make_piece(slot: dict, color: int, k: int, apply_dihedral: bool) -> dict:
    if int(slot["area"]) == 1:
        patch = np.full((k, k), int(color), dtype=int)
        mask = np.ones((k, k), dtype=bool)
        return {
            "color": int(color),
            "slot": slot,
            "out_patch": patch,
            "out_mask": mask,
            "in_patch": patch,
            "in_mask": mask,
        }

    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, int(color), 0).astype(int)
    if apply_dihedral:
        variants = _dihedral_masks(slot["mask"])
        chosen = random.choice(variants)
        scale = np.ones((k, k), dtype=bool)
        in_mask = np.kron(chosen, scale)
        in_patch = np.where(in_mask, int(color), 0).astype(int)
    else:
        in_mask, in_patch = out_mask, out_patch

    return {
        "color": int(color),
        "slot": slot,
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
    }


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for piece in pieces:
        coords = piece["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * int(k)
        left = (int(coords[:, 1].min()) + 1) * int(k)
        paste(out, piece["out_patch"], int(top), int(left), mask=piece["out_mask"], overwrite=True)
    return out


def _build_input_canvas(
    pieces: list[dict],
    layout_mask: np.ndarray,
    layout_h: int,
    layout_w: int,
    k: int,
    frame_color: int,
    distractor_color: int,
) -> np.ndarray | None:
    frame_h, frame_w = int(layout_h + 2), int(layout_w + 2)
    total_piece_area = sum(int(p["in_mask"].sum()) for p in pieces)
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
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, int(frame_color))
    fr_max, fc_max = canvas_h - frame_h - 1, canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    for piece in sorted(pieces, key=lambda p: -int(p["in_mask"].sum())):
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
        if np.any(canvas[r_lo:r_hi, c_lo:c_hi] != 0):
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
    n_slots = random.randint(*N_SLOTS_RANGE)
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 2))
    n_non_unit = int(n_slots - n_units)

    area_choices = [a for a in NON_UNIT_AREA_CHOICES if len(SHAPES_BY_AREA[a]) >= n_non_unit]
    if not area_choices:
        return None
    common_area = random.choice(area_choices)

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    needed_piece_colors = n_non_unit + 1
    if len(palette) < needed_piece_colors:
        return None
    non_unit_colors = palette[:n_non_unit]
    unit_color = palette[n_non_unit]

    stage_h = random.randint(*LAYOUT_H_RANGE)
    stage_w = random.randint(*LAYOUT_W_RANGE)
    chosen_shapes = random.sample(SHAPES_BY_AREA[common_area], n_non_unit)
    slots, layout_mask = _gen_layout(stage_h, stage_w, chosen_shapes, n_units)
    if slots is None or layout_mask is None:
        return None

    unit_slots = [s for s in slots if int(s["area"]) == 1]
    non_unit_slots = [s for s in slots if int(s["area"]) > 1]
    if len(non_unit_slots) < 2:
        return None
    if {int(s["area"]) for s in non_unit_slots} != {int(common_area)}:
        return None

    pieces = [_make_piece(slot, non_unit_colors[i], k, apply_dihedral=True) for i, slot in enumerate(non_unit_slots)]
    pieces += [_make_piece(slot, unit_color, k, apply_dihedral=False) for slot in unit_slots]

    layout_h, layout_w = map(int, layout_mask.shape)
    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
    canvas = _build_input_canvas(pieces, layout_mask, layout_h, layout_w, k, frame_color, distractor_color)
    if canvas is None:
        return None

    try:
        produced = main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected_out):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError(f"generate_input: failed to generate after {GENERATE_MAX_RETRIES} attempts")
