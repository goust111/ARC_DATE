from __future__ import annotations

from collections import Counter
from math import gcd

import numpy as np


_RNG = np.random.default_rng()
_BG = 0
_COLOR_POOL = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
_GEN_RETRIES = 300
_FRAME_THICKNESS = 1
_SLOT_GAP = 1
_TEMPLATE_SIDE = 3
_INSTRUCTION_CHOICES = (0, 1, 2, 3)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_in_bbox(coords: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[r - r0, c - c0] = True
    return out


def _is_rectangular_frame(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    expected = np.zeros_like(mask, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(mask, expected)


def _downscale_binary(mask: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return mask.copy()
    h, w = mask.shape
    assert h % scale == 0 and w % scale == 0
    blocks = mask.reshape(h // scale, scale, w // scale, scale)
    block_any = blocks.any(axis=(1, 3))
    block_all = blocks.all(axis=(1, 3))
    assert np.array_equal(block_any, block_all)
    return block_any


def _upscale_binary(mask: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return mask.copy()
    return np.kron(mask.astype(int), np.ones((scale, scale), dtype=int)).astype(bool)


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color


def _rectangles_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    pad: int = 0,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _random_template() -> np.ndarray:
    for _ in range(200):
        area = int(_RNG.integers(5, 8))
        start = (int(_RNG.integers(_TEMPLATE_SIDE)), int(_RNG.integers(_TEMPLATE_SIDE)))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < _TEMPLATE_SIDE and 0 <= nc < _TEMPLATE_SIDE and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(_RNG.integers(len(frontier)))])
        if len(cells) != area:
            continue
        mask = np.zeros((_TEMPLATE_SIDE, _TEMPLATE_SIDE), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not mask.any(axis=0).all() or not mask.any(axis=1).all():
            continue
        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        return mask
    raise RuntimeError("failed to build asymmetric template")


def _component_info(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict[str, object]:
    box = _bbox(coords)
    mask = _mask_in_bbox(coords, box)
    colors = Counter(int(grid[r, c]) for r, c in coords)
    r0, c0, r1, c1 = box
    return {
        "bbox": box,
        "mask": mask,
        "colors": colors,
        "frame_like": _is_rectangular_frame(mask),
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "h": r1 - r0 + 1,
        "w": c1 - c0 + 1,
    }


def _find_slot_group(parts: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, int, int, int], list[dict[str, object]]] = {}
    for part in parts:
        if not bool(part["frame_like"]):
            continue
        colors = list(part["colors"].keys())
        if len(colors) != 1:
            continue
        key = (int(colors[0]), int(part["h"]), int(part["w"]), int(part["r0"]))
        groups.setdefault(key, []).append(part)
    candidates = [group for group in groups.values() if len(group) >= 2]
    assert candidates, "no slot frames detected"
    slots = max(candidates, key=lambda group: (len(group), max(int(p["r0"]) for p in group)))
    slots = sorted(slots, key=lambda part: int(part["c0"]))
    return slots


def _major_body_color(shapes: list[dict[str, object]], bg: int, slot_color: int) -> int:
    counts: Counter[int] = Counter()
    for shape in shapes:
        for color, amount in shape["colors"].items():
            if color not in (bg, slot_color):
                counts[int(color)] += int(amount)
    assert counts, "no shape colors found"
    return max(counts, key=counts.get)


def _divisors(value: int) -> list[int]:
    return [d for d in range(1, value + 1) if value % d == 0]


def _infer_unit_template(
    shapes: list[dict[str, object]],
    slot_inner_h: int,
    slot_inner_w: int,
) -> tuple[np.ndarray, int]:
    height_gcd = slot_inner_h
    width_gcd = slot_inner_w
    for shape in shapes:
        height_gcd = gcd(height_gcd, int(shape["h"]))
        width_gcd = gcd(width_gcd, int(shape["w"]))

    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for base_h in _divisors(height_gcd):
        for base_w in _divisors(width_gcd):
            if slot_inner_h % base_h != 0 or slot_inner_w % base_w != 0:
                continue
            target_scale_h = slot_inner_h // base_h
            target_scale_w = slot_inner_w // base_w
            if target_scale_h != target_scale_w:
                continue

            unit_template: np.ndarray | None = None
            valid = True
            for shape in shapes:
                sh = int(shape["h"])
                sw = int(shape["w"])
                if sh % base_h != 0 or sw % base_w != 0:
                    valid = False
                    break
                scale_h = sh // base_h
                scale_w = sw // base_w
                if scale_h != scale_w:
                    valid = False
                    break
                try:
                    unit = _downscale_binary(np.asarray(shape["mask"], dtype=bool), scale_h)
                except AssertionError:
                    valid = False
                    break
                if unit_template is None:
                    unit_template = unit
                elif not np.array_equal(unit_template, unit):
                    valid = False
                    break
            if valid and unit_template is not None:
                candidates.append((target_scale_h, base_h * base_w, base_h, unit_template))

    assert candidates, "failed to infer the common template"
    target_scale, _, _, unit_template = max(candidates, key=lambda item: (item[0], -item[1]))
    return unit_template, int(target_scale)


def _render_output(
    shape_order: list[dict[str, object]],
    slots: list[dict[str, object]],
    bg: int,
    frame_color: int,
    body_color: int,
    unit_template: np.ndarray,
    target_scale: int,
) -> np.ndarray:
    max_r = max(int(slot["r1"]) for slot in slots) + 1
    max_c = max(int(slot["c1"]) for slot in slots) + 1
    out = np.full((max_r, max_c), bg, dtype=int)

    slot_inner_h = int(slots[0]["h"]) - 2 * _FRAME_THICKNESS
    slot_inner_w = int(slots[0]["w"]) - 2 * _FRAME_THICKNESS
    target_mask = _upscale_binary(unit_template, target_scale)
    assert target_mask.shape == (slot_inner_h, slot_inner_w)

    for shape, slot in zip(shape_order, slots):
        instruction_count = sum(
            int(amount)
            for color, amount in shape["colors"].items()
            if int(color) != body_color
        )
        assert instruction_count in _INSTRUCTION_CHOICES
        rotated = np.rot90(target_mask, -instruction_count)

        r0 = int(slot["r0"])
        c0 = int(slot["c0"])
        sh = int(slot["h"])
        sw = int(slot["w"])
        _draw_frame(out, r0, c0, sh, sw, frame_color)

        top = r0 + _FRAME_THICKNESS + (slot_inner_h - rotated.shape[0]) // 2
        left = c0 + _FRAME_THICKNESS + (slot_inner_w - rotated.shape[1]) // 2
        out[top : top + rotated.shape[0], left : left + rotated.shape[1]][rotated] = body_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each instructed scaled copy, normalize all copies to the largest size, and redraw them into the bottom slots.

    Input structure:
    - The background is black.
    - Several 4-connected copies of one asymmetric filled template are scattered above the bottom band; the copies differ only by integer scale.
    - Each copy uses one body color plus zero to three instruction pixels of another color inside the occupied shape.
    - Along the bottom there is a row of equal empty rectangular frames in one separator color; all frames have the same size, matching the largest copy's bounding box.

    Rule:
    - For each shape, count its instruction pixels: 0 means no rotation, 1 means rotate 90 degrees clockwise, 2 means 180 degrees, and 3 means 270 degrees clockwise.
    - Recover the common unscaled template from the occupied pixels, then resize every copy to the largest scale.
    - Sort the original shapes from left to right, remove all instruction pixels, and redraw the rotated large body-colored shapes into the bottom frames in that same order.
    - Clear the upper scattered copies to black and keep only the bottom frames with the transformed results.

    Key insight:
    - Rotation is encoded by counting the special interior pixels, while the shared template and the largest frame reveal the common target scale.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(np.bincount(g.ravel()).argmax())

    parts = [_component_info(g, coords) for coords in _components(g != bg)]
    slots = _find_slot_group(parts)
    slot_color = next(iter(slots[0]["colors"]))

    slot_boxes = {tuple(slot["bbox"]) for slot in slots}
    shapes = [part for part in parts if tuple(part["bbox"]) not in slot_boxes]
    shapes = sorted(shapes, key=lambda part: int(part["c0"]))
    assert len(shapes) == len(slots) >= 2
    assert all(int(shape["r1"]) < int(slots[0]["r0"]) for shape in shapes)

    body_color = _major_body_color(shapes, bg, int(slot_color))
    slot_inner_h = int(slots[0]["h"]) - 2 * _FRAME_THICKNESS
    slot_inner_w = int(slots[0]["w"]) - 2 * _FRAME_THICKNESS
    unit_template, target_scale = _infer_unit_template(shapes, slot_inner_h, slot_inner_w)

    out = np.full_like(g, bg)
    rendered = _render_output(
        shapes,
        slots,
        bg=bg,
        frame_color=int(slot_color),
        body_color=body_color,
        unit_template=unit_template,
        target_scale=target_scale,
    )
    out[: rendered.shape[0], : rendered.shape[1]] = rendered
    return out


def _sample_scales(num_objects: int) -> list[int]:
    if num_objects == 2:
        return [1, int(_RNG.choice([2, 3]))]
    choices = ([1, 1, 2], [1, 2, 2])
    return list(choices[int(_RNG.integers(len(choices)))])


def _build_input_patch(
    template: np.ndarray,
    scale: int,
    body_color: int,
    instruction_color: int,
    instruction_count: int,
) -> np.ndarray:
    occupied = _upscale_binary(template, scale)
    patch = np.full(occupied.shape, _BG, dtype=int)
    patch[occupied] = body_color
    cells = np.argwhere(occupied)
    if instruction_count:
        chosen = cells[_RNG.choice(len(cells), size=instruction_count, replace=False)]
        for r, c in chosen:
            patch[int(r), int(c)] = instruction_color
    return patch


def _place_shapes(
    patches: list[np.ndarray],
    slot_r0: int,
    grid_h: int,
    grid_w: int,
) -> list[tuple[int, int]] | None:
    order = sorted(range(len(patches)), key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)
    boxes: list[tuple[int, int, int, int]] = []
    positions: list[tuple[int, int] | None] = [None] * len(patches)
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        max_r0 = slot_r0 - ph - 1
        if max_r0 < 0 or grid_w - pw < 0:
            return None
        placed = False
        for _ in range(200):
            r0 = int(_RNG.integers(0, max_r0 + 1))
            c0 = int(_RNG.integers(0, grid_w - pw + 1))
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if any(_rectangles_overlap(box, other, pad=1) for other in boxes):
                continue
            boxes.append(box)
            positions[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None
    final = [pos for pos in positions if pos is not None]
    if len(final) != len(patches):
        return None
    if len({c0 for _, c0 in final}) != len(final):
        return None
    return [(int(r0), int(c0)) for r0, c0 in final]


def _expected_output(
    grid_shape: tuple[int, int],
    slots: list[tuple[int, int, int, int]],
    objects: list[dict[str, object]],
    frame_color: int,
    body_color: int,
    template: np.ndarray,
    target_scale: int,
) -> np.ndarray:
    out = np.full(grid_shape, _BG, dtype=int)
    target_mask = _upscale_binary(template, target_scale)
    ordered_objects = sorted(objects, key=lambda obj: int(obj["c0"]))
    ordered_slots = sorted(slots, key=lambda slot: int(slot[1]))
    for (r0, c0, sh, sw), obj in zip(ordered_slots, ordered_objects):
        _draw_frame(out, r0, c0, sh, sw, frame_color)
        turned = np.rot90(target_mask, -int(obj["turns"]))
        top = r0 + _FRAME_THICKNESS + (sh - 2 * _FRAME_THICKNESS - turned.shape[0]) // 2
        left = c0 + _FRAME_THICKNESS + (sw - 2 * _FRAME_THICKNESS - turned.shape[1]) // 2
        out[top : top + turned.shape[0], left : left + turned.shape[1]][turned] = body_color
    return out


def _gen_attempt() -> np.ndarray | None:
    num_objects = int(_RNG.choice([2, 3]))
    template = _random_template()
    colors = _RNG.choice(_COLOR_POOL, size=3, replace=False)
    body_color, instruction_color, frame_color = map(int, colors)
    turns = list(map(int, _RNG.permutation(4)[:num_objects]))
    scales = _sample_scales(num_objects)
    max_scale = max(scales)

    slot_inner = _TEMPLATE_SIDE * max_scale
    slot_outer = slot_inner + 2 * _FRAME_THICKNESS
    slots_width = num_objects * slot_outer + (num_objects - 1) * _SLOT_GAP
    if slots_width > 30:
        return None

    extra_w = int(_RNG.integers(0, min(5, 30 - slots_width) + 1))
    grid_w = slots_width + extra_w
    slot_left = int(_RNG.integers(0, grid_w - slots_width + 1))

    max_shape_side = _TEMPLATE_SIDE * max_scale
    slot_r0_min = max_shape_side + 2
    slot_r0_max = min(18, 30 - slot_outer)
    if slot_r0_max < slot_r0_min:
        return None
    slot_r0 = int(_RNG.integers(slot_r0_min, slot_r0_max + 1))
    bottom_margin = int(_RNG.integers(0, min(3, 30 - (slot_r0 + slot_outer)) + 1))
    grid_h = slot_r0 + slot_outer + bottom_margin
    if grid_h > 30:
        return None

    patches = [
        _build_input_patch(template, scale, body_color, instruction_color, turn_count)
        for scale, turn_count in zip(scales, turns)
    ]
    positions = _place_shapes(patches, slot_r0, grid_h, grid_w)
    if positions is None:
        return None

    grid = np.full((grid_h, grid_w), _BG, dtype=int)
    slot_specs: list[tuple[int, int, int, int]] = []
    for idx in range(num_objects):
        c0 = slot_left + idx * (slot_outer + _SLOT_GAP)
        slot_specs.append((slot_r0, c0, slot_outer, slot_outer))
        _draw_frame(grid, slot_r0, c0, slot_outer, slot_outer, frame_color)

    objects: list[dict[str, object]] = []
    for patch, (r0, c0), scale, turn_count in zip(patches, positions, scales, turns):
        ph, pw = patch.shape
        assert np.all(grid[r0 : r0 + ph, c0 : c0 + pw] == _BG)
        grid[r0 : r0 + ph, c0 : c0 + pw][patch != _BG] = patch[patch != _BG]
        objects.append({"r0": r0, "c0": c0, "scale": scale, "turns": turn_count})

    expected = _expected_output(
        grid.shape,
        slot_specs,
        objects,
        frame_color=frame_color,
        body_color=body_color,
        template=template,
        target_scale=max_scale,
    )
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
