from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, components_mask, crop_bbox, find_components, get_bg, paste


BACKGROUND = Color.BLACK
SPRITE_A_COLOR = Color.BLUE
SPRITE_B_COLOR = Color.RED
GEN_ATTEMPTS = 600
GEN_RNG = np.random.default_rng()


def _enumerate_sprite_masks(height: int, width: int) -> list[np.ndarray]:
    """Enumerate tight 4-connected irregular masks with 3-5 foreground cells."""
    masks: list[np.ndarray] = []
    cell_count = height * width
    for bits in range(1 << cell_count):
        area = bits.bit_count()
        if area < 3 or area > 5:
            continue
        mask = np.zeros((height, width), dtype=bool)
        for idx in range(cell_count):
            if bits & (1 << idx):
                mask[idx // width, idx % width] = True
        if mask.all():
            continue
        coords = np.argwhere(mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if (int(r0), int(c0), int(r1), int(c1)) != (0, 0, height - 1, width - 1):
            continue
        if len(components_mask(mask, connectivity=4)) != 1:
            continue
        masks.append(mask)
    return masks


SPRITE_MASKS: dict[tuple[int, int], list[np.ndarray]] = {
    (height, width): _enumerate_sprite_masks(height, width)
    for height in (2, 3)
    for width in (2, 3)
}


def _mask_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _component_color_set(grid: np.ndarray, comp: dict) -> set[int]:
    coords = comp["coords"]
    return {int(grid[r, c]) for r, c in coords}


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = int(get_bg(g))
    assert bg == BACKGROUND, "expected black background"

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert len(comps) == 3, "expected exactly two sprites and one control matrix"

    control_candidates = []
    sprite_candidates = []
    for comp in comps:
        box = tuple(map(int, comp["bbox"]))
        area = int(comp["area"])
        box_area = int(bbox_h(box) * bbox_w(box))
        colors = _component_color_set(g, comp)
        if area == box_area and colors <= {SPRITE_A_COLOR, SPRITE_B_COLOR}:
            control_candidates.append(comp)
        else:
            sprite_candidates.append(comp)

    assert len(control_candidates) == 1, "expected one dense control rectangle"
    assert len(sprite_candidates) == 2, "expected two sprite components"

    control_comp = control_candidates[0]
    sprites = sorted(sprite_candidates, key=lambda comp: int(comp["bbox"][1]))
    sprite_a_comp, sprite_b_comp = sprites

    a_bbox = tuple(map(int, sprite_a_comp["bbox"]))
    b_bbox = tuple(map(int, sprite_b_comp["bbox"]))
    a_h, a_w = bbox_h(a_bbox), bbox_w(a_bbox)
    b_h, b_w = bbox_h(b_bbox), bbox_w(b_bbox)
    assert (a_h, a_w) == (b_h, b_w), "sprite bounding boxes must match"
    assert 2 <= a_h <= 3 and 2 <= a_w <= 3, "unexpected sprite size"

    ar0, ac0, ar1, ac1 = a_bbox
    br0, bc0, br1, bc1 = b_bbox
    assert ar0 == 0 and ac0 == 0, "sprite A must occupy the top-left corner"
    assert br0 == 0, "sprite B must share sprite A's top row"
    assert bc0 == ac1 + 2, "sprite B must be one background column to the right of sprite A"
    assert br1 == ar1 and bc1 - bc0 == ac1 - ac0, "sprite boxes must align exactly"

    a_colors = _component_color_set(g, sprite_a_comp)
    b_colors = _component_color_set(g, sprite_b_comp)
    assert a_colors == {SPRITE_A_COLOR}, "sprite A must be blue"
    assert b_colors == {SPRITE_B_COLOR}, "sprite B must be red"

    a_mask = crop_bbox(g == SPRITE_A_COLOR, a_bbox)
    b_mask = crop_bbox(g == SPRITE_B_COLOR, b_bbox)
    assert 3 <= int(a_mask.sum()) <= 5 and 3 <= int(b_mask.sum()) <= 5
    assert not np.array_equal(a_mask, b_mask), "the two sprites must differ"
    overlap_mask = a_mask & b_mask
    assert overlap_mask.any(), "the two sprites must conflict when aligned"
    assert int(overlap_mask.sum()) < int(a_mask.sum()) and int(overlap_mask.sum()) < int(b_mask.sum())

    control_bbox = tuple(map(int, control_comp["bbox"]))
    control = crop_bbox(g, control_bbox)
    control_colors = set(map(int, np.unique(control)))
    assert control_colors == {SPRITE_A_COLOR, SPRITE_B_COLOR}, "control matrix must use both sprite colors"

    step_r = a_h + 1
    step_c = a_w + 1
    expected_rows = len(range(br0, g.shape[0], step_r))
    expected_cols = len(range(bc0, g.shape[1], step_c))
    assert control.shape == (expected_rows, expected_cols), "control matrix shape must match the shared stamp lattice"

    return {
        "grid_shape": tuple(map(int, g.shape)),
        "bg": bg,
        "a_bbox": a_bbox,
        "b_bbox": b_bbox,
        "control_bbox": control_bbox,
        "a_mask": a_mask,
        "b_mask": b_mask,
        "control": control,
        "step_r": step_r,
        "step_c": step_c,
    }


def _render_scene(scene: dict, control: np.ndarray | None = None) -> np.ndarray:
    """Compose the output from parsed scene data."""
    grid_shape = scene["grid_shape"]
    bg = int(scene["bg"])
    a_mask = scene["a_mask"]
    b_mask = scene["b_mask"]
    control = scene["control"] if control is None else np.asarray(control, dtype=int)
    step_r = int(scene["step_r"])
    step_c = int(scene["step_c"])
    a_r0, a_c0, _a_r1, _a_c1 = scene["a_bbox"]
    b_r0, b_c0, _b_r1, _b_c1 = scene["b_bbox"]
    ctl_r0, ctl_c0, ctl_r1, ctl_c1 = scene["control_bbox"]

    out = np.full(grid_shape, bg, dtype=int)
    a_patch = _mask_patch(a_mask, SPRITE_A_COLOR)
    overlap_patch = np.zeros(b_mask.shape, dtype=int)
    overlap_patch[b_mask & ~a_mask] = SPRITE_B_COLOR

    for rr in range(a_r0, grid_shape[0], step_r):
        for cc in range(a_c0, grid_shape[1], step_c):
            paste(out, a_patch, rr, cc, transparent=bg, clip=True)

    for gi, rr in enumerate(range(b_r0, grid_shape[0], step_r)):
        for gj, cc in enumerate(range(b_c0, grid_shape[1], step_c)):
            patch = overlap_patch.copy()
            patch[a_mask & b_mask] = int(control[gi, gj])
            paste(out, patch, rr, cc, transparent=bg, clip=True)

    out[ctl_r0 : ctl_r1 + 1, ctl_c0 : ctl_c1 + 1] = bg
    return out


def _visible_conflict_slots(scene: dict) -> list[tuple[int, int]]:
    """Return lattice slots where at least one conflict pixel survives outside the control bbox."""
    overlap = scene["a_mask"] & scene["b_mask"]
    if not overlap.any():
        return []

    h, w = scene["grid_shape"]
    b_r0, b_c0, _b_r1, _b_c1 = scene["b_bbox"]
    step_r = int(scene["step_r"])
    step_c = int(scene["step_c"])
    ctl_r0, ctl_c0, ctl_r1, ctl_c1 = scene["control_bbox"]
    slots: list[tuple[int, int]] = []
    for gi, rr in enumerate(range(b_r0, h, step_r)):
        for gj, cc in enumerate(range(b_c0, w, step_c)):
            visible = False
            for dr, dc in np.argwhere(overlap):
                ar = int(rr + dr)
                ac = int(cc + dc)
                if not (0 <= ar < h and 0 <= ac < w):
                    continue
                if ctl_r0 <= ar <= ctl_r1 and ctl_c0 <= ac <= ctl_c1:
                    continue
                visible = True
                break
            if visible:
                slots.append((gi, gj))
    return slots


def _semantic_checks(grid: np.ndarray) -> bool:
    """Validate that the generated input exercises the intended rule rather than a shortcut."""
    scene = _extract_scene(grid)
    out = _render_scene(scene)
    overlap = scene["a_mask"] & scene["b_mask"]
    conflict_slot_count = 0
    h, w = scene["grid_shape"]
    b_r0, b_c0, _b_r1, _b_c1 = scene["b_bbox"]
    step_r = int(scene["step_r"])
    step_c = int(scene["step_c"])
    for rr in range(b_r0, h, step_r):
        for cc in range(b_c0, w, step_c):
            for dr, dc in np.argwhere(overlap):
                ar = int(rr + dr)
                ac = int(cc + dc)
                if 0 <= ar < h and 0 <= ac < w:
                    conflict_slot_count += 1
                    break
    if conflict_slot_count < 3:
        return False

    visible_slots = _visible_conflict_slots(scene)
    if len(visible_slots) < 2:
        return False

    visible_values = {int(scene["control"][gi, gj]) for gi, gj in visible_slots}
    if visible_values != {SPRITE_A_COLOR, SPRITE_B_COLOR}:
        return False

    all_a = np.full_like(scene["control"], SPRITE_A_COLOR)
    all_b = np.full_like(scene["control"], SPRITE_B_COLOR)
    out_all_a = _render_scene(scene, control=all_a)
    out_all_b = _render_scene(scene, control=all_b)
    if np.array_equal(out, out_all_a) or np.array_equal(out, out_all_b):
        return False

    if not ((out == SPRITE_A_COLOR).any() and (out == SPRITE_B_COLOR).any()):
        return False

    return np.array_equal(main(grid), out)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp two adjacent sprites across a shared lattice and use a control matrix to resolve overlap colors.

    Input structure:
    - A black grid contains a blue sprite A in the top-left corner and a red sprite B one background column to its right.
    - The two sprites have the same 2x2, 2x3, 3x2, or 3x3 tight bounding box, but different irregular 4-connected masks.
    - A dense blue/red control matrix sits in the lower-right area; its shape matches the row/column count of B's stamp slots.

    Rule:
    - Stamp sprite A from its own upper-left corner every bbox_height+1 rows and bbox_width+1 columns, coloring its foreground blue.
    - Stamp sprite B from its own upper-left corner with the same spacings, coloring its foreground red.
    - Where only one layer is foreground, keep that layer's color; where both layers are foreground, use the control matrix entry for that B-slot position to choose blue or red; then erase the control-matrix rectangle back to black.

    Key insight:
    - The control matrix does not recolor an existing object directly; instead, each of its cells selects which layer wins at one periodic overlap slot, so the final overlay cannot be explained by a fixed paint order.
    """
    scene = _extract_scene(grid)
    return _render_scene(scene)


def _choose_sprite_pair(height: int, width: int) -> tuple[np.ndarray, np.ndarray] | None:
    masks = SPRITE_MASKS[(height, width)]
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for a_mask in masks:
        for b_mask in masks:
            if np.array_equal(a_mask, b_mask):
                continue
            overlap = a_mask & b_mask
            if not overlap.any():
                continue
            if int(overlap.sum()) >= int(a_mask.sum()) or int(overlap.sum()) >= int(b_mask.sum()):
                continue
            candidates.append((a_mask, b_mask))
    if not candidates:
        return None
    idx = int(GEN_RNG.integers(len(candidates)))
    return candidates[idx]


def _sample_axis_extent(period_count: int, sprite_size: int, force_clip: bool) -> int:
    step = sprite_size + 1
    if force_clip and sprite_size > 2:
        tail = int(GEN_RNG.integers(1, sprite_size))
    elif force_clip:
        tail = 1
    else:
        tail = int(GEN_RNG.integers(sprite_size, sprite_size + 2))
    return int((period_count - 1) * step + tail)


def _sample_control_position(grid_shape: tuple[int, int], sprite_shape: tuple[int, int], control_shape: tuple[int, int]) -> tuple[int, int] | None:
    height, width = grid_shape
    sprite_h, sprite_w = sprite_shape
    ctl_h, ctl_w = control_shape
    min_r = sprite_h + 2
    max_r = height - ctl_h
    min_c = 2 * sprite_w + 2
    max_c = width - ctl_w
    if min_r > max_r or min_c > max_c:
        return None

    low_r = max(min_r, max_r - 2)
    low_c = max(min_c, max_c - 2)
    ctl_r0 = int(GEN_RNG.integers(low_r, max_r + 1))
    ctl_c0 = int(GEN_RNG.integers(low_c, max_c + 1))
    return ctl_r0, ctl_c0


def _build_control(scene_stub: dict) -> np.ndarray | None:
    visible_slots = _visible_conflict_slots(scene_stub)
    if len(visible_slots) < 2:
        return None

    control = GEN_RNG.choice([SPRITE_A_COLOR, SPRITE_B_COLOR], size=scene_stub["control"].shape).astype(int)
    first_slot = visible_slots[int(GEN_RNG.integers(len(visible_slots)))]
    second_choices = [slot for slot in visible_slots if slot != first_slot]
    second_slot = second_choices[int(GEN_RNG.integers(len(second_choices)))]
    control[first_slot] = SPRITE_A_COLOR
    control[second_slot] = SPRITE_B_COLOR
    if not ((control == SPRITE_A_COLOR).any() and (control == SPRITE_B_COLOR).any()):
        return None
    return control


def _gen_attempt() -> np.ndarray | None:
    sprite_h = int(GEN_RNG.choice([2, 3]))
    sprite_w = int(GEN_RNG.choice([2, 3]))
    pair = _choose_sprite_pair(sprite_h, sprite_w)
    if pair is None:
        return None
    a_mask, b_mask = pair

    period_rows = int(GEN_RNG.integers(2, 6))
    period_cols = int(GEN_RNG.integers(2, 6))
    height = _sample_axis_extent(period_rows, sprite_h, force_clip=bool(GEN_RNG.integers(2)))
    width = period_cols * (sprite_w + 1) + int(
        _sample_axis_extent(1, sprite_w, force_clip=bool(GEN_RNG.integers(2)))
    )

    if height > 30 or width > 30 or height < sprite_h + 4 or width < 2 * sprite_w + 4:
        return None

    b_cols = len(range(sprite_w + 1, width, sprite_w + 1))
    if b_cols != period_cols:
        return None

    b_rows = len(range(0, height, sprite_h + 1))
    if b_rows != period_rows:
        return None

    control_pos = _sample_control_position((height, width), (sprite_h, sprite_w), (period_rows, period_cols))
    if control_pos is None:
        return None
    ctl_r0, ctl_c0 = control_pos

    grid = np.zeros((height, width), dtype=int)
    paste(grid, _mask_patch(a_mask, SPRITE_A_COLOR), 0, 0, transparent=BACKGROUND)
    paste(grid, _mask_patch(b_mask, SPRITE_B_COLOR), 0, sprite_w + 1, transparent=BACKGROUND)

    scene_stub = {
        "grid_shape": (height, width),
        "bg": BACKGROUND,
        "a_bbox": (0, 0, sprite_h - 1, sprite_w - 1),
        "b_bbox": (0, sprite_w + 1, sprite_h - 1, 2 * sprite_w),
        "control_bbox": (ctl_r0, ctl_c0, ctl_r0 + period_rows - 1, ctl_c0 + period_cols - 1),
        "a_mask": a_mask,
        "b_mask": b_mask,
        "control": np.full((period_rows, period_cols), SPRITE_A_COLOR, dtype=int),
        "step_r": sprite_h + 1,
        "step_c": sprite_w + 1,
    }
    control = _build_control(scene_stub)
    if control is None:
        return None
    scene_stub["control"] = control
    paste(grid, control, ctl_r0, ctl_c0)

    try:
        if not _semantic_checks(grid):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-stamp control input")
