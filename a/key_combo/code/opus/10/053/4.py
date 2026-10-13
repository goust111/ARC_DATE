from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, colors_present, components_mask, find_components, get_bg, paste


BACKGROUND = Color.BLACK
SEPARATOR = Color.GREY
SPRITE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 500
SHAPE_ATTEMPTS = 300
PATTERN_ATTEMPTS = 200


def _mask_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(mask.astype(np.uint8).ravel())


def _rotation_family(mask: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(mask, -turns) for turns in range(4)]


def _crop_mask(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return (grid[r0 : r1 + 1, c0 : c1 + 1] == color)


def _square_dims(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h == w, "template and slots must use square boxes"
    return h


def _find_template(grid: np.ndarray, fg: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    comps = find_components(
        grid,
        bg=BACKGROUND,
        colors=[fg],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert comps, "expected at least the template component"
    template = min(comps, key=lambda comp: tuple(map(int, comp["bbox"])))
    box = tuple(map(int, template["bbox"]))
    size = _square_dims(box)
    assert size in (3, 4), "template bbox must be 3x3 or 4x4"
    mask = _crop_mask(grid, box, fg)
    assert len(components_mask(mask, connectivity=4)) == 1, "template must be 4-connected"
    assert len({_mask_key(rot) for rot in _rotation_family(mask)}) == 4, "template must have four distinct rotations"
    return box, mask


def _find_slot_origins(grid: np.ndarray, size: int) -> tuple[str, list[tuple[int, int]]]:
    grey_comps = find_components(
        grid,
        bg=BACKGROUND,
        colors=[SEPARATOR],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert 4 <= len(grey_comps) <= 7, "expected 5-8 slots separated by grey lines"
    boxes = [tuple(map(int, comp["bbox"])) for comp in grey_comps]
    dims = {
        (box[2] - box[0] + 1, box[3] - box[1] + 1)
        for box in boxes
    }
    assert len(dims) == 1, "all separators must have the same size"
    sep_h, sep_w = dims.pop()

    if sep_h == 1 and sep_w == size:
        rows = sorted(box[0] for box in boxes)
        c0s = {box[1] for box in boxes}
        c1s = {box[3] for box in boxes}
        assert len(c0s) == 1 and len(c1s) == 1, "horizontal separators must align"
        assert all(box[0] == box[2] for box in boxes), "separator rows must be one pixel tall"
        diffs = [rows[i + 1] - rows[i] for i in range(len(rows) - 1)]
        assert len(set(diffs)) == 1 and diffs[0] == size + 1, "separator spacing must match slot size"
        start_r = rows[0] - size
        c0 = next(iter(c0s))
        origins = [(start_r + i * (size + 1), c0) for i in range(len(rows) + 1)]
        assert all(0 <= r <= grid.shape[0] - size for r, _ in origins)
        return "vertical", origins

    if sep_w == 1 and sep_h == size:
        cols = sorted(box[1] for box in boxes)
        r0s = {box[0] for box in boxes}
        r1s = {box[2] for box in boxes}
        assert len(r0s) == 1 and len(r1s) == 1, "vertical separators must align"
        assert all(box[1] == box[3] for box in boxes), "separator columns must be one pixel wide"
        diffs = [cols[i + 1] - cols[i] for i in range(len(cols) - 1)]
        assert len(set(diffs)) == 1 and diffs[0] == size + 1, "separator spacing must match slot size"
        start_c = cols[0] - size
        r0 = next(iter(r0s))
        origins = [(r0, start_c + i * (size + 1)) for i in range(len(cols) + 1)]
        assert all(0 <= c <= grid.shape[1] - size for _, c in origins)
        return "horizontal", origins

    raise AssertionError("separator dimensions do not match a valid slot sequence")


def _render_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    sprite = np.zeros(mask.shape, dtype=int)
    sprite[mask] = color
    return sprite


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    bg = int(get_bg(g))
    assert bg == BACKGROUND, "expected black background"
    fg_colors = colors_present(g, exclude={bg, SEPARATOR})
    assert len(fg_colors) == 1, "expected one sprite color plus grey separators"
    fg = int(fg_colors[0])

    template_box, template_mask = _find_template(g, fg)
    size = _square_dims(template_box)
    rotations = _rotation_family(template_mask)
    orientation, origins = _find_slot_origins(g, size)
    assert orientation in {"vertical", "horizontal"}

    occupied = 0
    empty = 0
    out = g.copy()
    for slot_i, (r0, c0) in enumerate(origins):
        slot = g[r0 : r0 + size, c0 : c0 + size]
        slot_colors = set(map(int, np.unique(slot)))
        assert slot_colors <= {bg, fg}, "slot interiors may only contain background or sprite color"
        expected_mask = rotations[slot_i % 4]
        if fg in slot_colors:
            occupied += 1
            matches = [idx for idx, rot in enumerate(rotations) if np.array_equal(slot == fg, rot)]
            assert matches == [slot_i % 4], "occupied slot rotation must match its absolute sequence index"
        else:
            empty += 1
            out = paste(out, _render_sprite(expected_mask, fg), r0, c0, transparent=bg)

    fg_comps = find_components(
        g,
        bg=BACKGROUND,
        colors=[fg],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert len(fg_comps) == occupied + 1, "foreground components should be the template plus one per filled slot"
    assert occupied >= 3 and empty >= 1, "input must expose both filled and empty slots"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each empty slot with the template sprite rotated by its absolute slot index.

    Input structure:
    - The background is black, with one small monochrome template sprite near the upper-left.
    - Elsewhere, 5-8 same-size square slots form a horizontal or vertical sequence.
    - Adjacent slots are separated by one-pixel grey divider lines, and some slots already contain rotated copies of the template.

    Rule:
    - Read the template sprite from its tight 3x3 or 4x4 bounding box.
    - Recover the ordered slot sequence from the evenly spaced grey separators.
    - Slot `i` must contain the template rotated clockwise by `i * 90` degrees modulo 360.
    - Leave the template, every already-correct filled slot, and all grey separators unchanged; fill each empty slot with its required rotated copy.

    Key insight:
    - The correct rotation is tied to the slot's absolute position in the evenly spaced sequence, not just to a local neighbor difference.
    """
    return _solve(grid)


def _has_forbidden_reflection(mask: np.ndarray) -> bool:
    reflections = [
        np.fliplr(mask),
        np.flipud(mask),
        mask.T,
        np.fliplr(mask.T),
        np.flipud(mask.T),
    ]
    return any(np.array_equal(mask, ref) for ref in reflections)


def _random_template_mask(size: int) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        density = random.uniform(0.35, 0.65)
        mask = (np.random.rand(size, size) < density)
        if mask.sum() < size + 1 or mask.sum() > size * size - 2:
            continue
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            continue
        if rows[0] != 0 or rows[-1] != size - 1 or cols[0] != 0 or cols[-1] != size - 1:
            continue
        if len(components_mask(mask, connectivity=4)) != 1:
            continue
        if len({_mask_key(rot) for rot in _rotation_family(mask)}) != 4:
            continue
        if _has_forbidden_reflection(mask):
            continue
        return mask.astype(bool)
    return None


def _sample_empty_slots(slot_count: int) -> set[int] | None:
    for _ in range(PATTERN_ATTEMPTS):
        empty_count = random.randint(1, min(4, slot_count - 2))
        empties = set(random.sample(range(slot_count), empty_count))
        occupied = [i for i in range(slot_count) if i not in empties]
        residues = {i % 4 for i in occupied}
        if len(occupied) < 3 or len(residues) < 2:
            continue
        if slot_count >= 5 and random.random() < 0.7:
            if not any((i + 4) in occupied for i in occupied if i + 4 < slot_count):
                continue
        if len(occupied) == slot_count or len(occupied) == 0:
            continue
        if occupied == list(range(min(occupied), max(occupied) + 1)) and random.random() < 0.7:
            continue
        return empties
    return None


def _build_vertical_case(size: int, slot_count: int, template_mask: np.ndarray, fg: int) -> np.ndarray | None:
    template_r0 = random.randint(0, 2)
    template_c0 = random.randint(0, 2)
    gap = random.randint(1, 3)
    max_slots = (30 - template_r0 - size - gap + 1) // (size + 1)
    if max_slots < slot_count:
        return None
    seq_r0 = template_r0 + size + gap
    seq_c0 = random.randint(template_c0, min(template_c0 + 4, 30 - size))
    total_seq_h = slot_count * size + (slot_count - 1)
    base_h = seq_r0 + total_seq_h
    if base_h > 30:
        return None
    bottom_pad = random.randint(0, min(2, 30 - base_h))
    width_base = max(template_c0 + size, seq_c0 + size)
    right_pad = random.randint(0, min(2, 30 - width_base))
    grid = np.zeros((base_h + bottom_pad, width_base + right_pad), dtype=int)

    template = _render_sprite(template_mask, fg)
    grid = paste(grid, template, template_r0, template_c0)
    rotations = _rotation_family(template_mask)
    empties = _sample_empty_slots(slot_count)
    if empties is None:
        return None

    for i in range(slot_count):
        r0 = seq_r0 + i * (size + 1)
        if i not in empties:
            grid = paste(grid, _render_sprite(rotations[i % 4], fg), r0, seq_c0, transparent=BACKGROUND)
        if i < slot_count - 1:
            grid[r0 + size, seq_c0 : seq_c0 + size] = SEPARATOR
    return grid


def _build_horizontal_case(size: int, slot_count: int, template_mask: np.ndarray, fg: int) -> np.ndarray | None:
    template_r0 = random.randint(0, 2)
    template_c0 = random.randint(0, 2)
    gap = random.randint(1, 3)
    max_slots = (30 - template_c0 - size - gap + 1) // (size + 1)
    if max_slots < slot_count:
        return None
    seq_c0 = template_c0 + size + gap
    seq_r0 = random.randint(template_r0, min(template_r0 + 4, 30 - size))
    total_seq_w = slot_count * size + (slot_count - 1)
    base_w = seq_c0 + total_seq_w
    if base_w > 30:
        return None
    right_pad = random.randint(0, min(2, 30 - base_w))
    height_base = max(template_r0 + size, seq_r0 + size)
    bottom_pad = random.randint(0, min(2, 30 - height_base))
    grid = np.zeros((height_base + bottom_pad, base_w + right_pad), dtype=int)

    template = _render_sprite(template_mask, fg)
    grid = paste(grid, template, template_r0, template_c0)
    rotations = _rotation_family(template_mask)
    empties = _sample_empty_slots(slot_count)
    if empties is None:
        return None

    for i in range(slot_count):
        c0 = seq_c0 + i * (size + 1)
        if i not in empties:
            grid = paste(grid, _render_sprite(rotations[i % 4], fg), seq_r0, c0, transparent=BACKGROUND)
        if i < slot_count - 1:
            grid[seq_r0 : seq_r0 + size, c0 + size] = SEPARATOR
    return grid


def _expected_output(grid: np.ndarray) -> np.ndarray:
    return _solve(grid)


def _gen_attempt() -> np.ndarray | None:
    size = random.choice([3, 4])
    orientation = random.choice(["vertical", "horizontal"])
    slot_count = random.randint(5, 8)
    if size == 4:
        slot_count = 5

    template_mask = _random_template_mask(size)
    if template_mask is None:
        return None
    fg = random.choice(SPRITE_COLORS)

    if orientation == "vertical":
        grid = _build_vertical_case(size, slot_count, template_mask, fg)
    else:
        grid = _build_horizontal_case(size, slot_count, template_mask, fg)
    if grid is None:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None

    expected = grid.copy()
    template_rotations = _rotation_family(template_mask)
    grey_comps = find_components(
        grid,
        bg=BACKGROUND,
        colors=[SEPARATOR],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    slot_total = len(grey_comps) + 1
    _, origins = _find_slot_origins(grid, size)
    for i, (r0, c0) in enumerate(origins):
        slot = grid[r0 : r0 + size, c0 : c0 + size]
        if not np.any(slot == fg):
            expected = paste(expected, _render_sprite(template_rotations[i % 4], fg), r0, c0, transparent=BACKGROUND)
    if len(origins) != slot_total:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid equal-difference rotation-completion input")
