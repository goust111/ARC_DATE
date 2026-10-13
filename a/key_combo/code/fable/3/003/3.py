from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, components_mask, crop_bbox, find_components, paste


BG = Color.BLACK
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GENERATION_RETRIES = 400
_GEN_COUNTER = 0


def _source_and_marker(g: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int], int]:
    comps = find_components(g, bg=BG, connectivity=4, monochromatic=False, return_format="dict")
    source = [comp for comp in comps if int(comp["area"]) > 1]
    markers = [comp for comp in comps if int(comp["area"]) == 1]
    assert len(source) == 1, "expected one nontrivial multicolor sprite"
    assert len(markers) == 1, "expected one isolated target marker pixel"

    marker = tuple(map(int, markers[0]["coords"][0]))
    marker_color = int(g[marker])
    assert marker_color != BG, "marker must be non-background"
    return source[0]["coords"], source[0]["bbox"], marker, marker_color


def _defect_from_symmetry(
    g: np.ndarray,
    source_bbox: tuple[int, int, int, int],
    marker_color: int,
) -> tuple[tuple[int, int], tuple[int, int], int]:
    r0, c0, _r1, _c1 = source_bbox
    sprite = crop_bbox(g, source_bbox)
    h, w = sprite.shape

    mismatches: list[tuple[int, int, int, int, int, int]] = []
    for rr in range(h):
        for cc in range(w // 2):
            mirror_c = w - 1 - cc
            left = int(sprite[rr, cc])
            right = int(sprite[rr, mirror_c])
            if left != right:
                mismatches.append((rr, cc, rr, mirror_c, left, right))

    assert len(mismatches) == 1, "sprite must have exactly one left-right symmetry-breaking pair"
    rr, cc, _mr, mirror_c, left, right = mismatches[0]

    if left == marker_color and right != marker_color:
        local_defect = (rr, cc)
        repair_color = right
    elif right == marker_color and left != marker_color:
        local_defect = (rr, mirror_c)
        repair_color = left
    else:
        raise AssertionError("the marker color must identify exactly one side of the broken pair")

    assert repair_color != BG, "the broken color must be repaired to a real sprite color"
    absolute_defect = (r0 + local_defect[0], c0 + local_defect[1])
    return local_defect, absolute_defect, int(repair_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a mirror-breaking sprite pixel and copy the repaired sprite onto its matching target marker.

    Input structure:
    - A black grid contains one 4-connected multicolor sprite and one isolated non-black marker pixel.
    - The sprite's bounding box has left-right mirror symmetry except for one mirrored pair of cells.
    - The true defective cell is the side of that pair whose color matches the isolated marker pixel.

    Rule:
    - Detect the unique left-right symmetry-breaking pair inside the sprite bbox.
    - Replace the defective cell by the color of its mirror cell.
    - Translate-copy the repaired sprite so that the repaired defect position lands on the isolated marker.
    - Keep the repaired original sprite in place and leave all other background cells unchanged.

    Key insight:
    - The copy anchor is not explicitly drawn inside the sprite; it is the pixel revealed by the
      one symmetry break, and the isolated same-color pixel gives the matching destination.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    H, W = g.shape

    _source_coords, source_bbox, marker, marker_color = _source_and_marker(g)
    local_defect, absolute_defect, repair_color = _defect_from_symmetry(g, source_bbox, marker_color)

    out = g.copy()
    out[absolute_defect] = repair_color

    r0, c0, _r1, _c1 = source_bbox
    repaired_sprite = crop_bbox(out, source_bbox)
    mask = repaired_sprite != BG
    assert bool(mask[local_defect]), "the repaired anchor must still be part of the sprite"

    dst_r0 = int(marker[0] - local_defect[0])
    dst_c0 = int(marker[1] - local_defect[1])
    h, w = repaired_sprite.shape
    assert 0 <= dst_r0 and dst_r0 + h <= H and 0 <= dst_c0 and dst_c0 + w <= W, "translated sprite must remain in bounds"

    for rr, cc in np.argwhere(mask):
        tr = int(dst_r0 + rr)
        tc = int(dst_c0 + cc)
        if (tr, tc) == marker:
            continue
        assert int(g[tr, tc]) == BG, "translated sprite must land on background except for the marker"

    paste(out, repaired_sprite, dst_r0, dst_c0, transparent=BG)
    return out


def _is_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(components_mask(mask, connectivity=4)) == 1


def _touches_all_edges(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _mirror_groups(h: int, w: int) -> list[list[tuple[int, int]]]:
    groups: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range((w + 1) // 2):
            mc = w - 1 - c
            if c == mc:
                groups.append([(r, c)])
            else:
                groups.append([(r, c), (r, mc)])
    return groups


def _make_symmetric_mask(h: int, w: int) -> np.ndarray | None:
    mask = np.ones((h, w), dtype=bool)
    groups = _mirror_groups(h, w)
    random.shuffle(groups)
    target_area = random.randint(max(9, int(h * w * 0.45)), max(9, int(h * w * 0.78)))

    for group in groups:
        if int(mask.sum()) - len(group) < target_area:
            continue
        if random.random() < 0.2:
            continue
        trial = mask.copy()
        for r, c in group:
            trial[r, c] = False
        if _touches_all_edges(trial) and _is_connected(trial):
            mask = trial

    if not (_touches_all_edges(mask) and _is_connected(mask)):
        return None
    noncenter_pairs = [(r, c) for r in range(h) for c in range(w // 2) if mask[r, c] and mask[r, w - 1 - c]]
    if len(noncenter_pairs) < 2:
        return None
    return mask


def _color_symmetric_sprite(mask: np.ndarray, base_colors: list[int]) -> np.ndarray:
    h, w = mask.shape
    sprite = np.full((h, w), BG, dtype=int)
    colored_groups: list[list[tuple[int, int]]] = []

    for group in _mirror_groups(h, w):
        if all(mask[r, c] for r, c in group):
            color = random.choice(base_colors)
            for r, c in group:
                sprite[r, c] = color
            colored_groups.append(group)

    random.shuffle(colored_groups)
    if len(colored_groups) >= 2:
        for color, group in zip(base_colors[:2], colored_groups[:2]):
            for r, c in group:
                sprite[r, c] = color
    assert len(set(map(int, sprite[sprite != BG]))) >= 2, "sprite must be visibly multicolor"
    return sprite


def _make_sprite_with_defect(force_left: bool) -> tuple[np.ndarray, tuple[int, int], int, int] | None:
    h = random.randint(4, 8)
    w = random.randint(4, 9)
    mask = _make_symmetric_mask(h, w)
    if mask is None:
        return None

    color_count = random.randint(2, 4)
    colors = random.sample(PALETTE, color_count + 1)
    base_colors = colors[:-1]
    defect_color = colors[-1]
    sprite = _color_symmetric_sprite(mask, base_colors)

    candidates = [(r, c) for r in range(h) for c in range(w // 2) if mask[r, c] and mask[r, w - 1 - c]]
    if not candidates:
        return None
    r, left_c = random.choice(candidates)
    right_c = w - 1 - left_c
    defect_c = left_c if force_left else right_c
    mirror_c = right_c if force_left else left_c
    repair_color = int(sprite[r, mirror_c])
    sprite[r, defect_c] = defect_color
    return sprite, (int(r), int(defect_c)), int(repair_color), int(defect_color)


def _separated(a: tuple[int, int], b: tuple[int, int], h: int, w: int, gap: int = 1) -> bool:
    ar, ac = a
    br, bc = b
    return ar + h + gap <= br or br + h + gap <= ar or ac + w + gap <= bc or bc + w + gap <= ac


def _placement_pair(shape: tuple[int, int], sprite_shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    H, W = shape
    h, w = sprite_shape
    origins = [(r, c) for r in range(1, H - h) for c in range(1, W - w)]
    random.shuffle(origins)
    for src in origins:
        dests = [dst for dst in origins if _separated(src, dst, h, w, gap=1)]
        if dests:
            return src, random.choice(dests)
    return None


def _expected_output(
    shape: tuple[int, int],
    sprite: np.ndarray,
    defect: tuple[int, int],
    repair_color: int,
    defect_color: int,
    source: tuple[int, int],
    dest: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros(shape, dtype=int)
    paste(inp, sprite, source[0], source[1], transparent=BG)
    marker = (dest[0] + defect[0], dest[1] + defect[1])
    inp[marker] = defect_color

    repaired = sprite.copy()
    repaired[defect] = repair_color
    out = inp.copy()
    out[source[0] + defect[0], source[1] + defect[1]] = repair_color
    paste(out, repaired, dest[0], dest[1], transparent=BG)
    return inp, out


def _valid_generation(
    inp: np.ndarray,
    expected: np.ndarray,
    defect_color: int,
    repair_color: int,
    source: tuple[int, int],
    dest: tuple[int, int],
    defect: tuple[int, int],
) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if int(np.count_nonzero(inp == defect_color)) != 2:
        return False

    source_defect = (source[0] + defect[0], source[1] + defect[1])
    marker = (dest[0] + defect[0], dest[1] + defect[1])
    if int(inp[source_defect]) != defect_color or int(inp[marker]) != defect_color:
        return False
    if int(expected[source_defect]) != repair_color or int(expected[marker]) != repair_color:
        return False

    try:
        comps = find_components(inp, bg=BG, connectivity=4, monochromatic=False, return_format="dict")
        if sorted(int(comp["area"]) for comp in comps)[0] != 1 or len(comps) != 2:
            return False
        local, absolute, found_repair = _defect_from_symmetry(inp, [comp for comp in comps if int(comp["area"]) > 1][0]["bbox"], defect_color)
        if local != defect or absolute != source_defect or found_repair != repair_color:
            return False
        solved = main(inp)
    except Exception:
        return False
    return np.array_equal(solved, expected) and not np.array_equal(inp, expected)


def _gen_attempt(force_left: bool) -> np.ndarray | None:
    made = _make_sprite_with_defect(force_left)
    if made is None:
        return None
    sprite, defect, repair_color, defect_color = made
    h, w = sprite.shape

    min_h = max(12, h * 2 + 4)
    min_w = max(12, w * 2 + 4)
    H = random.randint(min_h, 30)
    W = random.randint(min_w, 30)

    placed = _placement_pair((H, W), (h, w))
    if placed is None:
        return None
    source, dest = placed
    inp, expected = _expected_output((H, W), sprite, defect, repair_color, defect_color, source, dest)
    if not _valid_generation(inp, expected, defect_color, repair_color, source, dest, defect):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    force_left = (_GEN_COUNTER % 2) == 0
    _GEN_COUNTER += 1
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(force_left)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for symmetry defect anchor")
