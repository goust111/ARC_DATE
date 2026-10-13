from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, components_mask, crop_bbox


Coord = tuple[int, int]
Shape = tuple[Coord, ...]
Placement = tuple[int, int, frozenset[Coord], int]

BLACK = Color.BLACK
RED = Color.RED
GREEN = Color.GREEN
GREY = Color.GREY

PANEL_COLORS = [Color.BLUE, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GEN_RETRIES = 500


def _normalize_coords(coords: np.ndarray) -> Shape:
    pts = np.asarray(coords, dtype=int)
    pts = pts - pts.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in pts))


def _mask_to_shape(mask: np.ndarray) -> Shape:
    coords = np.argwhere(mask)
    assert coords.size > 0
    return _normalize_coords(coords)


def _shape_size(shape: Shape) -> tuple[int, int]:
    return max(r for r, _c in shape) + 1, max(c for _r, c in shape) + 1


def _shape_to_mask(shape: Shape) -> np.ndarray:
    h, w = _shape_size(shape)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _dihedral_variants(mask: np.ndarray) -> list[Shape]:
    seen: set[Shape] = set()
    out: list[Shape] = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        for candidate in (rotated, np.fliplr(rotated)):
            shape = _mask_to_shape(candidate)
            if shape not in seen:
                seen.add(shape)
                out.append(shape)
    return out


def _placement_key(entry: Placement) -> tuple[int, int, int, tuple[Coord, ...]]:
    r, c, cells, _variant_idx = entry
    return (r, c, len(cells), tuple(sorted(cells)))


def _cells_at(shape: Shape, r0: int, c0: int) -> frozenset[Coord]:
    return frozenset((r0 + dr, c0 + dc) for dr, dc in shape)


def _all_placements(variants: list[Shape], H: int, W: int) -> list[Placement]:
    out: list[Placement] = []
    seen: set[tuple[Coord, ...]] = set()
    for variant_idx, shape in enumerate(variants):
        h, w = _shape_size(shape)
        for r in range(H - h + 1):
            for c in range(W - w + 1):
                cells = _cells_at(shape, r, c)
                key = tuple(sorted(cells))
                if key in seen:
                    continue
                seen.add(key)
                out.append((r, c, cells, variant_idx))
    out.sort(key=_placement_key)
    return out


def _find_valid_placements(field: np.ndarray, variants: list[Shape]) -> list[Placement]:
    H, W = field.shape
    out: list[Placement] = []
    seen: set[tuple[Coord, ...]] = set()
    for variant_idx, shape in enumerate(variants):
        h, w = _shape_size(shape)
        for r in range(H - h + 1):
            for c in range(W - w + 1):
                cells = _cells_at(shape, r, c)
                if not all(int(field[rr, cc]) == GREEN for rr, cc in cells):
                    continue
                key = tuple(sorted(cells))
                if key in seen:
                    continue
                seen.add(key)
                out.append((r, c, cells, variant_idx))
    out.sort(key=_placement_key)
    return out


def _select_non_overlapping(entries: list[Placement]) -> list[Placement]:
    selected: list[Placement] = []
    occupied: set[Coord] = set()
    for entry in sorted(entries, key=_placement_key):
        cells = entry[2]
        if occupied.isdisjoint(cells):
            selected.append(entry)
            occupied.update(cells)
    return selected


def _extract_layout(grid: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    divider_rows = [r for r in range(H) if np.all(g[r, :] == GREY)]
    assert len(divider_rows) == 1
    hdiv = divider_rows[0]
    top = g[:hdiv, :]
    bottom = g[hdiv + 1 :, :]
    assert top.size > 0 and bottom.size > 0

    divider_cols = [c for c in range(W) if np.all(top[:, c] == GREY)]
    assert len(divider_cols) == 1
    vdiv = divider_cols[0]
    left = top[:, :vdiv]
    right = top[:, vdiv + 1 :]
    assert left.shape == right.shape and left.size > 0
    return hdiv, vdiv, left, right, bottom


def _recover_template(grid: np.ndarray) -> np.ndarray:
    _hdiv, _vdiv, left, right, _bottom = _extract_layout(grid)
    left_mask = left != BLACK
    right_mask = right != BLACK
    template_full = left_mask & right_mask
    box = bbox(template_full)
    assert box is not None
    return crop_bbox(template_full.astype(int), box).astype(bool)


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden overlap template from the two top masks and recolor its green copies red in the lower field.

    Input structure:
    - A horizontal grey divider splits the grid into a top band and a lower field, and a vertical grey divider splits the top band into equal left and right panels.
    - Each top panel has black background plus one non-grey foreground color; neither panel alone is the target motif.
    - The lower field is a black-and-green noise field containing some rotated or reflected copies of the hidden motif plus unrelated green distractors.

    Rule:
    - Convert the left and right top panels to binary masks and intersect them at corresponding coordinates.
    - Crop that overlap mask to its bounding box, enumerate all distinct rotations and reflections, and scan the lower field in row-major order for placements whose motif cells are all green.
    - Recolor every selected placement red, skipping any later valid placement that overlaps already selected cells, and leave all other pixels unchanged.

    Key insight:
    - The target shape is never drawn directly; it only appears as the intersection of the two masked panels, and that recovered motif is the key for matching the lower-field copies.
    """
    g = np.asarray(input_grid, dtype=int)
    assert g.ndim == 2

    hdiv, _vdiv, _left, _right, bottom = _extract_layout(g)
    template = _recover_template(g)
    variants = _dihedral_variants(template)
    candidates = _find_valid_placements(bottom, variants)
    selected = _select_non_overlapping(candidates)

    out = g.copy()
    for r0, c0, cells, _variant_idx in selected:
        for rr, cc in cells:
            out[hdiv + 1 + rr, cc] = RED
    return out


def _random_connected_template() -> np.ndarray:
    for _ in range(200):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        max_area = min(h * w - 1, 7)
        min_area = min(4, max_area)
        if min_area > max_area:
            continue
        area = random.randint(min_area, max_area)

        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        while int(mask.sum()) < area:
            frontier: list[Coord] = []
            for r, c in np.argwhere(mask):
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = int(r + dr), int(c + dc)
                        if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                            frontier.append((rr, cc))
            if not frontier:
                break
            rr, cc = random.choice(frontier)
            mask[rr, cc] = True

        if int(mask.sum()) != area:
            continue
        if len(components_mask(mask, connectivity=8)) != 1:
            continue
        if bbox(mask) != (0, 0, h - 1, w - 1):
            continue
        if mask.all():
            continue
        return mask
    raise RuntimeError("failed to generate template")


def _neighbor_cells(mask: np.ndarray) -> list[Coord]:
    h, w = mask.shape
    neighbors: set[Coord] = set()
    for r, c in np.argwhere(mask):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                    neighbors.add((rr, cc))
    return sorted(neighbors)


def _grow_extra(mask: np.ndarray, forbidden: np.ndarray, area: int) -> np.ndarray | None:
    h, w = mask.shape
    seeds = [cell for cell in _neighbor_cells(mask) if not forbidden[cell]]
    if not seeds:
        return None
    extra = np.zeros((h, w), dtype=bool)
    sr, sc = random.choice(seeds)
    extra[sr, sc] = True
    while int(extra.sum()) < area:
        frontier: list[Coord] = []
        for r, c in np.argwhere(extra | mask):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = int(r + dr), int(c + dc)
                    if 0 <= rr < h and 0 <= cc < w and not forbidden[rr, cc] and not extra[rr, cc] and not mask[rr, cc]:
                        frontier.append((rr, cc))
        if not frontier:
            return None
        rr, cc = random.choice(frontier)
        extra[rr, cc] = True
    return extra


def _build_panel_masks(template: np.ndarray, panel_h: int, panel_w: int) -> tuple[np.ndarray, np.ndarray]:
    th, tw = template.shape
    for _ in range(200):
        r0 = random.randint(1, panel_h - th - 1)
        c0 = random.randint(1, panel_w - tw - 1)
        base = np.zeros((panel_h, panel_w), dtype=bool)
        base[r0 : r0 + th, c0 : c0 + tw] = template

        left_area = random.randint(1, 4)
        right_area = random.randint(1, 4)
        left_extra = _grow_extra(base, base.copy(), left_area)
        if left_extra is None:
            continue
        forbidden = base | left_extra
        right_extra = _grow_extra(base, forbidden, right_area)
        if right_extra is None:
            continue

        left_mask = base | left_extra
        right_mask = base | right_extra
        if np.array_equal(left_mask, base) or np.array_equal(right_mask, base):
            continue

        template_variants = set(_dihedral_variants(template))
        if _mask_to_shape(crop_bbox(left_mask.astype(int), bbox(left_mask)).astype(bool)) in template_variants:
            continue
        if _mask_to_shape(crop_bbox(right_mask.astype(int), bbox(right_mask)).astype(bool)) in template_variants:
            continue
        if bbox(left_mask & right_mask) is None:
            continue
        return left_mask, right_mask
    raise RuntimeError("failed to build panel masks")


def _add_safe_subset(
    green_cells: set[Coord],
    subset: set[Coord],
    allowed_sets: set[frozenset[Coord]],
    all_sets: list[frozenset[Coord]],
) -> bool:
    trial = set(green_cells)
    trial.update(subset)
    for placement in all_sets:
        if placement not in allowed_sets and placement.issubset(trial):
            return False
    green_cells.update(subset)
    return True


def _choose_allowed_placements(placements: list[Placement]) -> list[Placement]:
    assert placements
    placements = sorted(placements, key=_placement_key)
    chosen: list[Placement] = []
    occupied: set[Coord] = set()

    want_special = random.random() < 0.7
    special_entries = [entry for entry in placements if entry[3] != 0]
    if want_special and special_entries:
        seed = random.choice(special_entries)
        chosen.append(seed)
        occupied.update(seed[2])

    non_overlap_target = random.choice([1, 2, 2, 3, 4])
    shuffled = placements[:]
    random.shuffle(shuffled)
    for entry in shuffled:
        if entry[2].isdisjoint(occupied):
            chosen.append(entry)
            occupied.update(entry[2])
            if len(_select_non_overlapping(chosen)) >= non_overlap_target:
                break

    if not chosen:
        chosen = [random.choice(placements)]

    if random.random() < 0.45:
        selected_now = _select_non_overlapping(chosen)
        overlap_options = [
            entry
            for entry in placements
            if entry not in chosen
            and any(entry[2] & earlier[2] and (entry[0], entry[1]) > (earlier[0], earlier[1]) for earlier in selected_now)
        ]
        if overlap_options:
            chosen.append(random.choice(overlap_options))

    unique: dict[frozenset[Coord], Placement] = {}
    for entry in chosen:
        unique.setdefault(entry[2], entry)
    return sorted(unique.values(), key=_placement_key)


def _compose_expected(inp: np.ndarray, selected: list[Placement], hdiv: int) -> np.ndarray:
    out = inp.copy()
    for r0, c0, cells, _variant_idx in selected:
        for rr, cc in cells:
            out[hdiv + 1 + rr, cc] = RED
    return out


def _generate_one() -> np.ndarray | None:
    template = _random_connected_template()
    th, tw = template.shape
    panel_h = random.randint(max(th + 2, 5), min(th + 5, 9))
    panel_w = random.randint(max(tw + 2, 5), min(tw + 5, 10))
    field_h = random.randint(8, min(17, 30 - panel_h - 1))
    total_w = 2 * panel_w + 1

    left_mask, right_mask = _build_panel_masks(template, panel_h, panel_w)
    left_color, right_color = random.sample(PANEL_COLORS, 2)

    grid = np.zeros((panel_h + 1 + field_h, total_w), dtype=int)
    grid[panel_h, :] = GREY
    grid[:panel_h, panel_w] = GREY
    grid[:panel_h, :panel_w][left_mask] = left_color
    grid[:panel_h, panel_w + 1 :][right_mask] = right_color

    variants = _dihedral_variants(template)
    all_placements = _all_placements(variants, field_h, total_w)
    if not all_placements:
        return None

    allowed_entries = _choose_allowed_placements(all_placements)
    allowed_sets = {entry[2] for entry in allowed_entries}
    all_sets = [entry[2] for entry in all_placements]

    green_cells: set[Coord] = set()
    for entry in allowed_entries:
        green_cells.update(entry[2])

    near_match_goal = random.randint(1, 3)
    forbidden_entries = [entry for entry in all_placements if entry[2] not in allowed_sets]
    random.shuffle(forbidden_entries)
    near_matches = 0
    for entry in forbidden_entries:
        if near_matches >= near_match_goal:
            break
        cells = list(entry[2])
        random.shuffle(cells)
        missing = 1 if len(cells) <= 4 else random.choice([1, 1, 2])
        subset = set(cells[:-missing])
        if subset.issubset(green_cells):
            continue
        if _add_safe_subset(green_cells, subset, allowed_sets, all_sets):
            near_matches += 1

    singleton_goal = random.randint(1, 5)
    free_cells = [(r, c) for r in range(field_h) for c in range(total_w) if (r, c) not in green_cells]
    random.shuffle(free_cells)
    singleton_count = 0
    for cell in free_cells:
        if singleton_count >= singleton_goal:
            break
        if _add_safe_subset(green_cells, {cell}, allowed_sets, all_sets):
            singleton_count += 1

    field = np.zeros((field_h, total_w), dtype=int)
    for r, c in green_cells:
        field[r, c] = GREEN
    grid[panel_h + 1 :, :] = field

    recovered = _recover_template(grid)
    if not np.array_equal(recovered, template):
        return None

    parsed_variants = _dihedral_variants(recovered)
    found_entries = _find_valid_placements(field, parsed_variants)
    if {entry[2] for entry in found_entries} != allowed_sets:
        return None

    selected = _select_non_overlapping(found_entries)
    selected_union = set().union(*(entry[2] for entry in selected)) if selected else set()
    if not selected_union:
        return None
    if not (green_cells - selected_union):
        return None

    expected = _compose_expected(grid, selected, panel_h)
    if np.any(grid == RED):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
