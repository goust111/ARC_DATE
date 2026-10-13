from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, crop_bbox, in_bounds, neighbors4


Coord = tuple[int, int]
Shape = tuple[Coord, ...]


TEMPLATE_CATALOG: tuple[Shape, ...] = (
    ((0, 0), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
)

GEN_ATTEMPTS = 500
REGION_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 160
GRID_HEIGHT_RANGE = (18, 25)
GRID_WIDTH_RANGE = (19, 27)
REGION_HEIGHT_RANGE = (7, 10)
REGION_WIDTH_RANGE = (7, 10)
REGION_MASK_AREA_RANGE = (38, 49)
MIN_REGION_FILL_AREA = 30
MAX_REGION_FILL_AREA = 50
MIN_REGION_SEPARATION = 1
MIN_TEMPLATE_SEPARATION = 1
MIN_INTERIOR_CELLS = 12


def _normalize(coords: np.ndarray | Shape) -> Shape:
    pts = np.asarray(coords, dtype=int)
    pts = pts - pts.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in pts))


def _shape_size(shape: Shape) -> tuple[int, int]:
    return max(r for r, _c in shape) + 1, max(c for _r, c in shape) + 1


def _dihedral_variants(shape: Shape) -> list[Shape]:
    pts = np.asarray(shape, dtype=int)
    seen: set[Shape] = set()
    variants: list[Shape] = []
    for rotation in range(4):
        transformed = pts.copy()
        for _ in range(rotation):
            transformed = np.column_stack((-transformed[:, 1], transformed[:, 0]))
        for reflected in (False, True):
            candidate = transformed.copy()
            if reflected:
                candidate[:, 1] = -candidate[:, 1]
            normalized = _normalize(candidate)
            if normalized not in seen:
                seen.add(normalized)
                variants.append(normalized)
    variants.sort(key=lambda sh: (_shape_size(sh), sh))
    return variants


def _template_shape(grid: np.ndarray) -> Shape:
    comps = components_mask(grid == Color.TEAL, connectivity=4)
    assert len(comps) == 1, "expected one cyan template"
    box = bbox_coords(comps[0])
    crop = crop_bbox(grid == Color.TEAL, box)
    return tuple(sorted((int(r), int(c)) for r, c in np.argwhere(crop)))


def _shape_cells(shape: Shape, r0: int, c0: int) -> frozenset[Coord]:
    return frozenset((r0 + dr, c0 + dc) for dr, dc in shape)


def _region_defects(grid: np.ndarray, region_color: int) -> set[Coord]:
    defects: set[Coord] = set()
    for comp in components_mask(grid == Color.BLACK, connectivity=4):
        cells = {(int(r), int(c)) for r, c in comp}
        if any(r == 0 or c == 0 or r == grid.shape[0] - 1 or c == grid.shape[1] - 1 for r, c in cells):
            continue
        boundary_colors: set[int] = set()
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) not in cells:
                    boundary_colors.add(int(grid[nr, nc]))
        if boundary_colors == {int(region_color)}:
            defects.update(cells)
    return defects


def _match_candidates(defects: set[Coord], variants: list[Shape], grid_shape: tuple[int, int]) -> list[tuple[tuple[int, int], frozenset[Coord]]]:
    H, W = map(int, grid_shape)
    matches: list[tuple[tuple[int, int], frozenset[Coord]]] = []
    seen: set[frozenset[Coord]] = set()
    for variant in variants:
        h, w = _shape_size(variant)
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                cells = _shape_cells(variant, r0, c0)
                if cells in seen:
                    continue
                if cells.issubset(defects):
                    seen.add(cells)
                    matches.append(((r0, c0), cells))
    matches.sort(key=lambda item: (item[0][0], item[0][1]))
    return matches


def _other_region_color(region_color: int) -> int:
    return Color.RED if int(region_color) == Color.BLUE else Color.BLUE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the cyan defect template to outline only the matching black-hole cluster in the opposite region color.

    Input structure:
    - Black background contains exactly two large filled regions, one blue and one red, plus one small cyan template in a corner.
    - Each large region is 8-connected and contains several single-cell black dents whose 4-neighbors all belong to that region.
    - The cyan template is a 4-connected shape; its bbox-normalized mask may reappear among a region's dents after rotation or reflection.

    Rule:
    - Extract the cyan template mask, enumerate its unique rotations and reflections, and search each region's dent set for the first row-major exact placement of any variant.
    - Keep the matched dent cells black, repaint every 4-neighbor region cell around those dents to the other large-region color, and fill every other dent back with its own region color.
    - Erase the cyan template to black and leave every unrelated pixel unchanged.

    Key insight:
    The template does not mark foreground pixels to copy; it classifies which arrangement of missing pixels is special, and only that defect signature receives opposite-color outlining instead of repair.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[g == Color.TEAL] = Color.BLACK

    template = _template_shape(g)
    variants = _dihedral_variants(template)
    assert len(variants) >= 1, "template must be nonempty"

    for region_color in (Color.BLUE, Color.RED):
        defects = _region_defects(g, region_color)
        matches = _match_candidates(defects, variants, g.shape)
        assert matches, f"expected one template match inside region {region_color}"
        matched = set(matches[0][1])
        repaired = defects - matched
        for r, c in repaired:
            out[r, c] = int(region_color)

        outline: set[Coord] = set()
        for r, c in matched:
            for nr, nc in neighbors4(r, c):
                if in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == int(region_color):
                    outline.add((nr, nc))
        opposite = _other_region_color(region_color)
        for r, c in outline:
            out[r, c] = opposite

    return out.astype(int)


def _interior_cells(mask: np.ndarray) -> set[Coord]:
    interior: set[Coord] = set()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if all(in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) for nr, nc in neighbors4(rr, cc)):
            interior.add((rr, cc))
    return interior


def _boundary_cells(mask: np.ndarray) -> list[Coord]:
    cells: list[Coord] = []
    H, W = mask.shape
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if rr in (0, H - 1) or cc in (0, W - 1):
            cells.append((rr, cc))
            continue
        if any(not bool(mask[nr, nc]) for nr, nc in neighbors4(rr, cc)):
            cells.append((rr, cc))
    return cells


def _mask_cells(mask: np.ndarray, r0: int = 0, c0: int = 0) -> set[Coord]:
    return {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)}


def _separated(a: set[Coord], b: set[Coord], margin: int) -> bool:
    for ar, ac in a:
        for br, bc in b:
            if max(abs(ar - br), abs(ac - bc)) <= margin:
                return False
    return True


def _region_components_ok(mask: np.ndarray) -> bool:
    return len(components_mask(mask, connectivity=8)) == 1


def _build_region_mask(template_size: int) -> np.ndarray | None:
    for _ in range(REGION_ATTEMPTS):
        H = random.randint(*REGION_HEIGHT_RANGE)
        W = random.randint(*REGION_WIDTH_RANGE)
        mask = np.ones((H, W), dtype=bool)
        target = random.randint(REGION_MASK_AREA_RANGE[0], min(REGION_MASK_AREA_RANGE[1], H * W - 1))
        while int(mask.sum()) > target:
            removable = _boundary_cells(mask)
            random.shuffle(removable)
            removed = False
            for r, c in removable:
                trial = mask.copy()
                trial[r, c] = False
                if not _region_components_ok(trial):
                    continue
                if len(_interior_cells(trial)) < max(MIN_INTERIOR_CELLS, template_size + 3):
                    continue
                mask = trial
                removed = True
                break
            if not removed:
                break

        if int(mask.sum()) > target or int(mask.sum()) < REGION_MASK_AREA_RANGE[0]:
            continue
        if int(mask.sum()) == H * W:
            continue
        if random.choice((False, True)):
            mask = np.fliplr(mask)
        mask = np.rot90(mask, random.randrange(4)).copy()
        if not _region_components_ok(mask):
            continue
        if len(_interior_cells(mask)) < max(MIN_INTERIOR_CELLS, template_size + 3):
            continue
        return mask
    return None


def _all_region_placements(mask: np.ndarray, variants: list[Shape]) -> list[tuple[tuple[int, int], frozenset[Coord]]]:
    interior = _interior_cells(mask)
    placements: list[tuple[tuple[int, int], frozenset[Coord]]] = []
    seen: set[frozenset[Coord]] = set()
    H, W = mask.shape
    for variant in variants:
        h, w = _shape_size(variant)
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                cells = _shape_cells(variant, r0, c0)
                if cells in seen:
                    continue
                if cells.issubset(interior):
                    seen.add(cells)
                    placements.append(((r0, c0), cells))
    placements.sort(key=lambda item: (item[0][0], item[0][1]))
    return placements


def _noise_choices(pool: list[Coord], count: int, matched: set[Coord]) -> list[tuple[Coord, ...]]:
    far = [
        cell
        for cell in pool
        if all(abs(cell[0] - mr) + abs(cell[1] - mc) >= 2 for mr, mc in matched)
    ]
    ordered_pools = [far, pool] if len(far) >= count else [pool]
    combos: list[tuple[Coord, ...]] = []
    for current in ordered_pools:
        local = list(itertools.combinations(current, count))
        random.shuffle(local)
        combos.extend(local[:80])
    return combos


def _select_region_defects(mask: np.ndarray, variants: list[Shape]) -> tuple[set[Coord], set[Coord], list[tuple[tuple[int, int], frozenset[Coord]]]] | None:
    placements = _all_region_placements(mask, variants)
    if not placements:
        return None
    interior = _interior_cells(mask)
    order = list(range(len(placements)))
    random.shuffle(order)
    for idx in order:
        matched = set(placements[idx][1])
        extra_counts = [1] if len(matched) >= 5 else [2, 1]
        remaining = list(interior - matched)
        for extra_count in extra_counts:
            if len(remaining) < extra_count:
                continue
            for extra_cells in _noise_choices(remaining, extra_count, matched):
                defects = matched | set(extra_cells)
                matches = _match_candidates(defects, variants, mask.shape)
                if len(matches) == 1 and set(matches[0][1]) == matched:
                    return matched, set(extra_cells), placements
    return None


def _corner_position(shape: Shape, canvas_shape: tuple[int, int], corner: str) -> tuple[int, int]:
    H, W = canvas_shape
    h, w = _shape_size(shape)
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, W - w
    if corner == "bl":
        return H - h, 0
    return H - h, W - w


def _globalize(cells: set[Coord] | frozenset[Coord], r0: int, c0: int) -> set[Coord]:
    return {(r0 + r, c0 + c) for r, c in cells}


def _place_regions(
    grid_shape: tuple[int, int],
    template_cells: set[Coord],
    blue_mask: np.ndarray,
    red_mask: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    H, W = grid_shape
    layout = random.choice(("horizontal", "vertical"))
    blue_h, blue_w = blue_mask.shape
    red_h, red_w = red_mask.shape
    blue_local = _mask_cells(blue_mask)
    red_local = _mask_cells(red_mask)

    for _ in range(PLACEMENT_ATTEMPTS):
        if layout == "horizontal":
            split = random.randint(W // 2 - 1, W // 2 + 1)
            blue_r = random.randint(0, H - blue_h)
            red_r = random.randint(0, H - red_h)
            blue_c_max = split - blue_w - 1
            red_c_min = split + 1
            if blue_c_max < 0 or red_c_min > W - red_w:
                continue
            blue_c = random.randint(0, blue_c_max)
            red_c = random.randint(red_c_min, W - red_w)
        else:
            split = random.randint(H // 2 - 1, H // 2 + 1)
            blue_c = random.randint(0, W - blue_w)
            red_c = random.randint(0, W - red_w)
            blue_r_max = split - blue_h - 1
            red_r_min = split + 1
            if blue_r_max < 0 or red_r_min > H - red_h:
                continue
            blue_r = random.randint(0, blue_r_max)
            red_r = random.randint(red_r_min, H - red_h)

        blue_cells = _globalize(blue_local, blue_r, blue_c)
        red_cells = _globalize(red_local, red_r, red_c)
        if not _separated(blue_cells, red_cells, MIN_REGION_SEPARATION):
            continue
        if not _separated(blue_cells, template_cells, MIN_TEMPLATE_SEPARATION):
            continue
        if not _separated(red_cells, template_cells, MIN_TEMPLATE_SEPARATION):
            continue
        return (blue_r, blue_c), (red_r, red_c)
    return None


def _render_input(
    grid_shape: tuple[int, int],
    template_shape: Shape,
    template_corner: str,
    blue_mask: np.ndarray,
    blue_pos: tuple[int, int],
    blue_defects: set[Coord],
    red_mask: np.ndarray,
    red_pos: tuple[int, int],
    red_defects: set[Coord],
) -> tuple[np.ndarray, set[Coord]]:
    H, W = grid_shape
    grid = np.zeros((H, W), dtype=int)

    template_r, template_c = _corner_position(template_shape, grid_shape, template_corner)
    template_cells = {(template_r + r, template_c + c) for r, c in template_shape}
    for r, c in template_cells:
        grid[r, c] = Color.TEAL

    blue_cells = _globalize(_mask_cells(blue_mask), *blue_pos)
    red_cells = _globalize(_mask_cells(red_mask), *red_pos)
    for r, c in blue_cells:
        grid[r, c] = Color.BLUE
    for r, c in red_cells:
        grid[r, c] = Color.RED

    for r, c in _globalize(blue_defects, *blue_pos):
        grid[r, c] = Color.BLACK
    for r, c in _globalize(red_defects, *red_pos):
        grid[r, c] = Color.BLACK
    return grid, template_cells


def _oracle_output(
    grid: np.ndarray,
    template_cells: set[Coord],
    blue_matched: set[Coord],
    blue_noise: set[Coord],
    red_matched: set[Coord],
    red_noise: set[Coord],
) -> np.ndarray:
    out = grid.copy()
    for r, c in template_cells:
        out[r, c] = Color.BLACK

    for r, c in blue_noise:
        out[r, c] = Color.BLUE
    for r, c in red_noise:
        out[r, c] = Color.RED

    blue_outline: set[Coord] = set()
    for r, c in blue_matched:
        for nr, nc in neighbors4(r, c):
            if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == Color.BLUE:
                blue_outline.add((nr, nc))
    for r, c in blue_outline:
        out[r, c] = Color.RED

    red_outline: set[Coord] = set()
    for r, c in red_matched:
        for nr, nc in neighbors4(r, c):
            if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == Color.RED:
                red_outline.add((nr, nc))
    for r, c in red_outline:
        out[r, c] = Color.BLUE

    return out


def _validate_generated(
    grid: np.ndarray,
    expected: np.ndarray,
    template_cells: set[Coord],
    template_shape: Shape,
    blue_defects: set[Coord],
    blue_matched: set[Coord],
    red_defects: set[Coord],
    red_matched: set[Coord],
) -> bool:
    if not np.array_equal(main(grid), expected):
        return False
    if len(_dihedral_variants(template_shape)) < 2:
        return False
    if len(components_mask(grid == Color.TEAL, connectivity=4)) != 1:
        return False
    if len(components_mask(grid == Color.BLUE, connectivity=8)) != 1:
        return False
    if len(components_mask(grid == Color.RED, connectivity=8)) != 1:
        return False
    if _region_defects(grid, Color.BLUE) != blue_defects:
        return False
    if _region_defects(grid, Color.RED) != red_defects:
        return False

    variants = _dihedral_variants(_template_shape(grid))
    blue_matches = _match_candidates(blue_defects, variants, grid.shape)
    red_matches = _match_candidates(red_defects, variants, grid.shape)
    if len(blue_matches) != 1 or set(blue_matches[0][1]) != blue_matched:
        return False
    if len(red_matches) != 1 or set(red_matches[0][1]) != red_matched:
        return False
    if len(blue_defects - blue_matched) < 1 or len(red_defects - red_matched) < 1:
        return False
    if int(np.sum(grid == Color.BLACK)) <= len(template_cells):
        return False
    if np.array_equal(grid, expected):
        return False
    return True


def _generate_once() -> np.ndarray | None:
    H = random.randint(*GRID_HEIGHT_RANGE)
    W = random.randint(*GRID_WIDTH_RANGE)

    base_template = random.choice(TEMPLATE_CATALOG)
    display_shape = random.choice(_dihedral_variants(base_template))
    variants = _dihedral_variants(display_shape)

    blue_mask = _build_region_mask(len(display_shape))
    red_mask = _build_region_mask(len(display_shape))
    if blue_mask is None or red_mask is None:
        return None

    blue_choice = _select_region_defects(blue_mask, variants)
    red_choice = _select_region_defects(red_mask, variants)
    if blue_choice is None or red_choice is None:
        return None
    blue_local_matched, blue_local_noise, _blue_placements = blue_choice
    red_local_matched, red_local_noise, _red_placements = red_choice

    template_corner = random.choice(("tl", "tr", "bl", "br"))
    template_pos = _corner_position(display_shape, (H, W), template_corner)
    template_cells = {(template_pos[0] + r, template_pos[1] + c) for r, c in display_shape}

    placements = _place_regions((H, W), template_cells, blue_mask, red_mask)
    if placements is None:
        return None
    blue_pos, red_pos = placements

    grid, template_cells = _render_input(
        (H, W),
        display_shape,
        template_corner,
        blue_mask,
        blue_pos,
        blue_local_matched | blue_local_noise,
        red_mask,
        red_pos,
        red_local_matched | red_local_noise,
    )

    blue_matched = _globalize(blue_local_matched, *blue_pos)
    blue_noise = _globalize(blue_local_noise, *blue_pos)
    red_matched = _globalize(red_local_matched, *red_pos)
    red_noise = _globalize(red_local_noise, *red_pos)
    blue_defects = blue_matched | blue_noise
    red_defects = red_matched | red_noise

    blue_fill_area = int(np.sum(grid == Color.BLUE))
    red_fill_area = int(np.sum(grid == Color.RED))
    if not (MIN_REGION_FILL_AREA <= blue_fill_area <= MAX_REGION_FILL_AREA):
        return None
    if not (MIN_REGION_FILL_AREA <= red_fill_area <= MAX_REGION_FILL_AREA):
        return None

    expected = _oracle_output(grid, template_cells, blue_matched, blue_noise, red_matched, red_noise)
    if not _validate_generated(
        grid,
        expected,
        template_cells,
        display_shape,
        blue_defects,
        blue_matched,
        red_defects,
        red_matched,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
