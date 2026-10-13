import itertools
import random

import numpy as np


Coord = tuple[int, int]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None, "expected a non-empty mask"
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_coords(mask: np.ndarray) -> list[Coord]:
    return [tuple(map(int, rc)) for rc in np.argwhere(mask)]


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for k in range(4):
            transformed = np.rot90(base, k)
            key = (tuple(map(int, transformed.shape)), transformed.astype(np.uint8).tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(transformed.copy())
    return variants


def _legend_entries(grid: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    assert w >= 2, "legend requires the last two columns"
    entries: list[tuple[int, int, int]] = []
    for r in range(h):
        left = int(grid[r, w - 2])
        right = int(grid[r, w - 1])
        if left == 0 and right == 0:
            continue
        assert left != 0 and right != 0, "legend rows must contain two non-black cells"
        entries.append((r, left, right))
    assert 2 <= len(entries) <= 3, "expected 2-3 legend rows on the right edge"
    return entries


def _candidate_placements(
    grid: np.ndarray, fragment_cells: set[Coord], variants: list[np.ndarray], bg: int = 0
) -> dict[tuple[int, int, int], set[Coord]]:
    h, w = grid.shape
    candidates: dict[tuple[int, int, int], set[Coord]] = {}
    for variant_id, mask in enumerate(variants):
        rel_cells = _mask_coords(mask)
        for fr, fc in fragment_cells:
            for mr, mc in rel_cells:
                r0 = fr - mr
                c0 = fc - mc
                placed: set[Coord] = set()
                ok = True
                for rr, cc in rel_cells:
                    ar = r0 + rr
                    ac = c0 + cc
                    if not (0 <= ar < h and 0 <= ac < w):
                        ok = False
                        break
                    placed.add((ar, ac))
                if not ok or not fragment_cells.issubset(placed):
                    continue
                for cell in placed:
                    if cell in fragment_cells:
                        continue
                    if int(grid[cell]) != bg:
                        ok = False
                        break
                if ok:
                    candidates[(variant_id, r0, c0)] = placed
    return candidates


def _find_unique_fragment_match(
    grid: np.ndarray, fragment_color: int, variants: list[np.ndarray]
) -> set[Coord]:
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape
    frag_mask = g == int(fragment_color)
    frag_mask[:, w - 2 : w] = False
    fragment_cells = set(_mask_coords(frag_mask))
    assert fragment_cells, f"expected visible fragment cells for color {fragment_color}"
    candidates = _candidate_placements(g, fragment_cells, variants, bg=0)
    assert len(candidates) == 1, f"fragment color {fragment_color} must match exactly one transformed template"
    return next(iter(candidates.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each colored fragment into a rotated/reflected copy of the grey template using the right-edge legend color.

    Input structure:
    - The grid has black background, one complete grey template near the upper-left, and a 2-column legend strip on the right edge.
    - Each legend row is a (fragment color, completion color) pair.
    - Away from the template and legend, each fragment color appears as a small subset of one rotated/reflected copy of the template.

    Rule:
    - Tight-crop the grey template mask and generate all distinct dihedral transforms of that mask.
    - For each fragment color listed in the legend, find the unique transformed template placement whose occupied cells contain that fragment and whose remaining template cells are black.
    - Keep the fragment pixels unchanged, fill the missing cells of that placement with the legend's completion color, then erase the right-edge legend strip to black.

    Key insight:
    - The fragment geometry determines which dihedral transform and placement to use, while the legend independently determines the color of the newly completed cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    entries = _legend_entries(g)
    legend_map: dict[int, int] = {}
    for _, fragment_color, fill_color in entries:
        assert fragment_color not in legend_map, "legend fragment colors must be unique"
        legend_map[int(fragment_color)] = int(fill_color)

    template_mask = g == 5
    assert template_mask.any(), "expected a grey template"
    template = _crop_mask(template_mask)
    variants = _dihedral_variants(template)

    out = g.copy()
    for fragment_color, fill_color in legend_map.items():
        frag_mask = g == fragment_color
        frag_mask[:, g.shape[1] - 2 : g.shape[1]] = False
        if not frag_mask.any():
            continue
        placed = _find_unique_fragment_match(g, fragment_color, variants)
        for r, c in placed:
            if out[r, c] == 0:
                out[r, c] = int(fill_color)

    out[:, g.shape[1] - 2 : g.shape[1]] = 0
    return out


GRID_H_RANGE = (17, 23)
GRID_W_RANGE = (18, 24)
TEMPLATE_H_RANGE = (3, 5)
TEMPLATE_W_RANGE = (4, 5)
TEMPLATE_AREA_RANGE = (6, 8)
LEGEND_COUNT_CHOICES = (2, 3)
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
PLACEMENT_GAP = 2
GEN_RETRIES = 400
TEMPLATE_RETRIES = 120
PLACEMENT_RETRIES = 120


def _neighbors8(r: int, c: int) -> list[Coord]:
    out: list[Coord] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _random_template_mask() -> np.ndarray:
    for _ in range(TEMPLATE_RETRIES):
        h = random.randint(*TEMPLATE_H_RANGE)
        w = random.randint(*TEMPLATE_W_RANGE)
        area = random.randint(*TEMPLATE_AREA_RANGE)
        cells: set[Coord] = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            frontier: set[Coord] = set()
            for r, c in cells:
                for nr, nc in _neighbors8(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if not (TEMPLATE_H_RANGE[0] <= mask.shape[0] <= TEMPLATE_H_RANGE[1]):
            continue
        if not (3 <= mask.shape[1] <= TEMPLATE_W_RANGE[1]):
            continue
        if len(_dihedral_variants(mask)) != 8:
            continue
        if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
            continue
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            continue
        if mask.sum() < TEMPLATE_AREA_RANGE[0]:
            continue
        return mask
    raise RuntimeError("failed to build a non-symmetric template")


def _bbox_from_origin(mask: np.ndarray, r0: int, c0: int) -> tuple[int, int, int, int]:
    h, w = mask.shape
    return int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)


def _bbox_far(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _placed_cells(mask: np.ndarray, r0: int, c0: int) -> set[Coord]:
    return {(r0 + r, c0 + c) for r, c in _mask_coords(mask)}


def _build_expected(
    grid: np.ndarray, fragments: list[dict[str, object]], legend_cols: slice
) -> np.ndarray:
    out = grid.copy()
    for item in fragments:
        fill_color = int(item["fill_color"])
        for r, c in item["placed"]:
            if out[r, c] == 0:
                out[r, c] = fill_color
    out[:, legend_cols] = 0
    return out


def _valid_anchor_subsets(
    base: np.ndarray,
    fragment_color: int,
    placed: set[Coord],
    variants: list[np.ndarray],
    intended: set[Coord],
) -> list[tuple[Coord, ...]]:
    cells = sorted(placed)
    valid: list[tuple[Coord, ...]] = []
    max_size = min(4, len(cells) - 1)
    for size in range(2, max_size + 1):
        for subset in itertools.combinations(cells, size):
            trial = base.copy()
            for r, c in subset:
                trial[r, c] = int(fragment_color)
            candidates = _candidate_placements(trial, set(subset), variants, bg=0)
            if len(candidates) != 1:
                continue
            only = next(iter(candidates.values()))
            if only == intended:
                valid.append(subset)
    return valid


def _choose_subset(valid_subsets: list[tuple[Coord, ...]]) -> tuple[Coord, ...]:
    by_size: dict[int, list[tuple[Coord, ...]]] = {}
    for subset in valid_subsets:
        by_size.setdefault(len(subset), []).append(subset)
    sizes = sorted(by_size)
    weights = [2 if size == 2 else 4 for size in sizes]
    chosen_size = random.choices(sizes, weights=weights, k=1)[0]
    return random.choice(by_size[chosen_size])


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    legend_count = random.choice(LEGEND_COUNT_CHOICES)
    if w < 8:
        return None

    template = _random_template_mask()
    variants = _dihedral_variants(template)
    if len(variants) < legend_count:
        return None

    colors = random.sample(PALETTE, 2 * legend_count)
    fragment_colors = colors[:legend_count]
    fill_colors = colors[legend_count:]

    legend_start = random.randint(max(0, h - legend_count - 3), h - legend_count)
    legend_rows = list(range(legend_start, legend_start + legend_count))
    legend_cols = slice(w - 2, w)

    grid = np.zeros((h, w), dtype=int)
    reserved_boxes: list[tuple[int, int, int, int]] = []

    template_r = random.randint(0, max(0, min(3, h - template.shape[0])))
    template_c = random.randint(0, max(0, min(3, w - 5 - template.shape[1])))
    template_cells = _placed_cells(template, template_r, template_c)
    for r, c in template_cells:
        grid[r, c] = 5
    reserved_boxes.append(_bbox_from_origin(template, template_r, template_c))

    legend_box = (legend_rows[0], w - 2, legend_rows[-1], w - 1)
    reserved_boxes.append(legend_box)
    for row, left, right in zip(legend_rows, fragment_colors, fill_colors):
        grid[row, w - 2] = left
        grid[row, w - 1] = right

    transform_ids = random.sample(range(len(variants)), legend_count)
    fragments: list[dict[str, object]] = []

    for color, fill_color, variant_id in zip(fragment_colors, fill_colors, transform_ids):
        mask = variants[variant_id]
        placed_box = None
        placed_cells = None
        for _ in range(PLACEMENT_RETRIES):
            max_c0 = w - 3 - mask.shape[1]
            if max_c0 < 0 or h - mask.shape[0] < 0:
                return None
            r0 = random.randint(0, h - mask.shape[0])
            c0 = random.randint(0, max_c0)
            box = _bbox_from_origin(mask, r0, c0)
            if any(not _bbox_far(box, other, PLACEMENT_GAP) for other in reserved_boxes):
                continue
            cells = _placed_cells(mask, r0, c0)
            placed_box = box
            placed_cells = cells
            break
        if placed_box is None or placed_cells is None:
            return None
        reserved_boxes.append(placed_box)
        fragments.append(
            {
                "fragment_color": int(color),
                "fill_color": int(fill_color),
                "mask": mask,
                "placed": placed_cells,
            }
        )

    base = grid.copy()
    for item in fragments:
        valid_subsets = _valid_anchor_subsets(
            base,
            int(item["fragment_color"]),
            item["placed"],
            variants,
            item["placed"],
        )
        if not valid_subsets:
            return None
        subset = _choose_subset(valid_subsets)
        item["anchors"] = subset
        for r, c in subset:
            grid[r, c] = int(item["fragment_color"])

    expected = _build_expected(grid, fragments, legend_cols)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dihedral-completion input")
