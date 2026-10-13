import random

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

SEED_SHAPES = ((2, 2), (2, 3), (3, 2), (3, 3))
MAX_GENERATION_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 250


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_border(cells: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _extract_seeds(grid: np.ndarray) -> tuple[list[dict], dict[int, np.ndarray]]:
    mask = (grid != BG) & (grid != GREY)
    seeds: list[dict] = []
    color_to_pattern: dict[int, np.ndarray] = {}
    for comp in _components(mask):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 + 1 < 2 or c1 - c0 + 1 < 2:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(sub == BG) or np.any(sub == GREY):
            continue
        colors = sorted(set(map(int, sub.reshape(-1))))
        if len(colors) != 2:
            continue
        if len(comp) != sub.size:
            continue
        pattern = sub.copy()
        seeds.append({"bbox": (r0, c0, r1, c1), "coords": comp, "pattern": pattern, "colors": colors})
        for color in colors:
            assert color not in color_to_pattern, "each seed color must belong to exactly one seed"
            color_to_pattern[color] = pattern
    assert seeds, "expected at least one seed rectangle"
    return seeds, color_to_pattern


def _find_regions(grid: np.ndarray) -> list[dict]:
    passable = grid != GREY
    regions: list[dict] = []
    for comp in _components(passable):
        if _touches_border(comp, grid.shape):
            continue
        marker_cells = [(r, c) for r, c in comp if int(grid[r, c]) != BG]
        if not marker_cells:
            continue
        assert len(marker_cells) == 1, "each enclosed region must contain exactly one colored marker"
        border: set[tuple[int, int]] = set()
        for r, c in comp:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == GREY:
                    border.add((nr, nc))
        assert border, "each enclosed region must be bounded by grey cells"
        anchor_bbox = _bbox(list(border) + comp)
        mr, mc = marker_cells[0]
        regions.append({"cells": comp, "marker_color": int(grid[mr, mc]), "bbox": anchor_bbox})
    assert regions, "expected at least one enclosed region"
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each marker-bearing closed region to a two-color seed rectangle, then tile that seed pattern through the region interior.

    Input structure:
    - The background is black.
    - Several small dense rectangles elsewhere in the grid act as seed patterns; each rectangle uses exactly two non-grey colors.
    - Separate larger closed regions are enclosed by grey boundary pixels, with black interiors and one single colored marker pixel inside each region.

    Rule:
    - Find every seed rectangle and map each of its two colors back to that rectangle's full pattern.
    - Find every non-grey area that is enclosed by grey walls and does not touch the grid edge; its lone colored pixel is the marker.
    - For each enclosed region, choose the seed pattern that contains the marker color, then tile that pattern periodically over the region using the closed region's bounding box top-left as the alignment origin.
    - Keep the grey walls unchanged, overwrite the marker with the tiled color, and clear the original seed rectangles to black.

    Key insight:
    - The colored marker does not tell you a fill color directly; it identifies which distant seed rectangle supplies a repeating two-color tile that must be phase-aligned to the whole closed region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    seeds, color_to_pattern = _extract_seeds(g)
    regions = _find_regions(g)

    out = g.copy()
    for region in regions:
        pattern = color_to_pattern[region["marker_color"]]
        ph, pw = pattern.shape
        r0, c0, _, _ = region["bbox"]
        for r, c in region["cells"]:
            out[r, c] = int(pattern[(r - r0) % ph, (c - c0) % pw])

    for seed in seeds:
        for r, c in seed["coords"]:
            out[r, c] = BG
    return out


def _make_seed_pattern(colors: tuple[int, int]) -> np.ndarray:
    h, w = random.choice(SEED_SHAPES)
    c1, c2 = colors
    styles = ["checker", "hstripe", "vstripe"]
    if (h, w) == (3, 3):
        styles.extend(["frame", "cross"])
    while True:
        style = random.choice(styles)
        patch = np.zeros((h, w), dtype=int)
        for r in range(h):
            for c in range(w):
                if style == "checker":
                    use_first = (r + c) % 2 == 0
                elif style == "hstripe":
                    use_first = r % 2 == 0
                elif style == "vstripe":
                    use_first = c % 2 == 0
                elif style == "frame":
                    use_first = r in (0, h - 1) or c in (0, w - 1)
                else:
                    use_first = r == h // 2 or c == w // 2
                patch[r, c] = c1 if use_first else c2
        if len(set(map(int, patch.reshape(-1)))) == 2:
            return patch


def _region_size_range(region_count: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if region_count <= 2:
        return (7, 12), (7, 12)
    if region_count == 3:
        return (6, 10), (6, 10)
    return (6, 9), (6, 9)


def _connected(mask: np.ndarray) -> bool:
    return mask.any() and len(_components(mask)) == 1


def _add_notch(mask: np.ndarray) -> np.ndarray | None:
    ih, iw = mask.shape
    side = random.choice(("top", "bottom", "left", "right"))
    new_mask = mask.copy()
    if side in ("top", "bottom"):
        if ih < 2 or iw < 3:
            return None
        width = random.randint(1, min(3, iw - 1))
        start = random.randint(0, iw - width)
        max_depth = min(3, ih - 1)
        depth = random.randint(1, max_depth)
        if side == "top":
            new_mask[:depth, start : start + width] = False
        else:
            new_mask[ih - depth :, start : start + width] = False
    else:
        if iw < 2 or ih < 3:
            return None
        height = random.randint(1, min(3, ih - 1))
        start = random.randint(0, ih - height)
        max_depth = min(3, iw - 1)
        depth = random.randint(1, max_depth)
        if side == "left":
            new_mask[start : start + height, :depth] = False
        else:
            new_mask[start : start + height, iw - depth :] = False

    min_area = max(6, (ih * iw) // 2)
    if int(np.count_nonzero(new_mask)) < min_area:
        return None
    if not _connected(new_mask):
        return None
    return new_mask


def _make_region_patch(height: int, width: int, force_irregular: bool) -> tuple[np.ndarray, list[tuple[int, int]], bool]:
    ih, iw = height - 2, width - 2
    assert ih >= 4 and iw >= 4
    for _ in range(80):
        interior = np.ones((ih, iw), dtype=bool)
        irregular = False
        target_notches = random.randint(1 if force_irregular else 0, 2 if min(ih, iw) >= 5 else 1)
        for _ in range(target_notches):
            updated = _add_notch(interior)
            if updated is not None and not np.array_equal(updated, interior):
                interior = updated
                irregular = True
        if force_irregular and not irregular:
            continue
        patch = np.full((height, width), GREY, dtype=int)
        patch[1:-1, 1:-1] = np.where(interior, BG, GREY)
        cells = [(r + 1, c + 1) for r, c in np.argwhere(interior)]
        if cells:
            return patch, cells, irregular
    raise RuntimeError("failed to build a valid region patch")


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    if top < 1 or left < 1 or top + height > h - 1 or left + width > w - 1:
        return False
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + height + 1)
    c1 = min(w, left + width + 1)
    return not occupied[r0:r1, c0:c1].any()


def _mark_occupied(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    h, w = occupied.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + height + 1)
    c1 = min(w, left + width + 1)
    occupied[r0:r1, c0:c1] = True


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    ph, pw = patch.shape
    max_top = grid.shape[0] - ph - 1
    max_left = grid.shape[1] - pw - 1
    if max_top < 1 or max_left < 1:
        return None
    for _ in range(PLACEMENT_ATTEMPTS):
        top = random.randint(1, max_top)
        left = random.randint(1, max_left)
        if not _can_place(occupied, top, left, ph, pw):
            continue
        sub = grid[top : top + ph, left : left + pw]
        sub[patch != BG] = patch[patch != BG]
        _mark_occupied(occupied, top, left, ph, pw)
        return top, left
    return None


def _build_assignment(seed_count: int, region_count: int) -> list[int]:
    assignment = list(range(seed_count))
    while len(assignment) < region_count:
        assignment.append(random.randrange(seed_count))
    random.shuffle(assignment)
    return assignment


def _generate_attempt() -> np.ndarray | None:
    seed_count = random.randint(2, 4)
    region_count = seed_count if seed_count == 4 else random.choice((seed_count, seed_count + 1))
    region_count = min(region_count, 4)

    colors = list(random.sample(COLOR_POOL, 2 * seed_count))
    random.shuffle(colors)
    seed_specs = []
    for idx in range(seed_count):
        pair = (colors[2 * idx], colors[2 * idx + 1])
        patch = _make_seed_pattern(pair)
        seed_specs.append({"colors": pair, "patch": patch})

    h_range, w_range = _region_size_range(region_count)
    assignment = _build_assignment(seed_count, region_count)
    region_specs = []
    irregular_count = 0
    for idx, seed_idx in enumerate(assignment):
        height = random.randint(*h_range)
        width = random.randint(*w_range)
        patch, cells, irregular = _make_region_patch(height, width, force_irregular=idx == 0)
        marker_color = random.choice(seed_specs[seed_idx]["colors"])
        marker_r, marker_c = random.choice(cells)
        patch = patch.copy()
        patch[marker_r, marker_c] = marker_color
        region_specs.append({"patch": patch, "seed_idx": seed_idx, "marker_color": marker_color})
        irregular_count += int(irregular)
    if irregular_count == 0:
        return None

    min_side = max(18, 4 + sum(spec["patch"].shape[0] for spec in region_specs) // 2)
    grid_h = random.randint(min_side, 30)
    grid_w = random.randint(min_side, 30)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    for spec in sorted(region_specs, key=lambda item: item["patch"].size, reverse=True):
        if _place_patch(grid, occupied, spec["patch"]) is None:
            return None

    seed_order = list(range(seed_count))
    random.shuffle(seed_order)
    for idx in seed_order:
        if _place_patch(grid, occupied, seed_specs[idx]["patch"]) is None:
            return None

    seeds, color_to_pattern = _extract_seeds(grid)
    regions = _find_regions(grid)
    if len(seeds) != seed_count or len(regions) != region_count:
        return None
    if any(region["marker_color"] not in color_to_pattern for region in regions):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
