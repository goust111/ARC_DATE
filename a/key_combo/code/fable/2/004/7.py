from __future__ import annotations

from collections import Counter, deque
import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import components_mask, find_components, get_bg, paste


NOTCH_SIDES = ("top", "bottom", "left", "right")
GEN_RETRY_LIMIT = 500
OUTLINE_RETRY_LIMIT = 120
PLACEMENT_RETRY_LIMIT = 80
MARKER_TOTAL_CHOICES = (3, 4, 5, 6, 7)
PALETTE = tuple(range(10))


def _enclosed_region_coords(wall_coords: np.ndarray, box: tuple[int, int, int, int]) -> list[np.ndarray]:
    r0, c0, r1, c1 = map(int, box)
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    wall = np.zeros((h, w), dtype=bool)
    local = wall_coords - np.array([r0, c0], dtype=int)
    wall[local[:, 0], local[:, 1]] = True
    passable = ~wall

    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for rr in range(h):
        for cc in (0, w - 1):
            if passable[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))
    for cc in range(w):
        for rr in (0, h - 1):
            if passable[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))

    while q:
        rr, cc = q.popleft()
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    enclosed = passable & ~outside
    return [coords + np.array([r0, c0], dtype=int) for coords in components_mask(enclosed, connectivity=4)]


def _filled_patch(
    bg: int,
    bbox: tuple[int, int, int, int],
    outline_coords: np.ndarray,
    region_coords: np.ndarray,
    outline_color: int,
    fill_color: int,
) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    local_outline = outline_coords - np.array([r0, c0], dtype=int)
    local_region = region_coords - np.array([r0, c0], dtype=int)
    patch[local_outline[:, 0], local_outline[:, 1]] = int(outline_color)
    patch[local_region[:, 0], local_region[:, 1]] = int(fill_color)
    return patch


def _parse_objects(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert components, "expected non-background components"

    markers: dict[tuple[int, int], int] = {}
    outlines: list[dict] = []
    for comp in components:
        coords = comp["coords"]
        if int(comp["area"]) == 1:
            markers[(int(coords[0, 0]), int(coords[0, 1]))] = int(comp["color"])
        else:
            outlines.append(comp)

    assert outlines, "expected at least one outline"

    objects: list[dict] = []
    totals: list[int] = []
    for outline in outlines:
        regions = _enclosed_region_coords(outline["coords"], outline["bbox"])
        assert len(regions) == 1, "each outline must enclose exactly one region"
        region = regions[0]
        marker_colors = [
            int(markers[(int(r), int(c))])
            for r, c in region
            if (int(r), int(c)) in markers
        ]
        assert marker_colors, "each enclosed region must contain marker pixels"
        counts = Counter(marker_colors)
        ranked = counts.most_common()
        assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "majority marker color must be unique"

        total = int(len(marker_colors))
        totals.append(total)
        objects.append(
            {
                "bbox": tuple(map(int, outline["bbox"])),
                "coords": outline["coords"],
                "outline_color": int(outline["color"]),
                "region": region,
                "count": total,
                "fill_color": int(ranked[0][0]),
            }
        )

    assert len(totals) == len(set(totals)), "internal marker totals must be distinct"
    return bg, objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Queue hollow outlines along the bottom by increasing internal marker count and fill each cavity with its majority marker color.

    Input structure:
    - The background is dominant.
    - Foreground consists of monochrome closed outline components plus isolated one-pixel markers of one or more colors.
    - Some markers lie inside outline-enclosed regions and some lie outside as distractors.

    Rule:
    - Remove every marker and clear the canvas.
    - For each closed outline, count only the markers inside its enclosed region, keep the outline's original shape and color, and fill that enclosed region with the unique majority marker color found there.
    - Reposition the resulting filled outlines left-to-right in increasing internal-marker-count order, with their bottom edges aligned to the grid bottom and exactly one blank column between consecutive outlines.

    Key insight:
    - The marker count is a sorting key, not a drawing size; only markers inside the truly enclosed cavity matter, while bbox-contained or outside distractors are irrelevant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, objects = _parse_objects(g)
    out = np.full_like(g, bg)

    left = 0
    for obj in sorted(objects, key=lambda item: item["count"]):
        patch = _filled_patch(
            bg,
            obj["bbox"],
            obj["coords"],
            obj["region"],
            obj["outline_color"],
            obj["fill_color"],
        )
        ph, pw = patch.shape
        top = g.shape[0] - ph
        assert left + pw <= g.shape[1], "queued outlines must fit in the output grid"
        paste(out, patch, top, left, transparent=bg)
        left += pw + 1

    return out


def _add_segment(cells: set[tuple[int, int]], r0: int, c0: int, r1: int, c1: int) -> None:
    if int(r0) == int(r1):
        a, b = sorted((int(c0), int(c1)))
        for cc in range(a, b + 1):
            cells.add((int(r0), cc))
        return
    a, b = sorted((int(r0), int(r1)))
    for rr in range(a, b + 1):
        cells.add((rr, int(c0)))


def _rect_perimeter_cells(h: int, w: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for cc in range(w):
        cells.add((0, cc))
        cells.add((h - 1, cc))
    for rr in range(h):
        cells.add((rr, 0))
        cells.add((rr, w - 1))
    return cells


def _apply_notch(cells: set[tuple[int, int]], h: int, w: int, side: str) -> bool:
    if side in ("top", "bottom"):
        max_span = min(3, w - 3)
        max_depth = min(2, (h - 3) // 2)
        if max_span < 2 or max_depth < 1:
            return False
        span = random.randint(2, max_span)
        a = random.randint(1, w - span - 2)
        b = a + span
        depth = random.randint(1, max_depth)
        if side == "top":
            for cc in range(a + 1, b):
                cells.discard((0, cc))
            _add_segment(cells, 0, a, depth, a)
            _add_segment(cells, 0, b, depth, b)
            _add_segment(cells, depth, a, depth, b)
        else:
            r = h - 1
            inner = r - depth
            for cc in range(a + 1, b):
                cells.discard((r, cc))
            _add_segment(cells, r, a, inner, a)
            _add_segment(cells, r, b, inner, b)
            _add_segment(cells, inner, a, inner, b)
        return True

    max_span = min(3, h - 3)
    max_depth = min(2, (w - 3) // 2)
    if max_span < 2 or max_depth < 1:
        return False
    span = random.randint(2, max_span)
    a = random.randint(1, h - span - 2)
    b = a + span
    depth = random.randint(1, max_depth)
    if side == "left":
        for rr in range(a + 1, b):
            cells.discard((rr, 0))
        _add_segment(cells, a, 0, a, depth)
        _add_segment(cells, b, 0, b, depth)
        _add_segment(cells, a, depth, b, depth)
    else:
        c = w - 1
        inner = c - depth
        for rr in range(a + 1, b):
            cells.discard((rr, c))
        _add_segment(cells, a, c, a, inner)
        _add_segment(cells, b, c, b, inner)
        _add_segment(cells, a, inner, b, inner)
    return True


def _outline_patch(color: int, bg: int, *, require_notch: bool) -> dict | None:
    for _ in range(OUTLINE_RETRY_LIMIT):
        h = random.randint(6, 8)
        w = random.randint(6, 8)
        cells = _rect_perimeter_cells(h, w)
        min_notches = 1 if require_notch else 0
        target_notches = random.randint(min_notches, 2)
        applied = 0
        for side in random.sample(NOTCH_SIDES, k=target_notches):
            applied += int(_apply_notch(cells, h, w, side))
        if require_notch and applied == 0:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for rr, cc in cells:
            mask[rr, cc] = True

        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, random.randrange(4))

        coords = np.argwhere(mask)
        if len(components_mask(mask, connectivity=4)) != 1:
            continue
        bbox = (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)
        regions = _enclosed_region_coords(coords, bbox)
        if len(regions) != 1:
            continue
        region = regions[0]
        parity = (region[:, 0] + region[:, 1]) % 2
        max_markers = int(max(np.sum(parity == 0), np.sum(parity == 1)))
        if int(region.shape[0]) < 14 or max_markers < max(MARKER_TOTAL_CHOICES):
            continue

        patch = np.full(mask.shape, int(bg), dtype=int)
        patch[mask] = int(color)
        region_mask = np.zeros(mask.shape, dtype=bool)
        region_mask[region[:, 0], region[:, 1]] = True
        pocket = np.argwhere((patch == int(bg)) & ~region_mask)

        return {
            "outline_patch": patch,
            "outline_coords": coords,
            "bbox": bbox,
            "region_local": region,
            "pocket_local": pocket,
            "outline_color": int(color),
            "outline_area": int(coords.shape[0]),
            "max_markers": max_markers,
        }
    return None


def _sample_marker_totals(specs: list[dict]) -> list[int] | None:
    available = list(MARKER_TOTAL_CHOICES)
    totals = [0] * len(specs)
    for idx in sorted(range(len(specs)), key=lambda i: specs[i]["max_markers"]):
        feasible = [value for value in available if value <= specs[idx]["max_markers"]]
        if not feasible:
            return None
        totals[idx] = random.choice(feasible)
        available.remove(totals[idx])
    return totals


def _positive_partition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = [cuts[0]]
    for a, b in zip(cuts, cuts[1:]):
        values.append(b - a)
    values.append(total - cuts[-1])
    return [int(v) for v in values]


def _marker_recipe(total: int, majority_color: int, palette: list[int]) -> list[tuple[int, int]]:
    color_choices = [2] if total == 3 else [2, 3]
    for _ in range(80):
        n_colors = random.choice(color_choices)
        max_majority = total - (n_colors - 1)
        majority_count = random.randint(2, max_majority)
        remainder = total - majority_count
        if remainder < n_colors - 1:
            continue
        minority_counts = _positive_partition(remainder, n_colors - 1)
        if majority_count <= max(minority_counts):
            continue
        minority_colors = random.sample([c for c in palette if c != int(majority_color)], n_colors - 1)
        recipe = [(int(majority_color), int(majority_count))]
        recipe.extend((int(color), int(count)) for color, count in zip(minority_colors, minority_counts))
        random.shuffle(recipe)
        return recipe
    raise RuntimeError("failed to sample a unique-majority marker recipe")


def _global_coords(local_coords: np.ndarray, top: int, left: int) -> np.ndarray:
    return local_coords + np.array([int(top), int(left)], dtype=int)


def _touches4(cell: tuple[int, int], used: set[tuple[int, int]]) -> bool:
    r, c = cell
    if cell in used:
        return True
    return (
        (r - 1, c) in used
        or (r + 1, c) in used
        or (r, c - 1) in used
        or (r, c + 1) in used
    )


def _choose_region_cells(region_global: np.ndarray, count: int) -> list[tuple[int, int]] | None:
    coords = [tuple(map(int, rc)) for rc in region_global]
    viable: list[list[tuple[int, int]]] = []
    for parity in (0, 1):
        group = [cell for cell in coords if (cell[0] + cell[1]) % 2 == parity]
        if len(group) >= count:
            viable.append(group)
    if not viable:
        return None
    cells = random.sample(random.choice(viable), count)
    return [(int(r), int(c)) for r, c in cells]


def _choose_free_rect(
    reserved: np.ndarray,
    patch_shape: tuple[int, int],
    *,
    border: int,
    padding: int,
) -> tuple[int, int] | None:
    h, w = reserved.shape
    ph, pw = map(int, patch_shape)
    candidates: list[tuple[int, int]] = []
    for r0 in range(border, h - border - ph + 1):
        for c0 in range(border, w - border - pw + 1):
            rr0 = max(0, r0 - padding)
            cc0 = max(0, c0 - padding)
            rr1 = min(h, r0 + ph + padding)
            cc1 = min(w, c0 + pw + padding)
            if not bool(reserved[rr0:rr1, cc0:cc1].any()):
                candidates.append((int(r0), int(c0)))
    if not candidates:
        return None
    return random.choice(candidates)


def _place_outlines(specs: list[dict], shape: tuple[int, int]) -> list[tuple[int, int]] | None:
    reserved = np.zeros(shape, dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(specs)
    order = sorted(range(len(specs)), key=lambda i: specs[i]["outline_patch"].shape[0] * specs[i]["outline_patch"].shape[1], reverse=True)
    for idx in order:
        loc = _choose_free_rect(reserved, specs[idx]["outline_patch"].shape, border=1, padding=2)
        if loc is None:
            return None
        positions[idx] = loc
        r0, c0 = loc
        ph, pw = specs[idx]["outline_patch"].shape
        reserved[r0 : r0 + ph, c0 : c0 + pw] = True
    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _place_scattered(
    candidates: list[tuple[int, int]],
    count: int,
    used: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    shuffled = list(candidates)
    random.shuffle(shuffled)
    picked: list[tuple[int, int]] = []
    blocked = set(used)
    for cell in shuffled:
        if _touches4(cell, blocked):
            continue
        picked.append(cell)
        blocked.add(cell)
        if len(picked) == count:
            return picked
    return None


def _expected_output(bg: int, shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    left = 0
    for spec in sorted(specs, key=lambda item: item["count"]):
        patch = spec["filled_patch"]
        ph, pw = patch.shape
        paste(out, patch, shape[0] - ph, left, transparent=bg)
        left += pw + 1
    return out


def _attempt() -> np.ndarray | None:
    n_objects = random.choice((2, 3))
    bg = random.choice(PALETTE)
    outline_colors = random.sample([c for c in PALETTE if c != bg], n_objects)
    marker_palette = [c for c in PALETTE if c not in {bg, *outline_colors}]
    assert len(marker_palette) >= 4

    specs: list[dict] = []
    for idx, color in enumerate(outline_colors):
        patch = _outline_patch(color, bg, require_notch=(idx == 0))
        if patch is None:
            return None
        specs.append(patch)

    totals = _sample_marker_totals(specs)
    if totals is None:
        return None
    majority_colors = random.sample(marker_palette, n_objects)
    for spec, total, majority_color in zip(specs, totals, majority_colors):
        spec["count"] = int(total)
        spec["fill_color"] = int(majority_color)
        spec["marker_recipe"] = _marker_recipe(total, majority_color, marker_palette)
        filled_patch = spec["outline_patch"].copy()
        region = spec["region_local"]
        filled_patch[region[:, 0], region[:, 1]] = int(majority_color)
        spec["filled_patch"] = filled_patch

    output_width = sum(spec["outline_patch"].shape[1] for spec in specs) + (n_objects - 1)
    max_h = max(spec["outline_patch"].shape[0] for spec in specs)
    min_h = max(max_h + 6, 16)
    min_w = max(output_width + 4, 18)
    if min_h > 28 or min_w > 30:
        return None

    for _ in range(PLACEMENT_RETRY_LIMIT):
        height = random.randint(min_h, 28)
        width = random.randint(min_w, 30)
        positions = _place_outlines(specs, (height, width))
        if positions is None:
            continue

        grid = np.full((height, width), int(bg), dtype=int)
        marker_cells: set[tuple[int, int]] = set()
        bbox_mask = np.zeros((height, width), dtype=bool)
        pocket_candidates: list[tuple[int, int]] = []

        for spec, (top, left) in zip(specs, positions):
            paste(grid, spec["outline_patch"], top, left, transparent=bg)
            region_global = _global_coords(spec["region_local"], top, left)
            pocket_global = _global_coords(spec["pocket_local"], top, left)
            spec["input_top"] = int(top)
            spec["input_left"] = int(left)
            spec["region_global"] = region_global
            spec["pocket_global"] = pocket_global
            ph, pw = spec["outline_patch"].shape
            bbox_mask[top : top + ph, left : left + pw] = True
            pocket_candidates.extend([tuple(map(int, rc)) for rc in pocket_global])

        input_order = sorted(
            range(n_objects),
            key=lambda i: (int(specs[i]["input_left"]), int(specs[i]["input_top"])),
        )
        count_order = sorted(range(n_objects), key=lambda i: int(specs[i]["count"]))
        area_order = sorted(range(n_objects), key=lambda i: (int(specs[i]["outline_area"]), i))
        if input_order == count_order or area_order == count_order:
            continue

        ok = True
        for spec in specs:
            chosen = _choose_region_cells(spec["region_global"], int(spec["count"]))
            if chosen is None:
                ok = False
                break
            if any(_touches4(cell, marker_cells) for cell in chosen):
                ok = False
                break
            random.shuffle(chosen)
            offset = 0
            for color, amount in spec["marker_recipe"]:
                for r, c in chosen[offset : offset + amount]:
                    grid[r, c] = int(color)
                    marker_cells.add((int(r), int(c)))
                offset += amount
        if not ok:
            continue

        pocket_pool = [cell for cell in pocket_candidates if int(grid[cell]) == bg]
        pocket_distractors = _place_scattered(pocket_pool, 1, marker_cells)
        if pocket_distractors is None:
            continue
        for cell in pocket_distractors:
            grid[cell] = int(random.choice(marker_palette))
            marker_cells.add(cell)

        outside_candidates = [
            (int(r), int(c))
            for r, c in np.argwhere((grid == int(bg)) & ~bbox_mask)
        ]
        extra_distractors = random.randint(1, 4)
        scattered = _place_scattered(outside_candidates, extra_distractors, marker_cells)
        if scattered is None:
            continue
        for cell in scattered:
            grid[cell] = int(random.choice(marker_palette))
            marker_cells.add(cell)

        expected = _expected_output(bg, grid.shape, specs)
        try:
            actual = main(grid)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue
        if int(np.sum(grid == bg)) <= grid.size // 2:
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid counting-queue input")
