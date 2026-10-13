from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
COLORS = list(range(1, 10))
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]

GEN_RETRIES = 400


def _neighbors(r: int, c: int, shape: tuple[int, int], connectivity: int) -> list[tuple[int, int]]:
    dirs = DIRS4 if connectivity == 4 else DIRS8
    out = []
    h, w = shape
    for dr, dc in dirs:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, mask.shape, connectivity):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        sprite[r - r0, c - c0] = int(grid[r, c])
    return sprite, (r0, c0, r1, c1)


def _touches_mask(coords: list[tuple[int, int]], mask: np.ndarray) -> bool:
    for r, c in coords:
        for nr, nc in _neighbors(r, c, mask.shape, 4):
            if mask[nr, nc]:
                return True
    return False


def _largest_nonblack_component(grid: np.ndarray) -> tuple[int, np.ndarray]:
    best_color = None
    best_coords: list[tuple[int, int]] | None = None
    for color in np.unique(grid):
        color = int(color)
        if color == BLACK:
            continue
        for comp in _components(grid == color, connectivity=4):
            if best_coords is None or len(comp) > len(best_coords):
                best_color = color
                best_coords = comp
    assert best_color is not None and best_coords is not None
    mask = np.zeros_like(grid, dtype=bool)
    for r, c in best_coords:
        mask[r, c] = True
    return best_color, mask


def _extract_templates(
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[np.ndarray, tuple[int, int]]]]:
    catalog_color, catalog_bg_mask = _largest_nonblack_component(grid)
    candidate_mask = (grid != BLACK) & (grid != catalog_color)

    template_cells = np.zeros_like(grid, dtype=bool)
    templates: dict[int, tuple[np.ndarray, tuple[int, int]]] = {}

    for comp in _components(candidate_mask, connectivity=8):
        if len(comp) <= 1 or not _touches_mask(comp, catalog_bg_mask):
            continue

        counts = Counter(int(grid[r, c]) for r, c in comp)
        if len(counts) != 2:
            continue
        minority_colors = [color for color, count in counts.items() if count == 1]
        if len(minority_colors) != 1:
            continue

        anchor_color = minority_colors[0]
        sprite, bbox = _crop_component(grid, comp)
        anchor_abs = next((r, c) for r, c in comp if int(grid[r, c]) == anchor_color)
        anchor = (anchor_abs[0] - bbox[0], anchor_abs[1] - bbox[1])

        if anchor_color in templates:
            raise AssertionError("expected unique minority colors across templates")
        templates[anchor_color] = (sprite, anchor)
        for r, c in comp:
            template_cells[r, c] = True

    if not templates:
        raise AssertionError("expected at least one catalog template")
    return catalog_bg_mask, template_cells, templates


def _stamp(
    out: np.ndarray,
    sprite: np.ndarray,
    top: int,
    left: int,
    protected: np.ndarray,
) -> None:
    h, w = out.shape
    sh, sw = sprite.shape
    for sr in range(sh):
        for sc in range(sw):
            value = int(sprite[sr, sc])
            if value == BLACK:
                continue
            rr = top + sr
            cc = left + sc
            if 0 <= rr < h and 0 <= cc < w and not protected[rr, cc]:
                out[rr, cc] = value


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each workspace key pixel with its matching catalog sprite, aligning on the template's unique minority-color cell.

    Input structure:
    - One connected non-black catalog background region coexists with an otherwise black workspace.
    - Inside the catalog sit several two-color templates; each template has many cells of one main color and exactly one cell of a different minority color.
    - The workspace contains scattered single-pixel markers whose colors may or may not match template minority colors.

    Rule:
    - Find the largest non-black monochrome component as the catalog background and extract every adjacent multicolor template.
    - For each template, treat its unique minority-color pixel both as the lookup key and as the anchor coordinate inside the template sprite.
    - For every single workspace pixel whose color matches a template key, stamp the whole template so that the template's minority-color cell lands on that pixel, clipping at the grid boundary and leaving unmatched markers unchanged.

    Key insight:
    - The same minority-color pixel inside each catalog template serves two roles at once: it identifies which template to use and it tells you exactly how to align that template when stamping.
    """
    g = np.asarray(grid, dtype=int)
    catalog_bg_mask, template_cells, templates = _extract_templates(g)
    protected = catalog_bg_mask | template_cells

    workspace_mask = (g != BLACK) & (~protected)
    out = g.copy()
    for comp in _components(workspace_mask, connectivity=4):
        if len(comp) != 1:
            continue
        r, c = comp[0]
        color = int(g[r, c])
        if color not in templates:
            continue
        sprite, anchor = templates[color]
        _stamp(out, sprite, r - anchor[0], c - anchor[1], protected)
    return out


def _catalog_rect(h: int, w: int, side: str, thickness: int) -> tuple[int, int, int, int]:
    if side == "top":
        return 0, 0, thickness - 1, w - 1
    if side == "bottom":
        return h - thickness, 0, h - 1, w - 1
    if side == "left":
        return 0, 0, h - 1, thickness - 1
    return 0, w - thickness, h - 1, w - 1


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    cells = [start]
    frontier = set()
    frontier.update(_neighbors(start[0], start[1], mask.shape, 4))

    while len(cells) < area and frontier:
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        if mask[r, c]:
            continue
        mask[r, c] = True
        cells.append((r, c))
        for nr, nc in _neighbors(r, c, mask.shape, 4):
            if not mask[nr, nc]:
                frontier.add((nr, nc))
    return mask


def _generate_template_sprite(max_h: int, max_w: int, main_color: int, anchor_color: int) -> tuple[np.ndarray, tuple[int, int]]:
    height = random.randint(2, max_h)
    width = random.randint(2, max_w)
    max_area = min(height * width, 7)
    area = random.randint(3, max_area)
    mask = _random_connected_mask(height, width, area)
    coords = [(r, c) for r in range(height) for c in range(width) if mask[r, c]]
    anchor = random.choice(coords)

    sprite = np.zeros((height, width), dtype=int)
    for r, c in coords:
        sprite[r, c] = main_color
    sprite[anchor] = anchor_color
    return sprite, anchor


def _can_place_bbox(
    occupied: np.ndarray,
    bounds: tuple[int, int, int, int],
    top: int,
    left: int,
    height: int,
    width: int,
) -> bool:
    r0, c0, r1, c1 = bounds
    if top < r0 + 1 or left < c0 + 1 or top + height - 1 > r1 - 1 or left + width - 1 > c1 - 1:
        return False
    rr0 = max(r0, top - 1)
    cc0 = max(c0, left - 1)
    rr1 = min(r1, top + height)
    cc1 = min(c1, left + width)
    return not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _visible_stamp_cells(
    shape: tuple[int, int],
    sprite: np.ndarray,
    anchor: tuple[int, int],
    marker: tuple[int, int],
) -> tuple[list[tuple[int, int, int]], bool]:
    h, w = shape
    top = marker[0] - anchor[0]
    left = marker[1] - anchor[1]
    visible: list[tuple[int, int, int]] = []
    clipped = False
    for sr in range(sprite.shape[0]):
        for sc in range(sprite.shape[1]):
            value = int(sprite[sr, sc])
            if value == BLACK:
                continue
            rr = top + sr
            cc = left + sc
            if 0 <= rr < h and 0 <= cc < w:
                visible.append((rr, cc, value))
            else:
                clipped = True
    return visible, clipped


def _expand_cells(cells: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    h, w = shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))
    return expanded


def _pick_marker_position(
    shape: tuple[int, int],
    catalog_zone: np.ndarray,
    sprite: np.ndarray,
    anchor: tuple[int, int],
    blocked: set[tuple[int, int]],
    require_clipped: bool,
) -> tuple[tuple[int, int], list[tuple[int, int, int]]] | None:
    h, w = shape
    candidates = []
    for r in range(h):
        for c in range(w):
            if catalog_zone[r, c]:
                continue
            visible, clipped = _visible_stamp_cells(shape, sprite, anchor, (r, c))
            if not visible:
                continue
            if require_clipped:
                if not clipped or len(visible) < 2:
                    continue
            elif clipped:
                continue

            coords = [(rr, cc) for rr, cc, _ in visible]
            if any(catalog_zone[rr, cc] for rr, cc in coords):
                continue
            if any((rr, cc) in blocked for rr, cc in coords):
                continue
            candidates.append(((r, c), visible))
    if not candidates:
        return None
    return random.choice(candidates)


def _pick_unmatched_positions(
    shape: tuple[int, int],
    catalog_zone: np.ndarray,
    blocked: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    h, w = shape
    choices = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if not catalog_zone[r, c] and (r, c) not in blocked
    ]
    random.shuffle(choices)
    picked: list[tuple[int, int]] = []
    local_blocked = set(blocked)
    for pos in choices:
        if pos in local_blocked:
            continue
        picked.append(pos)
        local_blocked |= _expand_cells([pos], shape)
        if len(picked) == count:
            return picked
    return None


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(["top", "bottom", "left", "right"])
    height = random.randint(18, 26)
    width = random.randint(18, 26)
    thickness = random.randint(7, 10)
    if side in ("top", "bottom") and thickness >= height - 4:
        return None
    if side in ("left", "right") and thickness >= width - 4:
        return None

    catalog_box = _catalog_rect(height, width, side, thickness)
    catalog_color = random.choice(COLORS)
    grid = np.zeros((height, width), dtype=int)
    catalog_zone = np.zeros_like(grid, dtype=bool)
    r0, c0, r1, c1 = catalog_box
    grid[r0 : r1 + 1, c0 : c1 + 1] = catalog_color
    catalog_zone[r0 : r1 + 1, c0 : c1 + 1] = True

    n_templates = random.randint(2, 4)
    remaining = [c for c in COLORS if c != catalog_color]
    minority_colors = random.sample(remaining, n_templates)
    main_palette = [c for c in remaining if c not in minority_colors]
    if len(main_palette) < n_templates:
        return None
    main_colors = random.sample(main_palette, n_templates)

    occupied_catalog = np.zeros_like(grid, dtype=bool)
    template_specs = []
    for idx in range(n_templates):
        max_h = min(5, r1 - r0 - 1)
        max_w = min(5, c1 - c0 - 1)
        if max_h < 2 or max_w < 2:
            return None
        sprite, anchor = _generate_template_sprite(max_h, max_w, main_colors[idx], minority_colors[idx])

        placements = []
        for top in range(r0 + 1, r1 - sprite.shape[0] + 2):
            for left in range(c0 + 1, c1 - sprite.shape[1] + 2):
                if _can_place_bbox(occupied_catalog, catalog_box, top, left, sprite.shape[0], sprite.shape[1]):
                    placements.append((top, left))
        if not placements:
            return None

        top, left = random.choice(placements)
        occupied_catalog[top : top + sprite.shape[0], left : left + sprite.shape[1]] = True
        for sr in range(sprite.shape[0]):
            for sc in range(sprite.shape[1]):
                value = int(sprite[sr, sc])
                if value != BLACK:
                    grid[top + sr, left + sc] = value
        template_specs.append(
            {
                "sprite": sprite,
                "anchor": anchor,
                "anchor_color": minority_colors[idx],
                "main_color": main_colors[idx],
            }
        )

    want_unused = n_templates > 2 and random.random() < 0.65
    max_used = n_templates - 1 if want_unused else n_templates
    used_count = random.randint(1, max_used)
    used_indices = random.sample(range(n_templates), used_count)

    copy_counts = {idx: 1 for idx in used_indices}
    if random.random() < 0.45:
        copy_counts[random.choice(used_indices)] += 1

    plans = [(idx, False) for idx in used_indices for _ in range(copy_counts[idx])]
    if random.random() < 0.55:
        clip_i = random.randrange(len(plans))
        plans[clip_i] = (plans[clip_i][0], True)
    random.shuffle(plans)
    plans.sort(key=lambda item: not item[1])

    blocked: set[tuple[int, int]] = set()
    matched_placements = []
    for idx, require_clipped in plans:
        spec = template_specs[idx]
        picked = _pick_marker_position(
            grid.shape,
            catalog_zone,
            spec["sprite"],
            spec["anchor"],
            blocked,
            require_clipped=require_clipped,
        )
        if picked is None:
            return None
        marker, visible = picked
        coords = [(r, c) for r, c, _ in visible]
        blocked |= _expand_cells(coords, grid.shape)
        matched_placements.append((spec, marker, visible))

    unmatched_colors = [color for color in COLORS if color not in minority_colors and color != catalog_color]
    unmatched_count = random.choice([0, 1, 1, 2])
    unmatched_positions = _pick_unmatched_positions(grid.shape, catalog_zone, blocked, unmatched_count)
    if unmatched_positions is None:
        return None

    input_grid = grid.copy()
    expected = grid.copy()

    for spec, marker, visible in matched_placements:
        input_grid[marker] = spec["anchor_color"]
        for r, c, value in visible:
            expected[r, c] = value

    if unmatched_count:
        colors = random.choices(unmatched_colors, k=unmatched_count)
        for (r, c), color in zip(unmatched_positions, colors):
            input_grid[r, c] = color

    if not np.array_equal(main(input_grid), expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
