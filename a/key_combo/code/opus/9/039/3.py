from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
ORANGE = 7
MAROON = 9

SPRITE_COLORS = {ORANGE, BLUE, MAROON}
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def bbox_of(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def enclosed_background_cells(subgrid: np.ndarray) -> list[tuple[int, int]]:
    h, w = subgrid.shape
    bg = (subgrid == BLACK)
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    holes = np.argwhere(bg & ~seen)
    return [(int(r), int(c)) for r, c in holes]


def sprite_components(grid: np.ndarray) -> list[dict]:
    comps = []
    for coords in component_coords(np.isin(grid, list(SPRITE_COLORS))):
        colors = [int(grid[r, c]) for r, c in coords]
        control_cells = [(r, c) for r, c in coords if grid[r, c] in (BLUE, MAROON)]
        if len(control_cells) != 1:
            raise AssertionError("each sprite must contain exactly one control pixel")
        control = control_cells[0]
        control_color = int(grid[control])
        if any(color not in SPRITE_COLORS for color in colors):
            raise AssertionError("unexpected sprite color")
        comps.append(
            {
                "coords": coords,
                "coord_set": set(coords),
                "bbox": bbox_of(coords),
                "control": control,
                "control_color": control_color,
            }
        )
    return comps


def fill_blue_holes(grid: np.ndarray, comps: list[dict]) -> np.ndarray:
    out = grid.copy()
    for comp in comps:
        if comp["control_color"] != BLUE:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        sub = out[r0 : r1 + 1, c0 : c1 + 1]
        for rr, cc in enclosed_background_cells(sub):
            out[r0 + rr, c0 + cc] = YELLOW
    return out


def erase_maroon_sprites(grid: np.ndarray, comps: list[dict]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    out = grid.copy()
    sources: list[tuple[int, int]] = []
    for comp in comps:
        if comp["control_color"] != MAROON:
            continue
        sources.append(comp["control"])
        for r, c in comp["coords"]:
            out[r, c] = BLACK
    return out, sources


def ray_cells(base: np.ndarray, source: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = source
    dr, dc = direction
    rr, cc = r + dr, c + dc
    cells: list[tuple[int, int]] = []
    while in_bounds(base.shape, rr, cc) and base[rr, cc] == BLACK:
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def apply_rays(base: np.ndarray, sources: list[tuple[int, int]], left_red: tuple[int, int]) -> np.ndarray:
    out = base.copy()
    all_red = {left_red}
    for source in sources:
        all_red.add(source)
        for direction in DIRS4:
            all_red.update(ray_cells(base, source, direction))
    all_red.update(ray_cells(base, left_red, (0, 1)))
    for r, c in all_red:
        out[r, c] = RED
    return out


def _transform(grid: np.ndarray, *, fill_first: bool = True) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    red_sources = np.argwhere(g == RED)
    if len(red_sources) != 1:
        raise AssertionError("expected exactly one original red source")
    left_red = (int(red_sources[0, 0]), int(red_sources[0, 1]))
    if left_red[1] != 0:
        raise AssertionError("original red source must be on the left edge")

    comps = sprite_components(g)
    if not comps:
        raise AssertionError("expected at least one sprite")

    if fill_first:
        stage1 = fill_blue_holes(g, comps)
        stage2, maroon_sources = erase_maroon_sprites(stage1, comps)
        return apply_rays(stage2, maroon_sources, left_red)

    stage1, maroon_sources = erase_maroon_sprites(g, comps)
    stage2 = apply_rays(stage1, maroon_sources, left_red)
    blue_comps = sprite_components(g)
    return fill_blue_holes(stage2, blue_comps)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill holes inside blue-controlled orange sprites, then erase maroon-controlled sprites and emit red rays.

    Input structure:
    - The grid has black background, 4-connected orange sprites, and one original red pixel on the left edge.
    - Each orange sprite contains exactly one internal control pixel, either blue or maroon, replacing one orange cell.
    - Blue-controlled sprites may enclose background holes; maroon-controlled sprites are erased before emitting rays.

    Rule:
    - Identify each connected sprite formed by orange plus its blue or maroon control cell.
    - First, for every blue-controlled sprite, recolor each enclosed black hole inside that sprite's bounding box to yellow while keeping the sprite itself unchanged.
    - Second, erase every maroon-controlled sprite to black, then from each erased control position emit red rays up, down, left, and right across background cells until the first non-background blocker.
    - Keep the original left-edge red pixel and emit one additional red ray to the right by the same stopping rule.

    Key insight:
    - The control color selects one of two incompatible behaviors, and the order matters: blue hole filling changes which cells become solid blockers before the maroon rays are fired.
    """
    return _transform(grid, fill_first=True)


def make_patch(mask: np.ndarray, control_color: int, control: tuple[int, int]) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = ORANGE
    patch[control] = control_color
    return patch


def solid_mask(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def ring_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def l_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = True
    mask[-1, :] = True
    if h >= 3 and w >= 3:
        mask[-2, 1:] = True
    return mask


def choose_control(mask: np.ndarray) -> tuple[int, int]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    h, w = mask.shape
    interior = [(r, c) for r, c in coords if 0 < r < h - 1 and 0 < c < w - 1]
    edge_non_corner = [
        (r, c)
        for r, c in coords
        if (r in (0, h - 1) or c in (0, w - 1)) and (r, c) not in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    ]
    pool = interior or edge_non_corner or coords
    return random.choice(pool)


def place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w][patch != BLACK] = patch[patch != BLACK]


def expanded_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        for dr, dc in DIRS8:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def can_place_bbox(forbidden: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    if top < 0 or left < 0 or top + h > forbidden.shape[0] or left + w > forbidden.shape[1]:
        return False
    return not np.any(forbidden[top : top + h, left : left + w])


def make_special_outer_and_inner() -> tuple[np.ndarray, np.ndarray]:
    outer_h, outer_w = random.choice([(6, 7), (7, 6), (7, 7), (6, 8), (8, 6)])
    outer_mask = ring_mask(outer_h, outer_w)
    usable_h = outer_h - 4
    usable_w = outer_w - 4
    inner_h, inner_w = random.choice([(2, 3), (3, 2)])
    if inner_h > usable_h or inner_w > usable_w:
        inner_h, inner_w = min(inner_h, usable_h), min(inner_w, usable_w)
    inner_mask = solid_mask(inner_h, inner_w)

    max_top = 2 + (usable_h - inner_h)
    max_left = 2 + (usable_w - inner_w)
    inner_top = random.randint(2, max_top)
    inner_left = random.randint(2, max_left)

    inner_control = choose_control(inner_mask)
    inner_global_control = (inner_top + inner_control[0], inner_left + inner_control[1])

    control_choices = []
    if 0 < inner_global_control[0] < outer_h - 1:
        control_choices.extend([(inner_global_control[0], 0), (inner_global_control[0], outer_w - 1)])
    if 0 < inner_global_control[1] < outer_w - 1:
        control_choices.extend([(0, inner_global_control[1]), (outer_h - 1, inner_global_control[1])])
    outer_control = random.choice(control_choices)

    outer_patch = make_patch(outer_mask, BLUE, outer_control)
    inner_patch = make_patch(inner_mask, MAROON, inner_control)
    return outer_patch, (inner_patch, inner_top, inner_left)


def sample_extra_patch(control_color: int) -> np.ndarray:
    choices: list[np.ndarray] = []
    choices.append(make_patch(solid_mask(2, 3), control_color, choose_control(solid_mask(2, 3))))
    choices.append(make_patch(solid_mask(3, 2), control_color, choose_control(solid_mask(3, 2))))
    choices.append(make_patch(solid_mask(2, 4), control_color, choose_control(solid_mask(2, 4))))
    choices.append(make_patch(solid_mask(3, 3), control_color, choose_control(solid_mask(3, 3))))
    choices.append(make_patch(l_mask(3, 3), control_color, choose_control(l_mask(3, 3))))
    if control_color == BLUE or random.random() < 0.35:
        choices.append(make_patch(ring_mask(3, 4), control_color, choose_control(ring_mask(3, 4))))
        choices.append(make_patch(ring_mask(4, 3), control_color, choose_control(ring_mask(4, 3))))
        choices.append(make_patch(ring_mask(4, 4), control_color, choose_control(ring_mask(4, 4))))
    return random.choice(choices)


def ray_hit_color(base: np.ndarray, source: tuple[int, int], direction: tuple[int, int]) -> int | None:
    r, c = source
    dr, dc = direction
    rr, cc = r + dr, c + dc
    while in_bounds(base.shape, rr, cc):
        if base[rr, cc] != BLACK:
            return int(base[rr, cc])
        rr += dr
        cc += dc
    return None


def first_nonblack_run(grid: np.ndarray, row: int) -> int:
    count = 0
    for c in range(1, grid.shape[1]):
        if grid[row, c] != BLACK:
            break
        count += 1
    return count


def validate_generated(grid: np.ndarray) -> bool:
    try:
        comps = sprite_components(grid)
    except AssertionError:
        return False

    if not (4 <= len(comps) <= 6):
        return False

    control_colors = [comp["control_color"] for comp in comps]
    if BLUE not in control_colors or MAROON not in control_colors:
        return False

    blue_comps = [comp for comp in comps if comp["control_color"] == BLUE]
    maroon_comps = [comp for comp in comps if comp["control_color"] == MAROON]
    if not blue_comps or not maroon_comps:
        return False

    blue_hole_counts = []
    for comp in blue_comps:
        r0, c0, r1, c1 = comp["bbox"]
        holes = enclosed_background_cells(grid[r0 : r1 + 1, c0 : c1 + 1])
        blue_hole_counts.append(len(holes))
    if max(blue_hole_counts, default=0) == 0:
        return False

    aligned = False
    for m in maroon_comps:
        mr, mc = m["control"]
        for b in blue_comps:
            br, bc = b["control"]
            if mr == br or mc == bc:
                aligned = True
                break
        if aligned:
            break
    if not aligned:
        return False

    red_sources = np.argwhere(grid == RED)
    if len(red_sources) != 1 or int(red_sources[0, 1]) != 0:
        return False
    left_red = (int(red_sources[0, 0]), int(red_sources[0, 1]))
    if first_nonblack_run(grid, left_red[0]) < 2:
        return False

    filled = fill_blue_holes(grid, comps)
    erased_after_fill, maroon_sources = erase_maroon_sprites(filled, comps)
    erased_no_fill, _ = erase_maroon_sprites(grid, comps)

    order_interaction = False
    for source in maroon_sources:
        for direction in DIRS4:
            with_fill = ray_cells(erased_after_fill, source, direction)
            without_fill = ray_cells(erased_no_fill, source, direction)
            if len(with_fill) < len(without_fill) and ray_hit_color(erased_after_fill, source, direction) == YELLOW:
                order_interaction = True
                break
        if order_interaction:
            break
    if not order_interaction:
        return False

    out = main(grid)
    swapped = _transform(grid, fill_first=False)
    if np.array_equal(out, swapped):
        return False
    if np.array_equal(out, grid):
        return False
    if int(np.count_nonzero(out == RED)) < 6:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        h = random.randint(16, 24)
        w = random.randint(18, 28)
        grid = np.zeros((h, w), dtype=int)

        outer_patch, (inner_patch, inner_top, inner_left) = make_special_outer_and_inner()
        outer_h, outer_w = outer_patch.shape
        outer_r = random.randint(1, h - outer_h - 2)
        outer_c = random.randint(2, w - outer_w - 2)
        place_patch(grid, outer_patch, outer_r, outer_c)
        place_patch(grid, inner_patch, outer_r + inner_top, outer_c + inner_left)

        forbidden = np.zeros_like(grid, dtype=bool)
        outer_bbox_mask = np.zeros_like(grid, dtype=bool)
        outer_bbox_mask[outer_r : outer_r + outer_h, outer_c : outer_c + outer_w] = True
        forbidden |= expanded_mask(outer_bbox_mask)

        total = random.randint(4, 6)
        remaining = total - 2
        extra_colors = []
        if remaining >= 1:
            extra_colors.append(MAROON)
        if remaining >= 2:
            extra_colors.append(BLUE)
        while len(extra_colors) < remaining:
            extra_colors.append(random.choice([BLUE, MAROON]))
        random.shuffle(extra_colors)

        ok = True
        for control_color in extra_colors:
            patch = sample_extra_patch(control_color)
            ph, pw = patch.shape
            candidates = [
                (r, c)
                for r in range(1, h - ph - 1)
                for c in range(1, w - pw - 1)
                if can_place_bbox(forbidden, r, c, ph, pw)
            ]
            random.shuffle(candidates)
            placed = False
            for r, c in candidates:
                place_patch(grid, patch, r, c)
                mark = np.zeros_like(grid, dtype=bool)
                mark[r : r + ph, c : c + pw] = patch != BLACK
                forbidden |= expanded_mask(mark)
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        red_rows = [
            r
            for r in range(h)
            if grid[r, 0] == BLACK and first_nonblack_run(grid, r) >= 2
        ]
        if not red_rows:
            continue
        red_r = random.choice(red_rows)
        grid[red_r, 0] = RED

        if validate_generated(grid):
            return grid

    raise RuntimeError("failed to generate a valid input after many attempts")
