from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS) not in sys.path:
    sys.path.append(str(_ARC_UTILS))

from arc_utils import colors_present, components_mask, in_bounds, is_contiguous, paste


GREEN = 3
RETRY_LIMIT = 500
PALETTE = tuple(color for color in range(10) if color != GREEN)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _extract_candidate(grid: np.ndarray, orientation: str) -> dict | None:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)

    if orientation == "vertical":
        if W < 5 or (W - 2) % 3 != 0:
            return None
        panel_h = H
        panel_w = (W - 2) // 3
        sep1 = panel_w
        sep2 = 2 * panel_w + 1
        line_color = int(g[0, sep1])
        if not np.all(g[:, sep1] == line_color) or not np.all(g[:, sep2] == line_color):
            return None
        panels = [
            g[:, :panel_w].copy(),
            g[:, sep1 + 1 : sep2].copy(),
            g[:, sep2 + 1 :].copy(),
        ]
    else:
        if H < 5 or (H - 2) % 3 != 0:
            return None
        panel_h = (H - 2) // 3
        panel_w = W
        sep1 = panel_h
        sep2 = 2 * panel_h + 1
        line_color = int(g[sep1, 0])
        if not np.all(g[sep1, :] == line_color) or not np.all(g[sep2, :] == line_color):
            return None
        panels = [
            g[:panel_h, :].copy(),
            g[sep1 + 1 : sep2, :].copy(),
            g[sep2 + 1 :, :].copy(),
        ]

    panel_color_sets = [set(map(int, np.unique(panel))) for panel in panels]
    if any(line_color in color_set for color_set in panel_color_sets):
        return None

    common = panel_color_sets[0] & panel_color_sets[1] & panel_color_sets[2]
    if len(common) != 1:
        return None
    bg = int(next(iter(common)))
    if bg == line_color:
        return None

    fg_areas: list[int] = []
    comp_counts: list[int] = []
    fg_colors: list[int] = []
    for panel in panels:
        fg_colors_here = colors_present(panel, exclude={bg})
        if len(fg_colors_here) != 1:
            return None
        fg_colors.append(int(fg_colors_here[0]))
        fg_mask = panel != bg
        fg_area = int(fg_mask.sum())
        if fg_area == 0:
            return None
        fg_areas.append(fg_area)
        comp_counts.append(len(components_mask(fg_mask, connectivity=8)))

    sprite_area = min(fg_areas)
    if fg_areas.count(sprite_area) != 1:
        return None
    sprite_idx = fg_areas.index(sprite_area)
    if sprite_idx not in (0, 2):
        return None
    if comp_counts[sprite_idx] != 1:
        return None
    if not all(fg_areas[idx] >= sprite_area + 2 for idx in range(3) if idx != sprite_idx):
        return None

    return {
        "orientation": orientation,
        "panel_shape": (panel_h, panel_w),
        "panels": panels,
        "bg": bg,
        "line_color": line_color,
        "sprite_idx": sprite_idx,
        "fg_colors": fg_colors,
        "fg_areas": fg_areas,
    }


def _parse_layout(grid: np.ndarray) -> dict:
    candidates = []
    for orientation in ("vertical", "horizontal"):
        candidate = _extract_candidate(grid, orientation)
        if candidate is not None:
            candidates.append(candidate)
    assert len(candidates) == 1, "expected exactly one valid three-panel split"
    return candidates[0]


def _move_direction(info: dict) -> tuple[int, int]:
    if info["orientation"] == "vertical":
        return (0, 1) if int(info["sprite_idx"]) == 0 else (0, -1)
    return (1, 0) if int(info["sprite_idx"]) == 0 else (-1, 0)


def _sprite_and_masks(info: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    panels = info["panels"]
    if int(info["sprite_idx"]) == 0:
        return panels[0], panels[1], panels[2]
    return panels[2], panels[0], panels[1]


def _leading_edge(coords: list[tuple[int, int]], direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = map(int, direction)
    coord_set = set(coords)
    return [(r, c) for r, c in coords if (r + dr, c + dc) not in coord_set]


def _slide_sprite(
    sprite_coords: list[tuple[int, int]],
    wall_mask: np.ndarray,
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    dr, dc = map(int, direction)
    current = [(int(r), int(c)) for r, c in sprite_coords]
    H, W = map(int, wall_mask.shape)

    for _ in range(H + W + 5):
        blocked = False
        for r, c in current:
            rr = int(r + dr)
            cc = int(c + dc)
            assert in_bounds(wall_mask.shape, rr, cc), "sprite must meet a wall before leaving the panel"
            if bool(wall_mask[rr, cc]):
                blocked = True
                break
        if blocked:
            return current
        current = [(int(r + dr), int(c + dc)) for r, c in current]
    raise AssertionError("sprite failed to stop on a wall")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden XOR wall from two adjacent masks and slide the end-panel sprite toward it.

    Input structure:
    - Two full same-colored separator lines split the grid into three equal panels, either side-by-side or stacked.
    - The two adjacent panels are foreground masks on a shared background, one foreground color per panel.
    - The remaining end panel contains the same background plus one small monochrome sprite.

    Rule:
    - Align the two mask panels cell-by-cell and mark a wall wherever exactly one mask has foreground.
    - Output a single panel of the same size, coloring those XOR-wall cells green.
    - Start the sprite from its position in the end panel and slide it toward the mask side according to the panel layout.
    - Overlap cells where both masks have foreground are passable; stop when some sprite cell has a green wall immediately ahead.

    Key insight:
    - The obstacle is not drawn explicitly in the input: it is the XOR of the two mask foregrounds, and the panel topology alone determines the slide direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    info = _parse_layout(g)
    bg = int(info["bg"])
    sprite_panel, mask_a, mask_b = _sprite_and_masks(info)
    direction = _move_direction(info)

    wall_mask = np.logical_xor(mask_a != bg, mask_b != bg)
    assert bool(wall_mask.any()), "expected a non-empty hidden wall"

    sprite_coords = [tuple(map(int, coord)) for coord in np.argwhere(sprite_panel != bg)]
    assert sprite_coords, "expected a non-empty sprite"
    final_coords = _slide_sprite(sprite_coords, wall_mask, direction)

    out = np.full(wall_mask.shape, bg, dtype=int)
    out[wall_mask] = GREEN
    sprite_color = int(colors_present(sprite_panel, exclude={bg})[0])
    for r, c in final_coords:
        assert int(out[r, c]) == bg, "sprite must stop before the wall, not on it"
        out[r, c] = sprite_color
    return out


def _random_connected_sprite(rng: np.random.Generator, h: int, w: int, area: int) -> np.ndarray | None:
    for _ in range(120):
        start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(0, len(frontier)))])

        if len(cells) != area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if not is_contiguous(mask.astype(int), background=0, connectivity=4):
            continue
        if bool(mask.all()):
            continue
        return mask
    return None


def _anchored_cells(mask: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    return {
        (int(r0 + r), int(c0 + c))
        for r, c in np.argwhere(mask)
    }


def _anchored_offsets(offsets: list[tuple[int, int]], r0: int, c0: int) -> set[tuple[int, int]]:
    return {(int(r0 + r), int(c0 + c)) for r, c in offsets}


def _sample_patch(
    rng: np.random.Generator,
    panel_h: int,
    panel_w: int,
    forbidden: set[tuple[int, int]],
    *,
    max_h: int = 3,
    max_w: int = 3,
) -> set[tuple[int, int]] | None:
    for _ in range(80):
        patch_h = int(rng.integers(1, min(max_h, panel_h) + 1))
        patch_w = int(rng.integers(1, min(max_w, panel_w) + 1))
        r0 = int(rng.integers(0, panel_h - patch_h + 1))
        c0 = int(rng.integers(0, panel_w - patch_w + 1))
        cells = {(r, c) for r in range(r0, r0 + patch_h) for c in range(c0, c0 + patch_w)}
        if cells & forbidden:
            continue
        return cells
    return None


def _sample_free_cell(
    rng: np.random.Generator,
    panel_h: int,
    panel_w: int,
    forbidden: set[tuple[int, int]],
) -> tuple[int, int] | None:
    free = [
        (r, c)
        for r in range(panel_h)
        for c in range(panel_w)
        if (r, c) not in forbidden
    ]
    if not free:
        return None
    return free[int(rng.integers(0, len(free)))]


def _panel_from_cells(
    shape: tuple[int, int],
    bg: int,
    color: int,
    cells: set[tuple[int, int]],
) -> np.ndarray:
    panel = np.full(shape, int(bg), dtype=int)
    for r, c in cells:
        panel[int(r), int(c)] = int(color)
    return panel


def _compose_input(
    orientation: str,
    sprite_first: bool,
    sprite_panel: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    line_color: int,
) -> np.ndarray:
    panel_h, panel_w = map(int, sprite_panel.shape)
    if sprite_first:
        panels = [sprite_panel, mask_a, mask_b]
    else:
        panels = [mask_a, mask_b, sprite_panel]

    if orientation == "vertical":
        grid = np.full((panel_h, panel_w * 3 + 2), int(line_color), dtype=int)
        for idx, panel in enumerate(panels):
            paste(grid, panel, 0, idx * (panel_w + 1))
        return grid

    grid = np.full((panel_h * 3 + 2, panel_w), int(line_color), dtype=int)
    for idx, panel in enumerate(panels):
        paste(grid, panel, idx * (panel_h + 1), 0)
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    orientation = str(rng.choice(["vertical", "horizontal"]))
    sprite_first = bool(rng.integers(0, 2))

    if orientation == "vertical":
        panel_h = int(rng.integers(8, 15))
        panel_w = int(rng.integers(6, 10))
        direction = (0, 1) if sprite_first else (0, -1)
    else:
        panel_h = int(rng.integers(6, 10))
        panel_w = int(rng.integers(8, 15))
        direction = (1, 0) if sprite_first else (-1, 0)

    bg, line_color, mask_a_color, mask_b_color, sprite_color = rng.choice(PALETTE, size=5, replace=False)
    move_vertical = direction[0] != 0

    if move_vertical:
        sprite_h = int(rng.integers(2, min(3, panel_h - 3) + 1))
        sprite_w = int(rng.integers(2, min(4, panel_w - 1) + 1))
    else:
        sprite_h = int(rng.integers(2, min(4, panel_h - 1) + 1))
        sprite_w = int(rng.integers(2, min(3, panel_w - 3) + 1))

    max_area = min(5, sprite_h * sprite_w - 1)
    if max_area < 3:
        return None
    sprite_area = int(rng.integers(3, max_area + 1))
    sprite_mask = _random_connected_sprite(rng, sprite_h, sprite_w, sprite_area)
    if sprite_mask is None:
        return None

    sprite_offsets = [tuple(map(int, coord)) for coord in np.argwhere(sprite_mask)]
    lead_offsets = _leading_edge(sprite_offsets, direction)
    if not lead_offsets:
        return None

    if direction == (0, 1):
        max_dist = panel_w - sprite_w - 1
        if max_dist < 2:
            return None
        distance = int(rng.integers(2, min(5, max_dist) + 1))
        start_r = int(rng.integers(0, panel_h - sprite_h + 1))
        start_c = int(rng.integers(0, panel_w - sprite_w - 1 - distance + 1))
    elif direction == (0, -1):
        max_dist = panel_w - sprite_w - 1
        if max_dist < 2:
            return None
        distance = int(rng.integers(2, min(5, max_dist) + 1))
        start_r = int(rng.integers(0, panel_h - sprite_h + 1))
        start_c = int(rng.integers(distance + 1, panel_w - sprite_w + 1))
    elif direction == (1, 0):
        max_dist = panel_h - sprite_h - 1
        if max_dist < 2:
            return None
        distance = int(rng.integers(2, min(5, max_dist) + 1))
        start_r = int(rng.integers(0, panel_h - sprite_h - 1 - distance + 1))
        start_c = int(rng.integers(0, panel_w - sprite_w + 1))
    else:
        max_dist = panel_h - sprite_h - 1
        if max_dist < 2:
            return None
        distance = int(rng.integers(2, min(5, max_dist) + 1))
        start_r = int(rng.integers(distance + 1, panel_h - sprite_h + 1))
        start_c = int(rng.integers(0, panel_w - sprite_w + 1))

    dr, dc = map(int, direction)
    swept: set[tuple[int, int]] = set()
    ahead_bands: list[set[tuple[int, int]]] = []
    for step in range(distance + 1):
        anchor_r = int(start_r + step * dr)
        anchor_c = int(start_c + step * dc)
        swept |= _anchored_cells(sprite_mask, anchor_r, anchor_c)
        if step < distance:
            ahead_bands.append(_anchored_offsets(lead_offsets, anchor_r + dr, anchor_c + dc))

    final_barrier = set(ahead_bands[-1])
    overlap_cells: set[tuple[int, int]] = set()
    use_overlap_trap = bool(rng.random() < 0.7)
    if use_overlap_trap:
        overlap_step = int(rng.integers(0, distance - 1))
        overlap_cells |= set(ahead_bands[overlap_step])

    exclusive_a: set[tuple[int, int]] = set()
    exclusive_b: set[tuple[int, int]] = set()
    for cell in final_barrier:
        if bool(rng.integers(0, 2)):
            exclusive_a.add(cell)
        else:
            exclusive_b.add(cell)

    reserved_for_extras = set(swept)
    for band in ahead_bands:
        reserved_for_extras |= set(band)

    forbidden = set(reserved_for_extras) | set(exclusive_a) | set(exclusive_b) | set(overlap_cells)
    if use_overlap_trap and bool(rng.random() < 0.5):
        patch = _sample_patch(rng, panel_h, panel_w, forbidden)
        if patch is not None:
            overlap_cells |= patch
            forbidden |= patch

    for _ in range(int(rng.integers(1, 3))):
        patch = _sample_patch(rng, panel_h, panel_w, forbidden)
        if patch is None:
            break
        exclusive_a |= patch
        forbidden |= patch

    for _ in range(int(rng.integers(1, 3))):
        patch = _sample_patch(rng, panel_h, panel_w, forbidden)
        if patch is None:
            break
        exclusive_b |= patch
        forbidden |= patch

    target_a = sprite_area + int(rng.integers(4, 8))
    target_b = sprite_area + int(rng.integers(4, 8))

    while len(exclusive_a | overlap_cells) < target_a:
        choice = _sample_patch(rng, panel_h, panel_w, forbidden, max_h=2, max_w=2)
        if choice is None:
            cell = _sample_free_cell(rng, panel_h, panel_w, forbidden)
            if cell is None:
                break
            choice = {cell}
        exclusive_a |= choice
        forbidden |= choice

    while len(exclusive_b | overlap_cells) < target_b:
        choice = _sample_patch(rng, panel_h, panel_w, forbidden, max_h=2, max_w=2)
        if choice is None:
            cell = _sample_free_cell(rng, panel_h, panel_w, forbidden)
            if cell is None:
                break
            choice = {cell}
        exclusive_b |= choice
        forbidden |= choice

    if len(exclusive_a | overlap_cells) <= sprite_area or len(exclusive_b | overlap_cells) <= sprite_area:
        return None

    sprite_panel = _panel_from_cells((panel_h, panel_w), bg, sprite_color, _anchored_cells(sprite_mask, start_r, start_c))
    mask_a_panel = _panel_from_cells((panel_h, panel_w), bg, mask_a_color, exclusive_a | overlap_cells)
    mask_b_panel = _panel_from_cells((panel_h, panel_w), bg, mask_b_color, exclusive_b | overlap_cells)

    grid = _compose_input(orientation, sprite_first, sprite_panel, mask_a_panel, mask_b_panel, int(line_color))
    if max(grid.shape) > 30:
        return None

    info = _parse_layout(grid)
    parsed_bg = int(info["bg"])
    parsed_direction = _move_direction(info)
    parsed_sprite_panel, parsed_mask_a, parsed_mask_b = _sprite_and_masks(info)
    wall_xor = np.logical_xor(parsed_mask_a != parsed_bg, parsed_mask_b != parsed_bg)
    wall_union = np.logical_or(parsed_mask_a != parsed_bg, parsed_mask_b != parsed_bg)
    sprite_coords = [tuple(map(int, coord)) for coord in np.argwhere(parsed_sprite_panel != parsed_bg)]
    if not sprite_coords:
        return None

    final_xor = _slide_sprite(sprite_coords, wall_xor, parsed_direction)
    if any(not in_bounds(wall_xor.shape, r, c) for r, c in final_xor):
        return None
    if any(bool(wall_xor[r, c]) for r, c in final_xor):
        return None

    if use_overlap_trap:
        final_union = _slide_sprite(sprite_coords, wall_union, parsed_direction)
        if set(final_union) == set(final_xor):
            return None

    overlap_real = np.logical_and(parsed_mask_a != parsed_bg, parsed_mask_b != parsed_bg)
    if use_overlap_trap and not bool(overlap_real.any()):
        return None
    if not use_overlap_trap and bool(overlap_real.any()):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid xor-wall slide input")
