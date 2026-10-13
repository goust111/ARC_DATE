from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
NON_BLACK_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 500

PANEL_HEIGHT_RANGE = (8, 14)
PANEL_WIDTH_RANGE = (8, 14)
SPRITE_BBOX_HEIGHT_RANGE = (2, 4)
SPRITE_BBOX_WIDTH_RANGE = (2, 4)
SPRITE_AREA_RANGE = (3, 7)
STEP_CHOICES = (1, 1, 1, 2, 2, 3, 3, 3)
PIVOT_RING = 1


def get_bg(grid: np.ndarray) -> int:
    """Return the most common color in the grid."""
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as (N,2) coord arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        stack = [(rr, cc)]
        coords = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for nr, nc in neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(np.array(coords, dtype=int))
    return components


def rotate_offsets(offsets: frozenset[tuple[int, int]], k: int) -> frozenset[tuple[int, int]]:
    """Rotate (dr,dc) offsets around the origin by k quarter-turns counterclockwise."""
    steps = int(k) % 4
    rotated = set(offsets)
    for _ in range(steps):
        rotated = {(-dc, dr) for dr, dc in rotated}
    return frozenset(rotated)


def _parse_panel(panel: np.ndarray, bg: int) -> dict:
    """Identify the unique pivot singleton and the unique monochrome sprite component."""
    colors = sorted(set(map(int, panel.reshape(-1))) - {int(bg)})
    components = []
    for color in colors:
        for coords in components_mask(panel == color):
            components.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )

    assert len(components) == 2, "each panel must contain exactly one pivot and one sprite"
    pivot_candidates = [component for component in components if component["area"] == 1]
    sprite_candidates = [component for component in components if component["area"] > 1]
    assert len(pivot_candidates) == 1, "expected a unique singleton pivot pixel"
    assert len(sprite_candidates) == 1, "expected a unique multi-cell sprite"

    pivot_component = pivot_candidates[0]
    sprite_component = sprite_candidates[0]
    pivot = tuple(map(int, pivot_component["coords"][0]))
    sprite_offsets = frozenset(
        (int(r - pivot[0]), int(c - pivot[1])) for r, c in sprite_component["coords"]
    )
    assert sprite_offsets and (0, 0) not in sprite_offsets
    return {
        "pivot": pivot,
        "pivot_color": int(pivot_component["color"]),
        "sprite_color": int(sprite_component["color"]),
        "sprite_offsets": sprite_offsets,
        "sprite_area": int(sprite_component["area"]),
    }


def _infer_scene(grid: np.ndarray) -> dict:
    """Parse the two-frame scene and infer the unique quarter-turn step."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    height, width = g.shape
    assert height % 2 == 1, "expected two equal-height panels separated by one row"
    separator = height // 2
    assert 0 < separator < height - 1, "separator row must leave two non-empty panels"
    assert np.all(g[separator, :] == g[separator, 0]), "expected a solid horizontal separator"

    bg = get_bg(g)
    top = _parse_panel(g[:separator, :], bg)
    bottom = _parse_panel(g[separator + 1 :, :], bg)

    assert top["pivot"] == bottom["pivot"], "pivot position must match between frames"
    assert top["pivot_color"] == bottom["pivot_color"], "pivot color must match between frames"
    assert top["sprite_color"] == bottom["sprite_color"], "sprite color must match between frames"
    assert top["sprite_area"] == bottom["sprite_area"], "sprite area must be preserved by rotation"

    candidates = [
        steps
        for steps in range(4)
        if rotate_offsets(top["sprite_offsets"], steps) == bottom["sprite_offsets"]
    ]
    assert len(candidates) == 1, "frames must imply a unique rotation amount"

    return {
        "panel_shape": (separator, width),
        "pivot": top["pivot"],
        "pivot_color": top["pivot_color"],
        "sprite_color": top["sprite_color"],
        "top": top,
        "bottom": bottom,
        "steps": int(candidates[0]),
    }


def _render_panel(
    shape: tuple[int, int],
    pivot: tuple[int, int],
    pivot_color: int,
    sprite_color: int,
    sprite_offsets: frozenset[tuple[int, int]],
) -> np.ndarray:
    """Render one frame on a black background."""
    out = np.full(shape, BLACK, dtype=int)
    pr, pc = map(int, pivot)
    out[pr, pc] = int(pivot_color)
    for dr, dc in sprite_offsets:
        rr = pr + int(dr)
        cc = pc + int(dc)
        assert 0 <= rr < shape[0] and 0 <= cc < shape[1], "rotated sprite must stay in bounds"
        out[rr, cc] = int(sprite_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the quarter-turn between two scene frames and rotate the sprite once more around the fixed pivot.

    Input structure:
    - A solid horizontal separator row splits the grid into two equal-height panels.
    - Each panel contains one monochrome multi-cell sprite and one singleton pivot pixel on a dominant background.
    - The pivot has the same local position in both panels, and the sprite keeps its color and area across frames.

    Rule:
    - Treat the upper panel as frame 1 and the lower panel as frame 2 of the same scene.
    - Find the unique quarter-turn count that rotates the upper-frame sprite around the pivot onto the lower-frame sprite.
    - Apply that same rotation once more to the lower-frame sprite.
    - Output one panel-sized grid with a black background, the pivot at its original position, and the third-frame sprite.

    Key insight:
    - The two panels are consecutive time steps, not layers to merge: the fixed pivot reveals the rotation already applied, and the answer is the next frame after repeating that motion.
    """
    scene = _infer_scene(grid)
    third_offsets = rotate_offsets(scene["bottom"]["sprite_offsets"], scene["steps"])
    return _render_panel(
        scene["panel_shape"],
        scene["pivot"],
        scene["pivot_color"],
        scene["sprite_color"],
        third_offsets,
    )


def _mask_from_cells(cells: set[tuple[int, int]], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _sample_connected_patch(height: int, width: int, area: int) -> set[tuple[int, int]] | None:
    cells = [(r, c) for r in range(height) for c in range(width)]
    chosen = {random.choice(cells)}
    while len(chosen) < area:
        frontier = {
            (nr, nc)
            for r, c in chosen
            for nr, nc in neighbors4(r, c)
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in chosen
        }
        if not frontier:
            return None
        chosen.add(random.choice(tuple(frontier)))

    if not _is_connected(chosen):
        return None
    rows = {r for r, _ in chosen}
    cols = {c for _, c in chosen}
    if rows != set(range(height)) or cols != set(range(width)):
        return None

    mask = _mask_from_cells(chosen, height, width)
    if any(np.array_equal(mask, np.rot90(mask, k=steps)) for steps in (1, 2, 3)):
        return None
    return chosen


def _pivot_candidates(
    patch_cells: set[tuple[int, int]], height: int, width: int
) -> list[tuple[int, int]]:
    candidates = []
    for pr in range(-PIVOT_RING, height + PIVOT_RING):
        for pc in range(-PIVOT_RING, width + PIVOT_RING):
            if (pr, pc) in patch_cells:
                continue
            distance = min(abs(r - pr) + abs(c - pc) for r, c in patch_cells)
            if 1 <= distance <= 3:
                candidates.append((pr, pc))
    random.shuffle(candidates)
    return candidates


def _feasible_pivot_positions(
    rotations: list[frozenset[tuple[int, int]]], panel_height: int, panel_width: int
) -> list[tuple[int, int]]:
    union = set().union(*rotations)
    min_dr = min(dr for dr, _ in union)
    max_dr = max(dr for dr, _ in union)
    min_dc = min(dc for _, dc in union)
    max_dc = max(dc for _, dc in union)

    row_lo = max(1, 1 - min_dr)
    row_hi = min(panel_height - 2, panel_height - 2 - max_dr)
    col_lo = max(1, 1 - min_dc)
    col_hi = min(panel_width - 2, panel_width - 2 - max_dc)
    if row_lo > row_hi or col_lo > col_hi:
        return []

    positions = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
    random.shuffle(positions)
    return positions


def _render_input(
    panel_height: int,
    panel_width: int,
    separator_color: int,
    pivot: tuple[int, int],
    pivot_color: int,
    sprite_color: int,
    top_offsets: frozenset[tuple[int, int]],
    bottom_offsets: frozenset[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((panel_height * 2 + 1, panel_width), BLACK, dtype=int)
    grid[panel_height, :] = int(separator_color)
    grid[:panel_height, :] = _render_panel(
        (panel_height, panel_width), pivot, pivot_color, sprite_color, top_offsets
    )
    grid[panel_height + 1 :, :] = _render_panel(
        (panel_height, panel_width), pivot, pivot_color, sprite_color, bottom_offsets
    )
    return grid


def _expected_output_from_offsets(
    panel_shape: tuple[int, int],
    pivot: tuple[int, int],
    pivot_color: int,
    sprite_color: int,
    offsets: frozenset[tuple[int, int]],
) -> np.ndarray:
    return _render_panel(panel_shape, pivot, pivot_color, sprite_color, offsets)


def _semantic_checks(grid: np.ndarray, expected: np.ndarray, true_step: int) -> bool:
    try:
        scene = _infer_scene(grid)
    except AssertionError:
        return False

    if scene["steps"] != int(true_step):
        return False
    if scene["steps"] not in {1, 2, 3}:
        return False

    top_offsets = scene["top"]["sprite_offsets"]
    bottom_offsets = scene["bottom"]["sprite_offsets"]
    if rotate_offsets(top_offsets, scene["steps"]) != bottom_offsets:
        return False
    if len([k for k in range(4) if rotate_offsets(top_offsets, k) == bottom_offsets]) != 1:
        return False

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if actual.shape != scene["panel_shape"]:
        return False

    pivot_mask = actual == scene["pivot_color"]
    if int(np.count_nonzero(pivot_mask)) != 1:
        return False
    return True


def _generate_instance() -> tuple[np.ndarray, np.ndarray, int] | None:
    panel_height = random.randint(*PANEL_HEIGHT_RANGE)
    panel_width = random.randint(*PANEL_WIDTH_RANGE)
    separator_color, pivot_color, sprite_color = random.sample(NON_BLACK_COLORS, 3)

    bbox_h = random.randint(*SPRITE_BBOX_HEIGHT_RANGE)
    bbox_w = random.randint(*SPRITE_BBOX_WIDTH_RANGE)
    max_area = min(SPRITE_AREA_RANGE[1], bbox_h * bbox_w)
    if max_area < SPRITE_AREA_RANGE[0]:
        return None
    area = random.randint(SPRITE_AREA_RANGE[0], max_area)

    patch_cells = _sample_connected_patch(bbox_h, bbox_w, area)
    if patch_cells is None:
        return None

    for pivot_local in _pivot_candidates(patch_cells, bbox_h, bbox_w):
        base_offsets = frozenset(
            (r - pivot_local[0], c - pivot_local[1]) for r, c in patch_cells
        )
        rotations = [rotate_offsets(base_offsets, steps) for steps in range(4)]
        if len({tuple(sorted(rotation)) for rotation in rotations}) != 4:
            continue

        pivot_positions = _feasible_pivot_positions(rotations, panel_height, panel_width)
        if not pivot_positions:
            continue

        pivot = random.choice(pivot_positions)
        top_rotation = random.randrange(4)
        step = random.choice(STEP_CHOICES)
        top_offsets = rotations[top_rotation]
        bottom_offsets = rotations[(top_rotation + step) % 4]
        third_offsets = rotations[(top_rotation + 2 * step) % 4]

        grid = _render_input(
            panel_height,
            panel_width,
            separator_color,
            pivot,
            pivot_color,
            sprite_color,
            top_offsets,
            bottom_offsets,
        )
        expected = _expected_output_from_offsets(
            (panel_height, panel_width), pivot, pivot_color, sprite_color, third_offsets
        )
        return grid, expected, int(step)
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        instance = _generate_instance()
        if instance is None:
            continue
        grid, expected, step = instance
        if _semantic_checks(grid, expected, step):
            return grid
    raise RuntimeError("failed to generate a valid two-frame rotation scene")
