from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import Color, bbox, crop_bbox, neighbors4, paste


BG = Color.BLACK
WALL = Color.GREY
SPRITE_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
MIN_INTERIOR = 5
MAX_ATTEMPTS = 400

SYMMETRIC_SHAPES = (
    ((0, 0), (1, 0), (2, 0)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 1), (1, 1), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 2)),
)

ASYMMETRIC_SHAPES = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
)


def _shape_size(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _room_interior(room: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = room
    return r1 - r0 - 1, c1 - c0 - 1


def _eligible_wall_lines(room: tuple[int, int, int, int], axis: str) -> list[int]:
    r0, c0, r1, c1 = room
    if axis == "h":
        lo = r0 + MIN_INTERIOR + 1
        hi = r1 - MIN_INTERIOR - 1
    else:
        lo = c0 + MIN_INTERIOR + 1
        hi = c1 - MIN_INTERIOR - 1
    if lo > hi:
        return []
    return list(range(lo, hi + 1))


def _split_room(
    room: tuple[int, int, int, int], rng: random.Random
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[str, int]] | None:
    axes = []
    if _eligible_wall_lines(room, "h"):
        axes.append("h")
    if _eligible_wall_lines(room, "v"):
        axes.append("v")
    if not axes:
        return None
    axis = rng.choice(axes)
    line = rng.choice(_eligible_wall_lines(room, axis))
    r0, c0, r1, c1 = room
    if axis == "h":
        return (r0, c0, line, c1), (line, c0, r1, c1), (axis, line)
    return (r0, c0, r1, line), (r0, line, r1, c1), (axis, line)


def _build_rooms(
    height: int, width: int, chamber_count: int, rng: random.Random
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    grid = np.full((height, width), BG, dtype=int)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL

    rooms = [(0, 0, height - 1, width - 1)]
    while len(rooms) < chamber_count:
        candidates = [i for i, room in enumerate(rooms) if _split_room(room, rng) is not None]
        if not candidates:
            return None
        idx = rng.choice(candidates)
        split = _split_room(rooms[idx], rng)
        if split is None:
            return None
        room_a, room_b, (axis, line) = split
        if axis == "h":
            grid[line, rooms[idx][1] : rooms[idx][3] + 1] = WALL
        else:
            grid[rooms[idx][0] : rooms[idx][2] + 1, line] = WALL
        rooms[idx : idx + 1] = [room_a, room_b]

    return grid, rooms


def _choose_shape(
    symmetric: bool, room: tuple[int, int, int, int], rng: random.Random
) -> tuple[tuple[tuple[int, int], ...], np.ndarray] | None:
    interior_h, interior_w = _room_interior(room)
    max_h = interior_h - 2
    max_w = interior_w - 2
    catalog = SYMMETRIC_SHAPES if symmetric else ASYMMETRIC_SHAPES
    candidates = [shape for shape in catalog if _shape_size(shape)[0] <= max_h and _shape_size(shape)[1] <= max_w]
    if not candidates:
        return None
    shape = rng.choice(candidates)
    patch_h, patch_w = _shape_size(shape)
    patch = np.zeros((patch_h, patch_w), dtype=int)
    for r, c in shape:
        patch[r, c] = 1
    return shape, patch


def _place_sprite(
    grid: np.ndarray,
    room: tuple[int, int, int, int],
    color: int,
    symmetric: bool,
    rng: random.Random,
) -> bool:
    chosen = _choose_shape(symmetric, room, rng)
    if chosen is None:
        return False
    _shape, mask_patch = chosen
    patch = mask_patch * int(color)
    r0, c0, r1, c1 = room
    patch_h, patch_w = patch.shape
    top_min = r0 + 2
    top_max = r1 - patch_h - 1
    left_min = c0 + 2
    left_max = c1 - patch_w - 1
    if top_min > top_max or left_min > left_max:
        return False
    top = rng.randint(top_min, top_max)
    left = rng.randint(left_min, left_max)
    paste(grid, patch, top, left, transparent=0)
    return True


def _non_wall_components(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == WALL:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] != WALL:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _color_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and mask[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _sprite_info(grid: np.ndarray, component: np.ndarray) -> tuple[int, np.ndarray]:
    values = {int(grid[r, c]) for r, c in component}
    colors = sorted(values - {BG})
    assert len(colors) == 1, "each chamber must contain exactly one sprite color"
    color = colors[0]
    sprite_mask = np.zeros(grid.shape, dtype=bool)
    color_coords = component[np.array([grid[r, c] == color for r, c in component], dtype=bool)]
    sprite_mask[color_coords[:, 0], color_coords[:, 1]] = True
    assert 3 <= int(sprite_mask.sum()) <= 6, "sprite area must be in [3, 6]"
    assert _color_components(sprite_mask) == 1, "sprite must be 4-connected"
    return color, sprite_mask


def _is_vertically_symmetric(sprite_mask: np.ndarray) -> bool:
    sprite_bbox = bbox(sprite_mask)
    assert sprite_bbox is not None
    crop = crop_bbox(sprite_mask.astype(int), sprite_bbox)
    return np.array_equal(crop, np.fliplr(crop))


def main(grid: np.ndarray) -> np.ndarray:
    """Gate chamber flood-filling by the vertical symmetry of each chamber's sprite.

    Input structure:
    - A black background is partitioned into disjoint chambers by one-pixel-thick grey walls.
    - Each chamber contains black empty space plus exactly one monochrome 4-connected sprite in a distinct non-grey color.
    - Sprites are separated from the walls by at least one background cell.

    Rule:
    - Treat grey cells as barriers and find every 4-connected non-wall chamber.
    - In each chamber, crop the unique sprite to its bounding box and test whether that crop is invariant under left-right reflection.
    - If the sprite is vertically symmetric, recolor the entire chamber with the sprite's color; otherwise erase only the sprite to black.

    Key insight:
    - The sprite plays two roles at once: its cropped geometry decides whether filling is allowed, and its own visible color is the fill color used when the symmetry gate opens.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for component in _non_wall_components(g):
        color, sprite_mask = _sprite_info(g, component)
        if _is_vertically_symmetric(sprite_mask):
            out[component[:, 0], component[:, 1]] = color
        else:
            out[sprite_mask] = BG
    return out


def _validate_sample(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30
    assert int((grid == WALL).sum()) > 0

    wall_mask = grid == WALL
    assert _color_components(wall_mask) == 1, "wall network must stay connected"

    chambers = _non_wall_components(grid)
    assert 3 <= len(chambers) <= 5
    branch_flags = []
    colors_seen = set()
    out = main(grid)
    assert np.array_equal(out[wall_mask], grid[wall_mask])

    for chamber in chambers:
        color, sprite_mask = _sprite_info(grid, chamber)
        colors_seen.add(color)
        branch = _is_vertically_symmetric(sprite_mask)
        branch_flags.append(branch)

        assert int(np.sum(np.array([grid[r, c] == BG for r, c in chamber], dtype=int))) > 0
        sprite_coords = np.argwhere(sprite_mask)
        for r, c in sprite_coords:
            assert all(
                not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == WALL)
                for nr, nc in neighbors4(int(r), int(c))
            ), "sprite must keep a one-cell gap from walls"

        chamber_rows = chamber[:, 0]
        chamber_cols = chamber[:, 1]
        chamber_values = out[chamber_rows, chamber_cols]
        if branch:
            assert np.all(chamber_values == color), "symmetric chamber must be fully filled"
        else:
            assert np.all(chamber_values == BG), "asymmetric chamber must end empty"

    assert len(colors_seen) == len(chambers), "sprite colors must be distinct"
    assert any(branch_flags) and not all(branch_flags), "both branches must appear"
    assert not np.array_equal(grid, out), "sample should produce a nontrivial output"


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        height = rng.randint(16, 22)
        width = rng.randint(16, 22)
        max_rooms = 5 if min(height, width) >= 19 else 4
        chamber_count = rng.randint(3, max_rooms)
        built = _build_rooms(height, width, chamber_count, rng)
        if built is None:
            continue
        grid, rooms = built
        rng.shuffle(rooms)

        branch_flags = [True, False]
        while len(branch_flags) < len(rooms):
            branch_flags.append(bool(rng.randint(0, 1)))
        rng.shuffle(branch_flags)
        colors = rng.sample(SPRITE_COLORS, len(rooms))

        ok = True
        for room, color, symmetric in zip(rooms, colors, branch_flags):
            if not _place_sprite(grid, room, color, symmetric, rng):
                ok = False
                break
        if not ok:
            continue

        try:
            _validate_sample(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid symmetry-gated flood-fill puzzle")
