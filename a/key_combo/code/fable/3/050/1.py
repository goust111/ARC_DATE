from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]


def _background(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q: deque[Coord] = deque([(r, c)])
        seen[r, c] = True
        coords: list[Coord] = []
        while q:
            rr, cc = q.popleft()
            coords.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _find_wall_and_doors(grid: np.ndarray, bg: int) -> tuple[str, int, int, dict[int, Coord]]:
    h, w = grid.shape
    row_candidates = [r for r in range(h) if np.all(grid[r, :] != bg)]
    col_candidates = [c for c in range(w) if np.all(grid[:, c] != bg)]

    assert (len(row_candidates) == 1) ^ (len(col_candidates) == 1), "expected one full-span wall"
    if row_candidates:
        orientation = "horizontal"
        wall_index = int(row_candidates[0])
        line = grid[wall_index, :]
    else:
        orientation = "vertical"
        wall_index = int(col_candidates[0])
        line = grid[:, wall_index]

    counts = Counter(map(int, line))
    wall_color, wall_count = counts.most_common(1)[0]
    assert wall_color != bg and wall_count > 3, "expected a dominant wall color"

    centers: dict[int, Coord] = {}
    i = 0
    n = len(line)
    while i < n:
        color = int(line[i])
        if color == wall_color:
            i += 1
            continue
        j = i
        while j < n and int(line[j]) == color:
            j += 1
        assert j - i == 3, "each door segment must be a straight three-pixel run"
        center_axis = (i + j - 1) // 2
        assert color not in centers, "door core colors must be unique"
        if orientation == "horizontal":
            centers[color] = (wall_index, center_axis)
        else:
            centers[color] = (center_axis, wall_index)
        i = j

    assert centers, "expected at least one door segment"
    return orientation, wall_index, int(wall_color), centers


def _source_side_mask(grid: np.ndarray, bg: int, orientation: str, wall_index: int) -> np.ndarray:
    h, w = grid.shape
    mask = np.zeros((h, w), dtype=bool)
    if orientation == "horizontal":
        above = grid[:wall_index, :] != bg
        below = grid[wall_index + 1 :, :] != bg
        assert bool(np.any(above)) ^ bool(np.any(below)), "sprites must occupy exactly one side"
        if np.any(above):
            mask[:wall_index, :] = above
        else:
            mask[wall_index + 1 :, :] = below
    else:
        left = grid[:, :wall_index] != bg
        right = grid[:, wall_index + 1 :] != bg
        assert bool(np.any(left)) ^ bool(np.any(right)), "sprites must occupy exactly one side"
        if np.any(left):
            mask[:, :wall_index] = left
        else:
            mask[:, wall_index + 1 :] = right
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project every two-tone sprite through its same-colored door core as a pinhole inverted image.

    Input structure:
    - A uniform background is split by one full-span horizontal or vertical wall.
    - The wall contains several straight three-pixel door segments; each segment's center/core color
      is unique and differs from the wall and background colors.
    - On exactly one side of the wall are 4-connected sprites, each made of one body color plus
      exactly one key pixel whose color matches one door core; the other side has no foreground.

    Rule:
    - Preserve the wall and all door segments.
    - Erase every original-side sprite.
    - For each sprite, find the door core with the same color as its key pixel and reflect every
      sprite pixel 180 degrees about that core, keeping each pixel's color.
    - Reflected pixels that fall outside the grid or on the wall are discarded.

    Key insight:
    - The door core is a pinhole pivot, not a passage: the key color selects the pivot, and the
      sprite is point-reflected through that pivot so both wall-normal and wall-parallel offsets flip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background(g)
    orientation, wall_index, _wall_color, door_centers = _find_wall_and_doors(g, bg)
    key_colors = set(door_centers)

    source_mask = _source_side_mask(g, bg, orientation, wall_index)
    sprites = _components(source_mask)
    assert sprites, "expected at least one sprite"

    out = g.copy()
    for coords in sprites:
        values = g[coords[:, 0], coords[:, 1]]
        key_locs = [tuple(map(int, rc)) for rc, val in zip(coords, values) if int(val) in key_colors]
        assert len(key_locs) == 1, "each sprite must contain exactly one door-colored key pixel"
        key_color = int(g[key_locs[0]])
        pr, pc = door_centers[key_color]

        out[coords[:, 0], coords[:, 1]] = bg
        for r, c in coords:
            rr = 2 * pr - int(r)
            cc = 2 * pc - int(c)
            if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
                continue
            if (orientation == "horizontal" and rr == wall_index) or (
                orientation == "vertical" and cc == wall_index
            ):
                continue
            assert out[rr, cc] == bg, "reflected sprites must not overlap existing foreground"
            out[rr, cc] = int(g[int(r), int(c)])

    return out


# ---------------- generator ----------------

COLORS = tuple(range(10))
GRID_H_RANGE = (15, 30)
GRID_W_RANGE = (15, 30)
DOOR_COUNT_RANGE = (2, 4)
EXTRA_SPRITE_RANGE = (0, 2)
GENERATION_RETRIES = 500

SPRITE_SHAPES = (
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (3, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (2, 3)),
    ((0, 2), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)),
    ((0, 0), (0, 1), (1, 0), (2, 0), (2, 1), (2, 2)),
)


def _sample_spaced_positions(count: int, lo: int, hi: int, gap: int) -> list[int] | None:
    positions = list(range(lo, hi + 1))
    random.shuffle(positions)
    chosen: list[int] = []
    for pos in positions:
        if all(abs(pos - old) >= gap for old in chosen):
            chosen.append(pos)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _shape_array() -> tuple[np.ndarray, tuple[int, int]]:
    cells = list(random.choice(SPRITE_SHAPES))
    if random.random() < 0.5:
        max_c = max(c for _r, c in cells)
        cells = [(r, max_c - c) for r, c in cells]
    if random.random() < 0.5:
        max_r = max(r for r, _c in cells)
        cells = [(max_r - r, c) for r, c in cells]
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    cells = sorted((r - r0, c - c0) for r, c in cells)
    h = max(r for r, _c in cells) + 1
    w = max(c for _r, c in cells) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    key = random.choice(cells)
    return mask, key


def _paint_wall(
    grid: np.ndarray,
    orientation: str,
    wall_index: int,
    wall_color: int,
    door_colors: list[int],
    door_axes: list[int],
) -> dict[int, Coord]:
    centers: dict[int, Coord] = {}
    if orientation == "horizontal":
        grid[wall_index, :] = wall_color
        for color, c in zip(door_colors, door_axes):
            grid[wall_index, c - 1 : c + 2] = int(color)
            centers[int(color)] = (wall_index, int(c))
    else:
        grid[:, wall_index] = wall_color
        for color, r in zip(door_colors, door_axes):
            grid[r - 1 : r + 2, wall_index] = int(color)
            centers[int(color)] = (int(r), wall_index)
    return centers


def _too_close(cells: set[Coord], occupied: set[Coord], padding: int) -> bool:
    for r, c in cells:
        for dr in range(-padding, padding + 1):
            for dc in range(-padding, padding + 1):
                if (r + dr, c + dc) in occupied:
                    return True
    return False


def _candidate_sprite_positions(
    shape: np.ndarray,
    orientation: str,
    wall_index: int,
    source_side: str,
    h: int,
    w: int,
):
    sh, sw = shape.shape
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            cells = {(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(shape)}
            if orientation == "horizontal":
                if source_side == "before" and all(r < wall_index for r, _c in cells):
                    yield r0, c0, cells
                elif source_side == "after" and all(r > wall_index for r, _c in cells):
                    yield r0, c0, cells
            else:
                if source_side == "before" and all(c < wall_index for _r, c in cells):
                    yield r0, c0, cells
                elif source_side == "after" and all(c > wall_index for _r, c in cells):
                    yield r0, c0, cells


def _reflected_cells(
    cells: set[Coord],
    center: Coord,
    orientation: str,
    wall_index: int,
    shape: tuple[int, int],
) -> tuple[set[Coord], int]:
    h, w = shape
    pr, pc = center
    visible: set[Coord] = set()
    clipped = 0
    for r, c in cells:
        rr = 2 * pr - r
        cc = 2 * pc - c
        on_wall = (orientation == "horizontal" and rr == wall_index) or (
            orientation == "vertical" and cc == wall_index
        )
        if 0 <= rr < h and 0 <= cc < w and not on_wall:
            visible.add((rr, cc))
        else:
            clipped += 1
    return visible, clipped


def _axis_flip_would_differ(cells: set[Coord], center: Coord, orientation: str) -> bool:
    pr, pc = center
    point = {(2 * pr - r, 2 * pc - c) for r, c in cells}
    if orientation == "horizontal":
        axis = {(2 * pr - r, c) for r, c in cells}
    else:
        axis = {(r, 2 * pc - c) for r, c in cells}
    return point != axis


def _place_sprite(
    grid: np.ndarray,
    expected: np.ndarray,
    center: Coord,
    key_color: int,
    body_color: int,
    orientation: str,
    wall_index: int,
    source_side: str,
    occupied_in: set[Coord],
    occupied_out: set[Coord],
    require_clip: bool,
) -> bool:
    shape, key_local = _shape_array()
    h, w = grid.shape
    candidates = list(_candidate_sprite_positions(shape, orientation, wall_index, source_side, h, w))
    random.shuffle(candidates)

    for r0, c0, cells in candidates:
        if _too_close(cells, occupied_in, 1):
            continue
        visible, clipped = _reflected_cells(cells, center, orientation, wall_index, grid.shape)
        if len(visible) < 2:
            continue
        if require_clip and clipped == 0:
            continue
        if not require_clip and clipped > 1 and random.random() < 0.8:
            continue
        if _too_close(visible, occupied_out, 0):
            continue
        if any(expected[r, c] != grid[0, 0] for r, c in visible):
            continue
        if not _axis_flip_would_differ(cells, center, orientation):
            continue

        for r, c in cells:
            grid[r, c] = int(body_color)
        kr, kc = r0 + key_local[0], c0 + key_local[1]
        assert (kr, kc) in cells
        grid[kr, kc] = int(key_color)

        for r, c in cells:
            rr = 2 * center[0] - r
            cc = 2 * center[1] - c
            if (rr, cc) in visible:
                expected[rr, cc] = int(grid[r, c])

        occupied_in.update(cells)
        occupied_out.update(visible)
        return True
    return False


def _nondegenerate(inp: np.ndarray, out: np.ndarray, bg: int, orientation: str, wall_index: int) -> bool:
    changed = inp != out
    if int(np.sum(changed)) < 6:
        return False
    if orientation == "horizontal":
        projected = (out != bg).copy()
        projected[wall_index, :] = False
    else:
        projected = (out != bg).copy()
        projected[:, wall_index] = False
    return len(_components(projected)) >= 2


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    orientation = random.choice(("horizontal", "vertical"))

    line_len = w if orientation == "horizontal" else h
    max_doors = min(DOOR_COUNT_RANGE[1], (line_len - 4) // 3)
    if max_doors < DOOR_COUNT_RANGE[0]:
        return None
    door_count = random.randint(DOOR_COUNT_RANGE[0], max_doors)

    colors = list(COLORS)
    bg, wall_color = random.sample(colors, 2)
    remaining = [c for c in colors if c not in (bg, wall_color)]
    if len(remaining) < door_count + 1:
        return None
    door_colors = random.sample(remaining, door_count)
    body_palette = [c for c in remaining if c not in door_colors]

    if orientation == "horizontal":
        wall_index = random.randint(5, h - 6)
        door_axes = _sample_spaced_positions(door_count, 1, w - 2, 4)
    else:
        wall_index = random.randint(5, w - 6)
        door_axes = _sample_spaced_positions(door_count, 1, h - 2, 4)
    if door_axes is None:
        return None

    grid = np.full((h, w), int(bg), dtype=int)
    expected = np.full((h, w), int(bg), dtype=int)
    centers = _paint_wall(grid, orientation, wall_index, wall_color, door_colors, door_axes)
    _paint_wall(expected, orientation, wall_index, wall_color, door_colors, door_axes)

    before_extent = wall_index
    after_extent = (h - wall_index - 1) if orientation == "horizontal" else (w - wall_index - 1)
    if before_extent < 4 or after_extent < 4:
        return None
    source_side = random.choice(("before", "after"))

    sprite_keys = door_colors[:]
    for _ in range(random.randint(*EXTRA_SPRITE_RANGE)):
        sprite_keys.append(random.choice(door_colors))
    random.shuffle(sprite_keys)

    occupied_in: set[Coord] = set()
    occupied_out: set[Coord] = set()
    clip_index = random.randrange(len(sprite_keys))
    for i, key_color in enumerate(sprite_keys):
        body_color = random.choice(body_palette)
        if not _place_sprite(
            grid,
            expected,
            centers[int(key_color)],
            int(key_color),
            int(body_color),
            orientation,
            wall_index,
            source_side,
            occupied_in,
            occupied_out,
            require_clip=(i == clip_index),
        ):
            return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _nondegenerate(grid, expected, bg, orientation, wall_index):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pinhole reflection input")
