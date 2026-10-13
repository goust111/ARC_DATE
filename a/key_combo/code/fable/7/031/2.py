"""ARC task: cast diagonal rays, then settle rigid sprites under gravity."""

from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
YELLOW = 4
CYAN = 8
SPRITE_COLORS = (1, 2, 3, 5, 6, 7, 9)
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as arrays of (row, column) coordinates."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue = deque([(r, c)])
        cells: list[tuple[int, int]] = []
        while queue:
            rr, cc = queue.popleft()
            cells.append((rr, cc))
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _read_scene(grid: np.ndarray) -> tuple[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]]:
    source_cells = np.argwhere(grid == YELLOW)
    assert len(source_cells) == 1, "expected one isolated yellow source pixel"
    source = tuple(map(int, source_cells[0]))

    sprite_mask = (grid != BACKGROUND) & (grid != YELLOW) & (grid != CYAN)
    components = _components(sprite_mask)
    assert components, "expected at least one foreground sprite"
    sprites = []
    for coords in components:
        values = grid[coords[:, 0], coords[:, 1]].astype(int)
        sprites.append((coords.copy(), values.copy()))
    return source, sprites


def _cast_rays(shape: tuple[int, int], source: tuple[int, int], blockers: np.ndarray) -> np.ndarray:
    """Cast four 45-degree rays, excluding the source and each first blocker."""
    h, w = shape
    rays = np.zeros(shape, dtype=bool)
    sr, sc = source
    for dr, dc in DIAGONALS:
        r, c = sr + dr, sc + dc
        while 0 <= r < h and 0 <= c < w:
            if blockers[r, c]:
                break
            rays[r, c] = True
            r += dr
            c += dc
    return rays


def _settle(
    shape: tuple[int, int],
    sprites: list[tuple[np.ndarray, np.ndarray]],
    terrain: np.ndarray,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[int]]:
    """Simultaneously move every currently unobstructed rigid sprite down one row."""
    h, w = shape
    objects = [(coords.copy(), values.copy()) for coords, values in sprites]
    shifts = [0] * len(objects)

    # A sprite can descend at most h-1 times; the extra factor permits alternating
    # motion in stacks when a lower sprite opens a cell for an upper sprite.
    for _ in range(h * max(1, len(objects)) + 1):
        owner = np.full((h, w), -1, dtype=int)
        for idx, (coords, _values) in enumerate(objects):
            assert np.all(owner[coords[:, 0], coords[:, 1]] == -1), "sprites overlap"
            owner[coords[:, 0], coords[:, 1]] = idx

        movable: list[int] = []
        for idx, (coords, _values) in enumerate(objects):
            target_rows = coords[:, 0] + 1
            if np.any(target_rows >= h):
                continue
            target_cols = coords[:, 1]
            if np.any(terrain[target_rows, target_cols]):
                continue
            occupants = owner[target_rows, target_cols]
            if np.any((occupants != -1) & (occupants != idx)):
                continue
            movable.append(idx)

        if not movable:
            return objects, shifts
        for idx in movable:
            objects[idx][0][:, 0] += 1
            shifts[idx] += 1
    raise AssertionError("gravity did not settle within its finite move bound")


def _solve_with_meta(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30x30"
    assert not np.any(g == CYAN), "cyan is reserved for generated rays"

    source, sprites = _read_scene(g)
    initial_sprite_mask = (g != BACKGROUND) & (g != YELLOW)
    rays = _cast_rays(g.shape, source, initial_sprite_mask)
    terrain = rays.copy()
    terrain[source] = True
    settled, shifts = _settle(g.shape, sprites, terrain)

    out = np.zeros_like(g)
    out[rays] = CYAN
    out[source] = YELLOW
    for coords, values in settled:
        assert np.all(out[coords[:, 0], coords[:, 1]] == BACKGROUND), "settled sprites overlap terrain"
        out[coords[:, 0], coords[:, 1]] = values
    return out, source, sprites, rays, settled, shifts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast four sprite-truncated diagonal rays, then settle every floating sprite downward onto the resulting terrain.

    Input structure:
    - A black grid contains one isolated yellow source in its lower half.
    - Several mutually separate 4-connected multicolor sprites float above it; cyan is absent initially.

    Rule:
    - From the yellow source, paint cyan cells along all four 45-degree diagonals until the boundary or an initial sprite pixel, leaving that blocker unchanged.
    - Keeping those rays and the source fixed, repeatedly move each rigid sprite down one row whenever all of its destination cells are in bounds and contain neither terrain nor another sprite. Apply all eligible moves simultaneously until none is possible.
    - Return only the yellow source, cyan rays, and the sprites at their settled positions on black.

    Key insight:
    - Rays are cast against the sprites' initial positions before gravity acts, so truncated ray remnants become permanent terrain that controls the later settling and collisions.
    """
    return _solve_with_meta(grid)[0]


def _colored_patch(mask: np.ndarray) -> np.ndarray:
    """Give every connected mask a visible two- or three-color internal pattern."""
    colors = random.sample(SPRITE_COLORS, random.choice((2, 2, 3)))
    patch = np.zeros(mask.shape, dtype=int)
    cells = list(map(tuple, np.argwhere(mask)))
    random.shuffle(cells)
    for i, (r, c) in enumerate(cells):
        patch[r, c] = colors[i % len(colors)]
    # Ensure each selected color actually occurs.
    for i, color in enumerate(colors):
        r, c = cells[i]
        patch[r, c] = color
    return patch


def _shape_options() -> list[np.ndarray]:
    return [
        np.ones((2, 2), dtype=bool),
        np.array([[1, 1], [1, 0]], dtype=bool),
        np.array([[1, 1], [0, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    ]


def _can_place(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, gap: int = 1) -> bool:
    h, w = mask.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    cells = {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)}
    occupied = set(map(tuple, np.argwhere(grid != BACKGROUND)))
    if cells & occupied:
        return False
    if gap:
        for r, c in cells:
            if any((r + dr, c + dc) in occupied for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                return False
    return True


def _place(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    patch = _colored_patch(mask)
    view = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    view[mask] = patch[mask]


def _rays_after_gravity_first(grid: np.ndarray) -> np.ndarray:
    """Counterfactual used only to reject examples where operation order is invisible."""
    g = np.asarray(grid, dtype=int)
    source, sprites = _read_scene(g)
    source_only = np.zeros_like(g, dtype=bool)
    source_only[source] = True
    settled, _ = _settle(g.shape, sprites, source_only)
    blocker = np.zeros_like(g, dtype=bool)
    for coords, _values in settled:
        blocker[coords[:, 0], coords[:, 1]] = True
    rays = _cast_rays(g.shape, source, blocker)
    out = np.zeros_like(g)
    out[rays] = CYAN
    out[source] = YELLOW
    for coords, values in settled:
        if np.any(out[coords[:, 0], coords[:, 1]] != BACKGROUND):
            return np.full_like(g, -1)
        out[coords[:, 0], coords[:, 1]] = values
    return out


def _semantically_strong(grid: np.ndarray) -> bool:
    """Check rule-level dependencies rather than only main(generator) closure."""
    try:
        out, source, sprites, rays, settled, shifts = _solve_with_meta(grid)
    except (AssertionError, IndexError):
        return False
    if not (2 <= len(sprites) <= 5) or not all(shift > 0 for shift in shifts):
        return False

    # At least one upward ray must be visibly truncated by an initial sprite.
    sr, sc = source
    initial = (grid != BACKGROUND) & (grid != YELLOW)
    truncated = False
    for dr, dc in ((-1, -1), (-1, 1)):
        r, c = sr + dr, sc + dc
        while 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            if initial[r, c]:
                truncated = True
                break
            r, c = r + dr, c + dc
    if not truncated or int(rays.sum()) < 8:
        return False

    # The ray terrain must actually alter settlement.
    source_terrain = np.zeros_like(rays)
    source_terrain[source] = True
    no_ray_settled, _ = _settle(grid.shape, sprites, source_terrain)
    if all(np.array_equal(a[0], b[0]) for a, b in zip(settled, no_ray_settled)):
        return False

    # At least one terminal sprite must rest directly on another sprite.
    owner = np.full(grid.shape, -1, dtype=int)
    for idx, (coords, _values) in enumerate(settled):
        owner[coords[:, 0], coords[:, 1]] = idx
    sprite_collision = False
    for idx, (coords, _values) in enumerate(settled):
        rows = coords[:, 0] + 1
        valid = rows < grid.shape[0]
        labels = owner[rows[valid], coords[:, 1][valid]]
        if np.any((labels != -1) & (labels != idx)):
            sprite_collision = True
    if not sprite_collision:
        return False

    # Explicit counterexample to swapping the two operations.
    if np.array_equal(out, _rays_after_gravity_first(grid)):
        return False

    # Shape/color conservation and pure vertical translation for every sprite.
    for (before, before_values), (after, after_values), shift in zip(sprites, settled, shifts):
        if not np.array_equal(before_values, after_values):
            return False
        if not np.array_equal(before[:, 1], after[:, 1]):
            return False
        if not np.array_equal(before[:, 0] + shift, after[:, 0]):
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    H = random.randint(16, 25)
    W = random.randint(15, 25)
    grid = np.zeros((H, W), dtype=int)
    sr = random.randint(max(10, H // 2 + 2), H - 4)
    sc = random.randint(5, W - 6)
    grid[sr, sc] = YELLOW

    # The lower member of a vertical pair blocks one selected upward ray.
    dc = random.choice((-1, 1))
    distance = random.randint(4, min(7, sr - 5, sc if dc < 0 else W - 1 - sc))
    hit_r, hit_c = sr - distance, sc + dc * distance
    lower_mask = random.choice(_shape_options()[:3]).copy()
    # Put a guaranteed occupied cell exactly at the ray hit coordinate.
    lower_r = hit_r - 1
    lower_c = hit_c if dc > 0 else hit_c - 1
    local_r, local_c = hit_r - lower_r, hit_c - lower_c
    lower_mask[local_r, local_c] = True
    if not _can_place(grid, lower_mask, lower_r, lower_c):
        return None
    _place(grid, lower_mask, lower_r, lower_c)

    # An aligned upper sprite guarantees that sprite-to-sprite blocking matters.
    upper_mask = random.choice(_shape_options()).copy()
    upper_r = lower_r - upper_mask.shape[0] - random.randint(2, 4)
    overlap_lo = max(lower_c, 1)
    overlap_hi = min(lower_c + lower_mask.shape[1] - 1, W - upper_mask.shape[1] - 1)
    if overlap_lo > overlap_hi:
        return None
    upper_c = random.randint(max(1, overlap_lo - upper_mask.shape[1] + 1), overlap_hi)
    if not _can_place(grid, upper_mask, upper_r, upper_c):
        return None
    _place(grid, upper_mask, upper_r, upper_c)

    # Add zero to three independently shaped distractor sprites in the upper field.
    target_count = random.randint(2, 5)
    for _ in range(target_count - 2):
        mask = random.choice(_shape_options()).copy()
        candidates = [
            (r, c)
            for r in range(1, sr - mask.shape[0] - 1)
            for c in range(1, W - mask.shape[1])
            if _can_place(grid, mask, r, c)
        ]
        if not candidates:
            return None
        r, c = random.choice(candidates)
        _place(grid, mask, r, c)

    return grid if _semantically_strong(grid) else None


def generate_input() -> np.ndarray:
    """Generate a varied non-degenerate instance satisfying all puzzle assumptions."""
    for _ in range(1200):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-then-gravity instance")
