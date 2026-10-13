"""ARC task: repair same-palette tampering in a mirror-alternating tiling."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

import numpy as np


MAX_GENERATION_ATTEMPTS = 500
PALETTE = tuple(range(10))


@dataclass(frozen=True)
class _Structure:
    tile_h: int
    tile_w: int
    direction: str  # "rows" or "cols"
    errors: int
    repaired: np.ndarray


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _canonical_coord(
    r: int, c: int, tile_h: int, tile_w: int, direction: str
) -> tuple[int, int]:
    """Map a canvas (r,c) to its coordinate in the unmirrored base tile."""
    tile_row, local_r = divmod(int(r), int(tile_h))
    tile_col, local_c = divmod(int(c), int(tile_w))
    if direction == "rows" and tile_row % 2:
        local_r = tile_h - 1 - local_r
    elif direction == "cols" and tile_col % 2:
        local_c = tile_w - 1 - local_c
    return int(local_r), int(local_c)


def _repair_for_structure(
    grid: np.ndarray, tile_h: int, tile_w: int, direction: str
) -> _Structure | None:
    """Return an orbit-vote repair if every orbit has at most one dissenter."""
    height, width = map(int, grid.shape)
    tile_rows, tile_cols = height // tile_h, width // tile_w
    if tile_rows * tile_cols < 3:
        return None
    if direction == "rows" and tile_rows < 2:
        return None
    if direction == "cols" and tile_cols < 2:
        return None

    orbit_members: dict[tuple[int, int], list[tuple[int, int]]] = {
        (r, c): [] for r in range(tile_h) for c in range(tile_w)
    }
    for r in range(height):
        for c in range(width):
            key = _canonical_coord(r, c, tile_h, tile_w, direction)
            orbit_members[key].append((r, c))

    repaired = grid.copy()
    errors = 0
    for members in orbit_members.values():
        counts = Counter(int(grid[r, c]) for r, c in members)
        winner, winner_count = counts.most_common(1)[0]
        # This is the puzzle's semantic promise, not a soft similarity score.
        if winner_count < len(members) - 1:
            return None
        if len(counts) > 1 and len(counts.most_common()) > 1:
            if counts.most_common()[0][1] == counts.most_common()[1][1]:
                return None
        for r, c in members:
            if int(grid[r, c]) != winner:
                repaired[r, c] = winner
                errors += 1

    # The task always contains camouflage patches; an error-free hypothesis is
    # an overfitted super-tile rather than the intended repair structure.
    if errors == 0:
        return None
    base = repaired[:tile_h, :tile_w]
    if direction == "rows" and np.array_equal(base, np.flipud(base)):
        return None
    if direction == "cols" and np.array_equal(base, np.fliplr(base)):
        return None
    return _Structure(tile_h, tile_w, direction, errors, repaired)


def _infer_structure(grid: np.ndarray) -> _Structure:
    height, width = map(int, grid.shape)
    candidates: list[_Structure] = []
    for tile_h in _divisors(height):
        for tile_w in _divisors(width):
            if tile_h == height and tile_w == width:
                continue
            for direction in ("rows", "cols"):
                candidate = _repair_for_structure(
                    grid, tile_h, tile_w, direction
                )
                if candidate is not None:
                    candidates.append(candidate)

    if not candidates:
        raise ValueError("no mirror-alternating tiling satisfies the orbit promise")

    # The base tile is the smallest repeating explanatory unit. A doubled plain
    # period may also fit, but is not the fundamental mirror-alternating tile.
    min_area = min(c.tile_h * c.tile_w for c in candidates)
    fundamental = [c for c in candidates if c.tile_h * c.tile_w == min_area]
    if len(fundamental) != 1:
        raise ValueError("tiling structure is not uniquely determined")
    return fundamental[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair same-palette camouflage patches by voting within mirror-alternating tile orbits.

    Input structure:
    - The full rectangular grid is tiled by copies of one smallest base tile.
    - Successive tile columns are left-right mirrors, or successive tile rows are top-bottom mirrors; the tile size and alternating direction are unmarked.
    - Several rectangular patches use only existing pattern colors, and every normalized tile orbit contains at most one patched cell.

    Rule:
    - Infer the unique smallest tile dimensions and whether row-blocks or column-blocks alternate by reflection.
    - Normalize every cell to its coordinate in the unmirrored tile, then replace the sole color dissenter in each orbit by the common color of all other orbit members.
    - Leave every already unanimous orbit unchanged and preserve the grid dimensions and palette.

    Key insight:
    - Ordinary repetition creates systematic disagreements on reflected blocks, whereas reflection-aware orbit voting makes all disagreements collapse exactly onto the unmarked patches.
    """
    g = np.asarray(grid)
    if g.ndim != 2:
        raise ValueError("expected a two-dimensional grid")
    if not np.issubdtype(g.dtype, np.integer):
        raise TypeError("ARC colors must be integer ids")
    if min(g.shape) < 6 or max(g.shape) > 30:
        raise ValueError("grid dimensions must lie within the ARC range")
    return _infer_structure(g.astype(int, copy=False)).repaired


def _render_tiling(
    tile: np.ndarray, tile_rows: int, tile_cols: int, direction: str
) -> np.ndarray:
    tile_h, tile_w = map(int, tile.shape)
    out = np.empty((tile_h * tile_rows, tile_w * tile_cols), dtype=int)
    for block_r in range(tile_rows):
        for block_c in range(tile_cols):
            patch = tile
            if direction == "rows" and block_r % 2:
                patch = np.flipud(tile)
            elif direction == "cols" and block_c % 2:
                patch = np.fliplr(tile)
            r0, c0 = block_r * tile_h, block_c * tile_w
            out[r0 : r0 + tile_h, c0 : c0 + tile_w] = patch
    return out


def _random_tile(tile_h: int, tile_w: int) -> np.ndarray | None:
    n_colors = random.choice((3, 3, 4))
    colors = random.sample(PALETTE, n_colors)
    tile = np.array(
        random.choices(colors, weights=[5, 3, 2, 2][:n_colors], k=tile_h * tile_w),
        dtype=int,
    ).reshape(tile_h, tile_w)
    counts = Counter(map(int, tile.flat))
    if any(counts[color] < 2 for color in colors):
        return None
    if np.array_equal(tile, np.fliplr(tile)) or np.array_equal(tile, np.flipud(tile)):
        return None
    # Avoid a smaller ordinary row or column period inside the sampled tile.
    for d in range(1, tile_h):
        if tile_h % d == 0 and np.array_equal(tile, np.tile(tile[:d], (tile_h // d, 1))):
            return None
    for d in range(1, tile_w):
        if tile_w % d == 0 and np.array_equal(tile, np.tile(tile[:, :d], (1, tile_w // d))):
            return None
    return tile


def _rect_candidates(tile_h: int, tile_w: int) -> list[tuple[int, int, int, int]]:
    result = []
    for rect_h in range(1, min(3, tile_h) + 1):
        for rect_w in range(1, min(3, tile_w) + 1):
            area = rect_h * rect_w
            if not 2 <= area <= 6:
                continue
            for r0 in range(tile_h - rect_h + 1):
                for c0 in range(tile_w - rect_w + 1):
                    result.append((r0, c0, rect_h, rect_w))
    random.shuffle(result)
    return result


def _patch_grid(
    clean: np.ndarray,
    tile_h: int,
    tile_w: int,
    tile_rows: int,
    tile_cols: int,
    direction: str,
) -> tuple[np.ndarray, list[set[tuple[int, int]]]] | None:
    inp = clean.copy()
    patch_count = random.choice((2, 3, 3, 4))
    blocks = [(br, bc) for br in range(tile_rows) for bc in range(tile_cols)]
    random.shuffle(blocks)

    used_orbits: set[tuple[int, int]] = set()
    used_blocks: list[tuple[int, int]] = []
    patch_orbits: list[set[tuple[int, int]]] = []
    phases: set[int] = set()
    palette = sorted(set(map(int, clean.flat)))

    for block_r, block_c in blocks:
        if len(patch_orbits) == patch_count:
            break
        if any(abs(block_r - br) + abs(block_c - bc) < 2 for br, bc in used_blocks):
            continue
        phase = block_r % 2 if direction == "rows" else block_c % 2
        # Reserve room for both an unreflected and reflected patch.
        if len(patch_orbits) == patch_count - 1 and len(phases) == 1 and phase in phases:
            continue
        for local_r, local_c, rect_h, rect_w in _rect_candidates(tile_h, tile_w):
            physical = []
            normalized = set()
            for dr in range(rect_h):
                for dc in range(rect_w):
                    r = block_r * tile_h + local_r + dr
                    c = block_c * tile_w + local_c + dc
                    physical.append((r, c))
                    normalized.add(
                        _canonical_coord(r, c, tile_h, tile_w, direction)
                    )
            if normalized & used_orbits:
                continue
            for r, c in physical:
                choices = [color for color in palette if color != int(clean[r, c])]
                inp[r, c] = random.choice(choices)
            used_orbits.update(normalized)
            patch_orbits.append(normalized)
            used_blocks.append((block_r, block_c))
            phases.add(phase)
            break

    if len(patch_orbits) != patch_count or phases != {0, 1}:
        return None
    return inp, patch_orbits


def _plain_period_area(grid: np.ndarray) -> int:
    """Area of the smallest exact top-left aligned ordinary repeating tile."""
    height, width = map(int, grid.shape)
    areas = []
    for h in range(1, height + 1):
        if height % h:
            continue
        for w in range(1, width + 1):
            if width % w:
                continue
            candidate = np.tile(grid[:h, :w], (height // h, width // w))
            if np.array_equal(candidate, grid):
                areas.append(h * w)
    return min(areas)


def _gen_attempt(direction: str) -> np.ndarray | None:
    tile_h = random.randint(3, 6)
    tile_w = random.randint(3, 6)
    tile_rows = random.randint(3, min(5, 30 // tile_h))
    tile_cols = random.randint(3, min(5, 30 // tile_w))

    tile = _random_tile(tile_h, tile_w)
    if tile is None:
        return None
    clean = _render_tiling(tile, tile_rows, tile_cols, direction)
    patched = _patch_grid(
        clean, tile_h, tile_w, tile_rows, tile_cols, direction
    )
    if patched is None:
        return None
    inp, patch_orbits = patched

    changed = inp != clean
    if int(changed.sum()) < 4 or len(patch_orbits) < 2:
        return None
    if not set(map(int, inp.flat)) <= set(map(int, clean.flat)):
        return None
    if any(len(a & b) for i, a in enumerate(patch_orbits) for b in patch_orbits[i + 1 :]):
        return None
    changed_coords = np.argwhere(changed)
    changed_orbits = {
        _canonical_coord(int(r), int(c), tile_h, tile_w, direction)
        for r, c in changed_coords
    }
    if len(changed_orbits) != int(changed.sum()):
        return None

    # The reflected step must visibly matter at the stated tile scale, and the
    # fundamental mirror tile must be strictly smaller than any plain period.
    plain_same_size = np.tile(tile, (tile_rows, tile_cols))
    if int(np.sum(plain_same_size != clean)) <= int(changed.sum()):
        return None
    if _plain_period_area(clean) <= tile_h * tile_w:
        return None

    try:
        inferred = _infer_structure(inp)
        actual = main(inp)
    except (TypeError, ValueError):
        return None
    if (inferred.tile_h, inferred.tile_w, inferred.direction) != (
        tile_h,
        tile_w,
        direction,
    ):
        return None
    if inferred.errors != int(changed.sum()) or not np.array_equal(actual, clean):
        return None
    if np.array_equal(inp, clean):
        return None
    return inp


_GENERATION_INDEX = random.randrange(2)
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def generate_input() -> np.ndarray:
    """Generate a varied, oracle-checked instance; consecutive calls cover both axes."""
    global _GENERATION_INDEX
    direction = "rows" if _GENERATION_INDEX % 2 == 0 else "cols"
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(direction)
        if grid is not None:
            output = main(grid)
            signature = (tuple(map(int, output.shape)), output.tobytes())
            if signature in _SEEN_OUTPUTS:
                continue
            _SEEN_OUTPUTS.add(signature)
            _GENERATION_INDEX += 1
            return grid
    raise RuntimeError("failed to generate a unique mirror-alternating repair instance")
