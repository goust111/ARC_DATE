from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


ROTATIONS = (1, 2, 3)
MAX_GENERATION_ATTEMPTS = 500
_generation_counter = 0


def _background(grid: np.ndarray) -> int:
    """Return the unique most frequent color."""
    counts = Counter(map(int, grid.ravel()))
    ordered = counts.most_common()
    assert len(ordered) == 1 or ordered[0][1] > ordered[1][1], "background must be uniquely dominant"
    return int(ordered[0][0])


def _foreground_components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 8-connected foreground components as (N, 2) arrays of (r, c)."""
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[np.ndarray] = []
    for raw_r, raw_c in np.argwhere(mask):
        r, c = int(raw_r), int(raw_c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue = deque([(r, c)])
        coords: list[tuple[int, int]] = []
        while queue:
            cr, cc = queue.popleft()
            coords.append((cr, cc))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == dc == 0:
                        continue
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
        components.append(np.asarray(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rotation_candidates(blocks: np.ndarray, miniature: np.ndarray, bg: int) -> list[int]:
    candidates = []
    for quarter_turns in ROTATIONS:
        rotated = np.rot90(blocks, quarter_turns)
        overlap = (rotated != bg) & (miniature != bg)
        if overlap.any() and np.array_equal(rotated[overlap], miniature[overlap]):
            candidates.append(quarter_turns)
    return candidates


def _parse(grid: np.ndarray):
    """Locate both representations and recover their block-level arrays."""
    bg = _background(grid)
    components = _foreground_components(grid != bg)
    assert len(components) == 2, "expected exactly two separated foreground regions"

    regions = []
    for coords in components:
        r0, c0, r1, c1 = _bbox(coords)
        height, width = r1 - r0 + 1, c1 - c0 + 1
        assert height == width, "both regions must be square"
        regions.append((height, r0, c0, grid[r0 : r1 + 1, c0 : c1 + 1]))
    regions.sort(key=lambda item: item[0])

    n, small_r, small_c, miniature = regions[0]
    large_side, large_r, large_c, large = regions[1]
    assert 2 <= n and large_side > n and large_side % n == 0
    block_size = large_side // n
    assert block_size >= 2

    small_holes = np.argwhere(miniature == bg)
    assert small_holes.shape == (1, 2), "miniature must have exactly one missing pixel"

    blocks = np.full((n, n), bg, dtype=int)
    large_holes: list[tuple[int, int]] = []
    for br in range(n):
        for bc in range(n):
            patch = large[
                br * block_size : (br + 1) * block_size,
                bc * block_size : (bc + 1) * block_size,
            ]
            values = set(map(int, np.unique(patch)))
            if values == {bg}:
                large_holes.append((br, bc))
            else:
                assert bg not in values and len(values) == 1, "visible large blocks must be solid"
                blocks[br, bc] = values.pop()
    assert len(large_holes) == 1, "large mosaic must have exactly one missing block"

    candidates = _rotation_candidates(blocks, miniature, bg)
    assert len(candidates) == 1, "visible evidence must determine one non-zero rotation"
    quarter_turns = candidates[0]
    large_hole = large_holes[0]
    small_hole = tuple(map(int, small_holes[0]))

    rotated_ids = np.rot90(np.arange(n * n).reshape(n, n), quarter_turns)
    projected = tuple(map(int, np.argwhere(rotated_ids == large_hole[0] * n + large_hole[1])[0]))
    assert projected != small_hole, "the two omissions must not correspond to one another"

    return {
        "bg": bg,
        "n": n,
        "block_size": block_size,
        "large_origin": (large_r, large_c),
        "small_origin": (small_r, small_c),
        "blocks": blocks,
        "miniature": miniature.copy(),
        "large_hole": large_hole,
        "small_hole": small_hole,
        "rotation": quarter_turns,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair one missing large block and one missing miniature pixel after inferring their hidden rotation.

    Input structure:
    - A dominant background separates two square regions: an n×n mosaic of solid k×k
      color blocks and an n×n one-pixel-per-block miniature.
    - Exactly one whole block is background in the mosaic and exactly one pixel is
      background in the miniature; those omissions refer to different source cells.
    - The miniature is the block-level mosaic rotated by 90, 180, or 270 degrees.

    Rule:
    - Compare only mutually visible block/pixel pairs under the three allowed rotations
      and select the unique matching rotation.
    - Through that correspondence, copy the visible miniature color into every pixel of
      the missing large block, and copy the visible large-block color into the missing
      miniature pixel. Preserve every other grid cell.

    Key insight:
    - Neither scale is complete: their non-corresponding holes make the two rotated
      representations complementary, so the rotation must be inferred before they can
      repair one another in opposite directions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    info = _parse(g)

    bg = info["bg"]
    blocks = info["blocks"]
    miniature = info["miniature"]
    quarter_turns = info["rotation"]
    br, bc = info["large_hole"]
    sr, sc = info["small_hole"]

    miniature_in_large_frame = np.rot90(miniature, -quarter_turns)
    large_fill = int(miniature_in_large_frame[br, bc])
    small_fill = int(np.rot90(blocks, quarter_turns)[sr, sc])
    assert large_fill != bg and small_fill != bg, "each missing cell must be visible at the other scale"

    out = g.copy()
    large_r, large_c = info["large_origin"]
    block_size = info["block_size"]
    out[
        large_r + br * block_size : large_r + (br + 1) * block_size,
        large_c + bc * block_size : large_c + (bc + 1) * block_size,
    ] = large_fill
    small_r, small_c = info["small_origin"]
    out[small_r + sr, small_c + sc] = small_fill
    return out


def _projected_coordinate(n: int, coord: tuple[int, int], quarter_turns: int) -> tuple[int, int]:
    ids = np.arange(n * n).reshape(n, n)
    target = coord[0] * n + coord[1]
    return tuple(map(int, np.argwhere(np.rot90(ids, quarter_turns) == target)[0]))


def _make_pattern(
    n: int,
    bg: int,
    palette: list[int],
    quarter_turns: int,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]] | None:
    """Create a pattern whose damaged views uniquely expose the intended rotation."""
    base = np.asarray(random.choices(palette, k=n * n), dtype=int).reshape(n, n)
    if len(set(map(int, base.ravel()))) < min(4, len(palette)):
        return None

    large_hole = (random.randrange(n), random.randrange(n))
    projected_hole = _projected_coordinate(n, large_hole, quarter_turns)
    choices = [
        (r, c)
        for r in range(n)
        for c in range(n)
        if (r, c) not in {projected_hole, large_hole}
    ]
    if not choices:
        return None
    small_hole = random.choice(choices)
    full_miniature = np.rot90(base, quarter_turns)

    # Direct same-coordinate copying must visibly fail in both directions.
    if int(full_miniature[large_hole]) == int(base[large_hole]):
        return None
    if int(full_miniature[small_hole]) == int(base[small_hole]):
        return None
    if int(base[large_hole]) == int(full_miniature[small_hole]):
        return None

    damaged_blocks = base.copy()
    damaged_miniature = full_miniature.copy()
    damaged_blocks[large_hole] = bg
    damaged_miniature[small_hole] = bg

    candidates = _rotation_candidates(damaged_blocks, damaged_miniature, bg)
    if candidates != [quarter_turns]:
        return None

    # Each disallowed rotation, including 0 degrees, must have multiple visible conflicts.
    for wrong in (0, 1, 2, 3):
        if wrong == quarter_turns:
            continue
        rotated = np.rot90(damaged_blocks, wrong)
        overlap = (rotated != bg) & (damaged_miniature != bg)
        if int(np.count_nonzero(rotated[overlap] != damaged_miniature[overlap])) < 2:
            return None
    return base, large_hole, small_hole


def _paint_mosaic(canvas: np.ndarray, pattern: np.ndarray, r0: int, c0: int, block_size: int) -> None:
    for br in range(pattern.shape[0]):
        for bc in range(pattern.shape[1]):
            canvas[
                r0 + br * block_size : r0 + (br + 1) * block_size,
                c0 + bc * block_size : c0 + (bc + 1) * block_size,
            ] = int(pattern[br, bc])


def _place_regions(large_side: int, n: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Choose a random horizontal or vertical layout with a clear blank gap."""
    gap = random.randint(2, 4)
    lead, trail = random.randint(1, 3), random.randint(1, 3)
    large_first = random.choice((False, True))

    if random.choice((False, True)):
        height = max(large_side, n) + random.randint(2, 6)
        width = lead + large_side + gap + n + trail
        first_side, second_side = (large_side, n) if large_first else (n, large_side)
        first_origin = (random.randint(1, height - first_side - 1), lead)
        second_origin = (random.randint(1, height - second_side - 1), lead + first_side + gap)
    else:
        height = lead + large_side + gap + n + trail
        width = max(large_side, n) + random.randint(2, 6)
        first_side, second_side = (large_side, n) if large_first else (n, large_side)
        first_origin = (lead, random.randint(1, width - first_side - 1))
        second_origin = (lead + first_side + gap, random.randint(1, width - second_side - 1))

    assert height <= 30 and width <= 30
    large_origin, small_origin = (first_origin, second_origin) if large_first else (second_origin, first_origin)
    return (height, width), large_origin, small_origin


def _generate_attempt(quarter_turns: int) -> np.ndarray | None:
    n = random.choice((3, 4, 5))
    block_size = random.choice((2, 3))
    large_side = n * block_size
    bg = random.randrange(10)
    available = [color for color in range(10) if color != bg]
    palette = random.sample(available, random.randint(4, min(7, len(available))))

    made = _make_pattern(n, bg, palette, quarter_turns)
    if made is None:
        return None
    base, large_hole, small_hole = made
    full_miniature = np.rot90(base, quarter_turns)

    shape, large_origin, small_origin = _place_regions(large_side, n)
    inp = np.full(shape, bg, dtype=int)
    expected = np.full(shape, bg, dtype=int)
    _paint_mosaic(inp, base, *large_origin, block_size)
    _paint_mosaic(expected, base, *large_origin, block_size)
    sr0, sc0 = small_origin
    inp[sr0 : sr0 + n, sc0 : sc0 + n] = full_miniature
    expected[sr0 : sr0 + n, sc0 : sc0 + n] = full_miniature

    br, bc = large_hole
    lr0, lc0 = large_origin
    inp[
        lr0 + br * block_size : lr0 + (br + 1) * block_size,
        lc0 + bc * block_size : lc0 + (bc + 1) * block_size,
    ] = bg
    inp[sr0 + small_hole[0], sc0 + small_hole[1]] = bg

    if _background(inp) != bg:
        return None
    try:
        info = _parse(inp)
        out = main(inp)
    except AssertionError:
        return None
    if info["rotation"] != quarter_turns or not np.array_equal(out, expected):
        return None

    changed = out != inp
    if int(changed.sum()) != block_size * block_size + 1:
        return None
    if len(set(map(int, out[changed]))) != 2:
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate diverse, oracle-checked examples while cycling all three rotations."""
    global _generation_counter
    quarter_turns = ROTATIONS[_generation_counter % len(ROTATIONS)]
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(quarter_turns)
        if grid is not None:
            _generation_counter += 1
            return grid
    raise RuntimeError("failed to generate a unique, non-degenerate cross-scale repair puzzle")
