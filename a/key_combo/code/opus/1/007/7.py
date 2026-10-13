from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
WALL = 2
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SPRITE_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
PARTITIONS = {
    2: [(2,), (1, 1)],
    3: [(3,), (2, 1), (1, 1, 1)],
    4: [(4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1)],
}
BOUNDARY_SIDES = ("top", "bottom", "left", "right")
GEN_ATTEMPTS = 500
GEN_RNG = random.Random()
GEN_CALL_INDEX = 0
GEN_SCHEDULE = (
    (2, "compound"),
    (3, "boundary"),
    (4, "distractor"),
    (2, "boundary"),
    (3, "compound"),
    (4, "distractor"),
)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            components.append(coords)

    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _exterior_black(grid: np.ndarray) -> np.ndarray:
    black = grid == BACKGROUND
    h, w = grid.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if black[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if black[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if _in_bounds(grid.shape, nr, nc) and black[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    return exterior


def _count_enclosed_regions(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    exterior = _exterior_black(grid)
    enclosed = (grid == BACKGROUND) & ~exterior
    return len(_mask_components(enclosed)), enclosed, exterior


def _extract_sprite(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], list[tuple[int, int]]]:
    sprite_mask = (grid != BACKGROUND) & (grid != WALL)
    components = _mask_components(sprite_mask)
    assert len(components) == 1, "expected exactly one non-red, non-black object"

    coords = components[0]
    r0, c0, r1, c1 = _bbox(coords)
    crop = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    h, w = crop.shape
    assert 3 <= h <= 4 and 3 <= w <= 4, "sprite bbox must be 3x3 or 4x4"
    assert not np.any(crop == WALL), "sprite bbox must not contain wall pixels"
    return crop, (r0, c0, r1, c1), coords


def _scale_patch(patch: np.ndarray, factor: int) -> np.ndarray:
    assert factor >= 1
    return np.repeat(np.repeat(np.asarray(patch, dtype=int), factor, axis=0), factor, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count black regions enclosed by red walls, then crop and magnify the unique sprite by that count.

    Input structure:
    - A black background contains red one-cell-thick wall pixels that may form closed rooms,
      open barriers, or chambers that use the grid border as part of their outline.
    - Exactly one small multicolor sprite uses non-black, non-red colors inside a 3x3 or 4x4
      tight bounding box, possibly with some black cells still present inside that box.
    - The sprite is separate from the wall system and sits in an open area rather than inside any
      red-enclosed black region.

    Rule:
    - Flood-fill 4-neighbor black cells from the grid border while treating red pixels as the walls.
    - Count the 4-connected black regions that remain unreachable from the border; call this count N.
    - Find the unique connected non-red, non-black object, and crop the grid to its tight bounding box.
    - Magnify that cropped sprite by a factor of N in both directions, including any black cells inside
      the sprite box, and output only the enlarged crop.

    Key insight:
    - The output scale is hidden in the wall topology, so solving requires combining a hole-counting
      flood fill with a separate sprite-extraction step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    n_regions, _enclosed, _exterior = _count_enclosed_regions(g)
    assert n_regions >= 1, "expected at least one enclosed black region"

    sprite, _bbox_coords, _sprite_coords = _extract_sprite(g)
    return _scale_patch(sprite, n_regions)


def _inflate_bbox(
    bbox: tuple[int, int, int, int], shape: tuple[int, int], pad: int = 1
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(shape[0] - 1, r1 + pad),
        min(shape[1] - 1, c1 + pad),
    )


def _bbox_is_free(reserve: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    rr0, cc0, rr1, cc1 = _inflate_bbox(bbox, reserve.shape, pad=pad)
    return not bool(reserve[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _reserve_bbox(reserve: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    rr0, cc0, rr1, cc1 = _inflate_bbox(bbox, reserve.shape, pad=pad)
    reserve[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _make_block_shape(
    size: int, orientation: str, chamber_minor: int, chamber_sizes: list[int]
) -> tuple[list[tuple[int, int]], int, int]:
    assert size == len(chamber_sizes)

    cells: set[tuple[int, int]] = set()
    if orientation == "h":
        block_h = chamber_minor + 2
        block_w = sum(chamber_sizes) + size + 1
        for c in range(block_w):
            cells.add((0, c))
            cells.add((block_h - 1, c))
        divider = 0
        cells.update((r, divider) for r in range(block_h))
        for width in chamber_sizes:
            divider += width + 1
            cells.update((r, divider) for r in range(block_h))
    else:
        block_h = sum(chamber_sizes) + size + 1
        block_w = chamber_minor + 2
        for r in range(block_h):
            cells.add((r, 0))
            cells.add((r, block_w - 1))
        divider = 0
        cells.update((divider, c) for c in range(block_w))
        for height in chamber_sizes:
            divider += height + 1
            cells.update((divider, c) for c in range(block_w))

    return sorted(cells), block_h, block_w


def _candidate_positions(
    shape: tuple[int, int], block_h: int, block_w: int, attach: str | None
) -> list[tuple[int, int]]:
    h, w = shape
    positions: list[tuple[int, int]] = []

    if block_h > h or block_w > w:
        return positions

    if attach is None:
        if h - block_h < 2 or w - block_w < 2:
            return positions
        for r0 in range(1, h - block_h):
            for c0 in range(1, w - block_w):
                positions.append((r0, c0))
        return positions

    if attach == "top":
        r0 = 0
        for c0 in range(0, w - block_w + 1):
            positions.append((r0, c0))
    elif attach == "bottom":
        r0 = h - block_h
        for c0 in range(0, w - block_w + 1):
            positions.append((r0, c0))
    elif attach == "left":
        c0 = 0
        for r0 in range(0, h - block_h + 1):
            positions.append((r0, c0))
    elif attach == "right":
        c0 = w - block_w
        for r0 in range(0, h - block_h + 1):
            positions.append((r0, c0))
    else:
        raise AssertionError(f"unknown attachment side: {attach}")

    return positions


def _draw_cells(grid: np.ndarray, cells: list[tuple[int, int]], r0: int, c0: int) -> None:
    for dr, dc in cells:
        grid[r0 + dr, c0 + dc] = WALL


def _place_block(
    grid: np.ndarray,
    reserve: np.ndarray,
    rng: random.Random,
    size: int,
    force_attach: bool | None = None,
) -> dict[str, object] | None:
    attachments: list[str | None]
    if force_attach is True:
        attachments = list(BOUNDARY_SIDES)
    elif force_attach is False:
        attachments = [None]
    else:
        attachments = [None] + list(BOUNDARY_SIDES)

    options: list[tuple[str, int, list[int], str | None]] = []
    for orientation in ("h", "v"):
        minor = rng.randint(2, 4)
        sizes = [rng.randint(2, 4) for _ in range(size)]
        for attach in attachments:
            options.append((orientation, minor, sizes, attach))
    rng.shuffle(options)

    for orientation, minor, sizes, attach in options:
        cells, block_h, block_w = _make_block_shape(size, orientation, minor, sizes)
        positions = _candidate_positions(grid.shape, block_h, block_w, attach)
        rng.shuffle(positions)
        for r0, c0 in positions:
            bbox = (r0, c0, r0 + block_h - 1, c0 + block_w - 1)
            if not _bbox_is_free(reserve, bbox, pad=1):
                continue
            _draw_cells(grid, cells, r0, c0)
            _reserve_bbox(reserve, bbox, pad=1)
            return {
                "size": size,
                "orientation": orientation,
                "attach": attach,
                "bbox": bbox,
            }

    return None


def _make_segment(length: int, horizontal: bool) -> tuple[list[tuple[int, int]], int, int]:
    if horizontal:
        return [(0, c) for c in range(length)], 1, length
    return [(r, 0) for r in range(length)], length, 1


def _make_l_shape(horizontal_len: int, vertical_len: int, variant: int) -> tuple[list[tuple[int, int]], int, int]:
    cells = {(0, c) for c in range(horizontal_len)}
    cells.update((r, 0) for r in range(vertical_len))
    block_h, block_w = vertical_len, horizontal_len

    transformed = set(cells)
    if variant & 1:
        transformed = {(r, block_w - 1 - c) for r, c in transformed}
    if variant & 2:
        transformed = {(block_h - 1 - r, c) for r, c in transformed}
    if variant & 4:
        transformed = {(c, r) for r, c in transformed}
        block_h, block_w = block_w, block_h

    return sorted(transformed), block_h, block_w


def _place_distractor(grid: np.ndarray, reserve: np.ndarray, rng: random.Random) -> dict[str, object] | None:
    choices: list[tuple[list[tuple[int, int]], int, int]] = []
    for _ in range(4):
        length = rng.randint(3, 7)
        choices.append(_make_segment(length, horizontal=bool(rng.randint(0, 1))))
    for _ in range(4):
        choices.append(_make_l_shape(rng.randint(3, 6), rng.randint(3, 6), rng.randint(0, 7)))

    rng.shuffle(choices)
    for cells, block_h, block_w in choices:
        if grid.shape[0] - block_h < 2 or grid.shape[1] - block_w < 2:
            continue
        positions = [
            (r0, c0)
            for r0 in range(1, grid.shape[0] - block_h)
            for c0 in range(1, grid.shape[1] - block_w)
        ]
        rng.shuffle(positions)
        for r0, c0 in positions:
            bbox = (r0, c0, r0 + block_h - 1, c0 + block_w - 1)
            if not _bbox_is_free(reserve, bbox, pad=1):
                continue
            _draw_cells(grid, cells, r0, c0)
            _reserve_bbox(reserve, bbox, pad=1)
            return {"bbox": bbox}
    return None


def _zero_components_touch_border(mask: np.ndarray) -> bool:
    zero_mask = ~mask
    for comp in _mask_components(zero_mask):
        if not any(r in (0, mask.shape[0] - 1) or c in (0, mask.shape[1] - 1) for r, c in comp):
            return False
    return True


def _generate_sprite_patch(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        side = rng.choice((3, 4))
        full_count = side * side
        max_count = full_count if rng.random() < 0.3 else full_count - 1
        target = rng.randint(side + 1, max_count)

        start = (rng.randrange(side), rng.randrange(side))
        cells = {start}
        while len(cells) < target:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < side and 0 <= nc < side and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(sorted(frontier)))

        if len(cells) != target:
            continue

        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if rows != set(range(side)) or cols != set(range(side)):
            continue

        mask = np.zeros((side, side), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not _zero_components_touch_border(mask):
            continue

        color_count = rng.randint(2, min(4, len(cells)))
        palette = rng.sample(SPRITE_COLORS, color_count)
        patch = np.zeros((side, side), dtype=int)
        assigned = []
        for r, c in sorted(cells):
            color = rng.choice(palette)
            patch[r, c] = color
            assigned.append(color)
        if len(set(assigned)) < 2:
            continue
        return patch

    raise RuntimeError("failed to generate a valid sprite patch")


def _bbox_margin_has_no_walls(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    rr0, cc0, rr1, cc1 = _inflate_bbox(bbox, grid.shape, pad=1)
    return not bool((grid[rr0 : rr1 + 1, cc0 : cc1 + 1] == WALL).any())


def _place_sprite(
    wall_grid: np.ndarray, patch: np.ndarray, expected_n: int, rng: random.Random
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    h, w = wall_grid.shape
    ph, pw = patch.shape
    _n0, _enclosed0, exterior = _count_enclosed_regions(wall_grid)

    candidates = [
        (r0, c0)
        for r0 in range(1, h - ph)
        for c0 in range(1, w - pw)
        if exterior[r0 : r0 + ph, c0 : c0 + pw].all()
        and np.all(wall_grid[r0 : r0 + ph, c0 : c0 + pw] == BACKGROUND)
        and _bbox_margin_has_no_walls(wall_grid, (r0, c0, r0 + ph - 1, c0 + pw - 1))
    ]
    rng.shuffle(candidates)

    for r0, c0 in candidates:
        grid = wall_grid.copy()
        color_mask = patch != BACKGROUND
        grid[r0 : r0 + ph, c0 : c0 + pw][color_mask] = patch[color_mask]

        n_regions, _enclosed, _exterior = _count_enclosed_regions(grid)
        if n_regions != expected_n:
            continue

        try:
            sprite, bbox, _coords = _extract_sprite(grid)
        except AssertionError:
            continue
        if not np.array_equal(sprite, patch):
            continue
        return grid, bbox

    return None


def _choose_partition(rng: random.Random, n_regions: int, mode: str) -> tuple[int, ...]:
    options = list(PARTITIONS[n_regions])
    if mode == "compound":
        options = [part for part in options if max(part) > 1]
    return rng.choice(options)


def _validate_instance(
    grid: np.ndarray,
    expected_n: int,
    expected_patch: np.ndarray,
    mode: str,
    block_count: int,
    distractor_count: int,
) -> bool:
    try:
        n_regions, _enclosed, _exterior = _count_enclosed_regions(grid)
        if n_regions != expected_n:
            return False

        sprite, bbox, coords = _extract_sprite(grid)
        if not np.array_equal(sprite, expected_patch):
            return False
        if not _bbox_margin_has_no_walls(grid, bbox):
            return False
        for r, c in coords:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == WALL:
                    return False

        red_components = len(_mask_components(grid == WALL))
        touches_boundary = bool((grid[0] == WALL).any() or (grid[-1] == WALL).any() or (grid[:, 0] == WALL).any() or (grid[:, -1] == WALL).any())
        if mode == "compound" and red_components >= expected_n:
            return False
        if mode == "boundary" and not touches_boundary:
            return False
        if mode == "distractor" and (distractor_count < 1 or red_components <= block_count):
            return False

        out = main(grid)
        expected_out = _scale_patch(expected_patch, expected_n)
        if not np.array_equal(out, expected_out):
            return False
        return True
    except AssertionError:
        return False


def _gen_attempt(
    rng: random.Random,
    forced_n: int | None = None,
    forced_mode: str | None = None,
) -> np.ndarray | None:
    height = rng.randint(20, 26)
    width = rng.randint(20, 26)
    expected_n = forced_n if forced_n is not None else rng.choice((2, 3, 4))
    mode = forced_mode if forced_mode is not None else rng.choice(("compound", "boundary", "distractor"))

    grid = np.full((height, width), BACKGROUND, dtype=int)
    reserve = np.zeros((height, width), dtype=bool)

    partition = list(_choose_partition(rng, expected_n, mode))
    partition.sort(reverse=True)

    boundary_index = rng.randrange(len(partition)) if mode == "boundary" else None
    blocks: list[dict[str, object]] = []
    for index, size in enumerate(partition):
        force_attach = True if boundary_index == index else None
        block = _place_block(grid, reserve, rng, size=size, force_attach=force_attach)
        if block is None:
            return None
        blocks.append(block)

    distractor_target = 0
    if mode == "distractor":
        distractor_target = 1
    distractor_target += 1 if rng.random() < 0.55 else 0
    distractors = []
    for _ in range(distractor_target):
        distractor = _place_distractor(grid, reserve, rng)
        if distractor is None:
            if mode == "distractor" and not distractors:
                return None
            break
        distractors.append(distractor)

    n_before_sprite, _enclosed, _exterior = _count_enclosed_regions(grid)
    if n_before_sprite != expected_n:
        return None

    patch = _generate_sprite_patch(rng)
    placed = _place_sprite(grid, patch, expected_n, rng)
    if placed is None:
        return None
    final_grid, _sprite_bbox = placed

    if not _validate_instance(
        final_grid,
        expected_n,
        patch,
        mode,
        block_count=len(blocks),
        distractor_count=len(distractors),
    ):
        return None
    return final_grid


def generate_input() -> np.ndarray:
    global GEN_CALL_INDEX

    rng = GEN_RNG
    forced_n, forced_mode = GEN_SCHEDULE[GEN_CALL_INDEX % len(GEN_SCHEDULE)]
    GEN_CALL_INDEX += 1
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng, forced_n=forced_n, forced_mode=forced_mode)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle input after {GEN_ATTEMPTS} attempts")
