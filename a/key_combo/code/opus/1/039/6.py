from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
FILL = 3
WALL = 5

MAX_GEN_ATTEMPTS = 500
HEIGHT_CHOICES = (12, 14, 16, 18, 20)
WIDTH_MIN = 13
WIDTH_MAX = 30
ROOM_GAP_BASE = 2
OUTER_MARGIN_BASE = 1


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = map(int, mask.shape)
    seen = np.zeros((height, width), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _enclosed_background_regions(wall_mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = map(int, wall_mask.shape)
    bg = ~wall_mask
    exterior = np.zeros((height, width), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(height):
        for c in (0, width - 1):
            if bool(bg[r, c]) and not bool(exterior[r, c]):
                exterior[r, c] = True
                q.append((r, c))
    for c in range(width):
        for r in (0, height - 1):
            if bool(bg[r, c]) and not bool(exterior[r, c]):
                exterior[r, c] = True
                q.append((r, c))

    while q:
        cr, cc = q.popleft()
        for nr, nc in _neighbors4(cr, cc):
            if 0 <= nr < height and 0 <= nc < width and bool(bg[nr, nc]) and not bool(exterior[nr, nc]):
                exterior[nr, nc] = True
                q.append((nr, nc))

    enclosed = bg & ~exterior
    return _mask_components(enclosed)


def _merged_wall_mask(grid: np.ndarray) -> np.ndarray:
    wall = np.asarray(grid, dtype=int) == WALL
    return wall | np.flipud(wall)


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    wall = _merged_wall_mask(grid)
    out = np.zeros_like(np.asarray(grid, dtype=int))
    for region in _enclosed_background_regions(wall):
        for r, c in region:
            out[r, c] = FILL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror-merge the gray wall fragments, flood-fill the enclosed voids, and erase the walls.

    Input structure:
    - The grid is mostly black background.
    - Gray wall fragments appear in the upper and lower halves as several separated
      4-connected line segments, and green is absent from the input.
    - In the input alone, every black background cell is still 4-connected to the
      outer border, so no enclosed background region exists yet.

    Rule:
    - Reflect the whole grid across the horizontal midline and take the logical-or
      of the original and reflected gray-wall masks.
    - In that merged wall layout, find all 4-connected black background regions.
    - Mark every background region that touches the grid border as exterior, then
      flood-fill every remaining enclosed region with green.
    - Erase all gray wall pixels to black, leaving only the green filled regions.

    Key insight:
    - The cavities do not exist in the input itself; they emerge only after the
      original fragments are combined with their vertical mirror image.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    return _oracle_output(g)


def _room_layout() -> list[int]:
    if random.random() < 0.62:
        return [random.choices([1, 2, 3, 4], weights=[1, 3, 4, 3], k=1)[0]]
    return random.choice(([1, 1], [1, 2], [2, 1], [1, 3], [3, 1], [2, 2]))


def _sample_room_widths(region_count: int) -> list[int]:
    if region_count >= 3:
        return [random.randint(3, 5) for _ in range(region_count)]
    widths = [random.randint(3, 5) for _ in range(region_count)]
    if random.random() < 0.35:
        widths[random.randrange(region_count)] = random.randint(6, 8)
    return widths


def _sample_vertical_len(height: int) -> int:
    max_len = min(8, height // 2 - 1)
    choices = list(range(3, max_len + 1))
    weights = [1 + min(v - 3, 3) for v in choices]
    return random.choices(choices, weights=weights, k=1)[0]


def _distribute_extra(total: int, slots: int) -> list[int]:
    parts = [0] * slots
    for _ in range(total):
        parts[random.randrange(slots)] += 1
    return parts


def _segment_cells_horizontal(row: int, c0: int, c1: int) -> list[tuple[int, int]]:
    return [(row, c) for c in range(c0, c1 + 1)]


def _segment_cells_vertical(col: int, r0: int, r1: int) -> list[tuple[int, int]]:
    return [(r, col) for r in range(r0, r1 + 1)]


def _mirror_cells(height: int, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(height - 1 - r, c) for r, c in cells]


def _place_room(
    grid: np.ndarray,
    merged_wall: np.ndarray,
    left: int,
    widths: list[int],
    vertical_len: int,
    verticals_on_top: bool,
) -> None:
    height, _width = map(int, grid.shape)
    top_row = height // 2 - vertical_len
    bridge_row = height // 2 - 1
    x_positions = [left]
    for width in widths:
        x_positions.append(x_positions[-1] + width + 1)

    for idx, x in enumerate(x_positions):
        vertical = _segment_cells_vertical(x, top_row, bridge_row)
        actual_vertical = vertical if verticals_on_top else _mirror_cells(height, vertical)
        for r, c in actual_vertical:
            grid[r, c] = WALL
        for r, c in vertical:
            merged_wall[r, c] = True
            merged_wall[height - 1 - r, c] = True

        if idx < len(widths):
            horizontal = _segment_cells_horizontal(top_row, x + 1, x_positions[idx + 1] - 1)
            actual_horizontal = _mirror_cells(height, horizontal) if verticals_on_top else horizontal
            for r, c in actual_horizontal:
                grid[r, c] = WALL
            for r, c in horizontal:
                merged_wall[r, c] = True
                merged_wall[height - 1 - r, c] = True


def _touches_existing(mask: np.ndarray, cells: list[tuple[int, int]], pad: int = 0) -> bool:
    height, width = map(int, mask.shape)
    for r, c in cells:
        for nr in range(r - pad, r + pad + 1):
            for nc in range(c - pad, c + pad + 1):
                if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]):
                    return True
    return False


def _add_distractors(grid: np.ndarray, merged_wall: np.ndarray) -> None:
    height, width = map(int, grid.shape)
    count = random.choices([0, 1, 2], weights=[2, 3, 2], k=1)[0]
    attempts = 0
    added = 0
    while added < count and attempts < 80:
        attempts += 1
        length = random.randint(3, min(8, max(height, width)))
        horizontal = bool(random.getrandbits(1))
        on_top = bool(random.getrandbits(1))
        if horizontal:
            if width - length < 2:
                continue
            row_lo = 1
            row_hi = height // 2 - 2
            if row_lo > row_hi:
                continue
            row = random.randint(row_lo, row_hi)
            col = random.randint(1, width - length - 1)
            base = _segment_cells_horizontal(row, col, col + length - 1)
        else:
            row_lo = 1
            row_hi = height // 2 - length
            if row_lo > row_hi:
                continue
            row = random.randint(row_lo, row_hi)
            col = random.randint(1, width - 2)
            base = _segment_cells_vertical(col, row, row + length - 1)

        actual = base if on_top else _mirror_cells(height, base)
        mirrored_union = base + _mirror_cells(height, base)
        if _touches_existing(grid == WALL, actual, pad=1):
            continue
        if _touches_existing(merged_wall, mirrored_union, pad=1):
            continue

        for r, c in actual:
            grid[r, c] = WALL
        for r, c in mirrored_union:
            merged_wall[r, c] = True
        added += 1


def _component_is_straight(comp: list[tuple[int, int]]) -> bool:
    rows = {r for r, _ in comp}
    cols = {c for _, c in comp}
    return len(rows) == 1 or len(cols) == 1


def _validate_input(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "generator must return a 2D grid"
    assert max(g.shape) <= 30, "grid exceeds ARC size limit"
    assert FILL not in set(map(int, np.unique(g))), "green must be absent from input"
    assert set(map(int, np.unique(g))) <= {BACKGROUND, WALL}, "generator uses only black and gray"

    wall = g == WALL
    comps = _mask_components(wall)
    assert comps, "expected at least one gray fragment"
    assert any(r < g.shape[0] // 2 for comp in comps for r, _ in comp), "need upper-half fragments"
    assert any(r >= g.shape[0] // 2 for comp in comps for r, _ in comp), "need lower-half fragments"
    for comp in comps:
        assert 3 <= len(comp) <= 8, f"fragment length must be 3..8, got {len(comp)}"
        assert _component_is_straight(comp), "generator should emit only line-segment fragments"

    assert not _enclosed_background_regions(wall), "input itself must contain no enclosed background region"

    merged = _merged_wall_mask(g)
    regions = _enclosed_background_regions(merged)
    assert 1 <= len(regions) <= 4, f"merged wall must create 1..4 enclosed regions, got {len(regions)}"
    assert all(len(region) >= 4 for region in regions), "every enclosed region must have area at least 4"

    out = _oracle_output(g)
    assert np.any(out == FILL), "output must contain at least one filled cell"
    assert not np.any(out == WALL), "output must erase the walls"

    input_like_output = np.zeros_like(g)
    for region in _enclosed_background_regions(wall):
        for r, c in region:
            input_like_output[r, c] = FILL
    assert not np.array_equal(out, input_like_output), "generator must require the mirror-merge step"

    assert np.array_equal(main(g), out), "main() must agree with the independent oracle"


def _attempt_generate() -> np.ndarray | None:
    height = random.choice(HEIGHT_CHOICES)
    room_region_counts = _room_layout()

    room_specs = []
    for region_count in room_region_counts:
        widths = _sample_room_widths(region_count)
        room_specs.append(
            {
                "regions": region_count,
                "widths": widths,
                "vertical_len": _sample_vertical_len(height),
                "outer_width": sum(widths) + region_count + 1,
                "verticals_on_top": bool(random.getrandbits(1)),
            }
        )

    required_width = 2 * OUTER_MARGIN_BASE + ROOM_GAP_BASE * (len(room_specs) - 1)
    required_width += sum(spec["outer_width"] for spec in room_specs)
    if required_width > WIDTH_MAX:
        return None

    width = max(WIDTH_MIN, required_width)
    width += random.randint(0, min(5, WIDTH_MAX - width))
    slot_count = len(room_specs) + 1
    slots = [OUTER_MARGIN_BASE] * slot_count
    for idx in range(1, slot_count - 1):
        slots[idx] = ROOM_GAP_BASE
    extra = width - required_width
    additions = _distribute_extra(extra, slot_count)
    slots = [base + inc for base, inc in zip(slots, additions)]

    left_positions: list[int] = []
    cursor = slots[0]
    for idx, spec in enumerate(room_specs):
        left_positions.append(cursor)
        cursor += spec["outer_width"]
        if idx + 1 < len(room_specs):
            cursor += slots[idx + 1]

    grid = np.zeros((height, width), dtype=int)
    merged_wall = np.zeros((height, width), dtype=bool)
    for left, spec in zip(left_positions, room_specs):
        _place_room(
            grid,
            merged_wall,
            left=left,
            widths=spec["widths"],
            vertical_len=spec["vertical_len"],
            verticals_on_top=bool(spec["verticals_on_top"]),
        )

    _add_distractors(grid, merged_wall)

    try:
        _validate_input(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-merge flood-fill input")
