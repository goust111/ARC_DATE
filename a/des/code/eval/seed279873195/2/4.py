from __future__ import annotations

import random

import numpy as np


BG = 0
GRAY = 5
SEPARATOR = 9
SHAPE_COLORS = [1, 2, 3, 4, 6, 7, 8]
DIRECTIONS = ("up", "down", "left", "right")
MAX_GENERATION_ATTEMPTS = 400
MAX_SHAPE_ATTEMPTS = 200


def _intervals_from_separator_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start = 0
    for idx, is_separator in enumerate(map(bool, mask)):
        if is_separator:
            if start < idx:
                intervals.append((start, idx))
            start = idx + 1
    if start < len(mask):
        intervals.append((start, len(mask)))
    return intervals


def _on_border(pos: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = pos
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _parse_slot_region(region: np.ndarray) -> dict | None:
    coords = np.argwhere(region != BG)
    if len(coords) != 2:
        return None

    values = [int(region[tuple(coord)]) for coord in coords]
    if values.count(GRAY) != 1:
        return None

    grey_idx = values.index(GRAY)
    color_idx = 1 - grey_idx
    color_value = values[color_idx]
    if color_value in (BG, GRAY, SEPARATOR):
        return None

    grey_pos = tuple(map(int, coords[grey_idx]))
    color_pos = tuple(map(int, coords[color_idx]))
    dr = grey_pos[0] - color_pos[0]
    dc = grey_pos[1] - color_pos[1]
    direction_map = {
        (-1, 0): "up",
        (1, 0): "down",
        (0, -1): "left",
        (0, 1): "right",
    }
    if (dr, dc) not in direction_map:
        return None
    if not (_on_border(grey_pos, region.shape) or _on_border(color_pos, region.shape)):
        return None

    return {
        "color": int(color_value),
        "direction": direction_map[(dr, dc)],
    }


def _extract_patterns(reference: np.ndarray) -> dict[int, np.ndarray]:
    patterns: dict[int, np.ndarray] = {}
    colors = sorted(set(map(int, np.unique(reference))) - {BG, GRAY, SEPARATOR})
    for color in colors:
        coords = np.argwhere(reference == color)
        if len(coords) == 0:
            continue
        r0, c0 = np.min(coords, axis=0)
        r1, c1 = np.max(coords, axis=0)
        patch = np.where(
            reference[r0 : r1 + 1, c0 : c1 + 1] == color,
            color,
            BG,
        ).astype(int)
        patterns[int(color)] = patch
    return patterns


def _transform_patch(patch: np.ndarray, direction: str) -> np.ndarray:
    if direction == "up":
        return patch.copy()
    if direction == "down":
        return np.flipud(patch)
    if direction == "left":
        return np.fliplr(patch)
    if direction == "right":
        return np.flipud(np.fliplr(patch))
    raise ValueError(direction)


def _paste_top_left(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    ph, pw = map(int, patch.shape)
    sub = canvas[r0 : r0 + ph, c0 : c0 + pw]
    mask = patch != BG
    sub[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each empty slot with the color-matched reference pattern transformed by the slot arrow.

    The grid is partitioned by full maroon separator rows and columns. One region contains
    multiple colored reference shapes; every other active region is an empty slot containing a
    two-pixel marker made of one grey cell and one colored cell. The colored marker cell selects
    which reference shape to use, and the grey cell's offset from the colored cell selects the
    flip transform:
    - grey above  -> no flip
    - grey below  -> vertical flip
    - grey left   -> horizontal flip
    - grey right  -> both flips (180-degree rotation)
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_bands = _intervals_from_separator_mask(np.all(g == SEPARATOR, axis=1))
    col_bands = _intervals_from_separator_mask(np.all(g == SEPARATOR, axis=0))
    assert row_bands and col_bands

    slot_regions: list[dict] = []
    reference_candidates: list[dict] = []

    for row_idx, (r0, r1) in enumerate(row_bands):
        for col_idx, (c0, c1) in enumerate(col_bands):
            region = g[r0:r1, c0:c1]
            slot = _parse_slot_region(region)
            if slot is not None:
                slot_regions.append(
                    {
                        **slot,
                        "row_idx": int(row_idx),
                        "col_idx": int(col_idx),
                        "r0": int(r0),
                        "r1": int(r1),
                        "c0": int(c0),
                        "c1": int(c1),
                    }
                )
                continue

            non_bg_count = int(np.count_nonzero(region != BG))
            if non_bg_count == 0:
                continue
            shape_colors = sorted(set(map(int, np.unique(region))) - {BG, GRAY, SEPARATOR})
            if not shape_colors:
                continue
            area = int(region.shape[0] * region.shape[1])
            reference_candidates.append(
                {
                    "row_idx": int(row_idx),
                    "col_idx": int(col_idx),
                    "r0": int(r0),
                    "r1": int(r1),
                    "c0": int(c0),
                    "c1": int(c1),
                    "region": region,
                    "n_shape_colors": int(len(shape_colors)),
                    "non_bg_count": int(non_bg_count),
                    "area": int(area),
                }
            )

    assert reference_candidates
    reference = max(
        reference_candidates,
        key=lambda item: (
            int(item["n_shape_colors"]),
            int(item["non_bg_count"]),
            int(item["area"]),
            -int(item["r0"]),
            -int(item["c0"]),
        ),
    )
    patterns = _extract_patterns(reference["region"])
    assert patterns

    out = g.copy()
    for slot in slot_regions:
        r0, r1 = int(slot["r0"]), int(slot["r1"])
        c0, c1 = int(slot["c0"]), int(slot["c1"])
        color = int(slot["color"])
        assert color in patterns
        patch = _transform_patch(patterns[color], slot["direction"])
        slot_h, slot_w = r1 - r0, c1 - c0
        ph, pw = map(int, patch.shape)
        assert ph <= slot_h and pw <= slot_w
        out[r0:r1, c0:c1] = BG
        _paste_top_left(out, patch, r0, c0)

    return out


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier: set[tuple[int, int]] = set()
    mask[start] = True

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add((nr, nc))

    add_frontier(*start)
    while len(cells) < area and frontier:
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        if mask[r, c]:
            continue
        mask[r, c] = True
        cells.add((r, c))
        add_frontier(r, c)
    return mask


def _interesting_mask(mask: np.ndarray) -> bool:
    if mask.shape[0] < 2 or mask.shape[1] < 2:
        return False
    if mask.all():
        return False
    if not mask[0, :].any():
        return False
    if not mask[-1, :].any():
        return False
    if not mask[:, 0].any():
        return False
    if not mask[:, -1].any():
        return False
    if np.array_equal(mask, np.flipud(mask)):
        return False
    if np.array_equal(mask, np.fliplr(mask)):
        return False
    if np.array_equal(mask, np.flipud(np.fliplr(mask))):
        return False
    return True


def _make_shape_patch(max_h: int, max_w: int, color: int) -> np.ndarray | None:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        height = random.randint(2, max_h)
        width = random.randint(2, max_w)
        max_area = height * width - 1
        min_area = max(3, (height * width) // 2)
        if min_area > max_area:
            continue
        area = random.randint(min_area, max_area)
        mask = _random_connected_mask(height, width, area)
        if not _interesting_mask(mask):
            continue
        patch = np.full((height, width), BG, dtype=int)
        patch[mask] = color
        return patch
    return None


def _zone_specs(ref_h: int, ref_w: int) -> list[tuple[int, int, int, int]]:
    top_h = (ref_h - 3) // 2
    bottom_h = ref_h - 3 - top_h
    left_w = (ref_w - 3) // 2
    right_w = ref_w - 3 - left_w
    return [
        (1, 1, top_h, left_w),
        (1, left_w + 2, top_h, right_w),
        (top_h + 2, 1, bottom_h, left_w),
        (top_h + 2, left_w + 2, bottom_h, right_w),
    ]


def _build_reference_region() -> tuple[np.ndarray, dict[int, np.ndarray]]:
    ref_h = random.randint(9, 11)
    ref_w = random.randint(9, 11)
    reference = np.full((ref_h, ref_w), BG, dtype=int)
    zones = _zone_specs(ref_h, ref_w)

    n_shapes = random.randint(2, 4)
    zone_ids = random.sample(range(len(zones)), n_shapes)
    colors = random.sample(SHAPE_COLORS, n_shapes)
    patterns: dict[int, np.ndarray] = {}

    for zone_id, color in zip(zone_ids, colors):
        r0, c0, zh, zw = zones[zone_id]
        patch = _make_shape_patch(min(zh, 4), min(zw, 4), color)
        assert patch is not None
        ph, pw = map(int, patch.shape)
        rr = r0 + random.randint(0, zh - ph)
        cc = c0 + random.randint(0, zw - pw)
        sub = reference[rr : rr + ph, cc : cc + pw]
        sub[patch != BG] = color
        patterns[int(color)] = patch

    return reference, patterns


def _slot_sizes(reference_dim: int, min_patch_dim: int, count: int) -> list[int]:
    sizes: list[int] = []
    low = max(4, min_patch_dim)
    high = max(low, reference_dim - 2)
    for _ in range(count):
        sizes.append(random.randint(low, high))
    return sizes


def _segment_starts(sizes: list[int]) -> list[int]:
    starts: list[int] = []
    pos = 0
    for idx, size in enumerate(sizes):
        starts.append(pos)
        pos += size
        if idx + 1 < len(sizes):
            pos += 1
    return starts


def _compatible_sides(
    row_idx: int,
    col_idx: int,
    n_rows: int,
    n_cols: int,
    direction: str,
) -> list[str]:
    sides: list[str] = []
    if direction in ("up", "down"):
        if row_idx > 0:
            sides.append("top")
        if row_idx < n_rows - 1:
            sides.append("bottom")
    else:
        if col_idx > 0:
            sides.append("left")
        if col_idx < n_cols - 1:
            sides.append("right")
    return sides


def _edge_anchor(length: int) -> int:
    if length <= 2:
        return 0
    if length == 3:
        return 1
    return random.randint(1, length - 2)


def _arrow_cells(
    height: int,
    width: int,
    side: str,
    direction: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if side == "top":
        c = _edge_anchor(width)
        if direction == "up":
            return (1, c), (0, c)
        return (0, c), (1, c)
    if side == "bottom":
        c = _edge_anchor(width)
        if direction == "up":
            return (height - 1, c), (height - 2, c)
        return (height - 2, c), (height - 1, c)
    if side == "left":
        r = _edge_anchor(height)
        if direction == "left":
            return (r, 1), (r, 0)
        return (r, 0), (r, 1)
    if side == "right":
        r = _edge_anchor(height)
        if direction == "left":
            return (r, width - 1), (r, width - 2)
        return (r, width - 2), (r, width - 1)
    raise ValueError(side)


def _repeat_shuffled(values: list[int] | list[str], length: int) -> list:
    out: list = []
    while len(out) < length:
        block = list(values)
        random.shuffle(block)
        out.extend(block)
    return out[:length]


def _render_expected(
    grid: np.ndarray,
    row_starts: list[int],
    col_starts: list[int],
    row_sizes: list[int],
    col_sizes: list[int],
    assignments: list[dict],
    patterns: dict[int, np.ndarray],
) -> np.ndarray:
    out = grid.copy()
    for assignment in assignments:
        row_idx = int(assignment["row_idx"])
        col_idx = int(assignment["col_idx"])
        r0 = int(row_starts[row_idx])
        c0 = int(col_starts[col_idx])
        r1 = r0 + int(row_sizes[row_idx])
        c1 = c0 + int(col_sizes[col_idx])
        out[r0:r1, c0:c1] = BG
        patch = _transform_patch(patterns[int(assignment["color"])], assignment["direction"])
        _paste_top_left(out, patch, r0, c0)
    return out


def _gen_attempt() -> np.ndarray | None:
    n_rows, n_cols = random.choice([(2, 3), (3, 2), (3, 3)])
    reference, patterns = _build_reference_region()

    max_patch_h = max(int(patch.shape[0]) for patch in patterns.values())
    max_patch_w = max(int(patch.shape[1]) for patch in patterns.values())

    row_sizes = [int(reference.shape[0])] + _slot_sizes(int(reference.shape[0]), max_patch_h, n_rows - 1)
    col_sizes = [int(reference.shape[1])] + _slot_sizes(int(reference.shape[1]), max_patch_w, n_cols - 1)
    height = sum(row_sizes) + (n_rows - 1)
    width = sum(col_sizes) + (n_cols - 1)
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    row_starts = _segment_starts(row_sizes)
    col_starts = _segment_starts(col_sizes)

    for idx in range(1, n_rows):
        grid[row_starts[idx] - 1, :] = SEPARATOR
    for idx in range(1, n_cols):
        grid[:, col_starts[idx] - 1] = SEPARATOR

    ref_h, ref_w = map(int, reference.shape)
    grid[row_starts[0] : row_starts[0] + ref_h, col_starts[0] : col_starts[0] + ref_w] = reference

    slots = [(r, c) for r in range(n_rows) for c in range(n_cols) if (r, c) != (0, 0)]
    random.shuffle(slots)
    directions = _repeat_shuffled(list(DIRECTIONS), len(slots))
    colors = _repeat_shuffled(list(patterns.keys()), len(slots))
    assignments: list[dict] = []

    for (row_idx, col_idx), direction, color in zip(slots, directions, colors):
        slot_h = int(row_sizes[row_idx])
        slot_w = int(col_sizes[col_idx])
        side = random.choice(_compatible_sides(row_idx, col_idx, n_rows, n_cols, direction))
        color_pos, grey_pos = _arrow_cells(slot_h, slot_w, side, direction)
        r0 = int(row_starts[row_idx])
        c0 = int(col_starts[col_idx])
        grid[r0 + color_pos[0], c0 + color_pos[1]] = int(color)
        grid[r0 + grey_pos[0], c0 + grey_pos[1]] = GRAY
        assignments.append(
            {
                "row_idx": int(row_idx),
                "col_idx": int(col_idx),
                "direction": direction,
                "color": int(color),
                "side": side,
            }
        )

    expected = _render_expected(
        grid,
        row_starts,
        col_starts,
        row_sizes,
        col_sizes,
        assignments,
        patterns,
    )

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
