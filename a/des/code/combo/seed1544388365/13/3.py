import random

import numpy as np


DIRECTIONS = ("up", "right", "down", "left")
DIR_VECTORS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
MAX_GENERATION_ATTEMPTS = 500


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, height: int, width: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, height, width):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _border_mask(height: int, width: int) -> np.ndarray:
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    border = _border_mask(r1 - r0 + 1, c1 - c0 + 1)
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = border
    return mask


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for value in np.unique(grid):
        color = int(value)
        if color == bg:
            continue
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height < 3 or width < 3:
                continue
            if len(coords) != 2 * (height + width) - 4:
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = _border_mask(height, width)
            if not np.all(patch[border] == color):
                continue
            if np.any(patch[~border] == color):
                continue
            inner_colors = set(map(int, np.unique(patch[1:-1, 1:-1]))) - {bg}
            if len(inner_colors) != 1:
                continue
            candidates.append((len(coords), color, (r0, c0, r1, c1)))
    assert candidates, "failed to find the frame"
    candidates.sort(key=lambda item: item[0], reverse=True)
    _area, color, bbox = candidates[0]
    return color, bbox


def _rotate_offsets(offsets: np.ndarray, turns_cw: int) -> np.ndarray:
    turns_cw %= 4
    rotated = offsets.copy().astype(int)
    for _ in range(turns_cw):
        rotated = np.column_stack((rotated[:, 1], -rotated[:, 0]))
    return rotated


def _extract_template(
    grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int
) -> tuple[int, str, np.ndarray]:
    r0, c0, r1, c1 = bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    colors = list(set(map(int, np.unique(interior))) - {bg})
    assert len(colors) == 1, "frame must contain exactly one template color"
    color = colors[0]
    mask = interior == color
    assert np.any(mask), "template must be non-empty"

    side_counts = {
        "up": int(mask[0, :].sum()),
        "down": int(mask[-1, :].sum()),
        "left": int(mask[:, 0].sum()),
        "right": int(mask[:, -1].sum()),
    }
    base_sides = [side for side, count in side_counts.items() if count == 1]
    assert len(base_sides) == 1, "template must have one unique touching side"
    base_side = base_sides[0]
    for side, count in side_counts.items():
        if side != base_side:
            assert count == 0, "template must touch only the base side"

    if base_side == "up":
        contact = (0, int(np.flatnonzero(mask[0, :])[0]))
    elif base_side == "down":
        contact = (mask.shape[0] - 1, int(np.flatnonzero(mask[-1, :])[0]))
    elif base_side == "left":
        contact = (int(np.flatnonzero(mask[:, 0])[0]), 0)
    else:
        contact = (int(np.flatnonzero(mask[:, -1])[0]), mask.shape[1] - 1)

    offsets = np.argwhere(mask).astype(int)
    offsets[:, 0] -= int(contact[0])
    offsets[:, 1] -= int(contact[1])
    return color, base_side, offsets


def _find_anchors(
    grid: np.ndarray, frame_mask: np.ndarray, frame_color: int, marker_color: int, bg: int
) -> list[tuple[int, int, str]]:
    height, width = grid.shape
    anchors: list[tuple[int, int, str]] = []
    coords = np.argwhere((grid == frame_color) & (~frame_mask))
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if any(grid[nr, nc] == frame_color for nr, nc in _neighbors4(rr, cc, height, width)):
            continue
        directions: list[str] = []
        valid = True
        for side in DIRECTIONS:
            dr, dc = DIR_VECTORS[side]
            nr = rr + dr
            nc = cc + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            value = int(grid[nr, nc])
            if value == bg:
                continue
            if value == marker_color:
                directions.append(side)
            else:
                valid = False
                break
        if valid and len(directions) == 1:
            anchors.append((rr, cc, directions[0]))
    assert anchors, "no valid anchors found"
    return anchors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the pattern inside the frame as a stamp.

    The unique pixel touching the frame determines the template's default orientation.
    Each outside frame-color anchor has one adjacent marker whose direction selects the
    rotation. The rotated template is stamped with its contact pixel aligned to the
    anchor position. The output keeps only the stamped copies on a black background.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    bg = _bg_color(g)

    frame_color, frame_bbox = _find_frame(g, bg)
    marker_color, base_side, base_offsets = _extract_template(g, frame_bbox, bg)
    frame_mask = _frame_mask(g.shape, frame_bbox)
    anchors = _find_anchors(g, frame_mask, frame_color, marker_color, bg)

    out = np.full_like(g, bg)
    base_index = DIRECTIONS.index(base_side)
    for anchor_r, anchor_c, target_side in anchors:
        turns = (DIRECTIONS.index(target_side) - base_index) % 4
        stamp_offsets = _rotate_offsets(base_offsets, turns)
        for dr, dc in stamp_offsets:
            rr = anchor_r + int(dr)
            cc = anchor_c + int(dc)
            if 0 <= rr < height and 0 <= cc < width:
                out[rr, cc] = marker_color
    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _has_nontrivial_symmetry(mask: np.ndarray) -> bool:
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for turns in range(4):
            transformed = np.rot90(base, -turns)
            if not flip and turns == 0:
                continue
            if transformed.shape == mask.shape and np.array_equal(transformed, mask):
                return True
    return False


def _generate_canonical_offsets(rng: random.Random) -> np.ndarray | None:
    for _ in range(300):
        height = rng.randint(3, 5)
        width = rng.randint(3, 5)
        contact_col = rng.randrange(width)
        mask = np.zeros((height, width), dtype=bool)
        mask[0, contact_col] = True

        next_cols = [contact_col]
        if contact_col > 0:
            next_cols.append(contact_col - 1)
        if contact_col + 1 < width:
            next_cols.append(contact_col + 1)
        mask[1, rng.choice(next_cols)] = True

        target_area = rng.randint(4, min(8, height * width - (width - 1)))
        while int(mask.sum()) < target_area:
            cells = np.argwhere(mask)
            cr, cc = cells[rng.randrange(len(cells))]
            dr, dc = rng.choice(((1, 0), (0, -1), (0, 1), (-1, 0)))
            nr = int(cr + dr)
            nc = int(cc + dc)
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if nr == 0 and nc != contact_col:
                continue
            mask[nr, nc] = True

        cropped = _crop_mask(mask)
        if int(cropped[0, :].sum()) != 1:
            continue
        if cropped.shape[0] < 2 or cropped.shape[1] < 2:
            continue
        if len(_components(cropped)) != 1:
            continue
        if _has_nontrivial_symmetry(cropped):
            continue

        contact = (0, int(np.flatnonzero(cropped[0, :])[0]))
        offsets = np.argwhere(cropped).astype(int)
        offsets[:, 0] -= int(contact[0])
        offsets[:, 1] -= int(contact[1])
        return offsets
    return None


def _offset_bounds(offsets: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(offsets[:, 0].min()),
        int(offsets[:, 1].min()),
        int(offsets[:, 0].max()),
        int(offsets[:, 1].max()),
    )


def _build_interior_pattern(
    offsets: np.ndarray, base_side: str, rng: random.Random
) -> tuple[np.ndarray, tuple[int, int]]:
    min_r, min_c, max_r, max_c = _offset_bounds(offsets)
    margin_top = 0 if base_side == "up" else rng.randint(1, 3)
    margin_bottom = 0 if base_side == "down" else rng.randint(1, 3)
    margin_left = 0 if base_side == "left" else rng.randint(1, 3)
    margin_right = 0 if base_side == "right" else rng.randint(1, 3)

    shift_r = margin_top - min_r
    shift_c = margin_left - min_c
    interior_height = margin_top + (max_r - min_r + 1) + margin_bottom
    interior_width = margin_left + (max_c - min_c + 1) + margin_right
    pattern = np.zeros((interior_height, interior_width), dtype=bool)
    for dr, dc in offsets:
        pattern[shift_r + int(dr), shift_c + int(dc)] = True

    side_counts = {
        "up": int(pattern[0, :].sum()),
        "down": int(pattern[-1, :].sum()),
        "left": int(pattern[:, 0].sum()),
        "right": int(pattern[:, -1].sum()),
    }
    assert side_counts[base_side] == 1
    for side, count in side_counts.items():
        if side != base_side:
            assert count == 0
    return pattern, (int(shift_r), int(shift_c))


def _make_direction_list(count: int, rng: random.Random) -> list[str]:
    base = list(DIRECTIONS)
    rng.shuffle(base)
    result = base[: min(count, 4)]
    while len(result) < count:
        result.append(rng.choice(DIRECTIONS))
    rng.shuffle(result)
    return result


def _in_frame_box(r: int, c: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _can_place_single(
    cell: tuple[int, int],
    occupied: set[tuple[int, int]],
    height: int,
    width: int,
    allowed: set[tuple[int, int]],
) -> bool:
    r, c = cell
    if not (0 <= r < height and 0 <= c < width):
        return False
    if cell in occupied and cell not in allowed:
        return False
    for nr, nc in _neighbors4(r, c, height, width):
        if (nr, nc) in occupied and (nr, nc) not in allowed:
            return False
    return True


def _candidate_anchors(
    height: int,
    width: int,
    direction: str,
    stamp_offsets: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    occupied_input: set[tuple[int, int]],
    occupied_output: set[tuple[int, int]],
) -> list[tuple[int, int, tuple[int, int], set[tuple[int, int]], bool]]:
    candidates: list[tuple[int, int, tuple[int, int], set[tuple[int, int]], bool]] = []
    mark_dr, mark_dc = DIR_VECTORS[direction]
    for r in range(height):
        for c in range(width):
            marker = (r + mark_dr, c + mark_dc)
            if _in_frame_box(r, c, frame_bbox) or _in_frame_box(marker[0], marker[1], frame_bbox):
                continue
            if not _can_place_single((r, c), occupied_input, height, width, {marker}):
                continue
            if not _can_place_single(marker, occupied_input, height, width, {(r, c)}):
                continue

            all_cells = {(r + int(dr), c + int(dc)) for dr, dc in stamp_offsets}
            in_bounds = {
                (rr, cc) for rr, cc in all_cells if 0 <= rr < height and 0 <= cc < width
            }
            if len(in_bounds) < 3:
                continue
            if any(_in_frame_box(rr, cc, frame_bbox) for rr, cc in in_bounds):
                continue
            if in_bounds & occupied_output:
                continue
            clipped = len(in_bounds) < len(all_cells)
            candidates.append((r, c, marker, in_bounds, clipped))
    return candidates


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    canonical = _generate_canonical_offsets(rng)
    if canonical is None:
        return None

    bg = 0
    colors = rng.sample(range(1, 10), 2)
    frame_color, pattern_color = int(colors[0]), int(colors[1])
    base_side = rng.choice(DIRECTIONS)
    base_offsets = _rotate_offsets(canonical, DIRECTIONS.index(base_side))

    interior_pattern, _contact = _build_interior_pattern(base_offsets, base_side, rng)
    interior_height, interior_width = interior_pattern.shape
    frame_height = interior_height + 2
    frame_width = interior_width + 2

    max_height = 24
    max_width = 24
    min_height = frame_height + 6
    min_width = frame_width + 6
    if min_height > max_height or min_width > max_width:
        return None

    height = rng.randint(min_height, max_height)
    width = rng.randint(min_width, max_width)
    frame_r0 = rng.randint(2, height - frame_height - 2)
    frame_c0 = rng.randint(2, width - frame_width - 2)
    frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_height - 1, frame_c0 + frame_width - 1)

    grid = np.full((height, width), bg, dtype=int)
    r0, c0, r1, c1 = frame_bbox
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 + 1 : r1, c0 + 1 : c1][interior_pattern] = pattern_color

    occupied_input: set[tuple[int, int]] = set(map(tuple, np.argwhere(grid != bg)))
    occupied_output: set[tuple[int, int]] = set()
    expected = np.full_like(grid, bg)

    anchor_count = rng.randint(2, 5)
    directions = _make_direction_list(anchor_count, rng)
    need_clipped = rng.random() < 0.5
    clipped_used = False

    for index, direction in enumerate(directions):
        turns = (DIRECTIONS.index(direction) - DIRECTIONS.index(base_side)) % 4
        stamp_offsets = _rotate_offsets(base_offsets, turns)
        candidates = _candidate_anchors(
            height,
            width,
            direction,
            stamp_offsets,
            frame_bbox,
            occupied_input,
            occupied_output,
        )
        if not candidates:
            return None

        preferred = candidates
        if need_clipped and not clipped_used:
            clipped_candidates = [cand for cand in candidates if cand[4]]
            remaining = anchor_count - index
            if clipped_candidates:
                preferred = clipped_candidates
            elif remaining == 1:
                return None
        choice = rng.choice(preferred)
        anchor_r, anchor_c, marker, stamp_cells, clipped = choice
        if clipped:
            clipped_used = True

        grid[anchor_r, anchor_c] = frame_color
        grid[marker[0], marker[1]] = pattern_color
        occupied_input.add((anchor_r, anchor_c))
        occupied_input.add(marker)
        occupied_output |= stamp_cells
        for rr, cc in stamp_cells:
            expected[rr, cc] = pattern_color

    if need_clipped and not clipped_used:
        return None

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
