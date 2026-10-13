from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    stack = [tuple(map(int, coords[0]))]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, mask.shape):
            if not mask[nr, nc] or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _find_divider_column(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(w):
        color = int(grid[0, c])
        if color == bg:
            continue
        if np.all(grid[:, c] == color) and int(np.count_nonzero(grid == color)) == h:
            candidates.append((c, color))
    assert len(candidates) == 1, "expected exactly one full-height divider column"
    return candidates[0]


def _complete_sprite(left_patch: np.ndarray) -> np.ndarray:
    full = np.concatenate((left_patch, np.fliplr(left_patch)), axis=1)
    assert full.shape[0] == full.shape[1], "completed sprite must be square"
    return full


def _mark_color(sprite: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, sprite[sprite != bg].ravel()))
    mark = [color for color, count in counts.items() if count == 2]
    assert len(mark) == 1, "completed sprite must contain one mirrored marker pair"
    return int(mark[0])


def _rotation_marker_sets(sprite: np.ndarray, mark_color: int) -> list[frozenset[tuple[int, int]]]:
    marker_sets = []
    for k in range(4):
        rotated = np.rot90(sprite, k=k)
        marker_sets.append(frozenset(map(tuple, np.argwhere(rotated == mark_color))))
    assert len(set(marker_sets)) == 4, "marker pair must distinguish all four rotations"
    return marker_sets


def _partition_starts(length: int, slot_size: int) -> list[int]:
    candidates: list[list[int]] = []

    if length % slot_size == 0:
        candidates.append([i * slot_size for i in range(length // slot_size)])

    gap = 1
    if (length + gap) % (slot_size + gap) == 0:
        count = (length + gap) // (slot_size + gap)
        if count >= 1 and count * slot_size + (count - 1) * gap == length:
            candidates.append([i * (slot_size + gap) for i in range(count)])

    unique = []
    for starts in candidates:
        if starts not in unique:
            unique.append(starts)
    assert len(unique) == 1, "right-side slots must partition unambiguously"
    return unique[0]


def _template_parts(grid: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray, int]:
    bg = _get_bg(grid)
    divider_c, _divider_color = _find_divider_column(grid, bg)
    left_zone = grid[:, :divider_c]
    box = _bbox(left_zone != bg)
    assert box is not None, "expected a half sprite to the left of the divider"
    left_patch = _crop_bbox(left_zone, box)
    full_sprite = _complete_sprite(left_patch)
    mark_color = _mark_color(full_sprite, bg)
    return bg, divider_c, left_patch, full_sprite, mark_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror-complete the half sprite beside the divider, then stamp the full sprite into marked slots using marker-pair rotation.

    Input structure:
    - A full-height single-color divider column splits the grid into a narrow left template area and a wider right slot area.
    - The left area contains one vertically placed half sprite touching the divider side; it uses one body color plus one distinct marker cell.
    - The right area is partitioned into equal square slots matching the completed sprite size; some slots are blank and some contain exactly the two marker pixels of one rotated full sprite.

    Rule:
    - Crop the left half sprite and mirror it left-right to reconstruct the complete square sprite, which therefore contains a symmetric pair of marker cells.
    - For each marked slot, find the unique rotation whose marker-pair coordinates exactly match the two slot markers.
    - Replace that whole slot with the full sprite in the matched rotation.
    - Leave blank slots, the divider column, and the left half template unchanged.

    Key insight:
    - The divider is both a layout separator and the mirror cue for rebuilding the hidden template, while each slot's two-pixel marker pair is the only orientation signal for the stamp.
    """
    g = np.asarray(grid, dtype=int)
    bg, divider_c, _left_patch, full_sprite, mark_color = _template_parts(g)
    slot_size = int(full_sprite.shape[0])
    row_starts = _partition_starts(g.shape[0], slot_size)
    col_starts = _partition_starts(g.shape[1] - divider_c - 1, slot_size)

    rotated_sprites = [np.rot90(full_sprite, k=k) for k in range(4)]
    marker_sets = _rotation_marker_sets(full_sprite, mark_color)

    out = g.copy()
    base_c = divider_c + 1
    for r0 in row_starts:
        for c0_rel in col_starts:
            c0 = base_c + c0_rel
            slot = g[r0 : r0 + slot_size, c0 : c0 + slot_size]
            coords = frozenset(map(tuple, np.argwhere(slot == mark_color)))
            non_bg = set(map(int, np.unique(slot))) - {bg}
            if not coords:
                assert not non_bg, "blank slot may not contain stray colors"
                continue
            assert non_bg == {mark_color}, "marked slot must contain only the marker color"
            assert len(coords) == 2, "marked slot must contain exactly two marker pixels"
            matches = [k for k, marker_coords in enumerate(marker_sets) if marker_coords == coords]
            assert len(matches) == 1, "marker pair must select a unique rotation"
            _paste(out, rotated_sprites[matches[0]], r0, c0)

    return out


def _connect_points(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    tr, tc = end
    mask[r, c] = True
    while (r, c) != (tr, tc):
        vertical_first = bool(random.getrandbits(1))
        if vertical_first and r != tr:
            r += 1 if tr > r else -1
        elif c != tc:
            c += 1 if tc > c else -1
        elif r != tr:
            r += 1 if tr > r else -1
        mask[r, c] = True


def _random_half_mask(full_size: int) -> np.ndarray:
    half_w = full_size // 2
    area = full_size * half_w
    for _ in range(200):
        mask = np.zeros((full_size, half_w), dtype=bool)
        border_points = [
            (0, random.randrange(half_w)),
            (full_size - 1, random.randrange(half_w)),
            (random.randrange(full_size), 0),
            (random.randrange(full_size), half_w - 1),
        ]
        points = []
        for point in border_points:
            if point not in points:
                points.append(point)
                mask[point] = True

        random.shuffle(points)
        current = points[0]
        for point in points[1:]:
            _connect_points(mask, current, point)
            current = point

        target = random.randint(max(full_size + 1, area // 2), area - 1)
        while int(mask.sum()) < target:
            frontier = set()
            for r, c in np.argwhere(mask):
                for nr, nc in _neighbors4(int(r), int(c), mask.shape):
                    if not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            mask[random.choice(tuple(frontier))] = True

        if (
            _is_connected(mask)
            and mask[0].any()
            and mask[-1].any()
            and mask[:, 0].any()
            and mask[:, -1].any()
            and int(mask.sum()) < area
        ):
            return mask
    raise RuntimeError("failed to generate a valid half sprite mask")


def _make_left_patch(full_size: int, body_color: int, mark_color: int) -> tuple[np.ndarray, np.ndarray]:
    mask = _random_half_mask(full_size)
    patch = np.full(mask.shape, BACKGROUND, dtype=int)
    patch[mask] = int(body_color)

    body_cells = list(map(tuple, np.argwhere(mask)))
    mark_r, mark_c = random.choice(body_cells)
    patch[mark_r, mark_c] = int(mark_color)

    full_sprite = _complete_sprite(patch)
    assert len(set(_rotation_marker_sets(full_sprite, mark_color))) == 4
    return patch, full_sprite


def _slot_layout(full_size: int) -> tuple[int, int, int]:
    if full_size == 6:
        return 2, 3, 1
    return random.choice(((2, 3, 1), (2, 4, 1)))


def _slot_starts(count: int, slot_size: int, gap: int) -> list[int]:
    return [i * (slot_size + gap) for i in range(count)]


def _marker_patch(full_sprite: np.ndarray, mark_color: int, rotation: int) -> np.ndarray:
    rotated = np.rot90(full_sprite, k=rotation)
    patch = np.full(rotated.shape, BACKGROUND, dtype=int)
    patch[rotated == mark_color] = int(mark_color)
    return patch


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    full_size = random.choice((4, 6))
    half_w = full_size // 2
    slot_rows, slot_cols, gap = _slot_layout(full_size)

    divider_color, body_color, mark_color = random.sample(PALETTE, 3)
    left_patch, full_sprite = _make_left_patch(full_size, body_color, mark_color)

    height = slot_rows * full_size + (slot_rows - 1) * gap
    right_width = slot_cols * full_size + (slot_cols - 1) * gap
    width = half_w + 1 + right_width
    assert max(height, width) <= 30

    grid = np.full((height, width), BACKGROUND, dtype=int)
    divider_c = half_w
    grid[:, divider_c] = int(divider_color)

    left_top = random.randint(0, height - full_size)
    _paste(grid, left_patch, left_top, 0)

    slot_positions = [
        (r0, divider_c + 1 + c0)
        for r0 in _slot_starts(slot_rows, full_size, gap)
        for c0 in _slot_starts(slot_cols, full_size, gap)
    ]
    total_slots = len(slot_positions)
    empty_slots = random.randint(1, min(2, total_slots - 4))
    marked_count = total_slots - empty_slots

    marked_indices = random.sample(range(total_slots), marked_count)
    rotations = [0, 1, 2, 3]
    while len(rotations) < marked_count:
        rotations.append(random.randrange(4))
    random.shuffle(rotations)

    expected = grid.copy()
    for idx, rotation in zip(marked_indices, rotations):
        r0, c0 = slot_positions[idx]
        _paste(grid, _marker_patch(full_sprite, mark_color, rotation), r0, c0)
        _paste(expected, np.rot90(full_sprite, k=rotation), r0, c0)

    marker_sets = _rotation_marker_sets(full_sprite, mark_color)
    assert len(set(rotations)) == 4
    for idx, rotation in zip(marked_indices, rotations):
        r0, c0 = slot_positions[idx]
        slot = grid[r0 : r0 + full_size, c0 : c0 + full_size]
        coords = frozenset(map(tuple, np.argwhere(slot == mark_color)))
        assert coords == marker_sets[rotation]
    for idx in set(range(total_slots)) - set(marked_indices):
        r0, c0 = slot_positions[idx]
        slot = grid[r0 : r0 + full_size, c0 : c0 + full_size]
        assert not np.any(slot != BACKGROUND)

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid, expected = _build_case()
        out = main(grid)
        if np.array_equal(out, expected):
            return grid
    raise RuntimeError("failed to generate a valid mirror-stamp puzzle")
