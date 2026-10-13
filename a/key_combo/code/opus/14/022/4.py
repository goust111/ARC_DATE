from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BACKGROUND = 0
MARKER = 9
BLOCK_SIZE = 3
GEN_ATTEMPTS = 800
GRID_MIN = 15
GRID_MAX = 20
MIN_SOURCE_GAP = 2

SIDE_NORMALS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

OVERLAP_DELTAS = [
    (-2, 0),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -2),
    (0, -1),
    (0, 1),
    (0, 2),
    (1, -1),
    (1, 0),
    (1, 1),
    (2, 0),
]


@dataclass(frozen=True)
class BlockSpec:
    color: int
    src_r0: int
    src_c0: int
    marker_side: str
    distance: int
    marker_start: int


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_anchor(r0: int, c0: int) -> tuple[int, int, int, int]:
    return int(r0), int(c0), int(r0 + BLOCK_SIZE - 1), int(c0 + BLOCK_SIZE - 1)


def _marker_cells(block: BlockSpec) -> list[tuple[int, int]]:
    r0, c0 = int(block.src_r0), int(block.src_c0)
    start = int(block.marker_start)
    count = int(block.distance)
    if block.marker_side == "up":
        return [(r0 - 1, c0 + start + i) for i in range(count)]
    if block.marker_side == "down":
        return [(r0 + BLOCK_SIZE, c0 + start + i) for i in range(count)]
    if block.marker_side == "left":
        return [(r0 + start + i, c0 - 1) for i in range(count)]
    return [(r0 + start + i, c0 + BLOCK_SIZE) for i in range(count)]


def _target_anchor(block: BlockSpec, *, reverse: bool = True, distance_override: int | None = None) -> tuple[int, int]:
    side_dr, side_dc = SIDE_NORMALS[block.marker_side]
    move_dr, move_dc = -side_dr, -side_dc
    if not reverse:
        move_dr, move_dc = side_dr, side_dc
    distance = int(block.distance if distance_override is None else distance_override)
    return int(block.src_r0 + move_dr * distance), int(block.src_c0 + move_dc * distance)


def _box_gaps(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> tuple[int, int]:
    row_gap = max(0, box_b[0] - box_a[2] - 1, box_a[0] - box_b[2] - 1)
    col_gap = max(0, box_b[1] - box_a[3] - 1, box_a[1] - box_b[3] - 1)
    return int(row_gap), int(col_gap)


def _boxes_have_min_gap(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], gap: int) -> bool:
    row_gap, col_gap = _box_gaps(box_a, box_b)
    return max(row_gap, col_gap) >= int(gap)


def _boxes_overlap(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> bool:
    return not (
        box_a[2] < box_b[0]
        or box_b[2] < box_a[0]
        or box_a[3] < box_b[1]
        or box_b[3] < box_a[1]
    )


def _overlap_area(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> int:
    if not _boxes_overlap(box_a, box_b):
        return 0
    rows = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]) + 1
    cols = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]) + 1
    return int(rows * cols)


def _extract_blocks(grid: np.ndarray) -> list[BlockSpec]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    blocks: list[BlockSpec] = []
    claimed_markers: set[tuple[int, int]] = set()
    colors = [int(color) for color in np.unique(g) if int(color) not in (bg, MARKER)]
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.shape[0] == 9, "each block color must form one solid 3x3 square"
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        assert (r1 - r0 + 1, c1 - c0 + 1) == (BLOCK_SIZE, BLOCK_SIZE)
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color)

        side_markers = {
            "up": [(r0 - 1, c) for c in range(c0, c1 + 1) if r0 > 0 and g[r0 - 1, c] == MARKER],
            "down": [(r1 + 1, c) for c in range(c0, c1 + 1) if r1 + 1 < g.shape[0] and g[r1 + 1, c] == MARKER],
            "left": [(r, c0 - 1) for r in range(r0, r1 + 1) if c0 > 0 and g[r, c0 - 1] == MARKER],
            "right": [(r, c1 + 1) for r in range(r0, r1 + 1) if c1 + 1 < g.shape[1] and g[r, c1 + 1] == MARKER],
        }
        used_sides = [side for side, pts in side_markers.items() if pts]
        assert len(used_sides) == 1, "each block must have markers on exactly one side"
        side = used_sides[0]
        pts = side_markers[side]
        assert 1 <= len(pts) <= 3
        if side in {"up", "down"}:
            coords_on_side = sorted(c for _, c in pts)
        else:
            coords_on_side = sorted(r for r, _ in pts)
        assert coords_on_side[-1] - coords_on_side[0] + 1 == len(coords_on_side), "markers must be contiguous"
        marker_start = int(coords_on_side[0] - (c0 if side in {"up", "down"} else r0))
        blocks.append(
            BlockSpec(
                color=int(color),
                src_r0=int(r0),
                src_c0=int(c0),
                marker_side=side,
                distance=int(len(pts)),
                marker_start=marker_start,
            )
        )
        claimed_markers.update((int(r), int(c)) for r, c in pts)

    all_markers = {(int(r), int(c)) for r, c in np.argwhere(g == MARKER)}
    assert claimed_markers == all_markers, "every marker must belong to exactly one block"
    assert 3 <= len(blocks) <= 4
    return blocks


def _render_output(
    shape: tuple[int, int],
    blocks: list[BlockSpec],
    *,
    reverse: bool = True,
    distance_override: int | None = None,
    shorter_on_top: bool = True,
    smaller_row_on_top: bool = True,
) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)

    def draw_key(block: BlockSpec) -> tuple[int, int]:
        dist_key = block.distance if shorter_on_top else -block.distance
        row_key = block.src_r0 if smaller_row_on_top else -block.src_r0
        return int(dist_key), int(row_key)

    for block in sorted(blocks, key=draw_key, reverse=True):
        tr0, tc0 = _target_anchor(block, reverse=reverse, distance_override=distance_override)
        out[tr0 : tr0 + BLOCK_SIZE, tc0 : tc0 + BLOCK_SIZE] = int(block.color)
    return out


def _scene_has_required_overlap(blocks: list[BlockSpec]) -> bool:
    targets = [_bbox_from_anchor(*_target_anchor(block)) for block in blocks]
    return any(
        _overlap_area(targets[i], targets[j]) > 0
        for i in range(len(targets))
        for j in range(i + 1, len(targets))
    )


def main(grid: np.ndarray) -> np.ndarray:
    """Project each marked 3x3 block opposite its marker side, using marker count as distance and shorter jumps as the top overlay.

    Input structure:
    - A black-background grid contains 3-4 solid 3x3 colored blocks, each in a distinct non-black non-maroon color.
    - Every block has maroon marker pixels on exactly one orthogonally adjacent side; the markers form a contiguous line of length 1-3.
    - Blocks are initially separated, while markers only touch their own block.

    Rule:
    - For each block, read the marker side and project the whole 3x3 block in the opposite direction.
    - The projection distance equals the number of maroon marker pixels on that side.
    - Remove all original blocks and markers, then draw only the projected blocks at their destination positions.
    - If projected blocks overlap, the block with the shorter distance is on top; if distances tie, the block whose original top row is smaller is on top.

    Key insight:
    - The maroon strip carries two pieces of information at once: its side gives a reversed direction, and its length gives the translation distance; overlaps reveal a separate layer-priority rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    blocks = _extract_blocks(g)
    return _render_output(g.shape, blocks)


def _render_input(shape: tuple[int, int], blocks: list[BlockSpec]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for block in blocks:
        grid[block.src_r0 : block.src_r0 + BLOCK_SIZE, block.src_c0 : block.src_c0 + BLOCK_SIZE] = int(block.color)
        for r, c in _marker_cells(block):
            grid[r, c] = MARKER
    return grid


def _block_fits(shape: tuple[int, int], block: BlockSpec) -> bool:
    h, w = shape
    if not (0 <= block.src_r0 <= h - BLOCK_SIZE and 0 <= block.src_c0 <= w - BLOCK_SIZE):
        return False
    for r, c in _marker_cells(block):
        if not (0 <= r < h and 0 <= c < w):
            return False
    tr0, tc0 = _target_anchor(block)
    return 0 <= tr0 <= h - BLOCK_SIZE and 0 <= tc0 <= w - BLOCK_SIZE


def _sample_marker_start(distance: int) -> int:
    return random.randint(0, BLOCK_SIZE - distance)


def _sample_colors(count: int) -> list[int]:
    palette = [1, 2, 3, 4, 5, 6, 7, 8]
    random.shuffle(palette)
    return palette[:count]


def _random_non_overlapping_target(
    shape: tuple[int, int],
    taken: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    h, w = shape
    for _ in range(120):
        r0 = random.randint(1, h - BLOCK_SIZE - 1)
        c0 = random.randint(1, w - BLOCK_SIZE - 1)
        box = _bbox_from_anchor(r0, c0)
        if all(not _boxes_overlap(box, other) and _boxes_have_min_gap(box, other, 1) for other in taken):
            return int(r0), int(c0)
    return None


def _sample_blocks() -> tuple[tuple[int, int], list[BlockSpec]] | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    shape = (h, w)
    count = random.choice([3, 4])
    colors = _sample_colors(count)
    use_tie_branch = random.random() < 0.45
    overlap_delta = random.choice(OVERLAP_DELTAS)

    pair_distance_a = random.randint(1, 3)
    pair_distance_b = pair_distance_a if use_tie_branch else random.choice([d for d in (1, 2, 3) if d != pair_distance_a])

    target_a = None
    target_b = None
    for _ in range(150):
        r0 = random.randint(3, h - BLOCK_SIZE - 3)
        c0 = random.randint(3, w - BLOCK_SIZE - 3)
        other_r0 = r0 + overlap_delta[0]
        other_c0 = c0 + overlap_delta[1]
        if 0 <= other_r0 <= h - BLOCK_SIZE and 0 <= other_c0 <= w - BLOCK_SIZE:
            target_a = (int(r0), int(c0))
            target_b = (int(other_r0), int(other_c0))
            break
    if target_a is None or target_b is None:
        return None

    blocks: list[BlockSpec] = []
    used_source_boxes: list[tuple[int, int, int, int]] = []
    target_boxes = [_bbox_from_anchor(*target_a), _bbox_from_anchor(*target_b)]
    distances = [pair_distance_a, pair_distance_b]
    all_sides = list(SIDE_NORMALS)
    for idx, target in enumerate((target_a, target_b)):
        choices = all_sides[:]
        random.shuffle(choices)
        built = None
        for side in choices:
            dr, dc = SIDE_NORMALS[side]
            src_r0 = target[0] + dr * distances[idx]
            src_c0 = target[1] + dc * distances[idx]
            block = BlockSpec(
                color=colors[idx],
                src_r0=int(src_r0),
                src_c0=int(src_c0),
                marker_side=side,
                distance=int(distances[idx]),
                marker_start=_sample_marker_start(distances[idx]),
            )
            if not _block_fits(shape, block):
                continue
            source_box = _bbox_from_anchor(block.src_r0, block.src_c0)
            if any(not _boxes_have_min_gap(source_box, other, MIN_SOURCE_GAP) for other in used_source_boxes):
                continue
            built = block
            used_source_boxes.append(source_box)
            blocks.append(built)
            break
        if built is None:
            return None

    taken_targets = target_boxes[:]
    for idx in range(2, count):
        target = _random_non_overlapping_target(shape, taken_targets)
        if target is None:
            return None
        side_choices = all_sides[:]
        random.shuffle(side_choices)
        if idx == 2 and len({block.distance for block in blocks}) == 1:
            candidate_distances = [d for d in (1, 2, 3) if d != blocks[0].distance]
        else:
            candidate_distances = [1, 2, 3]
        random.shuffle(candidate_distances)
        built = None
        for distance in candidate_distances:
            for side in side_choices:
                dr, dc = SIDE_NORMALS[side]
                src_r0 = target[0] + dr * distance
                src_c0 = target[1] + dc * distance
                block = BlockSpec(
                    color=colors[idx],
                    src_r0=int(src_r0),
                    src_c0=int(src_c0),
                    marker_side=side,
                    distance=int(distance),
                    marker_start=_sample_marker_start(distance),
                )
                if not _block_fits(shape, block):
                    continue
                source_box = _bbox_from_anchor(block.src_r0, block.src_c0)
                if any(not _boxes_have_min_gap(source_box, other, MIN_SOURCE_GAP) for other in used_source_boxes):
                    continue
                built = block
                used_source_boxes.append(source_box)
                blocks.append(block)
                taken_targets.append(_bbox_from_anchor(*target))
                break
            if built is not None:
                break
        if built is None:
            return None

    return shape, blocks


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        blocks = _extract_blocks(grid)
    except Exception:
        return False

    if len({block.marker_side for block in blocks}) < 2:
        return False
    if len({block.distance for block in blocks}) < 2:
        return False
    if not _scene_has_required_overlap(blocks):
        return False

    true_out = _render_output(grid.shape, blocks)
    if np.array_equal(true_out, grid):
        return False

    target_boxes = [_bbox_from_anchor(*_target_anchor(block)) for block in blocks]
    overlap_pairs = []
    has_distance_priority_case = False
    has_row_priority_case = False
    for i in range(len(target_boxes)):
        for j in range(i + 1, len(target_boxes)):
            overlap = _overlap_area(target_boxes[i], target_boxes[j])
            if overlap <= 0:
                continue
            overlap_pairs.append((i, j))
            if blocks[i].distance != blocks[j].distance:
                has_distance_priority_case = True
            else:
                if blocks[i].src_r0 == blocks[j].src_r0:
                    return False
                has_row_priority_case = True
    if not overlap_pairs:
        return False

    reverse_variant = _render_output(grid.shape, blocks, reverse=False)
    fixed_one_variant = _render_output(grid.shape, blocks, distance_override=1)
    fixed_three_variant = _render_output(grid.shape, blocks, distance_override=3)
    long_first_variant = _render_output(grid.shape, blocks, shorter_on_top=False)
    large_row_variant = _render_output(grid.shape, blocks, smaller_row_on_top=False)

    if np.array_equal(reverse_variant, true_out):
        return False
    if np.array_equal(fixed_one_variant, true_out) or np.array_equal(fixed_three_variant, true_out):
        return False
    if has_distance_priority_case and np.array_equal(long_first_variant, true_out):
        return False
    if has_row_priority_case and np.array_equal(large_row_variant, true_out):
        return False

    if not np.array_equal(main(grid), true_out):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        scene = _sample_blocks()
        if scene is None:
            continue
        shape, blocks = scene
        grid = _render_input(shape, blocks)
        if max(shape) > 30:
            continue
        if _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid marked-projection ARC input")
