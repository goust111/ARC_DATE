from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))

# Generator configuration.
GRID_HEIGHT_RANGE = (20, 28)
GRID_WIDTH_RANGE = (20, 28)
GROUP_COUNT_RANGE = (2, 4)
ODD_COUNTS = (3, 5, 7)
ODD_WEIGHTS = (1, 4, 4)
EVEN_COUNTS = (4, 6)
ZONE_RADIUS_RANGE = (4, 5)
ANCHOR_PADDING = 1
ANCHOR_ATTEMPTS = 300
POINT_ATTEMPTS = 800
GENERATION_ATTEMPTS = 400


def _integer_median(values: list[int]) -> int:
    ordered = sorted(int(v) for v in values)
    n = len(ordered)
    if n % 2 == 1:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) // 2


def _compact_mask(count: int) -> np.ndarray:
    width = min(3, int(count))
    height = (int(count) + width - 1) // width
    mask = np.zeros((height, width), dtype=bool)
    for idx in range(int(count)):
        mask[idx // width, idx % width] = True
    if count % 2 == 1:
        mask = mask[:, ::-1]
    return mask


def _bbox_overlaps(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    padding: int = 0,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + padding < br0
        or br1 + padding < ar0
        or ac1 + padding < bc0
        or bc1 + padding < ac0
    )


def _output_cells(anchor_r: int, anchor_c: int, count: int) -> set[tuple[int, int]]:
    mask = _compact_mask(count)
    cells: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        cells.add((anchor_r + int(r), anchor_c + int(c)))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered pixels by color, compress each color into a compact block,
    and flip odd-sized blocks horizontally before placing them at the color's
    median anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    height, width = g.shape
    out = np.full_like(g, BACKGROUND)

    for color in (int(v) for v in np.unique(g) if int(v) != BACKGROUND):
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue

        rows = [int(r) for r in coords[:, 0]]
        cols = [int(c) for c in coords[:, 1]]
        anchor_r = _integer_median(rows)
        anchor_c = _integer_median(cols)

        mask = _compact_mask(int(coords.shape[0]))
        block_h, block_w = map(int, mask.shape)
        assert 0 <= anchor_r <= height - block_h
        assert 0 <= anchor_c <= width - block_w

        region = out[anchor_r : anchor_r + block_h, anchor_c : anchor_c + block_w]
        assert not bool(np.any(region[mask] != BACKGROUND))
        region[mask] = color

    return out


def _sample_group_counts(group_count: int) -> list[int]:
    parities = [random.choice([0, 1]) for _ in range(group_count)]
    if group_count >= 2 and (all(parities) or not any(parities)):
        flip_idx = random.randrange(group_count)
        parities[flip_idx] = 1 - parities[flip_idx]

    counts = []
    for parity in parities:
        if parity:
            counts.append(random.choices(ODD_COUNTS, weights=ODD_WEIGHTS, k=1)[0])
        else:
            counts.append(random.choice(EVEN_COUNTS))
    random.shuffle(counts)
    return counts


def _sample_anchor_layout(
    grid_shape: tuple[int, int],
    counts: list[int],
    row_phase: int,
    col_phase: int,
) -> tuple[list[tuple[int, int]], list[int]] | None:
    height, width = grid_shape
    radii = [random.randint(*ZONE_RADIUS_RANGE) for _ in counts]
    order = sorted(range(len(counts)), key=lambda i: radii[i], reverse=True)

    anchors: list[tuple[int, int] | None] = [None] * len(counts)
    zone_boxes: list[tuple[int, int, int, int]] = []
    output_boxes: list[tuple[int, int, int, int]] = []

    for idx in order:
        count = counts[idx]
        radius = radii[idx]
        mask = _compact_mask(count)
        block_h, block_w = map(int, mask.shape)

        row_candidates = [
            r
            for r in range(radius, height - radius)
            if r % 2 == row_phase and r + block_h <= height
        ]
        col_candidates = [
            c
            for c in range(radius, width - radius)
            if c % 2 == col_phase and c + block_w <= width
        ]
        if not row_candidates or not col_candidates:
            return None

        placed = False
        for _ in range(ANCHOR_ATTEMPTS):
            anchor_r = random.choice(row_candidates)
            anchor_c = random.choice(col_candidates)
            zone_box = (
                anchor_r - radius,
                anchor_c - radius,
                anchor_r + radius,
                anchor_c + radius,
            )
            output_box = (
                anchor_r,
                anchor_c,
                anchor_r + block_h - 1,
                anchor_c + block_w - 1,
            )
            if any(_bbox_overlaps(zone_box, other, padding=ANCHOR_PADDING) for other in zone_boxes):
                continue
            if any(_bbox_overlaps(output_box, other, padding=1) for other in output_boxes):
                continue

            anchors[idx] = (anchor_r, anchor_c)
            zone_boxes.append(zone_box)
            output_boxes.append(output_box)
            placed = True
            break

        if not placed:
            return None

    return [anchor for anchor in anchors if anchor is not None], radii


def _sample_group_points(
    grid_shape: tuple[int, int],
    anchor: tuple[int, int],
    count: int,
    radius: int,
    row_phase: int,
    col_phase: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    height, width = grid_shape
    anchor_r, anchor_c = anchor
    target_output = _output_cells(anchor_r, anchor_c, count)

    candidates = [
        (r, c)
        for r in range(max(0, anchor_r - radius), min(height, anchor_r + radius + 1))
        for c in range(max(0, anchor_c - radius), min(width, anchor_c + radius + 1))
        if r % 2 == row_phase and c % 2 == col_phase and (r, c) not in occupied
    ]
    if len(candidates) < count:
        return None

    for _ in range(POINT_ATTEMPTS):
        pts = random.sample(candidates, count)
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]

        if _integer_median(rows) != anchor_r or _integer_median(cols) != anchor_c:
            continue
        if min(rows) >= anchor_r or max(rows) <= anchor_r:
            continue
        if min(cols) >= anchor_c or max(cols) <= anchor_c:
            continue
        if max(rows) - min(rows) < 2 or max(cols) - min(cols) < 2:
            continue
        if set(pts) == target_output:
            continue

        return pts

    return None


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    group_count = random.randint(*GROUP_COUNT_RANGE)
    counts = _sample_group_counts(group_count)
    colors = random.sample(PALETTE, group_count)
    row_phase = random.randint(0, 1)
    col_phase = random.randint(0, 1)

    layout = _sample_anchor_layout((height, width), counts, row_phase, col_phase)
    if layout is None:
        return None
    anchors, radii = layout

    grid = np.full((height, width), BACKGROUND, dtype=int)
    occupied: set[tuple[int, int]] = set()

    for color, count, anchor, radius in zip(colors, counts, anchors, radii):
        pts = _sample_group_points(
            (height, width),
            anchor,
            count,
            radius,
            row_phase,
            col_phase,
            occupied,
        )
        if pts is None:
            return None
        for r, c in pts:
            grid[r, c] = color
            occupied.add((r, c))

    out = main(grid)
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
