from __future__ import annotations

import random
from collections import Counter

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def label_mask(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    comp_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            comp_id += 1
            stack = [(r, c)]
            labels[r, c] = comp_id
            while stack:
                rr, cc = stack.pop()
                for dr, dc in deltas:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = comp_id
                        stack.append((nr, nc))
    return labels, comp_id


def _runs(bits: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(np.asarray(bits, dtype=bool).tolist()):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(bits) - 1))
    return runs


def _split_panels(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    occupied_cols = np.any(grid != bg, axis=0)
    runs = _runs(occupied_cols)
    assert len(runs) >= 2, "expected left barrier panel and right candidate panel"

    gap_choices = []
    for i in range(len(runs) - 1):
        left_end = runs[i][1]
        right_start = runs[i + 1][0]
        gap_choices.append((right_start - left_end - 1, i))
    gap, idx = max(gap_choices)
    assert gap >= 1, "expected at least one empty separator column"

    left = grid[:, : runs[idx][1] + 1]
    right = grid[:, runs[idx + 1][0] :]
    left_box = bbox(left != bg)
    right_box = bbox(right != bg)
    assert left_box is not None and right_box is not None
    return crop_bbox(left, left_box), crop_bbox(right, right_box)


def _count_enclosed_regions(panel: np.ndarray, bg: int) -> tuple[int, int, int]:
    bg_labels, bg_count = label_mask(panel == bg, connectivity=4)
    enclosed = 0
    boundary_touching = 0
    h, w = panel.shape
    for comp_id in range(1, bg_count + 1):
        coords = np.argwhere(bg_labels == comp_id)
        touches = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)
        if touches:
            boundary_touching += 1
        else:
            enclosed += 1
    return enclosed, boundary_touching, bg_count


def _candidate_bboxes(panel: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    labels, comp_count = label_mask(panel != bg, connectivity=4)
    boxes: list[tuple[int, int, int, int]] = []
    for comp_id in range(1, comp_count + 1):
        box = bbox(labels == comp_id)
        assert box is not None
        boxes.append(box)
    boxes.sort(key=lambda box: (box[0], box[1]))
    return boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count enclosed background chambers in the left barrier drawing and use that count to index the right-side sprite queue.

    Input structure:
    - The left part of the grid is a larger single-color barrier drawing on one background color.
    - That barrier drawing splits its local background into several 4-connected regions, of which only some are fully enclosed.
    - The right part is a top-to-bottom queue of separated candidate sprites on the same background.

    Rule:
    - Split the scene into the left barrier panel and the right candidate panel using the empty separator gap.
    - In the left panel, count 4-connected background regions that do not touch the panel boundary.
    - Take the Nth candidate sprite from the right panel in top-to-bottom order, where N is that enclosed-region count, and crop it to its bounding box.

    Key insight:
    - The barrier drawing does not visually match the answer; it only encodes a hidden index through the number of closed background chambers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    left_panel, right_panel = _split_panels(g, bg)
    enclosed, _open_count, _total_bg = _count_enclosed_regions(left_panel, bg)
    assert enclosed >= 1, "expected at least one enclosed chamber"

    boxes = _candidate_bboxes(right_panel, bg)
    assert enclosed <= len(boxes), "candidate queue shorter than enclosed-region index"
    return crop_bbox(right_panel, boxes[enclosed - 1])


ARC_COLORS = tuple(range(10))
NON_BG_COLORS = tuple(range(1, 10))
MAX_TRIES = 400


def _draw_rect_border(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color


def _partitions_with_large_part(n: int) -> list[list[int]]:
    if n == 1:
        return [[1]]
    out = [[n]]
    for a in range(1, n):
        b = n - a
        if max(a, b) > 1:
            out.append([a, b])
    return out


def _build_barrier_panel(bg: int, barrier: int) -> tuple[np.ndarray, int, list[int]]:
    for _ in range(200):
        target_n = random.choice([1, 2, 2, 3, 3, 4, 4])
        chambers_per_box = random.choice(_partitions_with_large_part(target_n))
        trunk_col = random.randint(1, 2)
        box_left = trunk_col + random.randint(2, 3)
        top_margin = random.randint(1, 2)
        bottom_margin = random.randint(1, 2)

        box_specs: list[tuple[int, list[int]]] = []
        for chambers in chambers_per_box:
            width_hi = 3 if chambers <= 2 else 2
            chamber_widths = [random.randint(1, width_hi) for _ in range(chambers)]
            box_height = random.randint(5, 7)
            box_specs.append((box_height, chamber_widths))

        gaps = [random.randint(2, 4) for _ in range(max(0, len(box_specs) - 1))]
        panel_h = top_margin + sum(spec[0] for spec in box_specs) + sum(gaps) + bottom_margin
        box_widths = [2 + sum(widths) + (len(widths) - 1) for _h, widths in box_specs]
        panel_w = box_left + max(box_widths) + random.randint(1, 3)
        if panel_h > 22 or panel_w > 22:
            continue

        panel = np.full((panel_h, panel_w), bg, dtype=int)
        panel[:, trunk_col] = barrier

        row = top_margin
        connector_rows: list[int] = []
        gap_steps = list(gaps)
        for (box_height, chamber_widths), box_width in zip(box_specs, box_widths):
            _draw_rect_border(panel, row, box_left, box_height, box_width, barrier)
            offset = box_left + 1
            for width in chamber_widths[:-1]:
                offset += width
                panel[row : row + box_height, offset] = barrier
                offset += 1

            mid_row = row + box_height // 2
            panel[mid_row, trunk_col : box_left + 1] = barrier
            connector_rows.append(mid_row)
            row += box_height
            if gap_steps:
                row += gap_steps.pop(0)

        free_rows = [r for r in range(panel_h) if r not in connector_rows]
        random.shuffle(free_rows)
        for r in free_rows[: random.randint(0, min(2, len(free_rows)))]:
            panel[r, 0 : trunk_col + 1] = barrier

        enclosed, boundary_touching, _total_bg = _count_enclosed_regions(panel, bg)
        occupied_cols = np.any(panel != bg, axis=0)
        runs = _runs(occupied_cols)
        if enclosed == target_n and boundary_touching >= 2 and len(runs) == 1:
            return panel, enclosed, chambers_per_box

    raise RuntimeError("failed to build a valid barrier panel")


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(x) for x in row) for row in mask.astype(int))


def _gen_sprite_mask(h: int, w: int, used: set[tuple[tuple[int, ...], ...]]) -> np.ndarray:
    for _ in range(200):
        spine = random.randrange(w)
        left_force = random.randrange(h)
        right_force = random.randrange(h)
        mask = np.zeros((h, w), dtype=bool)
        intervals = []
        for r in range(h):
            lo = random.randint(0, spine)
            hi = random.randint(spine, w - 1)
            if r == left_force:
                lo = 0
            if r == right_force:
                hi = w - 1
            mask[r, lo : hi + 1] = True
            intervals.append((lo, hi))
        if not np.all(mask.any(axis=0)):
            continue
        if len(set(intervals)) == 1 and random.random() < 0.8:
            continue
        sig = _mask_signature(mask)
        if sig in used:
            continue
        used.add(sig)
        return mask
    raise RuntimeError("failed to generate a distinct sprite mask")


def _render_sprite(mask: np.ndarray, bg: int, color: int) -> np.ndarray:
    sprite = np.full(mask.shape, bg, dtype=int)
    sprite[mask] = color
    return sprite


def _place_panel(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _validate_instance(
    grid: np.ndarray,
    expected_output: np.ndarray,
    expected_n: int,
    chambers_per_box: list[int],
    candidate_count: int,
) -> bool:
    bg = get_bg(grid)
    left_panel, right_panel = _split_panels(grid, bg)
    enclosed, boundary_touching, total_bg = _count_enclosed_regions(left_panel, bg)
    if enclosed != expected_n:
        return False
    if boundary_touching < 2:
        return False
    if total_bg <= candidate_count:
        return False
    boxes = _candidate_bboxes(right_panel, bg)
    if len(boxes) != candidate_count:
        return False
    if len({_mask_signature(crop_bbox(right_panel, box) != bg) for box in boxes}) != len(boxes):
        return False
    if expected_n > 1 and max(chambers_per_box) <= 1:
        return False
    if not np.array_equal(crop_bbox(right_panel, boxes[expected_n - 1]), expected_output):
        return False
    if not np.array_equal(main(grid), expected_output):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_TRIES):
        bg = random.choice(ARC_COLORS)
        barrier = random.choice([c for c in ARC_COLORS if c != bg])
        left_panel, enclosed_count, chambers_per_box = _build_barrier_panel(bg, barrier)

        slot_w = random.randint(4, 6)
        candidate_count = random.choice([enclosed_count, enclosed_count + 1])
        if candidate_count > 5:
            candidate_count = enclosed_count

        gap_rows = random.randint(2, 3)
        heights = [random.randint(3, 5) for _ in range(candidate_count)]
        queue_h = sum(heights) + gap_rows * (candidate_count - 1)
        grid_h = max(left_panel.shape[0] + random.randint(0, 2), queue_h + random.randint(0, 2))
        if grid_h > 28:
            continue

        left_top = random.randint(0, grid_h - left_panel.shape[0])
        right_top = random.randint(0, grid_h - queue_h)
        separator = random.randint(2, 4)
        left_margin = random.randint(0, 1)
        right_margin = random.randint(1, 2)
        grid_w = left_margin + left_panel.shape[1] + separator + slot_w + right_margin
        if grid_w > 28:
            continue

        grid = np.full((grid_h, grid_w), bg, dtype=int)
        _place_panel(grid, left_panel, left_top, left_margin)

        available_candidate_colors = [c for c in ARC_COLORS if c != bg]
        random.shuffle(available_candidate_colors)
        if barrier in available_candidate_colors and random.random() < 0.45:
            available_candidate_colors.remove(barrier)
            candidate_colors = [barrier] + available_candidate_colors[: candidate_count - 1]
            random.shuffle(candidate_colors)
        else:
            candidate_colors = available_candidate_colors[:candidate_count]

        used_masks: set[tuple[tuple[int, ...], ...]] = set()
        row = right_top
        right_col = left_margin + left_panel.shape[1] + separator
        expected_output: np.ndarray | None = None
        for idx, (sprite_h, color) in enumerate(zip(heights, candidate_colors), start=1):
            mask = _gen_sprite_mask(sprite_h, slot_w, used_masks)
            sprite = _render_sprite(mask, bg, int(color))
            _place_panel(grid, sprite, row, right_col)
            if idx == enclosed_count:
                expected_output = sprite.copy()
            row += sprite_h + gap_rows

        assert expected_output is not None
        if _validate_instance(grid, expected_output, enclosed_count, chambers_per_box, candidate_count):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
