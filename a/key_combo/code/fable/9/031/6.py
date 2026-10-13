from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


ANCHOR = 8
MARKER = 2
BACKGROUND = 0
BASE_CHOICES = (1, 3, 4, 5, 6, 7, 9)
SIDE_CHOICES = (5, 6, 7)
PANEL_COUNT_CHOICES = (3, 4, 5)
MIN_GAP = 1
MAX_GAP = 3
MAX_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _orient_to_anchor_tl(patch: np.ndarray) -> np.ndarray:
    matches = []
    for k in range(4):
        oriented = np.rot90(patch, k=k)
        if int(oriented[0, 0]) == ANCHOR:
            matches.append(oriented)
    assert len(matches) == 1, "anchor must identify a unique corner orientation"
    return matches[0].copy()


def _panel_base(patch: np.ndarray) -> int:
    counts = Counter(map(int, patch.ravel()))
    base, _freq = counts.most_common(1)[0]
    assert base not in {BACKGROUND, ANCHOR, MARKER}
    return int(base)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate anchored square panels into a common orientation, then reveal only multi-panel marker coincidences.

    Input structure:
    - A background canvas contains several separated square panels of the same size.
    - Every panel is filled with one shared base color, has one cyan anchor pixel in a corner,
      and contains several red marker pixels elsewhere.
    - Different panels may be rotated by 0/90/180/270 degrees, and the anchor corner indicates each rotation.

    Rule:
    - Extract each square panel and rotate it so the cyan anchor moves to the top-left corner.
    - Create one output square of the shared panel size, filled with the shared base color; do not copy anchors.
    - For every cell position, paint it red in the output iff at least two aligned panels have a red marker there.

    Key insight:
    - The anchor is an orientation key, not part of the picture. After registration, red pixels are combined by
      coincidence threshold rather than union, so singleton markers disappear as noise.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    panels: list[np.ndarray] = []
    for coords in _components_mask(g != bg):
        bbox = _bbox_coords(coords)
        patch = _crop_bbox(g, bbox)
        h, w = patch.shape
        assert h == w >= 3
        assert np.all(patch != bg)

        anchor_locs = np.argwhere(patch == ANCHOR)
        assert len(anchor_locs) == 1
        ar, ac = map(int, anchor_locs[0])
        assert (ar, ac) in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
        panels.append(patch)

    assert len(panels) >= 2
    assert len({panel.shape for panel in panels}) == 1

    aligned = [_orient_to_anchor_tl(panel) for panel in panels]
    base_colors = {_panel_base(panel) for panel in aligned}
    assert len(base_colors) == 1
    base = next(iter(base_colors))

    side = aligned[0].shape[0]
    votes = np.zeros((side, side), dtype=int)
    for panel in aligned:
        assert int(panel[0, 0]) == ANCHOR
        votes += (panel == MARKER).astype(int)

    out = np.full((side, side), base, dtype=int)
    out[votes >= 2] = MARKER
    return out


def _expected_output(side: int, base: int, marker_sets: list[set[tuple[int, int]]], threshold: int = 2) -> np.ndarray:
    votes = np.zeros((side, side), dtype=int)
    for cells in marker_sets:
        for r, c in cells:
            votes[r, c] += 1
    out = np.full((side, side), base, dtype=int)
    out[votes >= threshold] = MARKER
    return out


def _random_partition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _sample_marker_sets(side: int, panel_count: int) -> list[set[tuple[int, int]]] | None:
    interior = [(r, c) for r in range(1, side - 1) for c in range(1, side - 1)]
    max_survivors = min(5, len(interior) - panel_count)
    if max_survivors < 2:
        return None

    for _ in range(200):
        survivor_count = random.randint(2, max_survivors)
        survivors = random.sample(interior, survivor_count)
        survivor_supports: list[set[int]] = []
        marker_sets = [set() for _ in range(panel_count)]

        for cell in survivors:
            support_size = 2 if random.random() < 0.7 else random.randint(2, panel_count)
            support = set(random.sample(range(panel_count), support_size))
            survivor_supports.append(support)
            for idx in support:
                marker_sets[idx].add(cell)

        if any(not any(idx in support for support in survivor_supports) for idx in range(panel_count)):
            continue
        if not any(len(support) == 2 for support in survivor_supports):
            continue
        if not any(len(support) < panel_count for support in survivor_supports):
            continue

        remaining = [cell for cell in interior if cell not in survivors]
        random.shuffle(remaining)
        if len(remaining) < panel_count:
            continue

        for idx in range(panel_count):
            marker_sets[idx].add(remaining.pop())

        extra_noise = random.randint(0, min(2, len(remaining)))
        for _ in range(extra_noise):
            marker_sets[random.randrange(panel_count)].add(remaining.pop())

        sizes = [len(cells) for cells in marker_sets]
        if min(sizes) < 2 or max(sizes) > 6:
            continue

        votes = np.zeros((side, side), dtype=int)
        for cells in marker_sets:
            for r, c in cells:
                votes[r, c] += 1
        if not np.any(votes >= 2):
            continue
        if not np.any(votes == 1):
            continue
        return marker_sets

    return None


def _render_panel(side: int, base: int, markers: set[tuple[int, int]]) -> np.ndarray:
    panel = np.full((side, side), base, dtype=int)
    panel[0, 0] = ANCHOR
    for r, c in markers:
        panel[r, c] = MARKER
    return panel


def _sample_layout(panel_count: int, side: int) -> tuple[int, int, list[tuple[int, int]]] | None:
    for _ in range(200):
        n_rows = random.randint(1, min(3, panel_count))
        row_counts = _random_partition(panel_count, n_rows)
        row_gaps = [random.randint(MIN_GAP, MAX_GAP) for _ in range(max(0, n_rows - 1))]
        top_border = random.randint(1, 3)
        bottom_border = random.randint(1, 3)
        min_h = top_border + bottom_border + n_rows * side + sum(row_gaps)
        if min_h > 30:
            continue

        row_widths = []
        gap_lists = []
        for count in row_counts:
            gaps = [random.randint(MIN_GAP, MAX_GAP) for _ in range(max(0, count - 1))]
            gap_lists.append(gaps)
            row_widths.append(count * side + sum(gaps))

        left_border = random.randint(1, 3)
        right_border = random.randint(1, 3)
        min_w = left_border + right_border + max(row_widths)
        if min_w > 30:
            continue

        height = random.randint(min_h, min(30, min_h + 5))
        width = random.randint(min_w, min(30, min_w + 5))

        slack_slots = [0] * (n_rows + 1)
        for _ in range(height - min_h):
            slack_slots[random.randrange(n_rows + 1)] += 1

        positions: list[tuple[int, int]] = []
        r = top_border + slack_slots[0]
        for row_idx, count in enumerate(row_counts):
            row_width = row_widths[row_idx]
            c = random.randint(left_border, width - right_border - row_width)
            for gap_idx in range(count):
                positions.append((r, c))
                c += side
                if gap_idx < count - 1:
                    c += gap_lists[row_idx][gap_idx]
            r += side
            if row_idx < n_rows - 1:
                r += row_gaps[row_idx] + slack_slots[row_idx + 1]

        random.shuffle(positions)
        return height, width, positions

    return None


def _raw_threshold_output(raw_panels: list[np.ndarray], base: int) -> np.ndarray:
    side = raw_panels[0].shape[0]
    votes = np.zeros((side, side), dtype=int)
    for panel in raw_panels:
        votes += (panel == MARKER).astype(int)
    out = np.full((side, side), base, dtype=int)
    out[votes >= 2] = MARKER
    return out


def _validate_generated(
    grid: np.ndarray,
    expected: np.ndarray,
    aligned_markers: list[set[tuple[int, int]]],
    raw_panels: list[np.ndarray],
    rotations: list[int],
    positions: list[tuple[int, int]],
) -> bool:
    if _get_bg(grid) != BACKGROUND:
        return False

    try:
        solved = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False

    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if len(set(rotations)) == 1 and rotations[0] == 0:
        return False
    if not any(rot != 0 for rot in rotations):
        return False

    union_like = _expected_output(expected.shape[0], int(expected[0, 0]), aligned_markers, threshold=1)
    if np.array_equal(union_like, expected):
        return False

    raw_threshold = _raw_threshold_output(raw_panels, int(expected[0, 0]))
    if np.array_equal(raw_threshold, expected):
        return False

    if np.count_nonzero(expected == MARKER) == 0:
        return False
    if not np.any(union_like != expected):
        return False

    occupied = np.zeros_like(grid, dtype=bool)
    side = raw_panels[0].shape[0]
    for panel, (r0, c0) in zip(raw_panels, positions):
        if panel.shape != (side, side):
            return False
        if np.count_nonzero(panel == ANCHOR) != 1:
            return False
        if grid[r0 : r0 + side, c0 : c0 + side].shape != panel.shape:
            return False
        if occupied[max(0, r0 - 1) : min(grid.shape[0], r0 + side + 1), max(0, c0 - 1) : min(grid.shape[1], c0 + side + 1)].any():
            return False
        occupied[r0 : r0 + side, c0 : c0 + side] = True
    return True


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(SIDE_CHOICES)
    panel_count = random.choice(PANEL_COUNT_CHOICES)
    base = random.choice(BASE_CHOICES)

    marker_sets = _sample_marker_sets(side, panel_count)
    if marker_sets is None:
        return None

    layout = _sample_layout(panel_count, side)
    if layout is None:
        return None
    height, width, positions = layout

    aligned_panels = [_render_panel(side, base, markers) for markers in marker_sets]
    rotations = [random.randrange(4) for _ in range(panel_count)]
    if not any(rot != 0 for rot in rotations):
        rotations[random.randrange(panel_count)] = random.randrange(1, 4)

    raw_panels = [np.rot90(panel, k=rot) for panel, rot in zip(aligned_panels, rotations)]
    expected = _expected_output(side, base, marker_sets, threshold=2)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for panel, (r0, c0) in zip(raw_panels, positions):
        grid[r0 : r0 + side, c0 : c0 + side] = panel

    if not _validate_generated(grid, expected, marker_sets, raw_panels, rotations, positions):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for anchored coincidence overlay task")
