from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, paste, random_free_location


INPUT_COLORS = [0, 1, 2, 3, 5, 6, 7, 8, 9]
TEMPLATE_H_RANGE = (4, 8)
TEMPLATE_W_RANGE = (4, 8)
COPY_COUNT_RANGE = (4, 7)
GEN_RETRIES = 400
_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _sorted_copies(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    copies = find_components(grid, bg=bg, connectivity=8, monochromatic=False)
    copies = sorted(copies, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    assert len(copies) >= 4, "expected at least four translated copies"

    shapes = {
        (
            int(obj["bbox"][2] - obj["bbox"][0] + 1),
            int(obj["bbox"][3] - obj["bbox"][1] + 1),
        )
        for obj in copies
    }
    assert len(shapes) == 1, "all copies must share one template bbox size"
    return [crop_bbox(grid, obj["bbox"]) for obj in copies]


def _majority_patch(patches: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(patches, axis=0)
    n_copies, height, width = stack.shape
    out = np.empty((height, width), dtype=int)

    for r in range(height):
        for c in range(width):
            values, counts = np.unique(stack[:, r, c], return_counts=True)
            best = int(np.argmax(counts))
            assert int(counts[best]) > n_copies // 2, "template recovery needs a strict majority"
            out[r, c] = int(values[best])
    return out


def _anomaly_coords(patches: list[np.ndarray], template: np.ndarray) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for patch in patches:
        diff = np.argwhere(patch != template)
        assert diff.shape[0] == 1, "each copy must contain exactly one swapped cell"
        coord = tuple(map(int, diff[0]))
        assert coord not in seen, "anomaly coordinates must be distinct across copies"
        seen.add(coord)
        coords.append(coord)

    return coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the shared two-color sprite by majority vote, then mark the distributed swapped cells in yellow.

    Input structure:
    - A dominant background surrounds at least four non-overlapping translated copies of one template.
    - Each copy uses the same two non-background colors inside one shared bbox size, with no rotation or reflection.
    - Exactly one cell in every copy is wrong: its value is swapped to the template's other pattern color.

    Rule:
    - Extract the repeated copies and align them by their bbox coordinates.
    - Recover the true template with per-cell majority vote across all aligned copies.
    - For each copy, locate its unique mismatch against the recovered template.
    - Output the recovered template cropped to one bbox, then recolor all gathered mismatch coordinates to yellow.

    Key insight:
    - No copy reveals the hidden figure alone; only cross-copy voting identifies the true sprite and isolates each one-cell swap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = int(get_bg(g))
    patches = _sorted_copies(g, bg)
    template = _majority_patch(patches)
    out = template.copy()

    for r, c in _anomaly_coords(patches, template):
        out[int(r), int(c)] = int(Color.YELLOW)

    return out


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in _NEIGHBORS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _grow_connected_cells(height: int, width: int, target: int) -> set[tuple[int, int]] | None:
    if target < 1 or target > height * width:
        return None

    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier: set[tuple[int, int]] = set(_neighbors4(*start, height, width))

    while len(cells) < target:
        if not frontier:
            return None

        ranked = sorted(
            frontier,
            key=lambda rc: sum(nb in cells for nb in _neighbors4(rc[0], rc[1], height, width)),
            reverse=True,
        )
        if random.random() < 0.75:
            cell = random.choice(ranked[: min(4, len(ranked))])
        else:
            cell = random.choice(ranked)

        frontier.remove(cell)
        cells.add(cell)
        for nb in _neighbors4(cell[0], cell[1], height, width):
            if nb not in cells:
                frontier.add(nb)

    return cells


def _make_template(height: int, width: int, color_a: int, color_b: int) -> np.ndarray | None:
    area = height * width
    low = max(4, (area + 2) // 3)
    high = min(area - 4, (2 * area) // 3)
    if low > high:
        return None

    for _ in range(120):
        blob = _grow_connected_cells(height, width, random.randint(low, high))
        if blob is None:
            continue

        rows = {r for r, _ in blob}
        cols = {c for _, c in blob}
        if len(rows) < 2 or len(cols) < 2:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in blob:
            mask[int(r), int(c)] = True

        mixed_rows = sum(0 < int(mask[r].sum()) < width for r in range(height))
        mixed_cols = sum(0 < int(mask[:, c].sum()) < height for c in range(width))
        if mixed_rows < 2 or mixed_cols < 2:
            continue

        patch = np.full((height, width), int(color_a), dtype=int)
        patch[mask] = int(color_b)
        return patch

    return None


def _choose_anomalies(template: np.ndarray, count: int) -> list[tuple[int, int]] | None:
    height, width = template.shape

    for require_both_colors in (True, False):
        for _ in range(160):
            cells = _grow_connected_cells(height, width, count)
            if cells is None:
                continue

            rows = {r for r, _ in cells}
            cols = {c for _, c in cells}
            if len(rows) < 2 or len(cols) < 2:
                continue

            colors = {int(template[r, c]) for r, c in cells}
            if require_both_colors and len(colors) < 2:
                continue

            return sorted((int(r), int(c)) for r, c in cells)

    return None


def _canvas_shape(template: np.ndarray, copy_count: int) -> tuple[int, int] | None:
    height, width = template.shape
    occupied = height * width * copy_count
    counts = [int(np.count_nonzero(template == color)) for color in np.unique(template)]
    max_copy_color = max(counts) * copy_count + copy_count
    min_area = max(
        copy_count * (height + 2) * (width + 2),
        occupied + max_copy_color + 5,
    )
    if min_area > 30 * 30:
        return None

    preferred: list[tuple[int, int]] = []
    fallback: list[tuple[int, int]] = []
    for canvas_h in range(max(height + 2, 15), 31):
        for canvas_w in range(max(width + 2, 15), 31):
            area = canvas_h * canvas_w
            if area < min_area:
                continue
            aspect = canvas_h / canvas_w
            if not (0.55 <= aspect <= 1.8):
                continue
            fallback.append((canvas_h, canvas_w))
            if area <= min(900, min_area + 180):
                preferred.append((canvas_h, canvas_w))

    options = preferred or fallback
    if not options:
        return None
    return random.choice(options)


def _mutated_copy(template: np.ndarray, coord: tuple[int, int]) -> np.ndarray:
    colors = [int(v) for v in np.unique(template)]
    assert len(colors) == 2

    r, c = coord
    out = template.copy()
    out[int(r), int(c)] = colors[1] if int(out[int(r), int(c)]) == colors[0] else colors[0]
    return out


def _expected_output(template: np.ndarray, anomalies: list[tuple[int, int]]) -> np.ndarray:
    out = template.copy()
    for r, c in anomalies:
        out[int(r), int(c)] = int(Color.YELLOW)
    return out


def _generate_case() -> np.ndarray | None:
    height = random.randint(*TEMPLATE_H_RANGE)
    width = random.randint(*TEMPLATE_W_RANGE)
    copy_count = random.randint(COPY_COUNT_RANGE[0], min(COPY_COUNT_RANGE[1], height * width - 1))
    bg, color_a, color_b = random.sample(INPUT_COLORS, 3)

    template = _make_template(height, width, color_a, color_b)
    if template is None:
        return None

    anomalies = _choose_anomalies(template, copy_count)
    if anomalies is None:
        return None

    canvas_shape = _canvas_shape(template, copy_count)
    if canvas_shape is None:
        return None

    grid = np.full(canvas_shape, int(bg), dtype=int)
    copies = [_mutated_copy(template, coord) for coord in anomalies]
    random.shuffle(copies)

    for patch in copies:
        try:
            top = random_free_location(
                grid,
                patch,
                transparent=int(bg),
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(grid, patch, int(top[0]), int(top[1]))

    if int(get_bg(grid)) != int(bg):
        return None
    if len(find_components(grid, bg=bg, connectivity=8, monochromatic=False)) != copy_count:
        return None

    expected = _expected_output(template, anomalies)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
