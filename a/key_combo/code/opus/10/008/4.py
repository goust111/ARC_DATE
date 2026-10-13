from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
LEFT_FG = 1
RED = 2
RIGHT_FG = 3
SEP = 4
ORANGE = 7

PANEL_SIZE = 15
PANEL_SIZE_OPTIONS = (12, 13, 14)
GEN_RETRIES = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _is_single_component(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                queue.append((rr, cc))
    return len(seen) == int(mask.sum())


def _parse_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h > 0 and w % 2 == 1 and w >= 3, "expected left|separator|right layout"

    mid = w // 2
    sep_vals = np.unique(g[:, mid])
    assert sep_vals.size == 1, "separator column must be solid"

    left = g[:, :mid]
    right = g[:, mid + 1 :]
    assert left.shape == right.shape, "panels must have equal shapes"

    left_colors = set(map(int, np.unique(left)))
    right_colors = set(map(int, np.unique(right)))
    common = left_colors & right_colors
    assert len(common) == 1, "expected exactly one shared background color"
    bg = next(iter(common))
    assert int(sep_vals[0]) != bg, "separator must differ from background"

    left_only = left_colors - {bg}
    right_only = right_colors - {bg}
    assert len(left_only) == 1 and len(right_only) == 1, "each panel must be a binary mask"

    return left != bg, right != bg


def _trace_network(marker_mask: np.ndarray) -> np.ndarray:
    h, w = map(int, marker_mask.shape)
    out = np.zeros((h, w), dtype=int)
    out[marker_mask] = ORANGE
    assert not marker_mask[0, 0], "start cell cannot be a marker"

    pending: list[tuple[int, int, int, int]] = [(0, 0, 0, 1)]
    seen_segments: set[tuple[int, int, int, int]] = set()
    out[0, 0] = RED

    while pending:
        r, c, dr, dc = pending.pop()
        key = (int(r), int(c), int(dr), int(dc))
        if key in seen_segments:
            continue
        seen_segments.add(key)

        if out[r, c] == BG:
            out[r, c] = RED

        rr, cc = r + dr, c + dc
        while 0 <= rr < h and 0 <= cc < w:
            if marker_mask[rr, cc]:
                if dr == 0:
                    pending.append((rr - dr, cc - dc, -1, 0))
                    pending.append((rr - dr, cc - dc, 1, 0))
                else:
                    pending.append((rr - dr, cc - dc, 0, -1))
                    pending.append((rr - dr, cc - dc, 0, 1))
                break

            if out[rr, cc] == RED:
                break

            out[rr, cc] = RED
            rr += dr
            cc += dc

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare the two aligned side panels, keep only XOR differences as markers, and grow a red turning ray network from them.

    Input structure:
    - The grid has odd width with a solid center separator column splitting two equal-size panels.
    - The left panel is a binary foreground mask on a shared background, and the right panel is another binary mask in a different foreground color.
    - The separator is only an alignment guide and is not part of the output canvas.

    Rule:
    - Remove the separator and compare the two panels cell by cell.
    - Wherever exactly one panel has foreground, place an orange marker on the aligned output canvas.
    - On that output canvas, start at the top-left cell and send a red ray to the right; when the next cell is an orange marker, stop before it and launch the two perpendicular rays from the previous red cell.
    - Every ray paints background cells red and stops at the grid edge, an orange marker, or an already red cell; markers stay orange.

    Key insight:
    - The panel comparison only defines where the control markers sit; the final pattern comes from recursively turning rays at those difference markers rather than from the XOR mask alone.
    """
    left_mask, right_mask = _parse_panels(np.asarray(grid, dtype=int))
    marker_mask = np.logical_xor(left_mask, right_mask)
    return _trace_network(marker_mask)


def _empty_run(layout: np.ndarray, r: int, c: int, dr: int, dc: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    rr, cc = r + dr, c + dc
    h, w = layout.shape
    while 0 <= rr < h and 0 <= cc < w and layout[rr, cc] == BG:
        cells.append((int(rr), int(cc)))
        rr += dr
        cc += dc
    return cells


def _choose_distance(cells: list[tuple[int, int]], is_root: bool) -> int:
    min_d = 2 if is_root and len(cells) >= 2 else 1
    choices = list(range(min_d, len(cells) + 1))
    if not choices:
        return -1
    window = choices[: min(len(choices), 5)]
    weights = [len(window) - i for i in range(len(window))]
    return int(random.choices(window, weights=weights, k=1)[0])


def _build_output_network(target_markers: int, panel_size: int) -> np.ndarray | None:
    layout = np.zeros((panel_size, panel_size), dtype=int)
    layout[0, 0] = RED
    pending: list[tuple[int, int, int, int]] = [(0, 0, 0, 1)]
    seen_segments: set[tuple[int, int, int, int]] = set()
    markers = 0

    while pending:
        r, c, dr, dc = pending.pop()
        key = (int(r), int(c), int(dr), int(dc))
        if key in seen_segments:
            continue
        seen_segments.add(key)

        cells = _empty_run(layout, r, c, dr, dc)
        if not cells:
            continue

        remaining = target_markers - markers
        must_branch = remaining > len(pending)
        should_branch = remaining > 0 and (must_branch or random.random() < 0.72 or key == (0, 0, 0, 1))

        if should_branch:
            dist = _choose_distance(cells, is_root=(key == (0, 0, 0, 1)))
            if dist != -1:
                marker_r, marker_c = cells[dist - 1]
                prev_r, prev_c = (r, c) if dist == 1 else cells[dist - 2]
                for rr, cc in cells[: dist - 1]:
                    layout[rr, cc] = RED
                layout[marker_r, marker_c] = ORANGE
                markers += 1

                if dr == 0:
                    pending.append((prev_r, prev_c, -1, 0))
                    pending.append((prev_r, prev_c, 1, 0))
                else:
                    pending.append((prev_r, prev_c, 0, -1))
                    pending.append((prev_r, prev_c, 0, 1))
                continue

        for rr, cc in cells:
            layout[rr, cc] = RED

    if markers != target_markers:
        return None

    marker_mask = layout == ORANGE
    traced = _trace_network(marker_mask)
    if not np.array_equal(traced, layout):
        return None
    return layout


def _grow_common_mask(
    base_mask: np.ndarray,
    target_common: int,
    forbidden: np.ndarray,
    panel_size: int,
) -> np.ndarray | None:
    common = base_mask.copy()
    while int(common.sum()) < target_common:
        frontier = []
        for r in range(panel_size):
            for c in range(panel_size):
                if common[r, c] or forbidden[r, c]:
                    continue
                if any(0 <= rr < panel_size and 0 <= cc < panel_size and common[rr, cc] for rr, cc in _neighbors4(r, c)):
                    frontier.append((r, c))
        if not frontier:
            return None
        common[random.choice(frontier)] = True
    return common


def _assign_markers(marker_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = marker_coords.copy()
    np.random.shuffle(order)
    total = len(order)
    left_count = random.randint(1, total - 1)
    left = order[:left_count]
    right = order[left_count:]
    return left, right


def _render_input(left_mask: np.ndarray, right_mask: np.ndarray) -> np.ndarray:
    panel_size = int(left_mask.shape[0])
    grid = np.zeros((panel_size, 2 * panel_size + 1), dtype=int)
    grid[:, :panel_size][left_mask] = LEFT_FG
    grid[:, panel_size] = SEP
    grid[:, panel_size + 1 :][right_mask] = RIGHT_FG
    return grid


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    left_mask, right_mask = _parse_panels(inp)
    if not (_is_single_component(left_mask) and _is_single_component(right_mask)):
        return False

    left_cells = int(left_mask.sum())
    right_cells = int(right_mask.sum())
    if not (15 <= left_cells <= 25 and 15 <= right_cells <= 25):
        return False

    diff = np.logical_xor(left_mask, right_mask)
    diff_count = int(diff.sum())
    if not (4 <= diff_count <= 8):
        return False
    if np.array_equal(left_mask, right_mask):
        return False

    left_only = int(np.sum(left_mask & ~right_mask))
    right_only = int(np.sum(right_mask & ~left_mask))
    if left_only == 0 or right_only == 0:
        return False

    marker_rows = {int(r) for r, _ in np.argwhere(diff)}
    marker_cols = {int(c) for _, c in np.argwhere(diff)}
    if len(marker_rows) < 2 or len(marker_cols) < 2:
        return False
    if diff[0, 0]:
        return False

    if int(np.sum(out == ORANGE)) != diff_count:
        return False
    if int(np.sum(out == RED)) < 10:
        return False
    if not np.any(np.argwhere(out == RED)[:, 0] > 0):
        return False
    if not np.any(np.argwhere(out == RED)[:, 1] > 0):
        return False

    rows_with_red = np.flatnonzero(np.any(out == RED, axis=1))
    cols_with_red = np.flatnonzero(np.any(out == RED, axis=0))
    if rows_with_red.size < 2 or cols_with_red.size < 2:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    panel_size = random.choice(PANEL_SIZE_OPTIONS)
    target_markers = random.randint(4, 8)
    network = _build_output_network(target_markers, panel_size)
    if network is None:
        return None

    base_mask = network == RED
    marker_mask = network == ORANGE
    base_cells = int(base_mask.sum())
    if not (8 <= base_cells <= 20):
        return None

    marker_coords = np.argwhere(marker_mask)
    left_markers, right_markers = _assign_markers(marker_coords)
    common_lo = max(base_cells, 15 - len(left_markers), 15 - len(right_markers))
    common_hi = min(panel_size * panel_size - target_markers, 25 - len(left_markers), 25 - len(right_markers), 20)
    if common_lo > common_hi:
        return None
    common_target = random.randint(common_lo, common_hi)

    common_mask = _grow_common_mask(base_mask, common_target, marker_mask, panel_size)
    if common_mask is None:
        return None

    left_mask = common_mask.copy()
    right_mask = common_mask.copy()
    if left_markers.size:
        left_mask[left_markers[:, 0], left_markers[:, 1]] = True
    if right_markers.size:
        right_mask[right_markers[:, 0], right_markers[:, 1]] = True

    if not (_is_single_component(left_mask) and _is_single_component(right_mask)):
        return None

    inp = _render_input(left_mask, right_mask)
    out = main(inp)
    if not np.array_equal(out, network):
        return None
    if not _semantic_checks(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-difference ray-network input")
