from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.asarray(coords, dtype=int))

    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_solid_rect(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox_coords(coords)
    return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _inside_bbox(pos: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = pos
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _rect_outside(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3]


def _find_template(grid: np.ndarray, bg: int) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    candidates: list[dict[str, object]] = []

    for color in np.unique(g):
        line_color = int(color)
        if line_color == int(bg):
            continue

        for coords in _components_mask(g == line_color):
            r0, c0, r1, c1 = _bbox_coords(coords)
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            line_mask = sub == line_color

            row_idx = [r for r in range(line_mask.shape[0]) if bool(line_mask[r, :].all())]
            col_idx = [c for c in range(line_mask.shape[1]) if bool(line_mask[:, c].all())]
            if len(row_idx) < 3 or len(col_idx) < 3:
                continue
            if row_idx[0] != 0 or row_idx[-1] != line_mask.shape[0] - 1:
                continue
            if col_idx[0] != 0 or col_idx[-1] != line_mask.shape[1] - 1:
                continue
            if any(b - a <= 1 for a, b in zip(row_idx, row_idx[1:])):
                continue
            if any(b - a <= 1 for a, b in zip(col_idx, col_idx[1:])):
                continue

            expected = np.zeros_like(line_mask, dtype=bool)
            expected[row_idx, :] = True
            expected[:, col_idx] = True
            if not np.array_equal(line_mask, expected):
                continue

            other_fg = (sub != int(bg)) & (sub != int(line_color))
            if bool((other_fg & expected).any()):
                continue

            cell_heights = [row_idx[i + 1] - row_idx[i] - 1 for i in range(len(row_idx) - 1)]
            cell_widths = [col_idx[i + 1] - col_idx[i] - 1 for i in range(len(col_idx) - 1)]
            if len(set(cell_heights)) != 1 or len(set(cell_widths)) != 1:
                continue

            candidates.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "line_color": int(line_color),
                    "rows": [r0 + r for r in row_idx],
                    "cols": [c0 + c for c in col_idx],
                    "cell_h": int(cell_heights[0]),
                    "cell_w": int(cell_widths[0]),
                }
            )

    assert len(candidates) == 1, "expected one template lattice"
    return candidates[0]


def _parse_labels(grid: np.ndarray, bg: int, template: dict[str, object]) -> dict[int, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    line_color = int(template["line_color"])
    rows = template["rows"]
    cols = template["cols"]

    labels: dict[int, dict[str, object]] = {}
    for sr in range(len(rows) - 1):
        for sc in range(len(cols) - 1):
            r0 = int(rows[sr]) + 1
            c0 = int(cols[sc]) + 1
            r1 = int(rows[sr + 1]) - 1
            c1 = int(cols[sc + 1]) - 1
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            marker_mask = (region != int(bg)) & (region != int(line_color))
            if not bool(marker_mask.any()):
                continue

            marker_coords = np.argwhere(marker_mask)
            marker_colors = np.unique(region[marker_mask])
            assert len(marker_coords) == 1, "expected a one-pixel label"
            assert marker_colors.size == 1, "expected a single label color per slot"

            color = int(marker_colors[0])
            assert color not in labels, "label colors must be unique"
            rr, cc = marker_coords[0]
            labels[color] = {
                "slot": (sr, sc),
                "cell_bbox": (r0, c0, r1, c1),
                "label_pos": (r0 + int(rr), c0 + int(cc)),
            }

    assert labels, "expected at least one labeled slot"
    return labels


def _find_fill_blocks(
    grid: np.ndarray, template: dict[str, object], labels: dict[int, dict[str, object]]
) -> dict[int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    template_bbox = template["bbox"]
    blocks: dict[int, np.ndarray] = {}

    for color, info in labels.items():
        comps = _components_mask(g == int(color))
        label_comp: np.ndarray | None = None
        outer_comps: list[np.ndarray] = []
        label_pos = info["label_pos"]

        for comp in comps:
            if np.any((comp[:, 0] == label_pos[0]) & (comp[:, 1] == label_pos[1])):
                label_comp = comp
            else:
                outer_comps.append(comp)

        assert label_comp is not None and len(label_comp) == 1
        assert len(outer_comps) == 1, "expected one external fill block per label color"

        block = outer_comps[0]
        assert _is_solid_rect(block), "fill blocks must be solid rectangles"
        br0, bc0, br1, bc1 = _bbox_coords(block)
        assert _rect_outside((br0, bc0, br1, bc1), template_bbox)

        r0, c0, r1, c1 = info["cell_bbox"]
        assert br1 - br0 == r1 - r0
        assert bc1 - bc0 == c1 - c0
        blocks[int(color)] = block

    return blocks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each external fill block into the template slot marked by the same color.

    Input structure
    - A grey rectangular lattice divides the grid into equally sized slots.
    - Some slots contain a single-pixel color label; unlabeled slots are empty.
    - Outside the lattice, each label color also appears as one solid rectangle whose
      size matches a slot interior.

    Rule
    - Use the label color to match each slot with the external block of the same color.
    - Remove the external block from its original position.
    - Fill the entire slot interior with that color, keeping the lattice lines unchanged.
    - Unlabeled slots remain background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    template = _find_template(g, bg)
    labels = _parse_labels(g, bg, template)
    blocks = _find_fill_blocks(g, template, labels)

    out = g.copy()
    for block in blocks.values():
        out[block[:, 0], block[:, 1]] = int(bg)

    for color, info in labels.items():
        r0, c0, r1, c1 = info["cell_bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)

    return out


_GRID_ROWS_RANGE = (2, 5)
_GRID_COLS_RANGE = (2, 5)
_CELL_SIZE_RANGE = (2, 4)
_MAX_LABELS = 7
_CANVAS_EXTRA_RANGE = (4, 10)
_MAX_GENERATION_ATTEMPTS = 300
_SIDES = ("top", "bottom", "left", "right")


def _cell_bbox(
    top: int, left: int, cell_h: int, cell_w: int, slot_r: int, slot_c: int
) -> tuple[int, int, int, int]:
    r0 = top + 1 + slot_r * (cell_h + 1)
    c0 = left + 1 + slot_c * (cell_w + 1)
    return r0, c0, r0 + cell_h - 1, c0 + cell_w - 1


def _mark_reserved(mask: np.ndarray, r0: int, c0: int, height: int, width: int) -> None:
    h, w = mask.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r0 + height + 1)
    cc1 = min(w, c0 + width + 1)
    mask[rr0:rr1, cc0:cc1] = True


def _candidate_side(
    r0: int, c0: int, height: int, width: int, template_bbox: tuple[int, int, int, int]
) -> str | None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    tr0, tc0, tr1, tc1 = template_bbox
    if r1 < tr0:
        return "top"
    if r0 > tr1:
        return "bottom"
    if c1 < tc0:
        return "left"
    if c0 > tc1:
        return "right"
    return None


def _block_candidates(
    reserved: np.ndarray,
    template_bbox: tuple[int, int, int, int],
    block_h: int,
    block_w: int,
) -> dict[str, list[tuple[int, int]]]:
    h, w = reserved.shape
    groups = {side: [] for side in _SIDES}
    for r0 in range(1, h - block_h):
        for c0 in range(1, w - block_w):
            if bool(reserved[r0 : r0 + block_h, c0 : c0 + block_w].any()):
                continue
            side = _candidate_side(r0, c0, block_h, block_w, template_bbox)
            if side is not None:
                groups[side].append((r0, c0))
    return groups


def _expected_output(
    grid: np.ndarray,
    bg: int,
    labels: dict[int, dict[str, object]],
    block_positions: dict[int, tuple[int, int]],
    cell_h: int,
    cell_w: int,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)

    for r0, c0 in block_positions.values():
        out[r0 : r0 + cell_h, c0 : c0 + cell_w] = int(bg)

    for color, info in labels.items():
        r0, c0, r1, c1 = info["cell_bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)

    return out


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*_GRID_ROWS_RANGE)
    cols = random.randint(*_GRID_COLS_RANGE)
    cell_h = random.randint(*_CELL_SIZE_RANGE)
    cell_w = random.randint(*_CELL_SIZE_RANGE)

    template_h = rows * cell_h + rows + 1
    template_w = cols * cell_w + cols + 1

    base_h = template_h + 2 * cell_h + 8
    base_w = template_w + 2 * cell_w + 8
    height = base_h + random.randint(*_CANVAS_EXTRA_RANGE)
    width = base_w + random.randint(*_CANVAS_EXTRA_RANGE)

    top = random.randint(cell_h + 2, height - template_h - cell_h - 2)
    left = random.randint(cell_w + 2, width - template_w - cell_w - 2)
    template_bbox = (top, left, top + template_h - 1, left + template_w - 1)

    colors = list(range(10))
    bg = random.choice(colors)
    colors.remove(bg)
    line_color = random.choice(colors)
    colors.remove(line_color)

    total_slots = rows * cols
    max_labels = min(total_slots - 1, len(colors), _MAX_LABELS)
    if max_labels < 1:
        return None
    label_count = random.randint(1, max_labels)

    label_colors = random.sample(colors, label_count)
    slot_ids = random.sample(range(total_slots), label_count)

    grid = np.full((height, width), int(bg), dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    grid[top : top + template_h, left] = int(line_color)
    grid[top : top + template_h, left + template_w - 1] = int(line_color)
    grid[top, left : left + template_w] = int(line_color)
    grid[top + template_h - 1, left : left + template_w] = int(line_color)
    for i in range(1, rows):
        rr = top + i * (cell_h + 1)
        grid[rr, left : left + template_w] = int(line_color)
    for j in range(1, cols):
        cc = left + j * (cell_w + 1)
        grid[top : top + template_h, cc] = int(line_color)

    _mark_reserved(reserved, top, left, template_h, template_w)

    labels: dict[int, dict[str, object]] = {}
    for color, slot_id in zip(label_colors, slot_ids):
        slot_r, slot_c = divmod(slot_id, cols)
        r0, c0, r1, c1 = _cell_bbox(top, left, cell_h, cell_w, slot_r, slot_c)
        label_r = random.randint(r0, r1)
        label_c = random.randint(c0, c1)
        grid[label_r, label_c] = int(color)
        labels[int(color)] = {
            "slot": (slot_r, slot_c),
            "cell_bbox": (r0, c0, r1, c1),
            "label_pos": (label_r, label_c),
        }

    block_positions: dict[int, tuple[int, int]] = {}
    side_counts = {side: 0 for side in _SIDES}
    for color in random.sample(label_colors, len(label_colors)):
        groups = _block_candidates(reserved, template_bbox, cell_h, cell_w)
        available_sides = [side for side in _SIDES if groups[side]]
        if not available_sides:
            return None

        weights = [1.0 / (1 + side_counts[side]) for side in available_sides]
        side = random.choices(available_sides, weights=weights, k=1)[0]
        r0, c0 = random.choice(groups[side])
        grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = int(color)
        block_positions[int(color)] = (r0, c0)
        side_counts[side] += 1
        _mark_reserved(reserved, r0, c0, cell_h, cell_w)

    expected = _expected_output(grid, bg, labels, block_positions, cell_h, cell_w)
    if np.array_equal(expected, grid):
        return None
    if int(np.count_nonzero(grid != int(bg))) >= grid.size // 2:
        return None
    if _get_bg(grid) != int(bg):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid template-fill puzzle input")
