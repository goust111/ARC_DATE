from __future__ import annotations

import random

import numpy as np


BG = 0
MAROON = 9
PAIR_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

FRAME_RANGE = (4, 6)
PAIR_COUNT_RANGE = (2, 4)
GRID_LIMIT = 30
GENERATION_RETRIES = 400
CORE_RETRIES = 150
LAYOUT_RETRIES = 80
DIFF_PATTERNS = [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2)]


def _neighbors(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(int(a1), int(b1)) - max(int(a0), int(b0)) + 1)


def _gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return int(b0 - a1 - 1)
    if b1 < a0:
        return int(a0 - b1 - 1)
    return 0


def _folded_cells(coords: np.ndarray, orientation: str, axis: int) -> set[tuple[int, int]]:
    if orientation == "stacked":
        return {
            (int(axis + abs(int(r) - axis) - 1), int(c))
            for r, c in coords
        }
    return {
        (int(r), int(axis + abs(int(c) - axis) - 1))
        for r, c in coords
    }


def _pair_orientation_and_axis(comp_a: np.ndarray, comp_b: np.ndarray) -> tuple[str, int]:
    a_box = _bbox(comp_a)
    b_box = _bbox(comp_b)
    row_gap = _gap(a_box[0], a_box[2], b_box[0], b_box[2])
    col_gap = _gap(a_box[1], a_box[3], b_box[1], b_box[3])
    row_overlap = _interval_overlap(a_box[0], a_box[2], b_box[0], b_box[2])
    col_overlap = _interval_overlap(a_box[1], a_box[3], b_box[1], b_box[3])

    if row_gap > 0 and col_overlap > 0 and row_gap >= col_gap:
        top, bottom = sorted((a_box, b_box), key=lambda box: box[0])
        return "stacked", int((top[2] + bottom[0]) // 2)
    if col_gap > 0 and row_overlap > 0 and col_gap >= row_gap:
        left, right = sorted((a_box, b_box), key=lambda box: box[1])
        return "side_by_side", int((left[3] + right[1]) // 2)

    mean_a = comp_a.mean(axis=0)
    mean_b = comp_b.mean(axis=0)
    if abs(float(mean_a[0] - mean_b[0])) >= abs(float(mean_a[1] - mean_b[1])):
        top, bottom = sorted((a_box, b_box), key=lambda box: box[0])
        return "stacked", int(round((top[2] + bottom[0]) / 2))
    left, right = sorted((a_box, b_box), key=lambda box: box[1])
    return "side_by_side", int(round((left[3] + right[1]) / 2))


def main(grid: np.ndarray) -> np.ndarray:
    """Fold each same-color fragment pair onto its symmetry axis and mark the non-overlapping folded pixels in maroon.

    Input structure:
    - The background is black.
    - Every non-background, non-maroon color appears as exactly two 4-connected fragments.
    - Each same-color pair is either stacked above/below one another around a horizontal axis or placed left/right around a vertical axis.
    - A single maroon edge pixel is present as a separate marker and does not belong to any fragment pair.

    Rule:
    - Ignore the lone maroon edge marker.
    - For each same-color pair, infer whether the pair is vertically separated or horizontally separated and locate the middle row/column between the two fragments.
    - Fold both fragments toward that axis, aligning equal distances from the axis onto one merged strip starting on the axis row/column.
    - Cells reached by both folded fragments keep the pair color; cells reached by only one folded fragment become maroon.
    - Clear the original fragment locations and keep only the merged folded result.

    Key insight:
    - The objects must first be paired by color, then interpreted as near-mirror fragments around a hidden axis, and finally overlaid so that overlap stays in the original color while disagreement turns into maroon.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    out = np.full((h, w), BG, dtype=int)

    maroon_cells = np.argwhere(g == MAROON)
    for r, c in maroon_cells:
        assert r in (0, h - 1) or c in (0, w - 1), "maroon input marker must lie on the edge"

    colors = sorted(int(color) for color in np.unique(g) if color not in (BG, MAROON))
    for color in colors:
        comps = _components(g == color)
        assert len(comps) == 2, f"color {color} must form exactly two components"

        orientation, axis = _pair_orientation_and_axis(comps[0], comps[1])
        folded_a = _folded_cells(comps[0], orientation, axis)
        folded_b = _folded_cells(comps[1], orientation, axis)

        both = folded_a & folded_b
        diff = folded_a ^ folded_b
        for r, c in both:
            assert 0 <= r < h and 0 <= c < w
            assert out[r, c] in (BG, color)
            out[r, c] = color
        for r, c in diff:
            assert 0 <= r < h and 0 <= c < w
            assert out[r, c] in (BG, MAROON)
            out[r, c] = MAROON

    return out


def _manhattan_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    cells = [(r, c)]
    while (r, c) != b:
        choices = []
        if r < b[0]:
            choices.append((1, 0))
        elif r > b[0]:
            choices.append((-1, 0))
        if c < b[1]:
            choices.append((0, 1))
        elif c > b[1]:
            choices.append((0, -1))
        dr, dc = random.choice(choices)
        r += dr
        c += dc
        cells.append((r, c))
    return cells


def _adjacent_background(cells: set[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    border = set()
    for r, c in cells:
        for nr, nc in _neighbors(r, c, h, w):
            if (nr, nc) not in cells:
                border.add((nr, nc))
    return list(border)


def _connected_mask_with_bbox(h: int, w: int, needed_extras: int) -> np.ndarray | None:
    for _ in range(CORE_RETRIES):
        hub = (random.randrange(h), random.randrange(w))
        anchors = [
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        ]
        cells = {hub}
        for anchor in anchors:
            cells.update(_manhattan_path(anchor, hub))

        max_area = h * w - needed_extras
        if len(cells) >= max_area:
            continue

        target = random.randint(len(cells), max(len(cells), min(max_area - 1, int(0.7 * h * w))))
        while len(cells) < target:
            frontier = _adjacent_background(cells, h, w)
            if not frontier:
                break
            cells.add(random.choice(frontier))

        frontier = _adjacent_background(cells, h, w)
        if len(frontier) < needed_extras:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        comps = _components(mask)
        if len(comps) != 1:
            continue
        if _bbox(comps[0]) != (0, 0, h - 1, w - 1):
            continue
        return mask
    return None


def _sample_pair_masks() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*FRAME_RANGE)
    w = random.randint(*FRAME_RANGE)
    extra_a, extra_b = random.choice(DIFF_PATTERNS)
    total_extras = extra_a + extra_b

    core = _connected_mask_with_bbox(h, w, total_extras)
    if core is None:
        return None

    frontier = _adjacent_background({tuple(rc) for rc in np.argwhere(core)}, h, w)
    if len(frontier) < total_extras:
        return None
    random.shuffle(frontier)

    mask_a = core.copy()
    mask_b = core.copy()
    for r, c in frontier[:extra_a]:
        mask_a[r, c] = True
    for r, c in frontier[extra_a : extra_a + extra_b]:
        mask_b[r, c] = True

    if len(_components(mask_a)) != 1 or len(_components(mask_b)) != 1:
        return None
    return mask_a, mask_b


def _make_pair_spec(color: int) -> dict[str, object] | None:
    masks = _sample_pair_masks()
    if masks is None:
        return None

    mask_a, mask_b = masks
    h, w = mask_a.shape
    orientation = random.choice(["side_by_side", "stacked"])
    block_h = h if orientation == "side_by_side" else 2 * h + 1
    block_w = 2 * w + 1 if orientation == "side_by_side" else w

    return {
        "color": int(color),
        "orientation": orientation,
        "mask_a": mask_a,
        "mask_b": mask_b,
        "frame_h": int(h),
        "frame_w": int(w),
        "block_h": int(block_h),
        "block_w": int(block_w),
    }


def _row_partition(count: int) -> list[int]:
    if count == 2:
        return [2] if random.random() < 0.5 else [1, 1]
    if count == 3:
        return [2, 1] if random.random() < 0.5 else [1, 2]
    return [2, 2]


def _layout_specs(specs: list[dict[str, object]]) -> tuple[int, int, list[tuple[dict[str, object], int, int]]] | None:
    shuffled = specs[:]
    random.shuffle(shuffled)
    row_sizes = _row_partition(len(shuffled))
    rows: list[list[dict[str, object]]] = []
    idx = 0
    for size in row_sizes:
        row = shuffled[idx : idx + size]
        random.shuffle(row)
        rows.append(row)
        idx += size

    positions: list[tuple[dict[str, object], int, int]] = []
    row_tops: list[int] = []
    row_heights: list[int] = []
    current_top = random.randint(1, 3)
    max_width = 0

    for row_index, row in enumerate(rows):
        row_tops.append(current_top)
        height = max(int(spec["block_h"]) for spec in row)
        row_heights.append(height)
        x = random.randint(1, 3)
        gap = random.randint(2, 4)
        for spec in row:
            positions.append((spec, current_top, x))
            x += int(spec["block_w"]) + gap
        max_width = max(max_width, x - gap + random.randint(1, 3))
        if row_index + 1 < len(rows):
            current_top += height + random.randint(2, 4)

    total_h = row_tops[-1] + row_heights[-1] + random.randint(1, 3)
    total_w = max_width
    if total_h > GRID_LIMIT or total_w > GRID_LIMIT:
        return None
    return total_h, total_w, positions


def _paint_input_pair(grid: np.ndarray, spec: dict[str, object], top: int, left: int) -> None:
    mask_a = np.asarray(spec["mask_a"], dtype=bool)
    mask_b = np.asarray(spec["mask_b"], dtype=bool)
    color = int(spec["color"])
    h, w = mask_a.shape

    if spec["orientation"] == "side_by_side":
        for r in range(h):
            for c in range(w):
                if mask_a[r, c]:
                    grid[top + r, left + (w - 1 - c)] = color
                if mask_b[r, c]:
                    grid[top + r, left + w + 1 + c] = color
    else:
        for r in range(h):
            for c in range(w):
                if mask_a[r, c]:
                    grid[top + (h - 1 - r), left + c] = color
                if mask_b[r, c]:
                    grid[top + h + 1 + r, left + c] = color


def _paint_output_pair(grid: np.ndarray, spec: dict[str, object], top: int, left: int) -> None:
    mask_a = np.asarray(spec["mask_a"], dtype=bool)
    mask_b = np.asarray(spec["mask_b"], dtype=bool)
    color = int(spec["color"])
    h, w = mask_a.shape

    if spec["orientation"] == "side_by_side":
        out_top = top
        out_left = left + w
    else:
        out_top = top + h
        out_left = left

    both = mask_a & mask_b
    diff = mask_a ^ mask_b
    for r in range(h):
        for c in range(w):
            if both[r, c]:
                grid[out_top + r, out_left + c] = color
            elif diff[r, c]:
                grid[out_top + r, out_left + c] = MAROON


def _place_marker(inp: np.ndarray) -> bool:
    h, w = inp.shape
    candidates = []
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if inp[r, c] != BG:
                continue
            ok = True
            for rr in range(max(0, r - 1), min(h, r + 2)):
                for cc in range(max(0, c - 1), min(w, c + 2)):
                    if inp[rr, cc] != BG:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                candidates.append((r, c))
    if not candidates:
        return False
    r, c = random.choice(candidates)
    inp[r, c] = MAROON
    return True


def _valid_input(inp: np.ndarray, pair_colors: list[int]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > GRID_LIMIT:
        return False
    if np.count_nonzero(inp == MAROON) != 1:
        return False
    marker = tuple(map(int, np.argwhere(inp == MAROON)[0]))
    if marker[0] not in (0, inp.shape[0] - 1) and marker[1] not in (0, inp.shape[1] - 1):
        return False
    for color in pair_colors:
        if len(_components(inp == color)) != 2:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample(PAIR_COLORS, pair_count)

    specs: list[dict[str, object]] = []
    for color in colors:
        spec = _make_pair_spec(color)
        if spec is None:
            return None
        specs.append(spec)

    for _ in range(LAYOUT_RETRIES):
        layout = _layout_specs(specs)
        if layout is None:
            continue
        h, w, positions = layout
        inp = np.full((h, w), BG, dtype=int)
        out = np.full((h, w), BG, dtype=int)

        for spec, top, left in positions:
            _paint_input_pair(inp, spec, top, left)
            _paint_output_pair(out, spec, top, left)

        if not _place_marker(inp):
            continue
        if not _valid_input(inp, colors):
            continue
        try:
            solved = main(inp)
        except Exception:
            continue
        if np.array_equal(solved, out):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid folded-fragment ARC input")
