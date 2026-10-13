from __future__ import annotations

from collections import Counter
import random

import numpy as np


ARC_MAX = 30
GEN_ATTEMPTS = 400
PATTERN_ATTEMPTS = 200
DECOR_ATTEMPTS = 120
GRID_LAYOUTS = ((2, 2), (2, 3), (3, 2))
CARDINALS = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAGONALS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
CORNERS = ((0, 0), (0, -1), (-1, 0), (-1, -1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _non_separator_blocks(mask: np.ndarray) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_sep in enumerate(mask.tolist()):
        if not is_sep and start is None:
            start = idx
        elif is_sep and start is not None:
            blocks.append((start, idx))
            start = None
    if start is not None:
        blocks.append((start, len(mask)))
    return blocks


def _extract_boxes(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_blocks = _non_separator_blocks(np.all(grid == bg, axis=1))
    col_blocks = _non_separator_blocks(np.all(grid == bg, axis=0))
    assert row_blocks and col_blocks, "expected a separator grid"
    heights = {r1 - r0 for r0, r1 in row_blocks}
    widths = {c1 - c0 for c0, c1 in col_blocks}
    assert len(heights) == 1 and len(widths) == 1, "boxes must be equal-sized"
    return row_blocks, col_blocks


def _box_counts(box: np.ndarray, bg: int) -> Counter[int]:
    vals = [int(v) for v in box.ravel() if int(v) != int(bg)]
    return Counter(vals)


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in CARDINALS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _minority_corner_candidate(box: np.ndarray, bg: int) -> tuple[int, int] | None:
    counts = _box_counts(box, bg)
    if len(counts) != 2:
        return None
    skeleton, decor = sorted(counts, key=lambda color: counts[color], reverse=True)
    if counts[skeleton] <= counts[decor]:
        return None
    h, w = box.shape
    for rr, cc in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        if int(box[rr, cc]) == int(decor):
            return skeleton, decor
    return None


def _seed_and_active_mask(grid: np.ndarray, bg: int) -> tuple[tuple[int, int], int, np.ndarray]:
    row_blocks, col_blocks = _extract_boxes(grid, bg)
    candidates: list[tuple[int, int, int, int]] = []
    fallback: list[tuple[int, int, int, int]] = []
    for rb, (r0, r1) in enumerate(row_blocks):
        for cb, (c0, c1) in enumerate(col_blocks):
            box = grid[r0:r1, c0:c1]
            counts = _box_counts(box, bg)
            if len(counts) != 2:
                continue
            picked = _minority_corner_candidate(box, bg)
            if picked is not None:
                skeleton, decor = picked
                candidates.append((rb, cb, skeleton, decor))
            else:
                skeleton, decor = sorted(counts, key=lambda color: counts[color], reverse=True)
                h, w = box.shape
                if any(int(box[rr, cc]) != int(bg) for rr, cc in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))):
                    fallback.append((rb, cb, skeleton, decor))

    seed_list = candidates if len(candidates) == 1 else fallback
    assert len(seed_list) == 1, "expected a unique seed box"
    seed_r, seed_c, skeleton, decor = seed_list[0]
    sr0, sr1 = row_blocks[seed_r]
    sc0, sc1 = col_blocks[seed_c]
    seed_box = grid[sr0:sr1, sc0:sc1]

    k = int(np.sum(seed_box == decor))
    comps = _connected_components(seed_box == skeleton)
    assert comps, "expected skeleton components in the seed box"
    comps.sort(key=lambda comp: (-len(comp), min(pt[0] for pt in comp), min(pt[1] for pt in comp)))

    active = np.zeros_like(seed_box, dtype=bool)
    for comp in comps[: min(k, len(comps))]:
        for r, c in comp:
            active[r, c] = True
    return (seed_r, seed_c), int(skeleton), active


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the top-K largest skeleton regions in each non-seed box and recolor them with that box's decoration color.

    Input structure:
    - The most common color is the background/separator color, and full separator
      lines split the grid into equal rectangular boxes.
    - Every box contains the same skeleton drawing in one non-background color,
      plus a second non-background decoration color at fixed interior positions.
    - Exactly one seed box also has a single corner marker in its decoration
      color, making it the only box whose minority color appears in a corner.

    Rule:
    - In the seed box, find the 4-connected components of the skeleton color.
    - Let K be the number of decoration-color pixels in the seed box.
    - Sort the skeleton components by area descending and mark the top K as active.
    - Leave the seed box unchanged; in every other box, recolor skeleton pixels in
      active components to that box's decoration color and erase the remaining
      skeleton pixels to background. Decoration pixels and separator lines stay unchanged.

    Key insight:
    - The seed box's decoration pixels are not the regions to keep; their count is
      only a counter that selects how many of the largest skeleton components survive.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    row_blocks, col_blocks = _extract_boxes(g, bg)
    (seed_r, seed_c), skeleton, active = _seed_and_active_mask(g, bg)

    out = g.copy()
    for rb, (r0, r1) in enumerate(row_blocks):
        for cb, (c0, c1) in enumerate(col_blocks):
            if (rb, cb) == (seed_r, seed_c):
                continue
            box = g[r0:r1, c0:c1]
            counts = _box_counts(box, bg)
            assert len(counts) == 2, "each box must have skeleton and decoration colors"
            decor_colors = [color for color in counts if int(color) != int(skeleton)]
            assert len(decor_colors) == 1, "expected exactly one decoration color per box"
            decor = int(decor_colors[0])
            for rr in range(box.shape[0]):
                for cc in range(box.shape[1]):
                    if int(box[rr, cc]) != int(skeleton):
                        continue
                    out[r0 + rr, c0 + cc] = decor if active[rr, cc] else bg
    return out


def _shape_cells(spec: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, h, w = spec
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _touches_existing(existing: set[tuple[int, int]], cells: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for dr, dc in DIAGONALS:
            if (r + dr, c + dc) in existing:
                return True
    return False


def _rows_cols_covered(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    return len(rows) == h and len(cols) == w


def _safe_removals(comp: set[tuple[int, int]]) -> list[tuple[int, int]]:
    safe: list[tuple[int, int]] = []
    for cell in comp:
        remaining = comp - {cell}
        if not remaining:
            continue
        rs = [r for r, _ in remaining]
        cs = [c for _, c in remaining]
        mask = np.zeros((max(rs) - min(rs) + 1, max(cs) - min(cs) + 1), dtype=bool)
        for r, c in remaining:
            mask[r - min(rs), c - min(cs)] = True
        if len(_connected_components(mask)) == 1:
            safe.append(cell)
    return safe


def _sample_component_specs(h: int, w: int, use_center: bool) -> list[tuple[int, int, int, int]] | None:
    specs: list[tuple[int, int, int, int]] = []

    h1 = random.randint(1, 2)
    w1 = random.randint(2, 3)
    c1_hi = min(2, w - w1 - 4)
    if c1_hi < 1:
        return None
    specs.append((0, random.randint(1, c1_hi), h1, w1))

    h2_hi = min(4, h - 3)
    if h2_hi < 3:
        return None
    h2 = random.randint(3, h2_hi)
    w2 = random.randint(1, 2)
    r2_hi = min(2, h - h2 - 2)
    if r2_hi < 1:
        return None
    specs.append((random.randint(1, r2_hi), w - w2, h2, w2))

    h3 = random.randint(2, min(3, h - 2))
    w3 = random.randint(1, 2)
    r3_lo = max(2, h - h3 - 3)
    r3_hi = h - h3 - 1
    if r3_lo > r3_hi:
        return None
    specs.append((random.randint(r3_lo, r3_hi), 0, h3, w3))

    h4 = random.randint(1, 2)
    w4 = random.randint(3, min(4, w - 2))
    c4_lo = max(2, w // 2 - 1)
    c4_hi = w - w4 - 1
    if c4_lo > c4_hi:
        return None
    specs.append((h - h4, random.randint(c4_lo, c4_hi), h4, w4))

    if use_center:
        h5_hi = min(3, h - 4)
        w5_hi = min(3, w - 4)
        if h5_hi < 2 or w5_hi < 2:
            return None
        h5 = random.randint(2, h5_hi)
        w5 = random.randint(2, w5_hi)
        r5_lo = 2
        r5_hi = h - h5 - 2
        c5_lo = 2
        c5_hi = w - w5 - 2
        if r5_lo > r5_hi or c5_lo > c5_hi:
            return None
        specs.append((random.randint(r5_lo, r5_hi), random.randint(c5_lo, c5_hi), h5, w5))

    return specs


def _build_pattern(h: int, w: int) -> tuple[list[set[tuple[int, int]]], list[tuple[int, int]]] | None:
    for _ in range(PATTERN_ATTEMPTS):
        use_center = random.random() < 0.45
        specs = _sample_component_specs(h, w, use_center)
        if specs is None:
            continue

        comps: list[set[tuple[int, int]]] = []
        occupied: set[tuple[int, int]] = set()
        ok = True
        for spec in specs:
            cells = _shape_cells(spec)
            if _touches_existing(occupied, cells):
                ok = False
                break
            comps.append(cells)
            occupied |= cells
        if not ok:
            continue
        if any(corner in occupied for corner in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))):
            continue
        if not _rows_cols_covered(occupied, h, w):
            continue

        safe_by_comp = [_safe_removals(comp) for comp in comps]
        if any(not safe for safe in safe_by_comp):
            continue

        n = len(comps)
        for _ in range(DECOR_ATTEMPTS):
            k = random.randint(2, n)
            chosen = random.sample(range(n), k - 1)
            areas = [len(comps[idx]) - (1 if idx in chosen else 0) for idx in range(n)]
            if len(set(areas)) != n:
                continue
            removals = [random.choice(safe_by_comp[idx]) for idx in chosen]
            return comps, removals
    return None


def _paint_box(
    grid: np.ndarray,
    top: int,
    left: int,
    h: int,
    w: int,
    skeleton_color: int,
    decor_color: int,
    comps: list[set[tuple[int, int]]],
    decor_cells: set[tuple[int, int]],
    marker_corner: tuple[int, int] | None,
) -> None:
    for comp in comps:
        for r, c in comp:
            grid[top + r, left + c] = skeleton_color
    for r, c in decor_cells:
        grid[top + r, left + c] = decor_color
    if marker_corner is not None:
        mr, mc = marker_corner
        grid[top + mr, left + mc] = decor_color


def _expected_output(
    inp: np.ndarray,
    rows: int,
    cols: int,
    h: int,
    w: int,
    seed_pos: tuple[int, int],
    skeleton_color: int,
    decor_colors: list[int],
    active_mask: np.ndarray,
) -> np.ndarray:
    out = inp.copy()
    for rb in range(rows):
        for cb in range(cols):
            if (rb, cb) == seed_pos:
                continue
            top = 1 + rb * (h + 1)
            left = 1 + cb * (w + 1)
            box = inp[top : top + h, left : left + w]
            decor = decor_colors[rb * cols + cb]
            for rr in range(h):
                for cc in range(w):
                    if int(box[rr, cc]) != int(skeleton_color):
                        continue
                    out[top + rr, left + cc] = decor if active_mask[rr, cc] else _most_common_color(inp)
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    rows, cols = random.choice(GRID_LAYOUTS)
    cell_h = random.randint(7, min(8, (ARC_MAX - (rows + 1)) // rows))
    cell_w = random.randint(7, min(8, (ARC_MAX - (cols + 1)) // cols))

    pattern = _build_pattern(cell_h, cell_w)
    if pattern is None:
        return None
    comps, decor_list = pattern
    decor_cells = set(decor_list)

    palette = list(range(10))
    bg = random.choice(palette)
    remaining = [color for color in palette if color != bg]
    skeleton_color = random.choice(remaining)
    remaining = [color for color in remaining if color != skeleton_color]

    n_boxes = rows * cols
    if len(remaining) < n_boxes:
        return None
    decor_colors = random.sample(remaining, n_boxes)
    seed_pos = (random.randrange(rows), random.randrange(cols))
    marker_corner = random.choice(
        [(0, 0), (0, cell_w - 1), (cell_h - 1, 0), (cell_h - 1, cell_w - 1)]
    )

    total_h = rows * cell_h + rows + 1
    total_w = cols * cell_w + cols + 1
    inp = np.full((total_h, total_w), bg, dtype=int)

    for rb in range(rows):
        for cb in range(cols):
            top = 1 + rb * (cell_h + 1)
            left = 1 + cb * (cell_w + 1)
            decor_color = decor_colors[rb * cols + cb]
            corner = marker_corner if (rb, cb) == seed_pos else None
            _paint_box(
                inp,
                top,
                left,
                cell_h,
                cell_w,
                skeleton_color,
                decor_color,
                comps,
                decor_cells,
                corner,
            )

    if _most_common_color(inp) != bg:
        return None

    seed_decor = decor_colors[seed_pos[0] * cols + seed_pos[1]]
    seed_box = inp[
        1 + seed_pos[0] * (cell_h + 1) : 1 + seed_pos[0] * (cell_h + 1) + cell_h,
        1 + seed_pos[1] * (cell_w + 1) : 1 + seed_pos[1] * (cell_w + 1) + cell_w,
    ]
    k = int(np.sum(seed_box == seed_decor))
    areas_and_comps: list[tuple[int, set[tuple[int, int]]]] = []
    for comp in comps:
        reduced = comp - decor_cells
        areas_and_comps.append((len(reduced), reduced))
    if any(len(reduced) == 0 for _, reduced in areas_and_comps):
        return None
    if len({area for area, _ in areas_and_comps}) != len(areas_and_comps):
        return None

    areas_and_comps.sort(key=lambda item: (-item[0], min(r for r, _ in item[1]), min(c for _, c in item[1])))
    active_mask = np.zeros((cell_h, cell_w), dtype=bool)
    for _, comp in areas_and_comps[: min(k, len(areas_and_comps))]:
        for r, c in comp:
            active_mask[r, c] = True

    out = inp.copy()
    for rb in range(rows):
        for cb in range(cols):
            if (rb, cb) == seed_pos:
                continue
            top = 1 + rb * (cell_h + 1)
            left = 1 + cb * (cell_w + 1)
            decor_color = decor_colors[rb * cols + cb]
            box = inp[top : top + cell_h, left : left + cell_w]
            for rr in range(cell_h):
                for cc in range(cell_w):
                    if int(box[rr, cc]) != int(skeleton_color):
                        continue
                    out[top + rr, left + cc] = decor_color if active_mask[rr, cc] else bg

    if np.array_equal(inp, out):
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input")
