from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GREY = 5
PINK = 6
PALETTE_COLORS = (1, 2, 3, 4, 7, 8, 9)

GEN_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    total = int(mask.sum())
    if total == 0:
        return False
    return len(_components(mask)) == 1


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected non-empty values"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique majority color"
    return int(ranked[0][0])


def _mask_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    box = _bbox((grid == GREY) | (grid == PINK))
    assert box is not None, "expected a lower mask made of grey cells and pink markers"
    return box


def _top_blocks(grid: np.ndarray, mask_top: int) -> list[np.ndarray]:
    upper = grid[:mask_top]
    comps = _components(upper != BG)
    assert comps, "expected lattice cells above the mask"

    boxes: list[tuple[int, int, int, int]] = []
    for comp in comps:
        r0, c0 = comp.min(axis=0)
        r1, c1 = comp.max(axis=0)
        box = (int(r0), int(c0), int(r1), int(c1))
        patch = _crop(upper, box)
        assert np.all(patch != BG), "expected each lattice cell bbox to be fully filled"
        boxes.append(box)

    heights = {r1 - r0 + 1 for r0, _c0, r1, _c1 in boxes}
    widths = {c1 - c0 + 1 for _r0, c0, _r1, c1 in boxes}
    row_tops = {r0 for r0, _c0, _r1, _c1 in boxes}
    row_bottoms = {r1 for _r0, _c0, r1, _c1 in boxes}
    assert len(heights) == 1 and len(widths) == 1, "expected same-sized lattice cells"
    assert len(row_tops) == 1 and len(row_bottoms) == 1, "expected a single row of cells"

    blocks = [_crop(upper, box) for box in sorted(boxes, key=lambda b: b[1])]
    return blocks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the noisy upper lattice into a palette strip, then repaint the lower mask column-by-column and erase its markers.

    Input structure:
    - The background is black.
    - The upper area contains one row of same-sized non-black lattice cells separated by full black rows and columns; each cell is a solid rectangle with a unique majority color plus minority noise.
    - The lower area contains a grey mask with black holes and 1-3 pink single-pixel markers inside its bounding box.

    Rule:
    - Find the lower mask bounding box and crop to it for the output frame.
    - Read the upper lattice cells from left to right and take the unique majority color of each cell to form a palette strip whose length matches the crop width.
    - For each crop column, recolor every grey pixel with that column's palette color, keep black holes black, and erase every pink marker to black.

    Key insight:
    - The palette is hidden inside the noisy lattice: only after majority-voting each upper cell do the crop columns receive their final colors, while pink cells are a separate erase layer rather than part of the recolored mask.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mask_box = _mask_bbox(g)
    r0, c0, r1, c1 = mask_box
    crop = _crop(g, mask_box)
    blocks = _top_blocks(g, r0)
    assert len(blocks) == crop.shape[1], "expected one upper cell per output column"

    palette = np.array([_unique_mode(block) for block in blocks], dtype=int)
    out = np.zeros_like(crop)
    grey_mask = crop == GREY
    for col, color in enumerate(palette):
        out[grey_mask[:, col], col] = int(color)
    out[crop == PINK] = BG
    return out


def _palette_sequence(n_cols: int) -> list[int]:
    distinct = random.randint(3, min(5, n_cols))
    pool = random.sample(PALETTE_COLORS, distinct)
    seq = pool[:]
    while len(seq) < n_cols:
        seq.append(random.choice(pool))
    random.shuffle(seq)
    if len(set(seq)) < 3:
        return _palette_sequence(n_cols)
    return [int(x) for x in seq]


def _make_noisy_block(height: int, width: int, majority: int) -> np.ndarray:
    area = height * width
    max_noise = min(2, (area - 1) // 2)
    noise_count = random.randint(1, max_noise)
    block = np.full((height, width), int(majority), dtype=int)
    choices = random.sample(range(area), noise_count)
    other_colors = [c for c in PALETTE_COLORS if c != majority]
    for idx in choices:
        block[idx // width, idx % width] = random.choice(other_colors)
    assert _unique_mode(block) == int(majority)
    assert np.any(block != int(majority))
    return block


def _grow_mask(height: int, width: int, target_area: int) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    steps = ["D"] * (height - 1) + ["R"] * (width - 1)
    random.shuffle(steps)
    r = c = 0
    mask[r, c] = True
    for step in steps:
        if step == "D":
            r += 1
        else:
            c += 1
        mask[r, c] = True

    frontier: set[tuple[int, int]] = set()

    def add_frontier(rr: int, cc: int) -> None:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add((nr, nc))

    for rr, cc in np.argwhere(mask):
        add_frontier(int(rr), int(cc))

    while int(mask.sum()) < target_area:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        rr, cc = cell
        mask[rr, cc] = True
        add_frontier(rr, cc)

    if not _is_connected(mask):
        return None
    return mask


def _make_mask(height: int, width: int) -> np.ndarray | None:
    area = height * width
    min_area = max(height + width - 1, int(np.ceil(area * 0.52)))
    max_area = int(np.floor(area * 0.74))
    if min_area > max_area:
        return None
    for _ in range(120):
        target = random.randint(min_area, max_area)
        mask = _grow_mask(height, width, target)
        if mask is None:
            continue
        if not np.all(mask.any(axis=0)):
            continue
        if not np.all(mask.any(axis=1)):
            continue
        if int((~mask).sum()) < max(3, width // 2):
            continue
        if height >= 3 and width >= 3 and not np.any(~mask[1:-1, 1:-1]):
            continue
        return mask
    return None


def _marker_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            trial = mask.copy()
            trial[r, c] = False
            if not np.all(trial.any(axis=0)):
                continue
            if not np.all(trial.any(axis=1)):
                continue
            if not _is_connected(trial):
                continue
            out.append((r, c))
    return out


def _pick_markers(mask: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    grey = mask.copy()
    markers: list[tuple[int, int]] = []
    target_count = random.randint(1, 3)
    for _ in range(target_count):
        candidates = [
            cell
            for cell in _marker_candidates(grey)
            if all(abs(cell[0] - rr) + abs(cell[1] - cc) >= 2 for rr, cc in markers)
        ]
        if not candidates:
            break
        cell = random.choice(candidates)
        grey[cell] = False
        markers.append(cell)
    if not markers:
        return None
    if not _is_connected(grey):
        return None
    if not np.all(grey.any(axis=0)):
        return None
    return grey, markers


def _render_input(
    n_cols: int,
    cell_h: int,
    cell_w: int,
    palette: list[int],
    grey_mask: np.ndarray,
    markers: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    lattice_h = cell_h + 2
    lattice_w = n_cols * cell_w + (n_cols + 1)
    mask_h = grey_mask.shape[0]
    gap = random.randint(2, 5)
    base_h = lattice_h + gap + mask_h
    base_w = max(lattice_w, n_cols)
    total_h = base_h + random.randint(0, min(4, 30 - base_h))
    total_w = base_w + random.randint(0, min(5, 30 - base_w))
    top_margin = random.randint(0, total_h - base_h)
    bottom_start = top_margin + lattice_h + gap

    grid = np.zeros((total_h, total_w), dtype=int)

    lattice_c0 = random.randint(0, total_w - lattice_w)
    for idx, color in enumerate(palette):
        block = _make_noisy_block(cell_h, cell_w, int(color))
        r0 = top_margin + 1
        c0 = lattice_c0 + 1 + idx * (cell_w + 1)
        grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = block

    mask_c0 = random.randint(0, total_w - n_cols)
    input_crop = np.zeros((mask_h, n_cols), dtype=int)
    input_crop[grey_mask] = GREY
    for rr, cc in markers:
        input_crop[rr, cc] = PINK
    grid[bottom_start : bottom_start + mask_h, mask_c0 : mask_c0 + n_cols] = input_crop

    oracle = np.zeros_like(input_crop)
    for cc, color in enumerate(palette):
        oracle[grey_mask[:, cc], cc] = int(color)
    return grid, oracle


def _validate_example(
    grid: np.ndarray,
    oracle: np.ndarray,
    palette: list[int],
    markers: list[tuple[int, int]],
) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(out, oracle):
        return False
    if not np.all(np.any(out != BG, axis=0)):
        return False
    if len(set(map(int, palette))) < 3:
        return False
    box = _mask_bbox(grid)
    crop = _crop(grid, box)
    if crop.shape[1] != len(palette):
        return False
    for rr, cc in markers:
        if crop[rr, cc] != PINK or out[rr, cc] != BG:
            return False
        col_color = palette[cc]
        if not np.any(out[:, cc] == int(col_color)):
            return False
    upper_blocks = _top_blocks(grid, box[0])
    if [_unique_mode(block) for block in upper_blocks] != [int(x) for x in palette]:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_cols = random.randint(4, 7)
    cell_h = random.randint(2, 3)
    cell_w = random.randint(2, 3)
    mask_h = random.randint(6, 10)
    palette = _palette_sequence(n_cols)

    base_mask = _make_mask(mask_h, n_cols)
    if base_mask is None:
        return None
    marker_data = _pick_markers(base_mask)
    if marker_data is None:
        return None
    grey_mask, markers = marker_data

    grid, oracle = _render_input(n_cols, cell_h, cell_w, palette, grey_mask, markers)
    if not _validate_example(grid, oracle, palette, markers):
        return None
    if max(grid.shape) > 30 or max(oracle.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
