from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400
LEGEND_ENTRY_RANGE = (2, 4)
FRAME_COUNT_RANGE = (2, 4)
GRID_H_RANGE = (14, 30)
GRID_W_RANGE = (18, 30)
LEGEND_FRAME_GAP_RANGE = (2, 4)
FRAME_MARGIN_RANGE = (1, 2)


HOLE_LIBRARY = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 0], [0, 1]], dtype=bool),
    np.array([[0, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _flood(mask: np.ndarray, seeds: list[tuple[int, int]]) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(mask, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r, c in seeds:
        if mask[r, c] and not seen[r, c]:
            seen[r, c] = True
            q.append((r, c))
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty shape"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    tight = _tight_mask(mask)
    return (tuple(map(int, tight.shape)), tight.astype(np.uint8).tobytes())


def _boundary_background_holes(tile: np.ndarray, bg: int) -> np.ndarray:
    bg_mask = tile == bg
    h, w = tile.shape
    seeds: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                if bg_mask[r, c]:
                    seeds.append((r, c))
    outside = _flood(bg_mask, seeds)
    return bg_mask & (~outside)


def _frame_regions(tile: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = tile.shape
    assert h >= 5 and w >= 5, "frame bbox must be at least 5x5"
    bg_mask = tile == bg
    seeds: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if r in (1, h - 2) or c in (1, w - 2):
                if bg_mask[r, c]:
                    seeds.append((r, c))
    fillable = _flood(bg_mask, seeds)
    holes = bg_mask & (~fillable)
    return fillable, holes


def _legend_column_end(grid: np.ndarray, bg: int) -> int:
    non_bg_cols = [int(c) for c in np.flatnonzero(np.any(grid != bg, axis=0))]
    assert non_bg_cols, "expected non-background content"
    legend_end = non_bg_cols[-1]
    for left, right in zip(non_bg_cols, non_bg_cols[1:]):
        if right - left >= 3:
            legend_end = int(left)
            break
    return legend_end


def _row_groups(active_rows: list[int]) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    if not active_rows:
        return groups
    start = active_rows[0]
    prev = active_rows[0]
    for row in active_rows[1:]:
        if row != prev + 1:
            groups.append((start, prev))
            start = row
        prev = row
    groups.append((start, prev))
    return groups


def _legend_mapping(grid: np.ndarray, bg: int, grey: int) -> tuple[int, dict[tuple[tuple[int, int], bytes], int]]:
    legend_end = _legend_column_end(grid, bg)
    legend = grid[:, : legend_end + 1]
    active_rows = [int(r) for r in np.flatnonzero(np.any(legend != bg, axis=1))]
    groups = _row_groups(active_rows)
    mapping: dict[tuple[tuple[int, int], bytes], int] = {}
    for r0, r1 in groups:
        panel = legend[r0 : r1 + 1]
        grey_coords = np.argwhere(panel == grey)
        assert grey_coords.size > 0, "every legend entry must include grey pixels"
        sr0, sc0 = grey_coords.min(axis=0)
        sr1, sc1 = grey_coords.max(axis=0)
        shape_box = panel[sr0 : sr1 + 1, sc0 : sc1 + 1]
        holes = _boundary_background_holes(shape_box, bg)
        key = _shape_key(holes)
        swatch_colors = [int(v) for v in np.unique(panel) if int(v) not in (bg, grey)]
        assert len(swatch_colors) == 1, "every legend entry must contain one swatch color"
        mapping[key] = swatch_colors[0]
    assert mapping, "legend mapping must be non-empty"
    return legend_end, mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each frame's internal hole pattern to the left legend and fill the remaining interior background.

    Input structure:
    - The background is a single dominant color.
    - On the left sits a vertical legend; each entry has a small grey hollow stencil and an adjacent 2x1 color swatch, with one background row between entries.
    - On the right are several grey rectangular frames. Each frame is a 1-pixel outer border connected to an internal grey stencil whose enclosed background holes encode a shape.

    Rule:
    - Read the legend by extracting each stencil's enclosed background-hole pattern and mapping that normalized shape to the adjacent swatch color.
    - For every grey frame on the right, extract the background cells inside its bbox that do not connect to the frame's inner perimeter; these cells are the frame's hole pattern.
    - Find the legend entry with the same normalized hole shape.
    - Fill the other background cells inside that frame's outer border with the matched color, while keeping the hole cells and all grey pixels unchanged.

    Key insight:
    - The holes are the true symbols: the grey material only defines which background pixels are protected holes and which are ordinary interior space to be filled.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    non_bg = [int(v) for v in np.unique(g) if int(v) != bg]
    assert non_bg, "expected at least one foreground color"
    grey = max(non_bg, key=lambda color: int(np.count_nonzero(g == color)))

    legend_end, mapping = _legend_mapping(g, bg, grey)

    out = g.copy()
    for comp in _components(g == grey):
        r0, c0 = comp.min(axis=0)
        r1, c1 = comp.max(axis=0)
        r0 = int(r0)
        c0 = int(c0)
        r1 = int(r1)
        c1 = int(c1)
        if c1 <= legend_end:
            continue
        tile = g[r0 : r1 + 1, c0 : c1 + 1]
        if not (
            np.all(tile[0, :] == grey)
            and np.all(tile[-1, :] == grey)
            and np.all(tile[:, 0] == grey)
            and np.all(tile[:, -1] == grey)
        ):
            continue
        fillable, holes = _frame_regions(tile, bg)
        assert holes.any(), "every frame must contain a hole pattern"
        key = _shape_key(holes)
        assert key in mapping, "frame hole pattern must appear in the legend"
        fill_color = mapping[key]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[fillable] = fill_color
    return out


def _display_sizes(hole: np.ndarray) -> list[int]:
    if hole.shape == (1, 1):
        return [3, 4]
    return [4]


def _make_stencil(hole: np.ndarray, size: int, rng: random.Random) -> np.ndarray:
    inner_h = size - 2
    inner_w = size - 2
    hh, hw = hole.shape
    assert hh <= inner_h and hw <= inner_w, "hole must fit inside the stencil interior"
    off_r = rng.randint(0, inner_h - hh)
    off_c = rng.randint(0, inner_w - hw)
    stencil = np.ones((size, size), dtype=bool)
    for rr, cc in np.argwhere(hole):
        stencil[1 + off_r + int(rr), 1 + off_c + int(cc)] = False
    return stencil


def _legend_entry(hole: np.ndarray, color: int, grey: int, bg: int, size: int, rng: random.Random) -> np.ndarray:
    stencil = _make_stencil(hole, size, rng)
    entry = np.full((size, size + 1), bg, dtype=int)
    entry[:, :size][stencil] = grey
    swatch_r = rng.randint(0, size - 2)
    entry[swatch_r : swatch_r + 2, size] = color
    return entry


def _frame_object(hole: np.ndarray, grey: int, bg: int, size: int, rng: random.Random) -> tuple[np.ndarray, tuple[tuple[int, int], bytes]]:
    stencil = _make_stencil(hole, size, rng)
    top = rng.randint(*FRAME_MARGIN_RANGE)
    bottom = rng.randint(*FRAME_MARGIN_RANGE)
    left = rng.randint(*FRAME_MARGIN_RANGE)
    right = rng.randint(*FRAME_MARGIN_RANGE)
    sh, sw = stencil.shape
    h = sh + top + bottom + 2
    w = sw + left + right + 2

    obj = np.zeros((h, w), dtype=bool)
    obj[0, :] = True
    obj[-1, :] = True
    obj[:, 0] = True
    obj[:, -1] = True

    sr = 1 + top
    sc = 1 + left
    obj[sr : sr + sh, sc : sc + sw] |= stencil

    side = rng.choice(["top", "bottom", "left", "right"])
    if side == "top":
        bc = sc + rng.randrange(sw)
        obj[1:sr, bc] = True
    elif side == "bottom":
        bc = sc + rng.randrange(sw)
        obj[sr + sh : h - 1, bc] = True
    elif side == "left":
        br = sr + rng.randrange(sh)
        obj[br, 1:sc] = True
    else:
        br = sr + rng.randrange(sh)
        obj[br, sc + sw : w - 1] = True

    tile = np.where(obj, grey, bg)
    fillable, holes = _frame_regions(tile, bg)
    assert fillable.any(), "frame must have fillable interior"
    key = _shape_key(holes)
    assert key == _shape_key(hole), "generated frame must preserve its intended hole key"
    return tile, key


def _can_place(occupied: np.ndarray, h: int, w: int, r0: int, c0: int) -> bool:
    H, W = occupied.shape
    r1 = r0 + h
    c1 = c0 + w
    if r0 < 0 or c0 < 0 or r1 > H or c1 > W:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(H, r1 + 1)
    cc1 = min(W, c1 + 1)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _place(grid: np.ndarray, occupied: np.ndarray, tile: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = tile.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    mask = tile != bg
    patch[mask] = tile[mask]
    occupied[r0 : r0 + h, c0 : c0 + w] |= mask


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(10))
    bg = rng.choice(palette)
    remaining = [c for c in palette if c != bg]
    grey = rng.choice(remaining)
    remaining.remove(grey)

    n_entries = rng.randint(*LEGEND_ENTRY_RANGE)
    n_frames = rng.randint(*FRAME_COUNT_RANGE)

    single_key = _shape_key(HOLE_LIBRARY[0])
    chosen_holes: list[np.ndarray] = []
    used_keys: set[tuple[tuple[int, int], bytes]] = set()
    if rng.random() < 0.5:
        chosen_holes.append(HOLE_LIBRARY[0])
        used_keys.add(single_key)
    pool = HOLE_LIBRARY[1:]
    rng.shuffle(pool)
    for hole in pool:
        if len(chosen_holes) >= n_entries:
            break
        key = _shape_key(hole)
        if key not in used_keys:
            chosen_holes.append(hole)
            used_keys.add(key)
    if len(chosen_holes) < n_entries:
        return None
    rng.shuffle(chosen_holes)

    swatch_colors = rng.sample(remaining, n_entries)

    entries: list[np.ndarray] = []
    mapping: dict[tuple[tuple[int, int], bytes], int] = {}
    for hole, color in zip(chosen_holes, swatch_colors):
        size = rng.choice(_display_sizes(hole))
        entry = _legend_entry(hole, color, grey, bg, size, rng)
        entries.append(entry)
        mapping[_shape_key(hole)] = color

    legend_h = sum(entry.shape[0] for entry in entries) + (len(entries) - 1)
    legend_w = max(entry.shape[1] for entry in entries)
    legend = np.full((legend_h, legend_w), bg, dtype=int)
    cursor = 0
    for entry in entries:
        eh, ew = entry.shape
        legend[cursor : cursor + eh, :ew] = entry
        cursor += eh + 1

    frame_specs: list[tuple[np.ndarray, int]] = []
    used_frame_keys = []
    for _ in range(n_frames):
        hole = rng.choice(chosen_holes)
        size = rng.choice(_display_sizes(hole))
        frame, key = _frame_object(hole, grey, bg, size, rng)
        frame_specs.append((frame, mapping[key]))
        used_frame_keys.append(key)
    if len(set(used_frame_keys)) < min(2, n_frames):
        return None

    gap = rng.randint(*LEGEND_FRAME_GAP_RANGE)
    max_frame_h = max(frame.shape[0] for frame, _ in frame_specs)
    max_frame_w = max(frame.shape[1] for frame, _ in frame_specs)

    if n_frames <= 2:
        frame_rows = rng.choice([1, 2])
        frame_cols = 2 if frame_rows == 1 else 1
    else:
        frame_rows = 2
        frame_cols = 2
    if frame_rows * frame_cols < n_frames:
        return None

    cell_h = max_frame_h + 1
    cell_w = max_frame_w + 1
    min_h = max(legend_h, frame_rows * cell_h)
    min_w = legend_w + gap + frame_cols * cell_w
    if min_h > 30 or min_w > 30:
        return None

    H = rng.randint(max(GRID_H_RANGE[0], min_h), GRID_H_RANGE[1])
    W = rng.randint(max(GRID_W_RANGE[0], min_w), GRID_W_RANGE[1])
    if H > 30 or W > 30:
        return None

    grid = np.full((H, W), bg, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    legend_r0 = rng.randint(0, H - legend_h)
    _place(grid, occupied, legend, legend_r0, 0, bg)

    right_c0 = legend_w + gap
    available_slots = list(range(frame_rows * frame_cols))
    rng.shuffle(available_slots)
    chosen_slots = available_slots[:n_frames]
    rng.shuffle(frame_specs)

    for (frame, _fill_color), slot in zip(frame_specs, chosen_slots):
        fr, fc = divmod(slot, frame_cols)
        h0 = fr * cell_h
        w0 = right_c0 + fc * cell_w
        fh, fw = frame.shape
        max_r = min(H - fh, h0 + cell_h - fh)
        max_c = min(W - fw, w0 + cell_w - fw)
        if max_r < h0 or max_c < w0:
            return None
        placed = False
        row_choices = list(range(h0, max_r + 1))
        col_choices = list(range(w0, max_c + 1))
        rng.shuffle(row_choices)
        rng.shuffle(col_choices)
        for r0 in row_choices:
            for c0 in col_choices:
                if _can_place(occupied, fh, fw, r0, c0):
                    _place(grid, occupied, frame, r0, c0, bg)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
