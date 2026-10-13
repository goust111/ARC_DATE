from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BACKGROUND = 0
MARKER_EXCLUDED_COLOR = 4
PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    steps = NEIGHBORS4 if int(connectivity) == 4 else NEIGHBORS8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None, clip: bool = False) -> None:
    h, w = patch.shape
    hh, ww = dst.shape
    if not clip:
        if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
            raise ValueError("patch out of bounds")
        rr0, cc0, rr1, cc1 = r0, c0, r0 + h, c0 + w
        pr0, pc0 = 0, 0
    else:
        rr0 = max(0, int(r0))
        cc0 = max(0, int(c0))
        rr1 = min(hh, int(r0) + h)
        cc1 = min(ww, int(c0) + w)
        if rr0 >= rr1 or cc0 >= cc1:
            return
        pr0 = rr0 - int(r0)
        pc0 = cc0 - int(c0)
    pr1 = pr0 + (rr1 - rr0)
    pc1 = pc0 + (cc1 - cc0)
    view = patch[pr0:pr1, pc0:pc1]
    if transparent is None:
        dst[rr0:rr1, cc0:cc1] = view
        return
    mask = view != int(transparent)
    dst_view = dst[rr0:rr1, cc0:cc1]
    dst_view[mask] = view[mask]


def _dense_runs(counts: np.ndarray) -> list[tuple[int, int]]:
    values = sorted(set(map(int, counts)))
    assert len(values) >= 2
    gaps = [(b - a, b) for a, b in zip(values, values[1:])]
    max_gap = max(gap for gap, _ in gaps)
    thresholds = [upper for gap, upper in gaps if gap == max_gap]
    assert len(thresholds) == 1
    high = np.asarray(counts, dtype=int) >= thresholds[0]
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(high):
        if not high[i]:
            i += 1
            continue
        j = i
        while j < len(high) and high[j]:
            j += 1
        runs.append((i, j))
        i = j
    assert len(runs) >= 2
    assert len({b - a for a, b in runs}) == 1
    if len(runs) > 1:
        assert len({runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)}) == 1
    return runs


def _mode_color(values: list[int]) -> int:
    counts = Counter(values)
    best = max(counts.values())
    modes = [color for color, n in counts.items() if n == best]
    assert len(modes) == 1
    return int(modes[0])


def _analyze_array(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], np.ndarray, tuple[int, int, int, int]]:
    bg = _get_bg(grid)
    fg = grid != bg
    row_runs = _dense_runs(fg.sum(axis=1))
    col_runs = _dense_runs(fg.sum(axis=0))
    slot_h = row_runs[0][1] - row_runs[0][0]
    slot_w = col_runs[0][1] - col_runs[0][0]
    assert slot_h >= 3 and slot_w >= 3
    assert len(col_runs) == 3
    assert len(row_runs) in (2, 3)
    blocks = []
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            block = np.asarray(grid[r0:r1, c0:c1], dtype=int)
            assert block.shape == (slot_h, slot_w)
            blocks.append(block)
    sprite = np.zeros((slot_h, slot_w), dtype=int)
    for r in range(slot_h):
        for c in range(slot_w):
            sprite[r, c] = _mode_color([int(block[r, c]) for block in blocks])
    bbox = (row_runs[0][0], col_runs[0][0], row_runs[-1][1] - 1, col_runs[-1][1] - 1)
    return row_runs, col_runs, sprite, bbox


def _anchor_for_color(sprite: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(sprite == int(color))
    assert len(coords) >= 1
    coords = sorted((int(r), int(c)) for r, c in coords)
    return coords[0]


def _render_from_sprite(
    shape: tuple[int, int],
    sprite: np.ndarray,
    markers: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c, color in markers:
        ar, ac = _anchor_for_color(sprite, color)
        _paste(out, sprite, r - ar, c - ac, transparent=BACKGROUND, clip=True)
    return out


def _slot_mask(row_runs: list[tuple[int, int]], col_runs: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            mask[r0:r1, c0:c1] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the repeated sprite by majority vote, then expand that sprite around each colored marker using the matched color position as the anchor.

    Input structure:
    - A black background contains a regular 2x3 or 3x3 slot array on the right half, with fixed row and column gaps.
    - Each slot holds a noisy copy of the same sparse 3x3 or 4x4 sprite; 1-2 cells per copy may be recolored.
    - The left half contains isolated single-pixel markers whose colors appear in the true sprite, and color 4 is unused.

    Rule:
    - Detect the dense row bands and column bands of the slot array, and treat their Cartesian product as the slot layout.
    - Recover the clean sprite by taking the unique per-coordinate majority color across all slots.
    - For each isolated marker, find the top-leftmost occurrence of that marker color inside the recovered sprite and place the whole sprite so that occurrence lands on the marker cell, clipping at the grid boundary.
    - Erase the entire original slot-array bounding box, including the gap area, to black.

    Key insight:
    - The right-side array is only a temporary codebook: majority voting reconstructs one clean sprite, and each marker color selects the anchor point that determines how that sprite expands elsewhere.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    row_runs, col_runs, sprite, array_bbox = _analyze_array(g)
    r0, c0, r1, c1 = array_bbox

    outside = np.ones_like(g, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    markers: list[tuple[int, int, int]] = []
    for comp in _components_mask((g != bg) & outside, connectivity=4):
        assert len(comp) == 1
        rr, cc = map(int, comp[0])
        markers.append((rr, cc, int(g[rr, cc])))
    assert markers

    out = np.full_like(g, bg)
    for r, c, color in markers:
        ar, ac = _anchor_for_color(sprite, color)
        _paste(out, sprite, r - ar, c - ac, transparent=bg, clip=True)
    out[r0 : r1 + 1, c0 : c1 + 1] = bg
    return out


def _occupied8(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    for dr, dc in NEIGHBORS8 + ((0, 0),):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            return True
    return False


def _sprite_connected(mask: np.ndarray) -> bool:
    comps = _components_mask(mask, connectivity=8)
    return len(comps) == 1


def _generate_sprite(rng: random.Random) -> tuple[np.ndarray, list[int]]:
    for _ in range(300):
        h = rng.choice((3, 4))
        w = rng.choice((3, 4))
        n_colors = rng.choice((2, 3))
        colors = rng.sample(PALETTE, n_colors)
        min_cells = max(h, w, n_colors)
        max_cells = min(h * w, 2 * n_colors)
        if min_cells > max_cells:
            continue
        total_cells = rng.randint(min_cells, max_cells)
        cells = rng.sample([(r, c) for r in range(h) for c in range(w)], total_cells)
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if len(rows) != h or len(cols) != w:
            continue
        if not _sprite_connected(mask):
            continue
        extras = total_cells - n_colors
        dup_colors = rng.sample(colors, extras) if extras else []
        counts = {color: 1 for color in colors}
        for color in dup_colors:
            counts[color] += 1
        color_list: list[int] = []
        for color in colors:
            color_list.extend([color] * counts[color])
        rng.shuffle(color_list)
        sprite = np.zeros((h, w), dtype=int)
        for (r, c), color in zip(cells, color_list):
            sprite[r, c] = color
        if MARKER_EXCLUDED_COLOR in sprite:
            continue
        return sprite, sorted(colors)
    raise RuntimeError("failed to generate sprite")


def _candidate_marker_coords(
    shape: tuple[int, int],
    sprite: np.ndarray,
    color: int,
    array_left: int,
    *,
    allow_clip: bool,
) -> list[tuple[int, int]]:
    h, w = shape
    sh, sw = sprite.shape
    ar, ac = _anchor_for_color(sprite, color)
    coords: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(array_left):
            start_r = r - ar
            start_c = c - ac
            if start_c + sw > array_left:
                continue
            clipped = start_r < 0 or start_c < 0 or start_r + sh > h or start_c + sw > w
            if clipped and not allow_clip:
                continue
            if not clipped and allow_clip:
                continue
            visible = np.zeros(shape, dtype=bool)
            rr0 = max(0, start_r)
            cc0 = max(0, start_c)
            rr1 = min(h, start_r + sh)
            cc1 = min(w, start_c + sw)
            if rr0 >= rr1 or cc0 >= cc1:
                continue
            patch = sprite[rr0 - start_r : rr1 - start_r, cc0 - start_c : cc1 - start_c]
            visible[rr0:rr1, cc0:cc1] = patch != BACKGROUND
            if visible.sum() < 2:
                continue
            coords.append((r, c))
    return coords


def _place_markers(
    rng: random.Random,
    shape: tuple[int, int],
    sprite: np.ndarray,
    colors: list[int],
    array_left: int,
) -> list[tuple[int, int, int]] | None:
    clip_color = rng.choice(colors) if rng.random() < 0.45 else None
    occupancy = np.zeros(shape, dtype=bool)
    markers: list[tuple[int, int, int]] = []
    for color in rng.sample(colors, len(colors)):
        candidates = _candidate_marker_coords(
            shape,
            sprite,
            color,
            array_left,
            allow_clip=color == clip_color,
        )
        rng.shuffle(candidates)
        placed = False
        for r, c in candidates:
            ar, ac = _anchor_for_color(sprite, color)
            start_r = r - ar
            start_c = c - ac
            visible = np.zeros(shape, dtype=bool)
            rr0 = max(0, start_r)
            cc0 = max(0, start_c)
            rr1 = min(shape[0], start_r + sprite.shape[0])
            cc1 = min(shape[1], start_c + sprite.shape[1])
            patch = sprite[rr0 - start_r : rr1 - start_r, cc0 - start_c : cc1 - start_c]
            visible[rr0:rr1, cc0:cc1] = patch != BACKGROUND
            if np.any(occupancy & visible):
                continue
            if _occupied8(occupancy, r, c):
                continue
            occupancy |= visible
            markers.append((r, c, color))
            placed = True
            break
        if not placed:
            return None
    return sorted(markers)


def _corrupt_slot(
    rng: random.Random,
    sprite: np.ndarray,
    wrong_counts: np.ndarray,
    max_wrong: int,
) -> np.ndarray | None:
    block = sprite.copy()
    h, w = sprite.shape
    n_corrupt = rng.choice((1, 2))
    cells = [(r, c) for r in range(h) for c in range(w) if wrong_counts[r, c] < max_wrong]
    rng.shuffle(cells)
    chosen = cells[:n_corrupt]
    if len(chosen) != n_corrupt:
        return None
    sprite_colors = sorted({int(v) for v in sprite.ravel() if v != BACKGROUND})
    for r, c in chosen:
        current = int(block[r, c])
        choices = [color for color in sprite_colors if color != current]
        if current == BACKGROUND:
            choices = sprite_colors
        if not choices:
            return None
        block[r, c] = rng.choice(choices)
        wrong_counts[r, c] += 1
    if np.array_equal(block, sprite):
        return None
    return block


def _build_input_from_layout(
    rng: random.Random,
    shape: tuple[int, int],
    sprite: np.ndarray,
    row_starts: list[int],
    col_starts: list[int],
    markers: list[tuple[int, int, int]],
) -> np.ndarray | None:
    grid = np.zeros(shape, dtype=int)
    for r, c, color in markers:
        grid[r, c] = color
    total_slots = len(row_starts) * len(col_starts)
    max_wrong = total_slots // 2 - 1
    if max_wrong < 1:
        return None
    wrong_counts = np.zeros(sprite.shape, dtype=int)
    placed_blocks: list[np.ndarray] = []
    for r0 in row_starts:
        for c0 in col_starts:
            block = _corrupt_slot(rng, sprite, wrong_counts, max_wrong)
            if block is None:
                return None
            placed_blocks.append(block)
            _paste(grid, block, r0, c0)
    array_r0 = row_starts[0]
    array_c0 = col_starts[0]
    array_r1 = row_starts[-1] + sprite.shape[0] - 1
    array_c1 = col_starts[-1] + sprite.shape[1] - 1
    gap_cells = [
        (r, c)
        for r in range(array_r0, array_r1 + 1)
        for c in range(array_c0, array_c1 + 1)
        if grid[r, c] == BACKGROUND
    ]
    rng.shuffle(gap_cells)
    if gap_cells and rng.random() < 0.6:
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        placed_noise = 0
        sprite_colors = sorted({int(v) for v in sprite.ravel() if v != BACKGROUND})
        for r, c in gap_cells:
            if r in used_rows or c in used_cols:
                continue
            grid[r, c] = rng.choice(sprite_colors)
            used_rows.add(r)
            used_cols.add(c)
            placed_noise += 1
            if placed_noise == 2:
                break
    return grid


def _layout_choices(rng: random.Random, sprite: np.ndarray) -> tuple[tuple[int, int], list[int], list[int]] | None:
    sh, sw = sprite.shape
    slot_rows = rng.choice((2, 3))
    slot_cols = 3
    row_gap = rng.choice((1, 2))
    col_gap = rng.choice((1, 2))
    top = rng.randint(1, 4)
    left = rng.randint(8, 12)
    right = rng.randint(1, 3)
    bottom = rng.randint(1, 4)
    h = top + slot_rows * sh + (slot_rows - 1) * row_gap + bottom
    w = left + slot_cols * sw + (slot_cols - 1) * col_gap + right
    if h > 30 or w > 30:
        return None
    row_starts = [top + i * (sh + row_gap) for i in range(slot_rows)]
    col_starts = [left + i * (sw + col_gap) for i in range(slot_cols)]
    return (h, w), row_starts, col_starts


def _single_slot_output(shape: tuple[int, int], slot_sprite: np.ndarray, markers: list[tuple[int, int, int]], bbox: tuple[int, int, int, int]) -> np.ndarray:
    out = _render_from_sprite(shape, slot_sprite, markers)
    r0, c0, r1, c1 = bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = BACKGROUND
    return out


def _validate_instance(inp: np.ndarray, sprite: np.ndarray, markers: list[tuple[int, int, int]], row_starts: list[int], col_starts: list[int], expected: np.ndarray) -> bool:
    try:
        row_runs, col_runs, recovered, bbox = _analyze_array(inp)
        if not np.array_equal(recovered, sprite):
            return False
        if [r0 for r0, _ in row_runs] != row_starts:
            return False
        if [c0 for c0, _ in col_runs] != col_starts:
            return False
        out = main(inp)
        if not np.array_equal(out, expected):
            return False
        r0, c0, r1, c1 = bbox
        if np.any(out[r0 : r1 + 1, c0 : c1 + 1] != BACKGROUND):
            return False
        outside = np.ones_like(inp, dtype=bool)
        outside[r0 : r1 + 1, c0 : c1 + 1] = False
        comps = _components_mask((inp != BACKGROUND) & outside, connectivity=4)
        if sorted(len(comp) for comp in comps) != [1] * len(markers):
            return False
        slot_mask = _slot_mask(row_runs, col_runs, inp.shape)
        blocks = [inp[r0a:r1a, c0a:c1a] for r0a, r1a in row_runs for c0a, c1a in col_runs]
        if not all(not np.array_equal(block, sprite) for block in blocks):
            return False
        for block in blocks:
            alt = _single_slot_output(inp.shape, block, markers, bbox)
            if np.array_equal(alt, expected):
                return False
        for color in sorted({color for _, _, color in markers}):
            coords = sorted((int(r), int(c)) for r, c in np.argwhere(sprite == color))
            if len(coords) >= 2:
                second = coords[1]
                alt = np.zeros_like(expected)
                for r, c, marker_color in markers:
                    anchor = second if marker_color == color else _anchor_for_color(sprite, marker_color)
                    _paste(alt, sprite, r - anchor[0], c - anchor[1], transparent=BACKGROUND, clip=True)
                if np.array_equal(alt, expected):
                    return False
        if np.count_nonzero(inp & (~slot_mask)) < len(markers):
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        sprite, colors = _generate_sprite(rng)
        layout = _layout_choices(rng, sprite)
        if layout is None:
            continue
        shape, row_starts, col_starts = layout
        markers = _place_markers(rng, shape, sprite, colors, col_starts[0])
        if markers is None:
            continue
        expected = _render_from_sprite(shape, sprite, markers)
        inp = _build_input_from_layout(rng, shape, sprite, row_starts, col_starts, markers)
        if inp is None:
            continue
        if not _validate_instance(inp, sprite, markers, row_starts, col_starts, expected):
            continue
        return inp
    raise RuntimeError("failed to generate valid puzzle instance")
