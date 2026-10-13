from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Inclusive (r0, c0, r1, c1) bbox, following arc_utils conventions."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N, 2) arrays of (row, column)."""
    h, w = mask.shape
    unseen = mask.astype(bool).copy()
    result: list[np.ndarray] = []
    for sr, sc in np.argwhere(unseen):
        sr, sc = int(sr), int(sc)
        if not unseen[sr, sc]:
            continue
        unseen[sr, sc] = False
        q = deque([(sr, sc)])
        cells: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and unseen[nr, nc]:
                    unseen[nr, nc] = False
                    q.append((nr, nc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _d4(array: np.ndarray) -> list[np.ndarray]:
    """The four rotations and their left-right reflections."""
    return [
        transformed.copy()
        for k in range(4)
        for transformed in (np.rot90(array, k), np.fliplr(np.rot90(array, k)))
    ]


def _separator_panel(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int], list[int], list[int]]:
    """Find the uniquely framed regular row/column separator lattice."""
    candidates = []
    for color in map(int, np.unique(grid)):
        box = _bbox(grid == color)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        view = grid[r0 : r1 + 1, c0 : c1 + 1]
        full_rows = np.flatnonzero(np.all(view == color, axis=1)).tolist()
        full_cols = np.flatnonzero(np.all(view == color, axis=0)).tolist()
        if len(full_rows) < 3 or len(full_cols) < 3:
            continue
        if full_rows[0] != 0 or full_rows[-1] != view.shape[0] - 1:
            continue
        if full_cols[0] != 0 or full_cols[-1] != view.shape[1] - 1:
            continue
        row_steps = np.diff(full_rows)
        col_steps = np.diff(full_cols)
        if not (np.all(row_steps == row_steps[0]) and np.all(col_steps == col_steps[0])):
            continue
        if int(row_steps[0]) < 2 or int(col_steps[0]) < 2:
            continue
        expected = np.zeros(view.shape, dtype=bool)
        expected[full_rows, :] = True
        expected[:, full_cols] = True
        if not np.array_equal(view == color, expected):
            continue
        if np.any(view[~expected] == color):
            continue
        candidates.append((color, box, full_rows, full_cols))
    assert len(candidates) == 1, "expected one visible separator lattice"
    return candidates[0]


def _decode(grid: np.ndarray) -> dict:
    """Parse both panels and return all semantic intermediate objects."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(g[0, 0])
    sep, panel_box, row_lines, col_lines = _separator_panel(g)
    pr0, pc0, pr1, pc1 = panel_box
    panel = g[pr0 : pr1 + 1, pc0 : pc1 + 1]

    color_map = np.empty((len(row_lines) - 1, len(col_lines) - 1), dtype=int)
    for r in range(color_map.shape[0]):
        for c in range(color_map.shape[1]):
            block = panel[row_lines[r] + 1 : row_lines[r + 1], col_lines[c] + 1 : col_lines[c + 1]]
            assert block.size >= 4 and not np.any(block == sep), "invalid noise block"
            counts = Counter(map(int, block.flat))
            top = max(counts.values())
            winners = [color for color, count in counts.items() if count == top]
            assert len(winners) == 1 and top > block.size / 2, "each block needs one strict majority"
            color_map[r, c] = winners[0]

    outside = g != bg
    outside[pr0 : pr1 + 1, pc0 : pc1 + 1] = False
    components = _components(outside)
    assert len(components) == color_map.size, "expected one motif per lattice cell"

    records = []
    reference_mask = None
    reference_shape = None
    for coords in components:
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        values = set(map(int, g[coords[:, 0], coords[:, 1]]))
        assert len(values) == 1, "each motif must be monochrome"
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        if reference_mask is None:
            reference_mask, reference_shape = local, local.shape
        assert local.shape == reference_shape and np.array_equal(local, reference_mask), "motifs must share one shape"
        records.append((r0, c0, r1, c1, next(iter(values))))

    row_starts = sorted({record[0] for record in records})
    col_starts = sorted({record[1] for record in records})
    assert len(row_starts) * len(col_starts) == len(records), "motifs must form a rectangular lattice"
    slots = {(r, c): color for r, c, _r1, _c1, color in records}
    assert set(slots) == {(r, c) for r in row_starts for c in col_starts}, "missing motif lattice slot"

    component_colors = Counter(slots.values())
    default_candidates = [color for color, count in component_colors.items() if count == len(records) - 1]
    assert len(default_candidates) == 1, "expected exactly one precolored anchor motif"
    default = default_candidates[0]
    anchor_records = [(r, c, color) for (r, c), color in slots.items() if color != default]
    assert len(anchor_records) == 1, "expected exactly one anchor motif"
    ar, ac, anchor_color = anchor_records[0]
    anchor_index = (row_starts.index(ar), col_starts.index(ac))

    matches = []
    for pose_index, posed in enumerate(_d4(color_map)):
        if posed.shape == (len(row_starts), len(col_starts)) and int(posed[anchor_index]) == anchor_color:
            matches.append((pose_index, posed))
    assert len(matches) == 1, "anchor position and color must determine one dihedral pose"
    pose_index, posed = matches[0]
    return {
        "bg": bg,
        "separator": sep,
        "color_map": color_map,
        "pose_index": pose_index,
        "posed": posed,
        "motif_mask": reference_mask,
        "records": records,
        "row_starts": row_starts,
        "col_starts": col_starts,
        "anchor_index": anchor_index,
        "anchor_color": anchor_color,
        "default": default,
    }


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress noisy majority blocks, use the lone precolored motif to fix a dihedral pose, and recolor the motif lattice.

    Input structure:
    - One region is a monochrome framed lattice whose cells are noisy blocks, each with a unique strict majority color.
    - A separate region is a rectangular or transposed lattice of identical, background-separated motifs; exactly one motif has a nondefault color.

    Rule:
    - Replace every noisy block conceptually by its majority color to form a color map. Among all eight rotations/reflections of that map, select the unique pose whose shape and color at the precolored motif both match the motif lattice. Recolor every motif from the posed map, preserve its binary shape and the anchor, and return the motif lattice's tight crop.

    Key insight:
    - The unique precolored motif is simultaneously an output cell and a visible orientation instruction; neither raw block pixels nor the unrotated compressed map can be copied directly.
    """
    info = _decode(grid)
    bg = info["bg"]
    posed = info["posed"]
    mask = info["motif_mask"]
    rows = info["row_starts"]
    cols = info["col_starts"]
    mh, mw = mask.shape
    r0, c0 = rows[0], cols[0]
    r1, c1 = rows[-1] + mh - 1, cols[-1] + mw - 1
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for ri, r in enumerate(rows):
        for ci, c in enumerate(cols):
            view = out[r - r0 : r - r0 + mh, c - c0 : c - c0 + mw]
            view[mask] = int(posed[ri, ci])
    return out


_PALETTE = list(range(1, 10))
_MOTIF_MASKS = (
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
)
_GEN_CALL_COUNT = 0
_GEN_POSE_OFFSET = random.randrange(8)


def _make_color_map(rows: int, cols: int, colors: list[int]) -> tuple[np.ndarray, int]:
    """Build an asymmetric three-color map with a unique off-axis anchor color."""
    anchor, first, second = colors
    cells = rows * cols
    values = [anchor] + [first] * ((cells - 1) // 2) + [second] * (cells - 1 - (cells - 1) // 2)
    tail = values[1:]
    random.shuffle(tail)
    anchor_col = random.choice([0, cols - 1])
    anchor_row = random.randrange(rows)
    positions = [(r, c) for r in range(rows) for c in range(cols) if (r, c) != (anchor_row, anchor_col)]
    random.shuffle(positions)
    result = np.empty((rows, cols), dtype=int)
    result[anchor_row, anchor_col] = anchor
    for (r, c), value in zip(positions, tail):
        result[r, c] = value
    return result, anchor


def _noise_panel(color_map: np.ndarray, separator: int, block_size: int = 3) -> np.ndarray:
    rows, cols = color_map.shape
    panel = np.full((rows * block_size + rows + 1, cols * block_size + cols + 1), separator, dtype=int)
    instruction_colors = set(map(int, color_map.flat))
    for r in range(rows):
        for c in range(cols):
            majority = int(color_map[r, c])
            minority = list(instruction_colors - {majority})
            cells = [majority] * 5 + [minority[0]] * 2 + [minority[1]] * 2
            random.shuffle(cells)
            center = 4
            if cells[center] == majority:
                swap = next(i for i, value in enumerate(cells) if value != majority)
                cells[center], cells[swap] = cells[swap], cells[center]
            block = np.asarray(cells, dtype=int).reshape(block_size, block_size)
            rr = 1 + r * (block_size + 1)
            cc = 1 + c * (block_size + 1)
            panel[rr : rr + block_size, cc : cc + block_size] = block
    return panel


def _motif_panel(posed: np.ndarray, anchor_color: int, default: int, mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    rows, cols = posed.shape
    mh, mw = mask.shape
    panel = np.zeros((rows * mh + rows - 1, cols * mw + cols - 1), dtype=int)
    anchor_positions = list(map(tuple, np.argwhere(posed == anchor_color)))
    assert len(anchor_positions) == 1
    anchor_index = anchor_positions[0]
    for r in range(rows):
        for c in range(cols):
            color = anchor_color if (r, c) == anchor_index else default
            rr, cc = r * (mh + 1), c * (mw + 1)
            panel[rr : rr + mh, cc : cc + mw][mask] = color
    return panel, anchor_index


def generate_input() -> np.ndarray:
    """Generate a diverse, nondegenerate anchored-instruction puzzle instance."""
    global _GEN_CALL_COUNT
    call_index = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1
    bg = 0
    rows, cols = ((2, 3), (2, 4))[call_index % 2]
    instruction = random.sample(_PALETTE, 3)
    color_map, anchor_color = _make_color_map(rows, cols, instruction)
    pose_index = (_GEN_POSE_OFFSET + call_index) % 8
    posed = _d4(color_map)[pose_index]

    remaining = [color for color in _PALETTE if color not in instruction]
    separator, default = random.sample(remaining, 2)
    motif_mask = random.choice(_MOTIF_MASKS).copy()
    noise = _noise_panel(color_map, separator)
    motifs, _anchor_index = _motif_panel(posed, anchor_color, default, motif_mask)

    gap = random.randint(2, 3)
    layout_options = []
    horizontal_shape = (max(noise.shape[0], motifs.shape[0]), noise.shape[1] + gap + motifs.shape[1])
    vertical_shape = (noise.shape[0] + gap + motifs.shape[0], max(noise.shape[1], motifs.shape[1]))
    if max(horizontal_shape[0] + 2, horizontal_shape[1] + 2) <= 30:
        layout_options.append((True, horizontal_shape))
    if max(vertical_shape[0] + 2, vertical_shape[1] + 2) <= 30:
        layout_options.append((False, vertical_shape))
    assert layout_options
    horizontal, (inner_h, inner_w) = layout_options[call_index % len(layout_options)]

    extra_h = random.randint(0, 30 - (inner_h + 2))
    extra_w = random.randint(0, 30 - (inner_w + 2))
    height, width = inner_h + 2 + extra_h, inner_w + 2 + extra_w
    grid = np.full((height, width), bg, dtype=int)
    top = random.randint(1, height - inner_h - 1)
    left = random.randint(1, width - inner_w - 1)
    if horizontal:
        nr = top + random.randint(0, inner_h - noise.shape[0])
        mr = top + random.randint(0, inner_h - motifs.shape[0])
        nc, mc = left, left + noise.shape[1] + gap
    else:
        nc = left + random.randint(0, inner_w - noise.shape[1])
        mc = left + random.randint(0, inner_w - motifs.shape[1])
        nr, mr = top, top + noise.shape[0] + gap
    grid[nr : nr + noise.shape[0], nc : nc + noise.shape[1]] = noise
    motif_view = grid[mr : mr + motifs.shape[0], mc : mc + motifs.shape[1]]
    motif_view[motifs != bg] = motifs[motifs != bg]

    info = _decode(grid)
    assert np.array_equal(info["color_map"], color_map)
    assert info["pose_index"] == pose_index
    assert info["anchor_color"] == anchor_color
    output = main(grid)
    assert output.shape == motifs.shape
    expected = np.zeros_like(motifs)
    for r in range(posed.shape[0]):
        for c in range(posed.shape[1]):
            rr, cc = r * 3, c * 3
            expected[rr : rr + 2, cc : cc + 2][motif_mask] = int(posed[r, c])
    assert np.array_equal(output, expected)
    assert not np.array_equal(output, motifs), "recoloring must change non-anchor motifs"
    return grid
