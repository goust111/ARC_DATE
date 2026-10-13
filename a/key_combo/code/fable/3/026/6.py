from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
MOTIF_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

PANEL_W_RANGE = (7, 9)
HEIGHT_RANGE = (9, 15)
MOTIF_SIZE_RANGE = (3, 5)
MAX_GENERATION_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size, "expected a visible motif"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rot_cw(arr: np.ndarray) -> np.ndarray:
    return np.rot90(arr, -1).copy()


def _background_without_separators(grid: np.ndarray, sep_cols: list[int]) -> int:
    keep = np.ones(grid.shape[1], dtype=bool)
    keep[sep_cols] = False
    values = [int(v) for v in grid[:, keep].ravel() if int(v) != GREY]
    assert values, "expected panel cells outside the grey separators"
    return Counter(values).most_common(1)[0][0]


def _split_panels(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = g.shape

    sep_cols = [c for c in range(W) if np.all(g[:, c] == GREY)]
    assert len(sep_cols) == 2, "expected exactly two complete grey separator columns"
    first, second = sep_cols
    panel_w = first
    assert panel_w > 0, "left panel must be non-empty"
    assert second == 2 * panel_w + 1, "the three panels must have equal width"
    assert W == 3 * panel_w + 2, "expected only three panels and two separators"

    bg = _background_without_separators(g, sep_cols)
    assert bg != GREY, "grey is reserved for the separators"
    panels = [
        g[:, 0:panel_w].copy(),
        g[:, panel_w + 1 : 2 * panel_w + 1].copy(),
        g[:, 2 * panel_w + 2 : 3 * panel_w + 2].copy(),
    ]
    for panel in panels:
        assert not np.any(panel == GREY), "grey may only appear in separator columns"
    return panels, bg


def _panel_motif(panel: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int], int]:
    mask = panel != int(bg)
    box = _bbox(mask)
    r0, c0, r1, c1 = box
    size_r = r1 - r0 + 1
    size_c = c1 - c0 + 1
    assert size_r == size_c, "each motif must have a square bounding box"
    motif = _crop_bbox(panel, box)
    assert np.array_equal(mask, _placed_mask(panel.shape, motif != bg, (r0, c0))), "motif must be the only foreground in its panel"
    return motif, (r0, c0), size_r


def _placed_mask(shape: tuple[int, int], mask: np.ndarray, pos: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    r0, c0 = pos
    h, w = mask.shape
    out[r0 : r0 + h, c0 : c0 + w] = mask
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extrapolate the fourth panel of a translating, rotating motif whose translation accelerates.

    Input structure:
    - Two complete grey vertical lines divide the input into three equal-width panels, read left to right as consecutive frames.
    - Every panel has the same non-grey motif in a square bounding box on the background.
    - From one panel to the next, the motif rotates ninety degrees clockwise around its own center and also translates inside the panel.

    Rule:
    - Read the first displacement from frame 1 to frame 2 and the second displacement from frame 2 to frame 3.
    - Their difference is the fixed acceleration, so the next displacement is the second displacement plus that same difference.
    - Rotate the third-frame motif once more clockwise and draw it in a single output panel at the extrapolated fourth-frame position.

    Key insight:
    The three panels form one timeline: the answer needs both the orientation cycle and the second-order translation, not a constant-velocity or rotation-only continuation.
    """
    panels, bg = _split_panels(grid)
    motifs: list[np.ndarray] = []
    positions: list[tuple[int, int]] = []
    sizes: list[int] = []
    for panel in panels:
        motif, pos, size = _panel_motif(panel, bg)
        motifs.append(motif)
        positions.append(pos)
        sizes.append(size)

    assert len(set(sizes)) == 1, "the same square motif size must be used in all frames"
    assert np.array_equal(motifs[1], _rot_cw(motifs[0])), "frame 2 must be frame 1 rotated clockwise"
    assert np.array_equal(motifs[2], _rot_cw(motifs[1])), "frame 3 must be frame 2 rotated clockwise"

    d1 = np.array(positions[1], dtype=int) - np.array(positions[0], dtype=int)
    d2 = np.array(positions[2], dtype=int) - np.array(positions[1], dtype=int)
    accel = d2 - d1
    d3 = d2 + accel
    next_pos = np.array(positions[2], dtype=int) + d3

    final_motif = _rot_cw(motifs[2])
    H, W = panels[0].shape
    size = final_motif.shape[0]
    r0, c0 = int(next_pos[0]), int(next_pos[1])
    assert 0 <= r0 <= H - size and 0 <= c0 <= W - size, "extrapolated motif must fit in one panel"

    out = np.full((H, W), bg, dtype=int)
    out[r0 : r0 + size, c0 : c0 + size] = final_motif
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _add_path(cells: set[tuple[int, int]], a: tuple[int, int], b: tuple[int, int], rng: random.Random) -> None:
    r, c = a
    br, bc = b
    if rng.random() < 0.5:
        while r != br:
            cells.add((r, c))
            r += 1 if br > r else -1
        while c != bc:
            cells.add((r, c))
            c += 1 if bc > c else -1
    else:
        while c != bc:
            cells.add((r, c))
            c += 1 if bc > c else -1
        while r != br:
            cells.add((r, c))
            r += 1 if br > r else -1
    cells.add((br, bc))


def _random_connected_mask(size: int, rng: random.Random) -> np.ndarray:
    center = (rng.randrange(size), rng.randrange(size))
    targets = [
        (0, rng.randrange(size)),
        (size - 1, rng.randrange(size)),
        (rng.randrange(size), 0),
        (rng.randrange(size), size - 1),
    ]
    cells = {center}
    for target in targets:
        _add_path(cells, center, target, rng)

    max_area = max(len(cells), int(size * size * 0.7))
    target_area = rng.randint(len(cells), max_area)
    while len(cells) < target_area:
        frontier = set()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(tuple(frontier)))

    mask = np.zeros((size, size), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_motif(size: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(100):
        mask = _random_connected_mask(size, rng)
        if not _is_connected(mask):
            continue
        rows, cols = np.where(mask)
        if {0, size - 1} - set(map(int, rows)) or {0, size - 1} - set(map(int, cols)):
            continue

        colors = rng.sample(MOTIF_COLORS, rng.choice((2, 3)))
        coords = list(zip(rows.tolist(), cols.tolist()))
        rng.shuffle(coords)
        motif = np.full((size, size), BLACK, dtype=int)
        for i, (r, c) in enumerate(coords):
            if i < len(colors):
                color = colors[i]
            else:
                color = rng.choice(colors)
            motif[r, c] = color

        rotations = [_rotate_n(motif, k) for k in range(4)]
        if len({arr.tobytes() for arr in rotations}) != 4:
            continue
        return motif
    return None


def _rotate_n(motif: np.ndarray, turns: int) -> np.ndarray:
    out = motif.copy()
    for _ in range(turns % 4):
        out = _rot_cw(out)
    return out


def _top_left_range(limit: int, offsets: list[int], size: int) -> tuple[int, int] | None:
    lo = -min(offsets)
    hi = limit - size - max(offsets)
    if lo > hi:
        return None
    return int(lo), int(hi)


def _sample_motion(height: int, panel_w: int, size: int, rng: random.Random) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]] | None:
    vectors = [-1, 0, 1]
    candidates = []
    for dr1 in vectors:
        for dc1 in vectors:
            d1 = np.array([dr1, dc1], dtype=int)
            if not np.any(d1):
                continue
            for ar in vectors:
                for ac in vectors:
                    accel = np.array([ar, ac], dtype=int)
                    if not np.any(accel):
                        continue
                    d2 = d1 + accel
                    d3 = d2 + accel
                    if not np.any(d2) or not np.any(d3):
                        continue
                    if int(np.abs(d2).sum()) <= int(np.abs(d1).sum()):
                        continue
                    if int(np.abs(d3).sum()) <= int(np.abs(d2).sum()):
                        continue

                    rel = [
                        np.array([0, 0], dtype=int),
                        d1,
                        2 * d1 + accel,
                        3 * d1 + 3 * accel,
                        3 * d1 + 2 * accel,
                    ]
                    row_range = _top_left_range(height, [int(p[0]) for p in rel], size)
                    col_range = _top_left_range(panel_w, [int(p[1]) for p in rel], size)
                    if row_range is None or col_range is None:
                        continue
                    candidates.append((d1, accel, row_range, col_range))

    if not candidates:
        return None
    d1, accel, row_range, col_range = rng.choice(candidates)
    r0 = rng.randint(*row_range)
    c0 = rng.randint(*col_range)

    p0 = np.array([r0, c0], dtype=int)
    p1 = p0 + d1
    p2 = p1 + d1 + accel
    p3 = p2 + d1 + 2 * accel
    positions = [tuple(map(int, p)) for p in (p0, p1, p2, p3)]
    return positions, tuple(map(int, d1)), tuple(map(int, accel))


def _render_input(
    height: int,
    panel_w: int,
    motif: np.ndarray,
    positions: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((height, 3 * panel_w + 2), BLACK, dtype=int)
    grid[:, panel_w] = GREY
    grid[:, 2 * panel_w + 1] = GREY
    starts = [0, panel_w + 1, 2 * panel_w + 2]
    size = motif.shape[0]
    for frame in range(3):
        r0, c0 = positions[frame]
        patch = _rotate_n(motif, frame)
        panel_c = starts[frame] + c0
        grid[r0 : r0 + size, panel_c : panel_c + size] = patch
    return grid


def _expected_output(height: int, panel_w: int, motif: np.ndarray, fourth_pos: tuple[int, int]) -> np.ndarray:
    out = np.full((height, panel_w), BLACK, dtype=int)
    r0, c0 = fourth_pos
    size = motif.shape[0]
    out[r0 : r0 + size, c0 : c0 + size] = _rotate_n(motif, 3)
    return out


def _wrong_constant_velocity_output(height: int, panel_w: int, motif: np.ndarray, positions: list[tuple[int, int]]) -> np.ndarray:
    p1 = np.array(positions[1], dtype=int)
    p2 = np.array(positions[2], dtype=int)
    wrong_pos = tuple(map(int, p2 + (p2 - p1)))
    return _expected_output(height, panel_w, motif, wrong_pos)


def _wrong_no_rotation_output(height: int, panel_w: int, motif: np.ndarray, fourth_pos: tuple[int, int]) -> np.ndarray:
    out = np.full((height, panel_w), BLACK, dtype=int)
    r0, c0 = fourth_pos
    size = motif.shape[0]
    out[r0 : r0 + size, c0 : c0 + size] = _rotate_n(motif, 2)
    return out


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    height: int,
    panel_w: int,
    motif: np.ndarray,
    positions: list[tuple[int, int]],
    d1: tuple[int, int],
    accel: tuple[int, int],
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False
    if accel == (0, 0):
        return False
    d1_arr = np.array(d1, dtype=int)
    d2_arr = d1_arr + np.array(accel, dtype=int)
    d3_arr = d2_arr + np.array(accel, dtype=int)
    if int(np.abs(d2_arr).sum()) <= int(np.abs(d1_arr).sum()):
        return False
    if int(np.abs(d3_arr).sum()) <= int(np.abs(d2_arr).sum()):
        return False
    if np.array_equal(expected, _wrong_constant_velocity_output(height, panel_w, motif, positions)):
        return False
    if np.array_equal(expected, _wrong_no_rotation_output(height, panel_w, motif, positions[3])):
        return False
    if len({arr.tobytes() for arr in (_rotate_n(motif, k) for k in range(4))}) != 4:
        return False
    return len(set(map(int, expected.ravel())) - {BLACK}) >= 2


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*HEIGHT_RANGE)
    panel_w = rng.randint(*PANEL_W_RANGE)
    max_size = min(MOTIF_SIZE_RANGE[1], height, panel_w)
    if max_size < MOTIF_SIZE_RANGE[0]:
        return None
    size = rng.randint(MOTIF_SIZE_RANGE[0], max_size)

    motif = _random_motif(size, rng)
    if motif is None:
        return None
    motion = _sample_motion(height, panel_w, size, rng)
    if motion is None:
        return None
    positions, d1, accel = motion

    inp = _render_input(height, panel_w, motif, positions)
    expected = _expected_output(height, panel_w, motif, positions[3])
    try:
        if not _semantic_checks(inp, expected, height, panel_w, motif, positions, d1, accel):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid accelerating extrapolation input")
