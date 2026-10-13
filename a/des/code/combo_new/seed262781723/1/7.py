from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIR_FROM_MISSING = {
    (0, 0): "up",
    (0, 1): "right",
    (1, 0): "left",
    (1, 1): "down",
}
MISSING_FROM_DIR = {direction: corner for corner, direction in DIR_FROM_MISSING.items()}
VALID_DIRS = {
    (0, 0): ("right", "down"),
    (0, 1): ("left", "down"),
    (1, 0): ("up", "right"),
    (1, 1): ("up", "left"),
}
MAX_GENERATION_ATTEMPTS = 400


def _quadrant_slices(qh: int, qw: int, qr: int, qc: int) -> tuple[slice, slice]:
    return slice(qr * qh, (qr + 1) * qh), slice(qc * qw, (qc + 1) * qw)


def _marker_info(coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if coords.shape[0] != 3:
        return None
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    if r1 - r0 != 1 or c1 - c0 != 1:
        return None
    present = {(int(r - r0), int(c - c0)) for r, c in coords}
    missing = {(0, 0), (0, 1), (1, 0), (1, 1)} - present
    if len(missing) != 1:
        return None
    return (r0, c0), next(iter(missing))


def _find_marker(g: np.ndarray) -> tuple[int, tuple[int, int], str]:
    h, w = g.shape
    qh, qw = h // 2, w // 2
    candidates: list[tuple[int, tuple[int, int], str]] = []
    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue
        coords = np.argwhere(g == color)
        info = _marker_info(coords)
        if info is None:
            continue
        _, missing = info
        qr = int(coords[0, 0] // qh)
        qc = int(coords[0, 1] // qw)
        if not np.all(coords[:, 0] // qh == qr) or not np.all(coords[:, 1] // qw == qc):
            continue
        direction = DIR_FROM_MISSING[missing]
        if direction not in VALID_DIRS[(qr, qc)]:
            continue
        candidates.append((color, (qr, qc), direction))
    assert len(candidates) == 1, "expected exactly one valid 3-pixel L marker"
    return candidates[0]


def _target_quadrant(source_quad: tuple[int, int], direction: str) -> tuple[int, int]:
    qr, qc = source_quad
    if direction == "up":
        return qr - 1, qc
    if direction == "down":
        return qr + 1, qc
    if direction == "left":
        return qr, qc - 1
    if direction == "right":
        return qr, qc + 1
    raise ValueError(direction)


def _fold_local(r: int, c: int, qh: int, qw: int, direction: str) -> tuple[int, int]:
    if direction in ("left", "right"):
        return r, qw - 1 - c
    return qh - 1 - r, c


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h % 2 == 0 and w % 2 == 0
    qh, qw = h // 2, w // 2

    marker_color, source_quad, direction = _find_marker(g)
    assert marker_color != BG
    target_quad = _target_quadrant(source_quad, direction)
    assert target_quad in VALID_DIRS

    src_rows, src_cols = _quadrant_slices(qh, qw, *source_quad)
    tgt_rows, tgt_cols = _quadrant_slices(qh, qw, *target_quad)
    source = g[src_rows, src_cols]
    target = g[tgt_rows, tgt_cols]

    out = g.copy()
    for r0, c0 in np.argwhere(source != BG):
        r = int(r0)
        c = int(c0)
        rr, cc = _fold_local(r, c, qh, qw, direction)
        gr = tgt_rows.start + rr
        gc = tgt_cols.start + cc
        if out[gr, gc] == BG:
            out[gr, gc] = int(source[r, c])

    out[src_rows, src_cols] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the quadrant marked by a 3-pixel L into the adjacent quadrant indicated by the L's missing corner.

    Input structure:
    - The background is black, and the grid has even height and even width so it splits exactly into four equal quadrants.
    - Exactly one quadrant contains a uniquely colored 3-pixel L marker occupying three cells of a 2x2 block.
    - The L marker's color is used nowhere else; the remaining quadrants contain other non-black patterns.

    Rule:
    - Find the unique L marker, identify its quadrant, and decode the missing corner as the fold direction using the compass cycle: top-left=up, top-right=right, bottom-right=down, bottom-left=left.
    - Mirror the entire marked quadrant, including the L itself, across the corresponding middle fold line into the adjacent quadrant in that direction.
    - When a folded pixel lands on an already colored target pixel, keep the target quadrant's original non-black color; otherwise copy the folded pixel. Then clear the source quadrant to black.

    Key insight:
    - The same L both selects the source quadrant and encodes which middle axis to fold across, so solving the task requires reading a tiny local shape as a global fold instruction.
    """
    return _solve(grid)


def _grow_connected_mask(box_h: int, box_w: int, area: int) -> np.ndarray | None:
    seed = (random.randrange(box_h), random.randrange(box_w))
    cells = {seed}
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(*seed)
    while len(cells) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        add_frontier(*cell)
    if len(cells) != area:
        return None

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _random_shape(max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 2 or max_w < 2:
        return None
    for _ in range(50):
        box_h = random.randint(2, min(5, max_h))
        box_w = random.randint(2, min(5, max_w))
        area = random.randint(4, min(7, box_h * box_w))
        mask = _grow_connected_mask(box_h, box_w, area)
        if mask is not None and mask.sum() >= 4:
            return mask
    return None


def _place_mask(
    quadrant: np.ndarray,
    mask: np.ndarray,
    color: int,
    *,
    must_intersect: set[tuple[int, int]] | None = None,
    avoid: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    qh, qw = quadrant.shape
    mh, mw = mask.shape
    positions = [(r0, c0) for r0 in range(qh - mh + 1) for c0 in range(qw - mw + 1)]
    random.shuffle(positions)
    mask_cells = np.argwhere(mask)
    for r0, c0 in positions:
        coords = [(r0 + int(r), c0 + int(c)) for r, c in mask_cells]
        if any(int(quadrant[r, c]) != BG for r, c in coords):
            continue
        if must_intersect is not None and not any((r, c) in must_intersect for r, c in coords):
            continue
        if avoid is not None and any((r, c) in avoid for r, c in coords):
            continue
        for r, c in coords:
            quadrant[r, c] = color
        return coords
    return None


def _marker_mask(direction: str) -> np.ndarray:
    missing = MISSING_FROM_DIR[direction]
    mask = np.ones((2, 2), dtype=bool)
    mask[missing] = False
    return mask


def _collect_folded_cells(source_quad: np.ndarray, direction: str) -> set[tuple[int, int]]:
    qh, qw = source_quad.shape
    folded = set()
    for r0, c0 in np.argwhere(source_quad != BG):
        rr, cc = _fold_local(int(r0), int(c0), qh, qw, direction)
        folded.add((rr, cc))
    return folded


def _render_full_grid(quads: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    any_quad = next(iter(quads.values()))
    qh, qw = any_quad.shape
    grid = np.full((qh * 2, qw * 2), BG, dtype=int)
    for (qr, qc), quad in quads.items():
        rows, cols = _quadrant_slices(qh, qw, qr, qc)
        grid[rows, cols] = quad
    return grid


def _shape_count_for_quadrant(qh: int, qw: int, *, source: bool) -> int:
    area = qh * qw
    if source:
        return 2 if area >= 64 and random.random() < 0.5 else 1
    return 2 if area >= 81 and random.random() < 0.3 else 1


def _gen_attempt() -> np.ndarray | None:
    qh = random.randint(5, 15)
    qw = random.randint(5, 15)
    source_quad = random.choice(tuple(VALID_DIRS))
    direction = random.choice(VALID_DIRS[source_quad])
    target_quad = _target_quadrant(source_quad, direction)
    overlap_mode = random.random() < 0.5

    quads = {(qr, qc): np.full((qh, qw), BG, dtype=int) for qr in range(2) for qc in range(2)}
    counts = {quad: _shape_count_for_quadrant(qh, qw, source=(quad == source_quad)) for quad in quads}
    total_objects = 1 + sum(counts.values())
    if total_objects > len(PALETTE):
        return None
    colors = random.sample(PALETTE, total_objects)
    color_idx = 0

    marker_color = colors[color_idx]
    color_idx += 1
    if _place_mask(quads[source_quad], _marker_mask(direction), marker_color) is None:
        return None

    for _ in range(counts[source_quad]):
        mask = _random_shape(qh, qw)
        if mask is None:
            return None
        if _place_mask(quads[source_quad], mask, colors[color_idx]) is None:
            return None
        color_idx += 1

    folded_cells = _collect_folded_cells(quads[source_quad], direction)
    if len(folded_cells) < 4:
        return None

    target_quad_array = quads[target_quad]
    for i in range(counts[target_quad]):
        mask = _random_shape(qh, qw)
        if mask is None:
            return None
        must_intersect = folded_cells if overlap_mode and i == 0 else None
        avoid = folded_cells if not overlap_mode else (folded_cells if i > 0 and random.random() < 0.7 else None)
        placed = _place_mask(
            target_quad_array,
            mask,
            colors[color_idx],
            must_intersect=must_intersect,
            avoid=avoid,
        )
        if placed is None:
            return None
        color_idx += 1

    for quad in quads:
        if quad in (source_quad, target_quad):
            continue
        for _ in range(counts[quad]):
            mask = _random_shape(qh, qw)
            if mask is None:
                return None
            if _place_mask(quads[quad], mask, colors[color_idx]) is None:
                return None
            color_idx += 1

    inp = _render_full_grid(quads)
    out = _solve(inp)

    src_rows, src_cols = _quadrant_slices(qh, qw, *source_quad)
    tgt_rows, tgt_cols = _quadrant_slices(qh, qw, *target_quad)
    source_non_bg = int(np.count_nonzero(inp[src_rows, src_cols]))
    target_changes = int(np.count_nonzero(inp[tgt_rows, tgt_cols] != out[tgt_rows, tgt_cols]))
    overlap_count = sum(1 for r, c in folded_cells if int(target_quad_array[r, c]) != BG)

    if source_non_bg < 4:
        return None
    if target_changes == 0:
        return None
    if overlap_mode and overlap_count == 0:
        return None
    if not overlap_mode and overlap_count != 0:
        return None
    if not np.all(out[src_rows, src_cols] == BG):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fold-quadrant ARC input")
