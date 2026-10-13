import random
from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 300
MIN_SLOTS = 2
MAX_SLOTS = 4
MIN_GRID = 18
MAX_GRID = 30
SHAPE_MIN_CELLS = 4
SHAPE_MAX_CELLS = 7
SHAPE_MAX_H = 4
SHAPE_MAX_W = 5
TEMPLATE_GAP_CHOICES = (2, 3)
PIECE_PADDING = 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...] | set[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    pts = list(coords)
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    rel = tuple(sorted((r - r0, c - c0) for r, c in pts))
    h = max(r for r, _ in rel) + 1
    w = max(c for _, c in rel) + 1
    return rel, h, w


def _rot90(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    _, h, _ = _normalize(coords)
    rotated = [(c, h - 1 - r) for r, c in coords]
    rel, _, _ = _normalize(rotated)
    return rel


def _rotations(coords: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    cur = coords
    out: list[tuple[tuple[int, int], ...]] = []
    for _ in range(4):
        out.append(cur)
        cur = _rot90(cur)
    unique: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for shape in out:
        if shape not in seen:
            seen.add(shape)
            unique.append(shape)
    return unique


def _canonical_shape(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return min(_rotations(coords))


def _build_slot_patch(shape: tuple[tuple[int, int], ...], color: int) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    rel, h, w = _normalize(shape)
    hole = tuple(sorted((r + 1, c + 1) for r, c in rel))
    hole_set = set(hole)
    patch = np.zeros((h + 2, w + 2), dtype=int)
    for r, c in hole:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in hole_set:
                patch[nr, nc] = color
    return patch, hole


def _build_piece_patch(shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    rel, h, w = _normalize(shape)
    patch = np.zeros((h, w), dtype=int)
    for r, c in rel:
        patch[r, c] = color
    return patch


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _extract_slots(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    slots: list[dict] = []
    for comp in _components(grid == 0):
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in comp):
            continue
        adj = set()
        for r, c in comp:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                val = int(grid[nr, nc])
                if val != 0:
                    adj.add(val)
        if len(adj) != 1:
            continue
        rel, _, _ = _normalize(comp)
        slots.append(
            {
                "coords": tuple(sorted(comp)),
                "shape": rel,
                "color": next(iter(adj)),
            }
        )
    return slots


def _extract_pieces(grid: np.ndarray, slot_colors: set[int]) -> list[dict]:
    pieces: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) not in slot_colors and int(c) != 0):
        for comp in _components(grid == color):
            rel, _, _ = _normalize(comp)
            pieces.append(
                {
                    "coords": tuple(sorted(comp)),
                    "shape": rel,
                    "rotations": _rotations(rel),
                    "color": color,
                }
            )
    return pieces


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    slots = _extract_slots(g)
    assert slots, "expected at least one slot cavity"
    slot_colors = {slot["color"] for slot in slots}
    pieces = _extract_pieces(g, slot_colors)
    assert len(pieces) == len(slots), "piece and slot counts must match"

    out = g.copy()
    for piece in pieces:
        for r, c in piece["coords"]:
            out[r, c] = 0

    remaining = {idx: slot for idx, slot in enumerate(slots)}
    for piece in pieces:
        matches = []
        for idx, slot in remaining.items():
            if slot["shape"] in piece["rotations"]:
                matches.append(idx)
        assert len(matches) == 1, "each piece must match exactly one slot"
        slot = remaining.pop(matches[0])
        for r, c in slot["coords"]:
            out[r, c] = slot["color"]

    assert not remaining, "all slots must be filled"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match scattered shape fragments to template cavities and insert them with the cavity border color.

    Input structure:
    - Background is black.
    - Near the grid center sit several colored one-cell-thick outlines whose interiors are black cavity shapes.
    - Each cavity is bordered by exactly one color, and different cavities use different border colors.
    - Outside the template area are monochrome filled fragments whose binary shapes match the cavities up to 90-degree rotation.

    Rule:
    - Find every black connected component that does not touch the grid boundary and is adjacent to exactly one non-black color; treat it as a cavity slot.
    - Extract every outside monochrome fragment whose color is not used by a cavity border.
    - Compare each fragment shape against every cavity shape under 0/90/180/270-degree rotation and find the unique match.
    - Clear the fragment from its original position, then fill the matched cavity with the cavity border color.

    Key insight:
    - Fragment colors are irrelevant noise: shape decides the destination slot, while the slot's visible border decides the final color.
    """
    return _solve(np.asarray(grid, dtype=int))


def _random_shape(rng: random.Random) -> tuple[tuple[int, int], ...] | None:
    target = rng.randint(SHAPE_MIN_CELLS, SHAPE_MAX_CELLS)
    cells = {(0, 0)}
    frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
    while len(cells) < target:
        options = []
        for cand in frontier:
            trial = cells | {cand}
            rel, h, w = _normalize(trial)
            if h <= SHAPE_MAX_H and w <= SHAPE_MAX_W:
                options.append((cand, rel, h, w))
        if not options:
            return None
        cand, rel, h, w = rng.choice(options)
        cells.add(cand)
        frontier.remove(cand)
        for dr, dc in DIR4:
            nxt = (cand[0] + dr, cand[1] + dc)
            if nxt not in cells:
                frontier.add(nxt)
    rel, h, w = _normalize(cells)
    if h < 2 or w < 2:
        return None
    if len(rel) == h * w and rng.random() < 0.7:
        return None
    return rel


def _choose_shapes(rng: random.Random, count: int) -> list[tuple[tuple[int, int], ...]] | None:
    shapes: list[tuple[tuple[int, int], ...]] = []
    sigs: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(200):
        shape = _random_shape(rng)
        if shape is None:
            continue
        sig = _canonical_shape(shape)
        if sig in sigs:
            continue
        sigs.add(sig)
        shapes.append(shape)
        if len(shapes) == count:
            return shapes
    return None


def _layout_slots(
    rng: random.Random,
    patches: list[np.ndarray],
) -> tuple[list[tuple[int, int]], int, int] | None:
    n = len(patches)
    layouts = [(1, n), (n, 1)]
    if n >= 3:
        layouts.append((2, 2))
    rows, cols = rng.choice(layouts)
    if rows * cols < n:
        return None

    cells = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(cells)
    chosen = cells[:n]
    assignment = list(zip(range(n), chosen))

    row_heights = [0] * rows
    col_widths = [0] * cols
    for idx, (rr, cc) in assignment:
        ph, pw = patches[idx].shape
        row_heights[rr] = max(row_heights[rr], ph)
        col_widths[cc] = max(col_widths[cc], pw)

    gap = rng.choice(TEMPLATE_GAP_CHOICES)
    template_h = sum(row_heights) + gap * (rows - 1)
    template_w = sum(col_widths) + gap * (cols - 1)
    starts_r = []
    cur = 0
    for size in row_heights:
        starts_r.append(cur)
        cur += size + gap
    starts_c = []
    cur = 0
    for size in col_widths:
        starts_c.append(cur)
        cur += size + gap

    positions = [(0, 0)] * n
    for idx, (rr, cc) in assignment:
        ph, pw = patches[idx].shape
        slack_r = row_heights[rr] - ph
        slack_c = col_widths[cc] - pw
        offset_r = rng.randint(0, slack_r) if slack_r else 0
        offset_c = rng.randint(0, slack_c) if slack_c else 0
        positions[idx] = (starts_r[rr] + offset_r, starts_c[cc] + offset_c)
    return positions, template_h, template_w


def _render_slot_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
) -> None:
    ph, pw = patch.shape
    sub = grid[top : top + ph, left : left + pw]
    mask = patch != 0
    assert not np.any(sub[mask] != 0)
    sub[mask] = patch[mask]


def _candidate_piece_positions(
    grid: np.ndarray,
    patch: np.ndarray,
    forbidden: tuple[int, int, int, int],
    existing: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    h, w = grid.shape
    ph, pw = patch.shape
    positions: list[tuple[int, int]] = []
    for top in range(1, h - ph):
        for left in range(1, w - pw):
            box = (top, left, top + ph - 1, left + pw - 1)
            if _boxes_overlap(box, forbidden, pad=2):
                continue
            if any(_boxes_overlap(box, other, pad=PIECE_PADDING) for other in existing):
                continue
            sub = grid[top : top + ph, left : left + pw]
            if np.any((patch != 0) & (sub != 0)):
                continue
            positions.append((top, left))
    return positions


def _expected_output(
    grid: np.ndarray,
    slot_specs: list[dict],
    piece_cells: list[list[tuple[int, int]]],
) -> np.ndarray:
    out = grid.copy()
    for cells in piece_cells:
        for r, c in cells:
            out[r, c] = 0
    for slot in slot_specs:
        for r, c in slot["abs_hole"]:
            out[r, c] = slot["color"]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    slot_count = rng.randint(MIN_SLOTS, MAX_SLOTS)
    shapes = _choose_shapes(rng, slot_count)
    if shapes is None:
        return None

    colors = list(range(1, 10))
    rng.shuffle(colors)
    if len(colors) < 2 * slot_count:
        return None
    slot_colors = colors[:slot_count]
    piece_colors = colors[slot_count : 2 * slot_count]

    slot_patches: list[np.ndarray] = []
    slot_holes: list[tuple[tuple[int, int], ...]] = []
    rotated_shapes: list[tuple[tuple[int, int], ...]] = []
    piece_patches: list[np.ndarray] = []
    for shape, slot_color, piece_color in zip(shapes, slot_colors, piece_colors):
        slot_patch, hole = _build_slot_patch(shape, slot_color)
        slot_patches.append(slot_patch)
        slot_holes.append(hole)
        rotated = shape
        for _ in range(rng.randrange(4)):
            rotated = _rot90(rotated)
        rotated_shapes.append(rotated)
        piece_patches.append(_build_piece_patch(rotated, piece_color))

    layout = _layout_slots(rng, slot_patches)
    if layout is None:
        return None
    slot_positions, template_h, template_w = layout

    min_h = max(MIN_GRID, template_h + 10)
    min_w = max(MIN_GRID, template_w + 10)
    if min_h > MAX_GRID or min_w > MAX_GRID:
        return None
    grid_h = rng.randint(min_h, MAX_GRID)
    grid_w = rng.randint(min_w, MAX_GRID)

    center_top = (grid_h - template_h) // 2
    center_left = (grid_w - template_w) // 2
    top_lo = max(1, center_top - 1)
    top_hi = min(grid_h - template_h - 1, center_top + 1)
    left_lo = max(1, center_left - 1)
    left_hi = min(grid_w - template_w - 1, center_left + 1)
    if top_lo > top_hi or left_lo > left_hi:
        return None
    template_top = rng.randint(top_lo, top_hi)
    template_left = rng.randint(left_lo, left_hi)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    slot_specs: list[dict] = []
    for patch, hole, color, (dr, dc) in zip(slot_patches, slot_holes, slot_colors, slot_positions):
        top = template_top + dr
        left = template_left + dc
        _render_slot_patch(grid, patch, top, left)
        abs_hole = [(top + r, left + c) for r, c in hole]
        slot_specs.append(
            {
                "color": color,
                "shape": _normalize(abs_hole)[0],
                "abs_hole": tuple(sorted(abs_hole)),
            }
        )

    template_box = (template_top, template_left, template_top + template_h - 1, template_left + template_w - 1)
    piece_indices = list(range(slot_count))
    rng.shuffle(piece_indices)
    piece_boxes: list[tuple[int, int, int, int]] = []
    piece_cells: list[list[tuple[int, int]]] = [[] for _ in range(slot_count)]
    for idx in sorted(piece_indices, key=lambda k: -int(np.count_nonzero(piece_patches[k]))):
        patch = piece_patches[idx]
        candidates = _candidate_piece_positions(grid, patch, template_box, piece_boxes)
        if not candidates:
            return None
        top, left = rng.choice(candidates)
        ph, pw = patch.shape
        sub = grid[top : top + ph, left : left + pw]
        mask = patch != 0
        sub[mask] = patch[mask]
        coords = [(top + r, left + c) for r, c in zip(*np.where(mask))]
        piece_cells[idx] = coords
        piece_boxes.append((top, left, top + ph - 1, left + pw - 1))

    expected = _expected_output(grid, slot_specs, piece_cells)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_RETRIES} attempts")
