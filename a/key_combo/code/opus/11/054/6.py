from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
RING_COLORS = (1, 2, 3, 4, 7)
PANEL_BORDER = 5
CW_SLOT = 6
CCW_SLOT = 8

CORNER_POS = ((0, 0), (0, 2), (2, 2), (2, 0))
SIDE_POS = ((0, 1), (1, 2), (2, 1), (1, 0))

RING_COUNT_RANGE = (2, 4)
HEIGHT_RANGE = (11, 19)
WIDTH_RANGE = (13, 21)
ROW_GAP_MIN = 4
PANEL_GAP = 2
GEN_RETRIES = 400


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _rotate_cycle(values: list[int], clockwise: bool) -> list[int]:
    if clockwise:
        return [values[-1], *values[:-1]]
    return [*values[1:], values[0]]


def _rotate_ring_patch(patch: np.ndarray, corner_cw: bool, side_cw: bool) -> np.ndarray:
    out = patch.copy()
    corners = [int(patch[r, c]) for r, c in CORNER_POS]
    sides = [int(patch[r, c]) for r, c in SIDE_POS]
    for (r, c), value in zip(CORNER_POS, _rotate_cycle(corners, corner_cw)):
        out[r, c] = value
    for (r, c), value in zip(SIDE_POS, _rotate_cycle(sides, side_cw)):
        out[r, c] = value
    out[1, 1] = BG
    return out


def _find_panel_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey = np.argwhere(grid == PANEL_BORDER)
    assert grey.size > 0, "expected one grey legend frame"
    bbox = _bbox_from_coords(grey)
    r0, c0, r1, c1 = bbox
    frame = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.count_nonzero(grid == PANEL_BORDER) == frame.size - max(frame.shape[0] - 2, 0) * max(frame.shape[1] - 2, 0)
    assert frame.shape[0] == 4 and frame.shape[1] >= 4, "legend must be a 2-row slot panel inside a grey frame"
    assert np.all(frame[0, :] == PANEL_BORDER)
    assert np.all(frame[-1, :] == PANEL_BORDER)
    assert np.all(frame[:, 0] == PANEL_BORDER)
    assert np.all(frame[:, -1] == PANEL_BORDER)
    assert np.all(np.isin(frame[1:-1, 1:-1], (CW_SLOT, CCW_SLOT)))
    return bbox


def _decode_panel(grid: np.ndarray, panel_bbox: tuple[int, int, int, int]) -> list[tuple[bool, bool]]:
    r0, c0, r1, c1 = panel_bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert interior.shape[0] == 2
    directions = []
    for col in range(interior.shape[1]):
        upper = int(interior[0, col])
        lower = int(interior[1, col])
        assert upper in (CW_SLOT, CCW_SLOT) and lower in (CW_SLOT, CCW_SLOT)
        directions.append((upper == CW_SLOT, lower == CW_SLOT))
    return directions


def _find_rings(grid: np.ndarray) -> list[dict]:
    mask = np.isin(grid, RING_COLORS)
    rings = []
    for coords in _components4(mask):
        assert coords.shape[0] == 8, "ring must contain exactly 8 colored cells"
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        assert r1 - r0 == 2 and c1 - c0 == 2, "ring bbox must be 3x3"
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        expected = np.ones((3, 3), dtype=bool)
        expected[1, 1] = False
        assert np.array_equal(np.isin(patch, RING_COLORS), expected), "ring must be a complete 3x3 loop"
        assert int(patch[1, 1]) == BG
        rings.append({"bbox": bbox, "patch": patch.copy()})
    assert rings, "expected at least one ring"
    rings.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    top_rows = [item["bbox"][0] for item in rings]
    assert len(set(top_rows)) == len(top_rows), "ring order must be uniquely determined by top-to-bottom position"
    return rings


def _render_output_from_rings(
    grid: np.ndarray,
    rings: list[dict],
    directions: list[tuple[bool, bool]],
) -> np.ndarray:
    assert len(rings) == len(directions)
    out = np.full_like(grid, BG)
    bboxes = []
    for ring, (corner_cw, side_cw) in zip(rings, directions):
        r0, c0, r1, c1 = ring["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = _rotate_ring_patch(ring["patch"], corner_cw, side_cw)
        bboxes.append(ring["bbox"])
    r0 = min(box[0] for box in bboxes)
    c0 = min(box[1] for box in bboxes)
    r1 = max(box[2] for box in bboxes)
    c1 = max(box[3] for box in bboxes)
    return out[r0 : r1 + 1, c0 : c1 + 1]


def _inspect(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[tuple[bool, bool]], list[dict]]:
    panel_bbox = _find_panel_bbox(grid)
    directions = _decode_panel(grid, panel_bbox)
    rings = _find_rings(grid)
    assert len(directions) == len(rings), "legend columns must match the number of rings"
    return panel_bbox, directions, rings


def main(grid: np.ndarray) -> np.ndarray:
    """Decode per-ring corner/side rotation directions from the legend slots, then rotate each 3x3 ring accordingly.

    Input structure:
    - The background is black.
    - The left region contains one grey rectangular frame enclosing a 2-row slot legend; each interior column has an upper slot and a lower slot colored magenta or cyan.
    - The right region contains 2-4 separated 3x3 foreground rings made from colors {blue, red, green, yellow, orange}, each with a black center.

    Rule:
    - Sort the rings from top to bottom; legend columns correspond to rings in that order.
    - In a legend column, the upper slot controls the four corner positions of the matching ring and the lower slot controls the four side-center positions.
    - Magenta means move colors one step clockwise along that four-cycle; cyan means move colors one step counterclockwise.
    - Apply both rotations to every ring independently, then discard the legend and crop to the minimal bounding box containing all transformed rings.

    Key insight:
    - The legend does not name colors directly; it encodes two independent direction parameters per ring, one for corners and one for side-centers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    _panel_bbox, directions, rings = _inspect(g)
    return _render_output_from_rings(g, rings, directions)


def _cycle_is_nondegenerate(values: list[int]) -> bool:
    return len(set(values)) >= 2 and not (values[0] == values[2] and values[1] == values[3])


def _sample_cycle(colors: list[int]) -> list[int]:
    for _ in range(200):
        values = [random.choice(colors) for _ in range(4)]
        if _cycle_is_nondegenerate(values):
            return values
    raise RuntimeError("failed to sample a nondegenerate four-cycle")


def _sample_ring_patch() -> np.ndarray:
    for _ in range(300):
        target_unique = random.randint(2, 4)
        colors = random.sample(list(RING_COLORS), target_unique)
        corners = _sample_cycle(colors)
        sides = _sample_cycle(colors)
        used = set(corners) | set(sides)
        if len(used) != target_unique:
            continue
        patch = np.zeros((3, 3), dtype=int)
        for (r, c), value in zip(CORNER_POS, corners):
            patch[r, c] = value
        for (r, c), value in zip(SIDE_POS, sides):
            patch[r, c] = value
        if len(set(int(v) for v in patch.flat if v != BG)) != target_unique:
            continue
        return patch
    raise RuntimeError("failed to sample a valid ring patch")


def _sample_rows(height: int, count: int) -> list[int]:
    base_rows = [i * ROW_GAP_MIN for i in range(count)]
    max_last = height - 3
    slack = max_last - base_rows[-1]
    assert slack >= 0, "grid is too short for the requested ring count"
    offsets = sorted(random.randint(0, slack) for _ in range(count))
    return [base + offset for base, offset in zip(base_rows, offsets)]


def _sample_cols(width: int, panel_width: int, count: int) -> list[int]:
    start = panel_width + PANEL_GAP
    allowed = list(range(start, width - 2))
    assert len(allowed) >= count
    base = sorted(random.sample(allowed, count))
    if count == 1:
        return base
    perm = list(range(count))
    while perm == list(range(count)):
        random.shuffle(perm)
    return [base[i] for i in perm]


def _sample_directions(count: int) -> list[tuple[bool, bool]]:
    while True:
        corners = [bool(random.getrandbits(1)) for _ in range(count)]
        sides = [bool(random.getrandbits(1)) for _ in range(count)]
        if count > 1:
            if all(v == corners[0] for v in corners):
                corners[-1] = not corners[-1]
            if all(v == sides[0] for v in sides):
                sides[-1] = not sides[-1]
        pairs = list(zip(corners, sides))
        if len(set(pairs)) >= 2:
            return pairs


def _build_expected_from_specs(
    rows: list[int],
    cols: list[int],
    patches: list[np.ndarray],
    directions: list[tuple[bool, bool]],
) -> np.ndarray:
    ring_boxes = [(r, c, r + 2, c + 2) for r, c in zip(rows, cols)]
    r0 = min(box[0] for box in ring_boxes)
    c0 = min(box[1] for box in ring_boxes)
    r1 = max(box[2] for box in ring_boxes)
    c1 = max(box[3] for box in ring_boxes)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for row, col, patch, (corner_cw, side_cw) in zip(rows, cols, patches, directions):
        rotated = _rotate_ring_patch(patch, corner_cw, side_cw)
        out[row - r0 : row - r0 + 3, col - c0 : col - c0 + 3] = rotated
    return out


def _build_input(
    height: int,
    width: int,
    panel_top: int,
    rows: list[int],
    cols: list[int],
    patches: list[np.ndarray],
    directions: list[tuple[bool, bool]],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)

    panel_width = len(directions) + 2
    grid[panel_top, :panel_width] = PANEL_BORDER
    grid[panel_top + 3, :panel_width] = PANEL_BORDER
    grid[panel_top : panel_top + 4, 0] = PANEL_BORDER
    grid[panel_top : panel_top + 4, panel_width - 1] = PANEL_BORDER
    for idx, (corner_cw, side_cw) in enumerate(directions, start=1):
        grid[panel_top + 1, idx] = CW_SLOT if corner_cw else CCW_SLOT
        grid[panel_top + 2, idx] = CW_SLOT if side_cw else CCW_SLOT

    for row, col, patch in zip(rows, cols, patches):
        grid[row : row + 3, col : col + 3] = patch
    return grid


def _validate_generated(
    grid: np.ndarray,
    expected: np.ndarray,
    rows: list[int],
    cols: list[int],
    directions: list[tuple[bool, bool]],
) -> bool:
    try:
        panel_bbox, decoded, rings = _inspect(grid)
    except AssertionError:
        return False

    if panel_bbox[1] != 0:
        return False
    if decoded != directions:
        return False
    if not np.array_equal(main(grid), expected):
        return False

    if len(set(cols)) != len(cols):
        return False
    col_order = sorted(range(len(cols)), key=lambda i: cols[i])
    if col_order == list(range(len(cols))):
        return False

    if len(set(directions)) < 2:
        return False

    for ring in rings:
        patch = ring["patch"]
        corners = [int(patch[r, c]) for r, c in CORNER_POS]
        sides = [int(patch[r, c]) for r, c in SIDE_POS]
        if not (_cycle_is_nondegenerate(corners) and _cycle_is_nondegenerate(sides)):
            return False

    for idx in range(len(directions)):
        flipped = directions.copy()
        corner_cw, side_cw = flipped[idx]
        flipped[idx] = (not corner_cw, side_cw)
        if np.array_equal(_render_output_from_rings(grid, rings, flipped), expected):
            return False
        flipped[idx] = (corner_cw, not side_cw)
        if np.array_equal(_render_output_from_rings(grid, rings, flipped), expected):
            return False

    wrong_mapping = [directions[i] for i in col_order]
    if np.array_equal(_render_output_from_rings(grid, rings, wrong_mapping), expected):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    ring_count = random.randint(*RING_COUNT_RANGE)
    panel_width = ring_count + 2

    min_height = max(HEIGHT_RANGE[0], 3 + ROW_GAP_MIN * (ring_count - 1))
    min_width = max(WIDTH_RANGE[0], panel_width + PANEL_GAP + ring_count + 2)
    height = random.randint(min_height, HEIGHT_RANGE[1])
    width = random.randint(min_width, WIDTH_RANGE[1])

    rows = _sample_rows(height, ring_count)
    cols = _sample_cols(width, panel_width, ring_count)
    patches = [_sample_ring_patch() for _ in range(ring_count)]
    directions = _sample_directions(ring_count)
    panel_top = random.randint(0, height - 4)

    grid = _build_input(height, width, panel_top, rows, cols, patches, directions)
    expected = _build_expected_from_specs(rows, cols, patches, directions)
    if not _validate_generated(grid, expected, rows, cols, directions):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-controlled ring-cycling input")
