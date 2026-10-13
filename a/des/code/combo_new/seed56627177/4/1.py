import numpy as np


BG_COLOR = 0
FRAME_COLOR = 5
PATTERN_COLOR = 2
FRAGMENT_COLORS = [1, 3, 4, 6, 7, 8, 9]
GAP_SIDES = ("top", "right", "bottom", "left")
ROTATION_K = {"top": 0, "right": 3, "bottom": 2, "left": 1}
FRAME_SIZES = [7, 9, 11, 13]
MAX_GENERATION_ATTEMPTS = 500

RNG = np.random.default_rng()


def _majority_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel())
    return int(np.argmax(counts))


def _find_frame(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray, str]:
    points = np.argwhere(grid == FRAME_COLOR)
    assert len(points) > 0, "frame not found"

    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    side_h = int(r1 - r0 + 1)
    side_w = int(c1 - c0 + 1)
    assert side_h == side_w, "frame must be square"

    frame_mask = grid[r0 : r1 + 1, c0 : c1 + 1] == FRAME_COLOR
    gap_candidates = []
    edge_slices = {
        "top": frame_mask[0, 1:-1],
        "right": frame_mask[1:-1, -1],
        "bottom": frame_mask[-1, 1:-1],
        "left": frame_mask[1:-1, 0],
    }
    for side, edge in edge_slices.items():
        missing = np.flatnonzero(~edge)
        if len(missing) == 0:
            continue
        assert np.all(np.diff(missing) == 1), "gap must be contiguous"
        gap_candidates.append(side)

    assert len(gap_candidates) == 1, "expected exactly one gapped side"
    return (int(r0), int(c0), int(r1), int(c1)), frame_mask, gap_candidates[0]


def _extract_pattern(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = frame_bbox
    rows, cols = np.where(grid[r0 + 1 : r1, c0 + 1 : c1] == PATTERN_COLOR)
    assert len(rows) > 0, "pattern not found"

    pr0 = int(r0 + 1 + rows.min())
    pc0 = int(c0 + 1 + cols.min())
    pr1 = int(r0 + 1 + rows.max())
    pc1 = int(c0 + 1 + cols.max())
    pattern = grid[pr0 : pr1 + 1, pc0 : pc1 + 1] == PATTERN_COLOR
    return (pr0, pc0, pr1, pc1), pattern


def _collect_fragment_colors(grid: np.ndarray, bg: int, frame_bbox: tuple[int, int, int, int]) -> list[int]:
    r0, c0, r1, c1 = frame_bbox
    fragments = []
    for r, c in np.argwhere((grid != bg) & (grid != FRAME_COLOR) & (grid != PATTERN_COLOR)):
        if r0 <= int(r) <= r1 and c0 <= int(c) <= c1:
            continue
        fragments.append((int(r), int(c), int(grid[r, c])))
    fragments.sort(key=lambda item: (item[0], item[1]))
    return [color for _, _, color in fragments]


def _render_output(
    shape: tuple[int, int],
    bg: int,
    frame_bbox: tuple[int, int, int, int],
    frame_mask: np.ndarray,
    pattern_bbox: tuple[int, int, int, int],
    pattern_mask: np.ndarray,
    gap_side: str,
    ordered_colors: list[int],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)

    r0, c0, r1, c1 = frame_bbox
    frame_view = out[r0 : r1 + 1, c0 : c1 + 1]
    frame_view[frame_mask] = FRAME_COLOR

    rotated = np.rot90(pattern_mask, k=ROTATION_K[gap_side])
    positions = [tuple(map(int, pos)) for pos in np.argwhere(rotated)]
    assert len(positions) == len(ordered_colors), "fragment count must match template seats"

    pr0, pc0, _, _ = pattern_bbox
    for color, (dr, dc) in zip(ordered_colors, positions):
        out[pr0 + dr, pc0 + dc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate the interior template from the frame gap direction, then seat outside color fragments into it by scan order.

    Input structure:
    - The background is black, and a single grey one-pixel-wide square frame sits near the center.
    - Exactly one frame side has a contiguous centered gap, while a small red binary pattern appears inside the frame on the side opposite that gap.
    - Outside the frame are scattered non-grey, non-red single-pixel color fragments whose count matches the number of red pattern cells.

    Rule:
    - Detect which frame side contains the gap.
    - Crop the interior red pattern and rotate it according to the gap direction: top=0 degrees, right=90 degrees clockwise, bottom=180 degrees, left=90 degrees counterclockwise.
    - Read the outside color fragments from top to bottom then left to right, and place them in that order onto the rotated template's foreground cells, also traversed in top-to-bottom then left-to-right order.
    - Return only the original grey frame with its gap preserved plus the newly seated colored pixels; remove the original red template and all scattered outside fragments.

    Key insight:
    - The gap direction determines the template rotation, and the rotated template acts as a seat map that is filled by the outside fragments in row-major order.
    """
    g = np.asarray(grid, dtype=int)
    bg = _majority_color(g)

    frame_bbox, frame_mask, gap_side = _find_frame(g)
    pattern_bbox, pattern_mask = _extract_pattern(g, frame_bbox)
    ordered_colors = _collect_fragment_colors(g, bg, frame_bbox)

    return _render_output(
        g.shape,
        bg,
        frame_bbox,
        frame_mask,
        pattern_bbox,
        pattern_mask,
        gap_side,
        ordered_colors,
    )


def _random_pattern(box_side: int, rng: np.random.Generator) -> np.ndarray | None:
    max_cells = min(len(FRAGMENT_COLORS), box_side * box_side)
    for _ in range(300):
        cell_count = int(rng.integers(4, max_cells + 1))
        choices = rng.choice(box_side * box_side, size=cell_count, replace=False)
        mask = np.zeros((box_side, box_side), dtype=bool)
        mask.reshape(-1)[choices] = True

        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if len({np.rot90(mask, k).tobytes() for k in range(4)}) != 4:
            continue
        return mask
    return None


def _gap_length(frame_side: int, rng: np.random.Generator) -> int:
    choices = [length for length in (1, 3, 5) if length <= frame_side - 4]
    return int(rng.choice(choices))


def _pattern_top_left(
    frame_bbox: tuple[int, int, int, int],
    box_side: int,
    gap_side: str,
    rng: np.random.Generator,
) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    inner_side = r1 - r0 - 1
    row_offset = int(rng.integers(0, inner_side - box_side + 1))
    col_offset = int(rng.integers(0, inner_side - box_side + 1))

    if gap_side == "top":
        return r1 - box_side, c0 + 1 + col_offset
    if gap_side == "bottom":
        return r0 + 1, c0 + 1 + col_offset
    if gap_side == "left":
        return r0 + 1 + row_offset, c1 - box_side
    return r0 + 1 + row_offset, c0 + 1


def _scatter_fragments(
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    h, w = shape
    r0, c0, r1, c1 = frame_bbox
    blocked = np.zeros((h, w), dtype=bool)
    blocked[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True

    candidates = [(r, c) for r in range(h) for c in range(w) if not blocked[r, c]]
    rng.shuffle(candidates)

    picked: list[tuple[int, int]] = []
    for r, c in candidates:
        if any(abs(r - pr) <= 1 and abs(c - pc) <= 1 for pr, pc in picked):
            continue
        picked.append((r, c))
        if len(picked) == count:
            return picked
    return None


def _draw_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], gap_side: str, gap_len: int) -> np.ndarray:
    r0, c0, r1, c1 = frame_bbox
    side = r1 - r0 + 1
    local = np.zeros((side, side), dtype=bool)
    local[0, :] = True
    local[-1, :] = True
    local[:, 0] = True
    local[:, -1] = True

    start = (side - gap_len) // 2
    end = start + gap_len
    if gap_side == "top":
        local[0, start:end] = False
    elif gap_side == "right":
        local[start:end, -1] = False
    elif gap_side == "bottom":
        local[-1, start:end] = False
    else:
        local[start:end, 0] = False

    view = grid[r0 : r1 + 1, c0 : c1 + 1]
    view[local] = FRAME_COLOR
    return local


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    gap_side = GAP_SIDES[int(rng.integers(len(GAP_SIDES)))]
    box_side = int(rng.integers(3, 6))
    pattern_mask = _random_pattern(box_side, rng)
    if pattern_mask is None:
        return None

    frame_choices = [size for size in FRAME_SIZES if size >= box_side + 3]
    frame_side = int(rng.choice(frame_choices))
    grid_h = int(rng.integers(frame_side + 6, min(31, frame_side + 15)))
    grid_w = int(rng.integers(frame_side + 6, min(31, frame_side + 15)))

    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    frame_r0 = (grid_h - frame_side) // 2
    frame_c0 = (grid_w - frame_side) // 2
    frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_side - 1, frame_c0 + frame_side - 1)

    gap_len = _gap_length(frame_side, rng)
    frame_mask = _draw_frame(grid, frame_bbox, gap_side, gap_len)

    pattern_r0, pattern_c0 = _pattern_top_left(frame_bbox, box_side, gap_side, rng)
    pattern_bbox = (
        pattern_r0,
        pattern_c0,
        pattern_r0 + box_side - 1,
        pattern_c0 + box_side - 1,
    )
    grid[pattern_r0 : pattern_r0 + box_side, pattern_c0 : pattern_c0 + box_side][pattern_mask] = PATTERN_COLOR

    fragment_count = int(pattern_mask.sum())
    fragment_positions = _scatter_fragments(grid.shape, frame_bbox, fragment_count, rng)
    if fragment_positions is None:
        return None

    fragment_colors = list(map(int, rng.choice(FRAGMENT_COLORS, size=fragment_count, replace=False)))
    rng.shuffle(fragment_colors)
    for (r, c), color in zip(fragment_positions, fragment_colors):
        grid[r, c] = color

    ordered_colors = [
        color
        for _, _, color in sorted(
            [(r, c, color) for (r, c), color in zip(fragment_positions, fragment_colors)],
            key=lambda item: (item[0], item[1]),
        )
    ]
    expected = _render_output(
        grid.shape,
        BG_COLOR,
        frame_bbox,
        frame_mask,
        pattern_bbox,
        pattern_mask,
        gap_side,
        ordered_colors,
    )

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
