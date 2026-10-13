import numpy as np

from arc_utils import crop_bbox, find_components, get_bg, paste
import random


def _is_hollow_frame(patch: np.ndarray, bg: int, color: int) -> bool:
    h, w = patch.shape
    if h < 3 or w < 3:
        return False
    return (
        np.all(patch[0, :] == color)
        and np.all(patch[-1, :] == color)
        and np.all(patch[:, 0] == color)
        and np.all(patch[:, -1] == color)
        and np.all(patch[1:-1, 1:-1] == bg)
    )


def main(grid: np.ndarray) -> np.ndarray:
    """Pack the hollow square frames into a contiguous strip ordered by their original positions.

    Input structure
    - A black-background scene of separated monochrome 4x4 objects: hollow square frames and solid square distractors.
    Rule
    - Keep only the hollow frames and paste them edge-to-edge into one strip.
    - If the hollow-frame positions spread more by column than row, pack left-to-right ordered by column; otherwise pack top-to-bottom ordered by row.
    Key insight
    - The solid blocks are distractors; the hollow frames are the retained sprites, and their dominant layout axis determines the output strip.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frames: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        color = int(comp["color"])
        patch = crop_bbox(g, comp["bbox"])
        is_solid = bool(np.all(patch == color))
        is_frame = _is_hollow_frame(patch, bg, color)
        assert is_solid or is_frame
        if is_frame:
            frames.append((comp["bbox"], patch))

    assert len(frames) >= 2
    shapes = {patch.shape for _bbox, patch in frames}
    assert len(shapes) == 1
    frame_h, frame_w = next(iter(shapes))

    rows = [bbox[0] for bbox, _patch in frames]
    cols = [bbox[1] for bbox, _patch in frames]
    row_span = max(rows) - min(rows)
    col_span = max(cols) - min(cols)
    assert row_span != col_span

    if col_span > row_span:
        assert len(set(cols)) == len(cols)
        ordered = sorted(frames, key=lambda item: item[0][1])
        out = np.full((frame_h, frame_w * len(ordered)), bg, dtype=int)
        for k, (_bbox, patch) in enumerate(ordered):
            paste(out, patch, 0, k * frame_w)
    else:
        assert len(set(rows)) == len(rows)
        ordered = sorted(frames, key=lambda item: item[0][0])
        out = np.full((frame_h * len(ordered), frame_w), bg, dtype=int)
        for k, (_bbox, patch) in enumerate(ordered):
            paste(out, patch, k * frame_h, 0)

    return out

# --------------- generator (o5x) ---------------

# Canvas and object geometry.
BG = 0
OBJECT_SIZE = 4
OBJECT_GAP = 1
GRID_HEIGHT_RANGE = (18, 24)
GRID_WIDTH_RANGE = (16, 22)
# Object counts and colors.
FRAME_COUNT_RANGE = (2, 3)
SOLID_COUNT_RANGE = (2, 3)
PALETTE_COLORS = list(range(1, 10))
REQUIRED_SOLID_COLOR = 8
# Layout choices and retry budget.
PACKING_AXES = ("horizontal", "vertical")
MAX_ATTEMPTS = 300


def _gen_frame_patch(color: int) -> np.ndarray:
    patch = np.full((OBJECT_SIZE, OBJECT_SIZE), int(color), dtype=int)
    patch[1:-1, 1:-1] = BG
    return patch


def _gen_anchor_bounds(size: int) -> tuple[int, int]:
    return OBJECT_GAP, int(size) - OBJECT_SIZE - OBJECT_GAP


def _gen_spaced_positions(size: int, count: int) -> list[int] | None:
    lo, hi = _gen_anchor_bounds(size)
    step = OBJECT_SIZE + OBJECT_GAP
    slack = hi - lo - (int(count) - 1) * step
    if slack < 0: return None
    offsets = sorted(random.choices(range(slack + 1), k=int(count)))
    return [lo + i * step + offsets[i] for i in range(int(count))]


def _gen_limited_positions(size: int, count: int, max_span: int) -> list[int] | None:
    lo, hi = _gen_anchor_bounds(size)
    span_limit = min(int(max_span), hi - lo)
    if span_limit < 0: return None
    span = random.randint(0, span_limit)
    base = random.randint(lo, hi - span)
    return [base + random.randint(0, span) for _ in range(int(count))]


def _gen_bbox_clear(r: int, c: int, occupied: list[tuple[int, int]]) -> bool:
    r = int(r)
    c = int(c)
    last = OBJECT_SIZE - 1
    for rr, cc in occupied:
        if not (r + last < rr - OBJECT_GAP or rr + last < r - OBJECT_GAP
                or c + last < cc - OBJECT_GAP or cc + last < c - OBJECT_GAP):
            return False
    return True


def _gen_free_positions(height: int, width: int, occupied: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r_lo, r_hi = _gen_anchor_bounds(height)
    c_lo, c_hi = _gen_anchor_bounds(width)
    return [
        (r, c)
        for r in range(r_lo, r_hi + 1)
        for c in range(c_lo, c_hi + 1)
        if _gen_bbox_clear(r, c, occupied)
    ]


def _gen_colors(n_frames: int, n_solids: int) -> tuple[list[int], list[int]]:
    """Select distinct frame/solid colors, forcing color 8 into a solid."""
    solid_colors = [REQUIRED_SOLID_COLOR] + random.sample(
        [c for c in PALETTE_COLORS if c != REQUIRED_SOLID_COLOR], n_solids - 1
    )
    random.shuffle(solid_colors)
    frame_colors = random.sample([c for c in PALETTE_COLORS if c not in solid_colors], n_frames)
    return frame_colors, solid_colors


def _gen_frame_layout(axis: str, height: int, width: int, frame_patches: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    """Return expected strip and frame anchors for an axis, or None if constraints fail."""
    horizontal = axis == "horizontal"
    n_frames = len(frame_patches)
    shape = (OBJECT_SIZE, OBJECT_SIZE * n_frames) if horizontal else (OBJECT_SIZE * n_frames, OBJECT_SIZE)
    expected = np.full(shape, BG, dtype=int)
    for i, patch in enumerate(frame_patches):
        paste(expected, patch, 0 if horizontal else i * OBJECT_SIZE, i * OBJECT_SIZE if horizontal else 0)

    if horizontal:
        cols = _gen_spaced_positions(width, n_frames)
        if cols is None: return None
        col_span = max(cols) - min(cols)
        rows = _gen_limited_positions(height, n_frames, col_span - 1)
        if rows is None: return None
        return expected, list(zip(rows, cols))

    rows = _gen_spaced_positions(height, n_frames)
    if rows is None: return None
    row_span = max(rows) - min(rows)
    cols = _gen_limited_positions(width, n_frames, row_span - 1)
    if cols is None: return None
    return expected, list(zip(rows, cols))


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(*FRAME_COUNT_RANGE)
    n_solids = random.randint(*SOLID_COUNT_RANGE)
    frame_colors, solid_colors = _gen_colors(n_frames, n_solids)
    axis = random.choice(PACKING_AXES)
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)

    frame_patches = [_gen_frame_patch(color) for color in frame_colors]
    layout = _gen_frame_layout(axis, height, width, frame_patches)
    if layout is None: return None
    expected, frame_positions = layout

    grid = np.full((height, width), BG, dtype=int)
    occupied: list[tuple[int, int]] = []
    for (r, c), patch in zip(frame_positions, frame_patches):
        paste(grid, patch, r, c)
        occupied.append((r, c))

    for color in solid_colors:
        free = _gen_free_positions(height, width, occupied)
        if not free: return None
        r, c = random.choice(free)
        solid = np.full((OBJECT_SIZE, OBJECT_SIZE), int(color), dtype=int)
        paste(grid, solid, r, c)
        occupied.append((r, c))

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected): return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
