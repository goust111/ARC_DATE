import numpy as np

from arc_utils import find_components, get_bg, is_contiguous, paste
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Shift monochromatic foreground objects one column left/right in alternating top-to-bottom order.

    Input structure
    - The grid has a dominant background color.
    - Every foreground shape is a single-color 4-connected object.
    - The objects have distinct top bbox rows, giving a unique top-to-bottom order.

    Rule
    - Sort the objects by their top bbox row.
    - Shift the 1st, 3rd, 5th, ... objects one column left.
    - Shift the 2nd, 4th, 6th, ... objects one column right.
    - Vacated cells become background, and colors/shapes are preserved.

    Key insight
    - The movement direction is not tied to color or shape; it is the object's rank in the vertical order.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    objects = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert objects, "expected at least one foreground object"

    top_rows = [int(obj["bbox"][0]) for obj in objects]
    assert len(set(top_rows)) == len(top_rows), "foreground objects must have distinct top rows"
    ordered = sorted(objects, key=lambda obj: int(obj["bbox"][0]))

    out = np.full_like(g, bg)
    occupied = np.zeros(g.shape, dtype=bool)
    height, width = g.shape

    for rank, obj in enumerate(ordered):
        dc = -1 if rank % 2 == 0 else 1
        coords = np.asarray(obj["coords"], dtype=int)
        shifted = coords + np.array([0, dc], dtype=int)
        in_bounds = (
            (0 <= shifted[:, 0])
            & (shifted[:, 0] < height)
            & (0 <= shifted[:, 1])
            & (shifted[:, 1] < width)
        )
        assert bool(np.all(in_bounds)), "shifted objects must stay in bounds"
        assert not bool(np.any(occupied[shifted[:, 0], shifted[:, 1]])), "shifted objects must not overlap"

        out[shifted[:, 0], shifted[:, 1]] = int(obj["color"])
        occupied[shifted[:, 0], shifted[:, 1]] = True

    return out

# --------------- generator (o5x) ---------------

# Grid and palette bounds.
GRID_HEIGHT_RANGE = (12, 16)
GRID_WIDTH_RANGE = (12, 13)
PALETTE_SIZE = 10
# Pair and object layout bounds.
PAIR_COUNT_RANGE = (1, 2)
OBJECTS_PER_PAIR = 2
OUTER_ROW_MARGIN = 1
INTER_PAIR_GAP = 1
OBJECT_HEIGHT_RANGE = (1, 5)
SPAN_WIDTH_RANGE = (3, 7)
OUTPUT_SIDE_CLEARANCE = 2
# Sprite density, movement, and retry policy.
MIN_SPRITE_FILL = 0.79
CONNECTIVITY = 4
HORIZONTAL_SHIFT = 1
MAX_GENERATION_ATTEMPTS = 300

def _gen_composition(total: int, parts: int) -> list[int]:
    """Split total into nonnegative row-gap extras using the original sampling rule."""
    total = int(total)
    parts = int(parts)
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    cuts = [-1] + cuts + [total + parts - 1]
    return [right - left - 1 for left, right in zip(cuts, cuts[1:])]

def _gen_height_combo(count: int, max_sum: int) -> list[int]:
    """Sample bounded object heights uniformly from feasible height combinations."""
    combos = []

    def rec(prefix: list[int], remaining: int, budget: int) -> None:
        if remaining == 0:
            combos.append(prefix)
            return
        lo, max_height = OBJECT_HEIGHT_RANGE
        hi = min(max_height, budget - lo * (remaining - 1))
        for h in range(lo, hi + 1):
            rec(prefix + [h], remaining - 1, budget - h)

    rec([], int(count), int(max_sum))
    return list(random.choice(combos))

def _removable_hole_cells(sprite: np.ndarray, bg: int) -> list[tuple[int, int]]:
    """Return cells removable without shrinking the bbox or splitting the sprite."""
    safe = []
    foreground = sprite != int(bg)
    for r in range(sprite.shape[0]):
        for c in range(sprite.shape[1]):
            if not foreground[r, c]:
                continue
            mask = foreground.copy()
            mask[r, c] = False
            if not (mask[0, :].any() and mask[-1, :].any()):
                continue
            if not (mask[:, 0].any() and mask[:, -1].any()):
                continue
            if is_contiguous(mask, background=0, connectivity=CONNECTIVITY):
                safe.append((r, c))
    return safe

def _gen_dense_sprite(height: int, width: int, color: int, bg: int) -> np.ndarray:
    """Build one dense monochrome sprite, optionally carving safe background holes."""
    sprite = np.full((int(height), int(width)), int(color), dtype=int)
    area = int(height) * int(width)
    max_holes = area - int(np.ceil(MIN_SPRITE_FILL * area))
    if max_holes <= 0:
        return sprite
    for _ in range(random.randint(0, max_holes)):
        safe = _removable_hole_cells(sprite, bg)
        if not safe: break
        rr, cc = random.choice(safe)
        sprite[rr, cc] = int(bg)
    return sprite

def _sample_row_layout(height: int, pair_count: int, obj_count: int) -> tuple[list[int], list[int]]:
    """Sample object heights and extra background rows between pair blocks."""
    max_height_sum = height - 2 * OUTER_ROW_MARGIN - INTER_PAIR_GAP * (pair_count - 1)
    obj_heights = _gen_height_combo(obj_count, max_height_sum)
    extra_rows = height - sum(obj_heights) - 2 * OUTER_ROW_MARGIN - INTER_PAIR_GAP * (pair_count - 1)
    return obj_heights, _gen_composition(extra_rows, pair_count + 1)

def _place_output_pairs(out: np.ndarray, obj_heights: list[int], gap_extras: list[int], pair_count: int, bg: int) -> list[tuple[int, np.ndarray, int, int]] | None:
    """Render output pair blocks and return metadata for inverse shifting."""
    placements = []
    row = OUTER_ROW_MARGIN + gap_extras[0]
    colors = [c for c in range(PALETTE_SIZE) if c != bg]
    rank = 0
    for pair_i in range(pair_count):
        span_width = random.randint(*SPAN_WIDTH_RANGE)
        col_min = OUTPUT_SIDE_CLEARANCE
        col_max = out.shape[1] - span_width - OUTPUT_SIDE_CLEARANCE
        if col_max < col_min:
            return None
        col = random.randint(col_min, col_max)
        pair_colors = random.sample(colors, OBJECTS_PER_PAIR)
        for color in pair_colors:
            sprite = _gen_dense_sprite(obj_heights[rank], span_width, color, bg)
            paste(out, sprite, row, col, transparent=bg)
            placements.append((rank, sprite, row, col))
            row += obj_heights[rank]
            rank += 1
        if pair_i < pair_count - 1:
            row += INTER_PAIR_GAP + gap_extras[pair_i + 1]
    return placements

def _render_input(out: np.ndarray, bg: int, placements: list[tuple[int, np.ndarray, int, int]]) -> np.ndarray:
    """Reverse the solver shift to produce the paired input grid."""
    inp = np.full_like(out, bg)
    for rank, sprite, row, col in placements:
        input_col = col + HORIZONTAL_SHIFT if rank % 2 == 0 else col - HORIZONTAL_SHIFT
        paste(inp, sprite, row, input_col, transparent=bg)
    return inp

def _passes_oracle(inp: np.ndarray, out: np.ndarray, bg: int, obj_count: int) -> bool:
    """Keep only grids that the solver maps exactly to the sampled output."""
    if get_bg(inp) != bg:
        return False
    objects = find_components(inp, bg=bg, connectivity=CONNECTIVITY, monochromatic=True, return_format="dict")
    top_rows = [int(obj["bbox"][0]) for obj in objects]
    if len(objects) != obj_count or len(set(top_rows)) != len(top_rows):
        return False
    try:
        solved = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(solved, out))

def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    bg = random.randint(0, PALETTE_SIZE - 1)
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    obj_count = pair_count * OBJECTS_PER_PAIR
    obj_heights, gap_extras = _sample_row_layout(height, pair_count, obj_count)
    out = np.full((height, width), bg, dtype=int)
    placements = _place_output_pairs(out, obj_heights, gap_extras, pair_count, bg)
    if placements is None: return None
    inp = _render_input(out, bg, placements)
    return inp if _passes_oracle(inp, out, bg, obj_count) else None

def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None: return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
