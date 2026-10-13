import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

PAIR_MAP = {
    BLUE: RED,
    RED: BLUE,
    GREEN: YELLOW,
    YELLOW: GREEN,
    MAGENTA: ORANGE,
    ORANGE: MAGENTA,
    AZURE: MAROON,
    MAROON: AZURE,
}

MODE_SEED_COLORS = (
    (RED, GREEN, MAGENTA, AZURE),
    (BLUE, YELLOW, ORANGE, MAROON),
    (BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, AZURE, MAROON),
)

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400

_GEN_CALL_COUNT = 0


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_bbox(component: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in component]
    cols = [c for _, c in component]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    expected_size = 2 * (h + w) - 4
    if len(component) != expected_size:
        return None

    cells = set(component)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in cells):
                return None
    return r0, c0, r1, c1


def _find_frames(grid: np.ndarray) -> list[dict[str, int]]:
    grey_components = _components(grid == GREY)
    frames: list[dict[str, int]] = []
    for comp in grey_components:
        bbox = _frame_bbox(comp)
        assert bbox is not None, "every grey component must be a rectangular frame"
        r0, c0, r1, c1 = bbox
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        assert interior.size > 0, "frame must contain an interior"
        seed_coords = np.argwhere((interior != BLACK) & (interior != GREY))
        assert len(seed_coords) == 1, "each frame must contain exactly one seed"
        sr, sc = map(int, seed_coords[0])
        seed_color = int(interior[sr, sc])
        assert seed_color in PAIR_MAP, "seed color must belong to the fixed pair table"
        assert np.count_nonzero(interior == seed_color) == 1, "seed color must be a single pixel"
        assert np.all((interior == BLACK) | (interior == seed_color)), "interior may only contain black plus the seed"
        frames.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "seed_r": int(r0 + 1 + sr),
                "seed_c": int(c0 + 1 + sc),
                "seed_color": seed_color,
            }
        )

    assert frames, "expected at least one frame"
    frames.sort(key=lambda frame: (frame["r0"], frame["c0"]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey rectangular enclosure with its seed color and recolor the frame with that seed's fixed paired complement.

    Input structure:
    - Background is black.
    - Several one-pixel-thick grey rectangular frames appear on the grid.
    - Each frame encloses exactly one non-black seed pixel, and the remaining interior cells are black.
    - Frames are separate and do not touch each other.

    Rule:
    - Detect every grey frame and read the unique seed color C inside it.
    - Recolor all black interior cells of that frame to C, leaving the seed pixel unchanged.
    - Repaint the whole grey border using C's paired complement from the learned fixed table:
      red<->blue, green<->yellow, magenta<->orange, azure<->maroon.
    - Leave the outside black background unchanged.

    Key insight:
    The first layer is a direct seed-driven flood fill, but the border color comes from a second hidden rule:
    a fixed cross-example color-pair mapping rather than a constant or self-colored frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for frame in _find_frames(g):
        r0 = frame["r0"]
        c0 = frame["c0"]
        r1 = frame["r1"]
        c1 = frame["c1"]
        seed_color = frame["seed_color"]
        border_color = PAIR_MAP[seed_color]

        out[r0 + 1 : r1, c0 + 1 : c1] = seed_color
        out[r0, c0 : c1 + 1] = border_color
        out[r1, c0 : c1 + 1] = border_color
        out[r0 : r1 + 1, c0] = border_color
        out[r0 : r1 + 1, c1] = border_color

    return out


def _partition_counts(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), parts - 1))
    counts: list[int] = []
    prev = 0
    for cut in cuts + [total]:
        counts.append(cut - prev)
        prev = cut
    return counts


def _sample_specs(seed_colors: tuple[int, ...]) -> list[dict[str, int]]:
    specs: list[dict[str, int]] = []
    compact = len(seed_colors) > 4
    for seed_color in seed_colors:
        if compact:
            h = random.randint(4, 7)
            w = random.randint(4, 7)
        else:
            h = random.randint(4, 9)
            w = random.randint(4, 9)
        specs.append({"seed_color": int(seed_color), "h": h, "w": w})
    random.shuffle(specs)
    return specs


def _place_specs(specs: list[dict[str, int]]) -> tuple[np.ndarray, list[dict[str, int]]] | None:
    n = len(specs)
    row_options = (2, 3) if n == 4 else (3, 4)
    row_count = random.choice(row_options)
    counts = _partition_counts(n, row_count)

    rows: list[list[dict[str, int]]] = []
    idx = 0
    for count in counts:
        rows.append(specs[idx : idx + count])
        idx += count

    col_gaps = [[random.randint(1, 3) for _ in range(max(0, len(row) - 1))] for row in rows]
    row_heights = [max(spec["h"] for spec in row) for row in rows]
    row_widths = []
    for row, gaps in zip(rows, col_gaps):
        row_widths.append(sum(spec["w"] for spec in row) + sum(gaps))

    usable_w = max(row_widths)
    row_spacings = [random.randint(1, 3) for _ in range(row_count - 1)]
    usable_h = sum(row_heights) + sum(row_spacings)

    min_h = usable_h + 2
    min_w = usable_w + 2
    if min_h > 30 or min_w > 30:
        return None

    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)
    extra_top = random.randint(0, grid_h - min_h)
    extra_left = random.randint(0, grid_w - min_w)

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    placed: list[dict[str, int]] = []

    y = 1 + extra_top
    for row_idx, row in enumerate(rows):
        row_height = row_heights[row_idx]
        row_width = row_widths[row_idx]
        row_left = 1 + extra_left + random.randint(0, usable_w - row_width)
        x = row_left
        for spec_idx, spec in enumerate(row):
            top = y + random.randint(0, row_height - spec["h"])
            left = x
            h = spec["h"]
            w = spec["w"]
            bottom = top + h - 1
            right = left + w - 1

            grid[top, left : right + 1] = GREY
            grid[bottom, left : right + 1] = GREY
            grid[top : bottom + 1, left] = GREY
            grid[top : bottom + 1, right] = GREY

            seed_r = random.randint(top + 1, bottom - 1)
            seed_c = random.randint(left + 1, right - 1)
            grid[seed_r, seed_c] = spec["seed_color"]

            placed.append(
                {
                    "r0": int(top),
                    "c0": int(left),
                    "r1": int(bottom),
                    "c1": int(right),
                    "seed_r": int(seed_r),
                    "seed_c": int(seed_c),
                    "seed_color": int(spec["seed_color"]),
                }
            )

            if spec_idx < len(row) - 1:
                x += w + col_gaps[row_idx][spec_idx]
        if row_idx < row_count - 1:
            y += row_height + row_spacings[row_idx]

    return grid, placed


def _expected_output(grid: np.ndarray, placed: list[dict[str, int]]) -> np.ndarray:
    out = grid.copy()
    for frame in placed:
        r0 = frame["r0"]
        c0 = frame["c0"]
        r1 = frame["r1"]
        c1 = frame["c1"]
        seed_color = frame["seed_color"]
        border_color = PAIR_MAP[seed_color]
        out[r0 + 1 : r1, c0 + 1 : c1] = seed_color
        out[r0, c0 : c1 + 1] = border_color
        out[r1, c0 : c1 + 1] = border_color
        out[r0 : r1 + 1, c0] = border_color
        out[r0 : r1 + 1, c1] = border_color
    return out


def _valid_generated_case(grid: np.ndarray, placed: list[dict[str, int]]) -> bool:
    try:
        found = _find_frames(grid)
    except AssertionError:
        return False

    if len(found) != len(placed):
        return False
    if len({frame["seed_color"] for frame in found}) != len(found):
        return False

    expected = _expected_output(grid, placed)
    actual = main(grid)
    return np.array_equal(actual, expected)


def _attempt_generate(seed_colors: tuple[int, ...]) -> np.ndarray | None:
    specs = _sample_specs(seed_colors)
    placed_case = _place_specs(specs)
    if placed_case is None:
        return None
    grid, placed = placed_case
    if not _valid_generated_case(grid, placed):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    mode_idx = _GEN_CALL_COUNT % len(MODE_SEED_COLORS)
    _GEN_CALL_COUNT += 1
    seed_colors = MODE_SEED_COLORS[mode_idx]

    for _ in range(GEN_RETRIES):
        grid = _attempt_generate(seed_colors)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-frame ARC input")
