from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg, paste, random_free_location
import random


def _dominant_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size > 0, "expected foreground pixels"
    max_count = int(counts.max())
    winners = vals[counts == max_count]
    assert winners.size == 1, "expected one largest foreground color for lattice frames"
    return int(winners[0])


def _parse_frames(g: np.ndarray, bg: int, frame_color: int) -> list[tuple[list[int], list[int]]]:
    frames: list[tuple[list[int], list[int]]] = []
    components = find_components(
        g,
        bg=int(bg),
        colors=[int(frame_color)],
        connectivity=4,
        monochromatic=True,
    )
    assert components, "expected at least one lattice frame"

    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        sub = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(frame_color)

        row_idx = [r for r in range(sub.shape[0]) if bool(sub[r, :].all())]
        col_idx = [c for c in range(sub.shape[1]) if bool(sub[:, c].all())]
        assert len(row_idx) >= 2 and len(col_idx) >= 2, "frame must divide slots"
        assert row_idx[0] == 0 and row_idx[-1] == sub.shape[0] - 1
        assert col_idx[0] == 0 and col_idx[-1] == sub.shape[1] - 1
        assert all(a + 1 < b for a, b in zip(row_idx, row_idx[1:]))
        assert all(a + 1 < b for a, b in zip(col_idx, col_idx[1:]))

        expected = np.zeros_like(sub, dtype=bool)
        expected[row_idx, :] = True
        expected[:, col_idx] = True
        assert np.array_equal(sub, expected), "frame must be a complete rectangular lattice"

        rows = [int(r0) + r for r in row_idx]
        cols = [int(c0) + c for c in col_idx]
        frames.append((rows, cols))

    slot_rows = len(frames[0][0]) - 1
    slot_cols = len(frames[0][1]) - 1
    for rows, cols in frames:
        assert len(rows) - 1 == slot_rows
        assert len(cols) - 1 == slot_cols
    return frames


def _infer_slot_colors(
    g: np.ndarray,
    bg: int,
    frame_color: int,
    frames: list[tuple[list[int], list[int]]],
) -> dict[tuple[int, int], int]:
    slot_rows = len(frames[0][0]) - 1
    slot_cols = len(frames[0][1]) - 1
    colors: dict[tuple[int, int], int] = {}

    for rows, cols in frames:
        for sr in range(slot_rows):
            for sc in range(slot_cols):
                region = g[rows[sr] + 1 : rows[sr + 1], cols[sc] + 1 : cols[sc + 1]]
                marker = region[(region != int(bg)) & (region != int(frame_color))]
                if marker.size == 0:
                    continue
                vals = np.unique(marker)
                assert vals.size == 1, "each colored slot must contain one color"
                color = int(vals[0])
                key = (sr, sc)
                assert key not in colors or colors[key] == color
                colors[key] = color

    expected = {(sr, sc) for sr in range(slot_rows) for sc in range(slot_cols)}
    assert set(colors) == expected, "colored slots must identify every logical slot"
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill every lattice slot from the color assigned to that slot position by the other lattices.

    Input structure:
    - The background is the dominant color.
    - The largest foreground color forms several one-pixel-thick rectangular lattice frames.
    - The frame lines divide each lattice into the same row-by-column set of rectangular slots; some slots already contain one non-frame color.

    Rule:
    - Infer a color for each logical slot position from the colored slots across all lattice frames.
    - Repaint every slot interior in every lattice with the inferred color for its row/column position.
    - Keep the frame lines and all background outside frames unchanged.

    Key insight:
    - Slot sizes vary by frame, but the complete color pattern is indexed by lattice slot position rather than by absolute pixel size or location.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    frame_color = _dominant_foreground_color(g, bg)
    frames = _parse_frames(g, bg, frame_color)
    slot_colors = _infer_slot_colors(g, bg, frame_color, frames)

    out = g.copy()
    for rows, cols in frames:
        for sr in range(len(rows) - 1):
            for sc in range(len(cols) - 1):
                out[rows[sr] + 1 : rows[sr + 1], cols[sc] + 1 : cols[sc + 1]] = slot_colors[
                    (sr, sc)
                ]
    return out

# --------------- generator (o5x) ---------------

# Grid and palette choices.
BG = 0
GRID_SIZE_RANGE = (22, 28)
NEAR_SQUARE_PROB = 0.65
NEAR_SQUARE_DELTAS = [-2, -1, 0, 1, 2]
FOREGROUND_COLORS = range(1, 10)
PALETTE_SIZE = 5

# Lattice geometry.
SLOT_SIZE_RANGE = (1, 8)
FRAME_HEIGHT_RANGE = (5, 18)
FRAME_WIDTH_RANGE = (5, 14)
LATTICE_LINE_COUNT = 3
SLOT_POSITIONS = [(0, 0), (0, 1), (1, 0), (1, 1)]

# Frame count and area allocation.
FRAME_COUNT_RANGE = (4, 6)
MIN_FRAME_AREA = 25
AREA_BUDGET_RANGE = (0.55, 0.88)
AREA_WEIGHT_SCALE = 35.0

# Placement and retry limits.
EDGE_BORDER = 1
PLACEMENT_PADDING = 1
PLACEMENT_CONNECTIVITY = 8
MAX_GENERATION_ATTEMPTS = 300

FrameSpec = tuple[tuple[int, int], tuple[int, int]]
PlacedFrame = tuple[int, int, tuple[int, int], tuple[int, int]]
SlotColors = dict[tuple[int, int], int]


def _line_indices(lengths: tuple[int, int]) -> list[int]:
    return [0, int(lengths[0]) + 1, int(lengths[0]) + int(lengths[1]) + 2]


def _slot_options() -> dict[tuple[int, int], list[FrameSpec]]:
    options: dict[tuple[int, int], list[FrameSpec]] = {}
    slot_min, slot_max = SLOT_SIZE_RANGE
    height_min, height_max = FRAME_HEIGHT_RANGE
    width_min, width_max = FRAME_WIDTH_RANGE
    for h0 in range(slot_min, slot_max + 1):
        for h1 in range(slot_min, slot_max + 1):
            height = h0 + h1 + LATTICE_LINE_COUNT
            if not height_min <= height <= height_max:
                continue
            for w0 in range(slot_min, slot_max + 1):
                for w1 in range(slot_min, slot_max + 1):
                    width = w0 + w1 + LATTICE_LINE_COUNT
                    if not width_min <= width <= width_max:
                        continue
                    options.setdefault((height, width), []).append(((h0, h1), (w0, w1)))
    return options


_SIZE_OPTIONS = _slot_options()


def _choose_frame_specs(n_frames: int, rows: int, cols: int) -> list[FrameSpec] | None:
    keys = [size for size in _SIZE_OPTIONS if size[0] <= rows - 2 * EDGE_BORDER and size[1] <= cols - 2 * EDGE_BORDER]
    random.shuffle(keys)

    usable = (rows - 2 * EDGE_BORDER) * (cols - 2 * EDGE_BORDER)
    budget = int(usable * random.uniform(*AREA_BUDGET_RANGE))
    chosen_sizes: list[tuple[int, int]] = []
    remaining = budget

    for i in range(n_frames):
        slots_left = n_frames - i - 1
        candidates = [
            size
            for size in keys
            if size not in chosen_sizes
            and size[0] * size[1] <= remaining - slots_left * MIN_FRAME_AREA
        ]
        if not candidates:
            return None
        weights = []
        target = max(MIN_FRAME_AREA, remaining / (slots_left + 1))
        for height, width in candidates:
            area = height * width
            weights.append(1.0 / (1.0 + abs(area - target) / AREA_WEIGHT_SCALE))
        size = random.choices(candidates, weights=weights, k=1)[0]
        chosen_sizes.append(size)
        remaining -= size[0] * size[1]

    specs = [random.choice(_SIZE_OPTIONS[size]) for size in chosen_sizes]
    specs.sort(key=lambda spec: (sum(spec[0]) + LATTICE_LINE_COUNT) * (sum(spec[1]) + LATTICE_LINE_COUNT), reverse=True)
    return specs


def _choose_canvas_shape() -> tuple[int, int]:
    grid_min, grid_max = GRID_SIZE_RANGE
    rows = random.randint(grid_min, grid_max)
    cols = random.randint(grid_min, grid_max)
    if random.random() < NEAR_SQUARE_PROB:
        delta = random.choice(NEAR_SQUARE_DELTAS)
        cols = min(grid_max, max(grid_min, rows + delta))
    return rows, cols


def _choose_palette() -> tuple[int, SlotColors]:
    colors = random.sample(FOREGROUND_COLORS, PALETTE_SIZE)
    frame_color = colors[0]
    color_values = colors[1:]
    random.shuffle(color_values)
    slot_colors = {pos: color_values[i] for i, pos in enumerate(SLOT_POSITIONS)}
    return frame_color, slot_colors


def _make_output_patch(heights: tuple[int, int], widths: tuple[int, int], frame_color: int, slot_colors: SlotColors) -> tuple[np.ndarray, list[int], list[int]]:
    row_lines = _line_indices(heights)
    col_lines = _line_indices(widths)
    patch = np.zeros((row_lines[-1] + 1, col_lines[-1] + 1), dtype=int)
    patch[row_lines, :] = int(frame_color)
    patch[:, col_lines] = int(frame_color)

    for sr, sc in SLOT_POSITIONS:
        patch[row_lines[sr] + 1 : row_lines[sr + 1], col_lines[sc] + 1 : col_lines[sc + 1]] = slot_colors[(sr, sc)]
    return patch, row_lines, col_lines


def _hint_providers(n_frames: int) -> dict[tuple[int, int], int]:
    providers = [random.randrange(n_frames) for _ in SLOT_POSITIONS]
    if len(set(providers)) == 1:
        providers[random.randrange(len(providers))] = random.choice(
            [idx for idx in range(n_frames) if idx != providers[0]]
        )
    return {pos: providers[i] for i, pos in enumerate(SLOT_POSITIONS)}


def _place_frames(rows: int, cols: int, specs: list[FrameSpec]) -> list[PlacedFrame] | None:
    occ = np.zeros((rows, cols), dtype=int)
    placed: list[PlacedFrame] = []
    for heights, widths in specs:
        height = sum(heights) + LATTICE_LINE_COUNT
        width = sum(widths) + LATTICE_LINE_COUNT
        block = np.ones((height, width), dtype=int)
        try:
            r0, c0 = random_free_location(occ, block, transparent=BG, border=EDGE_BORDER, padding=PLACEMENT_PADDING, padding_connectivity=PLACEMENT_CONNECTIVITY)
        except ValueError:
            return None
        paste(occ, block, r0, c0)
        placed.append((r0, c0, heights, widths))
    return placed


def _gen_attempt() -> np.ndarray | None:
    rows, cols = _choose_canvas_shape()
    n_frames = random.randint(*FRAME_COUNT_RANGE)
    frame_color, slot_colors = _choose_palette()

    specs = _choose_frame_specs(n_frames, rows, cols)
    if specs is None:
        return None

    placed = _place_frames(rows, cols, specs)
    if placed is None:
        return None

    out = np.full((rows, cols), BG, dtype=int)
    line_data: list[tuple[int, int, list[int], list[int]]] = []
    for r0, c0, heights, widths in placed:
        patch, row_lines, col_lines = _make_output_patch(heights, widths, frame_color, slot_colors)
        paste(out, patch, r0, c0)
        line_data.append((r0, c0, row_lines, col_lines))

    inp = out.copy()
    providers = _hint_providers(n_frames)
    for frame_idx, (r0, c0, row_lines, col_lines) in enumerate(line_data):
        for sr, sc in SLOT_POSITIONS:
            if providers[(sr, sc)] == frame_idx:
                continue
            rr0 = r0 + row_lines[sr] + 1
            rr1 = r0 + row_lines[sr + 1]
            cc0 = c0 + col_lines[sc] + 1
            cc1 = c0 + col_lines[sc + 1]
            inp[rr0:rr1, cc0:cc1] = BG

    try:
        if int(get_bg(inp)) != BG:
            return None
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 2e65ae53 input")
