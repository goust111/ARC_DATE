from __future__ import annotations

import random

import numpy as np


BG = 0
LEGEND_INK = 5
LEGEND_MARKER_POOL = (1, 2, 3)
FRAME_COLOR_POOL = (4, 6, 7, 8, 9)
INNER_SIZE_POOL = ((4, 4), (4, 6), (6, 4), (6, 6))
GENERATION_RETRIES = 400

MASK_POOL = (
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [0, 1]], dtype=bool),
    np.array([[0, 1], [1, 0]], dtype=bool),
)


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "bbox expects a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _boxes_overlap(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    pad: int = 0,
) -> bool:
    return not (
        box_a[2] + pad < box_b[0]
        or box_b[2] + pad < box_a[0]
        or box_a[3] + pad < box_b[1]
        or box_b[3] + pad < box_a[1]
    )


def _scale_mask(mask: np.ndarray, scale_y: int, scale_x: int) -> np.ndarray:
    tile = np.ones((scale_y, scale_x), dtype=int)
    return np.kron(mask.astype(int), tile).astype(bool)


def _extract_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        mask = grid == color
        if not np.any(mask):
            continue
        r0, c0, r1, c1 = _bbox(mask)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 3:
            continue
        local = mask[r0 : r1 + 1, c0 : c1 + 1]
        border = _border_mask(height, width)
        if np.array_equal(local, border):
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "inner_shape": (height - 2, width - 2),
                }
            )
    assert frames, "expected at least one rectangular frame"
    return frames


def _parse_legend(
    grid: np.ndarray,
    bg: int,
    frame_boxes: list[tuple[int, int, int, int]],
) -> dict[int, np.ndarray]:
    legend: dict[int, np.ndarray] = {}
    height, width = grid.shape
    for r in range(height - 2):
        for c in range(width - 2):
            color = int(grid[r, c])
            if color in (bg, LEGEND_INK):
                continue
            if any(r0 <= r <= r1 and c0 <= c <= c1 for r0, c0, r1, c1 in frame_boxes):
                continue
            patch = grid[r + 1 : r + 3, c + 1 : c + 3]
            if np.all((patch == bg) | (patch == LEGEND_INK)) and np.any(patch == LEGEND_INK):
                mask = patch == LEGEND_INK
                if color in legend:
                    assert np.array_equal(legend[color], mask), "legend color must map to one template"
                legend[color] = mask
    assert legend, "legend markers were not found"
    return legend


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp a legend-selected 2x2 template into each frame after scaling it to the frame's inner size.

    Input structure:
    - The background is black.
    - A legend in the upper-left area contains colored marker pixels, each paired with a 2x2 grey binary template placed one row down and one column right.
    - Elsewhere the grid contains several one-cell-thick hollow rectangular frames of distinct colors, each with exactly one colored control pixel in its interior.

    Rule:
    - Read the legend as a mapping from control color to a 2x2 binary template mask.
    - For each hollow frame, use the interior control pixel color to choose the template, and use the frame border color as the ink color.
    - Scale the 2x2 template by `(inner_height / 2, inner_width / 2)` so it exactly fills the frame interior, clear the control pixel, and stamp the scaled template into the interior while leaving zero-cells black.

    Key insight:
    - The control color determines which template to draw, but the frame color determines which color the drawn ink should have, so pattern selection and recoloring are structurally separated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background(g)

    frames = _extract_frames(g, bg)
    legend = _parse_legend(g, bg, [frame["bbox"] for frame in frames])

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        inner_h, inner_w = inner.shape
        assert inner_h >= 4 and inner_w >= 4
        assert inner_h % 2 == 0 and inner_w % 2 == 0, "frame interior must be an integer multiple of 2x2"

        control_cells = np.argwhere(inner != bg)
        assert control_cells.shape[0] == 1, "each frame interior must contain exactly one control pixel"
        cr, cc = map(int, control_cells[0])
        control_color = int(inner[cr, cc])
        assert control_color in legend, "control pixel must reference a legend entry"

        scaled = _scale_mask(legend[control_color], inner_h // 2, inner_w // 2)
        target = out[r0 + 1 : r1, c0 + 1 : c1]
        target.fill(bg)
        target[scaled] = frame["color"]

    return out


def _choose_legend_entries(num_entries: int) -> tuple[list[int], dict[int, np.ndarray]]:
    marker_colors = random.sample(list(LEGEND_MARKER_POOL), num_entries)
    while True:
        masks = [mask.copy() for mask in random.sample(list(MASK_POOL), num_entries)]
        if len({int(mask.sum()) for mask in masks}) >= 2:
            break
    return marker_colors, {color: mask for color, mask in zip(marker_colors, masks)}


def _build_frame_specs(marker_colors: list[int]) -> list[dict]:
    size_a, size_b, size_c, _size_d = random.sample(list(INNER_SIZE_POOL), len(INNER_SIZE_POOL))

    if len(marker_colors) == 2:
        c0, c1 = marker_colors
        controls = [c0, c0, c1]
        sizes = [size_a, size_b, size_a]
        if random.random() < 0.5:
            controls.append(c1)
            sizes.append(size_b)
    else:
        c0, c1, c2 = marker_colors
        controls = [c0, c0, c1, c2]
        sizes = [size_a, size_b, size_a, random.choice((size_b, size_c))]

    frame_colors = random.sample(list(FRAME_COLOR_POOL), len(controls))
    specs = []
    for frame_color, control_color, (inner_h, inner_w) in zip(frame_colors, controls, sizes):
        specs.append(
            {
                "frame_color": frame_color,
                "control_color": control_color,
                "inner_h": inner_h,
                "inner_w": inner_w,
            }
        )
    random.shuffle(specs)
    return specs


def _specs_cover_counterexamples(specs: list[dict], marker_colors: list[int]) -> bool:
    size_to_controls: dict[tuple[int, int], set[int]] = {}
    control_to_sizes: dict[int, set[tuple[int, int]]] = {}
    used_controls = []
    for spec in specs:
        size = (spec["inner_h"], spec["inner_w"])
        control = spec["control_color"]
        size_to_controls.setdefault(size, set()).add(control)
        control_to_sizes.setdefault(control, set()).add(size)
        used_controls.append(control)

    same_control_diff_size = any(len(sizes) >= 2 for sizes in control_to_sizes.values())
    same_size_diff_control = any(len(controls) >= 2 for controls in size_to_controls.values())
    return same_control_diff_size and same_size_diff_control and set(used_controls) == set(marker_colors)


def _legend_box(
    marker_row: int,
    marker_col0: int,
    spacing: int,
    num_entries: int,
) -> tuple[int, int, int, int]:
    last_marker_col = marker_col0 + spacing * (num_entries - 1)
    return marker_row, marker_col0, marker_row + 2, last_marker_col + 2


def _place_frames(
    shape: tuple[int, int],
    legend_box: tuple[int, int, int, int],
    specs: list[dict],
) -> list[tuple[int, int, int, int]] | None:
    height, width = shape
    boxes: list[tuple[int, int, int, int]] = [legend_box]
    placements: list[tuple[int, int, int, int] | None] = [None] * len(specs)
    order = sorted(
        range(len(specs)),
        key=lambda idx: (specs[idx]["inner_h"] + 2) * (specs[idx]["inner_w"] + 2),
        reverse=True,
    )

    for idx in order:
        outer_h = specs[idx]["inner_h"] + 2
        outer_w = specs[idx]["inner_w"] + 2
        candidates = []
        for r0 in range(height - outer_h + 1):
            for c0 in range(width - outer_w + 1):
                box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
                if any(_boxes_overlap(box, other, pad=1) for other in boxes):
                    continue
                candidates.append(box)
        if not candidates:
            return None
        chosen = random.choice(candidates)
        placements[idx] = chosen
        boxes.append(chosen)

    return [box for box in placements if box is not None]


def _render_input(
    shape: tuple[int, int],
    marker_row: int,
    marker_col0: int,
    spacing: int,
    legend: dict[int, np.ndarray],
    specs: list[dict],
    frame_boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)

    for index, (color, mask) in enumerate(legend.items()):
        c = marker_col0 + spacing * index
        grid[marker_row, c] = color
        grid[marker_row + 1 : marker_row + 3, c + 1 : c + 3] = np.where(mask, LEGEND_INK, BG)

    for spec, (r0, c0, r1, c1) in zip(specs, frame_boxes):
        color = spec["frame_color"]
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color

        ctrl_r = random.randint(r0 + 1, r1 - 1)
        ctrl_c = random.randint(c0 + 1, c1 - 1)
        grid[ctrl_r, ctrl_c] = spec["control_color"]
        spec["bbox"] = (r0, c0, r1, c1)
        spec["control_pos"] = (ctrl_r, ctrl_c)

    return grid


def _render_output(inp: np.ndarray, legend: dict[int, np.ndarray], specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        stamp = _scale_mask(legend[spec["control_color"]], inner_h // 2, inner_w // 2)
        target = out[r0 + 1 : r1, c0 + 1 : c1]
        target.fill(BG)
        target[stamp] = spec["frame_color"]
    return out


def _oracle_accepts(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _rendered_signatures(legend: dict[int, np.ndarray], specs: list[dict]) -> set[tuple[tuple[int, int], bytes]]:
    signatures = set()
    for spec in specs:
        size = (spec["inner_h"], spec["inner_w"])
        stamp = _scale_mask(legend[spec["control_color"]], spec["inner_h"] // 2, spec["inner_w"] // 2)
        signatures.add((size, stamp.tobytes()))
    return signatures


def _gen_attempt() -> np.ndarray | None:
    num_entries = random.randint(2, 3)
    marker_colors, legend = _choose_legend_entries(num_entries)
    specs = _build_frame_specs(marker_colors)
    if not _specs_cover_counterexamples(specs, marker_colors):
        return None

    spacing = random.randint(4, 5)
    marker_row = random.randint(0, 1)
    marker_col0 = random.randint(0, 1)
    legend_box = _legend_box(marker_row, marker_col0, spacing, num_entries)

    min_height = max(18, legend_box[2] + 4)
    min_width = max(18, legend_box[3] + 4)
    height = random.randint(min_height, 24)
    width = random.randint(min_width, 24)

    frame_boxes = _place_frames((height, width), legend_box, specs)
    if frame_boxes is None:
        return None

    inp = _render_input((height, width), marker_row, marker_col0, spacing, legend, specs, frame_boxes)
    expected = _render_output(inp, legend, specs)

    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if np.array_equal(inp, expected):
        return None
    if len(_rendered_signatures(legend, specs)) < 2:
        return None
    if not _oracle_accepts(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
