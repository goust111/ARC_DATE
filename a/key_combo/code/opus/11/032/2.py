from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
SPRITE_COLORS = (BLUE, RED, GREEN)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 400

LAYOUTS = (
    {
        "frame_shape": (15, 18),
        "holes": [(9, 1, 1), (6, 1, 11), (3, 11, 7)],
    },
    {
        "frame_shape": (18, 17),
        "holes": [(9, 1, 1), (3, 1, 13), (6, 11, 6)],
    },
    {
        "frame_shape": (18, 14),
        "holes": [(6, 1, 1), (3, 1, 10), (9, 8, 3)],
    },
    {
        "frame_shape": (15, 21),
        "holes": [(6, 1, 1), (9, 1, 9), (3, 11, 15)],
    },
)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_coords(coords)


def _crop_bbox(mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask.astype(bool))
    return len(comps) == 1


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((factor, factor), dtype=int)).astype(bool)


def _paint_template_into_hole(
    out: np.ndarray,
    template: np.ndarray,
    hole_box: tuple[int, int, int, int],
    color: int,
) -> None:
    r0, c0, r1, c1 = hole_box
    side = r1 - r0 + 1
    assert side == c1 - c0 + 1
    assert side % 3 == 0
    factor = side // 3
    out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    scaled = _scale_mask(template, factor)
    out[r0 : r1 + 1, c0 : c1 + 1][scaled] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the full 3x3 template from three complementary partial sprites, then scale and place it into marked square holes.

    Input structure:
    - Black background with three 3x3 monochrome partial sprites above one large gray rectangular frame.
    - The sprite colors are blue, red, and green; each sprite occupies a separate 3x3 window and is missing different template cells.
    - Inside the gray frame are three isolated square holes, each mostly black with one colored marker pixel at its top-left cell.

    Rule:
    - Recover the complete 3x3 template mask by taking the top-left-aligned union of the three partial sprite windows.
    - For each hole, compute the scale factor as hole_side / 3, clear the whole hole, and draw the scaled template anchored at the hole's top-left corner.
    - Use the hole's marker color for every filled template pixel, erase the original upper sprites, and keep the gray frame plus outer black background unchanged.

    Key insight:
    - The upper sprites jointly determine shape, the hole size determines scale, and the hole marker determines color; all three clues must be combined to produce the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    grey_components = _components(g == GREY)
    assert grey_components, "expected a gray frame"
    frame_coords = max(grey_components, key=lambda comp: int(comp.shape[0]))
    frame_r0, frame_c0, frame_r1, frame_c1 = _bbox_coords(frame_coords)

    upper = g[:frame_r0, :]
    template = np.zeros((3, 3), dtype=bool)
    for color in SPRITE_COLORS:
        mask = upper == color
        box = _bbox(mask)
        assert box is not None, f"missing upper sprite color {color}"
        r0, c0, r1, c1 = box
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "each sprite window must be 3x3"
        template |= _crop_bbox(mask, box)

    assert 5 <= int(template.sum()) <= 8
    assert _is_connected(template)

    frame = g[frame_r0 : frame_r1 + 1, frame_c0 : frame_c1 + 1]
    hole_components = _components(frame != GREY)
    assert len(hole_components) == 3, "expected exactly three holes"

    out = g.copy()
    upper_mask = np.isin(out[:frame_r0, :], SPRITE_COLORS)
    out[:frame_r0, :][upper_mask] = BLACK

    seen_sizes: list[int] = []
    seen_markers: set[int] = set()
    for comp in hole_components:
        hr0, hc0, hr1, hc1 = _bbox_coords(comp)
        side = hr1 - hr0 + 1
        assert side == hc1 - hc0 + 1, "hole must be square"
        hole = frame[hr0 : hr1 + 1, hc0 : hc1 + 1]
        marker_colors = sorted(int(v) for v in np.unique(hole) if int(v) in SPRITE_COLORS)
        assert len(marker_colors) == 1, "each hole must have exactly one marker color"
        color = marker_colors[0]
        seen_sizes.append(side)
        seen_markers.add(color)
        _paint_template_into_hole(
            out,
            template,
            (frame_r0 + hr0, frame_c0 + hc0, frame_r0 + hr1, frame_c0 + hc1),
            color,
        )

    assert sorted(seen_sizes) == [3, 6, 9]
    assert seen_markers == set(SPRITE_COLORS)
    return out


def _random_template(rng: random.Random) -> np.ndarray | None:
    cells = [(r, c) for r in range(3) for c in range(3)]
    area = rng.randint(6, 8)
    target_top_left = rng.choice([False, True])
    for _ in range(300):
        chosen = rng.sample(cells, area)
        mask = np.zeros((3, 3), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        if mask[0, 0] != target_top_left:
            continue
        if _bbox(mask) != (0, 0, 2, 2):
            continue
        if not _is_connected(mask):
            continue
        return mask
    return None


def _assign_partials(
    template: np.ndarray, rng: random.Random
) -> tuple[dict[int, np.ndarray], list[np.ndarray]] | None:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(template)]
    for _ in range(500):
        unique_cells = rng.sample(cells, 3)
        memberships: dict[tuple[int, int], int] = {}
        for idx, cell in enumerate(unique_cells):
            memberships[cell] = 1 << idx
        for cell in cells:
            if cell in memberships:
                continue
            memberships[cell] = rng.choice([3, 5, 6, 7])

        partials: list[np.ndarray] = []
        ok = True
        for bit in (1, 2, 4):
            mask = np.zeros((3, 3), dtype=bool)
            for (r, c), bits in memberships.items():
                if bits & bit:
                    mask[r, c] = True
            area = int(mask.sum())
            if not (4 <= area <= 7):
                ok = False
                break
            if _bbox(mask) != (0, 0, 2, 2):
                ok = False
                break
            if np.array_equal(mask, template):
                ok = False
                break
            partials.append(mask)
        if not ok:
            continue

        if not np.array_equal(np.logical_or.reduce(partials), template):
            continue
        pair_indices = ((0, 1), (0, 2), (1, 2))
        if any(np.array_equal(np.logical_or(partials[i], partials[j]), template) for i, j in pair_indices):
            continue
        return dict(zip(SPRITE_COLORS, partials, strict=True)), partials
    return None


def _build_expected_from_spec(
    input_grid: np.ndarray,
    frame_box: tuple[int, int, int, int],
    template: np.ndarray,
    hole_specs: list[dict[str, int]],
) -> np.ndarray:
    out = input_grid.copy()
    frame_r0 = frame_box[0]
    upper_mask = np.isin(out[:frame_r0, :], SPRITE_COLORS)
    out[:frame_r0, :][upper_mask] = BLACK
    for spec in hole_specs:
        _paint_template_into_hole(out, template, spec["box"], spec["color"])
    return out


def _render_case(
    rng: random.Random,
    template: np.ndarray,
    color_partials: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]] | None:
    layout = rng.choice(LAYOUTS)
    frame_h, frame_w = layout["frame_shape"]
    top_gap1 = rng.randint(2, 4)
    top_gap2 = rng.randint(2, 4)
    top_span = 9 + top_gap1 + top_gap2
    min_w = max(frame_w + 2, top_span + 2)
    max_w = min(29, min_w + 4)
    if min_w > max_w:
        return None
    grid_w = rng.randint(min_w, max_w)

    top_r0 = rng.randint(1, 3)
    between_top_and_frame = rng.randint(2, 4)
    frame_r0 = top_r0 + 3 + between_top_and_frame
    min_h = frame_r0 + frame_h + 1
    max_h = min(29, min_h + 3)
    if min_h > max_h:
        return None
    grid_h = rng.randint(min_h, max_h)

    top_c0 = rng.randint(1, grid_w - top_span - 1)
    frame_c0 = rng.randint(1, grid_w - frame_w - 1)

    grid = np.zeros((grid_h, grid_w), dtype=int)

    color_order = list(SPRITE_COLORS)
    rng.shuffle(color_order)
    sprite_cols = [top_c0, top_c0 + 3 + top_gap1, top_c0 + 6 + top_gap1 + top_gap2]
    for color, c0 in zip(color_order, sprite_cols, strict=True):
        sprite = color_partials[color]
        region = grid[top_r0 : top_r0 + 3, c0 : c0 + 3]
        region[sprite] = color

    grid[frame_r0 : frame_r0 + frame_h, frame_c0 : frame_c0 + frame_w] = GREY

    marker_colors = list(SPRITE_COLORS)
    rng.shuffle(marker_colors)
    hole_specs: list[dict[str, int]] = []
    for (side, local_r0, local_c0), color in zip(layout["holes"], marker_colors, strict=True):
        r0 = frame_r0 + local_r0
        c0 = frame_c0 + local_c0
        grid[r0 : r0 + side, c0 : c0 + side] = BLACK
        grid[r0, c0] = color
        hole_specs.append({"box": (r0, c0, r0 + side - 1, c0 + side - 1), "color": color, "side": side})

    expected = _build_expected_from_spec(
        grid,
        (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1),
        template,
        hole_specs,
    )
    return grid, expected, [color_partials[color].copy() for color in SPRITE_COLORS]


def _variant_with_template(
    input_grid: np.ndarray,
    frame_box: tuple[int, int, int, int],
    hole_specs: list[dict[str, int]],
    template: np.ndarray,
) -> np.ndarray:
    return _build_expected_from_spec(input_grid, frame_box, template, hole_specs)


def _extract_hole_specs(
    input_grid: np.ndarray, frame_box: tuple[int, int, int, int]
) -> list[dict[str, int]]:
    frame_r0, frame_c0, frame_r1, frame_c1 = frame_box
    frame = input_grid[frame_r0 : frame_r1 + 1, frame_c0 : frame_c1 + 1]
    specs: list[dict[str, int]] = []
    for comp in _components(frame != GREY):
        r0, c0, r1, c1 = _bbox_coords(comp)
        hole = frame[r0 : r1 + 1, c0 : c1 + 1]
        colors = [int(v) for v in np.unique(hole) if int(v) in SPRITE_COLORS]
        specs.append({"box": (frame_r0 + r0, frame_c0 + c0, frame_r0 + r1, frame_c0 + c1), "color": colors[0], "side": r1 - r0 + 1})
    return specs


def _validate_case(
    input_grid: np.ndarray,
    expected: np.ndarray,
    template: np.ndarray,
    partials: list[np.ndarray],
) -> bool:
    if input_grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(input_grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if not np.array_equal(main(input_grid), expected):
        return False

    grey_components = _components(input_grid == GREY)
    if len(grey_components) != 1:
        return False
    frame_box = _bbox_coords(grey_components[0])
    hole_specs = _extract_hole_specs(input_grid, frame_box)
    if sorted(spec["side"] for spec in hole_specs) != [3, 6, 9]:
        return False
    if {spec["color"] for spec in hole_specs} != set(SPRITE_COLORS):
        return False

    if _bbox(template) != (0, 0, 2, 2):
        return False
    if not _is_connected(template):
        return False

    for part in partials:
        if _bbox(part) != (0, 0, 2, 2):
            return False
        if not (4 <= int(part.sum()) <= 7):
            return False
        if np.array_equal(part, template):
            return False

    for subset in ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2)):
        merged = np.zeros((3, 3), dtype=bool)
        for idx in subset:
            merged |= partials[idx]
        if np.array_equal(merged, template):
            return False
        alt = _variant_with_template(input_grid, frame_box, hole_specs, merged)
        if np.array_equal(alt, expected):
            return False

    return True


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    template = _random_template(rng)
    if template is None:
        return None
    assignment = _assign_partials(template, rng)
    if assignment is None:
        return None
    color_partials, partials = assignment
    rendered = _render_case(rng, template, color_partials)
    if rendered is None:
        return None
    input_grid, expected, ordered_partials = rendered
    if not _validate_case(input_grid, expected, template, ordered_partials):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-template hole-filling input")
