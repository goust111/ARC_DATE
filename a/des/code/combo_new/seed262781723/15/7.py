from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
LEGEND_DEMO_COLOR = 5
LEGEND_PATTERN_COLOR = 8
SEED_COLORS = (1, 2, 3, 4, 6, 7, 9)
SYMMETRY_KINDS = ("H", "V", "HV")
PATTERN_SIZES = (5, 7)
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[dict[str, object]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )

    return components


def _component_mask(component: dict[str, object]) -> np.ndarray:
    coords = np.asarray(component["coords"], dtype=int)
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _symmetry_kind(mask: np.ndarray) -> str:
    horizontal = bool(np.array_equal(mask, mask[::-1, :]))
    vertical = bool(np.array_equal(mask, mask[:, ::-1]))
    assert horizontal or vertical, "seed/demo must match one of the legend symmetry types"
    if horizontal and vertical:
        return "HV"
    if horizontal:
        return "H"
    return "V"


def _component_row_center(component: dict[str, object]) -> float:
    r0, _, r1, _ = component["bbox"]
    return 0.5 * (r0 + r1)


def _component_offsets(component: dict[str, object]) -> tuple[tuple[int, int], ...]:
    coords = np.asarray(component["coords"], dtype=int)
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h % 2 == 1 and w % 2 == 1, "legend patterns and seeds must have odd dimensions"
    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    return tuple(sorted((int(r - cr), int(c - cc)) for r, c in coords))


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    grid[top : top + h, left : left + w][mask] = color


def _make_horizontal_mask(size: int) -> np.ndarray:
    mid = size // 2
    widths = [size]
    prev = random.randint(1, size - 1)
    widths.append(prev)
    for _ in range(2, mid + 1):
        prev = random.randint(1, prev)
        widths.append(prev)

    mask = np.zeros((size, size), dtype=bool)
    for r in range(size):
        width = widths[abs(r - mid)]
        mask[r, :width] = True
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return mask


def _make_both_mask(size: int) -> np.ndarray:
    mid = size // 2
    half_widths = [mid]
    prev = mid
    for _ in range(1, mid + 1):
        prev = random.randint(0, prev)
        half_widths.append(prev)

    mask = np.zeros((size, size), dtype=bool)
    for r in range(size):
        half = half_widths[abs(r - mid)]
        mask[r, mid - half : mid + half + 1] = True
    return mask


def _make_mask(kind: str, size: int) -> np.ndarray:
    if kind == "H":
        return _make_horizontal_mask(size)
    if kind == "V":
        return _make_horizontal_mask(size).T
    assert kind == "HV"
    return _make_both_mask(size)


def main(grid: np.ndarray) -> np.ndarray:
    """Grow each colored 3x3 seed by selecting the legend pattern that matches its reflection symmetry.

    Input structure:
    - The background is black.
    - On the left, the legend has three rows; each row contains a grey 3x3 demo sprite and, to its right, an azure growth pattern.
    - The three grey demo sprites encode the three seed symmetry classes: horizontal-only, vertical-only, and symmetric across both axes.
    - On the right, several non-black, non-grey, non-azure 3x3 monochrome seed objects appear, each belonging to exactly one of those symmetry classes.

    Rule:
    - Keep the legend unchanged.
    - Determine the symmetry class of each seed by checking whether its 3x3 shape is horizontally symmetric, vertically symmetric, or both.
    - Find the legend row whose grey demo has the same symmetry class.
    - Center the matched azure growth pattern on the seed's center and recolor those pattern cells with the seed's color, preserving the seed's original pixels.

    Key insight:
    - The grey legend sprites are not copied directly; they classify seeds by hidden symmetry, and that class chooses which larger azure template gets stamped around each seed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _dominant_color(g)

    demo_components = _components(g == LEGEND_DEMO_COLOR)
    pattern_components = _components(g == LEGEND_PATTERN_COLOR)
    assert len(demo_components) == 3 and len(pattern_components) == 3

    demo_components.sort(key=_component_row_center)
    pattern_components.sort(key=_component_row_center)

    legend_patterns: dict[str, tuple[tuple[int, int], ...]] = {}
    for demo, pattern in zip(demo_components, pattern_components):
        demo_mask = _component_mask(demo)
        assert demo_mask.shape == (3, 3), "legend demos must be 3x3"
        assert pattern["bbox"][1] > demo["bbox"][3], "each growth pattern must sit to the right of its demo"
        kind = _symmetry_kind(demo_mask)
        assert kind not in legend_patterns, "legend must cover each symmetry class exactly once"
        legend_patterns[kind] = _component_offsets(pattern)

    assert set(legend_patterns) == set(SYMMETRY_KINDS)

    out = g.copy()
    seed_colors = [int(color) for color in np.unique(g) if int(color) not in {bg, LEGEND_DEMO_COLOR, LEGEND_PATTERN_COLOR}]
    assert seed_colors, "expected at least one seed object"

    for color in seed_colors:
        for component in _components(g == color):
            mask = _component_mask(component)
            assert mask.shape == (3, 3), "seed objects must be 3x3"
            kind = _symmetry_kind(mask)
            r0, c0, r1, c1 = component["bbox"]
            cr = (r0 + r1) // 2
            cc = (c0 + c1) // 2
            for dr, dc in legend_patterns[kind]:
                rr = cr + dr
                cc2 = cc + dc
                assert 0 <= rr < g.shape[0] and 0 <= cc2 < g.shape[1]
                current = int(out[rr, cc2])
                assert current in (bg, color), "grown patterns must not overlap other objects"
                out[rr, cc2] = color

    return out


def _choose_seed_kinds() -> list[str]:
    extra_count = random.randint(0, 2)
    kinds = list(SYMMETRY_KINDS) + random.sample(SYMMETRY_KINDS, extra_count)
    random.shuffle(kinds)
    return kinds


def _layout_legend(
    row_order: list[str], pattern_masks: dict[str, np.ndarray]
) -> tuple[list[tuple[str, int, int, int, int]], int, int]:
    left_margin = random.randint(0, 1)
    top_margin = random.randint(0, 2)
    demo_gap = random.randint(1, 2)
    row_gaps = [random.randint(1, 2) for _ in range(len(row_order) - 1)]

    cursor = top_margin
    placements: list[tuple[str, int, int, int, int]] = []
    legend_right = 0
    for idx, kind in enumerate(row_order):
        pattern_size = pattern_masks[kind].shape[0]
        row_height = max(3, pattern_size)
        demo_top = cursor + (row_height - 3) // 2
        pattern_top = cursor + (row_height - pattern_size) // 2
        demo_left = left_margin
        pattern_left = left_margin + 3 + demo_gap
        placements.append((kind, demo_top, demo_left, pattern_top, pattern_left))
        legend_right = max(legend_right, pattern_left + pattern_size - 1)
        cursor += row_height
        if idx < len(row_gaps):
            cursor += row_gaps[idx]

    total_height = cursor + random.randint(0, 2)
    return placements, legend_right, total_height


def _place_seed_boxes(
    grid_shape: tuple[int, int],
    seed_specs: list[dict[str, object]],
    min_col: int,
) -> list[tuple[int, int]] | None:
    h, w = grid_shape
    order = sorted(range(len(seed_specs)), key=lambda idx: -int(seed_specs[idx]["box_size"]))
    occupied = np.zeros((h, w), dtype=bool)
    boxes: list[tuple[int, int] | None] = [None] * len(seed_specs)

    for idx in order:
        size = int(seed_specs[idx]["box_size"])
        if min_col > w - size:
            return None
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(0, h - size)
            left = random.randint(min_col, w - size)
            r0 = max(0, top - 1)
            c0 = max(0, left - 1)
            r1 = min(h, top + size + 1)
            c1 = min(w, left + size + 1)
            if occupied[r0:r1, c0:c1].any():
                continue
            occupied[r0:r1, c0:c1] = True
            boxes[idx] = (top, left)
            break
        else:
            return None

    return [box for box in boxes if box is not None]


def _build_expected_pair(
    grid_shape: tuple[int, int],
    row_order: list[str],
    demo_masks: dict[str, np.ndarray],
    pattern_masks: dict[str, np.ndarray],
    legend_placements: list[tuple[str, int, int, int, int]],
    seed_specs: list[dict[str, object]],
    seed_boxes: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(grid_shape, BACKGROUND, dtype=int)
    out = inp.copy()

    for kind, demo_top, demo_left, pattern_top, pattern_left in legend_placements:
        _paint_mask(inp, demo_masks[kind], demo_top, demo_left, LEGEND_DEMO_COLOR)
        _paint_mask(out, demo_masks[kind], demo_top, demo_left, LEGEND_DEMO_COLOR)
        _paint_mask(inp, pattern_masks[kind], pattern_top, pattern_left, LEGEND_PATTERN_COLOR)
        _paint_mask(out, pattern_masks[kind], pattern_top, pattern_left, LEGEND_PATTERN_COLOR)

    for spec, (top, left) in zip(seed_specs, seed_boxes):
        color = int(spec["color"])
        box_size = int(spec["box_size"])
        seed_mask = np.asarray(spec["seed_mask"], dtype=bool)
        pattern_mask = np.asarray(spec["pattern_mask"], dtype=bool)
        seed_top = top + box_size // 2 - 1
        seed_left = left + box_size // 2 - 1

        _paint_mask(inp, seed_mask, seed_top, seed_left, color)
        _paint_mask(out, pattern_mask, top, left, color)
        _paint_mask(out, seed_mask, seed_top, seed_left, color)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    row_order = list(SYMMETRY_KINDS)
    random.shuffle(row_order)

    demo_masks = {kind: _make_mask(kind, 3) for kind in SYMMETRY_KINDS}
    pattern_masks = {kind: _make_mask(kind, random.choice(PATTERN_SIZES)) for kind in SYMMETRY_KINDS}
    legend_placements, legend_right, legend_height = _layout_legend(row_order, pattern_masks)

    seed_kinds = _choose_seed_kinds()
    seed_colors = random.sample(SEED_COLORS, len(seed_kinds))
    seed_specs = []
    for kind, color in zip(seed_kinds, seed_colors):
        pattern_mask = pattern_masks[kind]
        seed_specs.append(
            {
                "kind": kind,
                "color": color,
                "seed_mask": _make_mask(kind, 3),
                "pattern_mask": pattern_mask,
                "box_size": int(pattern_mask.shape[0]),
            }
        )

    max_box = max(int(spec["box_size"]) for spec in seed_specs)
    seed_gap = random.randint(3, 4)
    min_seed_col = legend_right + 1 + seed_gap

    min_height = max(legend_height, max_box, 18)
    min_width = max(min_seed_col + max_box, 22)
    if min_height > 30 or min_width > 30:
        return None

    height = random.randint(min_height, 30)
    width = random.randint(min_width, 30)
    seed_boxes = _place_seed_boxes((height, width), seed_specs, min_seed_col)
    if seed_boxes is None:
        return None

    inp, expected = _build_expected_pair(
        (height, width), row_order, demo_masks, pattern_masks, legend_placements, seed_specs, seed_boxes
    )
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-legend ARC input")
