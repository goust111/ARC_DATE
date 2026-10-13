from __future__ import annotations

import random

import numpy as np


BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]

    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _count_enclosed_holes(coords: np.ndarray) -> int:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True

    holes = 0
    for bg_comp in _components(~local, connectivity=4):
        rows = bg_comp[:, 0]
        cols = bg_comp[:, 1]
        touches_border = (
            np.any(rows == 0)
            or np.any(rows == local.shape[0] - 1)
            or np.any(cols == 0)
            or np.any(cols == local.shape[1] - 1)
        )
        if not touches_border:
            holes += 1
    return holes


def _extract_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    strip_mask = grid[:, :2] != bg
    legend_mask = np.zeros_like(grid, dtype=bool)
    mapping: dict[int, int] = {}

    for comp in _components(strip_mask, connectivity=4):
        r0, c0, r1, c1 = _bbox_from_coords(comp)
        if comp.shape[0] != 2 or r0 != r1 or c0 != 0 or c1 != 1:
            continue
        source = int(grid[r0, 0])
        target = int(grid[r0, 1])
        assert source != bg and target != bg and source != target, "legend pairs must be 1x2 non-background mappings"
        assert source not in mapping, "each source color must have exactly one legend pair"
        mapping[source] = target
        legend_mask[r0, 0:2] = True

    assert mapping, "expected at least one left-edge 1x2 legend pair"
    return mapping, legend_mask


def _iterate_color(color: int, steps: int, mapping: dict[int, int]) -> int:
    cur = int(color)
    for _ in range(steps):
        assert cur in mapping, "legend chain must define every requested iteration step"
        cur = int(mapping[cur])
    return cur


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each object by iterating the left-edge legend mapping once per enclosed hole.

    Input structure:
    - The grid has a dominant background color.
    - Along the left edge sit several isolated 1x2 legend pairs; in each pair the left cell is a
      source color and the right cell is its mapped color.
    - Elsewhere are separated monochrome 4-connected objects whose colors appear in the legend's
      source set, and each object encloses zero or more background holes.

    Rule:
    - Read the left-edge 1x2 pairs as a color-to-color function.
    - For each non-legend object, count how many 4-connected background components are enclosed
      strictly inside the object's bbox.
    - Starting from the object's original color, apply the legend function that many times and
      recolor the whole object to the resulting color, leaving holes and background unchanged.

    Key insight:
    - The legend is a reusable function, not a one-step lookup table; each object's topology
      determines how many times the same mapping must be iterated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    mapping, legend_mask = _extract_legend(g, bg)

    object_mask = (g != bg) & ~legend_mask
    objects = _components(object_mask, connectivity=4)
    assert objects, "expected at least one non-legend object"

    out = g.copy()
    for coords in objects:
        colors = np.unique(g[coords[:, 0], coords[:, 1]])
        assert colors.size == 1, "each object must be monochrome"
        source = int(colors[0])
        assert source in mapping, "object colors must come from the legend source set"

        holes = _count_enclosed_holes(coords)
        target = _iterate_color(source, holes, mapping)
        out[coords[:, 0], coords[:, 1]] = target

    return out


BACKGROUND = 0
GRID_HEIGHT_RANGE = (20, 29)
GRID_WIDTH_RANGE = (20, 29)
OBJECT_MIN_COL = 3
MAX_PLACEMENT_ATTEMPTS = 250
MAX_GENERATION_ATTEMPTS = 400
PALETTE = tuple(range(1, 10))


def _boxes_overlap(a: BBox, b: BBox) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def _expand_box(r0: int, c0: int, h: int, w: int, shape: tuple[int, int], pad: int = 1) -> BBox:
    max_r, max_c = shape
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(max_r - 1, r0 + h - 1 + pad),
        min(max_c - 1, c0 + w - 1 + pad),
    )


def _make_object_patch(color: int, holes: int) -> np.ndarray:
    if holes == 0:
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        return np.full((h, w), int(color), dtype=int)

    orient = random.choice(["horizontal", "vertical"])
    if orient == "horizontal":
        top = random.randint(1, 2)
        bottom = random.randint(1, 2)
        left = random.randint(1, 2)
        right = random.randint(1, 2)
        hole_h = random.randint(1, 2)
        hole_w = 1 if holes >= 2 else random.randint(1, 2)
        gap = 1
        patch = np.full((top + hole_h + bottom, left + holes * hole_w + (holes - 1) * gap + right), int(color), dtype=int)
        for idx in range(holes):
            c0 = left + idx * (hole_w + gap)
            patch[top : top + hole_h, c0 : c0 + hole_w] = BACKGROUND
        return patch

    top = random.randint(1, 2)
    bottom = random.randint(1, 2)
    left = random.randint(1, 2)
    right = random.randint(1, 2)
    hole_h = 1 if holes >= 2 else random.randint(1, 2)
    hole_w = random.randint(1, 2)
    gap = 1
    patch = np.full((top + holes * hole_h + (holes - 1) * gap + bottom, left + hole_w + right), int(color), dtype=int)
    for idx in range(holes):
        r0 = top + idx * (hole_h + gap)
        patch[r0 : r0 + hole_h, left : left + hole_w] = BACKGROUND
    return patch


def _sample_mapping() -> tuple[dict[int, int], list[int], str]:
    if random.random() < 0.5:
        count = random.randint(4, 6)
        order = random.sample(PALETTE, count)
        mapping = {order[idx]: order[(idx + 1) % count] for idx in range(count)}
        return mapping, order, "ring"

    source_count = random.randint(5, 6)
    colors = random.sample(PALETTE, source_count + 1)
    sources = colors[:source_count]
    random.shuffle(sources)
    sink = colors[-1]
    mapping = {sources[idx]: sources[idx + 1] for idx in range(source_count - 1)}
    mapping[sources[-1]] = sink
    return mapping, sources, "chain"


def _legend_rows(height: int, count: int) -> list[int]:
    gaps = [random.randint(1, 2) for _ in range(count - 1)]
    span = count + sum(gaps)
    start = random.randint(0, height - span)
    rows = [start]
    for gap in gaps:
        rows.append(rows[-1] + 1 + gap)
    return rows


def _valid_sources(mapping: dict[int, int], sources: list[int], holes: int) -> list[int]:
    valid = []
    for color in sources:
        try:
            _iterate_color(color, holes, mapping)
        except AssertionError:
            continue
        valid.append(color)
    return valid


def _choose_object_specs(mapping: dict[int, int], sources: list[int], mode: str) -> list[tuple[int, int]]:
    specs: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for holes in range(4):
        valid = _valid_sources(mapping, sources, holes)
        assert valid, "each required hole count must have at least one valid source"
        unused = [color for color in valid if (holes, color) not in used]
        color = random.choice(unused or valid)
        specs.append((holes, color))
        used.add((holes, color))

    extra_count = random.randint(0, 2)
    for _ in range(extra_count):
        holes = random.randint(0, 3)
        color = random.choice(_valid_sources(mapping, sources, holes))
        specs.append((holes, color))

    if mode == "chain" and random.random() < 0.8:
        terminal_holes = random.randint(1, 3)
        valid = []
        for color in sources:
            try:
                target = _iterate_color(color, terminal_holes, mapping)
            except AssertionError:
                continue
            if target not in sources:
                valid.append(color)
        if valid:
            specs.append((terminal_holes, random.choice(valid)))

    random.shuffle(specs)
    return specs


def _place_patches(
    shape: tuple[int, int],
    patches: list[np.ndarray],
    legend_rows: list[int],
) -> list[tuple[int, int]] | None:
    height, width = shape
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    occupied_boxes: list[BBox] = []
    legend_row_set = set(legend_rows)

    order = sorted(range(len(patches)), key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)
    for idx in order:
        patch = patches[idx]
        ph, pw = map(int, patch.shape)
        if pw > width - OBJECT_MIN_COL:
            return None

        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, height - ph)
            c0 = random.randint(OBJECT_MIN_COL, width - pw)
            if set(range(r0, r0 + ph)) & legend_row_set:
                continue

            box = _expand_box(r0, c0, ph, pw, shape)
            if any(_boxes_overlap(box, other) for other in occupied_boxes):
                continue

            occupied_boxes.append(box)
            placements[idx] = (r0, c0)
            break
        else:
            return None

    assert all(item is not None for item in placements)
    return [(int(r0), int(c0)) for r0, c0 in placements if r0 is not None and c0 is not None]


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    mask = patch != BACKGROUND
    region = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    region[mask] = patch[mask]


def _render_grids(
    height: int,
    width: int,
    mapping: dict[int, int],
    legend_rows: list[int],
    specs: list[tuple[int, int]],
    patches: list[np.ndarray],
    placements: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), BACKGROUND, dtype=int)
    expected = np.full((height, width), BACKGROUND, dtype=int)

    legend_pairs = list(mapping.items())
    random.shuffle(legend_pairs)
    for row, (source, target) in zip(legend_rows, legend_pairs):
        inp[row, 0] = source
        inp[row, 1] = target
        expected[row, 0] = source
        expected[row, 1] = target

    for (holes, source), patch, (r0, c0) in zip(specs, patches, placements):
        source_patch = patch.copy()
        source_patch[source_patch != BACKGROUND] = source
        _paint_patch(inp, source_patch, r0, c0)

        target = _iterate_color(source, holes, mapping)
        target_patch = patch.copy()
        target_patch[target_patch != BACKGROUND] = target
        _paint_patch(expected, target_patch, r0, c0)

    return inp, expected


def _validate_structure(grid: np.ndarray, specs: list[tuple[int, int]]) -> bool:
    bg = _get_bg(grid)
    mapping, legend_mask = _extract_legend(grid, bg)
    object_mask = (grid != bg) & ~legend_mask
    comps = _components(object_mask, connectivity=4)
    if len(comps) != len(specs):
        return False
    hole_counts = {_count_enclosed_holes(comp) for comp in comps}
    return {0, 1, 2, 3}.issubset(hole_counts)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)

    mapping, sources, mode = _sample_mapping()
    legend_rows = _legend_rows(height, len(mapping))
    specs = _choose_object_specs(mapping, sources, mode)
    patches = [_make_object_patch(source, holes) for holes, source in specs]
    placements = _place_patches((height, width), patches, legend_rows)
    if placements is None or len(placements) != len(patches):
        return None

    inp, expected = _render_grids(height, width, mapping, legend_rows, specs, patches, placements)
    if not _validate_structure(inp, specs):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
