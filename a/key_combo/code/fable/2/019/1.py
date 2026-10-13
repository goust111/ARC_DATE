from __future__ import annotations

from collections import deque
from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

Color = _ARC_UTILS.Color
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
paste = _ARC_UTILS.paste


BG = int(Color.BLACK)


def _enclosed_hole_mask(outline_mask: np.ndarray) -> np.ndarray:
    """Return the cells strictly inside a 4-connected outline crop."""
    h, w = map(int, outline_mask.shape)
    outside = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not outline_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not outline_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not outline_mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                queue.append((nr, nc))

    return (~outline_mask) & (~outside)


def _extract_shapes(grid: np.ndarray) -> list[dict]:
    components = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    outlines = [comp for comp in components if int(comp["area"]) > 1]
    markers = [comp for comp in components if int(comp["area"]) == 1]

    assert len(outlines) >= 2, "expected at least two hollow outline shapes"
    assert len(markers) == len(outlines), "expected exactly one marker per outline"

    outline_colors = [int(comp["color"]) for comp in outlines]
    marker_colors = [int(comp["color"]) for comp in markers]
    assert len(set(outline_colors)) == len(outline_colors), "outline colors must be unique"
    assert set(marker_colors) == set(outline_colors), "marker colors must permute the outline colors"

    used_markers: set[int] = set()
    shapes = []
    for outline in outlines:
        color = int(outline["color"])
        bbox = tuple(map(int, outline["bbox"]))
        r0, c0, r1, c1 = bbox
        patch = crop_bbox(grid, bbox)
        outline_mask = patch == color
        hole_mask = _enclosed_hole_mask(outline_mask)
        assert hole_mask.any(), "each outline must enclose an interior region"

        candidate_ids = []
        for idx, marker in enumerate(markers):
            mr, mc = map(int, marker["coords"][0])
            if r0 <= mr <= r1 and c0 <= mc <= c1 and hole_mask[mr - r0, mc - c0]:
                candidate_ids.append(idx)

        assert len(candidate_ids) == 1, "each outline must contain exactly one marker pixel"
        marker_id = candidate_ids[0]
        assert marker_id not in used_markers, "marker assignments must be unique"
        used_markers.add(marker_id)

        marker = markers[marker_id]
        shapes.append(
            {
                "outline_color": color,
                "marker_color": int(marker["color"]),
                "bbox": bbox,
                "top_left": (r0, c0),
                "patch": patch,
            }
        )

    assert len(used_markers) == len(markers), "every marker must belong to one outline"
    return sorted(shapes, key=lambda shape: shape["top_left"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each hollow shape to the original position of the shape named by its interior marker color.

    Input structure:
    - The black background contains several separated monochrome hollow outlines, and each outline color is unique.
    - Each outline encloses exactly one isolated marker pixel somewhere in its interior hole.
    - The marker colors are a permutation of the outline colors, so every outline color is pointed to exactly once.

    Rule:
    - For each outline, find the target outline whose outline color matches the marker color inside the source outline.
    - Translate the whole source object, including both its outline and its marker pixel, so the source bbox top-left lands on the target outline's original bbox top-left.
    - Resolve every move simultaneously from the input snapshot, leaving black background elsewhere.

    Key insight:
    - The interior dot is an address, not a repaint instruction: colors define a permutation on the outline slots, so whole objects cycle through the original bbox anchors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    shapes = _extract_shapes(g)
    by_color = {shape["outline_color"]: shape for shape in shapes}

    out = np.full_like(g, BG)
    occupied = np.zeros(g.shape, dtype=bool)
    for shape in shapes:
        target = by_color[shape["marker_color"]]
        top, left = target["top_left"]
        patch = shape["patch"]
        h, w = map(int, patch.shape)
        assert 0 <= top <= g.shape[0] - h and 0 <= left <= g.shape[1] - w, "moved shape must stay in bounds"

        mask = patch != BG
        region = occupied[top : top + h, left : left + w]
        assert not np.any(region & mask), "moved shapes must not overlap"
        region |= mask
        paste(out, patch, top, left, transparent=BG)

    return out


# --------------- generator (o5x) ---------------

MIN_SHAPES = 3
MAX_SHAPES = 5
SLOT_H = 7
SLOT_W = 7
MIN_GAP = 2
OUTER_MARGIN = 1
MAX_GRID = 30
MAX_ATTEMPTS = 400
OUTLINE_COLORS = [int(c) for c in Color.NOT_BLACK]

_COUNT_BAG: list[int] = []
_STYLE_BAG: list[str] = []


def _draw_frame(mask: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    bottom = top + height - 1
    right = left + width - 1
    mask[top, left : right + 1] = True
    mask[bottom, left : right + 1] = True
    mask[top : bottom + 1, left] = True
    mask[top : bottom + 1, right] = True


def _random_composition(total: int, parts: int) -> list[int]:
    total = int(total)
    parts = int(parts)
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    cuts = [-1] + cuts + [total + parts - 1]
    return [right - left - 1 for left, right in zip(cuts, cuts[1:])]


def _next_shape_count() -> int:
    global _COUNT_BAG
    if not _COUNT_BAG:
        _COUNT_BAG = list(range(MIN_SHAPES, MAX_SHAPES + 1))
        random.shuffle(_COUNT_BAG)
    return _COUNT_BAG.pop()


def _next_style(n_shapes: int) -> str:
    global _STYLE_BAG
    feasible = {"with_fixed", "single_cycle"}
    if n_shapes >= 4:
        feasible.add("split_cycles")

    while True:
        if not _STYLE_BAG:
            _STYLE_BAG = ["with_fixed", "single_cycle", "split_cycles"]
            random.shuffle(_STYLE_BAG)
        style = _STYLE_BAG.pop()
        if style in feasible:
            return style


def _make_geometry() -> dict:
    frame_h = random.randint(3, 5)
    frame_w = random.randint(3, 5)
    sides = ["top", "bottom", "left", "right"]
    tab_count = random.choices([0, 1, 2, 3], weights=[1, 3, 3, 1], k=1)[0]
    chosen = set(random.sample(sides, tab_count))

    top_ext = int("top" in chosen)
    bottom_ext = int("bottom" in chosen)
    left_ext = int("left" in chosen)
    right_ext = int("right" in chosen)

    mask = np.zeros((frame_h + top_ext + bottom_ext, frame_w + left_ext + right_ext), dtype=bool)
    frame_top = top_ext
    frame_left = left_ext
    _draw_frame(mask, frame_top, frame_left, frame_h, frame_w)

    if "top" in chosen:
        mask[0, frame_left + random.randrange(frame_w)] = True
    if "bottom" in chosen:
        mask[-1, frame_left + random.randrange(frame_w)] = True
    if "left" in chosen:
        mask[frame_top + random.randrange(frame_h), 0] = True
    if "right" in chosen:
        mask[frame_top + random.randrange(frame_h), -1] = True

    marker_pos = (
        random.randint(frame_top + 1, frame_top + frame_h - 2),
        random.randint(frame_left + 1, frame_left + frame_w - 2),
    )
    signature = (
        mask.shape,
        tuple(int(v) for v in mask.ravel()),
        marker_pos[0] - frame_top,
        marker_pos[1] - frame_left,
    )
    return {"mask": mask, "marker_pos": marker_pos, "signature": signature}


def _layout_candidates(n_shapes: int) -> list[tuple[int, int]]:
    candidates = []
    for rows in range(1, 4):
        for cols in range(1, 4):
            capacity = rows * cols
            if capacity < n_shapes or capacity > n_shapes + 2:
                continue
            base_h = rows * SLOT_H + (rows - 1) * MIN_GAP + 2 * OUTER_MARGIN
            base_w = cols * SLOT_W + (cols - 1) * MIN_GAP + 2 * OUTER_MARGIN
            if base_h <= MAX_GRID and base_w <= MAX_GRID:
                candidates.append((rows, cols))
    return candidates


def _sample_anchors(n_shapes: int) -> tuple[tuple[int, int], dict[int, tuple[int, int]]]:
    rows, cols = random.choice(_layout_candidates(n_shapes))
    base_h = rows * SLOT_H + (rows - 1) * MIN_GAP + 2 * OUTER_MARGIN
    base_w = cols * SLOT_W + (cols - 1) * MIN_GAP + 2 * OUTER_MARGIN

    extra_h = random.randint(0, MAX_GRID - base_h)
    extra_w = random.randint(0, MAX_GRID - base_w)
    row_extra = _random_composition(extra_h, rows + 1)
    col_extra = _random_composition(extra_w, cols + 1)

    anchor_rows = []
    cursor = OUTER_MARGIN + row_extra[0]
    for idx in range(rows):
        anchor_rows.append(cursor)
        if idx < rows - 1:
            cursor += SLOT_H + MIN_GAP + row_extra[idx + 1]
    grid_h = anchor_rows[-1] + SLOT_H + OUTER_MARGIN + row_extra[-1]

    anchor_cols = []
    cursor = OUTER_MARGIN + col_extra[0]
    for idx in range(cols):
        anchor_cols.append(cursor)
        if idx < cols - 1:
            cursor += SLOT_W + MIN_GAP + col_extra[idx + 1]
    grid_w = anchor_cols[-1] + SLOT_W + OUTER_MARGIN + col_extra[-1]

    slots = [(r, c) for r in anchor_rows for c in anchor_cols]
    chosen = random.sample(slots, n_shapes)
    random.shuffle(chosen)
    return (grid_h, grid_w), {idx: anchor for idx, anchor in enumerate(chosen)}


def _sample_permutation(colors: list[int], style: str) -> dict[int, int]:
    colors = list(colors)
    mapping = {color: color for color in colors}

    if style == "with_fixed":
        cycle = random.sample(colors, random.randint(2, len(colors) - 1))
        random.shuffle(cycle)
        for src, dst in zip(cycle, cycle[1:] + cycle[:1]):
            mapping[src] = dst
        return mapping

    if style == "single_cycle":
        cycle = colors[:]
        random.shuffle(cycle)
        return {src: dst for src, dst in zip(cycle, cycle[1:] + cycle[:1])}

    assert style == "split_cycles"
    order = colors[:]
    random.shuffle(order)
    if len(order) == 4:
        groups = [order[:2], order[2:]]
    else:
        pivot = random.choice([2, 3])
        groups = [order[:pivot], order[pivot:]]

    out = {}
    for group in groups:
        for src, dst in zip(group, group[1:] + group[:1]):
            out[src] = dst
    return out


def _paint_shape(mask: np.ndarray, outline_color: int, marker_pos: tuple[int, int], marker_color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(outline_color)
    patch[marker_pos] = int(marker_color)
    return patch


def _expected_output(shape_records: list[dict], grid_shape: tuple[int, int]) -> np.ndarray:
    out = np.full(grid_shape, BG, dtype=int)
    occupied = np.zeros(grid_shape, dtype=bool)
    anchors = {record["outline_color"]: record["top_left"] for record in shape_records}

    for record in shape_records:
        top, left = anchors[record["marker_color"]]
        patch = record["patch"]
        h, w = map(int, patch.shape)
        mask = patch != BG
        region = occupied[top : top + h, left : left + w]
        if np.any(region & mask):
            raise AssertionError("sampled output placement overlaps")
        region |= mask
        paste(out, patch, top, left, transparent=BG)

    return out


def _semantic_check(grid: np.ndarray, expected: np.ndarray, mapping: dict[int, int], top_lefts: dict[int, tuple[int, int]]) -> bool:
    if np.array_equal(grid, expected):
        return False

    try:
        recovered = _extract_shapes(grid)
        solved = main(grid)
    except Exception:
        return False

    recovered_map = {shape["outline_color"]: shape["marker_color"] for shape in recovered}
    recovered_anchors = {shape["outline_color"]: shape["top_left"] for shape in recovered}
    return recovered_map == mapping and recovered_anchors == top_lefts and np.array_equal(solved, expected)


def _gen_attempt() -> np.ndarray | None:
    n_shapes = _next_shape_count()
    colors = random.sample(OUTLINE_COLORS, n_shapes)
    geometries = [_make_geometry() for _ in range(n_shapes)]

    if len({geom["signature"] for geom in geometries}) < 2:
        return None

    grid_shape, anchors_by_index = _sample_anchors(n_shapes)
    style = _next_style(n_shapes)
    mapping = _sample_permutation(colors, style)

    shape_records = []
    grid = np.full(grid_shape, BG, dtype=int)
    for idx, (outline_color, geometry) in enumerate(zip(colors, geometries)):
        marker_color = mapping[outline_color]
        patch = _paint_shape(geometry["mask"], outline_color, geometry["marker_pos"], marker_color)
        top_left = anchors_by_index[idx]
        paste(grid, patch, top_left[0], top_left[1], transparent=BG)
        shape_records.append(
            {
                "outline_color": outline_color,
                "marker_color": marker_color,
                "top_left": top_left,
                "patch": patch,
            }
        )

    expected = _expected_output(shape_records, grid_shape)
    top_lefts = {record["outline_color"]: record["top_left"] for record in shape_records}

    moved = sum(int(mapping[color] != color) for color in colors)
    if moved == 0:
        return None
    if not _semantic_check(grid, expected, mapping, top_lefts):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
