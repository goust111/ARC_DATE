from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


GEN_RETRIES = 400
GRID_MIN = 18
GRID_MAX = 30
FRAME_COUNT_RANGE = (2, 4)
SECOND_ROOT_PROB = 0.35
ROOT_PLACEMENT_RETRIES = 120
CORNER_NAMES = ("tl", "tr", "bl", "br")
LEGEND_STEP = 4
LEGEND_LAYOUTS = {
    2: [[(0, 0), (0, 1)], [(0, 0), (1, 0)]],
    3: [[(0, 0), (0, 1), (0, 2)], [(0, 0), (1, 0), (2, 0)]],
    4: [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
    ],
}


@dataclass(frozen=True)
class FrameSpec:
    bbox: tuple[int, int, int, int]
    border: int
    fill: int


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _bboxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _perimeter_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _find_legends(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    legend_mask = np.zeros((h, w), dtype=bool)
    mapping: dict[int, int] = {}
    ring_positions = [
        (0, 0), (0, 1), (0, 2),
        (1, 0),         (1, 2),
        (2, 0), (2, 1), (2, 2),
    ]
    for r in range(h - 2):
        for c in range(w - 2):
            if legend_mask[r : r + 3, c : c + 3].any():
                continue
            patch = grid[r : r + 3, c : c + 3]
            ring_vals = [int(patch[rr, cc]) for rr, cc in ring_positions]
            ring_color = ring_vals[0]
            if ring_color == bg or any(v != ring_color for v in ring_vals):
                continue
            center = int(patch[1, 1])
            if center == bg or center == ring_color:
                continue
            legend_mask[r : r + 3, c : c + 3] = True
            mapping[ring_color] = center
    return mapping, legend_mask


def _detect_frames(grid: np.ndarray, bg: int, legend_mask: np.ndarray) -> list[dict[str, int | tuple[int, int, int, int]]]:
    frames: list[dict[str, int | tuple[int, int, int, int]]] = []
    colors = {int(v) for v in np.unique(grid) if int(v) != bg}
    for color in colors:
        color_mask = (grid == color) & ~legend_mask
        for coords in _components(color_mask):
            bbox = _bbox_from_coords(coords)
            r0, c0, r1, c1 = bbox
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            coord_set = set(coords)
            if coord_set == _perimeter_cells(bbox):
                frames.append({"color": color, "bbox": bbox})
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill nested rectangular frame cavities using corner 3x3 legends.

    Each legend block uses the frame color on its 8-pixel border and the
    corresponding cavity fill color at its center. Frames are filled from the
    deepest nested level back out to the roots, and the legend area is erased.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)

    mapping, legend_mask = _find_legends(g, bg)
    assert mapping, "expected at least one 3x3 legend block"

    frames = _detect_frames(g, bg, legend_mask)
    assert frames, "expected at least one rectangular frame"

    out = g.copy()
    out[legend_mask] = bg

    ordered = []
    for frame in frames:
        bbox = frame["bbox"]
        depth = sum(
            1
            for other in frames
            if other is not frame and _bbox_contains(other["bbox"], bbox)
        )
        ordered.append((depth, _bbox_area(bbox), int(frame["color"]), bbox))

    for _depth, _area, color, bbox in sorted(ordered, key=lambda item: (-item[0], item[1])):
        assert color in mapping, "every frame color must have a legend entry"
        r0, c0, r1, c1 = bbox
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == bg] = mapping[color]

    return out


def _partition_total(total: int, root_count: int, rng: random.Random) -> list[int]:
    if root_count == 1:
        return [total]
    cut = rng.randint(1, total - 1)
    parts = [cut, total - cut]
    rng.shuffle(parts)
    return parts


def _legend_layout(count: int, rng: random.Random) -> tuple[list[tuple[int, int]], int, int]:
    slots = list(rng.choice(LEGEND_LAYOUTS[count]))
    rows = 1 + max(r for r, _ in slots)
    cols = 1 + max(c for _, c in slots)
    height = rows * 3 + (rows - 1)
    width = cols * 3 + (cols - 1)
    return slots, height, width


def _place_legend_bbox(h: int, w: int, legend_h: int, legend_w: int, corner: str, rng: random.Random) -> tuple[int, int, int, int]:
    pad_r = rng.randint(0, 1)
    pad_c = rng.randint(0, 1)
    if corner[0] == "t":
        r0 = pad_r
    else:
        r0 = h - legend_h - pad_r
    if corner[1] == "l":
        c0 = pad_c
    else:
        c0 = w - legend_w - pad_c
    return r0, c0, r0 + legend_h - 1, c0 + legend_w - 1


def _place_root_bboxes(
    h: int,
    w: int,
    depths: list[int],
    blocked_bbox: tuple[int, int, int, int],
    rng: random.Random,
) -> list[tuple[int, int, int, int]] | None:
    ordered = sorted(enumerate(depths), key=lambda item: item[1], reverse=True)
    placed: list[tuple[int, tuple[int, int, int, int]]] = []
    occupied = [blocked_bbox]
    for index, depth in ordered:
        min_side = 4 * depth - 1
        max_h = min(h - 2, min_side + 8)
        max_w = min(w - 2, min_side + 8)
        if max_h < min_side or max_w < min_side:
            return None
        bbox = None
        for _ in range(ROOT_PLACEMENT_RETRIES):
            box_h = rng.randint(min_side, max_h)
            box_w = rng.randint(min_side, max_w)
            if h - box_h - 1 < 1 or w - box_w - 1 < 1:
                continue
            r0 = rng.randint(1, h - box_h - 1)
            c0 = rng.randint(1, w - box_w - 1)
            candidate = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
            if any(_bboxes_too_close(candidate, other, gap=1) for other in occupied):
                continue
            bbox = candidate
            break
        if bbox is None:
            return None
        placed.append((index, bbox))
        occupied.append(bbox)

    result = [None] * len(depths)
    for index, bbox in placed:
        result[index] = bbox
    return result


def _child_bbox(parent: tuple[int, int, int, int], remaining_depth: int, rng: random.Random) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = parent
    ph = r1 - r0 + 1
    pw = c1 - c0 + 1
    min_side = 4 * remaining_depth - 1
    child_h = rng.randint(min_side, ph - 4)
    child_w = rng.randint(min_side, pw - 4)

    shrink_h = ph - child_h
    shrink_w = pw - child_w
    top = rng.randint(2, shrink_h - 2)
    left = rng.randint(2, shrink_w - 2)
    bottom = shrink_h - top
    right = shrink_w - left
    return (r0 + top, c0 + left, r1 - bottom, c1 - right)


def _make_chain(
    root_bbox: tuple[int, int, int, int],
    depth: int,
    border_colors: list[int],
    fill_colors: list[int],
    rng: random.Random,
) -> list[FrameSpec]:
    frames: list[FrameSpec] = []
    bbox = root_bbox
    for level in range(depth):
        frames.append(FrameSpec(bbox=bbox, border=border_colors[level], fill=fill_colors[level]))
        remaining = depth - level - 1
        if remaining:
            bbox = _child_bbox(bbox, remaining, rng)
    return frames


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _legend_positions(
    legend_bbox: tuple[int, int, int, int],
    slots: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    r0, c0, _r1, _c1 = legend_bbox
    return [(r0 + rr * LEGEND_STEP, c0 + cc * LEGEND_STEP) for rr, cc in slots]


def _render_scene(
    shape: tuple[int, int],
    frames: list[FrameSpec],
    legend_bbox: tuple[int, int, int, int],
    slots: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros(shape, dtype=int)
    for frame in frames:
        _paint_frame(grid, frame.bbox, frame.border)

    legend_positions = _legend_positions(legend_bbox, slots)
    legend_order = frames[:]
    rng.shuffle(legend_order)
    for (r0, c0), frame in zip(legend_positions, legend_order):
        grid[r0 : r0 + 3, c0 : c0 + 3] = frame.border
        grid[r0 + 1, c0 + 1] = frame.fill

    expected = np.zeros_like(grid)
    for frame in frames:
        _paint_frame(expected, frame.bbox, frame.border)

    ordered = sorted(
        frames,
        key=lambda frame: (
            -sum(1 for other in frames if other is not frame and _bbox_contains(other.bbox, frame.bbox)),
            _bbox_area(frame.bbox),
        ),
    )
    for frame in ordered:
        r0, c0, r1, c1 = frame.bbox
        interior = expected[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == 0] = frame.fill

    return grid, expected


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    total_frames = rng.randint(*FRAME_COUNT_RANGE)
    root_count = 1
    if total_frames >= 3 and rng.random() < SECOND_ROOT_PROB:
        root_count = 2
    depths = _partition_total(total_frames, root_count, rng)
    largest_depth = max(depths)

    slots, legend_h, legend_w = _legend_layout(total_frames, rng)
    min_dim = max(GRID_MIN, 4 * largest_depth + 3, 24 if root_count == 2 else 0, legend_h + 4, legend_w + 4)
    if min_dim > GRID_MAX:
        return None
    h = rng.randint(min_dim, GRID_MAX)
    w = rng.randint(min_dim, GRID_MAX)

    corner = rng.choice(CORNER_NAMES)
    legend_bbox = _place_legend_bbox(h, w, legend_h, legend_w, corner, rng)
    root_bboxes = _place_root_bboxes(h, w, depths, legend_bbox, rng)
    if root_bboxes is None:
        return None

    border_colors = rng.sample(range(1, 10), total_frames)
    fill_pool = [color for color in range(1, 10) if color not in border_colors]
    rng.shuffle(fill_pool)
    fill_colors = fill_pool[:total_frames]

    frames: list[FrameSpec] = []
    offset = 0
    for bbox, depth in zip(root_bboxes, depths):
        chain_borders = border_colors[offset : offset + depth]
        chain_fills = fill_colors[offset : offset + depth]
        frames.extend(_make_chain(bbox, depth, chain_borders, chain_fills, rng))
        offset += depth

    grid, expected = _render_scene((h, w), frames, legend_bbox, slots, rng)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
