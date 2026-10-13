from __future__ import annotations

import random
from collections import Counter

import numpy as np


PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)
EMITTER_COLOR = 4
BACKGROUND = 0
GEN_ATTEMPTS = 300
TRACE_LIMIT = 800


Box = tuple[int, int, int, int]
Cell = tuple[int, int]


def _perimeter_coords(box: Box) -> list[Cell]:
    r0, c0, r1, c1 = box
    pts = [(r0, c) for c in range(c0, c1 + 1)]
    pts += [(r1, c) for c in range(c0, c1 + 1)]
    pts += [(r, c0) for r in range(r0 + 1, r1)]
    pts += [(r, c1) for r in range(r0 + 1, r1)]
    return pts


def _inner_side_cells(box: Box, side: str) -> list[Cell]:
    r0, c0, r1, c1 = box
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    if side == "right":
        return [(r, c1) for r in range(r0 + 1, r1)]
    raise ValueError(f"unknown side: {side}")


def _legal_gap_positions(box: Box) -> list[Cell]:
    return (
        _inner_side_cells(box, "top")
        + _inner_side_cells(box, "bottom")
        + _inner_side_cells(box, "left")
        + _inner_side_cells(box, "right")
    )


def _box_contains(box: Box, cell: Cell) -> bool:
    r, c = cell
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _boundary_kind(box: Box, cell: Cell) -> str:
    r, c = cell
    r0, c0, r1, c1 = box
    if (r, c) in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
        return "corner"
    if r in (r0, r1):
        return "horizontal"
    return "vertical"


def _emitter_direction(box: Box, emitter: Cell) -> tuple[int, int]:
    r, c = emitter
    r0, c0, r1, c1 = box
    mid_r = (r0 + r1) / 2.0
    mid_c = (c0 + c1) / 2.0
    if r == r0:
        return -1, -1 if c < mid_c else 1
    if r == r1:
        return 1, -1 if c < mid_c else 1
    if c == c0:
        return (-1 if r < mid_r else 1), -1
    if c == c1:
        return (-1 if r < mid_r else 1), 1
    raise AssertionError("emitter must lie on the innermost ring perimeter")


def _bbox_from_mask(mask: np.ndarray) -> Box | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_boxes(grid: np.ndarray) -> list[Box]:
    mask = grid != BACKGROUND
    boxes: list[Box] = []
    remaining = mask.copy()
    while remaining.any():
        box = _bbox_from_mask(remaining)
        assert box is not None
        r0, c0, r1, c1 = box
        assert r1 > r0 and c1 > c0 and (r1 - r0) == (c1 - c0), "rings must be square"
        boxes.append(box)
        remaining[r0, c0 : c1 + 1] = False
        remaining[r1, c0 : c1 + 1] = False
        remaining[r0 : r1 + 1, c0] = False
        remaining[r0 : r1 + 1, c1] = False
    return boxes[::-1]


def _region_id(cell: Cell, boxes: list[Box], boundary_owner: dict[Cell, int]) -> int | None:
    if cell in boundary_owner:
        return None
    if _box_contains(boxes[0], cell):
        return None
    for idx in range(len(boxes) - 1):
        if _box_contains(boxes[idx + 1], cell) and not _box_contains(boxes[idx], cell):
            return idx
    if not _box_contains(boxes[-1], cell):
        return len(boxes) - 1
    return None


def _region_masks(shape: tuple[int, int], boxes: list[Box], boundary_owner: dict[Cell, int]) -> list[np.ndarray]:
    h, w = shape
    masks: list[np.ndarray] = []
    for idx in range(len(boxes) - 1):
        b_in = boxes[idx]
        b_out = boxes[idx + 1]
        mask = np.zeros((h, w), dtype=bool)
        r0, c0, r1, c1 = b_out
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
        ir0, ic0, ir1, ic1 = b_in
        mask[ir0 : ir1 + 1, ic0 : ic1 + 1] = False
        for cell in boundary_owner:
            mask[cell] = False
        masks.append(mask)

    outer = np.ones((h, w), dtype=bool)
    r0, c0, r1, c1 = boxes[-1]
    outer[r0 : r1 + 1, c0 : c1 + 1] = False
    for cell in boundary_owner:
        outer[cell] = False
    masks.append(outer)
    return masks


def _extract_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 2 <= max(g.shape) <= 30

    emitters = [tuple(map(int, cell)) for cell in np.argwhere(g == EMITTER_COLOR)]
    assert len(emitters) == 1, "expected exactly one emitter"
    emitter = emitters[0]

    boxes = _extract_boxes(g)
    assert 2 <= len(boxes) <= 3, "expected 2-3 nested rings"

    centers = {(box[0] + box[2], box[1] + box[3]) for box in boxes}
    assert len(centers) == 1, "rings must be concentric"
    for inner, outer in zip(boxes[:-1], boxes[1:]):
        annulus_width = inner[0] - outer[0] - 1
        assert annulus_width >= 2, "annulus width must be at least 2"

    gaps: list[Cell] = []
    ring_colors: list[int] = []
    boundary_owner: dict[Cell, int] = {}
    wall_color: dict[Cell, int] = {}

    for idx, box in enumerate(boxes):
        perimeter = _perimeter_coords(box)
        gap_cells = [cell for cell in perimeter if int(g[cell]) == BACKGROUND]
        assert len(gap_cells) == 1, "each ring must have exactly one gap"
        gap = gap_cells[0]
        kind = _boundary_kind(box, gap)
        assert kind != "corner", "gaps cannot be at corners"
        gaps.append(gap)

        ring_cells = [cell for cell in perimeter if cell not in {gap, emitter}]
        colors = {int(g[cell]) for cell in ring_cells}
        assert len(colors) == 1 and BACKGROUND not in colors and EMITTER_COLOR not in colors
        ring_color = next(iter(colors))
        ring_colors.append(ring_color)

        for cell in perimeter:
            boundary_owner[cell] = idx
            if cell != gap:
                wall_color[cell] = int(g[cell])

    assert emitter in boundary_owner and boundary_owner[emitter] == 0, "emitter must be on the innermost ring"
    assert abs(emitter[0] - gaps[0][0]) + abs(emitter[1] - gaps[0][1]) > 1, "emitter cannot touch inner gap"

    return {
        "grid": g,
        "boxes": boxes,
        "gaps": gaps,
        "ring_colors": ring_colors,
        "emitter": emitter,
        "boundary_owner": boundary_owner,
        "wall_color": wall_color,
    }


def _trace(structure: dict, *, fill_mode: str = "entry") -> dict | None:
    g: np.ndarray = structure["grid"]
    boxes: list[Box] = structure["boxes"]
    gaps: list[Cell] = structure["gaps"]
    emitter: Cell = structure["emitter"]
    boundary_owner: dict[Cell, int] = structure["boundary_owner"]
    wall_color: dict[Cell, int] = structure["wall_color"]

    h, w = g.shape
    gap_set = set(gaps)
    region_masks = _region_masks(g.shape, boxes, boundary_owner)
    out = g.copy()

    pos = emitter
    dr, dc = _emitter_direction(boxes[0], emitter)
    current_color = EMITTER_COLOR
    current_region: int | None = None
    entry_color: int | None = None
    region_records: list[dict] = []
    filled_regions: set[int] = set()
    hits: list[dict] = []
    gap_cross: list[Cell] = []
    seen: set[tuple[Cell, int, int, int, int | None]] = set()

    def finalize_region(exit_color: int) -> None:
        nonlocal current_region, entry_color
        if current_region is None or current_region in filled_regions:
            return
        assert entry_color is not None
        fill_color = entry_color if fill_mode == "entry" else exit_color
        out[region_masks[current_region]] = fill_color
        region_records.append(
            {
                "region": current_region,
                "entry_color": int(entry_color),
                "exit_color": int(exit_color),
                "fill_color": int(fill_color),
            }
        )
        filled_regions.add(current_region)

    for _ in range(TRACE_LIMIT):
        state = (pos, int(dr), int(dc), int(current_color), current_region)
        if state in seen:
            return None
        seen.add(state)

        nxt = (pos[0] + dr, pos[1] + dc)
        if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
            finalize_region(current_color)
            return {
                "out": out,
                "region_records": region_records,
                "hits": hits,
                "gap_cross": gap_cross,
            }

        if nxt in boundary_owner:
            if nxt in gap_set:
                finalize_region(current_color)
                pos = nxt
                current_region = None
                entry_color = None
                gap_cross.append(nxt)
                continue

            hit_color = int(wall_color[nxt])
            kind = _boundary_kind(boxes[boundary_owner[nxt]], nxt)
            if kind in ("horizontal", "corner"):
                dr = -dr
            if kind in ("vertical", "corner"):
                dc = -dc
            current_color = hit_color
            hits.append(
                {
                    "cell": nxt,
                    "ring": int(boundary_owner[nxt]),
                    "kind": kind,
                    "color": hit_color,
                }
            )
            continue

        pos = nxt
        new_region = _region_id(pos, boxes, boundary_owner)
        if new_region != current_region:
            finalize_region(current_color)
            current_region = new_region
            entry_color = None if new_region is None else int(current_color)

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Send the emitter's diagonal ray through nested square-ring gaps and fill each traversed annulus with the ray color carried into it.

    Input structure:
    - The grid has a black background with 2-3 concentric one-pixel-thick square rings, each ring using its own color.
    - Every ring has exactly one one-cell gap on a non-corner wall cell.
    - The innermost ring also contains one yellow emitter pixel on its perimeter, away from the gap.

    Rule:
    - Start a 45-degree ray at the emitter, pointing into the outward quadrant relative to the common ring center.
    - If the next diagonal step reaches a ring wall pixel, reflect on that wall: top/bottom walls flip `dr`, left/right walls flip `dc`, corners flip both; after every bounce the ray adopts the wall color.
    - If the next step reaches a gap cell, the ray passes through unchanged.
    - Whenever the ray leaves an annulus (or the outer area), fill every background cell of that region with the color the ray had when it entered that region.
    - Ring walls, gap cells, and the emitter stay unchanged; untouched regions remain black.

    Key insight:
    - The visible output is determined jointly by ring nesting, gap placement, and bounce color inheritance: gaps choose which regions are reached, bounces choose the carried color, and filling uses the entry color rather than the exit color.
    """
    structure = _extract_structure(np.asarray(grid, dtype=int))
    traced = _trace(structure, fill_mode="entry")
    assert traced is not None, "ray must reach the boundary without looping"
    return traced["out"]


def _sample_boxes(rng: random.Random) -> tuple[int, int, list[Box]]:
    n_rings = rng.choice((2, 3))
    if n_rings == 2:
        inner_size = rng.randint(5, 8)
        insets = [rng.randint(3, 5)]
    else:
        inner_size = rng.randint(5, 7)
        insets = [rng.randint(3, 4) for _ in range(2)]
    sizes = [inner_size]
    for inset in insets:
        sizes.append(sizes[-1] + 2 * inset)

    outer_size = sizes[-1]
    h = outer_size + rng.randint(3, 7)
    w = outer_size + rng.randint(3, 7)
    top = rng.randint(1, h - outer_size - 1)
    left = rng.randint(1, w - outer_size - 1)

    boxes = [(top, left, top + outer_size - 1, left + outer_size - 1)]
    for inset in insets[::-1]:
        r0, c0, r1, c1 = boxes[-1]
        boxes.append((r0 + inset, c0 + inset, r1 - inset, c1 - inset))
    return h, w, boxes[::-1]


def _emitter_candidates(box: Box, gap: Cell) -> list[Cell]:
    mid_r = (box[0] + box[2]) / 2.0
    mid_c = (box[1] + box[3]) / 2.0
    candidates: list[Cell] = []
    for side in ("top", "bottom", "left", "right"):
        for cell in _inner_side_cells(box, side):
            if cell == gap:
                continue
            if abs(cell[0] - gap[0]) + abs(cell[1] - gap[1]) == 1:
                continue
            if side in ("top", "bottom") and cell[1] == mid_c:
                continue
            if side in ("left", "right") and cell[0] == mid_r:
                continue
            candidates.append(cell)
    return candidates


def _build_input(shape: tuple[int, int], boxes: list[Box], ring_colors: list[int], gaps: list[Cell], emitter: Cell) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    for box, color, gap in zip(boxes, ring_colors, gaps):
        for cell in _perimeter_coords(box):
            if cell != gap:
                grid[cell] = int(color)
    grid[emitter] = EMITTER_COLOR
    return grid


def _structure_from_layout(
    shape: tuple[int, int], boxes: list[Box], ring_colors: list[int], gaps: list[Cell], emitter: Cell
) -> dict:
    grid = _build_input(shape, boxes, ring_colors, gaps, emitter)
    boundary_owner: dict[Cell, int] = {}
    wall_color: dict[Cell, int] = {}
    for idx, (box, gap) in enumerate(zip(boxes, gaps)):
        for cell in _perimeter_coords(box):
            boundary_owner[cell] = idx
            if cell != gap:
                wall_color[cell] = int(grid[cell])
    return {
        "grid": grid,
        "boxes": boxes,
        "gaps": gaps,
        "ring_colors": ring_colors,
        "emitter": emitter,
        "boundary_owner": boundary_owner,
        "wall_color": wall_color,
    }


def _good_trace(trace: dict, boxes: list[Box], gaps: list[Cell], *, require_corner: bool) -> bool:
    region_records = trace["region_records"]
    fill_colors = [record["fill_color"] for record in region_records]
    entry_colors = [record["entry_color"] for record in region_records]
    exit_colors = [record["exit_color"] for record in region_records]
    hit_kinds = [hit["kind"] for hit in trace["hits"]]

    if len(region_records) != len(boxes):
        return False
    if trace["gap_cross"] != gaps[1:]:
        return False
    if len(fill_colors) < 2 or len(set(fill_colors)) < 2:
        return False
    if fill_colors[0] != EMITTER_COLOR or fill_colors[-1] == EMITTER_COLOR:
        return False
    if not trace["hits"]:
        return False
    if require_corner and "corner" not in hit_kinds:
        return False
    if not any(entry != exit_ for entry, exit_ in zip(entry_colors, exit_colors)):
        return False
    return True


def _search_gaps(
    h: int,
    w: int,
    boxes: list[Box],
    ring_colors: list[int],
    gap0: Cell,
    emitter: Cell,
    rng: random.Random,
    *,
    require_corner: bool,
) -> tuple[list[Cell], dict] | None:
    if len(boxes) == 2:
        gap1s = _legal_gap_positions(boxes[1])
        rng.shuffle(gap1s)
        for gap1 in gap1s:
            gaps = [gap0, gap1]
            structure = _structure_from_layout((h, w), boxes, ring_colors, gaps, emitter)
            trace = _trace(structure, fill_mode="entry")
            if trace is not None and _good_trace(trace, boxes, gaps, require_corner=require_corner):
                return gaps, trace
        return None

    gap1s = _legal_gap_positions(boxes[1])
    gap2s = _legal_gap_positions(boxes[2])
    rng.shuffle(gap1s)
    rng.shuffle(gap2s)
    for gap1 in gap1s:
        for gap2 in gap2s:
            gaps = [gap0, gap1, gap2]
            structure = _structure_from_layout((h, w), boxes, ring_colors, gaps, emitter)
            trace = _trace(structure, fill_mode="entry")
            if trace is not None and _good_trace(trace, boxes, gaps, require_corner=require_corner):
                return gaps, trace
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h, w, boxes = _sample_boxes(rng)
    ring_colors = rng.sample(PALETTE, len(boxes))
    gap0s = _legal_gap_positions(boxes[0])
    rng.shuffle(gap0s)

    for gap0 in gap0s[:8]:
        emitters = _emitter_candidates(boxes[0], gap0)
        rng.shuffle(emitters)
        for emitter in emitters[:8]:
            found = _search_gaps(
                h,
                w,
                boxes,
                ring_colors,
                gap0,
                emitter,
                rng,
                require_corner=False,
            )
            if found is None:
                continue
            gaps, _trace_result = found
            return _build_input((h, w), boxes, ring_colors, gaps, emitter)
    return None


def _validate_generated(grid: np.ndarray) -> bool:
    structure = _extract_structure(grid)
    trace = _trace(structure, fill_mode="entry")
    if trace is None:
        return False

    out = main(grid)
    if not np.array_equal(out, trace["out"]):
        return False

    boxes = structure["boxes"]
    gaps = structure["gaps"]
    if not _good_trace(trace, boxes, gaps, require_corner=False):
        return False

    if np.count_nonzero(out != grid) < 6:
        return False

    # Wrong fill rule: using exit color instead of entry color must change the output.
    wrong_trace = _trace(structure, fill_mode="exit")
    if wrong_trace is None or np.array_equal(wrong_trace["out"], out):
        return False

    # Gap crossings must advance strictly outward through ring indices.
    crossed = [structure["boundary_owner"][cell] for cell in trace["gap_cross"]]
    if crossed != list(range(1, len(boxes))):
        return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None and _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid nested-ring ray puzzle")
