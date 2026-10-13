from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


Coord = tuple[int, int]

BG = 0
PALETTE = tuple(range(1, 10))
DIRECTIONS = ("up", "down", "left", "right")
DIR_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


@dataclass(frozen=True)
class Blueprint:
    direction: str
    body_h: int
    body_w: int
    stem_len: int
    copies: int
    attach_offset: int


@dataclass(frozen=True)
class Structure:
    direction: str
    body_h: int
    body_w: int
    stem_len: int
    copies: int
    attach_offset: int
    body_color: int
    stem_color: int
    bbox_r: int
    bbox_c: int


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[counts.argmax()])


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _middle_offsets(length: int) -> tuple[int, ...]:
    mid = length // 2
    if length % 2 == 1:
        return (mid,)
    return (mid - 1, mid)


def _is_solid_rectangle(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox_coords(coords)
    return int(coords.shape[0]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _is_body_rectangle(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox_coords(coords)
    return _is_solid_rectangle(coords) and (r1 > r0) and (c1 > c0)


def _is_straight_line(coords: np.ndarray) -> bool:
    rows = sorted(set(map(int, coords[:, 0])))
    cols = sorted(set(map(int, coords[:, 1])))
    if len(rows) == 1:
        return int(coords.shape[0]) == cols[-1] - cols[0] + 1
    if len(cols) == 1:
        return int(coords.shape[0]) == rows[-1] - rows[0] + 1
    return False


def _axis_extent(obj: Blueprint | Structure) -> int:
    if obj.direction in {"left", "right"}:
        return int(obj.body_w)
    return int(obj.body_h)


def _step(obj: Blueprint | Structure) -> int:
    return _axis_extent(obj) + int(obj.stem_len)


def _bbox_size(obj: Blueprint | Structure) -> tuple[int, int]:
    step = _step(obj)
    if obj.direction in {"left", "right"}:
        return int(obj.body_h), int(obj.body_w + obj.copies * step)
    return int(obj.body_h + obj.copies * step), int(obj.body_w)


def _body_top_left(obj: Structure) -> Coord:
    step = _step(obj)
    if obj.direction == "right":
        return int(obj.bbox_r), int(obj.bbox_c)
    if obj.direction == "left":
        return int(obj.bbox_r), int(obj.bbox_c + obj.copies * step)
    if obj.direction == "down":
        return int(obj.bbox_r), int(obj.bbox_c)
    return int(obj.bbox_r + obj.copies * step), int(obj.bbox_c)


def _stem_cells(obj: Structure) -> list[Coord]:
    br, bc = _body_top_left(obj)
    cells: list[Coord] = []
    if obj.direction == "right":
        row = br + obj.attach_offset
        for cc in range(bc + obj.body_w, bc + obj.body_w + obj.stem_len):
            cells.append((row, cc))
    elif obj.direction == "left":
        row = br + obj.attach_offset
        for cc in range(bc - obj.stem_len, bc):
            cells.append((row, cc))
    elif obj.direction == "down":
        col = bc + obj.attach_offset
        for rr in range(br + obj.body_h, br + obj.body_h + obj.stem_len):
            cells.append((rr, col))
    else:
        col = bc + obj.attach_offset
        for rr in range(br - obj.stem_len, br):
            cells.append((rr, col))
    return cells


def _in_bounds_rect(shape: tuple[int, int], r0: int, c0: int, h: int, w: int) -> bool:
    return 0 <= r0 and 0 <= c0 and r0 + h <= shape[0] and c0 + w <= shape[1]


def _infer_stem(
    body_bbox: tuple[int, int, int, int],
    stem_coords: np.ndarray,
) -> tuple[str, int, int]:
    r0, c0, r1, c1 = body_bbox
    body_h = r1 - r0 + 1
    body_w = c1 - c0 + 1
    rows = sorted(set(map(int, stem_coords[:, 0])))
    cols = sorted(set(map(int, stem_coords[:, 1])))

    if len(rows) == 1:
        row = rows[0]
        attach = row - r0
        assert attach in _middle_offsets(body_h), "horizontal stem must leave from the body's middle rows"
        assert cols == list(range(cols[0], cols[-1] + 1)), "horizontal stem must be contiguous"
        if cols[-1] == c0 - 1:
            return "left", c0 - cols[0], attach
        if cols[0] == c1 + 1:
            return "right", cols[-1] - c1, attach
    if len(cols) == 1:
        col = cols[0]
        attach = col - c0
        assert attach in _middle_offsets(body_w), "vertical stem must leave from the body's middle columns"
        assert rows == list(range(rows[0], rows[-1] + 1)), "vertical stem must be contiguous"
        if rows[-1] == r0 - 1:
            return "up", r0 - rows[0], attach
        if rows[0] == r1 + 1:
            return "down", rows[-1] - r1, attach
    raise AssertionError("stem must be a straight one-cell-wide segment attached to one body side")


def _extract_structure(grid: np.ndarray, comp: np.ndarray, bg: int) -> tuple[int, int, tuple[int, int, int, int], str, int]:
    values = grid[comp[:, 0], comp[:, 1]]
    colors = sorted(set(map(int, values)))
    assert len(colors) == 2, "each structure must use exactly two colors"
    by_color = {color: comp[values == color] for color in colors}

    body_candidates = [color for color, coords in by_color.items() if _is_body_rectangle(coords)]
    stem_candidates = [color for color, coords in by_color.items() if _is_straight_line(coords)]
    assert len(body_candidates) == 1, "expected exactly one solid rectangle body color"
    assert len(stem_candidates) == 1, "expected exactly one straight stem color"

    body_color = body_candidates[0]
    stem_color = stem_candidates[0]
    assert body_color != stem_color, "body and stem colors must differ"
    body_coords = by_color[body_color]
    stem_coords = by_color[stem_color]
    body_bbox = _bbox_coords(body_coords)
    direction, stem_len, attach = _infer_stem(body_bbox, stem_coords)
    extent = (body_bbox[3] - body_bbox[1] + 1) if direction in {"left", "right"} else (body_bbox[2] - body_bbox[0] + 1)
    assert stem_len >= extent, "stem length must encode a gap at least as large as the body extent"
    return body_color, stem_color, body_bbox, direction, attach


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat-stamp each rectangle body along its stem direction, leaving stem-length gaps and alternating stem/body colors.

    Input structure:
    - A black background contains one or more independent 4-connected two-color structures.
    - Each structure has one solid monochrome rectangular body and one differently colored straight one-cell-wide stem.
    - The stem protrudes from the middle of one body side, and its length is at least the body's extent along that direction.

    Rule:
    - Erase every stem while keeping each original body in place.
    - Along the stem direction, place full body-sized copies separated by gaps exactly equal to the stem length.
    - Color the first copy with the stem color, the second with the body color, and keep alternating until the next full copy would leave the grid.

    Key insight:
    - The stem is not kept as geometry; it is a ruler that simultaneously encodes copy direction, repetition period, and the color phase of the stamped sequence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    comps = _components_mask(g != bg)
    out = g.copy()

    for comp in comps:
        body_color, stem_color, body_bbox, direction, _attach = _extract_structure(g, comp, bg)
        body_r0, body_c0, body_r1, body_c1 = body_bbox
        body_h = body_r1 - body_r0 + 1
        body_w = body_c1 - body_c0 + 1
        body_mask = np.zeros_like(g, dtype=bool)
        body_mask[body_r0 : body_r1 + 1, body_c0 : body_c1 + 1] = True
        stem_mask = np.zeros_like(g, dtype=bool)
        stem_mask[comp[:, 0], comp[:, 1]] = True
        stem_mask &= ~body_mask
        stem_coords = np.argwhere(stem_mask)
        direction, stem_len, _attach = _infer_stem(body_bbox, stem_coords)
        step = (body_w if direction in {"left", "right"} else body_h) + stem_len
        dr, dc = DIR_VEC[direction]

        out[stem_mask] = bg
        copy_index = 1
        while True:
            nr = body_r0 + dr * step * copy_index
            nc = body_c0 + dc * step * copy_index
            if not _in_bounds_rect(out.shape, nr, nc, body_h, body_w):
                break
            patch = out[nr : nr + body_h, nc : nc + body_w]
            assert np.all(patch == bg), "copy paths must not overlap existing objects"
            out[nr : nr + body_h, nc : nc + body_w] = stem_color if copy_index % 2 == 1 else body_color
            copy_index += 1

    return out


COUNT_CHOICES = (1, 2, 3, 4)
COUNT_WEIGHTS = (1, 3, 4, 2)
GRID_MIN_BY_COUNT = {
    1: (12, 12),
    2: (16, 16),
    3: (20, 20),
    4: (24, 24),
}
GRID_MAX = 30
BODY_H_RANGE = range(2, 6)
BODY_W_RANGE = range(2, 6)
EXTRA_STEM_RANGE = range(0, 4)
COPY_OPTIONS = (1, 2, 3)
PLACEMENT_PADDING = 1
GEN_RETRIES = 500


def _blueprint_options(direction: str, h: int, w: int) -> list[Blueprint]:
    options: list[Blueprint] = []
    horizontal = direction in {"left", "right"}
    for body_h in BODY_H_RANGE:
        for body_w in BODY_W_RANGE:
            extent = body_w if horizontal else body_h
            for extra in EXTRA_STEM_RANGE:
                stem_len = extent + extra
                step = extent + stem_len
                for copies in COPY_OPTIONS:
                    if horizontal:
                        bbox_h = body_h
                        bbox_w = body_w + copies * step
                    else:
                        bbox_h = body_h + copies * step
                        bbox_w = body_w
                    if bbox_h > h or bbox_w > w:
                        continue
                    cross = body_h if horizontal else body_w
                    for attach in _middle_offsets(cross):
                        options.append(Blueprint(direction, body_h, body_w, stem_len, copies, attach))
    return options


def _sample_blueprints(directions: list[str], h: int, w: int) -> list[Blueprint] | None:
    options_by_dir = {direction: _blueprint_options(direction, h, w) for direction in directions}
    if any(not opts for opts in options_by_dir.values()):
        return None

    blueprints = [random.choice(options_by_dir[direction]) for direction in directions]
    if all(bp.copies == 1 for bp in blueprints):
        expandable = [direction for direction in directions if any(opt.copies >= 2 for opt in options_by_dir[direction])]
        if not expandable:
            return None
        chosen_dir = random.choice(expandable)
        idx = directions.index(chosen_dir)
        rich = [opt for opt in options_by_dir[chosen_dir] if opt.copies >= 2]
        blueprints[idx] = random.choice(rich)
    return blueprints


def _candidate_positions(bp: Blueprint, h: int, w: int) -> list[Coord]:
    bbox_h, bbox_w = _bbox_size(bp)
    step = _step(bp)
    positions: list[Coord] = []
    if bp.direction == "right":
        c_lo = max(0, w - bbox_w - (step - 1))
        c_hi = w - bbox_w
        for r0 in range(0, h - bbox_h + 1):
            for c0 in range(c_lo, c_hi + 1):
                positions.append((r0, c0))
    elif bp.direction == "left":
        c_hi = min(step - 1, w - bbox_w)
        for r0 in range(0, h - bbox_h + 1):
            for c0 in range(0, c_hi + 1):
                positions.append((r0, c0))
    elif bp.direction == "down":
        r_lo = max(0, h - bbox_h - (step - 1))
        r_hi = h - bbox_h
        for r0 in range(r_lo, r_hi + 1):
            for c0 in range(0, w - bbox_w + 1):
                positions.append((r0, c0))
    else:
        r_hi = min(step - 1, h - bbox_h)
        for r0 in range(0, r_hi + 1):
            for c0 in range(0, w - bbox_w + 1):
                positions.append((r0, c0))
    random.shuffle(positions)
    return positions


def _box_free(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _place_blueprints(blueprints: list[Blueprint], h: int, w: int) -> list[tuple[int, int]] | None:
    order = sorted(range(len(blueprints)), key=lambda i: _bbox_size(blueprints[i])[0] * _bbox_size(blueprints[i])[1], reverse=True)
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(blueprints)

    def backtrack(k: int) -> bool:
        if k == len(order):
            return True
        idx = order[k]
        bp = blueprints[idx]
        bbox_h, bbox_w = _bbox_size(bp)
        for r0, c0 in _candidate_positions(bp, h, w):
            if not _box_free(occupied, r0, c0, bbox_h, bbox_w, PLACEMENT_PADDING):
                continue
            occupied[r0 : r0 + bbox_h, c0 : c0 + bbox_w] = True
            placements[idx] = (r0, c0)
            if backtrack(k + 1):
                return True
            occupied[r0 : r0 + bbox_h, c0 : c0 + bbox_w] = False
            placements[idx] = None
        return False

    if backtrack(0):
        return [(int(r0), int(c0)) for r0, c0 in placements if r0 is not None and c0 is not None]
    return None


def _paint_rect(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0 : r0 + h, c0 : c0 + w] = int(color)


def _render_input(structures: list[Structure], h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for obj in structures:
        br, bc = _body_top_left(obj)
        body_patch = grid[br : br + obj.body_h, bc : bc + obj.body_w]
        assert np.all(body_patch == BG), "body placement must be isolated"
        _paint_rect(grid, br, bc, obj.body_h, obj.body_w, obj.body_color)
        for rr, cc in _stem_cells(obj):
            assert 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == BG, "stem must fit and stay isolated"
            grid[rr, cc] = obj.stem_color
    return grid


def _render_output(structures: list[Structure], h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for obj in structures:
        br, bc = _body_top_left(obj)
        base_patch = grid[br : br + obj.body_h, bc : bc + obj.body_w]
        assert np.all(base_patch == BG), "original bodies must not overlap"
        _paint_rect(grid, br, bc, obj.body_h, obj.body_w, obj.body_color)
        dr, dc = DIR_VEC[obj.direction]
        step = _step(obj)
        for idx in range(1, obj.copies + 1):
            nr = br + dr * step * idx
            nc = bc + dc * step * idx
            assert _in_bounds_rect(grid.shape, nr, nc, obj.body_h, obj.body_w), "declared copy must fit"
            patch = grid[nr : nr + obj.body_h, nc : nc + obj.body_w]
            assert np.all(patch == BG), "declared copy paths must stay disjoint"
            color = obj.stem_color if idx % 2 == 1 else obj.body_color
            _paint_rect(grid, nr, nc, obj.body_h, obj.body_w, color)
    return grid


def _validate_structures(inp: np.ndarray, out: np.ndarray, structures: list[Structure]) -> bool:
    if len(_components_mask(inp != BG)) != len(structures):
        return False
    if len({obj.direction for obj in structures}) != len(structures):
        return False
    if not any(obj.copies >= 2 for obj in structures):
        return False
    for obj in structures:
        br, bc = _body_top_left(obj)
        if not _in_bounds_rect(inp.shape, br, bc, obj.body_h, obj.body_w):
            return False
        dr, dc = DIR_VEC[obj.direction]
        step = _step(obj)
        last_r = br + dr * step * obj.copies
        last_c = bc + dc * step * obj.copies
        next_r = br + dr * step * (obj.copies + 1)
        next_c = bc + dc * step * (obj.copies + 1)
        if not _in_bounds_rect(inp.shape, last_r, last_c, obj.body_h, obj.body_w):
            return False
        if _in_bounds_rect(inp.shape, next_r, next_c, obj.body_h, obj.body_w):
            return False
    try:
        solved = main(inp)
    except AssertionError:
        return False
    return np.array_equal(solved, out)


def _gen_attempt() -> np.ndarray | None:
    count = random.choices(COUNT_CHOICES, weights=COUNT_WEIGHTS, k=1)[0]
    min_h, min_w = GRID_MIN_BY_COUNT[count]
    h = random.randint(min_h, GRID_MAX)
    w = random.randint(min_w, GRID_MAX)

    directions = random.sample(DIRECTIONS, count)
    blueprints = _sample_blueprints(directions, h, w)
    if blueprints is None:
        return None

    placements = _place_blueprints(blueprints, h, w)
    if placements is None:
        return None

    colors = random.sample(PALETTE, 2 * count)
    structures = [
        Structure(
            direction=bp.direction,
            body_h=bp.body_h,
            body_w=bp.body_w,
            stem_len=bp.stem_len,
            copies=bp.copies,
            attach_offset=bp.attach_offset,
            body_color=colors[2 * i],
            stem_color=colors[2 * i + 1],
            bbox_r=placements[i][0],
            bbox_c=placements[i][1],
        )
        for i, bp in enumerate(blueprints)
    ]

    inp = _render_input(structures, h, w)
    out = _render_output(structures, h, w)
    if _validate_structures(inp, out, structures):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-overlay input")
