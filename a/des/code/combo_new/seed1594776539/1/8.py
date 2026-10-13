from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
GREY = 5
FRAGMENT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

RNG = np.random.default_rng()

SHAPE_LIBRARY = [
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)),
]

GEN_ATTEMPTS = 400
TEMPLATE_H_RANGE = (10, 15)
TEMPLATE_W_RANGE = (11, 17)
MARGIN_RANGE = (3, 7)
SLOT_COUNT_RANGE = (2, 5)
PLACEMENT_GAP = 2


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize_shape(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0, c0 = min(rs), min(cs)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _transform_shape(shape: tuple[tuple[int, int], ...], rot: int, flip: bool) -> tuple[tuple[int, int], ...]:
    pts = np.array(shape, dtype=int)
    if flip:
        pts[:, 1] *= -1
    for _ in range(rot % 4):
        pts = np.stack([pts[:, 1], -pts[:, 0]], axis=1)
    return _normalize_shape([tuple(map(int, x)) for x in pts])


def _mask_from_coords(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = _shape_dims(shape)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _shape_has_cavity(shape: tuple[tuple[int, int], ...]) -> bool:
    empty = ~_mask_from_coords(shape)
    h, w = empty.shape
    for comp in _components(empty):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            return True
    return False


def _dilate(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    pts = np.argwhere(mask)
    for r, c in pts:
        rr, cc = int(r), int(c)
        for dr, dc in DIR8:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _largest_template_mask(grid: np.ndarray) -> np.ndarray:
    comps = _components(grid == GREY)
    assert comps, "grey template must exist"
    largest = max(comps, key=len)
    mask = np.zeros_like(grid, dtype=bool)
    for r, c in largest:
        mask[r, c] = True
    return mask


def _find_holes(grid: np.ndarray, template_mask: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    holes: list[dict[str, object]] = []
    for comp in _components(grid == BLACK):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        enclosed = True
        for r, c in comp:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if grid[nr, nc] != BLACK and not template_mask[nr, nc]:
                    enclosed = False
                    break
            if not enclosed:
                break
        if not enclosed:
            continue
        holes.append({"coords": comp, "shape": _normalize_shape(comp)})
    return holes


def _find_fragments(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    fragments: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BLACK, GREY) or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            fragments.append({"coords": comp, "shape": _normalize_shape(comp), "color": color})
    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed slot in a grey template using the matching external colored fragment.

    Input structure:
    - The background is black.
    - One large grey connected template occupies the middle of the grid and contains several black holes fully enclosed by grey.
    - Outside the template, there are several monochrome colored fragments, each disconnected from the template and from the other fragments.

    Rule:
    - Find every black 4-connected hole whose boundary is entirely grey; these are the template slots.
    - Find every colored fragment outside the template and normalize each fragment and slot by translation only.
    - Match fragments to slots by exact normalized shape equality, with no rotation or reflection.
    - Fill each slot with the color of its matching fragment, and erase all external fragment pixels back to black.

    Key insight:
    The colored pieces are not moved by position cues; they are paired to the template solely by exact shape identity.
    """
    g = np.asarray(grid, dtype=int)
    template_mask = _largest_template_mask(g)
    holes = _find_holes(g, template_mask)
    fragments = _find_fragments(g)

    assert holes and fragments and len(holes) == len(fragments)
    hole_map = {hole["shape"]: hole for hole in holes}
    assert len(hole_map) == len(holes), "hole shapes must be unique"

    out = g.copy()
    for fragment in fragments:
        for r, c in fragment["coords"]:
            out[r, c] = BLACK
        shape = fragment["shape"]
        assert shape in hole_map, "every fragment must match a slot"
        for r, c in hole_map[shape]["coords"]:
            out[r, c] = int(fragment["color"])
    return out


def _sample_shapes(count: int) -> list[tuple[tuple[int, int], ...]]:
    usable_shapes = [shape for shape in SHAPE_LIBRARY if not _shape_has_cavity(shape)]
    choices = RNG.choice(len(usable_shapes), size=count, replace=False)
    shapes: list[tuple[tuple[int, int], ...]] = []
    seen = set()
    for idx in choices:
        base = usable_shapes[int(idx)]
        variants = []
        for rot in range(4):
            for flip in (False, True):
                variants.append(_transform_shape(base, rot, flip))
        variants = sorted(set(variants))
        shape = variants[int(RNG.integers(len(variants)))]
        if shape in seen:
            return []
        seen.add(shape)
        shapes.append(shape)
    return shapes


def _add_protrusions(template_mask: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    h, w = template_mask.shape
    bottom = top + height - 1
    right = left + width - 1

    if top >= 2 and RNG.random() < 0.6:
        depth = int(RNG.integers(1, min(2, top - 1) + 1))
        span = int(RNG.integers(2, min(width, 6) + 1))
        start = int(RNG.integers(left, left + width - span + 1))
        template_mask[top - depth : top, start : start + span] = True
    if h - 1 - bottom >= 2 and RNG.random() < 0.6:
        depth = int(RNG.integers(1, min(2, h - 2 - bottom) + 1))
        span = int(RNG.integers(2, min(width, 6) + 1))
        start = int(RNG.integers(left, left + width - span + 1))
        template_mask[bottom + 1 : bottom + depth + 1, start : start + span] = True
    if left >= 2 and RNG.random() < 0.6:
        depth = int(RNG.integers(1, min(2, left - 1) + 1))
        span = int(RNG.integers(2, min(height, 6) + 1))
        start = int(RNG.integers(top, top + height - span + 1))
        template_mask[start : start + span, left - depth : left] = True
    if w - 1 - right >= 2 and RNG.random() < 0.6:
        depth = int(RNG.integers(1, min(2, w - 2 - right) + 1))
        span = int(RNG.integers(2, min(height, 6) + 1))
        start = int(RNG.integers(top, top + height - span + 1))
        template_mask[start : start + span, right + 1 : right + depth + 1] = True


def _place_holes(
    grid: np.ndarray,
    core_top: int,
    core_left: int,
    core_height: int,
    core_width: int,
    shapes: list[tuple[tuple[int, int], ...]],
) -> list[dict[str, object]] | None:
    blocked = np.zeros((core_height, core_width), dtype=bool)
    holes: list[dict[str, object]] = []
    for shape in shapes:
        sh, sw = _shape_dims(shape)
        if sh > core_height - 2 or sw > core_width - 2:
            return None
        placed = False
        for _ in range(150):
            top = int(RNG.integers(1, core_height - sh))
            left = int(RNG.integers(1, core_width - sw))
            cells = [(top + r, left + c) for r, c in shape]
            if any(blocked[r, c] for r, c in cells):
                continue
            placed = True
            abs_cells = [(core_top + r, core_left + c) for r, c in cells]
            holes.append({"coords": abs_cells, "shape": shape})
            for r, c in cells:
                for dr, dc in DIR8 + [(0, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < core_height and 0 <= nc < core_width:
                        blocked[nr, nc] = True
            break
        if not placed:
            return None
    for hole in holes:
        for r, c in hole["coords"]:
            grid[r, c] = BLACK
    return holes


def _can_place_shape(blocked: np.ndarray, shape: tuple[tuple[int, int], ...], top: int, left: int) -> list[tuple[int, int]] | None:
    h, w = blocked.shape
    cells = []
    for r, c in shape:
        rr, cc = top + r, left + c
        if not (0 <= rr < h and 0 <= cc < w) or blocked[rr, cc]:
            return None
        cells.append((rr, cc))
    return cells


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr, dc in DIR8 + [(0, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _random_fragment_anchor(
    grid_shape: tuple[int, int],
    shape: tuple[tuple[int, int], ...],
    template_bbox: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    h, w = grid_shape
    sh, sw = _shape_dims(shape)
    tr0, tc0, tr1, tc1 = template_bbox
    side_options = []
    if tr0 - PLACEMENT_GAP - sh >= 0:
        side_options.append("top")
    if tr1 + PLACEMENT_GAP + sh < h:
        side_options.append("bottom")
    if tc0 - PLACEMENT_GAP - sw >= 0:
        side_options.append("left")
    if tc1 + PLACEMENT_GAP + sw < w:
        side_options.append("right")
    if side_options and RNG.random() < 0.85:
        side = side_options[int(RNG.integers(len(side_options)))]
        if side == "top":
            return int(RNG.integers(0, tr0 - PLACEMENT_GAP - sh + 1)), int(RNG.integers(0, w - sw + 1))
        if side == "bottom":
            return int(RNG.integers(tr1 + PLACEMENT_GAP, h - sh + 1)), int(RNG.integers(0, w - sw + 1))
        if side == "left":
            return int(RNG.integers(0, h - sh + 1)), int(RNG.integers(0, tc0 - PLACEMENT_GAP - sw + 1))
        return int(RNG.integers(0, h - sh + 1)), int(RNG.integers(tc1 + PLACEMENT_GAP, w - sw + 1))

    top = int(RNG.integers(0, h - sh + 1))
    left = int(RNG.integers(0, w - sw + 1))
    rr1, cc1 = top + sh - 1, left + sw - 1
    if not (rr1 < tr0 - 1 or top > tr1 + 1 or cc1 < tc0 - 1 or left > tc1 + 1):
        return None
    return top, left


def _place_fragments(
    grid: np.ndarray,
    template_mask: np.ndarray,
    shapes: list[tuple[tuple[int, int], ...]],
    colors: list[int],
) -> list[dict[str, object]] | None:
    blocked = _dilate(template_mask)
    template_bbox = _bbox(template_mask)
    fragments: list[dict[str, object]] = []
    for shape, color in zip(shapes, colors):
        placed = False
        for _ in range(250):
            anchor = _random_fragment_anchor(grid.shape, shape, template_bbox)
            if anchor is None:
                continue
            top, left = anchor
            cells = _can_place_shape(blocked, shape, top, left)
            if cells is None:
                continue
            placed = True
            fragments.append({"coords": cells, "shape": shape, "color": color})
            for r, c in cells:
                grid[r, c] = color
            _mark_blocked(blocked, cells)
            break
        if not placed:
            return None
    return fragments


def _expected_output(grid: np.ndarray, holes: list[dict[str, object]], fragments: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    hole_map = {hole["shape"]: hole["coords"] for hole in holes}
    for fragment in fragments:
        for r, c in fragment["coords"]:
            out[r, c] = BLACK
        for r, c in hole_map[fragment["shape"]]:
            out[r, c] = int(fragment["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    slot_count = int(RNG.integers(*SLOT_COUNT_RANGE))
    shapes = _sample_shapes(slot_count)
    if len(shapes) != slot_count:
        return None

    core_height = int(RNG.integers(*TEMPLATE_H_RANGE))
    core_width = int(RNG.integers(*TEMPLATE_W_RANGE))
    top_margin = int(RNG.integers(*MARGIN_RANGE))
    bottom_margin = int(RNG.integers(*MARGIN_RANGE))
    left_margin = int(RNG.integers(*MARGIN_RANGE))
    right_margin = int(RNG.integers(*MARGIN_RANGE))
    h = core_height + top_margin + bottom_margin
    w = core_width + left_margin + right_margin
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    template_mask = np.zeros((h, w), dtype=bool)
    core_top, core_left = top_margin, left_margin
    template_mask[core_top : core_top + core_height, core_left : core_left + core_width] = True
    _add_protrusions(template_mask, core_top, core_left, core_height, core_width)
    grid[template_mask] = GREY

    holes = _place_holes(grid, core_top, core_left, core_height, core_width, shapes)
    if holes is None:
        return None

    colors = list(RNG.choice(FRAGMENT_COLORS, size=slot_count, replace=False))
    fragments = _place_fragments(grid, template_mask, shapes, colors)
    if fragments is None:
        return None

    expected = _expected_output(grid, holes, fragments)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after retries")
