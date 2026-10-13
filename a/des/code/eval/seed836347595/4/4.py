from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
PALETTE = (1, 3, 4, 6, 7, 8, 9)

MAX_GENERATION_ATTEMPTS = 400

_DIRS4 = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
_SHAPE_TEMPLATES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)


def _normalize_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    return tuple(sorted((r - min_r, c - min_c) for r, c in shape))


def _shape_variants(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    variants: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()

    for flip in (False, True):
        for rot_k in range(4):
            pts = []
            for r, c in shape:
                rr, cc = r, c
                if flip:
                    cc = -cc
                for _ in range(rot_k):
                    rr, cc = cc, -rr
                pts.append((rr, cc))
            norm = _normalize_shape(tuple(pts))
            if norm not in seen:
                seen.add(norm)
                variants.append(norm)

    return variants


_ALL_SHAPES: list[tuple[tuple[int, int], ...]] = []
_SEEN_SHAPES: set[tuple[tuple[int, int], ...]] = set()
for _template in _SHAPE_TEMPLATES:
    for _variant in _shape_variants(_template):
        if _variant not in _SEEN_SHAPES:
            _SEEN_SHAPES.add(_variant)
            _ALL_SHAPES.append(_variant)


def _is_dashed_line(line: np.ndarray) -> bool:
    vals = set(map(int, line))
    if GREY not in vals or not vals.issubset({BLACK, GREY}):
        return False

    n = int(line.size)
    for phase in (0, 1):
        expected = np.full(n, BLACK, dtype=int)
        expected[phase::2] = GREY
        if np.array_equal(line, expected):
            return True
    return False


def _find_fold_line(grid: np.ndarray) -> tuple[str, int]:
    candidates: list[tuple[str, int]] = []

    for r in range(grid.shape[0]):
        if _is_dashed_line(grid[r, :]):
            candidates.append(("horizontal", int(r)))
    for c in range(grid.shape[1]):
        if _is_dashed_line(grid[:, c]):
            candidates.append(("vertical", int(c)))

    assert len(candidates) == 1, "expected exactly one dashed fold line"
    return candidates[0]


def _merge_layers(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    out = base.copy()
    base_fg = base != BLACK
    overlay_fg = overlay != BLACK

    only_overlay = overlay_fg & ~base_fg
    conflicts = overlay_fg & base_fg & (overlay != base)

    out[only_overlay] = overlay[only_overlay]
    out[conflicts] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the smaller side of a dashed grey line onto the larger side.

    Input structure
    - Background is black.
    - Exactly one full dashed grey fold line is present, horizontal or vertical.
    - The two sides contain colored patterns using non-black, non-grey colors.

    Rule
    - Detect the dashed fold line.
    - Reflect the smaller side across that line onto the larger side.
    - On overlap: keep the visible color if only one side is colored; keep it if
      both colors are equal; otherwise paint red.
    - Return only the larger side after composition.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3, "expected a non-empty 2D grid"

    orient, axis = _find_fold_line(g)

    if orient == "horizontal":
        top = g[:axis, :]
        bottom = g[axis + 1 :, :]
        assert top.shape[0] > 0 and bottom.shape[0] > 0, "fold line must split the grid"
        assert top.shape[0] != bottom.shape[0], "folded regions must have unequal sizes"

        if top.shape[0] > bottom.shape[0]:
            big = top.copy()
            folded = np.flipud(bottom)
            overlay = np.zeros_like(big)
            overlay[-bottom.shape[0] :, :] = folded
        else:
            big = bottom.copy()
            folded = np.flipud(top)
            overlay = np.zeros_like(big)
            overlay[: top.shape[0], :] = folded
    else:
        left = g[:, :axis]
        right = g[:, axis + 1 :]
        assert left.shape[1] > 0 and right.shape[1] > 0, "fold line must split the grid"
        assert left.shape[1] != right.shape[1], "folded regions must have unequal sizes"

        if left.shape[1] > right.shape[1]:
            big = left.copy()
            folded = np.fliplr(right)
            overlay = np.zeros_like(big)
            overlay[:, -right.shape[1] :] = folded
        else:
            big = right.copy()
            folded = np.fliplr(left)
            overlay = np.zeros_like(big)
            overlay[:, : left.shape[1]] = folded

    assert GREY not in set(map(int, big.ravel())), "grey is reserved for the fold line"
    return _merge_layers(big, overlay)


def _zone_slice(
    big_shape: tuple[int, int],
    small_span: int,
    orientation: str,
    big_side: str,
) -> tuple[slice, slice]:
    if orientation == "horizontal":
        if big_side == "top":
            return slice(big_shape[0] - small_span, big_shape[0]), slice(0, big_shape[1])
        return slice(0, small_span), slice(0, big_shape[1])

    if big_side == "left":
        return slice(0, big_shape[0]), slice(big_shape[1] - small_span, big_shape[1])
    return slice(0, big_shape[0]), slice(0, small_span)


def _map_zone_coords(
    coords: list[tuple[int, int]],
    big_shape: tuple[int, int],
    small_span: int,
    orientation: str,
    big_side: str,
) -> list[tuple[int, int]]:
    rows, cols = _zone_slice(big_shape, small_span, orientation, big_side)
    row_offset = int(rows.start)
    col_offset = int(cols.start)
    return [(row_offset + r, col_offset + c) for r, c in coords]


def _paint(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        assert int(grid[r, c]) == BLACK, "generator paint overlap detected"
        grid[r, c] = int(color)


def _mark_blocked(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for dr, dc in _DIRS4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                blocked[rr, cc] = True


def _candidate_placements(
    allowed: np.ndarray,
    blocked: np.ndarray,
    shape: tuple[tuple[int, int], ...],
) -> list[list[tuple[int, int]]]:
    h, w = allowed.shape
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    placements: list[list[tuple[int, int]]] = []

    for r0 in range(h - max_r):
        for c0 in range(w - max_c):
            coords = [(r0 + r, c0 + c) for r, c in shape]
            if all(bool(allowed[r, c]) and not bool(blocked[r, c]) for r, c in coords):
                placements.append(coords)

    return placements


def _place_component(
    allowed: np.ndarray,
    blocked: np.ndarray,
    area_choices: tuple[int, ...],
) -> list[tuple[int, int]] | None:
    shapes = [shape for shape in _ALL_SHAPES if len(shape) in area_choices]
    random.shuffle(shapes)

    placements: list[list[tuple[int, int]]] = []
    for shape in shapes:
        placements.extend(_candidate_placements(allowed, blocked, shape))

    if not placements:
        return None

    coords = random.choice(placements)
    _mark_blocked(blocked, coords)
    return coords


def _make_dashed_line(length: int) -> np.ndarray:
    line = np.full(length, BLACK, dtype=int)
    phase = random.choice((0, 1))
    line[phase::2] = GREY
    assert np.any(line == GREY)
    return line


def _compose_expected(
    big: np.ndarray,
    overlay_zone: np.ndarray,
    orientation: str,
    big_side: str,
) -> np.ndarray:
    overlay = np.zeros_like(big)
    small_span = int(overlay_zone.shape[0] if orientation == "horizontal" else overlay_zone.shape[1])
    rows, cols = _zone_slice(big.shape, small_span, orientation, big_side)
    overlay[rows, cols] = overlay_zone
    return _merge_layers(big, overlay)


def _sample_geometry() -> tuple[str, str, tuple[int, int], tuple[int, int]]:
    orientation = random.choice(("horizontal", "vertical"))
    big_side = random.choice(("top", "bottom")) if orientation == "horizontal" else random.choice(("left", "right"))

    cross_span = random.randint(10, 16)
    small_span = random.randint(5, 8)
    big_span = random.randint(small_span + 2, min(14, small_span + 6))

    if orientation == "horizontal":
        return orientation, big_side, (big_span, cross_span), (small_span, cross_span)
    return orientation, big_side, (cross_span, big_span), (cross_span, small_span)


def _branch_masks(big: np.ndarray, overlay_zone: np.ndarray, orientation: str, big_side: str) -> dict[str, np.ndarray]:
    overlay = np.zeros_like(big)
    small_span = int(overlay_zone.shape[0] if orientation == "horizontal" else overlay_zone.shape[1])
    rows, cols = _zone_slice(big.shape, small_span, orientation, big_side)
    overlay[rows, cols] = overlay_zone

    big_fg = big != BLACK
    overlay_fg = overlay != BLACK
    return {
        "same": big_fg & overlay_fg & (big == overlay),
        "conflict": big_fg & overlay_fg & (big != overlay),
        "big_only": big_fg & ~overlay_fg,
        "small_only": overlay_fg & ~big_fg,
    }


def _build_attempt() -> np.ndarray | None:
    orientation, big_side, big_shape, zone_shape = _sample_geometry()
    big = np.full(big_shape, BLACK, dtype=int)
    overlay_zone = np.full(zone_shape, BLACK, dtype=int)

    blocked_zone = np.zeros(zone_shape, dtype=bool)
    blocked_big = np.zeros(big_shape, dtype=bool)

    colors = random.sample(list(PALETTE), 6)
    same_color = int(colors[0])
    conflict_big_color = int(colors[1])
    conflict_small_color = int(colors[2])
    small_only_color = int(colors[3])
    big_only_zone_color = int(colors[4])
    far_color = int(colors[5])

    small_span = int(zone_shape[0] if orientation == "horizontal" else zone_shape[1])

    def place_zone(area_choices: tuple[int, ...]) -> list[tuple[int, int]] | None:
        allowed = np.ones(zone_shape, dtype=bool)
        return _place_component(allowed, blocked_zone, area_choices)

    same_coords = place_zone((2, 3, 4))
    conflict_coords = place_zone((2, 3, 4))
    small_only_coords = place_zone((1, 2, 3))
    big_only_zone_coords = place_zone((1, 2, 3))
    if any(coords is None for coords in (same_coords, conflict_coords, small_only_coords, big_only_zone_coords)):
        return None

    same_big_coords = _map_zone_coords(same_coords, big_shape, small_span, orientation, big_side)
    conflict_big_coords = _map_zone_coords(conflict_coords, big_shape, small_span, orientation, big_side)
    big_only_big_coords = _map_zone_coords(big_only_zone_coords, big_shape, small_span, orientation, big_side)

    for coords in (same_big_coords, conflict_big_coords, big_only_big_coords):
        _mark_blocked(blocked_big, coords)

    _paint(big, same_big_coords, same_color)
    _paint(big, conflict_big_coords, conflict_big_color)
    _paint(big, big_only_big_coords, big_only_zone_color)

    _paint(overlay_zone, same_coords, same_color)
    _paint(overlay_zone, conflict_coords, conflict_small_color)
    _paint(overlay_zone, small_only_coords, small_only_color)

    zone_rows, zone_cols = _zone_slice(big_shape, small_span, orientation, big_side)
    far_allowed = np.ones(big_shape, dtype=bool)
    far_allowed[zone_rows, zone_cols] = False

    far_coords = _place_component(far_allowed, blocked_big, (1, 2, 3, 4))
    if far_coords is None:
        return None
    _paint(big, far_coords, far_color)

    extra_zone_count = random.randint(0, 3)
    for _ in range(extra_zone_count):
        coords = place_zone((1, 2, 3, 4))
        if coords is None:
            continue

        mode = random.choice(("same", "conflict", "big", "small"))
        if mode == "same":
            color = int(random.choice(PALETTE))
            big_coords = _map_zone_coords(coords, big_shape, small_span, orientation, big_side)
            _mark_blocked(blocked_big, big_coords)
            _paint(big, big_coords, color)
            _paint(overlay_zone, coords, color)
        elif mode == "conflict":
            big_color, small_color = random.sample(list(PALETTE), 2)
            big_coords = _map_zone_coords(coords, big_shape, small_span, orientation, big_side)
            _mark_blocked(blocked_big, big_coords)
            _paint(big, big_coords, int(big_color))
            _paint(overlay_zone, coords, int(small_color))
        elif mode == "big":
            color = int(random.choice(PALETTE))
            big_coords = _map_zone_coords(coords, big_shape, small_span, orientation, big_side)
            _mark_blocked(blocked_big, big_coords)
            _paint(big, big_coords, color)
        else:
            color = int(random.choice(PALETTE))
            _paint(overlay_zone, coords, color)

    extra_far_count = random.randint(0, 2)
    for _ in range(extra_far_count):
        coords = _place_component(far_allowed, blocked_big, (1, 2, 3, 4))
        if coords is None:
            continue
        _paint(big, coords, int(random.choice(PALETTE)))

    masks = _branch_masks(big, overlay_zone, orientation, big_side)
    if not all(bool(mask.any()) for mask in masks.values()):
        return None

    expected = _compose_expected(big, overlay_zone, orientation, big_side)
    if not np.any(expected == RED):
        return None

    if orientation == "horizontal":
        dashed = _make_dashed_line(big_shape[1])[None, :]
        small = np.flipud(overlay_zone)
        if big_side == "top":
            inp = np.vstack([big, dashed, small])
        else:
            inp = np.vstack([small, dashed, big])
    else:
        dashed = _make_dashed_line(big_shape[0])[:, None]
        small = np.fliplr(overlay_zone)
        if big_side == "left":
            inp = np.hstack([big, dashed, small])
        else:
            inp = np.hstack([small, dashed, big])

    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _build_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid dashed-fold puzzle input")
