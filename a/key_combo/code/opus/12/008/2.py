from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7
TEAL = 8

_RNG = np.random.default_rng()
_GEN_CALL_COUNT = 0
_GEN_MAX_ATTEMPTS = 300
_GEN_HEIGHT_RANGE = (8, 13)  # output-layer height; input height is 2H+1
_GEN_WIDTH_RANGE = (12, 19)
_GEN_BOX_GAP = 1


def _overlap(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
    ar, ac, aside = a
    br, bc, bside = b
    return ar < br + bside and br < ar + aside and ac < bc + bside and bc < ac + aside


def _separator_row(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rows = [r for r in range(h) if np.all(g[r, :] == RED)]
    assert len(rows) == 1, "expected exactly one full-width red separator row"
    sep = rows[0]
    assert 2 * sep + 1 == h, "separator must split two equal-height masks"
    assert int(np.count_nonzero(g == RED)) == w, "red may only appear in the separator row"
    return sep


def _selected_squares(active: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = map(int, active.shape)
    selected: list[tuple[int, int, int]] = []
    for side in range(min(h, w), 1, -1):
        for r in range(h - side + 1):
            for c in range(w - side + 1):
                square = (r, c, side)
                if any(_overlap(square, other) for other in selected):
                    continue
                if bool(np.all(active[r : r + side, c : c + side])):
                    selected.append(square)
    return selected


def _color_for_side(side: int) -> int:
    if side == 2:
        return TEAL
    if side == 3:
        return YELLOW
    if side == 4:
        return ORANGE
    assert side >= 5
    return PINK


def main(grid: np.ndarray) -> np.ndarray:
    """Collapse the xor of two stacked masks, then greedily recolor its largest non-overlapping squares by side length.

    Input structure
    - A full-width red separator row splits the grid into equal-height upper and lower masks.
    - The upper mask uses black background with blue foreground cells; the lower mask uses black background with yellow foreground cells.
    - Foreground cells in the two masks may overlap spatially or appear independently.

    Rule
    - Remove the separator and align the two masks cell by cell.
    - Mark a cell active iff exactly one aligned mask cell is foreground; inactive cells become black.
    - In the active region, greedily select all axis-aligned all-active squares of side at least 2, processing larger sides first and using row-major tie-breaking within a side length.
    - Recolor selected squares by size: side 2 -> teal, side 3 -> yellow, side 4 -> orange, side >= 5 -> pink; active cells not claimed by a selected square stay green.

    Key insight
    - The output depends on two linked steps: first compute the symmetric difference of the stacked masks, then apply largest-first non-overlapping square extraction on that derived active region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE, RED, YELLOW}), "unexpected input colors"

    sep = _separator_row(g)
    upper = g[:sep, :]
    lower = g[sep + 1 :, :]
    assert upper.shape == lower.shape
    assert set(map(int, np.unique(upper))).issubset({BLACK, BLUE}), "upper mask must use black/blue only"
    assert set(map(int, np.unique(lower))).issubset({BLACK, YELLOW}), "lower mask must use black/yellow only"

    active = (upper != BLACK) ^ (lower != BLACK)
    out = np.zeros_like(upper, dtype=int)
    out[active] = GREEN
    for r, c, side in _selected_squares(active):
        out[r : r + side, c : c + side] = _color_for_side(side)
    return out


def _mask_from_coords(shape: tuple[int, int], coords: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


_SPLIT5_A = _mask_from_coords(
    (5, 5),
    [
        (0, 0), (0, 1),
        (1, 0), (1, 1),
        (2, 0), (2, 1), (2, 2),
        (3, 0), (3, 1),
        (4, 0),
    ],
)

_SPLIT4_A = _mask_from_coords(
    (4, 4),
    [
        (0, 0), (0, 1),
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1),
        (3, 0),
    ],
)

_AMBIG2 = np.ones((3, 3), dtype=bool)
_AMBIG2[2, 2] = False

_AMBIG3 = np.ones((4, 4), dtype=bool)
_AMBIG3[3, 3] = False

_SHARED = _mask_from_coords(
    (3, 3),
    [
        (0, 0),
        (1, 0), (1, 1), (1, 2),
        (2, 1),
    ],
)


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, copy=True)
    if bool(_RNG.integers(2)):
        out = np.fliplr(out)
    out = np.rot90(out, int(_RNG.integers(4)))
    return out


def _box_separated(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_boxes(
    layer_shape: tuple[int, int],
    boxes: dict[str, int],
    gap: int,
) -> dict[str, tuple[int, int]]:
    h, w = layer_shape
    ordered = sorted(boxes.items(), key=lambda item: item[1], reverse=True)
    placed: dict[str, tuple[int, int, int, int]] = {}

    def backtrack(index: int) -> bool:
        if index == len(ordered):
            return True

        name, side = ordered[index]
        candidates = [(r, c) for r in range(h - side + 1) for c in range(w - side + 1)]
        _RNG.shuffle(candidates)
        for r, c in candidates:
            box = (r, c, r + side - 1, c + side - 1)
            if all(_box_separated(box, other, gap) for other in placed.values()):
                placed[name] = box
                if backtrack(index + 1):
                    return True
                del placed[name]
        return False

    if not backtrack(0):
        raise ValueError("failed to place non-touching generation boxes")
    return {name: (box[0], box[1]) for name, box in placed.items()}


def _paint_mask(target: np.ndarray, top_left: tuple[int, int], mask: np.ndarray) -> None:
    r0, c0 = top_left
    h, w = mask.shape
    target[r0 : r0 + h, c0 : c0 + w] |= mask


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(comp)
    return parts


def _component_areas(mask: np.ndarray) -> list[int]:
    return sorted(len(comp) for comp in _components(mask))


def _expected_output(upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    active = upper ^ lower
    out = np.zeros_like(upper, dtype=int)
    out[active] = GREEN
    for r, c, side in _selected_squares(active):
        out[r : r + side, c : c + side] = _color_for_side(side)
    return out


def _build_pattern(pattern: str) -> tuple[np.ndarray, list[int]]:
    layer_h = int(_RNG.integers(*_GEN_HEIGHT_RANGE))
    layer_w = int(_RNG.integers(*_GEN_WIDTH_RANGE))

    if pattern == "5_and_2":
        boxes = {"split": 5, "ambig": 3, "shared": 3}
        expected_sizes = [5, 2]
        split_a = _random_transform(_SPLIT5_A)
        split_b = ~split_a
        ambig = _random_transform(_AMBIG2)
    elif pattern == "4_and_3":
        boxes = {"split": 4, "ambig": 4, "shared": 3}
        expected_sizes = [4, 3]
        split_a = _random_transform(_SPLIT4_A)
        split_b = ~split_a
        ambig = _random_transform(_AMBIG3)
    else:
        raise AssertionError(f"unknown pattern: {pattern}")

    shared = _random_transform(_SHARED)
    positions = _place_boxes((layer_h, layer_w), boxes, _GEN_BOX_GAP)

    upper = np.zeros((layer_h, layer_w), dtype=bool)
    lower = np.zeros((layer_h, layer_w), dtype=bool)

    if bool(_RNG.integers(2)):
        _paint_mask(upper, positions["split"], split_a)
        _paint_mask(lower, positions["split"], split_b)
    else:
        _paint_mask(upper, positions["split"], split_b)
        _paint_mask(lower, positions["split"], split_a)

    if bool(_RNG.integers(2)):
        _paint_mask(upper, positions["ambig"], ambig)
    else:
        _paint_mask(lower, positions["ambig"], ambig)

    _paint_mask(upper, positions["shared"], shared)
    _paint_mask(lower, positions["shared"], shared)

    grid = np.zeros((2 * layer_h + 1, layer_w), dtype=int)
    grid[:layer_h, :][upper] = BLUE
    grid[layer_h, :] = RED
    grid[layer_h + 1 :, :][lower] = YELLOW
    return grid, expected_sizes


def _verify_generated(grid: np.ndarray, expected_sizes: list[int]) -> bool:
    try:
        sep = _separator_row(grid)
    except AssertionError:
        return False

    upper = grid[:sep, :] == BLUE
    lower = grid[sep + 1 :, :] == YELLOW

    upper_areas = _component_areas(upper)
    lower_areas = _component_areas(lower)
    if not (2 <= len(upper_areas) <= 4 and 2 <= len(lower_areas) <= 4):
        return False
    if any(area < 5 or area > 15 for area in upper_areas + lower_areas):
        return False

    overlap = upper & lower
    active = upper ^ lower
    if int(np.count_nonzero(overlap)) < 5:
        return False
    if not np.any(active):
        return False
    if not np.any(~active):
        return False

    selected = _selected_squares(active)
    sizes = [side for _r, _c, side in selected]
    if sizes != expected_sizes:
        return False

    out = main(grid)
    if not np.array_equal(out, _expected_output(upper, lower)):
        return False

    if GREEN not in set(map(int, np.unique(out))):
        return False
    return True


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    patterns = ("5_and_2", "4_and_3")
    pattern = patterns[_GEN_CALL_COUNT % len(patterns)]
    _GEN_CALL_COUNT += 1

    for _ in range(_GEN_MAX_ATTEMPTS):
        grid, expected_sizes = _build_pattern(pattern)
        if _verify_generated(grid, expected_sizes):
            return grid
    raise RuntimeError("failed to generate a valid input grid")
