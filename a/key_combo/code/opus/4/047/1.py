import random

import numpy as np


BACKGROUND = 0
FOREGROUND = 3
GRID_SIZE = 5
AREA_BOUNDS = (4, 8)
_GEN_CALL_COUNT = 0


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _row_intervals(mask: np.ndarray) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    for r in range(mask.shape[0]):
        cols = np.flatnonzero(mask[r])
        if cols.size:
            rows.append((int(r), int(cols[0]), int(cols[-1])))
    return rows


def _fill_outline_rows(grid: np.ndarray, *, bg: int, fg: int) -> np.ndarray:
    filled = np.full_like(grid, int(bg))
    for r in range(grid.shape[0]):
        cols = np.flatnonzero(grid[r] == fg)
        if cols.size == 0:
            continue
        assert cols.size == 2, "each occupied row must have exactly two boundary pixels"
        filled[r, int(cols[0]) : int(cols[-1]) + 1] = int(fg)
    return filled


def _outline_from_filled(filled_mask: np.ndarray) -> np.ndarray:
    outline = np.zeros_like(filled_mask, dtype=bool)
    for r, c0, c1 in _row_intervals(filled_mask):
        outline[r, c0] = True
        outline[r, c1] = True
    return outline


def _connected8(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    pending = {tuple(map(int, rc)) for rc in coords}
    stack = [next(iter(pending))]
    seen = set()
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nxt = (r + dr, c + dc)
                if nxt in pending and nxt not in seen:
                    stack.append(nxt)
    return len(seen) == len(pending)


def _expand_from_filled_mask(filled_mask: np.ndarray) -> np.ndarray:
    bg_mask = ~filled_mask
    return np.kron(bg_mask.astype(int), bg_mask.astype(int)).astype(bool)


def _validate_filled_mask(filled_mask: np.ndarray) -> None:
    assert filled_mask.shape == (GRID_SIZE, GRID_SIZE)
    bbox = _bbox(filled_mask)
    assert bbox is not None, "filled mask must be non-empty"
    r0, c0, r1, c1 = bbox
    assert r1 > r0, "row span must cover at least two rows"
    assert c1 - c0 + 1 >= 3, "column span must be at least three cells so filling matters"

    rows = _row_intervals(filled_mask)
    assert len(rows) >= 2, "shape must occupy at least two rows"
    occupied = [r for r, _, _ in rows]
    assert occupied == list(range(occupied[0], occupied[-1] + 1)), "occupied rows must be contiguous"

    widths = []
    area = 0
    for _r, c0, c1 in rows:
        width = c1 - c0 + 1
        widths.append(width)
        area += width
        assert width >= 2, "each occupied row must have two distinct outline pixels"

    lo, hi = AREA_BOUNDS
    assert lo <= area <= hi, "filled area must stay within the required bounds"
    assert max(widths) >= 3, "at least one row must widen beyond the sparse outline"
    assert widths[0] == 2 or widths[-1] == 2, "top or bottom row must connect the two outline sides"

    for (r_prev, l_prev, rr_prev), (r_cur, l_cur, rr_cur) in zip(rows, rows[1:]):
        assert r_cur == r_prev + 1
        assert abs(l_cur - l_prev) <= 1, "left boundary must stay 8-connected"
        assert abs(rr_cur - rr_prev) <= 1, "right boundary must stay 8-connected"

    outline = _outline_from_filled(filled_mask)
    assert _connected8(outline), "sparse outline must be a single 8-connected object"


def _validate_input_grid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.shape == (GRID_SIZE, GRID_SIZE)
    assert set(map(int, np.unique(grid))).issubset({BACKGROUND, FOREGROUND})
    fg_rows = [r for r in range(GRID_SIZE) if np.any(grid[r] == FOREGROUND)]
    assert fg_rows, "input must contain one outline"
    for r in fg_rows:
        assert int(np.sum(grid[r] == FOREGROUND)) == 2, "each occupied row must expose exactly two boundary pixels"

    filled = _fill_outline_rows(grid, bg=BACKGROUND, fg=FOREGROUND) == FOREGROUND
    _validate_filled_mask(filled)

    outline_mask = grid == FOREGROUND
    assert not np.array_equal(outline_mask, filled), "generator must not allow filling to be skipped"

    filled_out = _expand_from_filled_mask(filled)
    outline_out = _expand_from_filled_mask(outline_mask)
    assert not np.array_equal(filled_out, outline_out), "Kronecker stage must depend on the fill result"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the sparse row-paired outline, then use the solid result as a self-expanding Kronecker mask.

    Input structure:
    - The grid is a 5x5 black background with exactly one green sparse outline object.
    - On every occupied row, the outline has exactly two green pixels marking that row's left and right boundary.
    - The outline spans at least two rows and at least three columns overall, and its filled area is small but nontrivial.

    Rule:
    - First perform row-span filling: on each occupied row, paint every cell between the two green boundary pixels green.
    - Treat this filled 5x5 binary image as the mask for a self-expansion into a 25x25 output.
    - Each black input cell becomes the foreground/background-inverted 5x5 filled mask, while each green input cell becomes an all-black 5x5 block.

    Key insight:
    - The visible sparse contour is not itself the expansion mask; only the row-filled solid shape defines the Kronecker structure, so both stages are necessary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape == (GRID_SIZE, GRID_SIZE), "expected a 5x5 grid"

    colors = sorted(int(v) for v in np.unique(g) if int(v) != BACKGROUND)
    assert colors == [FOREGROUND], "expected exactly one green foreground outline"

    filled = _fill_outline_rows(g, bg=BACKGROUND, fg=FOREGROUND)
    filled_mask = filled == FOREGROUND
    _validate_filled_mask(filled_mask)

    out_mask = _expand_from_filled_mask(filled_mask)
    out = np.full(out_mask.shape, BACKGROUND, dtype=int)
    out[out_mask] = FOREGROUND
    return out


def _canonical_shape_masks() -> list[np.ndarray]:
    masks: dict[bytes, np.ndarray] = {}
    for height in range(2, GRID_SIZE):
        sequences: list[list[tuple[int, int]]] = []

        def build(rows: list[tuple[int, int]]) -> None:
            if len(rows) == height:
                widths = [right - left + 1 for left, right in rows]
                area = sum(widths)
                if not (AREA_BOUNDS[0] <= area <= AREA_BOUNDS[1]):
                    return
                if max(widths) < 3:
                    return
                if widths[0] != 2 and widths[-1] != 2:
                    return

                min_left = min(left for left, _ in rows)
                norm = [(left - min_left, right - min_left) for left, right in rows]
                max_right = max(right for _, right in norm)
                if max_right >= GRID_SIZE:
                    return

                tight = np.zeros((height, max_right + 1), dtype=bool)
                for r, (left, right) in enumerate(norm):
                    tight[r, left : right + 1] = True

                try:
                    _validate_filled_mask(_place_tight_mask(tight, 0, 0))
                except AssertionError:
                    return

                masks[tight.tobytes() + str(tight.shape).encode()] = tight
                return

            if not rows:
                starts = [(left, right) for left in range(GRID_SIZE - 1) for right in range(left + 1, GRID_SIZE)]
            else:
                prev_left, prev_right = rows[-1]
                starts = []
                for left in range(prev_left - 1, prev_left + 2):
                    for right in range(prev_right - 1, prev_right + 2):
                        if 0 <= left < right < GRID_SIZE:
                            starts.append((left, right))

            for interval in starts:
                build(rows + [interval])

        build([])

    return list(masks.values())


def _place_tight_mask(tight: np.ndarray, top: int, left: int) -> np.ndarray:
    mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    h, w = tight.shape
    mask[top : top + h, left : left + w] = tight
    return mask


def _shape_family(tight: np.ndarray) -> str:
    widths = [c1 - c0 + 1 for _, c0, c1 in _row_intervals(tight)]
    if tight.shape[0] == 2:
        return "two_row"
    if widths[0] == 2 and widths[-1] == 2:
        return "double_cap"
    if widths[0] == 2:
        return "top_cap"
    return "bottom_cap"


def _build_shape_buckets() -> dict[str, list[np.ndarray]]:
    buckets = {"two_row": [], "top_cap": [], "bottom_cap": [], "double_cap": []}
    for tight in _canonical_shape_masks():
        buckets[_shape_family(tight)].append(tight)
    for family, items in buckets.items():
        assert items, f"missing generator family: {family}"
    return buckets


_SHAPE_BUCKETS = _build_shape_buckets()
_FAMILY_ORDER = tuple(_SHAPE_BUCKETS.keys())


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    family = _FAMILY_ORDER[_GEN_CALL_COUNT % len(_FAMILY_ORDER)]
    _GEN_CALL_COUNT += 1
    tight = random.choice(_SHAPE_BUCKETS[family]).copy()

    if random.random() < 0.5:
        tight = np.fliplr(tight)

    h, w = tight.shape
    top = random.randint(0, GRID_SIZE - h)
    left = random.randint(0, GRID_SIZE - w)
    filled_mask = _place_tight_mask(tight, top, left)
    outline_mask = _outline_from_filled(filled_mask)

    grid = np.full((GRID_SIZE, GRID_SIZE), BACKGROUND, dtype=int)
    grid[outline_mask] = FOREGROUND
    _validate_input_grid(grid)
    return grid
