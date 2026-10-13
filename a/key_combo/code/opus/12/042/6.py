import numpy as np
from collections import deque


class Color:
    BLACK = 0
    RED = 2
    GREEN = 3
    YELLOW = 4


GEN_PANEL_H_RANGE = (8, 13)
GEN_PANEL_W_RANGE = (12, 19)
GEN_FRAME_COUNT_RANGE = (3, 6)
GEN_OUTER_H_RANGE = (3, 7)
GEN_OUTER_W_RANGE = (3, 8)
GEN_MAX_LAYOUT_ATTEMPTS = 400
GEN_MAX_INPUT_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == Color.YELLOW)]
    assert len(rows) == 1, "expected exactly one full yellow separator row"
    return rows[0]


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sep = _find_separator_row(grid)
    top = grid[:sep, :]
    bottom = grid[sep + 1 :, :]
    assert top.shape == bottom.shape, "top and bottom panels must align"
    return top, bottom


def _black_components(panel: np.ndarray) -> list[np.ndarray]:
    h, w = panel.shape
    black = panel == Color.BLACK
    seen = np.zeros_like(black, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not black[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(panel.shape, nr, nc) and black[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            mask = np.zeros_like(black, dtype=bool)
            for rr, cc in coords:
                mask[rr, cc] = True
            comps.append(mask)
    return comps


def _hole_masks(panel: np.ndarray) -> list[np.ndarray]:
    holes = []
    for comp in _black_components(panel):
        coords = np.argwhere(comp)
        if coords.size == 0:
            continue
        touches_border = np.any(
            (coords[:, 0] == 0)
            | (coords[:, 0] == panel.shape[0] - 1)
            | (coords[:, 1] == 0)
            | (coords[:, 1] == panel.shape[1] - 1)
        )
        if not touches_border:
            holes.append(comp)
    return holes


def _xor_active(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    return (top == Color.RED) ^ (bottom == Color.RED)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each top-panel hole whose aligned XOR signal contains at least one active pixel.

    Input structure:
    - The input is split by one full yellow separator row into equal-sized top and bottom panels.
    - The top panel contains red one-pixel rectangular frames on a black background, producing black enclosed holes.
    - The bottom panel is another red/black mask aligned cell-for-cell with the top panel.

    Rule:
    - Remove the yellow separator row and align the two panels.
    - Compute the pixelwise XOR of red occupancy: a cell is active exactly when one panel is red and the other is black.
    - Detect every black hole enclosed by red walls in the top panel, and fill an entire hole green iff the XOR map has at least one active cell inside that hole.

    Key insight:
    - The XOR result is only an intermediate selector: it never appears in the output directly, but it decides which top-panel holes become green.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.RED, Color.YELLOW}, "unexpected colors"

    top, bottom = _split_panels(g)
    holes = _hole_masks(top)
    active = _xor_active(top, bottom)

    out = top.copy()
    for hole in holes:
        if np.any(active & hole):
            out[hole] = Color.GREEN
    return out


def _expanded_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a[0] - pad, a[1] - pad, a[2] + pad, a[3] + pad
    br0, bc0, br1, bc1 = b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _draw_frame(panel: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = rect
    panel[r0, c0 : c1 + 1] = Color.RED
    panel[r1, c0 : c1 + 1] = Color.RED
    panel[r0 : r1 + 1, c0] = Color.RED
    panel[r0 : r1 + 1, c1] = Color.RED
    return panel


def _sample_layout() -> tuple[np.ndarray, list[np.ndarray]]:
    h = int(np.random.randint(*GEN_PANEL_H_RANGE))
    w = int(np.random.randint(*GEN_PANEL_W_RANGE))
    frame_count = int(np.random.randint(*GEN_FRAME_COUNT_RANGE))

    panel = np.full((h, w), Color.BLACK, dtype=int)
    rects: list[tuple[int, int, int, int]] = []
    hole_masks: list[np.ndarray] = []

    for _ in range(frame_count):
        placed = False
        for _ in range(GEN_MAX_LAYOUT_ATTEMPTS):
            outer_h = int(np.random.randint(*GEN_OUTER_H_RANGE))
            outer_w = int(np.random.randint(*GEN_OUTER_W_RANGE))
            outer_h = min(outer_h, h)
            outer_w = min(outer_w, w)
            if outer_h < 3 or outer_w < 3:
                continue
            if outer_h > h or outer_w > w:
                continue

            r0 = int(np.random.randint(0, h - outer_h + 1))
            c0 = int(np.random.randint(0, w - outer_w + 1))
            rect = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if any(_expanded_overlap(rect, other, pad=1) for other in rects):
                continue

            rects.append(rect)
            _draw_frame(panel, rect)
            hole = np.zeros((h, w), dtype=bool)
            hole[r0 + 1 : r0 + outer_h - 1, c0 + 1 : c0 + outer_w - 1] = True
            hole_masks.append(hole)
            placed = True
            break
        if not placed:
            raise ValueError("failed to place enough frames")

    sizes = sorted(int(mask.sum()) for mask in hole_masks)
    if len(set(sizes)) < 2:
        raise ValueError("need varied hole sizes")
    return panel, hole_masks


def _choose_signal_positions(mask: np.ndarray, min_count: int, max_count: int) -> list[tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if len(coords) < min_count:
        raise ValueError("not enough candidate signal positions")
    count = int(np.random.randint(min_count, min(max_count, len(coords)) + 1))
    chosen_idx = np.random.choice(len(coords), size=count, replace=False)
    return [coords[int(i)] for i in np.atleast_1d(chosen_idx)]


def _build_bottom(top: np.ndarray, holes: list[np.ndarray]) -> tuple[np.ndarray, set[int], np.ndarray]:
    hole_count = len(holes)
    selected_count = int(np.random.randint(1, hole_count))
    selected = set(map(int, np.random.choice(hole_count, size=selected_count, replace=False)))

    active = np.zeros_like(top, dtype=bool)
    union_holes = np.zeros_like(top, dtype=bool)
    for hole in holes:
        union_holes |= hole

    for idx, hole in enumerate(holes):
        if idx not in selected:
            continue
        for r, c in _choose_signal_positions(hole, 1, 3):
            active[r, c] = True

    red_border = top == Color.RED
    exterior_black = (top == Color.BLACK) & ~union_holes

    for r, c in _choose_signal_positions(red_border, 1, 4):
        active[r, c] = True
    for r, c in _choose_signal_positions(exterior_black, 1, 5):
        active[r, c] = True

    if np.random.rand() < 0.5:
        for idx, hole in enumerate(holes):
            if idx in selected and np.random.rand() < 0.5:
                coords = [tuple(map(int, rc)) for rc in np.argwhere(hole & ~active)]
                if coords:
                    r, c = coords[int(np.random.randint(len(coords)))]
                    active[r, c] = True

    bottom = top.copy()
    bottom[active & (top == Color.RED)] = Color.BLACK
    bottom[active & (top == Color.BLACK)] = Color.RED
    return bottom, selected, active


def _expected_output(top: np.ndarray, holes: list[np.ndarray], selected: set[int]) -> np.ndarray:
    out = top.copy()
    for idx, hole in enumerate(holes):
        if idx in selected:
            out[hole] = Color.GREEN
    return out


def _validate_generated(grid: np.ndarray) -> None:
    top, bottom = _split_panels(grid)
    holes = _hole_masks(top)
    assert 3 <= len(holes) <= 5, "generator must build 3-5 holes"

    active = _xor_active(top, bottom)
    hit_flags = [bool(np.any(active & hole)) for hole in holes]
    assert any(hit_flags), "at least one hole must be selected"
    assert not all(hit_flags), "at least one hole must remain unselected"

    union_holes = np.zeros_like(top, dtype=bool)
    for hole in holes:
        union_holes |= hole
    assert np.any(active & (top == Color.RED)), "need XOR activity on frame walls"
    assert np.any(active & ((top == Color.BLACK) & ~union_holes)), "need XOR activity outside holes"

    out = main(grid)
    assert out.shape == top.shape
    for hole, hit in zip(holes, hit_flags):
        vals = set(map(int, np.unique(out[hole])))
        if hit:
            assert vals == {Color.GREEN}, "selected hole must be fully green"
        else:
            assert vals == {Color.BLACK}, "unselected hole must stay black"
    assert np.all(out[top == Color.RED] == Color.RED), "frame walls must remain red"


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_INPUT_ATTEMPTS):
        try:
            top, holes = _sample_layout()
            bottom, selected, _active = _build_bottom(top, holes)
        except ValueError:
            continue

        sep = np.full((1, top.shape[1]), Color.YELLOW, dtype=int)
        grid = np.vstack([top, sep, bottom])

        expected = _expected_output(top, holes, selected)
        try:
            out = main(grid)
            if not np.array_equal(out, expected):
                continue
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
