import numpy as np


BG_COLOR = 0
PALETTE = list(range(1, 10))
MAX_GEN_ATTEMPTS = 500
_RNG = np.random.default_rng()

DIR_TO_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# The center is the control color. The remaining colored cells form the arrow.
DIR_TO_MASK = {
    "up": frozenset({(0, 1), (1, 0), (1, 2)}),
    "down": frozenset({(1, 0), (1, 2), (2, 1)}),
    "left": frozenset({(0, 1), (1, 0), (2, 1)}),
    "right": frozenset({(0, 1), (1, 2), (2, 1)}),
}
MASK_TO_DIR = {mask: direction for direction, mask in DIR_TO_MASK.items()}
RING_CELLS = [(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _make_arrow_patch(border_color: int, control_color: int, direction: str, bg: int = BG_COLOR) -> np.ndarray:
    patch = np.full((3, 3), int(bg), dtype=int)
    patch[1, 1] = int(control_color)
    for r, c in DIR_TO_MASK[direction]:
        patch[r, c] = int(border_color)
    return patch


def _find_arrows(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    arrows = []
    for r in range(h - 2):
        for c in range(w - 2):
            patch = g[r : r + 3, c : c + 3]
            control_color = int(patch[1, 1])
            if control_color == int(bg):
                continue
            border_color = None
            colored_ring = []
            ok = True
            for rr, cc in RING_CELLS:
                val = int(patch[rr, cc])
                if val == int(bg):
                    continue
                if border_color is None:
                    border_color = val
                elif val != border_color:
                    ok = False
                    break
                colored_ring.append((rr, cc))
            if not ok or border_color is None or border_color == control_color:
                continue
            direction = MASK_TO_DIR.get(frozenset(colored_ring))
            if direction is None:
                continue
            arrows.append(
                {
                    "top_left": (int(r), int(c)),
                    "direction": direction,
                    "border_color": int(border_color),
                    "control_color": int(control_color),
                    "patch": patch.copy(),
                }
            )
    return arrows


def _find_square_sizes(grid: np.ndarray, bg: int) -> dict[int, int]:
    g = np.asarray(grid, dtype=int)
    square_sizes: dict[int, int] = {}
    for color in sorted(int(v) for v in np.unique(g) if int(v) != int(bg)):
        candidates = []
        for coords in _components(g == int(color)):
            area = int(coords.shape[0])
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            side_r = int(r1 - r0 + 1)
            side_c = int(c1 - c0 + 1)
            if side_r != side_c or side_r < 2 or area != side_r * side_c:
                continue
            if np.all(g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(color)):
                candidates.append(side_r)
        if len(candidates) == 1:
            square_sizes[int(color)] = int(candidates[0])
    return square_sizes


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    arrows = _find_arrows(g, bg)
    square_sizes = _find_square_sizes(g, bg)

    out = g.copy()
    for arrow in arrows:
        size = square_sizes.get(int(arrow["control_color"]))
        if size is None:
            continue
        r0, c0 = arrow["top_left"]
        dr, dc = DIR_TO_VEC[arrow["direction"]]
        patch = arrow["patch"]
        for step in range(1, int(size) + 1):
            rr = int(r0 + 3 * step * dr)
            cc = int(c0 + 3 * step * dc)
            if rr < 0 or cc < 0 or rr + 3 > h or cc + 3 > w:
                break
            out[rr : rr + 3, cc : cc + 3] = patch
    return out


def _in_bounds(top_left: tuple[int, int], height: int, width: int, grid_h: int, grid_w: int) -> bool:
    r, c = top_left
    return 0 <= r and 0 <= c and r + height <= grid_h and c + width <= grid_w


def _area_is_free(reserved: np.ndarray, r0: int, c0: int, height: int, width: int, margin: int = 1) -> bool:
    h, w = reserved.shape
    rr0 = max(0, int(r0) - margin)
    cc0 = max(0, int(c0) - margin)
    rr1 = min(h, int(r0 + height + margin))
    cc1 = min(w, int(c0 + width + margin))
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def _mark_reserved(reserved: np.ndarray, r0: int, c0: int, height: int, width: int, margin: int = 1) -> None:
    h, w = reserved.shape
    rr0 = max(0, int(r0) - margin)
    cc0 = max(0, int(c0) - margin)
    rr1 = min(h, int(r0 + height + margin))
    cc1 = min(w, int(c0 + width + margin))
    reserved[rr0:rr1, cc0:cc1] = True


def _candidate_arrow_positions(
    grid_h: int,
    grid_w: int,
    direction: str,
    square_size: int,
    visible_copies: int,
) -> list[tuple[int, int]]:
    candidates = []
    if direction == "right":
        if visible_copies == square_size:
            cols = list(range(grid_w - 3 - 3 * square_size + 1))
        else:
            cols = [grid_w - 3 - 3 * visible_copies]
        rows = list(range(grid_h - 2))
        _RNG.shuffle(rows)
        _RNG.shuffle(cols)
        for c in cols:
            for r in rows:
                candidates.append((int(r), int(c)))
    elif direction == "left":
        if visible_copies == square_size:
            cols = list(range(3 * square_size, grid_w - 2))
        else:
            cols = [3 * visible_copies]
        rows = list(range(grid_h - 2))
        _RNG.shuffle(rows)
        _RNG.shuffle(cols)
        for c in cols:
            for r in rows:
                candidates.append((int(r), int(c)))
    elif direction == "down":
        if visible_copies == square_size:
            rows = list(range(grid_h - 3 - 3 * square_size + 1))
        else:
            rows = [grid_h - 3 - 3 * visible_copies]
        cols = list(range(grid_w - 2))
        _RNG.shuffle(rows)
        _RNG.shuffle(cols)
        for r in rows:
            for c in cols:
                candidates.append((int(r), int(c)))
    else:
        if visible_copies == square_size:
            rows = list(range(3 * square_size, grid_h - 2))
        else:
            rows = [3 * visible_copies]
        cols = list(range(grid_w - 2))
        _RNG.shuffle(rows)
        _RNG.shuffle(cols)
        for r in rows:
            for c in cols:
                candidates.append((int(r), int(c)))
    return candidates


def _place_arrow(
    grid: np.ndarray,
    reserved: np.ndarray,
    direction: str,
    border_color: int,
    control_color: int,
    square_size: int,
    visible_copies: int,
) -> dict | None:
    grid_h, grid_w = grid.shape
    patch = _make_arrow_patch(border_color, control_color, direction)
    dr, dc = DIR_TO_VEC[direction]
    for r0, c0 in _candidate_arrow_positions(grid_h, grid_w, direction, square_size, visible_copies):
        boxes = []
        ok = True
        for step in range(visible_copies + 1):
            rr = int(r0 + 3 * step * dr)
            cc = int(c0 + 3 * step * dc)
            if not _in_bounds((rr, cc), 3, 3, grid_h, grid_w):
                ok = False
                break
            boxes.append((rr, cc))
        if not ok:
            continue
        if any(not _area_is_free(reserved, rr, cc, 3, 3, margin=1) for rr, cc in boxes):
            continue
        grid[r0 : r0 + 3, c0 : c0 + 3] = patch
        for rr, cc in boxes:
            _mark_reserved(reserved, rr, cc, 3, 3, margin=1)
        return {
            "top_left": (int(r0), int(c0)),
            "direction": direction,
            "border_color": int(border_color),
            "control_color": int(control_color),
            "patch": patch,
            "square_size": int(square_size),
            "visible_copies": int(visible_copies),
        }
    return None


def _place_square(grid: np.ndarray, reserved: np.ndarray, color: int, side: int) -> tuple[int, int] | None:
    grid_h, grid_w = grid.shape
    rows = list(range(grid_h - side + 1))
    cols = list(range(grid_w - side + 1))
    _RNG.shuffle(rows)
    _RNG.shuffle(cols)
    for r0 in rows:
        for c0 in cols:
            if not _area_is_free(reserved, r0, c0, side, side, margin=1):
                continue
            grid[r0 : r0 + side, c0 : c0 + side] = int(color)
            _mark_reserved(reserved, r0, c0, side, side, margin=1)
            return (int(r0), int(c0))
    return None


def _render_expected(inp: np.ndarray, arrow_records: list[dict]) -> np.ndarray:
    out = np.asarray(inp, dtype=int).copy()
    h, w = out.shape
    for rec in arrow_records:
        r0, c0 = rec["top_left"]
        dr, dc = DIR_TO_VEC[rec["direction"]]
        patch = rec["patch"]
        for step in range(1, int(rec["square_size"]) + 1):
            rr = int(r0 + 3 * step * dr)
            cc = int(c0 + 3 * step * dc)
            if rr < 0 or cc < 0 or rr + 3 > h or cc + 3 > w:
                break
            out[rr : rr + 3, cc : cc + 3] = patch
    return out


def _gen_attempt() -> np.ndarray | None:
    num_pairs = int(_RNG.integers(2, 5))
    min_side = 18 + 2 * num_pairs
    grid_h = int(_RNG.integers(min_side, 31))
    grid_w = int(_RNG.integers(min_side, 31))
    if grid_h < 9 or grid_w < 9:
        return None

    all_colors = list(_RNG.choice(PALETTE, size=2 * num_pairs, replace=False))
    control_colors = all_colors[:num_pairs]
    border_colors = all_colors[num_pairs:]
    square_sizes = [int(_RNG.integers(2, 6)) for _ in range(num_pairs)]
    directions = [str(_RNG.choice(list(DIR_TO_VEC))) for _ in range(num_pairs)]
    if num_pairs > 1 and len(set(directions)) == 1:
        others = [d for d in DIR_TO_VEC if d != directions[0]]
        directions[0] = str(_RNG.choice(others))

    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)
    arrow_records = []

    for border_color, control_color, square_size, direction in zip(
        border_colors,
        control_colors,
        square_sizes,
        directions,
    ):
        if bool(_RNG.integers(0, 2)):
            visible_copies = int(square_size)
        else:
            visible_copies = int(_RNG.integers(1, square_size))
        record = _place_arrow(
            grid,
            reserved,
            direction,
            int(border_color),
            int(control_color),
            int(square_size),
            int(visible_copies),
        )
        if record is None:
            return None
        arrow_records.append(record)

    for control_color, square_size in zip(control_colors, square_sizes):
        if _place_square(grid, reserved, int(control_color), int(square_size)) is None:
            return None

    detected_arrows = _find_arrows(grid, BG_COLOR)
    if len(detected_arrows) != num_pairs:
        return None
    detected_squares = _find_square_sizes(grid, BG_COLOR)
    if {int(rec["control_color"]): int(rec["square_size"]) for rec in arrow_records} != detected_squares:
        return None

    expected = _render_expected(grid, arrow_records)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
