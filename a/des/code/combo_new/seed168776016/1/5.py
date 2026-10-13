import numpy as np


BG = 0
GRAY = 5
AZURE = 8
BORDER = 2
MAX_GENERATION_ATTEMPTS = 400
_RNG = np.random.default_rng()
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
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
                for dr, dc in _NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _quadrant_boxes(line_r: int, line_c: int, h: int, w: int) -> dict[str, tuple[int, int, int, int]]:
    return {
        "TL": (BORDER, line_r - 1, BORDER, line_c - 1),
        "TR": (BORDER, line_r - 1, line_c + 1, w - BORDER - 1),
        "BL": (line_r + 1, h - BORDER - 1, BORDER, line_c - 1),
        "BR": (line_r + 1, h - BORDER - 1, line_c + 1, w - BORDER - 1),
    }


def _center2(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, r1, c0, c1 = box
    return r0 + r1, c0 + c1


def _quadrant_colors(top: int, bottom: int, left: int, right: int) -> dict[str, int]:
    return {"TL": left, "TR": top, "BL": bottom, "BR": right}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy one gray stamp prototype into all four quadrants using a rotated border-color legend.

    Input structure:
    - Black fills the background, and a two-pixel-thick colored frame surrounds the grid.
    - Inside the frame, one azure horizontal dashed line and one azure vertical dashed line split the interior into four quadrants.
    - Exactly one small gray 4-connected object appears fully inside a single quadrant; it is the stamp prototype.

    Rule:
    - Locate the azure dashed cross to recover the four interior quadrants.
    - Find the gray prototype and record each of its pixels by local position relative to the center of its quadrant.
    - Map border colors to quadrant fill colors with a clockwise-shifted legend: top-left uses the left border color, top-right uses the top border color, bottom-right uses the right border color, and bottom-left uses the bottom border color.
    - Stamp the prototype into every quadrant at the same local coordinates, recoloring each copy with that quadrant's mapped color; the original gray copy is recolored as well.
    - Preserve the frame, black background, and azure dashed lines.

    Key insight:
    - The gray object is not moved by mirroring or rotation; it is copied by center-relative offsets, and the border-to-quadrant color mapping is rotated one side clockwise from the obvious reading.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h > 2 * BORDER and w > 2 * BORDER, "grid too small for a two-pixel border"

    inner = g[BORDER : h - BORDER, BORDER : w - BORDER]
    row_counts = np.sum(inner == AZURE, axis=1)
    col_counts = np.sum(inner == AZURE, axis=0)
    line_r_inner = int(np.argmax(row_counts))
    line_c_inner = int(np.argmax(col_counts))
    assert row_counts[line_r_inner] > 0 and col_counts[line_c_inner] > 0, "missing azure dashed cross"
    assert int(np.sum(row_counts == row_counts[line_r_inner])) == 1, "expected a unique horizontal dashed line"
    assert int(np.sum(col_counts == col_counts[line_c_inner])) == 1, "expected a unique vertical dashed line"

    line_r = BORDER + line_r_inner
    line_c = BORDER + line_c_inner

    top_color = int(g[0, w // 2])
    bottom_color = int(g[h - 1, w // 2])
    left_color = int(g[h // 2, 0])
    right_color = int(g[h // 2, w - 1])
    qcolors = _quadrant_colors(top_color, bottom_color, left_color, right_color)
    boxes = _quadrant_boxes(line_r, line_c, h, w)

    gray_mask = g == GRAY
    gray_comps = _connected_components(gray_mask)
    assert len(gray_comps) == 1, "expected exactly one gray prototype"
    prototype = gray_comps[0]

    source_name = None
    for name, box in boxes.items():
        r0, r1, c0, c1 = box
        if all(r0 <= r <= r1 and c0 <= c <= c1 for r, c in prototype):
            source_name = name
            break
    assert source_name is not None, "gray prototype must lie entirely within one quadrant"

    source_center_r2, source_center_c2 = _center2(boxes[source_name])
    offsets2 = [(2 * r - source_center_r2, 2 * c - source_center_c2) for r, c in prototype]

    out = g.copy()
    for name, box in boxes.items():
        center_r2, center_c2 = _center2(box)
        color = qcolors[name]
        for dr2, dc2 in offsets2:
            rr2 = center_r2 + dr2
            cc2 = center_c2 + dc2
            assert rr2 % 2 == 0 and cc2 % 2 == 0, "stamp coordinates must land on cells"
            rr = rr2 // 2
            cc = cc2 // 2
            r0, r1, c0, c1 = box
            assert r0 <= rr <= r1 and c0 <= cc <= c1, "stamp copy leaves its quadrant"
            out[rr, cc] = color
    return out


def _draw_border(grid: np.ndarray, top: int, bottom: int, left: int, right: int) -> None:
    grid[:BORDER, :] = top
    grid[-BORDER:, :] = bottom
    grid[BORDER:-BORDER, :BORDER] = left
    grid[BORDER:-BORDER, -BORDER:] = right


def _draw_dashed_cross(grid: np.ndarray, qh: int, qw: int, row_phase: int, col_phase: int) -> tuple[int, int]:
    line_r = BORDER + qh
    line_c = BORDER + qw
    inner_rows = range(BORDER, grid.shape[0] - BORDER)
    inner_cols = range(BORDER, grid.shape[1] - BORDER)

    for c in inner_cols:
        if (c - BORDER + row_phase) % 2 == 0:
            grid[line_r, c] = AZURE
    for r in inner_rows:
        if (r - BORDER + col_phase) % 2 == 0:
            grid[r, line_c] = AZURE
    return line_r, line_c


def _random_connected_mask(h: int, w: int) -> np.ndarray:
    area = h * w
    if area == 1:
        return np.ones((1, 1), dtype=bool)

    full_prob = 0.2
    if _RNG.random() < full_prob:
        target = area
    else:
        max_cells = min(area - 1 if area > 1 else 1, 8)
        target = int(_RNG.integers(1, max_cells + 1))

    start = (int(_RNG.integers(h)), int(_RNG.integers(w)))
    cells = {start}
    frontier = set()
    while len(cells) < target:
        cell_list = list(cells)
        cr, cc = cell_list[int(_RNG.integers(len(cell_list)))]
        options = []
        for dr, dc in _NEIGHBORS4:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                options.append((nr, nc))
        if options:
            pick = options[int(_RNG.integers(len(options)))]
            cells.add(pick)
            continue
        for cell in cells:
            for dr, dc in _NEIGHBORS4:
                nr, nc = cell[0] + dr, cell[1] + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        pick = tuple(frontier)[int(_RNG.integers(len(frontier)))]
        cells.add(pick)
        frontier.discard(pick)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _sample_local_stamp(qh: int, qw: int) -> np.ndarray:
    stamp_h = int(_RNG.integers(1, min(6, qh - 1)))
    stamp_w = int(_RNG.integers(1, min(6, qw - 1)))
    mask = _random_connected_mask(stamp_h, stamp_w)

    local = np.zeros((qh, qw), dtype=bool)
    row_lo = 1
    row_hi = qh - stamp_h - 1
    col_lo = 1
    col_hi = qw - stamp_w - 1
    top = int(_RNG.integers(row_lo, row_hi + 1))
    left = int(_RNG.integers(col_lo, col_hi + 1))
    local[top : top + stamp_h, left : left + stamp_w] = mask
    return local


def _render_case(
    qh: int,
    qw: int,
    border_colors: tuple[int, int, int, int],
    row_phase: int,
    col_phase: int,
    source_name: str,
    local_stamp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h = 2 * qh + 2 * BORDER + 1
    w = 2 * qw + 2 * BORDER + 1
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    top, bottom, left, right = border_colors
    _draw_border(inp, top, bottom, left, right)
    _draw_border(out, top, bottom, left, right)
    line_r, line_c = _draw_dashed_cross(inp, qh, qw, row_phase, col_phase)
    _draw_dashed_cross(out, qh, qw, row_phase, col_phase)

    boxes = _quadrant_boxes(line_r, line_c, h, w)
    qcolors = _quadrant_colors(top, bottom, left, right)

    local_coords = np.argwhere(local_stamp)
    for name, box in boxes.items():
        r0, _r1, c0, _c1 = box
        for lr, lc in local_coords:
            if name == source_name:
                inp[r0 + int(lr), c0 + int(lc)] = GRAY
            out[r0 + int(lr), c0 + int(lc)] = qcolors[name]
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    qh = int(_RNG.integers(4, 13))
    qw = int(_RNG.integers(4, 13))

    colors = [c for c in range(1, 10) if c not in {GRAY, AZURE}]
    border_colors = tuple(int(c) for c in _RNG.choice(colors, size=4, replace=False))
    row_phase = int(_RNG.integers(2))
    col_phase = int(_RNG.integers(2))
    source_name = ("TL", "TR", "BL", "BR")[int(_RNG.integers(4))]
    local_stamp = _sample_local_stamp(qh, qw)

    inp, expected = _render_case(qh, qw, border_colors, row_phase, col_phase, source_name, local_stamp)

    if inp.ndim != 2 or expected.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if not np.any(local_stamp):
        return None
    if len(_connected_components(local_stamp)) != 1:
        return None
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
