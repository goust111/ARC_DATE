import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _normalized(cells):
    if not cells:
        return frozenset()
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _border_holes(panel, color):
    coords = np.argwhere(panel == color)
    if coords.size == 0:
        return None
    top, left = coords.min(axis=0)
    bottom, right = coords.max(axis=0)
    if bottom - top < 2 or right - left < 2:
        return None

    holes = []
    colored_border = 0
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            if r in (top, bottom) or c in (left, right):
                if panel[r, c] == color:
                    colored_border += 1
                elif panel[r, c] == BLACK:
                    holes.append((r, c))
                else:
                    return None

    if colored_border == 0 or not holes:
        return None
    hole_rows = {r for r, _ in holes}
    hole_cols = {c for _, c in holes}
    on_one_edge = (
        len(hole_rows) == 1
        and next(iter(hole_rows)) in (top, bottom)
        or len(hole_cols) == 1
        and next(iter(hole_cols)) in (left, right)
    )
    if not on_one_edge:
        return None
    return (int(top), int(left), int(bottom), int(right), holes)


def main(grid: np.ndarray) -> np.ndarray:
    """Fill left-panel rectangle frames whose same-colored right-panel gap pieces are present.

    Input structure: The grid has a black background, a full-height grey vertical divider,
    incomplete single-color rectangular frames on the left, and small 4-connected
    same-color pieces on the right.  Each frame has one continuous gap on one border.

    Rule: Remove the divider and right panel.  For each left frame, compare its missing
    border cells with the normalized shape of any right-side piece of the same color.
    If a matching piece exists, repair the border and fill the whole rectangle with
    that frame color; otherwise leave that frame exactly as it was.

    Key insight: Color gives the pairing, but the decisive condition is structural: a
    frame is filled only when the matching piece completes its missing border segment.
    """
    grid = np.asarray(grid)
    grey_cols = np.where(np.all(grid == GREY, axis=0))[0]
    if len(grey_cols) == 0:
        counts = np.sum(grid == GREY, axis=0)
        if counts.max() == 0:
            return grid.copy()
        sep = int(np.argmax(counts))
    else:
        sep = int(grey_cols[0])

    left_panel = grid[:, :sep]
    right_panel = grid[:, sep + 1 :]
    out = left_panel.copy()

    colors = [int(c) for c in np.unique(left_panel) if c not in (BLACK, GREY)]
    right_shapes = {}
    for color in colors:
        comps = _components(right_panel == color)
        right_shapes[color] = {_normalized(comp) for comp in comps}

    for color in colors:
        frame = _border_holes(left_panel, color)
        if frame is None:
            continue
        top, left, bottom, right, holes = frame
        if _normalized(holes) in right_shapes.get(color, set()):
            out[top : bottom + 1, left : right + 1] = color

    return out


def _draw_frame(grid, top, left, height, width, color):
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _gap_cells(top, left, height, width, side, start, length):
    bottom = top + height - 1
    right = left + width - 1
    if side == 0:
        return [(top, left + start + i) for i in range(length)]
    if side == 1:
        return [(bottom, left + start + i) for i in range(length)]
    if side == 2:
        return [(top + start + i, left) for i in range(length)]
    return [(top + start + i, right) for i in range(length)]


def _place_piece(grid, occupied, rel_shape, color, sep, right_w, rng):
    shape_h = max(r for r, _ in rel_shape) + 1
    shape_w = max(c for _, c in rel_shape) + 1
    h = grid.shape[0]
    if h - shape_h < 2 or right_w - shape_w < 2:
        return False

    for _ in range(200):
        row = int(rng.integers(1, h - shape_h))
        col = int(rng.integers(1, right_w - shape_w))
        cells = [(row + r, col + c) for r, c in rel_shape]
        clear = True
        for rr, cc in cells:
            for nr in range(max(0, rr - 1), min(h, rr + 2)):
                for nc in range(max(0, cc - 1), min(right_w, cc + 2)):
                    if occupied[nr, nc]:
                        clear = False
                        break
                if not clear:
                    break
            if not clear:
                break
        if not clear:
            continue
        for rr, cc in cells:
            occupied[rr, cc] = True
            grid[rr, sep + 1 + cc] = color
        return True
    return False


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(1000):
        n_frames = int(rng.integers(2, 5))
        heights = [int(rng.integers(5, 8)) for _ in range(n_frames)]
        gaps = [int(rng.integers(1, 3)) for _ in range(n_frames - 1)]
        top_margin = int(rng.integers(1, 3))
        bottom_margin = int(rng.integers(1, 3))
        total_h = top_margin + bottom_margin + sum(heights) + sum(gaps)
        if total_h > 30:
            continue

        left_w = int(rng.integers(11, 18))
        max_right_w = 30 - left_w - 1
        if max_right_w < 6:
            continue
        right_w = int(rng.integers(6, max_right_w + 1))
        width = left_w + 1 + right_w
        grid = np.zeros((total_h, width), dtype=int)
        sep = left_w
        grid[:, sep] = GREY

        colors = list(rng.choice(COLORS, size=n_frames, replace=False))
        missing_index = int(rng.integers(0, n_frames)) if rng.random() < 0.5 else -1
        pieces = []

        row = top_margin
        ok = True
        for i in range(n_frames):
            height = heights[i]
            max_frame_w = min(left_w - 2, 9)
            if max_frame_w < 5:
                ok = False
                break
            frame_w = int(rng.integers(5, max_frame_w + 1))
            col = int(rng.integers(1, left_w - frame_w))
            color = int(colors[i])
            _draw_frame(grid, row, col, height, frame_w, color)

            side = int(rng.integers(0, 4))
            span = frame_w - 2 if side in (0, 1) else height - 2
            gap_len = int(rng.integers(1, min(3, span) + 1))
            start = int(rng.integers(1, span - gap_len + 2))
            holes = _gap_cells(row, col, height, frame_w, side, start, gap_len)
            for rr, cc in holes:
                grid[rr, cc] = BLACK
            if i != missing_index:
                pieces.append((color, _normalized(holes)))

            row += height
            if i < len(gaps):
                row += gaps[i]

        if not ok:
            continue

        occupied = np.zeros((total_h, right_w), dtype=bool)
        rng.shuffle(pieces)
        for color, rel_shape in pieces:
            if not _place_piece(grid, occupied, rel_shape, color, sep, right_w, rng):
                ok = False
                break
        if ok:
            return grid

    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0, 2, 2, 0, 0, 0, 5, 0, 2, 2, 0, 3, 0],
            [0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 5, 0, 0, 0, 0, 3, 0],
            [0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
            [0, 3, 3, 3, 3, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
            [0, 3, 3, 0, 3, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )
