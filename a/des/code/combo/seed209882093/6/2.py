from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            cells = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _template_info(grid: np.ndarray) -> tuple[dict[int, tuple[int, int]], tuple[int, int, int, int], int, int]:
    assert grid[0, 0] == GREY, "template must start at the top-left corner"
    grey_comps = _components(grid == GREY)
    template_comp = None
    for comp in grey_comps:
        if np.any((comp[:, 0] == 0) & (comp[:, 1] == 0)):
            template_comp = comp
            break
    assert template_comp is not None, "expected a grey template grid at the top-left"

    r0, c0, r1, c1 = _bbox(template_comp)
    assert (r0, c0) == (0, 0)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = sub.shape
    assert h >= 3 and w >= 3 and h % 2 == 1 and w % 2 == 1

    grey_rows = [r for r in range(h) if np.all(sub[r, :] == GREY)]
    grey_cols = [c for c in range(w) if np.all(sub[:, c] == GREY)]
    assert grey_rows == list(range(0, h, 2))
    assert grey_cols == list(range(0, w, 2))

    cell_rows = (h - 1) // 2
    cell_cols = (w - 1) // 2
    color_to_pos: dict[int, tuple[int, int]] = {}
    for rr in range(cell_rows):
        for cc in range(cell_cols):
            vr = 2 * rr + 1
            vc = 2 * cc + 1
            cell_color = int(sub[vr, vc])
            assert cell_color not in {BLACK, GREY}
            assert cell_color not in color_to_pos
            color_to_pos[cell_color] = (rr, cc)

    assert len(color_to_pos) == cell_rows * cell_cols
    return color_to_pos, (r0, c0, r1, c1), cell_rows, cell_cols


def _extract_pieces(
    grid: np.ndarray,
    template_bbox: tuple[int, int, int, int],
    color_to_pos: dict[int, tuple[int, int]],
) -> tuple[list[tuple[tuple[int, int], np.ndarray]], int, int]:
    mask = (grid != BLACK) & (grid != GREY)
    r0, c0, r1, c1 = template_bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    comps = _components(mask)
    assert comps, "expected at least one fragment"

    pieces: list[tuple[tuple[int, int], np.ndarray]] = []
    piece_shape = None
    seen_positions: set[tuple[int, int]] = set()

    for comp in comps:
        br0, bc0, br1, bc1 = _bbox(comp)
        piece = grid[br0 : br1 + 1, bc0 : bc1 + 1].copy()
        assert np.all(piece != BLACK) and np.all(piece != GREY), "fragments must be solid rectangles"

        if piece_shape is None:
            piece_shape = piece.shape
        assert piece.shape == piece_shape

        anchor = int(piece[0, 0])
        assert anchor in color_to_pos
        pos = color_to_pos[anchor]
        assert pos not in seen_positions, "fragment addresses must be unique"
        seen_positions.add(pos)
        pieces.append((pos, piece))

    assert piece_shape is not None
    return pieces, int(piece_shape[0]), int(piece_shape[1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble same-size fragments into a compact mosaic using a top-left grey template grid.

    The template's unique cell colors encode the row/column destination for each fragment:
    a fragment's top-left pixel color selects its slot in the final packed arrangement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    color_to_pos, template_bbox, n_rows, n_cols = _template_info(g)
    pieces, piece_h, piece_w = _extract_pieces(g, template_bbox, color_to_pos)
    assert len(pieces) == len(color_to_pos)

    out = np.full((n_rows * piece_h, n_cols * piece_w), BLACK, dtype=int)
    for (rr, cc), piece in pieces:
        out[rr * piece_h : (rr + 1) * piece_h, cc * piece_w : (cc + 1) * piece_w] = piece
    return out


def _sample_template_shape() -> tuple[int, int]:
    options = [
        (rows, cols)
        for rows in range(1, 5)
        for cols in range(1, 5)
        if 2 <= rows * cols <= len(PALETTE)
    ]
    return random.choice(options)


def _make_template(colors: np.ndarray) -> np.ndarray:
    rows, cols = colors.shape
    out = np.full((2 * rows + 1, 2 * cols + 1), GREY, dtype=int)
    out[1::2, 1::2] = colors
    return out


def _make_piece(anchor: int, h: int, w: int) -> np.ndarray:
    mode = random.choice(("rows", "cols", "frame", "checker", "diag", "noise"))
    palette_size = random.randint(2, min(5, len(PALETTE)))
    colors = random.sample(PALETTE, palette_size)
    if anchor not in colors:
        colors[0] = anchor

    piece = np.empty((h, w), dtype=int)
    if mode == "rows":
        row_colors = [random.choice(colors) for _ in range(h)]
        for r in range(h):
            piece[r, :] = row_colors[r]
    elif mode == "cols":
        col_colors = [random.choice(colors) for _ in range(w)]
        for c in range(w):
            piece[:, c] = col_colors[c]
    elif mode == "frame":
        border = random.choice(colors)
        inner = random.choice(colors)
        piece[:, :] = border
        if h > 2 and w > 2:
            piece[1:-1, 1:-1] = inner
        if h > 3 and w > 3 and random.random() < 0.5:
            piece[2:-2, 2:-2] = random.choice(colors)
    elif mode == "checker":
        a, b = random.sample(colors, 2)
        rr, cc = np.indices((h, w))
        piece[:, :] = np.where((rr + cc) % 2 == 0, a, b)
    elif mode == "diag":
        band_colors = [random.choice(colors) for _ in range(h + w - 1)]
        for r in range(h):
            for c in range(w):
                piece[r, c] = band_colors[r + c]
    else:
        piece[:, :] = np.random.choice(colors, size=(h, w))

    piece[0, 0] = anchor
    if np.all(piece == anchor):
        piece[-1, -1] = random.choice([color for color in PALETTE if color != anchor])
    return piece


def _reserve(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(blocked.shape[0], r0 + h + pad)
    cc1 = min(blocked.shape[1], c0 + w + pad)
    if blocked[rr0:rr1, cc0:cc1].any():
        return False
    blocked[rr0:rr1, cc0:cc1] = True
    return True


def _sample_canvas_size(template: np.ndarray, piece_count: int, piece_h: int, piece_w: int) -> tuple[int, int]:
    occupied = template.size + piece_count * piece_h * piece_w
    min_side = max(
        template.shape[0] + 2,
        template.shape[1] + 2,
        piece_h + 2,
        piece_w + 2,
        int(math.ceil(math.sqrt(2 * occupied))),
    )
    max_side = max(min_side, 30)
    return random.randint(min_side, max_side), random.randint(min_side, max_side)


def _place_fragments(
    grid: np.ndarray,
    template: np.ndarray,
    pieces: list[np.ndarray],
) -> bool:
    th, tw = template.shape
    grid[:th, :tw] = template
    blocked = np.zeros_like(grid, dtype=bool)
    if not _reserve(blocked, 0, 0, th, tw, pad=1):
        return False

    order = list(range(len(pieces)))
    random.shuffle(order)
    for idx in order:
        piece = pieces[idx]
        ph, pw = piece.shape
        for _ in range(300):
            r0 = random.randint(0, grid.shape[0] - ph)
            c0 = random.randint(0, grid.shape[1] - pw)
            if not _reserve(blocked, r0, c0, ph, pw, pad=1):
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw] = piece
            break
        else:
            return False
    return True


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    rows, cols = _sample_template_shape()
    anchors = random.sample(PALETTE, rows * cols)
    template_colors = np.asarray(anchors, dtype=int).reshape(rows, cols)
    template = _make_template(template_colors)

    piece_h = random.randint(2, 5)
    piece_w = random.randint(2, 5)

    pieces: list[np.ndarray] = []
    expected = np.full((rows * piece_h, cols * piece_w), BLACK, dtype=int)
    for rr in range(rows):
        for cc in range(cols):
            anchor = int(template_colors[rr, cc])
            piece = _make_piece(anchor, piece_h, piece_w)
            pieces.append(piece)
            expected[rr * piece_h : (rr + 1) * piece_h, cc * piece_w : (cc + 1) * piece_w] = piece

    canvas_h, canvas_w = _sample_canvas_size(template, len(pieces), piece_h, piece_w)
    grid = np.full((canvas_h, canvas_w), BLACK, dtype=int)
    if not _place_fragments(grid, template, pieces):
        return None

    if np.count_nonzero(grid == BLACK) <= np.count_nonzero(grid != BLACK):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
