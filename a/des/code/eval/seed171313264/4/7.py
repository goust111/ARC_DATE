import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
AZURE = 8
MAROON = 9

BOARD_COLORS = (GREY, AZURE, RED, BLUE)
PIECE_COLORS = (RED, BLUE)

LEGEND_OUTER = 7
LEGEND_INNER = 5
BOARD_SIZES = (6, 7, 8)
GEN_RETRIES = 300
LAYOUT_RETRIES = 400

ATTACK_PATTERNS = (
    ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)),
    ((-1, 0), (0, -1), (0, 1), (1, 0)),
    ((-1, -1), (-1, 1), (1, -1), (1, 1)),
    ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)),
    ((-2, 0), (-1, 0), (0, -2), (0, -1), (0, 1), (0, 2), (1, 0), (2, 0)),
    ((-2, -2), (-1, -1), (-2, 2), (-1, 1), (1, -1), (2, -2), (1, 1), (2, 2)),
)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(pts)
    return parts


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _find_legend_offsets(grid: np.ndarray) -> tuple[tuple[int, int], ...]:
    candidates: list[tuple[tuple[int, int], ...]] = []
    for pts in _components(grid == MAROON):
        r0, c0, r1, c1 = _bbox(pts)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5 or h % 2 == 0 or w % 2 == 0:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(sub[border] == MAROON):
            continue
        if np.any(sub[1:-1, 1:-1] == MAROON):
            continue
        orange_pts = np.argwhere(sub == ORANGE)
        if orange_pts.shape[0] != 1:
            continue
        cr, cc = map(int, orange_pts[0])
        if (cr, cc) != (h // 2, w // 2):
            continue
        yellow_pts = np.argwhere(sub == YELLOW)
        offsets = sorted({(int(r - cr), int(c - cc)) for r, c in yellow_pts})
        if not offsets:
            continue
        candidates.append(tuple(offsets))
    assert len(candidates) == 1, f"expected exactly one legend, found {len(candidates)}"
    return candidates[0]


def _find_board_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    mask = np.isin(grid, BOARD_COLORS)
    for pts in _components(mask):
        r0, c0, r1, c1 = _bbox(pts)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != w or h not in BOARD_SIZES or len(pts) != h * w:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.all(np.isin(sub, BOARD_COLORS)):
            candidates.append((h * w, (r0, c0, r1, c1)))
    assert candidates, "board not found"
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_area = candidates[0][0]
    best = [bbox for area, bbox in candidates if area == best_area]
    assert len(best) == 1, "board detection is ambiguous"
    return best[0]


def _infer_base_board(board: np.ndarray) -> np.ndarray:
    rows, cols = np.indices(board.shape)
    parity = (rows + cols) % 2
    candidates: list[tuple[int, np.ndarray]] = []
    for top_left in (GREY, AZURE):
        other = AZURE if top_left == GREY else GREY
        expected = np.where(parity == 0, top_left, other).astype(int)
        score = int(np.count_nonzero((board == GREY) & (expected == GREY)))
        score += int(np.count_nonzero((board == AZURE) & (expected == AZURE)))
        candidates.append((score, expected))
    candidates.sort(key=lambda item: item[0], reverse=True)
    assert candidates[0][0] > candidates[1][0], "checkerboard orientation is ambiguous"
    return candidates[0][1]


def _threat_counts(shape: tuple[int, int], piece_positions: list[tuple[int, int]], offsets: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = shape
    counts = np.zeros((h, w), dtype=int)
    for pr, pc in piece_positions:
        for dr, dc in offsets:
            rr = pr + dr
            cc = pc + dc
            if 0 <= rr < h and 0 <= cc < w:
                counts[rr, cc] += 1
    return counts


def _recolor_board(board: np.ndarray, offsets: tuple[tuple[int, int], ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = _infer_base_board(board)
    piece_mask = np.isin(board, PIECE_COLORS)
    piece_positions = [tuple(map(int, pos)) for pos in np.argwhere(piece_mask)]
    counts = _threat_counts(board.shape, piece_positions, offsets)
    out = board.copy()
    empty = ~piece_mask
    out[empty & (counts == 0)] = GREEN
    out[empty & (counts == 1)] = base[empty & (counts == 1)]
    out[empty & (counts >= 2)] = YELLOW
    return out, counts, base


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a legend-defined attack pattern, stack all piece threats over a checkerboard,
    and recolor empty cells by threat multiplicity.
    """
    g = np.asarray(grid, dtype=int)
    offsets = _find_legend_offsets(g)
    r0, c0, r1, c1 = _find_board_bbox(g)
    out = g.copy()
    board_out, _, _ = _recolor_board(g[r0 : r1 + 1, c0 : c1 + 1], offsets)
    out[r0 : r1 + 1, c0 : c1 + 1] = board_out
    return out


def _make_checkerboard(size: int, top_left: int) -> np.ndarray:
    rows, cols = np.indices((size, size))
    other = AZURE if top_left == GREY else GREY
    return np.where((rows + cols) % 2 == 0, top_left, other).astype(int)


def _sample_layout(
    rng: random.Random, size: int, offsets: tuple[tuple[int, int], ...]
) -> tuple[list[tuple[tuple[int, int], int]], np.ndarray] | None:
    cells = [(r, c) for r in range(size) for c in range(size)]
    piece_count_choices = (4, 5, 6) if len(offsets) <= 4 else (3, 4, 5)
    for _ in range(LAYOUT_RETRIES):
        count = rng.choice(piece_count_choices)
        positions = rng.sample(cells, count)
        piece_mask = np.zeros((size, size), dtype=bool)
        for r, c in positions:
            piece_mask[r, c] = True
        threats = _threat_counts((size, size), positions, offsets)
        empty = ~piece_mask
        if not np.any(empty & (threats == 0)):
            continue
        if not np.any(empty & (threats == 1)):
            continue
        if not np.any(empty & (threats >= 2)):
            continue
        colors = [RED, BLUE]
        while len(colors) < count:
            colors.append(rng.choice(PIECE_COLORS))
        rng.shuffle(colors)
        layout = list(zip(positions, colors))
        return layout, threats
    return None


def _compose_legend(offsets: tuple[tuple[int, int], ...]) -> np.ndarray:
    legend = np.full((LEGEND_OUTER, LEGEND_OUTER), BG, dtype=int)
    legend[0, :] = MAROON
    legend[-1, :] = MAROON
    legend[:, 0] = MAROON
    legend[:, -1] = MAROON
    center = LEGEND_OUTER // 2
    legend[center, center] = ORANGE
    for dr, dc in offsets:
        legend[center + dr, center + dc] = YELLOW
    return legend


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    size = rng.choice(BOARD_SIZES)
    offsets = rng.choice(ATTACK_PATTERNS)
    base = _make_checkerboard(size, rng.choice((GREY, AZURE)))
    layout = _sample_layout(rng, size, offsets)
    if layout is None:
        return None
    pieces, _ = layout

    board = base.copy()
    for (r, c), color in pieces:
        board[r, c] = color

    board_out, threats, recovered_base = _recolor_board(board, offsets)
    empty = ~np.isin(board, PIECE_COLORS)
    if not np.any(empty & (threats == 0)):
        return None
    if not np.any(empty & (threats == 1)):
        return None
    if not np.any(empty & (threats >= 2)):
        return None
    if not np.any(empty & (board_out == GREEN)):
        return None
    if not np.any(empty & (board_out == YELLOW)):
        return None
    if not np.any(empty & (board_out == recovered_base)):
        return None

    left_margin = rng.randint(1, 3)
    gap = rng.randint(2, 4)
    right_margin = rng.randint(1, 4)
    width = left_margin + LEGEND_OUTER + gap + size + right_margin + rng.randint(0, 3)
    height = max(size, LEGEND_OUTER) + rng.randint(4, 8)

    board_top = _clamp((height - size) // 2 + rng.randint(-1, 1), 1, height - size - 1)
    legend_top = _clamp(
        (height - LEGEND_OUTER) // 2 + rng.randint(-2, 2),
        1,
        height - LEGEND_OUTER - 1,
    )
    board_left = left_margin + LEGEND_OUTER + gap
    legend_left = left_margin

    grid = np.full((height, width), BG, dtype=int)
    legend = _compose_legend(offsets)
    grid[legend_top : legend_top + LEGEND_OUTER, legend_left : legend_left + LEGEND_OUTER] = legend
    grid[board_top : board_top + size, board_left : board_left + size] = board

    expected = grid.copy()
    expected[board_top : board_top + size, board_left : board_left + size] = board_out
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
