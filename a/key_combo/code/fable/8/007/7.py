import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _fingerprint(col: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in np.asarray(col, dtype=int).tolist())


def _column_groups(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    active = np.any(grid != int(bg), axis=0)
    groups: list[tuple[int, int]] = []
    c = 0
    while c < grid.shape[1]:
        if not active[c]:
            c += 1
            continue
        c0 = c
        while c + 1 < grid.shape[1] and active[c + 1]:
            c += 1
        groups.append((c0, c))
        c += 1
    return groups


def _extract_pieces(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    pieces: list[np.ndarray] = []
    for c0, c1 in _column_groups(grid, bg):
        rows = np.flatnonzero(np.any(grid[:, c0 : c1 + 1] != int(bg), axis=1))
        assert rows.size > 0, "each fragment must contain foreground"
        r0, r1 = int(rows[0]), int(rows[-1])
        pieces.append(np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy())
    assert pieces, "expected at least one fragment"
    heights = {piece.shape[0] for piece in pieces}
    assert len(heights) == 1, "all fragments must have the same height"
    return pieces


def _infer_order(pieces: list[np.ndarray]) -> list[int]:
    left_to_piece: dict[tuple[int, ...], list[int]] = {}
    right_to_piece: dict[tuple[int, ...], list[int]] = {}
    left_cols: list[tuple[int, ...]] = []
    right_cols: list[tuple[int, ...]] = []

    for idx, piece in enumerate(pieces):
        left_fp = _fingerprint(piece[:, 0])
        right_fp = _fingerprint(piece[:, -1])
        left_cols.append(left_fp)
        right_cols.append(right_fp)
        left_to_piece.setdefault(left_fp, []).append(idx)
        right_to_piece.setdefault(right_fp, []).append(idx)

    starts = [idx for idx, left_fp in enumerate(left_cols) if left_fp not in right_to_piece]
    assert len(starts) == 1, "expected a unique chain start"

    order: list[int] = []
    used: set[int] = set()
    cur = starts[0]
    while True:
        assert cur not in used, "chain must not contain a cycle"
        used.add(cur)
        order.append(cur)
        candidates = [idx for idx in left_to_piece.get(right_cols[cur], []) if idx != cur]
        if not candidates:
            break
        assert len(candidates) == 1, "expected a unique successor"
        cur = candidates[0]

    assert len(order) == len(pieces), "chain must consume every fragment exactly once"
    return order


def _merge_pieces(pieces: list[np.ndarray], order: list[int]) -> np.ndarray:
    out = pieces[order[0]].copy()
    for idx in order[1:]:
        piece = pieces[idx]
        assert out.shape[0] == piece.shape[0]
        assert np.array_equal(out[:, -1], piece[:, 0]), "adjacent seam columns must match"
        out = np.concatenate([out, piece[:, 1:]], axis=1)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder shuffled rectangular fragments by matching edge-column fingerprints, then merge them with one-column seam overlap.

    Input structure:
    - The canvas contains several equal-height rectangular colored fragments on a uniform background.
    - Each fragment occupies a contiguous run of non-background columns, and different fragments are separated by one or more all-background columns.
    - The left-to-right input order is scrambled, but the fragments are not rotated or recolored.

    Rule:
    - Treat each fragment's leftmost and rightmost columns as its edge fingerprints.
    - Find the unique first fragment whose left edge does not match any other fragment's right edge, then follow exact right-edge to left-edge matches to order the whole chain.
    - Reconstruct the full image by concatenating fragments in that inferred order while overlapping each matched seam column into a single shared column.

    Key insight:
    The edge-column pattern plays two roles at once: it determines the unique fragment order and also identifies the exact one-column overlap that must be merged away at every seam.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    pieces = _extract_pieces(g, bg)
    order = _infer_order(pieces)
    return _merge_pieces(pieces, order)


def _random_column(height: int, palette: tuple[int, ...]) -> np.ndarray:
    while True:
        runs = random.randint(2, min(4, height))
        cuts = sorted(random.sample(range(1, height), runs - 1))
        bounds = [0, *cuts, height]
        colors = [random.choice(palette)]
        for _ in range(runs - 1):
            options = [color for color in palette if color != colors[-1]]
            colors.append(random.choice(options))

        col = np.empty(height, dtype=int)
        for idx, color in enumerate(colors):
            col[bounds[idx] : bounds[idx + 1]] = int(color)
        if len(set(map(int, col.tolist()))) >= 2:
            return col


def _bridged_piece(left: np.ndarray, right: np.ndarray, width: int, palette: tuple[int, ...]) -> np.ndarray:
    height = left.shape[0]
    piece = np.empty((height, width), dtype=int)
    piece[:, 0] = left
    piece[:, -1] = right
    for j in range(1, width - 1):
        t = j / (width - 1)
        col = left.copy()
        segment_count = random.randint(1, min(3, height))
        for _ in range(segment_count):
            a = random.randint(0, height - 1)
            b = random.randint(a, height - 1)
            if random.random() < 0.65:
                col[a : b + 1] = right[a : b + 1]
            else:
                col[a : b + 1] = random.choice(palette)
        flip_rows = max(1, int(round(t * height / 2)))
        for row in random.sample(range(height), flip_rows):
            if random.random() < t:
                col[row] = right[row]
        if np.array_equal(col, left):
            row = random.randrange(height)
            col[row] = right[row] if right[row] != left[row] else random.choice(palette)
        piece[:, j] = col
    return piece


def _sample_widths(piece_count: int, out_width: int) -> list[int] | None:
    total = out_width + piece_count - 1
    widths = [2] * piece_count
    extras = total - 2 * piece_count
    for _ in range(extras):
        widths[random.randrange(piece_count)] += 1
    if max(widths) > 7:
        return None
    if len(set(widths)) < 2:
        return None
    if widths == sorted(widths) or widths == sorted(widths, reverse=True):
        return None
    return widths


def _build_output(height: int, widths: list[int], palette: tuple[int, ...]) -> tuple[np.ndarray, list[np.ndarray]]:
    boundary_count = len(widths) + 1
    boundaries: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    while len(boundaries) < boundary_count:
        col = _random_column(height, palette)
        fp = _fingerprint(col)
        if fp in seen:
            continue
        seen.add(fp)
        boundaries.append(col)

    pieces: list[np.ndarray] = []
    for idx, width in enumerate(widths):
        pieces.append(_bridged_piece(boundaries[idx], boundaries[idx + 1], width, palette))

    out = pieces[0].copy()
    for piece in pieces[1:]:
        assert np.array_equal(out[:, -1], piece[:, 0])
        out = np.concatenate([out, piece[:, 1:]], axis=1)

    return out, pieces


def _input_order_merge(pieces: list[np.ndarray]) -> np.ndarray:
    out = pieces[0].copy()
    for piece in pieces[1:]:
        out = np.concatenate([out, piece[:, 1:]], axis=1)
    return out


def _validate_instance(grid: np.ndarray, expected: np.ndarray, piece_count: int, perm: list[int]) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if get_bg(grid) != BG:
        return False

    pieces = _extract_pieces(grid, BG)
    if len(pieces) != piece_count:
        return False
    if any(np.any(piece == BG) for piece in pieces):
        return False

    order = _infer_order(pieces)
    rebuilt = _merge_pieces(pieces, order)
    if not np.array_equal(rebuilt, expected):
        return False
    if not np.array_equal(main(grid), expected):
        return False

    input_merge = _input_order_merge(pieces)
    if np.array_equal(input_merge, expected):
        return False

    correct_positions = [perm.index(idx) for idx in range(piece_count)]
    asc = sorted(range(piece_count), key=lambda idx: pieces[idx].shape[1])
    desc = sorted(range(piece_count), key=lambda idx: (-pieces[idx].shape[1], idx))
    if correct_positions == asc or correct_positions == desc:
        return False

    left_edges = [_fingerprint(piece[:, 0]) for piece in pieces]
    right_edges = [_fingerprint(piece[:, -1]) for piece in pieces]
    if len(set(left_edges + right_edges)) != piece_count + 1:
        return False

    if len(set(map(int, expected.ravel().tolist()))) < 3:
        return False
    if any(len(set(map(int, piece.ravel().tolist()))) < 2 for piece in pieces):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        piece_count = random.randint(3, 5)
        height = random.randint(4, 8)
        out_width = random.randint(max(10, piece_count + 5), 18)
        widths = _sample_widths(piece_count, out_width)
        if widths is None:
            continue

        palette = tuple(random.sample(PALETTE, random.randint(3, 6)))
        expected, ordered_pieces = _build_output(height, widths, palette)
        if expected.shape[1] != out_width:
            continue

        perm = list(range(piece_count))
        random.shuffle(perm)
        if perm == list(range(piece_count)):
            perm[0], perm[1] = perm[1], perm[0]

        input_pieces = [ordered_pieces[idx] for idx in perm]
        sep_widths = [random.randint(1, 3) for _ in range(piece_count - 1)]
        top = random.randint(0, 3)
        bottom = random.randint(0, 3)
        left = random.randint(0, 3)
        right = random.randint(0, 3)
        canvas_h = top + height + bottom
        canvas_w = left + sum(piece.shape[1] for piece in input_pieces) + sum(sep_widths) + right
        if canvas_h > 30 or canvas_w > 30:
            continue

        grid = np.full((canvas_h, canvas_w), BG, dtype=int)
        c = left
        for idx, piece in enumerate(input_pieces):
            grid[top : top + height, c : c + piece.shape[1]] = piece
            c += piece.shape[1]
            if idx < len(sep_widths):
                c += sep_widths[idx]

        if _validate_instance(grid, expected, piece_count, perm):
            return grid

    raise RuntimeError("failed to generate a valid seam-fingerprint instance")
