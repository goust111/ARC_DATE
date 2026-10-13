import random
from dataclasses import dataclass

import numpy as np


BLACK = 0
ORANGE = 7
AZURE = 8
INTERFACE = {ORANGE, AZURE}
BASE_COLORS = [1, 2, 3, 4, 5, 6, 9]
GEN_RETRIES = 400


@dataclass(frozen=True)
class Piece:
    data: np.ndarray
    bbox: tuple[int, int, int, int]

    @property
    def h(self) -> int:
        return int(self.data.shape[0])

    @property
    def w(self) -> int:
        return int(self.data.shape[1])

    @property
    def top(self) -> np.ndarray:
        return self.data[0, :]

    @property
    def bottom(self) -> np.ndarray:
        return self.data[-1, :]

    @property
    def left(self) -> np.ndarray:
        return self.data[:, 0]

    @property
    def right(self) -> np.ndarray:
        return self.data[:, -1]


def _component_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in pts]
            cols = [p[1] for p in pts]
            boxes.append((min(rows), min(cols), max(rows), max(cols)))
    return boxes


def _extract_pieces(grid: np.ndarray) -> list[Piece]:
    g = np.asarray(grid, dtype=int)
    boxes = _component_bboxes(g != BLACK)
    pieces: list[Piece] = []
    for r0, c0, r1, c1 in boxes:
        sub = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        pieces.append(Piece(sub, (r0, c0, r1, c1)))
    pieces.sort(key=lambda p: (p.bbox[0], p.bbox[1]))
    return pieces


def _has_interface(edge: np.ndarray) -> bool:
    return any(int(v) in INTERFACE for v in edge)


def _edge_score(first: np.ndarray, second: np.ndarray) -> int:
    if len(first) != len(second):
        return -1
    visible_matches = 0
    interface_count = 0
    for a, b in zip(first, second):
        a = int(a)
        b = int(b)
        a_special = a in INTERFACE
        b_special = b in INTERFACE

        # The generator uses directional coding: left/top pieces expose azure,
        # right/bottom pieces expose orange.
        if a == ORANGE or b == AZURE:
            return -1
        if not a_special and not b_special:
            if a != b:
                return -1
            visible_matches += 1
        else:
            interface_count += 1
    if interface_count == 0:
        return -1
    return 10 * visible_matches + interface_count


def _search_layout(pieces: list[Piece]) -> list[list[int]]:
    n = len(pieces)
    right_scores = [[-1] * n for _ in range(n)]
    down_scores = [[-1] * n for _ in range(n)]
    for i, pi in enumerate(pieces):
        for j, pj in enumerate(pieces):
            if i == j:
                continue
            right_scores[i][j] = _edge_score(pi.right, pj.left)
            down_scores[i][j] = _edge_score(pi.bottom, pj.top)

    best_score = None
    best_layout = None

    for rows in range(1, n + 1):
        if n % rows != 0:
            continue
        cols = n // rows
        order = [-1] * n
        used = [False] * n

        def border_ok(piece_idx: int, r: int, c: int) -> bool:
            piece = pieces[piece_idx]
            if r == 0 and _has_interface(piece.top):
                return False
            if r == rows - 1 and _has_interface(piece.bottom):
                return False
            if c == 0 and _has_interface(piece.left):
                return False
            if c == cols - 1 and _has_interface(piece.right):
                return False
            return True

        def future_ok(piece_idx: int, r: int, c: int) -> bool:
            if c < cols - 1:
                if not any((not used[j]) and right_scores[piece_idx][j] > 0 for j in range(n)):
                    return False
            if r < rows - 1:
                if not any((not used[j]) and down_scores[piece_idx][j] > 0 for j in range(n)):
                    return False
            return True

        def backtrack(pos: int, score: int) -> None:
            nonlocal best_score, best_layout
            if pos == n:
                layout = [order[r * cols : (r + 1) * cols] for r in range(rows)]
                if best_score is None or score > best_score:
                    best_score = score
                    best_layout = layout
                return

            r, c = divmod(pos, cols)
            candidates = []
            for piece_idx in range(n):
                if used[piece_idx] or not border_ok(piece_idx, r, c):
                    continue
                add = 0
                if c > 0:
                    left_idx = order[pos - 1]
                    edge = right_scores[left_idx][piece_idx]
                    if edge <= 0:
                        continue
                    add += edge
                if r > 0:
                    up_idx = order[pos - cols]
                    edge = down_scores[up_idx][piece_idx]
                    if edge <= 0:
                        continue
                    add += edge
                if not future_ok(piece_idx, r, c):
                    continue
                candidates.append((-(add), piece_idx))
            candidates.sort()

            for _neg_add, piece_idx in candidates:
                add = 0
                if c > 0:
                    add += right_scores[order[pos - 1]][piece_idx]
                if r > 0:
                    add += down_scores[order[pos - cols]][piece_idx]
                order[pos] = piece_idx
                used[piece_idx] = True
                backtrack(pos + 1, score + add)
                used[piece_idx] = False
                order[pos] = -1

        backtrack(0, 0)

    if best_layout is None:
        raise ValueError("failed to infer piece layout")
    return best_layout


def _merge_value(a: int, b: int) -> int:
    a_special = a in INTERFACE
    b_special = b in INTERFACE
    if not a_special and not b_special:
        if a != b:
            raise ValueError("conflicting visible colors while merging")
        return a
    if a_special and b_special:
        return ORANGE
    return b if not b_special else a


def _assemble(layout: list[list[int]], pieces: list[Piece]) -> np.ndarray:
    rows = len(layout)
    cols = len(layout[0])

    row_heights = []
    for r in range(rows):
        vals = {pieces[idx].h for idx in layout[r]}
        if len(vals) != 1:
            raise ValueError("row height mismatch")
        row_heights.append(vals.pop())

    col_widths = []
    for c in range(cols):
        vals = {pieces[layout[r][c]].w for r in range(rows)}
        if len(vals) != 1:
            raise ValueError("column width mismatch")
        col_widths.append(vals.pop())

    out_h = sum(row_heights) - (rows - 1)
    out_w = sum(col_widths) - (cols - 1)
    out = np.full((out_h, out_w), -1, dtype=int)

    row_offsets = []
    acc = 0
    for h in row_heights:
        row_offsets.append(acc)
        acc += h - 1

    col_offsets = []
    acc = 0
    for w in col_widths:
        col_offsets.append(acc)
        acc += w - 1

    for r in range(rows):
        for c in range(cols):
            piece = pieces[layout[r][c]].data
            rr0 = row_offsets[r]
            cc0 = col_offsets[c]
            for rr in range(piece.shape[0]):
                for cc in range(piece.shape[1]):
                    v = int(piece[rr, cc])
                    tr = rr0 + rr
                    tc = cc0 + cc
                    cur = int(out[tr, tc])
                    if cur == -1:
                        out[tr, tc] = v
                    else:
                        out[tr, tc] = _merge_value(cur, v)
    if np.any(out < 0):
        raise ValueError("incomplete assembly")
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    pieces = _extract_pieces(g)
    if not pieces:
        return g.copy()
    layout = _search_layout(pieces)
    return _assemble(layout, pieces)


def _ensure_palette_coverage(arr: np.ndarray, palette: list[int], rng: random.Random) -> None:
    cells = [(r, c) for r in range(arr.shape[0]) for c in range(arr.shape[1])]
    rng.shuffle(cells)
    for color, (r, c) in zip(palette, cells):
        arr[r, c] = color


def _make_full_pattern(h: int, w: int, rng: random.Random) -> np.ndarray:
    palette = rng.sample(BASE_COLORS, rng.randint(3, 5))
    mode = rng.choice(["noise", "row_bands", "col_bands", "blocks", "diagonal"])
    arr = np.zeros((h, w), dtype=int)

    if mode == "noise":
        for r in range(h):
            for c in range(w):
                arr[r, c] = rng.choice(palette)
    elif mode == "row_bands":
        for r in range(h):
            last = None
            c = 0
            while c < w:
                span = rng.randint(1, min(3, w - c))
                choices = [x for x in palette if x != last] or palette
                color = rng.choice(choices)
                arr[r, c : c + span] = color
                last = color
                c += span
    elif mode == "col_bands":
        for c in range(w):
            last = None
            r = 0
            while r < h:
                span = rng.randint(1, min(3, h - r))
                choices = [x for x in palette if x != last] or palette
                color = rng.choice(choices)
                arr[r : r + span, c] = color
                last = color
                r += span
    elif mode == "blocks":
        block_h = rng.randint(2, 3)
        block_w = rng.randint(2, 3)
        for r in range(h):
            for c in range(w):
                idx = ((r // block_h) + 2 * (c // block_w) + rng.randint(0, 2)) % len(palette)
                arr[r, c] = palette[idx]
    else:
        band = rng.randint(1, 2)
        flip = rng.random() < 0.5
        for r in range(h):
            for c in range(w):
                k = ((r + c) // band) if not flip else ((r + (w - 1 - c)) // band)
                arr[r, c] = palette[(k + rng.randint(0, len(palette) - 1)) % len(palette)]

    _ensure_palette_coverage(arr, palette, rng)
    return arr


def _sample_boundary_states(length: int, rng: random.Random, need_double: bool) -> list[int]:
    inner = list(range(1, length - 1))
    if not inner:
        return [0] * length
    while True:
        states = [0] * length
        counts = [0, 0, 0, 0]
        for idx in inner:
            state = rng.choices([0, 1, 2, 3], weights=[3, 2, 2, 1])[0]
            states[idx] = state
            counts[state] += 1
        if counts[1] == 0 or counts[2] == 0:
            continue
        if counts[1] + counts[2] + counts[3] < 2:
            continue
        if need_double and counts[3] == 0 and len(inner) >= 3:
            continue
        return states


def _make_pieces(rng: random.Random) -> tuple[list[np.ndarray], np.ndarray, tuple[int, int]]:
    rows, cols = rng.choice([(2, 2), (2, 3), (3, 2), (3, 3)])
    piece_h = rng.randint(4, 7)
    piece_w = rng.randint(4, 7)
    if piece_h == piece_w:
        piece_w = 4 if piece_h != 4 else 7

    full_h = rows * (piece_h - 1) + 1
    full_w = cols * (piece_w - 1) + 1
    full = _make_full_pattern(full_h, full_w, rng)
    expected = full.copy()

    pieces = [
        [
            full[r * (piece_h - 1) : r * (piece_h - 1) + piece_h, c * (piece_w - 1) : c * (piece_w - 1) + piece_w].copy()
            for c in range(cols)
        ]
        for r in range(rows)
    ]

    boundaries: list[tuple[str, int, int, list[int], int]] = []
    force_double = rng.random() < 0.7
    total_boundaries = rows * (cols - 1) + (rows - 1) * cols
    double_target = rng.randrange(total_boundaries) if force_double and total_boundaries > 0 else -1
    boundary_idx = 0

    for r in range(rows):
        for c in range(cols - 1):
            states = _sample_boundary_states(piece_h, rng, boundary_idx == double_target)
            boundaries.append(("v", r, c, states, boundary_idx))
            boundary_idx += 1

    for r in range(rows - 1):
        for c in range(cols):
            states = _sample_boundary_states(piece_w, rng, boundary_idx == double_target)
            boundaries.append(("h", r, c, states, boundary_idx))
            boundary_idx += 1

    for kind, r, c, states, _idx in boundaries:
        if kind == "v":
            left = pieces[r][c]
            right = pieces[r][c + 1]
            out_col = (c + 1) * (piece_w - 1)
            row0 = r * (piece_h - 1)
            for i, state in enumerate(states):
                if state in (1, 3):
                    left[i, -1] = AZURE
                if state in (2, 3):
                    right[i, 0] = ORANGE
                if state == 3:
                    expected[row0 + i, out_col] = ORANGE
        else:
            top = pieces[r][c]
            bottom = pieces[r + 1][c]
            out_row = (r + 1) * (piece_h - 1)
            col0 = c * (piece_w - 1)
            for i, state in enumerate(states):
                if state in (1, 3):
                    top[-1, i] = AZURE
                if state in (2, 3):
                    bottom[0, i] = ORANGE
                if state == 3:
                    expected[out_row, col0 + i] = ORANGE

    flat_pieces = [pieces[r][c] for r in range(rows) for c in range(cols)]
    return flat_pieces, expected, (piece_h, piece_w)


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = max(br0 - ar1 - 1, ar0 - br1 - 1, 0)
    col_gap = max(bc0 - ac1 - 1, ac0 - bc1 - 1, 0)
    return row_gap >= 2 or col_gap >= 2


def _scatter_pieces(pieces: list[np.ndarray], piece_shape: tuple[int, int], rng: random.Random) -> np.ndarray | None:
    ph, pw = piece_shape
    n = len(pieces)
    canvas_h = max(18, int((ph + 3) * np.ceil(np.sqrt(n))) + rng.randint(2, 8))
    canvas_w = max(18, int((pw + 3) * np.ceil(np.sqrt(n))) + rng.randint(2, 8))
    grid = np.full((canvas_h, canvas_w), BLACK, dtype=int)
    placed: list[tuple[int, int, int, int]] = []

    order = list(range(n))
    rng.shuffle(order)
    for idx in order:
        piece = pieces[idx]
        done = False
        for _ in range(300):
            r0 = rng.randint(1, canvas_h - ph - 1)
            c0 = rng.randint(1, canvas_w - pw - 1)
            bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if all(_separated(bbox, other) for other in placed):
                grid[r0 : r0 + ph, c0 : c0 + pw] = piece
                placed.append(bbox)
                done = True
                break
        if not done:
            return None
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pieces, expected, piece_shape = _make_pieces(rng)
    inp = _scatter_pieces(pieces, piece_shape, rng)
    if inp is None:
        return None
    try:
        out = main(inp)
    except Exception:
        return None
    if np.array_equal(out, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
