from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BG = 0
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
LAYOUT_CHOICES = ((1, 4), (4, 1), (2, 2), (2, 3), (3, 2))
GEN_RETRIES = 400


@dataclass(frozen=True)
class Side:
    kind: str
    pos: int = -1


@dataclass
class Piece:
    color: int
    mask: np.ndarray
    sides: dict[str, Side]
    top_pad: int
    left_pad: int


def _side(kind: str, pos: int = -1) -> Side:
    return Side(kind, int(pos))


def _is_flat(side: Side) -> bool:
    return side.kind == "flat"


def _complements(a: Side, b: Side) -> bool:
    return (
        a.kind in {"tab", "slot"}
        and b.kind in {"tab", "slot"}
        and a.kind != b.kind
        and a.pos == b.pos
    )


def _components_by_color(g: np.ndarray) -> list[tuple[int, np.ndarray]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    pieces: list[tuple[int, np.ndarray]] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            pieces.append((color, np.asarray(coords, dtype=int)))
    return pieces


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    start_r, start_c = map(int, pts[0])
    seen = np.zeros((h, w), dtype=bool)
    seen[start_r, start_c] = True
    stack = [(start_r, start_c)]
    count = 0
    while stack:
        rr, cc = stack.pop()
        count += 1
        for dr, dc in DIRS4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return count == int(mask.sum())


def _crop_mask(coords: np.ndarray) -> np.ndarray:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    c0 = int(cols.min())
    h = int(rows.max() - r0 + 1)
    w = int(cols.max() - c0 + 1)
    mask = np.zeros((h, w), dtype=bool)
    mask[rows - r0, cols - c0] = True
    return mask


def _infer_base_dims(masks: list[np.ndarray]) -> tuple[int, int]:
    row_values = [int(v) for mask in masks for v in mask.sum(axis=1) if int(v) > 1]
    col_values = [int(v) for mask in masks for v in mask.sum(axis=0) if int(v) > 1]
    assert row_values and col_values

    def mode(values: list[int]) -> int:
        uniq, cnts = np.unique(np.asarray(values, dtype=int), return_counts=True)
        best = np.flatnonzero(cnts == cnts.max())
        return int(uniq[best[-1]])

    base_h = mode(col_values)
    base_w = mode(row_values)
    assert base_h >= 3 and base_w >= 3, "base rectangle must support interior tab positions"
    return base_h, base_w


def _slot_or_flat(line: np.ndarray) -> Side:
    missing = np.flatnonzero(~line)
    if missing.size == 0:
        return _side("flat")
    assert missing.size == 1, "each side may have at most one indentation"
    pos = int(missing[0])
    assert 0 < pos < line.size - 1, "indentation must not use a corner cell"
    return _side("slot", pos)


def _parse_piece(color: int, mask: np.ndarray, base_h: int, base_w: int) -> Piece:
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)
    top_pad = int(row_counts[0] == 1)
    bottom_pad = int(row_counts[-1] == 1)
    left_pad = int(col_counts[0] == 1)
    right_pad = int(col_counts[-1] == 1)

    assert mask.shape == (base_h + top_pad + bottom_pad, base_w + left_pad + right_pad)
    main = mask[top_pad : top_pad + base_h, left_pad : left_pad + base_w]
    assert main.shape == (base_h, base_w)

    sides: dict[str, Side] = {}

    if top_pad:
        cols = np.flatnonzero(mask[0])
        assert cols.size == 1
        sides["top"] = _side("tab", int(cols[0] - left_pad))
    else:
        sides["top"] = _slot_or_flat(main[0, :])

    if bottom_pad:
        cols = np.flatnonzero(mask[-1])
        assert cols.size == 1
        sides["bottom"] = _side("tab", int(cols[0] - left_pad))
    else:
        sides["bottom"] = _slot_or_flat(main[-1, :])

    if left_pad:
        rows = np.flatnonzero(mask[:, 0])
        assert rows.size == 1
        sides["left"] = _side("tab", int(rows[0] - top_pad))
    else:
        sides["left"] = _slot_or_flat(main[:, 0])

    if right_pad:
        rows = np.flatnonzero(mask[:, -1])
        assert rows.size == 1
        sides["right"] = _side("tab", int(rows[0] - top_pad))
    else:
        sides["right"] = _slot_or_flat(main[:, -1])

    for side_name, side in sides.items():
        if side.kind == "flat":
            continue
        limit = base_w if side_name in {"top", "bottom"} else base_h
        assert 0 < side.pos < limit - 1, "tab position must stay away from corners"

    return Piece(int(color), mask.copy(), sides, top_pad, left_pad)


def _solve_layout(pieces: list[Piece]) -> tuple[list[list[int]], int, int]:
    n = len(pieces)
    n_rows = sum(_is_flat(piece.sides["left"]) for piece in pieces)
    n_cols = sum(_is_flat(piece.sides["top"]) for piece in pieces)
    assert n_rows > 0 and n_cols > 0 and n_rows * n_cols == n
    assert sum(_is_flat(piece.sides["right"]) for piece in pieces) == n_rows
    assert sum(_is_flat(piece.sides["bottom"]) for piece in pieces) == n_cols

    order = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    placed = [[-1] * n_cols for _ in range(n_rows)]
    used = [False] * n
    solutions: list[list[list[int]]] = []

    def fits(idx: int, r: int, c: int) -> bool:
        piece = pieces[idx]
        top = piece.sides["top"]
        bottom = piece.sides["bottom"]
        left = piece.sides["left"]
        right = piece.sides["right"]

        if _is_flat(top) != (r == 0):
            return False
        if _is_flat(bottom) != (r == n_rows - 1):
            return False
        if _is_flat(left) != (c == 0):
            return False
        if _is_flat(right) != (c == n_cols - 1):
            return False

        if r > 0:
            above = pieces[placed[r - 1][c]]
            if not _complements(above.sides["bottom"], top):
                return False
        if c > 0:
            prev = pieces[placed[r][c - 1]]
            if not _complements(prev.sides["right"], left):
                return False
        return True

    def search(i: int) -> None:
        if len(solutions) > 1:
            return
        if i == n:
            solutions.append([row[:] for row in placed])
            return
        r, c = order[i]
        for idx in range(n):
            if used[idx] or not fits(idx, r, c):
                continue
            used[idx] = True
            placed[r][c] = idx
            search(i + 1)
            placed[r][c] = -1
            used[idx] = False

    search(0)
    assert len(solutions) == 1, "puzzle must assemble into a unique rectangle"
    return solutions[0], n_rows, n_cols


def _compose_output(layout: list[list[int]], pieces: list[Piece], base_h: int, base_w: int) -> np.ndarray:
    n_rows = len(layout)
    n_cols = len(layout[0])
    out = np.zeros((n_rows * base_h, n_cols * base_w), dtype=int)
    occupied = np.zeros_like(out, dtype=bool)

    for r, row in enumerate(layout):
        for c, idx in enumerate(row):
            piece = pieces[idx]
            start_r = r * base_h - piece.top_pad
            start_c = c * base_w - piece.left_pad
            for dr, dc in np.argwhere(piece.mask):
                rr = int(start_r + dr)
                cc = int(start_c + dc)
                assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]
                assert not occupied[rr, cc], "composed rectangle may not overlap conflicting pixels"
                occupied[rr, cc] = True
                out[rr, cc] = piece.color

    assert occupied.all(), "assembled output must be a full rectangle"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered interlocking jigsaw-like pieces into the unique rectangle.

    Each piece is a monochrome near-rectangle with optional 1-pixel tabs or slots
    on its sides. Tabs and slots on adjacent pieces are complementary, and the
    outer border pieces have flat exterior edges.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    raw_pieces = _components_by_color(g)
    assert raw_pieces, "expected at least one non-background piece"

    masks = [_crop_mask(coords) for _, coords in raw_pieces]
    base_h, base_w = _infer_base_dims(masks)
    pieces = [
        _parse_piece(color=color, mask=mask, base_h=base_h, base_w=base_w)
        for (color, _coords), mask in zip(raw_pieces, masks, strict=True)
    ]

    layout, _rows, _cols = _solve_layout(pieces)
    return _compose_output(layout, pieces, base_h, base_w)


def _blank_sides() -> dict[str, Side]:
    return {
        "top": _side("flat"),
        "bottom": _side("flat"),
        "left": _side("flat"),
        "right": _side("flat"),
    }


def _render_piece(color: int, sides: dict[str, Side], base_h: int, base_w: int) -> Piece:
    top_pad = int(sides["top"].kind == "tab")
    bottom_pad = int(sides["bottom"].kind == "tab")
    left_pad = int(sides["left"].kind == "tab")
    right_pad = int(sides["right"].kind == "tab")

    mask = np.zeros((base_h + top_pad + bottom_pad, base_w + left_pad + right_pad), dtype=bool)
    core_r0 = top_pad
    core_c0 = left_pad
    mask[core_r0 : core_r0 + base_h, core_c0 : core_c0 + base_w] = True

    if sides["top"].kind == "slot":
        mask[core_r0, core_c0 + sides["top"].pos] = False
    elif sides["top"].kind == "tab":
        mask[core_r0 - 1, core_c0 + sides["top"].pos] = True

    if sides["bottom"].kind == "slot":
        mask[core_r0 + base_h - 1, core_c0 + sides["bottom"].pos] = False
    elif sides["bottom"].kind == "tab":
        mask[core_r0 + base_h, core_c0 + sides["bottom"].pos] = True

    if sides["left"].kind == "slot":
        mask[core_r0 + sides["left"].pos, core_c0] = False
    elif sides["left"].kind == "tab":
        mask[core_r0 + sides["left"].pos, core_c0 - 1] = True

    if sides["right"].kind == "slot":
        mask[core_r0 + sides["right"].pos, core_c0 + base_w - 1] = False
    elif sides["right"].kind == "tab":
        mask[core_r0 + sides["right"].pos, core_c0 + base_w] = True

    assert _is_connected(mask), "generated piece must stay 4-connected"
    return Piece(int(color), mask, dict(sides), top_pad, left_pad)


def _sample_base_size(rows: int, cols: int) -> tuple[int, int]:
    needed_h = rows * max(cols - 1, 0)
    needed_w = cols * max(rows - 1, 0)
    base_h = random.randint(max(4, needed_h + 2), max(5, needed_h + 4))
    base_w = random.randint(max(4, needed_w + 2), max(5, needed_w + 4))
    return base_h, base_w


def _build_pieces(rows: int, cols: int, base_h: int, base_w: int) -> tuple[list[Piece], list[list[int]], np.ndarray]:
    n = rows * cols
    colors = random.sample(range(1, 10), n)

    side_specs = [[_blank_sides() for _ in range(cols)] for _ in range(rows)]

    horizontal_positions = random.sample(list(range(1, base_h - 1)), rows * max(cols - 1, 0))
    vertical_positions = random.sample(list(range(1, base_w - 1)), cols * max(rows - 1, 0))

    h_idx = 0
    for r in range(rows):
        for c in range(cols - 1):
            pos = horizontal_positions[h_idx]
            h_idx += 1
            if random.random() < 0.5:
                side_specs[r][c]["right"] = _side("tab", pos)
                side_specs[r][c + 1]["left"] = _side("slot", pos)
            else:
                side_specs[r][c]["right"] = _side("slot", pos)
                side_specs[r][c + 1]["left"] = _side("tab", pos)

    v_idx = 0
    for r in range(rows - 1):
        for c in range(cols):
            pos = vertical_positions[v_idx]
            v_idx += 1
            if random.random() < 0.5:
                side_specs[r][c]["bottom"] = _side("tab", pos)
                side_specs[r + 1][c]["top"] = _side("slot", pos)
            else:
                side_specs[r][c]["bottom"] = _side("slot", pos)
                side_specs[r + 1][c]["top"] = _side("tab", pos)

    layout = [[r * cols + c for c in range(cols)] for r in range(rows)]
    pieces: list[Piece] = []
    for r in range(rows):
        for c in range(cols):
            pieces.append(_render_piece(colors[layout[r][c]], side_specs[r][c], base_h, base_w))

    expected = _compose_output(layout, pieces, base_h, base_w)
    return pieces, layout, expected


def _scatter_pieces(pieces: list[Piece]) -> np.ndarray | None:
    max_h = max(piece.mask.shape[0] for piece in pieces)
    max_w = max(piece.mask.shape[1] for piece in pieces)
    total_bbox_area = sum(int(piece.mask.shape[0] * piece.mask.shape[1]) for piece in pieces)
    min_side = max(max_h + 2, max_w + 2, int(np.ceil((total_bbox_area * 2.2) ** 0.5)))

    for _ in range(80):
        h = random.randint(min_side, min(30, min_side + 8))
        w = random.randint(min_side, min(30, min_side + 10))
        grid = np.zeros((h, w), dtype=int)
        blocked = np.zeros((h, w), dtype=bool)
        order = pieces[:]
        random.shuffle(order)
        ok = True

        for piece in order:
            ph, pw = piece.mask.shape
            choices: list[tuple[int, int]] = []
            for r in range(1, h - ph):
                for c in range(1, w - pw):
                    r0 = max(0, r - 1)
                    c0 = max(0, c - 1)
                    r1 = min(h, r + ph + 1)
                    c1 = min(w, c + pw + 1)
                    if blocked[r0:r1, c0:c1].any():
                        continue
                    choices.append((r, c))
            if not choices:
                ok = False
                break
            r, c = random.choice(choices)
            window = grid[r : r + ph, c : c + pw]
            window[piece.mask] = piece.color
            blocked[max(0, r - 1) : min(h, r + ph + 1), max(0, c - 1) : min(w, c + pw + 1)] = True

        if ok:
            return grid
    return None


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    rows, cols = random.choice(LAYOUT_CHOICES)
    base_h, base_w = _sample_base_size(rows, cols)
    pieces, _layout, expected = _build_pieces(rows, cols, base_h, base_w)
    grid = _scatter_pieces(pieces)
    assert grid is not None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid, expected = _generate_case()
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_RETRIES} attempts")
