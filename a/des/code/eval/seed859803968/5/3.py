from __future__ import annotations

import random

import numpy as np


BG = 8
PALETTE = [c for c in range(10) if c != BG]
GEN_RETRIES = 400
LAYOUT_OPTIONS = [(2, 2), (2, 3), (3, 2), (3, 3), (3, 3)]
BODY_SIZE_CHOICES = (3, 4, 5)
SCATTER_RETRIES = 40
POSITION_RETRIES = 200


def _dominant_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    winners = np.flatnonzero(counts == counts.max())
    assert winners.size == 1, "expected a unique dominant color"
    return int(winners[0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
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
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _apply_transform(arr: np.ndarray, code: int) -> np.ndarray:
    if code == 0:
        return arr.copy()
    if code == 1:
        return np.rot90(arr, 1).copy()
    if code == 2:
        return np.rot90(arr, 2).copy()
    if code == 3:
        return np.rot90(arr, 3).copy()
    if code == 4:
        return np.fliplr(arr).copy()
    if code == 5:
        return np.flipud(arr).copy()
    if code == 6:
        return arr.T.copy()
    if code == 7:
        return np.fliplr(np.rot90(arr, 1)).copy()
    raise ValueError(code)


def _render_piece_mask(
    body_h: int, body_w: int, sides: dict[str, tuple[str, int | None]]
) -> np.ndarray:
    top_tooth = sides["top"][0] == "tooth"
    bottom_tooth = sides["bottom"][0] == "tooth"
    left_tooth = sides["left"][0] == "tooth"
    right_tooth = sides["right"][0] == "tooth"

    body_r0 = 1 if top_tooth else 0
    body_c0 = 1 if left_tooth else 0
    h = body_h + int(top_tooth) + int(bottom_tooth)
    w = body_w + int(left_tooth) + int(right_tooth)
    mask = np.zeros((h, w), dtype=bool)
    mask[body_r0 : body_r0 + body_h, body_c0 : body_c0 + body_w] = True

    for side, (kind, offset) in sides.items():
        if kind == "smooth":
            continue
        assert offset is not None
        if kind == "slot":
            if side == "top":
                mask[body_r0, body_c0 + offset] = False
            elif side == "bottom":
                mask[body_r0 + body_h - 1, body_c0 + offset] = False
            elif side == "left":
                mask[body_r0 + offset, body_c0] = False
            else:
                mask[body_r0 + offset, body_c0 + body_w - 1] = False
        elif kind == "tooth":
            if side == "top":
                mask[body_r0 - 1, body_c0 + offset] = True
            elif side == "bottom":
                mask[body_r0 + body_h, body_c0 + offset] = True
            elif side == "left":
                mask[body_r0 + offset, body_c0 - 1] = True
            else:
                mask[body_r0 + offset, body_c0 + body_w] = True
        else:
            raise ValueError(kind)

    return mask


def _make_piece(
    color: int,
    body_h: int,
    body_w: int,
    sides: dict[str, tuple[str, int | None]],
    mask: np.ndarray | None = None,
) -> dict:
    piece_mask = _render_piece_mask(body_h, body_w, sides) if mask is None else mask.copy()
    return {
        "color": int(color),
        "mask": piece_mask.astype(bool, copy=False),
        "body_h": int(body_h),
        "body_w": int(body_w),
        "body_r0": 1 if sides["top"][0] == "tooth" else 0,
        "body_c0": 1 if sides["left"][0] == "tooth" else 0,
        "sides": {
            side: (kind, None if offset is None else int(offset))
            for side, (kind, offset) in sides.items()
        },
    }


def _analyze_edge(
    line: np.ndarray, strip: np.ndarray, has_tooth: bool
) -> tuple[str, int | None] | None:
    line = np.asarray(line, dtype=bool)
    strip = np.asarray(strip, dtype=bool)
    length = int(line.size)
    if length < 3:
        return None

    if has_tooth:
        if not np.all(line):
            return None
        tooth_pos = np.flatnonzero(strip)
        if tooth_pos.size != 1:
            return None
        offset = int(tooth_pos[0])
        if offset == 0 or offset == length - 1:
            return None
        return "tooth", offset

    if strip.any():
        return None
    missing = np.flatnonzero(~line)
    if missing.size == 0:
        return "smooth", None
    if missing.size == 1 and int(missing[0]) not in (0, length - 1) and np.count_nonzero(line) == length - 1:
        return "slot", int(missing[0])
    return None


def _infer_piece(mask: np.ndarray, color: int) -> dict:
    h, w = mask.shape
    solutions = []
    for top_tooth in (0, 1):
        for bottom_tooth in (0, 1):
            for left_tooth in (0, 1):
                for right_tooth in (0, 1):
                    body_r0 = top_tooth
                    body_c0 = left_tooth
                    body_h = h - top_tooth - bottom_tooth
                    body_w = w - left_tooth - right_tooth
                    if body_h < 3 or body_w < 3:
                        continue
                    body_r1 = body_r0 + body_h - 1
                    body_c1 = body_c0 + body_w - 1
                    body = mask[body_r0 : body_r1 + 1, body_c0 : body_c1 + 1]
                    if body.shape != (body_h, body_w):
                        continue
                    if not body[1:-1, 1:-1].all():
                        continue

                    top_strip = mask[body_r0 - 1, body_c0 : body_c1 + 1] if top_tooth else np.zeros(body_w, dtype=bool)
                    bottom_strip = (
                        mask[body_r1 + 1, body_c0 : body_c1 + 1] if bottom_tooth else np.zeros(body_w, dtype=bool)
                    )
                    left_strip = mask[body_r0 : body_r1 + 1, body_c0 - 1] if left_tooth else np.zeros(body_h, dtype=bool)
                    right_strip = (
                        mask[body_r0 : body_r1 + 1, body_c1 + 1] if right_tooth else np.zeros(body_h, dtype=bool)
                    )

                    sides = {
                        "top": _analyze_edge(body[0, :], top_strip, bool(top_tooth)),
                        "bottom": _analyze_edge(body[-1, :], bottom_strip, bool(bottom_tooth)),
                        "left": _analyze_edge(body[:, 0], left_strip, bool(left_tooth)),
                        "right": _analyze_edge(body[:, -1], right_strip, bool(right_tooth)),
                    }
                    if any(value is None for value in sides.values()):
                        continue
                    sides = {k: v for k, v in sides.items() if v is not None}
                    recon = _render_piece_mask(body_h, body_w, sides)
                    if recon.shape != mask.shape or not np.array_equal(recon, mask):
                        continue
                    solutions.append(_make_piece(color, body_h, body_w, sides, mask))

    assert len(solutions) == 1, "piece shape must decode uniquely"
    return solutions[0]


def _matches_boundary(piece: dict, r: int, c: int, rows: int, cols: int) -> bool:
    expectations = {
        "top": r == 0,
        "bottom": r == rows - 1,
        "left": c == 0,
        "right": c == cols - 1,
    }
    for side, needs_smooth in expectations.items():
        if (piece["sides"][side][0] == "smooth") != needs_smooth:
            return False
    return True


def _complement(a: tuple[str, int | None], b: tuple[str, int | None]) -> bool:
    return {a[0], b[0]} == {"tooth", "slot"} and a[1] == b[1]


def _horizontal_match(left_piece: dict, right_piece: dict) -> bool:
    return left_piece["body_h"] == right_piece["body_h"] and _complement(
        left_piece["sides"]["right"], right_piece["sides"]["left"]
    )


def _vertical_match(top_piece: dict, bottom_piece: dict) -> bool:
    return top_piece["body_w"] == bottom_piece["body_w"] and _complement(
        top_piece["sides"]["bottom"], bottom_piece["sides"]["top"]
    )


def _solve_layout(pieces: list[dict], max_solutions: int = 2) -> tuple[int, int, list[tuple[tuple[tuple[int, ...], ...], tuple[int, ...], tuple[int, ...]]]]:
    rows = sum(piece["sides"]["left"][0] == "smooth" for piece in pieces)
    cols = sum(piece["sides"]["top"][0] == "smooth" for piece in pieces)
    assert rows > 0 and cols > 0
    assert rows == sum(piece["sides"]["right"][0] == "smooth" for piece in pieces)
    assert cols == sum(piece["sides"]["bottom"][0] == "smooth" for piece in pieces)
    assert rows * cols == len(pieces)

    grid = [[-1 for _ in range(cols)] for _ in range(rows)]
    used = [False] * len(pieces)
    row_heights: list[int | None] = [None] * rows
    col_widths: list[int | None] = [None] * cols
    boundary_candidates = {
        (r, c): [i for i, piece in enumerate(pieces) if _matches_boundary(piece, r, c, rows, cols)]
        for r in range(rows)
        for c in range(cols)
    }
    solutions = []

    def fits(piece_idx: int, r: int, c: int) -> bool:
        if used[piece_idx]:
            return False
        piece = pieces[piece_idx]
        if row_heights[r] is not None and row_heights[r] != piece["body_h"]:
            return False
        if col_widths[c] is not None and col_widths[c] != piece["body_w"]:
            return False
        if r > 0 and grid[r - 1][c] != -1 and not _vertical_match(pieces[grid[r - 1][c]], piece):
            return False
        if c > 0 and grid[r][c - 1] != -1 and not _horizontal_match(pieces[grid[r][c - 1]], piece):
            return False
        if r + 1 < rows and grid[r + 1][c] != -1 and not _vertical_match(piece, pieces[grid[r + 1][c]]):
            return False
        if c + 1 < cols and grid[r][c + 1] != -1 and not _horizontal_match(piece, pieces[grid[r][c + 1]]):
            return False
        return True

    def search() -> None:
        if len(solutions) >= max_solutions:
            return

        best_pos = None
        best_choices = None
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != -1:
                    continue
                choices = [i for i in boundary_candidates[(r, c)] if fits(i, r, c)]
                if not choices:
                    return
                if best_choices is None or len(choices) < len(best_choices):
                    best_pos = (r, c)
                    best_choices = choices
                    if len(best_choices) == 1:
                        break
            if best_choices is not None and len(best_choices) == 1:
                break

        if best_pos is None:
            layout = tuple(tuple(int(cell) for cell in row) for row in grid)
            assert all(value is not None for value in row_heights)
            assert all(value is not None for value in col_widths)
            solutions.append(
                (
                    layout,
                    tuple(int(value) for value in row_heights),
                    tuple(int(value) for value in col_widths),
                )
            )
            return

        r, c = best_pos
        old_row = row_heights[r]
        old_col = col_widths[c]
        for piece_idx in best_choices:
            used[piece_idx] = True
            grid[r][c] = piece_idx
            if row_heights[r] is None:
                row_heights[r] = pieces[piece_idx]["body_h"]
            if col_widths[c] is None:
                col_widths[c] = pieces[piece_idx]["body_w"]
            search()
            used[piece_idx] = False
            grid[r][c] = -1
            row_heights[r] = old_row
            col_widths[c] = old_col

    search()
    return rows, cols, solutions


def _assemble_solution(
    pieces: list[dict],
    layout: tuple[tuple[int, ...], ...],
    row_heights: tuple[int, ...],
    col_widths: tuple[int, ...],
) -> np.ndarray:
    row_starts = [0]
    for value in row_heights[:-1]:
        row_starts.append(row_starts[-1] + int(value))
    col_starts = [0]
    for value in col_widths[:-1]:
        col_starts.append(col_starts[-1] + int(value))

    out = np.full((sum(row_heights), sum(col_widths)), -1, dtype=int)
    for r, row in enumerate(layout):
        for c, piece_idx in enumerate(row):
            piece = pieces[piece_idx]
            top = row_starts[r] - piece["body_r0"]
            left = col_starts[c] - piece["body_c0"]
            rr, cc = np.where(piece["mask"])
            for dr, dc in zip(rr, cc):
                gr = int(top + dr)
                gc = int(left + dc)
                assert 0 <= gr < out.shape[0] and 0 <= gc < out.shape[1]
                assert out[gr, gc] == -1, "assembled pieces must not overlap"
                out[gr, gc] = int(piece["color"])
    assert np.all(out != -1), "assembled output must be a filled rectangle"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered rectangular puzzle pieces by matching smooth edges and interlocking teeth/slots.

    Smooth edges belong to the outside of the final rectangle. Non-smooth edges must be paired with an
    adjacent piece that has the opposite connector type at the same offset. The assembled output is the
    tight filled rectangle formed by all pieces.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _dominant_color(g)

    pieces = []
    for coords in _components(g != bg):
        colors = np.unique(g[coords[:, 0], coords[:, 1]])
        assert colors.size == 1, "each puzzle piece must be monochrome"
        r0, c0, r1, c1 = _bbox(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        pieces.append(_infer_piece(mask, int(colors[0])))

    assert pieces, "expected at least one piece"
    _rows, _cols, solutions = _solve_layout(pieces, max_solutions=2)
    assert len(solutions) == 1, "puzzle layout must have a unique assembly"
    layout, row_heights, col_widths = solutions[0]
    return _assemble_solution(pieces, layout, row_heights, col_widths)


def _random_layout() -> tuple[int, int]:
    return random.choice(LAYOUT_OPTIONS)


def _random_sides(rows: int, cols: int, row_heights: list[int], col_widths: list[int]) -> list[list[dict[str, tuple[str, int | None]]]]:
    sides = [[{} for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if r == 0:
                sides[r][c]["top"] = ("smooth", None)
            if r == rows - 1:
                sides[r][c]["bottom"] = ("smooth", None)
            if c == 0:
                sides[r][c]["left"] = ("smooth", None)
            if c == cols - 1:
                sides[r][c]["right"] = ("smooth", None)

    for r in range(rows):
        for c in range(cols - 1):
            offset = random.randint(1, row_heights[r] - 2)
            if random.random() < 0.5:
                sides[r][c]["right"] = ("tooth", offset)
                sides[r][c + 1]["left"] = ("slot", offset)
            else:
                sides[r][c]["right"] = ("slot", offset)
                sides[r][c + 1]["left"] = ("tooth", offset)

    for r in range(rows - 1):
        for c in range(cols):
            offset = random.randint(1, col_widths[c] - 2)
            if random.random() < 0.5:
                sides[r][c]["bottom"] = ("tooth", offset)
                sides[r + 1][c]["top"] = ("slot", offset)
            else:
                sides[r][c]["bottom"] = ("slot", offset)
                sides[r + 1][c]["top"] = ("tooth", offset)

    for r in range(rows):
        for c in range(cols):
            assert set(sides[r][c]) == {"top", "bottom", "left", "right"}
    return sides


def _build_canonical_pieces() -> tuple[list[dict], np.ndarray] | None:
    rows, cols = _random_layout()
    row_heights = [random.choice(BODY_SIZE_CHOICES) for _ in range(rows)]
    col_widths = [random.choice(BODY_SIZE_CHOICES) for _ in range(cols)]
    side_grid = _random_sides(rows, cols, row_heights, col_widths)
    colors = random.sample(PALETTE, rows * cols)

    pieces = []
    layout = []
    idx = 0
    for r in range(rows):
        row = []
        for c in range(cols):
            piece = _make_piece(colors[idx], row_heights[r], col_widths[c], side_grid[r][c])
            if len(_components(piece["mask"])) != 1:
                return None
            pieces.append(piece)
            row.append(idx)
            idx += 1
        layout.append(tuple(row))

    expected = _assemble_solution(pieces, tuple(layout), tuple(row_heights), tuple(col_widths))
    return pieces, expected


def _scatter_pieces(pieces: list[dict]) -> np.ndarray | None:
    masks = [piece["mask"] for piece in pieces]
    max_h = max(mask.shape[0] for mask in masks)
    max_w = max(mask.shape[1] for mask in masks)
    padded_area = sum((mask.shape[0] + 2) * (mask.shape[1] + 2) for mask in masks)
    base = int(np.ceil(np.sqrt(padded_area)))

    for extra in range(SCATTER_RETRIES):
        height = random.randint(max(base, max_h + 3, 18), max(base, max_h + 3, 18) + 6 + extra)
        width = random.randint(max(base, max_w + 3, 18), max(base, max_w + 3, 18) + 6 + extra)
        occupied = np.zeros((height, width), dtype=bool)
        order = sorted(range(len(pieces)), key=lambda i: masks[i].shape[0] * masks[i].shape[1], reverse=True)
        placements: list[tuple[int, int] | None] = [None] * len(pieces)
        ok = True

        for idx in order:
            mask = masks[idx]
            h, w = mask.shape
            if height <= h + 2 or width <= w + 2:
                ok = False
                break
            placed = False
            for _ in range(POSITION_RETRIES):
                top = random.randint(1, height - h - 1)
                left = random.randint(1, width - w - 1)
                r0 = max(0, top - 1)
                r1 = min(height, top + h + 1)
                c0 = max(0, left - 1)
                c1 = min(width, left + w + 1)
                if occupied[r0:r1, c0:c1].any():
                    continue
                occupied[r0:r1, c0:c1] = True
                placements[idx] = (top, left)
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        grid = np.full((height, width), BG, dtype=int)
        for idx, placement in enumerate(placements):
            assert placement is not None
            top, left = placement
            patch = grid[top : top + masks[idx].shape[0], left : left + masks[idx].shape[1]]
            patch[masks[idx]] = pieces[idx]["color"]
        return grid

    return None


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    built = _build_canonical_pieces()
    if built is None:
        return None
    pieces, expected = built
    transform = random.randrange(8)
    transformed = []
    for piece in pieces:
        transformed.append(
            {
                "color": piece["color"],
                "mask": _apply_transform(piece["mask"], transform).astype(bool, copy=False),
            }
        )
    expected = _apply_transform(expected, transform)
    grid = _scatter_pieces(transformed)
    if grid is None:
        return None
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        attempt = _gen_attempt()
        if attempt is not None:
            return attempt[0]
    raise RuntimeError("failed to generate a valid puzzle input")
