from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
RECT_COLORS = (1, 4, 5, 6, 7, 8, 9)
MAX_ATTEMPTS = 800
ROW_ATTEMPTS = 300


@dataclass(frozen=True)
class Rect:
    color: int
    top: int
    left: int
    height: int
    width: int
    guide_row: int


@dataclass(frozen=True)
class Move:
    rect: Rect
    new_left: int
    next_guide_row: int | None


def _wrapped_cols(start: int, width: int, total_w: int) -> tuple[int, ...]:
    return tuple(int((start + dc) % total_w) for dc in range(width))


def _find_guide_rows(grid: np.ndarray) -> list[int]:
    guides = [int(r) for r in range(grid.shape[0]) if np.all(grid[r] != BG)]
    assert 2 <= len(guides) <= 3, "expected 2-3 full-width horizontal guide lines"
    assert all(guides[i + 1] - guides[i] >= 4 for i in range(len(guides) - 1))
    return guides


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _extract_rectangles(grid: np.ndarray, guide_rows: list[int]) -> list[Rect]:
    fg = (grid != BG) & (grid != RED)
    rects: list[Rect] = []
    for coords in _components(fg):
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        color = int(patch[0, 0])
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert color not in (BG, RED, GREEN)
        assert np.all(patch == color), "each foreground object must be a solid monochrome rectangle"
        assert coords.shape[0] == height * width
        assert r1 in guide_rows, "rectangle bottom edge must coincide with a guide line"
        assert 2 <= height <= 3
        assert 2 <= width <= 4
        rects.append(Rect(color=color, top=r0, left=c0, height=height, width=width, guide_row=r1))

    assert rects, "expected at least one rectangle"
    per_row = {row: 0 for row in guide_rows}
    for rect in rects:
        per_row[rect.guide_row] += 1
    assert all(1 <= per_row[row] <= 2 for row in guide_rows)
    return sorted(rects, key=lambda rect: (rect.guide_row, rect.left))


def _compute_moves(rects: list[Rect], guide_rows: list[int], total_w: int) -> list[Move]:
    next_rows = {guide_rows[i]: guide_rows[i + 1] for i in range(len(guide_rows) - 1)}
    next_rows[guide_rows[-1]] = None
    moves: list[Move] = []
    covered_by_row: dict[int, set[int]] = {row: set() for row in guide_rows}
    for rect in rects:
        new_left = (rect.left + rect.width) % total_w
        cols = set(_wrapped_cols(new_left, rect.width, total_w))
        assert len(cols) == rect.width
        assert not (cols & covered_by_row[rect.guide_row]), "shifted rectangles on one row must not overlap"
        covered_by_row[rect.guide_row].update(cols)
        moves.append(Move(rect=rect, new_left=new_left, next_guide_row=next_rows[rect.guide_row]))
    return moves


def _paint_wrapped_rect(out: np.ndarray, rect: Rect, new_left: int) -> None:
    total_w = out.shape[1]
    cols = _wrapped_cols(new_left, rect.width, total_w)
    for rr in range(rect.top, rect.guide_row + 1):
        for cc in cols:
            out[rr, cc] = rect.color


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    guide_rows = _find_guide_rows(g)
    rects = _extract_rectangles(g, guide_rows)
    moves = _compute_moves(rects, guide_rows, g.shape[1])

    out = g.copy()
    for rect in rects:
        if rect.height > 1:
            out[rect.top : rect.guide_row, rect.left : rect.left + rect.width] = BG
        out[rect.guide_row, rect.left : rect.left + rect.width] = RED

    for move in moves:
        _paint_wrapped_rect(out, move.rect, move.new_left)

    for move in moves:
        if move.next_guide_row is None:
            continue
        ray_col = move.new_left
        for rr in range(move.rect.top, move.next_guide_row):
            if out[rr, ray_col] == BG:
                out[rr, ray_col] = BLUE
            else:
                assert out[rr, ray_col] != RED, "ray must stop at the next lower guide line"
        assert out[move.next_guide_row, ray_col] == RED, "ray endpoint must be an exposed guide-line pixel"
        out[move.next_guide_row, ray_col] = GREEN

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move each guide-line rectangle right by its own width, then drop a ray to the next guide line.

    Input structure:
    - Black background with 2-3 full-width horizontal red guide lines.
    - Each guide line contains 1-2 solid non-red rectangles whose bottom edges lie on that guide line.
    - Rectangles are separated, axis-aligned, and use heights 2-3 and widths 2-4.

    Rule:
    - Erase each rectangle from its original position, restoring black above the guide line and red on the guide-line row.
    - Repaint the same rectangle shifted right by exactly its own width, wrapping around the left edge if needed.
    - From the shifted rectangle's top-left column, emit a blue vertical ray downward, painting only black cells until the next lower guide line; replace that hit guide-line pixel with green. Rectangles on the lowest guide line only shift.

    Key insight:
    - A rectangle's own width determines its horizontal shift, and that shifted anchor column then determines where the downward ray meets the next guide line.
    """
    return _solve(grid)


def _sample_guide_rows(height: int, count: int) -> list[int] | None:
    rows: list[int] = []
    choices = list(range(4, height - 1))
    for idx in range(count):
        remaining = count - idx - 1
        valid = [
            row
            for row in choices
            if (not rows or row - rows[-1] >= 4) and row + 4 * remaining <= height - 2
        ]
        if not valid:
            return None
        picked = random.choice(valid)
        rows.append(picked)
        choices = [row for row in choices if row > picked]
    return rows


def _old_row_cols(left: int, width: int) -> set[int]:
    return set(range(left, left + width))


def _old_row_separated(left: int, width: int, existing: list[Rect]) -> bool:
    right = left + width - 1
    for rect in existing:
        other_right = rect.left + rect.width - 1
        if not (right + 1 < rect.left or other_right + 1 < left):
            return False
    return True


def _make_row_rects(
    total_w: int,
    guide_row: int,
    blocked_targets: set[int],
    want_wrap: bool,
) -> list[Rect] | None:
    for _ in range(ROW_ATTEMPTS):
        rect_count = random.choice([1, 2])
        rects: list[Rect] = []
        shifted_cover: set[int] = set()
        wrap_count = 0

        order = list(range(rect_count))
        random.shuffle(order)
        for idx in order:
            candidates: list[tuple[int, int, int]] = []
            for height in (2, 3):
                top = guide_row - height + 1
                if top < 0:
                    continue
                for width in (2, 3, 4):
                    for left in range(0, total_w - width + 1):
                        if not _old_row_separated(left, width, rects):
                            continue
                        new_left = (left + width) % total_w
                        new_cols = set(_wrapped_cols(new_left, width, total_w))
                        if new_cols & shifted_cover:
                            continue
                        if blocked_targets and new_left in blocked_targets:
                            continue
                        candidates.append((height, width, left))
            if not candidates:
                rects = []
                break

            weighted: list[tuple[int, int, int]] = []
            for height, width, left in candidates:
                wraps = left + 2 * width > total_w
                copies = 3 if wraps else 1
                weighted.extend([(height, width, left)] * copies)
            height, width, left = random.choice(weighted)
            wraps = left + 2 * width > total_w
            wrap_count += int(wraps)
            rects.append(
                Rect(
                    color=-1,
                    top=guide_row - height + 1,
                    left=left,
                    height=height,
                    width=width,
                    guide_row=guide_row,
                )
            )
            shifted_cover.update(_wrapped_cols((left + width) % total_w, width, total_w))

        if not rects:
            continue
        if want_wrap and wrap_count == 0:
            continue
        return sorted(rects, key=lambda rect: rect.left)
    return None


def _assign_colors(rows: list[list[Rect]]) -> list[Rect]:
    total = sum(len(row) for row in rows)
    colors = random.sample(list(RECT_COLORS), total)
    out: list[Rect] = []
    idx = 0
    for row in rows:
        for rect in row:
            out.append(
                Rect(
                    color=colors[idx],
                    top=rect.top,
                    left=rect.left,
                    height=rect.height,
                    width=rect.width,
                    guide_row=rect.guide_row,
                )
            )
            idx += 1
    return out


def _render_input(shape: tuple[int, int], guide_rows: list[int], rects: list[Rect]) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    grid[guide_rows, :] = RED
    for rect in rects:
        grid[rect.top : rect.guide_row + 1, rect.left : rect.left + rect.width] = rect.color
    return grid


def _semantic_check(inp: np.ndarray, out: np.ndarray) -> None:
    guide_rows = _find_guide_rows(inp)
    rects = _extract_rectangles(inp, guide_rows)
    moves = _compute_moves(rects, guide_rows, inp.shape[1])

    move_map = {(move.rect.guide_row, move.rect.left, move.rect.width, move.rect.height): move for move in moves}
    row_cover: dict[int, set[int]] = {row: set() for row in guide_rows}
    for move in moves:
        row_cover[move.rect.guide_row].update(_wrapped_cols(move.new_left, move.rect.width, inp.shape[1]))

    assert len(rects) >= 3
    assert any(move.next_guide_row is not None for move in moves)

    for rect in rects:
        move = move_map[(rect.guide_row, rect.left, rect.width, rect.height)]
        new_cols = _wrapped_cols(move.new_left, rect.width, inp.shape[1])
        for rr in range(rect.top, rect.guide_row + 1):
            assert all(out[rr, cc] == rect.color for cc in new_cols)

        if move.next_guide_row is not None:
            assert move.new_left not in row_cover[move.next_guide_row]
            assert out[move.next_guide_row, move.new_left] == GREEN
            blue_cells = 0
            for rr in range(rect.guide_row + 1, move.next_guide_row):
                if out[rr, move.new_left] == BLUE:
                    blue_cells += 1
                else:
                    assert out[rr, move.new_left] == BG
            assert blue_cells >= 1

    green_points = np.argwhere(out == GREEN)
    assert len(green_points) == sum(move.next_guide_row is not None for move in moves)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 24)
    width = random.randint(14, 22)
    guide_count = random.choice([2, 3, 3])
    guide_rows = _sample_guide_rows(height, guide_count)
    if guide_rows is None:
        return None

    row_rects: list[list[Rect]] = [None] * guide_count  # type: ignore[list-item]
    next_row_cover: set[int] = set()
    wrap_required = random.random() < 0.55
    wrap_row_index = random.randrange(guide_count)

    for idx in range(guide_count - 1, -1, -1):
        want_wrap = wrap_required and idx == wrap_row_index
        rects = _make_row_rects(width, guide_rows[idx], next_row_cover, want_wrap)
        if rects is None:
            return None
        row_rects[idx] = rects
        next_row_cover = set()
        for rect in rects:
            next_row_cover.update(_wrapped_cols((rect.left + rect.width) % width, rect.width, width))

    rects = _assign_colors(row_rects)
    inp = _render_input((height, width), guide_rows, rects)

    try:
        out = _solve(inp)
        _semantic_check(inp, out)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
