from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
LINE = 5
MARKER = 7

MULTI_COLORS = (1, 2, 3, 6, 8, 9)
SINGLE_COLORS = (1, 2, 3, 4, 6, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
GEN_RETRIES = 500


def _on_boundary(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components_by_color(grid: np.ndarray, marker_mask: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or marker_mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or marker_mask[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append({"color": color, "coords": coords, "area": len(coords)})
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw gray boundary-defined through-lines while letting multi-pixel objects occlude them and single pixels get absorbed into the line.

    Input structure:
    - The background is black.
    - Orange boundary marker pixels are the only non-background cells on the outer border.
    - Interior foreground objects are 4-connected colored components, including both multi-pixel shapes and isolated single pixels.

    Rule:
    - Each orange marker on the top or bottom edge defines a full vertical gray line through its column, and each orange marker on the left or right edge defines a full horizontal gray line through its row.
    - Draw all such full-length gray lines across the grid.
    - For every line/object overlap, inspect the overlapped pixel's original 4-connected monochromatic component: components with area greater than one keep their original color and occlude the line, while area-one single pixels are overwritten to gray.
    - Keep all non-overlapped interior objects unchanged and preserve the orange boundary markers.

    Key insight:
    - The final color at a crossed pixel depends on two coupled cues at once: whether a boundary marker induces a through-line there, and whether the touched foreground pixel belongs to a multi-pixel component or an isolated singleton.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    marker_mask = np.zeros_like(g, dtype=bool)
    rows = set()
    cols = set()
    for r, c in np.argwhere(g == MARKER):
        rr, cc = int(r), int(c)
        if not _on_boundary(rr, cc, h, w):
            continue
        marker_mask[rr, cc] = True
        if rr == 0 or rr == h - 1:
            cols.add(cc)
        if cc == 0 or cc == w - 1:
            rows.add(rr)

    out = g.copy()
    for c in cols:
        out[:, c] = LINE
    for r in rows:
        out[r, :] = LINE

    for comp in _components_by_color(g, marker_mask):
        if comp["area"] > 1:
            for r, c in comp["coords"]:
                if r in rows or c in cols:
                    out[r, c] = int(g[r, c])

    out[marker_mask] = MARKER
    return out


def _line_overlap(cell: tuple[int, int], rows: set[int], cols: set[int]) -> bool:
    r, c = cell
    return r in rows or c in cols


def _cell_clear(occupied: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if occupied[r, c]:
        return False
    for dr, dc in DIR8:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and occupied[rr, cc]:
            return False
    return True


def _grow_blob(
    h: int,
    w: int,
    occupied: np.ndarray,
    size: int,
    *,
    must_hit: tuple[str, int] | None = None,
    avoid_lines: tuple[set[int], set[int]] | None = None,
) -> list[tuple[int, int]] | None:
    rows, cols = avoid_lines if avoid_lines is not None else (set(), set())
    for _ in range(200):
        if must_hit is not None:
            axis, value = must_hit
            if axis == "v":
                candidates = [
                    (r, value)
                    for r in range(1, h - 1)
                    if _cell_clear(occupied, r, value, h, w)
                ]
            else:
                candidates = [
                    (value, c)
                    for c in range(1, w - 1)
                    if _cell_clear(occupied, value, c, h, w)
                ]
        else:
            candidates = [
                (r, c)
                for r in range(1, h - 1)
                for c in range(1, w - 1)
                if _cell_clear(occupied, r, c, h, w) and not _line_overlap((r, c), rows, cols)
            ]
        if not candidates:
            return None

        seed = random.choice(candidates)
        blob = {seed}
        while len(blob) < size:
            frontier = []
            for r, c in blob:
                for nr, nc in _neighbors4(r, c, h, w):
                    if (nr, nc) in blob:
                        continue
                    if must_hit is None and _line_overlap((nr, nc), rows, cols):
                        continue
                    if occupied[nr, nc]:
                        continue
                    if not (1 <= nr < h - 1 and 1 <= nc < w - 1):
                        continue
                    bad = False
                    for dr, dc in DIR8:
                        rr, cc = nr + dr, nc + dc
                        if 0 <= rr < h and 0 <= cc < w and occupied[rr, cc]:
                            bad = True
                            break
                    if not bad:
                        frontier.append((nr, nc))
            if not frontier:
                break
            blob.add(random.choice(frontier))

        if len(blob) != size:
            continue
        if must_hit is not None:
            axis, value = must_hit
            if axis == "v" and all(c == value for _, c in blob):
                continue
            if axis == "h" and all(r == value for r, _ in blob):
                continue
        return list(blob)
    return None


def _choose_line_pattern(line_count: int) -> tuple[int, int]:
    if line_count == 2:
        patterns = [(1, 1), (2, 0), (0, 2)]
        weights = (0.5, 0.25, 0.25)
    else:
        patterns = [(2, 1), (1, 2), (3, 0), (0, 3)]
        weights = (0.4, 0.4, 0.1, 0.1)
    return random.choices(patterns, weights=weights, k=1)[0]


def _sample_markers(h: int, w: int) -> tuple[list[tuple[int, int]], list[tuple[str, int]]]:
    line_count = random.randint(2, 3)
    v_count, h_count = _choose_line_pattern(line_count)
    cols = random.sample(range(1, w - 1), k=v_count)
    rows = random.sample(range(1, h - 1), k=h_count)

    markers = []
    lines = []
    for c in cols:
        markers.append((random.choice((0, h - 1)), c))
        lines.append(("v", c))
    for r in rows:
        markers.append((r, random.choice((0, w - 1))))
        lines.append(("h", r))
    random.shuffle(markers)
    random.shuffle(lines)
    return markers, lines


def _pick_single_cell(
    h: int,
    w: int,
    occupied: np.ndarray,
    rows: set[int],
    cols: set[int],
    *,
    must_hit: tuple[str, int] | None = None,
    off_lines: bool = False,
) -> tuple[int, int] | None:
    candidates = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not _cell_clear(occupied, r, c, h, w):
                continue
            on_line = _line_overlap((r, c), rows, cols)
            if must_hit is not None:
                axis, value = must_hit
                if axis == "v" and c != value:
                    continue
                if axis == "h" and r != value:
                    continue
            elif off_lines and on_line:
                continue
            candidates.append((r, c))
    if not candidates:
        return None
    return random.choice(candidates)


def _paint_cells(grid: np.ndarray, occupied: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color
        occupied[r, c] = True


def _validate_instance(grid: np.ndarray) -> bool:
    h, w = grid.shape
    if max(h, w) > 30:
        return False

    non_bg_boundary = [
        (int(r), int(c))
        for r, c in np.argwhere(grid != BG)
        if _on_boundary(int(r), int(c), h, w)
    ]
    if not (2 <= len(non_bg_boundary) <= 3):
        return False
    if any(int(grid[r, c]) != MARKER for r, c in non_bg_boundary):
        return False

    marker_mask = np.zeros_like(grid, dtype=bool)
    rows = set()
    cols = set()
    for r, c in non_bg_boundary:
        marker_mask[r, c] = True
        if r == 0 or r == h - 1:
            cols.add(c)
        if c == 0 or c == w - 1:
            rows.add(r)

    comps = _components_by_color(grid, marker_mask)
    multi = [comp for comp in comps if comp["area"] > 1]
    singles = [comp for comp in comps if comp["area"] == 1]
    if not (3 <= len(multi) <= 5 and 3 <= len(singles) <= 6):
        return False

    if not any(any(_line_overlap(cell, rows, cols) for cell in comp["coords"]) for comp in multi):
        return False
    if not any(_line_overlap(comp["coords"][0], rows, cols) for comp in singles):
        return False
    if not any(any(not _line_overlap(cell, rows, cols) for cell in comp["coords"]) for comp in multi):
        return False
    if not any(not _line_overlap(comp["coords"][0], rows, cols) for comp in singles):
        return False

    target_lines = []
    for axis, value in [("v", c) for c in cols] + [("h", r) for r in rows]:
        has_multi = any(
            any((c == value if axis == "v" else r == value) for r, c in comp["coords"])
            for comp in multi
        )
        has_single = any(
            (comp["coords"][0][1] == value if axis == "v" else comp["coords"][0][0] == value)
            for comp in singles
        )
        if has_multi and has_single:
            target_lines.append((axis, value))
    if not target_lines:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(12, 18)
    w = random.randint(12, 18)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    markers, lines = _sample_markers(h, w)
    rows = {value for axis, value in lines if axis == "h"}
    cols = {value for axis, value in lines if axis == "v"}
    target_line = random.choice(lines)

    for r, c in markers:
        grid[r, c] = MARKER

    num_multi = random.randint(3, 5)
    num_single = random.randint(3, 6)

    target_blob = _grow_blob(
        h,
        w,
        occupied,
        random.randint(3, 8),
        must_hit=target_line,
    )
    if target_blob is None:
        return None
    _paint_cells(grid, occupied, target_blob, random.choice(MULTI_COLORS))

    target_single = _pick_single_cell(
        h,
        w,
        occupied,
        rows,
        cols,
        must_hit=target_line,
    )
    if target_single is None:
        return None
    _paint_cells(grid, occupied, [target_single], random.choice(SINGLE_COLORS))

    need_off_multi = True
    for i in range(num_multi - 1):
        mode = "off"
        if i == 0 and len(lines) > 1 and random.random() < 0.5:
            mode = "other-line"
        elif random.random() < 0.3:
            mode = "target-line"

        must_hit = None
        avoid_lines = None
        if mode == "off" or need_off_multi:
            avoid_lines = (rows, cols)
            must_hit = None
            need_off_multi = False
        elif mode == "other-line":
            other_lines = [line for line in lines if line != target_line]
            if other_lines:
                must_hit = random.choice(other_lines)
            else:
                avoid_lines = (rows, cols)
        else:
            must_hit = target_line

        blob = _grow_blob(
            h,
            w,
            occupied,
            random.randint(3, 8),
            must_hit=must_hit,
            avoid_lines=avoid_lines,
        )
        if blob is None:
            return None
        _paint_cells(grid, occupied, blob, random.choice(MULTI_COLORS))

    need_off_single = True
    for i in range(num_single - 1):
        must_hit = None
        off_lines = False
        if need_off_single or (i > 0 and random.random() < 0.5):
            off_lines = True
            need_off_single = False
        else:
            candidate_lines = lines if random.random() < 0.6 else [target_line]
            must_hit = random.choice(candidate_lines)

        cell = _pick_single_cell(
            h,
            w,
            occupied,
            rows,
            cols,
            must_hit=must_hit,
            off_lines=off_lines,
        )
        if cell is None:
            return None
        _paint_cells(grid, occupied, [cell], random.choice(SINGLE_COLORS))

    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-8 input")


if __name__ == "__main__":
    sample = generate_input()
    print(sample)
    print(main(sample))
