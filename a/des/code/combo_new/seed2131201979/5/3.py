from __future__ import annotations

import random

import numpy as np


DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}

GEN_RETRY_LIMIT = 400
GEN_COLOR_POOL = tuple(range(10))
GEN_BG = 0


def _mode_color(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_grid_structure(grid: np.ndarray, bg: int) -> tuple[list[int], list[int], int]:
    h, w = grid.shape

    candidate_rows = []
    for r in range(h):
        row_vals = set(map(int, grid[r, :]))
        if bg in row_vals:
            continue
        if len(row_vals) <= 2:
            candidate_rows.append(r)

    candidate_cols = []
    for c in range(w):
        col_vals = set(map(int, grid[:, c]))
        if bg in col_vals:
            continue
        if len(col_vals) <= 2:
            candidate_cols.append(c)

    assert len(candidate_rows) >= 2 and len(candidate_cols) >= 2, "failed to detect separator rows/cols"

    stripe_samples = []
    for r in candidate_rows:
        stripe_samples.extend(int(v) for v in grid[r, :])
    for c in candidate_cols:
        stripe_samples.extend(int(v) for v in grid[:, c])
    sep = _mode_color(np.array(stripe_samples, dtype=int))

    line_rows = [r for r in candidate_rows if int(np.count_nonzero(grid[r, :] == sep)) > w // 2]
    line_cols = [c for c in candidate_cols if int(np.count_nonzero(grid[:, c] == sep)) > h // 2]

    line_rows.sort()
    line_cols.sort()
    assert len(line_rows) >= 2 and len(line_cols) >= 2, "failed to refine separator rows/cols"
    return line_rows, line_cols, int(sep)


def _cell_patch(grid: np.ndarray, line_rows: list[int], line_cols: list[int], r: int, c: int) -> np.ndarray:
    return grid[line_rows[r] + 1 : line_rows[r + 1], line_cols[c] + 1 : line_cols[c + 1]].copy()


def _cell_nonempty(patch: np.ndarray, bg: int) -> bool:
    return bool(np.any(patch != int(bg)))


def _cell_direction(
    grid: np.ndarray,
    line_rows: list[int],
    line_cols: list[int],
    r: int,
    c: int,
    sep: int,
) -> str | None:
    r0 = line_rows[r] + 1
    r1 = line_rows[r + 1]
    c0 = line_cols[c] + 1
    c1 = line_cols[c + 1]

    sides = []
    if bool(np.any(grid[line_rows[r], c0:c1] != int(sep))):
        sides.append("up")
    if bool(np.any(grid[line_rows[r + 1], c0:c1] != int(sep))):
        sides.append("down")
    if bool(np.any(grid[r0:r1, line_cols[c]] != int(sep))):
        sides.append("left")
    if bool(np.any(grid[r0:r1, line_cols[c + 1]] != int(sep))):
        sides.append("right")

    if len(sides) != 1:
        return None
    return sides[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each cell's interior stamp through empty neighboring cells in the direction indicated by its border marker.

    Input structure:
    - The background is the dominant color.
    - One separator color forms full horizontal and vertical lattice lines that partition the grid into equal rectangular cells.
    - Some cell interiors contain a small non-background stamp pattern made of one or more non-separator colors.
    - Every non-empty source cell has exactly one marker pixel placed on one of its bordering separator segments, outside the cell interior.

    Rule:
    - Detect the lattice rows and columns, then read each non-empty cell interior as a reusable stamp.
    - The separator side containing the marker determines the propagation direction: top, right, bottom, or left.
    - March cell by cell in that direction; whenever the next cell interior is completely empty, copy the full stamp into it, and stop when the next cell is already non-empty or the grid boundary is reached.
    - Keep separator lines, marker pixels, and original source stamps unchanged.

    Key insight:
    - The marker does not sit inside the stamp cell, so direction must be inferred from the cell-border segment carrying the out-of-cell marker before the stamp can be tiled through the lattice.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    line_rows, line_cols, sep = _find_grid_structure(g, bg)

    n_rows = len(line_rows) - 1
    n_cols = len(line_cols) - 1
    out = g.copy()

    occupied = np.zeros((n_rows, n_cols), dtype=bool)
    sources: list[tuple[int, int, str, np.ndarray]] = []
    for r in range(n_rows):
        for c in range(n_cols):
            patch = _cell_patch(g, line_rows, line_cols, r, c)
            if not _cell_nonempty(patch, bg):
                continue
            occupied[r, c] = True
            direction = _cell_direction(g, line_rows, line_cols, r, c, sep)
            if direction is None:
                continue
            sources.append((r, c, direction, patch))

    for r, c, direction, patch in sources:
        dr, dc = DIRS[direction]
        rr, cc = r + dr, c + dc
        while 0 <= rr < n_rows and 0 <= cc < n_cols and not bool(occupied[rr, cc]):
            out[line_rows[rr] + 1 : line_rows[rr + 1], line_cols[cc] + 1 : line_cols[cc + 1]] = patch
            occupied[rr, cc] = True
            rr += dr
            cc += dc

    return out


def _grow_connected_cells(height: int, width: int, area: int) -> list[tuple[int, int]] | None:
    for _ in range(40):
        start = (random.randrange(height), random.randrange(width))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) == area:
            return sorted(cells)
    return None


def _make_pattern(height: int, width: int, colors: list[int]) -> np.ndarray | None:
    max_area = min(height * width - 1, 7)
    if max_area < 2:
        return None
    area = random.randint(2, max_area)
    cells = _grow_connected_cells(height, width, area)
    if cells is None:
        return None

    patch = np.full((height, width), GEN_BG, dtype=int)
    primary = colors[0]
    secondary = colors[1] if len(colors) > 1 else colors[0]
    use_secondary = len(colors) > 1 and area >= 3 and random.random() < 0.4
    secondary_count = random.randint(1, max(1, area // 2)) if use_secondary else 0
    secondary_cells = set(random.sample(cells, k=secondary_count)) if secondary_count else set()

    for cell in cells:
        patch[cell] = secondary if cell in secondary_cells else primary

    return patch


def _enumerate_candidates(
    cell_rows: int,
    cell_cols: int,
    reserved: np.ndarray,
    source_cells: set[tuple[int, int]],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for r in range(cell_rows):
        for c in range(cell_cols):
            if bool(reserved[r, c]):
                continue
            if any(abs(r - sr) + abs(c - sc) == 1 for sr, sc in source_cells):
                continue
            for direction, (dr, dc) in DIRS.items():
                rr, cc = r + dr, c + dc
                if not (0 <= rr < cell_rows and 0 <= cc < cell_cols):
                    continue
                if bool(reserved[rr, cc]):
                    continue

                path = []
                while 0 <= rr < cell_rows and 0 <= cc < cell_cols and not bool(reserved[rr, cc]):
                    path.append((rr, cc))
                    rr += dr
                    cc += dc
                if not path:
                    continue

                stop = "boundary" if not (0 <= rr < cell_rows and 0 <= cc < cell_cols) else "blocker"
                candidates.append(
                    {
                        "src": (r, c),
                        "dir": direction,
                        "path": path,
                        "stop": stop,
                    }
                )
    return candidates


def _choose_candidate(candidates: list[dict[str, object]], preferred_stop: str | None, dir_counts: dict[str, int]) -> dict[str, object]:
    pool = candidates
    if preferred_stop is not None:
        narrowed = [cand for cand in pool if str(cand["stop"]) == preferred_stop]
        if narrowed:
            pool = narrowed

    min_count = min(dir_counts[str(cand["dir"])] for cand in pool)
    pool = [cand for cand in pool if dir_counts[str(cand["dir"])] == min_count]
    weights = [1.0 / (len(cand["path"]) + 0.5) for cand in pool]
    return random.choices(pool, weights=weights, k=1)[0]


def _render_input(
    cell_rows: int,
    cell_cols: int,
    cell_h: int,
    cell_w: int,
    sep: int,
    arrow: int,
    placements: list[dict[str, object]],
) -> np.ndarray:
    h = cell_rows * cell_h + cell_rows + 1
    w = cell_cols * cell_w + cell_cols + 1
    grid = np.full((h, w), GEN_BG, dtype=int)

    line_rows = [idx * (cell_h + 1) for idx in range(cell_rows + 1)]
    line_cols = [idx * (cell_w + 1) for idx in range(cell_cols + 1)]
    grid[line_rows, :] = sep
    grid[:, line_cols] = sep

    for item in placements:
        r, c = item["src"]
        patch = item["patch"]
        r0 = line_rows[r] + 1
        c0 = line_cols[c] + 1
        grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = patch

        direction = str(item["dir"])
        if direction in ("up", "down"):
            offset = int(item["offset"])
            rr = line_rows[r] if direction == "up" else line_rows[r + 1]
            cc = c0 + offset
        else:
            offset = int(item["offset"])
            rr = r0 + offset
            cc = line_cols[c] if direction == "left" else line_cols[c + 1]
        grid[rr, cc] = arrow

    return grid


def _expected_output(
    grid: np.ndarray,
    cell_rows: int,
    cell_cols: int,
    cell_h: int,
    cell_w: int,
    placements: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    line_rows = [idx * (cell_h + 1) for idx in range(cell_rows + 1)]
    line_cols = [idx * (cell_w + 1) for idx in range(cell_cols + 1)]
    for item in placements:
        patch = item["patch"]
        for rr, cc in item["path"]:
            r0 = line_rows[rr] + 1
            c0 = line_cols[cc] + 1
            out[r0 : r0 + cell_h, c0 : c0 + cell_w] = patch
    return out


def _attempt_generate() -> np.ndarray | None:
    for _ in range(120):
        cell_rows = random.randint(2, 5)
        cell_cols = random.randint(2, 5)
        cell_h = random.randint(4, 6)
        cell_w = random.randint(4, 6)
        h = cell_rows * cell_h + cell_rows + 1
        w = cell_cols * cell_w + cell_cols + 1
        if h > 30 or w > 30:
            continue

        palette = list(GEN_COLOR_POOL)
        palette.remove(GEN_BG)
        random.shuffle(palette)
        sep = palette.pop()
        arrow = palette.pop()

        max_sources = min(6, max(2, (cell_rows * cell_cols) // 2))
        n_sources = random.randint(2, max_sources)
        reserved = np.zeros((cell_rows, cell_cols), dtype=bool)
        source_cells: set[tuple[int, int]] = set()
        dir_counts = {name: 0 for name in DIRS}
        placements: list[dict[str, object]] = []

        for index in range(n_sources):
            candidates = _enumerate_candidates(cell_rows, cell_cols, reserved, source_cells)
            if not candidates:
                break

            preferred_stop = None
            if index == 0:
                preferred_stop = "boundary"
            elif index % 2 == 1:
                preferred_stop = "blocker"
            else:
                preferred_stop = "boundary"

            chosen = _choose_candidate(candidates, preferred_stop, dir_counts)
            src = chosen["src"]
            direction = str(chosen["dir"])
            path = list(chosen["path"])

            color_choices = random.sample(palette, k=min(3, len(palette)))
            patch = _make_pattern(cell_h, cell_w, color_choices)
            if patch is None:
                break

            offset = random.randrange(cell_w if direction in ("up", "down") else cell_h)
            placements.append(
                {
                    "src": src,
                    "dir": direction,
                    "path": path,
                    "patch": patch,
                    "offset": offset,
                    "stop": chosen["stop"],
                }
            )

            sr, sc = src
            reserved[sr, sc] = True
            for rr, cc in path:
                reserved[rr, cc] = True
            source_cells.add(src)
            dir_counts[direction] += 1

        if len(placements) < 2:
            continue

        grid = _render_input(cell_rows, cell_cols, cell_h, cell_w, sep, arrow, placements)
        if _mode_color(grid) != GEN_BG:
            continue

        stops = {str(item["stop"]) for item in placements}
        if "blocker" not in stops and len(placements) >= 3:
            continue

        expected = _expected_output(grid, cell_rows, cell_cols, cell_h, cell_w, placements)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
