from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 500


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _door_data(grid: np.ndarray) -> list[dict[str, tuple[int, int]]]:
    centers = np.argwhere(grid == RED)
    assert len(centers) == 2, "expected exactly two red door markers"

    doors = []
    for r0, c0 in centers:
        r, c = int(r0), int(c0)
        gaps = []
        blue_neighbors = 0
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            assert _in_bounds(grid.shape, rr, cc), "door center must be away from the boundary"
            value = int(grid[rr, cc])
            if value == BLACK:
                gaps.append((dr, dc, rr, cc))
            elif value == BLUE:
                blue_neighbors += 1

        assert len(gaps) == 1, "each marker must have exactly one doorway gap"
        assert blue_neighbors == 3, "each marker must sit inside a 3x3 blue frame"

        dr, dc, gap_r, gap_c = gaps[0]
        ext_r, ext_c = gap_r + dr, gap_c + dc
        assert _in_bounds(grid.shape, ext_r, ext_c), "exterior doorway point must be in bounds"
        assert int(grid[ext_r, ext_c]) == BLACK, "exterior doorway point must be black"
        doors.append(
            {
                "center": (r, c),
                "gap": (gap_r, gap_c),
                "ext": (ext_r, ext_c),
            }
        )

    return doors


def _monotone_path_data(
    grid: np.ndarray, start: tuple[int, int], end: tuple[int, int]
) -> tuple[np.ndarray, int]:
    row_step = _sign(end[0] - start[0])
    col_step = _sign(end[1] - start[1])
    assert row_step != 0 or col_step != 0, "doorway endpoints must differ"

    row_dist = abs(end[0] - start[0])
    col_dist = abs(end[1] - start[1])
    open_rel = np.zeros((row_dist + 1, col_dist + 1), dtype=bool)

    for i in range(row_dist + 1):
        rr = start[0] + i * row_step
        for j in range(col_dist + 1):
            cc = start[1] + j * col_step
            open_rel[i, j] = int(grid[rr, cc]) == BLACK

    assert open_rel[0, 0] and open_rel[row_dist, col_dist], "doorway endpoints must be unblocked"

    forward = [[0] * (col_dist + 1) for _ in range(row_dist + 1)]
    backward = [[0] * (col_dist + 1) for _ in range(row_dist + 1)]

    for i in range(row_dist + 1):
        for j in range(col_dist + 1):
            if not open_rel[i, j]:
                continue
            if i == 0 and j == 0:
                forward[i][j] = 1
                continue
            count = 0
            if i > 0:
                count += forward[i - 1][j]
            if j > 0:
                count += forward[i][j - 1]
            forward[i][j] = count

    for i in range(row_dist, -1, -1):
        for j in range(col_dist, -1, -1):
            if not open_rel[i, j]:
                continue
            if i == row_dist and j == col_dist:
                backward[i][j] = 1
                continue
            count = 0
            if i < row_dist:
                count += backward[i + 1][j]
            if j < col_dist:
                count += backward[i][j + 1]
            backward[i][j] = count

    path_mask = np.zeros_like(grid, dtype=bool)
    for i in range(row_dist + 1):
        rr = start[0] + i * row_step
        for j in range(col_dist + 1):
            if open_rel[i, j] and forward[i][j] and backward[i][j]:
                cc = start[1] + j * col_step
                path_mask[rr, cc] = True

    path_count = int(forward[row_dist][col_dist])
    assert path_count > 0, "expected at least one monotone shortest path"
    return path_mask, path_count


def _shear_positions(mask: np.ndarray, shape: tuple[int, int], by_rows: bool) -> set[tuple[int, int]]:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    h, w = shape
    shifted = set()

    if by_rows:
        base = int(coords[:, 0].min())
        for r0, c0 in coords:
            r, c = int(r0), int(c0)
            nc = c + (r - base)
            if nc < w:
                shifted.add((r, nc))
    else:
        base = int(coords[:, 1].min())
        for r0, c0 in coords:
            r, c = int(r0), int(c0)
            nr = r + (c - base)
            if nr < h:
                shifted.add((nr, c))

    return shifted


def _analyze(grid: np.ndarray) -> dict[str, object]:
    doors = _door_data(grid)
    start, end = doors[0]["ext"], doors[1]["ext"]
    path_mask, path_count = _monotone_path_data(grid, start, end)
    for door in doors:
        path_mask[door["gap"]] = True

    (r1, c1), (r2, c2) = doors[0]["center"], doors[1]["center"]
    by_rows = abs(c2 - c1) >= abs(r2 - r1)
    shifted = _shear_positions(path_mask, grid.shape, by_rows)
    return {
        "doors": doors,
        "path_mask": path_mask,
        "path_count": path_count,
        "by_rows": by_rows,
        "shifted": shifted,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the union of all monotone doorway-to-doorway shortest paths, then shear that region by rows or columns.

    Input structure:
    - A black background contains exactly two blue 3x3 hollow door frames whose centers are red markers.
    - Each red marker has exactly one black 4-neighbor gap, and the next cell outward in the same direction is the exterior doorway point.
    - Additional grey horizontal or vertical 1-cell-thick segments act as blockers between the frames.

    Rule:
    - Treat every blue, red, and grey cell as blocked.
    - Between the two exterior doorway points, collect every black cell that lies on at least one shortest monotone 4-neighbor path, and also include both gap cells.
    - Color that whole path union yellow, then shear it: if the frame centers differ more in columns than rows, shift each covered row right by its offset from the topmost covered row; otherwise shift each covered column down by its offset from the leftmost covered column.
    - Drop sheared yellow pixels that leave the grid, restore vacated path cells to black, and keep existing frames and obstacles unchanged unless a sheared yellow pixel lands on top of them.

    Key insight:
    - The monotone shortest-path union is only an intermediate shape; the final answer comes from applying a direction chosen from the two door frames' relative orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    details = _analyze(g)
    path_mask = details["path_mask"]
    shifted = details["shifted"]

    out = g.copy()
    out[path_mask] = BLACK
    for r, c in shifted:
        out[r, c] = YELLOW
    return out


def _randint(lo: int, hi: int) -> int:
    return int(np.random.randint(lo, hi + 1))


def _buffer_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    buffered = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if _in_bounds(shape, rr, cc):
                    buffered.add((rr, cc))
    return buffered


def _frame_geometry(center: tuple[int, int], gap_dir: tuple[int, int]):
    r, c = center
    dr, dc = gap_dir
    frame = set()
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if (rr, cc) == center:
                continue
            if (rr, cc) == (r + dr, c + dc):
                continue
            frame.add((rr, cc))
    gap = (r + dr, c + dc)
    ext = (r + 2 * dr, c + 2 * dc)
    return frame, gap, ext


def _draw_frame(grid: np.ndarray, center: tuple[int, int], gap_dir: tuple[int, int]) -> dict[str, tuple[int, int]]:
    frame, gap, ext = _frame_geometry(center, gap_dir)
    for r, c in frame:
        grid[r, c] = BLUE
    grid[center] = RED
    return {"center": center, "gap": gap, "ext": ext}


def _sample_centers(h: int, w: int, by_rows: bool) -> tuple[tuple[int, int], tuple[int, int]] | None:
    for _ in range(200):
        c1 = (_randint(2, h - 3), _randint(2, w - 3))
        c2 = (_randint(2, h - 3), _randint(2, w - 3))
        dr = abs(c2[0] - c1[0])
        dc = abs(c2[1] - c1[1])
        if dr < 4 or dc < 4:
            continue
        if max(dr, dc) < 7:
            continue
        if by_rows and dc < dr + 2:
            continue
        if not by_rows and dr < dc + 2:
            continue
        if abs(c1[0] - c2[0]) <= 2 and abs(c1[1] - c2[1]) <= 2:
            continue
        return c1, c2
    return None


def _valid_gap_dirs(center: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = center
    dirs = []
    for dr, dc in DIRS4:
        if 0 <= r + 2 * dr < h and 0 <= c + 2 * dc < w:
            dirs.append((dr, dc))
    return dirs


def _segment_cells(
    anchor_box: tuple[int, int, int, int], orientation: str, length: int, offset_a: int, offset_b: int
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = anchor_box
    if orientation == "h":
        row = r0 + offset_a
        col = c0 + offset_b
        return [(row, col + k) for k in range(length)]
    row = r0 + offset_b
    col = c0 + offset_a
    return [(row + k, col) for k in range(length)]


def _place_obstacles(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    forbidden: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    if inner_h < 2 or inner_w < 2:
        return None

    obstacles: set[tuple[int, int]] = set()
    buffered_obstacles: set[tuple[int, int]] = set()
    target_segments = _randint(2, 4)

    for _ in range(target_segments):
        placed = False
        for _ in range(120):
            orientation = "h" if np.random.randint(0, 2) == 0 else "v"
            if orientation == "h":
                max_len = min(5, max(2, inner_w))
                if max_len < 2:
                    continue
                length = _randint(2, max_len)
                if inner_w - length < 0:
                    continue
                offset_a = _randint(1, inner_h)
                offset_b = _randint(1, inner_w - length + 1)
            else:
                max_len = min(5, max(2, inner_h))
                if max_len < 2:
                    continue
                length = _randint(2, max_len)
                if inner_h - length < 0:
                    continue
                offset_a = _randint(1, inner_w)
                offset_b = _randint(1, inner_h - length + 1)

            cells = _segment_cells(bbox, orientation, length, offset_a, offset_b)
            if any(cell in forbidden or cell in buffered_obstacles for cell in cells):
                continue
            placed = True
            for cell in cells:
                grid[cell] = GREY
                obstacles.add(cell)
            buffered_obstacles |= _buffer_cells(set(cells), grid.shape)
            break
        if not placed:
            return None

    return obstacles


def _quality_ok(grid: np.ndarray, details: dict[str, object], obstacles: set[tuple[int, int]]) -> bool:
    path_mask = details["path_mask"]
    shifted = details["shifted"]
    doors = details["doors"]
    path_size = int(path_mask.sum())
    if path_size < 10:
        return False

    start = doors[0]["ext"]
    end = doors[1]["ext"]
    r0, r1 = sorted([start[0], end[0]])
    c0, c1 = sorted([start[1], end[1]])
    rect_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    if path_size >= rect_area:
        return False

    kept_ratio = len(shifted) / path_size
    if kept_ratio < 0.72:
        return False

    overlap = sum(1 for cell in shifted if int(grid[cell]) in (BLUE, RED, GREY))
    if overlap > max(3, path_size // 4):
        return False

    if details["path_count"] < 2:
        return False

    if not any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in obstacles):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = _randint(17, 24)
    w = _randint(17, 24)
    by_rows = bool(np.random.randint(0, 2))
    centers = _sample_centers(h, w, by_rows)
    if centers is None:
        return None

    dirs1 = _valid_gap_dirs(centers[0], (h, w))
    dirs2 = _valid_gap_dirs(centers[1], (h, w))
    np.random.shuffle(dirs1)
    np.random.shuffle(dirs2)

    for gap1 in dirs1:
        frame1, gap_cell1, ext1 = _frame_geometry(centers[0], gap1)
        occupied1 = frame1 | {centers[0]}
        for gap2 in dirs2:
            frame2, gap_cell2, ext2 = _frame_geometry(centers[1], gap2)
            occupied2 = frame2 | {centers[1]}

            if occupied1 & occupied2:
                continue
            if gap_cell1 in occupied2 or gap_cell2 in occupied1:
                continue
            if ext1 in occupied2 or ext2 in occupied1:
                continue
            if ext1 == ext2 or ext1[0] == ext2[0] or ext1[1] == ext2[1]:
                continue
            if abs(ext1[0] - ext2[0]) < 3 or abs(ext1[1] - ext2[1]) < 3:
                continue

            grid = np.zeros((h, w), dtype=int)
            _draw_frame(grid, centers[0], gap1)
            _draw_frame(grid, centers[1], gap2)

            reserved = occupied1 | occupied2 | {gap_cell1, gap_cell2, ext1, ext2}
            forbidden = _buffer_cells(reserved, grid.shape)
            bbox = (
                min(ext1[0], ext2[0]),
                min(ext1[1], ext2[1]),
                max(ext1[0], ext2[0]),
                max(ext1[1], ext2[1]),
            )
            obstacles = _place_obstacles(grid, bbox, forbidden)
            if obstacles is None:
                continue

            try:
                details = _analyze(grid)
            except AssertionError:
                continue

            if bool(details["by_rows"]) != by_rows:
                continue
            if _quality_ok(grid, details, obstacles):
                return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
