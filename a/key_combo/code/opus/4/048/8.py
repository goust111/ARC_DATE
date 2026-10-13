import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
TILE_COLORS = (1, 3, 4, 6, 7, 8, 9)
MAX_RETRIES = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _line_ranges(line_indices: np.ndarray, limit: int) -> tuple[list[tuple[int, int]], int]:
    assert line_indices.size >= 3, "expected outer border plus at least one internal divider"
    assert int(line_indices[0]) == 0 and int(line_indices[-1]) == limit - 1, "grey frame must bound the grid"
    steps = np.diff(line_indices)
    pitch = int(steps[0])
    assert pitch >= 2, "cell interior must be non-empty"
    assert np.all(steps == pitch), "divider spacing must be uniform"
    ranges = [(int(line_indices[i] + 1), int(line_indices[i + 1])) for i in range(len(line_indices) - 1)]
    return ranges, pitch - 1


def _partition_grid(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    row_lines = np.where(np.all(grid == GREY, axis=1))[0]
    col_lines = np.where(np.all(grid == GREY, axis=0))[0]
    row_ranges, cell_h = _line_ranges(row_lines, grid.shape[0])
    col_ranges, cell_w = _line_ranges(col_lines, grid.shape[1])
    assert cell_h == cell_w, "90-degree rotation requires square cells"
    return row_ranges, col_ranges, cell_h


def _cell_view(
    grid: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
    rr: int,
    cc: int,
) -> np.ndarray:
    r0, r1 = row_ranges[rr]
    c0, c1 = col_ranges[cc]
    return grid[r0:r1, c0:c1]


def _find_reference_cell(
    grid: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
) -> tuple[int, int, int]:
    candidates = []
    for rr in range(len(row_ranges)):
        for cc in range(len(col_ranges)):
            cell = _cell_view(grid, row_ranges, col_ranges, rr, cc)
            extra = sorted(set(map(int, np.unique(cell))) - {BLACK, RED, GREY})
            if extra:
                assert len(extra) == 1, "reference cell must use a single tile color"
                candidates.append((rr, cc, extra[0]))
            else:
                assert np.all((cell == BLACK) | (cell == RED)), "non-reference cells may only contain black and red"
    assert len(candidates) == 1, "there must be exactly one reference cell"
    return candidates[0]


def _random_connected_shape() -> np.ndarray:
    for _ in range(200):
        area = random.randint(3, 5)
        cells = {(random.randint(0, 2), random.randint(0, 2))}
        while len(cells) < area:
            base_r, base_c = random.choice(tuple(cells))
            options = []
            for nr, nc in neighbors4(base_r, base_c):
                if 0 <= nr < 3 and 0 <= nc < 3 and (nr, nc) not in cells:
                    options.append((nr, nc))
            if not options:
                break
            cells.add(random.choice(options))
        if len(cells) != area:
            continue
        mask = np.zeros((3, 3), dtype=int)
        for r, c in cells:
            mask[r, c] = 1
        box = bbox(mask.astype(bool))
        assert box is not None
        sprite = crop_bbox(mask, box)
        if len(components_mask(sprite.astype(bool))) != 1:
            continue
        if len({np.rot90(sprite, -k).tobytes() for k in range(4)}) != 4:
            continue
        return sprite
    raise RuntimeError("failed to sample a non-symmetric connected tile")


def _centered_reference_cell(cell_size: int, tile_color: int) -> np.ndarray:
    for _ in range(200):
        sprite = _random_connected_shape()
        h, w = sprite.shape
        if h > cell_size or w > cell_size:
            continue
        cell = np.zeros((cell_size, cell_size), dtype=int)
        r0 = (cell_size - h) // 2
        c0 = (cell_size - w) // 2
        cell[r0 : r0 + h, c0 : c0 + w][sprite == 1] = tile_color
        if len(components_mask(cell == tile_color)) != 1:
            continue
        if len({np.rot90(cell, -k).tobytes() for k in range(4)}) != 4:
            continue
        return cell
    raise RuntimeError("failed to embed a valid reference tile")


def _place_markers(cell_size: int, count: int) -> np.ndarray:
    cell = np.zeros((cell_size, cell_size), dtype=int)
    if count == 0:
        return cell
    positions = random.sample([(r, c) for r in range(cell_size) for c in range(cell_size)], k=count)
    for r, c in positions:
        cell[r, c] = RED
    return cell


def _semantic_check(grid: np.ndarray) -> None:
    row_ranges, col_ranges, cell_size = _partition_grid(grid)
    ref_r, ref_c, tile_color = _find_reference_cell(grid, row_ranges, col_ranges)
    reference_cell = _cell_view(grid, row_ranges, col_ranges, ref_r, ref_c)

    assert tile_color in TILE_COLORS
    assert not np.any(reference_cell == RED)
    assert len(components_mask(reference_cell == tile_color)) == 1
    assert len({np.rot90(reference_cell, -k).tobytes() for k in range(4)}) == 4

    counts_seen = set()
    out = main(grid)
    assert np.array_equal(out[grid == GREY], grid[grid == GREY])
    assert np.array_equal(
        _cell_view(out, row_ranges, col_ranges, ref_r, ref_c),
        reference_cell,
    )

    rotated_variants = {np.rot90(reference_cell, -k).tobytes() for k in range(4)}
    assert len(rotated_variants) == 4

    for rr in range(len(row_ranges)):
        for cc in range(len(col_ranges)):
            if (rr, cc) == (ref_r, ref_c):
                continue
            inp_cell = _cell_view(grid, row_ranges, col_ranges, rr, cc)
            out_cell = _cell_view(out, row_ranges, col_ranges, rr, cc)
            count = int(np.sum(inp_cell == RED))
            assert 0 <= count <= 3
            counts_seen.add(count)
            assert np.all((inp_cell == BLACK) | (inp_cell == RED))
            assert not np.any(out_cell == RED)
            assert np.array_equal(out_cell, np.rot90(reference_cell, -count))

    assert cell_size in (4, 5)
    assert counts_seen == {0, 1, 2, 3}
    assert not np.array_equal(grid, out)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the reference tile into every grey-bounded cell, rotating it by the red marker count in that cell.

    Input structure:
    - Black background cells are partitioned by one-pixel grey divider lines into a uniform square grid of cells.
    - Exactly one cell contains a connected non-red tile in a single non-black, non-grey color; this is the reference tile.
    - Every other cell contains only black pixels and zero to three red marker pixels.

    Rule:
    - Read the square cell lattice from the grey divider lines and locate the unique reference-tile cell.
    - For each non-reference cell, count its red pixels: 0 -> keep the reference orientation, 1 -> rotate 90 degrees clockwise, 2 -> rotate 180 degrees, 3 -> rotate 270 degrees clockwise.
    - Apply the rotation in the cell's own coordinate system around the cell center, replace that cell with the rotated tile on black, remove all red markers, and leave the grey lines plus the original reference cell unchanged.

    Key insight:
    - The red pixels are not part of the output pattern; only their count matters, and that count is a discrete code for which quarter-turn of the reference tile should fill the cell.
    """
    g = np.asarray(grid, dtype=int)
    row_ranges, col_ranges, _cell_size = _partition_grid(g)
    ref_r, ref_c, _tile_color = _find_reference_cell(g, row_ranges, col_ranges)
    reference_cell = _cell_view(g, row_ranges, col_ranges, ref_r, ref_c).copy()

    out = g.copy()
    for rr in range(len(row_ranges)):
        for cc in range(len(col_ranges)):
            if (rr, cc) == (ref_r, ref_c):
                continue
            cell = _cell_view(g, row_ranges, col_ranges, rr, cc)
            count = int(np.sum(cell == RED))
            assert 0 <= count <= 3, "marker counts must encode one of four quarter-turns"
            rotated = np.rot90(reference_cell, -count)
            r0, r1 = row_ranges[rr]
            c0, c1 = col_ranges[cc]
            out[r0:r1, c0:c1] = rotated
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        cell_size = random.choice((4, 5))
        max_cells = 29 // (cell_size + 1)
        row_count = random.randint(2, max_cells)
        col_count = random.randint(3, max_cells)
        if row_count * col_count < 6:
            continue

        h = row_count * (cell_size + 1) + 1
        w = col_count * (cell_size + 1) + 1
        grid = np.zeros((h, w), dtype=int)
        grid[:: cell_size + 1, :] = GREY
        grid[:, :: cell_size + 1] = GREY

        row_ranges, col_ranges, _ = _partition_grid(grid)
        ref_r = random.randrange(row_count)
        ref_c = random.randrange(col_count)
        tile_color = random.choice(TILE_COLORS)
        reference_cell = _centered_reference_cell(cell_size, tile_color)

        r0, r1 = row_ranges[ref_r]
        c0, c1 = col_ranges[ref_c]
        grid[r0:r1, c0:c1] = reference_cell

        counts = [0, 1, 2, 3]
        extra_cells = row_count * col_count - 1 - len(counts)
        counts.extend(random.randint(0, 3) for _ in range(extra_cells))
        random.shuffle(counts)

        idx = 0
        for rr in range(row_count):
            for cc in range(col_count):
                if (rr, cc) == (ref_r, ref_c):
                    continue
                r0, r1 = row_ranges[rr]
                c0, c1 = col_ranges[cc]
                grid[r0:r1, c0:c1] = _place_markers(cell_size, counts[idx])
                idx += 1

        _semantic_check(grid)
        return grid

    raise RuntimeError("failed to generate a valid rotation-coded tiling input")
