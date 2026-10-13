import numpy as np
from collections import deque


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

SEED_COLORS = (BLUE, RED, GREEN)
SEPARATOR = -1


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _seed_rows(grid: np.ndarray) -> list[tuple[int, int]]:
    seeds: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if np.all(row == row[0]) and int(row[0]) in SEED_COLORS:
            seeds.append((r, int(row[0])))
    assert 2 <= len(seeds) <= 3, f"expected 2-3 seed rows, got {seeds}"
    assert len({color for _r, color in seeds}) == len(seeds), "seed colors must be distinct"
    return seeds


def _row_owner(seed_rows: list[tuple[int, int]], h: int) -> np.ndarray:
    rows = np.array([r for r, _color in seed_rows], dtype=int)
    colors = np.array([color for _r, color in seed_rows], dtype=int)
    owner = np.full(h, SEPARATOR, dtype=int)
    for r in range(h):
        if r in rows:
            owner[r] = int(np.where(rows == r)[0][0])
            continue
        dists = np.abs(rows - r)
        best = int(dists.min())
        tied = np.flatnonzero(dists == best)
        if tied.size == 1:
            owner[r] = int(tied[0])
        else:
            tied = tied[np.argsort(rows[tied])]
            tied_colors = {int(colors[i]) for i in tied}
            if len(tied_colors) == 1:
                owner[r] = int(tied[0])
            else:
                owner[r] = SEPARATOR
    return owner


def _naive_reflect_without_partitions(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    seeds = _seed_rows(out)
    h, _w = out.shape
    for seed_r, _color in seeds:
        for r in range(max(0, seed_r - 3), seed_r):
            rr = 2 * seed_r - r
            if not (0 <= rr < h):
                continue
            cols = np.flatnonzero(out[r] == ORANGE)
            out[rr, cols] = ORANGE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each partition's orange pattern across its own full-width seed bar, and turn equal-distance color boundaries into gray separators.

    Input structure:
    - The grid has a black background plus 2-3 full-width horizontal seed bars, each one row thick and a distinct color.
    - Orange pixels form one or more small 4-connected patterns only in the 1-3 rows directly above a seed bar.
    - Rows below every seed bar are otherwise black in the input.

    Rule:
    - Detect the seed-bar rows and order them from top to bottom.
    - Assign every non-seed row to its nearest seed row; if a row is exactly tied between different-colored seeds, make that entire row gray.
    - Within each seed's own row partition, keep the original orange pattern above the seed and copy it to the mirror-symmetric cells below the seed row.
    - Mirror copies are only placed on rows that belong to the same partition; separator rows stay fully gray.

    Key insight:
    - Each colored bar has a double role: it is both the nearest-row partition seed and the mirror axis for the orange pattern inside that partition.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    seeds = _seed_rows(g)
    owner = _row_owner(seeds, g.shape[0])

    for r in range(g.shape[0]):
        if owner[r] == SEPARATOR:
            out[r, :] = GREY

    for seed_idx, (seed_r, _color) in enumerate(seeds):
        for r in range(max(0, seed_r - 3), seed_r):
            if owner[r] != seed_idx:
                continue
            rr = 2 * seed_r - r
            if not (0 <= rr < g.shape[0]) or owner[rr] != seed_idx:
                continue
            cols = np.flatnonzero(g[r] == ORANGE)
            out[rr, cols] = ORANGE

    return out


GRID_ATTEMPTS = 400


def _random_connected_shape(height: int, width: int, area: int, required: tuple[int, int] | None) -> np.ndarray | None:
    cells = set()
    if required is not None:
        if not (0 <= required[0] < height and 0 <= required[1] < width):
            return None
        cells.add(required)
    else:
        cells.add((int(np.random.randint(0, height)), int(np.random.randint(0, width))))

    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        frontier = list(dict.fromkeys(frontier))
        nr, nc = frontier[int(np.random.randint(0, len(frontier)))]
        cells.add((nr, nc))

    coords = np.array(sorted(cells), dtype=int)
    return coords


def _sample_boxes(width: int, count: int, widths: list[int]) -> list[tuple[int, int]] | None:
    for _ in range(120):
        starts = []
        intervals = []
        for box_w in widths:
            c0 = int(np.random.randint(0, width - box_w + 1))
            intervals.append((c0, c0 + box_w - 1))
        intervals.sort()
        ok = True
        for i in range(1, len(intervals)):
            if intervals[i][0] - intervals[i - 1][1] < 2:
                ok = False
                break
        if ok:
            starts = intervals
            return starts
    return None


def _build_pattern_cells(
    seed_r: int,
    seed_idx: int,
    owner: np.ndarray,
    width: int,
    occupied: set[tuple[int, int]],
    required_depth: int | None,
) -> tuple[list[np.ndarray], bool] | None:
    allowed_rows = [r for r in range(max(0, seed_r - 3), seed_r) if owner[r] == seed_idx]
    if not allowed_rows:
        return None

    must_hit_required = required_depth is not None
    required_row = None if required_depth is None else seed_r - required_depth
    if required_row is not None and required_row not in allowed_rows:
        return None

    max_components = 2 if width >= 12 else 1
    comp_count = 1 + int(max_components == 2 and np.random.rand() < 0.4)

    for _ in range(160):
        widths = [int(np.random.randint(2, min(6, width) + 1)) for _ in range(comp_count)]
        boxes = _sample_boxes(width, comp_count, widths)
        if boxes is None:
            continue

        components: list[np.ndarray] = []
        used = set()
        hit_required = False
        ok = True

        required_box = None
        if required_row is not None:
            candidates = [i for i, (c0, c1) in enumerate(boxes) if c1 - c0 + 1 >= 2]
            required_box = int(np.random.choice(candidates)) if candidates else None
            if required_box is None:
                continue

        for idx, (c0, c1) in enumerate(boxes):
            box_w = c1 - c0 + 1
            box_h = len(allowed_rows)
            area_hi = min(6, box_h * box_w)
            area_lo = 3
            if area_hi < area_lo:
                ok = False
                break

            required = None
            if required_box == idx:
                req_local_r = allowed_rows.index(required_row)
                req_local_c = int(np.random.randint(0, box_w))
                required = (req_local_r, req_local_c)

            shape = _random_connected_shape(box_h, box_w, int(np.random.randint(area_lo, area_hi + 1)), required)
            if shape is None:
                ok = False
                break

            global_cells = np.array([[allowed_rows[r], c0 + c] for r, c in shape], dtype=int)
            for cell in map(tuple, global_cells):
                if cell in occupied or cell in used:
                    ok = False
                    break
            if not ok:
                break

            components.append(global_cells)
            used |= set(map(tuple, global_cells))
            if required_box == idx and any(r == required_row for r, _c in map(tuple, global_cells)):
                hit_required = True

        if ok and (not must_hit_required or hit_required):
            return components, hit_required

    return None


def _validate_input(grid: np.ndarray) -> None:
    seeds = _seed_rows(grid)
    h, w = grid.shape
    owner = _row_owner(seeds, h)
    orange_mask = grid == ORANGE
    orange_comps = _components4(orange_mask)
    assert orange_comps, "expected orange patterns"

    for comp in orange_comps:
        assert 3 <= len(comp) <= 6, f"orange component area out of range: {len(comp)}"
        seed_ids = set()
        for r, c in map(tuple, comp):
            assert 0 <= r < h and 0 <= c < w
            assert grid[r, c] == ORANGE
            matches = []
            for seed_idx, (seed_r, _color) in enumerate(seeds):
                if r < seed_r and 1 <= seed_r - r <= 3 and owner[r] == seed_idx:
                    matches.append(seed_idx)
            assert len(matches) == 1, f"orange cell {(r, c)} does not belong to exactly one seed"
            seed_ids.add(matches[0])
        assert len(seed_ids) == 1, "one orange component must belong to exactly one partition"

    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out == ORANGE), "expected orange to remain present"
    assert np.any(out != grid), "transformation must change the grid"

    even_gap_count = 0
    mirrored_below = 0
    for i in range(len(seeds) - 1):
        gap = seeds[i + 1][0] - seeds[i][0]
        if gap % 2 == 0:
            even_gap_count += 1
            mid = (seeds[i][0] + seeds[i + 1][0]) // 2
            assert np.all(out[mid] == GREY), "even-gap midpoint must be gray"

    for seed_idx, (seed_r, _color) in enumerate(seeds):
        for r in range(max(0, seed_r - 3), seed_r):
            if owner[r] != seed_idx:
                continue
            rr = 2 * seed_r - r
            if not (0 <= rr < h):
                continue
            cols = np.flatnonzero(grid[r] == ORANGE)
            if owner[rr] == seed_idx:
                if cols.size:
                    assert np.all(out[rr, cols] == ORANGE), "mirror copy missing"
                    mirrored_below += int(cols.size)
            elif cols.size:
                assert owner[rr] == SEPARATOR and np.all(out[rr] == GREY), "separator handling incorrect"

    assert mirrored_below > 0, "expected at least one mirrored orange pixel below a seed"

    naive = _naive_reflect_without_partitions(grid)
    assert not np.array_equal(naive, out), "instance degenerates to naive mirror without partitions"
    assert h <= 30 and w <= 30


def _gen_attempt() -> np.ndarray | None:
    seed_count = 3 if np.random.rand() < 0.6 else 2
    colors = list(np.random.permutation(SEED_COLORS)[:seed_count])

    first_gap = int(np.random.choice([4, 6]))
    gaps: list[int] = [first_gap]
    if seed_count == 3:
        second_gap_choices = [4, 5, 6, 7, 8]
        if np.random.rand() < 0.55:
            second_gap_choices = [5, 7]
        gaps.append(int(np.random.choice(second_gap_choices)))

    top_margin = int(np.random.randint(first_gap // 2, 5))
    bottom_margin = int(np.random.randint(3, 7))
    seed_rows = [top_margin]
    for gap in gaps:
        seed_rows.append(seed_rows[-1] + gap)

    h = seed_rows[-1] + 1 + bottom_margin
    if h < 14:
        h = 14
    if h > 28:
        return None

    w = int(np.random.randint(10, min(18, h) + 1))
    if h <= w:
        return None

    grid = np.zeros((h, w), dtype=int)
    for r, color in zip(seed_rows, colors):
        grid[r, :] = color

    owner = _row_owner(list(zip(seed_rows, colors)), h)
    occupied = set((r, c) for r in seed_rows for c in range(w))

    required_depths = {0: first_gap // 2}
    placed_required = False
    for seed_idx, seed_r in enumerate(seed_rows):
        forced_depth = required_depths.get(seed_idx)
        built = _build_pattern_cells(seed_r, seed_idx, owner, w, occupied, forced_depth)
        if built is None:
            return None
        components, hit_required = built
        if seed_idx == 0:
            placed_required = hit_required
        for comp in components:
            for r, c in map(tuple, comp):
                grid[r, c] = ORANGE
            occupied |= set(map(tuple, comp))

    if not placed_required:
        return None

    try:
        _validate_input(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GRID_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
