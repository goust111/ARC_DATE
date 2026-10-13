from __future__ import annotations

import random
from collections import Counter

import numpy as np


RED = 2
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = tuple(range(10))
GENERATION_ATTEMPTS = 400


def _mirror_col(width: int, col: int) -> int:
    return width - 1 - col


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _blob_mask(height: int, width: int, size: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    frontier = {start}

    while int(mask.sum()) < size:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        r, c = cell
        options = []
        for nr, nc in _neighbors4(r, c, height, width):
            if not mask[nr, nc]:
                options.append((nr, nc))
        if not options:
            frontier.update(tuple(map(int, rc)) for rc in np.argwhere(mask))
            continue
        pick = random.choice(options)
        mask[pick] = True
        frontier.add(pick)
        frontier.add(cell)

    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return mask.copy()


def _place_mirrored_blob(
    grid: np.ndarray,
    occupied: np.ndarray,
    half_width: int,
    color: int,
) -> bool:
    height, width = grid.shape
    box_h = random.randint(1, min(4, height))
    box_w = random.randint(1, min(4, half_width))
    max_size = min(box_h * box_w, 6)
    min_size = 1 if max_size == 1 else 2
    size = random.randint(min_size, max_size)
    mask = _blob_mask(box_h, box_w, size)

    rows = list(range(0, height - box_h + 1))
    cols = list(range(0, half_width - box_w + 1))
    random.shuffle(rows)
    random.shuffle(cols)
    for r0 in rows:
        for c0 in cols:
            local = np.argwhere(mask)
            coords = [(int(r0 + r), int(c0 + c)) for r, c in local]
            mirror_coords = [(r, _mirror_col(width, c)) for r, c in coords]
            if any(occupied[r, c] for r, c in coords):
                continue
            if any(occupied[r, c] for r, c in mirror_coords):
                continue
            for r, c in coords:
                grid[r, c] = color
                occupied[r, c] = True
            for r, c in mirror_coords:
                grid[r, c] = color
                occupied[r, c] = True
            return True
    return False


def _build_symmetric_base() -> np.ndarray | None:
    height = random.randint(9, 20)
    half_width = random.randint(4, 12)
    width = 2 * half_width + 1

    bg_choices = [color for color in PALETTE if color != RED]
    bg = random.choice(bg_choices)
    motif_pool = [color for color in PALETTE if color not in {bg, RED}]
    motif_count = random.randint(2, 4)
    motif_colors = random.sample(motif_pool, motif_count)

    grid = np.full((height, width), bg, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    mid = half_width

    axis_segments = random.randint(0, 3)
    for _ in range(axis_segments):
        length = random.randint(1, min(4, height))
        start = random.randint(0, height - length)
        color = random.choice(motif_colors)
        grid[start : start + length, mid] = color
        occupied[start : start + length, mid] = True

    wanted = random.randint(5, 10)
    placed = 0
    for _ in range(100):
        color = random.choice(motif_colors)
        if _place_mirrored_blob(grid, occupied, half_width, color):
            placed += 1
        if placed >= wanted:
            break

    if placed < 4:
        return None

    used_colors = set(map(int, np.unique(grid))) - {bg}
    if len(used_colors) < 2:
        return None

    fill_ratio = float(np.count_nonzero(grid != bg)) / float(grid.size)
    if not (0.10 <= fill_ratio <= 0.55):
        return None
    return grid


def _candidate_dirs(
    grid: np.ndarray,
    cell: tuple[int, int],
    blocked: set[tuple[int, int]],
    marker_rows: set[int],
) -> list[tuple[int, int]]:
    height, width = grid.shape
    mid = width // 2
    r, c = cell
    dirs = DIRS4[:]
    random.shuffle(dirs)
    options = []
    for dr, dc in dirs:
        mr, mc = r + dr, c + dc
        if not (0 <= mr < height and 0 <= mc < width):
            continue
        if mc == mid:
            continue
        marker = (mr, mc)
        if mr in marker_rows:
            continue
        marker_mirror = (mr, _mirror_col(width, mc))
        if marker in blocked or marker_mirror in blocked:
            continue
        if marker == cell:
            continue
        if any(nei in blocked for nei in _neighbors4(mr, mc, height, width)):
            continue
        if any(nei in blocked for nei in _neighbors4(r, c, height, width)):
            continue
        options.append(marker)
    return options


def _choose_pairs(base: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    height, width = base.shape
    mid = width // 2
    wanted = random.randint(1, 3)
    bg = Counter(map(int, base.ravel())).most_common(1)[0][0]

    prefer_fg = random.random() < 0.5
    candidates = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if c != mid
    ]
    random.shuffle(candidates)
    candidates.sort(key=lambda pt: 0 if (int(base[pt]) != bg) == prefer_fg else 1)

    blocked: set[tuple[int, int]] = set()
    marker_rows: set[int] = set()
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for cell in candidates:
        if len(pairs) >= wanted:
            break
        mirror = (cell[0], _mirror_col(width, cell[1]))
        if cell in blocked or mirror in blocked:
            continue
        options = _candidate_dirs(base, cell, blocked, marker_rows)
        if not options:
            continue
        marker = random.choice(options)
        marker_mirror = (marker[0], _mirror_col(width, marker[1]))
        blocked.update({cell, mirror, marker, marker_mirror})
        pairs.append((cell, marker))
        marker_rows.add(marker[0])

    if len(pairs) != wanted:
        return None
    return pairs


def _expected_output(
    base: np.ndarray,
    pairs: list[tuple[tuple[int, int], tuple[int, int]]],
    intruder_colors: list[int],
) -> np.ndarray:
    out = base.copy()
    mid = out.shape[1] // 2
    for (_, marker), color in zip(pairs, intruder_colors, strict=False):
        out[marker[0], mid] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a nearly left-right symmetric grid by copying mirror colors back onto anomaly cells and moving each intruder color onto the center column.

    Input structure:
    - The background is the dominant color, and the overall drawing is almost mirror-symmetric across the vertical center column.
    - One to three non-axis cells are intruders: each differs from its horizontal mirror partner and is adjacent in the 4-neighborhood to a red marker pixel.
    - Red appears only in those marker pixels, while the rest of the visible pattern uses ordinary background and motif colors.

    Rule:
    - For each red marker, find the adjacent non-red cell whose value disagrees with its mirror partner; this is the intruder linked to that marker.
    - Restore the symmetric picture by replacing every intruder cell and every red marker cell with the color from its mirror position.
    - Then place the intruder's original color on the vertical center column in the marker's row, overwriting whatever was previously on that axis cell.

    Key insight:
    - The red pixels do not belong to the symmetric motif; they point to the true anomaly cells, and the missing information needed to repair both marker and intruder positions already exists at their mirror locations.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape
    assert width % 2 == 1, "the mirror axis must be a center column"
    mid = width // 2

    red_cells = [tuple(map(int, rc)) for rc in np.argwhere(g == RED)]
    assert 1 <= len(red_cells) <= 3, "expected one to three red marker pixels"
    assert len({r for r, _ in red_cells}) == len(red_cells), "marker rows must be unique"

    counts = Counter(map(int, g.ravel()))
    intruders: dict[tuple[int, int], tuple[int, int]] = {}

    for marker in red_cells:
        mr, mc = marker
        assert mc != mid, "markers may not lie on the symmetry axis"
        candidates = []
        for nr, nc in _neighbors4(mr, mc, height, width):
            if nc == mid or int(g[nr, nc]) == RED:
                continue
            mirror_nc = _mirror_col(width, nc)
            if int(g[nr, nc]) == int(g[nr, mirror_nc]):
                continue
            candidates.append((nr, nc))

        assert candidates, f"marker at {marker} must touch an intruder"
        if len(candidates) > 1:
            candidates.sort(
                key=lambda pt: (
                    counts[int(g[pt])],
                    0 if counts[int(g[pt])] <= counts[int(g[pt[0], _mirror_col(width, pt[1])])] else 1,
                    pt[0],
                    pt[1],
                )
            )
        intruders[marker] = candidates[0]

    assert len(set(intruders.values())) == len(intruders), "each marker must map to a distinct intruder"

    repaired = g.copy()
    moved_colors: list[tuple[int, int]] = []
    for marker, intruder in intruders.items():
        ir, ic = intruder
        moved_colors.append((marker[0], int(g[ir, ic])))
        repaired[ir, ic] = int(g[ir, _mirror_col(width, ic)])

    for mr, mc in red_cells:
        repaired[mr, mc] = int(repaired[mr, _mirror_col(width, mc)])

    out = repaired.copy()
    for row, color in moved_colors:
        out[row, mid] = color

    assert np.array_equal(out, np.fliplr(out)), "output must be mirror-symmetric"
    return out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        base = _build_symmetric_base()
        if base is None:
            continue

        bg = Counter(map(int, base.ravel())).most_common(1)[0][0]
        used = set(map(int, np.unique(base)))
        available = [color for color in PALETTE if color not in used and color != RED]
        pairs = _choose_pairs(base)
        if pairs is None or len(available) < len(pairs):
            continue

        intruder_colors = random.sample(available, len(pairs))
        inp = base.copy()
        for (intruder, marker), color in zip(pairs, intruder_colors, strict=False):
            inp[intruder] = int(color)
            inp[marker] = RED

        out = _expected_output(base, pairs, intruder_colors)
        if Counter(map(int, inp.ravel())).most_common(1)[0][0] != bg:
            continue
        if not np.array_equal(main(inp), out):
            continue
        return inp

    raise RuntimeError("failed to generate a valid symmetric-anomaly ARC input")
