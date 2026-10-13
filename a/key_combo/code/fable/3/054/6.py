import random

import numpy as np


BG = 0
LATTICE = 8
PALETTE = [1, 2, 3, 4, 5, 6, 7, 9]
DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_lattice(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, list[int], list[int]]] = []

    for color in map(int, np.unique(grid)):
        if color == bg:
            continue

        row_lines = [r for r in range(h) if np.all(grid[r, :] == color)]
        col_lines = [c for c in range(w) if np.all(grid[:, c] == color)]
        if len(row_lines) < 3 or len(col_lines) < 3:
            continue
        if row_lines[0] != 0 or row_lines[-1] != h - 1:
            continue
        if col_lines[0] != 0 or col_lines[-1] != w - 1:
            continue

        row_steps = np.diff(row_lines)
        col_steps = np.diff(col_lines)
        if row_steps.size == 0 or col_steps.size == 0:
            continue
        if len(set(map(int, row_steps))) != 1 or len(set(map(int, col_steps))) != 1:
            continue
        if int(row_steps[0]) <= 1 or int(col_steps[0]) <= 1:
            continue

        score = len(row_lines) + len(col_lines)
        candidates.append((score, color, row_lines, col_lines))

    assert candidates, "failed to detect the lattice color"
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, color, row_lines, col_lines = candidates[0]
    return int(color), row_lines, col_lines


def _cavity_slices(
    row_lines: list[int], col_lines: list[int], chamber_r: int, chamber_c: int
) -> tuple[slice, slice]:
    return (
        slice(row_lines[chamber_r] + 1, row_lines[chamber_r + 1]),
        slice(col_lines[chamber_c] + 1, col_lines[chamber_c + 1]),
    )


def _parse_chambers(
    grid: np.ndarray,
) -> tuple[int, list[int], list[int], tuple[int, int, int, np.ndarray], list[tuple[int, int, int]]]:
    bg = _get_bg(grid)
    _lattice_color, row_lines, col_lines = _find_lattice(grid, bg)

    sprite_info = None
    markers: list[tuple[int, int, int]] = []

    for chamber_r in range(len(row_lines) - 1):
        for chamber_c in range(len(col_lines) - 1):
            rs, cs = _cavity_slices(row_lines, col_lines, chamber_r, chamber_c)
            patch = grid[rs, cs]
            values = patch[patch != bg]
            if values.size == 0:
                continue

            colors = np.unique(values)
            assert colors.size == 1, "each chamber should be monochrome apart from the black background"
            color = int(colors[0])
            if values.size == 1:
                markers.append((chamber_r, chamber_c, color))
            else:
                assert sprite_info is None, "expected exactly one sprite chamber"
                sprite_info = (chamber_r, chamber_c, color, patch.copy())

    assert sprite_info is not None, "missing source sprite chamber"
    assert markers, "missing marker chambers"
    return bg, row_lines, col_lines, sprite_info, markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal parity-controlled reflections of the source sprite inside the marked lattice chambers.

    Input structure:
    - A teal one-cell-thick lattice, including the outer border, partitions the canvas into equal black chambers.
    - Exactly one chamber contains a monochrome sprite that is asymmetric under both top-bottom and left-right reflection.
    - Several other chambers are empty except for one isolated marker pixel, with a distinct color per marked chamber.

    Rule:
    - Treat the sprite chamber as the origin in chamber coordinates.
    - For each marked chamber, compare its row and column offsets from that origin: an odd row offset flips the sprite top-to-bottom, an odd column offset flips it left-to-right, and even offsets keep that axis unchanged.
    - Replace the marker with the transformed sprite recolored to the marker color. Keep the source chamber and the lattice unchanged, and leave unmarked chambers black.

    Key insight:
    - The markers do not encode different shapes; they only choose which parity-based mirror image of the single source sprite should become visible and which color it should use.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, row_lines, col_lines, sprite_info, markers = _parse_chambers(g)
    sprite_r, sprite_c, _sprite_color, sprite_patch = sprite_info

    out = g.copy()
    for chamber_r, chamber_c, marker_color in markers:
        rendered = sprite_patch.copy()
        if abs(chamber_r - sprite_r) % 2 == 1:
            rendered = np.flipud(rendered)
        if abs(chamber_c - sprite_c) % 2 == 1:
            rendered = np.fliplr(rendered)

        recolored = np.full_like(rendered, bg)
        recolored[rendered != bg] = int(marker_color)

        rs, cs = _cavity_slices(row_lines, col_lines, chamber_r, chamber_c)
        out[rs, cs] = recolored

    return out


def _grow_connected_mask(height: int, width: int, target_area: int) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True

    while int(mask.sum()) < target_area:
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            for dr, dc in DIR4:
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        nr, nc = random.choice(frontier)
        mask[nr, nc] = True

    return mask


def _make_sprite_mask(height: int, width: int) -> np.ndarray | None:
    for _ in range(200):
        sub_h = random.randint(2, height)
        sub_w = random.randint(2, width)
        max_area = min(sub_h * sub_w - 1, max(4, (3 * sub_h * sub_w) // 5))
        if max_area < 4:
            continue
        target_area = random.randint(4, max_area)
        submask = _grow_connected_mask(sub_h, sub_w, target_area)
        if submask is None:
            continue
        if np.ptp(np.argwhere(submask)[:, 0]) == 0 or np.ptp(np.argwhere(submask)[:, 1]) == 0:
            continue

        mask = np.zeros((height, width), dtype=bool)
        offset_r = random.randint(0, height - sub_h)
        offset_c = random.randint(0, width - sub_w)
        mask[offset_r : offset_r + sub_h, offset_c : offset_c + sub_w] = submask

        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask

    return None


def _make_base_grid(
    chamber_rows: int, chamber_cols: int, cell_h: int, cell_w: int
) -> tuple[np.ndarray, list[int], list[int]]:
    grid_h = chamber_rows * cell_h + chamber_rows + 1
    grid_w = chamber_cols * cell_w + chamber_cols + 1
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    row_lines = [idx * (cell_h + 1) for idx in range(chamber_rows + 1)]
    col_lines = [idx * (cell_w + 1) for idx in range(chamber_cols + 1)]
    grid[row_lines, :] = LATTICE
    grid[:, col_lines] = LATTICE
    return grid, row_lines, col_lines


def _pick_layout() -> tuple[int, int, int, int]:
    for _ in range(100):
        chamber_rows = random.randint(4, 5)
        chamber_cols = random.randint(4, 5)
        max_cell_h = min(5, (30 - (chamber_rows + 1)) // chamber_rows)
        max_cell_w = min(5, (30 - (chamber_cols + 1)) // chamber_cols)
        if max_cell_h < 4 or max_cell_w < 4:
            continue
        cell_h = random.randint(4, max_cell_h)
        cell_w = random.randint(4, max_cell_w)
        return chamber_rows, chamber_cols, cell_h, cell_w
    raise RuntimeError("failed to sample a valid lattice layout")


def _pick_marker_chambers(
    chamber_rows: int, chamber_cols: int, sprite_rc: tuple[int, int], total_markers: int
) -> list[tuple[int, int]] | None:
    sprite_r, sprite_c = sprite_rc
    by_parity: dict[tuple[int, int], list[tuple[int, int]]] = {
        (0, 0): [],
        (1, 0): [],
        (0, 1): [],
        (1, 1): [],
    }

    all_chambers: list[tuple[int, int]] = []
    for chamber_r in range(chamber_rows):
        for chamber_c in range(chamber_cols):
            if (chamber_r, chamber_c) == sprite_rc:
                continue
            all_chambers.append((chamber_r, chamber_c))
            parity = (abs(chamber_r - sprite_r) % 2, abs(chamber_c - sprite_c) % 2)
            by_parity[parity].append((chamber_r, chamber_c))

    chosen: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for parity in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        options = [rc for rc in by_parity[parity] if rc not in used]
        if not options:
            return None
        rc = random.choice(options)
        chosen.append(rc)
        used.add(rc)

    remaining = [rc for rc in all_chambers if rc not in used]
    random.shuffle(remaining)
    extra = total_markers - len(chosen)
    chosen.extend(remaining[:extra])
    random.shuffle(chosen)
    return chosen


def _render_expected(
    grid: np.ndarray,
    row_lines: list[int],
    col_lines: list[int],
    sprite_rc: tuple[int, int],
    marker_specs: list[tuple[int, int, int]],
) -> np.ndarray:
    out = grid.copy()
    sprite_r, sprite_c = sprite_rc
    sprite_rs, sprite_cs = _cavity_slices(row_lines, col_lines, sprite_r, sprite_c)
    sprite_patch = grid[sprite_rs, sprite_cs].copy()

    for chamber_r, chamber_c, marker_color in marker_specs:
        rendered = sprite_patch.copy()
        if abs(chamber_r - sprite_r) % 2 == 1:
            rendered = np.flipud(rendered)
        if abs(chamber_c - sprite_c) % 2 == 1:
            rendered = np.fliplr(rendered)

        patch = np.full_like(rendered, BG)
        patch[rendered != BG] = int(marker_color)
        rs, cs = _cavity_slices(row_lines, col_lines, chamber_r, chamber_c)
        out[rs, cs] = patch

    return out


def _gen_attempt() -> np.ndarray | None:
    chamber_rows, chamber_cols, cell_h, cell_w = _pick_layout()
    grid, row_lines, col_lines = _make_base_grid(chamber_rows, chamber_cols, cell_h, cell_w)

    sprite_color = random.choice(PALETTE)
    marker_palette = [color for color in PALETTE if color != sprite_color]
    max_markers = min(len(marker_palette), chamber_rows * chamber_cols - 1)
    total_markers = random.randint(4, max_markers)

    sprite_rc = (random.randrange(chamber_rows), random.randrange(chamber_cols))
    marker_chambers = _pick_marker_chambers(chamber_rows, chamber_cols, sprite_rc, total_markers)
    if marker_chambers is None:
        return None

    sprite_mask = _make_sprite_mask(cell_h, cell_w)
    if sprite_mask is None:
        return None

    sprite_rs, sprite_cs = _cavity_slices(row_lines, col_lines, sprite_rc[0], sprite_rc[1])
    sprite_patch = np.full((cell_h, cell_w), BG, dtype=int)
    sprite_patch[sprite_mask] = int(sprite_color)
    grid[sprite_rs, sprite_cs] = sprite_patch

    marker_colors = random.sample(marker_palette, total_markers)
    marker_specs: list[tuple[int, int, int]] = []
    for color, (chamber_r, chamber_c) in zip(marker_colors, marker_chambers):
        rs, cs = _cavity_slices(row_lines, col_lines, chamber_r, chamber_c)
        marker_r = random.randrange(cell_h)
        marker_c = random.randrange(cell_w)
        grid[rs.start + marker_r, cs.start + marker_c] = int(color)
        marker_specs.append((chamber_r, chamber_c, int(color)))

    expected = _render_expected(grid, row_lines, col_lines, sprite_rc, marker_specs)
    try:
        observed = main(grid)
    except Exception:
        return None
    if not np.array_equal(observed, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for mirror-hall reflection puzzle")
