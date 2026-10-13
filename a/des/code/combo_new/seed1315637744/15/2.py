from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
SEPARATOR_COLOR = 5
PALETTE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIMENSION_OPTIONS = [(2, 2), (2, 3), (3, 2), (2, 4), (4, 2)]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GENERATION_ATTEMPTS = 400


def _mode_color(values: np.ndarray) -> int:
    flat = values.reshape(-1)
    colors, counts = np.unique(flat, return_counts=True)
    return int(colors[np.argmax(counts)])


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in NEIGHBORS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc, height, width):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(cells)
    return parts


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _expand_box(box: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    return box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _build_palette(palette: np.ndarray, separator: int) -> np.ndarray:
    n_rows, n_cols = palette.shape
    patch = np.full((2 * n_rows + 1, 2 * n_cols + 1), separator, dtype=int)
    for r in range(n_rows):
        for c in range(n_cols):
            patch[2 * r + 1, 2 * c + 1] = int(palette[r, c])
    return patch


def _make_motif_patch(
    shape_cells: set[tuple[int, int]],
    marker_cell: tuple[int, int],
    body_color: int,
    marker_color: int,
) -> np.ndarray:
    height = 1 + max(r for r, _ in shape_cells)
    width = 1 + max(c for _, c in shape_cells)
    patch = np.full((height, width), BG_COLOR, dtype=int)
    for r, c in shape_cells:
        patch[r, c] = int(body_color)
    patch[marker_cell[0], marker_cell[1]] = int(marker_color)
    return patch


def _assemble_output(
    n_rows: int,
    n_cols: int,
    separator: int,
    background: int,
    placements: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    assert placements, "expected at least one motif placement"
    max_h = max(int(patch.shape[0]) for patch in placements.values())
    max_w = max(int(patch.shape[1]) for patch in placements.values())
    cell_h = max_h + 2
    cell_w = max_w + 2

    out = np.full(
        (n_rows * cell_h + (n_rows + 1), n_cols * cell_w + (n_cols + 1)),
        background,
        dtype=int,
    )
    out[np.arange(0, out.shape[0], cell_h + 1), :] = int(separator)
    out[:, np.arange(0, out.shape[1], cell_w + 1)] = int(separator)

    for (row_idx, col_idx), patch in placements.items():
        patch_h, patch_w = patch.shape
        top = row_idx * (cell_h + 1) + 2 + (max_h - patch_h) // 2
        left = col_idx * (cell_w + 1) + 2 + (max_w - patch_w) // 2
        mask = patch != background
        out[top : top + patch_h, left : left + patch_w][mask] = patch[mask]

    return out


def _extract_palette(
    grid: np.ndarray,
    background: int,
) -> tuple[int, tuple[int, int, int, int], np.ndarray, dict[int, tuple[int, int]]]:
    best = None
    best_score = None

    for separator in np.unique(grid):
        separator = int(separator)
        if separator == background:
            continue
        coords = np.argwhere(grid == separator)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        height, width = patch.shape

        sep_rows = [r for r in range(height) if np.all(patch[r, :] == separator)]
        sep_cols = [c for c in range(width) if np.all(patch[:, c] == separator)]
        if len(sep_rows) < 3 or len(sep_cols) < 3:
            continue
        if sep_rows[0] != 0 or sep_rows[-1] != height - 1:
            continue
        if sep_cols[0] != 0 or sep_cols[-1] != width - 1:
            continue

        row_steps = np.diff(sep_rows)
        col_steps = np.diff(sep_cols)
        if len(set(map(int, row_steps.tolist()))) != 1 or len(set(map(int, col_steps.tolist()))) != 1:
            continue

        cell_h = int(row_steps[0] - 1)
        cell_w = int(col_steps[0] - 1)
        if cell_h <= 0 or cell_w <= 0:
            continue

        sep_row_set = set(sep_rows)
        sep_col_set = set(sep_cols)
        ok = True
        for r in range(height):
            for c in range(width):
                should_be_separator = (r in sep_row_set) or (c in sep_col_set)
                if should_be_separator != (int(patch[r, c]) == separator):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        row_intervals = [(sep_rows[i] + 1, sep_rows[i + 1]) for i in range(len(sep_rows) - 1)]
        col_intervals = [(sep_cols[i] + 1, sep_cols[i + 1]) for i in range(len(sep_cols) - 1)]

        palette = np.zeros((len(row_intervals), len(col_intervals)), dtype=int)
        color_to_pos: dict[int, tuple[int, int]] = {}
        seen_colors: set[int] = set()

        for rr, (rs, re) in enumerate(row_intervals):
            for cc, (cs, ce) in enumerate(col_intervals):
                cell = patch[rs:re, cs:ce]
                colors = np.unique(cell)
                if len(colors) != 1:
                    ok = False
                    break
                color = int(colors[0])
                if color == separator or color in seen_colors:
                    ok = False
                    break
                seen_colors.add(color)
                palette[rr, cc] = color
                color_to_pos[color] = (rr, cc)
            if not ok:
                break
        if not ok:
            continue

        score = (palette.size, len(seen_colors), height * width)
        if best is None or score > best_score:
            best = (separator, (int(r0), int(c0), int(r1), int(c1)), palette, color_to_pos)
            best_score = score

    assert best is not None, "failed to locate palette matrix"
    return best


def _extract_motifs(
    grid: np.ndarray,
    background: int,
    palette_box: tuple[int, int, int, int],
) -> list[dict[str, object]]:
    mask = grid != background
    r0, c0, r1, c1 = palette_box
    mask[r0 : r1 + 1, c0 : c1 + 1] = False

    motifs: list[dict[str, object]] = []
    for cells in _components(mask):
        br0, bc0, br1, bc1 = _bbox(cells)
        patch = np.full((br1 - br0 + 1, bc1 - bc0 + 1), background, dtype=int)
        for r, c in cells:
            patch[r - br0, c - bc0] = int(grid[r, c])

        colors = [int(color) for color in np.unique(patch) if int(color) != background]
        assert len(colors) == 2, "each motif must use exactly two visible colors"
        counts = {color: int(np.count_nonzero(patch == color)) for color in colors}
        marker_colors = [color for color, count in counts.items() if count == 1]
        assert len(marker_colors) == 1, "each motif must have a unique marker color"
        marker_color = marker_colors[0]
        body_color = next(color for color in colors if color != marker_color)
        motifs.append(
            {
                "patch": patch,
                "body_color": int(body_color),
                "marker_color": int(marker_color),
            }
        )

    assert motifs, "expected at least one scattered motif"
    return motifs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rearrange scattered two-color motifs into the row-column cells encoded by a central palette matrix.

    Input structure:
    - The background is black.
    - A central palette matrix is enclosed and partitioned by single-pixel grey separator lines into N×M cells.
    - Every palette cell is a solid unique color.
    - Outside the palette, several disconnected motifs share the same shape; each motif uses one body color and one unique marker-pixel color.

    Rule:
    - Locate the grey separator lattice and read the palette cell colors together with their row and column positions.
    - For each scattered motif, treat the majority color as the body color and the single rare color as the marker color.
    - The palette row containing the body color gives the output row, and the palette column containing the marker color gives the output column.
    - Build an output grid with the same N×M separator structure, using cells large enough to contain the motifs, and place every motif into its decoded cell without changing its orientation.

    Key insight:
    - The palette is a two-axis lookup table: one motif color addresses a palette row, the other addresses a palette column, and their combination decodes a destination cell.
    """
    g = np.asarray(grid, dtype=int)
    background = _mode_color(g)
    separator, palette_box, palette, color_to_pos = _extract_palette(g, background)
    motifs = _extract_motifs(g, background, palette_box)

    placements: dict[tuple[int, int], np.ndarray] = {}
    for motif in motifs:
        body_color = int(motif["body_color"])
        marker_color = int(motif["marker_color"])
        assert body_color in color_to_pos and marker_color in color_to_pos, "motif colors must come from the palette"
        row_idx = int(color_to_pos[body_color][0])
        col_idx = int(color_to_pos[marker_color][1])
        key = (row_idx, col_idx)
        assert key not in placements, "decoded output cells must be unique"
        placements[key] = np.asarray(motif["patch"], dtype=int)

    return _assemble_output(int(palette.shape[0]), int(palette.shape[1]), separator, background, placements)


def _random_connected_shape() -> tuple[set[tuple[int, int]], tuple[int, int]]:
    for _ in range(120):
        box_h = random.randint(2, 4)
        box_w = random.randint(2, 4)
        max_area = min(box_h * box_w, 7)
        min_area = min(max_area, 4)
        area = random.randint(min_area, max_area)

        start = (random.randrange(box_h), random.randrange(box_w))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c, box_h, box_w):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != area:
            continue

        cells = _normalize_cells(cells)
        height = 1 + max(r for r, _ in cells)
        width = 1 + max(c for _, c in cells)
        if height < 2 or width < 2 or len(cells) == height * width:
            continue

        degrees = {}
        for r, c in cells:
            degree = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c, height, width))
            degrees[(r, c)] = degree
        marker_candidates = [cell for cell, degree in degrees.items() if degree <= 2]
        if not marker_candidates:
            continue
        marker_cell = random.choice(marker_candidates)
        return cells, marker_cell

    raise RuntimeError("failed to generate a connected motif shape")


def _transform_shape(
    shape_cells: set[tuple[int, int]],
    marker_cell: tuple[int, int],
) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    height = 1 + max(r for r, _ in shape_cells)
    width = 1 + max(c for _, c in shape_cells)
    mask = np.zeros((height, width), dtype=int)
    for r, c in shape_cells:
        mask[r, c] = 1
    mask[marker_cell[0], marker_cell[1]] = 2

    mask = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    if random.random() < 0.5:
        mask = np.flipud(mask)

    new_cells = {tuple(map(int, cell)) for cell in np.argwhere(mask > 0)}
    new_marker = tuple(map(int, np.argwhere(mask == 2)[0]))
    return new_cells, new_marker


def _choose_targets(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    all_targets = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    chosen: list[tuple[int, int]] = []
    rows_left = set(range(n_rows))
    cols_left = set(range(n_cols))

    while rows_left or cols_left:
        candidates = [
            pos
            for pos in all_targets
            if pos not in chosen and (pos[0] in rows_left or pos[1] in cols_left)
        ]
        pos = random.choice(candidates)
        chosen.append(pos)
        rows_left.discard(pos[0])
        cols_left.discard(pos[1])

    target_count = random.randint(len(chosen), len(all_targets))
    remaining = [pos for pos in all_targets if pos not in chosen]
    random.shuffle(remaining)
    chosen.extend(remaining[: target_count - len(chosen)])
    return chosen


def _choose_color_pair(palette: np.ndarray, row_idx: int, col_idx: int) -> tuple[int, int]:
    body_choices = [int(color) for color in palette[row_idx, :].tolist()]
    marker_choices = [int(color) for color in palette[:, col_idx].tolist()]
    pairs = [(body, marker) for body in body_choices for marker in marker_choices if body != marker]
    return random.choice(pairs)


def _candidate_positions(
    height: int,
    width: int,
    patch_h: int,
    patch_w: int,
    palette_box: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    pal_expanded = _expand_box(palette_box, 1)
    candidates = []
    for r in range(height - patch_h + 1):
        for c in range(width - patch_w + 1):
            box = (r, c, r + patch_h - 1, c + patch_w - 1)
            if _boxes_overlap(box, pal_expanded):
                continue
            candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _place_scattered_motifs(
    height: int,
    width: int,
    palette_box: tuple[int, int, int, int],
    patches: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    patch_h = int(max(patch.shape[0] for patch in patches))
    patch_w = int(max(patch.shape[1] for patch in patches))
    candidates = _candidate_positions(height, width, patch_h, patch_w, palette_box)
    expanded_boxes: list[tuple[int, int, int, int]] = []
    placements: list[tuple[int, int]] = []

    for patch in patches:
        this_h, this_w = map(int, patch.shape)
        placed = False
        for r, c in candidates:
            box = (r, c, r + this_h - 1, c + this_w - 1)
            expanded = _expand_box(box, 1)
            if any(_boxes_overlap(expanded, other) for other in expanded_boxes):
                continue
            placements.append((r, c))
            expanded_boxes.append(expanded)
            placed = True
            break
        if not placed:
            return None
    return placements


def _render_input(
    height: int,
    width: int,
    palette_patch: np.ndarray,
    palette_origin: tuple[int, int],
    motif_patches: list[np.ndarray],
    motif_origins: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    pr, pc = palette_origin
    ph, pw = palette_patch.shape
    grid[pr : pr + ph, pc : pc + pw] = palette_patch
    for patch, (r0, c0) in zip(motif_patches, motif_origins, strict=False):
        mask = patch != BG_COLOR
        grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]
    return grid


def _generate_attempt() -> np.ndarray | None:
    n_rows, n_cols = random.choice(DIMENSION_OPTIONS)
    palette_colors = random.sample(PALETTE_COLORS, n_rows * n_cols)
    palette = np.array(palette_colors, dtype=int).reshape(n_rows, n_cols)
    palette_patch = _build_palette(palette, SEPARATOR_COLOR)

    shape_cells, marker_cell = _random_connected_shape()
    shape_cells, marker_cell = _transform_shape(shape_cells, marker_cell)
    target_cells = _choose_targets(n_rows, n_cols)

    motif_patches = []
    expected_placements: dict[tuple[int, int], np.ndarray] = {}
    for row_idx, col_idx in target_cells:
        body_color, marker_color = _choose_color_pair(palette, row_idx, col_idx)
        patch = _make_motif_patch(shape_cells, marker_cell, body_color, marker_color)
        motif_patches.append(patch)
        expected_placements[(row_idx, col_idx)] = patch

    expected = _assemble_output(n_rows, n_cols, SEPARATOR_COLOR, BG_COLOR, expected_placements)

    patch_h = max(int(patch.shape[0]) for patch in motif_patches)
    patch_w = max(int(patch.shape[1]) for patch in motif_patches)
    pal_h, pal_w = palette_patch.shape

    min_h = max(16, pal_h + patch_h + 8)
    min_w = max(16, pal_w + patch_w + 8)
    for _ in range(40):
        height = random.randint(min_h, 30)
        width = random.randint(min_w, 30)
        pal_r = (height - pal_h) // 2
        pal_c = (width - pal_w) // 2
        palette_box = (pal_r, pal_c, pal_r + pal_h - 1, pal_c + pal_w - 1)
        motif_origins = _place_scattered_motifs(height, width, palette_box, motif_patches)
        if motif_origins is None:
            continue

        inp = _render_input(height, width, palette_patch, (pal_r, pal_c), motif_patches, motif_origins)
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the palette-indexed motif task")
