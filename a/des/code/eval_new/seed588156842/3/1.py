from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OPPOSITE_QUADRANT = {"tl": "br", "tr": "bl", "bl": "tr", "br": "tl"}
ROTATIONS = {"top": 0, "right": 3, "bottom": 2, "left": 1}
MAX_ATTEMPTS = 400


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
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(pts, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_separator_index(full_lines: list[int]) -> int:
    assert full_lines, "expected at least one separator line"
    runs: list[list[int]] = [[full_lines[0]]]
    for idx in full_lines[1:]:
        if idx == runs[-1][-1] + 1:
            runs[-1].append(idx)
        else:
            runs.append([idx])
    longest = max(runs, key=len)
    return int(longest[len(longest) // 2])


def _split_quadrants(grid: np.ndarray) -> tuple[dict[str, tuple[np.ndarray, int, int]], int, int]:
    row_lines = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    col_lines = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    row_sep = _find_separator_index(row_lines)
    col_sep = _find_separator_index(col_lines)
    quads = {
        "tl": (grid[:row_sep, :col_sep], 0, 0),
        "tr": (grid[:row_sep, col_sep + 1 :], 0, col_sep + 1),
        "bl": (grid[row_sep + 1 :, :col_sep], row_sep + 1, 0),
        "br": (grid[row_sep + 1 :, col_sep + 1 :], row_sep + 1, col_sep + 1),
    }
    return quads, row_sep, col_sep


def _gap_side(
    missing: list[tuple[int, int]], r0: int, c0: int, r1: int, c1: int
) -> str | None:
    if all(r == r0 for r, _ in missing):
        vals = sorted(c for _, c in missing)
        return "top" if vals == list(range(vals[0], vals[0] + len(vals))) else None
    if all(r == r1 for r, _ in missing):
        vals = sorted(c for _, c in missing)
        return "bottom" if vals == list(range(vals[0], vals[0] + len(vals))) else None
    if all(c == c0 for _, c in missing):
        vals = sorted(r for r, _ in missing)
        return "left" if vals == list(range(vals[0], vals[0] + len(vals))) else None
    if all(c == c1 for _, c in missing):
        vals = sorted(r for r, _ in missing)
        return "right" if vals == list(range(vals[0], vals[0] + len(vals))) else None
    return None


def _find_frames(patch: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(set(map(int, np.unique(patch))) - {BLACK, GREY}):
        for coords in _components(patch == color):
            if len(coords) < 12:
                continue
            r0, c0, r1, c1 = _bbox(coords)
            side = r1 - r0 + 1
            if side != c1 - c0 + 1 or side < 5:
                continue
            pts = {(int(r), int(c)) for r, c in coords}
            perimeter = {
                (r, c)
                for r in range(r0, r1 + 1)
                for c in range(c0, c1 + 1)
                if r in (r0, r1) or c in (c0, c1)
            }
            if pts - perimeter:
                continue
            missing = sorted(perimeter - pts)
            if len(missing) not in (2, 3):
                continue
            side_name = _gap_side(missing, r0, c0, r1, c1)
            if side_name is None:
                continue
            interior = patch[r0 + 1 : r1, c0 + 1 : c1]
            color_key = frozenset(int(v) for v in np.unique(interior) if int(v) != BLACK)
            if len(color_key) < 2:
                continue
            frames.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "gap": side_name,
                    "inner_size": side - 2,
                    "colors": color_key,
                }
            )
    return frames


def _choose_main_and_legend(quads: dict[str, tuple[np.ndarray, int, int]]) -> tuple[str, str, list[dict]]:
    candidates: list[tuple[int, str, list[dict]]] = []
    for name, (patch, _r0, _c0) in quads.items():
        frames = _find_frames(patch)
        if len(frames) >= 2 and len({frame["inner_size"] for frame in frames}) == 1:
            candidates.append((len(frames), name, frames))
    assert candidates, "expected one quadrant with at least two valid frames"
    candidates.sort(reverse=True)
    _count, main_name, frames = candidates[0]
    return main_name, OPPOSITE_QUADRANT[main_name], frames


def _extract_templates(legend_patch: np.ndarray, inner_size: int) -> dict[frozenset[int], np.ndarray]:
    templates: dict[frozenset[int], np.ndarray] = {}
    for coords in _components(legend_patch != BLACK):
        r0, c0, r1, c1 = _bbox(coords)
        block = legend_patch[r0 : r1 + 1, c0 : c1 + 1]
        if block.shape != (inner_size + 1, inner_size):
            continue
        label = [int(v) for v in block[0].tolist() if int(v) != BLACK]
        if len(label) < 2 or len(label) != len(set(label)):
            continue
        template = block[1:, :].copy()
        colors = frozenset(label)
        if {int(v) for v in np.unique(template) if int(v) != BLACK} != set(colors):
            continue
        templates[colors] = template
    assert templates, "expected at least one template block"
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gapped square frame with the legend template whose color label matches the interior clue set.

    Input structure:
    - The background is black.
    - One corner region holds several compact multicolor templates, separated from the rest of the grid by full grey divider lines.
    - In the opposite quadrant there are several same-sized hollow square frames; each border has one contiguous 2-3 cell gap on exactly one side, and each interior contains a few clue pixels using at least two colors.

    Rule:
    - Identify the quadrant that contains the square frames and crop the output to that quadrant, removing the grey dividers and the legend corner.
    - For each frame, match the set of clue-pixel colors inside the frame to the labeled color set on one legend template.
    - Treat each legend template as the "gap on top" orientation, rotate it to match the frame's gap direction, and overwrite the frame interior with the rotated template while leaving the border and the gap unchanged.

    Key insight:
    - The interior clue pixels are a color-set key for template lookup, and the missing side of the frame supplies the rotation needed to orient that template correctly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    quads, _row_sep, _col_sep = _split_quadrants(g)
    main_name, legend_name, frames = _choose_main_and_legend(quads)
    main_patch = quads[main_name][0].copy()
    legend_patch = quads[legend_name][0]

    inner_size = frames[0]["inner_size"]
    templates = _extract_templates(legend_patch, inner_size)

    for frame in frames:
        template = templates[frame["colors"]]
        rotated = np.rot90(template, ROTATIONS[frame["gap"]])
        r0, c0, r1, c1 = frame["bbox"]
        main_patch[r0 + 1 : r1, c0 + 1 : c1] = rotated
    return main_patch


def _random_color_sets(template_palette: list[int], inner_size: int, count: int) -> list[tuple[int, ...]]:
    max_size = min(4, inner_size, len(template_palette))
    choices = [
        combo
        for size in range(2, max_size + 1)
        for combo in combinations(template_palette, size)
    ]
    assert len(choices) >= count
    return random.sample(choices, count)


def _is_rotation_asymmetric(template: np.ndarray) -> bool:
    return all(not np.array_equal(template, np.rot90(template, k)) for k in (1, 2, 3))


def _random_template(inner_size: int, colors: tuple[int, ...]) -> np.ndarray:
    colors_arr = np.asarray(colors, dtype=int)
    for _ in range(200):
        template = np.random.choice(colors_arr, size=(inner_size, inner_size)).astype(int)
        flat_positions = list(np.ndindex(template.shape))
        random.shuffle(flat_positions)
        for color, pos in zip(colors, flat_positions, strict=False):
            template[pos] = int(color)
        if _is_rotation_asymmetric(template):
            return template
    raise RuntimeError("failed to create an asymmetric template")


def _legend_block(template: np.ndarray, ordered_colors: list[int]) -> np.ndarray:
    inner_size = template.shape[0]
    block = np.full((inner_size + 1, inner_size), BLACK, dtype=int)
    cols = sorted(random.sample(range(inner_size), len(ordered_colors)))
    for color, col in zip(ordered_colors, cols, strict=False):
        block[0, col] = int(color)
    block[1:, :] = template
    return block


def _build_legend_patch(
    templates: list[np.ndarray], color_sets: list[tuple[int, ...]], inner_size: int
) -> np.ndarray:
    count = len(templates)
    cols_options = [c for c in (1, 2) if c <= count]
    legend_cols = random.choice(cols_options)
    legend_rows = (count + legend_cols - 1) // legend_cols
    block_h = inner_size + 1
    block_w = inner_size
    pad_top = random.randint(0, 1)
    pad_left = random.randint(0, 1)
    pad_bottom = random.randint(0, 1)
    pad_right = random.randint(0, 1)
    legend_h = pad_top + legend_rows * block_h + max(0, legend_rows - 1) + pad_bottom
    legend_w = pad_left + legend_cols * block_w + max(0, legend_cols - 1) + pad_right
    patch = np.full((legend_h, legend_w), BLACK, dtype=int)

    indices = list(range(count))
    random.shuffle(indices)
    for slot, idx in enumerate(indices):
        rr = slot // legend_cols
        cc = slot % legend_cols
        r0 = pad_top + rr * (block_h + 1)
        c0 = pad_left + cc * (block_w + 1)
        colors = list(color_sets[idx])
        random.shuffle(colors)
        block = _legend_block(templates[idx], colors)
        patch[r0 : r0 + block_h, c0 : c0 + block_w] = block
    return patch


def _distribute(total_extra: int, slots: int) -> list[int]:
    values = [1] * slots
    for _ in range(total_extra):
        values[random.randrange(slots)] += 1
    return values


def _slot_positions(count: int, outer_size: int, limit_h: int, limit_w: int) -> list[tuple[int, int]] | None:
    layouts = []
    for rows in range(1, count + 1):
        cols = (count + rows - 1) // rows
        base_h = rows * outer_size + (rows + 1)
        base_w = cols * outer_size + (cols + 1)
        if base_h <= limit_h and base_w <= limit_w:
            layouts.append((rows, cols, base_h, base_w))
    if not layouts:
        return None

    rows, cols, base_h, base_w = random.choice(layouts)
    extra_h = random.randint(0, limit_h - base_h)
    extra_w = random.randint(0, limit_w - base_w)
    vspaces = _distribute(extra_h, rows + 1)
    hspaces = _distribute(extra_w, cols + 1)

    row_starts: list[int] = []
    col_starts: list[int] = []
    cur = vspaces[0]
    for idx in range(rows):
        row_starts.append(cur)
        cur += outer_size + vspaces[idx + 1]
    cur = hspaces[0]
    for idx in range(cols):
        col_starts.append(cur)
        cur += outer_size + hspaces[idx + 1]

    slots = [(r0, c0) for r0 in row_starts for c0 in col_starts]
    random.shuffle(slots)
    return slots[:count]


def _frame_with_gap(
    outer_size: int, border_color: int, gap_side: str, gap_len: int
) -> np.ndarray:
    patch = np.full((outer_size, outer_size), BLACK, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    start = random.randint(1, outer_size - gap_len - 1)
    if gap_side == "top":
        patch[0, start : start + gap_len] = BLACK
    elif gap_side == "bottom":
        patch[-1, start : start + gap_len] = BLACK
    elif gap_side == "left":
        patch[start : start + gap_len, 0] = BLACK
    else:
        patch[start : start + gap_len, -1] = BLACK
    return patch


def _scatter_clues(inner_size: int, colors: tuple[int, ...]) -> np.ndarray:
    clue = np.full((inner_size, inner_size), BLACK, dtype=int)
    cells = list(np.ndindex(inner_size, inner_size))
    random.shuffle(cells)
    assert len(cells) >= len(colors)
    for color, (r, c) in zip(colors, cells, strict=False):
        clue[r, c] = int(color)
    extra = random.randint(0, max(1, inner_size - 1))
    for r, c in cells[len(colors) : len(colors) + extra]:
        clue[r, c] = int(random.choice(colors))
    return clue


def _expected_main_patch(
    main_patch: np.ndarray,
    placements: list[tuple[int, int]],
    templates: list[np.ndarray],
    gap_sides: list[str],
    outer_size: int,
) -> np.ndarray:
    out = main_patch.copy()
    for (r0, c0), template, gap_side in zip(placements, templates, gap_sides, strict=False):
        rotated = np.rot90(template, ROTATIONS[gap_side])
        out[r0 + 1 : r0 + outer_size - 1, c0 + 1 : c0 + outer_size - 1] = rotated
    return out


def _compose_grid(legend_patch: np.ndarray, main_patch: np.ndarray, corner: str) -> np.ndarray:
    legend_h, legend_w = legend_patch.shape
    main_h, main_w = main_patch.shape
    total_h = legend_h + 1 + main_h
    total_w = legend_w + 1 + main_w
    grid = np.full((total_h, total_w), BLACK, dtype=int)
    row_sep = legend_h if corner in ("tl", "tr") else main_h
    col_sep = legend_w if corner in ("tl", "bl") else main_w
    grid[row_sep, :] = GREY
    grid[:, col_sep] = GREY

    if corner == "tl":
        grid[:legend_h, :legend_w] = legend_patch
        grid[row_sep + 1 :, col_sep + 1 :] = main_patch
    elif corner == "tr":
        grid[:legend_h, col_sep + 1 :] = legend_patch
        grid[row_sep + 1 :, :main_w] = main_patch
    elif corner == "bl":
        grid[row_sep + 1 :, :legend_w] = legend_patch
        grid[:main_h, col_sep + 1 :] = main_patch
    else:
        grid[row_sep + 1 :, col_sep + 1 :] = legend_patch
        grid[:main_h, :main_w] = main_patch
    return grid


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(2, 4)
    inner_size = random.choice([3, 4, 5])
    outer_size = inner_size + 2

    border_colors = random.sample(COLOR_POOL, count)
    template_palette = [color for color in COLOR_POOL if color not in border_colors]
    if len(template_palette) < 4:
        return None

    color_sets = _random_color_sets(template_palette, inner_size, count)
    templates = [_random_template(inner_size, colors) for colors in color_sets]
    legend_patch = _build_legend_patch(templates, color_sets, inner_size)

    max_main_h = 30 - legend_patch.shape[0] - 1
    max_main_w = 30 - legend_patch.shape[1] - 1
    if max_main_h < outer_size + 2 or max_main_w < outer_size + 2:
        return None

    placements = _slot_positions(count, outer_size, max_main_h, max_main_w)
    if placements is None:
        return None

    main_h = max(r0 + outer_size for r0, _c0 in placements) + random.randint(0, 1)
    main_w = max(c0 + outer_size for _r0, c0 in placements) + random.randint(0, 1)
    if main_h > max_main_h or main_w > max_main_w:
        return None

    main_patch = np.full((main_h, main_w), BLACK, dtype=int)
    gap_sides = [random.choice(["top", "right", "bottom", "left"]) for _ in range(count)]
    gap_lengths = [random.choice([2, 3]) for _ in range(count)]

    items = list(zip(placements, templates, color_sets, border_colors, gap_sides, gap_lengths, strict=False))
    random.shuffle(items)
    for (r0, c0), template, colors, border_color, gap_side, gap_len in items:
        frame = _frame_with_gap(outer_size, border_color, gap_side, gap_len)
        clue = _scatter_clues(inner_size, colors)
        block = frame.copy()
        block[1:-1, 1:-1] = clue
        main_patch[r0 : r0 + outer_size, c0 : c0 + outer_size] = block

    expected = _expected_main_patch(main_patch, [item[0] for item in items], [item[1] for item in items], [item[4] for item in items], outer_size)
    corner = random.choice(["tl", "tr", "bl", "br"])
    grid = _compose_grid(legend_patch, main_patch, corner)
    if grid.shape[0] > 30 or grid.shape[1] > 30:
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
