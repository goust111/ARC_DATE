from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _is_exact_frame(pts: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 2 or w < 2:
        return False
    if len(pts) != 2 * h + 2 * w - 4:
        return False
    mask = np.zeros((h, w), dtype=bool)
    for r, c in pts:
        mask[r - r0, c - c0] = True
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(mask, expected))


def _is_legend_block(patch: np.ndarray, bg: int) -> bool:
    if patch.shape != (5, 5):
        return False
    border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
    color = int(border[0])
    if color == bg:
        return False
    if not np.all(border == color):
        return False
    return not bool(np.all(patch[1:4, 1:4] == bg))


def _legend_blocks(grid: np.ndarray, sep_row: int, bg: int) -> list[tuple[int, tuple[int, int, int, int], np.ndarray]]:
    h, w = grid.shape
    if sep_row < 5:
        return []
    used = np.zeros((sep_row, w), dtype=bool)
    blocks: list[tuple[int, tuple[int, int, int, int], np.ndarray]] = []
    for r in range(sep_row - 4):
        for c in range(w - 4):
            if used[r : r + 5, c : c + 5].any():
                continue
            patch = grid[r : r + 5, c : c + 5]
            if not _is_legend_block(patch, bg):
                continue
            border_color = int(patch[0, 0])
            blocks.append((border_color, (r, c, r + 4, c + 4), patch[1:4, 1:4].copy()))
            used[r : r + 5, c : c + 5] = True
    return blocks


def _find_hollow_rectangles(
    grid: np.ndarray, sep_row: int, bg: int
) -> list[tuple[int, tuple[int, int, int, int]]]:
    lower = grid[sep_row + 1 :, :]
    if lower.size == 0:
        return []
    rects: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in [int(v) for v in np.unique(lower) if int(v) != bg]:
        for pts in _components(lower == color):
            box = _bbox(pts)
            if not _is_exact_frame(pts, box):
                continue
            r0, c0, r1, c1 = box
            if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
                continue
            if not np.all(lower[r0 + 1 : r1, c0 + 1 : c1] == bg):
                continue
            rects.append((color, (sep_row + 1 + r0, c0, sep_row + 1 + r1, c1)))
    rects.sort(key=lambda item: item[1])
    return rects


def _find_separator_row(grid: np.ndarray, bg: int) -> int:
    h = grid.shape[0]
    candidates = [
        r
        for r in range(1, h - 1)
        if int(grid[r, 0]) != bg and bool(np.all(grid[r, :] == grid[r, 0]))
    ]
    assert candidates, "expected a horizontal separator row"

    best_row = None
    best_score = None
    for r in candidates:
        legend_count = len(_legend_blocks(grid, r, bg))
        rect_count = len(_find_hollow_rectangles(grid, r, bg))
        if legend_count == 0 or rect_count == 0:
            continue
        score = (legend_count, rect_count, -abs(r - h // 3))
        if best_score is None or score > best_score:
            best_score = score
            best_row = r
    if best_row is not None:
        return int(best_row)
    assert len(candidates) == 1
    return int(candidates[0])


def _tile(pattern: np.ndarray, height: int, width: int) -> np.ndarray:
    reps_r = (height + pattern.shape[0] - 1) // pattern.shape[0]
    reps_c = (width + pattern.shape[1] - 1) // pattern.shape[1]
    return np.tile(pattern, (reps_r, reps_c))[:height, :width]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a border-color -> 3x3 pattern lookup from the legend above the separator row,
    then fill each matching hollow rectangle below by tiling that pattern from the
    rectangle interior's top-left corner.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    sep_row = _find_separator_row(g, bg)

    legend = _legend_blocks(g, sep_row, bg)
    assert legend, "expected at least one legend block"
    patterns: dict[int, np.ndarray] = {}
    for color, _box, pattern in legend:
        patterns[color] = pattern

    out = g.copy()
    for color, (r0, c0, r1, c1) in _find_hollow_rectangles(g, sep_row, bg):
        assert color in patterns, "every rectangle border color must appear in the legend"
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile(patterns[color], inner_h, inner_w)
    return out


# --------------- generator ---------------

BG = 0
ALL_COLORS = list(range(1, 10))
GRID_H_RANGE = (25, 34)
GRID_W_RANGE = (26, 34)
SEP_ROW_RANGE = (7, 13)
LEGEND_COUNT_RANGE = (2, 4)
RECT_COUNT_RANGE = (3, 6)
INNER_H_RANGE = (3, 9)
INNER_W_RANGE = (3, 10)
PADDING = 1
MARGIN = 1
GEN_RETRIES = 400


def _pattern_library(a: int, b: int) -> list[tuple[str, np.ndarray]]:
    return [
        ("solid", np.full((3, 3), a, dtype=int)),
        ("checker", np.array([[a, b, a], [b, a, b], [a, b, a]], dtype=int)),
        ("vstripe", np.array([[a, b, a], [a, b, a], [a, b, a]], dtype=int)),
        ("hstripe", np.array([[a, a, a], [b, b, b], [a, a, a]], dtype=int)),
        ("diag", np.array([[a, b, b], [b, a, b], [b, b, a]], dtype=int)),
        ("cross", np.array([[b, a, b], [a, a, a], [b, a, b]], dtype=int)),
    ]


def _choose_patterns(border_colors: list[int], pattern_colors: list[int]) -> dict[int, np.ndarray]:
    patterns: dict[int, np.ndarray] = {}
    used_keys: set[bytes] = set()
    need_non_solid = True
    for i, border in enumerate(border_colors):
        for _ in range(80):
            a, b = random.sample(pattern_colors, 2)
            options = _pattern_library(a, b)
            if need_non_solid or (i == len(border_colors) - 1 and not patterns):
                options = [item for item in options if item[0] != "solid"]
            elif random.random() < 0.25:
                options = [item for item in options if item[0] == "solid"] or options
            name, pattern = random.choice(options)
            key = pattern.tobytes()
            if key in used_keys:
                continue
            used_keys.add(key)
            patterns[border] = pattern
            if name != "solid":
                need_non_solid = False
            break
        else:
            raise RuntimeError("failed to create distinct patterns")
    return patterns


def _find_place(
    occ: np.ndarray, patch_h: int, patch_w: int, padding: int = PADDING, margin: int = MARGIN
) -> tuple[int, int] | None:
    h, w = occ.shape
    candidates: list[tuple[int, int]] = []
    r_start = margin
    r_stop = h - patch_h - margin
    c_start = margin
    c_stop = w - patch_w - margin
    if r_start > r_stop or c_start > c_stop:
        return None
    for r in range(r_start, r_stop + 1):
        for c in range(c_start, c_stop + 1):
            rr0 = max(0, r - padding)
            rr1 = min(h, r + patch_h + padding)
            cc0 = max(0, c - padding)
            cc1 = min(w, c + patch_w + padding)
            if not occ[rr0:rr1, cc0:cc1].any():
                candidates.append((r, c))
    if not candidates:
        return None
    return random.choice(candidates)


def _draw_legend_patch(border_color: int, pattern: np.ndarray) -> np.ndarray:
    patch = np.zeros((5, 5), dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    patch[1:4, 1:4] = pattern
    return patch


def _draw_hollow_rectangle(grid: np.ndarray, r: int, c: int, inner_h: int, inner_w: int, color: int) -> None:
    r1 = r + inner_h + 1
    c1 = c + inner_w + 1
    grid[r, c : c1 + 1] = color
    grid[r1, c : c1 + 1] = color
    grid[r : r1 + 1, c] = color
    grid[r : r1 + 1, c1] = color


def _attempt_generate() -> tuple[np.ndarray, np.ndarray] | None:
    legend_count = random.randint(*LEGEND_COUNT_RANGE)
    rect_count = random.randint(max(legend_count, RECT_COUNT_RANGE[0]), RECT_COUNT_RANGE[1])

    border_colors = random.sample(ALL_COLORS, legend_count + 1)
    separator_color = border_colors[0]
    border_colors = border_colors[1:]
    pattern_colors = [c for c in ALL_COLORS if c not in {separator_color, *border_colors}]
    if len(pattern_colors) < 2:
        return None
    try:
        patterns = _choose_patterns(border_colors, pattern_colors)
    except RuntimeError:
        return None

    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    sep_row_max = min(SEP_ROW_RANGE[1], height - 12)
    if sep_row_max < SEP_ROW_RANGE[0]:
        return None
    sep_row = random.randint(SEP_ROW_RANGE[0], sep_row_max)

    grid = np.full((height, width), BG, dtype=int)
    expected = grid.copy()
    grid[sep_row, :] = separator_color
    expected[sep_row, :] = separator_color

    top_occ = np.zeros((sep_row, width), dtype=bool)
    for border_color in border_colors:
        pos = _find_place(top_occ, 5, 5)
        if pos is None:
            return None
        r, c = pos
        patch = _draw_legend_patch(border_color, patterns[border_color])
        grid[r : r + 5, c : c + 5] = patch
        expected[r : r + 5, c : c + 5] = patch
        top_occ[r : r + 5, c : c + 5] = True

    rect_colors = border_colors[:]
    while len(rect_colors) < rect_count:
        rect_colors.append(random.choice(border_colors))
    random.shuffle(rect_colors)

    specs: list[tuple[int, int, int]] = []
    for color in rect_colors:
        inner_h = random.randint(*INNER_H_RANGE)
        inner_w = random.randint(*INNER_W_RANGE)
        specs.append((color, inner_h, inner_w))
    if all(h % 3 == 0 and w % 3 == 0 for _color, h, w in specs):
        color, inner_h, inner_w = specs[0]
        specs[0] = (color, min(INNER_H_RANGE[1], inner_h + 1), inner_w)

    lower_h = height - sep_row - 1
    bottom_occ = np.zeros((lower_h, width), dtype=bool)
    for color, inner_h, inner_w in specs:
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        pos = _find_place(bottom_occ, outer_h, outer_w)
        if pos is None:
            return None
        lr, c = pos
        r = sep_row + 1 + lr
        _draw_hollow_rectangle(grid, r, c, inner_h, inner_w, color)
        _draw_hollow_rectangle(expected, r, c, inner_h, inner_w, color)
        expected[r + 1 : r + 1 + inner_h, c + 1 : c + 1 + inner_w] = _tile(
            patterns[color], inner_h, inner_w
        )
        bottom_occ[lr : lr + outer_h, c : c + outer_w] = True

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _attempt_generate()
        if result is not None:
            return result[0]
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
