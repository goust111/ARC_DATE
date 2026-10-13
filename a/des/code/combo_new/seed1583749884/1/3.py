import random
from collections import Counter

import numpy as np


BG = 0
GEN_SEPARATOR = 7
GEN_RETRIES = 400
SIDE_PATTERNS = (
    ("left", "left", "right"),
    ("left", "right", "left"),
    ("right", "left", "left"),
    ("right", "right", "left"),
    ("right", "left", "right"),
    ("left", "right", "right"),
)


def _find_separator_color(grid: np.ndarray) -> int:
    row_colors = [int(grid[r, 0]) for r in range(grid.shape[0]) if np.all(grid[r, :] == grid[r, 0])]
    col_colors = [int(grid[0, c]) for c in range(grid.shape[1]) if np.all(grid[:, c] == grid[0, c])]
    row_counts = Counter(row_colors)
    col_counts = Counter(col_colors)
    candidates = [
        color
        for color in row_counts
        if row_counts[color] >= 2 and col_counts.get(color, 0) >= 2
    ]
    if not candidates:
        candidates = list(set(row_colors) & set(col_colors))
    assert candidates, "expected a separator color occupying full rows and columns"
    return max(candidates, key=lambda color: row_counts[color] + col_counts.get(color, 0))


def _content_intervals(size: int, separators: list[int]) -> list[tuple[int, int]]:
    sep = sorted(set(int(x) for x in separators))
    assert sep, "expected separator lines"

    runs: list[tuple[int, int]] = []
    start = prev = sep[0]
    for idx in sep[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev + 1))
        start = prev = idx
    runs.append((start, prev + 1))

    intervals: list[tuple[int, int]] = []
    last = 0
    for start, end in runs:
        if last < start:
            intervals.append((last, start))
        last = end
    if last < size:
        intervals.append((last, size))
    return [(a, b) for a, b in intervals if b > a]


def _panel_layout(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]:
    sep = _find_separator_color(grid)
    row_seps = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == sep)]
    col_seps = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == sep)]
    row_intervals = _content_intervals(grid.shape[0], row_seps)
    col_intervals = _content_intervals(grid.shape[1], col_seps)
    assert len(row_intervals) == 3 and len(col_intervals) == 3, "expected a 3x3 panel layout"
    assert len({r1 - r0 for r0, r1 in row_intervals}) == 1, "panel rows must be equal-sized"
    assert len({c1 - c0 for c0, c1 in col_intervals}) == 1, "panel cols must be equal-sized"

    counts = Counter(int(x) for x in grid.ravel() if int(x) != sep)
    assert counts, "expected panel contents"
    bg = int(counts.most_common(1)[0][0])
    return sep, bg, row_intervals, col_intervals


def _panel_color(panel: np.ndarray, bg: int) -> int:
    colors = {int(x) for x in panel.ravel() if int(x) != bg}
    assert len(colors) == 1, "each panel should contain exactly one foreground color"
    return next(iter(colors))


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    _sep, bg, row_intervals, col_intervals = _panel_layout(g)

    out = g.copy()
    panel_h = row_intervals[0][1] - row_intervals[0][0]
    panel_w = col_intervals[0][1] - col_intervals[0][0]
    template_shape = (panel_h, panel_w)

    for r0, r1 in row_intervals:
        template = np.zeros(template_shape, dtype=bool)
        colors: list[int] = []
        for c0, c1 in col_intervals:
            panel = g[r0:r1, c0:c1]
            colors.append(_panel_color(panel, bg))
            template |= panel != bg

        for (c0, c1), color in zip(col_intervals, colors):
            repaired = np.full(template_shape, bg, dtype=int)
            repaired[template] = color
            out[r0:r1, c0:c1] = repaired

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each panel row by unioning complementary fragments into a shared symmetric stencil.

    Input structure:
    - The background fills empty space, and one separator color occupies full rows and full columns.
    - Those separator lines split the canvas into a 3x3 array of equal rectangular panels.
    - Each panel contains a single-color fragment of a left-right symmetric pattern, and the three panels in a row use complementary missing pixels.

    Rule:
    - Detect the separator rows and columns and extract the nine panels in their local coordinates.
    - For each row of three panels, take the positionwise union of all non-background pixels to recover that row's complete template.
    - Repaint every panel in the row with the recovered template, using the panel's own foreground color.
    - Leave the separator lines and untouched background cells unchanged.

    Key insight:
    - The missing cells are distributed across sibling panels in the same row, so the full figure is recovered by row-wise aggregation rather than by treating each panel independently.
    """
    return _solve(grid)


def _pair_reps(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    reps: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w // 2):
            if mask[r, c] and mask[r, w - 1 - c]:
                reps.append((r, c))
    return reps


def _make_template(h: int, w: int) -> np.ndarray | None:
    left_w = w // 2
    if left_w < 2:
        return None

    max_full = max(8, int(0.48 * h * w))
    max_left = max(4, max_full // 2)
    primitive_modes = ("rect", "rect", "hline", "vline", "lshape", "dots")

    for _ in range(80):
        left = np.zeros((h, left_w), dtype=bool)
        primitive_count = random.randint(2, 5)
        for _ in range(primitive_count):
            mode = random.choice(primitive_modes)
            if mode == "rect":
                ph = random.randint(1, min(3, h))
                pw = random.randint(1, min(2, left_w))
                r0 = random.randint(0, h - ph)
                c0 = random.randint(0, left_w - pw)
                left[r0 : r0 + ph, c0 : c0 + pw] = True
            elif mode == "hline":
                r = random.randint(0, h - 1)
                c0 = random.randint(0, left_w - 1)
                c1 = random.randint(c0, left_w - 1)
                left[r, c0 : c1 + 1] = True
            elif mode == "vline":
                c = random.randint(0, left_w - 1)
                r0 = random.randint(0, h - 1)
                r1 = random.randint(r0, h - 1)
                left[r0 : r1 + 1, c] = True
            elif mode == "lshape":
                r0 = random.randint(0, h - 2)
                c0 = random.randint(0, left_w - 1)
                arm_h = random.randint(2, h - r0)
                arm_w = random.randint(1, left_w - c0)
                left[r0 : r0 + arm_h, c0] = True
                left[r0 + arm_h - 1, c0 : c0 + arm_w] = True
            else:
                count = random.randint(1, min(3, h * left_w))
                cells = random.sample(range(h * left_w), count)
                for idx in cells:
                    rr, cc = divmod(idx, left_w)
                    left[rr, cc] = True

        if not (4 <= int(left.sum()) <= max_left):
            continue

        full = np.zeros((h, w), dtype=bool)
        full[:, :left_w] = left
        full[:, w - left_w :] = np.fliplr(left)

        if w % 2 == 1 and random.random() < 0.65:
            center = w // 2
            if random.random() < 0.5:
                r0 = random.randint(0, h - 1)
                r1 = random.randint(r0, h - 1)
                full[r0 : r1 + 1, center] = True
            else:
                count = random.randint(1, max(1, h // 3))
                rows = random.sample(range(h), count)
                full[rows, center] = True

        if not (8 <= int(full.sum()) <= max_full):
            continue
        if len(_pair_reps(full)) < 4:
            continue

        pts = np.argwhere(full)
        if len(set(pts[:, 0].tolist())) < 2:
            continue
        if len(set(pts[:, 1].tolist())) < 3:
            continue
        return full

    return None


def _sample_missing_plan(pair_reps: list[tuple[int, int]], sides: tuple[str, str, str]) -> list[tuple[str, frozenset[tuple[int, int]]]] | None:
    seen_by_side = {"left": set(), "right": set()}
    plan: list[tuple[str, frozenset[tuple[int, int]]]] = []
    max_remove = len(pair_reps) - 1
    if max_remove < 1:
        return None
    max_k = max(1, min(max_remove, len(pair_reps) // 2 + 1))

    for side in sides:
        for _ in range(50):
            k = random.randint(1, max_k)
            subset = frozenset(random.sample(pair_reps, k))
            if subset not in seen_by_side[side]:
                seen_by_side[side].add(subset)
                plan.append((side, subset))
                break
        else:
            return None
    return plan


def _apply_missing(template: np.ndarray, side: str, missing: frozenset[tuple[int, int]]) -> np.ndarray:
    panel = template.copy()
    w = panel.shape[1]
    for r, c in missing:
        col = c if side == "left" else w - 1 - c
        panel[r, col] = False
    return panel


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(5, 9)
    panel_w = random.randint(5, 9)
    if 3 * panel_h + 2 > 30 or 3 * panel_w + 2 > 30:
        return None

    row_colors = random.sample([color for color in range(1, 10) if color != GEN_SEPARATOR], 3)
    row_panels: list[list[np.ndarray]] = []

    for _ in range(3):
        for _ in range(80):
            template = _make_template(panel_h, panel_w)
            if template is None:
                continue
            pair_reps = _pair_reps(template)
            sides = random.choice(SIDE_PATTERNS)
            plan = _sample_missing_plan(pair_reps, sides)
            if plan is None:
                continue

            panels = [_apply_missing(template, side, missing) for side, missing in plan]
            if any(np.array_equal(panel, template) for panel in panels):
                continue
            if any(np.array_equal(panel, np.fliplr(panel)) for panel in panels):
                continue
            if not np.array_equal(np.logical_or.reduce(panels), template):
                continue
            row_panels.append(panels)
            break
        else:
            return None

    grid = np.full((3 * panel_h + 2, 3 * panel_w + 2), BG, dtype=int)
    grid[panel_h, :] = GEN_SEPARATOR
    grid[2 * panel_h + 1, :] = GEN_SEPARATOR
    grid[:, panel_w] = GEN_SEPARATOR
    grid[:, 2 * panel_w + 1] = GEN_SEPARATOR

    for row_idx in range(3):
        r0 = row_idx * (panel_h + 1)
        for col_idx in range(3):
            c0 = col_idx * (panel_w + 1)
            block = np.full((panel_h, panel_w), BG, dtype=int)
            block[row_panels[row_idx][col_idx]] = row_colors[row_idx]
            grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = block

    solved = _solve(grid)
    if solved.shape != grid.shape:
        return None
    if not np.any(solved != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
