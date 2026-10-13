from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
STAGE_SIZE = 30
WINDOW_COUNT_RANGE = (3, 5)
OUTER_SIZE_RANGE = (5, 8)
PLACEMENT_PADDING_RANGE = (1, 2)
MAX_GENERATION_ATTEMPTS = 300


def _border_majority_color(grid: np.ndarray) -> int:
    border = np.concatenate([grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]])
    values, counts = np.unique(border, return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    exclude = set() if exclude is None else set(exclude)
    return [int(v) for v in np.unique(grid) if int(v) not in exclude]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(cells)
    return parts


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_square_frame(cells: list[tuple[int, int]]) -> tuple[bool, tuple[int, int, int, int] | None]:
    r0, c0, r1, c1 = _bbox(cells)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height != width or height < 3:
        return False, None

    sub = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        sub[r - r0, c - c0] = True

    expected = np.zeros_like(sub, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if not np.array_equal(sub, expected):
        return False, None
    return True, (int(r0), int(c0), int(r1), int(c1))


def _find_windows(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    candidates: list[tuple[int, int, int, list[tuple[int, int, int, int]]]] = []
    for color in _colors_present(grid, exclude={bg}):
        components = _components(grid == color)
        if len(components) < 3:
            continue

        boxes: list[tuple[int, int, int, int]] = []
        side_lengths: set[int] = set()
        ok = True
        for cells in components:
            is_frame, box = _is_square_frame(cells)
            if not is_frame or box is None:
                ok = False
                break
            boxes.append(box)
            side_lengths.add(box[2] - box[0] + 1)

        if ok and len(side_lengths) == 1:
            side = next(iter(side_lengths))
            candidates.append((side, len(boxes), int(color), boxes))

    assert candidates, "expected a repeated square frame color"
    candidates.sort(key=lambda item: (item[0], item[1]))
    best_side, best_count, frame_color, boxes = candidates[-1]
    tied = [cand for cand in candidates if cand[0] == best_side and cand[1] == best_count]
    assert len(tied) == 1, "frame color must be uniquely identifiable"
    assert len(boxes) >= 3, "expected at least three windows"
    boxes.sort(key=lambda box: (box[0], box[1]))
    return int(frame_color), boxes


def _infer_target_pattern(interiors: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(interiors, axis=0)
    window_count, inner_h, inner_w = stack.shape
    target = np.empty((inner_h, inner_w), dtype=int)

    for r in range(inner_h):
        for c in range(inner_w):
            values, counts = np.unique(stack[:, r, c], return_counts=True)
            if len(values) == 1:
                target[r, c] = int(values[0])
                continue

            assert len(values) == 2, "each position must have at most one unique color"
            unique_values = values[counts == 1]
            common_values = values[counts == window_count - 1]
            assert len(unique_values) == 1 and len(common_values) == 1, "ambiguous unique-color position"
            target[r, c] = int(unique_values[0])

    return target


def main(grid: np.ndarray) -> np.ndarray:
    """
    Broadcast every window's unique interior pixel color to the matching position in all windows.

    Input structure:
    - The grid has a black background and several equal-size square windows with one-pixel-thick grey borders.
    - Every window contains an interior pattern of 2-3 non-border colors, and the windows share the same local layout.
    - At any local interior position, either all windows agree or exactly one window shows a different color there.

    Rule:
    - Locate the repeated square frame color and extract each window interior.
    - For each local interior coordinate, keep the color if all windows already match.
    - If one window has the lone differing color at that coordinate, replace that coordinate in every window with that unique color.

    Key insight:
    - The minority color at a mismatching position is the signal to preserve and copy everywhere, not an error to overwrite with the majority.
    """
    g = np.asarray(grid, dtype=int)
    bg = _border_majority_color(g)
    frame_color, windows = _find_windows(g, bg)
    assert frame_color != bg

    sizes = {(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in windows}
    assert len(sizes) == 1, "all windows must have the same size"

    interiors = [g[r0 + 1 : r1, c0 + 1 : c1].copy() for r0, c0, r1, c1 in windows]
    target = _infer_target_pattern(interiors)

    out = g.copy()
    for r0, c0, r1, c1 in windows:
        out[r0 + 1 : r1, c0 + 1 : c1] = target
    return out


def _pattern_checker(size: int, palette: list[int]) -> np.ndarray:
    k = len(palette)
    return np.array([[palette[(r + c) % k] for c in range(size)] for r in range(size)], dtype=int)


def _pattern_stripes_h(size: int, palette: list[int]) -> np.ndarray:
    k = len(palette)
    return np.array([[palette[r % k] for _ in range(size)] for r in range(size)], dtype=int)


def _pattern_stripes_v(size: int, palette: list[int]) -> np.ndarray:
    k = len(palette)
    return np.array([[palette[c % k] for c in range(size)] for _ in range(size)], dtype=int)


def _pattern_diagonals(size: int, palette: list[int]) -> np.ndarray:
    k = len(palette)
    pattern = np.full((size, size), palette[0], dtype=int)
    for i in range(size):
        pattern[i, i] = palette[1]
        pattern[i, size - 1 - i] = palette[1 if k == 2 else 2]
    if k == 3:
        pattern[size // 2, :] = palette[2]
    return pattern


def _pattern_quadrants(size: int, palette: list[int]) -> np.ndarray:
    split_r = random.randint(1, size - 1)
    split_c = random.randint(1, size - 1)
    pattern = np.full((size, size), palette[0], dtype=int)
    if len(palette) == 2:
        pattern[:split_r, split_c:] = palette[1]
        pattern[split_r:, :split_c] = palette[1]
    else:
        pattern[:split_r, split_c:] = palette[1]
        pattern[split_r:, :split_c] = palette[2]
        pattern[split_r:, split_c:] = palette[1]
    return pattern


def _random_transform(pattern: np.ndarray) -> np.ndarray:
    out = pattern.copy()
    turns = random.randrange(4)
    if turns:
        out = np.rot90(out, turns)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return np.asarray(out, dtype=int)


def _make_base_pattern(size: int, palette: list[int]) -> np.ndarray | None:
    builders = [
        _pattern_checker,
        _pattern_stripes_h,
        _pattern_stripes_v,
        _pattern_diagonals,
        _pattern_quadrants,
    ]
    random.shuffle(builders)
    for builder in builders:
        pattern = _random_transform(builder(size, palette))
        used = {int(v) for v in np.unique(pattern)}
        if used == set(palette):
            return pattern
    return None


def _draw_window(grid: np.ndarray, top: int, left: int, pattern: np.ndarray) -> None:
    outer = pattern.shape[0] + 2
    bottom = top + outer - 1
    right = left + outer - 1
    grid[top : bottom + 1, left] = FRAME_COLOR
    grid[top : bottom + 1, right] = FRAME_COLOR
    grid[top, left : right + 1] = FRAME_COLOR
    grid[bottom, left : right + 1] = FRAME_COLOR
    grid[top + 1 : bottom, left + 1 : right] = pattern


def _place_windows(count: int, outer: int, padding: int) -> list[tuple[int, int]] | None:
    reserved = np.zeros((STAGE_SIZE, STAGE_SIZE), dtype=bool)
    positions: list[tuple[int, int]] = []
    for _ in range(count):
        candidates: list[tuple[int, int]] = []
        for r in range(1, STAGE_SIZE - outer):
            for c in range(1, STAGE_SIZE - outer):
                r0 = max(0, r - padding)
                c0 = max(0, c - padding)
                r1 = min(STAGE_SIZE, r + outer + padding)
                c1 = min(STAGE_SIZE, c + outer + padding)
                if not reserved[r0:r1, c0:c1].any():
                    candidates.append((r, c))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        positions.append((top, left))
        r0 = max(0, top - padding)
        c0 = max(0, left - padding)
        r1 = min(STAGE_SIZE, top + outer + padding)
        c1 = min(STAGE_SIZE, left + outer + padding)
        reserved[r0:r1, c0:c1] = True
    return positions


def _crop_with_margin(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    foreground = out != BG
    coords = np.argwhere(foreground)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top_margin = random.randint(1, min(2, int(r0)))
    left_margin = random.randint(1, min(2, int(c0)))
    bottom_margin = random.randint(1, min(2, int(STAGE_SIZE - 1 - r1)))
    right_margin = random.randint(1, min(2, int(STAGE_SIZE - 1 - c1)))
    rr0 = int(r0 - top_margin)
    cc0 = int(c0 - left_margin)
    rr1 = int(r1 + bottom_margin)
    cc1 = int(c1 + right_margin)
    return inp[rr0 : rr1 + 1, cc0 : cc1 + 1].copy(), out[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()


def _gen_attempt() -> np.ndarray | None:
    window_count = random.randint(*WINDOW_COUNT_RANGE)
    min_outer = OUTER_SIZE_RANGE[0] + (1 if window_count >= 5 else 0)
    outer = random.randint(min_outer, OUTER_SIZE_RANGE[1])
    inner = outer - 2
    padding = random.randint(*PLACEMENT_PADDING_RANGE)

    color_count = random.choice([2, 3])
    palette = random.sample(PALETTE, color_count)
    base_pattern = _make_base_pattern(inner, palette)
    if base_pattern is None:
        return None

    anomaly_budget = [random.choice([1, 2]) for _ in range(window_count)]
    total_anomalies = sum(anomaly_budget)
    if total_anomalies >= inner * inner or total_anomalies > max(1, (inner * inner) // 2):
        return None

    positions = _place_windows(window_count, outer, padding)
    if positions is None:
        return None

    target_pattern = base_pattern.copy()
    window_patterns = [base_pattern.copy() for _ in range(window_count)]
    interior_positions = [(r, c) for r in range(inner) for c in range(inner)]
    anomaly_positions = random.sample(interior_positions, total_anomalies)
    owners = [idx for idx, budget in enumerate(anomaly_budget) for _ in range(budget)]
    random.shuffle(owners)

    for owner, (r, c) in zip(owners, anomaly_positions, strict=False):
        choices = [color for color in palette if color != int(base_pattern[r, c])]
        new_color = random.choice(choices)
        window_patterns[owner][r, c] = new_color
        target_pattern[r, c] = new_color

    inp = np.full((STAGE_SIZE, STAGE_SIZE), BG, dtype=int)
    out = np.full((STAGE_SIZE, STAGE_SIZE), BG, dtype=int)
    for (top, left), pattern in zip(positions, window_patterns, strict=False):
        _draw_window(inp, top, left, pattern)
    for top, left in positions:
        _draw_window(out, top, left, target_pattern)

    inp, out = _crop_with_margin(inp, out)
    if _border_majority_color(inp) != BG:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
