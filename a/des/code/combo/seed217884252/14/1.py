from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 400

PANEL_ROWS_RANGE = (2, 5)
PANEL_COLS_RANGE = (2, 5)
PANEL_HEIGHT_RANGE = (4, 8)
PANEL_WIDTH_RANGE = (4, 8)
SEPARATOR_THICKNESS_CHOICES = (1, 2)
MASK_ATTEMPTS = 120

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _group_consecutive(indices: list[int]) -> list[tuple[int, int]]:
    assert indices, "expected at least one separator line"
    groups: list[list[int]] = []
    for idx in indices:
        if not groups or idx != groups[-1][1] + 1:
            groups.append([idx, idx])
        else:
            groups[-1][1] = idx
    return [(int(a), int(b)) for a, b in groups]


def _intervals(length: int, groups: list[tuple[int, int]]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    prev_end = -1
    for start, end in groups:
        if start > prev_end + 1:
            spans.append((prev_end + 1, start))
        prev_end = end
    if prev_end < length - 1:
        spans.append((prev_end + 1, length))
    assert spans and all(end > start for start, end in spans), "separator lines must leave non-empty panels"
    return spans


def _panel_layout(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_sep = [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    col_sep = [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert row_sep and col_sep, "expected grey separator rows and columns"

    row_intervals = _intervals(grid.shape[0], _group_consecutive(row_sep))
    col_intervals = _intervals(grid.shape[1], _group_consecutive(col_sep))
    assert len(row_intervals) >= 2 and len(col_intervals) >= 2, "expected at least a 2x2 panel grid"

    row_sizes = {r1 - r0 for r0, r1 in row_intervals}
    col_sizes = {c1 - c0 for c0, c1 in col_intervals}
    assert len(row_sizes) == 1 and len(col_sizes) == 1, "all panels must share one size"
    return row_intervals, col_intervals


def _panel(grid: np.ndarray, row_intervals: list[tuple[int, int]], col_intervals: list[tuple[int, int]], r: int, c: int) -> np.ndarray:
    r0, r1 = row_intervals[r]
    c0, c1 = col_intervals[c]
    return grid[r0:r1, c0:c1]


def _template_mask(panel: np.ndarray) -> np.ndarray:
    values = set(map(int, np.unique(panel)))
    assert values.issubset({BLACK, GREY}), "template panels must contain only black and grey"
    mask = panel == GREY
    assert mask.any(), "template mask cannot be empty"
    assert not np.all(mask), "template mask cannot fill the entire panel"
    return mask


def _palette_color(panel: np.ndarray) -> int:
    values = set(map(int, np.unique(panel)))
    assert len(values) == 1, "palette panels must be uniformly filled"
    color = next(iter(values))
    assert color not in {BLACK, GREY}, "palette color must be a visible non-grey color"
    return int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    In a separator-defined panel grid, each interior panel copies the top-row mask
    from its column and fills it with the solid color from its row's left panel.
    """
    g = np.asarray(grid, dtype=int)
    row_intervals, col_intervals = _panel_layout(g)

    masks = [_template_mask(_panel(g, row_intervals, col_intervals, 0, c)) for c in range(1, len(col_intervals))]
    colors = [_palette_color(_panel(g, row_intervals, col_intervals, r, 0)) for r in range(1, len(row_intervals))]

    out = g.copy()
    for r, color in enumerate(colors, start=1):
        for c, mask in enumerate(masks, start=1):
            target = _panel(out, row_intervals, col_intervals, r, c)
            assert target.shape == mask.shape, "template and target panels must match in size"
            target[:] = BLACK
            target[mask] = color
    return out


def _valid_mask(mask: np.ndarray) -> bool:
    count = int(mask.sum())
    return 0 < count < mask.size and count >= 3


def _grow_blob(height: int, width: int) -> np.ndarray:
    target = random.randint(3, max(3, min(height * width - 1, (height * width * 2) // 3)))
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _diag_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mode = random.choice(("main", "anti", "x"))
    for r in range(height):
        c = round(r * (width - 1) / max(1, height - 1))
        if mode in {"main", "x"}:
            mask[r, c] = True
        if mode in {"anti", "x"}:
            mask[r, width - 1 - c] = True
    return mask


def _plus_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    cr = random.randrange(height)
    cc = random.randrange(width)
    up = random.randint(0, cr)
    down = random.randint(0, height - 1 - cr)
    left = random.randint(0, cc)
    right = random.randint(0, width - 1 - cc)
    mask[cr - up : cr + down + 1, cc] = True
    mask[cr, cc - left : cc + right + 1] = True
    return mask


def _rect_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    rect_h = random.randint(1, height)
    rect_w = random.randint(1, width)
    if rect_h == height and rect_w == width:
        rect_h = max(1, height - 1)
    top = random.randint(0, height - rect_h)
    left = random.randint(0, width - rect_w)
    mask[top : top + rect_h, left : left + rect_w] = True
    return mask


def _frame_mask(height: int, width: int) -> np.ndarray | None:
    if height < 3 or width < 3:
        return None
    outer_h = random.randint(3, height)
    outer_w = random.randint(3, width)
    top = random.randint(0, height - outer_h)
    left = random.randint(0, width - outer_w)
    mask = np.zeros((height, width), dtype=bool)
    bottom = top + outer_h - 1
    right = left + outer_w - 1
    mask[top, left : right + 1] = True
    mask[bottom, left : right + 1] = True
    mask[top : bottom + 1, left] = True
    mask[top : bottom + 1, right] = True
    return mask


def _l_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    corner = random.choice(("tl", "tr", "bl", "br"))
    vert = random.randint(1, height)
    horiz = random.randint(1, width)
    if corner == "tl":
        mask[:vert, 0] = True
        mask[0, :horiz] = True
    elif corner == "tr":
        mask[:vert, width - 1] = True
        mask[0, width - horiz :] = True
    elif corner == "bl":
        mask[height - vert :, 0] = True
        mask[height - 1, :horiz] = True
    else:
        mask[height - vert :, width - 1] = True
        mask[height - 1, width - horiz :] = True
    return mask


def _stripe_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if random.random() < 0.5:
        take = random.randint(1, height - 1)
        rows = random.sample(range(height), take)
        mask[rows, :] = True
    else:
        take = random.randint(1, width - 1)
        cols = random.sample(range(width), take)
        mask[:, cols] = True
    return mask


def _random_mask(height: int, width: int) -> np.ndarray:
    families = ("blob", "diag", "plus", "rect", "frame", "l", "stripe")
    for _ in range(MASK_ATTEMPTS):
        family = random.choice(families)
        if family == "blob":
            mask = _grow_blob(height, width)
        elif family == "diag":
            mask = _diag_mask(height, width)
        elif family == "plus":
            mask = _plus_mask(height, width)
        elif family == "rect":
            mask = _rect_mask(height, width)
        elif family == "frame":
            mask = _frame_mask(height, width)
            if mask is None:
                continue
        elif family == "l":
            mask = _l_mask(height, width)
        else:
            mask = _stripe_mask(height, width)

        if _valid_mask(mask):
            return mask
    raise RuntimeError("failed to sample a valid mask")


def _distinct_masks(count: int, height: int, width: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    seen: set[bytes] = set()
    for _ in range(MASK_ATTEMPTS * 3):
        if len(masks) == count:
            return masks
        mask = _random_mask(height, width)
        key = mask.tobytes()
        if key in seen:
            continue
        seen.add(key)
        masks.append(mask)
    raise RuntimeError("failed to sample enough distinct masks")


def _panel_starts(count: int, size: int, sep: int) -> list[int]:
    return [i * (size + sep) for i in range(count)]


def _compose_input(
    row_count: int,
    col_count: int,
    panel_h: int,
    panel_w: int,
    sep: int,
    masks: list[np.ndarray],
    colors: list[int],
) -> np.ndarray:
    height = row_count * panel_h + (row_count - 1) * sep
    width = col_count * panel_w + (col_count - 1) * sep
    grid = np.full((height, width), BLACK, dtype=int)

    row_starts = _panel_starts(row_count, panel_h, sep)
    col_starts = _panel_starts(col_count, panel_w, sep)

    for r in range(row_count - 1):
        start = row_starts[r] + panel_h
        grid[start : start + sep, :] = GREY
    for c in range(col_count - 1):
        start = col_starts[c] + panel_w
        grid[:, start : start + sep] = GREY

    for idx, mask in enumerate(masks, start=1):
        r0 = row_starts[0]
        c0 = col_starts[idx]
        grid[r0 : r0 + panel_h, c0 : c0 + panel_w][mask] = GREY

    for idx, color in enumerate(colors, start=1):
        r0 = row_starts[idx]
        c0 = col_starts[0]
        grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = color

    return grid


def _oracle(
    inp: np.ndarray,
    row_count: int,
    col_count: int,
    panel_h: int,
    panel_w: int,
    sep: int,
    masks: list[np.ndarray],
    colors: list[int],
) -> np.ndarray:
    out = inp.copy()
    row_starts = _panel_starts(row_count, panel_h, sep)
    col_starts = _panel_starts(col_count, panel_w, sep)
    for r, color in enumerate(colors, start=1):
        r0 = row_starts[r]
        for c, mask in enumerate(masks, start=1):
            c0 = col_starts[c]
            patch = out[r0 : r0 + panel_h, c0 : c0 + panel_w]
            patch[:] = BLACK
            patch[mask] = color
    return out


def _generate_attempt() -> np.ndarray | None:
    row_count = random.randint(*PANEL_ROWS_RANGE)
    col_count = random.randint(*PANEL_COLS_RANGE)
    panel_h = random.randint(*PANEL_HEIGHT_RANGE)
    panel_w = random.randint(*PANEL_WIDTH_RANGE)
    sep = random.choice(SEPARATOR_THICKNESS_CHOICES)

    colors = random.sample(PALETTE, row_count - 1)
    masks = _distinct_masks(col_count - 1, panel_h, panel_w)

    inp = _compose_input(row_count, col_count, panel_h, panel_w, sep, masks, colors)
    out = _oracle(inp, row_count, col_count, panel_h, panel_w, sep, masks, colors)
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        inp = _generate_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid panel-grid instance")
