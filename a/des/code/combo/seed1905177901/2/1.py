from __future__ import annotations

import random

import numpy as np


DIR_STEP = {
    "left": (0, -1),
    "right": (0, 1),
    "up": (-1, 0),
    "down": (1, 0),
}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "up": "down",
    "down": "up",
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()),
        int(cols.max()),
    )


def _split_segments(size: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        if start >= sep:
            return []
        spans.append((start, sep))
        start = sep + 1
    if start >= size:
        return []
    spans.append((start, size))
    return spans


def _detect_layout(grid: np.ndarray) -> tuple[int, int, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    mono_rows = {r: int(grid[r, 0]) for r in range(h) if np.all(grid[r, :] == grid[r, 0])}
    mono_cols = {c: int(grid[0, c]) for c in range(w) if np.all(grid[:, c] == grid[0, c])}
    candidate_colors = set(mono_rows.values()) | set(mono_cols.values())

    candidates = []
    for color in candidate_colors:
        sep_rows = sorted(r for r, v in mono_rows.items() if v == color)
        sep_cols = sorted(c for c, v in mono_cols.items() if v == color)
        if not sep_rows or not sep_cols:
            continue
        row_spans = _split_segments(h, sep_rows)
        col_spans = _split_segments(w, sep_cols)
        if not row_spans or not col_spans:
            continue
        if len({r1 - r0 for r0, r1 in row_spans}) != 1:
            continue
        if len({c1 - c0 for c0, c1 in col_spans}) != 1:
            continue
        if len(row_spans) < 2 or len(col_spans) < 2:
            continue
        candidates.append((len(sep_rows) + len(sep_cols), color, sep_rows, sep_cols, row_spans, col_spans))

    assert candidates, "expected a regular grid of panels split by monochrome separator lines"
    _score, sep_color, sep_rows, sep_cols, row_spans, col_spans = max(candidates, key=lambda x: (x[0], x[1]))

    non_sep = grid[grid != sep_color]
    vals, counts = np.unique(non_sep, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    return bg, int(sep_color), sep_rows, sep_cols, row_spans, col_spans


def _find_notch(mask: np.ndarray) -> tuple[str, tuple[int, int, int, int]]:
    h, w = mask.shape
    candidates: list[tuple[str, tuple[int, int, int, int]]] = []
    for coords in _components(~mask):
        r0, c0, r1, c1 = _bbox(coords)
        area = int(coords.shape[0])
        if area != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        touches = []
        if r0 == 0:
            touches.append("up")
        if r1 == h - 1:
            touches.append("down")
        if c0 == 0:
            touches.append("left")
        if c1 == w - 1:
            touches.append("right")
        if len(touches) != 1:
            continue
        side = touches[0]
        if side in {"left", "right"} and not (0 < r0 and r1 < h - 1):
            continue
        if side in {"up", "down"} and not (0 < c0 and c1 < w - 1):
            continue
        candidates.append((side, (r0, c0, r1, c1)))

    assert len(candidates) == 1, "expected exactly one inset rectangular notch"
    return candidates[0]


def _analyze_panel(panel: np.ndarray, bg: int) -> dict:
    colors = set(map(int, np.unique(panel))) - {int(bg)}
    assert len(colors) == 1, "each panel must contain exactly one foreground color"
    (color,) = tuple(colors)

    mask = panel == color
    comps = _components(mask)
    assert len(comps) == 1, "panel foreground must be one 4-connected object"

    br0, bc0, br1, bc1 = _bbox(comps[0])
    sub = mask[br0 : br1 + 1, bc0 : bc1 + 1]
    side, (nr0, nc0, nr1, nc1) = _find_notch(sub)
    h, w = sub.shape

    if side == "left":
        depth = nc1 - nc0 + 1
        slot_h = nr1 - nr0 + 1
        base_w = w - depth
        expected = np.zeros_like(sub, dtype=bool)
        expected[:, :base_w] = True
        expected[nr0 : nr1 + 1, :depth] = False
        expected[nr0 : nr1 + 1, base_w:] = True
        rect = (br0, br1 + 1, bc0, bc0 + base_w)
        notch_size = (slot_h, depth)
    elif side == "right":
        depth = nc1 - nc0 + 1
        slot_h = nr1 - nr0 + 1
        expected = np.zeros_like(sub, dtype=bool)
        expected[:, depth:] = True
        expected[nr0 : nr1 + 1, w - depth :] = False
        expected[nr0 : nr1 + 1, :depth] = True
        rect = (br0, br1 + 1, bc0 + depth, bc1 + 1)
        notch_size = (slot_h, depth)
    elif side == "up":
        depth = nr1 - nr0 + 1
        slot_w = nc1 - nc0 + 1
        base_h = h - depth
        expected = np.zeros_like(sub, dtype=bool)
        expected[:base_h, :] = True
        expected[:depth, nc0 : nc1 + 1] = False
        expected[base_h:, nc0 : nc1 + 1] = True
        rect = (br0, br0 + base_h, bc0, bc1 + 1)
        notch_size = (depth, slot_w)
    else:
        depth = nr1 - nr0 + 1
        slot_w = nc1 - nc0 + 1
        expected = np.zeros_like(sub, dtype=bool)
        expected[depth:, :] = True
        expected[h - depth :, nc0 : nc1 + 1] = False
        expected[:depth, nc0 : nc1 + 1] = True
        rect = (br0 + depth, br1 + 1, bc0, bc1 + 1)
        notch_size = (depth, slot_w)

    assert np.array_equal(sub, expected), "panel shape must be a notched rectangle plus opposite tab"
    return {
        "color": int(color),
        "direction": side,
        "notch_size": notch_size,
        "rect": rect,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair panel objects by notch direction, swap paired colors, and complete each notch with
    the partner's complementary tab, leaving a solid rectangle in every used panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, sep, sep_rows, sep_cols, row_spans, col_spans = _detect_layout(g)
    infos: dict[tuple[int, int], dict] = {}
    for pr, (r0, r1) in enumerate(row_spans):
        for pc, (c0, c1) in enumerate(col_spans):
            info = _analyze_panel(g[r0:r1, c0:c1], bg)
            infos[(pr, pc)] = info

    out = np.full_like(g, int(bg))
    if sep_rows:
        out[sep_rows, :] = int(sep)
    if sep_cols:
        out[:, sep_cols] = int(sep)

    for (pr, pc), info in infos.items():
        dr, dc = DIR_STEP[info["direction"]]
        partner = (pr + dr, pc + dc)
        assert partner in infos, "notch direction must point to an adjacent panel"
        partner_info = infos[partner]
        assert partner_info["direction"] == OPPOSITE[info["direction"]], "pairing must be reciprocal"
        assert partner_info["notch_size"] == info["notch_size"], "paired notches must share one shape"

        panel_r0, _panel_r1 = row_spans[pr]
        panel_c0, _panel_c1 = col_spans[pc]
        rr0, rr1, cc0, cc1 = info["rect"]
        out[panel_r0 + rr0 : panel_r0 + rr1, panel_c0 + cc0 : panel_c0 + cc1] = int(partner_info["color"])

    return out


# --------------- generator ---------------

GRID_SHAPES = ((2, 2), (2, 3), (3, 2))
PANEL_H_RANGE = (7, 10)
PANEL_W_RANGE = (7, 10)
BG = 0
SEP = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _panel_spans(count: int, panel_size: int) -> tuple[list[int], list[tuple[int, int]]]:
    separators = [i * panel_size + (i - 1) for i in range(1, count)]
    spans = []
    start = 0
    for sep in separators:
        spans.append((start, sep))
        start = sep + 1
    spans.append((start, start + panel_size))
    return separators, spans


def _tile_dominoes(rows: int, cols: int) -> list[tuple[str, tuple[int, int], tuple[int, int]]]:
    used = np.zeros((rows, cols), dtype=bool)
    placements: list[tuple[str, tuple[int, int], tuple[int, int]]] = []

    def backtrack() -> bool:
        empty = np.argwhere(~used)
        if empty.size == 0:
            return True
        r, c = map(int, empty[0])
        options = []
        if c + 1 < cols and not used[r, c + 1]:
            options.append(("h", (r, c), (r, c + 1)))
        if r + 1 < rows and not used[r + 1, c]:
            options.append(("v", (r, c), (r + 1, c)))
        random.shuffle(options)
        for orient, a, b in options:
            used[a] = True
            used[b] = True
            placements.append((orient, a, b))
            if backtrack():
                return True
            placements.pop()
            used[a] = False
            used[b] = False
        return False

    assert backtrack(), "board must admit a domino tiling"
    return placements


def _make_piece(panel_h: int, panel_w: int, side: str, notch_size: tuple[int, int], color: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    panel = np.full((panel_h, panel_w), BG, dtype=int)

    if side in {"left", "right"}:
        slot_h, depth = notch_size
        min_base_h = max(4, slot_h + 2)
        min_base_w = max(4, depth + 2)
        max_base_w = panel_w - depth
        if min_base_h > panel_h or min_base_w > max_base_w:
            raise ValueError("horizontal notch does not fit panel")
        base_h = random.randint(min_base_h, panel_h)
        base_w = random.randint(min_base_w, max_base_w)
        slot_top = random.randint(1, base_h - slot_h - 1)
        total_h = base_h
        total_w = base_w + depth
        top = random.randint(0, panel_h - total_h)
        left = random.randint(0, panel_w - total_w)
        patch = np.full((total_h, total_w), BG, dtype=int)
        if side == "left":
            patch[:, :base_w] = color
            patch[slot_top : slot_top + slot_h, :depth] = BG
            patch[slot_top : slot_top + slot_h, base_w:] = color
            rect = (top, top + base_h, left, left + base_w)
        else:
            patch[:, depth:] = color
            patch[slot_top : slot_top + slot_h, total_w - depth :] = BG
            patch[slot_top : slot_top + slot_h, :depth] = color
            rect = (top, top + base_h, left + depth, left + total_w)
    else:
        depth, slot_w = notch_size
        min_base_h = 4
        min_base_w = max(4, slot_w + 2)
        max_base_h = panel_h - depth
        if min_base_h > max_base_h or min_base_w > panel_w:
            raise ValueError("vertical notch does not fit panel")
        base_h = random.randint(min_base_h, max_base_h)
        base_w = random.randint(min_base_w, panel_w)
        slot_left = random.randint(1, base_w - slot_w - 1)
        total_h = base_h + depth
        total_w = base_w
        top = random.randint(0, panel_h - total_h)
        left = random.randint(0, panel_w - total_w)
        patch = np.full((total_h, total_w), BG, dtype=int)
        if side == "up":
            patch[:base_h, :] = color
            patch[:depth, slot_left : slot_left + slot_w] = BG
            patch[base_h:, slot_left : slot_left + slot_w] = color
            rect = (top, top + base_h, left, left + base_w)
        else:
            patch[depth:, :] = color
            patch[total_h - depth :, slot_left : slot_left + slot_w] = BG
            patch[:depth, slot_left : slot_left + slot_w] = color
            rect = (top + depth, top + total_h, left, left + base_w)

    panel[top : top + total_h, left : left + total_w] = patch
    return panel, rect


def _build_expected(
    directions: dict[tuple[int, int], str],
    rects: dict[tuple[int, int], tuple[int, int, int, int]],
    colors: dict[tuple[int, int], int],
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    sep_rows: list[int],
    sep_cols: list[int],
    shape: tuple[int, int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    if sep_rows:
        out[sep_rows, :] = SEP
    if sep_cols:
        out[:, sep_cols] = SEP

    for (pr, pc), side in directions.items():
        dr, dc = DIR_STEP[side]
        partner = (pr + dr, pc + dc)
        rr0, rr1, cc0, cc1 = rects[(pr, pc)]
        panel_r0, _panel_r1 = row_spans[pr]
        panel_c0, _panel_c1 = col_spans[pc]
        out[panel_r0 + rr0 : panel_r0 + rr1, panel_c0 + cc0 : panel_c0 + cc1] = colors[partner]

    return out


def _gen_attempt() -> np.ndarray | None:
    rows, cols = random.choice(GRID_SHAPES)
    panel_h = random.randint(*PANEL_H_RANGE)
    panel_w = random.randint(*PANEL_W_RANGE)
    if rows * cols > len(PALETTE):
        return None

    sep_rows, row_spans = _panel_spans(rows, panel_h)
    sep_cols, col_spans = _panel_spans(cols, panel_w)
    grid_h = rows * panel_h + len(sep_rows)
    grid_w = cols * panel_w + len(sep_cols)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    if sep_rows:
        grid[sep_rows, :] = SEP
    if sep_cols:
        grid[:, sep_cols] = SEP

    tile = _tile_dominoes(rows, cols)
    directions: dict[tuple[int, int], str] = {}
    notch_sizes: dict[tuple[int, int], tuple[int, int]] = {}
    for orient, a, b in tile:
        if orient == "h":
            depth = random.randint(1, 2)
            slot_h = random.randint(2, panel_h - 2)
            shared = (slot_h, depth)
            directions[a] = "right"
            directions[b] = "left"
        else:
            depth = random.randint(1, 2)
            slot_w = random.randint(2, panel_w - 2)
            shared = (depth, slot_w)
            directions[a] = "down"
            directions[b] = "up"
        notch_sizes[a] = shared
        notch_sizes[b] = shared

    colors = random.sample(PALETTE, rows * cols)
    rects: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    panel_colors: dict[tuple[int, int], int] = {}
    for idx, (pr, pc) in enumerate(sorted(directions)):
        panel_color = int(colors[idx])
        side = directions[(pr, pc)]
        notch_size = notch_sizes[(pr, pc)]
        try:
            panel, rect = _make_piece(panel_h, panel_w, side, notch_size, panel_color)
        except ValueError:
            return None
        r0, r1 = row_spans[pr]
        c0, c1 = col_spans[pc]
        grid[r0:r1, c0:c1] = panel
        rects[(pr, pc)] = rect
        panel_colors[(pr, pc)] = panel_color

    expected = _build_expected(
        directions=directions,
        rects=rects,
        colors=panel_colors,
        row_spans=row_spans,
        col_spans=col_spans,
        sep_rows=sep_rows,
        sep_cols=sep_cols,
        shape=grid.shape,
    )

    try:
        produced = main(grid)
    except Exception:
        return None
    if np.array_equal(produced, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
