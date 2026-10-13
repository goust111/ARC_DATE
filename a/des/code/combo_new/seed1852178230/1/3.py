from __future__ import annotations

import random

import numpy as np


SEPARATOR_COLOR = 5
PANEL_GRID_CHOICES = ((2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2))
PANEL_SIZE_RANGE = range(4, 8)
EDGE_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_RETRIES = 300


def main(grid: np.ndarray) -> np.ndarray:
    """Reassemble the shuffled panel grid so every shared border pattern matches.

    Input structure:
    - The grid is partitioned by single-pixel grey separator rows and columns into an equal-sized panel array.
    - Each panel has a black interior background with colored distractor motifs, and its outermost row/column pixels form border patterns.
    - The panel order in the input array is scrambled, but panel orientations stay fixed.

    Rule:
    - Detect the separator lattice and extract every panel without altering its pixels.
    - Find the unique arrangement where each horizontal neighbor pair has identical touching vertical-edge sequences and each vertical neighbor pair has identical touching horizontal-edge sequences.
    - Put the panels back into the same m x n frame and restore the grey separators.

    Key insight:
    - The puzzle is an exact jigsaw on panel borders: interior artwork is irrelevant, while the four edge color sequences are the only signals that determine adjacency.
    """
    g = np.asarray(grid, dtype=int)
    sep_color, row_spans, col_spans = _find_panel_layout(g)
    panels = _extract_panels(g, row_spans, col_spans)
    order = _solve_panel_order(panels, len(row_spans), len(col_spans))

    arranged = [[panels[int(order[r, c])] for c in range(order.shape[1])] for r in range(order.shape[0])]
    return _compose_grid(arranged, sep_color)


def _axis_spans(length: int, separators: list[int]) -> list[tuple[int, int]] | None:
    seps = sorted(int(x) for x in separators)
    if not seps:
        return [(0, int(length))]
    if len(seps) != len(set(seps)):
        return None
    n_panels = len(seps) + 1
    if len(seps) == 1:
        step = int(seps[0] + 1)
    else:
        gaps = [b - a for a, b in zip(seps, seps[1:])]
        if len(set(gaps)) != 1:
            return None
        step = int(gaps[0])
    if step < 2 or seps[0] != step - 1:
        return None
    if length + 1 != n_panels * step:
        return None
    expected = [step * k - 1 for k in range(1, n_panels)]
    if seps != expected:
        return None
    panel_size = step - 1
    return [(i * step, i * step + panel_size) for i in range(n_panels)]


def _find_panel_layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = map(int, grid.shape)
    candidates: list[tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]] = []

    for color in np.unique(grid):
        c = int(color)
        row_seps = [r for r in range(h) if np.all(grid[r, :] == c)]
        col_seps = [cc for cc in range(w) if np.all(grid[:, cc] == c)]
        if not row_seps and not col_seps:
            continue
        row_spans = _axis_spans(h, row_seps)
        col_spans = _axis_spans(w, col_seps)
        if row_spans is None or col_spans is None:
            continue
        if len(row_spans) * len(col_spans) <= 1:
            continue
        score = len(row_seps) + len(col_seps)
        candidates.append((score, c, row_spans, col_spans))

    assert candidates, "failed to detect separator lattice"
    _, sep_color, row_spans, col_spans = max(candidates, key=lambda item: item[0])
    return int(sep_color), row_spans, col_spans


def _extract_panels(
    grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]
) -> list[np.ndarray]:
    panels: list[np.ndarray] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            panels.append(grid[r0:r1, c0:c1].copy())
    return panels


def _panel_info(panel: np.ndarray) -> dict[str, object]:
    return {
        "grid": panel,
        "top": tuple(int(x) for x in panel[0, :]),
        "bottom": tuple(int(x) for x in panel[-1, :]),
        "left": tuple(int(x) for x in panel[:, 0]),
        "right": tuple(int(x) for x in panel[:, -1]),
    }


def _solve_panel_order(panels: list[np.ndarray], rows: int, cols: int) -> np.ndarray:
    info = [_panel_info(panel) for panel in panels]
    n_panels = len(info)
    assert n_panels == rows * cols

    for i, panel in enumerate(info):
        panel["top_matches"] = tuple(
            j for j, other in enumerate(info) if j != i and panel["top"] == other["bottom"]
        )
        panel["bottom_matches"] = tuple(
            j for j, other in enumerate(info) if j != i and panel["bottom"] == other["top"]
        )
        panel["left_matches"] = tuple(
            j for j, other in enumerate(info) if j != i and panel["left"] == other["right"]
        )
        panel["right_matches"] = tuple(
            j for j, other in enumerate(info) if j != i and panel["right"] == other["left"]
        )

    board = np.full((rows, cols), -1, dtype=int)
    unused = set(range(n_panels))
    solutions: list[np.ndarray] = []
    _search_orders(board, unused, info, rows, cols, solutions, limit=2)
    assert len(solutions) == 1, "panel arrangement is not unique"
    return solutions[0]


def _position_can_use(panel: dict[str, object], r: int, c: int, rows: int, cols: int) -> bool:
    if r > 0 and not panel["top_matches"]:
        return False
    if r < rows - 1 and not panel["bottom_matches"]:
        return False
    if c > 0 and not panel["left_matches"]:
        return False
    if c < cols - 1 and not panel["right_matches"]:
        return False
    return True


def _has_future_match(
    match_indices: tuple[int, ...], unused: set[int], banned: int, required_side: tuple[int, ...], panels: list[dict[str, object]], side: str
) -> bool:
    for idx in match_indices:
        if idx == banned or idx not in unused:
            continue
        if panels[idx][side] == required_side:
            return True
    return False


def _candidate_indices(
    board: np.ndarray, unused: set[int], panels: list[dict[str, object]], r: int, c: int, rows: int, cols: int
) -> list[int]:
    candidates: list[int] = []
    for idx in sorted(unused):
        panel = panels[idx]
        if not _position_can_use(panel, r, c, rows, cols):
            continue

        if r > 0 and board[r - 1, c] != -1:
            up = panels[int(board[r - 1, c])]
            if up["bottom"] != panel["top"]:
                continue
        if r + 1 < rows and board[r + 1, c] != -1:
            down = panels[int(board[r + 1, c])]
            if panel["bottom"] != down["top"]:
                continue
        if c > 0 and board[r, c - 1] != -1:
            left = panels[int(board[r, c - 1])]
            if left["right"] != panel["left"]:
                continue
        if c + 1 < cols and board[r, c + 1] != -1:
            right = panels[int(board[r, c + 1])]
            if panel["right"] != right["left"]:
                continue

        if r > 0 and board[r - 1, c] == -1:
            if not _has_future_match(panel["top_matches"], unused, idx, panel["top"], panels, "bottom"):
                continue
        if r + 1 < rows and board[r + 1, c] == -1:
            if not _has_future_match(panel["bottom_matches"], unused, idx, panel["bottom"], panels, "top"):
                continue
        if c > 0 and board[r, c - 1] == -1:
            if not _has_future_match(panel["left_matches"], unused, idx, panel["left"], panels, "right"):
                continue
        if c + 1 < cols and board[r, c + 1] == -1:
            if not _has_future_match(panel["right_matches"], unused, idx, panel["right"], panels, "left"):
                continue

        candidates.append(idx)
    return candidates


def _choose_next_slot(
    board: np.ndarray, unused: set[int], panels: list[dict[str, object]], rows: int, cols: int
) -> tuple[tuple[int, int] | None, list[int]]:
    best_pos: tuple[int, int] | None = None
    best_candidates: list[int] | None = None
    for r in range(rows):
        for c in range(cols):
            if board[r, c] != -1:
                continue
            candidates = _candidate_indices(board, unused, panels, r, c, rows, cols)
            if not candidates:
                return (r, c), []
            if best_candidates is None or len(candidates) < len(best_candidates):
                best_pos = (r, c)
                best_candidates = candidates
    return best_pos, ([] if best_candidates is None else best_candidates)


def _search_orders(
    board: np.ndarray,
    unused: set[int],
    panels: list[dict[str, object]],
    rows: int,
    cols: int,
    solutions: list[np.ndarray],
    limit: int,
) -> None:
    if len(solutions) >= limit:
        return
    if not unused:
        solutions.append(board.copy())
        return

    pos, candidates = _choose_next_slot(board, unused, panels, rows, cols)
    if pos is None or not candidates:
        return

    r, c = pos
    for idx in candidates:
        board[r, c] = int(idx)
        unused.remove(int(idx))
        _search_orders(board, unused, panels, rows, cols, solutions, limit)
        unused.add(int(idx))
        board[r, c] = -1
        if len(solutions) >= limit:
            return


def _compose_grid(panel_rows: list[list[np.ndarray]], sep_color: int) -> np.ndarray:
    rows = len(panel_rows)
    cols = len(panel_rows[0])
    panel_h, panel_w = map(int, panel_rows[0][0].shape)
    out_h = rows * panel_h + (rows - 1)
    out_w = cols * panel_w + (cols - 1)
    out = np.full((out_h, out_w), int(sep_color), dtype=int)

    for r in range(rows):
        for c in range(cols):
            rr = r * (panel_h + 1)
            cc = c * (panel_w + 1)
            out[rr : rr + panel_h, cc : cc + panel_w] = panel_rows[r][c]
    return out


def _random_edge_sequence(
    length: int, start: int, end: int, palette: tuple[int, ...], used: set[tuple[int, ...]]
) -> np.ndarray | None:
    for _ in range(200):
        if length == 2:
            seq = (int(start), int(end))
        else:
            middle = [random.choice(palette) for _ in range(length - 2)]
            seq = (int(start), *middle, int(end))
        if seq in used:
            continue
        used.add(seq)
        return np.array(seq, dtype=int)
    return None


def _decorate_panel(panel: np.ndarray, fill_palette: tuple[int, ...]) -> None:
    h, w = map(int, panel.shape)
    if h <= 2 or w <= 2:
        return

    inner_h = h - 2
    inner_w = w - 2
    motif_count = random.randint(1, 3)
    for _ in range(motif_count):
        color = int(random.choice(fill_palette))
        mode = random.choice(("rect", "hline", "vline", "diag", "dots"))

        if mode == "rect":
            r0 = random.randint(1, h - 2)
            r1 = random.randint(r0, h - 2)
            c0 = random.randint(1, w - 2)
            c1 = random.randint(c0, w - 2)
            panel[r0 : r1 + 1, c0 : c1 + 1] = color
        elif mode == "hline":
            rr = random.randint(1, h - 2)
            c0 = random.randint(1, w - 2)
            c1 = random.randint(c0, w - 2)
            panel[rr, c0 : c1 + 1] = color
        elif mode == "vline":
            cc = random.randint(1, w - 2)
            r0 = random.randint(1, h - 2)
            r1 = random.randint(r0, h - 2)
            panel[r0 : r1 + 1, cc] = color
        elif mode == "diag":
            steps = min(inner_h, inner_w)
            if steps <= 0:
                continue
            r0 = random.randint(1, h - steps)
            c0 = random.randint(1, w - steps)
            direction = random.choice((-1, 1))
            for t in range(steps):
                panel[r0 + t, c0 + (t if direction > 0 else steps - 1 - t)] = color
        else:
            count = random.randint(1, max(1, inner_h * inner_w // 3))
            for _ in range(count):
                rr = random.randint(1, h - 2)
                cc = random.randint(1, w - 2)
                panel[rr, cc] = color

    if not np.any(panel[1:-1, 1:-1] != 0):
        panel[random.randint(1, h - 2), random.randint(1, w - 2)] = int(random.choice(fill_palette))


def _build_solved_panels(rows: int, cols: int, panel_h: int, panel_w: int) -> list[np.ndarray] | None:
    edge_palette = tuple(random.sample(EDGE_COLOR_POOL, random.randint(4, 6)))
    fill_palette = edge_palette
    junctions = np.array(
        [[random.choice(edge_palette) for _ in range(cols + 1)] for _ in range(rows + 1)],
        dtype=int,
    )

    used_vertical: set[tuple[int, ...]] = set()
    used_horizontal: set[tuple[int, ...]] = set()
    vertical: list[list[np.ndarray]] = []
    horizontal: list[list[np.ndarray]] = []

    for r in range(rows):
        segs: list[np.ndarray] = []
        for c in range(cols + 1):
            seq = _random_edge_sequence(
                panel_h,
                int(junctions[r, c]),
                int(junctions[r + 1, c]),
                edge_palette,
                used_vertical,
            )
            if seq is None:
                return None
            segs.append(seq)
        vertical.append(segs)

    for r in range(rows + 1):
        segs = []
        for c in range(cols):
            seq = _random_edge_sequence(
                panel_w,
                int(junctions[r, c]),
                int(junctions[r, c + 1]),
                edge_palette,
                used_horizontal,
            )
            if seq is None:
                return None
            segs.append(seq)
        horizontal.append(segs)

    panels: list[np.ndarray] = []
    for r in range(rows):
        for c in range(cols):
            panel = np.zeros((panel_h, panel_w), dtype=int)
            panel[0, :] = horizontal[r][c]
            panel[-1, :] = horizontal[r + 1][c]
            panel[:, 0] = vertical[r][c]
            panel[:, -1] = vertical[r][c + 1]
            _decorate_panel(panel, fill_palette)
            panels.append(panel)
    return panels


def _gen_attempt() -> np.ndarray | None:
    rows, cols = random.choice(PANEL_GRID_CHOICES)
    valid_sizes = [
        (h, w)
        for h in PANEL_SIZE_RANGE
        for w in PANEL_SIZE_RANGE
        if rows * h + (rows - 1) <= 30 and cols * w + (cols - 1) <= 30
    ]
    panel_h, panel_w = random.choice(valid_sizes)

    panels = _build_solved_panels(rows, cols, panel_h, panel_w)
    if panels is None:
        return None

    solved_rows = [[panels[r * cols + c] for c in range(cols)] for r in range(rows)]
    solved_grid = _compose_grid(solved_rows, SEPARATOR_COLOR)

    order = list(range(len(panels)))
    random.shuffle(order)
    if order == list(range(len(panels))):
        order[0], order[1] = order[1], order[0]

    shuffled_rows = [[panels[order[r * cols + c]] for c in range(cols)] for r in range(rows)]
    inp = _compose_grid(shuffled_rows, SEPARATOR_COLOR)

    try:
        recovered = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(recovered, solved_grid):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-reordering input")
