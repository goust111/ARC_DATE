import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

PANEL_COUNT_RANGE = (3, 5)
GRID_HEIGHT_RANGE = (9, 13)
PANEL_WIDTH_RANGE = (8, 13)
GENERATION_ATTEMPTS = 400
LOCAL_ATTEMPTS = 200


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
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask_1d.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask_1d)))
    return runs


def _panel_spans(grid: np.ndarray) -> list[tuple[int, int]]:
    sep_mask = np.all(grid == GREY, axis=0)
    sep_runs = _runs(sep_mask)
    assert sep_runs, "expected at least one full-height separator column"

    spans: list[tuple[int, int]] = []
    prev = 0
    for start, end in sep_runs:
        if prev < start:
            spans.append((prev, start))
        prev = end
    if prev < grid.shape[1]:
        spans.append((prev, grid.shape[1]))

    assert len(spans) >= 2, "expected multiple panels"
    return spans


def main(grid: np.ndarray) -> np.ndarray:
    """Pair the small signal in each separator-delimited panel to recolor the larger receiver."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    out = g.copy()
    for c0, c1 in _panel_spans(g):
        panel = g[:, c0:c1]
        comps = _components(panel != BLACK)
        assert len(comps) == 2, "each panel must contain exactly two foreground sprites"

        comps.sort(key=len)
        signal, receiver = comps
        assert len(signal) < len(receiver), "signal must be strictly smaller than receiver"

        sig_color = int(panel[signal[0]])
        assert sig_color in {RED, BLUE}, "signal color must be red or blue"
        assert {int(panel[r, c]) for r, c in receiver} == {GREY}, "receiver must be grey"

        r0, c0_local, r1, c1_local = _bbox(signal)
        sig_h = r1 - r0 + 1
        sig_w = c1_local - c0_local + 1
        if sig_w > sig_h:
            receiver_color = GREEN
        elif sig_h > sig_w:
            receiver_color = YELLOW
        else:
            receiver_color = sig_color

        for r, c in signal:
            out[r, c0 + c] = BLACK
        for r, c in receiver:
            out[r, c0 + c] = receiver_color

    return out


def _signal_mask_and_color(
    rng: np.random.Generator, branch: str
) -> tuple[np.ndarray, int]:
    if branch == "horizontal":
        height = int(rng.integers(1, 3))
        width = int(rng.integers(height + 1, 6))
        color = RED if rng.random() < 0.5 else BLUE
        return np.ones((height, width), dtype=bool), color

    if branch == "vertical":
        width = int(rng.integers(1, 3))
        height = int(rng.integers(width + 1, 6))
        color = RED if rng.random() < 0.5 else BLUE
        return np.ones((height, width), dtype=bool), color

    side = int(rng.integers(2, 4))
    color = RED if rng.random() < 0.5 else BLUE
    return np.ones((side, side), dtype=bool), color


def _random_connected_mask(
    rng: np.random.Generator,
    height: int,
    width: int,
    min_area: int,
    max_area: int,
) -> np.ndarray | None:
    max_area = min(max_area, height * width)
    min_area = max(min_area, 1)
    if min_area > max_area:
        return None

    for _ in range(LOCAL_ATTEMPTS):
        target = int(rng.integers(min_area, max_area + 1))
        mask = np.zeros((height, width), dtype=bool)
        seed = (int(rng.integers(0, height)), int(rng.integers(0, width)))
        mask[seed] = True

        while int(mask.sum()) < target:
            frontier: list[tuple[int, int]] = []
            frontier_set: set[tuple[int, int]] = set()
            for r, c in zip(*np.where(mask)):
                for dr, dc in DIRS4:
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                        if (nr, nc) not in frontier_set:
                            frontier_set.add((nr, nc))
                            frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = frontier[int(rng.integers(0, len(frontier)))]
            mask[nr, nc] = True

        if int(mask.sum()) != target:
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        return mask
    return None


def _can_place(occupied: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = occupied.shape
    mh, mw = mask.shape
    if top < 1 or left < 1 or top + mh > h - 1 or left + mw > w - 1:
        return False

    rows, cols = np.where(mask)
    for rr_raw, cc_raw in zip(rows.tolist(), cols.tolist()):
        rr = top + rr_raw
        cc = left + cc_raw
        for dr, dc in ((0, 0),) + DIRS4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                return False
    return True


def _place_mask(panel: np.ndarray, occupied: np.ndarray, mask: np.ndarray, color: int, rng: np.random.Generator) -> bool:
    h, w = panel.shape
    mh, mw = mask.shape
    positions: list[tuple[int, int]] = []
    for top in range(1, h - mh):
        for left in range(1, w - mw):
            if _can_place(occupied, mask, top, left):
                positions.append((top, left))
    if not positions:
        return False

    top, left = positions[int(rng.integers(0, len(positions)))]
    block = panel[top : top + mh, left : left + mw]
    block[mask] = color
    occupied[top : top + mh, left : left + mw] |= mask
    return True


def _build_panel(rng: np.random.Generator, height: int, branch: str) -> np.ndarray | None:
    for _ in range(LOCAL_ATTEMPTS):
        width = int(rng.integers(PANEL_WIDTH_RANGE[0], PANEL_WIDTH_RANGE[1] + 1))
        panel = np.full((height, width), BLACK, dtype=int)
        occupied = np.zeros((height, width), dtype=bool)

        signal_mask, signal_color = _signal_mask_and_color(rng, branch)
        signal_area = int(signal_mask.sum())

        max_receiver_h = min(6, height - 2)
        max_receiver_w = min(6, width - 2)
        if max_receiver_h < 3 or max_receiver_w < 3:
            continue

        receiver_h = int(rng.integers(3, max_receiver_h + 1))
        receiver_w = int(rng.integers(3, max_receiver_w + 1))
        receiver_min = max(signal_area + 2, (receiver_h * receiver_w + 1) // 2)
        receiver_max = max(receiver_min, receiver_h * receiver_w - 1)
        receiver_mask = _random_connected_mask(rng, receiver_h, receiver_w, receiver_min, receiver_max)
        if receiver_mask is None:
            continue

        if not _place_mask(panel, occupied, receiver_mask, GREY, rng):
            continue
        if not _place_mask(panel, occupied, signal_mask, signal_color, rng):
            continue

        comps = _components(panel != BLACK)
        if len(comps) != 2:
            continue
        areas = sorted(len(comp) for comp in comps)
        if areas[0] >= areas[1]:
            continue
        return panel

    return None


def _branch_schedule(rng: np.random.Generator, panel_count: int) -> list[str]:
    branches = ["horizontal", "vertical", "square"]
    while len(branches) < panel_count:
        branches.append(["horizontal", "vertical", "square"][int(rng.integers(0, 3))])
    order = np.array(branches, dtype=object)
    rng.shuffle(order)
    return [str(x) for x in order.tolist()]


def _is_valid_input(grid: np.ndarray) -> bool:
    try:
        spans = _panel_spans(grid)
        seen_square = False
        seen_horizontal = False
        seen_vertical = False
        for c0, c1 in spans:
            panel = grid[:, c0:c1]
            comps = _components(panel != BLACK)
            if len(comps) != 2:
                return False
            comps.sort(key=len)
            signal, receiver = comps
            if len(signal) >= len(receiver):
                return False
            if int(panel[signal[0]]) not in {RED, BLUE}:
                return False
            if {int(panel[r, c]) for r, c in receiver} != {GREY}:
                return False
            r0, c0_local, r1, c1_local = _bbox(signal)
            sig_h = r1 - r0 + 1
            sig_w = c1_local - c0_local + 1
            if sig_w > sig_h:
                seen_horizontal = True
            elif sig_h > sig_w:
                seen_vertical = True
            else:
                seen_square = True
        if len(spans) >= 3 and not (seen_horizontal and seen_vertical and seen_square):
            return False
        _ = main(grid)
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    height = int(rng.integers(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1] + 1))
    panel_count = int(rng.integers(PANEL_COUNT_RANGE[0], PANEL_COUNT_RANGE[1] + 1))
    branches = _branch_schedule(rng, panel_count)

    panels: list[np.ndarray] = []
    for branch in branches:
        panel = _build_panel(rng, height, branch)
        if panel is None:
            return None
        panels.append(panel)

    total_width = sum(panel.shape[1] for panel in panels) + (len(panels) - 1)
    grid = np.full((height, total_width), BLACK, dtype=int)

    col = 0
    for idx, panel in enumerate(panels):
        width = panel.shape[1]
        grid[:, col : col + width] = panel
        col += width
        if idx + 1 < len(panels):
            grid[:, col] = GREY
            col += 1

    return grid if _is_valid_input(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
