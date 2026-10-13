import random
from collections import Counter

import numpy as np


WINDOW_SIZE = 7
BLOCK_SIZE = 2
BLOCK_STEP = 3

PANE_SLICES = {
    "tl": (slice(1, 3), slice(1, 3)),
    "tr": (slice(1, 3), slice(4, 6)),
    "bl": (slice(4, 6), slice(1, 3)),
    "br": (slice(4, 6), slice(4, 6)),
}
SIGNATURES = [(a, b, c) for a in (False, True) for b in (False, True) for c in (False, True)]

COUNT_CHOICES = (3, 4)
MARGIN_RANGE = (0, 1)
SMALL_GAP_RANGE = (1, 2)
WORK_GAP_RANGE = (1, 3)
EXTRA_PAD_RANGE = (0, 2)
PALETTE_SIZE_RANGE = (3, 6)
GEN_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _parse_window_patch(patch: np.ndarray, bg: int) -> dict | None:
    if patch.shape != (WINDOW_SIZE, WINDOW_SIZE):
        return None

    color = int(patch[0, 0])
    if color == bg:
        return None

    if not (
        np.all(patch[0, :] == color)
        and np.all(patch[-1, :] == color)
        and np.all(patch[:, 0] == color)
        and np.all(patch[:, -1] == color)
        and np.all(patch[3, :] == color)
        and np.all(patch[:, 3] == color)
    ):
        return None

    patch_colors = {int(v) for v in np.unique(patch)}
    if patch_colors - {bg, color}:
        return None

    panes: dict[str, bool] = {}
    for name, (rows, cols) in PANE_SLICES.items():
        pane = patch[rows, cols]
        values = np.unique(pane)
        if values.size != 1:
            return None
        value = int(values[0])
        if value not in (bg, color):
            return None
        panes[name] = value == bg

    return {
        "color": color,
        "active": panes["tl"],
        "sig": (panes["tr"], panes["bl"], panes["br"]),
    }


def _find_windows(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    windows: list[dict] = []
    for r in range(h - WINDOW_SIZE + 1):
        for c in range(w - WINDOW_SIZE + 1):
            info = _parse_window_patch(grid[r : r + WINDOW_SIZE, c : c + WINDOW_SIZE], bg)
            if info is None:
                continue
            info = info.copy()
            info["r"] = int(r)
            info["c"] = int(c)
            windows.append(info)
    return windows


def _split_windows(windows: list[dict]) -> tuple[list[dict], list[dict]]:
    assert windows, "expected at least one window"

    top_row = min(int(w["r"]) for w in windows)
    left_col = min(int(w["c"]) for w in windows)

    top = [w for w in windows if int(w["r"]) == top_row]
    left = [w for w in windows if int(w["c"]) == left_col]

    top_pos = {(int(w["r"]), int(w["c"])) for w in top}
    left_pos = {(int(w["r"]), int(w["c"])) for w in left}
    assert top and left, "expected both top and left window groups"
    assert top_pos.isdisjoint(left_pos), "top-left corner must remain empty"
    assert len(top) + len(left) == len(windows), "every window must belong to exactly one group"

    top.sort(key=lambda w: int(w["c"]))
    left.sort(key=lambda w: int(w["r"]))
    return left, top


def _workspace_origin(grid: np.ndarray, n_rows: int, n_cols: int) -> tuple[int, int]:
    h, w = grid.shape
    work_h = BLOCK_STEP * n_rows - 1
    work_w = BLOCK_STEP * n_cols - 1
    assert work_h <= h and work_w <= w, "workspace does not fit in grid"
    return h - work_h, w - work_w


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read window-pane activation/signature patterns and mark matching row/column intersections.

    Input structure:
    - bg = most common color.
    - The top band contains 7x7 window icons ordered left-to-right.
    - The left band contains 7x7 window icons ordered top-to-bottom.
    - Each window has a colored frame plus a colored cross, splitting the inside into four 2x2 panes.
    - The bottom-right workspace is a blank bg rectangle whose logical cells are 2x2 blocks separated by 1 bg pixel.

    Rule:
    - A window is active iff its top-left pane is open (bg).
    - The remaining three panes form a signature in (top-right, bottom-left, bottom-right) order.
    - For each active row window and active column window with identical signatures, paint a 2x2 block at their workspace intersection.
    - Use the row-window border color when both borders are the same; otherwise use the column-window border color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    windows = _find_windows(g, bg)
    rows, cols = _split_windows(windows)

    work_r0, work_c0 = _workspace_origin(g, len(rows), len(cols))
    assert np.all(g[work_r0:, work_c0:] == bg), "workspace must be blank in the input"

    out = g.copy()
    for i, row_win in enumerate(rows):
        if not bool(row_win["active"]):
            continue
        for j, col_win in enumerate(cols):
            if not bool(col_win["active"]):
                continue
            if tuple(row_win["sig"]) != tuple(col_win["sig"]):
                continue
            rr = work_r0 + BLOCK_STEP * i
            cc = work_c0 + BLOCK_STEP * j
            color = int(row_win["color"])
            if int(row_win["color"]) != int(col_win["color"]):
                color = int(col_win["color"])
            out[rr : rr + BLOCK_SIZE, cc : cc + BLOCK_SIZE] = color

    return out


# --------------- generator (codex) ---------------

def _sample_positions(count: int, start: int, rng: random.Random) -> list[int]:
    positions = [start]
    for _ in range(count - 1):
        positions.append(positions[-1] + WINDOW_SIZE + rng.randint(*SMALL_GAP_RANGE))
    return positions


def _make_specs(count: int, match_sig: tuple[bool, bool, bool], alt_sig: tuple[bool, bool, bool], rng: random.Random) -> tuple[list[dict], int]:
    order = list(range(count))
    rng.shuffle(order)

    specs = [None] * count
    specs[order[0]] = {"active": True, "sig": match_sig}
    specs[order[1]] = {"active": True, "sig": alt_sig}
    specs[order[2]] = {"active": False, "sig": rng.choice(SIGNATURES)}

    for idx in order[3:]:
        specs[idx] = {"active": rng.random() < 0.6, "sig": rng.choice(SIGNATURES)}

    return specs, order[0]


def _draw_window(canvas: np.ndarray, r: int, c: int, color: int, bg: int, active: bool, sig: tuple[bool, bool, bool]) -> None:
    patch = canvas[r : r + WINDOW_SIZE, c : c + WINDOW_SIZE]
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    patch[3, :] = color
    patch[:, 3] = color

    pane_states = {
        "tl": active,
        "tr": bool(sig[0]),
        "bl": bool(sig[1]),
        "br": bool(sig[2]),
    }
    for name, is_open in pane_states.items():
        rows, cols = PANE_SLICES[name]
        patch[rows, cols] = bg if is_open else color


def _expected_hits(row_specs: list[dict], col_specs: list[dict]) -> list[tuple[int, int]]:
    hits = []
    for i, row_spec in enumerate(row_specs):
        if not bool(row_spec["active"]):
            continue
        for j, col_spec in enumerate(col_specs):
            if not bool(col_spec["active"]):
                continue
            if tuple(row_spec["sig"]) == tuple(col_spec["sig"]):
                hits.append((i, j))
    return hits


def _strict_bg_dominance(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.reshape(-1)))
    bg = get_bg(grid)
    return sum(count == counts[bg] for count in counts.values()) == 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rows = rng.choice(COUNT_CHOICES)
    n_cols = rng.choice(COUNT_CHOICES)

    work_h = BLOCK_STEP * n_rows - 1
    work_w = BLOCK_STEP * n_cols - 1

    top_r = rng.randint(*MARGIN_RANGE)
    left_c = rng.randint(*MARGIN_RANGE)
    top_c0 = left_c + WINDOW_SIZE + rng.randint(*SMALL_GAP_RANGE)
    left_r0 = top_r + WINDOW_SIZE + rng.randint(*SMALL_GAP_RANGE)

    row_positions = _sample_positions(n_rows, left_r0, rng)
    col_positions = _sample_positions(n_cols, top_c0, rng)
    left_span = row_positions[-1] + WINDOW_SIZE
    top_span = col_positions[-1] + WINDOW_SIZE

    min_h = max(top_r + WINDOW_SIZE + rng.randint(*WORK_GAP_RANGE) + work_h, left_span)
    min_w = max(left_c + WINDOW_SIZE + rng.randint(*WORK_GAP_RANGE) + work_w, top_span)
    h = min_h + rng.randint(*EXTRA_PAD_RANGE)
    w = min_w + rng.randint(*EXTRA_PAD_RANGE)

    bg = rng.randrange(10)
    grid = np.full((h, w), bg, dtype=int)

    color_pool = [c for c in range(10) if c != bg]
    palette_size = min(len(color_pool), rng.randint(*PALETTE_SIZE_RANGE))
    palette = rng.sample(color_pool, k=palette_size)

    match_sig = rng.choice(SIGNATURES)
    nonmatch_choices = [sig for sig in SIGNATURES if sig != match_sig]
    row_alt_sig = rng.choice(nonmatch_choices)
    col_alt_sig = rng.choice(nonmatch_choices)

    row_specs, row_match_idx = _make_specs(n_rows, match_sig, row_alt_sig, rng)
    col_specs, col_match_idx = _make_specs(n_cols, match_sig, col_alt_sig, rng)

    row_colors = [rng.choice(palette) for _ in range(n_rows)]
    col_colors = [rng.choice(palette) for _ in range(n_cols)]
    if rng.random() < 0.5:
        shared = rng.choice(palette)
        row_colors[row_match_idx] = shared
        col_colors[col_match_idx] = shared
    else:
        row_colors[row_match_idx] = rng.choice(palette)
        diff_pool = [c for c in palette if c != row_colors[row_match_idx]]
        if not diff_pool:
            return None
        col_colors[col_match_idx] = rng.choice(diff_pool)

    for i, r0 in enumerate(row_positions):
        _draw_window(
            grid,
            r0,
            left_c,
            row_colors[i],
            bg,
            bool(row_specs[i]["active"]),
            tuple(row_specs[i]["sig"]),
        )
    for j, c0 in enumerate(col_positions):
        _draw_window(
            grid,
            top_r,
            c0,
            col_colors[j],
            bg,
            bool(col_specs[j]["active"]),
            tuple(col_specs[j]["sig"]),
        )

    hits = _expected_hits(row_specs, col_specs)
    active_pair_count = sum(
        1
        for row_spec in row_specs
        for col_spec in col_specs
        if bool(row_spec["active"]) and bool(col_spec["active"])
    )
    if not hits or len(hits) == active_pair_count:
        return None
    if not any(not bool(spec["active"]) for spec in row_specs):
        return None
    if not any(not bool(spec["active"]) for spec in col_specs):
        return None
    if not _strict_bg_dominance(grid):
        return None

    try:
        parsed = _find_windows(grid, bg)
        rows, cols = _split_windows(parsed)
        if len(rows) != n_rows or len(cols) != n_cols:
            return None
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to produce a valid sample")
