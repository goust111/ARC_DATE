from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5

PANEL_SIZE = 5
PANEL_ROWS = 2
PANEL_COL_CHOICES = (2, 3)
MID = PANEL_SIZE // 2
GEN_RETRIES = 600


def _build_sequence(n: int) -> np.ndarray:
    run_lengths = list(range(1, n + 1)) + list(range(n - 1, 0, -1))
    seq: list[int] = []
    for i, run_len in enumerate(run_lengths):
        seq.extend([BLUE] * run_len)
        if i != len(run_lengths) - 1:
            seq.append(BLACK)
    return np.asarray(seq, dtype=int)


def _control_length(grid: np.ndarray) -> int:
    top = np.asarray(grid[0], dtype=int)
    n = 0
    while n < top.size and int(top[n]) == BLUE:
        n += 1
    assert 2 <= n <= 5, "control bar length must be in [2, 5]"
    assert np.all(top[:n] == BLUE), "control bar must be contiguous"
    assert np.all(top[n:] == BLACK), "only the control bar may appear in the top row"
    return n


def _separator_rows() -> tuple[int, ...]:
    return tuple(k * (PANEL_SIZE + 1) for k in range(1, PANEL_ROWS))


def _separator_cols(nc: int) -> tuple[int, ...]:
    return tuple(PANEL_SIZE + k * (PANEL_SIZE + 1) for k in range(nc - 1))


def _panel_origin(pr: int, pc: int) -> tuple[int, int]:
    return 1 + pr * (PANEL_SIZE + 1), pc * (PANEL_SIZE + 1)


def _panel_slice(pr: int, pc: int) -> tuple[slice, slice]:
    r0, c0 = _panel_origin(pr, pc)
    return slice(r0, r0 + PANEL_SIZE), slice(c0, c0 + PANEL_SIZE)


def _infer_layout(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h == PANEL_ROWS * (PANEL_SIZE + 1), "unexpected row count"
    assert (w + 1) % (PANEL_SIZE + 1) == 0, "unexpected column count"
    nc = (w + 1) // (PANEL_SIZE + 1)
    assert nc in PANEL_COL_CHOICES, "expected a 2x2 or 2x3 panel lattice"

    sep_rows = _separator_rows()
    sep_cols = _separator_cols(nc)

    for r in sep_rows:
        assert np.all(g[r, :] == GREY), "horizontal grid lines must be grey"
    for c in sep_cols:
        assert np.all(g[1:, c] == GREY), "vertical grid lines must be grey below the control row"
        assert int(g[0, c]) in (BLACK, BLUE), "control row cannot contain grid separators"

    orientations: list[list[str]] = []
    for pr in range(PANEL_ROWS):
        row_orientations: list[str] = []
        for pc in range(nc):
            rs, cs = _panel_slice(pr, pc)
            panel = g[rs, cs]
            assert panel.shape == (PANEL_SIZE, PANEL_SIZE)

            has_h = bool(np.all(panel[MID, :] == GREY))
            has_v = bool(np.all(panel[:, MID] == GREY))
            assert has_h ^ has_v, "each panel must contain exactly one guide line"
            assert np.count_nonzero(panel == GREY) == PANEL_SIZE, "panel greys must belong only to the guide line"
            assert np.all((panel == BLACK) | (panel == GREY)), "panel interiors must start empty"

            row_orientations.append("H" if has_h else "V")
        orientations.append(row_orientations)

    flat = [ori for row in orientations for ori in row]
    assert "H" in flat and "V" in flat, "need both mirror directions"

    return {
        "nc": nc,
        "sep_rows": set(sep_rows),
        "sep_cols": set(sep_cols),
        "orientations": orientations,
    }


def _fill_positions(shape: tuple[int, int], meta: dict, *, skip_guides: bool) -> list[tuple[int, int, int, int]]:
    h, w = map(int, shape)
    cells: list[tuple[int, int, int, int]] = []
    for r in range(1, h):
        if r in meta["sep_rows"]:
            continue
        pr = (r - 1) // (PANEL_SIZE + 1)
        lr = (r - 1) % (PANEL_SIZE + 1)
        for c in range(w):
            if c in meta["sep_cols"]:
                continue
            pc = c // (PANEL_SIZE + 1)
            lc = c % (PANEL_SIZE + 1)
            if skip_guides:
                ori = meta["orientations"][pr][pc]
                if ori == "H" and lr == MID:
                    continue
                if ori == "V" and lc == MID:
                    continue
            cells.append((r, c, pr, pc))
    return cells


def _transform(
    grid: np.ndarray,
    meta: dict | None = None,
    *,
    skip_guides: bool = True,
    do_mirror: bool = True,
    return_pre_mirror: bool = False,
):
    g = np.asarray(grid, dtype=int)
    meta = _infer_layout(g) if meta is None else meta
    out = g.copy()

    seq = _build_sequence(_control_length(g))
    cells = _fill_positions(g.shape, meta, skip_guides=skip_guides)
    for idx, value in enumerate(seq):
        if idx >= len(cells):
            break
        r, c, _pr, _pc = cells[idx]
        out[r, c] = int(value)

    pre_mirror = out.copy()
    if do_mirror:
        for pr in range(PANEL_ROWS):
            for pc in range(meta["nc"]):
                rs, cs = _panel_slice(pr, pc)
                panel = out[rs, cs].copy()
                if meta["orientations"][pr][pc] == "H":
                    out[rs, cs] = np.flipud(panel)
                else:
                    out[rs, cs] = np.fliplr(panel)

    if return_pre_mirror:
        return out, pre_mirror, cells, seq
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a panel lattice with a wrapped control-bar sequence, then mirror each panel by its own guide line.

    Input structure:
    - The top row contains a left-aligned blue control bar on black background.
    - Below it is a 2x2 or 2x3 lattice of 5x5 panels separated by one-cell-wide grey grid lines.
    - Each panel is otherwise black and contains exactly one grey center guide: either a full middle row or a full middle column.

    Rule:
    - Read the top-row bar length n.
    - Build a 1D blue/black sequence whose blue run lengths are 1, 2, ..., n, n-1, ..., 1 with one black separator between adjacent runs.
    - Write that sequence in row-major order through the panel area, skipping grey separator rows and columns and also skipping each panel's grey guide cells; untouched panel cells stay black.
    - Then mirror each panel independently: a middle column means left-right reflection, and a middle row means top-bottom reflection. The top row and every grey line stay in place.

    Key insight:
    - The control bar defines one shared global signal, but the lattice distributes different fragments of that signal to different panels before each panel applies its own local mirror rule.
    """
    return _transform(grid)


def _build_input(n: int, nc: int, orientations: list[list[str]]) -> np.ndarray:
    h = PANEL_ROWS * (PANEL_SIZE + 1)
    w = nc * (PANEL_SIZE + 1) - 1
    grid = np.zeros((h, w), dtype=int)
    grid[0, :n] = BLUE

    for r in _separator_rows():
        grid[r, :] = GREY
    for c in _separator_cols(nc):
        grid[1:, c] = GREY

    for pr in range(PANEL_ROWS):
        for pc in range(nc):
            rs, cs = _panel_slice(pr, pc)
            if orientations[pr][pc] == "H":
                grid[rs.start + MID, cs] = GREY
            else:
                grid[rs, cs.start + MID] = GREY
    return grid


def _sample_orientations(nc: int) -> list[list[str]]:
    while True:
        orientations = [[random.choice(("H", "V")) for _ in range(nc)] for _ in range(PANEL_ROWS)]
        flat = [ori for row in orientations for ori in row]
        if "H" not in flat or "V" not in flat:
            continue
        if any(all(orientations[pr][pc] == "H" for pc in range(nc)) for pr in range(PANEL_ROWS)):
            continue
        if any(all(orientations[pr][pc] == "V" for pr in range(PANEL_ROWS)) for pc in range(nc)):
            continue
        return orientations


def _panel_stats(pre: np.ndarray, post: np.ndarray, meta: dict) -> dict[str, int]:
    stats = {
        "blue_panels_H": 0,
        "blue_panels_V": 0,
        "changed_H": 0,
        "changed_V": 0,
        "blue_panel_count": 0,
        "blue_panel_cols": 0,
    }
    blue_cols: set[int] = set()

    for pr in range(PANEL_ROWS):
        for pc in range(meta["nc"]):
            rs, cs = _panel_slice(pr, pc)
            before = pre[rs, cs]
            after = post[rs, cs]
            ori = meta["orientations"][pr][pc]
            has_blue = bool(np.any(after == BLUE))
            changed = bool(np.any(before != after))

            if has_blue:
                stats[f"blue_panels_{ori}"] += 1
                stats["blue_panel_count"] += 1
                blue_cols.add(pc)
            if has_blue and changed:
                stats[f"changed_{ori}"] += 1

    stats["blue_panel_cols"] = len(blue_cols)
    return stats


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        meta = _infer_layout(grid)
        correct, pre_mirror, cells, seq = _transform(grid, meta, return_pre_mirror=True)
        no_mirror = pre_mirror
        no_skip = _transform(grid, meta, skip_guides=False, do_mirror=True)
    except Exception:
        return False

    if len(cells) <= len(seq):
        return False
    if np.array_equal(correct, grid):
        return False
    if np.array_equal(correct, no_mirror):
        return False
    if np.array_equal(correct, no_skip):
        return False

    stats = _panel_stats(pre_mirror, correct, meta)
    if stats["blue_panels_H"] == 0 or stats["blue_panels_V"] == 0:
        return False
    if stats["changed_H"] == 0 or stats["changed_V"] == 0:
        return False
    if stats["blue_panel_count"] < 2:
        return False
    if stats["blue_panel_cols"] < 2:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        nc = random.choice(PANEL_COL_CHOICES)
        n = random.randint(2, 5)
        orientations = _sample_orientations(nc)
        grid = _build_input(n, nc, orientations)
        if _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
