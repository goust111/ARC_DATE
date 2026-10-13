import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )

    return components


def _is_regular_line_sequence(lines: np.ndarray) -> bool:
    if lines.size < 2:
        return False
    gaps = np.diff(lines)
    return bool(gaps.size >= 1 and np.all(gaps == gaps[0]) and int(gaps[0]) >= 2)


def _window_cell_count(grid: np.ndarray, component: dict) -> int:
    r0, c0, r1, c1 = component["bbox"]
    color = int(component["color"])
    sub = grid[r0 : r1 + 1, c0 : c1 + 1] == color

    full_rows = np.flatnonzero(np.all(sub, axis=1))
    full_cols = np.flatnonzero(np.all(sub, axis=0))

    if not _is_regular_line_sequence(full_rows):
        raise ValueError("window rows are not regularly spaced")
    if not _is_regular_line_sequence(full_cols):
        raise ValueError("window cols are not regularly spaced")
    if int(full_rows[0]) != 0 or int(full_rows[-1]) != sub.shape[0] - 1:
        raise ValueError("window must span its bbox vertically")
    if int(full_cols[0]) != 0 or int(full_cols[-1]) != sub.shape[1] - 1:
        raise ValueError("window must span its bbox horizontally")

    row_mask = np.zeros(sub.shape[0], dtype=bool)
    col_mask = np.zeros(sub.shape[1], dtype=bool)
    row_mask[full_rows] = True
    col_mask[full_cols] = True
    expected = row_mask[:, None] | col_mask[None, :]
    if not np.array_equal(sub, expected):
        raise ValueError("component is not a window grid")

    return int((full_rows.size - 1) * (full_cols.size - 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each gridded rectangular window into a bottom-aligned bar whose
    height equals the number of cells induced by the window's internal lines.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    windows: list[tuple[int, int, int, int]] = []
    for component in _find_components(g, bg):
        cells = _window_cell_count(g, component)
        r0, c0, _r1, _c1 = component["bbox"]
        windows.append((int(c0), int(r0), int(component["color"]), int(cells)))

    if not windows:
        return np.full((1, 1), bg, dtype=int)

    windows.sort(key=lambda item: (item[0], item[1]))
    out_h = max(cells for _c0, _r0, _color, cells in windows)
    out = np.full((out_h, len(windows)), bg, dtype=int)

    for out_c, (_c0, _r0, color, cells) in enumerate(windows):
        out[out_h - cells :, out_c] = int(color)

    return out


# --------------- generator ---------------

_RNG = np.random.default_rng()
_BRANCHES = ("horizontal", "vertical", "both")
_GRID_MAX_H = 22
_GRID_MAX_W = 30
_MAX_ATTEMPTS = 400


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(lo, hi + 1))


def _build_window(row_cells: int, col_cells: int, cell_h: int, cell_w: int, color: int, bg: int) -> np.ndarray:
    height = row_cells * cell_h + row_cells + 1
    width = col_cells * cell_w + col_cells + 1
    patch = np.full((height, width), bg, dtype=int)

    for rr in range(0, height, cell_h + 1):
        patch[rr, :] = color
    for cc in range(0, width, cell_w + 1):
        patch[:, cc] = color

    return patch


def _sample_branches(count: int) -> list[str]:
    branches = list(_BRANCHES)
    _RNG.shuffle(branches)
    if count <= len(branches):
        return branches[:count]

    out = branches[:]
    out.extend(_RNG.choice(branches, size=count - len(branches), replace=True).tolist())
    _RNG.shuffle(out)
    return out


def _sample_window(branch: str, color: int, bg: int) -> dict:
    if branch == "horizontal":
        row_cells = _randint(2, 5)
        col_cells = 1
        cell_h = _randint(1, 2)
        cell_w = _randint(2, 4)
    elif branch == "vertical":
        row_cells = 1
        col_cells = _randint(2, 3)
        cell_h = _randint(2, 4)
        cell_w = _randint(1, 2)
    else:
        row_cells = _randint(2, 3)
        col_cells = _randint(2, 3)
        cell_h = _randint(1, 2)
        cell_w = _randint(1, 2)

    patch = _build_window(row_cells, col_cells, cell_h, cell_w, color, bg)
    return {
        "branch": branch,
        "color": int(color),
        "cells": int(row_cells * col_cells),
        "patch": patch,
        "height": int(patch.shape[0]),
        "width": int(patch.shape[1]),
    }


def _expected_output(windows: list[dict], bg: int) -> np.ndarray:
    ordered = sorted(windows, key=lambda item: (int(item["c0"]), int(item["r0"])))
    out_h = max(int(item["cells"]) for item in ordered)
    out = np.full((out_h, len(ordered)), bg, dtype=int)
    for c, item in enumerate(ordered):
        out[out_h - int(item["cells"]) :, c] = int(item["color"])
    return out


def _bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    bg_count = int(np.count_nonzero(grid == bg))
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        if bg_count <= int(np.count_nonzero(grid == color)):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_windows = int(_RNG.choice([2, 3, 4], p=[0.25, 0.5, 0.25]))
    bg = _randint(0, 9)
    palette = [color for color in range(10) if color != bg]
    colors = _RNG.choice(palette, size=n_windows, replace=False)

    windows = [_sample_window(branch, int(color), bg) for branch, color in zip(_sample_branches(n_windows), colors)]
    counts = [int(window["cells"]) for window in windows]
    if len(set(counts)) < 2:
        return None

    widths = [int(window["width"]) for window in windows]
    heights = [int(window["height"]) for window in windows]
    left_margin = _randint(1, 2)
    right_margin = _randint(1, 2)
    gaps = [_randint(1, 3) for _ in range(n_windows - 1)]
    base_w = left_margin + sum(widths) + sum(gaps) + right_margin
    if base_w > _GRID_MAX_W:
        return None

    max_h = max(heights)
    if max_h + 2 > _GRID_MAX_H:
        return None

    grid_h = max_h + _randint(2, _GRID_MAX_H - max_h)
    grid_w = base_w + _randint(0, _GRID_MAX_W - base_w)
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    c0 = left_margin
    for idx, window in enumerate(windows):
        h = int(window["height"])
        w = int(window["width"])
        r0 = _randint(1, grid_h - h - 1)
        patch = window["patch"]
        region = grid[r0 : r0 + h, c0 : c0 + w]
        if np.any(region != bg):
            return None
        mask = patch != bg
        region[mask] = patch[mask]
        window["r0"] = int(r0)
        window["c0"] = int(c0)
        c0 += w
        if idx < len(gaps):
            c0 += gaps[idx]

    if not _bg_is_dominant(grid, bg):
        return None

    expected = _expected_output(windows, bg)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_MAX_ATTEMPTS} attempts")
