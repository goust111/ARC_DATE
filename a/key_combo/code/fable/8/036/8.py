from __future__ import annotations

import importlib.util
import random
from collections import Counter
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
if _ARC_UTILS_PATH.exists():
    _arc_utils_spec = importlib.util.spec_from_file_location("arc_utils_fable_8_036", _ARC_UTILS_PATH)
    _arc_utils = importlib.util.module_from_spec(_arc_utils_spec)
    assert _arc_utils_spec.loader is not None
    _arc_utils_spec.loader.exec_module(_arc_utils)
    get_bg = _arc_utils.get_bg
    colors_present = _arc_utils.colors_present
else:
    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])

    def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
        colors = set(map(int, grid.flatten()))
        if exclude is not None:
            colors -= set(map(int, exclude))
        return sorted(colors)


PALETTE = list(range(10))
GEN_RETRIES = 400


def _split_periodic_lines(counts: np.ndarray) -> list[int] | None:
    values = sorted({int(v) for v in counts})
    best: tuple[tuple[int, int, int], list[int]] | None = None
    for low, high in zip(values, values[1:]):
        indices = [idx for idx, value in enumerate(counts) if int(value) >= high]
        if len(indices) < 2:
            continue
        diffs = np.diff(indices)
        if len(set(map(int, diffs))) != 1:
            continue
        step = int(diffs[0])
        if step < 2:
            continue
        score = (int(high - low), len(indices), int(high))
        if best is None or score > best[0]:
            best = (score, indices)
    return None if best is None else best[1]


def _is_valid_lattice(
    grid: np.ndarray,
    bg: int,
    line_color: int,
    line_rows: list[int],
    line_cols: list[int],
) -> bool:
    h, w = grid.shape
    row_set = set(map(int, line_rows))
    col_set = set(map(int, line_cols))
    if len(row_set) < 2 or len(col_set) < 2:
        return False
    if len(set(np.diff(line_rows))) != 1 or len(set(np.diff(line_cols))) != 1:
        return False

    line_mask = np.zeros((h, w), dtype=bool)
    line_mask[line_rows, :] = True
    line_mask[:, line_cols] = True

    if np.any((grid == line_color) & (~line_mask)):
        return False

    for r in line_rows:
        for c in range(w):
            value = int(grid[r, c])
            if c in col_set:
                if value not in (bg, line_color):
                    return False
            elif value != line_color:
                return False

    for c in line_cols:
        for r in range(h):
            value = int(grid[r, c])
            if r in row_set:
                if value not in (bg, line_color):
                    return False
            elif value != line_color:
                return False

    return True


def _find_lattice(grid: np.ndarray) -> tuple[int, int, list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    candidates: list[tuple[tuple[int, int], int, list[int], list[int]]] = []
    for color in colors_present(g, exclude={bg}):
        row_counts = np.sum(g == int(color), axis=1)
        col_counts = np.sum(g == int(color), axis=0)
        line_rows = _split_periodic_lines(row_counts)
        line_cols = _split_periodic_lines(col_counts)
        if line_rows is None or line_cols is None:
            continue
        if not _is_valid_lattice(g, bg, int(color), line_rows, line_cols):
            continue
        score = (len(line_rows) * len(line_cols), int(np.sum(g == int(color))))
        candidates.append((score, int(color), line_rows, line_cols))

    assert len(candidates) == 1, "expected exactly one lattice line color"
    _, line_color, line_rows, line_cols = candidates[0]
    return bg, line_color, line_rows, line_cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the missing intersections of a regular lattice as a coarse pixel picture.

    Input structure:
    - One non-background color draws equally spaced full-width horizontal lines and full-height vertical lines, forming a regular lattice.
    - Some lattice intersections are punched out to background, while colored distractor pixels may appear only inside lattice cells.
    - The line color never appears away from the lattice itself, and every non-intersection lattice pixel is intact.

    Rule:
    - Identify the unique lattice line color together with its ordered intersection rows and columns.
    - Build an output grid whose coordinates are the lattice intersections.
    - Mark an output cell with the line color exactly when the corresponding lattice intersection is missing in the input; otherwise use the background color.

    Key insight:
    - The background holes at lattice intersections are the message, while the more salient colored pixels inside the cells are pure noise and must be ignored.
    """
    g = np.asarray(grid, dtype=int)
    bg, line_color, line_rows, line_cols = _find_lattice(g)

    out = np.full((len(line_rows), len(line_cols)), bg, dtype=int)
    for rr, r in enumerate(line_rows):
        for cc, c in enumerate(line_cols):
            value = int(g[r, c])
            assert value in (bg, line_color), "intersection must be either intact or punched out"
            if value == bg:
                out[rr, cc] = line_color
    return out


def _sample_output_mask(rows: int, cols: int) -> np.ndarray:
    modes = ("scatter", "walk", "bars", "two_blobs")
    for _ in range(200):
        mask = np.zeros((rows, cols), dtype=bool)
        mode = random.choice(modes)

        if mode == "scatter":
            count = random.randint(3, max(3, rows * cols - 2))
            picks = random.sample(range(rows * cols), count)
            for idx in picks:
                mask[idx // cols, idx % cols] = True
        elif mode == "walk":
            count = random.randint(3, max(3, rows * cols - 3))
            r = random.randrange(rows)
            c = random.randrange(cols)
            mask[r, c] = True
            while int(mask.sum()) < count:
                dr, dc = random.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
                r = min(rows - 1, max(0, r + dr))
                c = min(cols - 1, max(0, c + dc))
                mask[r, c] = True
        elif mode == "bars":
            row = random.randrange(rows)
            col = random.randrange(cols)
            c0 = random.randrange(cols)
            c1 = random.randrange(c0, cols)
            r0 = random.randrange(rows)
            r1 = random.randrange(r0, rows)
            mask[row, c0 : c1 + 1] = True
            mask[r0 : r1 + 1, col] = True
            extras = random.randint(0, max(0, rows * cols // 5))
            for _ in range(extras):
                mask[random.randrange(rows), random.randrange(cols)] = True
        else:
            seeds = random.sample(range(rows * cols), 2)
            for seed in seeds:
                r = seed // cols
                c = seed % cols
                mask[r, c] = True
                for _ in range(random.randint(1, max(1, rows + cols - 3))):
                    dr, dc = random.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
                    r = min(rows - 1, max(0, r + dr))
                    c = min(cols - 1, max(0, c + dc))
                    mask[r, c] = True

        coords = np.argwhere(mask)
        if coords.shape[0] < 3 or coords.shape[0] >= rows * cols:
            continue
        if len(set(map(int, coords[:, 0]))) < 2 or len(set(map(int, coords[:, 1]))) < 2:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        bbox_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if int(coords.shape[0]) >= bbox_area:
            continue
        if coords.shape[0] == 4:
            corners = {(int(r0), int(c0)), (int(r0), int(c1)), (int(r1), int(c0)), (int(r1), int(c1))}
            if set(map(lambda rc: (int(rc[0]), int(rc[1])), coords)) == corners:
                continue
        return mask
    raise RuntimeError("failed to sample a non-degenerate hole pattern")


def _spec_output(bg: int, line_color: int, hole_mask: np.ndarray) -> np.ndarray:
    out = np.full(hole_mask.shape, bg, dtype=int)
    out[hole_mask] = line_color
    return out


def _place_distractors(
    grid: np.ndarray,
    bg: int,
    line_color: int,
    line_rows: list[int],
    line_cols: list[int],
) -> list[tuple[int, int]]:
    distractor_colors = [c for c in PALETTE if c not in (bg, line_color)]
    positions: list[tuple[int, int]] = []
    cell_count = (len(line_rows) - 1) * (len(line_cols) - 1)
    total = random.randint(1, max(1, min(14, cell_count * 2)))
    used: set[tuple[int, int]] = set()

    for _ in range(total):
        cell_r = random.randrange(len(line_rows) - 1)
        cell_c = random.randrange(len(line_cols) - 1)
        r = random.randint(line_rows[cell_r] + 1, line_rows[cell_r + 1] - 1)
        c = random.randint(line_cols[cell_c] + 1, line_cols[cell_c + 1] - 1)
        if (r, c) in used:
            continue
        used.add((r, c))
        grid[r, c] = random.choice(distractor_colors)
        positions.append((r, c))

    assert positions, "expected at least one distractor pixel"
    return positions


def _build_instance() -> tuple[np.ndarray, dict[str, object]]:
    bg, line_color = random.sample(PALETTE, 2)

    rows = random.randint(3, 7)
    cols = random.randint(3, 7)
    step_r = random.randint(4, 6)
    step_c = random.randint(4, 6)
    height = 1 + (rows - 1) * step_r
    width = 1 + (cols - 1) * step_c
    if height > 30 or width > 30:
        raise RuntimeError("sampled lattice exceeds ARC size limit")

    line_rows = [step_r * idx for idx in range(rows)]
    line_cols = [step_c * idx for idx in range(cols)]
    hole_mask = _sample_output_mask(rows, cols)

    grid = np.full((height, width), bg, dtype=int)
    grid[line_rows, :] = line_color
    grid[:, line_cols] = line_color
    for rr, r in enumerate(line_rows):
        for cc, c in enumerate(line_cols):
            if hole_mask[rr, cc]:
                grid[r, c] = bg

    distractors = _place_distractors(grid, bg, line_color, line_rows, line_cols)
    spec = {
        "bg": bg,
        "line_color": line_color,
        "line_rows": line_rows,
        "line_cols": line_cols,
        "hole_mask": hole_mask,
        "distractors": distractors,
    }
    return grid, spec


def _validate_instance(grid: np.ndarray, spec: dict[str, object]) -> None:
    bg = int(spec["bg"])
    line_color = int(spec["line_color"])
    line_rows = list(spec["line_rows"])
    line_cols = list(spec["line_cols"])
    hole_mask = np.asarray(spec["hole_mask"], dtype=bool)
    distractors = list(spec["distractors"])

    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert len(line_rows) == hole_mask.shape[0]
    assert len(line_cols) == hole_mask.shape[1]
    assert int(hole_mask.sum()) >= 3
    assert not np.any((grid == line_color) & (~((np.isin(np.arange(grid.shape[0])[:, None], line_rows)) | (np.isin(np.arange(grid.shape[1])[None, :], line_cols)))))

    for r in line_rows:
        for c in range(grid.shape[1]):
            if c not in line_cols:
                assert int(grid[r, c]) == line_color
    for c in line_cols:
        for r in range(grid.shape[0]):
            if r not in line_rows:
                assert int(grid[r, c]) == line_color

    for rr, r in enumerate(line_rows):
        for cc, c in enumerate(line_cols):
            expected = bg if hole_mask[rr, cc] else line_color
            assert int(grid[r, c]) == expected

    for r, c in distractors:
        assert r not in line_rows and c not in line_cols
        assert int(grid[r, c]) not in (bg, line_color)

    expected = _spec_output(bg, line_color, hole_mask)
    actual = main(grid)
    assert np.array_equal(actual, expected)

    coords = np.argwhere(hole_mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    bbox_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
    assert int(coords.shape[0]) < bbox_area
    if coords.shape[0] == 4:
        corners = {(int(r0), int(c0)), (int(r0), int(c1)), (int(r1), int(c0)), (int(r1), int(c1))}
        assert set(map(lambda rc: (int(rc[0]), int(rc[1])), coords)) != corners


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid, spec = _build_instance()
            _validate_instance(grid, spec)
            return grid
        except (AssertionError, RuntimeError):
            continue
    raise RuntimeError("failed to generate a valid lattice-hole instance")
