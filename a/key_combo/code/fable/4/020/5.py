from __future__ import annotations

from pathlib import Path
import importlib.util
import numpy as np

from typing import Iterable
from collections.abc import Sequence

try:
    from arc_utils import Color, get_bg
except Exception:
    _ROOT = Path(__file__).resolve()
    _spec = None
    _module = None
    for _parent in (_ROOT, *_ROOT.parents):
        _candidate = _parent / "new" / "solvers" / "arc_utils.py"
        if _candidate.exists():
            _spec = importlib.util.spec_from_file_location("arc_utils", _candidate)
            if _spec is not None and _spec.loader is not None:
                _module = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_module)  # type: ignore[union-attr]
                break
    if _module is None:
        raise
    Color = _module.Color
    get_bg = _module.get_bg


BACKGROUND = Color.BLACK
PATH_COLOR = Color.GREY
START_COLOR = Color.GREEN

SEP_COLOR_CHOICES = [Color.RED, Color.BLUE, Color.YELLOW, Color.ORANGE, Color.PINK, Color.TEAL, Color.MAROON]
COLOR_POOL = [Color.BLUE, Color.RED, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]

GEN_TOP_ROWS = (8, 24)
GEN_WIDTH = (10, 28)
GEN_BEAD_COUNT = (2, 3)
GEN_PATH_LEN_MIN = 9
GEN_PATH_ATTEMPTS = 400
MAX_GENERATE_TRIES = 250


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r, c + 1
    yield r + 1, c
    yield r, c - 1


def _find_separator_row(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h - 1):
        row = grid[r]
        sep = int(row[0])
        if sep == bg:
            continue
        if not np.all(row == sep):
            continue
        if np.any(grid[r + 1] != bg):
            candidates.append((r, sep))
    assert candidates, "expected a full separator row"
    assert len(candidates) == 1, "expected a single separator row"
    return candidates[0]


def _read_palette_row(grid: np.ndarray, sep_row: int, sep_color: int, bg: int) -> list[int]:
    row = grid[sep_row + 1]
    palette: list[int] = []
    for c, value in enumerate(row):
        color = int(value)
        if color == bg:
            break
        assert color != sep_color, "palette should not use separator color"
        assert color not in palette, "palette colors must be unique"
        palette.append(color)
    assert palette, "expected at least one palette swatch"
    return palette


def _collect_path_component(grid: np.ndarray, start: tuple[int, int], bg: int) -> set[tuple[int, int]]:
    h, w = grid.shape
    stack = [start]
    component: set[tuple[int, int]] = set()
    while stack:
        r, c = stack.pop()
        if (r, c) in component:
            continue
        component.add((r, c))
        for nr, nc in _neighbors4(r, c):
            if _in_bounds((h, w), nr, nc) and (nr, nc) not in component and grid[nr, nc] != bg:
                stack.append((nr, nc))
    return component


def _trace_path(component: set[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = [start]
    cur = start
    prev: tuple[int, int] | None = None
    seen: set[tuple[int, int]] = {start}

    while True:
        next_steps = [(nr, nc) for nr, nc in _neighbors4(*cur) if (nr, nc) in component and (nr, nc) != prev]
        if not next_steps:
            break
        assert len(next_steps) == 1, "path must be unbranched"
        nxt = next_steps[0]
        assert nxt not in seen, "path must not loop"
        path.append(nxt)
        prev, cur = cur, nxt
        seen.add(nxt)
    assert len(seen) == len(component), "path component must be a single chain"
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor beads in order while tracing an open non-branching gray path from the green endpoint.

    Input structure:
    - The grid has a black background, a full separator row near the bottom, and a palette row directly beneath it.
    - Above the separator row is one open 4-connected path made of gray cells with one green endpoint and several colored beads.
    - The bead count equals the number of palette colors in the row below the separator.

    Rule:
    - Determine the separator row, remove it and the row(s) below from the output.
    - Starting from the green endpoint, walk the path in its unique order to the other endpoint, and recolor every bead encountered (non-green, non-gray pixel on the path) by palette order from left to right.
    - Keep gray path pixels, green marker, and background unchanged.

    Key insight:
    - The recoloring source is path traversal order, not geometric order; a snake-like path breaks left/right spatial priors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    sep_row, sep_color = _find_separator_row(g, bg)
    palette = _read_palette_row(g, sep_row, sep_color, bg)

    main_area = g[:sep_row]
    if main_area.size == 0:
        return main_area.copy()

    greens = np.argwhere(main_area == START_COLOR)
    assert greens.shape[0] == 1, "expected exactly one green start pixel"
    start = tuple(map(int, greens[0]))
    component = _collect_path_component(main_area, start, bg)
    assert component, "expected a path component"

    path = _trace_path(component, start)
    out = main_area.copy()

    palette_i = 0
    for r, c in path[1:]:
        pix = int(out[r, c])
        if pix in {bg, PATH_COLOR, START_COLOR}:
            continue
        assert palette_i < len(palette), "more beads than palette entries"
        out[r, c] = palette[palette_i]
        palette_i += 1

    assert palette_i == len(palette), "bead count and palette size must match"
    return out


def _path_is_chain(path: Sequence[tuple[int, int]], h: int, w: int) -> bool:
    path_set = set(path)
    if len(path) != len(path_set):
        return False
    for i, (r, c) in enumerate(path):
        deg = 0
        for nr, nc in _neighbors4(r, c):
            if _in_bounds((h, w), nr, nc) and (nr, nc) in path_set:
                deg += 1
        if i == 0 or i == len(path) - 1:
            if deg != 1:
                return False
        elif deg != 2:
            return False
    return True


def _has_turn(path: Sequence[tuple[int, int]]) -> bool:
    if len(path) < 3:
        return False
    prev_dr = path[1][0] - path[0][0]
    prev_dc = path[1][1] - path[0][1]
    for i in range(2, len(path)):
        dr = path[i][0] - path[i - 1][0]
        dc = path[i][1] - path[i - 1][1]
        if (dr, dc) != (prev_dr, prev_dc):
            return True
        prev_dr, prev_dc = dr, dc
    return False


def _touches_other_path(candidate: tuple[int, int], used: set[tuple[int, int]], anchor: tuple[int, int]) -> bool:
    cr, cc = candidate
    for nr, nc in _neighbors4(cr, cc):
        if (nr, nc) in used and (nr, nc) != anchor:
            return True
    return False


def _build_chain(h: int, w: int, length: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    for _ in range(GEN_PATH_ATTEMPTS):
        start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        path = [start]
        used = {start}

        while len(path) < length:
            r, c = path[-1]
            options: list[tuple[int, int]] = []
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds((h, w), nr, nc):
                    continue
                if (nr, nc) in used:
                    continue
                if _touches_other_path((nr, nc), used, (r, c)):
                    continue
                options.append((nr, nc))

            if not options:
                break

            rng.shuffle(options)
            nxt = options[0]
            path.append(nxt)
            used.add(nxt)

        if len(path) != length:
            continue
        if not _path_is_chain(path, h, w):
            continue
        if len(path) >= 5 and rng.random() < 0.85 and not _has_turn(path):
            continue
        return path
    return None


def _expected_from_path(main_area: np.ndarray, path: Sequence[tuple[int, int]], palette: Sequence[int]) -> np.ndarray:
    out = main_area.copy()
    palette_i = 0
    for r, c in path[1:]:
        if out[r, c] in (bg := BACKGROUND, PATH_COLOR, START_COLOR):
            continue
        out[r, c] = palette[palette_i]
        palette_i += 1
    assert palette_i == len(palette)
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_TRIES):
        rng = np.random.default_rng()
        top_h = int(rng.integers(GEN_TOP_ROWS[0], GEN_TOP_ROWS[1]))
        width = int(rng.integers(GEN_WIDTH[0], GEN_WIDTH[1]))
        total_h = top_h + 2
        if total_h > 30 or width > 30:
            continue

        sep_color = int(rng.choice(SEP_COLOR_CHOICES))
        bead_count = int(rng.integers(GEN_BEAD_COUNT[0], GEN_BEAD_COUNT[1] + 1))
        max_len = min(top_h * width // 2, 24)
        if max_len < bead_count + 3:
            continue
        path_len = int(rng.integers(bead_count + 3, max_len + 1))

        path = _build_chain(top_h, width, path_len, rng)
        if path is None:
            continue

        palette_pool = [c for c in COLOR_POOL if c not in {BACKGROUND, PATH_COLOR, START_COLOR, sep_color}]
        if len(palette_pool) < bead_count:
            continue
        palette = [int(c) for c in rng.choice(palette_pool, size=bead_count, replace=False)]
        bead_pool = [c for c in palette_pool if c not in palette]
        bead_colors = [int(c) for c in rng.choice(
            bead_pool,
            size=bead_count,
            replace=len(bead_pool) < bead_count,
        )]

        top = np.full((top_h, width), BACKGROUND, dtype=int)
        for r, c in path:
            top[r, c] = PATH_COLOR
        top[path[0]] = START_COLOR

        bead_idx = sorted(rng.choice(range(1, len(path) - 1), size=bead_count, replace=False))
        for j, idx in enumerate(bead_idx):
            top[path[idx]] = bead_colors[j]

        full = np.full((total_h, width), BACKGROUND, dtype=int)
        full[:top_h] = top
        full[top_h, :] = sep_color
        full[top_h + 1, :bead_count] = palette

        expected = _expected_from_path(full[:top_h], path, palette)
        if np.array_equal(main(full), expected):
            return full

    raise RuntimeError("failed to generate a valid 5 input")
