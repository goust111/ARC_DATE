"""Cascade cyan bars from one yellow start through pink controls row by row.

Input structure:
- Dominant background color with one visible yellow start pixel.
- Exactly one color is used for control pixels (pink), with one control per row.
- Each control row may contain counting pixels in the same row: non-background, non-control pixels.
- Exactly one control pixel is directly below the start pixel.

Rule:
- Start from an empty background canvas and keep the start pixel.
- Activate the control pixel directly under the start pixel.
- For each activated control at column c in row r, count non-background, non-control pixels strictly left (L) and right (R).
- Paint row r from column c-L to c+R with cyan.
- If an unlit control is in row r±1 and shares a painted column, activate it and repeat.
- Controls that are never activated must not appear in output.

Key insight:
- The propagation width in each row is encoded by counting pixels and drives the only channel to the next row.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path

import importlib.util
import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers from arc_utils
# ---------------------------------------------------------------------------

def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    if not path.exists():
        raise FileNotFoundError(f"arc_utils not found: {path}")
    spec = importlib.util.spec_from_file_location("arc_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load arc_utils from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_utils = _load_arc_utils()
get_bg = _utils.get_bg
colors_present = _utils.colors_present


# Output color is fixed by the task text to a cyan-like value.
_FILL_COLOR = 8


def _build_control_row_map(grid: np.ndarray, control_color: int) -> dict[int, list[int]]:
    m: dict[int, list[int]] = {}
    coords = np.argwhere(grid == int(control_color))
    for r, c in coords:
        rr, cc = int(r), int(c)
        m.setdefault(rr, []).append(cc)
    for rr, cols in m.items():
        if len(cols) != 1:
            raise AssertionError(
                f"control color appears {len(cols)} times in row {rr}, expected one per row"
            )
        m[rr] = [int(cols[0])]
    return m


def _locate_start_and_controls(
    grid: np.ndarray,
) -> tuple[int, tuple[int, int], int, int, dict[int, list[int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    colors = colors_present(g, exclude={bg})
    if len(colors) < 2:
        raise AssertionError("expected at least two non-background colors")

    # Candidate control colors should be one pixel per row (task assumption).
    control_candidates: list[int] = []
    for color in colors:
        rows = np.argwhere(g == int(color))[:, 0]
        if rows.size == 0:
            continue
        counts = Counter(map(int, rows))
        if all(v == 1 for v in counts.values()):
            control_candidates.append(int(color))

    start_candidates: list[tuple[int, int, int, int]] = []
    for r in range(H - 1):
        for c in range(W):
            color = int(g[r, c])
            if color == bg:
                continue
            below = int(g[r + 1, c])
            if below in control_candidates:
                start_candidates.append((int(r), int(c), color, below))

    if len(start_candidates) != 1:
        raise AssertionError(
            f"expected exactly one start-to-control adjacency pair, got {len(start_candidates)}"
        )

    start_r, start_c, start_color, control_color = start_candidates[0]
    control_map = _build_control_row_map(g, control_color)
    control_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == int(control_color))]

    if (start_r + 1, start_c) not in control_coords:
        raise AssertionError("start pixel must be directly above a control pixel")

    return bg, (start_r, start_c), int(start_color), control_color, control_map, control_coords


def _paint_from_map(
    grid: np.ndarray,
    start_pos: tuple[int, int],
    start_color: int,
    control_color: int,
    fill_color: int,
    control_map: dict[int, list[int]],
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    out = np.full((H, W), bg, dtype=int)
    out[start_pos] = int(start_color)

    queue: deque[tuple[int, int]] = deque([(start_pos[0] + 1, start_pos[1])])
    activated: set[tuple[int, int]] = set()
    painted_cols_by_row: dict[int, set[int]] = defaultdict(set)

    while queue:
        r, c = queue.popleft()
        if (r, c) in activated:
            continue
        if c not in control_map.get(r, ()):
            continue

        row = g[r]
        left_count = int(np.count_nonzero((row[:c] != bg) & (row[:c] != control_color)))
        right_count = int(np.count_nonzero((row[c + 1 :] != bg) & (row[c + 1 :] != control_color)))
        left = max(0, c - left_count)
        right = min(W - 1, c + right_count)

        painted = painted_cols_by_row[r]
        for cc in range(left, right + 1):
            painted.add(int(cc))
            out[r, cc] = int(fill_color)

        activated.add((r, c))

        # Propagate only by same column in adjacent rows.
        for nr in (r - 1, r + 1):
            if 0 <= nr < H:
                for nc in control_map.get(nr, ()):
                    if nc in painted and (nr, nc) not in activated:
                        queue.append((nr, int(nc)))

    return out


def _painted_cols_for_controls(
    grid: np.ndarray,
    start_pos: tuple[int, int],
    control_color: int,
    control_map: dict[int, list[int]],
) -> dict[int, set[int]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    queue: deque[tuple[int, int]] = deque([(start_pos[0] + 1, start_pos[1])])
    activated: set[tuple[int, int]] = set()
    painted_cols_by_row: dict[int, set[int]] = defaultdict(set)

    while queue:
        r, c = queue.popleft()
        if (r, c) in activated:
            continue
        if c not in control_map.get(r, ()):
            continue

        row = g[r]
        left_count = int(np.count_nonzero((row[:c] != bg) & (row[:c] != control_color)))
        right_count = int(np.count_nonzero((row[c + 1 :] != bg) & (row[c + 1 :] != control_color)))
        left = max(0, c - left_count)
        right = min(W - 1, c + right_count)

        painted = painted_cols_by_row[r]
        for cc in range(left, right + 1):
            painted.add(int(cc))

        activated.add((r, c))

        for nr in (r - 1, r + 1):
            if 0 <= nr < H:
                for nc in control_map.get(nr, ()):
                    if nc in painted and (nr, nc) not in activated:
                        queue.append((nr, int(nc)))

    return painted_cols_by_row


def main(grid: np.ndarray) -> np.ndarray:
    """Cascade cyan bars from the start marker through row-neighbor controls."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, start_pos, start_color, control_color, control_map, _ = _locate_start_and_controls(g)
    if start_color == bg or control_color == bg or start_color == control_color:
        raise AssertionError("start and control colors must be distinct from background")

    fill_color = _FILL_COLOR
    if fill_color in {start_color, control_color, bg}:
        for c in range(11):
            if c not in {start_color, control_color, bg}:
                fill_color = int(c)
                break

    return _paint_from_map(g, start_pos, start_color, control_color, fill_color, control_map)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

_BG_COLOR = 0
_CONTROL_PALETTE = [1, 2, 3, 4, 5, 7, 9]
_RNG = np.random.default_rng()


def _rand_count(rng: np.random.Generator, needed: int, max_count: int) -> int:
    if max_count <= 0:
        return 0
    if needed > max_count:
        raise AssertionError("count requirement exceeds row boundary")

    if needed > 0:
        extra_cap = max_count - needed
        extra = int(rng.integers(0, min(3, int(extra_cap) + 1)))
        return int(needed + extra)

    if rng.random() < 0.45:
        return 0
    return int(rng.integers(1, min(4, max_count) + 1))


def _place_control_row(
    rng: np.random.Generator,
    grid: np.ndarray,
    row: int,
    col: int,
    control_color: int,
    counter_color: int,
    needed_left: int,
    needed_right: int,
) -> None:
    H, W = grid.shape
    if not (0 <= row < H):
        raise AssertionError("row out of bounds")
    if not (0 <= col < W):
        raise AssertionError("col out of bounds")

    left = _rand_count(rng, needed_left, col)
    right = _rand_count(rng, needed_right, W - 1 - col)

    left_choices = np.arange(0, col, dtype=int)
    right_choices = np.arange(col + 1, W, dtype=int)

    if left > 0:
        for c in rng.choice(left_choices, size=left, replace=False):
            grid[row, int(c)] = int(counter_color)
    if right > 0:
        for c in rng.choice(right_choices, size=right, replace=False):
            grid[row, int(c)] = int(counter_color)

    grid[row, col] = int(control_color)


def _build_chain_rows(
    rng: np.random.Generator,
    start_row: int,
    start_col: int,
    height: int,
    width: int,
) -> tuple[list[int], list[int]]:
    max_len = int(rng.integers(1, min(7, height - 2)))
    chain_rows = [start_row + 1]
    chain_cols = [start_col]
    used_rows = {start_row + 1}

    for _ in range(max_len - 1):
        r0 = chain_rows[-1]
        options = []
        if r0 > 0:
            options.append(-1)
        if r0 + 1 < height:
            options.append(1)

        if not options:
            break

        placed = False
        for _ in range(12):
            r1 = int(r0 + int(rng.choice(options)))
            if r1 in used_rows:
                continue
            drift = int(rng.integers(-2, 3))
            c1 = int(np.clip(chain_cols[-1] + drift, 1, width - 2))

            chain_rows.append(r1)
            chain_cols.append(c1)
            used_rows.add(r1)
            placed = True
            break

        if not placed:
            break

    return chain_rows, chain_cols


def _sample_colors(rng: np.random.Generator) -> tuple[int, int, list[int]]:
    start_color, control_color = rng.choice(_CONTROL_PALETTE, size=2, replace=False)
    start_color = int(start_color)
    control_color = int(control_color)
    if start_color == _FILL_COLOR or control_color == _FILL_COLOR:
        raise AssertionError("control palette must exclude fill color")

    counter_colors = [c for c in _CONTROL_PALETTE if c not in {start_color, control_color}]
    if not counter_colors:
        raise AssertionError("not enough counter colors")
    return start_color, control_color, counter_colors


def _generate_once() -> np.ndarray:
    H = int(_RNG.integers(12, 30))
    W = int(_RNG.integers(12, 30))

    start_color, control_color, counter_colors = _sample_colors(_RNG)
    start_row = int(_RNG.integers(1, H - 2))
    start_col = int(_RNG.integers(1, W - 1))

    grid = np.full((H, W), _BG_COLOR, dtype=int)
    grid[start_row, start_col] = int(start_color)

    chain_rows, chain_cols = _build_chain_rows(_RNG, start_row, start_col, H, W)
    if not chain_rows:
        raise RuntimeError("failed to build control chain")

    chain_controls: list[tuple[int, int]] = []
    chain_map: dict[int, list[int]] = {}
    for idx, (r, c) in enumerate(zip(chain_rows, chain_cols)):
        if idx + 1 < len(chain_rows):
            nxt = chain_cols[idx + 1]
            need_left = max(0, c - nxt)
            need_right = max(0, nxt - c)
        else:
            need_left = 0
            need_right = 0

        _place_control_row(
            _RNG,
            grid,
            r,
            c,
            control_color,
            int(_RNG.choice(counter_colors)),
            need_left,
            need_right,
        )
        chain_controls.append((r, c))
        chain_map[r] = [c]

    # Precompute lit columns from chain-only controls for near-miss construction.
    chain_lit_cols = _painted_cols_for_controls(grid, (start_row, start_col), control_color, chain_map)

    # Near rows: adjacent to chain rows but intentionally not reached.
    near_rows = [
        r
        for r in range(H)
        if r not in chain_map and any(abs(r - cr) == 1 for cr in chain_rows)
    ]
    _RNG.shuffle(near_rows)

    controls_all: list[tuple[int, int]] = list(chain_controls)
    control_map_all: dict[int, list[int]] = dict(chain_map)

    for _ in range(int(_RNG.integers(0, 3))):
        if not near_rows:
            break
        r = int(near_rows.pop(0))
        forbidden = set()
        for nr in (r - 1, r + 1):
            if 0 <= nr < H:
                forbidden.update(chain_lit_cols.get(nr, set()))

        free = [c for c in range(1, W - 1) if c not in forbidden]
        if not free:
            continue

        c = int(_RNG.choice(free))
        _place_control_row(
            _RNG,
            grid,
            r,
            c,
            control_color,
            int(_RNG.choice(counter_colors)),
            0,
            0,
        )
        controls_all.append((r, c))

    # Far rows: definitely unreachable.
    far_rows = [
        r
        for r in range(H)
        if r not in control_map_all and all(abs(r - cr) > 1 for cr in chain_rows)
    ]
    _RNG.shuffle(far_rows)
    for _ in range(int(_RNG.integers(0, 3))):
        if not far_rows:
            break
        r = int(far_rows.pop(0))
        c = int(_RNG.integers(1, W - 1))
        _place_control_row(
            _RNG,
            grid,
            r,
            c,
            control_color,
            int(_RNG.choice(counter_colors)),
            0,
            0,
        )
        controls_all.append((r, c))

    # Validate against solver assumptions and semantics.
    try:
        out = main(grid)
    except Exception as exc:
        raise RuntimeError("generated input does not satisfy solver assumptions") from exc

    if not np.any(out == _FILL_COLOR):
        raise RuntimeError("no fill color painted")

    for r, c in chain_controls:
        if out[r, c] != _FILL_COLOR:
            raise RuntimeError("chain control not activated")

    for r, c in controls_all[len(chain_controls) :]:
        if out[r, c] == _FILL_COLOR:
            raise RuntimeError("distractor control activated")

    if not np.all((out == _BG_COLOR) | (out == _FILL_COLOR) | (out == start_color)):
        raise RuntimeError("unexpected color in output")

    return grid


def generate_input() -> np.ndarray:
    """Generate random ARC-style inputs for the chain-like lighting task."""
    for _ in range(400):
        try:
            return _generate_once()
        except RuntimeError:
            continue
    raise RuntimeError("failed to generate a valid chain-like lighting input")
