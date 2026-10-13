"""Close wall fragments by repeatedly adding a single blocking wall segment that creates a new enclosure.

Input structure:
- Black background with one or more cyan wall fragments.
- Wall endpoints are wall pixels with exactly one 4-neighbor wall pixel.
- Some pairs of endpoints are collinear (horizontal/vertical/diagonal), and the path
  between them is background.

Rule:
- Repeatedly choose the unique endpoint pair (if any) whose connecting wall segment
  makes a previously outside-connected background component become fully enclosed.
- Draw that segment in cyan, then recolor exactly the newly enclosed background cells to
  yellow.
- Continue until no such unique qualifying pair remains.

Key insight:
- Pair collinearity and clear background path are necessary but not sufficient;
  the decisive condition is a topological change of background connectivity.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


try:  # pragma: no cover
    from arc_utils import Color, components_mask
except ModuleNotFoundError:  # pragma: no cover
    _file = Path(__file__).resolve()
    for _parent in (_file, *_file.parents):
        _utils = _parent / "new" / "solvers" / "arc_utils.py"
        if _utils.exists():
            _utils_dir = str(_utils.parent)
            if _utils_dir not in sys.path:
                sys.path.append(_utils_dir)
            break
    from arc_utils import Color, components_mask


BG_COLOR = Color.BLACK
WALL_COLOR = Color.TEAL
FILL_COLOR = Color.YELLOW
MAX_STEPS = 12
GEN_ATTEMPTS = 260
MULTISTEP_PROB = 0.48


def _line_points(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """Return all integer points on the axis-aligned or 45°-diagonal segment."""
    r0, c0 = map(int, start)
    r1, c1 = map(int, end)
    dr = r1 - r0
    dc = c1 - c0
    if dr == 0 and dc == 0:
        return []
    if dr == 0:
        step = 1 if dc > 0 else -1
        return [(r0, c0 + i * step) for i in range(abs(dc) + 1)]
    if dc == 0:
        step = 1 if dr > 0 else -1
        return [(r0 + i * step, c0) for i in range(abs(dr) + 1)]
    if abs(dr) == abs(dc):
        rs = 1 if dr > 0 else -1
        cs = 1 if dc > 0 else -1
        return [(r0 + i * rs, c0 + i * cs) for i in range(abs(dr) + 1)]
    return []


def _wall_endpoints(grid: np.ndarray, wall: int) -> list[tuple[int, int]]:
    """Return wall pixels that currently have exactly one 4-neighbor wall pixel."""
    h, w = grid.shape
    wall_mask = grid == wall
    endpoints: list[tuple[int, int]] = []
    for r, c in np.argwhere(wall_mask):
        r = int(r)
        c = int(c)
        deg = 0
        if r > 0 and wall_mask[r - 1, c]:
            deg += 1
        if r + 1 < h and wall_mask[r + 1, c]:
            deg += 1
        if c > 0 and wall_mask[r, c - 1]:
            deg += 1
        if c + 1 < w and wall_mask[r, c + 1]:
            deg += 1
        if deg == 1:
            endpoints.append((r, c))
    return endpoints


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    if coords.size == 0:
        return False
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _enclosed_background_cells(grid: np.ndarray, background: int) -> np.ndarray:
    """Background cells that are NOT connected to border through background."""
    bg_mask = (grid == background)
    enclosed = np.zeros_like(bg_mask, dtype=bool)
    for comp in components_mask(bg_mask, connectivity=4):
        if comp.size and not _touches_border(comp, grid.shape):
            enclosed[comp[:, 0], comp[:, 1]] = True
    return enclosed


def _find_candidates(
    grid: np.ndarray, background: int, wall: int
) -> list[dict[str, object]]:
    """Return all qualifying segments that create a new enclosed area."""
    current_enclosed = _enclosed_background_cells(grid, background)
    eps = _wall_endpoints(grid, wall)
    candidates: list[dict[str, object]] = []

    if len(eps) < 2:
        return candidates

    for i in range(len(eps)):
        r0, c0 = eps[i]
        for j in range(i + 1, len(eps)):
            r1, c1 = eps[j]
            pts = _line_points((r0, c0), (r1, c1))
            if not pts:
                continue
            # Candidate walls must be drawable on a background-only channel.
            if any(grid[int(r), int(c)] != background for r, c in pts[1:-1]):
                continue

            trial = grid.copy()
            for r, c in pts:
                trial[int(r), int(c)] = wall

            after_enclosed = _enclosed_background_cells(trial, background)
            newly = after_enclosed & ~current_enclosed
            if np.any(newly):
                candidates.append({"line": pts, "fill": newly})
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """Close wall chains and recolor every newly enclosed cavity in yellow."""
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("main expects a 2D grid")

    background = int(np.bincount(g.ravel()).argmax())
    color_counts = {
        int(v): int(np.sum(g == int(v)))
        for v in np.unique(g)
        if int(v) != background and int(v) != FILL_COLOR
    }
    wall = int(max(color_counts, key=color_counts.get)) if color_counts else WALL_COLOR

    out = g.copy()
    for _ in range(MAX_STEPS):
        actions = _find_candidates(out, background, wall)
        if len(actions) != 1:
            break
        action = actions[0]
        for r, c in action["line"]:
            out[int(r), int(c)] = wall
        out[action["fill"]] = FILL_COLOR
    return out


def _random_grid(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(16, 29))
    w = int(rng.integers(16, 29))
    grid = np.full((h, w), BG_COLOR, dtype=int)
    return grid


def _draw_open_box(grid: np.ndarray, rng: np.random.Generator) -> None:
    h, w = grid.shape
    if h < 10 or w < 10:
        return
    max_h = min(10, h - 3)
    max_w = min(11, w - 3)
    box_h = int(rng.integers(4, max_h + 1))
    box_w = int(rng.integers(4, max_w + 1))
    r0 = int(rng.integers(1, h - box_h - 1))
    c0 = int(rng.integers(1, w - box_w - 1))
    r1 = r0 + box_h - 1
    c1 = c0 + box_w - 1
    miss = int(rng.integers(0, 4))

    if miss != 0:
        grid[r0, c0 : c1 + 1] = WALL_COLOR
    if miss != 1:
        grid[r1, c0 : c1 + 1] = WALL_COLOR
    if miss != 2:
        grid[r0 : r1 + 1, c0] = WALL_COLOR
    if miss != 3:
        grid[r0 : r1 + 1, c1] = WALL_COLOR


def _draw_fragment(grid: np.ndarray, rng: np.random.Generator) -> None:
    h, w = grid.shape
    kind = int(rng.integers(0, 3))
    if kind == 0 and h > 4:
        r = int(rng.integers(1, h - 1))
        length = int(rng.integers(3, min(9, w - 2)))
        c0 = int(rng.integers(1, w - length))
        if rng.random() < 0.9:
            grid[r, c0 : c0 + length] = WALL_COLOR
        else:
            for c in range(c0, c0 + length):
                if rng.random() < 0.8:
                    grid[r, c] = WALL_COLOR
    elif kind == 1 and w > 4:
        c = int(rng.integers(1, w - 1))
        length = int(rng.integers(3, min(9, h - 2)))
        r0 = int(rng.integers(1, h - length))
        if rng.random() < 0.9:
            grid[r0 : r0 + length, c] = WALL_COLOR
        else:
            for r in range(r0, r0 + length):
                if rng.random() < 0.8:
                    grid[r, c] = WALL_COLOR
    else:
        if h <= 4 or w <= 4:
            return
        length = int(rng.integers(3, min(9, min(h - 2, w - 2))))
        r0 = int(rng.integers(1, h - length))
        c0 = int(rng.integers(1, w - length))
        for k in range(length):
            if rng.random() < 0.9:
                grid[r0 + k, c0 + k] = WALL_COLOR


def _random_scene(rng: np.random.Generator) -> np.ndarray:
    grid = _random_grid(rng)
    n_boxes = int(rng.integers(1, 5))
    for _ in range(n_boxes):
        _draw_open_box(grid, rng)

    n_fragments = int(rng.integers(4, 13))
    for _ in range(n_fragments):
        _draw_fragment(grid, rng)

    # optional geometry randomization
    if rng.random() < 0.55:
        grid = np.fliplr(grid)
    if rng.random() < 0.55:
        grid = np.flipud(grid)
    if rng.random() < 0.45:
        grid = grid.T
    return grid


def _wall_density(grid: np.ndarray) -> float:
    return float(np.mean(grid == WALL_COLOR))


def _solve_trace_len(grid: np.ndarray) -> int:
    current = grid.copy()
    background = int(np.bincount(current.ravel()).argmax())
    step = 0
    for _ in range(MAX_STEPS):
        acts = _find_candidates(current, background, WALL_COLOR)
        if len(acts) != 1:
            return step
        action = acts[0]
        for r, c in action["line"]:
            current[int(r), int(c)] = WALL_COLOR
        current[action["fill"]] = FILL_COLOR
        step += 1
    return step


def _enhance_for_chain(seed: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(26):
        candidate = seed.copy()
        add_count = int(rng.integers(1, 4))
        for _ in range(add_count):
            if rng.random() < 0.7:
                _draw_open_box(candidate, rng)
            else:
                _draw_fragment(candidate, rng)
        if _solve_trace_len(candidate) >= 2:
            return candidate
    return None


def generate_input() -> np.ndarray:
    """Create a valid ARC input with a deterministic chain-enclosure wall progression."""
    rng = np.random.default_rng()
    want_multi = rng.random() < MULTISTEP_PROB

    for _ in range(GEN_ATTEMPTS):
        grid = _random_scene(rng)
        if _wall_density(grid) < 0.05 or _wall_density(grid) > 0.40:
            continue

        if want_multi:
            steps = _solve_trace_len(grid)
            if steps >= 2:
                return grid
            if steps == 1:
                # try to create a latent second candidate without breaking determinism
                boosted = _enhance_for_chain(grid, rng)
                if boosted is not None:
                    return boosted
            continue

        steps = _solve_trace_len(grid)
        if steps >= 1:
            return grid

    # fallback: keep searching for any valid single-step case
    for _ in range(GEN_ATTEMPTS):
        grid = _random_scene(rng)
        if _wall_density(grid) < 0.05 or _wall_density(grid) > 0.40:
            continue
        if _solve_trace_len(grid) >= 1:
            return grid

    raise RuntimeError("failed to generate a valid wall-chain puzzle input")
