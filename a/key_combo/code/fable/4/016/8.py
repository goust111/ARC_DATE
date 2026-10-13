from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import numpy as np
import sys

try:
    from arc_utils import get_bg
except ModuleNotFoundError:
    _UTIL_ROOT: Path | None = None
    _HERE = Path(__file__).resolve().parent
    for parent in _HERE.parents:
        candidate = parent / "new" / "solvers"
        if candidate.is_dir():
            _UTIL_ROOT = candidate
            break
    if _UTIL_ROOT is None:
        raise
    sys.path.append(str(_UTIL_ROOT))
    from arc_utils import get_bg  # type: ignore


Cardinal = tuple[int, int]
DIRS: tuple[Cardinal, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_H_RANGE = (12, 28)
GEN_W_RANGE = (12, 28)
GEN_ATTEMPTS = 800
PLAN_ATTEMPTS = 80
EXTRA_ATTEMPTS = 100
GEN_PALETTE = tuple(range(10))
EXTRA_BLOCK_COUNT = (0, 3)
EXTRA_BLOCK_TRIES = 60


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _trace_ray(
    shape: tuple[int, int],
    source: tuple[int, int],
    direction: Cardinal,
    obstacle_mask: np.ndarray,
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    h, w = map(int, shape)
    dr, dc = map(int, direction)
    r0, c0 = map(int, source)
    r = r0 + dr
    c = c0 + dc
    if not _in_bounds((h, w), r, c):
        return None

    path: list[tuple[int, int]] = []
    while _in_bounds((h, w), r, c):
        if obstacle_mask[r, c]:
            return None

        path.append((r, c))
        nr = r + dr
        nc = c + dc

        if not _in_bounds((h, w), nr, nc):
            return path, (r, c)
        if obstacle_mask[nr, nc]:
            return path, (r, c)

        r, c = nr, nc

    # Unreachable.
    return None


def _source_signature(
    grid: np.ndarray,
    source: tuple[int, int],
    obstacle_mask: np.ndarray,
    splash_set: set[tuple[int, int]],
) -> tuple[dict[Cardinal, list[tuple[int, int]]], dict[Cardinal, tuple[int, int]]] | None:
    paths: dict[Cardinal, list[tuple[int, int]]] = {}
    endpoints: dict[Cardinal, tuple[int, int]] = {}
    h, w = map(int, grid.shape)

    for d in DIRS:
        res = _trace_ray((h, w), source, d, obstacle_mask)
        if res is None:
            return None
        path, end = res
        if end not in splash_set:
            return None
        if any(cell != end for cell in path[:-1] if cell in splash_set):
            return None
        paths[d] = path
        endpoints[d] = end

    if len(set(endpoints.values())) != 4:
        return None

    return paths, endpoints


def main(grid: np.ndarray) -> np.ndarray:
    """Reconstruct hidden sources by tracing horizontal/vertical residue rays.

    Input structure:
    - Dominant background color.
    - One obstacle color (all non-background colors except the red splashes), usually gray.
    - Exactly eight red splash pixels, each being a ray endpoint.
    - Two hidden source cells generate four axis-aligned rays each; a ray stops at the grid boundary or the cell just before an obstacle.

    Rule:
    - Identify the red splash color from count constraints.
    - Infer each source by checking all background cells and requiring that its 4 cardinal rays end on splash pixels with no intermediate splash pixels.
    - Select the unique pair of sources whose endpoints are exactly the eight splashes, four per source.
    - Draw all four rays from each source (including the source cell) in the splash color; keep obstacles and background unchanged elsewhere.

    Key insight:
    - Each source is the only point whose four directional first blocking/edge hits give exactly four designated endpoints, so pairing splash endpoints by this strict ray constraint removes fake row/column intersections.
    """

    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    counts = Counter(map(int, g.ravel()))
    splash_candidates = [int(c) for c in counts if int(c) != bg and counts[c] == 8]
    assert len(splash_candidates) == 1, "expected one splash color with exactly 8 pixels"
    splash_color = splash_candidates[0]

    splashes = np.argwhere(g == splash_color)
    splash_set = {tuple(map(int, rc)) for rc in splashes}
    assert len(splash_set) == 8, "expected exactly eight splash pixels"

    obstacle_mask = (g != bg) & (g != splash_color)
    candidates: list[tuple[tuple[int, int], dict[Cardinal, list[tuple[int, int]]], set[tuple[int, int]]]] = []

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if g[r, c] != bg:
                continue
            sig = _source_signature(g, (r, c), obstacle_mask, splash_set)
            if sig is None:
                continue
            paths, endpoints = sig
            candidates.append(((r, c), paths, set(endpoints.values())))

    assert candidates, "expected at least one valid source candidate"

    selected: tuple[tuple[int, int], dict[Cardinal, list[tuple[int, int]]], set[tuple[int, int]]] | None = None
    for i in range(len(candidates)):
        _, _, eps1 = candidates[i]
        for j in range(i + 1, len(candidates)):
            _, _, eps2 = candidates[j]
            if eps1.isdisjoint(eps2) and eps1 | eps2 == splash_set:
                selected = (candidates[i], candidates[j])
                break
        if selected is not None:
            break

    assert selected is not None, "failed to find two sources covering all splashes"

    out = g.copy()
    for src_data, paths, _eps in selected:
        out[src_data] = splash_color
        for path in paths.values():
            for r, c in path:
                out[r, c] = splash_color

    assert np.any(out != g), "expected to draw visible source rays"
    return out


def _build_random_palette() -> tuple[int, int, int]:
    bg = random.choice(list(GEN_PALETTE))
    splash = random.choice([c for c in GEN_PALETTE if c != bg])
    obstacle = random.choice([c for c in GEN_PALETTE if c != bg and c != splash])
    return bg, splash, obstacle


def _path_for_generation(
    source: tuple[int, int],
    direction: Cardinal,
    length: int,
) -> list[tuple[int, int]]:
    dr, dc = direction
    r, c = source[0] + dr, source[1] + dc
    out: list[tuple[int, int]] = []
    for _ in range(length):
        out.append((r, c))
        r += dr
        c += dc
    return out


def _plan_direction(
    source: tuple[int, int],
    direction: Cardinal,
    obstacle_mask: np.ndarray,
    used_endpoints: set[tuple[int, int]],
    used_obstacle_cells: set[tuple[int, int]],
    source_paths: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int] | None] | None:
    h, w = map(int, obstacle_mask.shape)
    dr, dc = map(int, direction)
    sr, sc = map(int, source)

    max_len = 0
    if dr == -1:
        max_len = sr
    elif dr == 1:
        max_len = h - 1 - sr
    elif dc == -1:
        max_len = sc
    else:
        max_len = w - 1 - sc

    if max_len <= 0:
        return None

    for _ in range(PLAN_ATTEMPTS):
        if max_len == 1:
            use_obstacle = False
            length = 1
        else:
            use_obstacle = random.random() < 0.65
            if use_obstacle:
                length = random.randint(1, max_len - 1)
            else:
                length = random.randint(1, max_len)

        end_r = sr + dr * length
        end_c = sc + dc * length
        endpoint = (end_r, end_c)

        obstacle_cell = None
        if use_obstacle:
            obstacle_cell = (sr + dr * (length + 1), sc + dc * (length + 1))

        if endpoint in used_endpoints:
            continue
        if obstacle_cell is not None:
            if obstacle_cell in used_obstacle_cells:
                continue
            if obstacle_cell[0] < 0 or obstacle_cell[0] >= h or obstacle_cell[1] < 0 or obstacle_cell[1] >= w:
                continue

        path = _path_for_generation(source, direction, length)

        blocked = False
        for cell in path:
            if obstacle_mask[cell]:
                blocked = True
                break
            if cell in used_obstacle_cells:
                blocked = True
                break
            if cell in source_paths:
                blocked = True
                break
        if blocked:
            continue

        if obstacle_cell is not None:
            if obstacle_mask[obstacle_cell] or obstacle_cell in used_endpoints:
                continue

        return path, endpoint, obstacle_cell

    return None


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_H_RANGE)
    w = random.randint(*GEN_W_RANGE)
    bg, splash, obstacle = _build_random_palette()

    source_cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    if len(source_cells) < 2:
        return None

    for _ in range(PLAN_ATTEMPTS):
        src1, src2 = random.sample(source_cells, 2)
        sources = (src1, src2)

        obstacle_mask = np.zeros((h, w), dtype=bool)
        used_obstacle_cells: set[tuple[int, int]] = set()
        used_endpoints: set[tuple[int, int]] = set()
        source_paths: set[tuple[int, int]] = set(sources)
        specs: list[tuple[tuple[int, int], dict[Cardinal, list[tuple[int, int]]], dict[Cardinal, tuple[int, int]]]] = []

        for src in sources:
            path_by_dir: dict[Cardinal, list[tuple[int, int]]] = {}
            end_by_dir: dict[Cardinal, tuple[int, int]] = {}

            for d in DIRS:
                plan = _plan_direction(
                    source=src,
                    direction=d,
                    obstacle_mask=obstacle_mask,
                    used_endpoints=used_endpoints,
                    used_obstacle_cells=used_obstacle_cells,
                    source_paths=source_paths,
                )
                if plan is None:
                    break
                path, endpoint, obs_cell = plan
                if endpoint in used_endpoints:
                    break

                path_by_dir[d] = path
                end_by_dir[d] = endpoint

                for cell in path:
                    source_paths.add(cell)

                if obs_cell is not None:
                    obstacle_mask[obs_cell] = True
                    used_obstacle_cells.add(obs_cell)

            if len(path_by_dir) != 4:
                break

            specs.append((src, path_by_dir, end_by_dir))

            for ep in end_by_dir.values():
                used_endpoints.add(ep)

        if len(specs) != 2:
            continue
        if len(used_endpoints) != 8:
            continue

        for src, path_by_dir, end_by_dir in specs:
            for d in DIRS:
                rec = _trace_ray((h, w), src, d, obstacle_mask)
                if rec is None:
                    break
                rec_path, rec_end = rec
                if rec_end != end_by_dir[d] or rec_path != path_by_dir[d]:
                    break
            else:
                continue
            break
        else:
            planned_obstacle_cells = set(used_obstacle_cells)

            # Optional additional obstacle blocks disconnected from ray cells.
            reserved: set[tuple[int, int]] = set(planned_obstacle_cells) | set(used_endpoints) | source_paths
            extra_count = random.randint(*EXTRA_BLOCK_COUNT)

            for _ in range(extra_count):
                for _ in range(EXTRA_ATTEMPTS):
                    bh = random.randint(1, 3)
                    bw = random.randint(1, 3)
                    if bh > h or bw > w:
                        continue

                    r0 = random.randint(0, h - bh)
                    c0 = random.randint(0, w - bw)
                    cells = {
                        (r, c)
                        for r in range(r0, r0 + bh)
                        for c in range(c0, c0 + bw)
                    }
                    if cells & reserved:
                        continue
                    planned_obstacle_cells.update(cells)
                    reserved.update(cells)
                    break

            input_grid = np.full((h, w), bg, dtype=int)
            if planned_obstacle_cells:
                ors = [int(r) for r, _ in planned_obstacle_cells]
                ocs = [int(c) for _, c in planned_obstacle_cells]
                input_grid[ors, ocs] = obstacle

            for _, _, end_by_dir in specs:
                for ep in end_by_dir.values():
                    input_grid[ep] = splash

            if np.sum(input_grid == splash) != 8:
                continue
            if np.sum(input_grid == obstacle) == 8:
                continue

            # Ensure the generated input is well formed and matches the target solver.
            solved = main(input_grid)

            expected = input_grid.copy()
            for src, path_by_dir, _ in specs:
                expected[src] = splash
                for path in path_by_dir.values():
                    for r, c in path:
                        expected[r, c] = splash

            if not np.array_equal(solved, expected):
                continue

            return input_grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 8.py input")
