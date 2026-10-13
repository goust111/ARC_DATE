from __future__ import annotations

import hashlib
import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

GRID_SIDE_RANGE = (13, 19)
LINE_GAP_MIN = 4
WALL_GAP_MIN = 3
NOISE_COUNT_RANGE = (3, 6)
MAX_GENERATION_ATTEMPTS = 800

_GEN_CALL_COUNT = 0
_USED_INPUT_SIGS: set[str] = set()
_USED_OUTPUT_SIGS: set[str] = set()


def _signature(grid: np.ndarray) -> str:
    arr = np.asarray(grid, dtype=int)
    return hashlib.md5(arr.tobytes() + str(arr.shape).encode()).hexdigest()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _coords_of_color(grid: np.ndarray, color: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(grid == int(color))]


def _component_coords(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(int(r), int(c))])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                nbrs = _neighbors4(rr, cc) if connectivity == 4 else _neighbors8(rr, cc)
                for nr, nc in nbrs:
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _guide_rows(grid: np.ndarray) -> list[int]:
    rows: list[int] = []
    for r in range(grid.shape[0]):
        vals = set(map(int, grid[r]))
        if vals and vals <= {RED, GREEN}:
            rows.append(int(r))
    return rows


def _apply_rays(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h, _w = out.shape
    for _r, c in _coords_of_color(out, GREEN):
        for rr in range(h):
            if int(out[rr, c]) == BLACK:
                out[rr, c] = BLUE
    return out


def _remove_singleton_grey(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    grey_components = _component_coords(out == GREY, connectivity=8)
    for comp in grey_components:
        if len(comp) == 1:
            r, c = map(int, comp[0])
            out[r, c] = BLACK
    return out


def _exterior_black_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def maybe_push(r: int, c: int) -> None:
        if 0 <= r < h and 0 <= c < w and int(g[r, c]) == BLACK and not bool(seen[r, c]):
            seen[r, c] = True
            q.append((r, c))

    for c in range(w):
        maybe_push(0, c)
        maybe_push(h - 1, c)
    for r in range(h):
        maybe_push(r, 0)
        maybe_push(r, w - 1)

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == BLACK and not bool(seen[nr, nc]):
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _fill_enclosed_black(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    exterior = _exterior_black_mask(out)
    out[(out == BLACK) & ~exterior] = YELLOW
    return out


def _enclosed_components(grid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    exterior = _exterior_black_mask(grid)
    enclosed = (np.asarray(grid, dtype=int) == BLACK) & ~exterior
    return enclosed, _component_coords(enclosed, connectivity=4), exterior


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit vertical wall rays from green markers, erase isolated grey noise, then fill the remaining enclosed background regions.

    Input structure:
    - The grid has black background, 2-3 full-width horizontal red guide lines, and 1-2 green marker pixels on each guide line.
    - Grey pixels are sparse singleton noise points, and some may lie on the marker columns between guide lines.
    - Blue and yellow are absent from the input.

    Rule:
    - For every green marker, draw a blue vertical ray through its column from top edge to bottom edge, painting only black cells and skipping over any non-black cells without stopping.
    - Group grey pixels by 8-connectivity and erase every size-1 grey component back to black, leaving larger grey components unchanged.
    - Treat every non-black pixel that remains as a wall, mark all black cells reachable from the border by 4-neighbor flood fill, and recolor every other black cell yellow.

    Key insight:
    - A grey singleton on a ray column can temporarily plug a one-cell gap in a vertical wall, so the final filled regions depend on both building the rays and then erasing the noise.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    assert not np.any(g == BLUE), "blue should be absent from the input"
    assert not np.any(g == YELLOW), "yellow should be absent from the input"

    guide_rows = _guide_rows(g)
    assert 2 <= len(guide_rows) <= 3, "expected 2-3 horizontal guide lines"
    for r in guide_rows:
        green_count = int(np.count_nonzero(g[r] == GREEN))
        assert 1 <= green_count <= 2, "each guide line must contain 1-2 markers"
        assert np.all(np.isin(g[r], [RED, GREEN])), "guide line rows may contain only red and green"

    marker_coords = _coords_of_color(g, GREEN)
    assert marker_coords, "expected at least one green marker"
    assert all(r in guide_rows for r, _c in marker_coords), "markers must lie on guide lines"

    ray_grid = _apply_rays(g)
    cleaned = _remove_singleton_grey(ray_grid)
    return _fill_enclosed_black(cleaned)


def _pick_spaced_positions(lo: int, hi: int, count: int, min_gap: int) -> list[int] | None:
    choices = list(range(int(lo), int(hi) + 1))
    if len(choices) < count:
        return None
    for _ in range(200):
        pos = sorted(random.sample(choices, count))
        if all(b - a >= min_gap for a, b in zip(pos, pos[1:])):
            return pos
    return None


def _assign_markers(guide_rows: list[int], wall_cols: list[int]) -> dict[int, list[int]]:
    markers = {r: [] for r in guide_rows}
    shuffled_rows = guide_rows[:]
    shuffled_cols = wall_cols[:]
    random.shuffle(shuffled_rows)
    random.shuffle(shuffled_cols)

    for row, col in zip(shuffled_rows, shuffled_cols[: len(guide_rows)]):
        markers[row].append(int(col))

    for col in shuffled_cols[len(guide_rows) :]:
        eligible = [r for r in guide_rows if len(markers[r]) < 2 and col not in markers[r]]
        if not eligible:
            return {}
        markers[random.choice(eligible)].append(int(col))

    total_capacity = 2 * len(guide_rows)
    total_markers = sum(len(v) for v in markers.values())
    extras = random.randint(0, max(0, total_capacity - total_markers))
    for _ in range(extras):
        eligible_rows = [r for r in guide_rows if len(markers[r]) < 2]
        if not eligible_rows:
            break
        row = random.choice(eligible_rows)
        options = [c for c in wall_cols if c not in markers[row]]
        if not options:
            continue
        markers[row].append(int(random.choice(options)))

    for row in guide_rows:
        if not (1 <= len(markers[row]) <= 2):
            return {}
        markers[row].sort()
    if sorted({c for cols in markers.values() for c in cols}) != sorted(wall_cols):
        return {}
    return markers


def _place_noise(
    grid: np.ndarray,
    guide_rows: list[int],
    wall_cols: list[int],
    branch: int,
) -> tuple[np.ndarray, list[tuple[int, int]], str] | tuple[None, None, None]:
    out = np.asarray(grid, dtype=int).copy()
    side = "left" if branch in (0, 2) else "right"
    key_col = wall_cols[0] if side == "left" else wall_cols[-1]
    band_indices = list(range(len(guide_rows) - 1))
    if not band_indices:
        return None, None, None

    if len(band_indices) == 1:
        key_band_count = 1
    else:
        key_band_count = 2 if branch in (2, 3) else 1
    key_band_count = min(key_band_count, len(band_indices))
    chosen_bands = sorted(random.sample(band_indices, key_band_count))

    noise_cells: list[tuple[int, int]] = []
    for band_idx in chosen_bands:
        top = guide_rows[band_idx]
        bottom = guide_rows[band_idx + 1]
        gap_row = random.randint(top + 1, bottom - 1)
        out[gap_row, key_col] = GREY
        noise_cells.append((gap_row, key_col))

    target_noise = random.randint(NOISE_COUNT_RANGE[0], NOISE_COUNT_RANGE[1])
    extras_needed = target_noise - len(noise_cells)
    left_wall = wall_cols[0]
    right_wall = wall_cols[-1]
    candidates: list[tuple[int, int]] = []
    for r in range(out.shape[0]):
        if r in guide_rows:
            continue
        for c in range(out.shape[1]):
            if c in wall_cols:
                continue
            if not (c < left_wall or c > right_wall or r < guide_rows[0] or r > guide_rows[-1]):
                continue
            if int(out[r, c]) != BLACK:
                continue
            if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in noise_cells):
                continue
            candidates.append((int(r), int(c)))
    random.shuffle(candidates)
    for r, c in candidates:
        if extras_needed <= 0:
            break
        if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in noise_cells):
            continue
        out[r, c] = GREY
        noise_cells.append((r, c))
        extras_needed -= 1
    if extras_needed > 0:
        return None, None, None
    return out, noise_cells, side


def _input_structure_ok(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if not np.all(np.isin(g, [BLACK, RED, GREEN, GREY])):
        return False
    guide_rows = _guide_rows(g)
    if not (2 <= len(guide_rows) <= 3):
        return False
    for r in guide_rows:
        green_count = int(np.count_nonzero(g[r] == GREEN))
        if not (1 <= green_count <= 2):
            return False
    green_coords = _coords_of_color(g, GREEN)
    if not green_coords or any(r not in guide_rows for r, _c in green_coords):
        return False
    grey_components = _component_coords(g == GREY, connectivity=8)
    grey_count = int(np.count_nonzero(g == GREY))
    if not (NOISE_COUNT_RANGE[0] <= grey_count <= NOISE_COUNT_RANGE[1]):
        return False
    if any(len(comp) != 1 for comp in grey_components):
        return False
    return True


def _semantic_stats(grid: np.ndarray) -> dict[str, object]:
    ray_grid = _apply_rays(grid)
    before_mask, before_components, _ = _enclosed_components(ray_grid)
    cleaned = _remove_singleton_grey(ray_grid)
    after_mask, after_components, after_exterior = _enclosed_components(cleaned)
    final = _fill_enclosed_black(cleaned)
    no_erase = _fill_enclosed_black(ray_grid)
    no_rays = _fill_enclosed_black(_remove_singleton_grey(grid))

    opened_components = 0
    opened_area = 0
    for comp in before_components:
        coords = tuple(comp.T)
        if bool(after_exterior[coords].any()):
            opened_components += 1
            opened_area += int(after_exterior[coords].sum())

    return {
        "ray_grid": ray_grid,
        "cleaned": cleaned,
        "before_count": len(before_components),
        "after_count": len(after_components),
        "before_area": int(np.count_nonzero(before_mask)),
        "after_area": int(np.count_nonzero(after_mask)),
        "opened_components": int(opened_components),
        "opened_area": int(opened_area),
        "final": final,
        "no_erase": no_erase,
        "no_rays": no_rays,
    }


def _build_attempt(branch: int) -> np.ndarray | None:
    if branch in (0, 1):
        line_count = 2
    else:
        line_count = 3
    wall_count = 3 if branch in (0, 2) else 4

    for _ in range(250):
        h = random.randint(*GRID_SIDE_RANGE)
        w = random.randint(*GRID_SIDE_RANGE)
        guide_rows = _pick_spaced_positions(2, h - 3, line_count, LINE_GAP_MIN)
        wall_cols = _pick_spaced_positions(2, w - 3, wall_count, WALL_GAP_MIN)
        if guide_rows is None or wall_cols is None:
            continue

        markers = _assign_markers(guide_rows, wall_cols)
        if not markers:
            continue

        grid = np.zeros((h, w), dtype=int)
        for r in guide_rows:
            grid[r, :] = RED
        for r, cols in markers.items():
            for c in cols:
                grid[r, c] = GREEN

        grid, noise_cells, side = _place_noise(grid, guide_rows, wall_cols, branch)
        if grid is None:
            continue
        assert noise_cells is not None
        assert side is not None

        if not _input_structure_ok(grid):
            continue

        stats = _semantic_stats(grid)
        final = stats["final"]
        predicted = main(grid)
        if not np.array_equal(predicted, final):
            continue
        if stats["before_count"] < 2:
            continue
        if stats["after_count"] < 1:
            continue
        if stats["opened_components"] < 1:
            continue
        if stats["opened_area"] < 3:
            continue
        if np.array_equal(final, stats["no_erase"]):
            continue
        if np.array_equal(final, stats["no_rays"]):
            continue
        if not np.any(final == YELLOW):
            continue
        if np.count_nonzero(stats["ray_grid"] == BLUE) < h:
            continue

        input_sig = _signature(grid)
        output_sig = _signature(final)
        if input_sig in _USED_INPUT_SIGS or output_sig in _USED_OUTPUT_SIGS:
            continue
        _USED_INPUT_SIGS.add(input_sig)
        _USED_OUTPUT_SIGS.add(output_sig)
        return grid
    return None


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    branch = _GEN_CALL_COUNT % 4
    _GEN_CALL_COUNT += 1

    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_attempt(branch)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-wall noise-filter flood-fill input")
