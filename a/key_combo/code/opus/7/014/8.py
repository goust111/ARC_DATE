from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, tile_fill


def _repeat_mask(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return tile_fill(shape, tile.astype(int)).astype(bool)


def _min_axis_period(mask: np.ndarray, axis: int) -> int:
    n = int(mask.shape[axis])
    for period in range(1, n + 1):
        if axis == 0:
            if np.array_equal(mask[period:, :], mask[:-period, :]):
                return period
        else:
            if np.array_equal(mask[:, period:], mask[:, :-period]):
                return period
    raise AssertionError("every finite grid has a valid axis period")


def _exact_period(mask: np.ndarray) -> tuple[int, int]:
    return _min_axis_period(mask, 0), _min_axis_period(mask, 1)


def _reconstruct_from_observed(observed: np.ndarray, period: tuple[int, int]) -> np.ndarray:
    pr, pc = map(int, period)
    tile = np.zeros((pr, pc), dtype=bool)
    coords = np.argwhere(observed)
    if coords.size:
        tile[coords[:, 0] % pr, coords[:, 1] % pc] = True
    return _repeat_mask(tile, observed.shape)


def _period_solutions(observed: np.ndarray) -> list[dict]:
    h, w = map(int, observed.shape)
    observed_count = int(observed.sum())
    out: list[dict] = []
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            recon = _reconstruct_from_observed(observed, (pr, pc))
            missing = int(recon.sum()) - observed_count
            if not 3 <= missing <= 6:
                continue
            if not np.any(recon) or np.all(recon):
                continue
            if _exact_period(recon) != (pr, pc):
                continue
            out.append(
                {
                    "period": (pr, pc),
                    "recon": recon,
                    "missing": missing,
                    "score": (pr * pc, pr, pc, missing),
                }
            )
    out.sort(key=lambda item: item["score"])
    return out


def _repair_panel(panel: np.ndarray, fg: int) -> np.ndarray:
    observed = panel == int(fg)
    solutions = _period_solutions(observed)
    assert solutions, "expected at least one valid periodic repair"
    if len(solutions) > 1:
        assert solutions[0]["score"] < solutions[1]["score"], "period repair must be uniquely determined"
    repaired = np.where(solutions[0]["recon"], int(fg), Color.BLACK)
    return repaired.astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the two damaged periodic panels and overlay their masks into one green-on-grey summary.

    Input structure:
    - The grid has black background and exactly one full magenta divider row.
    - The divider splits the grid into equal-sized top and bottom panels.
    - The top panel uses only blue and black; the bottom panel uses only red and black.
    - In each panel, several foreground cells of a periodic binary tile were replaced by black.

    Rule:
    - Find the divider row and split the grid into the two panels.
    - For each panel independently, enumerate periodic completions whose implied number of missing foreground cells is 3..6 and choose the uniquely smallest exact period.
    - Restore every missing cell in that panel by repeating the recovered tile, treating any non-black cell as mask foreground.
    - Remove the divider and output one grid of panel size: paint green where either repaired mask is foreground, otherwise grey.

    Key insight:
    - The periodic restoration is only an intermediate step: once the hidden cells are recovered, the final answer is the position-wise union of the two repaired binary masks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == Color.PINK)]
    assert len(divider_rows) == 1, "expected exactly one full divider row"
    div = int(divider_rows[0])
    assert g.shape[0] % 2 == 1 and div * 2 + 1 == g.shape[0], "divider must split equal panels"

    top = g[:div, :]
    bottom = g[div + 1 :, :]
    assert top.shape == bottom.shape and top.size > 0
    assert set(map(int, np.unique(top))) <= {Color.BLACK, Color.BLUE}
    assert set(map(int, np.unique(bottom))) <= {Color.BLACK, Color.RED}

    repaired_top = _repair_panel(top, Color.BLUE)
    repaired_bottom = _repair_panel(bottom, Color.RED)

    out = np.full(top.shape, Color.GREY, dtype=int)
    union_mask = (repaired_top != Color.BLACK) | (repaired_bottom != Color.BLACK)
    out[union_mask] = Color.GREEN
    return out


def _sample_period(limit: int) -> int:
    choices = [p for p in range(2, limit // 2 + 1)]
    assert choices
    return random.choice(choices)


def _sample_full_mask(shape: tuple[int, int], forbidden: tuple[int, int] | None = None) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = map(int, shape)
    for _ in range(600):
        pr = _sample_period(h)
        pc = _sample_period(w)
        if forbidden is not None and (pr, pc) == forbidden:
            continue

        tile = np.zeros((pr, pc), dtype=bool)
        area = pr * pc
        count = random.randint(max(2, area // 4), max(2, area // 2))
        for idx in random.sample(range(area), count):
            tile[idx // pc, idx % pc] = True
        if not np.any(tile) or np.all(tile):
            continue

        full = _repeat_mask(tile, shape)
        density = float(full.mean())
        if not 0.18 <= density <= 0.55:
            continue
        if _exact_period(full) != (pr, pc):
            continue
        return full, (pr, pc)
    raise RuntimeError("failed to sample a periodic panel")


def _delete_cells(full: np.ndarray, period: tuple[int, int], other_full: np.ndarray) -> np.ndarray | None:
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(full):
        groups.setdefault((int(r) % period[0], int(c) % period[1]), []).append((int(r), int(c)))

    fg_coords = [tuple(map(int, coord)) for coord in np.argwhere(full)]
    unique_contrib = [cell for cell in fg_coords if not bool(other_full[cell])]
    if not unique_contrib:
        return None

    for _ in range(800):
        missing_count = random.randint(3, 6)
        deleted = set()
        first = random.choice(unique_contrib)
        deleted.add(first)

        pool = fg_coords[:]
        random.shuffle(pool)
        visible_counts = {key: len(cells) for key, cells in groups.items()}
        key = (first[0] % period[0], first[1] % period[1])
        visible_counts[key] -= 1

        for cell in pool:
            if len(deleted) == missing_count:
                break
            if cell in deleted:
                continue
            residue = (cell[0] % period[0], cell[1] % period[1])
            if visible_counts[residue] <= 1:
                continue
            deleted.add(cell)
            visible_counts[residue] -= 1
        if len(deleted) != missing_count:
            continue

        observed = full.copy()
        for r, c in deleted:
            observed[r, c] = False

        solutions = _period_solutions(observed)
        if not solutions:
            continue
        if len(solutions) > 1 and not (solutions[0]["score"] < solutions[1]["score"]):
            continue
        if solutions[0]["period"] != period:
            continue
        if not np.array_equal(solutions[0]["recon"], full):
            continue
        if not np.any((full & ~observed) & ~other_full):
            continue
        return observed
    return None


def _build_input(top_obs: np.ndarray, bottom_obs: np.ndarray) -> np.ndarray:
    h, w = map(int, top_obs.shape)
    grid = np.zeros((2 * h + 1, w), dtype=int)
    grid[:h, :] = np.where(top_obs, Color.BLUE, Color.BLACK)
    grid[h, :] = Color.PINK
    grid[h + 1 :, :] = np.where(bottom_obs, Color.RED, Color.BLACK)
    return grid


def _expected_output(top_full: np.ndarray, bottom_full: np.ndarray) -> np.ndarray:
    out = np.full(top_full.shape, Color.GREY, dtype=int)
    out[top_full | bottom_full] = Color.GREEN
    return out


def _semantic_checks(
    top_full: np.ndarray,
    bottom_full: np.ndarray,
    top_obs: np.ndarray,
    bottom_obs: np.ndarray,
) -> bool:
    union_full = top_full | bottom_full
    union_obs = top_obs | bottom_obs
    if np.array_equal(union_full, union_obs):
        return False
    if not np.any(top_full & ~bottom_full):
        return False
    if not np.any(bottom_full & ~top_full):
        return False
    if np.array_equal(union_full, top_full) or np.array_equal(union_full, bottom_full):
        return False
    if not np.any((top_full & ~top_obs) & ~bottom_full):
        return False
    if not np.any((bottom_full & ~bottom_obs) & ~top_full):
        return False
    density = float(union_full.mean())
    return 0.2 <= density <= 0.7


def _full_pair_checks(top_full: np.ndarray, bottom_full: np.ndarray) -> bool:
    union_full = top_full | bottom_full
    if not np.any(top_full & ~bottom_full):
        return False
    if not np.any(bottom_full & ~top_full):
        return False
    if np.array_equal(union_full, top_full) or np.array_equal(union_full, bottom_full):
        return False
    density = float(union_full.mean())
    return 0.2 <= density <= 0.7


def _gen_attempt() -> np.ndarray | None:
    half_h = random.randint(7, 12)
    width = random.randint(8, 13)
    shape = (half_h, width)

    top_full, top_period = _sample_full_mask(shape)
    bottom_full, bottom_period = _sample_full_mask(shape, forbidden=top_period)
    if not _full_pair_checks(top_full, bottom_full):
        return None

    top_obs = _delete_cells(top_full, top_period, bottom_full)
    bottom_obs = _delete_cells(bottom_full, bottom_period, top_full)
    if top_obs is None or bottom_obs is None:
        return None
    if not _semantic_checks(top_full, bottom_full, top_obs, bottom_obs):
        return None

    inp = _build_input(top_obs, bottom_obs)
    expected = _expected_output(top_full, bottom_full)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-overlay ARC input")
