from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _UTILS_DIR.exists() and str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

try:
    from arc_utils import Color, bbox_coords
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2

    def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)


GRID_SIZE = 10
VISIBLE_HEIGHTS = (4, 5, 6)
PERIODS = (2, 3)
SHIFTS = (1, 2)
MAX_INTERVAL_WIDTH = 4
GEN_ATTEMPTS = 400
COMPONENT_HEIGHT_MIN = 4

_COMPONENT_LIBRARY: dict[tuple[int, int], list[dict[str, object]]] | None = None


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _shift_right(row: np.ndarray, shift: int) -> np.ndarray:
    out = np.zeros_like(row)
    if shift < row.size:
        out[shift:] = row[:-shift]
    return out


def _infer_period_and_shift(grid: np.ndarray) -> tuple[int, int]:
    h, _w = grid.shape
    candidates = []
    for period in PERIODS:
        if period >= h:
            continue
        for shift in SHIFTS:
            if all(np.array_equal(grid[r], _shift_right(grid[r - period], shift)) for r in range(period, h)):
                candidates.append((period, shift))
    assert candidates, "expected at least one valid (period, shift)"
    min_period = min(period for period, _shift in candidates)
    minimal = [pair for pair in candidates if pair[0] == min_period]
    assert len(minimal) == 1, "expected a unique minimal (period, shift)"
    return minimal[0]


def _complete_grid(grid: np.ndarray, period: int, shift: int) -> np.ndarray:
    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    out[: grid.shape[0], : grid.shape[1]] = grid
    for r in range(grid.shape[0], GRID_SIZE):
        out[r] = _shift_right(out[r - period], shift)
    return out


def _recolor_components(full_grid: np.ndarray) -> np.ndarray:
    out = full_grid.copy()
    for comp in _components_mask(full_grid == Color.BLUE):
        r0, _c0, r1, _c1 = bbox_coords(comp)
        height = r1 - r0 + 1
        assert height % 2 == 0, "component bbox height must be even"
        split_row = r0 + height // 2
        for r, c in comp:
            if int(r) >= split_row:
                out[int(r), int(c)] = Color.RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a row-shifted periodic pattern and recolor the lower half of each resulting component.

    Input structure:
    - The input is a 10-column grid with 4 to 6 visible rows on a black background.
    - Blue foreground pixels follow a row-periodic rule: row k is row k-p shifted right by s.
    - The hidden rows below the input continue the same periodic right-shift process.

    Rule:
    - Infer the unique minimal period p and shift s that explain the visible rows.
    - Extend the grid downward to a full 10x10 canvas by repeatedly shifting row k-p right by s.
    - Find every blue 4-connected component in the completed grid.
    - For each component, compute its own bounding box and recolor the foreground pixels in the lower half of that box red, leaving the upper half blue.

    Key insight:
    - The completion step determines which pixels connect into each component, and those completed component bboxes then determine the recoloring split line independently for each region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] == GRID_SIZE
    period, shift = _infer_period_and_shift(g)
    completed = _complete_grid(g, period, shift)
    return _recolor_components(completed)


def _build_component_mask(intervals: tuple[tuple[int, int], ...], period: int, shift: int) -> np.ndarray:
    mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    for r in range(GRID_SIZE):
        left, right = intervals[r % period]
        row_shift = (r // period) * shift
        left = max(0, left + row_shift)
        right = min(GRID_SIZE, right + row_shift)
        if left < right:
            mask[r, left:right] = True
    return mask


def _component_meta(mask: np.ndarray) -> dict[str, object] | None:
    comps = _components_mask(mask)
    if len(comps) != 1:
        return None
    comp = comps[0]
    r0, c0, r1, c1 = bbox_coords(comp)
    height = r1 - r0 + 1
    if height < COMPONENT_HEIGHT_MIN or height % 2 != 0:
        return None
    split_row = r0 + height // 2
    rows = comp[:, 0]
    if not np.any(rows < split_row) or not np.any(rows >= split_row):
        return None
    return {
        "mask": mask,
        "height": int(height),
        "split_row": int(split_row),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "area": int(mask.sum()),
    }


def _interval_choices() -> list[tuple[int, int]]:
    choices = []
    for width in range(1, MAX_INTERVAL_WIDTH + 1):
        for left in range(0, GRID_SIZE - width + 1):
            choices.append((left, left + width))
    return choices


def _init_component_library() -> dict[tuple[int, int], list[dict[str, object]]]:
    interval_choices = _interval_choices()
    library: dict[tuple[int, int], list[dict[str, object]]] = {}
    for period in PERIODS:
        for shift in SHIFTS:
            unique: dict[bytes, dict[str, object]] = {}
            if period == 2:
                for interval0 in interval_choices:
                    for interval1 in interval_choices:
                        mask = _build_component_mask((interval0, interval1), period, shift)
                        meta = _component_meta(mask)
                        if meta is not None:
                            unique[mask.tobytes()] = meta
            else:
                for interval0 in interval_choices:
                    for interval1 in interval_choices:
                        for interval2 in interval_choices:
                            mask = _build_component_mask((interval0, interval1, interval2), period, shift)
                            meta = _component_meta(mask)
                            if meta is not None:
                                unique[mask.tobytes()] = meta
            library[(period, shift)] = list(unique.values())
    return library


def _get_component_library() -> dict[tuple[int, int], list[dict[str, object]]]:
    global _COMPONENT_LIBRARY
    if _COMPONENT_LIBRARY is None:
        _COMPONENT_LIBRARY = _init_component_library()
    return _COMPONENT_LIBRARY


def _touches(mask: np.ndarray, occupied: np.ndarray) -> bool:
    if not occupied.any():
        return False
    expanded = occupied.copy()
    expanded[1:, :] |= occupied[:-1, :]
    expanded[:-1, :] |= occupied[1:, :]
    expanded[:, 1:] |= occupied[:, :-1]
    expanded[:, :-1] |= occupied[:, 1:]
    return bool(np.any(mask & expanded))


def _pick_component(
    rng: np.random.Generator,
    pool: list[dict[str, object]],
    occupied: np.ndarray,
    *,
    required_height: set[int] | None = None,
    forbidden_height: set[int] | None = None,
    sample_size: int = 200,
) -> dict[str, object] | None:
    if not pool:
        return None
    count = min(len(pool), sample_size)
    indices = rng.choice(len(pool), size=count, replace=False)
    for idx in indices:
        comp = pool[int(idx)]
        height = int(comp["height"])
        if required_height is not None and height not in required_height:
            continue
        if forbidden_height is not None and height in forbidden_height:
            continue
        if _touches(comp["mask"], occupied):
            continue
        return comp
    return None


def _global_split_recolor(mask: np.ndarray) -> np.ndarray:
    out = np.where(mask, Color.BLUE, Color.BLACK).astype(int)
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    height = r1 - r0 + 1
    split_row = r0 + height // 2
    rows = np.arange(GRID_SIZE)[:, None]
    out[mask & (rows >= split_row)] = Color.RED
    return out


def _instance_is_valid(mask: np.ndarray, visible_height: int, period: int, shift: int) -> bool:
    comps = _components_mask(mask)
    if len(comps) < 2:
        return False
    split_rows = set()
    for comp in comps:
        r0, _c0, r1, _c1 = bbox_coords(comp)
        height = r1 - r0 + 1
        if height < COMPONENT_HEIGHT_MIN or height % 2 != 0:
            return False
        split_row = r0 + height // 2
        rows = comp[:, 0]
        if not np.any(rows < split_row) or not np.any(rows >= split_row):
            return False
        split_rows.add(int(split_row))
    if len(split_rows) < 2:
        return False
    if not np.any(mask[visible_height:, :]):
        return False
    visible = mask[:visible_height].astype(int)
    try:
        inferred = _infer_period_and_shift(visible)
    except AssertionError:
        return False
    if inferred != (period, shift):
        return False
    expected = _recolor_components(np.where(mask, Color.BLUE, Color.BLACK).astype(int))
    if np.array_equal(expected, _global_split_recolor(mask)):
        return False
    if not np.any(expected == Color.BLUE) or not np.any(expected == Color.RED):
        return False
    return np.array_equal(main(visible), expected)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    library = _get_component_library()

    for _ in range(GEN_ATTEMPTS):
        period = int(rng.choice(PERIODS))
        shift = int(rng.choice(SHIFTS))
        visible_height = int(rng.choice(VISIBLE_HEIGHTS))
        pool = library[(period, shift)]
        short_pool = [comp for comp in pool if int(comp["height"]) < GRID_SIZE]
        long_pool = [comp for comp in pool if int(comp["height"]) > visible_height]
        if not short_pool or not long_pool:
            continue

        occupied = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        selected = []

        first = _pick_component(rng, short_pool, occupied)
        if first is None:
            continue
        occupied |= first["mask"]
        selected.append(first)

        second = _pick_component(
            rng,
            long_pool,
            occupied,
            forbidden_height={int(first["height"])},
        )
        if second is None:
            continue
        occupied |= second["mask"]
        selected.append(second)

        if rng.random() < 0.45:
            third = _pick_component(rng, pool, occupied)
            if third is not None:
                occupied |= third["mask"]
                selected.append(third)

        if not _instance_is_valid(occupied, visible_height, period, shift):
            continue
        return occupied[:visible_height].astype(int)

    raise RuntimeError("failed to generate a valid periodic-half-recolor input")
