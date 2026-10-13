from __future__ import annotations

from dataclasses import dataclass
import random
import sys
from pathlib import Path

import numpy as np


_THIS_FILE = Path(__file__).resolve()
for _parent in (_THIS_FILE.parent, *_THIS_FILE.parents):
    _candidate = _parent / "new" / "solvers"
    if (_candidate / "arc_utils.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
else:
    raise ImportError("Could not locate new/solvers/arc_utils.py")

from arc_utils import Color, bbox_coords, components_mask, paste


GEN_ATTEMPTS = 500
GRID_SIZES = (10, 11, 12)
PERIOD_CHOICES = (3, 4)
CYCLE_PALETTE = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK)
SMALL_RECT_DIMS = ((2, 3), (3, 2), (3, 3))
LARGE_RECT_DIMS = ((3, 4), (4, 3), (4, 4))
BLANK_PATCHES = (
    np.array([[1]], dtype=int),
    np.array([[1, 1]], dtype=int),
    np.array([[1], [1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [1, 1]], dtype=int),
)
BLANK_PATCH_COUNT = (2, 4)
BLANK_AREA_RANGE = (4, 11)
PLACEMENT_ATTEMPTS = 300


@dataclass(frozen=True)
class RectSpec:
    top: int
    left: int
    height: int
    width: int

    @property
    def area(self) -> int:
        return int(self.height * self.width)

    @property
    def bottom(self) -> int:
        return int(self.top + self.height - 1)

    @property
    def right(self) -> int:
        return int(self.left + self.width - 1)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return int(self.top), int(self.left), int(self.bottom), int(self.right)


@dataclass(frozen=True)
class CaseSpec:
    input_grid: np.ndarray
    output_grid: np.ndarray
    period: int
    cycle: tuple[int, ...]
    blank_mask: np.ndarray
    small_rect: RectSpec
    large_rect: RectSpec


def _phase_index(row: int, col: int, period: int) -> int:
    if row <= col:
        return int(col % period)
    return int((col + 1) % period)


def _pattern_from_cycle(shape: tuple[int, int], cycle: tuple[int, ...]) -> np.ndarray:
    period = len(cycle)
    cycle_array = np.asarray(cycle, dtype=int)
    rows, cols = np.indices(shape)
    phase = np.where(rows <= cols, cols % period, (cols + 1) % period)
    return cycle_array[phase]


def _rect_mask(shape: tuple[int, int], rect: RectSpec) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[rect.top : rect.top + rect.height, rect.left : rect.left + rect.width] = True
    return mask


def _boxes_overlap(a: RectSpec, b: RectSpec, *, padding: int = 0) -> bool:
    return not (
        a.bottom + padding < b.top
        or b.bottom + padding < a.top
        or a.right + padding < b.left
        or b.right + padding < a.left
    )


def _grey_rectangles(grid: np.ndarray) -> list[RectSpec]:
    rects: list[RectSpec] = []
    for coords in components_mask(grid == Color.GREY, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        area = int(coords.shape[0])
        assert area == height * width, "grey occluders must be solid rectangles"
        rects.append(RectSpec(int(r0), int(c0), height, width))
    rects.sort(key=lambda rect: rect.area)
    return rects


def _candidate_cycles(grid: np.ndarray) -> list[tuple[int, tuple[int, ...]]]:
    visible = (grid != Color.GREY) & (grid != Color.MAROON)
    assert np.any(visible), "expected visible periodic pixels"

    candidates: list[tuple[int, tuple[int, ...]]] = []
    for period in PERIOD_CHOICES:
        cycle: list[int | None] = [None] * period
        ok = True
        for row, col in np.argwhere(visible):
            idx = _phase_index(int(row), int(col), period)
            value = int(grid[row, col])
            if cycle[idx] is None:
                cycle[idx] = value
            elif cycle[idx] != value:
                ok = False
                break
        if not ok or any(value is None for value in cycle):
            continue
        inferred = tuple(int(value) for value in cycle if value is not None)
        if len(set(inferred)) != period:
            continue
        full_pattern = _pattern_from_cycle(grid.shape, inferred)
        if np.array_equal(grid[visible], full_pattern[visible]):
            candidates.append((period, inferred))
    return candidates


def _infer_cycle(grid: np.ndarray) -> tuple[int, tuple[int, ...]]:
    candidates = _candidate_cycles(grid)
    assert len(candidates) == 1, "expected a unique cycle and period"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the smaller grey occluder and refill all maroon gaps from the table's diagonal-shifted column cycle.

    Input structure:
    - The grid is a square color table whose visible non-grey, non-maroon cells follow a periodic column cycle.
    - Cells on or above the main diagonal use phase `col mod n`, while cells below the diagonal use phase `(col + 1) mod n`, with `n` equal to 3 or 4.
    - Two grey solid rectangles occlude the table, and additional maroon cells mark original missing pixels.

    Rule:
    - Find the two grey solid rectangles and rank them by area.
    - Keep the larger rectangle unchanged, but treat every pixel of the smaller rectangle as a new maroon gap.
    - Infer the unique cycle length and ordered color sequence from all visible non-grey, non-maroon pixels.
    - Fill every maroon cell, including both the original gaps and the erased smaller rectangle, with the color dictated by the inferred periodic table.

    Key insight:
    - The task combines object ranking with global pattern completion: area comparison decides which occluder survives, and only then can the remaining gaps be reconstructed from the diagonal phase-shifted cycle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] == g.shape[1], "expected a square table"

    rects = _grey_rectangles(g)
    assert len(rects) == 2, "expected exactly two grey rectangles"
    assert rects[0].area != rects[1].area, "grey rectangles must have distinct areas"

    work = g.copy()
    small_rect = rects[0]
    work[small_rect.top : small_rect.top + small_rect.height, small_rect.left : small_rect.left + small_rect.width] = Color.MAROON

    _period, cycle = _infer_cycle(work)
    full_pattern = _pattern_from_cycle(work.shape, cycle)

    out = work.copy()
    fill_mask = out == Color.MAROON
    out[fill_mask] = full_pattern[fill_mask]
    return out


def _sample_rect(size: int, dims: tuple[int, int]) -> RectSpec:
    height, width = dims
    return RectSpec(
        top=random.randint(0, size - height),
        left=random.randint(0, size - width),
        height=height,
        width=width,
    )


def _sample_rect_pair(size: int) -> tuple[RectSpec, RectSpec] | None:
    small_dims = random.choice(SMALL_RECT_DIMS)
    large_dims = random.choice(LARGE_RECT_DIMS)
    for _ in range(PLACEMENT_ATTEMPTS):
        small_rect = _sample_rect(size, small_dims)
        large_rect = _sample_rect(size, large_dims)
        if _boxes_overlap(small_rect, large_rect, padding=1):
            continue
        return small_rect, large_rect
    return None


def _can_place_patch(mask: np.ndarray, patch: np.ndarray, top: int, left: int, forbidden: np.ndarray) -> bool:
    height, width = patch.shape
    region = mask[top : top + height, left : left + width]
    blocked = forbidden[top : top + height, left : left + width]
    active = patch.astype(bool)
    return not np.any(region & active) and not np.any(blocked & active)


def _sample_blank_mask(size: int, forbidden: np.ndarray) -> np.ndarray | None:
    mask = np.zeros((size, size), dtype=bool)
    patch_count = random.randint(*BLANK_PATCH_COUNT)
    for _ in range(patch_count):
        patch = random.choice(BLANK_PATCHES)
        height, width = patch.shape
        placed = False
        positions = [
            (top, left)
            for top in range(size - height + 1)
            for left in range(size - width + 1)
        ]
        random.shuffle(positions)
        for top, left in positions:
            if _can_place_patch(mask, patch, top, left, forbidden):
                paste(mask, patch.astype(bool), top, left, transparent=False)
                placed = True
                break
        if not placed:
            return None

    area = int(np.sum(mask))
    if area < BLANK_AREA_RANGE[0] or area > BLANK_AREA_RANGE[1]:
        return None
    return mask


def _visible_palette(grid: np.ndarray) -> set[int]:
    return {
        int(value)
        for value in np.unique(grid)
        if int(value) not in {Color.GREY, Color.MAROON}
    }


def _expected_output(spec: CaseSpec) -> np.ndarray:
    out = _pattern_from_cycle(spec.input_grid.shape, spec.cycle)
    out[_rect_mask(out.shape, spec.large_rect)] = Color.GREY
    return out


def _build_case() -> CaseSpec | None:
    size = random.choice(GRID_SIZES)
    period = random.choice(PERIOD_CHOICES)
    cycle = tuple(random.sample(CYCLE_PALETTE, k=period))
    full_pattern = _pattern_from_cycle((size, size), cycle)

    rect_pair = _sample_rect_pair(size)
    if rect_pair is None:
        return None
    small_rect, large_rect = rect_pair

    rect_forbidden = _rect_mask(full_pattern.shape, small_rect) | _rect_mask(full_pattern.shape, large_rect)
    blank_mask = _sample_blank_mask(size, rect_forbidden)
    if blank_mask is None:
        return None

    input_grid = full_pattern.copy()
    input_grid[blank_mask] = Color.MAROON
    input_grid[_rect_mask(input_grid.shape, small_rect)] = Color.GREY
    input_grid[_rect_mask(input_grid.shape, large_rect)] = Color.GREY

    visible_colors = _visible_palette(input_grid)
    if visible_colors != set(cycle):
        return None

    candidates = _candidate_cycles(input_grid)
    if candidates != [(period, cycle)]:
        return None

    output_grid = full_pattern.copy()
    output_grid[_rect_mask(output_grid.shape, large_rect)] = Color.GREY

    spec = CaseSpec(
        input_grid=input_grid,
        output_grid=output_grid,
        period=period,
        cycle=cycle,
        blank_mask=blank_mask,
        small_rect=small_rect,
        large_rect=large_rect,
    )

    if not np.array_equal(_expected_output(spec), output_grid):
        return None
    if not np.array_equal(main(input_grid), output_grid):
        return None

    out_rects = _grey_rectangles(output_grid)
    if out_rects != [large_rect]:
        return None
    if np.any(output_grid == Color.MAROON):
        return None
    if np.any(input_grid[_rect_mask(input_grid.shape, small_rect)] != Color.GREY):
        return None
    if np.any(blank_mask & (_rect_mask(blank_mask.shape, small_rect) | _rect_mask(blank_mask.shape, large_rect))):
        return None
    return spec


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        spec = _build_case()
        if spec is not None:
            return spec.input_grid
    raise RuntimeError("failed to generate a valid periodic-occluder input")
