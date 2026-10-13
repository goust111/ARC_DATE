from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import py_compile
import random
import shutil

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
HINGE = 4
ENDPOINT = 5
ORANGE = 7
ROD_COLORS = (BLUE, RED, GREEN, ORANGE)

MIN_HEIGHT = 16
MAX_HEIGHT = 24
MIN_WIDTH = 18
MAX_WIDTH = 26
SELF_TEST_TRIALS = 100
MAX_CASE_ATTEMPTS = 2000
ROW_CHOICES = (0, 2, 4, 6, 8)


@dataclass(frozen=True)
class Rod:
    row: int
    color: int
    hinge_col: int
    fixed_len: int
    tail_len: int
    endpoint_left: bool

    @property
    def span_start(self) -> int:
        if self.endpoint_left:
            return self.hinge_col - (self.fixed_len - 1)
        return self.hinge_col - self.tail_len

    @property
    def span_end(self) -> int:
        if self.endpoint_left:
            return self.hinge_col + self.tail_len
        return self.hinge_col + (self.fixed_len - 1)

    @property
    def endpoint_col(self) -> int:
        return self.span_start if self.endpoint_left else self.span_end

    @property
    def tail_cols(self) -> tuple[int, ...]:
        if self.endpoint_left:
            return tuple(range(self.hinge_col + 1, self.span_end + 1))
        return tuple(range(self.span_start, self.hinge_col))


@dataclass(frozen=True)
class Case:
    input_grid: np.ndarray
    output_grid: np.ndarray
    signature: tuple[object, ...]


def _find_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) == BG:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(grid[nr, nc]) != BG:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(cells)
    return components


def _parse_rod(grid: np.ndarray, cells: list[tuple[int, int]]) -> Rod:
    rows = {r for r, _ in cells}
    assert len(rows) == 1, "each object must be a single horizontal rod"
    row = next(iter(rows))
    cols = sorted(c for _, c in cells)
    span_start = cols[0]
    span_end = cols[-1]
    assert cols == list(range(span_start, span_end + 1)), "rod cells must form one contiguous run"

    values = {c: int(grid[row, c]) for c in cols}
    endpoint_cols = [c for c, value in values.items() if value == ENDPOINT]
    hinge_cols = [c for c, value in values.items() if value == HINGE]
    assert len(endpoint_cols) == 1 and len(hinge_cols) == 1, "each rod needs one endpoint and one hinge marker"

    endpoint_col = endpoint_cols[0]
    hinge_col = hinge_cols[0]
    assert endpoint_col in (span_start, span_end), "endpoint marker must be at an end"
    assert span_start < hinge_col < span_end, "hinge must be strictly inside the rod"

    line_colors = {value for value in values.values() if value not in (HINGE, ENDPOINT)}
    assert len(line_colors) == 1, "rod body must use exactly one color"
    color = next(iter(line_colors))
    assert color in ROD_COLORS, "unexpected rod color"

    endpoint_left = endpoint_col == span_start
    if endpoint_left:
        fixed_len = hinge_col - span_start + 1
        tail_len = span_end - hinge_col
        assert all(int(grid[row, c]) == color for c in range(hinge_col + 1, span_end + 1))
        assert all(int(grid[row, c]) == color for c in range(span_start + 1, hinge_col))
    else:
        fixed_len = span_end - hinge_col + 1
        tail_len = hinge_col - span_start
        assert all(int(grid[row, c]) == color for c in range(span_start, hinge_col))
        assert all(int(grid[row, c]) == color for c in range(hinge_col + 1, span_end))

    assert fixed_len >= 2 and tail_len >= 1, "rod must have a fixed segment and a non-empty tail"
    return Rod(
        row=row,
        color=color,
        hinge_col=hinge_col,
        fixed_len=fixed_len,
        tail_len=tail_len,
        endpoint_left=endpoint_left,
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each horizontal rod at its hinge, then use the folded tail as a downward fill source.

    Input structure:
    - The background is black.
    - Three or four colored horizontal rods lie in the upper half of the grid.
    - Each rod has one gray endpoint marker on a fixed end, one yellow hinge marker
      in the interior, and a single body color on all remaining cells.

    Rule:
    - For each rod, keep the segment from the gray endpoint through the yellow hinge unchanged.
    - Erase the tail on the far side of the hinge.
    - Fold that tail downward into the hinge column, starting one row below the hinge.
    - Use the folded vertical tail as a source and fill that entire column downward
      to the bottom with the rod's color.

    Key insight:
    - The input contains no pre-existing vertical sources; the fold creates them, and
      only after that new structure exists does the column fill become possible.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape

    rods = [_parse_rod(g, cells) for cells in _find_components(g)]
    assert 3 <= len(rods) <= 4, "expected three or four rods"

    hinge_cols = [rod.hinge_col for rod in rods]
    assert len(set(hinge_cols)) == len(hinge_cols), "hinge columns must be unique"

    for rod in rods:
        assert rod.row + rod.tail_len < height - 1, "folded tail must leave room below itself"
        for rr in range(rod.row + 1, height):
            assert int(g[rr, rod.hinge_col]) == BG, "hinge column below each rod must be empty in the input"

    out = g.copy()
    for rod in rods:
        for c in rod.tail_cols:
            out[rod.row, c] = BG
        out[rod.row + 1 :, rod.hinge_col] = rod.color
    return out


def _render_input(height: int, width: int, rods: list[Rod]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for rod in rods:
        row = rod.row
        start = rod.span_start
        end = rod.span_end
        grid[row, start : end + 1] = rod.color
        grid[row, rod.hinge_col] = HINGE
        grid[row, rod.endpoint_col] = ENDPOINT
    return grid


def _render_output(height: int, width: int, rods: list[Rod]) -> np.ndarray:
    out = _render_input(height, width, rods)
    for rod in rods:
        for c in rod.tail_cols:
            out[rod.row, c] = BG
        out[rod.row + 1 :, rod.hinge_col] = rod.color
    return out


def _rod_signature(rod: Rod) -> tuple[int, int, int, int, int, int]:
    return (
        rod.row,
        rod.color,
        rod.hinge_col,
        rod.fixed_len,
        rod.tail_len,
        int(rod.endpoint_left),
    )


def _is_valid_with_existing(candidate: Rod, placed: list[Rod]) -> bool:
    for other in placed:
        if candidate.span_start <= other.hinge_col <= candidate.span_end:
            return False
        if other.span_start <= candidate.hinge_col <= other.span_end:
            return False
    return True


def _sample_rod(
    rng: random.Random,
    height: int,
    width: int,
    row: int,
    color: int,
    placed: list[Rod],
) -> Rod | None:
    max_tail = min(6, height - row - 2, 9)
    if max_tail < 2:
        return None

    for _ in range(400):
        tail_len = rng.randint(2, max_tail)
        fixed_min = max(3, 6 - tail_len)
        fixed_max = min(8, 12 - tail_len)
        if fixed_min > fixed_max:
            continue
        fixed_len = rng.randint(fixed_min, fixed_max)
        endpoint_left = rng.choice([True, False])

        if endpoint_left:
            hinge_low = fixed_len - 1
            hinge_high = width - 1 - tail_len
        else:
            hinge_low = tail_len
            hinge_high = width - fixed_len
        if hinge_low > hinge_high:
            continue

        hinge_col = rng.randint(hinge_low, hinge_high)
        candidate = Rod(
            row=row,
            color=color,
            hinge_col=hinge_col,
            fixed_len=fixed_len,
            tail_len=tail_len,
            endpoint_left=endpoint_left,
        )
        if _is_valid_with_existing(candidate, placed):
            return candidate
    return None


def _make_case(rng: random.Random) -> Case:
    for _ in range(MAX_CASE_ATTEMPTS):
        height = rng.randint(MIN_HEIGHT, MAX_HEIGHT)
        width = rng.randint(MIN_WIDTH, MAX_WIDTH)
        rod_count = rng.choice([3, 4])
        rows = sorted(rng.sample(ROW_CHOICES, rod_count))
        colors = rng.sample(ROD_COLORS, rod_count)

        rods: list[Rod] = []
        ok = True
        for row, color in zip(rows, colors):
            rod = _sample_rod(rng, height, width, row, color, rods)
            if rod is None:
                ok = False
                break
            rods.append(rod)
        if not ok:
            continue

        input_grid = _render_input(height, width, rods)
        output_grid = _render_output(height, width, rods)
        predicted = main(input_grid)
        if not np.array_equal(predicted, output_grid):
            continue

        signature = (
            height,
            width,
            rod_count,
            tuple(sorted(_rod_signature(rod) for rod in rods)),
        )
        return Case(input_grid=input_grid, output_grid=output_grid, signature=signature)
    raise RuntimeError("failed to generate a valid puzzle instance")


def generate_input() -> np.ndarray:
    seed = random.randrange(1 << 30)
    return _make_case(random.Random(seed)).input_grid


def _validate_random_cases(trials: int = SELF_TEST_TRIALS) -> dict[str, int]:
    variants: set[tuple[object, ...]] = set()
    for trial in range(trials):
        case = _make_case(random.Random(20_000 + trial))
        assert case.input_grid.shape == case.output_grid.shape
        assert case.input_grid.shape[0] <= 30 and case.input_grid.shape[1] <= 30
        predicted = main(case.input_grid)
        assert np.array_equal(predicted, case.output_grid), f"mismatch on trial {trial}"
        variants.add(case.signature)
    assert len(variants) >= 20, f"expected at least 20 variants, got {len(variants)}"
    return {"trials": trials, "variants": len(variants)}


def _compile_self() -> None:
    py_compile.compile(str(Path(__file__)), doraise=True)


def _cleanup_pycache() -> None:
    pycache_dir = Path(__file__).with_name("__pycache__")
    if pycache_dir.exists():
        shutil.rmtree(pycache_dir)


if __name__ == "__main__":
    stats = _validate_random_cases()
    _compile_self()
    _cleanup_pycache()
    print(f"validated={stats['trials']} variants={stats['variants']}")
