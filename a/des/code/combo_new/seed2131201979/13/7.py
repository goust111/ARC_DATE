import random
from dataclasses import dataclass

import numpy as np


BG = 0
GROUND_COLOR = 9
MAX_GENERATION_ATTEMPTS = 600


@dataclass(frozen=True)
class _Container:
    color: int
    top: int
    left: int
    outer_h: int
    outer_w: int
    gap_start: int
    gap_w: int
    sand_count: int
    sand_cells: tuple[tuple[int, int], ...]

    @property
    def bottom(self) -> int:
        return self.top + self.outer_h - 1

    @property
    def right(self) -> int:
        return self.left + self.outer_w - 1


def _find_ground_row(grid: np.ndarray) -> tuple[int, int]:
    for r in range(grid.shape[0] - 1, -1, -1):
        row = grid[r]
        if int(row[0]) != BG and np.all(row == row[0]):
            return r, int(row[0])
    raise AssertionError("expected a full-width non-background ground row")


def _locate_container(
    grid: np.ndarray, color: int, ground_r: int
) -> tuple[int, int, int, int, int, int, int]:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, "container color must appear"

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)

    assert r1 < ground_r, "container must float above ground"
    assert r1 - r0 >= 3 and c1 - c0 >= 4, "container bbox too small"
    assert np.all(grid[r0, c0 : c1 + 1] == color), "missing top border"
    assert np.all(grid[r0 : r1 + 1, c0] == color), "missing left border"
    assert np.all(grid[r0 : r1 + 1, c1] == color), "missing right border"

    bottom = grid[r1, c0 : c1 + 1]
    assert int(bottom[0]) == color and int(bottom[-1]) == color, "bottom corners must exist"
    inner_bottom = bottom[1:-1]
    gap_mask = inner_bottom != color
    assert np.any(gap_mask), "container needs a bottom opening"
    gap_idx = np.where(gap_mask)[0]
    assert int(gap_idx[-1]) - int(gap_idx[0]) + 1 == len(gap_idx), "opening must be contiguous"
    assert np.all(inner_bottom[gap_mask] == BG), "opening must be empty"
    assert np.all(inner_bottom[~gap_mask] == color), "remaining bottom border must stay intact"

    gap_w = int(len(gap_idx))
    assert 1 <= gap_w <= 3, "opening width must be in [1, 3]"
    gap_start = c0 + 1 + int(gap_idx[0])
    sand_count = int(np.count_nonzero(grid[r0 + 1 : r1, c0 + 1 : c1] == color))
    return r0, c0, r1, c1, gap_start, gap_w, sand_count


def _pour_sand(
    out: np.ndarray, ground_r: int, gap_start: int, gap_w: int, sand_count: int, color: int
) -> None:
    remaining = int(sand_count)
    row = ground_r - 1
    while remaining > 0:
        assert row >= 0, "sand pile exceeded grid height"
        fill = min(gap_w, remaining)
        out[row, gap_start : gap_start + fill] = color
        remaining -= fill
        row -= 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Leak each container's interior sand through its bottom opening and stack it on the ground below.

    Input structure:
    - The background is black.
    - The bottom of the grid contains a full-width non-black horizontal ground line.
    - Several distinct colors each form one rectangular container with a one-pixel border and a contiguous opening on the bottom edge.
    - Inside each container are scattered single-pixel sand grains in the same color as that container's border.

    Rule:
    - Keep the grid size, the ground line, and every container border unchanged, including the exact opening shape.
    - For each container, count all same-color pixels strictly inside the border and erase them from the interior.
    - Drop that many grains through the bottom opening and stack them directly above the ground using the opening's width.
    - Fill the pile row by row from the bottom upward; if the top row is incomplete, fill it from left to right.

    Key insight:
    - The opening determines both where the sand lands and how many columns are available per layer, so the sand-count converts into a rectangular pile with a possible partial top row.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    ground_r, ground_color = _find_ground_row(g)

    colors = sorted(int(v) for v in np.unique(g) if int(v) not in (BG, ground_color))
    for color in colors:
        r0, c0, r1, c1, gap_start, gap_w, sand_count = _locate_container(g, color, ground_r)
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == color] = BG
        _pour_sand(out, ground_r, gap_start, gap_w, sand_count, color)

    return out


def _sample_sand_cells(
    rng: random.Random, top: int, left: int, outer_h: int, outer_w: int, count: int
) -> tuple[tuple[int, int], ...]:
    all_cells = [
        (r, c)
        for r in range(top + 1, top + outer_h - 1)
        for c in range(left + 1, left + outer_w - 1)
    ]
    inset_cells = [
        (r, c)
        for r, c in all_cells
        if top + 1 < r < top + outer_h - 2 and left + 1 < c < left + outer_w - 2
    ]
    chosen: list[tuple[int, int]] = []
    if inset_cells:
        take = min(count, len(inset_cells))
        chosen.extend(rng.sample(inset_cells, take))
    if len(chosen) < count:
        remaining = [cell for cell in all_cells if cell not in chosen]
        chosen.extend(rng.sample(remaining, count - len(chosen)))
    rng.shuffle(chosen)
    return tuple(chosen)


def _choose_sand_count(rng: random.Random, gap_w: int, max_count: int) -> int:
    counts = list(range(1, max_count + 1))
    divisible = [n for n in counts if n % gap_w == 0]
    partial = [n for n in counts if n % gap_w != 0]
    if divisible and partial:
        pool = divisible if rng.random() < 0.5 else partial
        return rng.choice(pool)
    return rng.choice(counts)


def _draw_container(grid: np.ndarray, spec: _Container, with_sand: bool) -> None:
    r0 = spec.top
    c0 = spec.left
    r1 = spec.bottom
    c1 = spec.right
    color = spec.color

    grid[r0, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r1, spec.gap_start : spec.gap_start + spec.gap_w] = BG

    if with_sand:
        for r, c in spec.sand_cells:
            grid[r, c] = color


def _build_container_specs(
    rng: random.Random, h: int, w: int
) -> list[_Container] | None:
    max_containers = min(4, max(1, w // 8))
    n_containers = rng.randint(1, max_containers)
    colors = rng.sample([c for c in range(1, 10) if c != GROUND_COLOR], n_containers)

    raw_specs = []
    widths = []
    for color in colors:
        gap_w = rng.randint(1, 3)
        min_outer_w = gap_w + 5
        max_outer_w = min(10, w - 2)
        if min_outer_w > max_outer_w:
            return None
        outer_w = rng.randint(min_outer_w, max_outer_w)

        max_outer_h = min(7, h - 3)
        if 4 > max_outer_h:
            return None
        outer_h = rng.randint(4, max_outer_h)
        max_clearance = h - outer_h - 2
        if max_clearance < 1:
            return None
        clearance = rng.randint(1, min(6, max_clearance))
        top = h - 1 - clearance - outer_h
        if top < 1:
            return None

        inner_area = (outer_h - 2) * (outer_w - 2)
        max_sand = min(10, inner_area, clearance * gap_w)
        if max_sand < 1:
            return None
        sand_count = _choose_sand_count(rng, gap_w, max_sand)

        gap_offset = rng.randint(2, outer_w - gap_w - 2)
        raw_specs.append((color, top, outer_h, outer_w, gap_w, sand_count, gap_offset))
        widths.append(outer_w)

    gaps = [1] + [2] * (n_containers - 1) + [1]
    base_width = sum(widths) + sum(gaps)
    if base_width > w:
        return None

    extra = w - base_width
    for _ in range(extra):
        gaps[rng.randrange(len(gaps))] += 1

    specs: list[_Container] = []
    left = gaps[0]
    for i, (color, top, outer_h, outer_w, gap_w, sand_count, gap_offset) in enumerate(raw_specs):
        gap_start = left + gap_offset
        sand_cells = _sample_sand_cells(rng, top, left, outer_h, outer_w, sand_count)
        specs.append(
            _Container(
                color=color,
                top=top,
                left=left,
                outer_h=outer_h,
                outer_w=outer_w,
                gap_start=gap_start,
                gap_w=gap_w,
                sand_count=sand_count,
                sand_cells=sand_cells,
            )
        )
        left += outer_w + gaps[i + 1]

    return specs


def _build_pair(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    h = rng.randint(12, 24)
    w = rng.randint(15, 30)
    specs = _build_container_specs(rng, h, w)
    if not specs:
        return None

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    inp[-1, :] = GROUND_COLOR
    out[-1, :] = GROUND_COLOR

    for spec in specs:
        _draw_container(inp, spec, with_sand=True)
        _draw_container(out, spec, with_sand=False)
        _pour_sand(out, h - 1, spec.gap_start, spec.gap_w, spec.sand_count, spec.color)

    return inp, out


def _valid_pair(inp: np.ndarray, out: np.ndarray, specs: list[_Container]) -> bool:
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if inp.shape != out.shape or max(inp.shape) > 30:
        return False
    if not np.all(inp[-1, :] == GROUND_COLOR) or not np.all(out[-1, :] == GROUND_COLOR):
        return False
    if np.array_equal(inp, out):
        return False

    for spec in specs:
        clearance = inp.shape[0] - 1 - spec.bottom - 1
        if clearance <= 0:
            return False
        if spec.sand_count > clearance * spec.gap_w:
            return False
        if not (1 <= spec.gap_w <= 3):
            return False
        if spec.left < 1 or spec.right >= inp.shape[1] - 1:
            return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, out)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair = _build_pair(rng)
    if pair is None:
        return None
    inp, out = pair

    colors = sorted(int(v) for v in np.unique(inp) if int(v) not in (BG, GROUND_COLOR))
    specs = []
    ground_r, _ = _find_ground_row(inp)
    for color in colors:
        r0, c0, r1, c1, gap_start, gap_w, sand_count = _locate_container(inp, color, ground_r)
        sand_cells = tuple(map(tuple, np.argwhere(inp[r0 + 1 : r1, c0 + 1 : c1] == color)))
        specs.append(
            _Container(
                color=color,
                top=r0,
                left=c0,
                outer_h=r1 - r0 + 1,
                outer_w=c1 - c0 + 1,
                gap_start=gap_start,
                gap_w=gap_w,
                sand_count=sand_count,
                sand_cells=tuple((r0 + 1 + r, c0 + 1 + c) for r, c in sand_cells),
            )
        )

    if not _valid_pair(inp, out, specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid falling-sand container puzzle")
