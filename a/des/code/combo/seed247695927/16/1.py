import numpy as np


TEAL = 6
RED = 2
_GEN_RNG = np.random.default_rng()
_GEN_RETRIES = 400
_COLOR_POOL = tuple(c for c in range(10) if c not in {TEAL, RED})


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _classify_diagonal_pixels(grid: np.ndarray) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    bg = _most_common_color(g)
    fg_coords = [tuple(map(int, rc)) for rc in np.argwhere(g != bg)]
    assert fg_coords, "expected foreground pixels"

    fg_colors = {int(g[r, c]) for r, c in fg_coords}
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    fg = next(iter(fg_colors))

    fg_mask = g == fg
    nwse: set[tuple[int, int]] = set()
    nesw: set[tuple[int, int]] = set()

    for r, c in fg_coords:
        has_nwse = False
        has_nesw = False
        for dr, dc in ((-1, -1), (1, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and fg_mask[rr, cc]:
                has_nwse = True
        for dr, dc in ((-1, 1), (1, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and fg_mask[rr, cc]:
                has_nesw = True

        assert has_nwse != has_nesw, "each foreground pixel must belong to exactly one diagonal orientation"
        if has_nwse:
            nwse.add((r, c))
        else:
            nesw.add((r, c))

    assert nwse and nesw, "expected both diagonal orientations"
    return nwse, nesw


def main(grid: np.ndarray) -> np.ndarray:
    """Color the more common diagonal orientation teal and the other red."""
    g = np.asarray(grid, dtype=int)
    nwse, nesw = _classify_diagonal_pixels(g)
    assert len(nwse) != len(nesw), "diagonal orientation counts must differ"

    bg = _most_common_color(g)
    out = np.full_like(g, bg)
    major, minor = (nwse, nesw) if len(nwse) > len(nesw) else (nesw, nwse)

    for r, c in major:
        out[r, c] = TEAL
    for r, c in minor:
        out[r, c] = RED
    return out


def _partition_total(total: int, rng: np.random.Generator) -> list[int]:
    assert total >= 2
    if total <= 4:
        return [total]

    max_parts = max(1, total // 2)
    num_parts = int(rng.integers(1, min(5, max_parts) + 1))
    remaining = total
    parts: list[int] = []
    for idx in range(num_parts - 1):
        min_rest = 2 * (num_parts - idx - 1)
        high = remaining - min_rest
        part = int(rng.integers(2, high + 1))
        parts.append(part)
        remaining -= part
    parts.append(remaining)
    rng.shuffle(parts)
    return parts


def _cells_for_segment(start_r: int, start_c: int, length: int, orientation: str) -> list[tuple[int, int]]:
    if orientation == "nwse":
        return [(start_r + i, start_c + i) for i in range(length)]
    if orientation == "nesw":
        return [(start_r + i, start_c - i) for i in range(length)]
    raise ValueError(f"unknown orientation: {orientation}")


def _fits_with_gap(cells: list[tuple[int, int]], occupied: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied:
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied and (rr, cc) not in cell_set:
                    return False
    return True


def _place_segments(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    lengths: list[int],
    orientation: str,
    color: int,
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    for length in lengths:
        placed = False
        for _ in range(200):
            if orientation == "nwse":
                start_r = int(rng.integers(0, h - length + 1))
                start_c = int(rng.integers(0, w - length + 1))
            else:
                start_r = int(rng.integers(0, h - length + 1))
                start_c = int(rng.integers(length - 1, w))
            cells = _cells_for_segment(start_r, start_c, length, orientation)
            if not _fits_with_gap(cells, occupied, grid.shape):
                continue
            for r, c in cells:
                grid[r, c] = color
                occupied.add((r, c))
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(8, 19))
    w = int(rng.integers(8, 19))
    area = h * w

    bg = int(rng.choice(_COLOR_POOL))
    fg_choices = [c for c in _COLOR_POOL if c != bg]
    fg = int(rng.choice(fg_choices))

    max_total = min(area // 3, max(14, min(h, w) * 4))
    if max_total < 8:
        return None
    total = int(rng.integers(8, max_total + 1))

    dominant_orientation = "nwse" if rng.random() < 0.5 else "nesw"
    dominant = int(rng.integers(total // 2 + 1, total - 1))
    other = total - dominant
    if other < 2:
        return None

    counts = {
        dominant_orientation: dominant,
        "nesw" if dominant_orientation == "nwse" else "nwse": other,
    }

    lengths = {
        key: _partition_total(value, rng)
        for key, value in counts.items()
    }
    if max(max(parts) for parts in lengths.values()) > min(h, w):
        return None

    grid = np.full((h, w), bg, dtype=int)
    occupied: set[tuple[int, int]] = set()

    order = ["nwse", "nesw"]
    rng.shuffle(order)
    for orientation in order:
        if not _place_segments(grid, occupied, lengths[orientation], orientation, fg, rng):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    teal_count = int(np.count_nonzero(out == TEAL))
    red_count = int(np.count_nonzero(out == RED))
    if teal_count <= red_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(_GEN_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
