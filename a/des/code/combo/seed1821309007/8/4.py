import numpy as np


BG = 0
BLUE = 1
GREEN = 3
MAROON = 9
INPUT_COLORS = (2, 4, 5, 6, 7, 8)
MAX_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _classify_color(coords: np.ndarray) -> tuple[str, int | None]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if np.all(rows == rows[0]):
        return ("h", None)
    if np.all(cols == cols[0]):
        return ("v", None)
    diffs = rows - cols
    if np.all(diffs == diffs[0]):
        return ("d", 1)
    sums = rows + cols
    if np.all(sums == sums[0]):
        return ("d", -1)
    raise AssertionError("each foreground color must lie on one straight horizontal, vertical, or diagonal line")


def _thick_rows(center: int, limit: int) -> tuple[int, int]:
    if center + 1 < limit:
        return center, center + 1
    return max(0, center - 1), center


def _thick_cols(center: int, limit: int) -> tuple[int, int]:
    if center + 1 < limit:
        return center, center + 1
    return max(0, center - 1), center


def _median_int(values: list[int]) -> int:
    ordered = sorted(int(v) for v in values)
    return ordered[len(ordered) // 2]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group monochrome line colors by orientation and replace each orientation group
    with a single two-pixel-wide canonical line.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)
    out = np.full_like(g, bg)

    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert colors, "expected at least one foreground color"

    groups: dict[str, list[dict[str, object]]] = {"h": [], "v": [], "d": []}
    diag_slopes: list[int] = []

    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.size > 0
        orientation, slope = _classify_color(coords)
        groups[orientation].append({"color": color, "coords": coords, "slope": slope})
        if orientation == "d":
            assert slope in (1, -1)
            diag_slopes.append(int(slope))

    if groups["h"]:
        coords = np.concatenate([entry["coords"] for entry in groups["h"]], axis=0)
        rmin, rmax = int(coords[:, 0].min()), int(coords[:, 0].max())
        cmin, cmax = int(coords[:, 1].min()), int(coords[:, 1].max())
        center = (rmin + rmax) // 2
        r0, r1 = _thick_rows(center, h)
        out[r0, cmin : cmax + 1] = MAROON
        out[r1, cmin : cmax + 1] = MAROON

    if groups["v"]:
        coords = np.concatenate([entry["coords"] for entry in groups["v"]], axis=0)
        rmin, rmax = int(coords[:, 0].min()), int(coords[:, 0].max())
        cmin, cmax = int(coords[:, 1].min()), int(coords[:, 1].max())
        center = (cmin + cmax) // 2
        c0, c1 = _thick_cols(center, w)
        out[rmin : rmax + 1, c0] = GREEN
        out[rmin : rmax + 1, c1] = GREEN

    if groups["d"]:
        coords = np.concatenate([entry["coords"] for entry in groups["d"]], axis=0)
        rmin, rmax = int(coords[:, 0].min()), int(coords[:, 0].max())
        cmin, cmax = int(coords[:, 1].min()), int(coords[:, 1].max())
        slope = diag_slopes[0]
        assert all(s == slope for s in diag_slopes), "all diagonal colors must share the same 45-degree slope"
        if slope == 1:
            base = _median_int([int(r - c) for r, c in coords])
            for r in range(rmin, rmax + 1):
                for c in range(cmin, cmax + 1):
                    if (r - c == base) or (r - c == base + 1):
                        out[r, c] = BLUE
        else:
            base = _median_int([int(r + c) for r, c in coords])
            for r in range(rmin, rmax + 1):
                for c in range(cmin, cmax + 1):
                    if (r + c == base) or (r + c == base + 1):
                        out[r, c] = BLUE

    return out


def _paintable(blocked: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    h, w = blocked.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if blocked[r, c]:
            return False
    return True


def _block_neighbors(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        r0 = max(0, r - 1)
        r1 = min(h - 1, r + 1)
        c0 = max(0, c - 1)
        c1 = min(w - 1, c + 1)
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _random_line(rng: np.random.Generator, shape: tuple[int, int], orientation: str, slope: int | None) -> list[tuple[int, int]] | None:
    h, w = shape
    length = int(rng.integers(2, min(7, max(h, w)) + 1))

    if orientation == "h":
        if w < length:
            return None
        r = int(rng.integers(0, h))
        c0 = int(rng.integers(0, w - length + 1))
        return [(r, c0 + dc) for dc in range(length)]

    if orientation == "v":
        if h < length:
            return None
        c = int(rng.integers(0, w))
        r0 = int(rng.integers(0, h - length + 1))
        return [(r0 + dr, c) for dr in range(length)]

    assert orientation == "d"
    max_len = min(h, w, 6)
    if max_len < 2:
        return None
    length = int(rng.integers(2, max_len + 1))
    if slope == 1:
        r0 = int(rng.integers(0, h - length + 1))
        c0 = int(rng.integers(0, w - length + 1))
        return [(r0 + i, c0 + i) for i in range(length)]

    r0 = int(rng.integers(0, h - length + 1))
    c0 = int(rng.integers(length - 1, w))
    return [(r0 + i, c0 - i) for i in range(length)]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(10, 19))
    w = int(rng.integers(10, 19))
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    orientations = ["h", "v", "d"]
    group_count = int(rng.integers(1, 4))
    chosen = list(rng.choice(orientations, size=group_count, replace=False))
    rng.shuffle(chosen)
    counts = {orientation: int(rng.integers(1, 4)) for orientation in chosen}

    diag_slope = int(rng.choice((1, -1)))
    needed_colors = sum(counts.values())
    if needed_colors > len(INPUT_COLORS):
        return None
    color_pool = list(rng.permutation(INPUT_COLORS)[:needed_colors])
    color_idx = 0

    for orientation in chosen:
        count = counts[orientation]
        placed = 0
        for _ in range(120):
            if placed == count:
                break
            color = int(color_pool[color_idx])
            slope = diag_slope if orientation == "d" else None
            coords = _random_line(rng, (h, w), orientation, slope)
            if coords is None or not _paintable(blocked, coords):
                continue
            for r, c in coords:
                grid[r, c] = color
            _block_neighbors(blocked, coords)
            color_idx += 1
            placed += 1
        if placed != count:
            return None

    if not np.any(grid):
        return None

    # Reject degenerate cases where a produced band would completely disappear
    # into another group due to overlap in the output.
    out = main(grid)
    if not np.any(out == MAROON) and "h" in chosen:
        return None
    if not np.any(out == GREEN) and "v" in chosen:
        return None
    if not np.any(out == BLUE) and "d" in chosen:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
