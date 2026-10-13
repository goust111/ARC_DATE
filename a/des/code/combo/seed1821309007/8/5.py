import numpy as np


_RNG = np.random.default_rng()
_BG = 0
_MAX_RETRIES = 300
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_CORNERS = ("tl", "tr", "bl", "br")
_OPPOSITE = {"tl": "br", "tr": "bl", "bl": "tr", "br": "tl"}


def _corner_blocks(r0: int, c0: int, r1: int, c1: int) -> dict[str, tuple[slice, slice]]:
    return {
        "tl": (slice(r0, r0 + 2), slice(c0, c0 + 2)),
        "tr": (slice(r0, r0 + 2), slice(c1 - 1, c1 + 1)),
        "bl": (slice(r1 - 1, r1 + 1), slice(c0, c0 + 2)),
        "br": (slice(r1 - 1, r1 + 1), slice(c1 - 1, c1 + 1)),
    }


def _marker_positions(r0: int, c0: int, r1: int, c1: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (r0 - 1, c0 - 1),
        "tr": (r0 - 1, c1 + 1),
        "bl": (r1 + 1, c0 - 1),
        "br": (r1 + 1, c1 + 1),
    }


def _find_rectangle_color(grid: np.ndarray, bg: int) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    pairs = [(int(color), int(count)) for color, count in zip(colors, counts) if int(color) != bg]
    assert pairs, "expected foreground colors"
    rect_color, rect_count = max(pairs, key=lambda x: x[1])
    assert sum(count == rect_count for _, count in pairs) == 1, "rectangle color must be unique by size"
    return rect_color


def _analyze(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int], dict[str, str], dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    bg = _BG
    rect_color = _find_rectangle_color(g, bg)

    coords = np.argwhere(g == rect_color)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert r1 - r0 + 1 >= 4 and c1 - c0 + 1 >= 4, "rectangle must support 2x2 corners"

    corner_status: dict[str, str] = {}
    for name, (rs, cs) in _corner_blocks(r0, c0, r1, c1).items():
        block = g[rs, cs]
        if bool(np.all(block == rect_color)):
            corner_status[name] = "full"
        elif bool(np.all(block == bg)):
            corner_status[name] = "missing"
        else:
            raise AssertionError("corner must be either intact or a 2x2 cutout")

    missing = [name for name, status in corner_status.items() if status == "missing"]
    full = [name for name, status in corner_status.items() if status == "full"]
    assert 1 <= len(missing) <= 2
    assert len(full) + len(missing) == 4

    marker_colors: dict[str, int] = {}
    for name, (mr, mc) in _marker_positions(r0, c0, r1, c1).items():
        assert 0 <= mr < g.shape[0] and 0 <= mc < g.shape[1], "marker positions must stay in bounds"
        value = int(g[mr, mc])
        if corner_status[name] == "full":
            assert value not in (bg, rect_color), "full corners need an outside marker"
            marker_colors[name] = value
        else:
            assert value == bg, "missing corners should not already contain a marker"

    for name in missing:
        opposite = _OPPOSITE[name]
        assert corner_status[opposite] == "full", "missing corner must map to a full opposite corner"

    return rect_color, (r0, c0, r1, c1), corner_status, marker_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each 2x2 missing rectangle corner using the marker color from the diagonally opposite full corner.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _, (r0, c0, r1, c1), corner_status, marker_colors = _analyze(g)

    for name, status in corner_status.items():
        if status != "missing":
            continue
        fill_color = marker_colors[_OPPOSITE[name]]
        rs, cs = _corner_blocks(r0, c0, r1, c1)[name]
        out[rs, cs] = fill_color

    return out


def _choose_missing_corners() -> list[str]:
    if bool(_RNG.integers(0, 2)):
        return [str(_RNG.choice(_CORNERS))]
    adjacent_pairs = [
        ["tl", "tr"],
        ["tr", "br"],
        ["br", "bl"],
        ["bl", "tl"],
    ]
    return list(adjacent_pairs[int(_RNG.integers(len(adjacent_pairs)))])


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(8, 17))
    w = int(_RNG.integers(8, 17))
    grid = np.full((h, w), _BG, dtype=int)

    rect_h = int(_RNG.integers(4, h - 1))
    rect_w = int(_RNG.integers(4, w - 1))
    r0 = int(_RNG.integers(1, h - rect_h))
    c0 = int(_RNG.integers(1, w - rect_w))
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1

    color_pool = [c for c in range(1, 10)]
    rect_color = int(_RNG.choice(color_pool))
    grid[r0 : r1 + 1, c0 : c1 + 1] = rect_color

    missing = _choose_missing_corners()
    for name in missing:
        rs, cs = _corner_blocks(r0, c0, r1, c1)[name]
        grid[rs, cs] = _BG

    full_corners = [name for name in _CORNERS if name not in missing]
    marker_palette = list(_RNG.permutation([c for c in color_pool if c != rect_color]))
    for name, color in zip(full_corners, marker_palette, strict=False):
        mr, mc = _marker_positions(r0, c0, r1, c1)[name]
        grid[mr, mc] = int(color)

    try:
        out = main(grid)
    except AssertionError:
        return None

    for name in missing:
        rs, cs = _corner_blocks(r0, c0, r1, c1)[name]
        expected = marker_palette[full_corners.index(_OPPOSITE[name])]
        if not bool(np.all(out[rs, cs] == expected)):
            return None

    if len(np.unique(grid)) < 3:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {_MAX_RETRIES} attempts")
