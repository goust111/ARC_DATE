import numpy as np

from arc_utils import Color, get_bg


def _unique_position(g: np.ndarray, color: int) -> tuple[int, int]:
    locs = np.argwhere(g == color)
    assert len(locs) == 1, "each route marker color must appear exactly once"
    return tuple(map(int, locs[0]))


def _paint_horizontal_then_vertical(
    out: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    bg: int,
    color: int,
) -> None:
    r0, c0 = start
    r1, c1 = end

    if c0 != c1:
        dc = 1 if c1 > c0 else -1
        for c in range(c0 + dc, c1 + dc, dc):
            if out[r0, c] == bg:
                out[r0, c] = color

    if r0 != r1:
        dr = 1 if r1 > r0 else -1
        for r in range(r0 + dr, r1 + dr, dr):
            if out[r, c1] == bg:
                out[r, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a grey orthogonal route from the red marker to the yellow marker and then from the yellow marker to the green marker.

    Input structure
    - A black-background grid containing exactly one red pixel, one yellow pixel, and one green pixel.
    - The marker pixels are the only foreground cells in the input.

    Rule
    - Preserve the three colored marker pixels.
    - For red-to-yellow and yellow-to-green, draw the connecting path by moving horizontally to the target column, then vertically to the target row.
    - Paint only background cells along those paths grey.

    Key insight
    - The yellow marker is the bend waypoint between the red start and green finish; each hop uses the same horizontal-first Manhattan routing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"
    assert set(map(int, np.unique(g))) == {Color.BLACK, Color.RED, Color.GREEN, Color.YELLOW}

    red = _unique_position(g, Color.RED)
    yellow = _unique_position(g, Color.YELLOW)
    green = _unique_position(g, Color.GREEN)

    out = g.copy()
    _paint_horizontal_then_vertical(out, red, yellow, bg=bg, color=Color.GREY)
    _paint_horizontal_then_vertical(out, yellow, green, bg=bg, color=Color.GREY)
    return out

# --------------- generator (o5x) ---------------

# Canvas and colors.
_GEN_SIZE = 13
_GEN_BACKGROUND_COLOR = Color.BLACK
_GEN_MARKER_COLORS = (Color.RED, Color.YELLOW, Color.GREEN)

# Marker placement constraints.
_GEN_EDGE_DISTANCES = (1, 2)
_GEN_ROUTE_DISTS = {11, 12, 13}

# Generation retry limit.
_GEN_MAX_ATTEMPTS = 300


def _gen_edge_distance(pos: tuple[int, int]) -> int:
    r, c = pos
    return min(r, c, _GEN_SIZE - 1 - r, _GEN_SIZE - 1 - c)


def _gen_manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _gen_is_valid_pair(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return (
        a[0] != b[0]
        and a[1] != b[1]
        and _gen_manhattan(a, b) in _GEN_ROUTE_DISTS
    )


def _gen_all_marker_triples() -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    positions = [
        (r, c)
        for r in range(_GEN_SIZE)
        for c in range(_GEN_SIZE)
        if _gen_edge_distance((r, c)) in _GEN_EDGE_DISTANCES
    ]

    triples = []
    for red in positions:
        for yellow in positions:
            if not _gen_is_valid_pair(red, yellow):
                continue
            for green in positions:
                if (
                    _gen_is_valid_pair(yellow, green)
                    and _gen_is_valid_pair(red, green)
                    and len({red[0], yellow[0], green[0]}) == len(_GEN_MARKER_COLORS)
                    and len({red[1], yellow[1], green[1]}) == len(_GEN_MARKER_COLORS)
                ):
                    triples.append((red, yellow, green))
    return triples


_GEN_MARKER_TRIPLES = _gen_all_marker_triples()


def _gen_attempt() -> np.ndarray | None:
    if not _GEN_MARKER_TRIPLES:
        return None
    marker_triple = _GEN_MARKER_TRIPLES[
        int(np.random.randint(0, len(_GEN_MARKER_TRIPLES)))
    ]
    grid = np.full((_GEN_SIZE, _GEN_SIZE), _GEN_BACKGROUND_COLOR, dtype=int)
    for pos, color in zip(marker_triple, _GEN_MARKER_COLORS):
        grid[pos] = color
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            main(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid 0e671a1a input")
