import numpy as np

from arc_utils import Color, find_components, get_bg
import itertools
import random


def _squares(g: np.ndarray, bg: int) -> list[tuple[int, int, int, int, int]]:
    out: list[tuple[int, int, int, int, int]] = []
    for comp in find_components(
        g,
        bg=bg,
        colors=[Color.GREY, Color.PINK],
        connectivity=4,
    ):
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        assert comp["area"] == 9
        assert r1 - r0 == 2 and c1 - c0 == 2
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color)
        out.append((r0, c0, r1, c1, color))
    assert out
    return out


def _touch_side(
    r: int, c: int, square: tuple[int, int, int, int, int]
) -> str | None:
    r0, c0, r1, c1, _color = square
    if r0 <= r <= r1:
        if c == c0 - 1:
            return "left"
        if c == c1 + 1:
            return "right"
    if c0 <= c <= c1:
        if r == r0 - 1:
            return "above"
        if r == r1 + 1:
            return "below"
    return None


def _target_for_marker(
    r: int, c: int, side: str, square: tuple[int, int, int, int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1, color = square
    if color == Color.PINK:
        if side in {"left", "right"}:
            return r, (c0 + c1) // 2
        return (r0 + r1) // 2, c

    assert color == Color.GREY
    if side == "left":
        return r, c0
    if side == "right":
        return r, c1
    if side == "above":
        return r0, c
    assert side == "below"
    return r1, c


def _pink_targets_for_markers(
    markers: list[tuple[int, int, str]],
    square: tuple[int, int, int, int, int],
) -> set[tuple[int, int]]:
    _r0, _c0, _r1, _c1, color = square
    assert color == Color.PINK
    return {_target_for_marker(r, c, side, square) for r, c, side in markers}


def main(grid: np.ndarray) -> np.ndarray:
    """Project maroon markers into adjacent grey or pink 3x3 squares, erasing grey squares afterward.

    Input structure
    The grid has a dominant background, solid 3x3 grey and/or pink squares, and maroon marker pixels or segments directly touching one side of exactly one square.
    Rule
    Grey squares are erased; their adjacent maroon markers stay in place and stamp the nearest edge cell inside the grey square. Pink squares stay pink; their adjacent maroon markers disappear and stamp the corresponding cell on the square's middle row or middle column.
    Key insight
    The side touched by a marker determines the projection: grey uses the near edge, while pink uses the centerline, preserving the marker's coordinate along that side.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = set(map(int, np.unique(g)))
    assert colors <= {bg, Color.GREY, Color.PINK, Color.MAROON}
    assert Color.MAROON in colors

    squares = _squares(g, bg)
    out = g.copy()
    out[g == Color.GREY] = bg

    paint: set[tuple[int, int]] = set()
    remove_sources: list[tuple[int, int]] = []
    pink_markers_by_square: dict[int, list[tuple[int, int, str]]] = {}
    for r, c in np.argwhere(g == Color.MAROON):
        hits = []
        for idx, square in enumerate(squares):
            side = _touch_side(int(r), int(c), square)
            if side is not None:
                hits.append((side, idx, square))
        assert len(hits) == 1

        side, idx, square = hits[0]
        if square[4] == Color.PINK:
            pink_markers_by_square.setdefault(idx, []).append((int(r), int(c), side))
            remove_sources.append((int(r), int(c)))
        else:
            target = _target_for_marker(int(r), int(c), side, square)
            tr, tc = target
            assert g[tr, tc] == square[4]
            paint.add(target)

    for idx, markers in pink_markers_by_square.items():
        square = squares[idx]
        for target in _pink_targets_for_markers(markers, square):
            tr, tc = target
            assert g[tr, tc] == square[4]
            paint.add(target)

    for r, c in remove_sources:
        out[r, c] = bg
    for r, c in paint:
        out[r, c] = Color.MAROON

    return out

# --------------- generator (o5x) ---------------

from arc_utils import paste


# Colors.
_GEN_BG = 8
_GEN_GREY = 5
_GEN_PINK = 6
_GEN_MAROON = 9

# Canvas and square geometry.
_GEN_CANVAS_MIN = 9
_GEN_CANVAS_MAX = 15
_GEN_CANVAS_MAX_DELTA = 2
_GEN_SQUARE_SIZE = 3
_GEN_SQUARE_LAST = _GEN_SQUARE_SIZE - 1
_GEN_SQUARE_GAP = 1

# Square count bounds.
_GEN_TOTAL_SQUARE_MIN = 1
_GEN_TOTAL_SQUARE_MAX = 7
_GEN_GREY_SQUARE_MAX = 3
_GEN_PINK_SQUARE_MAX = 4

# Marker side and count bounds.
_GEN_SIDES = ("left", "right", "above", "below")
_GEN_MARKER_SIDE_MIN = 1
_GEN_MARKER_SIDE_MAX = 3
_GEN_MARKER_PER_SIDE_MAX = 3
_GEN_MARKER_TOTAL_MAX = 5

# Rejection sampling budget.
_GEN_MAX_ATTEMPTS = 300
_GEN_SQUARE_PATCHES = {color: np.full((_GEN_SQUARE_SIZE, _GEN_SQUARE_SIZE), color, dtype=int) for color in (_GEN_GREY, _GEN_PINK)}

_GenSquare = tuple[int, int, int]
_GenMarker = tuple[int, int, str]


def _gen_square_gap_ok(r0: int, c0: int, squares: list[_GenSquare]) -> bool:
    last = _GEN_SQUARE_LAST
    gap = _GEN_SQUARE_GAP
    r1, c1 = r0 + last, c0 + last
    for sr0, sc0, _color in squares:
        sr1, sc1 = sr0 + last, sc0 + last
        if not (
            r1 < sr0 - gap
            or sr1 < r0 - gap
            or c1 < sc0 - gap
            or sc1 < c0 - gap
        ):
            return False
    return True


def _gen_place_square(grid: np.ndarray, r0: int, c0: int, color: int) -> None:
    paste(grid, _GEN_SQUARE_PATCHES[int(color)], r0, c0)


def _gen_side_for_cell(r: int, c: int, square: _GenSquare) -> str | None:
    r0, c0, _color = square
    r1, c1 = r0 + _GEN_SQUARE_LAST, c0 + _GEN_SQUARE_LAST
    if r0 <= r <= r1:
        if c == c0 - 1:
            return "left"
        if c == c1 + 1:
            return "right"
    if c0 <= c <= c1:
        if r == r0 - 1:
            return "above"
        if r == r1 + 1:
            return "below"
    return None


def _gen_inside_square(r: int, c: int, square: _GenSquare) -> bool:
    r0, c0, _color = square
    r1, c1 = r0 + _GEN_SQUARE_LAST, c0 + _GEN_SQUARE_LAST
    return r0 <= r <= r1 and c0 <= c <= c1


def _gen_valid_marker_slot(r: int, c: int, squares: list[_GenSquare]) -> bool:
    if any(_gen_inside_square(r, c, square) for square in squares):
        return False
    hits = sum(
        _gen_side_for_cell(r, c, square) is not None for square in squares
    )
    return hits == 1


def _gen_side_slots(H: int, W: int, square: _GenSquare, squares: list[_GenSquare]) -> dict[str, list[tuple[int, int]]]:
    r0, c0, _color = square
    edge = _GEN_SQUARE_SIZE
    raw = {
        "left": [(r, c0 - 1) for r in range(r0, r0 + edge)] if c0 > 0 else [],
        "right": [(r, c0 + edge) for r in range(r0, r0 + edge)] if c0 + edge < W else [],
        "above": [(r0 - 1, c) for c in range(c0, c0 + edge)] if r0 > 0 else [],
        "below": [(r0 + edge, c) for c in range(c0, c0 + edge)] if r0 + edge < H else [],
    }
    return {
        side: [
            (r, c)
            for r, c in slots
            if _gen_valid_marker_slot(r, c, squares)
        ]
        for side, slots in raw.items()
    }


def _gen_split_marker_total(total: int, caps: list[int]) -> list[int] | None:
    counts: list[int] = []
    remaining = int(total)
    for i, cap in enumerate(caps):
        rest = caps[i + 1 :]
        lo = max(1, remaining - sum(rest))
        hi = min(int(cap), remaining - len(rest))
        if lo > hi:
            return None
        count = random.randint(lo, hi)
        counts.append(count)
        remaining -= count
    return counts if remaining == 0 else None


def _gen_choose_square_markers(H: int, W: int, square: _GenSquare, squares: list[_GenSquare]) -> list[_GenMarker] | None:
    slots_by_side = _gen_side_slots(H, W, square, squares)
    options = [
        (side, slots_by_side[side])
        for side in _GEN_SIDES
        if slots_by_side[side]
    ]
    if not options:
        return None

    possible_side_counts = [
        k
        for k in range(_GEN_MARKER_SIDE_MIN, min(_GEN_MARKER_SIDE_MAX, len(options)) + 1)
        if any(
            sum(min(_GEN_MARKER_PER_SIDE_MAX, len(slots)) for _side, slots in combo) >= k
            for combo in itertools.combinations(options, k)
        )
    ]
    if not possible_side_counts:
        return None

    side_count = random.choice(possible_side_counts)
    combos = [
        combo
        for combo in itertools.combinations(options, side_count)
        if sum(min(_GEN_MARKER_PER_SIDE_MAX, len(slots)) for _side, slots in combo) >= side_count
    ]
    combo = list(random.choice(combos))
    random.shuffle(combo)
    caps = [min(_GEN_MARKER_PER_SIDE_MAX, len(slots)) for _side, slots in combo]
    marker_total = random.randint(side_count, min(_GEN_MARKER_TOTAL_MAX, sum(caps)))
    counts = _gen_split_marker_total(marker_total, caps)
    if counts is None:
        return None

    markers: list[_GenMarker] = []
    for (side, slots), count in zip(combo, counts):
        for r, c in random.sample(slots, count):
            markers.append((r, c, side))
    return markers


def _gen_target_for_marker(r: int, c: int, side: str, square: _GenSquare) -> tuple[int, int]:
    r0, c0, color = square
    r1, c1 = r0 + _GEN_SQUARE_LAST, c0 + _GEN_SQUARE_LAST
    if color == _GEN_PINK:
        if side in {"left", "right"}:
            return r, (c0 + c1) // 2
        return (r0 + r1) // 2, c
    if side == "left":
        return r, c0
    if side == "right":
        return r, c1
    if side == "above":
        return r0, c
    return r1, c


def _gen_pink_targets_for_markers(markers: list[_GenMarker], square: _GenSquare) -> set[tuple[int, int]]:
    _r0, _c0, color = square
    assert color == _GEN_PINK
    return {_gen_target_for_marker(r, c, side, square) for r, c, side in markers}


def _gen_choose_square_colors() -> list[int]:
    total = random.randint(_GEN_TOTAL_SQUARE_MIN, _GEN_TOTAL_SQUARE_MAX)
    grey_min = max(0, total - _GEN_PINK_SQUARE_MAX)
    grey_max = min(_GEN_GREY_SQUARE_MAX, total)
    grey_count = random.randint(grey_min, grey_max)
    colors = [_GEN_GREY] * grey_count + [_GEN_PINK] * (total - grey_count)
    random.shuffle(colors)
    return colors


def _gen_choose_canvas_shape() -> tuple[int, int]:
    H = random.randint(_GEN_CANVAS_MIN, _GEN_CANVAS_MAX)
    W = random.randint(
        max(_GEN_CANVAS_MIN, H - _GEN_CANVAS_MAX_DELTA),
        min(_GEN_CANVAS_MAX, H + _GEN_CANVAS_MAX_DELTA),
    )
    return H, W


def _gen_place_squares(H: int, W: int, colors: list[int]) -> list[_GenSquare] | None:
    squares: list[_GenSquare] = []
    for color in colors:
        candidates = [
            (r0, c0)
            for r0 in range(H - _GEN_SQUARE_SIZE + 1)
            for c0 in range(W - _GEN_SQUARE_SIZE + 1)
            if _gen_square_gap_ok(r0, c0, squares)
        ]
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        squares.append((r0, c0, int(color)))
    return squares


def _gen_choose_all_markers(H: int, W: int, squares: list[_GenSquare]) -> list[list[_GenMarker]] | None:
    markers_by_square: list[list[_GenMarker]] = []
    for square in squares:
        markers = _gen_choose_square_markers(H, W, square, squares)
        if markers is None:
            return None
        markers_by_square.append(markers)
    return markers_by_square


def _gen_render_output(H: int, W: int, squares: list[_GenSquare], markers_by_square: list[list[_GenMarker]]) -> np.ndarray:
    out = np.full((H, W), _GEN_BG, dtype=int)
    for r0, c0, color in squares:
        if color == _GEN_PINK:
            _gen_place_square(out, r0, c0, color)

    for square, markers in zip(squares, markers_by_square):
        color = square[2]
        if color == _GEN_PINK:
            for tr, tc in _gen_pink_targets_for_markers(markers, square):
                out[tr, tc] = _GEN_MAROON
        else:
            for r, c, side in markers:
                tr, tc = _gen_target_for_marker(r, c, side, square)
                out[r, c] = _GEN_MAROON
                out[tr, tc] = _GEN_MAROON
    return out


def _gen_build_input(out: np.ndarray, squares: list[_GenSquare], markers_by_square: list[list[_GenMarker]]) -> np.ndarray:
    inp = out.copy()
    for (r0, c0, color), markers in zip(squares, markers_by_square):
        _gen_place_square(inp, r0, c0, color)
        if color == _GEN_PINK:
            for r, c, _side in markers:
                inp[r, c] = _GEN_MAROON
    return inp


def _gen_background_dominates(grid: np.ndarray) -> bool:
    counts = {int(v): int((grid == v).sum()) for v in np.unique(grid)}
    return counts.get(_GEN_BG, 0) > max(
        counts.get(_GEN_GREY, 0),
        counts.get(_GEN_PINK, 0),
        counts.get(_GEN_MAROON, 0),
    )


def _gen_oracle_accepts(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        solved = main(inp)
    except Exception:
        return False
    return np.array_equal(solved, expected)


def _gen_attempt() -> np.ndarray | None:
    H, W = _gen_choose_canvas_shape()
    colors = _gen_choose_square_colors()

    squares = _gen_place_squares(H, W, colors)
    if squares is None:
        return None

    markers_by_square = _gen_choose_all_markers(H, W, squares)
    if markers_by_square is None:
        return None

    out = _gen_render_output(H, W, squares, markers_by_square)
    inp = _gen_build_input(out, squares, markers_by_square)
    if not _gen_background_dominates(inp):
        return None
    if not _gen_oracle_accepts(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid e39e9282 input")
