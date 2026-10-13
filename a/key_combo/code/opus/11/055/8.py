from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
ORANGE = 7

DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
ORTHOGONALS = ((-1, 0), (1, 0), (0, -1), (0, 1))
MARKER_COLORS = {BLUE, RED}
INPUT_COLORS = {BLACK, BLUE, RED, GREY}

MAX_ATTEMPTS = 500
VARIANTS = (
    {"require_conflict": True, "require_red_edge": True, "require_blue_edge": False},
    {"require_conflict": True, "require_red_edge": False, "require_blue_edge": True},
    {"require_conflict": True, "require_red_edge": True, "require_blue_edge": True},
    {"require_conflict": False, "require_red_edge": True, "require_blue_edge": True},
    {"require_conflict": False, "require_red_edge": True, "require_blue_edge": False},
    {"require_conflict": False, "require_red_edge": False, "require_blue_edge": True},
)
VARIANT_WEIGHTS = (3, 3, 2, 2, 1, 1)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _extract_band_height(grid: np.ndarray) -> int:
    h = int(grid.shape[0])
    band_h = 0
    for r in range(h):
        if np.all(grid[r] != BLACK):
            band_h += 1
        else:
            break
    assert band_h > 0, "expected a non-empty top band"
    assert np.all(grid[:band_h] != BLACK), "band rows must be fully filled"
    assert np.all(grid[band_h:] == BLACK), "rows below the top band must be black"
    return band_h


def _marker_targets(band: np.ndarray) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    h, w = map(int, band.shape)
    yellow_targets: set[tuple[int, int]] = set()
    orange_targets: set[tuple[int, int]] = set()

    for r, c in np.argwhere(band == RED):
        r = int(r)
        c = int(c)
        for dr, dc in DIAGONALS:
            nr = r + dr
            nc = c + dc
            if _in_bounds((h, w), nr, nc) and int(band[nr, nc]) == GREY:
                yellow_targets.add((nr, nc))

    for r, c in np.argwhere(band == BLUE):
        r = int(r)
        c = int(c)
        for dr, dc in ORTHOGONALS:
            nr = r + dr
            nc = c + dc
            if _in_bounds((h, w), nr, nc) and int(band[nr, nc]) == GREY:
                orange_targets.add((nr, nc))

    return yellow_targets, orange_targets


def _apply_marks_to_band(band: np.ndarray) -> np.ndarray:
    out = np.array(band, dtype=int, copy=True)
    yellow_targets, orange_targets = _marker_targets(np.asarray(band, dtype=int))

    for r, c in yellow_targets:
        out[r, c] = YELLOW
    for r, c in orange_targets:
        out[r, c] = ORANGE
    return out


def _simulate_correct(grid: np.ndarray, *, reverse_rows: bool = True) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    band_h = _extract_band_height(g)
    source = g[:band_h]
    bottom = source[::-1] if reverse_rows else source.copy()

    out = g.copy()
    out[-band_h:] = _apply_marks_to_band(bottom)
    return out


def _simulate_decorate_then_mirror(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    band_h = _extract_band_height(g)
    decorated_top = _apply_marks_to_band(g[:band_h])

    out = g.copy()
    out[-band_h:] = decorated_top[::-1]
    return out


def _local_is_edge(pos: tuple[int, int], band_h: int, width: int) -> bool:
    r, c = pos
    return r in {0, band_h - 1} or c in {0, width - 1}


def _local_is_corner(pos: tuple[int, int], band_h: int, width: int) -> bool:
    r, c = pos
    return r in {0, band_h - 1} and c in {0, width - 1}


def _valid_input_marker(pos: tuple[int, int], band_h: int, width: int) -> bool:
    return not _local_is_corner(pos, band_h, width)


def _too_close(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _can_place(pos: tuple[int, int], existing: list[tuple[int, int]], band_h: int, width: int) -> bool:
    if not _valid_input_marker(pos, band_h, width):
        return False
    return pos not in existing and all(not _too_close(pos, other) for other in existing)


def _all_allowed_positions(band_h: int, width: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(band_h)
        for c in range(width)
        if _valid_input_marker((r, c), band_h, width)
    ]


def _choose_conflict_pair(
    band_h: int,
    width: int,
    require_red_edge: bool,
    require_blue_edge: bool,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    allowed = _all_allowed_positions(band_h, width)
    candidates = []
    for red_pos in allowed:
        for blue_pos in allowed:
            if red_pos == blue_pos or _too_close(red_pos, blue_pos):
                continue
            if require_red_edge and not _local_is_edge(red_pos, band_h, width):
                continue
            if require_blue_edge and not _local_is_edge(blue_pos, band_h, width):
                continue
            red_band = np.full((band_h, width), GREY, dtype=int)
            red_band[red_pos] = RED
            red_band[blue_pos] = BLUE
            yellow_targets, orange_targets = _marker_targets(red_band)
            overlap = yellow_targets & orange_targets
            if overlap:
                candidates.append((red_pos, blue_pos))
    return random.choice(candidates) if candidates else None


def _pick_position(
    band_h: int,
    width: int,
    occupied: list[tuple[int, int]],
    *,
    prefer_edge: bool,
) -> tuple[int, int] | None:
    positions = _all_allowed_positions(band_h, width)
    random.shuffle(positions)
    edge_positions = [p for p in positions if _local_is_edge(p, band_h, width)]
    inner_positions = [p for p in positions if not _local_is_edge(p, band_h, width)]
    pools = [edge_positions, inner_positions] if prefer_edge else [inner_positions, edge_positions]
    for pool in pools:
        for pos in pool:
            if _can_place(pos, occupied, band_h, width):
                return pos
    return None


def _top_from_bottom_positions(
    band_h: int,
    width: int,
    reds_bottom: list[tuple[int, int]],
    blues_bottom: list[tuple[int, int]],
) -> np.ndarray:
    top = np.full((band_h, width), GREY, dtype=int)
    for r, c in reds_bottom:
        top[band_h - 1 - r, c] = RED
    for r, c in blues_bottom:
        top[band_h - 1 - r, c] = BLUE
    return top


def _analyze_case(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    band_h = _extract_band_height(g)
    top = g[:band_h].copy()
    bottom = top[::-1].copy()
    yellow_targets, orange_targets = _marker_targets(bottom)
    marked_bottom = _apply_marks_to_band(bottom)

    reds_bottom = [tuple(map(int, rc)) for rc in np.argwhere(bottom == RED)]
    blues_bottom = [tuple(map(int, rc)) for rc in np.argwhere(bottom == BLUE)]

    return {
        "band_h": band_h,
        "top": top,
        "bottom": bottom,
        "marked_bottom": marked_bottom,
        "yellow_targets": yellow_targets,
        "orange_targets": orange_targets,
        "conflict_targets": yellow_targets & orange_targets,
        "reds_bottom": reds_bottom,
        "blues_bottom": blues_bottom,
    }


def _validate_input(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert isinstance(g, np.ndarray)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert set(map(int, np.unique(g))).issubset(INPUT_COLORS)

    band_h = _extract_band_height(g)
    h, w = map(int, g.shape)
    assert band_h in {3, 4}
    assert h >= 2 * band_h + 1
    assert np.all(g[:band_h] != BLACK)
    assert np.all(g[band_h:] == BLACK)

    top = g[:band_h]
    red_count = int(np.count_nonzero(top == RED))
    blue_count = int(np.count_nonzero(top == BLUE))
    assert 2 <= red_count <= 4
    assert 2 <= blue_count <= 4

    markers = [tuple(map(int, rc)) for rc in np.argwhere(np.isin(top, [RED, BLUE]))]
    for pos in markers:
        assert _valid_input_marker(pos, band_h, w)
    for i, a in enumerate(markers):
        for b in markers[i + 1 :]:
            assert not _too_close(a, b)


def _validate_semantics(
    grid: np.ndarray,
    variant: dict[str, bool],
) -> None:
    _validate_input(grid)
    analysis = _analyze_case(grid)
    band_h = int(analysis["band_h"])
    top = np.asarray(analysis["top"])
    marked_bottom = np.asarray(analysis["marked_bottom"])
    yellow_targets = set(analysis["yellow_targets"])
    orange_targets = set(analysis["orange_targets"])
    conflict_targets = set(analysis["conflict_targets"])
    reds_bottom = list(analysis["reds_bottom"])
    blues_bottom = list(analysis["blues_bottom"])

    out = main(grid)
    expected = np.asarray(grid, dtype=int).copy()
    expected[-band_h:] = marked_bottom
    assert np.array_equal(out, expected)
    assert np.array_equal(out[:band_h], grid[:band_h])
    assert np.array_equal(out[band_h:-band_h], grid[band_h:-band_h])

    assert len(yellow_targets) > 0
    assert len(orange_targets) > 0
    assert np.count_nonzero(out[-band_h:] == YELLOW) > 0
    assert np.count_nonzero(out[-band_h:] == ORANGE) > 0

    if variant["require_conflict"]:
        assert conflict_targets
    else:
        assert not conflict_targets

    if conflict_targets:
        for r, c in conflict_targets:
            assert int(out[-band_h + r, c]) == ORANGE

    if variant["require_red_edge"]:
        assert any(_local_is_edge(pos, band_h, grid.shape[1]) for pos in reds_bottom)
    if variant["require_blue_edge"]:
        assert any(_local_is_edge(pos, band_h, grid.shape[1]) for pos in blues_bottom)

    assert not np.array_equal(_apply_marks_to_band(top), top), "top-only asymmetry must be visible"
    assert not np.array_equal(out, _simulate_correct(grid, reverse_rows=False)), "row reversal must matter"
    mirror_only = np.asarray(grid, dtype=int).copy()
    mirror_only[-band_h:] = np.asarray(analysis["bottom"])
    assert not np.array_equal(out, mirror_only), "marking must visibly change the mirrored band"

    mark_both = np.asarray(grid, dtype=int).copy()
    mark_both[:band_h] = _apply_marks_to_band(top)
    mark_both[-band_h:] = marked_bottom
    assert not np.array_equal(out, mark_both), "the top band must remain unchanged"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the top grey band to the bottom, then mark only the mirrored copy with yellow/red-derived diagonals and orange/blue-derived orthogonals.

    Input structure:
    - A black background fills the grid below a contiguous full-width band touching the top edge.
    - The top band is 3-4 rows tall and is mostly grey, with scattered red and blue single-pixel markers.
    - Rows below the band are entirely black, leaving a gap large enough for a separate bottom copy.

    Rule:
    - Keep the original top band unchanged.
    - Copy the top band into the bottom rows in vertically reversed row order.
    - Only inside that mirrored bottom copy, recolor grey diagonal neighbors of red markers to yellow and grey orthogonal neighbors of blue markers to orange.
    - If a grey cell is selected by both a red diagonal and a blue orthogonal expansion, color it orange.

    Key insight:
    - The neighborhood expansion is asymmetric: it acts only on the mirrored bottom band, so the vertical flip changes which in-band neighbors exist and therefore changes the final markings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset(INPUT_COLORS), "unexpected input colors"

    band_h = _extract_band_height(g)
    out = g.copy()
    out[-band_h:] = _apply_marks_to_band(g[:band_h][::-1])
    return out


def _generate_attempt(variant: dict[str, bool]) -> np.ndarray | None:
    band_h = random.randint(3, 4)
    height = random.randint(max(2 * band_h + 1, 10), 18)
    width = random.randint(8, 18)

    reds_bottom: list[tuple[int, int]] = []
    blues_bottom: list[tuple[int, int]] = []
    occupied: list[tuple[int, int]] = []

    if variant["require_conflict"]:
        pair = _choose_conflict_pair(
            band_h,
            width,
            require_red_edge=variant["require_red_edge"],
            require_blue_edge=variant["require_blue_edge"],
        )
        if pair is None:
            return None
        red_pos, blue_pos = pair
        reds_bottom.append(red_pos)
        blues_bottom.append(blue_pos)
        occupied.extend(pair)

    if variant["require_red_edge"] and not any(_local_is_edge(p, band_h, width) for p in reds_bottom):
        extra_red = _pick_position(band_h, width, occupied, prefer_edge=True)
        if extra_red is None:
            return None
        reds_bottom.append(extra_red)
        occupied.append(extra_red)

    if variant["require_blue_edge"] and not any(_local_is_edge(p, band_h, width) for p in blues_bottom):
        extra_blue = _pick_position(band_h, width, occupied, prefer_edge=True)
        if extra_blue is None:
            return None
        blues_bottom.append(extra_blue)
        occupied.append(extra_blue)

    red_total = random.randint(max(2, len(reds_bottom)), 4)
    blue_total = random.randint(max(2, len(blues_bottom)), 4)

    while len(reds_bottom) < red_total:
        pos = _pick_position(band_h, width, occupied, prefer_edge=random.random() < 0.4)
        if pos is None:
            return None
        reds_bottom.append(pos)
        occupied.append(pos)

    while len(blues_bottom) < blue_total:
        pos = _pick_position(band_h, width, occupied, prefer_edge=random.random() < 0.4)
        if pos is None:
            return None
        blues_bottom.append(pos)
        occupied.append(pos)

    top = _top_from_bottom_positions(band_h, width, reds_bottom, blues_bottom)
    grid = np.zeros((height, width), dtype=int)
    grid[:band_h] = top

    try:
        _validate_semantics(grid, variant)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        variant = random.choices(VARIANTS, weights=VARIANT_WEIGHTS, k=1)[0]
        grid = _generate_attempt(variant)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid non-degenerate input")
