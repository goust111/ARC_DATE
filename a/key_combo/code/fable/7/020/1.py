from __future__ import annotations

import random
from collections import deque

import numpy as np


# The three structural colors are stated by the puzzle.  Wallpaper colors are
# deliberately chosen from the remaining ARC palette by the generator.
BACKGROUND = 0
WINDOW = 5
MARKER = 8


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (n, 2) arrays of (row, column)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue = deque([(sr, sc)])
        cells: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _rectangles(grid: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int]]]:
    """Extract solid nonblack rectangular components and their canvas origins."""
    rectangles: list[tuple[np.ndarray, tuple[int, int]]] = []
    for cells in _components(grid != BACKGROUND):
        r0, c0 = cells.min(axis=0)
        r1, c1 = cells.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert cells.shape[0] == patch.size, "components must be solid rectangles"
        assert np.all(patch != BACKGROUND), "a rectangular object may not contain black"
        rectangles.append((patch, (r0, c0)))
    return rectangles


def _registered_observations(pieces: list[np.ndarray]) -> dict[tuple[int, int], int]:
    """Overlay pieces in marker-relative coordinates, checking visible agreement."""
    observed: dict[tuple[int, int], int] = {}
    for piece in pieces:
        marker_cells = np.argwhere(piece == MARKER)
        assert marker_cells.shape == (1, 2), "each fragment needs exactly one marker"
        mr, mc = map(int, marker_cells[0])
        for r, c in np.argwhere(piece != MARKER):
            key = (int(r) - mr, int(c) - mc)
            color = int(piece[r, c])
            assert color not in (BACKGROUND, WINDOW, MARKER)
            if key in observed:
                assert observed[key] == color, "registered overlaps must agree"
            else:
                observed[key] = color
    assert observed
    return observed


def _period_candidates(
    observed: dict[tuple[int, int], int],
) -> list[tuple[int, int, np.ndarray]]:
    """Enumerate rectangular periods whose every phase is visibly determined."""
    rows = [r for r, _ in observed]
    cols = [c for _, c in observed]
    max_h = max(rows) - min(rows) + 1
    max_w = max(cols) - min(cols) + 1
    candidates: list[tuple[int, int, np.ndarray]] = []
    for ph in range(1, max_h + 1):
        for pw in range(1, max_w + 1):
            phases: dict[tuple[int, int], int] = {}
            valid = True
            for (r, c), color in observed.items():
                phase = (r % ph, c % pw)
                if phase in phases and phases[phase] != color:
                    valid = False
                    break
                phases[phase] = color
            if not valid or len(phases) != ph * pw:
                continue
            tile = np.empty((ph, pw), dtype=int)
            for phase, color in phases.items():
                tile[phase] = color
            candidates.append((ph, pw, tile))
    return candidates


def _infer_tile(pieces: list[np.ndarray]) -> np.ndarray:
    observed = _registered_observations(pieces)
    candidates = _period_candidates(observed)
    assert candidates, "registered union must determine a complete period"
    smallest_area = min(ph * pw for ph, pw, _ in candidates)
    smallest = [item for item in candidates if item[0] * item[1] == smallest_area]
    assert len(smallest) == 1, "the minimum wallpaper period must be unique"
    return smallest[0][2]


def main(grid: np.ndarray) -> np.ndarray:
    """Register the marked fragments, infer their wallpaper period, and tile the marked window.

    Input structure:
    - Black is the dominant background around several separated solid rectangular
      wallpaper fragments and one solid gray rectangular window.
    - Every fragment and the window contain exactly one cyan marker pixel; all
      other fragment cells are wallpaper colors.

    Rule:
    - Translate fragments in their own row-column coordinates until their cyan
      markers coincide, requiring visible overlaps to agree.
    - Infer the unique minimum rectangular period whose phases are all covered by
      the registered union, including the true color hidden beneath the marker.
    - Return a grid the size of the gray window, periodically tiled so phase (0, 0)
      lies at that window's cyan marker; neither gray nor cyan remains in the output.

    Key insight:
    - The fragments are overlapping observations of one registered location, not
      adjacent puzzle pieces; the window marker is only a phase anchor for cropping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size and max(g.shape) <= 30
    assert int(np.count_nonzero(g == BACKGROUND)) > g.size // 2, "black must dominate"

    rectangles = _rectangles(g)
    assert len(rectangles) >= 4, "expected at least three fragments and one window"
    window_items = []
    fragment_items = []
    for patch, origin in rectangles:
        markers = np.argwhere(patch == MARKER)
        assert markers.shape == (1, 2), "every rectangle needs one cyan marker"
        nonmarker = patch != MARKER
        if np.all(patch[nonmarker] == WINDOW):
            window_items.append((patch, origin))
        else:
            assert not np.any(patch == WINDOW), "gray is reserved for the window"
            fragment_items.append((patch, origin))

    assert len(window_items) == 1, "the gray window must be unique"
    assert len(fragment_items) >= 3
    window_patch, _ = window_items[0]
    tile = _infer_tile([patch for patch, _ in fragment_items])

    marker = np.argwhere(window_patch == MARKER)
    assert marker.shape == (1, 2)
    mr, mc = map(int, marker[0])
    ph, pw = tile.shape
    rr = (np.arange(window_patch.shape[0])[:, None] - mr) % ph
    cc = (np.arange(window_patch.shape[1])[None, :] - mc) % pw
    return tile[rr, cc]


# ------------------------------- generator ---------------------------------

_WALLPAPER_PALETTE = (1, 2, 3, 4, 6, 7, 9)
_PERIOD_SHAPES = tuple(
    (h, w) for h in range(3, 6) for w in range(3, 6) if (h, w) != (3, 3)
)
_MAX_ATTEMPTS = 500


def _primitive_tile(ph: int, pw: int) -> np.ndarray | None:
    color_count = random.randint(3, min(5, ph * pw, len(_WALLPAPER_PALETTE)))
    palette = random.sample(_WALLPAPER_PALETTE, color_count)
    flat = palette + [random.choice(palette) for _ in range(ph * pw - color_count)]
    random.shuffle(flat)
    tile = np.asarray(flat, dtype=int).reshape(ph, pw)

    # Both stated axes must be necessary; reject accidental smaller periods.
    for qh in range(1, ph + 1):
        for qw in range(1, pw + 1):
            if (qh, qw) == (ph, pw):
                continue
            if all(tile[r, c] == tile[r % qh, c % qw] for r in range(ph) for c in range(pw)):
                return None
    return tile


def _sample_piece_bounds(ph: int, pw: int, count: int) -> list[tuple[int, int, int, int]] | None:
    """Sample marker-containing crops; each must add registered information."""
    for _ in range(150):
        bounds = []
        for _piece_index in range(count):
            # Extents on both sides make overlaps visible, while occasional full
            # extents expose another occurrence of the marker's hidden phase.
            up = random.randint(0, ph)
            down = random.randint(0, ph)
            left = random.randint(0, pw)
            right = random.randint(0, pw)
            if up + down < 2 or left + right < 2:
                break
            if up + down + 1 > 7 or left + right + 1 > 7:
                break
            bounds.append((-up, down, -left, right))
        if len(bounds) != count:
            continue

        sets = [
            {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) != (0, 0)}
            for r0, r1, c0, c1 in bounds
        ]
        union = set().union(*sets)
        # Every photograph is useful, and every pair has visible overlap beyond
        # the common marker.  No photograph alone reveals every wallpaper phase.
        if any(not (cells - set().union(*(sets[:i] + sets[i + 1 :]))) for i, cells in enumerate(sets)):
            continue
        if any(len(sets[i] & sets[j]) < 2 for i in range(count) for j in range(i + 1, count)):
            continue
        if any(len({(r % ph, c % pw) for r, c in cells}) == ph * pw for cells in sets):
            continue
        if len({(r % ph, c % pw) for r, c in union}) != ph * pw:
            continue
        if not any((r % ph, c % pw) == (0, 0) for r, c in union):
            continue
        return bounds
    return None


def _render_piece(tile: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    r0, r1, c0, c1 = bounds
    ph, pw = tile.shape
    piece = np.empty((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            piece[r - r0, c - c0] = tile[r % ph, c % pw]
    piece[-r0, -c0] = MARKER
    return piece


def _place_rectangles(patches: list[np.ndarray], h: int, w: int) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    order = list(range(len(patches)))
    random.shuffle(order)
    for index in order:
        patch = patches[index]
        ph, pw = patch.shape
        choices = []
        for r0 in range(h - ph + 1):
            for c0 in range(w - pw + 1):
                # A one-cell black moat prevents even diagonal contact.
                rr0, rr1 = max(0, r0 - 1), min(h, r0 + ph + 1)
                cc0, cc1 = max(0, c0 - 1), min(w, c0 + pw + 1)
                if np.all(grid[rr0:rr1, cc0:cc1] == BACKGROUND):
                    choices.append((r0, c0))
        if not choices:
            return None
        r0, c0 = random.choice(choices)
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch
    return grid


def _oracle(tile: np.ndarray, shape: tuple[int, int], marker: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mr, mc = marker
    return np.asarray(
        [[tile[(r - mr) % tile.shape[0], (c - mc) % tile.shape[1]] for c in range(w)] for r in range(h)],
        dtype=int,
    )


def _gen_attempt() -> np.ndarray | None:
    ph, pw = random.choice(_PERIOD_SHAPES)
    tile = _primitive_tile(ph, pw)
    if tile is None:
        return None

    count = random.choice((3, 4))
    bounds = _sample_piece_bounds(ph, pw, count)
    if bounds is None:
        return None
    pieces = [_render_piece(tile, b) for b in bounds]
    if any(len(set(map(int, piece[piece != MARKER]))) < 2 for piece in pieces):
        return None

    # Validate period uniqueness from the registered evidence independently of
    # the eventual canvas placement.
    observed = _registered_observations(pieces)
    candidates = _period_candidates(observed)
    minimum = min((a * b for a, b, _ in candidates), default=-1)
    winners = [(a, b, t) for a, b, t in candidates if a * b == minimum]
    if len(winners) != 1 or winners[0][:2] != (ph, pw) or not np.array_equal(winners[0][2], tile):
        return None

    wh = random.randint(max(6, ph + 2), 10)
    ww = random.randint(max(6, pw + 2), 11)
    wm = (random.randint(1, wh - 2), random.randint(1, ww - 2))
    window = np.full((wh, ww), WINDOW, dtype=int)
    window[wm] = MARKER
    patches = pieces + [window]

    # Vary both dimensions; retry placement on a fresh random canvas if needed.
    h = random.randint(22, 30)
    w = random.randint(22, 30)
    grid = _place_rectangles(patches, h, w)
    if grid is None or np.count_nonzero(grid == BACKGROUND) <= grid.size // 2:
        return None

    expected = _oracle(tile, window.shape, wm)
    try:
        actual = main(grid)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(actual, expected):
        return None
    if actual.shape == grid.shape or np.any(np.isin(actual, (BACKGROUND, WINDOW, MARKER))):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, structurally validated marked-wallpaper instance."""
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a non-degenerate wallpaper instance")
