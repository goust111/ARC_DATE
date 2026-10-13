import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

_SIZE_CHOICES = (11, 13)
_MIN_PIXELS = 4
_MAX_PIXELS = 8
_GEN_RETRIES = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _find_cross(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != BLACK):
        full_rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        full_cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        for r in full_rows:
            for c in full_cols:
                candidates.append((color, r, c))
    assert len(candidates) == 1, "expected exactly one full cross"
    color, r, c = candidates[0]
    assert h == w and h % 2 == 1 and r == c and h == 2 * r + 1
    return color, r, c


def _mirror_upper_left(source: np.ndarray) -> np.ndarray:
    qh, qw = source.shape
    out = np.zeros((2 * qh, 2 * qw), dtype=int)
    out[:qh, :qw] = source
    out[:qh, qw:] = np.fliplr(source)
    out[qh:, :qw] = np.flipud(source)
    out[qh:, qw:] = np.flipud(np.fliplr(source))
    return out


def _apply_seam_majority(mirrored: np.ndarray, tie_color: int) -> np.ndarray:
    out = mirrored.copy()
    h, w = mirrored.shape
    q = h // 2
    seam_mask = np.zeros_like(mirrored, dtype=bool)
    seam_mask[q - 1 : q + 1, :] = True
    seam_mask[:, q - 1 : q + 1] = True

    for r, c in np.argwhere(seam_mask & (mirrored != BLACK)):
        red_count = 0
        blue_count = 0
        for nr, nc in _neighbors4(int(r), int(c)):
            if not _in_bounds((h, w), nr, nc):
                continue
            color = int(mirrored[nr, nc])
            if color == RED:
                red_count += 1
            elif color == BLUE:
                blue_count += 1
        if red_count > blue_count:
            out[r, c] = RED
        elif blue_count > red_count:
            out[r, c] = BLUE
        else:
            out[r, c] = tie_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete the central cross, mirror the upper-left red/blue pattern into all four quadrants, then recolor seam-edge pixels by local majority.

    Input structure:
    - The input is an odd square grid with black background.
    - A full horizontal line and a full vertical line of the same color cross at the center.
    - Only the upper-left quadrant contains a small red/blue pattern; the other three quadrants are empty.

    Rule:
    - Detect the full central cross, record its color, and remove that row and column.
    - Mirror the remaining upper-left quadrant horizontally and vertically to fill the four quadrants of the smaller even-sized grid.
    - On the seam-adjacent pixels of the mirrored result (the two middle rows and two middle columns), count red and blue 4-neighbors using the mirrored grid before recoloring.
    - Recolor each non-black seam pixel to red if red neighbors are more numerous, blue if blue neighbors are more numerous, and to the original cross color on ties.

    Key insight:
    - The deleted cross creates a seam where mirrored copies touch, and only those contact-band pixels need a second-pass majority vote.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    cross_color, cross_r, cross_c = _find_cross(g)
    source = g[:cross_r, :cross_c].copy()
    assert source.shape == (cross_r, cross_c)
    assert set(np.unique(g[cross_r, :])) == {cross_color}
    assert set(np.unique(g[:, cross_c])) == {cross_color}
    assert not np.any(g[:cross_r, cross_c + 1 :] != BLACK)
    assert not np.any(g[cross_r + 1 :, :cross_c] != BLACK)
    assert not np.any(g[cross_r + 1 :, cross_c + 1 :] != BLACK)

    mirrored = _mirror_upper_left(source)
    return _apply_seam_majority(mirrored, tie_color=cross_color)


def _motif_tie_blue(q: int) -> dict[tuple[int, int], int]:
    return {
        (q - 1, q - 1): BLUE,
        (q - 1, q - 2): RED,
        (q - 2, q - 1): RED,
        (q - 2, q - 2): BLUE,
        (q - 3, q - 1): BLUE,
    }


def _motif_tie_red(q: int) -> dict[tuple[int, int], int]:
    return {
        (q - 1, q - 1): RED,
        (q - 1, q - 2): BLUE,
        (q - 2, q - 1): BLUE,
        (q - 1, q - 3): RED,
        (q - 2, q - 2): RED,
    }


def _seam_positions(q: int) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(2 * q)
        for c in range(2 * q)
        if r in {q - 1, q} or c in {q - 1, q}
    }


def _raw_output_from_source(source: np.ndarray) -> np.ndarray:
    return _mirror_upper_left(source)


def _changed_seam_stats(source: np.ndarray) -> tuple[int, bool, bool]:
    raw = _raw_output_from_source(source)
    final = _apply_seam_majority(raw, tie_color=GREEN)
    seam = _seam_positions(source.shape[0])
    changed = 0
    saw_green = False
    saw_flip = False
    for r, c in seam:
        if raw[r, c] == final[r, c]:
            continue
        changed += 1
        if final[r, c] == GREEN:
            saw_green = True
        if raw[r, c] in (RED, BLUE) and final[r, c] in (RED, BLUE) and raw[r, c] != final[r, c]:
            saw_flip = True
    return changed, saw_green, saw_flip


def _random_fill(source: np.ndarray, occupied: set[tuple[int, int]], target_pixels: int) -> None:
    q = source.shape[0]
    choices = [(r, c) for r in range(q) for c in range(q) if (r, c) not in occupied]
    np.random.shuffle(choices)
    for r, c in choices:
        if int(np.count_nonzero(source)) >= target_pixels:
            break
        source[r, c] = int(np.random.choice([BLUE, RED]))
        occupied.add((r, c))


def _build_source(q: int) -> np.ndarray:
    source = np.zeros((q, q), dtype=int)
    tie_blue = bool(np.random.randint(2))
    motif_fn = _motif_tie_blue if tie_blue else _motif_tie_red
    motif = motif_fn(q)
    for (r, c), color in motif.items():
        source[r, c] = color

    occupied = set(motif)
    survivor_color = RED if tie_blue else BLUE
    safe_cells = [(r, c) for r in range(q - 2) for c in range(q - 2) if (r, c) not in occupied]
    assert safe_cells, "expected at least one interior cell away from the seam"
    sr, sc = safe_cells[int(np.random.randint(len(safe_cells)))]
    source[sr, sc] = survivor_color
    occupied.add((sr, sc))

    target_pixels = int(np.random.randint(max(len(occupied), _MIN_PIXELS), _MAX_PIXELS + 1))
    _random_fill(source, occupied, target_pixels)
    return source


def _render_input(source: np.ndarray) -> np.ndarray:
    q = source.shape[0]
    n = 2 * q + 1
    grid = np.zeros((n, n), dtype=int)
    grid[q, :] = GREEN
    grid[:, q] = GREEN
    grid[:q, :q] = source
    return grid


def _is_valid_input(grid: np.ndarray) -> bool:
    q = grid.shape[0] // 2
    source = grid[:q, :q]
    pixel_count = int(np.count_nonzero(source))
    if not (_MIN_PIXELS <= pixel_count <= _MAX_PIXELS):
        return False
    colors = {int(x) for x in np.unique(source) if int(x) != BLACK}
    if colors != {BLUE, RED}:
        return False
    if not (np.any(source[q - 1, :] != BLACK) or np.any(source[:, q - 1] != BLACK)):
        return False

    changed, saw_green, saw_flip = _changed_seam_stats(source)
    if changed == 0 or not saw_green or not saw_flip:
        return False

    expected = main(grid)
    raw = _raw_output_from_source(source)
    out_colors = {int(x) for x in np.unique(expected)}
    return (
        expected.shape == (2 * q, 2 * q)
        and not np.array_equal(expected, raw)
        and {BLUE, RED, GREEN}.issubset(out_colors)
    )


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        n = int(np.random.choice(_SIZE_CHOICES))
        q = (n - 1) // 2
        source = _build_source(q)
        grid = _render_input(source)
        if _is_valid_input(grid):
            return grid
    raise RuntimeError("generate_input: failed to construct a valid puzzle instance")
