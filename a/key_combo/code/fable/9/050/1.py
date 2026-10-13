import random
from collections import Counter

import numpy as np


BG = 5
BLACK = 0
TILE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rot_orbit(r: int, c: int, side: int) -> list[tuple[int, int]]:
    return [
        (r, c),
        (c, side - 1 - r),
        (side - 1 - r, side - 1 - c),
        (side - 1 - c, r),
    ]


def _restore_motif(motif: np.ndarray) -> np.ndarray:
    side = motif.shape[0]
    restored = motif.copy()
    for r in range(side):
        for c in range(side):
            if restored[r, c] != BLACK:
                continue
            colors = {
                int(motif[rr, cc])
                for rr, cc in _rot_orbit(r, c, side)
                if int(motif[rr, cc]) != BLACK
            }
            assert len(colors) == 1, "each black cell must have one visible orbit color"
            restored[r, c] = colors.pop()
    return restored


def _locate_motif_and_marker(grid: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    bg = _get_bg(grid)
    comps = _find_components(grid != bg)
    assert len(comps) == 2, "expected one collage and one marker"

    boxes = [(_bbox(comp), comp) for comp in comps]
    boxes.sort(key=lambda item: len(item[1]), reverse=True)

    motif_box, motif_comp = boxes[0]
    marker_box, marker_comp = boxes[1]

    mr0, mc0, mr1, mc1 = motif_box
    side = mr1 - mr0 + 1
    assert side == mc1 - mc0 + 1 and side % 2 == 0, "collage must be an even square"
    assert len(motif_comp) >= 4, "collage must be larger than marker"
    assert len(marker_comp) == 1 and marker_box[0] == marker_box[2] and marker_box[1] == marker_box[3]

    marker = tuple(map(int, marker_comp[0]))
    return motif_box, marker


def _select_quadrant(restored: np.ndarray, marker: tuple[int, int], motif_box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = motif_box
    side = r1 - r0 + 1
    half = side // 2
    mr, mc = marker
    top = mr < r0 + half
    left = mc < c0 + half

    local_r = 0 if top else half
    local_c = 0 if left else half
    return restored[local_r : local_r + half, local_c : local_c + half].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """Repair a rotational four-quadrant collage and return the marker-indicated quadrant tile.

    Input structure:
    - A gray background grid contains one even-sided square collage and exactly one isolated marker pixel.
    - The collage is divided into four equal quadrants.
    - The quadrants are the same tile shown in the four 90-degree rotational orientations.
    - Some collage cells are replaced by black occluders, while the marker lies outside the collage.

    Rule:
    - Find the square collage and the marker.
    - Treat the collage center as a 4-fold rotation center; every black cell belongs to a four-position rotational orbit.
    - Replace each black cell by the unique visible non-black color elsewhere in its orbit.
    - Use the marker's direction relative to the collage center to choose the upper-left, upper-right, lower-left, or lower-right quadrant.
    - Output that repaired quadrant exactly as it appears in the collage, without rotating it back.

    Key insight:
    - Missing colors are recovered from rotational orbits across the four quadrants, and the external marker only selects which repaired rotation to crop out.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    motif_box, marker = _locate_motif_and_marker(g)
    r0, c0, r1, c1 = motif_box
    motif = g[r0 : r1 + 1, c0 : c1 + 1]
    restored = _restore_motif(motif)
    return _select_quadrant(restored, marker, motif_box)


def _quadrant_tiles(tile: np.ndarray) -> np.ndarray:
    n = tile.shape[0]
    motif = np.empty((2 * n, 2 * n), dtype=int)
    motif[:n, :n] = tile
    motif[:n, n:] = np.rot90(tile, -1)
    motif[n:, :n] = np.rot90(tile, 1)
    motif[n:, n:] = np.rot90(tile, 2)
    return motif


def _make_tile(n: int) -> np.ndarray:
    for _ in range(200):
        n_colors = random.randint(3, min(6, len(TILE_COLORS)))
        palette = random.sample(TILE_COLORS, n_colors)
        tile = np.array([[random.choice(palette) for _ in range(n)] for _ in range(n)], dtype=int)
        if (
            not np.array_equal(tile, np.rot90(tile, 1))
            and not np.array_equal(tile, np.rot90(tile, 2))
            and not np.array_equal(tile, np.rot90(tile, 3))
        ):
            return tile
    raise RuntimeError("failed to build asymmetric tile")


def _orbit_representatives(n: int) -> list[list[tuple[int, int]]]:
    return [_rot_orbit(r, c, 2 * n) for r in range(n) for c in range(n)]


def _quadrant_index(pos: tuple[int, int], n: int) -> int:
    r, c = pos
    return (0 if r < n else 2) + (0 if c < n else 1)


def _occlude_motif(motif: np.ndarray, selected_quad: int) -> np.ndarray:
    n = motif.shape[0] // 2
    occluded = motif.copy()
    reps = _orbit_representatives(n)
    random.shuffle(reps)

    count = random.randint(max(2, n), min(len(reps), max(4, 2 * n + 1)))
    chosen = reps[:count]
    selected_hit = False
    multi_hit = False

    for orbit in chosen:
        orbit_indices = list(range(4))
        random.shuffle(orbit_indices)
        k = random.choices([1, 2, 3], weights=[4, 3, 1], k=1)[0]
        picks = orbit_indices[:k]
        if selected_quad in picks:
            selected_hit = True
        if k >= 2:
            multi_hit = True
        for idx in picks:
            rr, cc = orbit[idx]
            occluded[rr, cc] = BLACK

    if not selected_hit:
        orbit = chosen[0]
        for idx, pos in enumerate(orbit):
            if _quadrant_index(pos, n) == selected_quad:
                rr, cc = pos
                occluded[rr, cc] = BLACK
                break
    if not multi_hit and len(chosen) >= 2:
        orbit = chosen[1]
        idxs = list(range(4))
        random.shuffle(idxs)
        for idx in idxs[:2]:
            rr, cc = orbit[idx]
            occluded[rr, cc] = BLACK
    return occluded


def _corner_ranges(shape: tuple[int, int], motif_box: tuple[int, int, int, int], quad: int) -> tuple[range, range]:
    h, w = shape
    r0, c0, r1, c1 = motif_box
    if quad == 0:
        return range(0, r0), range(0, c0)
    if quad == 1:
        return range(0, r0), range(c1 + 1, w)
    if quad == 2:
        return range(r1 + 1, h), range(0, c0)
    return range(r1 + 1, h), range(c1 + 1, w)


def _expected_output(tile: np.ndarray, quad: int) -> np.ndarray:
    if quad == 0:
        return tile.copy()
    if quad == 1:
        return np.rot90(tile, -1)
    if quad == 2:
        return np.rot90(tile, 1)
    return np.rot90(tile, 2)


def _validate_instance(grid: np.ndarray, expected: np.ndarray, selected_quad: int) -> bool:
    try:
        motif_box, marker = _locate_motif_and_marker(grid)
        r0, c0, r1, c1 = motif_box
        motif = grid[r0 : r1 + 1, c0 : c1 + 1]
        restored = _restore_motif(motif)
        side = motif.shape[0]
        n = side // 2

        for r in range(side):
            for c in range(side):
                if motif[r, c] != BLACK:
                    continue
                colors = {
                    int(motif[rr, cc])
                    for rr, cc in _rot_orbit(r, c, side)
                    if int(motif[rr, cc]) != BLACK
                }
                if len(colors) != 1:
                    return False

        mr, mc = marker
        top = mr < r0 + n
        left = mc < c0 + n
        quad = (0 if top else 2) + (0 if left else 1)
        if quad != selected_quad:
            return False

        raw_quad = motif[(0 if top else n) : (n if top else 2 * n), (0 if left else n) : (n if left else 2 * n)]
        if not np.any(raw_quad == BLACK):
            return False

        repaired_quads = [
            restored[:n, :n],
            restored[:n, n:],
            restored[n:, :n],
            restored[n:, n:],
        ]
        if not np.array_equal(repaired_quads[1], np.rot90(repaired_quads[0], -1)):
            return False
        if not np.array_equal(repaired_quads[2], np.rot90(repaired_quads[0], 1)):
            return False
        if not np.array_equal(repaired_quads[3], np.rot90(repaired_quads[0], 2)):
            return False

        if np.array_equal(repaired_quads[0], np.rot90(repaired_quads[0], 1)):
            return False
        if np.array_equal(repaired_quads[0], np.rot90(repaired_quads[0], 2)):
            return False

        return np.array_equal(main(grid), expected)
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        n = random.randint(3, 6)
        motif_side = 2 * n

        h = random.randint(motif_side + 2, min(30, motif_side + 10))
        w = random.randint(motif_side + 2, min(30, motif_side + 10))
        if h <= motif_side or w <= motif_side:
            continue

        r0 = random.randint(1, h - motif_side - 1)
        c0 = random.randint(1, w - motif_side - 1)
        motif_box = (r0, c0, r0 + motif_side - 1, c0 + motif_side - 1)

        tile = _make_tile(n)
        selected_quad = random.randint(0, 3)
        motif = _occlude_motif(_quadrant_tiles(tile), selected_quad)

        marker_rows, marker_cols = _corner_ranges((h, w), motif_box, selected_quad)
        if not marker_rows or not marker_cols:
            continue

        used = set(map(int, np.unique(tile)))
        marker_color_choices = [c for c in TILE_COLORS if c not in used]
        marker_color = random.choice(marker_color_choices if marker_color_choices else TILE_COLORS)
        marker = (random.choice(list(marker_rows)), random.choice(list(marker_cols)))

        grid = np.full((h, w), BG, dtype=int)
        grid[r0 : r0 + motif_side, c0 : c0 + motif_side] = motif
        grid[marker] = marker_color

        expected = _expected_output(tile, selected_quad)
        if _validate_instance(grid, expected, selected_quad):
            return grid

    raise RuntimeError("failed to generate valid input")
