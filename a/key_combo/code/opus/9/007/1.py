import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
YELLOW = 4
GREY = 5

SWAP_A = {GREEN: YELLOW, YELLOW: GREEN}
SWAP_B = {BLUE: GREY, GREY: BLUE}

GEN_RETRY_LIMIT = 400
_GEN_COUNTER = 0


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(comp, dtype=int))
    return comps


def _apply_swap(patch: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = patch.copy()
    for src, dst in mapping.items():
        out[patch == src] = dst
    return out


def _orient_patch(patch: np.ndarray, quadrant: tuple[int, int]) -> np.ndarray:
    out = patch
    if quadrant[0] == 1:
        out = np.flipud(out)
    if quadrant[1] == 1:
        out = np.fliplr(out)
    return out


def _place_in_quadrant(patch: np.ndarray, half_shape: tuple[int, int], quadrant: tuple[int, int]) -> np.ndarray:
    hh, hw = half_shape
    ph, pw = patch.shape
    out = np.zeros((hh, hw), dtype=int)
    r0 = 0 if quadrant[0] == 0 else hh - ph
    c0 = 0 if quadrant[1] == 0 else hw - pw
    out[r0 : r0 + ph, c0 : c0 + pw] = patch
    return out


def _build_output_from_source(src_quad: np.ndarray, src_pos: tuple[int, int]) -> np.ndarray:
    hh, hw = src_quad.shape
    out = np.zeros((2 * hh, 2 * hw), dtype=int)

    horiz_pos = (src_pos[0], 1 - src_pos[1])
    vert_pos = (1 - src_pos[0], src_pos[1])
    diag_pos = (1 - src_pos[0], 1 - src_pos[1])

    quads = {
        src_pos: src_quad,
        horiz_pos: _apply_swap(np.fliplr(src_quad), SWAP_A),
        vert_pos: _apply_swap(np.flipud(src_quad), SWAP_B),
        diag_pos: _apply_swap(_apply_swap(np.flipud(np.fliplr(src_quad)), SWAP_A), SWAP_B),
    }

    for (qr, qc), patch in quads.items():
        r0 = qr * hh
        c0 = qc * hw
        out[r0 : r0 + hh, c0 : c0 + hw] = patch
    return out


def _extract_source_quadrant(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = grid.shape
    assert h % 2 == 0 and w % 2 == 0, "grid must split into four equal quadrants"
    hh, hw = h // 2, w // 2
    quads = {
        (0, 0): grid[:hh, :hw],
        (0, 1): grid[:hh, hw:],
        (1, 0): grid[hh:, :hw],
        (1, 1): grid[hh:, hw:],
    }
    filled = [pos for pos, patch in quads.items() if np.any(patch != BLACK)]
    assert len(filled) == 1, "expected exactly one non-black quadrant"
    return quads[filled[0]].copy(), filled[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror the unique sprite quadrant into a 2x2 tiling, recoloring by reflection axis.

    Input structure:
    - The grid has shape 2H x 2W with black background.
    - Exactly one quadrant contains a single 4-connected multicolor sprite; the other three quadrants are empty.
    - The sprite's bounding box includes that quadrant's outer corner pixel.

    Rule:
    - Treat the filled quadrant as the original tile and keep it unchanged in place.
    - In the quadrant sharing a vertical edge with the original, place the horizontal reflection and swap green(3) with yellow(4).
    - In the quadrant sharing a horizontal edge with the original, place the vertical reflection and swap blue(1) with grey(5).
    - In the diagonal quadrant, place the double reflection and apply the green/yellow swap followed by the blue/grey swap.

    Key insight:
    - Geometry and recoloring are coupled: which color swap to apply is determined by which reflection axis created that copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert BLACK in set(map(int, np.unique(g))), "background must be black"

    src_quad, src_pos = _extract_source_quadrant(g)
    return _build_output_from_source(src_quad, src_pos)


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray | None:
    for _ in range(120):
        mask = np.zeros((height, width), dtype=bool)
        mask[0, 0] = True
        cells = {(0, 0)}
        frontier = {(1, 0), (0, 1)}
        frontier = {(r, c) for r, c in frontier if 0 <= r < height and 0 <= c < width}

        while len(cells) < area and frontier:
            choices = sorted(
                frontier,
                key=lambda rc: (rc[0] + rc[1], random.random()),
                reverse=True,
            )
            take_count = min(len(choices), 3)
            r, c = random.choice(choices[:take_count])
            frontier.remove((r, c))
            cells.add((r, c))
            mask[r, c] = True
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        if len(cells) != area:
            continue
        if not np.any(mask[-1, :]) or not np.any(mask[:, -1]):
            continue
        if mask.all():
            continue
        comps = _components_mask(mask)
        if len(comps) != 1:
            continue
        return mask
    return None


def _choose_palette(mode: int) -> list[int]:
    if mode % 3 == 0:
        return [BLUE, GREEN, YELLOW, GREY]
    if mode % 3 == 1:
        return [BLUE, GREEN, YELLOW]
    return [GREEN, YELLOW, GREY]


def _colorize_mask(mask: np.ndarray, palette: list[int]) -> np.ndarray | None:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    random.shuffle(coords)
    if len(coords) < len(palette):
        return None

    patch = np.zeros(mask.shape, dtype=int)
    for idx, color in enumerate(palette):
        r, c = coords[idx]
        patch[r, c] = color
    for r, c in coords[len(palette) :]:
        patch[r, c] = random.choice(palette)
    return patch


def _sprite_is_valid(patch: np.ndarray) -> bool:
    mask = patch != BLACK
    if np.count_nonzero(mask) < 8 or np.count_nonzero(mask) > 20:
        return False
    if len(_components_mask(mask)) != 1:
        return False

    colors = set(map(int, patch[mask]))
    if GREEN not in colors or YELLOW not in colors:
        return False
    if BLUE not in colors and GREY not in colors:
        return False
    if len(colors) < 3:
        return False

    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_coords(coords)
    if (r0, c0) != (0, 0):
        return False
    if r1 != patch.shape[0] - 1 or c1 != patch.shape[1] - 1:
        return False
    if patch[0, 0] == BLACK:
        return False
    if np.array_equal(patch, np.flipud(patch)) or np.array_equal(patch, np.fliplr(patch)):
        return False

    out = _build_output_from_source(_place_in_quadrant(patch, patch.shape, (0, 0)), (0, 0))
    hh, hw = patch.shape
    quadrants = [
        out[:hh, :hw],
        out[:hh, hw:],
        out[hh:, :hw],
        out[hh:, hw:],
    ]
    sigs = {q.tobytes() for q in quadrants}
    return len(sigs) == 4


def _make_canonical_patch(max_h: int, max_w: int, mode: int) -> np.ndarray | None:
    for _ in range(200):
        box_h = random.randint(3, max_h)
        box_w = random.randint(3, max_w)
        max_area = min(20, box_h * box_w - 1)
        if max_area < 8:
            continue
        min_area = 8
        area = random.randint(min_area, max_area)
        mask = _random_connected_mask(box_h, box_w, area)
        if mask is None:
            continue
        patch = _colorize_mask(mask, _choose_palette(mode))
        if patch is None or not _sprite_is_valid(patch):
            continue
        return patch
    return None


def _quadrant_corner_is_touched(patch: np.ndarray, quadrant: tuple[int, int]) -> bool:
    mask = patch != BLACK
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_coords(coords)
    corners = {
        (0, 0): (r0, c0),
        (0, 1): (r0, c1),
        (1, 0): (r1, c0),
        (1, 1): (r1, c1),
    }
    h, w = patch.shape
    targets = {
        (0, 0): (0, 0),
        (0, 1): (0, w - 1),
        (1, 0): (h - 1, 0),
        (1, 1): (h - 1, w - 1),
    }
    return corners[quadrant] == targets[quadrant]


def _validate_generated_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if grid.shape[0] % 2 or grid.shape[1] % 2:
        return False

    src_quad, src_pos = _extract_source_quadrant(grid)
    mask = src_quad != BLACK
    comps = _components_mask(mask)
    if len(comps) != 1:
        return False
    if not (8 <= int(mask.sum()) <= 20):
        return False
    if not _quadrant_corner_is_touched(src_quad, src_pos):
        return False

    colors = set(map(int, src_quad[mask]))
    if GREEN not in colors or YELLOW not in colors or (BLUE not in colors and GREY not in colors):
        return False

    out = main(grid)
    hh, hw = src_quad.shape
    actual = {
        (0, 0): out[:hh, :hw],
        (0, 1): out[:hh, hw:],
        (1, 0): out[hh:, :hw],
        (1, 1): out[hh:, hw:],
    }
    expected = {
        src_pos: src_quad,
        (src_pos[0], 1 - src_pos[1]): _apply_swap(np.fliplr(src_quad), SWAP_A),
        (1 - src_pos[0], src_pos[1]): _apply_swap(np.flipud(src_quad), SWAP_B),
        (1 - src_pos[0], 1 - src_pos[1]): _apply_swap(
            _apply_swap(np.flipud(np.fliplr(src_quad)), SWAP_A),
            SWAP_B,
        ),
    }
    for pos, patch in expected.items():
        if not np.array_equal(actual[pos], patch):
            return False
    return True


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    for _ in range(GEN_RETRY_LIMIT):
        mode = _GEN_COUNTER
        _GEN_COUNTER += 1

        half_h = random.randint(5, 8)
        half_w = random.randint(5, 8)
        quadrant = divmod(mode % 4, 2)

        canonical = _make_canonical_patch(half_h, half_w, mode)
        if canonical is None:
            continue

        oriented = _orient_patch(canonical, quadrant)
        quad = _place_in_quadrant(oriented, (half_h, half_w), quadrant)
        grid = np.zeros((2 * half_h, 2 * half_w), dtype=int)

        r0 = quadrant[0] * half_h
        c0 = quadrant[1] * half_w
        grid[r0 : r0 + half_h, c0 : c0 + half_w] = quad

        if _validate_generated_input(grid):
            return grid

    raise RuntimeError("failed to generate a valid input for the reflection-tiling recolor task")
