import random
from itertools import combinations

import numpy as np


BG = 0
MASK_COLOR = 3
TEMPLATE_COLOR = 5
PATTERN_COLORS = (1, 2, 4, 6, 7, 8, 9)
GRID_SIZE_RANGE = (11, 15)
MASK_HEIGHT_RANGE = (3, 6)
MASK_WIDTH_RANGE = (3, 6)
MAX_GENERATION_ATTEMPTS = 500


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def d4_orbit(r: int, c: int, n: int) -> tuple[tuple[int, int], ...]:
    coords = {
        (r, c),
        (c, n - 1 - r),
        (n - 1 - r, n - 1 - c),
        (n - 1 - c, r),
        (r, n - 1 - c),
        (n - 1 - r, c),
        (c, r),
        (n - 1 - c, n - 1 - r),
    }
    return tuple(sorted(coords))


def all_d4_orbits(n: int) -> list[tuple[tuple[int, int], ...]]:
    seen = np.zeros((n, n), dtype=bool)
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(n):
        for c in range(n):
            if seen[r, c]:
                continue
            orbit = d4_orbit(r, c, n)
            for rr, cc in orbit:
                seen[rr, cc] = True
            orbits.append(orbit)
    return orbits


def mask_connected(mask: np.ndarray) -> bool:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not cells:
        return False
    seen = {cells[0]}
    stack = [cells[0]]
    all_cells = set(cells)
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in all_cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def build_template_shapes() -> tuple[np.ndarray, ...]:
    shapes: list[np.ndarray] = []
    for count in range(4, 7):
        for idxs in combinations(range(9), count):
            mask = np.zeros((3, 3), dtype=bool)
            for idx in idxs:
                mask[idx // 3, idx % 3] = True
            if not mask_connected(mask):
                continue
            box = bbox(mask)
            if box != (0, 0, 2, 2):
                continue
            shapes.append(mask)
    return tuple(shapes)


TEMPLATE_SHAPES = build_template_shapes()


def render_output(source: np.ndarray, template_mask: np.ndarray) -> np.ndarray:
    sh, sw = source.shape
    th, tw = template_mask.shape
    out = np.full((sh * th, sw * tw), BG, dtype=int)
    for r in range(sh):
        for c in range(sw):
            color = int(source[r, c])
            if color == BG:
                continue
            block = out[r * th : (r + 1) * th, c * tw : (c + 1) * tw]
            block[template_mask] = color
    return out


def boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def expanded_box(r0: int, c0: int, h: int, w: int, pad: int) -> tuple[int, int, int, int]:
    return r0 - pad, c0 - pad, r0 + h - 1 + pad, c0 + w - 1 + pad


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the masked symmetric crop, then stamp a gray template block for every recovered colored cell.

    Input structure:
    - The input is a square grid whose main pattern is D4-symmetric on a black background.
    - One solid green rectangle masks part of that symmetric pattern.
    - A separate gray template shape sits elsewhere and does not belong to the symmetric motif.

    Rule:
    - Find the green rectangle and reconstruct each masked cell from its visible D4-equivalent copies, ignoring green mask cells and gray template cells.
    - Crop the recovered rectangle to obtain the source grid, and crop the gray shape to its bounding box to obtain the stamp template.
    - Create an output tiled by source-grid positions; each non-black source cell recolors the gray template to that source color in the corresponding output block, while black source cells leave their blocks black.

    Key insight:
    - The green rectangle hides a symmetric patch, and that recovered patch is not the final answer itself: it is a color-coded control grid that expands into repeated copies of the gray template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    n, m = g.shape
    assert n == m

    mask = g == MASK_COLOR
    mask_box = bbox(mask)
    assert mask_box is not None
    r0, c0, r1, c1 = mask_box
    assert bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1]))

    template_pixels = g == TEMPLATE_COLOR
    template_box = bbox(template_pixels)
    assert template_box is not None
    template = crop_bbox(g, template_box)
    template_mask = template == TEMPLATE_COLOR
    assert bool(template_mask.any())

    source = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            visible: list[int] = []
            for rr, cc in d4_orbit(r, c, n):
                color = int(g[rr, cc])
                if color in (MASK_COLOR, TEMPLATE_COLOR):
                    continue
                visible.append(color)
            assert visible
            colors = set(visible)
            assert len(colors) == 1
            source[r - r0, c - c0] = colors.pop()

    return render_output(source, template_mask)


def sample_side_position(n: int) -> tuple[int, int]:
    side = random.choice(("top", "bottom", "left", "right"))
    if side == "top":
        return 0, random.randint(0, n - 3)
    if side == "bottom":
        return n - 3, random.randint(0, n - 3)
    if side == "left":
        return random.randint(0, n - 3), 0
    return random.randint(0, n - 3), n - 3


def forced_black_cells(
    n: int, template_shape: np.ndarray, top: int, left: int
) -> set[tuple[int, int]]:
    forced: set[tuple[int, int]] = set()
    local_gray = {tuple(map(int, rc)) for rc in np.argwhere(template_shape)}
    local_black_zone = {(r, c) for r in range(3) for c in range(3)}

    for r, c in local_black_zone:
        for cell in d4_orbit(top + r, left + c, n):
            forced.add(cell)

    for lr, lc in local_gray:
        for nr, nc in neighbors4(top + lr, left + lc):
            if not (0 <= nr < n and 0 <= nc < n):
                continue
            if (nr - top, nc - left) in local_gray:
                continue
            for cell in d4_orbit(nr, nc, n):
                forced.add(cell)

    return forced


def make_symmetric_pattern(n: int, forced_black: set[tuple[int, int]]) -> np.ndarray:
    palette = random.sample(PATTERN_COLORS, random.randint(2, 3))
    orbits = all_d4_orbits(n)
    blocked: list[int] = []
    free: list[int] = []
    for idx, orbit in enumerate(orbits):
        if any(cell in forced_black for cell in orbit):
            blocked.append(idx)
        else:
            free.append(idx)

    if len(free) < len(palette) + 2:
        raise ValueError("not enough free orbits")

    min_colored = min(len(free), len(palette) + 3)
    max_colored = max(min_colored, min(len(free), max(len(palette) + 6, len(free) // 2)))
    colored_count = random.randint(min_colored, max_colored)
    colored_orbits = random.sample(free, colored_count)
    forced_palette = palette[:]
    random.shuffle(forced_palette)

    grid = np.zeros((n, n), dtype=int)
    for idx, orbit in enumerate(orbits):
        if idx in blocked:
            color = BG
        elif idx in colored_orbits:
            color = forced_palette.pop() if forced_palette else random.choice(palette)
        else:
            color = BG
        for r, c in orbit:
            grid[r, c] = color
    return grid


def candidate_mask_sizes() -> list[tuple[int, int]]:
    sizes = [
        (h, w)
        for h in range(MASK_HEIGHT_RANGE[0], MASK_HEIGHT_RANGE[1] + 1)
        for w in range(MASK_WIDTH_RANGE[0], MASK_WIDTH_RANGE[1] + 1)
        if 3 * h <= 30 and 3 * w <= 30
    ]
    random.shuffle(sizes)
    return sizes


def valid_source_patch(source: np.ndarray) -> bool:
    non_bg_colors = set(map(int, source.flatten())) - {BG}
    return len(non_bg_colors) >= 2 and int(np.count_nonzero(source == BG)) >= 1 and int(np.count_nonzero(source != BG)) >= 3


def build_attempt() -> np.ndarray | None:
    n = random.randint(*GRID_SIZE_RANGE)
    template_shape = random.choice(TEMPLATE_SHAPES)
    top, left = sample_side_position(n)
    template_box = (top, left, top + 2, left + 2)

    try:
        full = make_symmetric_pattern(n, forced_black_cells(n, template_shape, top, left))
    except ValueError:
        return None

    expected_template_bbox = full[top : top + 3, left : left + 3]
    if np.any(expected_template_bbox != BG):
        return None

    for h, w in candidate_mask_sizes():
        positions = [(r0, c0) for r0 in range(n - h + 1) for c0 in range(n - w + 1)]
        random.shuffle(positions)
        for r0, c0 in positions:
            if boxes_overlap(expanded_box(r0, c0, h, w, 1), template_box):
                continue
            source = full[r0 : r0 + h, c0 : c0 + w]
            if not valid_source_patch(source):
                continue

            inp = full.copy()
            template_view = inp[top : top + 3, left : left + 3]
            template_view[template_shape] = TEMPLATE_COLOR
            inp[r0 : r0 + h, c0 : c0 + w] = MASK_COLOR

            expected = render_output(source, template_shape)
            try:
                solved = main(inp)
            except AssertionError:
                continue
            if np.array_equal(solved, expected):
                return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
