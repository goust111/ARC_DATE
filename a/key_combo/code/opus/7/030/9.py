import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
PINK = 6
MAROON = 9

NEAR = "near"
MIDDLE = "middle"
OFFSET = "offset"

GLYPH_PATTERNS = {
    NEAR: np.array([[RED], [RED]], dtype=int),
    MIDDLE: np.array([[BLACK, RED], [RED, RED]], dtype=int),
    OFFSET: np.array([[RED, BLACK], [RED, RED]], dtype=int),
}

ALL_SIDES = ("left", "right", "above", "below")
HORIZONTAL_SIDES = ("left", "right")
VERTICAL_SIDES = ("above", "below")
MAX_ATTEMPTS = 400


def _decode_glyphs(strip: np.ndarray) -> list[str]:
    """Read the 2-row red glyph strip from left to right."""
    assert strip.shape[0] == 2
    assert set(map(int, np.unique(strip))) <= {BLACK, RED}

    glyphs: list[str] = []
    c = 0
    w = int(strip.shape[1])
    while c < w:
        if int(strip[0, c]) == BLACK and int(strip[1, c]) == BLACK:
            c += 1
            continue

        if c + 1 < w:
            patch = strip[:, c : c + 2]
            if np.array_equal(patch, GLYPH_PATTERNS[MIDDLE]):
                glyphs.append(MIDDLE)
                c += 2
                continue
            if np.array_equal(patch, GLYPH_PATTERNS[OFFSET]):
                glyphs.append(OFFSET)
                c += 2
                continue

        patch = strip[:, c : c + 1]
        assert np.array_equal(patch, GLYPH_PATTERNS[NEAR])
        glyphs.append(NEAR)
        c += 1

    return glyphs


def _component(grid: np.ndarray, start: tuple[int, int], visited: np.ndarray) -> list[tuple[int, int]]:
    """Collect one 4-connected monochrome component."""
    h, w = grid.shape
    color = int(grid[start])
    stack = [start]
    visited[start] = True
    coords: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        coords.append((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not visited[rr, cc] and int(grid[rr, cc]) == color:
                visited[rr, cc] = True
                stack.append((rr, cc))
    return coords


def _marker_for_square(grid: np.ndarray, r0: int, c0: int) -> tuple[int, int, str]:
    """Find the unique maroon marker touching this 3x3 square."""
    h, w = grid.shape
    hits: list[tuple[int, int, str]] = []

    if c0 - 1 >= 0:
        for r in range(r0, r0 + 3):
            if int(grid[r, c0 - 1]) == MAROON:
                hits.append((r, c0 - 1, "left"))
    if c0 + 3 < w:
        for r in range(r0, r0 + 3):
            if int(grid[r, c0 + 3]) == MAROON:
                hits.append((r, c0 + 3, "right"))
    if r0 - 1 >= 0:
        for c in range(c0, c0 + 3):
            if int(grid[r0 - 1, c]) == MAROON:
                hits.append((r0 - 1, c, "above"))
    if r0 + 3 < h:
        for c in range(c0, c0 + 3):
            if int(grid[r0 + 3, c]) == MAROON:
                hits.append((r0 + 3, c, "below"))

    assert len(hits) == 1
    return hits[0]


def _find_pairs(grid: np.ndarray) -> list[dict]:
    """Detect all valid marker-square pairs in the upper area."""
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    pairs: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in (GREY, PINK) or visited[r, c]:
                continue

            coords = _component(grid, (r, c), visited)
            rows = [pt[0] for pt in coords]
            cols = [pt[1] for pt in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            assert len(coords) == 9
            assert r1 - r0 == 2 and c1 - c0 == 2
            for rr in range(r0, r0 + 3):
                for cc in range(c0, c0 + 3):
                    assert int(grid[rr, cc]) == color

            mr, mc, side = _marker_for_square(grid, r0, c0)
            pairs.append(
                {
                    "color": color,
                    "r0": r0,
                    "c0": c0,
                    "marker": (mr, mc),
                    "side": side,
                    "center_col": c0 + 1,
                }
            )

    pairs.sort(key=lambda item: (item["center_col"], item["r0"]))
    centers = [int(pair["center_col"]) for pair in pairs]
    assert len(set(centers)) == len(centers)
    return pairs


def _target_cell(pair: dict, mode: str) -> tuple[int, int]:
    """Compute the projected maroon target cell inside one square."""
    r0 = int(pair["r0"])
    c0 = int(pair["c0"])
    mr, mc = pair["marker"]
    side = str(pair["side"])

    if side == "left":
        if mode == NEAR:
            return mr, c0
        if mode == MIDDLE:
            return mr, c0 + 1
        return mr, c0 + 2
    if side == "right":
        if mode == NEAR:
            return mr, c0 + 2
        if mode == MIDDLE:
            return mr, c0 + 1
        return mr, c0
    if side == "above":
        if mode == NEAR:
            return r0, mc
        if mode == MIDDLE:
            return r0 + 1, mc
        return r0 + 2, mc

    assert side == "below"
    if mode == NEAR:
        return r0 + 2, mc
    if mode == MIDDLE:
        return r0 + 1, mc
    return r0, mc


def main(grid: np.ndarray) -> np.ndarray:
    """Decode bottom glyphs into projection modes, apply them to adjacent marker-square pairs, and crop away the glyph strip.

    Input structure
    - The background is black, and the bottom two rows contain a left-to-right strip of red glyphs separated by black gaps.
    - Above the strip are 3 or 4 disjoint objects, each made of one maroon marker pixel touching one side of a solid 3x3 grey or pink square.

    Rule
    - Decode a 1-column vertical glyph as near-edge projection, a left-gap elbow as centerline projection, and a right-gap elbow as far-edge offset projection.
    - Sort the marker-square pairs by square center column, pair them with the decoded glyphs in that order, and stamp maroon into the corresponding near, middle, or opposite row/column inside each square.
    - After stamping, erase grey squares to black while keeping their source markers, keep pink squares but erase their source markers, and finally remove the bottom glyph strip from the output.

    Key insight
    - The square color no longer chooses the projection depth; instead, the detached red glyph strip is a control sequence that must be decoded and aligned with the pairs by left-to-right order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 5
    assert not np.any(g[:-2] == RED)

    strip = g[-2:, :]
    glyphs = _decode_glyphs(strip)

    upper = g[:-2, :]
    assert set(map(int, np.unique(upper))) <= {BLACK, GREY, PINK, MAROON}
    pairs = _find_pairs(upper)
    assert 3 <= len(pairs) <= 4
    assert len(glyphs) == len(pairs)

    out = upper.copy()
    for pair, mode in zip(pairs, glyphs):
        r0 = int(pair["r0"])
        c0 = int(pair["c0"])
        color = int(pair["color"])
        mr, mc = pair["marker"]
        tr, tc = _target_cell(pair, mode)

        if color == GREY:
            out[r0 : r0 + 3, c0 : c0 + 3] = BLACK
        else:
            out[mr, mc] = BLACK
        out[tr, tc] = MAROON

    return out


def _pair_bbox(r0: int, c0: int, side: str) -> tuple[int, int, int, int]:
    """Inclusive bbox covering the square and its marker."""
    if side == "left":
        return r0, c0 - 1, r0 + 2, c0 + 2
    if side == "right":
        return r0, c0, r0 + 2, c0 + 3
    if side == "above":
        return r0 - 1, c0, r0 + 2, c0 + 2
    return r0, c0, r0 + 3, c0 + 2


def _rectangles_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    """Require a gap of at least `pad` cells between two inclusive rectangles."""
    return a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]


def _sample_modes(count: int) -> list[str]:
    modes = [NEAR, MIDDLE, OFFSET]
    if count == 4:
        modes.append(random.choice([NEAR, MIDDLE, OFFSET]))
    random.shuffle(modes)
    return modes


def _sample_colors(count: int) -> list[int]:
    colors = [GREY, PINK]
    while len(colors) < count:
        colors.append(random.choice([GREY, PINK]))
    random.shuffle(colors)
    return colors


def _sample_sides(count: int) -> list[str]:
    sides = [random.choice(HORIZONTAL_SIDES), random.choice(VERTICAL_SIDES)]
    while len(sides) < count:
        sides.append(random.choice(ALL_SIDES))
    random.shuffle(sides)
    return sides


def _glyph_strip(width: int, modes: list[str]) -> np.ndarray:
    strip = np.zeros((2, width), dtype=int)
    used_width = sum(GLYPH_PATTERNS[mode].shape[1] for mode in modes) + len(modes) - 1
    start = random.randint(0, width - used_width)
    c = start
    for i, mode in enumerate(modes):
        pattern = GLYPH_PATTERNS[mode]
        pw = pattern.shape[1]
        strip[:, c : c + pw] = pattern
        c += pw
        if i + 1 < len(modes):
            c += 1
    return strip


def _sample_centers(width: int, count: int) -> list[int] | None:
    min_center = 2
    max_center = width - 3
    gap = 6
    centers: list[int] = []
    lo = min_center
    for idx in range(count):
        remaining = count - idx - 1
        hi = max_center - gap * remaining
        if lo > hi:
            return None
        centers.append(random.randint(lo, hi))
        lo = centers[-1] + gap
    return centers


def _place_pairs(height: int, width: int, modes: list[str], colors: list[int], sides: list[str]) -> np.ndarray | None:
    upper_h = height - 2
    grid = np.zeros((height, width), dtype=int)
    centers = _sample_centers(width, len(modes))
    if centers is None:
        return None

    placed_boxes: list[tuple[int, int, int, int]] = []
    for center, color, side in zip(centers, colors, sides):
        c0 = center - 1
        if side == "above":
            rows = list(range(1, upper_h - 2))
        elif side == "below":
            rows = list(range(0, upper_h - 3))
        else:
            rows = list(range(0, upper_h - 2))
        random.shuffle(rows)

        placed = False
        for r0 in rows:
            bbox = _pair_bbox(r0, c0, side)
            if any(not _rectangles_separated(bbox, other, pad=1) for other in placed_boxes):
                continue

            grid[r0 : r0 + 3, c0 : c0 + 3] = color
            slot = random.randint(0, 2)
            if side == "left":
                mr, mc = r0 + slot, c0 - 1
            elif side == "right":
                mr, mc = r0 + slot, c0 + 3
            elif side == "above":
                mr, mc = r0 - 1, c0 + slot
            else:
                mr, mc = r0 + 3, c0 + slot
            grid[mr, mc] = MAROON
            placed_boxes.append(bbox)
            placed = True
            break

        if not placed:
            return None

    grid[-2:, :] = _glyph_strip(width, modes)
    return grid


def _valid_sample(grid: np.ndarray, modes: list[str], colors: list[int], sides: list[str]) -> bool:
    """Run structural checks so the generator only returns informative inputs."""
    out = main(grid)
    if out.shape != (grid.shape[0] - 2, grid.shape[1]):
        return False
    if np.array_equal(out, grid[:-2, :]):
        return False
    if set(map(int, np.unique(grid[-2:, :]))) - {BLACK, RED}:
        return False

    decoded = _decode_glyphs(grid[-2:, :])
    if decoded != modes:
        return False

    pairs = _find_pairs(grid[:-2, :])
    if len(pairs) != len(modes):
        return False
    if not any(color == GREY for color in colors):
        return False
    if not any(color == PINK for color in colors):
        return False
    if not any(side in HORIZONTAL_SIDES for side in sides):
        return False
    if not any(side in VERTICAL_SIDES for side in sides):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        count = random.choice([3, 4])
        modes = _sample_modes(count)
        colors = _sample_colors(count)
        sides = _sample_sides(count)

        strip_width = sum(GLYPH_PATTERNS[mode].shape[1] for mode in modes) + count - 1
        min_width = max(strip_width + 2, 6 * count + 4)
        width = random.randint(min_width, min(28, min_width + 4))
        height = random.randint(14, 20)

        grid = _place_pairs(height, width, modes, colors, sides)
        if grid is None:
            continue
        if _valid_sample(grid, modes, colors, sides):
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 9")
