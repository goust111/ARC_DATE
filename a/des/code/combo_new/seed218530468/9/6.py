import random
from collections import deque

import numpy as np


BG = 0
ARC_MAX = 30
MAX_GENERATION_ATTEMPTS = 400
SIGN_COLORS = (1, 3, 4, 6, 7, 8, 9)
TRAVELER_COLORS = (2, 5)

UP = "up"
RIGHT = "right"
DOWN = "down"
LEFT = "left"
DIR_ORDER = (UP, RIGHT, DOWN, LEFT)
DIR_STEP = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}
NEXT_DIR_OPTIONS = {
    UP: (UP, LEFT, RIGHT),
    RIGHT: (RIGHT, UP, DOWN),
    DOWN: (DOWN, LEFT, RIGHT),
    LEFT: (LEFT, UP, DOWN),
}
ARROW_OFFSETS = {
    UP: ((0, 0), (1, -1), (1, 1)),
    RIGHT: ((0, 0), (-1, -1), (1, -1)),
    DOWN: ((0, 0), (-1, -1), (-1, 1)),
    LEFT: ((0, 0), (-1, 1), (1, 1)),
}


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = rr + dr
                        nc = cc + dc
                        if not (0 <= nr < h and 0 <= nc < w):
                            continue
                        if seen[nr, nc] or int(grid[nr, nc]) != color:
                            continue
                        seen[nr, nc] = True
                        q.append((nr, nc))

            coords = np.array(cells, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return components


def _is_traveler(comp: dict) -> bool:
    if comp["area"] != 4:
        return False
    r0, c0, r1, c1 = comp["bbox"]
    return (r1 - r0, c1 - c0) == (1, 1)


def _decode_arrow(comp: dict) -> tuple[tuple[int, int], str] | None:
    if comp["area"] != 3:
        return None

    cells = [tuple(map(int, rc)) for rc in comp["coords"]]
    degrees: dict[tuple[int, int], int] = {}
    for r, c in cells:
        deg = 0
        for rr, cc in cells:
            if (rr, cc) == (r, c):
                continue
            if max(abs(rr - r), abs(cc - c)) == 1:
                deg += 1
        degrees[(r, c)] = deg

    tips = [cell for cell, deg in degrees.items() if deg == 2]
    wings = [cell for cell, deg in degrees.items() if deg == 1]
    if len(tips) != 1 or len(wings) != 2:
        return None

    tip = tips[0]
    dvr = 2 * tip[0] - sum(r for r, _c in wings)
    dvc = 2 * tip[1] - sum(c for _r, c in wings)
    dir_map = {
        (-2, 0): UP,
        (0, 2): RIGHT,
        (2, 0): DOWN,
        (0, -2): LEFT,
    }
    if (dvr, dvc) not in dir_map:
        return None
    return tip, dir_map[(dvr, dvc)]


def _traveler_center(comp: dict) -> tuple[int, int]:
    r0, c0, _r1, _c1 = comp["bbox"]
    return int(r0 + 1), int(c0 + 1)


def _sign_cells(tip: tuple[int, int], direction: str) -> set[tuple[int, int]]:
    tr, tc = tip
    return {(tr + dr, tc + dc) for dr, dc in ARROW_OFFSETS[direction]}


def _search_from(
    origin: tuple[int, int], direction: str, signs: dict[tuple[int, int], dict]
) -> dict | None:
    orow, ocol = origin
    best: tuple[int, dict] | None = None
    for tip, sign in signs.items():
        row, col = tip
        if direction == UP and col == ocol and row < orow:
            dist = orow - row
        elif direction == DOWN and col == ocol and row > orow:
            dist = row - orow
        elif direction == LEFT and row == orow and col < ocol:
            dist = ocol - col
        elif direction == RIGHT and row == orow and col > ocol:
            dist = col - ocol
        else:
            continue
        if best is None or dist < best[0]:
            best = (dist, sign)
    return None if best is None else best[1]


def _initial_sign(
    center: tuple[int, int], signs: dict[tuple[int, int], dict]
) -> dict:
    visible = []
    for direction in DIR_ORDER:
        sign = _search_from(center, direction, signs)
        if sign is not None:
            visible.append(sign)
    assert len(visible) == 1, "traveler must initially see exactly one sign direction"
    return visible[0]


def _trace_chain(
    traveler_center: tuple[int, int], signs: dict[tuple[int, int], dict]
) -> list[dict]:
    remaining = dict(signs)
    current = traveler_center
    visited: list[dict] = []

    sign = _initial_sign(current, remaining)
    while sign is not None:
        visited.append(sign)
        remaining.pop(sign["tip"])
        current = sign["tip"]
        sign = _search_from(current, sign["direction"], remaining)

    return visited


def _paint_nested_square(
    out: np.ndarray,
    center: tuple[int, int],
    core_color: int,
    shell_colors: list[int],
) -> None:
    colors = [int(core_color)] + [int(color) for color in shell_colors]
    size = 2 + 2 * len(shell_colors)
    top = center[0] - size // 2
    left = center[1] - size // 2
    assert top >= 0 and left >= 0
    assert top + size <= out.shape[0] and left + size <= out.shape[1]

    for inset, color in enumerate(reversed(colors)):
        span = size - 2 * inset
        out[top + inset : top + inset + span, left + inset : left + inset + span] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the directional sign chain and wrap the traveler with one colored ring per visited sign.

    Input structure:
    - Black background with exactly one solid 2x2 monochrome traveler block.
    - Several monochrome 3-pixel arrowhead signposts appear elsewhere; each signpost has one tip pixel and two diagonal wing pixels that indicate one cardinal direction.
    - The traveler center is aligned with exactly one initially visible signpost tip, and later signposts form a ray-by-ray chain.

    Rule:
    - Decode every 3-pixel signpost into its tip position and pointing direction.
    - Starting from the traveler center, move to the nearest visible signpost tip; after each visit remove that signpost and change the search direction to the signpost's own arrow direction.
    - Each visited signpost adds one new outer ring in the signpost color around the traveler, so a 2x2 core becomes a 4x4 square, then 6x6, and so on.
    - Stop when the current arrow direction sees no further signpost; erase the original traveler location and visited signposts, then paint the final concentric square at the last visited tip.

    Key insight:
    - The signposts do not mark final destinations directly: each one is both a teleport anchor and a routing instruction for the next hop, while its color records the next outer shell of the growing nested square.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _find_components(g, bg=BG)
    travelers = [comp for comp in components if _is_traveler(comp)]
    assert len(travelers) == 1, "expected exactly one 2x2 traveler"

    traveler = travelers[0]
    traveler_center = _traveler_center(traveler)
    traveler_color = int(traveler["color"])

    signs: dict[tuple[int, int], dict] = {}
    for comp in components:
        if comp is traveler:
            continue
        decoded = _decode_arrow(comp)
        assert decoded is not None, "every non-traveler object must be a 3-pixel arrowhead"
        tip, direction = decoded
        signs[tip] = {
            "color": int(comp["color"]),
            "tip": tip,
            "direction": direction,
            "coords": [tuple(map(int, rc)) for rc in comp["coords"]],
        }

    visited = _trace_chain(traveler_center, signs)
    assert visited, "expected at least one reachable signpost"

    out = g.copy()
    for r, c in traveler["coords"]:
        out[int(r), int(c)] = BG
    for sign in visited:
        for r, c in sign["coords"]:
            out[r, c] = BG

    final_center = visited[-1]["tip"]
    shell_colors = [sign["color"] for sign in visited]
    _paint_nested_square(out, final_center, traveler_color, shell_colors)
    return out


def _relative_walk(
    segment_dirs: list[str], lengths: list[int]
) -> list[tuple[int, int]]:
    points = [(0, 0)]
    for direction, length in zip(segment_dirs, lengths):
        dr, dc = DIR_STEP[direction]
        pr, pc = points[-1]
        points.append((pr + dr * length, pc + dc * length))
    return points


def _path_points_ok(points: list[tuple[int, int]]) -> bool:
    if len(set(points)) != len(points):
        return False

    start = points[0]
    aligned_with_start = 0
    for row, col in points[1:]:
        if row == start[0] or col == start[1]:
            aligned_with_start += 1
    return aligned_with_start == 1


def _content_bounds(
    points: list[tuple[int, int]],
    sign_dirs: list[str],
    chain_len: int,
) -> tuple[int, int, int, int]:
    cells: set[tuple[int, int]] = set()
    start = points[0]
    cells.update(
        {
            (start[0] - 1, start[1] - 1),
            (start[0] - 1, start[1]),
            (start[0], start[1] - 1),
            (start[0], start[1]),
        }
    )

    for tip, direction in zip(points[1:], sign_dirs):
        cells.update(_sign_cells(tip, direction))

    final_size = 2 + 2 * chain_len
    last = points[-1]
    top = last[0] - final_size // 2
    left = last[1] - final_size // 2
    for r in range(top, top + final_size):
        for c in range(left, left + final_size):
            cells.add((r, c))

    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _embed_margin(extra: int) -> tuple[int, int]:
    if extra == 0:
        return 0, 0
    if extra == 1:
        return random.choice([(0, 1), (1, 0)])
    lead = random.randint(1, extra - 1)
    return lead, extra - lead


def _build_expected(
    grid: np.ndarray,
    traveler_cells: list[tuple[int, int]],
    signs: list[dict],
    traveler_color: int,
) -> np.ndarray:
    out = grid.copy()
    for r, c in traveler_cells:
        out[r, c] = BG
    for sign in signs:
        for r, c in sign["coords"]:
            out[r, c] = BG
    _paint_nested_square(
        out,
        signs[-1]["tip"],
        traveler_color,
        [sign["color"] for sign in signs],
    )
    return out


def _gen_attempt() -> np.ndarray | None:
    chain_len = random.randint(1, 5)
    traveler_color = random.choice(TRAVELER_COLORS)
    sign_colors = random.sample(list(SIGN_COLORS), chain_len)

    segment_dirs = [random.choice(DIR_ORDER)]
    for _ in range(chain_len - 1):
        segment_dirs.append(random.choice(NEXT_DIR_OPTIONS[segment_dirs[-1]]))
    sign_dirs = segment_dirs[1:] + [random.choice(DIR_ORDER)]
    lengths = [random.randint(3, 5) for _ in range(chain_len)]

    points = _relative_walk(segment_dirs, lengths)
    if not _path_points_ok(points):
        return None

    arrow_cells = set()
    for tip, direction in zip(points[1:], sign_dirs):
        cells = _sign_cells(tip, direction)
        if arrow_cells & cells:
            return None
        arrow_cells |= cells

    traveler_cells_rel = [
        (points[0][0] - 1, points[0][1] - 1),
        (points[0][0] - 1, points[0][1]),
        (points[0][0], points[0][1] - 1),
        (points[0][0], points[0][1]),
    ]
    if any(cell in arrow_cells for cell in traveler_cells_rel):
        return None

    min_r, min_c, max_r, max_c = _content_bounds(points, sign_dirs, chain_len)
    content_h = max_r - min_r + 1
    content_w = max_c - min_c + 1
    if content_h > ARC_MAX or content_w > ARC_MAX:
        return None

    extra_h = random.randint(0, ARC_MAX - content_h)
    extra_w = random.randint(0, ARC_MAX - content_w)
    top_pad, bottom_pad = _embed_margin(extra_h)
    left_pad, right_pad = _embed_margin(extra_w)
    height = content_h + top_pad + bottom_pad
    width = content_w + left_pad + right_pad

    row_shift = top_pad - min_r
    col_shift = left_pad - min_c
    shifted_points = [(r + row_shift, c + col_shift) for r, c in points]

    grid = np.full((height, width), BG, dtype=int)
    traveler_cells = [
        (shifted_points[0][0] - 1, shifted_points[0][1] - 1),
        (shifted_points[0][0] - 1, shifted_points[0][1]),
        (shifted_points[0][0], shifted_points[0][1] - 1),
        (shifted_points[0][0], shifted_points[0][1]),
    ]
    for r, c in traveler_cells:
        grid[r, c] = traveler_color

    signs: list[dict] = []
    for idx, (tip, direction) in enumerate(zip(shifted_points[1:], sign_dirs)):
        cells = sorted(_sign_cells(tip, direction))
        color = sign_colors[idx]
        for r, c in cells:
            if grid[r, c] != BG:
                return None
            grid[r, c] = color
        signs.append(
            {
                "color": color,
                "tip": tip,
                "direction": direction,
                "coords": cells,
            }
        )

    expected = _build_expected(grid, traveler_cells, signs, traveler_color)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(
        f"failed to generate a valid directional-chain input after {MAX_GENERATION_ATTEMPTS} attempts"
    )
