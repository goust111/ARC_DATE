import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

DIRS = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}

REFLECT_BACKSLASH = {
    UP: LEFT,
    RIGHT: DOWN,
    DOWN: RIGHT,
    LEFT: UP,
}

REFLECT_SLASH = {
    UP: RIGHT,
    RIGHT: UP,
    DOWN: LEFT,
    LEFT: DOWN,
}

_RIGHT_ARROW = np.array(
    [
        [1, 0, 0],
        [1, 1, 1],
        [1, 0, 0],
    ],
    dtype=bool,
)

ARROW_PATCHES = {
    RIGHT: _RIGHT_ARROW,
    DOWN: np.rot90(_RIGHT_ARROW, -1),
    LEFT: np.rot90(_RIGHT_ARROW, 2),
    UP: np.rot90(_RIGHT_ARROW, 1),
}

TIP_OFFSETS = {
    RIGHT: (1, 2),
    DOWN: (2, 1),
    LEFT: (1, 0),
    UP: (0, 1),
}

ARROW_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 300
SCENARIO_ATTEMPTS = 120


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _same_color_components(grid: np.ndarray, skip_colors: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(grid[r0, c0])
            if color in skip_colors or seen[r0, c0]:
                continue

            q = deque([(r0, c0)])
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    return comps


def _arrow_cells_from_tip(tip: tuple[int, int], direction: int) -> list[tuple[int, int]]:
    patch = ARROW_PATCHES[direction]
    tr, tc = TIP_OFFSETS[direction]
    r0 = int(tip[0] - tr)
    c0 = int(tip[1] - tc)
    cells = []
    for dr, dc in np.argwhere(patch):
        cells.append((r0 + int(dr), c0 + int(dc)))
    return cells


def _paint_arrow(out: np.ndarray, tip: tuple[int, int], direction: int, color: int) -> None:
    for r, c in _arrow_cells_from_tip(tip, direction):
        out[int(r), int(c)] = int(color)


def _detect_arrows(grid: np.ndarray) -> list[dict]:
    arrows = []
    for comp in _same_color_components(grid, {BLACK, GREY}):
        r0, c0, r1, c1 = comp["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1] == int(comp["color"])
        matched = False
        for direction, tmpl in ARROW_PATCHES.items():
            if patch.shape == tmpl.shape and np.array_equal(patch, tmpl):
                tr, tc = TIP_OFFSETS[direction]
                tip = (int(r0 + tr), int(c0 + tc))
                arrows.append(
                    {
                        "color": int(comp["color"]),
                        "direction": int(direction),
                        "tip": tip,
                        "coords": {(int(r), int(c)) for r, c in comp["coords"]},
                    }
                )
                matched = True
                break
        assert matched, "every non-grey foreground component must be a 3x3 arrow"

    assert 1 <= len(arrows) <= 3
    arrows.sort(key=lambda item: (item["tip"][0], item["tip"][1], item["color"]))
    return arrows


def _detect_mirrors(grid: np.ndarray) -> dict[tuple[int, int], str]:
    grey = {(int(r), int(c)) for r, c in np.argwhere(grid == GREY)}
    assert 1 <= len(grey) <= 12, "expected a small set of mirror pixels"

    mirror_kind: dict[tuple[int, int], str] = {}
    for r, c in grey:
        has_backslash = (r - 1, c - 1) in grey or (r + 1, c + 1) in grey
        has_slash = (r - 1, c + 1) in grey or (r + 1, c - 1) in grey
        assert has_backslash ^ has_slash, "each grey pixel must belong to exactly one diagonal mirror"
        mirror_kind[(r, c)] = "\\" if has_backslash else "/"
    return mirror_kind


def _reflect(direction: int, mirror_kind: str) -> int:
    if mirror_kind == "\\":
        return REFLECT_BACKSLASH[direction]
    return REFLECT_SLASH[direction]


def _trace_ray(
    shape: tuple[int, int],
    start_tip: tuple[int, int],
    direction: int,
    arrow_pixels: set[tuple[int, int]],
    mirrors: dict[tuple[int, int], str],
) -> tuple[list[tuple[int, int]], tuple[int, int], int]:
    h, w = shape
    pos = (int(start_tip[0]), int(start_tip[1]))
    last_beam = pos
    beam: list[tuple[int, int]] = []
    seen: set[tuple[int, int, int]] = set()

    while True:
        state = (pos[0], pos[1], direction)
        assert state not in seen, "mirror layout must not trap a ray in a cycle"
        seen.add(state)

        dr, dc = DIRS[direction]
        nr = pos[0] + dr
        nc = pos[1] + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break

        nxt = (nr, nc)
        if nxt in arrow_pixels:
            break
        if nxt in mirrors:
            pos = nxt
            direction = _reflect(direction, mirrors[nxt])
            continue

        beam.append(nxt)
        last_beam = nxt
        pos = nxt

    return beam, last_beam, direction


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast each colored arrow's ray through grey diagonal mirrors and stamp the same arrow at the ray's endpoint.

    Input structure:
    - The background is black.
    - The foreground contains 1-3 monochrome 3x3 directional arrows and 1-3 grey diagonal mirror segments.
    - Each mirror is a short '/' or '\' chain of grey pixels, and each arrow points in one of the four cardinal directions.

    Rule:
    - Keep the original arrows and mirrors in place.
    - From each arrow's tip pixel, emit a one-cell-wide ray in the arrow's facing direction, coloring only non-mirror cells with that arrow's color.
    - When the ray enters a grey mirror pixel, it reflects: '\' swaps horizontal with vertical in the matching way, and '/' does the opposite pairing.
    - A ray stops when its next step would leave the grid or enter an arrow pixel; at the last ray pixel, draw a same-colored copy of the source arrow, rotated to the ray's final travel direction.

    Key insight:
    - The task combines three cues: decode the arrow orientation from its 3x3 sprite, treat grey diagonals as reflectors instead of blockers, and orient the copied endpoint arrow by the post-reflection travel direction rather than the source direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    arrows = _detect_arrows(g)
    mirrors = _detect_mirrors(g)
    arrow_pixels = set()
    for arrow in arrows:
        arrow_pixels.update(arrow["coords"])

    out = g.copy()
    for arrow in arrows:
        beam, terminal, final_direction = _trace_ray(
            g.shape,
            arrow["tip"],
            arrow["direction"],
            arrow_pixels,
            mirrors,
        )
        for r, c in beam:
            out[int(r), int(c)] = int(arrow["color"])
        _paint_arrow(out, terminal, final_direction, arrow["color"])

    return out


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _place_arrow_top_left(grid: np.ndarray, top_left: tuple[int, int], direction: int, color: int) -> None:
    patch = ARROW_PATCHES[direction]
    r0, c0 = map(int, top_left)
    coords = np.argwhere(patch)
    for dr, dc in coords:
        grid[r0 + int(dr), c0 + int(dc)] = int(color)


def _scenario_ok(
    scenario: np.ndarray,
    reserved: np.ndarray,
) -> tuple[bool, np.ndarray]:
    try:
        out = main(scenario)
    except Exception:
        return False, reserved
    occupied = out != BLACK
    padded = _dilate(occupied)
    if np.any(padded & reserved):
        return False, reserved
    return True, padded


def _build_simple_edge(
    h: int,
    w: int,
    color: int,
    reserved: np.ndarray,
) -> np.ndarray | None:
    for _ in range(SCENARIO_ATTEMPTS):
        if w < 7:
            return None
        r0 = random.randint(0, h - 3)
        c0 = random.randint(0, w - 7)
        grid = np.zeros((h, w), dtype=int)
        _place_arrow_top_left(grid, (r0, c0), RIGHT, color)
        ok, _ = _scenario_ok(grid, reserved)
        if ok:
            return grid
    return None


def _build_single_mirror(
    h: int,
    w: int,
    color: int,
    reserved: np.ndarray,
    kind: str,
) -> np.ndarray | None:
    for _ in range(SCENARIO_ATTEMPTS):
        if h < 5 or w < 8:
            return None
        r0 = random.randint(0, h - 3)
        c0 = random.randint(0, w - 7)
        y = r0 + 1
        m_lo = max(c0 + 4, 1)
        m_hi = w - 2
        if m_lo > m_hi:
            continue
        m = random.randint(m_lo, m_hi)

        grid = np.zeros((h, w), dtype=int)
        _place_arrow_top_left(grid, (r0, c0), RIGHT, color)
        grid[y, m] = GREY
        if kind == "down":
            if y + 1 >= h or m + 1 >= w:
                continue
            grid[y + 1, m + 1] = GREY
        else:
            if y + 1 >= h or m - 1 < 0:
                continue
            grid[y + 1, m - 1] = GREY

        ok, _ = _scenario_ok(grid, reserved)
        if ok:
            return grid
    return None


def _build_double_mirror(
    h: int,
    w: int,
    color: int,
    reserved: np.ndarray,
    kind: str,
) -> np.ndarray | None:
    for _ in range(SCENARIO_ATTEMPTS):
        if h < 6 or w < 9:
            return None
        grid = np.zeros((h, w), dtype=int)
        c0 = random.randint(0, w - 8)

        if kind == "down_right":
            r0 = random.randint(0, h - 5)
            y = r0 + 1
            m_lo = max(c0 + 4, 1)
            m_hi = w - 2
            if m_lo > m_hi:
                continue
            m = random.randint(m_lo, m_hi)
            r2_lo = y + 2
            r2_hi = h - 2
            if r2_lo > r2_hi:
                continue
            r2 = random.randint(r2_lo, r2_hi)
            if r2 + 1 >= h or m + 1 >= w:
                continue

            _place_arrow_top_left(grid, (r0, c0), RIGHT, color)
            grid[y, m] = GREY
            grid[y + 1, m + 1] = GREY
            grid[r2, m] = GREY
            grid[r2 + 1, m + 1] = GREY
        else:
            r0 = random.randint(2, h - 3)
            y = r0 + 1
            m_lo = max(c0 + 4, 1)
            m_hi = w - 2
            if m_lo > m_hi:
                continue
            m = random.randint(m_lo, m_hi)
            r2_lo = 1
            r2_hi = y - 2
            if r2_lo > r2_hi:
                continue
            r2 = random.randint(r2_lo, r2_hi)
            if y + 1 >= h or m - 1 < 0 or r2 + 1 >= h:
                continue

            _place_arrow_top_left(grid, (r0, c0), RIGHT, color)
            grid[y, m] = GREY
            grid[y + 1, m - 1] = GREY
            grid[r2, m] = GREY
            grid[r2 + 1, m - 1] = GREY

        ok, _ = _scenario_ok(grid, reserved)
        if ok:
            return grid
    return None


def _build_blocker(
    h: int,
    w: int,
    color_a: int,
    color_b: int,
    reserved: np.ndarray,
    variant: str,
) -> np.ndarray | None:
    for _ in range(SCENARIO_ATTEMPTS):
        if h < 7 or w < 10:
            return None
        grid = np.zeros((h, w), dtype=int)
        cb = random.randint(6, w - 3)
        ca_hi = cb - 6
        if ca_hi < 0:
            continue
        ca = random.randint(0, ca_hi)

        if variant == "b_down":
            rb = random.randint(1, h - 3)
            ra = rb - 1
            _place_arrow_top_left(grid, (ra, ca), RIGHT, color_a)
            _place_arrow_top_left(grid, (rb, cb), DOWN, color_b)
        else:
            rb = random.randint(1, h - 4)
            ra = rb + 1
            _place_arrow_top_left(grid, (ra, ca), RIGHT, color_a)
            _place_arrow_top_left(grid, (rb, cb), UP, color_b)

        ok, _ = _scenario_ok(grid, reserved)
        if ok:
            return grid
    return None


def _merge_with_reservation(
    base: np.ndarray,
    scenario: np.ndarray,
    reserved: np.ndarray,
) -> np.ndarray | None:
    ok, padded = _scenario_ok(scenario, reserved)
    if not ok:
        return None

    merged = base.copy()
    mask = scenario != BLACK
    if np.any(merged[mask] != BLACK):
        return None
    merged[mask] = scenario[mask]
    reserved |= padded
    return merged


def _scenario_arrow_count(name: str) -> int:
    if name == "blocker":
        return 2
    return 1


def _scenario_sort_key(name: str) -> tuple[int, int]:
    priority = {
        "double_mirror": 0,
        "blocker": 1,
        "single_mirror": 2,
        "simple_edge": 3,
    }
    return (priority[name], 0)


def generate_input() -> np.ndarray:
    combos = [
        ["single_mirror"],
        ["double_mirror"],
        ["single_mirror", "simple_edge"],
        ["single_mirror", "blocker"],
        ["double_mirror", "simple_edge"],
        ["single_mirror", "single_mirror"],
        ["double_mirror", "single_mirror"],
    ]

    for _ in range(GEN_ATTEMPTS):
        combo = random.choice(combos)
        arrow_count = sum(_scenario_arrow_count(name) for name in combo)
        if arrow_count >= 3 or "double_mirror" in combo and "single_mirror" in combo:
            h = random.randint(22, 28)
            w = random.randint(22, 28)
        elif len(combo) == 1:
            h = random.randint(12, 22)
            w = random.randint(12, 22)
        else:
            h = random.randint(18, 26)
            w = random.randint(18, 26)

        colors = random.sample(ARROW_COLORS, arrow_count)
        color_i = 0

        grid = np.zeros((h, w), dtype=int)
        reserved = np.zeros((h, w), dtype=bool)

        scenario_names = sorted(combo, key=_scenario_sort_key)
        success = True
        for name in scenario_names:
            scenario = None
            if name == "single_mirror":
                kind = random.choice(["up", "down"])
                scenario = _build_single_mirror(h, w, colors[color_i], reserved, kind)
                color_i += 1
            elif name == "double_mirror":
                kind = random.choice(["up_right", "down_right"])
                scenario = _build_double_mirror(h, w, colors[color_i], reserved, kind)
                color_i += 1
            elif name == "simple_edge":
                scenario = _build_simple_edge(h, w, colors[color_i], reserved)
                color_i += 1
            else:
                variant = random.choice(["b_up", "b_down"])
                scenario = _build_blocker(h, w, colors[color_i], colors[color_i + 1], reserved, variant)
                color_i += 2

            if scenario is None:
                success = False
                break

            merged = _merge_with_reservation(grid, scenario, reserved)
            if merged is None:
                success = False
                break
            grid = merged

        if not success:
            continue

        k = random.randint(0, 3)
        if k:
            grid = np.rot90(grid, k)

        try:
            out = main(grid)
        except Exception:
            continue

        if np.array_equal(out, grid):
            continue

        arrow_comps = _detect_arrows(grid)
        mirror_pixels = int(np.sum(grid == GREY))
        if not (1 <= len(arrow_comps) <= 3):
            continue
        if mirror_pixels < 2:
            continue

        return grid

    raise RuntimeError("failed to generate a valid mirror-ray ARC input")
