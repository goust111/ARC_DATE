import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)

RNG = np.random.default_rng()

ZONE_MIN = 8
ZONE_MAX = 11
OUTER_MARGIN = 1
ZONE_GAP = 2
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 200

COUNT_MODES = ("horizontal_only", "vertical_only", "both")
ORIENTATIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))  # (h_sign, v_sign)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comps.append(np.array(pts, dtype=int))

    return comps


def _run_length(cells: set[tuple[int, int]], corner: tuple[int, int], dr: int, dc: int) -> int:
    rr, cc = map(int, corner)
    length = 1
    while (int(rr + dr), int(cc + dc)) in cells:
        rr += dr
        cc += dc
        length += 1
    return int(length)


def _parse_l_component(coords: np.ndarray) -> tuple[tuple[int, int], int, int, int, int]:
    cells = {tuple(map(int, pt)) for pt in coords.tolist()}
    corner_candidates: list[tuple[int, int, list[tuple[int, int]]]] = []

    for r, c in cells:
        neighbors = [(dr, dc) for dr, dc in DIR4 if (int(r + dr), int(c + dc)) in cells]
        if len(neighbors) != 2:
            continue
        has_horizontal = any(dc != 0 for _dr, dc in neighbors)
        has_vertical = any(dr != 0 for dr, _dc in neighbors)
        if has_horizontal and has_vertical:
            corner_candidates.append((int(r), int(c), neighbors))

    assert len(corner_candidates) == 1, "expected exactly one L-corner"
    corner_r, corner_c, neighbors = corner_candidates[0]

    h_sign = next(int(dc) for dr, dc in neighbors if dr == 0)
    v_sign = next(int(dr) for dr, dc in neighbors if dc == 0)

    h_len = _run_length(cells, (corner_r, corner_c), 0, h_sign)
    v_len = _run_length(cells, (corner_r, corner_c), v_sign, 0)
    assert h_len >= 2 and v_len >= 2, "each arm must have length >= 2 including the corner"

    expected = {(corner_r, int(corner_c + h_sign * step)) for step in range(h_len)}
    expected |= {(int(corner_r + v_sign * step), corner_c) for step in range(v_len)}
    assert expected == cells, "component must be a single-pixel-wide L shape"

    return (corner_r, corner_c), h_sign, v_sign, h_len, v_len


def main(grid: np.ndarray) -> np.ndarray:
    """Aggregate aligned same-color counters into L corners, then redraw count-driven rays.

    Input structure:
    - The background is black.
    - Each non-black color contains one single-pixel-wide L-shaped corner made from one
      horizontal arm and one vertical arm that share a corner pixel.
    - Additional isolated pixels of the same color lie on the corner row or the corner column.

    Rule:
    - For each color, detect the L component and locate its corner plus the original horizontal
      and vertical directions.
    - Count same-color isolated pixels on the corner row as the horizontal group, and on the
      corner column as the vertical group.
    - Clear the whole color pattern, keep the corner pixel, then extend a horizontal ray in the
      original horizontal direction by the horizontal-group count and a vertical ray in the
      original vertical direction by the vertical-group count.

    Key insight:
    - The scattered pixels are counters selected by row/column alignment with the L corner, while
      the original arm lengths are irrelevant once the corner and its two directions are identified.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = np.full_like(g, BG)
    colors = [int(color) for color in np.unique(g) if int(color) != BG]

    for color in colors:
        comps = _components(g == color)
        assert comps, "missing color components"

        l_components = [comp for comp in comps if int(comp.shape[0]) > 1]
        assert len(l_components) == 1, "each color must contain exactly one L component"

        corner, h_sign, v_sign, _h_len, _v_len = _parse_l_component(l_components[0])
        corner_r, corner_c = map(int, corner)

        horizontal_count = 0
        vertical_count = 0

        for comp in comps:
            if int(comp.shape[0]) != 1:
                continue
            r, c = map(int, comp[0])
            if r == corner_r:
                horizontal_count += 1
            elif c == corner_c:
                vertical_count += 1
            else:
                raise AssertionError("isolated pixels must align with the corner row or column")

        if out[corner_r, corner_c] not in (BG, color):
            raise AssertionError("output objects must not overlap")
        out[corner_r, corner_c] = color

        for step in range(1, horizontal_count + 1):
            rr = corner_r
            cc = int(corner_c + h_sign * step)
            assert 0 <= cc < g.shape[1]
            if out[rr, cc] not in (BG, color):
                raise AssertionError("horizontal rays must not overlap")
            out[rr, cc] = color

        for step in range(1, vertical_count + 1):
            rr = int(corner_r + v_sign * step)
            cc = corner_c
            assert 0 <= rr < g.shape[0]
            if out[rr, cc] not in (BG, color):
                raise AssertionError("vertical rays must not overlap")
            out[rr, cc] = color

    return out


def _max_spaced_count(values: list[int]) -> int:
    count = 0
    last = None
    for value in sorted(set(int(v) for v in values)):
        if last is None or value > last + 1:
            count += 1
            last = value
    return int(count)


def _sample_spaced_positions(values: list[int], k: int) -> list[int] | None:
    values = sorted(set(int(v) for v in values))
    if k == 0:
        return []
    if _max_spaced_count(values) < k:
        return None

    def helper(start: int, last_value: int | None, need: int) -> list[int] | None:
        if need == 0:
            return []

        indices = list(range(start, len(values)))
        RNG.shuffle(indices)
        for idx in indices:
            value = int(values[idx])
            if last_value is not None and value <= last_value + 1:
                continue

            remaining = 0
            last = value
            for nxt in range(idx + 1, len(values)):
                candidate = int(values[nxt])
                if candidate > last + 1:
                    remaining += 1
                    last = candidate
            if 1 + remaining < need:
                continue

            tail = helper(idx + 1, value, need - 1)
            if tail is not None:
                return [value] + tail

        return None

    return helper(0, None, k)


def _row_candidates(c0: int, c1: int, corner_c: int, h_sign: int, h_len: int) -> list[int]:
    candidates: list[int] = []

    left_end = int(corner_c - 2)
    if h_sign == -1:
        left_end = int(corner_c - h_len - 1)
    if c0 <= left_end:
        candidates.extend(range(int(c0), int(left_end) + 1))

    right_start = int(corner_c + 2)
    if h_sign == 1:
        right_start = int(corner_c + h_len + 1)
    if right_start <= c1:
        candidates.extend(range(int(right_start), int(c1) + 1))

    return candidates


def _col_candidates(r0: int, r1: int, corner_r: int, v_sign: int, v_len: int) -> list[int]:
    candidates: list[int] = []

    top_end = int(corner_r - 2)
    if v_sign == -1:
        top_end = int(corner_r - v_len - 1)
    if r0 <= top_end:
        candidates.extend(range(int(r0), int(top_end) + 1))

    bottom_start = int(corner_r + 2)
    if v_sign == 1:
        bottom_start = int(corner_r + v_len + 1)
    if bottom_start <= r1:
        candidates.extend(range(int(bottom_start), int(r1) + 1))

    return candidates


def _paint_input_object(grid: np.ndarray, spec: dict) -> None:
    color = int(spec["color"])
    corner_r, corner_c = map(int, spec["corner"])
    h_sign = int(spec["h_sign"])
    v_sign = int(spec["v_sign"])
    h_len = int(spec["h_len"])
    v_len = int(spec["v_len"])

    for step in range(h_len):
        rr = corner_r
        cc = int(corner_c + h_sign * step)
        assert grid[rr, cc] == BG
        grid[rr, cc] = color

    for step in range(v_len):
        rr = int(corner_r + v_sign * step)
        cc = corner_c
        if step == 0:
            assert grid[rr, cc] == color
        else:
            assert grid[rr, cc] == BG
            grid[rr, cc] = color

    for cc in spec["row_positions"]:
        assert grid[corner_r, int(cc)] == BG
        grid[corner_r, int(cc)] = color

    for rr in spec["col_positions"]:
        assert grid[int(rr), corner_c] == BG
        grid[int(rr), corner_c] = color


def _paint_output_object(grid: np.ndarray, spec: dict) -> None:
    color = int(spec["color"])
    corner_r, corner_c = map(int, spec["corner"])
    h_sign = int(spec["h_sign"])
    v_sign = int(spec["v_sign"])

    if grid[corner_r, corner_c] not in (BG, color):
        raise AssertionError("output overlap at corner")
    grid[corner_r, corner_c] = color

    for step in range(1, int(spec["h_count"]) + 1):
        rr = corner_r
        cc = int(corner_c + h_sign * step)
        if grid[rr, cc] not in (BG, color):
            raise AssertionError("output overlap on horizontal ray")
        grid[rr, cc] = color

    for step in range(1, int(spec["v_count"]) + 1):
        rr = int(corner_r + v_sign * step)
        cc = corner_c
        if grid[rr, cc] not in (BG, color):
            raise AssertionError("output overlap on vertical ray")
        grid[rr, cc] = color


def _place_object(zone: tuple[int, int, int, int], color: int, orientation: tuple[int, int], mode: str) -> dict | None:
    r0, r1, c0, c1 = map(int, zone)
    h_sign, v_sign = map(int, orientation)

    for _ in range(PLACE_ATTEMPTS):
        corner_r = int(RNG.integers(r0 + 1, r1))
        corner_c = int(RNG.integers(c0 + 1, c1))

        h_room = int(c1 - corner_c) if h_sign == 1 else int(corner_c - c0)
        v_room = int(r1 - corner_r) if v_sign == 1 else int(corner_r - r0)
        if h_room < 1 or v_room < 1:
            continue

        h_len = int(RNG.integers(2, h_room + 2))
        v_len = int(RNG.integers(2, v_room + 2))

        row_candidates = _row_candidates(c0, c1, corner_c, h_sign, h_len)
        col_candidates = _col_candidates(r0, r1, corner_r, v_sign, v_len)

        h_max = min(h_room, _max_spaced_count(row_candidates))
        v_max = min(v_room, _max_spaced_count(col_candidates))

        if mode == "horizontal_only":
            if h_max <= 0:
                continue
            h_count = int(RNG.integers(1, h_max + 1))
            v_count = 0
        elif mode == "vertical_only":
            if v_max <= 0:
                continue
            h_count = 0
            v_count = int(RNG.integers(1, v_max + 1))
        else:
            if h_max <= 0 or v_max <= 0:
                continue
            h_count = int(RNG.integers(1, h_max + 1))
            v_count = int(RNG.integers(1, v_max + 1))

        row_positions = _sample_spaced_positions(row_candidates, h_count)
        col_positions = _sample_spaced_positions(col_candidates, v_count)
        if row_positions is None or col_positions is None:
            continue

        return {
            "color": int(color),
            "corner": (int(corner_r), int(corner_c)),
            "h_sign": int(h_sign),
            "v_sign": int(v_sign),
            "h_len": int(h_len),
            "v_len": int(v_len),
            "h_count": int(h_count),
            "v_count": int(v_count),
            "row_positions": [int(v) for v in row_positions],
            "col_positions": [int(v) for v in col_positions],
        }

    return None


def _zones() -> tuple[tuple[int, int, int, int], ...]:
    row_heights = [int(v) for v in RNG.integers(ZONE_MIN, ZONE_MAX + 1, size=2)]
    col_widths = [int(v) for v in RNG.integers(ZONE_MIN, ZONE_MAX + 1, size=2)]

    zones: list[tuple[int, int, int, int]] = []
    top = OUTER_MARGIN
    for row_h in row_heights:
        left = OUTER_MARGIN
        for col_w in col_widths:
            zones.append((int(top), int(top + row_h - 1), int(left), int(left + col_w - 1)))
            left += col_w + ZONE_GAP
        top += row_h + ZONE_GAP

    return tuple(zones)


def _count_modes(n_objects: int) -> list[str]:
    order = list(COUNT_MODES)
    RNG.shuffle(order)
    modes = order[: min(n_objects, len(order))]
    while len(modes) < n_objects:
        modes.append(str(order[int(RNG.integers(0, len(order)))]))
    RNG.shuffle(modes)
    return modes


def _gen_attempt() -> np.ndarray | None:
    zones = list(_zones())
    h = max(zone[1] for zone in zones) + OUTER_MARGIN + 1
    w = max(zone[3] for zone in zones) + OUTER_MARGIN + 1
    if max(h, w) > 30:
        return None

    n_objects = int(RNG.integers(2, 5))
    zone_indices = [int(v) for v in RNG.choice(len(zones), size=n_objects, replace=False)]
    colors = [int(v) for v in RNG.choice(np.array(PALETTE), size=n_objects, replace=False)]

    orientations = list(ORIENTATIONS)
    RNG.shuffle(orientations)
    while len(orientations) < n_objects:
        orientations.append(ORIENTATIONS[int(RNG.integers(0, len(ORIENTATIONS)))])
    orientations = orientations[:n_objects]

    modes = _count_modes(n_objects)

    grid = np.full((h, w), BG, dtype=int)
    expected = np.full_like(grid, BG)
    specs: list[dict] = []

    for zone_index, color, orientation, mode in zip(zone_indices, colors, orientations, modes):
        spec = _place_object(zones[zone_index], color, orientation, mode)
        if spec is None:
            return None
        specs.append(spec)
        _paint_input_object(grid, spec)
        _paint_output_object(expected, spec)

    if not any(int(spec["h_count"]) > 0 for spec in specs):
        return None
    if not any(int(spec["v_count"]) > 0 for spec in specs):
        return None
    if not any(
        int(spec["h_count"]) != int(spec["h_len"] - 1) or int(spec["v_count"]) != int(spec["v_len"] - 1)
        for spec in specs
    ):
        return None

    try:
        produced = main(grid)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
