import numpy as np


BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
QUADRANTS = ("tl", "tr", "bl", "br")
MAX_GEN_ATTEMPTS = 400
MAX_PLACE_ATTEMPTS = 200


def _most_common(arr: np.ndarray) -> int:
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(counts)])


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
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _corner_signal_colors(grid: np.ndarray) -> dict[str, int]:
    h, w = grid.shape
    blocks = {
        "tl": grid[:2, :2],
        "tr": grid[:2, w - 2 :],
        "bl": grid[h - 2 :, :2],
        "br": grid[h - 2 :, w - 2 :],
    }
    colors: dict[str, int] = {}
    for name, block in blocks.items():
        vals = np.unique(block)
        assert len(vals) == 1, "each corner signal must be a solid 2x2 block"
        colors[name] = int(vals[0])
    assert len(set(colors.values())) == 4, "signal colors must be distinct"
    return colors


def _quadrant_bounds(h: int, w: int, quadrant: str) -> tuple[int, int, int, int]:
    mid_r = h // 2
    mid_c = w // 2
    if quadrant == "tl":
        return 2, mid_r - 1, 2, mid_c - 1
    if quadrant == "tr":
        return 2, mid_r - 1, mid_c + 1, w - 3
    if quadrant == "bl":
        return mid_r + 1, h - 3, 2, mid_c - 1
    if quadrant == "br":
        return mid_r + 1, h - 3, mid_c + 1, w - 3
    raise AssertionError(f"unknown quadrant {quadrant}")


def _quadrant_of(center_r: float, center_c: float, h: int, w: int) -> str:
    mid_r = (h - 1) / 2.0
    mid_c = (w - 1) / 2.0
    assert center_r != mid_r and center_c != mid_c, "path center cannot lie on a quadrant boundary"
    top = center_r < mid_r
    left = center_c < mid_c
    if top and left:
        return "tl"
    if top:
        return "tr"
    if left:
        return "bl"
    return "br"


def _is_corner_component(coords: np.ndarray, h: int, w: int) -> bool:
    if len(coords) != 4:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return False
    cells = {tuple(map(int, rc)) for rc in coords}
    corner_sets = [
        {(0, 0), (0, 1), (1, 0), (1, 1)},
        {(0, w - 2), (0, w - 1), (1, w - 2), (1, w - 1)},
        {(h - 2, 0), (h - 2, 1), (h - 1, 0), (h - 1, 1)},
        {(h - 2, w - 2), (h - 2, w - 1), (h - 1, w - 2), (h - 1, w - 1)},
    ]
    return cells in corner_sets


def _find_paths(grid: np.ndarray, bg: int) -> tuple[list[dict], int]:
    h, w = grid.shape
    singletons: dict[tuple[int, int], int] = {}
    paths: list[dict] = []

    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for comp in _components(grid == color):
            if _is_corner_component(comp, h, w):
                continue
            area = int(comp.shape[0])
            r0, c0, r1, c1 = _bbox(comp)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if area == 1:
                singletons[(r0, c0)] = color
                continue
            if (hh == 1 or ww == 1) and area == max(hh, ww) and 3 <= area <= 8:
                coords = [tuple(map(int, rc)) for rc in comp]
                if hh == 1:
                    coords.sort(key=lambda rc: rc[1])
                    orientation = "h"
                else:
                    coords.sort(key=lambda rc: rc[0])
                    orientation = "v"
                paths.append(
                    {
                        "color": color,
                        "coords": coords,
                        "bbox": (r0, c0, r1, c1),
                        "orientation": orientation,
                    }
                )
                continue
            raise AssertionError("unexpected foreground object")

    used_markers: set[tuple[int, int]] = set()
    marker_colors: set[int] = set()
    for path in paths:
        r0, c0, r1, c1 = path["bbox"]
        if path["orientation"] == "h":
            candidates = [((r0, c0 - 1), (0, -1)), ((r0, c1 + 1), (0, 1))]
        else:
            candidates = [((r0 - 1, c0), (-1, 0)), ((r1 + 1, c0), (1, 0))]
        matches = [(coord, d) for coord, d in candidates if coord in singletons]
        assert len(matches) == 1, "each path needs exactly one arrow marker"
        marker, direction = matches[0]
        used_markers.add(marker)
        marker_colors.add(int(singletons[marker]))
        center_r = (r0 + r1) / 2.0
        center_c = (c0 + c1) / 2.0
        path["marker"] = marker
        path["dir"] = direction
        path["quadrant"] = _quadrant_of(center_r, center_c, h, w)

    assert used_markers == set(singletons), "all singleton pixels must be arrow markers"
    assert len(marker_colors) == 1, "all arrow markers should share one grey color"
    return sorted(paths, key=lambda p: (p["marker"][0], p["marker"][1], p["color"])), marker_colors.pop()


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common(g)
    signals = _corner_signal_colors(g)
    paths, _marker_color = _find_paths(g, bg)

    out = np.full((h, w), bg, dtype=int)
    out[:2, :2] = g[:2, :2]
    out[:2, w - 2 :] = g[:2, w - 2 :]
    out[h - 2 :, :2] = g[h - 2 :, :2]
    out[h - 2 :, w - 2 :] = g[h - 2 :, w - 2 :]

    kept: list[dict] = []
    for path in paths:
        if int(path["color"]) != int(signals[path["quadrant"]]):
            continue
        kept.append(path)
        for r, c in path["coords"]:
            out[r, c] = int(path["color"])

    for path in kept:
        r, c = path["marker"]
        dr, dc = path["dir"]
        while 0 <= r < h and 0 <= c < w and int(out[r, c]) == bg:
            out[r, c] = int(path["color"])
            r += dr
            c += dc

    return out


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _path_from_active(active_r: int, active_c: int, dr: int, dc: int, length: int, color: int, kind: str) -> dict:
    coords = [(active_r - k * dr, active_c - k * dc) for k in range(length)]
    return {
        "coords": coords,
        "marker": (active_r + dr, active_c + dc),
        "dir": (dr, dc),
        "color": int(color),
        "kind": kind,
    }


def _path_cells(path: dict) -> set[tuple[int, int]]:
    return set(path["coords"]) | {path["marker"]}


def _neighbors8(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in DIRS8:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _can_place(path: dict, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]], h: int, w: int) -> bool:
    cells = _path_cells(path)
    if len(cells) != len(path["coords"]) + 1:
        return False
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied or (r, c) in reserved:
            return False
        for nr, nc in _neighbors8(r, c, h, w):
            if (nr, nc) in occupied:
                return False
    return True


def _paint_path(grid: np.ndarray, path: dict, occupied: set[tuple[int, int]]) -> None:
    for r, c in path["coords"]:
        grid[r, c] = int(path["color"])
        occupied.add((r, c))
    mr, mc = path["marker"]
    grid[mr, mc] = GREY
    occupied.add((mr, mc))


def _cells_to_boundary(marker: tuple[int, int], direction: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    dr, dc = direction
    r = marker[0] + dr
    c = marker[1] + dc
    cells = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _cells_until_before(marker: tuple[int, int], direction: tuple[int, int], stop: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    r = marker[0] + dr
    c = marker[1] + dc
    cells = []
    while (r, c) != stop:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _quadrants_for_horizontal(row_half: str, driver_side: str) -> tuple[str, str, tuple[str, str]]:
    if row_half == "top" and driver_side == "left":
        return "tl", "tr", ("bl", "br")
    if row_half == "top":
        return "tr", "tl", ("bl", "br")
    if driver_side == "left":
        return "bl", "br", ("tl", "tr")
    return "br", "bl", ("tl", "tr")


def _quadrants_for_vertical(col_half: str, driver_side: str) -> tuple[str, str, tuple[str, str]]:
    if col_half == "left" and driver_side == "top":
        return "tl", "bl", ("tr", "br")
    if col_half == "left":
        return "bl", "tl", ("tr", "br")
    if driver_side == "top":
        return "tr", "br", ("tl", "bl")
    return "br", "tr", ("tl", "bl")


def _make_fixed_row_path(
    rng: np.random.Generator,
    h: int,
    w: int,
    quadrant: str,
    color: int,
    direction: tuple[int, int],
    row: int,
    kind: str,
) -> dict | None:
    r0, r1, c0, c1 = _quadrant_bounds(h, w, quadrant)
    if not (r0 <= row <= r1):
        return None
    dr, dc = direction
    max_len = min(7, c1 - c0 + 1)
    if max_len < 3:
        return None
    for _ in range(MAX_PLACE_ATTEMPTS):
        length = _randint(rng, 3, max_len)
        if (dr, dc) == (0, 1):
            lo = c0 + length - 1
            hi = c1
        elif (dr, dc) == (0, -1):
            lo = c0
            hi = c1 - length + 1
        else:
            raise AssertionError("fixed-row paths must be horizontal")
        if lo > hi:
            continue
        active_c = _randint(rng, lo, hi)
        return _path_from_active(row, active_c, dr, dc, length, color, kind)
    return None


def _make_fixed_col_path(
    rng: np.random.Generator,
    h: int,
    w: int,
    quadrant: str,
    color: int,
    direction: tuple[int, int],
    col: int,
    kind: str,
) -> dict | None:
    r0, r1, c0, c1 = _quadrant_bounds(h, w, quadrant)
    if not (c0 <= col <= c1):
        return None
    dr, dc = direction
    max_len = min(7, r1 - r0 + 1)
    if max_len < 3:
        return None
    for _ in range(MAX_PLACE_ATTEMPTS):
        length = _randint(rng, 3, max_len)
        if (dr, dc) == (1, 0):
            lo = r0 + length - 1
            hi = r1
        elif (dr, dc) == (-1, 0):
            lo = r0
            hi = r1 - length + 1
        else:
            raise AssertionError("fixed-col paths must be vertical")
        if lo > hi:
            continue
        active_r = _randint(rng, lo, hi)
        return _path_from_active(active_r, col, dr, dc, length, color, kind)
    return None


def _make_random_quadrant_path(
    rng: np.random.Generator,
    h: int,
    w: int,
    quadrant: str,
    color: int,
    kind: str,
    directions: list[tuple[int, int]] | None = None,
) -> dict | None:
    r0, r1, c0, c1 = _quadrant_bounds(h, w, quadrant)
    dirs = directions or [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for _ in range(MAX_PLACE_ATTEMPTS):
        dr, dc = dirs[_randint(rng, 0, len(dirs) - 1)]
        if dr == 0:
            max_len = min(7, c1 - c0 + 1)
            if max_len < 3:
                continue
            length = _randint(rng, 3, max_len)
            row = _randint(rng, r0, r1)
            if dc == 1:
                lo = c0 + length - 1
                hi = c1
            else:
                lo = c0
                hi = c1 - length + 1
            if lo > hi:
                continue
            active_c = _randint(rng, lo, hi)
            return _path_from_active(row, active_c, dr, dc, length, color, kind)
        max_len = min(7, r1 - r0 + 1)
        if max_len < 3:
            continue
        length = _randint(rng, 3, max_len)
        col = _randint(rng, c0, c1)
        if dr == 1:
            lo = r0 + length - 1
            hi = r1
        else:
            lo = r0
            hi = r1 - length + 1
        if lo > hi:
            continue
        active_r = _randint(rng, lo, hi)
        return _path_from_active(active_r, col, dr, dc, length, color, kind)
    return None


def _place_horizontal_collision_pair(
    rng: np.random.Generator,
    grid: np.ndarray,
    signals: dict[str, int],
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> tuple[list[dict], tuple[str, str]] | None:
    h, w = grid.shape
    mid_r = h // 2
    row_half = "top" if _randint(rng, 0, 1) == 0 else "bottom"
    driver_side = "left" if _randint(rng, 0, 1) == 0 else "right"
    q_driver, q_block, remaining = _quadrants_for_horizontal(row_half, driver_side)
    row_lo = 2 if row_half == "top" else mid_r + 1
    row_hi = mid_r - 1 if row_half == "top" else h - 3
    inward = (0, 1) if driver_side == "left" else (0, -1)
    outward = (-1, 0) if row_half == "top" else (1, 0)

    for _ in range(MAX_PLACE_ATTEMPTS):
        shared_row = _randint(rng, row_lo, row_hi)
        driver = _make_fixed_row_path(rng, h, w, q_driver, signals[q_driver], inward, shared_row, "collision_driver")
        if driver is None:
            continue
        marker_c = driver["marker"][1]
        br0, br1, bc0, bc1 = _quadrant_bounds(h, w, q_block)
        if driver_side == "left":
            col_lo = max(bc0, marker_c + 2)
            col_hi = bc1
        else:
            col_lo = bc0
            col_hi = min(bc1, marker_c - 2)
        if col_lo > col_hi:
            continue
        block_col = _randint(rng, col_lo, col_hi)
        blocker = _make_fixed_col_path(rng, h, w, q_block, signals[q_block], outward, block_col, "boundary_blocker")
        if blocker is None:
            continue

        occ2 = occupied | _path_cells(driver)
        if not _can_place(driver, occupied, reserved, h, w):
            continue
        if not _can_place(blocker, occ2, reserved, h, w):
            continue

        stop = (shared_row, block_col)
        if stop not in set(blocker["coords"]):
            continue
        driver_corridor = _cells_until_before(driver["marker"], driver["dir"], stop)
        blocker_corridor = _cells_to_boundary(blocker["marker"], blocker["dir"], h, w)
        if not driver_corridor or not blocker_corridor:
            continue

        occ_all = occupied | _path_cells(driver) | _path_cells(blocker)
        if any(cell in occ_all or cell in reserved for cell in driver_corridor):
            continue
        if any(cell in occ_all or cell in reserved for cell in blocker_corridor):
            continue

        _paint_path(grid, driver, occupied)
        _paint_path(grid, blocker, occupied)
        reserved.update(driver_corridor)
        reserved.update(blocker_corridor)
        return [driver, blocker], remaining
    return None


def _place_vertical_collision_pair(
    rng: np.random.Generator,
    grid: np.ndarray,
    signals: dict[str, int],
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> tuple[list[dict], tuple[str, str]] | None:
    h, w = grid.shape
    mid_c = w // 2
    col_half = "left" if _randint(rng, 0, 1) == 0 else "right"
    driver_side = "top" if _randint(rng, 0, 1) == 0 else "bottom"
    q_driver, q_block, remaining = _quadrants_for_vertical(col_half, driver_side)
    col_lo = 2 if col_half == "left" else mid_c + 1
    col_hi = mid_c - 1 if col_half == "left" else w - 3
    inward = (1, 0) if driver_side == "top" else (-1, 0)
    outward = (0, -1) if col_half == "left" else (0, 1)

    for _ in range(MAX_PLACE_ATTEMPTS):
        shared_col = _randint(rng, col_lo, col_hi)
        driver = _make_fixed_col_path(rng, h, w, q_driver, signals[q_driver], inward, shared_col, "collision_driver")
        if driver is None:
            continue
        marker_r = driver["marker"][0]
        br0, br1, bc0, bc1 = _quadrant_bounds(h, w, q_block)
        if driver_side == "top":
            row_lo = max(br0, marker_r + 2)
            row_hi = br1
        else:
            row_lo = br0
            row_hi = min(br1, marker_r - 2)
        if row_lo > row_hi:
            continue
        block_row = _randint(rng, row_lo, row_hi)
        blocker = _make_fixed_row_path(rng, h, w, q_block, signals[q_block], outward, block_row, "boundary_blocker")
        if blocker is None:
            continue

        occ2 = occupied | _path_cells(driver)
        if not _can_place(driver, occupied, reserved, h, w):
            continue
        if not _can_place(blocker, occ2, reserved, h, w):
            continue

        stop = (block_row, shared_col)
        if stop not in set(blocker["coords"]):
            continue
        driver_corridor = _cells_until_before(driver["marker"], driver["dir"], stop)
        blocker_corridor = _cells_to_boundary(blocker["marker"], blocker["dir"], h, w)
        if not driver_corridor or not blocker_corridor:
            continue

        occ_all = occupied | _path_cells(driver) | _path_cells(blocker)
        if any(cell in occ_all or cell in reserved for cell in driver_corridor):
            continue
        if any(cell in occ_all or cell in reserved for cell in blocker_corridor):
            continue

        _paint_path(grid, driver, occupied)
        _paint_path(grid, blocker, occupied)
        reserved.update(driver_corridor)
        reserved.update(blocker_corridor)
        return [driver, blocker], remaining
    return None


def _place_blocked_path(
    rng: np.random.Generator,
    grid: np.ndarray,
    quadrant: str,
    color: int,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> dict | None:
    h, w = grid.shape
    for _ in range(MAX_PLACE_ATTEMPTS):
        path = _make_random_quadrant_path(rng, h, w, quadrant, color, "blocked")
        if path is None or not _can_place(path, occupied, reserved, h, w):
            continue
        _paint_path(grid, path, occupied)
        return path
    return None


def _place_boundary_path(
    rng: np.random.Generator,
    grid: np.ndarray,
    quadrant: str,
    color: int,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> dict | None:
    h, w = grid.shape
    options = {
        "tl": [(0, -1), (-1, 0)],
        "tr": [(0, 1), (-1, 0)],
        "bl": [(0, -1), (1, 0)],
        "br": [(0, 1), (1, 0)],
    }[quadrant]
    for _ in range(MAX_PLACE_ATTEMPTS):
        path = _make_random_quadrant_path(rng, h, w, quadrant, color, "extra_allowed", directions=options)
        if path is None or not _can_place(path, occupied, reserved, h, w):
            continue
        corridor = _cells_to_boundary(path["marker"], path["dir"], h, w)
        occ_all = occupied | _path_cells(path)
        if not corridor or any(cell in occ_all or cell in reserved for cell in corridor):
            continue
        _paint_path(grid, path, occupied)
        reserved.update(corridor)
        return path
    return None


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        signals = _corner_signal_colors(grid)
        paths, marker_color = _find_paths(grid, BG)
        if marker_color != GREY:
            return False
        kept = [p for p in paths if int(p["color"]) == int(signals[p["quadrant"]])]
        blocked = [p for p in paths if int(p["color"]) != int(signals[p["quadrant"]])]
        if len(kept) < 2 or len(blocked) < 2:
            return False
        out = main(grid)
        if np.any(out == GREY):
            return False
    except AssertionError:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = _randint(rng, 19, 25)
    w = _randint(rng, 19, 25)
    if h % 2 == 0:
        h += 1
    if w % 2 == 0:
        w += 1

    grid = np.full((h, w), BG, dtype=int)
    signal_values = rng.choice(PALETTE, size=4, replace=False)
    signals = {q: int(c) for q, c in zip(QUADRANTS, signal_values)}
    grid[:2, :2] = signals["tl"]
    grid[:2, w - 2 :] = signals["tr"]
    grid[h - 2 :, :2] = signals["bl"]
    grid[h - 2 :, w - 2 :] = signals["br"]

    occupied = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if int(grid[r, c]) != BG
    }
    reserved: set[tuple[int, int]] = set()
    paths: list[dict] = []

    pair = (
        _place_horizontal_collision_pair(rng, grid, signals, occupied, reserved)
        if _randint(rng, 0, 1) == 0
        else _place_vertical_collision_pair(rng, grid, signals, occupied, reserved)
    )
    if pair is None:
        return None
    pair_paths, remaining = pair
    paths.extend(pair_paths)

    q1, q2 = remaining
    if _place_blocked_path(rng, grid, q1, signals[q2], occupied, reserved) is None:
        return None
    if _place_blocked_path(rng, grid, q2, signals[q1], occupied, reserved) is None:
        return None

    if _randint(rng, 0, 1) == 1:
        extra_q = QUADRANTS[_randint(rng, 0, 3)]
        if _randint(rng, 0, 1) == 0:
            _place_boundary_path(rng, grid, extra_q, signals[extra_q], occupied, reserved)
        else:
            choices = [signals[q] for q in QUADRANTS if q != extra_q]
            extra_color = int(choices[_randint(rng, 0, len(choices) - 1)])
            _place_blocked_path(rng, grid, extra_q, extra_color, occupied, reserved)

    return grid if _validate_generated(grid) else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
