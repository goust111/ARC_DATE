import random

import numpy as np


BG = 0
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]

GRID_MIN = 12
GRID_MAX = 18
CENTER_MARGIN = 2
PAIR_TRIES = 120
EXTRA_TRIES = 120
GENERATE_TRIES = 300


def _components4(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _parse_arrow(comp: dict) -> dict:
    pixels = {tuple(map(int, cell)) for cell in comp["coords"]}
    assert len(pixels) == 4, "each arrow must be a 4-cell T shape"

    centers: list[tuple[int, int]] = []
    for r, c in pixels:
        degree = sum((r + dr, c + dc) in pixels for dr, dc in DIRS)
        if degree == 3:
            centers.append((r, c))
        else:
            assert degree == 1, "non-center arrow pixels must be arms"
    assert len(centers) == 1, "each arrow must have exactly one center"

    center = centers[0]
    present = [(dr, dc) for dr, dc in DIRS if (center[0] + dr, center[1] + dc) in pixels]
    assert len(present) == 3, "arrow center must have three present arms"
    missing = next(direction for direction in DIRS if direction not in present)
    direction = (-missing[0], -missing[1])

    return {
        "color": int(comp["color"]),
        "pixels": pixels,
        "center": center,
        "direction": direction,
    }


def _extract_arrows(grid: np.ndarray) -> list[dict]:
    arrows = [_parse_arrow(comp) for comp in _components4(grid)]
    assert arrows, "expected at least one arrow"
    return arrows


def _trace_ray(shape: tuple[int, int], occupied: dict, arrow: dict) -> dict:
    h, w = shape
    own = arrow["pixels"]
    r, c = arrow["center"]
    dr, dc = arrow["direction"]

    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w and (r, c) in own:
        r += dr
        c += dc

    path: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        hit = occupied.get((r, c))
        if hit is not None and hit is not arrow:
            return {
                "source_color": arrow["color"],
                "path": path,
                "branch": "hit",
                "fill_color": hit["color"],
                "target_color": hit["color"],
            }
        path.append((r, c))
        r += dr
        c += dc

    return {
        "source_color": arrow["color"],
        "path": path,
        "branch": "boundary",
        "fill_color": arrow["color"],
        "target_color": None,
    }


def _compute_rays(arrows: list[dict], shape: tuple[int, int]) -> list[dict]:
    occupied: dict[tuple[int, int], dict] = {}
    for arrow in arrows:
        for cell in arrow["pixels"]:
            assert cell not in occupied, "arrows must not overlap"
            occupied[cell] = arrow
    return [_trace_ray(shape, occupied, arrow) for arrow in arrows]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    arrows = _extract_arrows(g)
    rays = _compute_rays(arrows, g.shape)

    out = g.copy()
    fills: dict[tuple[int, int], int] = {}
    for ray in rays:
        color = int(ray["fill_color"])
        for cell in ray["path"]:
            prev = fills.get(cell)
            assert prev is None or prev == color, "conflicting ray fills"
            fills[cell] = color

    for (r, c), color in fills.items():
        out[r, c] = color
    return out


def _arrow_pixels(center: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    missing = (-direction[0], -direction[1])
    pixels = {(r, c)}
    for dr, dc in DIRS:
        if (dr, dc) != missing:
            pixels.add((r + dr, c + dc))
    return pixels


def _halo(pixels: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    halo: set[tuple[int, int]] = set()
    for r, c in pixels:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    halo.add((nr, nc))
    return halo


def _placeable(
    pixels: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    h, w = shape
    for r, c in pixels:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in blocked or (r, c) in reserved_paths:
            return False
    return True


def _boundary_path(
    center: tuple[int, int],
    direction: tuple[int, int],
    own: set[tuple[int, int]],
    occupied_pixels: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[tuple[int, int]] | None:
    h, w = shape
    r, c = center
    dr, dc = direction
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w and (r, c) in own:
        r += dr
        c += dc

    path: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cell = (r, c)
        if cell in occupied_pixels or cell in reserved_paths:
            return None
        path.append(cell)
        r += dr
        c += dc
    return path if path else None


def _hit_path(
    center: tuple[int, int],
    direction: tuple[int, int],
    own: set[tuple[int, int]],
    target_pixels: set[tuple[int, int]],
    occupied_pixels: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[tuple[int, int]] | None:
    h, w = shape
    r, c = center
    dr, dc = direction
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w and (r, c) in own:
        r += dr
        c += dc

    path: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cell = (r, c)
        if cell in target_pixels:
            return path if path else None
        if cell in occupied_pixels or cell in reserved_paths:
            return None
        path.append(cell)
        r += dr
        c += dc
    return None


def _build_grid(arrows: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for arrow in arrows:
        for r, c in arrow["pixels"]:
            grid[r, c] = int(arrow["color"])
    return grid


def _try_pair(
    rng: random.Random,
    shape: tuple[int, int],
    occupied_pixels: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
    source_color: int,
    target_color: int,
) -> tuple[list[dict], list[list[tuple[int, int]]]] | None:
    h, w = shape
    for _ in range(PAIR_TRIES):
        axis = rng.choice(("h", "v"))
        if axis == "h":
            direction = rng.choice([(0, 1), (0, -1)])
            row = rng.randint(CENTER_MARGIN, h - 1 - CENTER_MARGIN)
            if direction == (0, 1):
                src_col = rng.randint(CENTER_MARGIN, w - 1 - CENTER_MARGIN - 4)
                tgt_col = rng.randint(src_col + 4, w - 1 - CENTER_MARGIN)
            else:
                src_col = rng.randint(CENTER_MARGIN + 4, w - 1 - CENTER_MARGIN)
                tgt_col = rng.randint(CENTER_MARGIN, src_col - 4)
            source_center = (row, src_col)
            target_center = (row, tgt_col)
        else:
            direction = rng.choice([(1, 0), (-1, 0)])
            col = rng.randint(CENTER_MARGIN, w - 1 - CENTER_MARGIN)
            if direction == (1, 0):
                src_row = rng.randint(CENTER_MARGIN, h - 1 - CENTER_MARGIN - 4)
                tgt_row = rng.randint(src_row + 4, h - 1 - CENTER_MARGIN)
            else:
                src_row = rng.randint(CENTER_MARGIN + 4, h - 1 - CENTER_MARGIN)
                tgt_row = rng.randint(CENTER_MARGIN, src_row - 4)
            source_center = (src_row, col)
            target_center = (tgt_row, col)

        target_direction = rng.choice([step for step in DIRS if step != (-direction[0], -direction[1])])

        source_pixels = _arrow_pixels(source_center, direction)
        target_pixels = _arrow_pixels(target_center, target_direction)

        if not _placeable(source_pixels, blocked, reserved_paths, shape):
            continue
        blocked_with_source = blocked | _halo(source_pixels, shape)
        if not _placeable(target_pixels, blocked_with_source, reserved_paths, shape):
            continue

        source_path = _hit_path(
            source_center,
            direction,
            source_pixels,
            target_pixels,
            occupied_pixels,
            reserved_paths,
            shape,
        )
        if source_path is None:
            continue

        target_path = _boundary_path(
            target_center,
            target_direction,
            target_pixels,
            occupied_pixels | source_pixels,
            reserved_paths,
            shape,
        )
        if target_path is None:
            continue

        if set(source_path) & set(target_path):
            continue

        arrows = [
            {
                "color": source_color,
                "center": source_center,
                "direction": direction,
                "pixels": source_pixels,
                "expected_branch": "hit",
                "expected_target": target_color,
            },
            {
                "color": target_color,
                "center": target_center,
                "direction": target_direction,
                "pixels": target_pixels,
                "expected_branch": "boundary",
                "expected_target": None,
            },
        ]
        return arrows, [source_path, target_path]
    return None


def _try_boundary_arrow(
    rng: random.Random,
    shape: tuple[int, int],
    occupied_pixels: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
    color: int,
) -> tuple[dict, list[tuple[int, int]]] | None:
    h, w = shape
    for _ in range(EXTRA_TRIES):
        center = (
            rng.randint(CENTER_MARGIN, h - 1 - CENTER_MARGIN),
            rng.randint(CENTER_MARGIN, w - 1 - CENTER_MARGIN),
        )
        direction = rng.choice(DIRS)
        pixels = _arrow_pixels(center, direction)
        if not _placeable(pixels, blocked, reserved_paths, shape):
            continue

        path = _boundary_path(center, direction, pixels, occupied_pixels, reserved_paths, shape)
        if path is None:
            continue

        arrow = {
            "color": color,
            "center": center,
            "direction": direction,
            "pixels": pixels,
            "expected_branch": "boundary",
            "expected_target": None,
        }
        return arrow, path
    return None


def _validate_generated(grid: np.ndarray, expected: dict[int, tuple[str, int | None]]) -> bool:
    try:
        arrows = _extract_arrows(grid)
        rays = _compute_rays(arrows, grid.shape)
    except AssertionError:
        return False

    if len(arrows) != len(expected):
        return False

    seen_hit = False
    seen_boundary = False
    filled: set[tuple[int, int]] = set()
    for ray in rays:
        branch, target_color = expected.get(int(ray["source_color"]), ("", None))
        if ray["branch"] != branch:
            return False
        if branch == "hit":
            seen_hit = True
            if int(ray["fill_color"]) != int(target_color):
                return False
        elif branch == "boundary":
            seen_boundary = True
            if int(ray["fill_color"]) != int(ray["source_color"]):
                return False
        else:
            return False

        if len(ray["path"]) < 1:
            return False
        for cell in ray["path"]:
            if cell in filled:
                return False
            filled.add(cell)

    if not (seen_hit and seen_boundary):
        return False

    out = main(grid)
    return bool(np.any(out != grid))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    shape = (h, w)

    pair_count = rng.choice([1, 2])
    extra_count = 1 if pair_count == 1 else rng.choice([0, 1])
    total_arrows = 2 * pair_count + extra_count
    colors = rng.sample(PALETTE, total_arrows)

    arrows: list[dict] = []
    occupied_pixels: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()
    reserved_paths: set[tuple[int, int]] = set()
    expected: dict[int, tuple[str, int | None]] = {}

    color_index = 0
    for _ in range(pair_count):
        pair = _try_pair(
            rng,
            shape,
            occupied_pixels,
            blocked,
            reserved_paths,
            colors[color_index],
            colors[color_index + 1],
        )
        color_index += 2
        if pair is None:
            return None
        pair_arrows, pair_paths = pair
        for arrow in pair_arrows:
            arrows.append(arrow)
            occupied_pixels |= arrow["pixels"]
            blocked |= _halo(arrow["pixels"], shape)
            expected[int(arrow["color"])] = (str(arrow["expected_branch"]), arrow["expected_target"])
        for path in pair_paths:
            reserved_paths |= set(path)

    for _ in range(extra_count):
        extra = _try_boundary_arrow(
            rng,
            shape,
            occupied_pixels,
            blocked,
            reserved_paths,
            colors[color_index],
        )
        color_index += 1
        if extra is None:
            return None
        arrow, path = extra
        arrows.append(arrow)
        occupied_pixels |= arrow["pixels"]
        blocked |= _halo(arrow["pixels"], shape)
        reserved_paths |= set(path)
        expected[int(arrow["color"])] = (str(arrow["expected_branch"]), arrow["expected_target"])

    grid = _build_grid(arrows, shape)
    if not _validate_generated(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATE_TRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-ray input")
