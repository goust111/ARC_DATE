import numpy as np


BG = 0
GREY = 5
LIQUID_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

MIN_CONTAINERS = 2
MAX_CONTAINERS = 4
INNER_WIDTH_CHOICES = (3, 5)
INTERIOR_HEIGHT_RANGE = (3, 7)  # inclusive low, exclusive high
BASE_SIDE_MARGIN_RANGE = (1, 3)
BASE_GAP_RANGE = (2, 5)
MAX_EXTRA_WIDTH = 6
MAX_EXTRA_HEIGHT = 5
MAX_GRID_W = 36
MAX_GRID_H = 24
MAX_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _parse_container(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, hole_r: int, hole_c: int):
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if hole_r <= r0 or hole_r >= r1:
        return None
    if hole_c == c0:
        hole_side = "left"
    elif hole_c == c1:
        hole_side = "right"
    else:
        return None

    if not np.all(grid[r1, c0 : c1 + 1] == GREY):
        return None
    for r in range(r0, r1):
        left_ok = grid[r, c0] == GREY
        right_ok = grid[r, c1] == GREY
        if r == hole_r:
            if hole_side == "left":
                if left_ok or not right_ok:
                    return None
            else:
                if right_ok or not left_ok:
                    return None
        elif not (left_ok and right_ok):
            return None
    if r0 > 0 and (grid[r0 - 1, c0] == GREY or grid[r0 - 1, c1] == GREY):
        return None

    if np.any(grid[r0:r1, c0 + 1 : c1] == GREY):
        return None

    inner = grid[r0:r1, c0 + 1 : c1]
    if inner.size == 0:
        return None

    liquid_mask = inner != BG
    if not liquid_mask.any():
        return None
    colors = np.unique(inner[liquid_mask])
    if colors.size != 1:
        return None
    color = int(colors[0])
    inner_w = inner.shape[1]
    area = int(liquid_mask.sum())
    if area % inner_w != 0:
        return None
    liquid_h = area // inner_w
    expected_liquid = np.zeros_like(liquid_mask, dtype=bool)
    expected_liquid[-liquid_h:, :] = True
    if not np.array_equal(liquid_mask, expected_liquid):
        return None

    liquid_top = r1 - liquid_h
    if hole_r < liquid_top:
        return None

    liquid_coords = [(r, c) for r in range(liquid_top, r1) for c in range(c0 + 1, c1)]
    return {
        "bbox": (r0, c0, r1, c1),
        "hole": (hole_r, hole_c),
        "hole_side": hole_side,
        "color": color,
        "inner_width": inner_w,
        "liquid_height": liquid_h,
        "liquid_coords": liquid_coords,
    }


def _find_containers(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    containers = []
    for r1 in range(h):
        c = 0
        while c < w:
            if grid[r1, c] != GREY:
                c += 1
                continue
            c0 = c
            while c + 1 < w and grid[r1, c + 1] == GREY:
                c += 1
            c1 = c
            if c1 - c0 + 1 >= 3:
                r = r1 - 1
                while r >= 0 and (grid[r, c0] == GREY or grid[r, c1] == GREY):
                    r -= 1
                r0 = r + 1
                if r0 < r1:
                    counts = [int(grid[rr, c0] == GREY) + int(grid[rr, c1] == GREY) for rr in range(r0, r1)]
                    if counts and counts[0] == 2 and all(cnt >= 1 for cnt in counts) and sum(cnt == 1 for cnt in counts) == 1:
                        hole_r = next(rr for rr in range(r0, r1) if counts[rr - r0] == 1)
                        hole_c = c0 if grid[hole_r, c0] != GREY else c1
                        parsed = _parse_container(grid, r0, c0, r1, c1, hole_r, hole_c)
                        if parsed is not None:
                            containers.append(parsed)
            c += 1
    containers.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return containers


def _landing_bottom(grid: np.ndarray, start_r: int, drop_c: int) -> int:
    h = grid.shape[0]
    for r in range(start_r + 1, h):
        if grid[r, drop_c] == GREY:
            return r - 1
    return h - 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Empty each leaky U-shaped grey container and rebuild the liquid as a puddle
    beneath the hole, resting on the first grey platform below or on the grid bottom.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    containers = _find_containers(g)
    if not containers:
        return out

    for container in containers:
        for r, c in container["liquid_coords"]:
            out[r, c] = BG

    for container in containers:
        hole_r, hole_c = container["hole"]
        width = int(container["inner_width"])
        height = int(container["liquid_height"])
        drop_c = hole_c - 1 if container["hole_side"] == "left" else hole_c + 1
        assert 0 <= drop_c < g.shape[1]
        bottom = _landing_bottom(g, hole_r, drop_c)

        left = hole_c - width // 2
        right = left + width - 1
        top = bottom - height + 1
        assert 0 <= left <= right < g.shape[1]
        assert 0 <= top <= bottom < g.shape[0]
        assert not np.any(g[top : bottom + 1, left : right + 1] == GREY)

        out[top : bottom + 1, left : right + 1] = container["color"]

    return out


def _expected_output(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    h = grid.shape[0]
    for spec in specs:
        bottom_r = spec["bottom_r"]
        out[bottom_r - spec["liquid_h"] : bottom_r, spec["c0"] + 1 : spec["c1"]] = BG
    for spec in specs:
        bottom = spec["platform_r"] - 1 if spec["platform_r"] is not None else h - 1
        top = bottom - spec["liquid_h"] + 1
        out[top : bottom + 1, spec["puddle_left"] : spec["puddle_right"] + 1] = spec["color"]
    return out


def _sample_specs(rng: np.random.Generator) -> list[dict]:
    n = int(rng.integers(MIN_CONTAINERS, MAX_CONTAINERS + 1))
    colors = rng.choice(LIQUID_COLORS, size=n, replace=False)
    use_platform = rng.random(n) < 0.5
    if n > 1 and np.all(use_platform == use_platform[0]):
        use_platform[int(rng.integers(0, n))] = not bool(use_platform[0])

    specs: list[dict] = []
    for i in range(n):
        inner_w = int(rng.choice(INNER_WIDTH_CHOICES))
        interior_h = int(rng.integers(*INTERIOR_HEIGHT_RANGE))
        liquid_h = int(rng.integers(1, interior_h + 1))
        side = str(rng.choice(["left", "right"]))
        hole_depth = int(rng.integers(1, min(liquid_h, interior_h - 1) + 1))
        outer_w = inner_w + 2
        outer_h = interior_h + 1
        band_w = outer_w + inner_w // 2
        specs.append(
            {
                "color": int(colors[i]),
                "inner_w": inner_w,
                "interior_h": interior_h,
                "liquid_h": liquid_h,
                "side": side,
                "hole_depth": hole_depth,
                "outer_w": outer_w,
                "outer_h": outer_h,
                "band_w": band_w,
                "use_platform": bool(use_platform[i]),
            }
        )
    return specs


def _place_horizontal_layout(rng: np.random.Generator, specs: list[dict]):
    left_margin = int(rng.integers(*BASE_SIDE_MARGIN_RANGE))
    right_margin = int(rng.integers(*BASE_SIDE_MARGIN_RANGE))
    gaps = [int(rng.integers(*BASE_GAP_RANGE)) for _ in range(max(0, len(specs) - 1))]

    min_w = left_margin + right_margin + sum(spec["band_w"] for spec in specs) + sum(gaps)
    if min_w > MAX_GRID_W:
        return None

    extra_w = int(rng.integers(0, min(MAX_EXTRA_WIDTH, MAX_GRID_W - min_w) + 1))
    slots = [left_margin, *gaps, right_margin]
    for _ in range(extra_w):
        slots[int(rng.integers(0, len(slots)))] += 1

    cursor = slots[0]
    for i, spec in enumerate(specs):
        band_left = cursor
        band_right = band_left + spec["band_w"] - 1
        if spec["side"] == "left":
            hole_c = band_left + spec["inner_w"] // 2
            c0 = hole_c
            c1 = c0 + spec["outer_w"] - 1
        else:
            hole_c = band_right - spec["inner_w"] // 2
            c1 = hole_c
            c0 = c1 - spec["outer_w"] + 1
        puddle_left = hole_c - spec["inner_w"] // 2
        puddle_right = puddle_left + spec["inner_w"] - 1
        spec["c0"] = c0
        spec["c1"] = c1
        spec["hole_c"] = hole_c
        spec["puddle_left"] = puddle_left
        spec["puddle_right"] = puddle_right
        cursor = band_right + 1
        if i < len(specs) - 1:
            cursor += slots[i + 1]

    width = slots[0] + slots[-1] + sum(spec["band_w"] for spec in specs) + sum(slots[1:-1])
    return width


def _place_vertical_layout(rng: np.random.Generator, specs: list[dict]):
    required_h = max(
        spec["outer_h"] + spec["liquid_h"] + (3 if spec["use_platform"] else 1) for spec in specs
    )
    if required_h > MAX_GRID_H:
        return None
    height = required_h + int(rng.integers(0, min(MAX_EXTRA_HEIGHT, MAX_GRID_H - required_h) + 1))

    for spec in specs:
        slack = 2 if spec["use_platform"] else 0
        top_max = height - spec["outer_h"] - spec["liquid_h"] - slack
        if top_max < 1:
            return None
        top_r = int(rng.integers(1, top_max + 1))
        bottom_r = top_r + spec["outer_h"] - 1
        hole_r = bottom_r - spec["hole_depth"]

        platform_r = None
        platform_pad_left = 0
        platform_pad_right = 0
        if spec["use_platform"]:
            platform_min = bottom_r + spec["liquid_h"] + 1
            platform_max = height - 2
            if platform_min > platform_max:
                return None
            platform_r = int(rng.integers(platform_min, platform_max + 1))
            platform_pad_left = int(rng.integers(0, 2))
            platform_pad_right = int(rng.integers(0, 2))

        spec["top_r"] = top_r
        spec["bottom_r"] = bottom_r
        spec["hole_r"] = hole_r
        spec["platform_r"] = platform_r
        spec["platform_pad_left"] = platform_pad_left
        spec["platform_pad_right"] = platform_pad_right

    return height


def _render_input(height: int, width: int, specs: list[dict]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for spec in specs:
        top_r = spec["top_r"]
        bottom_r = spec["bottom_r"]
        c0 = spec["c0"]
        c1 = spec["c1"]
        grid[top_r : bottom_r + 1, c0] = GREY
        grid[top_r : bottom_r + 1, c1] = GREY
        grid[bottom_r, c0 : c1 + 1] = GREY
        grid[spec["hole_r"], spec["hole_c"]] = BG

        liquid_top = bottom_r - spec["liquid_h"]
        grid[liquid_top:bottom_r, c0 + 1 : c1] = spec["color"]

        if spec["platform_r"] is not None:
            left = max(0, spec["puddle_left"] - spec["platform_pad_left"])
            right = min(width - 1, spec["puddle_right"] + spec["platform_pad_right"])
            grid[spec["platform_r"], left : right + 1] = GREY
    return grid


def _valid_specs(height: int, width: int, specs: list[dict]) -> bool:
    puddles = []
    for spec in specs:
        if not (0 <= spec["c0"] < spec["c1"] < width):
            return False
        if not (0 <= spec["puddle_left"] <= spec["puddle_right"] < width):
            return False
        if not (0 <= spec["top_r"] < spec["bottom_r"] < height):
            return False
        if not (spec["top_r"] < spec["hole_r"] < spec["bottom_r"]):
            return False

        bottom = spec["platform_r"] - 1 if spec["platform_r"] is not None else height - 1
        puddle_top = bottom - spec["liquid_h"] + 1
        if puddle_top <= spec["bottom_r"]:
            return False
        rect = (puddle_top, bottom, spec["puddle_left"], spec["puddle_right"])
        for other in puddles:
            if not (
                rect[1] < other[0]
                or other[1] < rect[0]
                or rect[3] < other[2]
                or other[3] < rect[2]
            ):
                return False
        puddles.append(rect)
    return True


def _gen_attempt(rng: np.random.Generator):
    specs = _sample_specs(rng)
    width = _place_horizontal_layout(rng, specs)
    if width is None:
        return None
    height = _place_vertical_layout(rng, specs)
    if height is None:
        return None
    if not _valid_specs(height, width, specs):
        return None

    grid = _render_input(height, width, specs)
    containers = _find_containers(grid)
    if len(containers) != len(specs):
        return None

    expected = _expected_output(grid, specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if np.array_equal(grid, actual):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
