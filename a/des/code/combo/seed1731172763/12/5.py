import numpy as np


BG_COLOR = 8
PALETTE = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (11, 24)
GRID_W_RANGE = (11, 24)
MAX_RADIUS = 4
MAX_GENERATION_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 200


def _background_color(grid: np.ndarray) -> int:
    if bool((grid == BG_COLOR).any()):
        return BG_COLOR
    return int(np.bincount(grid.reshape(-1)).argmax())


def _collect_component(mask: np.ndarray, start: tuple[int, int], seen: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    stack = [start]
    seen[start] = True
    pts = [start]

    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and not bool(seen[rr, cc]):
                seen[rr, cc] = True
                stack.append((rr, cc))
                pts.append((rr, cc))

    return pts


def _extract_crosses(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)
    non_bg = g != bg
    seen = np.zeros((h, w), dtype=bool)
    crosses: list[tuple[int, int, int, int]] = []

    for r, c in np.argwhere(non_bg):
        r = int(r)
        c = int(c)
        if bool(seen[r, c]):
            continue

        comp = _collect_component(non_bg, (r, c), seen)
        comp_set = set(comp)

        centers = []
        for rr, cc in comp:
            deg = 0
            for dr, dc in DIR4:
                if (rr + dr, cc + dc) in comp_set:
                    deg += 1
            if deg == 4:
                centers.append((rr, cc))

        assert len(centers) == 1, "each component must contain exactly one cross center"
        cr, cc = centers[0]
        body_color = int(g[cr, cc])

        expected = {(cr, cc)}
        radius = None
        endpoint_color = None

        for dr, dc in DIR4:
            ray = []
            rr = cr + dr
            cc2 = cc + dc
            while 0 <= rr < h and 0 <= cc2 < w and (rr, cc2) in comp_set:
                ray.append((rr, cc2))
                rr += dr
                cc2 += dc

            assert ray, "each arm must have positive length"
            if radius is None:
                radius = len(ray)
            else:
                assert len(ray) == radius, "all four arms must have equal length"

            for rr, cc2 in ray[:-1]:
                assert int(g[rr, cc2]) == body_color, "arm body must match the center color"

            ray_endpoint_color = int(g[ray[-1]])
            assert ray_endpoint_color != body_color, "endpoint color must differ from body color"
            assert ray_endpoint_color != bg, "endpoint color must be non-background"
            if endpoint_color is None:
                endpoint_color = ray_endpoint_color
            else:
                assert ray_endpoint_color == endpoint_color, "all endpoints must share one color"

            expected.update(ray)

        assert radius is not None and endpoint_color is not None
        assert expected == comp_set, "component must be exactly one thin cross"
        crosses.append((cr, cc, int(radius), int(endpoint_color)))

    assert crosses, "expected at least one cross"
    return bg, crosses


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify each thin cross and replace it with a filled square centered at the
    same location. If a cross arm has radius R (endpoint is R steps from the
    center), the output square has side 2R+1 and uses the shared endpoint color.
    """
    g = np.asarray(grid, dtype=int)
    bg, crosses = _extract_crosses(g)

    out = np.full(g.shape, bg, dtype=int)
    for cr, cc, radius, fill_color in crosses:
        r0 = cr - radius
        r1 = cr + radius + 1
        c0 = cc - radius
        c1 = cc + radius + 1
        assert np.all(out[r0:r1, c0:c1] == bg), "output squares must not overlap"
        out[r0:r1, c0:c1] = fill_color
    return out


def _sample_radii(rng: np.random.Generator, count: int, radius_max: int) -> list[int]:
    allowed = [r for r in range(1, min(MAX_RADIUS, radius_max) + 1)]
    assert allowed

    radii = [int(rng.choice(allowed)) for _ in range(count)]
    larger = [r for r in allowed if r >= 2]
    if count >= 2 and larger and rng.random() < 0.5:
        radii[0] = 1
        radii[1] = int(rng.choice(larger))
    elif count == 1 and larger and rng.random() < 0.5:
        radii[0] = int(rng.choice(larger))

    rng.shuffle(radii)
    radii.sort(reverse=True)
    return radii


def _bbox_is_free(reserved: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    h, w = reserved.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _reserve_bbox(reserved: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = reserved.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _draw_cross(
    grid: np.ndarray,
    center: tuple[int, int],
    radius: int,
    body_color: int,
    endpoint_color: int,
) -> None:
    cr, cc = center
    grid[cr, cc] = body_color
    for dr, dc in DIR4:
        for step in range(1, radius):
            grid[cr + dr * step, cc + dc * step] = body_color
        grid[cr + dr * radius, cc + dc * radius] = endpoint_color


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    radius_max = min(MAX_RADIUS, (h - 1) // 2, (w - 1) // 2)
    if radius_max < 1:
        return None

    area = h * w
    max_objects = 2
    if area >= 180:
        max_objects = 3
    if area >= 260:
        max_objects = 4
    count = int(rng.integers(1, max_objects + 1))

    if 2 * count > len(PALETTE):
        return None

    colors = list(map(int, rng.permutation(PALETTE)))
    body_colors = colors[:count]
    endpoint_colors = colors[count : 2 * count]
    radii = _sample_radii(rng, count, radius_max)

    grid = np.full((h, w), BG_COLOR, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    for body_color, endpoint_color, radius in zip(body_colors, endpoint_colors, radii):
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            cr = int(rng.integers(radius, h - radius))
            cc = int(rng.integers(radius, w - radius))
            bbox = (cr - radius, cc - radius, cr + radius, cc + radius)
            pad = 1 if rng.random() < 0.8 else 0
            if not _bbox_is_free(reserved, bbox, pad=pad):
                continue
            _reserve_bbox(reserved, bbox, pad=pad)
            _draw_cross(grid, (cr, cc), radius, body_color, endpoint_color)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
        assert out.shape == grid.shape
        assert not np.array_equal(out, grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cross-to-square ARC input")
