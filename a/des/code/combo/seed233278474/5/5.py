import random

import numpy as np


BG_COLOR = 0
TEMPLATE_COLOR = 8
MARKER_PALETTE = (1, 2, 3, 4, 5, 6, 7, 9)
QUADRANTS = ("tl", "tr", "bl", "br")
DIAGONALS = ("tlbr", "trbl")
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_H_RANGE = (18, 32)
GRID_W_RANGE = (18, 32)
RECT_COUNT_CHOICES = (1, 2, 2, 3, 3, 4)
QUAD_SIZE_RANGE = (2, 5)
PLACEMENT_RETRIES = 120
BLOB_RETRIES = 120
GEN_RETRIES = 400


def _source_to_tl(mask: np.ndarray, quadrant: str) -> np.ndarray:
    if quadrant == "tl":
        return mask.copy()
    if quadrant == "tr":
        return np.fliplr(mask)
    if quadrant == "bl":
        return np.flipud(mask)
    if quadrant == "br":
        return np.rot90(mask, 2)
    raise ValueError(f"unknown quadrant: {quadrant}")


def _compose_inner_from_tl(tl_mask: np.ndarray, color: int) -> np.ndarray:
    qh, qw = tl_mask.shape
    inner = np.zeros((qh * 2, qw * 2), dtype=int)
    inner[:qh, :qw] = np.where(tl_mask, color, BG_COLOR)
    inner[:qh, qw:] = np.where(np.fliplr(tl_mask), color, BG_COLOR)
    inner[qh:, :qw] = np.where(np.flipud(tl_mask), color, BG_COLOR)
    inner[qh:, qw:] = np.where(np.rot90(tl_mask, 2), color, BG_COLOR)
    return inner


def _paint_source_quadrant(inner: np.ndarray, source_mask: np.ndarray, quadrant: str, color: int) -> None:
    qh, qw = source_mask.shape
    if quadrant == "tl":
        quad = inner[:qh, :qw]
    elif quadrant == "tr":
        quad = inner[:qh, qw:]
    elif quadrant == "bl":
        quad = inner[qh:, :qw]
    elif quadrant == "br":
        quad = inner[qh:, qw:]
    else:
        raise ValueError(f"unknown quadrant: {quadrant}")
    quad[source_mask] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each marker-defined rectangle and mirror the azure template into all four quadrants.

    Each non-black, non-azure color appears exactly twice as single-pixel markers.
    Those two pixels are opposite corners of a rectangle whose interior is split into
    four equal quadrants. Exactly one quadrant contains an azure template; the output
    replaces that template by the marker color and fills the other three quadrants by
    horizontal, vertical, and 180-degree symmetry.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    colors, counts = np.unique(g, return_counts=True)
    marker_colors = [
        int(color)
        for color, count in zip(colors, counts)
        if int(color) not in (BG_COLOR, TEMPLATE_COLOR) and int(count) == 2
    ]

    for color in marker_colors:
        pts = np.argwhere(g == color)
        assert pts.shape == (2, 2)
        (r1, c1), (r2, c2) = pts
        top, bottom = sorted((int(r1), int(r2)))
        left, right = sorted((int(c1), int(c2)))
        assert top < bottom and left < right

        inner = g[top + 1 : bottom, left + 1 : right]
        inner_h, inner_w = inner.shape
        assert inner_h > 0 and inner_w > 0
        assert inner_h % 2 == 0 and inner_w % 2 == 0

        qh, qw = inner_h // 2, inner_w // 2
        quadrants = {
            "tl": inner[:qh, :qw],
            "tr": inner[:qh, qw:],
            "bl": inner[qh:, :qw],
            "br": inner[qh:, qw:],
        }

        source = []
        for quadrant, block in quadrants.items():
            mask = block == TEMPLATE_COLOR
            if mask.any():
                source.append((quadrant, mask))
        assert len(source) == 1

        source_quadrant, source_mask = source[0]
        tl_mask = _source_to_tl(source_mask, source_quadrant)
        out[top + 1 : bottom, left + 1 : right] = _compose_inner_from_tl(tl_mask, color)

    return out


def _random_blob_mask(qh: int, qw: int, rng: random.Random) -> np.ndarray | None:
    area = qh * qw
    min_cells = min(3, area - 1)
    max_cells = area - 1
    if max_cells < 2:
        return None

    for _ in range(BLOB_RETRIES):
        target = rng.randint(min_cells, max_cells)
        cells = {(rng.randrange(qh), rng.randrange(qw))}

        while len(cells) < target:
            frontier = []
            for r, c in cells:
                neighbors = []
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < qh and 0 <= nc < qw and (nr, nc) not in cells:
                        neighbors.append((nr, nc))
                if neighbors:
                    frontier.append(neighbors)
            if not frontier:
                break
            cells.add(rng.choice(rng.choice(frontier)))

        if len(cells) != target:
            continue

        mask = np.zeros((qh, qw), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if mask.all():
            continue
        if mask.any(axis=0).sum() < 2 or mask.any(axis=1).sum() < 2:
            continue
        if np.array_equal(mask, np.fliplr(mask)) and np.array_equal(mask, np.flipud(mask)):
            continue
        return mask

    return None


def _place_rectangles(
    h: int, w: int, num_rectangles: int, rng: random.Random
) -> list[dict[str, object]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    specs: list[dict[str, object]] = []
    colors = rng.sample(MARKER_PALETTE, num_rectangles)

    for color in colors:
        qh = rng.randint(*QUAD_SIZE_RANGE)
        qw = rng.randint(*QUAD_SIZE_RANGE)
        full_h = qh * 2 + 2
        full_w = qw * 2 + 2
        if h < full_h + 2 or w < full_w + 2:
            return None

        source_mask = _random_blob_mask(qh, qw, rng)
        if source_mask is None:
            return None

        quadrant = rng.choice(QUADRANTS)
        diagonal = rng.choice(DIAGONALS)
        placed = False

        for _ in range(PLACEMENT_RETRIES):
            top = rng.randint(1, h - full_h - 1)
            left = rng.randint(1, w - full_w - 1)
            r0 = max(0, top - 1)
            r1 = min(h, top + full_h + 1)
            c0 = max(0, left - 1)
            c1 = min(w, left + full_w + 1)
            if occupied[r0:r1, c0:c1].any():
                continue

            occupied[r0:r1, c0:c1] = True
            specs.append(
                {
                    "color": color,
                    "top": top,
                    "left": left,
                    "qh": qh,
                    "qw": qw,
                    "source_mask": source_mask,
                    "quadrant": quadrant,
                    "diagonal": diagonal,
                }
            )
            placed = True
            break

        if not placed:
            return None

    return specs


def _render_pair(specs: list[dict[str, object]], h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    input_grid = np.zeros((h, w), dtype=int)
    output_grid = np.zeros((h, w), dtype=int)

    for spec in specs:
        color = int(spec["color"])
        top = int(spec["top"])
        left = int(spec["left"])
        qh = int(spec["qh"])
        qw = int(spec["qw"])
        quadrant = str(spec["quadrant"])
        diagonal = str(spec["diagonal"])
        source_mask = np.asarray(spec["source_mask"], dtype=bool)

        bottom = top + qh * 2 + 1
        right = left + qw * 2 + 1
        if diagonal == "tlbr":
            markers = ((top, left), (bottom, right))
        else:
            markers = ((top, right), (bottom, left))

        for r, c in markers:
            input_grid[r, c] = color
            output_grid[r, c] = color

        inner = np.zeros((qh * 2, qw * 2), dtype=int)
        _paint_source_quadrant(inner, source_mask, quadrant, TEMPLATE_COLOR)
        input_grid[top + 1 : bottom, left + 1 : right] = inner

        tl_mask = _source_to_tl(source_mask, quadrant)
        output_grid[top + 1 : bottom, left + 1 : right] = _compose_inner_from_tl(tl_mask, color)

    return input_grid, output_grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_rectangles = rng.choice(RECT_COUNT_CHOICES)
    min_side = max(GRID_H_RANGE[0], 8 + num_rectangles * 4)
    h = rng.randint(min_side, GRID_H_RANGE[1])
    w = rng.randint(min_side, GRID_W_RANGE[1])

    specs = _place_rectangles(h, w, num_rectangles, rng)
    if specs is None:
        return None

    input_grid, expected = _render_pair(specs, h, w)
    if int((input_grid == TEMPLATE_COLOR).sum()) == 0:
        return None

    try:
        actual = main(input_grid)
    except Exception:
        return None

    if np.array_equal(actual, input_grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
