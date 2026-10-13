from collections import Counter, deque

import numpy as np


BG = 8
SIDES = ("top", "bottom", "left", "right")
PALETTE = tuple(color for color in range(10) if color != BG)

GRID_H_BOUNDS = (16, 26)
GRID_W_BOUNDS = (16, 26)
RECT_H_BOUNDS = (2, 6)
RECT_W_BOUNDS = (2, 7)
EXTRA_OBJECT_BOUNDS = (0, 2)
MAX_GEN_ATTEMPTS = 300


def _randint_closed(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _side_colors(grid: np.ndarray) -> dict[str, int]:
    slices = {
        "top": grid[0, 1:-1],
        "bottom": grid[-1, 1:-1],
        "left": grid[1:-1, 0],
        "right": grid[1:-1, -1],
    }
    result: dict[str, int] = {}
    for side, cells in slices.items():
        colors = [int(value) for value in np.asarray(cells).reshape(-1)]
        assert colors, f"{side} border must have non-corner cells"
        color, _ = Counter(colors).most_common(1)[0]
        result[side] = int(color)
    assert len(set(result.values())) == 4, "expected four distinct border colors"
    return result


def _find_rectangles(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h - 2, w - 2), dtype=bool)
    objects: list[dict[str, object]] = []

    for ir in range(h - 2):
        for ic in range(w - 2):
            if seen[ir, ic]:
                continue
            color = int(grid[ir + 1, ic + 1])
            if color == BG:
                continue

            seen[ir, ic] = True
            queue = deque([(ir, ic)])
            coords: list[tuple[int, int]] = []

            while queue:
                cr, cc = queue.popleft()
                gr, gc = cr + 1, cc + 1
                coords.append((gr, gc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h - 2 and 0 <= nc < w - 2):
                        continue
                    if seen[nr, nc] or int(grid[nr + 1, nc + 1]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            assert np.all(sub == color), "objects must be solid rectangles"
            area = len(coords)
            rect_h = r1 - r0 + 1
            rect_w = c1 - c0 + 1
            assert area == rect_h * rect_w, "foreground component must fill its bbox"
            objects.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "shape": (rect_h, rect_w),
                    "area": area,
                }
            )

    assert objects, "expected at least one interior rectangle"
    return objects


def _nearest_side_or_none(bbox: tuple[int, int, int, int], h: int, w: int) -> str | None:
    r0, c0, r1, c1 = bbox
    distances = {
        "top": int(r0),
        "bottom": int(h - 1 - r1),
        "left": int(c0),
        "right": int(w - 1 - c1),
    }
    best = min(distances.values())
    winners = [side for side, dist in distances.items() if dist == best]
    return winners[0] if len(winners) == 1 else None


def _nearest_side(bbox: tuple[int, int, int, int], h: int, w: int) -> str:
    side = _nearest_side_or_none(bbox, h, w)
    assert side is not None, "nearest border must be unique"
    return side


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only rectangles whose color matches the nearest border, then center-stack the survivors by area.

    Input structure:
    - The background is azure.
    - The outermost row/column form four 1-pixel border strips, one distinct color per side.
    - Inside the border are separated solid monochrome rectangles on the azure background.

    Rule:
    - Ignore the border and detect every interior rectangle as one monochrome connected component.
    - For each rectangle, compare its top/bottom/left/right distances to the grid edges and find the unique nearest side.
    - Keep the rectangle iff its color equals that nearest side's border color; delete all other rectangles.
    - Start a same-sized azure output with no border, sort kept rectangles by area from large to small, and paint each one centered on the canvas so later smaller rectangles overwrite earlier larger ones.

    Key insight:
    - The border colors encode a spatial filter: an object survives only when its color agrees with the border it is closest to, and the surviving rectangles are then reassembled by centered area-ranked stacking.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 5 and w >= 5, "grid must have an interior"

    border = _side_colors(g)
    objects = _find_rectangles(g)

    kept = []
    for obj in objects:
        side = _nearest_side(obj["bbox"], h, w)
        if int(obj["color"]) == int(border[side]):
            kept.append(obj)

    kept.sort(key=lambda obj: (-int(obj["area"]), int(obj["bbox"][0]), int(obj["bbox"][1])))

    out = np.full((h, w), BG, dtype=int)
    for obj in kept:
        rect_h, rect_w = obj["shape"]
        r0 = (h - rect_h) // 2
        c0 = (w - rect_w) // 2
        out[r0 : r0 + rect_h, c0 : c0 + rect_w] = int(obj["color"])
    return out


def _paint_border(grid: np.ndarray, border_colors: dict[str, int]) -> None:
    grid[0, :] = int(border_colors["top"])
    grid[-1, :] = int(border_colors["bottom"])
    grid[1:-1, 0] = int(border_colors["left"])
    grid[1:-1, -1] = int(border_colors["right"])


def _sample_unique_dims(
    rng: np.random.Generator, count: int
) -> list[tuple[int, int]]:
    candidates = [
        (h, w)
        for h in range(RECT_H_BOUNDS[0], RECT_H_BOUNDS[1] + 1)
        for w in range(RECT_W_BOUNDS[0], RECT_W_BOUNDS[1] + 1)
    ]
    rng.shuffle(candidates)
    used_areas: set[int] = set()
    dims: list[tuple[int, int]] = []
    for rect_h, rect_w in candidates:
        area = rect_h * rect_w
        if area in used_areas:
            continue
        dims.append((rect_h, rect_w))
        used_areas.add(area)
        if len(dims) == count:
            return dims
    raise RuntimeError("not enough unique rectangle areas available")


def _candidate_positions(
    h: int,
    w: int,
    rect_h: int,
    rect_w: int,
    side: str,
    occupied: np.ndarray,
) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - rect_h):
        r1 = r0 + rect_h - 1
        for c0 in range(1, w - rect_w):
            c1 = c0 + rect_w - 1
            if _nearest_side_or_none((r0, c0, r1, c1), h, w) != side:
                continue
            rr0 = max(1, r0 - 1)
            rr1 = min(h - 2, r1 + 1)
            cc0 = max(1, c0 - 1)
            cc1 = min(w - 2, c1 + 1)
            if np.any(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1]):
                continue
            positions.append((r0, c0))
    return positions


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = _randint_closed(rng, *GRID_H_BOUNDS)
    w = _randint_closed(rng, *GRID_W_BOUNDS)

    border_values = list(map(int, rng.choice(PALETTE, size=4, replace=False)))
    border_colors = dict(zip(SIDES, border_values))
    extra_colors = [color for color in PALETTE if color not in border_values]

    keep_count = int(rng.choice([2, 3]))
    keep_sides = set(rng.choice(SIDES, size=keep_count, replace=False).tolist())
    extra_count = _randint_closed(rng, *EXTRA_OBJECT_BOUNDS)

    removed_needed = sum(side not in keep_sides for side in SIDES) + extra_count
    if removed_needed > len(extra_colors):
        return None

    specs: list[dict[str, object]] = []
    for side in SIDES:
        keep = side in keep_sides
        specs.append({"side": side, "keep": keep})

    for _ in range(extra_count):
        specs.append({"side": str(rng.choice(SIDES)), "keep": False})

    removed_colors = rng.choice(extra_colors, size=removed_needed, replace=False).tolist()
    removed_iter = iter(int(color) for color in removed_colors)
    for spec in specs:
        if spec["keep"]:
            spec["color"] = int(border_colors[str(spec["side"])])
        else:
            spec["color"] = int(next(removed_iter))

    dims = _sample_unique_dims(rng, len(specs))
    rng.shuffle(dims)
    for spec, (rect_h, rect_w) in zip(specs, dims):
        spec["shape"] = (rect_h, rect_w)
        spec["area"] = rect_h * rect_w

    place_order = sorted(specs, key=lambda spec: (-int(spec["area"]), int(not bool(spec["keep"]))))
    occupied = np.zeros((h, w), dtype=bool)

    for spec in place_order:
        rect_h, rect_w = spec["shape"]
        positions = _candidate_positions(h, w, rect_h, rect_w, str(spec["side"]), occupied)
        if not positions:
            return None
        r0, c0 = positions[int(rng.integers(len(positions)))]
        r1 = r0 + rect_h - 1
        c1 = c0 + rect_w - 1
        spec["bbox"] = (r0, c0, r1, c1)
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True

    grid = np.full((h, w), BG, dtype=int)
    _paint_border(grid, border_colors)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(spec["color"])

    try:
        parsed = _find_rectangles(grid)
        if len(parsed) != len(specs):
            return None
        colors = [int(obj["color"]) for obj in parsed]
        if len(colors) != len(set(colors)):
            return None
        kept = [
            obj
            for obj in parsed
            if int(obj["color"]) == int(border_colors[_nearest_side(obj["bbox"], h, w)])
        ]
        if len(kept) < 2 or len(kept) == len(parsed):
            return None
        _ = main(grid)
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
