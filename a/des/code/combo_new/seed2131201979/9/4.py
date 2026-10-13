import numpy as np
from collections import deque


SHADOW_COLOR = 5
BG_COLOR = 0
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRECTION_STEPS = {
    "left": (0, -1),
    "down": (1, 0),
    "right": (0, 1),
}
GEN_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _direction_from_col(col: int, width: int) -> str:
    if 3 * col < width:
        return "left"
    if 3 * col >= 2 * width:
        return "right"
    return "down"


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each colored stamp into a grey shadow according to a top-edge direction legend.

    Input structure:
    - The background is the dominant color.
    - The top row contains isolated single-pixel color bars; each bar color matches one interior object color.
    - Each bar's horizontal zone encodes a direction: left third means project left, middle third means project downward, right third means project right.
    - Inside the grid there are several separated monochrome 4-connected objects that do not touch the top legend row.

    Rule:
    - Keep every original object and every top-row legend pixel unchanged.
    - Match each interior object to the top-row bar of the same color and decode its projection direction from that bar's column.
    - From every pixel of the object, march cell by cell in that direction; same-object pixels are transparent, but a different object stops the march.
    - Recolor every background cell visited by those marches to grey, and leave already-grey shadow cells unchanged.

    Key insight:
    - The top-edge color bars are a direction code rather than extra objects: their left/middle/right placement determines how the matching colored object casts its shadow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _background_color(g)

    top_colors = [(int(c), int(g[0, c])) for c in range(w) if int(g[0, c]) != bg]
    assert top_colors, "expected at least one top-edge direction bar"

    color_to_direction: dict[int, str] = {}
    for col, color in top_colors:
        assert color not in color_to_direction, "each legend color must appear exactly once"
        color_to_direction[color] = _direction_from_col(col, w)

    interior = g.copy()
    interior[0, :] = bg
    object_colors = sorted(set(int(v) for v in np.unique(interior)) - {bg})
    assert object_colors, "expected interior objects"
    assert set(object_colors) == set(color_to_direction), "legend colors must match interior object colors"

    occupancy = np.full((h, w), -1, dtype=int)
    objects: dict[int, list[tuple[int, int]]] = {}
    for color in object_colors:
        mask = interior == color
        comps = _components4(mask)
        assert len(comps) == 1, "each object color must form exactly one 4-connected component"
        coords = comps[0]
        objects[color] = coords
        for r, c in coords:
            occupancy[r, c] = color

    out = g.copy()
    for color, coords in objects.items():
        dr, dc = DIRECTION_STEPS[color_to_direction[color]]
        for r, c in coords:
            rr = r + dr
            cc = c + dc
            while 0 <= rr < h and 0 <= cc < w:
                occ = int(occupancy[rr, cc])
                if occ == color:
                    rr += dr
                    cc += dc
                    continue
                if occ != -1:
                    break
                if int(out[rr, cc]) == bg:
                    out[rr, cc] = SHADOW_COLOR
                rr += dr
                cc += dc

    return out


def _normalize_cells(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    min_r = min(rows)
    min_c = min(cols)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _random_shape(rng: np.random.Generator) -> list[tuple[int, int]]:
    for _ in range(100):
        target = int(rng.integers(2, 7))
        cells = {(0, 0)}
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if (nr, nc) in cells:
                        continue
                    trial = cells | {(nr, nc)}
                    rows = [x for x, _ in trial]
                    cols = [y for _, y in trial]
                    if max(rows) - min(rows) + 1 <= 4 and max(cols) - min(cols) + 1 <= 4:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
        if len(cells) < 2:
            continue
        shape = cells
        if rng.random() < 0.5:
            shape = {(c, r) for r, c in shape}
        if rng.random() < 0.5:
            shape = {(-r, c) for r, c in shape}
        if rng.random() < 0.5:
            shape = {(r, -c) for r, c in shape}
        norm = _normalize_cells(shape)
        h = 1 + max(r for r, _ in norm)
        w = 1 + max(c for _, c in norm)
        if h <= 4 and w <= 4:
            return norm
    return [(0, 0), (0, 1)]


def _coords_with_offset(shape: list[tuple[int, int]], top: int, left: int) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in shape]


def _can_place(coords: list[tuple[int, int]], forbidden: np.ndarray) -> bool:
    h, w = forbidden.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if forbidden[r, c]:
            return False
    return True


def _mark_forbidden(forbidden: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = forbidden.shape
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    forbidden[nr, nc] = True


def _zone_columns(width: int, direction: str) -> list[int]:
    return [c for c in range(width) if _direction_from_col(c, width) == direction]


def _analyze_stop_types(objects: list[dict[str, object]], h: int, w: int) -> tuple[bool, bool]:
    occupancy = np.full((h, w), -1, dtype=int)
    for obj in objects:
        color = int(obj["color"])
        for r, c in obj["coords"]:
            occupancy[r, c] = color

    hit_other = False
    hit_boundary = False
    for obj in objects:
        color = int(obj["color"])
        dr, dc = DIRECTION_STEPS[str(obj["direction"])]
        for r, c in obj["coords"]:
            rr = r + dr
            cc = c + dc
            painted_any = False
            while 0 <= rr < h and 0 <= cc < w:
                occ = int(occupancy[rr, cc])
                if occ == color:
                    rr += dr
                    cc += dc
                    continue
                if occ != -1:
                    if painted_any:
                        hit_other = True
                    break
                painted_any = True
                rr += dr
                cc += dc
            else:
                if painted_any:
                    hit_boundary = True
    return hit_other, hit_boundary


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(11, 19))
    w = int(rng.integers(12, 20))
    n_objects = int(rng.integers(3, 5))

    directions = ["left", "down", "right"]
    if n_objects == 4:
        directions.append(str(rng.choice(directions)))
    rng.shuffle(directions)

    colors = [int(v) for v in rng.choice(np.array(COLOR_POOL, dtype=int), size=n_objects, replace=False)]
    forbidden = np.zeros((h, w), dtype=bool)
    forbidden[0, :] = True
    forbidden[1, :] = True
    objects: list[dict[str, object]] = []

    for color, direction in zip(colors, directions):
        placed = False
        for _ in range(120):
            shape = _random_shape(rng)
            sh = 1 + max(r for r, _ in shape)
            sw = 1 + max(c for _, c in shape)
            top = int(rng.integers(2, h - sh + 1))
            left = int(rng.integers(0, w - sw + 1))
            coords = _coords_with_offset(shape, top, left)
            min_c = min(c for _, c in coords)
            max_c = max(c for _, c in coords)
            max_r = max(r for r, _ in coords)
            if direction == "left" and min_c < 1:
                continue
            if direction == "right" and max_c > w - 2:
                continue
            if direction == "down" and max_r > h - 2:
                continue
            if not _can_place(coords, forbidden):
                continue
            objects.append(
                {
                    "color": color,
                    "direction": direction,
                    "coords": coords,
                }
            )
            _mark_forbidden(forbidden, coords)
            placed = True
            break
        if not placed:
            return None

    zone_counts = {direction: directions.count(direction) for direction in DIRECTION_STEPS}
    color_to_bar_col: dict[int, int] = {}
    for direction, need in zone_counts.items():
        cols = _zone_columns(w, direction)
        if len(cols) < need:
            return None
        chosen = [int(v) for v in rng.choice(np.array(cols, dtype=int), size=need, replace=False)]
        rng.shuffle(chosen)
        idx = 0
        for obj in objects:
            if obj["direction"] == direction:
                color_to_bar_col[int(obj["color"])] = chosen[idx]
                idx += 1

    grid = np.full((h, w), BG_COLOR, dtype=int)
    for obj in objects:
        color = int(obj["color"])
        for r, c in obj["coords"]:
            grid[r, c] = color
        grid[0, color_to_bar_col[color]] = color

    hit_other, hit_boundary = _analyze_stop_types(objects, h, w)
    if not (hit_other and hit_boundary):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.any(out == SHADOW_COLOR):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-shadow ARC input")
