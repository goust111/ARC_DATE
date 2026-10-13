from __future__ import annotations

import numpy as np


BG = 0
MARKER_COLOR = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
OPENINGS = ("ul", "ur", "dl", "dr")
AREA_TO_SIZE = {3: 2, 5: 3}
MAX_GENERATION_ATTEMPTS = 400

_OPENING_TO_BEND = {
    "ul": lambda size: (size - 1, size - 1),
    "ur": lambda size: (size - 1, 0),
    "dl": lambda size: (0, size - 1),
    "dr": lambda size: (0, 0),
}
_OPENING_TO_VECTOR = {
    "ul": (-1, -1),
    "ur": (-1, 1),
    "dl": (1, -1),
    "dr": (1, 1),
}
_OPENING_TO_CORNER = {
    "ul": lambda h, w: (0, 0),
    "ur": lambda h, w: (0, w - 1),
    "dl": lambda h, w: (h - 1, 0),
    "dr": lambda h, w: (h - 1, w - 1),
}
_RNG = np.random.default_rng()


def _template_cells(size: int, opening: str) -> set[tuple[int, int]]:
    bend_r, bend_c = _OPENING_TO_BEND[opening](size)
    cells = {(bend_r, c) for c in range(size)}
    cells.update((r, bend_c) for r in range(size))
    return cells


TEMPLATES = {
    (size, opening): _template_cells(size, opening)
    for size in (2, 3)
    for opening in OPENINGS
}


def _component_cells(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BG:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            bbox = (min(rows), min(cols), max(rows), max(cols))
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "area": len(coords),
                }
            )

    return components


def _match_l_object(component: dict[str, object]) -> dict[str, object]:
    coords = component["coords"]
    assert isinstance(coords, list)
    area = int(component["area"])
    assert area in AREA_TO_SIZE, f"unexpected component area: {area}"

    r0, c0, r1, c1 = component["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    size = AREA_TO_SIZE[area]
    assert height == size and width == size, f"unexpected L bbox: {(height, width)}"

    rel = {(int(r) - r0, int(c) - c0) for r, c in coords}
    for opening in OPENINGS:
        if rel == TEMPLATES[(size, opening)]:
            return {
                "color": int(component["color"]),
                "area": area,
                "size": size,
                "opening": opening,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }

    raise AssertionError(f"component is not a supported L shape: {sorted(rel)}")


def _target_opening(grid: np.ndarray) -> str:
    h, w = grid.shape
    hits = []
    for opening in OPENINGS:
        r, c = _OPENING_TO_CORNER[opening](h, w)
        if int(grid[r, c]) == MARKER_COLOR:
            hits.append(opening)
    assert len(hits) == 1, f"expected one gray corner marker, got {hits}"
    return hits[0]


def _cells_with_bend(size: int, opening: str, bend: tuple[int, int]) -> set[tuple[int, int]]:
    bend_r, bend_c = _OPENING_TO_BEND[opening](size)
    abs_r, abs_c = bend
    cells = {(abs_r, abs_c + (c - bend_c)) for c in range(size)}
    cells.update((abs_r + (r - bend_r), abs_c) for r in range(size))
    return cells


def _next_bend(opening: str, bend: tuple[int, int]) -> tuple[int, int]:
    dr, dc = _OPENING_TO_VECTOR[opening]
    return bend[0] + dr, bend[1] + dc


def _render_nested(objects: list[dict[str, object]], opening: str) -> np.ndarray:
    painted: dict[tuple[int, int], int] = {}
    bend = (0, 0)

    for obj in objects:
        color = int(obj["color"])
        size = int(obj["size"])
        cells = _cells_with_bend(size, opening, bend)
        for cell in cells:
            assert cell not in painted, "nested L objects must not overlap"
            painted[cell] = color
        bend = _next_bend(opening, bend)

    rows = [r for r, _ in painted]
    cols = [c for _, c in painted]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for (r, c), color in painted.items():
        out[r - r0, c - c0] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate every colored L to the corner-marked opening and nest them into one stepped corner cluster.

    Input structure:
    - The background is black.
    - Several monochrome L-shaped objects are scattered on the canvas; each object is either a 3-pixel 2x2 corner or a 5-pixel 3x3 corner.
    - Exactly one gray marker pixel sits in a grid corner and indicates the target opening direction for every L.

    Rule:
    - Read the gray corner marker as one of the four diagonal opening directions: upper-left, upper-right, lower-left, or lower-right.
    - Detect every L object, determine its size and current opening, and rotate it so its opening matches the marker.
    - Sort the objects by area from large to small; ties keep scan order so equal-size inputs remain deterministic.
    - Starting from the largest object, place each next L so its outer bend lands on the previous L's inner corner, then crop to the tight bounding box of the nested result.

    Key insight:
    - The corner marker is a direction code, not an object to keep, and the final picture is not a rearrangement of input positions but a new nested corner structure ordered by L size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h <= 30 and w <= 30

    opening = _target_opening(g)
    work = g.copy()
    marker_r, marker_c = _OPENING_TO_CORNER[opening](h, w)
    work[marker_r, marker_c] = BG

    components = _component_cells(work)
    assert components, "expected at least one L object"

    objects = [_match_l_object(component) for component in components]
    objects.sort(key=lambda obj: (-int(obj["area"]), int(obj["bbox"][0]), int(obj["bbox"][1])))
    return _render_nested(objects, opening)


def _draw_object(grid: np.ndarray, color: int, size: int, opening: str, top: int, left: int) -> None:
    for r, c in TEMPLATES[(size, opening)]:
        grid[top + r, left + c] = color


def _candidate_positions(blocked: np.ndarray, size: int) -> list[tuple[int, int]]:
    h, w = blocked.shape
    candidates: list[tuple[int, int]] = []
    for top in range(1, h - size):
        for left in range(1, w - size):
            r0 = top - 1
            c0 = left - 1
            r1 = top + size + 1
            c1 = left + size + 1
            if blocked[r0:r1, c0:c1].any():
                continue
            candidates.append((top, left))
    return candidates


def _pick_sizes() -> list[int]:
    if int(_RNG.integers(0, 2)) == 0:
        return [3, 2]
    return [3, 3, 2] if int(_RNG.integers(0, 2)) == 0 else [3, 2, 2]


def _pick_openings(count: int, target_opening: str) -> list[str]:
    openings = [_RNG.choice(OPENINGS).item() for _ in range(count)]
    if all(opening == target_opening for opening in openings):
        idx = int(_RNG.integers(count))
        choices = [opening for opening in OPENINGS if opening != target_opening]
        openings[idx] = _RNG.choice(choices).item()
    return openings


def _gen_attempt() -> np.ndarray | None:
    target_opening = _RNG.choice(OPENINGS).item()
    sizes = _pick_sizes()
    count = len(sizes)

    height = int(_RNG.integers(9 + count, 19))
    width = int(_RNG.integers(9 + count, 19))
    grid = np.full((height, width), BG, dtype=int)
    marker_r, marker_c = _OPENING_TO_CORNER[target_opening](height, width)
    grid[marker_r, marker_c] = MARKER_COLOR

    colors = list(map(int, _RNG.choice(COLOR_POOL, size=count, replace=False)))
    openings = _pick_openings(count, target_opening)
    objects = [
        {"color": color, "size": size, "opening": opening}
        for color, size, opening in zip(colors, sizes, openings)
    ]

    blocked = np.zeros_like(grid, dtype=bool)
    blocked[marker_r, marker_c] = True
    for opening_name in OPENINGS:
        corner_r, corner_c = _OPENING_TO_CORNER[opening_name](height, width)
        blocked[corner_r, corner_c] = True

    for obj in sorted(objects, key=lambda item: int(item["size"]), reverse=True):
        candidates = _candidate_positions(blocked, int(obj["size"]))
        if not candidates:
            return None
        top, left = candidates[int(_RNG.integers(len(candidates)))]
        _draw_object(grid, int(obj["color"]), int(obj["size"]), str(obj["opening"]), top, left)

        r0 = max(0, top - 1)
        c0 = max(0, left - 1)
        r1 = min(height, top + int(obj["size"]) + 1)
        c1 = min(width, left + int(obj["size"]) + 1)
        blocked[r0:r1, c0:c1] = True

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid L-nesting puzzle input")
