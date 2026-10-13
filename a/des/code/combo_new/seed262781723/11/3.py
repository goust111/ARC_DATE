from __future__ import annotations

import random

import numpy as np


EDGE_ORDER = ("top", "bottom", "left", "right")
EDGE_DIRS = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}
PALETTE = tuple(range(1, 10))
GEN_MAX_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    if np.any(g == 0):
        return 0
    counts = np.bincount(g.ravel())
    return int(np.argmax(counts))


def _components4(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict[str, object]] = []

    for r0 in range(height):
        for c0 in range(width):
            color = int(g[r0, c0])
            if color == bg or seen[r0, c0]:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = r + dr
                    cc = c + dc
                    if not (0 <= rr < height and 0 <= cc < width):
                        continue
                    if seen[rr, cc] or int(g[rr, cc]) != color:
                        continue
                    seen[rr, cc] = True
                    stack.append((rr, cc))

            arr = np.array(coords, dtype=int)
            rmin = int(arr[:, 0].min())
            cmin = int(arr[:, 1].min())
            rmax = int(arr[:, 0].max())
            cmax = int(arr[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (rmin, cmin, rmax, cmax),
                }
            )

    return components


def _edge_contacts(coords: np.ndarray, height: int, width: int) -> list[str]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    contacts: list[str] = []
    if np.any(rows == 0):
        contacts.append("top")
    if np.any(rows == height - 1):
        contacts.append("bottom")
    if np.any(cols == 0):
        contacts.append("left")
    if np.any(cols == width - 1):
        contacts.append("right")
    return contacts


def _edge_anchor(coords: np.ndarray, edge: str, height: int, width: int) -> tuple[int, int]:
    if edge == "top":
        edge_coords = coords[coords[:, 0] == 0]
        mid = (width - 1) / 2.0
        return min(
            ((abs(int(c) - mid), int(c), (0, int(c))) for _r, c in edge_coords),
            key=lambda item: (item[0], item[1]),
        )[2]
    if edge == "bottom":
        edge_coords = coords[coords[:, 0] == height - 1]
        mid = (width - 1) / 2.0
        return min(
            (
                (abs(int(c) - mid), int(c), (height - 1, int(c)))
                for _r, c in edge_coords
            ),
            key=lambda item: (item[0], item[1]),
        )[2]
    if edge == "left":
        edge_coords = coords[coords[:, 1] == 0]
        mid = (height - 1) / 2.0
        return min(
            ((abs(int(r) - mid), int(r), (int(r), 0)) for r, _c in edge_coords),
            key=lambda item: (item[0], item[1]),
        )[2]
    assert edge == "right"
    edge_coords = coords[coords[:, 1] == width - 1]
    mid = (height - 1) / 2.0
    return min(
        (
            (abs(int(r) - mid), int(r), (int(r), width - 1))
            for r, _c in edge_coords
        ),
        key=lambda item: (item[0], item[1]),
    )[2]


def _paint_ray(
    out: np.ndarray,
    bg: int,
    color: int,
    start: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    own_cells: set[tuple[int, int]],
) -> None:
    height, width = map(int, out.shape)
    dr, dc = map(int, direction)
    r0, c0 = map(int, start)

    for step in range(int(length)):
        r = r0 + dr * step
        c = c0 + dc * step
        if not (0 <= r < height and 0 <= c < width):
            break
        if (r, c) in own_cells:
            out[r, c] = color
            continue
        if int(out[r, c]) != bg:
            break
        out[r, c] = color
        own_cells.add((r, c))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend inward rays from boundary-touching objects, using each object's touched edges for direction and its area for ray length.

    Input structure:
    - The background is black.
    - The grid contains several 4-connected monochrome foreground objects, and different objects use different colors.
    - Some objects touch one or more grid edges: top row, bottom row, left column, or right column.

    Rule:
    - For each foreground object, determine which edges it touches and compute its pixel area N.
    - For every touched edge, choose that object's edge pixel closest to the middle of that edge.
    - From that pixel, draw a same-colored straight ray inward with length N: top goes downward, bottom upward, left rightward, right leftward.
    - If the next ray cell would hit another non-background pixel, stop before it. Objects touching no edge stay unchanged and emit no rays.

    Key insight:
    - The number and directions of rays come from spatial edge contacts, while the ray length comes from a separate numeric property, the object's area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    height, width = map(int, g.shape)

    components = _components4(g, bg)
    components.sort(key=lambda comp: tuple(comp["bbox"]))  # type: ignore[arg-type]

    out = g.copy()
    for comp in components:
        color = int(comp["color"])
        coords = np.asarray(comp["coords"], dtype=int)
        area = int(comp["area"])
        contacts = _edge_contacts(coords, height, width)
        if not contacts:
            continue

        own_cells = {tuple(map(int, cell)) for cell in coords.tolist()}
        for edge in EDGE_ORDER:
            if edge not in contacts:
                continue
            anchor = _edge_anchor(coords, edge, height, width)
            _paint_ray(out, bg, color, anchor, EDGE_DIRS[edge], area, own_cells)

    return out


def _normalize(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    rmin = min(r for r, _ in coords)
    cmin = min(c for _, c in coords)
    return sorted((r - rmin, c - cmin) for r, c in coords)


def _transform_shape(
    coords: list[tuple[int, int]],
    *,
    rot: int = 0,
    flip_h: bool = False,
) -> list[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in coords]
    for _ in range(rot % 4):
        pts = [(-c, r) for r, c in pts]
    if flip_h:
        pts = [(r, -c) for r, c in pts]
    return _normalize(pts)


BASE_INTERIOR_SHAPES = [
    [(0, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1), (1, 0)],
    [(0, 0), (0, 1), (1, 1)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 1), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 1), (1, 0), (1, 1), (1, 2)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
]


def _shape_variants(coords: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    variants: list[list[tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for rot in range(4):
        for flip_h in (False, True):
            variant = _transform_shape(coords, rot=rot, flip_h=flip_h)
            key = tuple(variant)
            if key not in seen:
                seen.add(key)
                variants.append(variant)
    return variants


INTERIOR_VARIANTS = [
    variant for shape in BASE_INTERIOR_SHAPES for variant in _shape_variants(shape)
]


def _dilate8(mask: np.ndarray) -> np.ndarray:
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


def _can_place(grid: np.ndarray, forbidden: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    height, width = map(int, grid.shape)
    for r, c in cells:
        if not (0 <= r < height and 0 <= c < width):
            return False
        if int(grid[r, c]) != 0 or bool(forbidden[r, c]):
            return False
    return True


def _stamp_object(
    grid: np.ndarray,
    forbidden: np.ndarray,
    cells: list[tuple[int, int]],
    color: int,
) -> None:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    grid[rows, cols] = int(color)
    mask = np.zeros_like(forbidden, dtype=bool)
    mask[rows, cols] = True
    forbidden |= _dilate8(mask)


def _ray_path_from_component(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
) -> dict[str, list[tuple[int, int]]]:
    coords = np.array(cells, dtype=int)
    height, width = map(int, grid.shape)
    area = int(coords.shape[0])
    object_set = set(cells)
    paths: dict[str, list[tuple[int, int]]] = {}
    for edge in _edge_contacts(coords, height, width):
        anchor = _edge_anchor(coords, edge, height, width)
        dr, dc = EDGE_DIRS[edge]
        ray: list[tuple[int, int]] = []
        for step in range(1, area):
            r = int(anchor[0] + dr * step)
            c = int(anchor[1] + dc * step)
            if not (0 <= r < height and 0 <= c < width):
                break
            if (r, c) in object_set:
                continue
            ray.append((r, c))
        paths[edge] = ray
    return paths


def _place_random_interior(
    grid: np.ndarray,
    forbidden: np.ndarray,
    color: int,
    *,
    avoid: set[tuple[int, int]] | None = None,
) -> bool:
    height, width = map(int, grid.shape)
    avoid = avoid or set()

    variants = INTERIOR_VARIANTS[:]
    random.shuffle(variants)
    for shape in variants:
        sh = 1 + max(r for r, _ in shape)
        sw = 1 + max(c for _, c in shape)
        row_options = list(range(2, height - sh - 1))
        col_options = list(range(2, width - sw - 1))
        random.shuffle(row_options)
        random.shuffle(col_options)
        for r0 in row_options:
            for c0 in col_options:
                cells = [(r0 + r, c0 + c) for r, c in shape]
                if any(cell in avoid for cell in cells):
                    continue
                if _can_place(grid, forbidden, cells):
                    _stamp_object(grid, forbidden, cells, color)
                    return True
    return False


def _edge_bar(side: str, height: int, width: int, length: int) -> list[tuple[int, int]] | None:
    if side == "top":
        starts = list(range(2, width - length - 1))
        if not starts:
            return None
        c0 = random.choice(starts)
        return [(0, c0 + i) for i in range(length)]
    if side == "bottom":
        starts = list(range(2, width - length - 1))
        if not starts:
            return None
        c0 = random.choice(starts)
        return [(height - 1, c0 + i) for i in range(length)]
    if side == "left":
        starts = list(range(2, height - length - 1))
        if not starts:
            return None
        r0 = random.choice(starts)
        return [(r0 + i, 0) for i in range(length)]
    starts = list(range(2, height - length - 1))
    if not starts:
        return None
    r0 = random.choice(starts)
    return [(r0 + i, width - 1) for i in range(length)]


def _corner_l(corner: str, height: int, width: int) -> list[tuple[int, int]]:
    if corner == "tl":
        return [(0, 0), (0, 1), (1, 0)]
    if corner == "tr":
        return [(0, width - 1), (0, width - 2), (1, width - 1)]
    if corner == "bl":
        return [(height - 1, 0), (height - 2, 0), (height - 1, 1)]
    return [(height - 1, width - 1), (height - 1, width - 2), (height - 2, width - 1)]


def _mixed_scene() -> np.ndarray | None:
    height = random.randint(14, 22)
    width = random.randint(14, 22)
    grid = np.zeros((height, width), dtype=int)
    forbidden = np.zeros_like(grid, dtype=bool)
    colors = random.sample(PALETTE, 4)

    side = random.choice(("top", "bottom", "left", "right"))
    length = random.randint(3, 5)
    edge_cells = _edge_bar(side, height, width, length)
    if edge_cells is None or not _can_place(grid, forbidden, edge_cells):
        return None
    _stamp_object(grid, forbidden, edge_cells, colors[0])

    opposite_corners = {
        "top": ("bl", "br"),
        "bottom": ("tl", "tr"),
        "left": ("tr", "br"),
        "right": ("tl", "bl"),
    }
    corner = random.choice(opposite_corners[side])
    corner_cells = _corner_l(corner, height, width)
    if not _can_place(grid, forbidden, corner_cells):
        return None
    _stamp_object(grid, forbidden, corner_cells, colors[1])

    blocker_placed = False
    if random.random() < 0.5:
        paths = _ray_path_from_component(grid, edge_cells)
        candidates = []
        for ray in paths.values():
            if len(ray) >= 2:
                candidates.extend(ray[1:])
        random.shuffle(candidates)
        for cell in candidates:
            if _can_place(grid, forbidden, [cell]):
                _stamp_object(grid, forbidden, [cell], colors[2])
                blocker_placed = True
                break

    if not blocker_placed and not _place_random_interior(grid, forbidden, colors[2]):
        return None

    if random.random() < 0.6 and not _place_random_interior(grid, forbidden, colors[3]):
        return None

    return grid


def _three_edge_shape(kind: str, height: int, width: int) -> list[tuple[int, int]]:
    if kind == "top":
        return [(0, c) for c in range(width)] + [(1, 0), (1, width - 1)]
    if kind == "bottom":
        return [(height - 1, c) for c in range(width)] + [(height - 2, 0), (height - 2, width - 1)]
    if kind == "left":
        return [(r, 0) for r in range(height)] + [(0, 1), (height - 1, 1)]
    return [(r, width - 1) for r in range(height)] + [(0, width - 2), (height - 1, width - 2)]


def _three_edge_scene() -> np.ndarray | None:
    kind = random.choice(("top", "bottom", "left", "right"))
    if kind in ("top", "bottom"):
        height = random.randint(6, 9)
        width = random.choice((5, 7))
    else:
        height = random.choice((5, 7))
        width = random.randint(6, 9)

    grid = np.zeros((height, width), dtype=int)
    forbidden = np.zeros_like(grid, dtype=bool)
    colors = random.sample(PALETTE, 3)

    main_cells = _three_edge_shape(kind, height, width)
    if not _can_place(grid, forbidden, main_cells):
        return None
    _stamp_object(grid, forbidden, main_cells, colors[0])

    if random.random() < 0.6:
        paths = _ray_path_from_component(grid, main_cells)
        major_edge = {"top": "top", "bottom": "bottom", "left": "left", "right": "right"}[kind]
        candidates = paths.get(major_edge, [])[1:]
        random.shuffle(candidates)
        for cell in candidates:
            if _can_place(grid, forbidden, [cell]):
                _stamp_object(grid, forbidden, [cell], colors[1])
                break

    if random.random() < 0.5:
        if not _place_random_interior(grid, forbidden, colors[2]):
            return None

    return grid


def _frame_scene() -> np.ndarray:
    height = random.choice((5, 7))
    width = random.choice((5, 7))
    grid = np.zeros((height, width), dtype=int)
    color = random.choice(PALETTE)

    cells = []
    for r in range(height):
        for c in range(width):
            if r in (0, height - 1) or c in (0, width - 1):
                cells.append((r, c))
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    grid[rows, cols] = color

    if height == 7 and width == 7 and random.random() < 0.5:
        extra_color = random.choice([c for c in PALETTE if c != color])
        extra_cell = random.choice([(2, 2), (2, 4), (4, 2), (4, 4)])
        grid[extra_cell] = extra_color

    return grid


def _valid_generated_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False
    bg = _background_color(grid)
    components = _components4(grid, bg)
    colors = [int(comp["color"]) for comp in components]
    if len(colors) != len(set(colors)):
        return False
    if not any(_edge_contacts(np.asarray(comp["coords"]), *grid.shape) for comp in components):
        return False
    try:
        out = main(grid)
    except Exception:
        return False
    return bool(isinstance(out, np.ndarray) and out.shape == grid.shape and max(out.shape) <= 30)


def generate_input() -> np.ndarray:
    builders = (_mixed_scene, _three_edge_scene, _frame_scene)
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = random.choice(builders)()
        if grid is None:
            continue
        if _valid_generated_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid boundary-ray input")
