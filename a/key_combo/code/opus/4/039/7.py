from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
PALETTE = (1, 3, 4, 5)
HOLE_SIZE = 6
NO_HOLE_SIZES = (4, 6)
GRID_SIDE_RANGE = (27, 30)
HOLE_QUADRANTS = ("NW", "NE", "SW", "SE")


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    deltas = ((-1, 0), (1, 0), (0, -1), (0, 1)) if connectivity == 4 else tuple(
        (dr, dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    )

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if not _in_bounds(mask.shape, nr, nc):
                        continue
                    if seen[nr, nc] or not mask[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_color_components(grid: np.ndarray) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BLACK):
        for coords in _components_mask(grid == color, connectivity=4):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                }
            )
    return sorted(comps, key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    black = sprite == BLACK
    h, w = sprite.shape
    exterior = np.zeros_like(black, dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if black[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if black[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        rr, cc = q.popleft()
        for nr, nc in _neighbors4(rr, cc):
            if not _in_bounds(sprite.shape, nr, nc):
                continue
            if exterior[nr, nc] or not black[nr, nc]:
                continue
            exterior[nr, nc] = True
            q.append((nr, nc))

    enclosed = black & ~exterior
    return _components_mask(enclosed, connectivity=4)


def _hole_quadrant(hole_coords: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    size = r1 - r0 + 1
    mid_r = r0 + size / 2.0
    mid_c = c0 + size / 2.0
    centroid_r = float(hole_coords[:, 0].mean())
    centroid_c = float(hole_coords[:, 1].mean())
    vertical = "N" if centroid_r < mid_r else "S"
    horizontal = "W" if centroid_c < mid_c else "E"
    return vertical + horizontal


def _stamp_origins(bbox: tuple[int, int, int, int], quadrant: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    size = r1 - r0 + 1
    if quadrant in {"NW", "SE"}:
        return [(r0 - size, c1 + 1), (r1 + 1, c0 - size)]
    return [(r0 - size, c0 - size), (r1 + 1, c1 + 1)]


def _paste_block(dst: np.ndarray, r0: int, c0: int, size: int, color: int) -> None:
    h, w = dst.shape
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(h, int(r0) + int(size))
    cc1 = min(w, int(c0) + int(size))
    if rr0 < rr1 and cc0 < cc1:
        dst[rr0:rr1, cc0:cc1] = int(color)


def main(input_grid: np.ndarray) -> np.ndarray:
    """Use each sprite hole's quadrant to choose which opposite diagonal corners receive red square stamps.

    Input structure:
    - The canvas is black and contains 3 to 4 separated single-color sprites.
    - Each sprite has an even-sized square bounding box, and some sprites contain one enclosed black hole.
    - Hole-bearing sprites and hole-free sprites coexist, with the hole position serving as the only direction cue.

    Rule:
    - Find every non-black 4-connected sprite and compute its square bounding box.
    - Inside that bbox, split black pixels into 4-connected regions and keep only enclosed regions that do not touch the bbox boundary.
    - Leave sprites with no enclosed hole unchanged.
    - For a sprite with exactly one enclosed hole, locate the hole centroid quadrant: holes in NW or SE trigger red stamps at the outward NE and SW corners, while holes in NE or SW trigger red stamps at the outward NW and SE corners.
    - Each red stamp is a solid square matching the sprite bbox size; clip any out-of-bounds stamp pixels.

    Key insight:
    - The enclosed hole is not something to fill; its quadrant encodes which complementary diagonal outside the sprite must be stamped.
    """
    grid = np.asarray(input_grid, dtype=int)
    out = grid.copy()

    for comp in _find_color_components(grid):
        bbox = comp["bbox"]
        r0, c0, r1, c1 = bbox
        size = r1 - r0 + 1
        if size != c1 - c0 + 1 or size % 2 != 0:
            continue

        sprite = _crop(grid, bbox)
        holes_local = _enclosed_holes(sprite)
        if len(holes_local) != 1:
            continue

        hole_global = holes_local[0] + np.array([[r0, c0]])
        quadrant = _hole_quadrant(hole_global, bbox)
        for rr, cc in _stamp_origins(bbox, quadrant):
            _paste_block(out, rr, cc, size, RED)

    return out


def _connected_2x2_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for bits in range(1, 1 << 4):
        mask = np.zeros((2, 2), dtype=bool)
        coords = []
        for idx, (r, c) in enumerate(cells):
            if bits & (1 << idx):
                mask[r, c] = True
                coords.append((r, c))
        comp = _components_mask(mask, connectivity=4)
        if len(comp) == 1:
            masks.append(mask)
    return masks


HOLE_MASKS = _connected_2x2_masks()


def _hole_anchor(quadrant: str) -> tuple[int, int]:
    return {
        "NW": (1, 1),
        "NE": (1, 3),
        "SW": (3, 1),
        "SE": (3, 3),
    }[quadrant]


def _build_sprite(spec: dict) -> np.ndarray:
    size = int(spec["size"])
    sprite = np.full((size, size), int(spec["color"]), dtype=int)
    quadrant = spec["hole_quadrant"]
    if quadrant is not None:
        anchor_r, anchor_c = _hole_anchor(quadrant)
        mask = spec["hole_mask"]
        sprite[anchor_r : anchor_r + 2, anchor_c : anchor_c + 2][mask] = BLACK
    return sprite


def _render_specs(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        sprite = _build_sprite(spec)
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        size = int(spec["size"])
        grid[r0 : r0 + size, c0 : c0 + size] = sprite
    return grid


def _expected_output(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = _render_specs(shape, specs)
    for spec in specs:
        if spec["hole_quadrant"] is None:
            continue
        bbox = (
            int(spec["r0"]),
            int(spec["c0"]),
            int(spec["r0"]) + int(spec["size"]) - 1,
            int(spec["c0"]) + int(spec["size"]) - 1,
        )
        for rr, cc in _stamp_origins(bbox, spec["hole_quadrant"]):
            _paste_block(out, rr, cc, int(spec["size"]), RED)
    return out


def _rect_cells(shape: tuple[int, int], r0: int, c0: int, size: int) -> set[tuple[int, int]]:
    h, w = shape
    return {
        (r, c)
        for r in range(max(0, r0), min(h, r0 + size))
        for c in range(max(0, c0), min(w, c0 + size))
    }


def _component_chebyshev_distance(a: np.ndarray, b: np.ndarray) -> int:
    best = 10**9
    for ar, ac in a:
        diffs = np.abs(b - np.array([ar, ac]))
        best = min(best, int(np.max(diffs, axis=1).min()))
    return best


def _spec_by_color(specs: list[dict]) -> dict[int, dict]:
    return {int(spec["color"]): spec for spec in specs}


def _validate_scene(grid: np.ndarray, specs: list[dict], expected: np.ndarray) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False

    comps = _find_color_components(grid)
    if len(comps) != len(specs) or len(comps) not in {3, 4}:
        return False

    spec_map = _spec_by_color(specs)
    if set(spec_map) != {int(comp["color"]) for comp in comps}:
        return False

    hole_modes = set()
    hole_components = 0
    for comp in comps:
        spec = spec_map[int(comp["color"])]
        r0, c0, r1, c1 = comp["bbox"]
        if comp["bbox"] != (
            int(spec["r0"]),
            int(spec["c0"]),
            int(spec["r0"]) + int(spec["size"]) - 1,
            int(spec["c0"]) + int(spec["size"]) - 1,
        ):
            return False
        sprite = _crop(grid, comp["bbox"])
        holes = _enclosed_holes(sprite)
        if spec["hole_quadrant"] is None:
            if holes:
                return False
            continue
        if len(holes) != 1:
            return False
        hole_components += 1
        hole_global = holes[0] + np.array([[r0, c0]])
        quadrant = _hole_quadrant(hole_global, comp["bbox"])
        if quadrant != spec["hole_quadrant"]:
            return False
        hole_modes.add("A" if quadrant in {"NW", "SE"} else "B")

    if hole_components != 2 or hole_modes != {"A", "B"}:
        return False

    coords = [comp["coords"] for comp in comps]
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if _component_chebyshev_distance(coords[i], coords[j]) < 3:
                return False

    source_cells = {
        (int(r), int(c))
        for comp in comps
        for r, c in comp["coords"]
    }
    for spec in specs:
        if spec["hole_quadrant"] is None:
            continue
        bbox = (
            int(spec["r0"]),
            int(spec["c0"]),
            int(spec["r0"]) + int(spec["size"]) - 1,
            int(spec["c0"]) + int(spec["size"]) - 1,
        )
        for rr, cc in _stamp_origins(bbox, spec["hole_quadrant"]):
            stamp_cells = _rect_cells(grid.shape, rr, cc, int(spec["size"]))
            if not stamp_cells:
                return False
            if stamp_cells & source_cells:
                return False
            if any(int(grid[r, c]) != BLACK for r, c in stamp_cells):
                return False

    red_pixels = int(np.sum(expected == RED))
    if red_pixels != 2 * 2 * HOLE_SIZE * HOLE_SIZE:
        return False

    produced = main(grid)
    return np.array_equal(produced, expected)


def _hole_spec(r0: int, c0: int, color: int, quadrant: str) -> dict:
    return {
        "r0": int(r0),
        "c0": int(c0),
        "size": HOLE_SIZE,
        "color": int(color),
        "hole_quadrant": quadrant,
        "hole_mask": random.choice(HOLE_MASKS).copy(),
    }


def _no_hole_spec(r0: int, c0: int, color: int, size: int) -> dict:
    return {
        "r0": int(r0),
        "c0": int(c0),
        "size": int(size),
        "color": int(color),
        "hole_quadrant": None,
        "hole_mask": None,
    }


def _make_specs(shape: tuple[int, int]) -> list[dict]:
    h, w = shape
    colors = random.sample(PALETTE, random.choice((3, 4)))
    class_a_first = random.choice((True, False))
    quad_a = random.choice(("NW", "SE"))
    quad_b = random.choice(("NE", "SW"))

    hole_tl = _hole_spec(6, 6, colors[0], quad_a if class_a_first else quad_b)
    hole_br = _hole_spec(h - 12, w - 12, colors[1], quad_b if class_a_first else quad_a)

    specs = [hole_tl, hole_br]
    corner_specs = [
        _no_hole_spec(2 if size == 4 else 1, w - (6 if size == 4 else 7), colors[2], size)
        for size in [random.choice(NO_HOLE_SIZES)]
    ]
    specs.extend(corner_specs)

    if len(colors) == 4:
        size = random.choice(NO_HOLE_SIZES)
        specs.append(_no_hole_spec(h - (6 if size == 4 else 7), 2 if size == 4 else 1, colors[3], size))

    return specs


def generate_input() -> np.ndarray:
    for _ in range(300):
        side_h = random.randint(*GRID_SIDE_RANGE)
        side_w = random.randint(*GRID_SIDE_RANGE)
        specs = _make_specs((side_h, side_w))
        grid = _render_specs((side_h, side_w), specs)
        expected = _expected_output((side_h, side_w), specs)
        if _validate_scene(grid, specs, expected):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
