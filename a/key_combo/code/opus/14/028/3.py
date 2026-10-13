from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
TEAL = 8

DIR_BY_COLOR = {
    GREEN: "left",
    YELLOW: "down",
    PINK: "up",
    TEAL: "right",
}
COLOR_ORDER = (GREEN, YELLOW, PINK, TEAL)
DIR_STEP = {
    "left": (0, -1),
    "right": (0, 1),
    "up": (-1, 0),
    "down": (1, 0),
}

PROTOTYPE_SLOTS = (
    {"name": "A", "origin": (0, 0), "limit": (2, 3)},
    {"name": "B", "origin": (0, 4), "limit": (3, 2)},
    {"name": "C", "origin": (3, 0), "limit": (3, 2)},
    {"name": "D", "origin": (4, 3), "limit": (2, 3)},
)

# Four distinct run-compression classes. The first one intentionally compresses
# to a single cell; the others remain multi-cell and are unique under D4.
SHAPE_LIBRARY = {
    "d2": np.array([[1, 1]], dtype=bool),
    "l3": np.array([[1, 0], [1, 1]], dtype=bool),
    "t4": np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    "s4": np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
}

GRID_SIZE_RANGE = (20, 26)
NUM_CANDIDATES_RANGE = (4, 6)
STRETCH_FACTORS = (1, 2, 3)
POINT_STRETCH_FACTORS = (1, 2, 3, 4, 5, 6)
MIN_CANDIDATE_AREA = 6
MAX_CANDIDATE_AREA = 20
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _compress_runs(mask: np.ndarray) -> np.ndarray:
    assert mask.ndim == 2 and mask.size > 0
    row_keep = [0]
    for r in range(1, mask.shape[0]):
        if not np.array_equal(mask[r], mask[r - 1]):
            row_keep.append(r)
    reduced = mask[row_keep, :]

    col_keep = [0]
    for c in range(1, reduced.shape[1]):
        if not np.array_equal(reduced[:, c], reduced[:, c - 1]):
            col_keep.append(c)
    return reduced[:, col_keep]


def _orientations(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen = set()
    for turns in range(4):
        rotated = np.rot90(mask, turns)
        for variant in (rotated, np.fliplr(rotated)):
            key = _mask_key(variant)
            if key not in seen:
                seen.add(key)
                out.append(variant.copy())
    return out


def _signature_set(mask: np.ndarray) -> set[tuple[tuple[int, ...], ...]]:
    return {_mask_key(_compress_runs(variant)) for variant in _orientations(mask)}


def _round_half_up(x: float) -> int:
    return int(np.floor(float(x) + 0.5))


def _centroid(coords: np.ndarray) -> tuple[int, int]:
    return _round_half_up(coords[:, 0].mean()), _round_half_up(coords[:, 1].mean())


def _ray_cells(start: tuple[int, int], direction: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = DIR_STEP[direction]
    r, c = start
    out: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        out.append((r, c))
        r += dr
        c += dc
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rs0 = max(0, -dr)
            rs1 = mask.shape[0] - max(0, dr)
            cs0 = max(0, -dc)
            cs1 = mask.shape[1] - max(0, dc)
            out[rs0 + dr : rs1 + dr, cs0 + dc : cs1 + dc] |= mask[rs0:rs1, cs0:cs1]
    return out


def _parse_grid(grid: np.ndarray) -> tuple[dict[int, dict], list[dict]]:
    prototypes: dict[int, dict] = {}
    for color in COLOR_ORDER:
        comps = _components(grid == color)
        assert len(comps) == 1, f"expected exactly one prototype of color {color}"
        coords = comps[0]
        assert coords[:, 0].max() < 6 and coords[:, 1].max() < 6, "prototype must be in top-left 6x6"
        mask = _coords_to_mask(coords)
        prototypes[color] = {
            "coords": coords,
            "mask": mask,
            "signatures": _signature_set(mask),
        }

    grey_comps = _components(grid == GREY)
    assert 4 <= len(grey_comps) <= 6, "expected 4-6 grey candidate objects"
    candidates: list[dict] = []
    for coords in grey_comps:
        mask = _coords_to_mask(coords)
        compressed = _compress_runs(mask)
        key = _mask_key(compressed)
        matches = [color for color, proto in prototypes.items() if key in proto["signatures"]]
        assert len(matches) == 1, "each grey candidate must match exactly one prototype"
        color = matches[0]
        candidates.append(
            {
                "coords": coords,
                "mask": mask,
                "compressed": compressed,
                "match_color": color,
                "direction": DIR_BY_COLOR[color],
                "centroid": _centroid(coords),
            }
        )
    return prototypes, candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each stretched grey candidate to a prototype by run-compressing its mask, then use the matched prototype color to choose a ray direction.

    Input structure:
    - A black-background grid contains four small colored prototype objects inside the top-left 6x6 region: green, yellow, pink, and teal.
    - Elsewhere the grid contains 4-6 grey 4-connected candidate objects whose raw outlines may be stretched by duplicated rows and/or columns.
    - The prototypes are separated from one another, and the candidates are separate grey components on the same black canvas.

    Rule:
    - Crop every prototype and grey candidate to its tight bbox, collapse consecutive identical rows and consecutive identical columns, and compare the resulting binary masks up to rotation and reflection.
    - Each grey candidate matches exactly one prototype; green means cast left, yellow down, pink up, and teal right.
    - Compute the candidate centroid by averaging its pixel rows and columns and rounding half up, then paint every black cell on the ray from that centroid to the grid edge red while leaving all non-black cells unchanged.

    Key insight:
    - The grey objects are not recognizable from their stretched silhouettes alone; run-compression recovers the prototype shape, and the matched prototype color is a direction code rather than a recoloring target.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _prototypes, candidates = _parse_grid(g)
    out = g.copy()
    for candidate in candidates:
        for r, c in _ray_cells(candidate["centroid"], candidate["direction"], g.shape):
            if out[r, c] == BG:
                out[r, c] = RED
    return out


def _prototype_variants_for_slot(base_mask: np.ndarray, limit: tuple[int, int]) -> list[np.ndarray]:
    max_h, max_w = limit
    return [variant for variant in _orientations(base_mask) if variant.shape[0] <= max_h and variant.shape[1] <= max_w]


def _build_prototypes() -> tuple[np.ndarray, dict[int, str]]:
    grid = np.zeros((6, 6), dtype=int)
    colors = list(COLOR_ORDER)
    random.shuffle(colors)
    shape_names = list(SHAPE_LIBRARY)
    random.shuffle(shape_names)

    color_to_shape: dict[int, str] = {}
    for slot, color, shape_name in zip(PROTOTYPE_SLOTS, colors, shape_names):
        limit = slot["limit"]
        variants = _prototype_variants_for_slot(SHAPE_LIBRARY[shape_name], limit)
        assert variants, "prototype shape must fit its slot"
        variant = random.choice(variants)
        max_roff = limit[0] - variant.shape[0]
        max_coff = limit[1] - variant.shape[1]
        roff = random.randint(0, max_roff)
        coff = random.randint(0, max_coff)
        r0 = slot["origin"][0] + roff
        c0 = slot["origin"][1] + coff
        patch = np.where(variant, color, BG)
        grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch
        color_to_shape[color] = shape_name
    return grid, color_to_shape


def _stretch_options(mask: np.ndarray, mode: str) -> list[np.ndarray]:
    options: list[np.ndarray] = []
    seen = set()
    factor_choices = POINT_STRETCH_FACTORS if int(mask.sum()) == 1 else STRETCH_FACTORS
    row_products = list(itertools.product(factor_choices, repeat=mask.shape[0]))
    col_products = list(itertools.product(factor_choices, repeat=mask.shape[1]))
    for row_factors in row_products:
        for col_factors in col_products:
            row_stretched = any(f > 1 for f in row_factors)
            col_stretched = any(f > 1 for f in col_factors)
            if mode == "row" and (not row_stretched or col_stretched):
                continue
            if mode == "col" and (row_stretched or not col_stretched):
                continue
            if mode == "both" and (not row_stretched or not col_stretched):
                continue
            stretched = np.repeat(np.repeat(mask, row_factors, axis=0), col_factors, axis=1)
            if int(stretched.sum()) < MIN_CANDIDATE_AREA or int(stretched.sum()) > MAX_CANDIDATE_AREA:
                continue
            if np.array_equal(stretched, mask):
                continue
            if not np.array_equal(_compress_runs(stretched), mask):
                continue
            key = _mask_key(stretched)
            if key in seen:
                continue
            seen.add(key)
            options.append(stretched)
    return options


def _candidate_specs(color_to_shape: dict[int, str]) -> list[dict] | None:
    count = random.randint(*NUM_CANDIDATES_RANGE)
    colors = list(COLOR_ORDER)
    extras = [random.choice(COLOR_ORDER) for _ in range(count - 4)]
    assigned_colors = colors + extras
    random.shuffle(assigned_colors)

    stretch_modes = ["row", "col", "both"] + [random.choice(("row", "col", "both")) for _ in range(count - 3)]
    random.shuffle(stretch_modes)

    specs: list[dict] = []
    for color, stretch_mode in zip(assigned_colors, stretch_modes):
        shape_name = color_to_shape[color]
        base_mask = SHAPE_LIBRARY[shape_name]
        variants = _orientations(base_mask)
        random.shuffle(variants)
        candidate_mask = None
        for variant in variants:
            target = _compress_runs(variant)
            options = _stretch_options(target, stretch_mode)
            if options:
                candidate_mask = random.choice(options)
                break
        if candidate_mask is None:
            return None
        specs.append(
            {
                "color": color,
                "direction": DIR_BY_COLOR[color],
                "stretch_mode": stretch_mode,
                "mask": candidate_mask,
            }
        )

    if set(spec["color"] for spec in specs) != set(COLOR_ORDER):
        return None
    if not {"row", "col", "both"} <= {spec["stretch_mode"] for spec in specs}:
        return None
    return specs


def _placement_score(direction: str, centroid: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    r, c = centroid
    if direction == "left":
        return (100 if r >= 7 else 0, c)
    if direction == "right":
        return (100 if r >= 7 else 0, w - 1 - c)
    if direction == "up":
        return (100 if c >= 7 else 0, r)
    return (100 if c >= 7 else 0, h - 1 - r)


def _enumerate_positions(
    candidate_mask: np.ndarray,
    direction: str,
    occupied: np.ndarray,
    reserved_rays: np.ndarray,
) -> list[dict]:
    h, w = occupied.shape
    ph, pw = candidate_mask.shape
    local_coords = np.argwhere(candidate_mask)
    avoid = _dilate8(occupied | reserved_rays)
    positions: list[dict] = []

    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            coords = local_coords + np.array([r0, c0], dtype=int)
            if np.any(avoid[coords[:, 0], coords[:, 1]]):
                continue
            centroid = _centroid(coords)
            ray = _ray_cells(centroid, direction, occupied.shape)
            object_cells = {tuple(map(int, rc)) for rc in coords}
            paintable = 0
            valid = True
            for cell in ray:
                if cell in object_cells:
                    continue
                rr, cc = cell
                if occupied[rr, cc] or reserved_rays[rr, cc]:
                    valid = False
                    break
                paintable += 1
            if not valid or paintable < 2:
                continue
            score = _placement_score(direction, centroid, occupied.shape)
            positions.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "coords": coords,
                    "centroid": centroid,
                    "ray": ray,
                    "paintable": paintable,
                    "score": score,
                }
            )

    random.shuffle(positions)
    positions.sort(key=lambda item: (item["score"][0], item["score"][1], item["paintable"]), reverse=True)
    return positions


def _place_candidates(base_grid: np.ndarray, specs: list[dict]) -> np.ndarray | None:
    start_occupied = base_grid != BG

    def dfs(grid: np.ndarray, occupied: np.ndarray, reserved_rays: np.ndarray, remaining: list[dict]) -> np.ndarray | None:
        if not remaining:
            return grid

        best_spec = None
        best_positions = None
        for spec in remaining:
            positions = _enumerate_positions(spec["mask"], spec["direction"], occupied, reserved_rays)
            if not positions:
                return None
            if best_positions is None or len(positions) < len(best_positions):
                best_spec = spec
                best_positions = positions

        assert best_spec is not None and best_positions is not None
        tail = [spec for spec in remaining if spec is not best_spec]
        for pos in best_positions:
            new_grid = grid.copy()
            new_occupied = occupied.copy()
            new_reserved = reserved_rays.copy()

            coords = pos["coords"]
            new_grid[coords[:, 0], coords[:, 1]] = GREY
            new_occupied[coords[:, 0], coords[:, 1]] = True

            object_cells = {tuple(map(int, rc)) for rc in coords}
            for cell in pos["ray"]:
                if cell not in object_cells:
                    new_reserved[cell] = True

            solved = dfs(new_grid, new_occupied, new_reserved, tail)
            if solved is not None:
                return solved
        return None

    return dfs(base_grid.copy(), start_occupied.copy(), np.zeros_like(start_occupied, dtype=bool), list(specs))


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        prototypes, candidates = _parse_grid(inp)
    except AssertionError:
        return False

    if set(prototypes) != set(COLOR_ORDER):
        return False
    if set(candidate["match_color"] for candidate in candidates) != set(COLOR_ORDER):
        return False

    prototype_canonical = {color: min(proto["signatures"]) for color, proto in prototypes.items()}
    if len(set(prototype_canonical.values())) != 4:
        return False

    occupied = inp != BG
    reserved = np.zeros_like(occupied, dtype=bool)
    for candidate in candidates:
        coords = candidate["coords"]
        if not (MIN_CANDIDATE_AREA <= len(coords) <= MAX_CANDIDATE_AREA):
            return False
        if int(candidate["compressed"].sum()) >= len(coords):
            return False

        cell_set = {tuple(map(int, rc)) for rc in coords}
        for cell in _ray_cells(candidate["centroid"], candidate["direction"], inp.shape):
            if cell in cell_set:
                continue
            rr, cc = cell
            if occupied[rr, cc]:
                return False
            if reserved[rr, cc]:
                return False
            reserved[rr, cc] = True

    expected = main(inp)
    if not np.array_equal(expected, out):
        return False
    if np.array_equal(inp, out):
        return False
    if RED not in set(map(int, np.unique(out))):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    base = np.zeros((h, w), dtype=int)

    prototypes, color_to_shape = _build_prototypes()
    base[:6, :6] = prototypes

    specs = _candidate_specs(color_to_shape)
    if specs is None:
        return None

    inp = _place_candidates(base, specs)
    if inp is None:
        return None

    out = main(inp)
    if not _semantic_checks(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC task 3")
