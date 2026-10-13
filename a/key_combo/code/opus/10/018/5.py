from __future__ import annotations

import hashlib
import random

import numpy as np


BG = 4
BLUE = 1
RED = 2
GREEN = 3

DIRS = ("up", "down", "left", "right")
GEN_RETRIES = 500
GRID_SIZE_RANGE = (13, 16)
OBJECT_COUNT_CHOICES = (2, 3, 3)
MIN_INPUT_GAP = 4

_USED_INPUT_SIGNATURES: set[str] = set()
_USED_OUTPUT_SIGNATURES: set[str] = set()


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_coords(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _has_flanked_gap(line: np.ndarray) -> bool:
    for idx in range(1, len(line) - 1):
        if not line[idx] and line[:idx].any() and line[idx + 1 :].any():
            return True
    return False


def _notch_directions(mask: np.ndarray) -> list[str]:
    directions: list[str] = []
    if _has_flanked_gap(mask[0, :]):
        directions.append("up")
    if _has_flanked_gap(mask[-1, :]):
        directions.append("down")
    if _has_flanked_gap(mask[:, 0]):
        directions.append("left")
    if _has_flanked_gap(mask[:, -1]):
        directions.append("right")
    return directions


def _axis_size(mask: np.ndarray, direction: str) -> int:
    return mask.shape[1] if direction in {"left", "right"} else mask.shape[0]


def _final_origin(
    shape: tuple[int, int],
    direction: str,
    start_r0: int,
    start_c0: int,
    grid_shape: tuple[int, int],
) -> tuple[int, int]:
    h, w = shape
    H, W = grid_shape
    if direction == "left":
        assert 2 * w <= W
        return start_r0, w
    if direction == "right":
        assert 2 * w <= W
        return start_r0, W - 2 * w
    if direction == "up":
        assert 2 * h <= H
        return h, start_c0
    assert direction == "down"
    assert 2 * h <= H
    return H - 2 * h, start_c0


def _placed_coords(mask: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    return {
        (int(r0 + r), int(c0 + c))
        for r, c in np.argwhere(mask)
    }


def _dilate_chebyshev1(coords: set[tuple[int, int]], grid_shape: tuple[int, int]) -> set[tuple[int, int]]:
    H, W = grid_shape
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    out.add((rr, cc))
    return out


def _chebyshev_distance(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    return min(max(abs(r1 - r2), abs(c1 - c2)) for r1, c1 in a for r2, c2 in b)


def _grid_signature(grid: np.ndarray) -> str:
    payload = f"{grid.shape[0]}x{grid.shape[1]}|".encode() + grid.astype(np.uint8, copy=False).tobytes()
    return hashlib.sha1(payload).hexdigest()


def _extract_blue_objects(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    components = _component_coords(g == BLUE)
    objects: list[dict[str, object]] = []
    for coords in components:
        bbox = _bbox_from_coords(coords)
        mask = _mask_from_coords(coords, bbox)
        directions = _notch_directions(mask)
        assert len(directions) == 1, "each object must have exactly one flanked bbox-edge gap"
        objects.append(
            {
                "coords": coords,
                "bbox": bbox,
                "mask": mask,
                "direction": directions[0],
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each notched blue object toward its indicated wall, bounce it back by its own axis length, and stamp a green halo.

    Input structure:
    - The grid is yellow background with 2-3 separated blue 4-connected objects.
    - Each blue object occupies 4-8 cells and has exactly one bbox edge whose border contains a flanked background gap.
    - The unique gap edge is the object's slide direction, and objects do not touch each other.

    Rule:
    - For each blue object, inspect only its bounding-box top, bottom, left, and right edges; the edge containing a flanked background gap determines whether it slides up, down, left, or right.
    - Translate the whole object in that direction until its front reaches the grid boundary, reverse direction, then move it back by a distance equal to its bbox width for horizontal motion or bbox height for vertical motion.
    - Draw the object at that final location in red, clear the original location to yellow, and recolor every yellow cell in the Chebyshev-distance-1 neighborhood of each final red pixel to green without overwriting non-background cells.

    Key insight:
    - The gap edge chooses the wall, but the rebound distance comes from the object's own bbox size along that same axis, so both the notch direction and the object's width or height are necessary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = g.shape

    objects = _extract_blue_objects(g)
    assert 2 <= len(objects) <= 3

    out = np.full((H, W), BG, dtype=int)
    red_cells: list[tuple[int, int]] = []
    occupied: set[tuple[int, int]] = set()

    for obj in objects:
        bbox = obj["bbox"]
        mask = obj["mask"]
        direction = obj["direction"]
        r0, c0, r1, c1 = bbox  # type: ignore[misc]
        h, w = mask.shape  # type: ignore[union-attr]
        final_r0, final_c0 = _final_origin((h, w), direction, r0, c0, (H, W))  # type: ignore[arg-type]

        for rr, cc in _placed_coords(mask, final_r0, final_c0):  # type: ignore[arg-type]
            assert (rr, cc) not in occupied, "final objects must not overlap"
            occupied.add((rr, cc))
            red_cells.append((rr, cc))
            out[rr, cc] = RED

    for r, c in red_cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and out[rr, cc] == BG:
                    out[rr, cc] = GREEN

    return out


def _build_shape_catalog() -> dict[str, list[np.ndarray]]:
    catalog = {direction: [] for direction in DIRS}
    for h in range(2, 5):
        for w in range(2, 5):
            cell_count = h * w
            for bits in range(1 << cell_count):
                if bits.bit_count() < 4 or bits.bit_count() > 8:
                    continue
                flat = np.array([(bits >> idx) & 1 for idx in range(cell_count)], dtype=bool)
                mask = flat.reshape(h, w)
                if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
                    continue
                if len(_component_coords(mask)) != 1:
                    continue
                directions = _notch_directions(mask)
                if len(directions) != 1:
                    continue
                catalog[directions[0]].append(mask.copy())

    for direction in DIRS:
        catalog[direction].sort(key=lambda m: (m.sum(), m.shape[0], m.shape[1], m.astype(np.uint8).tobytes()))
    return catalog


_SHAPE_CATALOG = _build_shape_catalog()


def _sample_directions(n: int) -> list[str]:
    horizontal = random.choice(["left", "right"])
    vertical = random.choice(["up", "down"])
    if n == 2:
        directions = [horizontal, vertical]
    else:
        remaining = [d for d in DIRS if d not in {horizontal, vertical}]
        directions = [horizontal, vertical, random.choice(remaining)]
    random.shuffle(directions)
    return directions


def _candidate_shapes(direction: str, H: int, W: int) -> list[np.ndarray]:
    options = []
    for mask in _SHAPE_CATALOG[direction]:
        h, w = mask.shape
        if 2 * h <= H and 2 * w <= W:
            options.append(mask)
    return options


def _render_input(specs: list[dict[str, object]], grid_shape: tuple[int, int]) -> np.ndarray:
    H, W = grid_shape
    grid = np.full((H, W), BG, dtype=int)
    for spec in specs:
        mask = spec["mask"]  # type: ignore[assignment]
        r0 = spec["input_r0"]  # type: ignore[assignment]
        c0 = spec["input_c0"]  # type: ignore[assignment]
        for r, c in _placed_coords(mask, r0, c0):  # type: ignore[arg-type]
            grid[r, c] = BLUE
    return grid


def _render_output(specs: list[dict[str, object]], grid_shape: tuple[int, int]) -> np.ndarray:
    H, W = grid_shape
    out = np.full((H, W), BG, dtype=int)
    red_coords: set[tuple[int, int]] = set()
    for spec in specs:
        mask = spec["mask"]  # type: ignore[assignment]
        r0 = spec["final_r0"]  # type: ignore[assignment]
        c0 = spec["final_c0"]  # type: ignore[assignment]
        coords = _placed_coords(mask, r0, c0)  # type: ignore[arg-type]
        assert not (coords & red_coords)
        red_coords |= coords
    for r, c in red_coords:
        out[r, c] = RED
    for r, c in red_coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and out[rr, cc] == BG:
                    out[rr, cc] = GREEN
    return out


def _axis_candidates(direction: str, size: int, extent: int, final_origin: int) -> list[int]:
    candidates = [pos for pos in range(1, extent - size) if abs(pos - final_origin) >= 2]
    random.shuffle(candidates)
    return candidates


def _try_place_objects(H: int, W: int, shapes: list[np.ndarray], directions: list[str]) -> list[dict[str, object]] | None:
    order = sorted(range(len(shapes)), key=lambda i: (shapes[i].shape[0] * shapes[i].shape[1]), reverse=True)
    specs: list[dict[str, object] | None] = [None] * len(shapes)
    input_coords_used: list[set[tuple[int, int]]] = []
    final_halos_used: list[set[tuple[int, int]]] = []

    for idx in order:
        mask = shapes[idx]
        direction = directions[idx]
        h, w = mask.shape
        if direction in {"left", "right"}:
            final_c0 = w if direction == "left" else W - 2 * w
            rows = list(range(1, H - h))
            random.shuffle(rows)
            placed = False
            for r0 in rows:
                final_coords = _placed_coords(mask, r0, final_c0)
                final_halo = _dilate_chebyshev1(final_coords, (H, W))
                if any(final_halo & used for used in final_halos_used):
                    continue
                for c0 in _axis_candidates(direction, w, W, final_c0):
                    input_coords = _placed_coords(mask, r0, c0)
                    if any(_chebyshev_distance(input_coords, used) < MIN_INPUT_GAP for used in input_coords_used):
                        continue
                    specs[idx] = {
                        "mask": mask,
                        "direction": direction,
                        "input_r0": r0,
                        "input_c0": c0,
                        "final_r0": r0,
                        "final_c0": final_c0,
                    }
                    input_coords_used.append(input_coords)
                    final_halos_used.append(final_halo)
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                return None
        else:
            final_r0 = h if direction == "up" else H - 2 * h
            cols = list(range(1, W - w))
            random.shuffle(cols)
            placed = False
            for c0 in cols:
                final_coords = _placed_coords(mask, final_r0, c0)
                final_halo = _dilate_chebyshev1(final_coords, (H, W))
                if any(final_halo & used for used in final_halos_used):
                    continue
                for r0 in _axis_candidates(direction, h, H, final_r0):
                    input_coords = _placed_coords(mask, r0, c0)
                    if any(_chebyshev_distance(input_coords, used) < MIN_INPUT_GAP for used in input_coords_used):
                        continue
                    specs[idx] = {
                        "mask": mask,
                        "direction": direction,
                        "input_r0": r0,
                        "input_c0": c0,
                        "final_r0": final_r0,
                        "final_c0": c0,
                    }
                    input_coords_used.append(input_coords)
                    final_halos_used.append(final_halo)
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                return None

    return [spec for spec in specs if spec is not None]  # type: ignore[return-value]


def _validate_specs(specs: list[dict[str, object]], grid_shape: tuple[int, int]) -> bool:
    H, W = grid_shape
    if not (2 <= len(specs) <= 3):
        return False

    directions = [spec["direction"] for spec in specs]
    if not any(direction in {"left", "right"} for direction in directions):
        return False
    if not any(direction in {"up", "down"} for direction in directions):
        return False

    axis_sizes = [_axis_size(spec["mask"], spec["direction"]) for spec in specs]  # type: ignore[arg-type]
    if len(set(axis_sizes)) < 2:
        return False

    input_sets = [
        _placed_coords(spec["mask"], spec["input_r0"], spec["input_c0"])  # type: ignore[arg-type]
        for spec in specs
    ]
    for i in range(len(input_sets)):
        for j in range(i + 1, len(input_sets)):
            if _chebyshev_distance(input_sets[i], input_sets[j]) < MIN_INPUT_GAP:
                return False

    halos = []
    for spec in specs:
        mask = spec["mask"]  # type: ignore[assignment]
        final_r0 = spec["final_r0"]  # type: ignore[assignment]
        final_c0 = spec["final_c0"]  # type: ignore[assignment]
        final_coords = _placed_coords(mask, final_r0, final_c0)  # type: ignore[arg-type]
        if any(not (0 <= r < H and 0 <= c < W) for r, c in final_coords):
            return False
        halos.append(_dilate_chebyshev1(final_coords, (H, W)))

        input_r0 = spec["input_r0"]  # type: ignore[assignment]
        input_c0 = spec["input_c0"]  # type: ignore[assignment]
        if (input_r0, input_c0) == (final_r0, final_c0):
            return False

    for i in range(len(halos)):
        for j in range(i + 1, len(halos)):
            if halos[i] & halos[j]:
                return False

    input_grid = _render_input(specs, grid_shape)
    output_grid = _render_output(specs, grid_shape)

    if len(_extract_blue_objects(input_grid)) != len(specs):
        return False
    if not np.array_equal(main(input_grid), output_grid):
        return False

    input_sig = _grid_signature(input_grid)
    output_sig = _grid_signature(output_grid)
    if input_sig in _USED_INPUT_SIGNATURES or output_sig in _USED_OUTPUT_SIGNATURES:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_SIZE_RANGE)
    W = random.randint(*GRID_SIZE_RANGE)
    n = random.choice(OBJECT_COUNT_CHOICES)
    directions = _sample_directions(n)

    shapes: list[np.ndarray] = []
    for direction in directions:
        options = _candidate_shapes(direction, H, W)
        if not options:
            return None
        shapes.append(random.choice(options))

    if len(set(_axis_size(mask, direction) for mask, direction in zip(shapes, directions))) < 2:
        return None

    specs = _try_place_objects(H, W, shapes, directions)
    if specs is None:
        return None
    if not _validate_specs(specs, (H, W)):
        return None

    input_grid = _render_input(specs, (H, W))
    output_grid = _render_output(specs, (H, W))
    _USED_INPUT_SIGNATURES.add(_grid_signature(input_grid))
    _USED_OUTPUT_SIGNATURES.add(_grid_signature(output_grid))
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
