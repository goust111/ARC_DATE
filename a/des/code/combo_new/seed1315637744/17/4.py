from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
YELLOW = 4
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)

GRID_MIN = 14
GRID_MAX = 24
MAX_BBOX = 5
MIN_OBJECTS = 3
MAX_OBJECTS = 6
GEN_RETRIES = 400

LR_ONLY = "lr_only"
UD_ONLY = "ud_only"
BOTH_AXES = "both_axes"
KINDS = (LR_ONLY, UD_ONLY, BOTH_AXES)
COLOR_POOL = tuple(color for color in range(10) if color not in {BG, YELLOW})


def _normalize_cells(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _cells_to_mask(cells: tuple[tuple[int, int], ...]) -> np.ndarray:
    height = max(r for r, _c in cells) + 1
    width = max(c for _r, c in cells) + 1
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _generate_polyominoes(max_area: int = 8) -> dict[int, set[tuple[tuple[int, int], ...]]]:
    polyominoes: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for area in range(2, max_area + 1):
        shapes: set[tuple[tuple[int, int], ...]] = set()
        for shape in polyominoes[area - 1]:
            shape_cells = set(shape)
            for r, c in shape:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in shape_cells:
                        continue
                    shapes.add(_normalize_cells(shape_cells | {(nr, nc)}))
        polyominoes[area] = shapes
    return polyominoes


def _symmetry_kind(mask: np.ndarray) -> str | None:
    mirror_lr = bool(np.array_equal(mask, np.fliplr(mask)))
    mirror_ud = bool(np.array_equal(mask, np.flipud(mask)))
    if mirror_lr and not mirror_ud:
        return LR_ONLY
    if mirror_ud and not mirror_lr:
        return UD_ONLY
    if mirror_lr and mirror_ud:
        return BOTH_AXES
    return None


def _build_shape_pools() -> dict[str, tuple[np.ndarray, ...]]:
    pools: dict[str, list[np.ndarray]] = {kind: [] for kind in KINDS}
    for area, shapes in _generate_polyominoes().items():
        if area < 3:
            continue
        for shape in shapes:
            mask = _cells_to_mask(shape)
            if min(mask.shape) < 2 or max(mask.shape) > MAX_BBOX:
                continue
            kind = _symmetry_kind(mask)
            if kind is not None:
                pools[kind].append(mask)
    return {kind: tuple(masks) for kind, masks in pools.items()}


SHAPE_POOLS = _build_shape_pools()


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for start_r in range(height):
        for start_c in range(width):
            if not mask[start_r, start_c] or seen[start_r, start_c]:
                continue
            queue: deque[tuple[int, int]] = deque([(start_r, start_c)])
            seen[start_r, start_c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for dr, dc in NEIGHBORS4:
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < height and 0 <= cc < width):
                        continue
                    if seen[rr, cc] or not mask[rr, cc]:
                        continue
                    seen[rr, cc] = True
                    queue.append((rr, cc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_coords(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _cells_from_mask(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    return {
        (int(top + r), int(left + c))
        for r, c in np.argwhere(mask)
    }


def _expand8(cells: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    expanded.add((rr, cc))
    return expanded


def _transformed_cells(
    mask: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    kind: str,
) -> set[tuple[int, int]]:
    if kind == LR_ONLY:
        return {(int(top + r), int(c)) for r, c in np.argwhere(mask)}
    if kind == UD_ONLY:
        dst_top = int(height - mask.shape[0])
        return {(int(dst_top + r), int(left + c)) for r, c in np.argwhere(mask)}
    assert kind == BOTH_AXES
    return _cells_from_mask(mask, top, left)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify each small monochrome object by its bbox symmetries, then slide or recolor it.

    Input structure:
    - The background is black.
    - Several separated single-color 4-connected objects appear away from the bottom-right corner.
    - Each object has 3-8 pixels and is exactly one of three bbox-symmetry types:
      left-right mirror only, up-down mirror only, or symmetric in both directions.
    - The bottom-right cell is an isolated yellow marker pixel.

    Rule:
    - If an object is symmetric only under left-right mirroring across its bbox vertical midline,
      move it horizontally so its bbox touches the left edge and clear its old cells.
    - If an object is symmetric only under up-down mirroring across its bbox horizontal midline,
      move it vertically so its bbox touches the bottom edge and clear its old cells.
    - If an object is symmetric in both directions, keep it in place and recolor all its pixels
      to the marker color.
    - Remove the bottom-right marker pixel in the output.

    Key insight:
    - The transformation is driven by each object's intrinsic symmetry class, so the solver must
      analyze geometry first and only then choose between left-translation, downward translation,
      or recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    height, width = map(int, g.shape)
    assert int(g[height - 1, width - 1]) != BG, "expected a bottom-right marker pixel"
    marker_color = int(g[height - 1, width - 1])

    work = g.copy()
    work[height - 1, width - 1] = BG
    components = _component_coords(work != BG)
    assert components, "expected at least one foreground object"

    out = np.full_like(g, BG)
    for coords in components:
        colors = {int(work[r, c]) for r, c in coords}
        assert len(colors) == 1, "each object must be monochrome"
        color = colors.pop()
        assert 3 <= int(coords.shape[0]) <= 8, "objects must have 3-8 pixels"

        comp_bbox = _bbox(coords)
        mask = _mask_from_coords(coords, comp_bbox)
        kind = _symmetry_kind(mask)
        assert kind is not None, "each object must match one supported symmetry class"

        if kind == LR_ONLY:
            shifted = {(int(r), int(c - comp_bbox[1])) for r, c in coords}
            paint_color = color
        elif kind == UD_ONLY:
            row_shift = int(height - 1 - comp_bbox[2])
            shifted = {(int(r + row_shift), int(c)) for r, c in coords}
            paint_color = color
        else:
            shifted = {(int(r), int(c)) for r, c in coords}
            paint_color = marker_color

        for r, c in shifted:
            assert 0 <= r < height and 0 <= c < width, "transformed object left the grid"
            assert out[r, c] == BG, "transformed objects must not overlap"
            out[r, c] = int(paint_color)

    out[height - 1, width - 1] = BG
    return out


def _sample_specs() -> list[dict[str, object]]:
    total = random.randint(MIN_OBJECTS, MAX_OBJECTS)
    counts = {LR_ONLY: 1, UD_ONLY: 1, BOTH_AXES: 1}
    for _ in range(total - 3):
        counts[random.choice(KINDS)] += 1

    colors = random.sample(COLOR_POOL, total)
    random.shuffle(colors)
    specs: list[dict[str, object]] = []
    color_index = 0
    for kind in KINDS:
        for _ in range(counts[kind]):
            mask = random.choice(SHAPE_POOLS[kind]).copy()
            specs.append(
                {
                    "kind": kind,
                    "mask": mask,
                    "color": int(colors[color_index]),
                }
            )
            color_index += 1

    random.shuffle(specs)
    specs.sort(
        key=lambda spec: (
            0 if spec["kind"] != BOTH_AXES else 1,
            -int(np.count_nonzero(spec["mask"])),
            -int(spec["mask"].shape[0] * spec["mask"].shape[1]),
        )
    )
    return specs


def _legal_positions(
    mask: np.ndarray,
    kind: str,
    height: int,
    width: int,
    input_forbidden: set[tuple[int, int]],
    output_forbidden: set[tuple[int, int]],
) -> list[tuple[int, int, set[tuple[int, int]], set[tuple[int, int]]]]:
    h, w = map(int, mask.shape)
    legal: list[tuple[int, int, set[tuple[int, int]], set[tuple[int, int]]]] = []
    for top in range(height - h + 1):
        for left in range(width - w + 1):
            if kind == LR_ONLY and left == 0:
                continue
            if kind == UD_ONLY and top == height - h:
                continue

            input_cells = _cells_from_mask(mask, top, left)
            if input_cells & input_forbidden:
                continue

            output_cells = _transformed_cells(mask, top, left, height, width, kind)
            if (height - 1, width - 1) in output_cells:
                continue
            if output_cells & output_forbidden:
                continue

            legal.append((int(top), int(left), input_cells, output_cells))
    return legal


def _place_specs(height: int, width: int, specs: list[dict[str, object]]) -> list[dict[str, object]] | None:
    marker = {(height - 1, width - 1)}
    input_forbidden = _expand8(marker, height, width)
    output_forbidden = set(marker)
    placed: list[dict[str, object]] = []

    for spec in specs:
        mask = spec["mask"]
        kind = str(spec["kind"])
        positions = _legal_positions(mask, kind, height, width, input_forbidden, output_forbidden)
        if not positions:
            return None

        top, left, input_cells, output_cells = random.choice(positions)
        placed_spec = dict(spec)
        placed_spec["top"] = int(top)
        placed_spec["left"] = int(left)
        placed_spec["input_cells"] = input_cells
        placed_spec["output_cells"] = output_cells
        placed.append(placed_spec)

        input_forbidden |= _expand8(input_cells, height, width)
        output_forbidden |= _expand8(output_cells, height, width)
    return placed


def _render_input(height: int, width: int, placed: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for spec in placed:
        for r, c in spec["input_cells"]:
            grid[r, c] = int(spec["color"])
    grid[height - 1, width - 1] = YELLOW
    return grid


def _render_expected(height: int, width: int, placed: list[dict[str, object]]) -> np.ndarray:
    out = np.full((height, width), BG, dtype=int)
    for spec in placed:
        color = YELLOW if spec["kind"] == BOTH_AXES else int(spec["color"])
        for r, c in spec["output_cells"]:
            out[r, c] = color
    out[height - 1, width - 1] = BG
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    specs = _sample_specs()
    placed = _place_specs(height, width, specs)
    if placed is None:
        return None

    inp = _render_input(height, width, placed)
    expected = _render_expected(height, width, placed)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-classification ARC input")
