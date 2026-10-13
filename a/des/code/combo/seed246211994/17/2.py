from __future__ import annotations

from collections import Counter

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
ARC_COLORS = tuple(range(10))

# Generator controls.
GRID_HEIGHT_RANGE = (12, 19)
GRID_WIDTH_RANGE = (12, 19)
GROUP_COUNT_RANGE = (2, 5)
INSTANCE_COUNT_RANGE = (1, 4)
TOTAL_OBJECT_RANGE = (4, 9)
MAX_GROUPS = 5
MAX_SHAPE_SIZE = 5
PLACEMENT_TRIES = 300
GENERATION_TRIES = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _normalized_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _signature_from_mask(mask: np.ndarray) -> tuple[int, int, bytes]:
    tight = np.asarray(mask, dtype=bool)
    return int(tight.shape[0]), int(tight.shape[1]), tight.tobytes()


def _component_records(mask: np.ndarray) -> list[dict[str, object]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    records: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.asarray(cells, dtype=int)
            bbox = _bbox(coords)
            norm = _normalized_mask(coords)
            records.append(
                {
                    "coords": coords,
                    "bbox": bbox,
                    "mask": norm,
                    "signature": _signature_from_mask(norm),
                }
            )

    return records


def _placeholder_color(grid: np.ndarray, bg: int) -> int:
    fg = np.asarray(grid, dtype=int)
    fg = fg[fg != bg]
    if fg.size == 0:
        return bg
    values, counts = np.unique(fg, return_counts=True)
    return int(values[np.argmax(counts)])


def _available_output_colors(bg: int, placeholder: int) -> list[int]:
    colors = [color for color in ARC_COLORS if color not in {bg, placeholder}]
    if placeholder != bg:
        colors.append(int(placeholder))
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """Group 4-connected placeholder objects by normalized shape and recolor each shape class.

    Input structure
    - The background is the dominant color.
    - Every foreground object is a separate 4-connected component.
    - All foreground objects share the same placeholder color.

    Rule
    - Normalize each object's mask by translating its tight bounding box to the top-left.
    - Equal normalized masks belong to the same shape class.
    - Scan objects by top-left position; each newly encountered shape class gets the next
      available output color, and later copies of that shape reuse the same color.
    - Leave the background unchanged.

    Key insight
    - The placeholder foreground color is irrelevant; exact translated shape is the only
      feature that determines output grouping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    records = _component_records(g != bg)
    if not records:
        return g.copy()

    placeholder = _placeholder_color(g, bg)
    palette = _available_output_colors(bg, placeholder)

    ordered = sorted(records, key=lambda rec: (rec["bbox"][0], rec["bbox"][1]))
    signatures = {rec["signature"] for rec in ordered}
    assert len(signatures) <= len(palette), "too many distinct shapes for the ARC palette"

    out = g.copy()
    signature_to_color: dict[tuple[int, int, bytes], int] = {}
    next_color = 0
    for rec in ordered:
        signature = rec["signature"]
        if signature not in signature_to_color:
            signature_to_color[signature] = int(palette[next_color])
            next_color += 1
        color = signature_to_color[signature]
        coords = rec["coords"]
        out[coords[:, 0], coords[:, 1]] = color

    return out


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _transform_shape(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.rot90(mask, int(rng.integers(0, 4)))
    if bool(rng.integers(0, 2)):
        out = np.fliplr(out)
    if bool(rng.integers(0, 2)):
        out = np.flipud(out)
    return _tight_crop(out)


def _line_shape(rng: np.random.Generator) -> np.ndarray:
    length = int(rng.integers(2, MAX_SHAPE_SIZE + 1))
    return np.ones((1, length), dtype=bool)


def _rectangle_shape(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(2, MAX_SHAPE_SIZE))
    w = int(rng.integers(2, MAX_SHAPE_SIZE))
    return np.ones((h, w), dtype=bool)


def _l_shape(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(3, MAX_SHAPE_SIZE + 1))
    w = int(rng.integers(3, MAX_SHAPE_SIZE + 1))
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = True
    mask[-1, :] = True
    return mask


def _t_shape(rng: np.random.Generator) -> np.ndarray:
    stem = int(rng.integers(2, MAX_SHAPE_SIZE))
    bar = int(rng.choice([3, 5]))
    mask = np.zeros((stem + 1, bar), dtype=bool)
    mask[0, :] = True
    mask[:, bar // 2] = True
    return mask


def _plus_shape() -> np.ndarray:
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, :] = True
    mask[:, 1] = True
    return mask


def _polyomino_shape(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(2, MAX_SHAPE_SIZE + 1))
    w = int(rng.integers(2, MAX_SHAPE_SIZE + 1))
    target = int(rng.integers(3, min(h * w, 7) + 1))
    seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {seed}

    while len(cells) < target:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        choice = tuple(frontier)[int(rng.integers(0, len(frontier)))]
        cells.add(choice)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _tight_crop(mask)


def _random_shape(rng: np.random.Generator) -> np.ndarray:
    family = int(rng.integers(0, 6))
    if family == 0:
        base = np.array([[True]], dtype=bool)
    elif family == 1:
        base = _line_shape(rng)
    elif family == 2:
        base = _rectangle_shape(rng)
    elif family == 3:
        base = _l_shape(rng)
    elif family == 4:
        base = _t_shape(rng)
    else:
        base = _plus_shape() if bool(rng.integers(0, 2)) else _polyomino_shape(rng)
    return _transform_shape(base, rng)


def _sample_group_counts(rng: np.random.Generator, n_groups: int) -> list[int]:
    while True:
        counts = [int(rng.integers(*INSTANCE_COUNT_RANGE)) for _ in range(n_groups)]
        if max(counts) < 2:
            counts[int(rng.integers(0, n_groups))] = 2
        total = sum(counts)
        if TOTAL_OBJECT_RANGE[0] <= total <= TOTAL_OBJECT_RANGE[1]:
            return counts


def _sample_unique_shapes(rng: np.random.Generator, n_groups: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    signatures: set[tuple[int, int, bytes]] = set()
    attempts = 0
    while len(shapes) < n_groups and attempts < 200:
        attempts += 1
        shape = _random_shape(rng)
        sig = _signature_from_mask(shape)
        if sig in signatures:
            continue
        shapes.append(shape)
        signatures.add(sig)
    return shapes if len(shapes) == n_groups else None


def _reserve_bbox(guard: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(guard.shape[0], r0 + h + 1)
    cc1 = min(guard.shape[1], c0 + w + 1)
    guard[rr0:rr1, cc0:cc1] = True


def _can_place(guard: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(guard.shape[0], r0 + h + 1)
    cc1 = min(guard.shape[1], c0 + w + 1)
    return not bool(guard[rr0:rr1, cc0:cc1].any())


def _place_objects(
    grid: np.ndarray,
    shapes: list[np.ndarray],
    counts: list[int],
    placeholder: int,
    rng: np.random.Generator,
) -> bool:
    object_ids = [idx for idx, count in enumerate(counts) for _ in range(count)]
    rng.shuffle(object_ids)
    object_ids.sort(key=lambda idx: int(np.count_nonzero(shapes[idx])), reverse=True)

    guard = np.zeros_like(grid, dtype=bool)
    for idx in object_ids:
        shape = shapes[idx]
        h, w = shape.shape
        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = int(rng.integers(0, grid.shape[0] - h + 1))
            c0 = int(rng.integers(0, grid.shape[1] - w + 1))
            if not _can_place(guard, r0, c0, h, w):
                continue
            patch = grid[r0 : r0 + h, c0 : c0 + w]
            if np.any(patch[shape] != _background_color(grid)):
                continue
            patch[shape] = placeholder
            _reserve_bbox(guard, r0, c0, h, w)
            placed = True
            break
        if not placed:
            return False
    return True


def _validate_instance(grid: np.ndarray) -> bool:
    bg = _background_color(grid)
    if np.count_nonzero(grid == bg) <= grid.size // 2:
        return False

    records = _component_records(grid != bg)
    if not (TOTAL_OBJECT_RANGE[0] <= len(records) <= TOTAL_OBJECT_RANGE[1]):
        return False

    sig_counts = Counter(rec["signature"] for rec in records)
    if len(sig_counts) < 2 or len(sig_counts) > MAX_GROUPS or max(sig_counts.values()) < 2:
        return False

    if len(np.unique(grid[grid != bg])) != 1:
        return False

    out = main(grid)
    if not np.array_equal(out == bg, grid == bg):
        return False

    color_by_sig: dict[tuple[int, int, bytes], int] = {}
    sig_by_color: dict[int, tuple[int, int, bytes]] = {}
    for rec in records:
        coords = rec["coords"]
        colors = np.unique(out[coords[:, 0], coords[:, 1]])
        if len(colors) != 1:
            return False
        color = int(colors[0])
        if color == bg:
            return False
        sig = rec["signature"]
        if sig in color_by_sig and color_by_sig[sig] != color:
            return False
        if color in sig_by_color and sig_by_color[color] != sig:
            return False
        color_by_sig[sig] = color
        sig_by_color[color] = sig

    return len(color_by_sig) == len(sig_counts)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    rows = int(rng.integers(*GRID_HEIGHT_RANGE))
    cols = int(rng.integers(*GRID_WIDTH_RANGE))

    bg = int(rng.integers(0, 10))
    placeholder_choices = [color for color in ARC_COLORS if color != bg]
    placeholder = int(placeholder_choices[int(rng.integers(0, len(placeholder_choices)))])

    n_groups = int(rng.integers(*GROUP_COUNT_RANGE))
    counts = _sample_group_counts(rng, n_groups)
    shapes = _sample_unique_shapes(rng, n_groups)
    if shapes is None:
        return None

    grid = np.full((rows, cols), bg, dtype=int)
    if not _place_objects(grid, shapes, counts, placeholder, rng):
        return None
    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
