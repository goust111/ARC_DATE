import random
from collections import deque

import numpy as np


BLACK = 0
ARC_MAX = 30
COLORS = tuple(range(1, 10))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# --------------- generator config ---------------
OBJECT_COUNT_RANGE = (2, 4)
FULL_MASK_PROB = 0.5
MIN_OBJECT_AREA = 4
MAX_OBJECT_BBOX_AREA = 16
MIN_MASK_DENSITY = 0.55
PLACEMENT_PAD = 1
MAX_GENERATION_ATTEMPTS = 400

RECT_DIMS = tuple(
    (h, w)
    for h in range(2, 6)
    for w in range(2, 6)
    if MIN_OBJECT_AREA <= h * w <= MAX_OBJECT_BBOX_AREA
)
CARVED_DIMS = tuple((h, w) for h, w in RECT_DIMS if h * w >= 6)


def _extract_objects(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == BLACK:
                continue

            queue = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) == BLACK:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            objects.append(
                {
                    "coords": np.array(coords, dtype=int),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                    "colors": {int(grid[rr, cc]) for rr, cc in coords},
                }
            )

    return objects


def _snake_sequence(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> list[int]:
    r0, c0, r1, c1 = bbox
    seq: list[int] = []
    for r in range(r0, r1 + 1):
        cols = range(c0, c1 + 1) if (r - r0) % 2 == 0 else range(c1, c0 - 1, -1)
        for c in cols:
            value = int(grid[r, c])
            if value != BLACK:
                seq.append(value)
    return seq


def _render_sequences(sequences: list[list[int]]) -> np.ndarray:
    out_h = 2 * len(sequences) - 1
    out_w = max(len(seq) for seq in sequences)
    out = np.full((out_h, out_w), BLACK, dtype=int)
    for i, seq in enumerate(sequences):
        out[2 * i, : len(seq)] = np.array(seq, dtype=int)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Linearize each foreground object into a horizontal snake-order color line.

    Input structure:
    - The background is Black.
    - The grid contains several 4-connected multicolor foreground objects separated by black cells.
    - Each object may occupy an irregular subset of its tight bbox, but every object uses exactly 2 or 3 visible colors.

    Rule:
    - Find every non-black 4-connected object and sort the objects by the row-major order of their bbox upper-left corner.
    - For one object, read its tight bbox in snake order: the first bbox row left-to-right, the next row right-to-left, alternating while skipping black cells.
    - Emit one left-aligned horizontal line per object whose length equals the object area and whose colors follow that snake-order sequence; keep one full black row between consecutive lines.

    Key insight:
    - The 2D shape is discarded after its pixels are read: the solution is the alternating row-direction sequence of visible colors inside each object's bbox.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= ARC_MAX, "ARC grids must be at most 30x30"

    objects = _extract_objects(g)
    assert objects, "expected at least one foreground object"

    sequences: list[list[int]] = []
    for obj in sorted(objects, key=lambda item: (item["bbox"][0], item["bbox"][1])):
        colors = obj["colors"]
        assert 2 <= len(colors) <= 3, "each object should use 2 or 3 colors"
        seq = _snake_sequence(g, obj["bbox"])
        assert len(seq) == obj["area"], "sequence length must match object area"
        sequences.append(seq)

    return _render_sequences(sequences)


# --------------- generator helpers ---------------

def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    queue = deque([start])
    seen = {start}

    while queue:
        r, c = queue.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not bool(mask[nr, nc]):
                continue
            if (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            queue.append((nr, nc))

    return len(seen) == int(mask.sum())


def _supports_snake_mask(mask: np.ndarray) -> bool:
    if mask.shape[0] < 2 or mask.shape[1] < 2:
        return False
    row_counts = mask.sum(axis=1)
    return bool((row_counts[1::2] >= 2).any())


def _mask_density(mask: np.ndarray) -> float:
    return float(mask.sum()) / float(mask.size)


def _snake_coords(mask: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, list[int]]]]:
    coords: list[tuple[int, int]] = []
    row_indices: list[tuple[int, list[int]]] = []
    index = 0

    h, w = mask.shape
    for r in range(h):
        row: list[tuple[int, int]] = []
        cols = range(w) if r % 2 == 0 else range(w - 1, -1, -1)
        for c in cols:
            if bool(mask[r, c]):
                row.append((r, c))
                coords.append((r, c))
        row_indices.append((r, list(range(index, index + len(row)))))
        index += len(row)

    return coords, row_indices


def _make_full_mask() -> np.ndarray:
    h, w = random.choice(RECT_DIMS)
    mask = np.ones((h, w), dtype=bool)
    assert _supports_snake_mask(mask)
    return mask


def _make_carved_mask() -> np.ndarray | None:
    for _ in range(80):
        h, w = random.choice(CARVED_DIMS)
        mask = np.ones((h, w), dtype=bool)
        max_remove = min(
            4,
            int(mask.sum()) - MIN_OBJECT_AREA,
            int(mask.sum()) - int(np.ceil(mask.size * MIN_MASK_DENSITY)),
        )
        if max_remove < 1:
            continue

        target_remove = random.randint(1, max_remove)
        removed = 0
        for _ in range(target_remove):
            candidates: list[np.ndarray] = []
            for r, c in np.argwhere(mask):
                trial = mask.copy()
                trial[int(r), int(c)] = False
                if not trial.any():
                    continue
                trial = _crop_mask(trial)
                if int(trial.sum()) < MIN_OBJECT_AREA:
                    continue
                if _mask_density(trial) < MIN_MASK_DENSITY:
                    continue
                if not _mask_connected(trial) or not _supports_snake_mask(trial):
                    continue
                candidates.append(trial)

            if not candidates:
                break

            mask = random.choice(candidates)
            removed += 1

        if removed and int(mask.sum()) < mask.size and _supports_snake_mask(mask):
            return mask

    return None


def _colorize_mask(mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    color_count = random.choice((2, 3))
    palette = random.sample(COLORS, color_count)
    coords, row_indices = _snake_coords(mask)
    seq = [random.choice(palette) for _ in coords]

    for color, idx in zip(palette, random.sample(range(len(seq)), color_count), strict=True):
        seq[idx] = color

    odd_rows = [indices for r, indices in row_indices if r % 2 == 1 and len(indices) >= 2]
    assert odd_rows, "generated mask must reveal the snake-order reversal"
    target = random.choice(odd_rows)

    protected = {target[0], target[-1]}
    seq[target[0]] = palette[0]
    seq[target[-1]] = palette[1]
    if color_count == 3 and len(target) >= 3:
        mid = target[len(target) // 2]
        protected.add(mid)
        seq[mid] = palette[2]

    for color in palette:
        if color in seq:
            continue
        options = [i for i in range(len(seq)) if i not in protected]
        seq[random.choice(options or list(range(len(seq))))] = color

    patch = np.full(mask.shape, BLACK, dtype=int)
    for (r, c), color in zip(coords, seq, strict=True):
        patch[r, c] = int(color)
    return patch, seq


def _bbox_free(occupied: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    hh, ww = occupied.shape
    r0 = max(0, r - PLACEMENT_PAD)
    r1 = min(hh, r + h + PLACEMENT_PAD)
    c0 = max(0, c - PLACEMENT_PAD)
    c1 = min(ww, c + w + PLACEMENT_PAD)
    return not bool(occupied[r0:r1, c0:c1].any())


def _mark_bbox(occupied: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    hh, ww = occupied.shape
    r0 = max(0, r - PLACEMENT_PAD)
    r1 = min(hh, r + h + PLACEMENT_PAD)
    c0 = max(0, c - PLACEMENT_PAD)
    c1 = min(ww, c + w + PLACEMENT_PAD)
    occupied[r0:r1, c0:c1] = True


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r: int, c: int) -> None:
    h, w = patch.shape
    target = grid[r : r + h, c : c + w]
    mask = patch != BLACK
    target[mask] = patch[mask]


def _sample_scene_shape(patches: list[dict]) -> tuple[int, int]:
    heights = [int(obj["patch"].shape[0]) for obj in patches]
    widths = [int(obj["patch"].shape[1]) for obj in patches]
    n = len(patches)

    max_h, max_w = max(heights), max(widths)
    sum_h, sum_w = sum(heights), sum(widths)

    h_lo = max(max_h + 2, min(12, max_h + n + 2))
    h_hi = min(ARC_MAX, max(h_lo, min(24, sum_h + n + 4)))
    w_lo = max(max_w + 2, min(14, max_w + n + 4))
    w_hi = min(ARC_MAX, max(w_lo, min(28, sum_w + n + 6)))
    return random.randint(h_lo, h_hi), random.randint(w_lo, w_hi)


def _place_objects(patches: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    h, w = _sample_scene_shape(patches)
    grid = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    placed: list[dict] = []

    order = sorted(
        patches,
        key=lambda obj: (obj["patch"].shape[0] * obj["patch"].shape[1], obj["patch"].shape[1]),
        reverse=True,
    )

    for obj in order:
        patch = obj["patch"]
        ph, pw = patch.shape
        candidates = [
            (r, c)
            for r in range(1, h - ph)
            for c in range(1, w - pw)
            if _bbox_free(occupied, r, c, ph, pw)
        ]
        if not candidates:
            return None

        r, c = random.choice(candidates)
        _paste_patch(grid, patch, r, c)
        _mark_bbox(occupied, r, c, ph, pw)
        placed.append({"row": r, "col": c, "sequence": obj["sequence"]})

    return grid, placed


def _make_object() -> dict | None:
    if random.random() < FULL_MASK_PROB:
        mask = _make_full_mask()
    else:
        mask = _make_carved_mask()
        if mask is None:
            return None

    patch, sequence = _colorize_mask(mask)
    return {"patch": patch, "sequence": sequence}


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    patches = []
    for _ in range(n_objects):
        obj = _make_object()
        if obj is None:
            return None
        patches.append(obj)

    placed = _place_objects(patches)
    if placed is None:
        return None

    grid, layout = placed
    expected = _render_sequences(
        [obj["sequence"] for obj in sorted(layout, key=lambda item: (item["row"], item["col"]))]
    )

    if max(grid.shape) > ARC_MAX or max(expected.shape) > ARC_MAX:
        return None

    objects = _extract_objects(grid)
    if len(objects) != n_objects:
        return None
    if any(not (2 <= len(obj["colors"]) <= 3) for obj in objects):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
