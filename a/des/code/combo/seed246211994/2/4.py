from __future__ import annotations

import numpy as np


_DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in _DIRS4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))

            comps.append({"color": color, "coords": np.asarray(coords, dtype=int)})

    return comps


def _component_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    c0 = int(cols.min())
    h = int(rows.max() - r0 + 1)
    w = int(cols.max() - c0 + 1)

    mask = np.zeros((h, w), dtype=bool)
    mask[rows - r0, cols - c0] = True
    return mask, (r0, c0)


def _flip_mode(mask: np.ndarray) -> str:
    top_gap = int(np.count_nonzero(~mask[0, :]))
    bottom_gap = int(np.count_nonzero(~mask[-1, :]))
    left_gap = int(np.count_nonzero(~mask[:, 0]))
    right_gap = int(np.count_nonzero(~mask[:, -1]))

    vertical_signal = int(top_gap + bottom_gap)
    horizontal_signal = int(left_gap + right_gap)
    assert vertical_signal + horizontal_signal > 0, "expected a notch on one side of the object"

    if vertical_signal > horizontal_signal:
        return "ud"
    if horizontal_signal > vertical_signal:
        return "lr"

    return "ud" if max(top_gap, bottom_gap) >= max(left_gap, right_gap) else "lr"


def _flipped_mask(mask: np.ndarray) -> np.ndarray:
    mode = _flip_mode(mask)
    return mask[::-1, :] if mode == "ud" else mask[:, ::-1]


def _paint_mask(canvas: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    region = canvas[top : top + mask.shape[0], left : left + mask.shape[1]]
    region[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group objects by their notch direction and flip them across the crossed axis.

    Input structure:
    - The most common color is background.
    - Each foreground object is a monochrome 4-connected near-rectangle.
    - Every object has exactly one one-cell-deep notch on one side.

    Rule:
    - A top or bottom notch triggers a vertical flip within the object's bbox.
    - A left or right notch triggers a horizontal flip within the object's bbox.
    - Every object stays in place and keeps its color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = np.full(g.shape, int(bg), dtype=int)

    for comp in _components(g, bg):
        mask, (r0, c0) = _component_mask(comp["coords"])
        _paint_mask(out, r0, c0, _flipped_mask(mask), int(comp["color"]))

    return out


# --------------- generator (codex) ---------------

COLOR_COUNT = 10
GRID_H_RANGE = (18, 27)      # inclusive-exclusive
GRID_W_RANGE = (18, 27)      # inclusive-exclusive
OBJECT_COUNT_RANGE = (3, 6)  # inclusive
OBJECT_H_RANGE = (4, 8)      # inclusive
OBJECT_W_RANGE = (4, 8)      # inclusive
OUTER_MARGIN = 1
OBJECT_PADDING = 1
MAX_GENERATION_ATTEMPTS = 400

_ORIENTATIONS = ("up", "down", "left", "right")


def _make_object_mask(rng: np.random.Generator, orientation: str) -> np.ndarray:
    h = int(rng.integers(OBJECT_H_RANGE[0], OBJECT_H_RANGE[1] + 1))
    w = int(rng.integers(OBJECT_W_RANGE[0], OBJECT_W_RANGE[1] + 1))
    mask = np.ones((h, w), dtype=bool)

    if orientation in ("up", "down"):
        notch_w = int(rng.integers(1, w - 1))
        start = int(rng.integers(1, w - notch_w))
        row = 0 if orientation == "up" else h - 1
        mask[row, start : start + notch_w] = False
    else:
        notch_h = int(rng.integers(1, h - 1))
        start = int(rng.integers(1, h - notch_h))
        col = 0 if orientation == "left" else w - 1
        mask[start : start + notch_h, col] = False

    return mask


def _sample_orientations(rng: np.random.Generator, count: int) -> list[str]:
    picked = [
        str(rng.choice(("up", "down"))),
        str(rng.choice(("left", "right"))),
    ]
    while len(picked) < count:
        picked.append(str(rng.choice(_ORIENTATIONS)))
    rng.shuffle(picked)
    return picked


def _valid_positions(reserved: np.ndarray, patch_h: int, patch_w: int) -> list[tuple[int, int]]:
    h, w = reserved.shape
    positions: list[tuple[int, int]] = []
    max_r = h - patch_h - OUTER_MARGIN
    max_c = w - patch_w - OUTER_MARGIN
    if max_r < OUTER_MARGIN or max_c < OUTER_MARGIN:
        return positions

    for r0 in range(OUTER_MARGIN, max_r + 1):
        for c0 in range(OUTER_MARGIN, max_c + 1):
            if not reserved[r0 : r0 + patch_h, c0 : c0 + patch_w].any():
                positions.append((int(r0), int(c0)))
    return positions


def _reserve_bbox(reserved: np.ndarray, top: int, left: int, patch_h: int, patch_w: int) -> None:
    h, w = reserved.shape
    r0 = max(0, top - OBJECT_PADDING)
    c0 = max(0, left - OBJECT_PADDING)
    r1 = min(h, top + patch_h + OBJECT_PADDING)
    c1 = min(w, left + patch_w + OBJECT_PADDING)
    reserved[r0:r1, c0:c1] = True


def _bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(values, counts)}
    bg_count = int(freq.get(int(bg), 0))
    other_max = max((count for color, count in freq.items() if color != int(bg)), default=0)
    return bg_count > other_max


def _flip_from_orientation(mask: np.ndarray, orientation: str) -> np.ndarray:
    return mask[::-1, :] if orientation in ("up", "down") else mask[:, ::-1]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    bg = int(rng.integers(0, COLOR_COUNT))
    n_objects = int(rng.integers(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1] + 1))

    colors = [int(c) for c in rng.choice([c for c in range(COLOR_COUNT) if c != bg], size=n_objects, replace=False)]
    orientations = _sample_orientations(rng, n_objects)

    objects = []
    for color, orientation in zip(colors, orientations):
        mask = _make_object_mask(rng, orientation)
        objects.append({"color": int(color), "orientation": orientation, "mask": mask})

    grid_h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1]))
    grid_w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1]))
    inp = np.full((grid_h, grid_w), int(bg), dtype=int)
    expected = np.full((grid_h, grid_w), int(bg), dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    rng.shuffle(objects)
    for obj in objects:
        mask = obj["mask"]
        patch_h, patch_w = mask.shape
        positions = _valid_positions(reserved, int(patch_h), int(patch_w))
        if not positions:
            return None
        top, left = positions[int(rng.integers(0, len(positions)))]
        obj["top"] = int(top)
        obj["left"] = int(left)
        _paint_mask(inp, int(top), int(left), mask, int(obj["color"]))
        _paint_mask(expected, int(top), int(left), _flip_from_orientation(mask, str(obj["orientation"])), int(obj["color"]))
        _reserve_bbox(reserved, int(top), int(left), int(patch_h), int(patch_w))

    if not _bg_is_dominant(inp, bg):
        return None
    if len(_components(inp, bg)) != n_objects:
        return None
    if np.array_equal(inp, expected):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
