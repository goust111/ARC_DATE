from __future__ import annotations

from collections import deque

import numpy as np


BG = 8
PALETTE = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
MAX_GENERATION_ATTEMPTS = 400
GRID_H_RANGE = (13, 25)
GRID_W_RANGE = (13, 25)
EXTRA_OBJECTS_RANGE = (0, 5)
PADDING = 1

WIDE_SPECS = [(h, w) for h in range(2, 6) for w in range(h + 1, 12) if w % 2 == 1]
TALL_SPECS = [(h, w) for w in range(2, 6) for h in range(w + 1, 12) if h % 2 == 1]
SQUARE_SPECS = [(side, side) for side in (3, 5, 7)]


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            comps.append(
                {
                    "color": color,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "coords": coords,
                }
            )

    return comps


def _draw_compressed(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2

    if width > height:
        out[r0 : r1 + 1, mid_c] = int(color)
    elif height > width:
        out[mid_r, c0 : c1 + 1] = int(color)
    else:
        out[mid_r, mid_c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each solid rectangle according to its aspect ratio and keep it centered.

    Input structure:
    - Background is azure.
    - Foreground objects are separated 4-connected solid monochrome rectangles.

    Rule:
    - Clear the canvas to the background.
    - Wide rectangles collapse to a centered vertical column of the same height.
    - Tall rectangles collapse to a centered horizontal row of the same width.
    - Squares collapse to a centered single pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = BG if np.any(g == BG) else int(np.bincount(g.ravel()).argmax())
    out = np.full(g.shape, bg, dtype=int)

    for comp in _find_components(g, bg):
        r0, c0, r1, c1 = comp["bbox"]
        color = int(comp["color"])
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert bool(np.all(patch == color)), "expected solid rectangles"
        _draw_compressed(out, comp["bbox"], color)

    return out


def _sample_object_kinds(rng: np.random.Generator) -> list[str]:
    kinds = ["wide", "tall", "square"]
    extra = int(rng.integers(EXTRA_OBJECTS_RANGE[0], EXTRA_OBJECTS_RANGE[1]))
    if extra:
        kinds.extend(rng.choice(np.array(kinds), size=extra, replace=True).tolist())
    rng.shuffle(kinds)
    return kinds


def _sample_dims(rng: np.random.Generator, kind: str) -> tuple[int, int]:
    if kind == "wide":
        h, w = WIDE_SPECS[int(rng.integers(0, len(WIDE_SPECS)))]
    elif kind == "tall":
        h, w = TALL_SPECS[int(rng.integers(0, len(TALL_SPECS)))]
    else:
        h, w = SQUARE_SPECS[int(rng.integers(0, len(SQUARE_SPECS)))]
    return int(h), int(w)


def _valid_positions(occupied: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    grid_h, grid_w = map(int, occupied.shape)
    valid: list[tuple[int, int]] = []

    for r in range(grid_h - h + 1):
        for c in range(grid_w - w + 1):
            r0 = max(0, r - PADDING)
            r1 = min(grid_h, r + h + PADDING)
            c0 = max(0, c - PADDING)
            c1 = min(grid_w, c + w + PADDING)
            if not occupied[r0:r1, c0:c1].any():
                valid.append((int(r), int(c)))

    return valid


def _render_input(shape: tuple[int, int], objects: list[dict], bg: int) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for obj in objects:
        r = int(obj["row"])
        c = int(obj["col"])
        h = int(obj["height"])
        w = int(obj["width"])
        grid[r : r + h, c : c + w] = int(obj["color"])
    return grid


def _render_expected(shape: tuple[int, int], objects: list[dict], bg: int) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for obj in objects:
        r = int(obj["row"])
        c = int(obj["col"])
        h = int(obj["height"])
        w = int(obj["width"])
        _draw_compressed(out, (r, c, r + h - 1, c + w - 1), int(obj["color"]))
    return out


def _place_objects(
    rng: np.random.Generator, shape: tuple[int, int], specs: list[tuple[str, int, int]], colors: np.ndarray
) -> list[dict] | None:
    occupied = np.zeros(shape, dtype=bool)
    objects: list[dict] = []

    order = sorted(
        range(len(specs)),
        key=lambda idx: specs[idx][1] * specs[idx][2],
        reverse=True,
    )

    for idx in order:
        kind, h, w = specs[idx]
        valid = _valid_positions(occupied, h, w)
        if not valid:
            return None
        r, c = valid[int(rng.integers(0, len(valid)))]
        occupied[r : r + h, c : c + w] = True
        objects.append(
            {
                "kind": kind,
                "height": int(h),
                "width": int(w),
                "row": int(r),
                "col": int(c),
                "color": int(colors[idx]),
            }
        )

    return objects


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    kinds = _sample_object_kinds(rng)
    specs = [(kind, *_sample_dims(rng, kind)) for kind in kinds]
    total_area = sum(h * w for _, h, w in specs)

    min_h = max(GRID_H_RANGE[0], max(h for _, h, _ in specs) + 2)
    min_w = max(GRID_W_RANGE[0], max(w for _, _, w in specs) + 2)
    if min_h >= GRID_H_RANGE[1] or min_w >= GRID_W_RANGE[1]:
        return None

    grid_h = int(rng.integers(min_h, GRID_H_RANGE[1]))
    grid_w = int(rng.integers(min_w, GRID_W_RANGE[1]))
    if total_area * 3 >= grid_h * grid_w * 2:
        return None

    colors = rng.choice(PALETTE, size=len(specs), replace=False)
    objects = _place_objects(rng, (grid_h, grid_w), specs, colors)
    if objects is None:
        return None

    inp = _render_input((grid_h, grid_w), objects, BG)
    expected = _render_expected((grid_h, grid_w), objects, BG)
    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
