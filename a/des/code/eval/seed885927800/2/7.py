from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
AZURE = 8
BG_VALUES = (BLACK, AZURE)
OBJECT_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
SIDE_ORDER = (
    ("top", (-1, 0)),
    ("right", (0, 1)),
    ("bottom", (1, 0)),
    ("left", (0, -1)),
)
GEN_RETRIES = 400


def _checkerboard(height: int, width: int, phase: int) -> np.ndarray:
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    parity = (rows + cols + phase) % 2
    return np.where(parity == 0, BLACK, AZURE).astype(int)


def _neighbors4(r: int, c: int, height: int, width: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < height:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < width:
        out.append((r, c + 1))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, height, width):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_objects(grid: np.ndarray) -> list[dict[str, object]]:
    mask = ~np.isin(grid, BG_VALUES)
    objs: list[dict[str, object]] = []
    for coords in _components(mask):
        colors = {int(grid[r, c]) for r, c in coords}
        if len(colors) != 1:
            continue
        bbox = _bbox(coords)
        objs.append(
            {
                "color": next(iter(colors)),
                "coords": coords,
                "bbox": bbox,
            }
        )
    objs.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["color"]))  # type: ignore[index]
    return objs


def _count_holes(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    window = grid[r0 : r1 + 1, c0 : c1 + 1]
    open_mask = np.isin(window, BG_VALUES)
    height, width = open_mask.shape
    seen = np.zeros_like(open_mask, dtype=bool)
    holes = 0
    for r in range(height):
        for c in range(width):
            if not open_mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_border = False
            while q:
                cr, cc = q.popleft()
                if cr == 0 or cr == height - 1 or cc == 0 or cc == width - 1:
                    touches_border = True
                for nr, nc in _neighbors4(cr, cc, height, width):
                    if open_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _side_anchor(
    coords: list[tuple[int, int]], bbox: tuple[int, int, int, int], side: str
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        target = (c0 + c1) // 2
        edge = [(r, c) for r, c in coords if r == r0]
        return min(edge, key=lambda p: (abs(p[1] - target), p[1]))
    if side == "right":
        target = (r0 + r1) // 2
        edge = [(r, c) for r, c in coords if c == c1]
        return min(edge, key=lambda p: (abs(p[0] - target), p[0]))
    if side == "bottom":
        target = (c0 + c1) // 2
        edge = [(r, c) for r, c in coords if r == r1]
        return min(edge, key=lambda p: (abs(p[1] - target), p[1]))
    target = (r0 + r1) // 2
    edge = [(r, c) for r, c in coords if c == c0]
    return min(edge, key=lambda p: (abs(p[0] - target), p[0]))


def _draw_ray(out: np.ndarray, anchor: tuple[int, int], delta: tuple[int, int], color: int) -> None:
    height, width = out.shape
    dr, dc = delta
    r, c = anchor[0] + dr, anchor[1] + dc
    while 0 <= r < height and 0 <= c < width and int(out[r, c]) in BG_VALUES:
        out[r, c] = color
        r += dr
        c += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each object's enclosed checkerboard cavities and emit same-colored rays
    from the first N bounding-box sides in clockwise order.
    """
    g = np.asarray(grid, dtype=int)
    out = np.array(g, copy=True)
    for obj in _find_objects(g):
        holes = min(_count_holes(g, obj["bbox"]), 4)  # type: ignore[index]
        if holes == 0:
            continue
        color = int(obj["color"])  # type: ignore[index]
        coords = obj["coords"]  # type: ignore[assignment]
        bbox = obj["bbox"]  # type: ignore[assignment]
        for side, delta in SIDE_ORDER[:holes]:
            anchor = _side_anchor(coords, bbox, side)
            _draw_ray(out, anchor, delta, color)
    return out


def _odd_choice(rng: random.Random, low: int, high: int) -> int:
    vals = [v for v in range(low, high + 1) if v % 2 == 1]
    return rng.choice(vals)


def _frame_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _open_frame_mask(rng: random.Random) -> np.ndarray:
    height = _odd_choice(rng, 5, 9)
    width = _odd_choice(rng, 5, 9)
    mask = _frame_mask(height, width)
    side = rng.choice(("top", "right", "bottom", "left"))
    if side in ("top", "bottom"):
        span = width
        gap = 1 if span <= 5 else rng.choice((1, 1, 2))
        start = rng.randrange(1, span - gap)
        row = 0 if side == "top" else height - 1
        mask[row, start : start + gap] = False
    else:
        span = height
        gap = 1 if span <= 5 else rng.choice((1, 1, 2))
        start = rng.randrange(1, span - gap)
        col = 0 if side == "left" else width - 1
        mask[start : start + gap, col] = False
    return np.rot90(mask, rng.randrange(4)).copy()


def _solid_mask(rng: random.Random) -> np.ndarray:
    height = _odd_choice(rng, 5, 9)
    width = _odd_choice(rng, 5, 9)
    return np.ones((height, width), dtype=bool)


def _one_hole_mask(rng: random.Random) -> np.ndarray:
    height = _odd_choice(rng, 5, 9)
    width = _odd_choice(rng, 5, 9)
    return np.rot90(_frame_mask(height, width), rng.randrange(4)).copy()


def _two_hole_mask(rng: random.Random) -> np.ndarray:
    vertical = rng.choice((True, False))
    if vertical:
        height = _odd_choice(rng, 5, 9)
        width = _odd_choice(rng, 7, 9)
        mask = _frame_mask(height, width)
        cut_col = rng.randrange(2, width - 2)
        mask[:, cut_col] = True
    else:
        height = _odd_choice(rng, 7, 9)
        width = _odd_choice(rng, 5, 9)
        mask = _frame_mask(height, width)
        cut_row = rng.randrange(2, height - 2)
        mask[cut_row, :] = True
    return np.rot90(mask, rng.randrange(4)).copy()


def _three_hole_mask(rng: random.Random) -> np.ndarray:
    height = _odd_choice(rng, 7, 9)
    width = _odd_choice(rng, 7, 9)
    mask = _frame_mask(height, width)
    if rng.choice((True, False)):
        cut_col = rng.randrange(2, width - 2)
        cut_row = rng.randrange(2, height - 2)
        mask[:, cut_col] = True
        if rng.choice((True, False)):
            mask[cut_row, : cut_col + 1] = True
        else:
            mask[cut_row, cut_col:] = True
    else:
        cut_row = rng.randrange(2, height - 2)
        cut_col = rng.randrange(2, width - 2)
        mask[cut_row, :] = True
        if rng.choice((True, False)):
            mask[: cut_row + 1, cut_col] = True
        else:
            mask[cut_row:, cut_col] = True
    return np.rot90(mask, rng.randrange(4)).copy()


def _four_hole_mask(rng: random.Random) -> np.ndarray:
    height = _odd_choice(rng, 7, 9)
    width = _odd_choice(rng, 7, 9)
    mask = _frame_mask(height, width)
    cut_row = rng.randrange(2, height - 2)
    cut_col = rng.randrange(2, width - 2)
    mask[cut_row, :] = True
    mask[:, cut_col] = True
    return np.rot90(mask, rng.randrange(4)).copy()


def _build_mask(rng: random.Random, holes: int) -> np.ndarray:
    if holes == 0:
        if rng.random() < 0.65:
            mask = _open_frame_mask(rng)
        else:
            mask = _solid_mask(rng)
    elif holes == 1:
        mask = _one_hole_mask(rng)
    elif holes == 2:
        mask = _two_hole_mask(rng)
    elif holes == 3:
        mask = _three_hole_mask(rng)
    else:
        mask = _four_hole_mask(rng)

    probe = _checkerboard(mask.shape[0], mask.shape[1], 0)
    probe[mask] = 1
    if _count_holes(probe, (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)) != holes:
        raise RuntimeError(f"mask hole mismatch for {holes}")
    return mask


def _local_side_anchor(mask: np.ndarray, side: str) -> tuple[int, int]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return _side_anchor(coords, (0, 0, mask.shape[0] - 1, mask.shape[1] - 1), side)


def _ray_cells_for_placement(
    top: int, left: int, mask: np.ndarray, holes: int, height: int, width: int
) -> list[tuple[int, int]] | None:
    cells: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for side, (dr, dc) in SIDE_ORDER[:holes]:
        ar, ac = _local_side_anchor(mask, side)
        r, c = top + ar + dr, left + ac + dc
        length = 0
        while 0 <= r < height and 0 <= c < width:
            if (r, c) not in used:
                cells.append((r, c))
                used.add((r, c))
            length += 1
            r += dr
            c += dc
        if length == 0:
            return None
    return cells


def _expanded_bbox(
    top: int, left: int, obj_h: int, obj_w: int, height: int, width: int, margin: int = 1
) -> tuple[int, int, int, int]:
    return (
        max(0, top - margin),
        max(0, left - margin),
        min(height - 1, top + obj_h - 1 + margin),
        min(width - 1, left + obj_w - 1 + margin),
    )


def _paint_object(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    rows, cols = np.where(mask)
    grid[top + rows, left + cols] = color


def _validate_grid(grid: np.ndarray, expected_holes: list[int]) -> bool:
    objs = _find_objects(grid)
    if len(objs) != len(expected_holes):
        return False
    actual = sorted(_count_holes(grid, obj["bbox"]) for obj in objs)  # type: ignore[index]
    if actual != sorted(expected_holes):
        return False
    out = main(grid)
    return not np.array_equal(out, grid)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    obj_count = rng.choice((2, 3, 4))
    hole_counts = rng.sample((0, 1, 2, 3, 4), obj_count)
    colors = rng.sample(OBJECT_COLORS, obj_count)

    specs = []
    for holes, color in zip(hole_counts, colors):
        mask = _build_mask(rng, holes)
        specs.append(
            {
                "holes": holes,
                "color": color,
                "mask": mask,
                "area": int(mask.sum()),
            }
        )
    specs.sort(key=lambda spec: (-spec["holes"], -spec["area"]))

    height = rng.randint(20, 30)
    width = rng.randint(20, 30)
    grid = _checkerboard(height, width, rng.randrange(2))
    blocked = np.zeros((height, width), dtype=bool)

    placed_holes: list[int] = []
    for spec in specs:
        mask = spec["mask"]
        obj_h, obj_w = mask.shape
        placed = False
        for _ in range(250):
            top = rng.randint(1, height - obj_h - 1)
            left = rng.randint(1, width - obj_w - 1)
            r0, c0, r1, c1 = _expanded_bbox(top, left, obj_h, obj_w, height, width)
            if blocked[r0 : r1 + 1, c0 : c1 + 1].any():
                continue

            ray_cells = _ray_cells_for_placement(top, left, mask, spec["holes"], height, width)
            if ray_cells is None:
                continue
            if any(blocked[r, c] for r, c in ray_cells):
                continue

            _paint_object(grid, top, left, mask, spec["color"])
            blocked[r0 : r1 + 1, c0 : c1 + 1] = True
            for r, c in ray_cells:
                blocked[r, c] = True
            placed_holes.append(spec["holes"])
            placed = True
            break
        if not placed:
            return None

    if not _validate_grid(grid, placed_holes):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
