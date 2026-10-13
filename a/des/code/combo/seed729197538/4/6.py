from __future__ import annotations

import math
import random

import numpy as np


_BG = 0
_PALETTE = tuple(range(1, 10))
_MAX_GENERATION_ATTEMPTS = 400
_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_RNG = random.Random()


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _rect_frame_bbox(grid: np.ndarray, coords: np.ndarray, color: int) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    perimeter = 2 * (h + w) - 4
    if coords.shape[0] != perimeter:
        return None
    sub = grid[r0 : r1 + 1, c0 : c1 + 1] == color
    want = np.zeros((h, w), dtype=bool)
    want[0, :] = True
    want[-1, :] = True
    want[:, 0] = True
    want[:, -1] = True
    if not np.array_equal(sub, want):
        return None
    return r0, c0, r1, c1


def _extract_containers(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]

    containers: list[dict[str, object]] = []
    for color in colors:
        for coords in _components(g == color):
            frame_bbox = _rect_frame_bbox(g, coords, color)
            if frame_bbox is None:
                continue
            r0, c0, r1, c1 = frame_bbox
            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            if interior.size == 0:
                continue
            inner_colors = [int(c) for c in np.unique(interior) if int(c) != bg]
            if len(inner_colors) != 1:
                continue
            inner_color = inner_colors[0]
            if inner_color == color:
                continue
            mask = interior == inner_color
            if not np.any(mask):
                continue
            containers.append(
                {
                    "bbox": frame_bbox,
                    "border_color": int(color),
                    "inner_color": int(inner_color),
                    "mask": mask.copy(),
                }
            )

    assert len(containers) >= 2, "expected at least two containers"

    border_colors = [int(c["border_color"]) for c in containers]
    inner_colors = [int(c["inner_color"]) for c in containers]
    assert len(set(border_colors)) == len(containers), "frame colors must be unique"
    assert len(set(inner_colors)) == len(containers), "pattern colors must be unique"
    assert set(border_colors) == set(inner_colors), "pattern colors must permute the frame colors"

    return bg, containers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect colored frame containers, follow the chain where a container's inner
    color points to the next container's border color, and rotate only shapes.
    """
    g = np.asarray(grid, dtype=int)
    bg, containers = _extract_containers(g)
    out = g.copy()

    border_to_idx = {int(item["border_color"]): idx for idx, item in enumerate(containers)}
    next_idx = [border_to_idx[int(item["inner_color"])] for item in containers]
    prev_idx = [-1] * len(containers)
    for src, dst in enumerate(next_idx):
        assert prev_idx[dst] == -1, "each container must have a unique predecessor"
        prev_idx[dst] = src

    for item in containers:
        r0, c0, r1, c1 = item["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = bg

    for dst, src in enumerate(prev_idx):
        assert src >= 0, "each container must belong to a closed cycle"
        src_mask = np.asarray(containers[src]["mask"], dtype=bool)
        dst_mask = np.asarray(containers[dst]["mask"], dtype=bool)
        assert src_mask.shape == dst_mask.shape, "linked containers must share the same interior size"
        r0, c0, r1, c1 = containers[dst]["bbox"]
        fill_color = int(containers[dst]["inner_color"])
        patch = out[r0 + 1 : r1, c0 + 1 : c1]
        patch[src_mask] = fill_color

    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(coords[0])}
    stack = [tuple(coords[0])]
    cells = {tuple(x) for x in coords.tolist()}
    while stack:
        r, c = stack.pop()
        for dr, dc in _NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _looks_like_hollow_frame(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    want = np.zeros((h, w), dtype=bool)
    want[0, :] = True
    want[-1, :] = True
    want[:, 0] = True
    want[:, -1] = True
    return np.array_equal(sub, want)


def _grow_connected_mask(h: int, w: int, area: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (rng.randrange(h), rng.randrange(w))
    mask[start] = True
    cells = {start}
    frontier = set()
    for dr, dc in _NEIGHBORS4:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))
    while len(cells) < area and frontier:
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        mask[cell] = True
        for dr, dc in _NEIGHBORS4:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))
    return mask


def _random_polyline_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    steps = rng.randint(2, max(2, min(h * w - 1, h + w + 1)))
    r = rng.randrange(h)
    c = rng.randrange(w)
    mask[r, c] = True
    for _ in range(steps - 1):
        choices = []
        for dr, dc in _NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                choices.append((nr, nc))
        r, c = rng.choice(choices)
        mask[r, c] = True
    return mask


def _random_block_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    bh = rng.randint(1, h)
    bw = rng.randint(1, w)
    if bh == h and bw == w:
        if bh > 1:
            bh -= 1
        elif bw > 1:
            bw -= 1
    r0 = rng.randint(0, h - bh)
    c0 = rng.randint(0, w - bw)
    mask[r0 : r0 + bh, c0 : c0 + bw] = True
    if bh >= 2 and bw >= 2 and rng.random() < 0.6:
        corner = rng.choice(
            (
                (r0, c0),
                (r0, c0 + bw - 1),
                (r0 + bh - 1, c0),
                (r0 + bh - 1, c0 + bw - 1),
            )
        )
        mask[corner] = False
        if not np.any(mask):
            mask[corner] = True
    return mask


def _random_pattern(h: int, w: int, rng: random.Random) -> np.ndarray:
    max_area = h * w - 1
    for _ in range(200):
        mode = rng.choice(("grow", "line", "block"))
        if mode == "grow":
            area = rng.randint(2, max(2, max_area))
            mask = _grow_connected_mask(h, w, area, rng)
        elif mode == "line":
            mask = _random_polyline_mask(h, w, rng)
        else:
            mask = _random_block_mask(h, w, rng)

        area = int(mask.sum())
        if area < 2 or area >= h * w:
            continue
        if not _is_connected(mask):
            continue
        if _looks_like_hollow_frame(mask):
            continue
        return mask
    raise RuntimeError("failed to sample a valid pattern")


def _partition_total(total: int, parts: int, rng: random.Random) -> list[int]:
    remaining = total
    out: list[int] = []
    for idx in range(parts - 1):
        max_piece = remaining - 2 * (parts - idx - 1)
        piece = rng.randint(2, max_piece)
        out.append(piece)
        remaining -= piece
    out.append(remaining)
    rng.shuffle(out)
    return out


def _choose_cycle_lengths(rng: random.Random) -> list[int]:
    cycle_count = rng.choice((1, 2, 3))
    if cycle_count == 1:
        total = rng.randint(2, 6)
    elif cycle_count == 2:
        total = rng.randint(4, 6)
    else:
        total = 6
    return _partition_total(total, cycle_count, rng)


def _sample_cycle_shapes(length: int, h: int, w: int, rng: random.Random) -> list[np.ndarray]:
    seen: set[bytes] = set()
    masks: list[np.ndarray] = []
    for _ in range(length):
        for _ in range(200):
            mask = _random_pattern(h, w, rng)
            key = mask.tobytes()
            if key in seen:
                continue
            seen.add(key)
            masks.append(mask)
            break
        else:
            raise RuntimeError("failed to sample distinct shapes for a cycle")
    return masks


def _place_rectangles(
    dims: list[tuple[int, int]],
    grid_h: int,
    grid_w: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(dims)
    order = sorted(range(len(dims)), key=lambda idx: dims[idx][0] * dims[idx][1], reverse=True)
    for idx in order:
        h, w = dims[idx]
        placed = False
        for _ in range(500):
            if grid_h - h - 1 <= 0 or grid_w - w - 1 <= 0:
                return None
            r = rng.randint(1, grid_h - h - 1)
            c = rng.randint(1, grid_w - w - 1)
            if blocked[r - 1 : r + h + 1, c - 1 : c + w + 1].any():
                continue
            positions[idx] = (r, c)
            blocked[r - 1 : r + h + 1, c - 1 : c + w + 1] = True
            placed = True
            break
        if not placed:
            return None
    return [pos for pos in positions if pos is not None]


def _grid_shape_for(dims: list[tuple[int, int]], rng: random.Random) -> tuple[int, int] | None:
    padded_area = sum((h + 2) * (w + 2) for h, w in dims)
    side = int(math.ceil(math.sqrt(padded_area)))
    max_h = max(h for h, _ in dims)
    max_w = max(w for _, w in dims)
    h_lo = max(14, max_h + 2, side - 1)
    w_lo = max(14, max_w + 2, side - 1)
    h_hi = min(30, h_lo + 8)
    w_hi = min(30, w_lo + 8)
    if h_lo > h_hi or w_lo > w_hi:
        return None
    return rng.randint(h_lo, h_hi), rng.randint(w_lo, w_hi)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    cycle_lengths = _choose_cycle_lengths(rng)
    total = sum(cycle_lengths)
    colors = rng.sample(_PALETTE, total)

    containers: list[dict[str, object]] = []
    color_offset = 0
    for length in cycle_lengths:
        inner_h = rng.randint(3, 6)
        inner_w = rng.randint(3, 6)
        cycle_colors = colors[color_offset : color_offset + length]
        color_offset += length
        masks = _sample_cycle_shapes(length, inner_h, inner_w, rng)
        for idx, border_color in enumerate(cycle_colors):
            containers.append(
                {
                    "border_color": int(border_color),
                    "inner_color": int(cycle_colors[(idx + 1) % length]),
                    "mask": masks[idx],
                    "outer_h": inner_h + 2,
                    "outer_w": inner_w + 2,
                }
            )

    dims = [(int(item["outer_h"]), int(item["outer_w"])) for item in containers]
    for _ in range(30):
        shape = _grid_shape_for(dims, rng)
        if shape is None:
            return None
        grid_h, grid_w = shape
        positions = _place_rectangles(dims, grid_h, grid_w, rng)
        if positions is None:
            continue

        grid = np.full((grid_h, grid_w), _BG, dtype=int)
        for item, (r0, c0) in zip(containers, positions):
            outer_h = int(item["outer_h"])
            outer_w = int(item["outer_w"])
            border_color = int(item["border_color"])
            inner_color = int(item["inner_color"])
            mask = np.asarray(item["mask"], dtype=bool)

            grid[r0, c0 : c0 + outer_w] = border_color
            grid[r0 + outer_h - 1, c0 : c0 + outer_w] = border_color
            grid[r0 : r0 + outer_h, c0] = border_color
            grid[r0 : r0 + outer_h, c0 + outer_w - 1] = border_color
            patch = grid[r0 + 1 : r0 + outer_h - 1, c0 + 1 : c0 + outer_w - 1]
            patch[mask] = inner_color

        if np.count_nonzero(grid == _BG) <= max(np.count_nonzero(grid == c) for c in _PALETTE if np.any(grid == c)):
            continue

        try:
            out = main(grid)
        except AssertionError:
            continue
        if not np.any(out != grid):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
