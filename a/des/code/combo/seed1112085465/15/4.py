from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_ATTEMPTS = 400
GRID_MAX_SIZE = 30


def _connected_components(grid: np.ndarray) -> list[dict]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict] = []

    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue

            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    return components


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in component["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _is_convex(mask: np.ndarray) -> bool:
    for row in mask:
        cols = np.flatnonzero(row)
        if cols.size == 0:
            return False
        if not row[cols[0] : cols[-1] + 1].all():
            return False
    return True


def _centered_interval(start: int, length: int, side: int) -> tuple[int, int]:
    if side <= length:
        offset = (length - side) // 2
        return start + offset, start + offset + side

    offset = (side - length) // 2
    return start - offset, start - offset + side


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    height, width = g.shape
    out = g.copy()

    for component in _connected_components(g):
        color = int(component["color"])
        r0, c0, r1, c1 = component["bbox"]
        bbox_h = r1 - r0 + 1
        bbox_w = c1 - c0 + 1
        mask = _component_mask(component)

        if _is_convex(mask):
            side = min(bbox_h, bbox_w)
            sq_r0, sq_r1 = _centered_interval(r0, bbox_h, side)
            sq_c0, sq_c1 = _centered_interval(c0, bbox_w, side)
            for r, c in component["coords"]:
                if not (sq_r0 <= r < sq_r1 and sq_c0 <= c < sq_c1):
                    out[r, c] = BG
            continue

        side = max(bbox_h, bbox_w)
        sq_r0, sq_r1 = _centered_interval(r0, bbox_h, side)
        sq_c0, sq_c1 = _centered_interval(c0, bbox_w, side)

        rr0 = max(0, sq_r0)
        rr1 = min(height, sq_r1)
        cc0 = max(0, sq_c0)
        cc1 = min(width, sq_c1)

        top_r1 = min(r0, rr1)
        if rr0 < top_r1 and cc0 < cc1:
            out[rr0:top_r1, cc0:cc1] = color

        bottom_r0 = max(r1 + 1, rr0)
        if bottom_r0 < rr1 and cc0 < cc1:
            out[bottom_r0:rr1, cc0:cc1] = color

        mid_r0 = max(r0, rr0)
        mid_r1 = min(r1 + 1, rr1)
        left_c1 = min(c0, cc1)
        if mid_r0 < mid_r1 and cc0 < left_c1:
            out[mid_r0:mid_r1, cc0:left_c1] = color

        right_c0 = max(c1 + 1, cc0)
        if mid_r0 < mid_r1 and right_c0 < cc1:
            out[mid_r0:mid_r1, right_c0:cc1] = color

    return out


def _choose_dimensions(rng: random.Random, concave: bool) -> tuple[int, int, int]:
    min_side = 5 if concave else 4
    min_short = 3 if concave else 2

    for _ in range(50):
        side = rng.randint(min_side, 8)
        options = [short for short in range(min_short, side) if short % 2 == side % 2]
        if not options:
            continue
        short = rng.choice(options)
        if rng.random() < 0.5:
            return side, side, short
        return side, short, side

    raise RuntimeError("failed to sample dimensions")


def _make_convex_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    if w < 3 or rng.random() < 0.5:
        return np.ones((h, w), dtype=bool)

    mask = np.zeros((h, w), dtype=bool)
    anchor_left = rng.random() < 0.5
    max_trim = max(1, w - 2)
    trims = [rng.randint(0, max_trim) for _ in range(h)]
    if all(trim > 0 for trim in trims):
        trims[rng.randrange(h)] = 0

    for r, trim in enumerate(trims):
        if anchor_left:
            mask[r, : w - trim] = True
        else:
            mask[r, trim:] = True

    return mask


def _make_concave_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = True
    mask[:, -1] = True

    variant = rng.choice(("frame", "open_top", "open_bottom"))
    if variant in ("frame", "open_bottom"):
        mask[0, :] = True
    if variant in ("frame", "open_top"):
        mask[-1, :] = True

    return mask


def _build_object(kind: str, color: int, rng: random.Random) -> dict:
    concave = kind == "concave"
    side, h, w = _choose_dimensions(rng, concave=concave)
    bbox_r0 = (side - h) // 2
    bbox_c0 = (side - w) // 2

    if concave:
        mask = _make_concave_mask(h, w, rng)
    else:
        mask = _make_convex_mask(h, w, rng)

    patch = np.zeros((side, side), dtype=int)
    window = patch[bbox_r0 : bbox_r0 + h, bbox_c0 : bbox_c0 + w]
    window[mask] = color

    return {
        "kind": kind,
        "color": color,
        "side": side,
        "patch": patch,
    }


def _random_partition(items: list[dict], groups: int, rng: random.Random) -> list[list[dict]]:
    if groups == 1:
        return [items[:]]

    cuts = sorted(rng.sample(range(1, len(items)), groups - 1))
    parts: list[list[dict]] = []
    start = 0
    for end in cuts + [len(items)]:
        parts.append(items[start:end])
        start = end
    return parts


def _place_objects(objects: list[dict], rng: random.Random) -> np.ndarray | None:
    shuffled = objects[:]
    rng.shuffle(shuffled)
    shelf_count = rng.randint(1, min(3, len(shuffled)))
    shelves = _random_partition(shuffled, shelf_count, rng)

    shelf_layouts = []
    shelf_widths: list[int] = []
    shelf_heights: list[int] = []

    for shelf in shelves:
        rng.shuffle(shelf)
        gaps = [rng.randint(1, 3) for _ in range(len(shelf) - 1)]
        lead = rng.randint(0, 2)
        trail = rng.randint(0, 2)
        shelf_height = max(obj["side"] for obj in shelf) + rng.randint(0, 2)
        shelf_width = lead + trail + sum(obj["side"] for obj in shelf) + sum(gaps)
        shelf_layouts.append((shelf, gaps, lead, shelf_height, shelf_width))
        shelf_widths.append(shelf_width)
        shelf_heights.append(shelf_height)

    top = rng.randint(0, 2)
    bottom = rng.randint(0, 2)
    row_gaps = [rng.randint(1, 3) for _ in range(len(shelves) - 1)]
    grid_h = top + bottom + sum(shelf_heights) + sum(row_gaps)
    grid_w = max(shelf_widths) + rng.randint(0, 3)

    if grid_h > GRID_MAX_SIZE or grid_w > GRID_MAX_SIZE:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    y = top

    for idx, (shelf, gaps, lead, shelf_height, shelf_width) in enumerate(shelf_layouts):
        shelf_x0 = rng.randint(0, grid_w - shelf_width)
        x = shelf_x0 + lead
        for obj_index, obj in enumerate(shelf):
            side = int(obj["side"])
            y_offset = rng.randint(0, shelf_height - side)
            patch = obj["patch"]
            target = grid[y + y_offset : y + y_offset + side, x : x + side]
            assert not np.any((target != BG) & (patch != BG))
            target[patch != BG] = patch[patch != BG]
            x += side
            if obj_index < len(gaps):
                x += gaps[obj_index]
        y += shelf_height
        if idx < len(row_gaps):
            y += row_gaps[idx]

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    object_count = rng.randint(2, 5)
    convex_count = rng.randint(1, object_count - 1)
    kinds = ["convex"] * convex_count + ["concave"] * (object_count - convex_count)
    rng.shuffle(kinds)

    colors = rng.sample(list(range(1, 10)), object_count)
    objects = [_build_object(kind, color, rng) for kind, color in zip(kinds, colors)]
    grid = _place_objects(objects, rng)
    if grid is None:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
