from __future__ import annotations

import random

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_RETRIES = 300
OBJECT_RETRIES = 160
PLACEMENT_RETRIES = 240


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    assert connectivity in (4, 8)
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        dirs = DIRS4
    else:
        dirs = DIRS4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _touches_edge(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "cannot crop an empty mask"
    r0, c0 = np.min(coords, axis=0)
    r1, c1 = np.max(coords, axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (int(r0), int(c0), int(r1), int(c1))


def _hole_components_local(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    holes: list[list[tuple[int, int]]] = []
    for comp in _components_mask(~mask, connectivity=4):
        if not _touches_edge(comp, h, w):
            holes.append(comp)
    return holes


def _fill_holes_local(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    holes = _hole_components_local(mask)
    filled = mask.copy()
    for comp in holes:
        for r, c in comp:
            filled[r, c] = True
    return filled, holes


def _dilate4(mask: np.ndarray, steps: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(steps):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def _grow_on_background(shape: np.ndarray, bg_mask: np.ndarray, steps: int) -> np.ndarray:
    out = shape.copy()
    for _ in range(steps):
        nbr = np.zeros_like(out)
        nbr[1:, :] |= out[:-1, :]
        nbr[:-1, :] |= out[1:, :]
        nbr[:, 1:] |= out[:, :-1]
        nbr[:, :-1] |= out[:, 1:]
        out |= nbr & bg_mask
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each object's holes, then grow the object outward by as many 4-neighbor layers as holes it had.

    Input structure:
    - The dominant color is the background. The non-background pixels form several
      pairwise separated monochrome 4-connected objects.
    - Some objects contain one or more enclosed background holes: 4-connected
      background regions fully surrounded by that object and not connected to the
      grid edge.
    - Other objects have no holes.

    Rule:
    - Process each foreground object independently.
    - Count its enclosed holes, fill every hole with the object's own color, and
      then grow the filled shape outward by exactly that many 4-neighbor layers.
    - Objects with zero holes are unchanged because they have nothing to fill and
      no layers to grow.

    Key insight:
    - The hole count has two roles at once: the holes disappear, and that same
      count is the exact number of outward contour rings added around the object.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    bg_mask = g == bg
    out = g.copy()

    for comp in _components_mask(g != bg, connectivity=4):
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each foreground object must be monochrome"
        color = next(iter(colors))

        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)

        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        shape = np.zeros_like(g, dtype=bool)
        for r, c in comp:
            local[r - r0, c - c0] = True
            shape[r, c] = True

        _, holes = _fill_holes_local(local)
        for hole in holes:
            for lr, lc in hole:
                shape[r0 + lr, c0 + lc] = True

        grown = _grow_on_background(shape, bg_mask, len(holes))
        out[grown] = color

    return out


def _carve_holes(core: np.ndarray, hole_count: int, rng: random.Random) -> bool:
    if hole_count == 0:
        return True

    h, w = map(int, core.shape)
    reserved = np.zeros((h, w), dtype=bool)
    placed = 0
    attempts = 0
    while placed < hole_count and attempts < 240:
        attempts += 1
        max_hole_h = min(2, h - 2)
        max_hole_w = min(3, w - 2)
        if max_hole_h < 1 or max_hole_w < 1:
            return False
        hole_h = rng.randint(1, max_hole_h)
        hole_w = rng.randint(1, max_hole_w)
        top = rng.randint(1, h - hole_h - 1)
        left = rng.randint(1, w - hole_w - 1)

        if np.any(reserved[top - 1 : top + hole_h + 1, left - 1 : left + hole_w + 1]):
            continue

        core[top : top + hole_h, left : left + hole_w] = False
        reserved[top - 1 : top + hole_h + 1, left - 1 : left + hole_w + 1] = True
        placed += 1

    return placed == hole_count


def _add_protrusions(
    canvas: np.ndarray,
    core_top: int,
    core_left: int,
    core_h: int,
    core_w: int,
    rng: random.Random,
) -> None:
    for _ in range(rng.randint(0, 3)):
        side = rng.choice(("top", "bottom", "left", "right"))
        depth = rng.randint(1, 2)

        if side in ("top", "bottom"):
            max_span = max(1, core_w - 1)
            span = rng.randint(1, max_span)
            start = rng.randint(0, core_w - span)
            c0 = core_left + start
            c1 = c0 + span
            if side == "top":
                r0 = core_top - depth
                r1 = core_top
            else:
                r0 = core_top + core_h
                r1 = r0 + depth
        else:
            max_span = max(1, core_h - 1)
            span = rng.randint(1, max_span)
            start = rng.randint(0, core_h - span)
            r0 = core_top + start
            r1 = r0 + span
            if side == "left":
                c0 = core_left - depth
                c1 = core_left
            else:
                c0 = core_left + core_w
                c1 = c0 + depth

        canvas[r0:r1, c0:c1] = True


def _make_object_spec(hole_count: int, rng: random.Random) -> dict[str, object] | None:
    for _ in range(OBJECT_RETRIES):
        core_min = max(4, hole_count + 4)
        core_h = rng.randint(core_min, min(9, core_min + 2))
        core_w = rng.randint(core_min, min(10, core_min + 3))
        pad = hole_count + 3

        canvas = np.zeros((core_h + 2 * pad, core_w + 2 * pad), dtype=bool)
        core_top = pad
        core_left = pad
        canvas[core_top : core_top + core_h, core_left : core_left + core_w] = True
        _add_protrusions(canvas, core_top, core_left, core_h, core_w, rng)

        core = canvas[core_top : core_top + core_h, core_left : core_left + core_w]
        if not _carve_holes(core, hole_count, rng):
            continue

        if len(_components_mask(canvas, connectivity=4)) != 1:
            continue

        filled_canvas, holes = _fill_holes_local(canvas)
        if len(holes) != hole_count:
            continue

        output_canvas = _dilate4(filled_canvas, hole_count)
        output_mask, (r0, c0, r1, c1) = _crop_mask(output_canvas)
        input_mask = canvas[r0 : r1 + 1, c0 : c1 + 1].copy()

        if len(_components_mask(input_mask, connectivity=4)) != 1:
            continue

        return {
            "hole_count": hole_count,
            "input_mask": input_mask,
            "output_mask": output_mask,
            "output_area": int(np.count_nonzero(output_mask)),
        }
    return None


def _hole_counts_in_grid(grid: np.ndarray) -> list[int]:
    bg = _mode_color(grid)
    counts: list[int] = []
    for comp in _components_mask(grid != bg, connectivity=4):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            local[r - r0, c - c0] = True
        counts.append(len(_hole_components_local(local)))
    return sorted(counts)


def _choose_hole_counts(object_count: int, rng: random.Random) -> list[int]:
    positive_pool = [1, 2, 3]
    counts = [0, rng.choice(positive_pool)]
    while len(counts) < object_count:
        counts.append(rng.choice([0, 1, 2, 3]))
    rng.shuffle(counts)
    return counts


def _place_objects(specs: list[dict[str, object]], rng: random.Random) -> np.ndarray | None:
    max_h = max(int(spec["output_mask"].shape[0]) for spec in specs)  # type: ignore[index]
    max_w = max(int(spec["output_mask"].shape[1]) for spec in specs)  # type: ignore[index]
    total_area = sum(int(spec["output_area"]) for spec in specs)
    side_floor = max(max_h + 2, max_w + 2, int(np.ceil(np.sqrt(total_area * 2.1))) + 3)
    if side_floor > 30:
        return None

    h = rng.randint(side_floor, 30)
    w = rng.randint(side_floor, 30)
    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    colors = rng.sample(list(range(1, 10)), len(specs))

    order = sorted(range(len(specs)), key=lambda i: int(specs[i]["output_area"]), reverse=True)
    for color, idx in zip(colors, order):
        spec = specs[idx]
        input_mask = spec["input_mask"]  # type: ignore[assignment]
        output_mask = spec["output_mask"]  # type: ignore[assignment]
        oh, ow = map(int, output_mask.shape)
        ih, iw = map(int, input_mask.shape)
        assert (ih, iw) == (oh, ow)
        if h - oh < 2 or w - ow < 2:
            return None

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = rng.randint(1, h - oh - 1)
            left = rng.randint(1, w - ow - 1)

            candidate = np.zeros((h, w), dtype=bool)
            candidate[top : top + oh, left : left + ow] = output_mask
            reserve = _dilate4(candidate, 1)
            if np.any(reserved & reserve):
                continue

            rows, cols = np.where(input_mask)
            grid[top + rows, left + cols] = color
            reserved |= reserve
            placed = True
            break

        if not placed:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        object_count = rng.choices([2, 3, 4], weights=[4, 3, 1])[0]
        hole_counts = _choose_hole_counts(object_count, rng)

        specs: list[dict[str, object]] = []
        ok = True
        for hole_count in hole_counts:
            spec = _make_object_spec(hole_count, rng)
            if spec is None:
                ok = False
                break
            specs.append(spec)
        if not ok:
            continue

        grid = _place_objects(specs, rng)
        if grid is None:
            continue

        actual_counts = _hole_counts_in_grid(grid)
        if actual_counts != sorted(hole_counts):
            continue
        if not any(count == 0 for count in actual_counts):
            continue
        if not any(count > 0 for count in actual_counts):
            continue

        return grid

    raise RuntimeError("failed to generate a valid input")
