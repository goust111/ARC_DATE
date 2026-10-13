import random

import numpy as np


BG = 0
GEN_RETRIES = 400
OBJECT_RETRIES = 200


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _tight_mask(points: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    mask[points[:, 0] - r0, points[:, 1] - c0] = True
    return (int(r0), int(c0), int(r1), int(c1)), mask


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches_border = r in (0, h - 1) or c in (0, w - 1)
            while stack:
                cr, cc = stack.pop()
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        if nr in (0, h - 1) or nc in (0, w - 1):
                            touches_border = True
                        stack.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _find_header(grid: np.ndarray) -> tuple[int, list[int], int]:
    h, w = grid.shape
    best = None
    for sep_row in range(1, h):
        row = grid[sep_row]
        uniq = np.unique(row)
        if len(uniq) != 1 or int(uniq[0]) == BG:
            continue
        legend_row = grid[sep_row - 1]
        cols = [int(c) for c in np.flatnonzero(legend_row != BG)]
        if len(cols) < 2:
            continue
        if cols != list(range(cols[0], cols[-1] + 1)):
            continue
        colors = [int(legend_row[c]) for c in cols]
        if len(colors) != len(set(colors)):
            continue
        score = (len(colors), -sep_row, -abs(cols[0] - (w - cols[-1] - 1)))
        cand = (score, int(sep_row), colors, int(uniq[0]))
        if best is None or cand[0] > best[0]:
            best = cand
    assert best is not None, "expected a top legend row and a uniform separator row"
    return best[1], best[2], best[3]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each monochrome object as a bbox frame whose color is selected by its hole count from the top legend.

    Input structure:
    - Black is the background.
    - Near the top sits a one-row horizontal legend made of distinct colors, followed immediately by a full-width grey separator row.
    - Below the separator, the main area contains several 4-connected objects, all drawn in the same foreground color, and each object may enclose zero or more black holes.

    Rule:
    - Read the legend colors from left to right.
    - For every foreground object in the main area, count how many enclosed black holes it contains.
    - Use the (hole_count + 1)-th legend color as that object's output color, clear the object's bbox to black, and draw the full rectangular bbox border in the chosen color.
    - Keep the legend row and the grey separator row unchanged.

    Key insight:
    - The legend is not a direct color key for object identity; it is indexed by each object's number of enclosed holes, and the object is replaced by its bbox frame rather than recolored in place.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    sep_row, legend_colors, sep_color = _find_header(g)
    main_area = g[sep_row + 1 :]

    excluded = {BG, sep_color, *legend_colors}
    vals, cnts = np.unique(main_area, return_counts=True)
    obj_candidates = [(int(n), int(v)) for v, n in zip(vals, cnts, strict=False) if int(v) not in excluded]
    assert obj_candidates, "expected at least one foreground object color below the separator"
    object_color = max(obj_candidates)[1]

    for comp in _components(main_area == object_color):
        (r0, c0, r1, c1), obj_mask = _tight_mask(comp)
        hole_count = _count_holes(obj_mask)
        assert hole_count < len(legend_colors), "hole count must map inside the legend"
        color = int(legend_colors[hole_count])

        gr0 = sep_row + 1 + r0
        gc0 = c0
        gr1 = sep_row + 1 + r1
        gc1 = c1

        out[gr0 : gr1 + 1, gc0 : gc1 + 1] = BG
        out[gr0, gc0 : gc1 + 1] = color
        out[gr1, gc0 : gc1 + 1] = color
        out[gr0 : gr1 + 1, gc0] = color
        out[gr0 : gr1 + 1, gc1] = color

    return out


def _choose_span(length: int, rng: random.Random) -> tuple[int, int]:
    max_span = max(1, length - 1)
    span = rng.randint(1, max_span)
    if length - span >= 2:
        start = rng.randint(1, length - span - 1)
    else:
        start = rng.randint(0, length - span)
    return int(start), int(start + span)


def _make_object(hole_count: int, rng: random.Random) -> np.ndarray | None:
    pad = 2
    for _ in range(OBJECT_RETRIES):
        body_h = rng.randint(5, 8)
        body_w = rng.randint(5, 8)
        slots = [(r, c) for r in range(1, body_h - 1, 2) for c in range(1, body_w - 1, 2)]
        if len(slots) < hole_count:
            continue

        mask = np.zeros((body_h + 2 * pad, body_w + 2 * pad), dtype=bool)
        mask[pad : pad + body_h, pad : pad + body_w] = True
        for rr, cc in rng.sample(slots, hole_count):
            mask[pad + rr, pad + cc] = False

        top0, top1 = _choose_span(body_w, rng)
        bot0, bot1 = _choose_span(body_w, rng)
        left0, left1 = _choose_span(body_h, rng)
        right0, right1 = _choose_span(body_h, rng)

        mask[:pad, pad + top0 : pad + top1] = True
        mask[pad + body_h :, pad + bot0 : pad + bot1] = True
        mask[pad + left0 : pad + left1, :pad] = True
        mask[pad + right0 : pad + right1, pad + body_w :] = True

        protected = set()
        for c in range(top0, top1):
            protected.add((pad, pad + c))
        for c in range(bot0, bot1):
            protected.add((pad + body_h - 1, pad + c))
        for r in range(left0, left1):
            protected.add((pad + r, pad))
        for r in range(right0, right1):
            protected.add((pad + r, pad + body_w - 1))

        corners = [
            (pad, pad),
            (pad, pad + body_w - 1),
            (pad + body_h - 1, pad),
            (pad + body_h - 1, pad + body_w - 1),
        ]
        rng.shuffle(corners)
        for cell in corners[: rng.randint(0, 2)]:
            if cell not in protected:
                mask[cell] = False

        for _ in range(rng.randint(0, 4)):
            side = rng.choice(("top", "bottom", "left", "right"))
            if side in ("top", "bottom"):
                r = pad if side == "top" else pad + body_h - 1
                choices = [c for c in range(pad + 1, pad + body_w - 1) if (r, c) not in protected]
                if choices:
                    mask[r, rng.choice(choices)] = False
            else:
                c = pad if side == "left" else pad + body_w - 1
                choices = [r for r in range(pad + 1, pad + body_h - 1) if (r, c) not in protected]
                if choices:
                    mask[rng.choice(choices), c] = False

        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, rng.randrange(4))

        comps = _components(mask)
        if len(comps) != 1:
            continue
        if _count_holes(mask) != hole_count:
            continue
        return mask
    return None


def _sample_hole_counts(legend_len: int, object_count: int, rng: random.Random) -> list[int]:
    available = list(range(legend_len))
    rng.shuffle(available)
    counts = available[: min(legend_len, object_count)]
    while len(counts) < object_count:
        counts.append(rng.choice(available))
    rng.shuffle(counts)
    return counts


def _place_objects(main_h: int, main_w: int, objects: list[np.ndarray], rng: random.Random):
    occ = np.zeros((main_h, main_w), dtype=bool)
    order = list(range(len(objects)))
    rng.shuffle(order)
    placed: list[tuple[int, int, int]] = []
    for idx in order:
        obj = objects[idx]
        h, w = obj.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, main_h - h):
            for c0 in range(1, main_w - w):
                rr0 = max(0, r0 - 1)
                rr1 = min(main_h, r0 + h + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(main_w, c0 + w + 1)
                if occ[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = rng.choice(candidates)
        placed.append((idx, r0, c0))
        occ[max(0, r0 - 1) : min(main_h, r0 + h + 1), max(0, c0 - 1) : min(main_w, c0 + w + 1)] = True
    placed.sort()
    return placed


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    max_holes = rng.randint(2, 4)
    legend_len = max_holes + 1
    object_count = rng.randint(2, min(4, legend_len))
    hole_counts = _sample_hole_counts(legend_len, object_count, rng)

    objects = []
    for hole_count in hole_counts:
        obj = _make_object(hole_count, rng)
        if obj is None:
            return None
        objects.append(obj)

    sizes = [obj.shape for obj in objects]
    max_h = max(h for h, _ in sizes)
    max_w = max(w for _, w in sizes)
    spread_area = sum((h + 2) * (w + 2) for h, w in sizes)
    need_side = int(np.ceil(np.sqrt(spread_area))) + 1

    main_h_low = max(max_h + 2, min(28, need_side))
    main_w_low = max(max_w + 2, legend_len + 2, min(30, need_side))
    if main_h_low > 28 or main_w_low > 30:
        return None

    main_h = rng.randint(main_h_low, 28)
    main_w = rng.randint(main_w_low, 30)
    placements = _place_objects(main_h, main_w, objects, rng)
    if placements is None:
        return None

    palette_pool = list(range(1, 10))
    rng.shuffle(palette_pool)
    legend_colors = palette_pool[:legend_len]
    remaining = [c for c in range(1, 10) if c not in legend_colors]
    separator_color, object_color = rng.sample(remaining, 2)

    grid = np.zeros((main_h + 2, main_w), dtype=int)
    legend_start = rng.randint(0, main_w - legend_len)
    grid[0, legend_start : legend_start + legend_len] = legend_colors
    grid[1, :] = int(separator_color)

    for idx, r0, c0 in placements:
        obj = objects[idx]
        h, w = obj.shape
        patch = grid[2 + r0 : 2 + r0 + h, c0 : c0 + w]
        patch[obj] = int(object_color)

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
