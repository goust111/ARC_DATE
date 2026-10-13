import random
from collections import Counter

import numpy as np


MAX_GEN_ATTEMPTS = 400
BG = 0
FOUR_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
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
                for dr, dc in FOUR_DIRS:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _count_holes(mask: np.ndarray) -> int:
    holes = 0
    for comp in _components_from_mask(~mask):
        if np.any(comp[:, 0] == 0) or np.any(comp[:, 0] == mask.shape[0] - 1):
            continue
        if np.any(comp[:, 1] == 0) or np.any(comp[:, 1] == mask.shape[1] - 1):
            continue
        holes += 1
    return holes


def _component_hole_count(coords: np.ndarray) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return _count_holes(mask)


def _component_anchor(coords: np.ndarray) -> tuple[int, int]:
    return min((int(r), int(c)) for r, c in coords)


def _extract_band_sequence(panel: np.ndarray, bg: int) -> list[int]:
    colors_by_col: list[int] = []
    for c in range(panel.shape[1]):
        vals = [int(v) for v in np.unique(panel[:, c]) if int(v) != int(bg)]
        if len(vals) > 1:
            return []
        colors_by_col.append(vals[0] if vals else int(bg))

    seq: list[int] = []
    prev = int(bg)
    for color in colors_by_col:
        if int(color) == int(bg):
            prev = int(bg)
            continue
        if int(color) != int(prev):
            seq.append(int(color))
        prev = int(color)
    return seq


def _body_components(grid: np.ndarray, usable: np.ndarray, bg: int) -> list[np.ndarray]:
    return _components_from_mask((grid != int(bg)) & usable)


def _candidate_layout(
    grid: np.ndarray, bg: int, panel_rows: list[int], divider_row: int
) -> dict | None:
    if not panel_rows:
        return None
    divider_vals = np.unique(grid[divider_row, :])
    if divider_vals.size != 1 or int(divider_vals[0]) == int(bg):
        return None

    seq = _extract_band_sequence(grid[panel_rows, :], bg)
    if len(seq) < 2:
        return None

    usable = np.ones(grid.shape, dtype=bool)
    usable[panel_rows, :] = False
    usable[divider_row, :] = False
    comps = _body_components(grid, usable, bg)
    if not comps:
        return None

    mono = True
    objects = []
    for coords in comps:
        vals = np.unique(grid[coords[:, 0], coords[:, 1]])
        if vals.size != 1:
            mono = False
        objects.append(
            {
                "coords": coords,
                "anchor": _component_anchor(coords),
                "hole_count": _component_hole_count(coords),
                "color": int(vals[0]),
            }
        )

    score = (
        int(len(objects) == len(seq) and mono),
        int(len(objects) == len(seq)),
        int(mono),
        len(seq),
        -len(panel_rows),
    )
    return {
        "score": score,
        "panel_rows": tuple(panel_rows),
        "divider_row": int(divider_row),
        "colors": seq,
        "objects": objects,
        "usable": usable,
    }


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    h = g.shape[0]

    candidates: list[dict] = []
    for divider_row in range(1, min(4, h - 1)):
        cand = _candidate_layout(g, bg, list(range(divider_row)), divider_row)
        if cand is not None:
            candidates.append(cand)
    for divider_row in range(max(1, h - 4), h - 1):
        cand = _candidate_layout(g, bg, list(range(divider_row + 1, h)), divider_row)
        if cand is not None:
            candidates.append(cand)

    assert candidates, "expected a detectable edge color band and divider"
    best = max(candidates, key=lambda item: item["score"])
    assert best["score"][1] == 1, "color band length must match object count"
    best["bg"] = int(bg)
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each foreground object according to its rank in hole-count order, using the edge color band as the target sequence.

    Input structure:
    - The most common color is the black-like background.
    - Away from the edge band, the foreground consists of several separated 4-connected single-color objects; each object may contain zero or more enclosed background holes.
    - Along the top or bottom edge, a small horizontal color-band panel is separated from the main scene by a full-width monochrome divider row.

    Rule:
    - Detect the divider row and read the color-band sequence from left to right, treating each contiguous non-background run as one color block.
    - Ignore the band panel, then find every remaining foreground object and count the number of enclosed background holes inside that object's own bounding box.
    - Sort the objects by increasing hole count; break ties by the object's topmost-then-leftmost pixel.
    - Recolor the first sorted object with the first band color, the second object with the second band color, and so on, without changing object shape or position.
    - Fill the entire color-band panel and divider row with background in the output; holes stay background.

    Key insight:
    - The relevant ordering key is a topological property, the number of enclosed holes, while the edge color band supplies only the replacement color order.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_grid(g)
    bg = int(info["bg"])

    out = g.copy()
    out[list(info["panel_rows"]), :] = int(bg)
    out[info["divider_row"], :] = int(bg)

    objects = sorted(
        info["objects"],
        key=lambda obj: (int(obj["hole_count"]), int(obj["anchor"][0]), int(obj["anchor"][1])),
    )
    for obj, new_color in zip(objects, info["colors"]):
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = int(new_color)
    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    r1 = int(pts[:, 0].max())
    c1 = int(pts[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _add_bumps(mask: np.ndarray, rng: random.Random, max_bumps: int) -> np.ndarray:
    pad = max_bumps + 2
    canvas = np.zeros((mask.shape[0] + 2 * pad, mask.shape[1] + 2 * pad), dtype=bool)
    canvas[pad : pad + mask.shape[0], pad : pad + mask.shape[1]] = mask
    for _ in range(rng.randint(1, max_bumps)):
        pts = np.argwhere(canvas)
        r0 = int(pts[:, 0].min())
        c0 = int(pts[:, 1].min())
        r1 = int(pts[:, 0].max())
        c1 = int(pts[:, 1].max())
        side = rng.choice(["top", "bottom", "left", "right"])
        if side in ("top", "bottom"):
            span = c1 - c0 + 1
            seg = rng.randint(1, min(2, span))
            start = rng.randint(c0, c1 - seg + 1)
            rr = r0 - 1 if side == "top" else r1 + 1
            canvas[rr, start : start + seg] = True
        else:
            span = r1 - r0 + 1
            seg = rng.randint(1, min(2, span))
            start = rng.randint(r0, r1 - seg + 1)
            cc = c0 - 1 if side == "left" else c1 + 1
            canvas[start : start + seg, cc] = True
    return _crop_mask(canvas)


def _make_solid_object(rng: random.Random) -> np.ndarray:
    h = rng.randint(3, 5)
    w = rng.randint(3, 6)
    mask = np.ones((h, w), dtype=bool)
    mask = _add_bumps(mask, rng, max_bumps=3)
    return mask


def _make_holey_object(hole_count: int, rng: random.Random) -> np.ndarray:
    for _ in range(100):
        orientation = rng.choice(["vertical", "horizontal"])
        if orientation == "vertical":
            hole_widths = [rng.randint(1, 2) for _ in range(hole_count)]
            inner_w = sum(hole_widths) + max(0, hole_count - 1)
            inner_h = rng.randint(2, 4)
            mask = np.ones((inner_h + 2, inner_w + 2), dtype=bool)
            mask[1:-1, 1:-1] = False
            col = 1 + hole_widths[0]
            for width in hole_widths[1:]:
                mask[1:-1, col] = True
                col += 1 + width
        else:
            hole_heights = [rng.randint(1, 2) for _ in range(hole_count)]
            inner_h = sum(hole_heights) + max(0, hole_count - 1)
            inner_w = rng.randint(2, 4)
            mask = np.ones((inner_h + 2, inner_w + 2), dtype=bool)
            mask[1:-1, 1:-1] = False
            row = 1 + hole_heights[0]
            for height in hole_heights[1:]:
                mask[row, 1:-1] = True
                row += 1 + height
        mask = _add_bumps(mask, rng, max_bumps=2)
        if _count_holes(mask) == hole_count and len(_components_from_mask(mask)) == 1:
            return mask
    raise RuntimeError(f"failed to generate object with {hole_count} holes")


def _make_object(hole_count: int, rng: random.Random) -> np.ndarray:
    if hole_count == 0:
        for _ in range(100):
            mask = _make_solid_object(rng)
            if _count_holes(mask) == 0 and len(_components_from_mask(mask)) == 1:
                return mask
        raise RuntimeError("failed to generate solid object")
    return _make_holey_object(hole_count, rng)


def _halo_points(mask: np.ndarray, top: int, left: int, h: int, w: int) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    cells = np.argwhere(mask)
    for r, c in cells:
        gr = int(top + r)
        gc = int(left + c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = gr + dr
                nc = gc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    pts.add((nr, nc))
    return pts


def _place_objects(body_h: int, body_w: int, masks: list[np.ndarray], rng: random.Random) -> list[tuple[int, int]] | None:
    blocked = np.zeros((body_h, body_w), dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True

    order = sorted(range(len(masks)), key=lambda i: -int(masks[i].sum()))
    placements: list[tuple[int, int] | None] = [None] * len(masks)

    for idx in order:
        mask = masks[idx]
        mh, mw = mask.shape
        if mh + 2 > body_h or mw + 2 > body_w:
            return None
        options = [(r, c) for r in range(1, body_h - mh) for c in range(1, body_w - mw)]
        rng.shuffle(options)
        placed = False
        for top, left in options[:500]:
            halo = _halo_points(mask, top, left, body_h, body_w)
            if any(blocked[r, c] for r, c in halo):
                continue
            placements[idx] = (top, left)
            for r, c in halo:
                blocked[r, c] = True
            placed = True
            break
        if not placed:
            return None
    return [p for p in placements if p is not None]


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> np.ndarray:
    cells = np.argwhere(mask)
    for r, c in cells:
        grid[top + int(r), left + int(c)] = int(color)
    return grid


def _sample_hole_counts(n_objects: int, rng: random.Random) -> list[int]:
    patterns = {
        3: [
            (0, 1, 2),
            (0, 1, 1),
            (0, 2, 2),
            (1, 2, 3),
            (0, 0, 2),
            (1, 1, 3),
        ],
        4: [
            (0, 1, 2, 3),
            (0, 0, 1, 2),
            (0, 1, 1, 3),
            (0, 2, 2, 3),
            (1, 1, 2, 2),
            (0, 0, 2, 3),
        ],
    }
    return list(rng.choice(patterns[n_objects]))


def _place_color_band(grid: np.ndarray, colors: list[int], top_band: bool, rng: random.Random) -> tuple[int, tuple[int, ...]]:
    h, w = grid.shape
    divider_row = 1 if top_band else h - 2
    band_row = 0 if top_band else h - 1

    block_widths = [rng.randint(1, 3) for _ in colors]
    gaps = [rng.randint(1, 2) for _ in range(len(colors) - 1)]
    total = sum(block_widths) + sum(gaps)
    if total > w - 2:
        return -1, ()
    start = rng.randint(1, w - total - 1)
    cur = start
    for i, color in enumerate(colors):
        grid[band_row, cur : cur + block_widths[i]] = int(color)
        cur += block_widths[i]
        if i < len(gaps):
            cur += gaps[i]
    return divider_row, (band_row,)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.choice([3, 4])
    hole_counts = _sample_hole_counts(n_objects, rng)
    masks = [_make_object(hc, rng) for hc in hole_counts]

    max_h = max(mask.shape[0] for mask in masks)
    max_w = max(mask.shape[1] for mask in masks)
    body_h = rng.randint(max(12, max_h + 4), 24)
    body_w = rng.randint(max(16, max_w + 6), 28)
    placements = _place_objects(body_h, body_w, masks, rng)
    if placements is None:
        return None

    total_h = body_h + 2
    total_w = body_w
    if total_h > 30 or total_w > 30:
        return None

    color_pool = list(range(1, 10))
    rng.shuffle(color_pool)
    divider_color = color_pool[0]
    object_colors = color_pool[1 : 1 + n_objects]
    band_colors = color_pool[1 + n_objects : 1 + 2 * n_objects]

    grid = np.full((total_h, total_w), BG, dtype=int)
    top_band = rng.choice([False, True])
    divider_row, panel_rows = _place_color_band(grid, band_colors, top_band, rng)
    if divider_row < 0:
        return None
    grid[divider_row, :] = int(divider_color)

    row_offset = 2 if top_band else 0
    for mask, (top, left), color in zip(masks, placements, object_colors):
        _paint_mask(grid, mask, row_offset + top, left, color)

    info = _analyze_grid(grid)
    inferred_counts = sorted(obj["hole_count"] for obj in info["objects"])
    if inferred_counts != sorted(hole_counts):
        return None

    expected_band = tuple(band_colors)
    if tuple(info["colors"]) != expected_band:
        return None

    objects = sorted(
        info["objects"],
        key=lambda obj: (int(obj["hole_count"]), int(obj["anchor"][0]), int(obj["anchor"][1])),
    )
    if all(int(obj["color"]) == int(new_color) for obj, new_color in zip(objects, band_colors)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
