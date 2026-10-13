from __future__ import annotations

from collections import Counter, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MOVE_BY_CONTACT = {
    "above": (0, 1),
    "below": (0, -1),
    "left": (1, 0),
    "right": (-1, 0),
}

RECT_DIMS = tuple(
    (h, w) for h in range(2, 5) for w in range(2, 5) if 4 <= h * w <= 12
)
MAX_RECT_DIM = max(max(h, w) for h, w in RECT_DIMS)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 200
MARKER_RETRIES = 200


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        q: deque[tuple[int, int]] = deque([(rr, cc)])
        pts: list[tuple[int, int]] = []
        while q:
            cr, cc2 = q.popleft()
            pts.append((cr, cc2))
            for dr, dc in DIR4:
                nr = int(cr + dr)
                nc = int(cc2 + dc)
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _detect_rails(grid: np.ndarray, bg: int) -> tuple[int, set[int], set[int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, set[int], set[int]]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        rows = {r for r in range(h) if np.all(grid[r, :] == color)}
        cols = {c for c in range(w) if np.all(grid[:, c] == color)}
        if rows or cols:
            score = len(rows) + len(cols)
            pixels = int(np.count_nonzero(grid == color))
            candidates.append((score, pixels, color, rows, cols))
    if not candidates:
        raise ValueError("no rail color found")
    _, _, color, rows, cols = max(candidates, key=lambda item: (item[0], item[1]))
    return int(color), set(rows), set(cols)


def _bbox_contacts(
    r0: int,
    c0: int,
    h: int,
    w: int,
    grid_shape: tuple[int, int],
    rail_rows: set[int],
    rail_cols: set[int],
) -> list[str]:
    gh, gw = grid_shape
    contacts: list[str] = []
    if r0 > 0 and (r0 - 1) in rail_rows:
        contacts.append("below")
    if r0 + h < gh and (r0 + h) in rail_rows:
        contacts.append("above")
    if c0 > 0 and (c0 - 1) in rail_cols:
        contacts.append("right")
    if c0 + w < gw and (c0 + w) in rail_cols:
        contacts.append("left")
    return contacts


def _bbox_crosses_rail(
    r0: int, c0: int, h: int, w: int, rail_rows: set[int], rail_cols: set[int]
) -> bool:
    for rr in range(r0, r0 + h):
        if rr in rail_rows:
            return True
    for cc in range(c0, c0 + w):
        if cc in rail_cols:
            return True
    return False


def _choose_marker_color(
    grid: np.ndarray,
    bg: int,
    rail_color: int,
    rectangles: list[tuple[int, int, int, int]],
) -> int:
    colors = [int(c) for c in np.unique(grid) if int(c) not in {bg, rail_color}]
    metrics: list[tuple[int, int, int, int]] = []
    for color in colors:
        comps = _components(grid == color)
        if not comps:
            continue
        max_size = max(int(len(comp)) for comp in comps)
        total = int(np.count_nonzero(grid == color))
        appearances = 0
        for r0, c0, r1, c1 in rectangles:
            if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
                appearances += 1
        metrics.append((appearances, max_size, total, color))
    if not metrics:
        raise ValueError("no marker color candidates found")
    best = max(metrics, key=lambda item: (item[0], -item[1], -item[2]))
    return int(best[3])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored rectangle along its touching rail by the number of embedded marker pixels.

    Input structure:
    - The background is the dominant color.
    - One foreground color forms one-pixel-wide full-length horizontal and/or vertical rail lines.
    - Several non-overlapping axis-aligned filled rectangles of other colors sit next to those rails, and each rectangle contains embedded single-cell marker pixels of a shared marker color.

    Rule:
    - Treat every 4-connected non-background, non-rail filled rectangle as one movable object.
    - Determine which rail the rectangle touches and whether the rectangle is above, below, left, or right of that rail.
    - Horizontal-rail contacts move the rectangle left or right; vertical-rail contacts move it up or down; the step count equals the number of marker pixels inside the rectangle.
    - Erase all rails and marker pixels, then redraw every rectangle at its translated location as a pure solid block of its own rectangle color.

    Key insight:
    - The rail orientation gives the movement axis, the side of contact gives the sign of motion, and the hidden marker count gives the exact distance.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    rail_color, rail_rows, rail_cols = _detect_rails(g, bg)

    mask = (g != bg) & (g != rail_color)
    comps = _components(mask)
    rectangles: list[tuple[int, int, int, int]] = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        if int(len(comp)) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        rectangles.append((r0, c0, r1, c1))

    if not rectangles:
        return np.full_like(g, bg)

    marker_color = _choose_marker_color(g, bg, rail_color, rectangles)
    out = np.full_like(g, bg)

    for r0, c0, r1, c1 in rectangles:
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = [int(c) for c in np.unique(sub) if int(c) not in {bg, rail_color}]
        if not colors:
            continue

        if marker_color in colors and len(colors) >= 2:
            rect_color = max(
                (color for color in colors if color != marker_color),
                key=lambda color: int(np.count_nonzero(sub == color)),
            )
            distance = int(np.count_nonzero(sub == marker_color))
        else:
            comp_sizes = {}
            for color in colors:
                local_comps = _components(sub == color)
                comp_sizes[color] = max(int(len(comp)) for comp in local_comps)
            rect_color = max(colors, key=lambda color: (comp_sizes[color], np.count_nonzero(sub == color)))
            other_colors = [color for color in colors if color != rect_color]
            distance = int(sum(np.count_nonzero(sub == color) for color in other_colors))

        h = r1 - r0 + 1
        w = c1 - c0 + 1
        contacts = _bbox_contacts(r0, c0, h, w, g.shape, rail_rows, rail_cols)
        if len(contacts) != 1:
            continue
        dr, dc = MOVE_BY_CONTACT[contacts[0]]
        nr0 = r0 + dr * distance
        nc0 = c0 + dc * distance
        nr1 = nr0 + h - 1
        nc1 = nc0 + w - 1
        if not (0 <= nr0 <= nr1 < g.shape[0] and 0 <= nc0 <= nc1 < g.shape[1]):
            continue
        out[nr0 : nr1 + 1, nc0 : nc1 + 1] = int(rect_color)

    return out


def _sample_spaced_positions(
    length: int, count: int, margin: int, min_gap: int, rng: np.random.Generator
) -> list[int] | None:
    candidates = list(range(margin, length - margin))
    rng.shuffle(candidates)
    chosen: list[int] = []
    for pos in candidates:
        if all(abs(pos - other) >= min_gap for other in chosen):
            chosen.append(int(pos))
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _bbox_clear(
    used: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int = 1
) -> bool:
    gh, gw = used.shape
    if r0 < 0 or c0 < 0 or r0 + h > gh or c0 + w > gw:
        return False
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(gh, r0 + h + margin)
    cc1 = min(gw, c0 + w + margin)
    return not bool(np.any(used[rr0:rr1, cc0:cc1]))


def _mark_used(used: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int = 1) -> None:
    gh, gw = used.shape
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(gh, r0 + h + margin)
    cc1 = min(gw, c0 + w + margin)
    used[rr0:rr1, cc0:cc1] = True


def _sample_marker_offsets(
    h: int, w: int, count: int, rng: np.random.Generator
) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    if count > (h * w + 1) // 2:
        return None
    for _ in range(MARKER_RETRIES):
        order = rng.permutation(len(cells))
        chosen: list[tuple[int, int]] = []
        for idx in order:
            cell = cells[int(idx)]
            if all(abs(cell[0] - rr) + abs(cell[1] - cc) > 1 for rr, cc in chosen):
                chosen.append(cell)
                if len(chosen) == count:
                    break
        if len(chosen) != count:
            continue
        body = np.ones((h, w), dtype=bool)
        for rr, cc in chosen:
            body[rr, cc] = False
        body_comps = _components(body)
        if body_comps and max(int(len(comp)) for comp in body_comps) >= 2:
            return chosen
    return None


def _candidate_bbox_for_lane(
    lane: tuple[str, int, str],
    h: int,
    w: int,
    distance: int,
    grid_shape: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    gh, gw = grid_shape
    kind, index, side = lane
    if kind == "H":
        r0 = int(index - h) if side == "above" else int(index + 1)
        if not (0 <= r0 and r0 + h <= gh):
            return None
        if side == "above":
            min_c = distance
            max_c = gw - w
            if min_c > max_c:
                return None
            out_c0 = int(rng.integers(min_c, max_c + 1))
            in_c0 = out_c0 - distance
        else:
            min_c = 0
            max_c = gw - w - distance
            if min_c > max_c:
                return None
            out_c0 = int(rng.integers(min_c, max_c + 1))
            in_c0 = out_c0 + distance
        return (r0, in_c0, h, w), (r0, out_c0, h, w)

    c0 = int(index - w) if side == "left" else int(index + 1)
    if not (0 <= c0 and c0 + w <= gw):
        return None
    if side == "left":
        min_r = distance
        max_r = gh - h
        if min_r > max_r:
            return None
        out_r0 = int(rng.integers(min_r, max_r + 1))
        in_r0 = out_r0 - distance
    else:
        min_r = 0
        max_r = gh - h - distance
        if min_r > max_r:
            return None
        out_r0 = int(rng.integers(min_r, max_r + 1))
        in_r0 = out_r0 + distance
    return (in_r0, c0, h, w), (out_r0, c0, h, w)


def _valid_bbox_for_lane(
    bbox: tuple[int, int, int, int],
    lane: tuple[str, int, str],
    grid_shape: tuple[int, int],
    rail_rows: set[int],
    rail_cols: set[int],
) -> bool:
    r0, c0, h, w = bbox
    if _bbox_crosses_rail(r0, c0, h, w, rail_rows, rail_cols):
        return False
    contacts = _bbox_contacts(r0, c0, h, w, grid_shape, rail_rows, rail_cols)
    return contacts == [lane[2]]


def _build_lanes(
    grid_shape: tuple[int, int], rail_rows: list[int], rail_cols: list[int]
) -> list[tuple[str, int, str]]:
    gh, gw = grid_shape
    lanes: list[tuple[str, int, str]] = []
    for row in rail_rows:
        if row >= 2:
            lanes.append(("H", int(row), "above"))
        if gh - row - 1 >= 2:
            lanes.append(("H", int(row), "below"))
    for col in rail_cols:
        if col >= 2:
            lanes.append(("V", int(col), "left"))
        if gw - col - 1 >= 2:
            lanes.append(("V", int(col), "right"))
    return lanes


def _choose_rail_layout(
    h: int, w: int, rng: np.random.Generator
) -> tuple[list[int], list[int]] | None:
    mode = str(rng.choice(np.array(["horizontal", "vertical", "mixed"], dtype=object)))
    if mode == "horizontal":
        num_h = int(rng.integers(1, 3))
        num_v = 0
    elif mode == "vertical":
        num_h = 0
        num_v = int(rng.integers(1, 3))
    else:
        num_h = 1
        num_v = 1

    margin = MAX_RECT_DIM
    min_gap = MAX_RECT_DIM + 2
    rail_rows = [] if num_h == 0 else _sample_spaced_positions(h, num_h, margin, min_gap, rng)
    rail_cols = [] if num_v == 0 else _sample_spaced_positions(w, num_v, margin, min_gap, rng)
    if rail_rows is None or rail_cols is None:
        return None
    return rail_rows, rail_cols


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(16, 31))
    w = int(rng.integers(16, 31))
    layout = _choose_rail_layout(h, w, rng)
    if layout is None:
        return None
    rail_rows, rail_cols = layout
    lanes = _build_lanes((h, w), rail_rows, rail_cols)
    if not lanes:
        return None

    palette = list(range(10))
    bg = int(rng.choice(palette))
    remaining = [color for color in palette if color != bg]
    rail_color = int(rng.choice(remaining))
    remaining.remove(rail_color)
    marker_color = int(rng.choice(remaining))
    remaining.remove(marker_color)

    rect_count = int(rng.integers(2, min(5, len(remaining) + 1)))
    rect_colors = [int(color) for color in rng.choice(remaining, size=rect_count, replace=False)]

    used_in = np.zeros((h, w), dtype=bool)
    used_out = np.zeros((h, w), dtype=bool)
    lane_use = {lane: 0 for lane in lanes}
    rectangles: list[dict[str, object]] = []

    for color in rect_colors:
        placed = False
        for attempt in range(PLACEMENT_RETRIES):
            min_use = min(lane_use.values())
            preferred = [lane for lane in lanes if lane_use[lane] == min_use]
            lane_pool = preferred if attempt < PLACEMENT_RETRIES // 2 else lanes
            lane = lane_pool[int(rng.integers(0, len(lane_pool)))]

            distance = int(rng.integers(1, 6))
            dims = [(hh, ww) for hh, ww in RECT_DIMS if hh * ww - distance >= 2]
            if not dims:
                continue
            rng.shuffle(dims)

            for rh, rw in dims:
                markers = _sample_marker_offsets(rh, rw, distance, rng)
                if markers is None:
                    continue
                pair = _candidate_bbox_for_lane(lane, rh, rw, distance, (h, w), rng)
                if pair is None:
                    continue
                in_bbox, out_bbox = pair
                if not _valid_bbox_for_lane(in_bbox, lane, (h, w), set(rail_rows), set(rail_cols)):
                    continue
                if not _valid_bbox_for_lane(out_bbox, lane, (h, w), set(rail_rows), set(rail_cols)):
                    continue

                ir0, ic0, ih, iw = in_bbox
                or0, oc0, oh, ow = out_bbox
                if not _bbox_clear(used_in, ir0, ic0, ih, iw):
                    continue
                if not _bbox_clear(used_out, or0, oc0, oh, ow):
                    continue

                _mark_used(used_in, ir0, ic0, ih, iw)
                _mark_used(used_out, or0, oc0, oh, ow)
                lane_use[lane] += 1
                rectangles.append(
                    {
                        "color": color,
                        "input_bbox": in_bbox,
                        "output_bbox": out_bbox,
                        "markers": markers,
                    }
                )
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    for row in rail_rows:
        inp[int(row), :] = rail_color
    for col in rail_cols:
        inp[:, int(col)] = rail_color

    for rect in rectangles:
        color = int(rect["color"])
        ir0, ic0, ih, iw = rect["input_bbox"]
        or0, oc0, oh, ow = rect["output_bbox"]
        inp[ir0 : ir0 + ih, ic0 : ic0 + iw] = color
        out[or0 : or0 + oh, oc0 : oc0 + ow] = color
        for mr, mc in rect["markers"]:
            inp[ir0 + int(mr), ic0 + int(mc)] = marker_color

    if _dominant_color(inp) != bg:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rail-slide input")
