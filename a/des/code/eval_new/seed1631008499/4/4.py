import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
AZURE = 8
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_FROM_DELTA = {v: k for k, v in DIRS.items()}
COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    vals = np.asarray(grid, dtype=int).reshape(-1)
    return int(np.bincount(vals, minlength=10).argmax())


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _normalize(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    return sorted((int(r - r0), int(c - c0)) for r, c in cells)


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return int(max(rows) + 1), int(max(cols) + 1)


def _extreme_anchor(cells: list[tuple[int, int]], direction: str) -> tuple[int, int]:
    if direction == "up":
        best = max(r for r, _ in cells)
        cands = [(r, c) for r, c in cells if r == best]
    elif direction == "down":
        best = min(r for r, _ in cells)
        cands = [(r, c) for r, c in cells if r == best]
    elif direction == "left":
        best = max(c for _, c in cells)
        cands = [(r, c) for r, c in cells if c == best]
    else:
        best = min(c for _, c in cells)
        cands = [(r, c) for r, c in cells if c == best]
    return min(cands)


def _render_canvas(
    canvas_shape: tuple[int, int],
    bg: int,
    stamps: dict[int, list[tuple[int, int]]],
    anchors: list[dict],
) -> np.ndarray:
    out = np.full(canvas_shape, int(bg), dtype=int)
    for anchor in sorted(anchors, key=lambda item: (int(item["r"]), int(item["c"]))):
        color = int(anchor["color"])
        direction = str(anchor["direction"])
        shape = stamps[color]
        ar, ac = _extreme_anchor(shape, direction)
        for sr, sc in shape:
            rr = int(anchor["r"]) + int(sr - ar)
            cc = int(anchor["c"]) + int(sc - ac)
            if 0 <= rr < canvas_shape[0] and 0 <= cc < canvas_shape[1]:
                out[rr, cc] = color
    return out


def _parse_stamps(top: np.ndarray, bg: int) -> dict[int, list[tuple[int, int]]]:
    stamps: dict[int, list[tuple[int, int]]] = {}
    for comp in _components(top != bg):
        colors = {int(top[r, c]) for r, c in comp}
        assert len(colors) == 1, "each stamp must be monochrome"
        color = int(next(iter(colors)))
        stamps[color] = _normalize(comp)
    assert stamps, "expected at least one stamp"
    return stamps


def _find_separator(grid: np.ndarray, bg: int) -> int:
    rows = [
        r
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != bg and np.all(grid[r] == grid[r, 0])
    ]
    assert len(rows) == 1, "expected exactly one solid separator row"
    return int(rows[0])


def _parse_canvas_and_anchors(bottom: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], list[dict]]:
    h, w = bottom.shape
    anchors: list[dict] = []
    anchor_cells = []
    for r in range(h):
        for c in range(w):
            value = int(bottom[r, c])
            if value in (bg, AZURE):
                continue
            azure_neighbors = []
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and int(bottom[nr, nc]) == AZURE:
                    azure_neighbors.append((int(nr), int(nc)))
            assert len(azure_neighbors) == 1, "each anchor must touch exactly one azure marker"
            mr, mc = azure_neighbors[0]
            direction = DIR_FROM_DELTA[(int(mr - r), int(mc - c))]
            anchors.append({"r": int(r), "c": int(c), "color": value, "direction": direction})
            anchor_cells.append((int(r), int(c)))

    assert anchors, "expected at least one lower-half anchor"
    rows = [r for r, _ in anchor_cells]
    cols = [c for _, c in anchor_cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert any(r == r0 for r, _ in anchor_cells)
    assert any(r == r1 for r, _ in anchor_cells)
    assert any(c == c0 for _, c in anchor_cells)
    assert any(c == c1 for _, c in anchor_cells)
    for item in anchors:
        assert item["r"] in (r0, r1) or item["c"] in (c0, c1), "anchors must lie on the canvas border"
    return (int(r0), int(r1), int(c0), int(c1)), anchors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the lower canvas by stamping the upper templates from color-matched border anchors.

    Input structure:
    - bg is the most common color, with a single full-width grey separator row splitting the grid into an upper stamp bank and a lower canvas area.
    - The upper area contains several disconnected monochrome stamp templates on the background; each template color is unique.
    - The lower area contains one rectangular canvas whose border carries colored anchor pixels, and each anchor touches exactly one adjacent azure marker pixel.

    Rule:
    - Match every lower anchor color to the upper stamp of the same color.
    - The azure marker position encodes the stamp expansion direction: marker above/down/left/right of the anchor means the placed stamp expands up/down/left/right from that anchor.
    - Align the anchor with the stamp pixel farthest opposite that expansion direction, paint the whole stamp on the canvas in the anchor color, and process anchors in row-major order so later placements overwrite earlier ones.
    - Remove the separator, the upper stamp bank, and all border markers; return only the reconstructed canvas rectangle.

    Key insight:
    - The azure neighbor does not rotate the stamp. It only tells which extremal stamp pixel should coincide with the border anchor, while color links anchors to templates.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    sep = _find_separator(g, bg)

    stamps = _parse_stamps(g[:sep], bg)
    canvas_bbox, anchors = _parse_canvas_and_anchors(g[sep + 1 :], bg)
    r0, r1, c0, c1 = canvas_bbox
    for anchor in anchors:
        assert int(anchor["color"]) in stamps, "every anchor color must match one stamp"
        anchor["r"] -= r0
        anchor["c"] -= c0

    canvas_shape = (int(r1 - r0 + 1), int(c1 - c0 + 1))
    return _render_canvas(canvas_shape, bg, stamps, anchors)


def _shape_unique_extremes(cells: list[tuple[int, int]]) -> bool:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return (
        rows.count(min(rows)) == 1
        and rows.count(max(rows)) == 1
        and cols.count(min(cols)) == 1
        and cols.count(max(cols)) == 1
    )


def _random_shape(rng: random.Random) -> list[tuple[int, int]]:
    for _ in range(300):
        target = rng.randint(4, 7)
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS.values():
                    nxt = (int(r + dr), int(c + dc))
                    if nxt not in cells and -3 <= nxt[0] <= 3 and -3 <= nxt[1] <= 3:
                        frontier.append(nxt)
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        norm = _normalize(cells)
        h, w = _bbox(norm)
        if h < 2 or w < 2 or h > 5 or w > 5:
            continue
        if len(norm) == h * w:
            continue
        if not _shape_unique_extremes(norm):
            continue
        return norm
    raise RuntimeError("failed to sample a valid stamp shape")


def _place_stamps_in_top(
    rng: random.Random,
    top_h: int,
    width: int,
    shapes: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    top = np.full((top_h, width), BG, dtype=int)
    occupied = np.zeros((top_h, width), dtype=bool)
    for color, shape in shapes.items():
        h, w = _bbox(shape)
        placed = False
        for _ in range(200):
            r0 = rng.randint(0, top_h - h)
            c0 = rng.randint(0, width - w)
            ok = True
            for sr, sc in shape:
                rr, cc = r0 + sr, c0 + sc
                for nr in range(max(0, rr - 1), min(top_h, rr + 2)):
                    for nc in range(max(0, cc - 1), min(width, cc + 2)):
                        if occupied[nr, nc]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue
            for sr, sc in shape:
                rr, cc = r0 + sr, c0 + sc
                occupied[rr, cc] = True
                top[rr, cc] = int(color)
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place upper stamps")
    return top


def _candidate_anchor_positions(
    shape: list[tuple[int, int]],
    direction: str,
    edge: str,
    canvas_h: int,
    canvas_w: int,
) -> list[tuple[int, int]]:
    ar, ac = _extreme_anchor(shape, direction)
    offsets = [(int(sr - ar), int(sc - ac)) for sr, sc in shape]
    out = []
    if edge in ("top", "bottom"):
        row = 0 if edge == "top" else canvas_h - 1
        span = range(1, canvas_w - 1)
        if canvas_w <= 2:
            span = range(canvas_w)
        for col in span:
            if all(0 <= row + dr < canvas_h and 0 <= col + dc < canvas_w for dr, dc in offsets):
                out.append((int(row), int(col)))
    else:
        col = 0 if edge == "left" else canvas_w - 1
        span = range(1, canvas_h - 1)
        if canvas_h <= 2:
            span = range(canvas_h)
        for row in span:
            if all(0 <= row + dr < canvas_h and 0 <= col + dc < canvas_w for dr, dc in offsets):
                out.append((int(row), int(col)))
    return out


def _choose_anchor(
    rng: random.Random,
    canvas_origin: tuple[int, int],
    canvas_shape: tuple[int, int],
    reserved: set[tuple[int, int]],
    shape: list[tuple[int, int]],
    color: int,
    edge: str,
    direction: str,
) -> dict | None:
    local_candidates = _candidate_anchor_positions(shape, direction, edge, canvas_shape[0], canvas_shape[1])
    rng.shuffle(local_candidates)
    dr, dc = DIRS[direction]
    for lr, lc in local_candidates:
        gr = int(canvas_origin[0] + lr)
        gc = int(canvas_origin[1] + lc)
        mr = int(gr + dr)
        mc = int(gc + dc)
        if (gr, gc) in reserved or (mr, mc) in reserved:
            continue
        reserved.add((gr, gc))
        reserved.add((mr, mc))
        return {
            "r": int(lr),
            "c": int(lc),
            "color": int(color),
            "direction": direction,
            "marker_global": (int(mr), int(mc)),
            "anchor_global": (int(gr), int(gc)),
            "edge": edge,
        }
    return None


def _paint_spec(
    total_h: int,
    total_w: int,
    top: np.ndarray,
    sep: int,
    anchors: list[dict],
) -> np.ndarray:
    grid = np.full((total_h, total_w), BG, dtype=int)
    grid[: top.shape[0], : top.shape[1]] = top
    grid[sep, :] = GREY
    for anchor in anchors:
        gr, gc = anchor["anchor_global"]
        mr, mc = anchor["marker_global"]
        grid[gr, gc] = int(anchor["color"])
        grid[mr, mc] = AZURE
    return grid


def _anchor_color_schedule(rng: random.Random, colors: list[int], total: int) -> list[int]:
    seq = colors[:]
    rng.shuffle(seq)
    while len(seq) < total:
        refill = colors[:]
        rng.shuffle(refill)
        seq.extend(refill)
    return seq[:total]


def _extra_specs(rng: random.Random) -> list[tuple[str, str]]:
    base = [("top", "down"), ("bottom", "up"), ("left", "right"), ("right", "left")]
    lateral = [
        ("top", "left"),
        ("top", "right"),
        ("bottom", "left"),
        ("bottom", "right"),
        ("left", "up"),
        ("left", "down"),
        ("right", "up"),
        ("right", "down"),
    ]
    specs: list[tuple[str, str]] = []
    extra_count = rng.randint(0, 3)
    if extra_count == 0:
        return specs
    if rng.random() < 0.7:
        specs.append(rng.choice(lateral))
    while len(specs) < extra_count:
        specs.append(rng.choice(base + lateral))
    rng.shuffle(specs)
    return specs


def _overlap_ratio(stamps: dict[int, list[tuple[int, int]]], anchors: list[dict], canvas_shape: tuple[int, int]) -> float:
    covered: Counter[tuple[int, int]] = Counter()
    for anchor in anchors:
        shape = stamps[int(anchor["color"])]
        ar, ac = _extreme_anchor(shape, str(anchor["direction"]))
        for sr, sc in shape:
            rr = int(anchor["r"] + sr - ar)
            cc = int(anchor["c"] + sc - ac)
            if 0 <= rr < canvas_shape[0] and 0 <= cc < canvas_shape[1]:
                covered[(rr, cc)] += 1
    total = sum(covered.values())
    if total == 0:
        return 0.0
    overlap = sum(v - 1 for v in covered.values() if v > 1)
    return overlap / total


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_stamps = rng.randint(2, 4)
    colors = rng.sample(COLOR_POOL, n_stamps)

    shapes: dict[int, list[tuple[int, int]]] = {}
    seen_shapes = set()
    for color in colors:
        shape = _random_shape(rng)
        key = tuple(shape)
        if key in seen_shapes:
            return None
        seen_shapes.add(key)
        shapes[int(color)] = shape

    total_w = rng.randint(18, 28)
    top_h = rng.randint(7, 11)
    top_margin = rng.randint(1, 3)
    bottom_margin = rng.randint(1, 3)
    left_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)
    max_canvas_w = min(18, total_w - left_margin - right_margin)
    if max_canvas_w < 10:
        return None
    canvas_w = rng.randint(10, max_canvas_w)
    canvas_h = rng.randint(10, 15)
    total_h = top_h + 1 + top_margin + canvas_h + bottom_margin
    if total_h > 30:
        return None

    top = _place_stamps_in_top(rng, top_h, total_w, shapes)
    sep = top_h
    canvas_origin = (int(sep + 1 + top_margin), int(left_margin))
    canvas_shape = (int(canvas_h), int(canvas_w))

    reserved: set[tuple[int, int]] = set()
    anchors: list[dict] = []
    mandatory_specs = [("top", "down"), ("bottom", "up"), ("left", "right"), ("right", "left")]
    extra_specs = _extra_specs(rng)
    full_specs = mandatory_specs + extra_specs
    color_schedule = _anchor_color_schedule(rng, colors, len(full_specs))

    for idx, (edge, direction) in enumerate(full_specs):
        chosen = _choose_anchor(
            rng,
            canvas_origin,
            canvas_shape,
            reserved,
            shapes[int(color_schedule[idx])],
            int(color_schedule[idx]),
            edge,
            direction,
        )
        if chosen is None:
            return None
        anchors.append(chosen)

    grid = _paint_spec(total_h, total_w, top, sep, anchors)
    painted = _render_canvas(
        canvas_shape,
        BG,
        shapes,
        [{"r": a["r"], "c": a["c"], "color": a["color"], "direction": a["direction"]} for a in anchors],
    )
    density = float(np.count_nonzero(painted) / painted.size)
    if density < 0.08 or density > 0.48:
        return None
    if _overlap_ratio(shapes, anchors, canvas_shape) > 0.45:
        return None
    if not np.array_equal(main(grid), painted):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-and-canvas input")
