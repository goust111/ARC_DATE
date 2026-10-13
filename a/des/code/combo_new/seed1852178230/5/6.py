import random
from collections import defaultdict

import numpy as np


BLACK = 0
AZURE = 8
MAROON = 9
STAMP_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7]
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
TURN_LABELS = {1: "cw", -1: "ccw"}
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for coords in _components(grid == BLACK):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        perimeter = 2 * (h + w) - 4
        if len(coords) != perimeter:
            continue
        ok = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_edge = r in (r0, r1) or c in (c0, c1)
                if on_edge != bool(grid[r, c] == BLACK):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            frames.append((r0, c0, r1, c1))
    return frames


def _extract_stamps(
    grid: np.ndarray, frame_bboxes: list[tuple[int, int, int, int]]
) -> tuple[tuple[int, int, int, int], dict[int, np.ndarray]]:
    assert len(frame_bboxes) >= 2, "expected at least one canvas and one stamp"
    canvas_bbox = max(frame_bboxes, key=lambda box: (box[2] - box[0] + 1) * (box[3] - box[1] + 1))
    stamps: dict[int, np.ndarray] = {}
    for bbox in frame_bboxes:
        if bbox == canvas_bbox:
            continue
        r0, c0, r1, c1 = bbox
        stamp = grid[r0 + 1 : r1, c0 + 1 : c1].copy()
        colors = sorted(set(map(int, stamp.ravel())) - {AZURE})
        assert len(colors) == 1, "each stamp must have exactly one visible color"
        color = int(colors[0])
        assert color not in (BLACK, MAROON), "stamp color cannot collide with control colors"
        assert stamp.shape[0] % 2 == 1 and stamp.shape[1] % 2 == 1
        stamps[color] = stamp
    assert stamps, "expected at least one stamp"
    return canvas_bbox, stamps


def _marker_turn(v1: tuple[int, int], v2: tuple[int, int]) -> int:
    i1 = DIRS4.index(v1)
    i2 = DIRS4.index(v2)
    diff = (i2 - i1) % 4
    assert diff in (1, 3), "arrow must turn left or right"
    return 1 if diff == 1 else -1


def _extract_markers(inner: np.ndarray, stamp_colors: set[int]) -> list[dict]:
    h, w = inner.shape
    used_maroon: set[tuple[int, int]] = set()
    markers: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(inner[r, c])
            if color not in stamp_colors:
                continue
            candidates = []
            for dr1, dc1 in DIRS4:
                mr1, mc1 = r + dr1, c + dc1
                if not (0 <= mr1 < h and 0 <= mc1 < w):
                    continue
                if int(inner[mr1, mc1]) != MAROON:
                    continue
                second_steps = []
                for dr2, dc2 in DIRS4:
                    if (dr2, dc2) in ((dr1, dc1), (-dr1, -dc1)):
                        continue
                    mr2, mc2 = mr1 + dr2, mc1 + dc2
                    if not (0 <= mr2 < h and 0 <= mc2 < w):
                        continue
                    if int(inner[mr2, mc2]) == MAROON:
                        second_steps.append((mr2, mc2, dr2, dc2))
                if len(second_steps) == 1:
                    mr2, mc2, dr2, dc2 = second_steps[0]
                    candidates.append(((mr1, mc1), (mr2, mc2), (dr1, dc1), (dr2, dc2)))
            assert len(candidates) == 1, "each colored dot must match exactly one L arrow"
            m1, m2, v1, v2 = candidates[0]
            assert m1 not in used_maroon and m2 not in used_maroon, "arrow pixels cannot be shared"
            used_maroon.add(m1)
            used_maroon.add(m2)
            markers.append(
                {
                    "color": color,
                    "center": (r, c),
                    "m1": m1,
                    "m2": m2,
                    "turn": _marker_turn(v1, v2),
                }
            )
    maroon_cells = {(r, c) for r in range(h) for c in range(w) if int(inner[r, c]) == MAROON}
    assert used_maroon == maroon_cells, "all maroon pixels must belong to some marker"
    return markers


def _paste_stamp(inner: np.ndarray, stamp: np.ndarray, center: tuple[int, int]) -> None:
    sh, sw = stamp.shape
    cr, cc = center
    top = cr - sh // 2
    left = cc - sw // 2
    for sr in range(sh):
        for sc in range(sw):
            value = int(stamp[sr, sc])
            if value == AZURE:
                continue
            tr = top + sr
            tc = left + sc
            if 0 <= tr < inner.shape[0] and 0 <= tc < inner.shape[1] and int(inner[tr, tc]) == AZURE:
                inner[tr, tc] = value


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify framed stamps and a framed canvas, rotate each stamp from same-colored L-arrow markers, and print the rotated result onto the canvas.

    Input structure:
    - The background is Azure.
    - Several small black rectangular frames contain single-color stamp patterns on Azure interiors.
    - One larger black rectangular frame is the canvas; inside it, colored singleton dots are paired with two-pixel Maroon L arrows.
    - Each colored dot uses the same color as exactly one stamp.

    Rule:
    - Find every black rectangular frame; the largest one is the canvas and the others are stamps.
    - Match each canvas dot to its stamp by color.
    - For each marker, follow dot -> first Maroon -> second Maroon; a right turn means rotate clockwise 90 degrees, a left turn means rotate counterclockwise 90 degrees.
    - Count how many markers of each color appear on the canvas, rotate the matching stamp by that many quarter turns in the indicated direction, erase all marker pixels, then stamp the rotated pattern centered on every dot position without overwriting non-Azure cells.

    Key insight:
    - The transformation depends on combining three cues: frame size separates stamps from the canvas, color links each dot to one stamp, and the L-arrow turn plus same-color marker count determines the shared rotation applied at every placement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    canvas_bbox, stamps = _extract_stamps(g, _frame_bboxes(g))
    r0, c0, r1, c1 = canvas_bbox
    canvas = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    inner = canvas[1:-1, 1:-1]
    markers = _extract_markers(inner, set(stamps))

    by_color: dict[int, list[dict]] = defaultdict(list)
    for marker in markers:
        by_color[int(marker["color"])].append(marker)
        cr, cc = marker["center"]
        m1r, m1c = marker["m1"]
        m2r, m2c = marker["m2"]
        inner[cr, cc] = AZURE
        inner[m1r, m1c] = AZURE
        inner[m2r, m2c] = AZURE

    for color, stamp in stamps.items():
        group = by_color.get(int(color), [])
        assert group, "every stamp must have at least one matching marker"
        turns = {int(marker["turn"]) for marker in group}
        assert len(turns) == 1, "all markers of one color must agree on rotation direction"
        turn = turns.pop()
        k = len(group) % 4 if turn == -1 else (-len(group)) % 4
        rotated = np.rot90(stamp, k)
        for marker in group:
            _paste_stamp(inner, rotated, marker["center"])

    return canvas


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _random_stamp(size: int, color: int, rng: random.Random) -> np.ndarray | None:
    min_cells = 3 if size == 3 else 6
    max_cells = size * size - 2
    for _ in range(120):
        target = rng.randint(min_cells, max_cells)
        mask = np.zeros((size, size), dtype=bool)
        start = (rng.randrange(size), rng.randrange(size))
        mask[start] = True
        frontier = [start]
        while int(mask.sum()) < target and frontier:
            base_r, base_c = rng.choice(frontier)
            nbrs = []
            for nr, nc in _neighbors4(base_r, base_c, size, size):
                if not mask[nr, nc]:
                    nbrs.append((nr, nc))
            if not nbrs:
                frontier.remove((base_r, base_c))
                continue
            cell = rng.choice(nbrs)
            mask[cell] = True
            frontier.append(cell)
        if int(mask.sum()) != target:
            continue
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if len(rows) < 2 or len(cols) < 2:
            continue
        if not _mask_connected(mask):
            continue
        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        stamp = np.full((size, size), AZURE, dtype=int)
        stamp[mask] = color
        return stamp
    return None


def _choose_counts(num_stamps: int, rng: random.Random) -> list[int]:
    if num_stamps == 3:
        counts = [1, 2, 3]
        rng.shuffle(counts)
        return counts
    counts = rng.sample([1, 2, 3], 2)
    return counts


def _choose_turns(num_stamps: int, rng: random.Random) -> list[str]:
    turns = ["cw", "ccw"]
    if num_stamps == 3:
        turns.append(rng.choice(["cw", "ccw"]))
    rng.shuffle(turns)
    return turns


def _candidate_centers(occupied: np.ndarray, size: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    radius = size // 2
    centers: list[tuple[int, int]] = []
    for r in range(radius, h - radius):
        for c in range(radius, w - radius):
            if not occupied[r - radius : r + radius + 1, c - radius : c + radius + 1].any():
                centers.append((r, c))
    return centers


def _place_markers(
    canvas_h: int, canvas_w: int, specs: list[dict], rng: random.Random
) -> dict[int, list[tuple[int, int]]] | None:
    tasks = []
    for spec in specs:
        for _ in range(spec["count"]):
            tasks.append((spec["size"], spec["color"]))
    rng.shuffle(tasks)
    tasks.sort(key=lambda item: -item[0])

    occupied = np.zeros((canvas_h, canvas_w), dtype=bool)
    placed: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for size, color in tasks:
        centers = _candidate_centers(occupied, size)
        if not centers:
            return None
        center = rng.choice(centers)
        radius = size // 2
        r, c = center
        occupied[r - radius : r + radius + 1, c - radius : c + radius + 1] = True
        placed[int(color)].append(center)
    for color in placed:
        placed[color].sort()
    return dict(placed)


def _make_expected_canvas(canvas_h: int, canvas_w: int, specs: list[dict], markers: list[dict]) -> np.ndarray:
    canvas = np.full((canvas_h + 2, canvas_w + 2), BLACK, dtype=int)
    canvas[1:-1, 1:-1] = AZURE
    inner = canvas[1:-1, 1:-1]
    for marker in markers:
        cr, cc = marker["center"]
        m1r, m1c = marker["m1"]
        m2r, m2c = marker["m2"]
        inner[cr, cc] = marker["color"]
        inner[m1r, m1c] = MAROON
        inner[m2r, m2c] = MAROON

    painted = canvas.copy()
    out_inner = painted[1:-1, 1:-1]
    by_color = {int(spec["color"]): spec for spec in specs}
    grouped: dict[int, list[dict]] = defaultdict(list)
    for marker in markers:
        grouped[int(marker["color"])].append(marker)
        cr, cc = marker["center"]
        m1r, m1c = marker["m1"]
        m2r, m2c = marker["m2"]
        out_inner[cr, cc] = AZURE
        out_inner[m1r, m1c] = AZURE
        out_inner[m2r, m2c] = AZURE

    for color, group in grouped.items():
        spec = by_color[color]
        k = len(group) % 4 if spec["turn"] == "ccw" else (-len(group)) % 4
        rotated = np.rot90(spec["stamp"], k)
        for marker in group:
            _paste_stamp(out_inner, rotated, marker["center"])
    return canvas, painted


def _stamp_frame(stamp: np.ndarray) -> np.ndarray:
    framed = np.full((stamp.shape[0] + 2, stamp.shape[1] + 2), BLACK, dtype=int)
    framed[1:-1, 1:-1] = stamp
    return framed


def _build_scene(canvas: np.ndarray, stamp_frames: list[np.ndarray], rng: random.Random) -> np.ndarray | None:
    orientation = rng.choice(["top", "bottom", "left", "right"])
    count = len(stamp_frames)
    gaps = [rng.randint(1, 2) for _ in range(max(0, count - 1))]
    side_pad_a = rng.randint(1, 2)
    side_pad_b = rng.randint(1, 2)
    sep = rng.randint(1, 2)
    far_pad = rng.randint(1, 2)
    order = list(stamp_frames)
    rng.shuffle(order)

    if orientation in ("top", "bottom"):
        band_h = max(frame.shape[0] for frame in order)
        band_w = sum(frame.shape[1] for frame in order) + sum(gaps)
        canvas_h, canvas_w = canvas.shape
        grid_h = band_h + sep + canvas_h + side_pad_a + side_pad_b
        grid_w = max(band_w, canvas_w) + far_pad + side_pad_a
        if grid_h > 30 or grid_w > 30:
            return None
        scene = np.full((grid_h, grid_w), AZURE, dtype=int)
        if orientation == "top":
            band_y = side_pad_a
            canvas_y = band_y + band_h + sep
        else:
            canvas_y = side_pad_a
            band_y = canvas_y + canvas_h + sep
        band_x = rng.randint(0, grid_w - band_w)
        canvas_x = rng.randint(0, grid_w - canvas_w)
        scene[canvas_y : canvas_y + canvas_h, canvas_x : canvas_x + canvas_w] = canvas
        x = band_x
        for idx, frame in enumerate(order):
            y = band_y + rng.randint(0, band_h - frame.shape[0])
            h, w = frame.shape
            scene[y : y + h, x : x + w] = frame
            x += w + (gaps[idx] if idx < len(gaps) else 0)
        return scene

    band_w = max(frame.shape[1] for frame in order)
    band_h = sum(frame.shape[0] for frame in order) + sum(gaps)
    canvas_h, canvas_w = canvas.shape
    grid_h = max(band_h, canvas_h) + far_pad + side_pad_a
    grid_w = band_w + sep + canvas_w + side_pad_a + side_pad_b
    if grid_h > 30 or grid_w > 30:
        return None
    scene = np.full((grid_h, grid_w), AZURE, dtype=int)
    if orientation == "left":
        band_x = side_pad_a
        canvas_x = band_x + band_w + sep
    else:
        canvas_x = side_pad_a
        band_x = canvas_x + canvas_w + sep
    band_y = rng.randint(0, grid_h - band_h)
    canvas_y = rng.randint(0, grid_h - canvas_h)
    scene[canvas_y : canvas_y + canvas_h, canvas_x : canvas_x + canvas_w] = canvas
    y = band_y
    for idx, frame in enumerate(order):
        x = band_x + rng.randint(0, band_w - frame.shape[1])
        h, w = frame.shape
        scene[y : y + h, x : x + w] = frame
        y += h + (gaps[idx] if idx < len(gaps) else 0)
    return scene


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_stamps = rng.choice([2, 3])
    colors = rng.sample(STAMP_COLOR_POOL, num_stamps)
    counts = _choose_counts(num_stamps, rng)
    turns = _choose_turns(num_stamps, rng)
    sizes = [rng.choice([3, 5]) for _ in range(num_stamps)]

    specs: list[dict] = []
    for color, count, turn, size in zip(colors, counts, turns, sizes):
        stamp = _random_stamp(size, color, rng)
        if stamp is None:
            return None
        specs.append(
            {
                "color": int(color),
                "count": int(count),
                "turn": turn,
                "size": int(size),
                "stamp": stamp,
            }
        )

    total_markers = sum(spec["count"] for spec in specs)
    canvas_choices = [11, 13, 15]
    if total_markers >= 5:
        canvas_choices = [13, 15]

    placements = None
    canvas_h = canvas_w = 0
    for _ in range(20):
        canvas_h = rng.choice(canvas_choices)
        canvas_w = rng.choice(canvas_choices)
        placements = _place_markers(canvas_h, canvas_w, specs, rng)
        if placements is not None:
            break
    if placements is None:
        return None

    markers: list[dict] = []
    for spec in specs:
        color = int(spec["color"])
        for center in placements[color]:
            first_idx = rng.randrange(4)
            turn_step = 1 if spec["turn"] == "cw" else -1
            second_idx = (first_idx + turn_step) % 4
            dr1, dc1 = DIRS4[first_idx]
            dr2, dc2 = DIRS4[second_idx]
            cr, cc = center
            m1 = (cr + dr1, cc + dc1)
            m2 = (cr + dr1 + dr2, cc + dc1 + dc2)
            markers.append({"color": color, "center": center, "m1": m1, "m2": m2})

    input_canvas, expected = _make_expected_canvas(canvas_h, canvas_w, specs, markers)
    if not np.any(expected[1:-1, 1:-1] != AZURE):
        return None

    stamp_frames = [_stamp_frame(spec["stamp"]) for spec in specs]
    scene = _build_scene(input_canvas, stamp_frames, rng)
    if scene is None:
        return None

    if len(_frame_bboxes(scene)) != num_stamps + 1:
        return None
    if not np.array_equal(main(scene), expected):
        return None
    return scene


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        scene = _gen_attempt(rng)
        if scene is not None:
            return scene
    raise RuntimeError("failed to generate a valid stamp-and-canvas ARC input")
