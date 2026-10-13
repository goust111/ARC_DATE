import random
import numpy as np


def _bbox_from_coords(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _intervals_overlap(a0, a1, b0, b1):
    return not (a1 < b0 or b1 < a0)


def _find_interior_components(grid, bg, ignore=None):
    h, w = grid.shape
    ignore = set() if ignore is None else set(ignore)
    seen = np.zeros((h, w), dtype=bool)
    for r, c in ignore:
        if 0 <= r < h and 0 <= c < w:
            seen[r, c] = True

    comps = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if not (1 <= nr < h - 1 and 1 <= nc < w - 1):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            bbox = _bbox_from_coords(coords)
            comps.append(
                {
                    "color": color,
                    "coords": np.array(coords, dtype=int),
                    "bbox": bbox,
                    "area": len(coords),
                }
            )
    return comps


def _parse_scene(grid):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 6 and w >= 6

    corners = [int(g[0, 0]), int(g[0, w - 1]), int(g[h - 1, 0]), int(g[h - 1, w - 1])]
    assert len(set(corners)) == 1
    bg = corners[0]

    top = int(g[0, 1])
    bottom = int(g[h - 1, 1])
    left = int(g[1, 0])
    right = int(g[1, w - 1])
    sides = {"top": top, "bottom": bottom, "left": left, "right": right}
    assert bg not in sides.values()
    assert len(set(sides.values())) == 4
    assert np.all(g[0, 1 : w - 1] == top)
    assert np.all(g[h - 1, 1 : w - 1] == bottom)
    assert np.all(g[1 : h - 1, 0] == left)
    assert np.all(g[1 : h - 1, w - 1] == right)

    marker_cells = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) in sides.values():
                marker_cells.append((r, c))
    assert len(marker_cells) == 1
    marker = marker_cells[0]
    marker_color = int(g[marker])

    if marker_color == top:
        direction = "up"
    elif marker_color == bottom:
        direction = "down"
    elif marker_color == left:
        direction = "left"
    else:
        assert marker_color == right
        direction = "right"

    rects = _find_interior_components(g, bg, ignore={marker})
    assert rects
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        color = int(rect["color"])
        area = int(rect["area"])
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert area == height * width
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color)
        rect["h"] = height
        rect["w"] = width

    return g, bg, sides, marker, direction, rects


def _first_rect_index(rects, marker, direction):
    mr, mc = marker
    candidates = []
    for idx, rect in enumerate(rects):
        r0, c0, r1, c1 = rect["bbox"]
        if direction == "right" and r0 <= mr <= r1 and c0 > mc:
            candidates.append((c0, idx))
        elif direction == "left" and r0 <= mr <= r1 and c1 < mc:
            candidates.append((-c1, idx))
        elif direction == "down" and c0 <= mc <= c1 and r0 > mr:
            candidates.append((r0, idx))
        elif direction == "up" and c0 <= mc <= c1 and r1 < mr:
            candidates.append((-r1, idx))
    assert candidates
    candidates.sort()
    assert len(candidates) == 1 or candidates[0][0] != candidates[1][0]
    return candidates[0][1]


def _next_rect_index(rects, current_idx, direction, used):
    cur = rects[current_idx]
    cr0, cc0, cr1, cc1 = cur["bbox"]
    candidates = []
    for idx, rect in enumerate(rects):
        if idx in used:
            continue
        r0, c0, r1, c1 = rect["bbox"]
        if direction == "right":
            if c0 > cc1 and _intervals_overlap(cr0, cr1, r0, r1):
                candidates.append((c0, idx))
        elif direction == "left":
            if c1 < cc0 and _intervals_overlap(cr0, cr1, r0, r1):
                candidates.append((-c1, idx))
        elif direction == "down":
            if r0 > cr1 and _intervals_overlap(cc0, cc1, c0, c1):
                candidates.append((r0, idx))
        else:
            if r1 < cr0 and _intervals_overlap(cc0, cc1, c0, c1):
                candidates.append((-r1, idx))
    if not candidates:
        return None
    candidates.sort()
    assert len(candidates) == 1 or candidates[0][0] != candidates[1][0]
    return candidates[0][1]


def _build_chain(rects, marker, direction):
    chain = [_first_rect_index(rects, marker, direction)]
    used = set(chain)
    while True:
        nxt = _next_rect_index(rects, chain[-1], direction, used)
        if nxt is None:
            break
        chain.append(nxt)
        used.add(nxt)
    return chain


def _destination_bbox(rects, idx, successor_idx, direction, shape):
    h, w = shape
    r0, c0, r1, c1 = rects[idx]["bbox"]
    rh = r1 - r0 + 1
    rw = c1 - c0 + 1
    if direction == "right":
        new_c1 = (w - 2) if successor_idx is None else rects[successor_idx]["bbox"][1] - 1
        new_c0 = new_c1 - rw + 1
        return r0, new_c0, r1, new_c1
    if direction == "left":
        new_c0 = 1 if successor_idx is None else rects[successor_idx]["bbox"][3] + 1
        new_c1 = new_c0 + rw - 1
        return r0, new_c0, r1, new_c1
    if direction == "down":
        new_r1 = (h - 2) if successor_idx is None else rects[successor_idx]["bbox"][0] - 1
        new_r0 = new_r1 - rh + 1
        return new_r0, c0, new_r1, c1
    new_r0 = 1 if successor_idx is None else rects[successor_idx]["bbox"][2] + 1
    new_r1 = new_r0 + rh - 1
    return new_r0, c0, new_r1, c1


def main(grid: np.ndarray) -> np.ndarray:
    """Launch a momentum relay from the pusher marker so each struck rectangle slides until it hands motion to the next, with only the chain tail docking to the frame.

    Input structure:
    - The grid has a one-cell-thick outer rectangular frame whose four sides have distinct colors and whose corners are background.
    - Inside the frame are several solid monochrome rectangles plus one single-cell pusher marker.
    - The marker color matches exactly one frame side and differs from every rectangle color.

    Rule:
    - The matching side determines the push direction toward that side.
    - Along that direction, the first rectangle whose row-span or column-span intersects the marker line starts moving.
    - A moving rectangle travels straight until it reaches the nearest later rectangle with overlapping orthogonal span, stops flush against that rectangle's original position, and passes the motion onward; the last rectangle in the chain slides until it is flush with the frame.
    - Rectangles outside this relay chain do not move, and the pusher marker is erased.

    Key insight:
    - Motion is not a global gravity step and not a single-object shove: the affected set grows one rectangle at a time through successive span-overlap collisions, so each earlier rectangle stops at its own impact point while only the tail reaches the frame.
    """
    g, bg, _sides, marker, direction, rects = _parse_scene(grid)
    chain = _build_chain(rects, marker, direction)

    out = g.copy()
    out[marker] = bg
    for idx in chain:
        coords = rects[idx]["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    occupied = out != bg
    for pos, idx in enumerate(chain):
        successor = chain[pos + 1] if pos + 1 < len(chain) else None
        nr0, nc0, nr1, nc1 = _destination_bbox(rects, idx, successor, direction, g.shape)
        assert 1 <= nr0 <= nr1 <= g.shape[0] - 2
        assert 1 <= nc0 <= nc1 <= g.shape[1] - 2
        assert not np.any(occupied[nr0 : nr1 + 1, nc0 : nc1 + 1])
        occupied[nr0 : nr1 + 1, nc0 : nc1 + 1] = True
        out[nr0 : nr1 + 1, nc0 : nc1 + 1] = int(rects[idx]["color"])

    return out


GEN_ATTEMPTS = 400


def _render_frame(h, w, side_colors, bg=0):
    top, bottom, left, right = side_colors
    grid = np.full((h, w), bg, dtype=int)
    grid[0, 1 : w - 1] = top
    grid[h - 1, 1 : w - 1] = bottom
    grid[1 : h - 1, 0] = left
    grid[1 : h - 1, w - 1] = right
    return grid


def _choose_chain_rows(h, heights):
    for _ in range(120):
        starts = []
        first_h = heights[0]
        first_r0 = random.randint(1, h - 1 - first_h)
        starts.append(first_r0)
        marker_row = random.randint(first_r0, first_r0 + first_h - 1)
        prev_r0 = first_r0
        prev_r1 = first_r0 + first_h - 1
        for rect_h in heights[1:]:
            lo = max(1, prev_r0 - rect_h + 1)
            hi = min(h - 1 - rect_h, prev_r1)
            if lo > hi:
                break
            choices = list(range(lo, hi + 1))
            random.shuffle(choices)
            pick = None
            for cand in choices:
                if cand != prev_r0:
                    pick = cand
                    break
            if pick is None:
                pick = choices[0]
            starts.append(pick)
            prev_r0 = pick
            prev_r1 = pick + rect_h - 1
        if len(starts) != len(heights):
            continue
        intervals = [(r0, r0 + rect_h - 1) for r0, rect_h in zip(starts, heights)]
        if any(not (r0 <= marker_row <= r1) for r0, r1 in intervals[1:]):
            return intervals, marker_row
    return None


def _place_decoy(h, first_rect, marker_row, marker_c, first_c0):
    first_r0, _, first_r1, _ = first_rect
    max_w = first_c0 - marker_c - 1
    if max_w < 2:
        return None
    decoy_w = random.randint(2, max_w)
    possible = []
    for decoy_h in (2, 3):
        if first_r0 - decoy_h >= 1:
            for r0 in range(1, first_r0 - decoy_h + 1):
                possible.append((r0, decoy_h))
        below_max = h - 1 - decoy_h
        if first_r1 + 1 <= below_max:
            for r0 in range(first_r1 + 1, below_max + 1):
                possible.append((r0, decoy_h))
    if not possible:
        for decoy_h in (2, 3):
            for r0 in range(1, h - 1 - decoy_h + 1):
                if not (r0 <= marker_row <= r0 + decoy_h - 1):
                    possible.append((r0, decoy_h))
    if not possible:
        return None
    r0, decoy_h = random.choice(possible)
    c0 = random.randint(marker_c + 1, first_c0 - decoy_w)
    return r0, c0, decoy_h, decoy_w


def _build_right_scene():
    bg = 0
    side_colors = random.sample(range(1, 10), 4)
    top, bottom, left, right = side_colors
    free_colors = [c for c in range(1, 10) if c not in side_colors]

    chain_len = random.choice((2, 3))
    widths = [random.randint(2, 3) for _ in range(chain_len)]
    heights = [random.randint(2, 5) for _ in range(chain_len)]
    between_gaps = [random.randint(1, 2) for _ in range(chain_len - 1)]
    tail_gap = random.randint(1, 2)
    gap_signature = between_gaps + [tail_gap]
    if len(set(gap_signature)) == 1:
        gap_signature[random.randrange(len(gap_signature))] = 3
        if gap_signature[-1] > 2:
            tail_gap = gap_signature[-1]
        else:
            between_gaps = gap_signature[:-1]

    fixed_left_space = 5
    inner_rest = sum(widths) + sum(between_gaps) + tail_gap
    inner_width = fixed_left_space + inner_rest
    if inner_width > 20:
        return None
    extra_left = random.randint(0, 20 - inner_width)
    left_space = fixed_left_space + extra_left
    w = left_space + inner_rest + 2

    min_h = max(12, max(heights) + 4)
    if min_h > 20:
        return None
    h = random.randint(min_h, 20)

    row_setup = _choose_chain_rows(h, heights)
    if row_setup is None:
        return None
    intervals, marker_row = row_setup

    colors = free_colors[: chain_len + 1]
    rects = []
    inner_right = w - 2
    c1 = inner_right - tail_gap
    for idx in range(chain_len - 1, -1, -1):
        rect_w = widths[idx]
        rect_h = heights[idx]
        r0, r1 = intervals[idx]
        c0 = c1 - rect_w + 1
        rects.append(
            {
                "color": colors[idx],
                "r0": r0,
                "c0": c0,
                "h": rect_h,
                "w": rect_w,
                "chain": True,
            }
        )
        if idx > 0:
            c1 = c0 - between_gaps[idx - 1] - 1
    rects.reverse()

    first = rects[0]
    first_c0 = first["c0"]
    marker_c = random.randint(1, min(2, first_c0 - 3))
    decoy = _place_decoy(
        h,
        (first["r0"], first["c0"], first["r0"] + first["h"] - 1, first["c0"] + first["w"] - 1),
        marker_row,
        marker_c,
        first_c0,
    )
    if decoy is None:
        return None
    dr0, dc0, dh, dw = decoy
    rects.append({"color": colors[-1], "r0": dr0, "c0": dc0, "h": dh, "w": dw, "chain": False})

    scene = {
        "h": h,
        "w": w,
        "bg": bg,
        "side_colors": (top, bottom, left, right),
        "marker": (marker_row, marker_c),
        "rects": rects,
        "chain_len": chain_len,
        "gap_signature": gap_signature,
    }
    if not _scene_is_valid(scene):
        return None
    return scene


def _scene_is_valid(scene):
    h = scene["h"]
    w = scene["w"]
    marker_r, marker_c = scene["marker"]
    rects = scene["rects"]
    chain = rects[: scene["chain_len"]]
    decoy = rects[scene["chain_len"]]

    assert 1 <= marker_r <= h - 2 and 1 <= marker_c <= w - 2
    assert len(chain) >= 2
    assert any(
        not (rect["r0"] <= marker_r <= rect["r0"] + rect["h"] - 1) for rect in chain[1:]
    )
    assert len(set(scene["gap_signature"])) >= 2

    boxes = []
    for rect in rects:
        r0, c0 = rect["r0"], rect["c0"]
        r1 = r0 + rect["h"] - 1
        c1 = c0 + rect["w"] - 1
        if not (1 <= r0 <= r1 <= h - 2 and 1 <= c0 <= c1 <= w - 2):
            return False
        for br0, bc0, br1, bc1 in boxes:
            if not (r1 < br0 or br1 < r0 or c1 < bc0 or bc1 < c0):
                return False
        boxes.append((r0, c0, r1, c1))

    for left_rect, right_rect in zip(chain, chain[1:]):
        lr0, lc0 = left_rect["r0"], left_rect["c0"]
        lr1, lc1 = lr0 + left_rect["h"] - 1, lc0 + left_rect["w"] - 1
        rr0, rc0 = right_rect["r0"], right_rect["c0"]
        rr1, _ = rr0 + right_rect["h"] - 1, rc0 + right_rect["w"] - 1
        if rc0 <= lc1:
            return False
        if not _intervals_overlap(lr0, lr1, rr0, rr1):
            return False

    first = chain[0]
    if not (first["r0"] <= marker_r <= first["r0"] + first["h"] - 1 and first["c0"] > marker_c):
        return False

    dr0, dc0 = decoy["r0"], decoy["c0"]
    dr1 = dr0 + decoy["h"] - 1
    dc1 = dc0 + decoy["w"] - 1
    if not (dc0 > marker_c and dc1 < first["c0"]):
        return False
    if dr0 <= marker_r <= dr1:
        return False

    return True


def _render_input(scene):
    grid = _render_frame(scene["h"], scene["w"], scene["side_colors"], bg=scene["bg"])
    for rect in scene["rects"]:
        r0, c0 = rect["r0"], rect["c0"]
        grid[r0 : r0 + rect["h"], c0 : c0 + rect["w"]] = rect["color"]
    grid[scene["marker"]] = scene["side_colors"][3]
    return grid


def _render_expected(scene):
    out = _render_frame(scene["h"], scene["w"], scene["side_colors"], bg=scene["bg"])
    rects = scene["rects"]
    chain = rects[: scene["chain_len"]]
    stationary = rects[scene["chain_len"] :]

    for rect in stationary:
        r0, c0 = rect["r0"], rect["c0"]
        out[r0 : r0 + rect["h"], c0 : c0 + rect["w"]] = rect["color"]

    for idx, rect in enumerate(chain):
        if idx + 1 < len(chain):
            next_rect = chain[idx + 1]
            c1 = next_rect["c0"] - 1
        else:
            c1 = scene["w"] - 2
        c0 = c1 - rect["w"] + 1
        r0 = rect["r0"]
        out[r0 : r0 + rect["h"], c0 : c0 + rect["w"]] = rect["color"]
    return out


def _verify_scene(scene):
    inp = _render_input(scene)
    expected = _render_expected(scene)
    assert not np.array_equal(inp, expected)
    rotations = random.randint(0, 3)
    inp = np.rot90(inp, rotations).copy()
    expected = np.rot90(expected, rotations).copy()
    actual = main(inp)
    assert np.array_equal(actual, expected)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        scene = _build_right_scene()
        if scene is None:
            continue
        try:
            return _verify_scene(scene)
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid momentum-relay input")
