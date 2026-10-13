import random
from collections import Counter

import numpy as np


BG = 0
ARROW = 2
FRAME = 5
BLOCK_COLORS = (1, 3, 4, 6, 7, 8, 9)
DIRS = {
    "left": (0, 1),
    "right": (0, -1),
    "top": (1, 0),
    "bottom": (-1, 0),
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(grid: np.ndarray, bg: int):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "coord_set": set(coords),
                    "bbox": _bbox(coords),
                }
            )
    return comps


def _is_frame_component(comp) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0 + 1, r1):
        expected.add((r, c0))
        expected.add((r, c1))
    return comp["coord_set"] == expected


def _arrow_direction(comp):
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    pts = {(r - r0, c - c0) for r, c in comp["coords"]}

    if h == 3 and w >= 3:
        if not all((1, cc) in pts for cc in range(w)):
            return None
        top = [cc for rr, cc in pts if rr == 0]
        bottom = [cc for rr, cc in pts if rr == 2]
        if len(top) != 1 or len(bottom) != 1 or top[0] != bottom[0]:
            return None
        if top[0] == 1:
            return (0, -1)
        if top[0] == w - 2:
            return (0, 1)

    if w == 3 and h >= 3:
        if not all((rr, 1) in pts for rr in range(h)):
            return None
        left = [rr for rr, cc in pts if cc == 0]
        right = [rr for rr, cc in pts if cc == 2]
        if len(left) != 1 or len(right) != 1 or left[0] != right[0]:
            return None
        if left[0] == 1:
            return (-1, 0)
        if left[0] == h - 2:
            return (1, 0)

    return None


def _tail_seed(arrow_bbox, direction):
    r0, c0, r1, c1 = arrow_bbox
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if direction == (0, 1):
        return mid_r, c0 - 1
    if direction == (0, -1):
        return mid_r, c1 + 1
    if direction == (1, 0):
        return r0 - 1, mid_c
    return r1 + 1, mid_c


def _distance_from_frame(block_bbox, frame_bbox, direction):
    br0, bc0, br1, bc1 = block_bbox
    fr0, fc0, fr1, fc1 = frame_bbox
    if direction == (0, 1):
        return fc0 - bc1 - 1
    if direction == (0, -1):
        return bc0 - fc1 - 1
    if direction == (1, 0):
        return fr0 - br1 - 1
    return br0 - fr1 - 1


def _extract_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.flat)).most_common(1)[0][0])
    comps = _components(g, bg)

    frame = None
    frame_index = None
    for idx, comp in enumerate(comps):
        if _is_frame_component(comp):
            if frame is None or len(comp["coords"]) > len(frame["coords"]):
                frame = comp
                frame_index = idx
    if frame is None:
        raise ValueError("frame not found")

    coord_to_comp = {}
    for idx, comp in enumerate(comps):
        for cell in comp["coords"]:
            coord_to_comp[cell] = idx

    pushers = []
    seen_blocks = set()
    for idx, comp in enumerate(comps):
        if idx == frame_index:
            continue
        direction = _arrow_direction(comp)
        if direction is None:
            continue
        seed = _tail_seed(comp["bbox"], direction)
        block_idx = coord_to_comp.get(seed)
        if block_idx is None or block_idx in seen_blocks or block_idx == frame_index:
            continue
        block = comps[block_idx]
        if _arrow_direction(block) is not None:
            continue
        seen_blocks.add(block_idx)
        pushers.append(
            {
                "block": block,
                "arrow": comp,
                "direction": direction,
                "distance": _distance_from_frame(block["bbox"], frame["bbox"], direction),
            }
        )

    return frame, pushers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each colored source rectangle through the central frame along its adjacent arrow, stacking later arrivals on top.

    Input structure:
    - The background is black, with one central hollow rectangular grey border whose interior is empty.
    - Outside the frame sit several solid single-color rectangles, each paired with a nearby red arrow component that points toward the frame.
    - Every arrow is a 3-cell-thick arrowhead/shaft shape, and each source block touches the arrow on its tail side.

    Rule:
    - Keep the grey frame and all red arrows unchanged.
    - Remove every exterior colored block from its original position, then translate it straight along the arrow direction into the frame interior.
    - Each moved block stops when its front edge reaches the opposite inner wall of the frame; if multiple blocks from the same side overlap there, the one that started closer to the frame is painted last and therefore appears on top.

    Key insight:
    - The red shape identifies both the movement direction and the source block, while the block-to-frame distance encodes the stacking order for overlaps from the same side.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    try:
        frame, pushers = _extract_scene(g)
    except ValueError:
        return out

    if not pushers:
        return out

    fr0, fc0, fr1, fc1 = frame["bbox"]
    inner_r0, inner_c0 = fr0 + 1, fc0 + 1
    inner_r1, inner_c1 = fr1 - 1, fc1 - 1

    for spec in pushers:
        for r, c in spec["block"]["coords"]:
            out[r, c] = BG

    grouped = {}
    for spec in pushers:
        grouped.setdefault(spec["direction"], []).append(spec)

    for direction, items in grouped.items():
        items.sort(key=lambda item: item["distance"], reverse=True)
        for spec in items:
            br0, bc0, br1, bc1 = spec["block"]["bbox"]
            if direction == (0, 1):
                dr, dc = 0, inner_c1 - bc1
            elif direction == (0, -1):
                dr, dc = 0, inner_c0 - bc0
            elif direction == (1, 0):
                dr, dc = inner_r1 - br1, 0
            else:
                dr, dc = inner_r0 - br0, 0
            for r, c in spec["block"]["coords"]:
                out[r + dr, c + dc] = spec["block"]["color"]

    return out


def _rects_overlap(a, b) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _relative_output_rect(side: str, pos: int, bh: int, bw: int, ih: int, iw: int):
    if side == "left":
        return (pos, iw - bw, pos + bh - 1, iw - 1)
    if side == "right":
        return (pos, 0, pos + bh - 1, bw - 1)
    if side == "top":
        return (ih - bh, pos, ih - 1, pos + bw - 1)
    return (0, pos, bh - 1, pos + bw - 1)


def _sample_block(side: str, ih: int, iw: int, rng: random.Random, dist_lo: int = 3, dist_hi: int = 5):
    bh = rng.randint(3, min(4, ih))
    bw = rng.randint(3, min(4, iw))
    if side in ("left", "right"):
        pos = rng.randint(0, ih - bh)
    else:
        pos = rng.randint(0, iw - bw)
    dist = rng.randint(dist_lo, dist_hi)
    return {
        "side": side,
        "pos": pos,
        "bh": bh,
        "bw": bw,
        "dist": dist,
        "out_rect": _relative_output_rect(side, pos, bh, bw, ih, iw),
    }


def _assign_attach_offsets(blocks, side: str, rng: random.Random):
    if len(blocks) == 1:
        span = blocks[0]["bh"] if side in ("left", "right") else blocks[0]["bw"]
        blocks[0]["attach"] = rng.randrange(span)
        return True

    assert len(blocks) == 2
    first, second = blocks
    first_span = first["bh"] if side in ("left", "right") else first["bw"]
    second_span = second["bh"] if side in ("left", "right") else second["bw"]
    options = []
    for a0 in range(first_span):
        for a1 in range(second_span):
            center0 = first["pos"] + a0
            center1 = second["pos"] + a1
            if abs(center0 - center1) >= 4:
                options.append((a0, a1))
    if not options:
        return False
    first["attach"], second["attach"] = rng.choice(options)
    return True


def _axis_interval_outside_frame(block, side: str):
    if side == "left":
        return (-block["dist"] - block["bw"], -block["dist"] - 1)
    if side == "right":
        return (block["dist"] + 1, block["dist"] + block["bw"])
    if side == "top":
        return (-block["dist"] - block["bh"], -block["dist"] - 1)
    return (block["dist"] + 1, block["dist"] + block["bh"])


def _sample_side_group(side: str, count: int, overlap: bool, ih: int, iw: int, rng: random.Random):
    for _ in range(200):
        dist_hi = 7 if overlap else 5
        blocks = [_sample_block(side, ih, iw, rng, dist_lo=3, dist_hi=dist_hi) for _ in range(count)]
        if len({block["dist"] for block in blocks}) != count:
            continue
        if count == 2:
            first, second = blocks
            if side in ("left", "right"):
                a0, a1 = first["pos"], first["pos"] + first["bh"] - 1
                b0, b1 = second["pos"], second["pos"] + second["bh"] - 1
            else:
                a0, a1 = first["pos"], first["pos"] + first["bw"] - 1
                b0, b1 = second["pos"], second["pos"] + second["bw"] - 1
            has_overlap = _intervals_overlap(a0, a1, b0, b1)
            if overlap and (not has_overlap or first["out_rect"] == second["out_rect"]):
                continue
            if not overlap and has_overlap:
                continue
            if overlap and _intervals_overlap(*_axis_interval_outside_frame(first, side), *_axis_interval_outside_frame(second, side)):
                continue
        if not _assign_attach_offsets(blocks, side, rng):
            continue
        return blocks
    return None


def _place_cells(grid: np.ndarray, cells, color: int) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BG:
            return False
    for r, c in cells:
        grid[r, c] = color
    return True


def _block_and_arrow_cells(spec, frame_bbox):
    fr0, fc0, fr1, fc1 = frame_bbox
    side = spec["side"]
    pos = spec["pos"]
    bh = spec["bh"]
    bw = spec["bw"]
    dist = spec["dist"]

    if side == "left":
        r0 = fr0 + 1 + pos
        c0 = fc0 - dist - bw
        center_r = r0 + spec["attach"]
        block = [(r, c) for r in range(r0, r0 + bh) for c in range(c0, c0 + bw)]
        arrow = [(center_r, c) for c in range(fc0 - dist, fc0)]
        arrow.extend([(center_r - 1, fc0 - 2), (center_r + 1, fc0 - 2)])
        return block, arrow

    if side == "right":
        r0 = fr0 + 1 + pos
        c0 = fc1 + dist + 1
        center_r = r0 + spec["attach"]
        block = [(r, c) for r in range(r0, r0 + bh) for c in range(c0, c0 + bw)]
        arrow = [(center_r, c) for c in range(fc1 + 1, fc1 + dist + 1)]
        arrow.extend([(center_r - 1, fc1 + 2), (center_r + 1, fc1 + 2)])
        return block, arrow

    if side == "top":
        r0 = fr0 - dist - bh
        c0 = fc0 + 1 + pos
        center_c = c0 + spec["attach"]
        block = [(r, c) for r in range(r0, r0 + bh) for c in range(c0, c0 + bw)]
        arrow = [(r, center_c) for r in range(fr0 - dist, fr0)]
        arrow.extend([(fr0 - 2, center_c - 1), (fr0 - 2, center_c + 1)])
        return block, arrow

    r0 = fr1 + dist + 1
    c0 = fc0 + 1 + pos
    center_c = c0 + spec["attach"]
    block = [(r, c) for r in range(r0, r0 + bh) for c in range(c0, c0 + bw)]
    arrow = [(r, center_c) for r in range(fr1 + 1, fr1 + dist + 1)]
    arrow.extend([(fr1 + 2, center_c - 1), (fr1 + 2, center_c + 1)])
    return block, arrow


def _pairwise_output_ok(specs):
    for i, first in enumerate(specs):
        for second in specs[i + 1 :]:
            if first["side"] == second["side"]:
                continue
            if _rects_overlap(first["out_rect"], second["out_rect"]):
                return False
    return True


def _make_counts(active_sides, want_stack: bool, rng: random.Random, forced_stack_side: str | None = None):
    counts = {side: 1 for side in active_sides}
    stack_side = forced_stack_side if want_stack else None
    if stack_side is not None:
        counts[stack_side] = 2
    else:
        extra = [side for side in active_sides if rng.random() < 0.35]
        for side in extra:
            counts[side] = 2
    while sum(counts.values()) > len(BLOCK_COLORS):
        shrinkable = [side for side, count in counts.items() if count == 2 and side != stack_side]
        if not shrinkable:
            break
        counts[rng.choice(shrinkable)] = 1
    return counts, stack_side


def _gen_attempt(rng: random.Random, required_side: str, want_stack: bool):
    ih = rng.randint(7, 10)
    iw = rng.randint(7, 10)
    frame_h = ih + 2
    frame_w = iw + 2

    active_count = rng.randint(2, 3) if want_stack else rng.randint(2, 4)
    others = [side for side in ["left", "right", "top", "bottom"] if side != required_side]
    active_sides = [required_side] + rng.sample(others, active_count - 1)
    counts, stack_side = _make_counts(active_sides, want_stack, rng, forced_stack_side=required_side if want_stack else None)

    specs = []
    for side in active_sides:
        group = _sample_side_group(side, counts[side], want_stack and side == stack_side, ih, iw, rng)
        if group is None:
            return None
        specs.extend(group)

    if not _pairwise_output_ok(specs):
        return None

    left_margin = max((spec["dist"] + spec["bw"] for spec in specs if spec["side"] == "left"), default=1)
    right_margin = max((spec["dist"] + spec["bw"] for spec in specs if spec["side"] == "right"), default=1)
    top_margin = max((spec["dist"] + spec["bh"] for spec in specs if spec["side"] == "top"), default=1)
    bottom_margin = max((spec["dist"] + spec["bh"] for spec in specs if spec["side"] == "bottom"), default=1)

    base_h = top_margin + frame_h + bottom_margin
    base_w = left_margin + frame_w + right_margin
    if base_h > 30 or base_w > 30:
        return None

    extra_top = rng.randint(0, min(3, 30 - base_h))
    extra_bottom = rng.randint(0, min(3, 30 - base_h - extra_top))
    extra_left = rng.randint(0, min(3, 30 - base_w))
    extra_right = rng.randint(0, min(3, 30 - base_w - extra_left))

    h = base_h + extra_top + extra_bottom
    w = base_w + extra_left + extra_right
    fr0 = top_margin + extra_top
    fc0 = left_margin + extra_left
    fr1 = fr0 + frame_h - 1
    fc1 = fc0 + frame_w - 1

    grid = np.full((h, w), BG, dtype=int)
    grid[fr0, fc0 : fc1 + 1] = FRAME
    grid[fr1, fc0 : fc1 + 1] = FRAME
    grid[fr0 : fr1 + 1, fc0] = FRAME
    grid[fr0 : fr1 + 1, fc1] = FRAME

    colors = list(BLOCK_COLORS)
    rng.shuffle(colors)
    for spec, color in zip(specs, colors):
        spec["color"] = color
        block_cells, arrow_cells = _block_and_arrow_cells(spec, (fr0, fc0, fr1, fc1))
        if not _place_cells(grid, block_cells, color):
            return None
        if not _place_cells(grid, arrow_cells, ARROW):
            return None

    try:
        frame, pushers = _extract_scene(grid)
    except ValueError:
        return None
    if frame["bbox"] != (fr0, fc0, fr1, fc1):
        return None
    if len(pushers) != len(specs):
        return None
    if want_stack:
        stack_specs = [spec for spec in specs if spec["side"] == stack_side]
        if len(stack_specs) != 2 or not _rects_overlap(stack_specs[0]["out_rect"], stack_specs[1]["out_rect"]):
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    required_side = rng.choice(["left", "right", "top", "bottom"])
    want_stack = rng.random() < 0.5
    for _ in range(500):
        grid = _gen_attempt(rng, required_side, want_stack)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to produce a valid sample")
