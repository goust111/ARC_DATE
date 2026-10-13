import random
from collections import Counter

import numpy as np


BG_COLOR = 8
FRAME_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")

MIN_FRAME_SIDE = 5
MAX_GRID_SIDE = 30
GEN_ATTEMPTS = 300
FRAGMENT_TRIES = 200
PLACE_TRIES = 200
MIN_FRAGMENT_CELLS = 4
MAX_FRAGMENT_SPAN = 13


def most_common_color(arr: np.ndarray) -> int:
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(counts)])


def bbox_of(coords) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def normalize_coords(coords):
    r0, c0, _, _ = bbox_of(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def coords_to_mask(coords) -> np.ndarray:
    r0, c0, r1, c1 = bbox_of(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def mask_coords(mask: np.ndarray):
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def connected_components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def thin_frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def tooth_cell(h: int, w: int, side: str | None):
    if side is None:
        return None
    if side == "top":
        return 1, w // 2
    if side == "bottom":
        return h - 2, w // 2
    if side == "left":
        return h // 2, 1
    if side == "right":
        return h // 2, w - 2
    raise ValueError(side)


def colored_shape_mask(h: int, w: int, tooth_side: str | None) -> np.ndarray:
    mask = thin_frame_mask(h, w)
    tooth = tooth_cell(h, w, tooth_side)
    if tooth is not None:
        mask[tooth] = True
    return mask


def is_thin_rect_frame(coords) -> bool:
    mask = coords_to_mask(coords)
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    return np.array_equal(mask, thin_frame_mask(h, w))


def detect_frame_color_and_boxes(grid: np.ndarray, bg: int):
    best = None
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != int(bg)):
        boxes = []
        for comp in connected_components(grid == int(color)):
            if is_thin_rect_frame(comp):
                boxes.append(bbox_of(comp))
        if len(boxes) < 2:
            continue
        boxes.sort(key=lambda b: ((b[2] - b[0] + 1) * (b[3] - b[1] + 1), b[0], b[1]))
        if best is None or len(boxes) > len(best[1]):
            best = (int(color), boxes)
    assert best is not None, "nested grey frames not found"
    return best


def candidate_targets(total_area: int):
    candidates = []
    seen = set()
    for with_tooth in (False, True):
        numer = total_area + (3 if with_tooth else 4)
        if numer % 2 != 0:
            continue
        side_sum = numer // 2
        for h in range(MIN_FRAME_SIDE, side_sum - MIN_FRAME_SIDE + 1, 2):
            w = side_sum - h
            if w < MIN_FRAME_SIDE or w % 2 == 0:
                continue
            if with_tooth:
                for side in SIDES:
                    mask = colored_shape_mask(h, w, side)
                    sig = normalize_coords(mask_coords(mask))
                    if sig not in seen:
                        seen.add(sig)
                        candidates.append(mask)
            else:
                mask = colored_shape_mask(h, w, None)
                sig = normalize_coords(mask_coords(mask))
                if sig not in seen:
                    seen.add(sig)
                    candidates.append(mask)
    candidates.sort(key=lambda m: (m.shape[0] * m.shape[1], m.shape[0], m.shape[1], int(m.sum())))
    return candidates


def target_can_cover(fragments, target: np.ndarray) -> bool:
    target_coords = [(int(r), int(c)) for r, c in np.argwhere(target)]
    index = {coord: i for i, coord in enumerate(target_coords)}
    full_mask = (1 << len(target_coords)) - 1
    positions_by_fragment = []

    for frag in fragments:
        placements = set()
        for tr, tc in target_coords:
            for fr, fc in frag:
                dr = tr - fr
                dc = tc - fc
                bits = 0
                ok = True
                for rr, cc in frag:
                    bit = index.get((rr + dr, cc + dc))
                    if bit is None:
                        ok = False
                        break
                    bits |= 1 << bit
                if ok:
                    placements.add(bits)
        if not placements:
            return False
        positions_by_fragment.append(tuple(sorted(placements)))

    order = sorted(range(len(fragments)), key=lambda i: len(positions_by_fragment[i]))
    memo = set()

    def dfs(k: int, covered: int) -> bool:
        if k == len(order):
            return covered == full_mask
        state = (k, covered)
        if state in memo:
            return False
        idx = order[k]
        for bits in positions_by_fragment[idx]:
            if bits & covered:
                continue
            if dfs(k + 1, covered | bits):
                return True
        memo.add(state)
        return False

    return dfs(0, 0)


def assemble_fragments(fragments) -> np.ndarray:
    total_area = sum(len(frag) for frag in fragments)
    matches = {}
    for target in candidate_targets(total_area):
        if target_can_cover(fragments, target):
            sig = normalize_coords(mask_coords(target))
            matches[sig] = target
    assert matches, "no valid compact assembly found"
    assert len(matches) == 1, "assembly must be unique"
    return next(iter(matches.values())).copy()


def centered_top_left(outer_bbox, mask: np.ndarray):
    h, w = mask.shape
    center2_r = outer_bbox[0] + outer_bbox[2]
    center2_c = outer_bbox[1] + outer_bbox[3]
    top = (center2_r - (h - 1)) // 2
    left = (center2_c - (w - 1)) // 2
    return int(top), int(left)


def draw_mask(canvas: np.ndarray, top: int, left: int, mask: np.ndarray, color: int):
    coords = np.argwhere(mask)
    for rr, cc in coords:
        canvas[top + int(rr), left + int(cc)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group same-color fragments, reassemble each compact shape, then nest them by area into concentric frames.

    Input structure:
    - Background-colored canvas with several colors, each split into 2-3 disconnected fragments.
    - One side contains nested thin rectangular frames of a dedicated frame color.

    Rule:
    - Reassemble each color by translation only into its unique compact rectangular / near-rectangular form.
    - Sort the reassembled shapes by area from small to large.
    - Place them from the innermost frame outward, centered on the common frame center.
    - Remove the scattered source fragments.
    """
    g = np.asarray(grid, dtype=int)
    bg = most_common_color(g)
    frame_color, frame_boxes = detect_frame_color_and_boxes(g, bg)

    out = np.full_like(g, int(bg))
    out[g == int(frame_color)] = int(frame_color)

    objects = []
    for color in sorted(int(x) for x in np.unique(g) if int(x) not in {int(bg), int(frame_color)}):
        comps = connected_components(g == int(color))
        assert 2 <= len(comps) <= 3, "each color must appear in 2-3 fragments"
        fragments = [normalize_coords(comp) for comp in comps]
        assembled = assemble_fragments(fragments)
        objects.append((int(assembled.sum()), int(color), assembled))

    objects.sort(key=lambda item: (item[0], item[1]))
    frame_boxes.sort(key=lambda b: ((b[2] - b[0] + 1) * (b[3] - b[1] + 1), b[0], b[1]))
    assert len(objects) == len(frame_boxes), "frame count must match number of colors"

    for (_, color, mask), box in zip(objects, frame_boxes):
        top, left = centered_top_left(box, mask)
        draw_mask(out, top, left, mask, color)

    return out


def rand_odd(rng: random.Random, lo: int, hi: int) -> int | None:
    if lo % 2 == 0:
        lo += 1
    if hi % 2 == 0:
        hi -= 1
    if lo > hi:
        return None
    choices = list(range(lo, hi + 1, 2))
    return int(rng.choice(choices))


def frame_order(h: int, w: int, tooth_side: str | None):
    order = []
    for c in range(w):
        order.append((0, c))
    for r in range(1, h - 1):
        order.append((r, w - 1))
    for c in range(w - 1, -1, -1):
        order.append((h - 1, c))
    for r in range(h - 2, 0, -1):
        order.append((r, 0))

    forbidden_cut = None
    if tooth_side is not None:
        if tooth_side == "top":
            anchor = (0, w // 2)
            tooth = (1, w // 2)
        elif tooth_side == "bottom":
            anchor = (h - 1, w // 2)
            tooth = (h - 2, w // 2)
        elif tooth_side == "left":
            anchor = (h // 2, 0)
            tooth = (h // 2, 1)
        else:
            anchor = (h // 2, w - 1)
            tooth = (h // 2, w - 2)
        idx = order.index(anchor)
        order = order[: idx + 1] + [tooth] + order[idx + 1 :]
        forbidden_cut = idx + 1

    return order, forbidden_cut


def cyclic_segment(order, start: int, end: int):
    if end > start:
        return order[start:end]
    return order[start:] + order[:end]


def random_fragments_from_shape(mask: np.ndarray, tooth_side: str | None, rng: random.Random):
    h, w = mask.shape
    order, forbidden_cut = frame_order(h, w, tooth_side)
    n_cells = len(order)
    fragment_count = 2 if (n_cells <= 20 or rng.random() < 0.5) else 3
    fragment_count = 3 if max(h, w) > 11 else fragment_count
    min_len = MIN_FRAGMENT_CELLS
    allowed_cuts = [i for i in range(n_cells) if i != forbidden_cut]
    if len(allowed_cuts) < fragment_count:
        return None

    for _ in range(FRAGMENT_TRIES):
        cuts = sorted(rng.sample(allowed_cuts, fragment_count))
        lengths = []
        segments = []
        ok = True
        for idx, cut in enumerate(cuts):
            nxt = cuts[(idx + 1) % fragment_count]
            seg = cyclic_segment(order, cut, nxt)
            segments.append(seg)
            lengths.append(len(seg))
            if len(seg) < min_len:
                ok = False
                break
        if not ok:
            continue

        fragments = [coords_to_mask(seg) for seg in segments]
        if any(int(frag.sum()) < MIN_FRAGMENT_CELLS for frag in fragments):
            continue
        if any(max(frag.shape) > MAX_FRAGMENT_SPAN for frag in fragments):
            continue
        rels = [normalize_coords(mask_coords(frag)) for frag in fragments]
        try:
            rebuilt = assemble_fragments(rels)
        except AssertionError:
            continue
        if np.array_equal(rebuilt, mask):
            return fragments
    return None


def expand_blocked(blocked: np.ndarray, coords):
    h, w = blocked.shape
    for r, c in coords:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def can_place(mask: np.ndarray, top: int, left: int, blocked: np.ndarray) -> bool:
    h, w = blocked.shape
    mh, mw = mask.shape
    if top < 0 or left < 0 or top + mh > h or left + mw > w:
        return False
    for rr, cc in np.argwhere(mask):
        if blocked[top + int(rr), left + int(cc)]:
            return False
    return True


def place_fragment(
    grid: np.ndarray,
    blocked: np.ndarray,
    mask: np.ndarray,
    color: int,
    forbidden_box,
    zone,
    rng: random.Random,
):
    mh, mw = mask.shape
    candidates = []
    zr0, zc0, zr1, zc1 = zone
    if zr1 - zr0 + 1 < mh or zc1 - zc0 + 1 < mw:
        return False
    for top in range(zr0, zr1 - mh + 2):
        for left in range(zc0, zc1 - mw + 2):
            r0, c0, r1, c1 = forbidden_box
            if not (top + mh - 1 < r0 or top > r1 or left + mw - 1 < c0 or left > c1):
                continue
            if can_place(mask, top, left, blocked):
                candidates.append((top, left))
    if not candidates:
        return False
    top, left = rng.choice(candidates)
    coords = []
    for rr, cc in np.argwhere(mask):
        r = top + int(rr)
        c = left + int(cc)
        grid[r, c] = int(color)
        coords.append((r, c))
    expand_blocked(blocked, coords)
    return True


def build_specs(rng: random.Random, color_count: int):
    specs = []
    prev_grey_h = None
    prev_grey_w = None
    for idx in range(color_count):
        if idx == 0:
            base_hi = 7 if color_count == 3 else 9
            h = rand_odd(rng, 5, base_hi)
            w = rand_odd(rng, 5, base_hi + 2)
        else:
            gap_in = 1 if (color_count == 3 and rng.random() < 0.7) else rng.randint(1, 2)
            min_h = prev_grey_h + 2 + 2 * gap_in
            min_w = prev_grey_w + 2 + 2 * gap_in
            h = rand_odd(rng, min_h, min_h + 2)
            w = rand_odd(rng, min_w, min_w + 2)
        if h is None or w is None:
            return None
        force_tooth = max(h, w) >= 13 or h == w
        tooth_side = rng.choice(SIDES) if (force_tooth or rng.random() < 0.5) else None
        mask = colored_shape_mask(h, w, tooth_side)
        gap_out = 1
        grey_h = h + 2 * gap_out
        grey_w = w + 2 * gap_out
        if grey_h > 19 or grey_w > 19:
            return None
        specs.append(
            {
                "color": None,
                "mask": mask,
                "tooth_side": tooth_side,
                "grey_h": grey_h,
                "grey_w": grey_w,
            }
        )
        prev_grey_h = grey_h
        prev_grey_w = grey_w
    return specs


def choose_grid_and_outer_box(outer_h: int, outer_w: int, max_frag_h: int, max_frag_w: int, rng: random.Random):
    options = []
    min_h_lr = max(outer_h + 2, 27)
    min_w_lr = max(outer_w + max_frag_w + 2, 27)
    if min_h_lr <= MAX_GRID_SIDE and min_w_lr <= MAX_GRID_SIDE:
        for side in ("left", "right"):
            options.append((side, min_h_lr, min_w_lr))

    min_h_tb = max(outer_h + max_frag_h + 2, 27)
    min_w_tb = max(outer_w + 2, 27)
    if min_h_tb <= MAX_GRID_SIDE and min_w_tb <= MAX_GRID_SIDE:
        for side in ("top", "bottom"):
            options.append((side, min_h_tb, min_w_tb))

    if not options:
        return None

    side, min_h, min_w = rng.choice(options)
    grid_h = rng.randint(min_h, MAX_GRID_SIDE)
    grid_w = rng.randint(min_w, MAX_GRID_SIDE)
    near_choices = [0, 1]

    if side == "left":
        left = rng.choice(near_choices)
        top = rng.choice([rng.choice(near_choices), grid_h - outer_h - rng.choice(near_choices)])
    elif side == "right":
        left = grid_w - outer_w - rng.choice(near_choices)
        top = rng.choice([rng.choice(near_choices), grid_h - outer_h - rng.choice(near_choices)])
    elif side == "top":
        top = rng.choice(near_choices)
        left = rng.choice([rng.choice(near_choices), grid_w - outer_w - rng.choice(near_choices)])
    else:
        top = grid_h - outer_h - rng.choice(near_choices)
        left = rng.choice([rng.choice(near_choices), grid_w - outer_w - rng.choice(near_choices)])
    return grid_h, grid_w, top, left


def centered_boxes_for_specs(specs, outer_top: int, outer_left: int):
    outer_h = specs[-1]["grey_h"]
    outer_w = specs[-1]["grey_w"]
    center2_r = 2 * outer_top + outer_h - 1
    center2_c = 2 * outer_left + outer_w - 1
    boxes = []
    for spec in specs:
        gh = spec["grey_h"]
        gw = spec["grey_w"]
        top = (center2_r - (gh - 1)) // 2
        left = (center2_c - (gw - 1)) // 2
        boxes.append((int(top), int(left), int(top + gh - 1), int(left + gw - 1)))
    return boxes


def draw_frame(canvas: np.ndarray, box, color: int):
    r0, c0, r1, c1 = box
    canvas[r0, c0 : c1 + 1] = int(color)
    canvas[r1, c0 : c1 + 1] = int(color)
    canvas[r0 : r1 + 1, c0] = int(color)
    canvas[r0 : r1 + 1, c1] = int(color)


def render_expected(grid_shape, specs, boxes):
    out = np.full(grid_shape, BG_COLOR, dtype=int)
    for box in boxes:
        draw_frame(out, box, FRAME_COLOR)
    for spec, box in zip(specs, boxes):
        top, left = centered_top_left(box, spec["mask"])
        shape_coords = []
        for rr, cc in np.argwhere(spec["mask"]):
            r = top + int(rr)
            c = left + int(cc)
            shape_coords.append((r, c))
            if out[r, c] != BG_COLOR:
                return None
        for r, c in shape_coords:
            out[r, c] = int(spec["color"])
    return out


def _gen_attempt(rng: random.Random):
    color_count = rng.choice((2, 3))
    specs = build_specs(rng, color_count)
    if specs is None:
        return None

    colors = list(rng.sample(PALETTE, color_count))
    for spec, color in zip(specs, colors):
        spec["color"] = int(color)

    fragments_by_spec = []
    max_frag_h = 0
    max_frag_w = 0
    for spec in specs:
        fragments = random_fragments_from_shape(spec["mask"], spec["tooth_side"], rng)
        if fragments is None:
            return None
        for frag in fragments:
            max_frag_h = max(max_frag_h, frag.shape[0])
            max_frag_w = max(max_frag_w, frag.shape[1])
        fragments_by_spec.append(fragments)

    outer_h = specs[-1]["grey_h"]
    outer_w = specs[-1]["grey_w"]
    placed = choose_grid_and_outer_box(outer_h, outer_w, max_frag_h, max_frag_w, rng)
    if placed is None:
        return None
    grid_h, grid_w, outer_top, outer_left = placed
    boxes = centered_boxes_for_specs(specs, outer_top, outer_left)
    forbidden_box = boxes[-1]
    zone = (0, 0, grid_h - 1, grid_w - 1)

    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    for box in boxes:
        draw_frame(grid, box, FRAME_COLOR)
        coords = []
        r0, c0, r1, c1 = box
        for c in range(c0, c1 + 1):
            coords.append((r0, c))
            coords.append((r1, c))
        for r in range(r0 + 1, r1):
            coords.append((r, c0))
            coords.append((r, c1))
        expand_blocked(blocked, coords)

    all_frags = []
    for spec, fragments in zip(specs, fragments_by_spec):
        for frag in fragments:
            all_frags.append((frag, spec["color"]))
    rng.shuffle(all_frags)
    all_frags.sort(key=lambda item: (item[0].shape[0] * item[0].shape[1], max(item[0].shape)), reverse=True)

    for frag, color in all_frags:
        if not place_fragment(grid, blocked, frag, color, forbidden_box, zone, rng):
            return None

    expected = render_expected(grid.shape, specs, boxes)
    if expected is None:
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
