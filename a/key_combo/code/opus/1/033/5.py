import random

import numpy as np


BG = 0
FRAME = 1
TEMPLATE = 3
MARKER = 7

HOLE_TO_COLOR = {1: 2, 2: 4, 3: 6}
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _count_enclosed_regions(subgrid: np.ndarray, frame_color: int = FRAME) -> int:
    open_mask = subgrid != int(frame_color)
    holes = 0
    for comp in _mask_components(open_mask):
        if not _touches_edge(comp, subgrid.shape):
            holes += 1
    return holes


def _find_template_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    pattern = np.array(
        [
            [BG, TEMPLATE, BG],
            [TEMPLATE, BG, TEMPLATE],
            [BG, TEMPLATE, BG],
        ],
        dtype=int,
    )
    matches: list[tuple[int, int]] = []
    h, w = grid.shape
    for r in range(h - 2):
        for c in range(w - 2):
            if np.array_equal(grid[r : r + 3, c : c + 3], pattern):
                matches.append((r, c))
    assert len(matches) == 1, "expected exactly one 3x3 hollow-plus template"
    r0, c0 = matches[0]
    return [(r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 2), (r0 + 2, c0 + 1)]


def _is_rectangle_component(grid: np.ndarray, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 7 or w < 7:
        return False

    submask = np.zeros((h, w), dtype=bool)
    submask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if not (submask[0, 0] and submask[0, -1] and submask[-1, 0] and submask[-1, -1]):
        return False

    perim = np.zeros_like(submask, dtype=bool)
    perim[0, :] = True
    perim[-1, :] = True
    perim[:, 0] = True
    perim[:, -1] = True
    occupied_perim = int(np.count_nonzero(submask & perim))
    perim_len = int(np.count_nonzero(perim))
    missing = perim_len - occupied_perim
    return missing in (0, 1)


def _find_rectangles(grid: np.ndarray) -> list[dict]:
    rectangles: list[dict] = []
    for comp in _mask_components(grid == FRAME):
        if not _is_rectangle_component(grid, comp):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        rectangles.append(
            {
                "bbox": (r0, c0, r1, c1),
                "coords": comp,
            }
        )
    rectangles.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    assert 2 <= len(rectangles) <= 4, "expected 2 to 4 frame rectangles"
    return rectangles


def _stamp_rectangle(
    out: np.ndarray,
    source: np.ndarray,
    bbox: tuple[int, int, int, int],
    holes: int,
) -> None:
    if holes == 0:
        return
    color = HOLE_TO_COLOR[holes]
    r0, c0, r1, c1 = bbox
    sub = source[r0 : r1 + 1, c0 : c1 + 1]
    marker_rows, marker_cols = np.where(sub == MARKER)
    for mr0, mc0 in zip(marker_rows, marker_cols):
        mr = int(mr0)
        mc = int(mc0)
        for dr, dc in DIRS4:
            for step in range(1, holes + 1):
                rr = mr + dr * step
                cc = mc + dc * step
                if not (0 <= rr < sub.shape[0] and 0 <= cc < sub.shape[1]):
                    break
                if int(sub[rr, cc]) == FRAME:
                    break
                gr = r0 + rr
                gc = c0 + cc
                if int(out[gr, gc]) == BG:
                    out[gr, gc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each rectangle's enclosed regions, then stamp scaled colored crosses on its markers.

    Input structure:
    - The background is black.
    - There are 2 to 4 blue rectangular frame components, sometimes with one outer gap and
      sometimes with internal blue splitter walls that create multiple enclosed black regions.
    - Each rectangle contains 1 or 2 orange marker pixels, and there is one separate green
      3x3 hollow-plus template whose center and corners are black.

    Rule:
    - Identify every blue rectangle by its bounding box and count how many 4-connected
      non-blue regions inside that box are enclosed, meaning they do not touch the box edge.
    - For hole count H, use stamp color red/yellow/magenta for H=1/2/3; rectangles with
      H=0 keep their markers but receive no stamp.
    - Around every orange marker in a rectangle, stamp a hollow plus with arm length H:
      along the four cardinal directions, paint black cells up to H steps away, truncating
      an arm immediately before the first blue frame pixel it would hit.
    - Preserve the blue frames and orange markers, and erase the separate green template.

    Key insight:
    - The same outside template is only a geometric cue: each rectangle derives both the
      stamp size and the stamp color from its own enclosed-region count, not from the marker
      location or from a fixed template size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for r, c in _find_template_cells(g):
        out[r, c] = BG

    for rect in _find_rectangles(g):
        bbox = rect["bbox"]
        r0, c0, r1, c1 = bbox
        holes = _count_enclosed_regions(g[r0 : r1 + 1, c0 : c1 + 1], FRAME)
        assert 0 <= holes <= 3, "hole count must stay within the puzzle range"
        _stamp_rectangle(out, g, bbox, holes)

    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 400
GEN_CANVAS = 30
GEN_MIN_SIZE = 20
COUNT_PATTERNS = [
    (0, 1, 2, 3),
    (1, 2, 3),
    (0, 1, 2),
    (0, 2, 3),
    (0, 1, 3),
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 2),
    (1, 3),
]
_COUNT_BAG: list[tuple[int, ...]] = []


def _next_count_pattern(rng: random.Random) -> tuple[int, ...]:
    global _COUNT_BAG
    if not _COUNT_BAG:
        _COUNT_BAG = COUNT_PATTERNS[:]
        rng.shuffle(_COUNT_BAG)
    return _COUNT_BAG.pop()


def _random_composition(total: int, parts: int, rng: random.Random) -> list[int]:
    assert parts >= 1 and total >= parts
    if parts == 1:
        return [total]
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    vals: list[int] = []
    prev = 0
    for cut in cuts + [total]:
        vals.append(cut - prev)
        prev = cut
    return vals


def _candidate_truncates(local: np.ndarray, cell: tuple[int, int], holes: int) -> bool:
    r, c = cell
    for dr, dc in DIRS4:
        for step in range(1, holes + 1):
            rr = r + dr * step
            cc = c + dc * step
            if int(local[rr, cc]) == FRAME:
                return True
    return False


def _choose_markers(local: np.ndarray, holes: int, rng: random.Random) -> list[tuple[int, int]] | None:
    h, w = local.shape
    open_comps = _mask_components(local != FRAME)
    touches_edge: dict[tuple[int, int], bool] = {}
    for comp in open_comps:
        edge = _touches_edge(comp, local.shape)
        for r0, c0 in comp:
            touches_edge[(int(r0), int(c0))] = edge

    interior_open = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if int(local[r, c]) == BG
    ]
    if holes == 0:
        safe = interior_open
    else:
        safe = [
            (r, c)
            for r, c in interior_open
            if min(r, h - 1 - r, c, w - 1 - c) >= holes + 1
        ]
    if not safe:
        return None

    want_two = len(safe) >= 2 and rng.random() < 0.6
    target_count = 2 if want_two else 1
    truncated = [cell for cell in safe if holes > 0 and _candidate_truncates(local, cell, holes)]
    full = [cell for cell in safe if cell not in truncated]
    edge_cells = [cell for cell in safe if touches_edge.get(cell, False)]
    inner_cells = [cell for cell in safe if not touches_edge.get(cell, False)]

    def far_enough(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= 2

    chosen: list[tuple[int, int]] = []
    if holes > 0 and truncated:
        chosen.append(rng.choice(truncated if rng.random() < 0.8 else safe))
    else:
        chosen.append(rng.choice(safe))

    if target_count == 2:
        preferred: list[tuple[int, int]] = []
        first = chosen[0]
        if edge_cells and inner_cells:
            contrast_pool = [
                cell
                for cell in safe
                if touches_edge.get(cell, False) != touches_edge.get(first, False) and far_enough(cell, first)
            ]
            preferred.extend(contrast_pool)
        if holes > 0 and truncated and full:
            if first in truncated:
                preferred.extend([cell for cell in full if far_enough(cell, first)])
            else:
                preferred.extend([cell for cell in truncated if far_enough(cell, first)])
        if not preferred:
            preferred = [cell for cell in safe if cell != first and far_enough(cell, first)]
        if preferred:
            chosen.append(rng.choice(preferred))

    return chosen


def _build_rectangle(holes: int, rng: random.Random) -> dict | None:
    for _ in range(200):
        open_family = holes == 0 or (holes > 0 and rng.random() < 0.5)
        wall_count = holes if open_family else max(0, holes - 1)
        orientation = rng.choice(["vertical", "horizontal"])

        min_side = max(7, 2 * max(holes, 1) + 3, 2 * wall_count + 3)
        max_side = min(12, min_side + 3)
        if min_side > max_side:
            continue
        h = rng.randint(min_side, max_side)
        w = rng.randint(min_side, max_side)

        local = np.zeros((h, w), dtype=int)
        local[0, :] = FRAME
        local[-1, :] = FRAME
        local[:, 0] = FRAME
        local[:, -1] = FRAME

        if orientation == "vertical":
            free_cols = w - 2 - wall_count
            if free_cols < wall_count + 1:
                continue
            widths = _random_composition(free_cols, wall_count + 1, rng)
            cursor = 1
            for width in widths[:-1]:
                cursor += width
                local[1:-1, cursor] = FRAME
                cursor += 1
            if open_family:
                gap_side = rng.choice([0, w - 1])
                gap_row = rng.randint(1, h - 2)
                local[gap_row, gap_side] = BG
        else:
            free_rows = h - 2 - wall_count
            if free_rows < wall_count + 1:
                continue
            heights = _random_composition(free_rows, wall_count + 1, rng)
            cursor = 1
            for height in heights[:-1]:
                cursor += height
                local[cursor, 1:-1] = FRAME
                cursor += 1
            if open_family:
                gap_side = rng.choice([0, h - 1])
                gap_col = rng.randint(1, w - 2)
                local[gap_side, gap_col] = BG

        if _count_enclosed_regions(local, FRAME) != holes:
            continue

        markers = _choose_markers(local, holes, rng)
        if not markers:
            continue
        for r, c in markers:
            local[r, c] = MARKER

        return {
            "grid": local,
            "holes": holes,
            "markers": markers,
        }
    return None


def _template_sprite() -> np.ndarray:
    sprite = np.zeros((3, 3), dtype=int)
    sprite[0, 1] = TEMPLATE
    sprite[1, 0] = TEMPLATE
    sprite[1, 2] = TEMPLATE
    sprite[2, 1] = TEMPLATE
    return sprite


def _can_place(reserved: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(reserved.shape[0], top + h + pad)
    c1 = min(reserved.shape[1], left + w + pad)
    return not np.any(reserved[r0:r1, c0:c1])


def _paint_sprite(dst: np.ndarray, sprite: np.ndarray, top: int, left: int) -> None:
    h, w = sprite.shape
    sub = dst[top : top + h, left : left + w]
    mask = sprite != BG
    sub[mask] = sprite[mask]


def _place_objects(rectangles: list[dict], rng: random.Random) -> tuple[np.ndarray, list[dict]] | None:
    canvas = np.zeros((GEN_CANVAS, GEN_CANVAS), dtype=int)
    reserved = np.zeros((GEN_CANVAS, GEN_CANVAS), dtype=bool)
    placed_rects: list[dict] = []

    order = sorted(range(len(rectangles)), key=lambda idx: rectangles[idx]["grid"].size, reverse=True)
    for idx in order:
        rect = rectangles[idx]
        sprite = rect["grid"]
        h, w = sprite.shape
        placed = False
        for _ in range(250):
            top = rng.randint(0, GEN_CANVAS - h)
            left = rng.randint(0, GEN_CANVAS - w)
            if not _can_place(reserved, top, left, h, w, pad=1):
                continue
            _paint_sprite(canvas, sprite, top, left)
            reserved[top : top + h, left : left + w] = True
            placed_rects.append(
                {
                    "top": top,
                    "left": left,
                    "grid": sprite,
                    "holes": rect["holes"],
                    "markers": rect["markers"],
                }
            )
            placed = True
            break
        if not placed:
            return None

    template = _template_sprite()
    th, tw = template.shape
    template_pos: tuple[int, int] | None = None
    for _ in range(250):
        top = rng.randint(0, GEN_CANVAS - th)
        left = rng.randint(0, GEN_CANVAS - tw)
        if not _can_place(reserved, top, left, th, tw, pad=1):
            continue
        _paint_sprite(canvas, template, top, left)
        template_pos = (top, left)
        break
    if template_pos is None:
        return None

    placed_rects.sort(key=lambda item: (item["top"], item["left"]))
    return canvas, placed_rects + [{"top": template_pos[0], "left": template_pos[1], "grid": template, "kind": "template"}]


def _expected_output(inp: np.ndarray, placements: list[dict]) -> np.ndarray:
    out = inp.copy()
    template = next(item for item in placements if item.get("kind") == "template")
    ttop = template["top"]
    tleft = template["left"]
    for dr, dc in ((0, 1), (1, 0), (1, 2), (2, 1)):
        out[ttop + dr, tleft + dc] = BG

    for rect in placements:
        if rect.get("kind") == "template":
            continue
        holes = int(rect["holes"])
        if holes == 0:
            continue
        color = HOLE_TO_COLOR[holes]
        top = int(rect["top"])
        left = int(rect["left"])
        local = rect["grid"]
        for mr, mc in rect["markers"]:
            for dr, dc in DIRS4:
                for step in range(1, holes + 1):
                    rr = mr + dr * step
                    cc = mc + dc * step
                    if int(local[rr, cc]) == FRAME:
                        break
                    gr = top + rr
                    gc = left + cc
                    if int(out[gr, gc]) == BG:
                        out[gr, gc] = color
    return out


def _crop_pair(inp: np.ndarray, out: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    occ_h = int(r1 - r0 + 1)
    occ_w = int(c1 - c0 + 1)

    target_h = rng.randint(max(occ_h, GEN_MIN_SIZE), GEN_CANVAS)
    target_w = rng.randint(max(occ_w, GEN_MIN_SIZE), GEN_CANVAS)

    top_lo = max(0, int(r1) - target_h + 1)
    top_hi = min(int(r0), GEN_CANVAS - target_h)
    left_lo = max(0, int(c1) - target_w + 1)
    left_hi = min(int(c0), GEN_CANVAS - target_w)

    top = rng.randint(top_lo, top_hi)
    left = rng.randint(left_lo, left_hi)
    return (
        inp[top : top + target_h, left : left + target_w],
        out[top : top + target_h, left : left + target_w],
    )


def _generate_example() -> tuple[np.ndarray, np.ndarray]:
    rng = random
    holes_list = list(_next_count_pattern(rng))
    rng.shuffle(holes_list)

    rectangles: list[dict] = []
    for holes in holes_list:
        rect = _build_rectangle(holes, rng)
        if rect is None:
            raise RuntimeError("failed to build rectangle")
        rectangles.append(rect)

    placed = _place_objects(rectangles, rng)
    if placed is None:
        raise RuntimeError("failed to place objects")
    inp_full, placements = placed
    out_full = _expected_output(inp_full, placements)
    return _crop_pair(inp_full, out_full, rng)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            inp, expected = _generate_example()
        except RuntimeError:
            continue
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 5")
