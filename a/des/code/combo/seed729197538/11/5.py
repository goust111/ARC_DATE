import math
import random
from collections import deque

import numpy as np


AZURE = 8
GREY = 5

ROT_MARKER_TO_STEPS = {
    2: 0,  # red
    3: 1,  # green: 90 clockwise
    4: 2,  # yellow: 180
    6: 3,  # magenta: 270 clockwise
}
STEPS_TO_ROT_MARKER = {v: k for k, v in ROT_MARKER_TO_STEPS.items()}

PIECE_COLORS = [0, 1, 7, 9]

GEN_ATTEMPTS = 300
SLOT_COUNT_RANGE = (2, 4)
PIECE_DIM_RANGE = (2, 6)
BLUEPRINT_INNER_RANGE = (11, 18)
CANVAS_H_RANGE = (16, 26)
RIGHT_GAP_RANGE = (2, 4)
RIGHT_MARGIN_RANGE = (1, 3)
TOP_MARGIN_RANGE = (0, 3)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _rotate_clockwise(arr: np.ndarray, steps: int) -> np.ndarray:
    return np.rot90(arr, -int(steps % 4))


def _rotated_dims(h: int, w: int, steps: int) -> tuple[int, int]:
    return (h, w) if steps % 2 == 0 else (w, h)


def _find_blueprint_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(grid == GREY)]
    assert coords, "expected a grey blueprint"
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    blueprint = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert blueprint.shape[0] >= 5 and blueprint.shape[1] >= 5
    assert np.all(blueprint[0, :] == GREY)
    assert np.all(blueprint[-1, :] == GREY)
    assert np.all(blueprint[:, 0] == GREY)
    assert np.all(blueprint[:, -1] == GREY)
    return r0, c0, r1, c1


def _extract_slots(blueprint: np.ndarray) -> list[dict]:
    interior = blueprint[1:-1, 1:-1]
    comps = _components(interior != GREY)
    slots: list[dict] = []

    for comp in comps:
        r0, c0, r1, c1 = _bbox_from_coords(comp)
        sub = interior[r0 : r1 + 1, c0 : c1 + 1]
        if len(comp) != sub.size:
            continue
        marker_cells = [(r, c) for r, c in np.argwhere(np.isin(sub, list(ROT_MARKER_TO_STEPS)))]
        if len(marker_cells) != 1:
            continue
        if not np.all((sub == AZURE) | np.isin(sub, list(ROT_MARKER_TO_STEPS))):
            continue

        mr, mc = map(int, marker_cells[0])
        h, w = map(int, sub.shape)
        corner_offsets = {
            (0, 0): "tl",
            (0, w - 1): "tr",
            (h - 1, 0): "bl",
            (h - 1, w - 1): "br",
        }
        assert (mr, mc) in corner_offsets, "slot marker must sit on a slot corner"
        marker_color = int(sub[mr, mc])
        slots.append(
            {
                "top": int(r0 + 1),
                "left": int(c0 + 1),
                "h": h,
                "w": w,
                "rot": int(ROT_MARKER_TO_STEPS[marker_color]),
            }
        )

    assert slots, "expected at least one rectangular slot"
    slots.sort(key=lambda s: (s["top"], s["left"]))
    return slots


def _extract_pieces(grid: np.ndarray, blueprint_bbox: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = blueprint_bbox
    outside = np.array(grid, copy=True)
    outside[r0 : r1 + 1, c0 : c1 + 1] = AZURE

    pieces: list[dict] = []
    for comp in _components(outside != AZURE):
        br0, bc0, br1, bc1 = _bbox_from_coords(comp)
        patch = outside[br0 : br1 + 1, bc0 : bc1 + 1]
        mask = patch != AZURE
        colors = set(map(int, patch[mask]))
        assert len(colors) == 1, "each scattered piece must be monochrome"
        pieces.append(
            {
                "patch": patch.copy(),
                "mask": mask.copy(),
                "dims": tuple(map(int, patch.shape)),
                "color": int(next(iter(colors))),
            }
        )

    assert pieces, "expected scattered pieces on the right"
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each scattered piece according to a slot marker and insert it into the
    matching grey-bordered rectangular slot. The output is the cropped blueprint.
    """
    g = np.asarray(grid, dtype=int)
    blueprint_bbox = _find_blueprint_bbox(g)
    br0, bc0, br1, bc1 = blueprint_bbox
    blueprint = g[br0 : br1 + 1, bc0 : bc1 + 1].copy()

    slots = _extract_slots(blueprint)
    pieces = _extract_pieces(g, blueprint_bbox)
    assert len(pieces) == len(slots), "slot count and piece count must match"

    unused = set(range(len(pieces)))
    for slot in slots:
        matches = []
        for idx in unused:
            piece = pieces[idx]
            rotated_patch = _rotate_clockwise(piece["patch"], slot["rot"])
            if tuple(map(int, rotated_patch.shape)) != (slot["h"], slot["w"]):
                continue
            rotated_mask = _rotate_clockwise(piece["mask"], slot["rot"])
            matches.append((idx, rotated_patch, rotated_mask))

        assert len(matches) == 1, "each slot must match exactly one piece after rotation"
        idx, rotated_patch, rotated_mask = matches[0]
        unused.remove(idx)

        top = int(slot["top"])
        left = int(slot["left"])
        region = blueprint[top : top + slot["h"], left : left + slot["w"]]
        region[rotated_mask] = rotated_patch[rotated_mask]

    assert not unused, "all pieces must be consumed"
    return blueprint


def _manhattan_path(a: tuple[int, int], b: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = a
    tr, tc = b
    path = {(r, c)}
    if random.random() < 0.5:
        while r != tr:
            r += 1 if tr > r else -1
            path.add((r, c))
        while c != tc:
            c += 1 if tc > c else -1
            path.add((r, c))
    else:
        while c != tc:
            c += 1 if tc > c else -1
            path.add((r, c))
        while r != tr:
            r += 1 if tr > r else -1
            path.add((r, c))
    return path


def _random_piece_mask(h: int, w: int) -> np.ndarray:
    for _ in range(80):
        mask = np.zeros((h, w), dtype=bool)
        center = (random.randrange(h), random.randrange(w))
        endpoints = [
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        ]
        cells = {center}
        for end in endpoints:
            cells |= _manhattan_path(end, center)

        min_area = len(cells)
        max_area = h * w
        if min_area > max_area:
            continue

        if max_area == min_area:
            target_area = max_area
        else:
            target_area = random.randint(min_area, max_area)
            if h * w >= 6 and target_area == h * w and random.random() < 0.8:
                target_area -= 1

        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        while len(cells) < target_area and frontier:
            nr, nc = random.choice(tuple(frontier))
            frontier.remove((nr, nc))
            cells.add((nr, nc))
            for ar, ac in _neighbors4(nr, nc):
                if 0 <= ar < h and 0 <= ac < w and (ar, ac) not in cells:
                    frontier.add((ar, ac))

        for r, c in cells:
            mask[r, c] = True

        if not mask[0, :].any():
            continue
        if not mask[-1, :].any():
            continue
        if not mask[:, 0].any():
            continue
        if not mask[:, -1].any():
            continue
        if len(_components(mask)) != 1:
            continue
        return mask

    return np.ones((h, w), dtype=bool)


def _render_piece(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, AZURE, dtype=int)
    patch[mask] = int(color)
    return patch


def _can_place_rect(occupied: np.ndarray, top: int, left: int, h: int, w: int, padding: int) -> bool:
    oh, ow = occupied.shape
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(oh, top + h + padding)
    c1 = min(ow, left + w + padding)
    return not occupied[r0:r1, c0:c1].any()


def _place_rectangles(height: int, width: int, rects: list[tuple[int, int]], padding: int) -> list[tuple[int, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(rects)
    order = sorted(range(len(rects)), key=lambda i: rects[i][0] * rects[i][1], reverse=True)

    for idx in order:
        rh, rw = rects[idx]
        candidates = [
            (r, c)
            for r in range(height - rh + 1)
            for c in range(width - rw + 1)
            if _can_place_rect(occupied, r, c, rh, rw, padding)
        ]
        if not candidates:
            return None
        top, left = random.choice(candidates)
        placements[idx] = (top, left)
        occupied[top : top + rh, left : left + rw] = True

    return [(int(r), int(c)) for r, c in placements]


def _choose_piece_dims(n_slots: int) -> list[tuple[int, int]]:
    dims_pool = [
        (h, w)
        for h in range(PIECE_DIM_RANGE[0], PIECE_DIM_RANGE[1] + 1)
        for w in range(PIECE_DIM_RANGE[0], PIECE_DIM_RANGE[1] + 1)
    ]
    random.shuffle(dims_pool)
    return dims_pool[:n_slots]


def _choose_rotations(n_slots: int) -> list[int]:
    for _ in range(40):
        rots = [random.randrange(4) for _ in range(n_slots)]
        if len(set(rots)) >= min(2, n_slots):
            return rots
    return [random.randrange(4) for _ in range(n_slots)]


def _build_blueprint(slot_specs: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    slot_rects = [(spec["slot_h"] + 2, spec["slot_w"] + 2) for spec in slot_specs]
    min_h = max(BLUEPRINT_INNER_RANGE[0], max(h for h, _ in slot_rects) + 2)
    min_w = max(BLUEPRINT_INNER_RANGE[0], max(w for _, w in slot_rects) + 2)
    max_h = BLUEPRINT_INNER_RANGE[1]
    max_w = BLUEPRINT_INNER_RANGE[1]
    if min_h > max_h or min_w > max_w:
        return None

    total_area = sum(h * w for h, w in slot_rects)
    for _ in range(60):
        inner_h = random.randint(min_h, max_h)
        inner_w = random.randint(min_w, max_w)
        usable_h = inner_h - 2
        usable_w = inner_w - 2
        if usable_h <= 0 or usable_w <= 0:
            continue
        if usable_h * usable_w < total_area + 8 * len(slot_rects):
            continue

        positions = _place_rectangles(usable_h, usable_w, slot_rects, padding=1)
        if positions is None:
            continue

        blueprint = np.full((inner_h + 2, inner_w + 2), AZURE, dtype=int)
        blueprint[0, :] = GREY
        blueprint[-1, :] = GREY
        blueprint[:, 0] = GREY
        blueprint[:, -1] = GREY

        placed_specs: list[dict] = []
        for spec, (top0, left0) in zip(slot_specs, positions):
            border_top = top0 + 2
            border_left = left0 + 2
            slot_h = int(spec["slot_h"])
            slot_w = int(spec["slot_w"])

            blueprint[border_top, border_left : border_left + slot_w + 2] = GREY
            blueprint[border_top + slot_h + 1, border_left : border_left + slot_w + 2] = GREY
            blueprint[border_top : border_top + slot_h + 2, border_left] = GREY
            blueprint[border_top : border_top + slot_h + 2, border_left + slot_w + 1] = GREY

            corner = random.choice(
                [
                    (0, 0),
                    (0, slot_w - 1),
                    (slot_h - 1, 0),
                    (slot_h - 1, slot_w - 1),
                ]
            )
            marker_r = border_top + 1 + corner[0]
            marker_c = border_left + 1 + corner[1]
            blueprint[marker_r, marker_c] = STEPS_TO_ROT_MARKER[int(spec["rot"])]

            placed_specs.append(
                {
                    **spec,
                    "slot_top": int(border_top + 1),
                    "slot_left": int(border_left + 1),
                }
            )

        return blueprint, placed_specs

    return None


def _build_expected_output(blueprint: np.ndarray, slot_specs: list[dict]) -> np.ndarray:
    out = blueprint.copy()
    for spec in slot_specs:
        rotated_patch = _rotate_clockwise(spec["piece_patch"], spec["rot"])
        rotated_mask = _rotate_clockwise(spec["piece_mask"], spec["rot"])
        top = int(spec["slot_top"])
        left = int(spec["slot_left"])
        region = out[top : top + spec["slot_h"], left : left + spec["slot_w"]]
        region[rotated_mask] = rotated_patch[rotated_mask]
    return out


def _build_input_canvas(blueprint: np.ndarray, slot_specs: list[dict]) -> np.ndarray | None:
    blueprint_h, blueprint_w = map(int, blueprint.shape)
    max_piece_h = max(int(spec["piece_patch"].shape[0]) for spec in slot_specs)
    total_piece_area = sum(int(spec["piece_patch"].shape[0] * spec["piece_patch"].shape[1]) for spec in slot_specs)

    min_h = max(CANVAS_H_RANGE[0], blueprint_h, max_piece_h + 2)
    max_h = CANVAS_H_RANGE[1]
    if min_h > max_h:
        return None
    canvas_h = random.randint(min_h, max_h)

    gap = random.randint(*RIGHT_GAP_RANGE)
    right_margin = random.randint(*RIGHT_MARGIN_RANGE)
    top_margin = random.randint(*TOP_MARGIN_RANGE)
    bottom_margin = random.randint(*TOP_MARGIN_RANGE)
    canvas_h = max(canvas_h, blueprint_h + top_margin + bottom_margin)

    piece_rects = [tuple(map(int, spec["piece_patch"].shape)) for spec in slot_specs]
    max_piece_w = max(w for _, w in piece_rects)
    zone_w = max(
        10,
        blueprint_w - 3,
        int(math.ceil(total_piece_area / canvas_h)) + max_piece_w + 3,
    )
    zone_w += random.randint(0, 4)

    positions = _place_rectangles(canvas_h, zone_w, piece_rects, padding=1)
    if positions is None:
        return None

    canvas_w = 2 + blueprint_w + gap + zone_w + right_margin
    canvas = np.full((canvas_h, canvas_w), AZURE, dtype=int)

    blueprint_top_max = canvas_h - blueprint_h
    if blueprint_top_max < 0:
        return None
    blueprint_top = random.randint(0, blueprint_top_max)
    blueprint_left = 1
    canvas[blueprint_top : blueprint_top + blueprint_h, blueprint_left : blueprint_left + blueprint_w] = blueprint

    zone_left = blueprint_left + blueprint_w + gap
    for spec, (top, left) in zip(slot_specs, positions):
        patch = spec["piece_patch"]
        mask = spec["piece_mask"]
        region = canvas[top : top + patch.shape[0], zone_left + left : zone_left + left + patch.shape[1]]
        region[mask] = patch[mask]

    return canvas


def _gen_attempt() -> np.ndarray | None:
    n_slots = random.randint(*SLOT_COUNT_RANGE)
    colors = random.sample(PIECE_COLORS, n_slots)
    piece_dims = _choose_piece_dims(n_slots)
    rots = _choose_rotations(n_slots)

    slot_specs = []
    for color, (piece_h, piece_w), rot in zip(colors, piece_dims, rots):
        piece_mask = _random_piece_mask(piece_h, piece_w)
        piece_patch = _render_piece(piece_mask, color)
        slot_h, slot_w = _rotated_dims(piece_h, piece_w, rot)
        slot_specs.append(
            {
                "rot": int(rot),
                "slot_h": int(slot_h),
                "slot_w": int(slot_w),
                "piece_mask": piece_mask,
                "piece_patch": piece_patch,
                "color": int(color),
            }
        )

    built = _build_blueprint(slot_specs)
    if built is None:
        return None
    blueprint, slot_specs = built

    expected = _build_expected_output(blueprint, slot_specs)
    canvas = _build_input_canvas(blueprint, slot_specs)
    if canvas is None:
        return None

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle instance")
