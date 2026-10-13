import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


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
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((factor, factor), dtype=int)).astype(bool)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return tuple(mask.shape), tuple(int(x) for x in mask.astype(int).ravel())


def _all_transforms(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen = set()
    for rot in range(4):
        spun = np.rot90(mask, rot)
        for flipped in (False, True):
            variant = np.fliplr(spun) if flipped else spun
            key = _mask_key(variant)
            if key not in seen:
                seen.add(key)
                out.append(variant.copy())
    return out


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _build_shape_pool() -> tuple[list[np.ndarray], dict[tuple[int, int, int], list[np.ndarray]]]:
    bases = [
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
        np.array([[1, 1], [0, 1], [0, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1], [1, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=bool),
        np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    ]
    variants: list[np.ndarray] = []
    seen = set()
    for base in bases:
        for variant in _all_transforms(base):
            if not _touches_all_sides(variant):
                continue
            key = _mask_key(variant)
            if key not in seen:
                seen.add(key)
                variants.append(variant.copy())
    groups: dict[tuple[int, int, int], list[np.ndarray]] = {}
    for mask in variants:
        key = (int(mask.shape[0]), int(mask.shape[1]), int(mask.sum()))
        groups.setdefault(key, []).append(mask)
    return variants, groups


SHAPES, SHAPE_GROUPS = _build_shape_pool()
MULTI_GROUP_KEYS = [key for key, group in SHAPE_GROUPS.items() if len(group) >= 2]


def _extract_rect_and_holes(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[dict]]:
    grey_comps = _components(grid == GREY)
    assert grey_comps, "expected a grey rectangle"
    rect_coords = max(grey_comps, key=lambda x: x.shape[0])
    rect_bbox = _bbox_from_coords(rect_coords)
    r0, c0, r1, c1 = rect_bbox
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    hole_comps = _components(patch == BLACK)
    holes: list[dict] = []
    for coords in hole_comps:
        if (
            (coords[:, 0] == 0).any()
            or (coords[:, 0] == patch.shape[0] - 1).any()
            or (coords[:, 1] == 0).any()
            or (coords[:, 1] == patch.shape[1] - 1).any()
        ):
            continue
        hb = _bbox_from_coords(coords)
        local_mask = _crop(patch == BLACK, hb)
        holes.append(
            {
                "bbox": (r0 + hb[0], c0 + hb[1], r0 + hb[2], c0 + hb[3]),
                "mask": local_mask,
                "coords": coords + np.array([[r0, c0]], dtype=int),
            }
        )
    holes.sort(key=lambda hole: hole["bbox"])
    return rect_bbox, holes


def _extract_pieces(grid: np.ndarray) -> list[dict]:
    fg = (grid != BLACK) & (grid != GREY)
    pieces: list[dict] = []
    for coords in _components(fg):
        b = _bbox_from_coords(coords)
        patch = _crop(grid, b)
        mask = patch != BLACK
        color = int(patch[mask][0])
        pieces.append({"bbox": b, "coords": coords, "mask": mask, "color": color})
    pieces.sort(key=lambda piece: piece["bbox"])
    return pieces


def _scale_match(piece_mask: np.ndarray, hole_mask: np.ndarray) -> int | None:
    ph, pw = piece_mask.shape
    hh, hw = hole_mask.shape
    if hh % ph or hw % pw:
        return None
    factor_h = hh // ph
    factor_w = hw // pw
    if factor_h != factor_w or factor_h < 1:
        return None
    factor = int(factor_h)
    if np.array_equal(_scale_mask(piece_mask, factor), hole_mask):
        return factor
    return None


def _unique_assignment(pieces: list[dict], holes: list[dict]) -> tuple[dict[int, tuple[int, int]], list[list[tuple[int, int]]]]:
    candidates: list[list[tuple[int, int]]] = []
    for hole in holes:
        options: list[tuple[int, int]] = []
        for pi, piece in enumerate(pieces):
            factor = _scale_match(piece["mask"], hole["mask"])
            if factor is not None:
                options.append((pi, factor))
        candidates.append(options)

    order = sorted(range(len(holes)), key=lambda idx: len(candidates[idx]))
    solutions: list[dict[int, tuple[int, int]]] = []
    used: set[int] = set()
    cur: dict[int, tuple[int, int]] = {}

    def dfs(depth: int) -> None:
        if len(solutions) > 1:
            return
        if depth == len(order):
            solutions.append(cur.copy())
            return
        hi = order[depth]
        for pi, factor in candidates[hi]:
            if pi in used:
                continue
            used.add(pi)
            cur[hi] = (pi, factor)
            dfs(depth + 1)
            cur.pop(hi, None)
            used.remove(pi)

    dfs(0)
    assert len(solutions) == 1, "expected exactly one valid piece-to-hole assignment"
    return solutions[0], candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale each exterior piece by its own integer factor and embed it into the uniquely matching black hole.

    Input structure:
    - A single grey rectangle sits on a black background and contains several black holes.
    - Outside the rectangle are several monochrome colored pieces, including distractors.
    - Every true hole is the integer pixel-scaled version of exactly one exterior piece.

    Rule:
    - Extract the grey rectangle and each enclosed black hole as a binary shape.
    - For every exterior piece, test integer pixel scaling against every hole; a match requires exact equality after scaling.
    - Fill each matched hole with the original piece color, erase that matched piece from its old location, and leave unmatched distractors in place.

    Key insight:
    - Direct shape equality always fails: the hole-piece correspondence is only visible after inferring a separate integer scale factor from each hole's size and confirming the full scaled mask.
    """
    g = np.asarray(grid, dtype=int)
    rect_bbox, holes = _extract_rect_and_holes(g)
    assert rect_bbox is not None
    assert holes, "expected at least one hole"
    pieces = _extract_pieces(g)
    assignment, _ = _unique_assignment(pieces, holes)

    out = g.copy()
    matched_piece_ids = {pi for pi, _factor in assignment.values()}
    for pi in matched_piece_ids:
        coords = pieces[pi]["coords"]
        out[coords[:, 0], coords[:, 1]] = BLACK

    for hi, hole in enumerate(holes):
        pi, _factor = assignment[hi]
        color = pieces[pi]["color"]
        hb = hole["bbox"]
        mask = hole["mask"]
        r0, c0, _r1, _c1 = hb
        rr, cc = np.argwhere(mask).T
        out[r0 + rr, c0 + cc] = color

    return out


def _random_choice(rng: np.random.Generator, seq):
    return seq[int(rng.integers(len(seq)))]


def _scale_patterns(n_match: int) -> list[list[int]]:
    if n_match == 2:
        return [[2, 3], [2, 4], [3, 4]]
    return [[2, 2, 3], [2, 3, 3], [2, 3, 4]]


def _layout_holes(hole_masks: list[np.ndarray], rng: np.random.Generator) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    n = len(hole_masks)
    sizes = [(mask.shape[0], mask.shape[1]) for mask in hole_masks]
    gap = int(rng.integers(2, 4))
    margins = tuple(int(x) for x in rng.integers(1, 3, size=4))
    layouts: list[tuple[str, int, int, list[tuple[int, int]], list[np.ndarray]]] = []

    if n in (2, 3):
        top, left, bottom, right = margins
        max_h = max(h for h, _ in sizes)
        slot_h = max_h + int(rng.integers(0, 3))
        slot_ws = [w + int(rng.integers(0, 3)) for _, w in sizes]
        rect_h = top + slot_h + bottom
        rect_w = left + sum(slot_ws) + gap * (n - 1) + right
        pos: list[tuple[int, int]] = []
        cursor = left
        for (h, w), slot_w in zip(sizes, slot_ws):
            r = top + int(rng.integers(0, slot_h - h + 1))
            c = cursor + int(rng.integers(0, slot_w - w + 1))
            pos.append((r, c))
            cursor += slot_w + gap
        layouts.append(("row", rect_h, rect_w, pos, hole_masks))

        max_w = max(w for _, w in sizes)
        slot_w = max_w + int(rng.integers(0, 3))
        slot_hs = [h + int(rng.integers(0, 3)) for h, _ in sizes]
        rect_h = top + sum(slot_hs) + gap * (n - 1) + bottom
        rect_w = left + slot_w + right
        pos = []
        cursor = top
        for (h, w), slot_h_i in zip(sizes, slot_hs):
            r = cursor + int(rng.integers(0, slot_h_i - h + 1))
            c = left + int(rng.integers(0, slot_w - w + 1))
            pos.append((r, c))
            cursor += slot_h_i + gap
        layouts.append(("col", rect_h, rect_w, pos, hole_masks))

    if n == 3:
        top, left, bottom, right = margins
        (h0, w0), (h1, w1), (h2, w2) = sizes
        top_h = max(h0, h1) + int(rng.integers(0, 3))
        bot_h = h2 + int(rng.integers(0, 3))
        left_w = w0 + int(rng.integers(0, 3))
        right_w = w1 + int(rng.integers(0, 3))
        lower_w = max(w2 + int(rng.integers(0, 4)), max(left_w + right_w + gap - 1, w2))
        rect_h = top + top_h + gap + bot_h + bottom
        rect_w = left + max(left_w + gap + right_w, lower_w) + right
        top_y = top
        top_x = left
        lower_x0 = left + int(rng.integers(0, max(1, rect_w - left - right - w2 + 1)))
        pos = [
            (top_y + int(rng.integers(0, top_h - h0 + 1)), top_x + int(rng.integers(0, left_w - w0 + 1))),
            (
                top_y + int(rng.integers(0, top_h - h1 + 1)),
                top_x + left_w + gap + int(rng.integers(0, right_w - w1 + 1)),
            ),
            (top + top_h + gap + int(rng.integers(0, bot_h - h2 + 1)), lower_x0),
        ]
        layouts.append(("L", rect_h, rect_w, pos, hole_masks))

    rng.shuffle(layouts)
    for _name, rect_h, rect_w, pos, ordered_masks in layouts:
        if rect_h > 24 or rect_w > 24:
            continue
        rect = np.full((rect_h, rect_w), GREY, dtype=int)
        ok = True
        for mask, (r0, c0) in zip(ordered_masks, pos):
            h, w = mask.shape
            if r0 < 1 or c0 < 1 or r0 + h >= rect_h - 0 or c0 + w >= rect_w - 0:
                ok = False
                break
            rr, cc = np.argwhere(mask).T
            if np.any(rect[r0 + rr, c0 + cc] == BLACK):
                ok = False
                break
            rect[r0 + rr, c0 + cc] = BLACK
        if ok:
            return rect, pos
    return None


def _place_bbox(
    occupied: np.ndarray,
    shape: tuple[int, int],
    rng: np.random.Generator,
    *,
    border: int = 0,
) -> tuple[int, int] | None:
    h, w = occupied.shape
    ph, pw = shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(border, h - ph - border + 1):
        for c0 in range(border, w - pw - border + 1):
            if not occupied[r0 : r0 + ph, c0 : c0 + pw].any():
                candidates.append((r0, c0))
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _reserve(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 0) -> None:
    r0, c0, r1, c1 = bbox
    r0 = max(0, r0 - pad)
    c0 = max(0, c0 - pad)
    r1 = min(occupied.shape[0] - 1, r1 + pad)
    c1 = min(occupied.shape[1] - 1, c1 + pad)
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: tuple[int, int], color: int) -> None:
    r0, c0 = top
    rr, cc = np.argwhere(mask).T
    grid[r0 + rr, c0 + cc] = color


def _build_case(rng: np.random.Generator) -> dict | None:
    n_match = int(rng.integers(2, 4))
    scales = _random_choice(rng, _scale_patterns(n_match)).copy()
    rng.shuffle(scales)

    primary_group_key = _random_choice(rng, MULTI_GROUP_KEYS)
    primary_group = SHAPE_GROUPS[primary_group_key]
    true_mask = _random_choice(rng, primary_group)
    impostor_options = [mask for mask in primary_group if not np.array_equal(mask, true_mask)]
    if not impostor_options:
        return None
    distractor_anchor = _random_choice(rng, impostor_options)

    chosen = [true_mask]
    seen = {_mask_key(true_mask)}
    while len(chosen) < n_match:
        candidate = _random_choice(rng, SHAPES)
        key = _mask_key(candidate)
        if key in seen:
            continue
        chosen.append(candidate)
        seen.add(key)

    rng.shuffle(chosen)
    holes = [_scale_mask(mask, scale) for mask, scale in zip(chosen, scales)]
    layout = _layout_holes(holes, rng)
    if layout is None:
        return None
    rect_sprite, hole_positions = layout

    rect_h, rect_w = rect_sprite.shape
    margin_top = int(rng.integers(2, 5))
    margin_left = int(rng.integers(2, 5))
    margin_bottom = int(rng.integers(2, 6))
    margin_right = int(rng.integers(2, 6))
    grid_h = rect_h + margin_top + margin_bottom
    grid_w = rect_w + margin_left + margin_right
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    rect_r0, rect_c0 = margin_top, margin_left
    grid[rect_r0 : rect_r0 + rect_h, rect_c0 : rect_c0 + rect_w] = rect_sprite
    rect_bbox = (rect_r0, rect_c0, rect_r0 + rect_h - 1, rect_c0 + rect_w - 1)

    matched_colors = rng.permutation(PALETTE)[:n_match]
    pieces = []
    for idx, (mask, scale, color, hole_mask, hole_top) in enumerate(
        zip(chosen, scales, matched_colors, holes, hole_positions)
    ):
        hr, hc = hole_top
        pieces.append(
            {
                "mask": mask.copy(),
                "color": int(color),
                "scale": int(scale),
                "hole_top": (rect_r0 + hr, rect_c0 + hc),
                "hole_mask": hole_mask,
                "hole_bbox": (
                    rect_r0 + hr,
                    rect_c0 + hc,
                    rect_r0 + hr + hole_mask.shape[0] - 1,
                    rect_c0 + hc + hole_mask.shape[1] - 1,
                ),
                "slot_index": idx,
            }
        )

    distractors = [{"mask": distractor_anchor.copy()}]
    extra_distractor_count = int(rng.integers(0, 2))
    extra_tries = 0
    while len(distractors) < 1 + extra_distractor_count and extra_tries < 50:
        extra_tries += 1
        candidate = _random_choice(rng, SHAPES)
        if any(_scale_match(candidate, piece["hole_mask"]) is not None for piece in pieces):
            continue
        if any(np.array_equal(candidate, existing["mask"]) for existing in distractors):
            continue
        distractors.append({"mask": candidate.copy()})

    total_colors_needed = n_match + len(distractors)
    if total_colors_needed > len(PALETTE):
        return None
    colors = list(rng.permutation(PALETTE)[:total_colors_needed])
    for piece, color in zip(pieces, colors[:n_match]):
        piece["color"] = int(color)
    for distractor, color in zip(distractors, colors[n_match:]):
        distractor["color"] = int(color)

    occupied = np.zeros_like(grid, dtype=bool)
    _reserve(occupied, rect_bbox, pad=1)

    outside_objects = pieces + distractors
    rng.shuffle(outside_objects)
    for obj in outside_objects:
        ph, pw = obj["mask"].shape
        pos = _place_bbox(occupied, (ph, pw), rng, border=0)
        if pos is None:
            return None
        r0, c0 = pos
        _paint_mask(grid, obj["mask"], pos, obj["color"])
        obj["top"] = pos
        obj["bbox"] = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        _reserve(occupied, obj["bbox"], pad=1)

    state = {
        "input": grid,
        "rect_bbox": rect_bbox,
        "matched": pieces,
        "distractors": distractors,
    }
    if not _validate_state(state):
        return None
    return state


def _expected_output_from_state(state: dict) -> np.ndarray:
    out = state["input"].copy()
    for piece in state["matched"]:
        r0, c0 = piece["top"]
        rr, cc = np.argwhere(piece["mask"]).T
        out[r0 + rr, c0 + cc] = BLACK
        hr, hc = piece["hole_top"]
        hrr, hcc = np.argwhere(piece["hole_mask"]).T
        out[hr + hrr, hc + hcc] = piece["color"]
    return out


def _validate_state(state: dict) -> bool:
    try:
        inp = state["input"]
        exp = _expected_output_from_state(state)
        if inp.ndim != 2 or max(inp.shape) > 30 or max(exp.shape) > 30:
            return False

        rect_bbox, holes = _extract_rect_and_holes(inp)
        if rect_bbox != state["rect_bbox"]:
            return False
        pieces = _extract_pieces(inp)
        if len(holes) != len(state["matched"]):
            return False
        if len(pieces) != len(state["matched"]) + len(state["distractors"]):
            return False

        assignment, candidates = _unique_assignment(pieces, holes)
        scales = {factor for _hi, (_pi, factor) in assignment.items()}
        if min(scales) <= 1 or len(scales) < 2:
            return False
        if any(len(opts) == 0 for opts in candidates):
            return False

        direct_equal = False
        for piece in pieces:
            for hole in holes:
                if piece["mask"].shape == hole["mask"].shape and np.array_equal(piece["mask"], hole["mask"]):
                    direct_equal = True
        if direct_equal:
            return False

        has_bbox_area_impostor = False
        matched_piece_ids = {pi for pi, _factor in assignment.values()}
        for hi, (pi, factor) in assignment.items():
            true_piece = pieces[pi]
            true_key = (true_piece["mask"].shape[0], true_piece["mask"].shape[1], int(true_piece["mask"].sum()))
            for other_i, other_piece in enumerate(pieces):
                if other_i == pi or other_i in matched_piece_ids:
                    continue
                other_key = (
                    other_piece["mask"].shape[0],
                    other_piece["mask"].shape[1],
                    int(other_piece["mask"].sum()),
                )
                if other_key == true_key and _scale_match(other_piece["mask"], holes[hi]["mask"]) is None:
                    has_bbox_area_impostor = True
                    break
            if has_bbox_area_impostor:
                break
        if not has_bbox_area_impostor:
            return False

        if not np.array_equal(main(inp), exp):
            return False
    except AssertionError:
        return False

    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        state = _build_case(rng)
        if state is not None:
            return state["input"]
    raise RuntimeError("failed to generate a valid 变倍拼图 input")
