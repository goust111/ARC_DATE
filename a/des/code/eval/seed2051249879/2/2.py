from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
MAROON = 9
PIECE_COLORS = (1, 2, 3, 4, 6, 8)

DIRS = ("up", "right", "down", "left")
DIR_TO_CW = {"up": 0, "right": 1, "down": 2, "left": 3}

ARROW_SHAPES = {
    "up": ((0, 0), (0, 1), (1, 0), (2, 0)),
    "right": ((0, 0), (0, 1), (0, 2), (1, 2)),
    "down": ((0, 1), (1, 1), (2, 0), (2, 1)),
    "left": ((0, 0), (1, 0), (1, 1), (1, 2)),
}
ARROW_TO_DIR = {tuple(sorted(shape)): direction for direction, shape in ARROW_SHAPES.items()}


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
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _rotate_cw(patch: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(patch, -turns)


def _detect_arrow_direction(coords: list[tuple[int, int]]) -> str:
    key = _normalize(coords)
    assert key in ARROW_TO_DIR, f"unexpected arrow shape: {key}"
    return ARROW_TO_DIR[key]


def _arrow_slot_score(
    arrow_bbox: tuple[int, int, int, int],
    slot_bbox: tuple[int, int, int, int],
    direction: str,
) -> int | None:
    ar0, ac0, ar1, ac1 = arrow_bbox
    sr0, sc0, sr1, sc1 = slot_bbox
    if direction == "up":
        gap = ar0 - sr1 - 1
        overlap = _interval_overlap(ac0, ac1, sc0, sc1)
    elif direction == "down":
        gap = sr0 - ar1 - 1
        overlap = _interval_overlap(ac0, ac1, sc0, sc1)
    elif direction == "right":
        gap = sc0 - ac1 - 1
        overlap = _interval_overlap(ar0, ar1, sr0, sr1)
    else:
        gap = ac0 - sc1 - 1
        overlap = _interval_overlap(ar0, ar1, sr0, sr1)
    if gap < 0 or overlap <= 0:
        return None
    arrow_center = ar0 + ar1 + ac0 + ac1
    slot_center = sr0 + sr1 + sc0 + sc1
    return gap * 100 + abs(arrow_center - slot_center)


def _solve_weighted_assignment(candidates: list[list[tuple[int, int]]], count: int) -> list[int]:
    order = sorted(range(count), key=lambda i: (len(candidates[i]), i))
    chosen = [-1] * count
    used = [False] * count
    best: list[int] | None = None

    def dfs(pos: int) -> bool:
        nonlocal best
        if pos == len(order):
            best = chosen[:]
            return True
        item = order[pos]
        for target, _score in sorted(candidates[item], key=lambda item_score: (item_score[1], item_score[0])):
            if used[target]:
                continue
            used[target] = True
            chosen[item] = target
            if dfs(pos + 1):
                return True
            chosen[item] = -1
            used[target] = False
        return False

    assert dfs(0), "expected a unique complete assignment"
    assert best is not None
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each left-side template slot with the uniquely matching right-side fragment after
    rotating that fragment according to the slot's orange arrow direction.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    sep_cols = [c for c in range(w) if np.all(g[:, c] == GREY)]
    assert sep_cols, "expected a full-height grey separator column"
    sep = sep_cols[0]
    left = g[:, :sep].copy()
    right = g[:, sep + 1 :].copy()

    template_color = MAROON if np.any(left == MAROON) else int(
        Counter(int(v) for v in left.ravel() if int(v) not in (BG, ORANGE)).most_common(1)[0][0]
    )
    out = left.copy()
    out[out == ORANGE] = template_color

    slot_comps = []
    for comp in _components(left == BG):
        r0, c0, r1, c1 = _bbox(comp)
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == left.shape[1] - 1:
            continue
        assert len(comp) == (r1 - r0 + 1) * (c1 - c0 + 1), "slots must be black rectangles"
        slot_comps.append({"bbox": (r0, c0, r1, c1), "size": (r1 - r0 + 1, c1 - c0 + 1)})
    assert slot_comps, "expected at least one enclosed slot"

    left_arrows = []
    for comp in _components(left == ORANGE):
        left_arrows.append(
            {
                "bbox": _bbox(comp),
                "direction": _detect_arrow_direction(comp),
            }
        )
    assert len(left_arrows) == len(slot_comps), "expected one arrow per slot"

    arrow_candidates: list[list[tuple[int, int]]] = []
    for arrow in left_arrows:
        matches: list[tuple[int, int]] = []
        for slot_idx, slot in enumerate(slot_comps):
            score = _arrow_slot_score(arrow["bbox"], slot["bbox"], arrow["direction"])
            if score is not None:
                matches.append((slot_idx, score))
        assert matches, "each arrow must correspond to a slot"
        arrow_candidates.append(matches)
    arrow_to_slot = _solve_weighted_assignment(arrow_candidates, len(slot_comps))
    slot_dirs = [None] * len(slot_comps)
    for arrow_idx, slot_idx in enumerate(arrow_to_slot):
        slot_dirs[slot_idx] = left_arrows[arrow_idx]["direction"]
    assert all(direction is not None for direction in slot_dirs)

    piece_mask = (right != BG) & (right != ORANGE)
    pieces = []
    for comp in _components(piece_mask):
        r0, c0, r1, c1 = _bbox(comp)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        for r, c in comp:
            patch[r - r0, c - c0] = int(right[r, c])
        pieces.append({"patch": patch, "size": patch.shape})
    assert len(pieces) == len(slot_comps), "expected one fragment per slot"

    slot_piece_candidates: list[list[tuple[int, int]]] = []
    for slot_idx, slot in enumerate(slot_comps):
        direction = slot_dirs[slot_idx]
        assert direction is not None
        turns = DIR_TO_CW[direction]
        matches: list[tuple[int, int]] = []
        for piece_idx, piece in enumerate(pieces):
            rotated = _rotate_cw(piece["patch"], turns)
            if rotated.shape == slot["size"]:
                matches.append((piece_idx, 0))
        assert matches, "each slot must match at least one fragment"
        slot_piece_candidates.append(matches)
    slot_to_piece = _solve_weighted_assignment(slot_piece_candidates, len(slot_comps))

    for slot_idx, piece_idx in enumerate(slot_to_piece):
        slot = slot_comps[slot_idx]
        direction = slot_dirs[slot_idx]
        assert direction is not None
        rotated = _rotate_cw(pieces[piece_idx]["patch"], DIR_TO_CW[direction])
        r0, c0, r1, c1 = slot["bbox"]
        assert rotated.shape == (r1 - r0 + 1, c1 - c0 + 1)
        target = out[r0 : r1 + 1, c0 : c1 + 1]
        mask = rotated != BG
        target[mask] = rotated[mask]

    return out


def _has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    inv = ~mask
    seen = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in (0, w - 1):
            if inv[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if inv[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and inv[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return bool(np.any(inv & ~seen))


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _random_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    out = [(r, c)]
    while (r, c) != (tr, tc):
        choices = []
        if r != tr:
            choices.append((r + (1 if tr > r else -1), c))
        if c != tc:
            choices.append((r, c + (1 if tc > c else -1)))
        r, c = random.choice(choices)
        out.append((r, c))
    return out


def _make_piece_patch(height: int, width: int, color: int) -> np.ndarray:
    area = height * width
    for _ in range(300):
        anchors = [
            (0, random.randrange(width)),
            (height - 1, random.randrange(width)),
            (random.randrange(height), 0),
            (random.randrange(height), width - 1),
        ]
        random.shuffle(anchors)
        cells = {anchors[0]}
        for anchor in anchors[1:]:
            start = random.choice(tuple(cells))
            cells.update(_random_path(start, anchor))

        min_area = max(len(cells), max(height + width - 2, int(round(area * 0.45))))
        max_area = max(min_area, area - 1)
        target_area = random.randint(min_area, max_area)
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not _touches_all_sides(mask):
            continue
        if _has_hole(mask):
            continue
        if mask.all() and random.random() < 0.8:
            continue
        patch = np.zeros((height, width), dtype=int)
        patch[mask] = color
        return patch
    raise RuntimeError("failed to build a fragment patch")


def _arrow_patch(direction: str) -> np.ndarray:
    coords = ARROW_SHAPES[direction]
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in coords:
        patch[r, c] = ORANGE
    return patch


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, transparent: int = BG) -> None:
    h, w = patch.shape
    region = grid[top : top + h, left : left + w]
    mask = patch != transparent
    region[mask] = patch[mask]


def _slot_feature_bbox(
    top: int, left: int, height: int, width: int, direction: str, arrow_offset: int
) -> tuple[int, int, int, int]:
    arrow = _arrow_patch(direction)
    if direction == "up":
        ar, ac = top + height, left + arrow_offset
    elif direction == "down":
        ar, ac = top - arrow.shape[0], left + arrow_offset
    elif direction == "right":
        ar, ac = top + arrow_offset, left - arrow.shape[1]
    else:
        ar, ac = top + arrow_offset, left + width
    r0 = min(top, ar)
    c0 = min(left, ac)
    r1 = max(top + height - 1, ar + arrow.shape[0] - 1)
    c1 = max(left + width - 1, ac + arrow.shape[1] - 1)
    return r0, c0, r1, c1


def _draw_slot_and_arrow(
    template: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    direction: str,
    arrow_offset: int,
) -> None:
    template[top : top + height, left : left + width] = BG
    arrow = _arrow_patch(direction)
    if direction == "up":
        _place_patch(template, arrow, top + height, left + arrow_offset)
    elif direction == "down":
        _place_patch(template, arrow, top - arrow.shape[0], left + arrow_offset)
    elif direction == "right":
        _place_patch(template, arrow, top + arrow_offset, left - arrow.shape[1])
    else:
        _place_patch(template, arrow, top + arrow_offset, left + width)


def _expand_bbox(box: tuple[int, int, int, int], amount: int, h: int, w: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return max(0, r0 - amount), max(0, c0 - amount), min(h - 1, r1 + amount), min(w - 1, c1 + amount)


def _try_place_slots(
    template_h: int,
    template_w: int,
    slot_specs: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    blocked = np.zeros((template_h, template_w), dtype=bool)
    placed: list[dict[str, object]] = []
    order = list(range(len(slot_specs)))
    random.shuffle(order)
    for spec_idx in order:
        spec = slot_specs[spec_idx]
        height, width = spec["slot_size"]
        direction = spec["direction"]
        assert isinstance(height, int) and isinstance(width, int) and isinstance(direction, str)
        for _ in range(500):
            if direction == "up":
                r_min, r_max = 1, template_h - height - 3
                c_min, c_max = 1, template_w - width - 1
                offset = random.randrange(width - 1)
            elif direction == "down":
                r_min, r_max = 3, template_h - height - 1
                c_min, c_max = 1, template_w - width - 1
                offset = random.randrange(width - 1)
            elif direction == "right":
                r_min, r_max = 1, template_h - height - 1
                c_min, c_max = 3, template_w - width - 1
                offset = random.randrange(height - 1)
            else:
                r_min, r_max = 1, template_h - height - 1
                c_min, c_max = 1, template_w - width - 3
                offset = random.randrange(height - 1)
            if r_min > r_max or c_min > c_max:
                return None
            top = random.randint(r_min, r_max)
            left = random.randint(c_min, c_max)
            box = _slot_feature_bbox(top, left, height, width, direction, offset)
            ex0, ey0, ex1, ey1 = _expand_bbox(box, 1, template_h, template_w)
            if blocked[ex0 : ex1 + 1, ey0 : ey1 + 1].any():
                continue
            blocked[ex0 : ex1 + 1, ey0 : ey1 + 1] = True
            placed.append(
                {
                    "spec_idx": spec_idx,
                    "slot_top": top,
                    "slot_left": left,
                    "slot_size": (height, width),
                    "direction": direction,
                    "arrow_offset": offset,
                }
            )
            break
        else:
            return None
    return placed


def _draw_right_piece(
    canvas: np.ndarray,
    top: int,
    left: int,
    patch: np.ndarray,
    arrow_offset: int,
) -> None:
    _place_patch(canvas, patch, top, left)
    _place_patch(canvas, _arrow_patch("up"), top + patch.shape[0], left + arrow_offset)


def _try_place_right_pieces(height: int, width: int, pieces: list[np.ndarray]) -> np.ndarray | None:
    canvas = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    order = list(range(len(pieces)))
    random.shuffle(order)
    for piece_idx in order:
        patch = pieces[piece_idx]
        ph, pw = patch.shape
        for _ in range(400):
            top = random.randint(0, height - ph - 3)
            left = random.randint(0, width - pw)
            arrow_offset = random.randrange(pw - 1)
            feature_box = (top, left, top + ph + 2, left + pw - 1)
            ex0, ey0, ex1, ey1 = _expand_bbox(feature_box, 1, height, width)
            if blocked[ex0 : ex1 + 1, ey0 : ey1 + 1].any():
                continue
            blocked[ex0 : ex1 + 1, ey0 : ey1 + 1] = True
            _draw_right_piece(canvas, top, left, patch, arrow_offset)
            break
        else:
            return None
    return canvas


def _build_expected_output(
    left: np.ndarray,
    template_box: tuple[int, int, int, int],
    slot_infos: list[dict[str, object]],
    piece_patches: list[np.ndarray],
) -> np.ndarray:
    out = left.copy()
    out[out == ORANGE] = MAROON
    tr0, tc0, _, _ = template_box
    for info in slot_infos:
        spec_idx = int(info["spec_idx"])
        top = tr0 + int(info["slot_top"])
        left_c = tc0 + int(info["slot_left"])
        direction = str(info["direction"])
        rotated = _rotate_cw(piece_patches[spec_idx], DIR_TO_CW[direction])
        region = out[top : top + rotated.shape[0], left_c : left_c + rotated.shape[1]]
        mask = rotated != BG
        region[mask] = rotated[mask]
    return out


def _generate_attempt() -> np.ndarray | None:
    piece_count = random.randint(2, 4)
    directions = [random.choice(DIRS) for _ in range(piece_count)]

    dims_pool = [(h, w) for h in range(3, 6) for w in range(3, 6) if h != w]
    random.shuffle(dims_pool)
    piece_dims = dims_pool[:piece_count]
    colors = random.sample(PIECE_COLORS, piece_count)
    piece_patches = [_make_piece_patch(h, w, color) for (h, w), color in zip(piece_dims, colors)]

    slot_specs = []
    for idx, ((ph, pw), direction) in enumerate(zip(piece_dims, directions)):
        if direction in ("up", "down"):
            slot_size = (ph, pw)
        else:
            slot_size = (pw, ph)
        slot_specs.append({"spec_idx": idx, "slot_size": slot_size, "direction": direction})

    template_h = random.randint(max(12, 4 + piece_count * 3), 18)
    template_w = random.randint(max(12, 4 + piece_count * 3), 18)
    placed_slots = _try_place_slots(template_h, template_w, slot_specs)
    if placed_slots is None:
        return None

    pad_top = random.randint(1, 3)
    pad_bottom = random.randint(1, 3)
    total_h = template_h + pad_top + pad_bottom

    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    left_w = template_w + left_margin + right_margin

    max_piece_width = max(p.shape[1] for p in piece_patches)
    right_w = random.randint(max(10, max_piece_width + 6), 16)
    right_canvas = _try_place_right_pieces(total_h, right_w, piece_patches)
    if right_canvas is None:
        return None

    left = np.zeros((total_h, left_w), dtype=int)
    template_box = (pad_top, left_margin, pad_top + template_h - 1, left_margin + template_w - 1)
    tr0, tc0, tr1, tc1 = template_box
    left[tr0 : tr1 + 1, tc0 : tc1 + 1] = MAROON
    for info in placed_slots:
        _draw_slot_and_arrow(
            left[tr0 : tr1 + 1, tc0 : tc1 + 1],
            int(info["slot_top"]),
            int(info["slot_left"]),
            int(info["slot_size"][0]),
            int(info["slot_size"][1]),
            str(info["direction"]),
            int(info["arrow_offset"]),
        )

    full = np.full((total_h, left_w + 1 + right_w), BG, dtype=int)
    full[:, :left_w] = left
    full[:, left_w] = GREY
    full[:, left_w + 1 :] = right_canvas

    expected = _build_expected_output(left, template_box, placed_slots, piece_patches)
    try:
        if np.array_equal(main(full), expected):
            return full
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        candidate = _generate_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid puzzle instance")
