import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDES = ("top", "right", "bottom", "left")
SIDE_TO_ROTATION = {
    "top": 0,
    "right": 3,
    "bottom": 2,
    "left": 1,
}

MAX_SIDE = 30
MAX_GENERATION_ATTEMPTS = 500
SHAPE_ATTEMPTS = 600
TEMPLATE_ATTEMPTS = 300
PIECE_PLACEMENT_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(cells)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _coords_to_mask(coords: list[tuple[int, int]]) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return (r0, c0, r1, c1), mask


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _canonical_rotation_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return min(_mask_signature(np.rot90(mask, k)) for k in range(4))


def _extract_template_and_notches(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[dict]]:
    grey_components = _components(grid == GREY)
    assert grey_components, "expected a grey template"
    sizes = [len(comp) for comp in grey_components]
    max_size = max(sizes)
    assert sizes.count(max_size) == 1, "expected a unique largest grey component"
    template = grey_components[sizes.index(max_size)]

    (r0, c0, r1, c1), template_mask = _coords_to_mask(template)
    notch_mask = ~template_mask
    notches: list[dict] = []
    for comp in _components(notch_mask):
        if not any(
            rr == 0 or rr == notch_mask.shape[0] - 1 or cc == 0 or cc == notch_mask.shape[1] - 1
            for rr, cc in comp
        ):
            continue
        abs_coords = [(r0 + rr, c0 + cc) for rr, cc in comp]
        bbox, mask = _coords_to_mask(abs_coords)
        notches.append({"coords": abs_coords, "bbox": bbox, "mask": mask})

    assert notches, "expected at least one boundary notch"
    return template, notches


def _extract_pieces(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    pieces: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BLACK, GREY) or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            bbox, mask = _coords_to_mask(coords)
            pieces.append({"color": color, "coords": coords, "bbox": bbox, "mask": mask})
    assert pieces, "expected colored pieces"
    return pieces


def _can_match(piece_mask: np.ndarray, notch_mask: np.ndarray) -> bool:
    return any(np.array_equal(np.rot90(piece_mask, k), notch_mask) for k in range(4))


def _solve_matching(notches: list[dict], pieces: list[dict]) -> list[int]:
    candidates: list[list[int]] = []
    for notch in notches:
        options = [idx for idx, piece in enumerate(pieces) if _can_match(piece["mask"], notch["mask"])]
        assert options, "each notch must have a matching piece"
        candidates.append(options)

    order = sorted(range(len(notches)), key=lambda idx: (len(candidates[idx]), idx))
    assignment = [-1] * len(notches)
    used = [False] * len(pieces)
    solutions: list[list[int]] = []

    def backtrack(pos: int) -> None:
        if len(solutions) >= 2:
            return
        if pos == len(order):
            solutions.append(assignment.copy())
            return
        notch_idx = order[pos]
        for piece_idx in candidates[notch_idx]:
            if used[piece_idx]:
                continue
            used[piece_idx] = True
            assignment[notch_idx] = piece_idx
            backtrack(pos + 1)
            assignment[notch_idx] = -1
            used[piece_idx] = False

    backtrack(0)
    assert len(solutions) == 1, "matching must be unique"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit each scattered colored fragment into the uniquely matching notch of the grey template.

    Input structure:
    - The background is black.
    - One large grey connected component forms a nearly rectangular template.
    - The template's outer bounding rectangle contains several black boundary-connected notches cut inward from its edges.
    - Outside the template sit several disconnected monochrome colored fragments, each in a different color.

    Rule:
    - Treat every boundary-connected black region inside the template's bounding rectangle as a notch.
    - For each notch, find the one colored fragment whose shape matches the notch after a rotation by 0/90/180/270 degrees.
    - Repaint the notch cells with that fragment's original color and erase the fragment from its old location back to black.
    - Leave the grey template unchanged.

    Key insight:
    - The colored objects are matched to the missing grey cutouts by rotation-invariant shape identity, while the notch itself determines the final orientation inside the template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _, notches = _extract_template_and_notches(g)
    pieces = _extract_pieces(g)
    assert len(notches) == len(pieces), "piece count must match notch count"

    piece_for_notch = _solve_matching(notches, pieces)

    out = g.copy()
    for piece in pieces:
        for r, c in piece["coords"]:
            out[r, c] = BLACK
    for notch_idx, piece_idx in enumerate(piece_for_notch):
        color = int(pieces[piece_idx]["color"])
        for r, c in notches[notch_idx]["coords"]:
            out[r, c] = color
    return out


def _random_connected_mask(rng: random.Random, target_area: int | None = None) -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        area = target_area if target_area is not None else rng.randint(4, 7)
        dims = [(h, w) for h in range(2, 5) for w in range(2, 5) if h * w >= area]
        h, w = rng.choice(dims)

        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))

        if len(cells) != area:
            continue

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, c0 = min(rows), min(cols)
        r1, c1 = max(rows), max(cols)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            mask[r - r0, c - c0] = True

        if min(mask.shape) < 2 or int(mask.sum()) == mask.size:
            continue
        if len({_mask_signature(np.rot90(mask, k)) for k in range(4)}) < 4:
            continue
        return mask
    raise RuntimeError("failed to generate an asymmetric connected mask")


def _choose_shapes(count: int, rng: random.Random) -> list[np.ndarray]:
    repeat_area = rng.choice((4, 5, 6))
    while True:
        first = _random_connected_mask(rng, repeat_area)
        second = _random_connected_mask(rng, repeat_area)
        first_key = _canonical_rotation_signature(first)
        second_key = _canonical_rotation_signature(second)
        if first_key == second_key:
            continue
        shapes = [first, second]
        keys = {first_key, second_key}
        while len(shapes) < count:
            mask = _random_connected_mask(rng)
            key = _canonical_rotation_signature(mask)
            if key in keys:
                continue
            shapes.append(mask)
            keys.add(key)
        areas = [int(mask.sum()) for mask in shapes]
        if any(areas.count(area) >= 2 for area in set(areas)):
            rng.shuffle(shapes)
            return shapes


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expanded_rect(rect: tuple[int, int, int, int], h: int, w: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return max(0, r0 - 1), max(0, c0 - 1), min(h - 1, r1 + 1), min(w - 1, c1 + 1)


def _make_template(
    assignments: list[tuple[str, np.ndarray]],
    rng: random.Random,
) -> tuple[np.ndarray, list[dict]] | None:
    top_depth = max((mask.shape[0] for side, mask in assignments if side == "top"), default=0)
    bottom_depth = max((mask.shape[0] for side, mask in assignments if side == "bottom"), default=0)
    left_span = max((mask.shape[0] for side, mask in assignments if side == "left"), default=0)
    right_span = max((mask.shape[0] for side, mask in assignments if side == "right"), default=0)
    top_width = max((mask.shape[1] for side, mask in assignments if side == "top"), default=0)
    bottom_width = max((mask.shape[1] for side, mask in assignments if side == "bottom"), default=0)
    left_depth = max((mask.shape[1] for side, mask in assignments if side == "left"), default=0)
    right_depth = max((mask.shape[1] for side, mask in assignments if side == "right"), default=0)

    min_h = max(8, top_depth + bottom_depth + 3, left_span + 2, right_span + 2)
    min_w = max(8, left_depth + right_depth + 3, top_width + 2, bottom_width + 2)
    max_h = min(16, MAX_SIDE - 6)
    max_w = min(16, MAX_SIDE - 6)
    if min_h > max_h or min_w > max_w:
        return None

    for _ in range(TEMPLATE_ATTEMPTS):
        rect_h = rng.randint(min_h, max_h)
        rect_w = rng.randint(min_w, max_w)
        template = np.ones((rect_h, rect_w), dtype=bool)
        carved = np.zeros((rect_h, rect_w), dtype=bool)
        notch_records: list[dict] = []
        ok = True

        for side, mask in assignments:
            h, w = mask.shape
            if side in ("top", "bottom"):
                if rect_w - w - 1 < 1:
                    ok = False
                    break
                c0 = rng.randint(1, rect_w - w - 1)
                r0 = 0 if side == "top" else rect_h - h
            else:
                if rect_h - h - 1 < 1:
                    ok = False
                    break
                r0 = rng.randint(1, rect_h - h - 1)
                c0 = 0 if side == "left" else rect_w - w

            coords = [(r0 + rr, c0 + cc) for rr, cc in zip(*np.where(mask))]
            if any(carved[r, c] for r, c in coords):
                ok = False
                break
            for r, c in coords:
                carved[r, c] = True
                template[r, c] = False
            notch_records.append({"side": side, "mask": mask, "coords": coords})

        if not ok:
            continue
        if len(_components(template)) != 1:
            continue

        found = []
        for comp in _components(~template):
            if any(r == 0 or r == rect_h - 1 or c == 0 or c == rect_w - 1 for r, c in comp):
                found.append(comp)
        if len(found) != len(assignments):
            continue

        for record in notch_records:
            _, mask = _coords_to_mask(record["coords"])
            record["mask"] = mask
        return template, notch_records
    return None


def _place_piece(
    grid: np.ndarray,
    piece_mask: np.ndarray,
    color: int,
    reserved: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]] | None:
    h, w = grid.shape
    ph, pw = piece_mask.shape
    for _ in range(PIECE_PLACEMENT_ATTEMPTS):
        r0 = rng.randint(0, h - ph)
        c0 = rng.randint(0, w - pw)
        piece_rect = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        expanded = _expanded_rect(piece_rect, h, w)
        if any(_rects_overlap(expanded, rect) for rect in reserved):
            continue
        coords = [(r0 + rr, c0 + cc) for rr, cc in zip(*np.where(piece_mask))]
        for r, c in coords:
            grid[r, c] = color
        return coords, expanded
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    piece_count = rng.randint(2, 4)
    shapes = _choose_shapes(piece_count, rng)

    side_subset = rng.sample(SIDES, piece_count)
    assignments = []
    for side, shape in zip(side_subset, shapes):
        assignments.append((side, np.rot90(shape, SIDE_TO_ROTATION[side])))

    built = _make_template(assignments, rng)
    if built is None:
        return None
    template_mask, notch_records = built
    rect_h, rect_w = template_mask.shape

    min_canvas_h = max(18, rect_h + 8)
    min_canvas_w = max(18, rect_w + 8)
    if min_canvas_h > MAX_SIDE or min_canvas_w > MAX_SIDE:
        return None

    canvas_h = rng.randint(min_canvas_h, MAX_SIDE)
    canvas_w = rng.randint(min_canvas_w, MAX_SIDE)
    grid = np.full((canvas_h, canvas_w), BLACK, dtype=int)

    template_r0 = rng.randint(1, canvas_h - rect_h - 1)
    template_c0 = rng.randint(1, canvas_w - rect_w - 1)
    template_rect = (template_r0, template_c0, template_r0 + rect_h - 1, template_c0 + rect_w - 1)

    for rr in range(rect_h):
        for cc in range(rect_w):
            if template_mask[rr, cc]:
                grid[template_r0 + rr, template_c0 + cc] = GREY

    colors = rng.sample(PIECE_COLORS, piece_count)
    piece_rotations = [rng.randrange(4) for _ in range(piece_count)]
    if all(rot == 0 for rot in piece_rotations):
        piece_rotations[rng.randrange(piece_count)] = rng.choice((1, 2, 3))

    reserved = [_expanded_rect(template_rect, canvas_h, canvas_w)]
    expected = grid.copy()

    order = list(range(piece_count))
    rng.shuffle(order)
    placed_piece_coords: list[list[tuple[int, int]] | None] = [None] * piece_count

    for idx in order:
        notch = notch_records[idx]
        display_mask = np.rot90(notch["mask"], piece_rotations[idx])
        placement = _place_piece(grid, display_mask, colors[idx], reserved, rng)
        if placement is None:
            return None
        coords, expanded = placement
        reserved.append(expanded)
        placed_piece_coords[idx] = coords

    for idx, notch in enumerate(notch_records):
        for r, c in placed_piece_coords[idx]:
            expected[r, c] = BLACK
        for r, c in notch["coords"]:
            expected[template_r0 + r, template_c0 + c] = colors[idx]

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid notch-fitting puzzle")
