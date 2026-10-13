import itertools
import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered monochrome puzzle strips into a solid rectangle.

    The non-background components are monochrome pieces plus one multicolor
    horizontal strip in a corner. The strip's left-to-right colors define
    boundary-priority order. Pieces assemble into a rectangle by matching their
    complementary row- or column-wise contours without rotation.
    """
    g = np.asarray(grid, dtype=int)
    pieces, strip_order = _extract_input_parts(g)
    return _assemble_output(pieces, strip_order)


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            colors: set[int] = set()
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                colors.add(int(grid[cr, cc]))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [r0 for r0, _ in coords]
            cols = [c0 for _, c0 in coords]
            comps.append(
                {
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "colors": colors,
                }
            )
    return comps


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _extract_input_parts(grid: np.ndarray) -> tuple[list[dict], list[int]]:
    bg = _bg_color(grid)
    h, w = grid.shape
    strip_candidates = []
    pieces: list[dict] = []
    for comp in _components(grid, bg):
        r0, c0, r1, c1 = comp["bbox"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if len(comp["colors"]) > 1:
            touches_corner = (r0 in (0, h - 1)) and (c0 == 0 or c1 == w - 1)
            if height == 1 and len(comp["coords"]) == width and touches_corner:
                strip_candidates.append(comp)
            continue
        color = next(iter(comp["colors"]))
        pieces.append({"color": int(color), "mask": _component_mask(comp)})

    assert len(strip_candidates) == 1, "expected exactly one corner color strip"
    strip = strip_candidates[0]
    sr, sc0, _, sc1 = strip["bbox"]
    strip_order = [int(grid[sr, c]) for c in range(sc0, sc1 + 1)]

    piece_colors = [piece["color"] for piece in pieces]
    assert len(pieces) >= 3, "expected at least three puzzle pieces"
    assert len(piece_colors) == len(set(piece_colors)), "piece colors must be unique"
    assert len(strip_order) == len(set(strip_order)), "strip colors must be unique"
    assert set(piece_colors) == set(strip_order), "strip must mention every piece color exactly once"
    return pieces, strip_order


def _row_profiles(mask: np.ndarray):
    starts = np.empty(mask.shape[0], dtype=int)
    ends = np.empty(mask.shape[0], dtype=int)
    for r in range(mask.shape[0]):
        cols = np.flatnonzero(mask[r])
        if len(cols) == 0 or int(cols[-1] - cols[0] + 1) != len(cols):
            return None
        starts[r] = int(cols[0])
        ends[r] = int(cols[-1])
    return starts, ends


def _col_profiles(mask: np.ndarray):
    starts = np.empty(mask.shape[1], dtype=int)
    ends = np.empty(mask.shape[1], dtype=int)
    for c in range(mask.shape[1]):
        rows = np.flatnonzero(mask[:, c])
        if len(rows) == 0 or int(rows[-1] - rows[0] + 1) != len(rows):
            return None
        starts[c] = int(rows[0])
        ends[c] = int(rows[-1])
    return starts, ends


def _vertical_solutions(pieces: list[dict]) -> list[dict]:
    if len({piece["mask"].shape[0] for piece in pieces}) != 1:
        return []

    profiles = [_row_profiles(piece["mask"]) for piece in pieces]
    if any(profile is None for profile in profiles):
        return []

    starts = [profile[0] for profile in profiles]
    ends = [profile[1] for profile in profiles]
    height = pieces[0]["mask"].shape[0]
    n = len(pieces)
    solutions = []

    for perm in itertools.permutations(range(n)):
        if not np.all(starts[perm[0]] == 0):
            continue
        offsets = [0] * n
        ok = True
        for left, right in zip(perm, perm[1:]):
            delta = ends[left] + 1 - starts[right]
            if not np.all(delta == delta[0]):
                ok = False
                break
            offsets[right] = offsets[left] + int(delta[0])
        if not ok:
            continue

        widths = []
        for r in range(height):
            cursor = 0
            for idx in perm:
                start = offsets[idx] + int(starts[idx][r])
                end = offsets[idx] + int(ends[idx][r])
                if start != cursor:
                    ok = False
                    break
                cursor = end + 1
            if not ok:
                break
            widths.append(cursor)
        if ok and len(set(widths)) == 1:
            solutions.append(
                {
                    "orientation": "vertical",
                    "order": tuple(perm),
                    "offsets": offsets,
                    "starts": starts,
                    "ends": ends,
                    "shape": (height, widths[0]),
                }
            )
    return solutions


def _horizontal_solutions(pieces: list[dict]) -> list[dict]:
    if len({piece["mask"].shape[1] for piece in pieces}) != 1:
        return []

    profiles = [_col_profiles(piece["mask"]) for piece in pieces]
    if any(profile is None for profile in profiles):
        return []

    starts = [profile[0] for profile in profiles]
    ends = [profile[1] for profile in profiles]
    width = pieces[0]["mask"].shape[1]
    n = len(pieces)
    solutions = []

    for perm in itertools.permutations(range(n)):
        if not np.all(starts[perm[0]] == 0):
            continue
        offsets = [0] * n
        ok = True
        for top, bottom in zip(perm, perm[1:]):
            delta = ends[top] + 1 - starts[bottom]
            if not np.all(delta == delta[0]):
                ok = False
                break
            offsets[bottom] = offsets[top] + int(delta[0])
        if not ok:
            continue

        heights = []
        for c in range(width):
            cursor = 0
            for idx in perm:
                start = offsets[idx] + int(starts[idx][c])
                end = offsets[idx] + int(ends[idx][c])
                if start != cursor:
                    ok = False
                    break
                cursor = end + 1
            if not ok:
                break
            heights.append(cursor)
        if ok and len(set(heights)) == 1:
            solutions.append(
                {
                    "orientation": "horizontal",
                    "order": tuple(perm),
                    "offsets": offsets,
                    "starts": starts,
                    "ends": ends,
                    "shape": (heights[0], width),
                }
            )
    return solutions


def _unique_solution(pieces: list[dict]) -> dict:
    solutions = _vertical_solutions(pieces) + _horizontal_solutions(pieces)
    assert len(solutions) == 1, "assembly must have a unique rectangle solution"
    return solutions[0]


def _assemble_output(pieces: list[dict], strip_order: list[int]) -> np.ndarray:
    rank = {color: i for i, color in enumerate(strip_order)}
    solution = _unique_solution(pieces)
    out = np.full(solution["shape"], BG, dtype=int)

    if solution["orientation"] == "vertical":
        for idx in solution["order"]:
            mask = pieces[idx]["mask"]
            rr, cc = np.nonzero(mask)
            out[rr, cc + solution["offsets"][idx]] = pieces[idx]["color"]
        for left, right in zip(solution["order"], solution["order"][1:]):
            left_color = pieces[left]["color"]
            right_color = pieces[right]["color"]
            if rank[left_color] < rank[right_color]:
                paint_color = left_color
                for r in range(out.shape[0]):
                    c = solution["offsets"][right] + int(solution["starts"][right][r])
                    out[r, c] = paint_color
            else:
                paint_color = right_color
                for r in range(out.shape[0]):
                    c = solution["offsets"][left] + int(solution["ends"][left][r])
                    out[r, c] = paint_color
        return out

    for idx in solution["order"]:
        mask = pieces[idx]["mask"]
        rr, cc = np.nonzero(mask)
        out[rr + solution["offsets"][idx], cc] = pieces[idx]["color"]
    for top, bottom in zip(solution["order"], solution["order"][1:]):
        top_color = pieces[top]["color"]
        bottom_color = pieces[bottom]["color"]
        if rank[top_color] < rank[bottom_color]:
            paint_color = top_color
            for c in range(out.shape[1]):
                r = solution["offsets"][bottom] + int(solution["starts"][bottom][c])
                out[r, c] = paint_color
        else:
            paint_color = bottom_color
            for c in range(out.shape[1]):
                r = solution["offsets"][top] + int(solution["ends"][top][c])
                out[r, c] = paint_color
    return out


# --------------- generator ---------------

BG = 0
COLORS = tuple(range(1, 10))
PIECE_COUNT_RANGE = (3, 5)
COMMON_SPAN_RANGE = (6, 10)
TOTAL_SPAN_MAX = 20
MIN_THICKNESS = 3
BOUNDARY_RETRIES = 200
PLACEMENT_RETRIES = 200
GEN_RETRIES = 400


def _random_composition(rng: random.Random, total: int, parts: int, minimum: int) -> list[int]:
    values = [minimum] * parts
    for _ in range(total - minimum * parts):
        values[rng.randrange(parts)] += 1
    rng.shuffle(values)
    return values


def _random_walk(rng: random.Random, length: int, amplitude: int) -> np.ndarray:
    if amplitude <= 0:
        return np.zeros(length, dtype=int)
    current = rng.randint(-amplitude, amplitude)
    values = []
    for _ in range(length):
        current = max(-amplitude, min(amplitude, current + rng.choice((-1, 0, 1))))
        values.append(current)
    if len(set(values)) == 1:
        idx = rng.randrange(length)
        values[idx] = max(-amplitude, min(amplitude, values[idx] + (1 if values[idx] < amplitude else -1)))
    return np.asarray(values, dtype=int)


def _valid_boundaries(boundaries: list[np.ndarray], total_span: int) -> bool:
    common = len(boundaries[0]) if boundaries else 0
    for i in range(common):
        prev = 0
        for boundary in boundaries:
            if int(boundary[i]) - prev < MIN_THICKNESS:
                return False
            prev = int(boundary[i])
        if total_span - prev < MIN_THICKNESS:
            return False
    return True


def _vertical_chain_masks(rng: random.Random) -> list[np.ndarray] | None:
    for _ in range(BOUNDARY_RETRIES):
        count = rng.randint(*PIECE_COUNT_RANGE)
        common_span = rng.randint(*COMMON_SPAN_RANGE)
        total_span = rng.randint(max(MIN_THICKNESS * count + 2, 11), TOTAL_SPAN_MAX)
        widths = _random_composition(rng, total_span, count, MIN_THICKNESS)
        amplitude = max(1, min(2, min(widths) - 2))
        bases = np.cumsum(widths)[:-1]

        boundaries = []
        varied = False
        for base in bases:
            path = base + _random_walk(rng, common_span, amplitude)
            boundaries.append(path)
            varied |= len(set(map(int, path.tolist()))) > 1
        if not varied or not _valid_boundaries(boundaries, total_span):
            continue

        pieces = []
        left = np.zeros(common_span, dtype=int)
        for boundary in boundaries + [np.full(common_span, total_span, dtype=int)]:
            start_abs = left
            end_abs = boundary - 1
            min_start = int(start_abs.min())
            max_end = int(end_abs.max())
            mask = np.zeros((common_span, max_end - min_start + 1), dtype=bool)
            for r in range(common_span):
                mask[r, int(start_abs[r] - min_start) : int(end_abs[r] - min_start) + 1] = True
            pieces.append(mask)
            left = boundary
        return pieces
    return None


def _boxes_separate(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _scatter_grid(
    rng: random.Random,
    masks: list[np.ndarray],
    colors: list[int],
    strip_order: list[int],
) -> np.ndarray | None:
    max_h = max(mask.shape[0] for mask in masks)
    max_w = max(mask.shape[1] for mask in masks)
    total_area = sum(int(mask.sum()) for mask in masks)
    side_hint = int(np.ceil(np.sqrt(total_area))) + 6
    grid_h = max(18, max_h + 6, side_hint + rng.randint(0, 3))
    grid_w = max(20, max_w + 8, side_hint + rng.randint(2, 6))

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    strip_len = len(strip_order)
    corner = rng.choice(("tl", "tr", "bl", "br"))
    strip_r = 0 if corner[0] == "t" else grid_h - 1
    strip_c = 0 if corner[1] == "l" else grid_w - strip_len
    grid[strip_r, strip_c : strip_c + strip_len] = np.array(strip_order, dtype=int)
    occupied = [(strip_r, strip_c, strip_r, strip_c + strip_len - 1)]

    order = sorted(range(len(masks)), key=lambda i: masks[i].shape[0] * masks[i].shape[1], reverse=True)
    placements = [None] * len(masks)

    for idx in order:
        mask = masks[idx]
        h, w = mask.shape
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = rng.randint(0, grid_h - h)
            c0 = rng.randint(0, grid_w - w)
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if all(_boxes_separate(box, other) for other in occupied):
                placements[idx] = (r0, c0)
                occupied.append(box)
                placed = True
                break
        if not placed:
            return None

    for idx, (r0, c0) in enumerate(placements):
        rr, cc = np.nonzero(masks[idx])
        grid[rr + r0, cc + c0] = colors[idx]
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    masks = _vertical_chain_masks(rng)
    if masks is None:
        return None
    if rng.random() < 0.5:
        masks = [mask.T for mask in masks]

    count = len(masks)
    colors = rng.sample(list(COLORS), count)
    strip_order = rng.sample(colors, len(colors))
    pieces = [{"color": color, "mask": mask} for color, mask in zip(colors, masks)]

    try:
        expected = _assemble_output(pieces, strip_order)
    except AssertionError:
        return None

    grid = _scatter_grid(rng, masks, colors, strip_order)
    if grid is None:
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
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
