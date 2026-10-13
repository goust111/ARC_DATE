import random
from collections import Counter, deque

import numpy as np


BG = 0
MARKER_COLOR = 4
OBJECT_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)

GEN_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 120


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def bbox_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    colors: set[int] | None = None,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = get_bg(g)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if seen[r, c] or color == bg:
                continue
            if colors is not None and color not in colors:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords_arr = np.array(coords, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "bbox": bbox_coords(coords_arr),
                    "area": int(coords_arr.shape[0]),
                }
            )
    return out


def _corner_cover(points: set[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    candidates: list[tuple[tuple[int, int, int, int], frozenset[tuple[int, int]]]] = []
    pts = sorted(points)
    seen_boxes: set[tuple[int, int, int, int]] = set()
    for r0, c0 in pts:
        for r1, c1 in pts:
            if r1 <= r0 + 1 or c1 <= c0 + 1:
                continue
            corners = frozenset(((r0, c0), (r0, c1), (r1, c0), (r1, c1)))
            if corners.issubset(points):
                box = (r0, c0, r1, c1)
                if box not in seen_boxes:
                    seen_boxes.add(box)
                    candidates.append((box, corners))

    point_to_candidates: dict[tuple[int, int], list[tuple[tuple[int, int, int, int], frozenset[tuple[int, int]]]]] = {
        p: [] for p in points
    }
    for item in candidates:
        _, corners = item
        for p in corners:
            point_to_candidates[p].append(item)

    solutions: list[list[tuple[int, int, int, int]]] = []

    def search(uncovered: set[tuple[int, int]], chosen: list[tuple[int, int, int, int]]) -> bool:
        if not uncovered:
            solutions.append(chosen.copy())
            return len(solutions) < 2
        pivot = min(uncovered)
        for box, corners in point_to_candidates.get(pivot, []):
            if corners.issubset(uncovered):
                chosen.append(box)
                if not search(uncovered - set(corners), chosen):
                    return False
                chosen.pop()
        return True

    search(set(points), [])
    if len(solutions) != 1:
        return None
    return sorted(solutions[0])


def _solve_assignment(
    frames: list[tuple[int, int, int, int]],
    objects: list[dict],
) -> dict[int, tuple[int, bool]] | None:
    frame_dims = [(box[2] - box[0] - 1, box[3] - box[1] - 1) for box in frames]
    choices: dict[int, list[tuple[int, bool]]] = {}
    for obj_idx, comp in enumerate(objects):
        h = bbox_h(comp["bbox"])
        w = bbox_w(comp["bbox"])
        cand: list[tuple[int, bool]] = []
        for frame_idx, (fh, fw) in enumerate(frame_dims):
            if (h, w) == (fh, fw):
                cand.append((frame_idx, False))
            elif (w, h) == (fh, fw):
                cand.append((frame_idx, True))
        if not cand:
            return None
        choices[obj_idx] = cand

    order = sorted(range(len(objects)), key=lambda idx: len(choices[idx]))
    solutions: list[dict[int, tuple[int, bool]]] = []

    def search(step: int, used_frames: set[int], assign: dict[int, tuple[int, bool]]) -> bool:
        if step == len(order):
            solutions.append(assign.copy())
            return len(solutions) < 2
        obj_idx = order[step]
        for frame_idx, transpose in choices[obj_idx]:
            if frame_idx in used_frames:
                continue
            used_frames.add(frame_idx)
            assign[obj_idx] = (frame_idx, transpose)
            if not search(step + 1, used_frames, assign):
                return False
            del assign[obj_idx]
            used_frames.remove(frame_idx)
        return True

    search(0, set(), {})
    if len(solutions) != 1:
        return None
    return solutions[0]


def _parse_scene(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int, int]], list[dict], dict[int, tuple[int, bool]], int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg=bg)
    by_color: dict[int, list[dict]] = {}
    for comp in components:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    solutions = []
    for color, comps in by_color.items():
        if len(comps) < 4 or len(comps) % 4 != 0:
            continue
        if not all(int(comp["area"]) == 1 for comp in comps):
            continue
        points = {tuple(map(int, comp["coords"][0])) for comp in comps}
        frames = _corner_cover(points)
        if frames is None:
            continue
        objects = sorted(
            [comp for comp in components if int(comp["color"]) != color],
            key=lambda comp: int(comp["color"]),
        )
        if len(objects) != len(frames):
            continue
        if len({int(comp["color"]) for comp in objects}) != len(objects):
            continue
        assignment = _solve_assignment(frames, objects)
        if assignment is None:
            continue
        solutions.append((color, frames, objects, assignment, bg))

    assert len(solutions) == 1, "expected a unique marker-color interpretation"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each monochrome object into the corner-marked frame whose interior matches its bounding-box size.
    
    Input structure:
    - The background is black.
    - One foreground color appears only as isolated corner markers; every four markers define one empty rectangular frame.
    - The remaining foreground colors form one 4-connected monochrome object each, with distinct bounding-box sizes.
    
    Rule:
    - Match every object to the unique frame whose interior size equals the object's bounding-box size.
    - If a frame matches only the swapped width-height order, transpose the object's bounding-box patch across the main diagonal before placing it.
    - Clear the objects from their original positions, keep the corner markers fixed, and align each moved bbox to the frame interior's upper-left corner.
    
    Key insight:
    - The pairing is determined by bounding-box dimensions rather than distance or color, and swapped dimensions signal a transpose instead of a rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    marker_color, frames, objects, assignment, bg = _parse_scene(g)
    out = g.copy()

    for fr0, fc0, fr1, fc1 in frames:
        assert int(g[fr0, fc0]) == marker_color
        assert int(g[fr0, fc1]) == marker_color
        assert int(g[fr1, fc0]) == marker_color
        assert int(g[fr1, fc1]) == marker_color
        assert np.all(g[fr0 + 1 : fr1, fc0 + 1 : fc1] == bg)

    for comp in objects:
        for r, c in comp["coords"]:
            out[int(r), int(c)] = bg

    for obj_idx, comp in enumerate(objects):
        frame_idx, transpose = assignment[obj_idx]
        fr0, fc0, fr1, fc1 = frames[frame_idx]
        r0, c0, r1, c1 = comp["bbox"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if transpose:
            patch = patch.T
        mask = patch == int(comp["color"])
        tr = fr0 + 1
        tc = fc0 + 1
        view = out[tr : tr + patch.shape[0], tc : tc + patch.shape[1]]
        assert view.shape == patch.shape
        assert np.all(view[mask] == bg)
        view[mask] = int(comp["color"])

    return out


def _random_connected_mask(h: int, w: int) -> np.ndarray:
    r = 0
    c = 0
    cells = {(0, 0)}
    moves = ["D"] * (h - 1) + ["R"] * (w - 1)
    random.shuffle(moves)
    for move in moves:
        if move == "D":
            r += 1
        else:
            c += 1
        cells.add((r, c))

    max_area = h * w - 1
    min_area = len(cells)
    target_area = random.randint(min_area, max_area)
    while len(cells) < target_area:
        frontier = []
        for cr, cc in cells:
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        assert frontier
        cells.add(random.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for cr, cc in cells:
        mask[cr, cc] = True
    return mask


def _rects_touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] + 1 < b[0] or b[2] + 1 < a[0] or a[3] + 1 < b[1] or b[3] + 1 < a[1])


def _place_rectangles(
    shape: tuple[int, int],
    sizes: list[tuple[str, int, int, int]],
) -> dict[tuple[str, int], tuple[int, int]] | None:
    h, w = shape
    placed: list[tuple[int, int, int, int]] = []
    positions: dict[tuple[str, int], tuple[int, int]] = {}
    order = sorted(sizes, key=lambda item: item[2] * item[3], reverse=True)
    for kind, idx, rh, rw in order:
        if rh > h or rw > w:
            return None
        success = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, h - rh)
            c0 = random.randint(0, w - rw)
            rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if all(not _rects_touch(rect, other) for other in placed):
                positions[(kind, idx)] = (r0, c0)
                placed.append(rect)
                success = True
                break
        if not success:
            return None
    return positions


def _build_specs() -> list[dict]:
    count = random.randint(2, 4)
    dims: list[tuple[int, int]] = []
    used_dim_sets: set[tuple[int, int]] = set()
    while len(dims) < count:
        h = random.randint(2, 6)
        w = random.randint(2, 6)
        if h == w:
            continue
        key = tuple(sorted((h, w)))
        if key in used_dim_sets:
            continue
        used_dim_sets.add(key)
        dims.append((h, w))

    transpose_flags = [False, True]
    transpose_flags.extend(random.choice([False, True]) for _ in range(count - 2))
    random.shuffle(transpose_flags)
    colors = random.sample(list(OBJECT_COLORS), count)

    specs: list[dict] = []
    for idx, ((h, w), transpose, color) in enumerate(zip(dims, transpose_flags, colors)):
        frame_h, frame_w = (w, h) if transpose else (h, w)
        specs.append(
            {
                "idx": idx,
                "color": int(color),
                "bbox_hw": (h, w),
                "frame_hw": (frame_h, frame_w),
                "transpose": bool(transpose),
                "mask": _random_connected_mask(h, w),
            }
        )
    return specs


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    specs = _build_specs()
    items: list[tuple[str, int, int, int]] = []
    total_area = 0
    max_h = 0
    max_w = 0
    for spec in specs:
        oh, ow = spec["bbox_hw"]
        fh, fw = spec["frame_hw"]
        items.append(("object", spec["idx"], oh, ow))
        items.append(("frame", spec["idx"], fh + 2, fw + 2))
        total_area += oh * ow + (fh + 2) * (fw + 2)
        max_h = max(max_h, oh, fh + 2)
        max_w = max(max_w, ow, fw + 2)

    base = max(14, max_h, max_w, int(np.ceil(np.sqrt(total_area * 2.1))))
    if base > 30:
        return None
    grid_h = random.randint(base, 30)
    grid_w = random.randint(base, 30)
    positions = _place_rectangles((grid_h, grid_w), items)
    if positions is None:
        return None

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    out = np.full((grid_h, grid_w), BG, dtype=int)

    for spec in specs:
        idx = spec["idx"]
        color = spec["color"]
        mask = spec["mask"]
        oh, ow = spec["bbox_hw"]
        fh, fw = spec["frame_hw"]

        fr, fc = positions[("frame", idx)]
        corners = (
            (fr, fc),
            (fr, fc + fw + 1),
            (fr + fh + 1, fc),
            (fr + fh + 1, fc + fw + 1),
        )
        for r, c in corners:
            inp[r, c] = MARKER_COLOR
            out[r, c] = MARKER_COLOR

        orow, ocol = positions[("object", idx)]
        inp[orow : orow + oh, ocol : ocol + ow][mask] = color

        placed_mask = mask.T if spec["transpose"] else mask
        tr = fr + 1
        tc = fc + 1
        out[tr : tr + placed_mask.shape[0], tc : tc + placed_mask.shape[1]][placed_mask] = color

    if np.array_equal(inp, out):
        return None

    direct_count = sum(not spec["transpose"] for spec in specs)
    transpose_count = sum(spec["transpose"] for spec in specs)
    if direct_count == 0 or transpose_count == 0:
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        attempt = _gen_attempt()
        if attempt is None:
            continue
        inp, _out = attempt
        return inp
    raise RuntimeError("failed to generate valid input for puzzle 7")
