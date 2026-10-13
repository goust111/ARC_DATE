import random
from collections import deque

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# --------------- shared helpers ---------------


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(int(x) for x in coords[0])
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(mask.sum())


def _crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rows, cols = np.where(mask)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0, r1, c1)


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == 0 or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )

    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["bbox"][2], comp["bbox"][3], comp["color"]))
    return comps


def _pair_candidates(comps: list[dict]) -> list[dict]:
    candidates: list[dict] = []
    for i, anchor in enumerate(comps):
        holes = ~anchor["mask"]
        if not holes.any() or not _mask_connected(holes):
            continue
        hole_crop, _hole_bbox = _crop_mask(holes)
        for j, filler in enumerate(comps):
            if i == j or anchor["color"] == filler["color"]:
                continue
            for rot_k in range(4):
                rot_mask = np.rot90(filler["mask"], rot_k)
                if rot_mask.shape == hole_crop.shape and np.array_equal(rot_mask, hole_crop):
                    candidates.append({"anchor": i, "filler": j, "rot_k": rot_k})
                    break
    candidates.sort(key=lambda item: (min(item["anchor"], item["filler"]), max(item["anchor"], item["filler"]), item["anchor"]))
    return candidates


def _solve_pairing(comps: list[dict], limit: int = 2) -> list[list[dict]]:
    candidates = _pair_candidates(comps)
    incident: list[list[dict]] = [[] for _ in comps]
    for cand in candidates:
        incident[cand["anchor"]].append(cand)
        incident[cand["filler"]].append(cand)

    solutions: list[list[dict]] = []

    def dfs(remaining: frozenset[int], current: list[dict]) -> None:
        if len(solutions) >= limit:
            return
        if not remaining:
            solutions.append(current.copy())
            return

        idx = min(remaining)
        options = [
            cand
            for cand in incident[idx]
            if cand["anchor"] in remaining and cand["filler"] in remaining
        ]
        for cand in options:
            current.append(cand)
            dfs(remaining - {cand["anchor"], cand["filler"]}, current)
            current.pop()

    dfs(frozenset(range(len(comps))), [])
    return solutions


def _placement_from_pair(anchor: dict, filler: dict) -> tuple[np.ndarray, int, int, int]:
    rect = np.where(anchor["mask"], int(anchor["color"]), int(filler["color"])).astype(int)
    ar0, ac0, ar1, ac1 = anchor["bbox"]
    br0, bc0, br1, bc1 = filler["bbox"]
    mid_r = (ar0 + ar1 + br0 + br1) // 4
    mid_c = (ac0 + ac1 + bc0 + bc1) // 4
    top = mid_r - rect.shape[0] // 2
    left = mid_c - rect.shape[1] // 2
    return rect, top, left, int(rect.shape[0] * rect.shape[1])


def _render_solution(shape: tuple[int, int], comps: list[dict], pairing: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    placements = []
    for cand in pairing:
        rect, top, left, area = _placement_from_pair(comps[cand["anchor"]], comps[cand["filler"]])
        placements.append((area, top, left, cand["anchor"], cand["filler"], rect))

    placements.sort(key=lambda item: (-item[0], item[1], item[2], item[3], item[4]))
    h, w = shape
    for _area, top, left, _anchor_idx, _filler_idx, rect in placements:
        rh, rw = rect.shape
        for r in range(rh):
            for c in range(rw):
                rr, cc = top + r, left + c
                if 0 <= rr < h and 0 <= cc < w and out[rr, cc] == 0:
                    out[rr, cc] = int(rect[r, c])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge complementary fragment pairs into complete rectangles and place them at the midpoint of the original pair.

    Input structure:
    - Black background with an even number of monochrome 4-connected colored fragments.
    - Each fragment has bounding box at most 5x5, and each pair uses two different colors.
    - In every valid pair, one fragment sits inside a larger rectangle-shaped bbox with connected empty cells, and the other fragment matches that empty-cell shape after an optional 0/90/180/270-degree rotation.

    Rule:
    - Find the unique pairing by testing which fragment can fill the connected empty cells inside another fragment's bounding box.
    - For each pair, rebuild the full rectangle: keep the anchor fragment's original pixels in its own color, and fill the missing cells with the partner fragment's color.
    - Place the rebuilt rectangle on an output grid of the same size, centered at the midpoint of the two original fragment bounding-box centers.
    - If rebuilt rectangles overlap, place larger-area rectangles first; later ones do not overwrite existing pixels.

    Key insight:
    - The task is not to move single components independently: each component must first be matched with its unique rotated complement, then the merged rectangle is repositioned using the midpoint of the two original bounding boxes.
    """
    g = np.asarray(grid, dtype=int)
    comps = _find_components(g)
    pairing_solutions = _solve_pairing(comps, limit=1)
    assert pairing_solutions, "expected at least one valid complementary pairing"
    return _render_solution(g.shape, comps, pairing_solutions[0])


# --------------- generator ---------------

GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 30
PAIR_COUNT_CHOICES = (2, 3, 4)
RECT_OPTIONS = ((3, 5), (5, 3), (5, 5))
GEN_ATTEMPTS = 500
OUTPUT_OVERLAP_PROB = 0.45
MIN_INPUT_OFFSET = 3
MAX_INPUT_OFFSET = 10

BASE_HOLE_MASKS = (
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 1, 1]], dtype=bool),
    np.array([[0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]], dtype=bool),
)


def _build_shape_catalog() -> dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]]:
    catalog = {shape: [] for shape in RECT_OPTIONS}
    for rect_h, rect_w in RECT_OPTIONS:
        seen: set[tuple[bytes, bytes, tuple[int, int], tuple[int, int]]] = set()
        for base in BASE_HOLE_MASKS:
            for hole_crop in (base, base.T):
                hole_h, hole_w = hole_crop.shape
                if hole_h > rect_h or hole_w > rect_w:
                    continue
                if hole_crop.shape[0] % 2 == 0 or hole_crop.shape[1] % 2 == 0:
                    continue
                for top in range(rect_h - hole_h + 1):
                    for left in range(rect_w - hole_w + 1):
                        hole_full = np.zeros((rect_h, rect_w), dtype=bool)
                        hole_full[top : top + hole_h, left : left + hole_w] = hole_crop
                        anchor = ~hole_full
                        if not _mask_connected(anchor):
                            continue
                        if not _touches_all_sides(anchor):
                            continue
                        sig = (anchor.tobytes(), hole_crop.tobytes(), anchor.shape, hole_crop.shape)
                        if sig in seen:
                            continue
                        seen.add(sig)
                        catalog[(rect_h, rect_w)].append((anchor.copy(), hole_crop.copy()))
    return catalog


def _sample_anchor_and_hole(rect_h: int, rect_w: int) -> tuple[np.ndarray, np.ndarray] | None:
    choices = SHAPE_CATALOG.get((rect_h, rect_w))
    if not choices:
        return None
    anchor, hole = random.choice(choices)
    anchor = anchor.copy()
    hole = hole.copy()

    if rect_h == rect_w:
        if random.random() < 0.5:
            anchor = np.flipud(anchor)
            hole = np.flipud(hole)
        if random.random() < 0.5:
            anchor = np.fliplr(anchor)
            hole = np.fliplr(hole)
        rot_k = random.randrange(4)
        anchor = np.rot90(anchor, rot_k)
        hole = np.rot90(hole, rot_k)
    else:
        if random.random() < 0.5:
            anchor = np.flipud(anchor)
            hole = np.flipud(hole)
        if random.random() < 0.5:
            anchor = np.fliplr(anchor)
            hole = np.fliplr(hole)
        if random.random() < 0.5:
            anchor = np.rot90(anchor, 2)
            hole = np.rot90(hole, 2)
    return anchor, hole


SHAPE_CATALOG = _build_shape_catalog()


def _boxes_overlap(box_a: tuple[int, int], shape_a: tuple[int, int], box_b: tuple[int, int], shape_b: tuple[int, int]) -> bool:
    ar, ac = box_a
    br, bc = box_b
    ah, aw = shape_a
    bh, bw = shape_b
    return not (ar + ah <= br or br + bh <= ar or ac + aw <= bc or bc + bw <= ac)


def _random_box(shape: tuple[int, int], grid_h: int, grid_w: int) -> tuple[int, int]:
    h, w = shape
    return random.randint(0, grid_h - h), random.randint(0, grid_w - w)


def _sample_overlapping_box(base_box: tuple[int, int], base_shape: tuple[int, int], shape: tuple[int, int], grid_h: int, grid_w: int) -> tuple[int, int] | None:
    br, bc = base_box
    bh, bw = base_shape
    h, w = shape
    candidates = []
    for dr in range(-(h - 1), bh):
        for dc in range(-(w - 1), bw):
            top = br + dr
            left = bc + dc
            if not (0 <= top <= grid_h - h and 0 <= left <= grid_w - w):
                continue
            if _boxes_overlap(base_box, base_shape, (top, left), shape):
                candidates.append((top, left))
    if not candidates:
        return None
    return random.choice(candidates)


def _choose_output_boxes(rect_shapes: list[tuple[int, int]], grid_h: int, grid_w: int, want_overlap: bool) -> tuple[list[tuple[int, int]], bool] | None:
    n = len(rect_shapes)
    boxes: list[tuple[int, int] | None] = [None] * n
    by_area = sorted(range(n), key=lambda idx: rect_shapes[idx][0] * rect_shapes[idx][1], reverse=True)

    first = by_area[0]
    boxes[first] = _random_box(rect_shapes[first], grid_h, grid_w)
    overlap_done = False

    if want_overlap:
        overlap_candidates = [
            idx
            for idx in by_area[1:]
            if rect_shapes[idx][0] * rect_shapes[idx][1] != rect_shapes[first][0] * rect_shapes[first][1]
        ]
        if not overlap_candidates:
            return None
        second = random.choice(overlap_candidates)
        overlap_box = _sample_overlapping_box(boxes[first], rect_shapes[first], rect_shapes[second], grid_h, grid_w)
        if overlap_box is None:
            return None
        boxes[second] = overlap_box
        overlap_done = True

    remaining = [idx for idx in range(n) if boxes[idx] is None]
    random.shuffle(remaining)
    for idx in remaining:
        placed = False
        for _ in range(100):
            cand_box = _random_box(rect_shapes[idx], grid_h, grid_w)
            if any(
                boxes[j] is not None and _boxes_overlap(cand_box, rect_shapes[idx], boxes[j], rect_shapes[j])
                for j in range(n)
            ):
                continue
            boxes[idx] = cand_box
            placed = True
            break
        if not placed:
            return None

    return [box for box in boxes if box is not None], overlap_done


def _cells_from_mask(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    rows, cols = np.where(mask)
    return [(top + int(r), left + int(c)) for r, c in zip(rows, cols)]


def _too_close(cells_a: list[tuple[int, int]], cells_b: list[tuple[int, int]]) -> bool:
    for ar, ac in cells_a:
        for br, bc in cells_b:
            if max(abs(ar - br), abs(ac - bc)) <= 1:
                return True
    return False


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _place_input_pair(
    blocked: np.ndarray,
    grid_h: int,
    grid_w: int,
    target_box: tuple[int, int],
    anchor_mask: np.ndarray,
    filler_mask: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    rect_h, rect_w = anchor_mask.shape
    fill_h, fill_w = filler_mask.shape
    center_r = target_box[0] + rect_h // 2
    center_c = target_box[1] + rect_w // 2

    offsets = [
        (dr, dc)
        for dr in range(-MAX_INPUT_OFFSET, MAX_INPUT_OFFSET + 1)
        for dc in range(-MAX_INPUT_OFFSET, MAX_INPUT_OFFSET + 1)
        if max(abs(dr), abs(dc)) >= MIN_INPUT_OFFSET
    ]
    random.shuffle(offsets)

    for dr, dc in offsets:
        anchor_top = center_r + dr - rect_h // 2
        anchor_left = center_c + dc - rect_w // 2
        filler_top = center_r - dr - fill_h // 2
        filler_left = center_c - dc - fill_w // 2

        if not (0 <= anchor_top <= grid_h - rect_h and 0 <= anchor_left <= grid_w - rect_w):
            continue
        if not (0 <= filler_top <= grid_h - fill_h and 0 <= filler_left <= grid_w - fill_w):
            continue

        anchor_cells = _cells_from_mask(anchor_mask, anchor_top, anchor_left)
        filler_cells = _cells_from_mask(filler_mask, filler_top, filler_left)
        all_cells = anchor_cells + filler_cells

        if len(set(all_cells)) != len(all_cells):
            continue
        if any(blocked[r, c] for r, c in all_cells):
            continue
        if _too_close(anchor_cells, filler_cells):
            continue

        return (anchor_top, anchor_left), (filler_top, filler_left)

    return None


def _draw_piece(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.where(mask)
    grid[rows + top, cols + left] = int(color)


def _build_expected(shape: tuple[int, int], pair_specs: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    ordered = sorted(pair_specs, key=lambda spec: (-spec["area"], spec["index"]))
    h, w = shape
    for spec in ordered:
        rect = np.where(spec["anchor_mask"], spec["color_a"], spec["color_b"]).astype(int)
        top, left = spec["output_box"]
        rh, rw = rect.shape
        for r in range(rh):
            for c in range(rw):
                rr, cc = top + r, left + c
                if 0 <= rr < h and 0 <= cc < w and out[rr, cc] == 0:
                    out[rr, cc] = int(rect[r, c])
    return out


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    grid_w = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    colors = random.sample(range(1, 10), 2 * pair_count)

    pair_specs: list[dict] = []
    for idx in range(pair_count):
        rect_h, rect_w = random.choices(RECT_OPTIONS, weights=[3, 3, 2], k=1)[0]
        sample = _sample_anchor_and_hole(rect_h, rect_w)
        if sample is None:
            return None
        anchor_mask, hole_crop = sample
        rot_k = random.randrange(4)
        pair_specs.append(
            {
                "index": idx,
                "anchor_mask": anchor_mask,
                "hole_crop": hole_crop,
                "filler_mask": np.rot90(hole_crop, rot_k),
                "rot_k": rot_k,
                "color_a": colors[2 * idx],
                "color_b": colors[2 * idx + 1],
                "area": rect_h * rect_w,
            }
        )

    rect_shapes = [tuple(spec["anchor_mask"].shape) for spec in pair_specs]
    want_overlap = pair_count >= 2 and random.random() < OUTPUT_OVERLAP_PROB
    chosen = _choose_output_boxes(rect_shapes, grid_h, grid_w, want_overlap)
    if chosen is None:
        return None
    output_boxes, overlap_done = chosen
    if want_overlap and not overlap_done:
        return None

    for spec, output_box in zip(pair_specs, output_boxes):
        spec["output_box"] = output_box

    inp = np.zeros((grid_h, grid_w), dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    for spec in pair_specs:
        placed = _place_input_pair(
            blocked,
            grid_h,
            grid_w,
            spec["output_box"],
            spec["anchor_mask"],
            spec["filler_mask"],
        )
        if placed is None:
            return None
        (anchor_top, anchor_left), (filler_top, filler_left) = placed
        _draw_piece(inp, spec["anchor_mask"], anchor_top, anchor_left, spec["color_a"])
        _draw_piece(inp, spec["filler_mask"], filler_top, filler_left, spec["color_b"])
        _mark_blocked(blocked, _cells_from_mask(spec["anchor_mask"], anchor_top, anchor_left))
        _mark_blocked(blocked, _cells_from_mask(spec["filler_mask"], filler_top, filler_left))

    comps = _find_components(inp)
    if len(comps) != 2 * pair_count:
        return None
    solutions = _solve_pairing(comps, limit=2)
    if len(solutions) != 1:
        return None

    expected = _build_expected(inp.shape, pair_specs)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-fragment puzzle")
