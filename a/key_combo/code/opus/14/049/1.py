import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
CTRL = 5
ORANGE = 7

MAX_GENERATION_ATTEMPTS = 400
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _draw_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _draw_ring(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    thickness: int,
    color: int,
    bg: int = BG,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color
    if (r1 - r0 + 1) > 2 * thickness and (c1 - c0 + 1) > 2 * thickness:
        grid[r0 + thickness : r1 - thickness + 1, c0 + thickness : c1 - thickness + 1] = bg


def _detect_source_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    comps = _components(grid == BLUE)
    assert comps, "expected at least one blue rectangular source"
    boxes = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        assert np.all(grid[r0, c0 : c1 + 1] == BLUE)
        assert np.all(grid[r1, c0 : c1 + 1] == BLUE)
        assert np.all(grid[r0 : r1 + 1, c0] == BLUE)
        assert np.all(grid[r0 : r1 + 1, c1] == BLUE)
        assert (r1 - r0 + 1) >= 3 and (c1 - c0 + 1) >= 3
        boxes.append((r0, c0, r1, c1))
    return boxes


def _frame_grid(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]], bg: int) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for box in boxes:
        _draw_border(out, box, BLUE)
    return out


def _ray_crossings(grid: np.ndarray, r: int, c: int, color: int = BLUE) -> int:
    h, w = grid.shape
    total = 0
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        while 0 <= rr < h and 0 <= cc < w:
            if int(grid[rr, cc]) == color:
                total += 1
            rr += dr
            cc += dc
    return total


def _recolor_for_count(count: int) -> int:
    if count == 0:
        return CTRL
    if count == 2:
        return RED
    if count == 4:
        return GREEN
    if count == 6:
        return YELLOW
    if count >= 8:
        return ORANGE
    raise AssertionError(f"unexpected crossing count {count}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Hollow each blue rectangle source to a 1-pixel frame, then recolor gray control pixels by frame-crossing count.

    Input structure:
    - The background is black, blue pixels form one or more nested axis-aligned
      rectangular sources, and gray pixels are isolated control points in black
      regions outside or between those sources.
    - Every blue source has a complete rectangular outer perimeter, so its
      bounding box uniquely defines one rectangle even when the blue region is
      thicker than one pixel.
    - Control pixels never 4-touch blue pixels.

    Rule:
    - Treat each blue source bbox as a solid rectangle and hollow it to a
      1-pixel blue frame on the same bbox.
    - From every gray control pixel, cast four rays (up, down, left, right) to
      the grid boundary and count how many blue frame pixels those rays cross.
    - Recolor each control by the total crossing count:
      0 -> gray, 2 -> red, 4 -> green, 6 -> yellow, 8 or more -> orange.
    - Keep all blue frame pixels and the black background unchanged.

    Key insight:
    - The hollowing step turns thick rectangular sources into permeable frames,
      so the control color depends on boundary-crossing count rather than on raw
      filled area; controls at the same containment depth can still differ by
      whether their row or column lines up with deeper frames.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    boxes = _detect_source_rectangles(g)
    frames = _frame_grid(g.shape, boxes, bg)
    out = frames.copy()

    controls = np.argwhere(g == CTRL)
    for r, c in controls:
        rr = int(r)
        cc = int(c)
        count = _ray_crossings(frames, rr, cc, BLUE)
        out[rr, cc] = _recolor_for_count(count)

    return out


def _sample_bboxes_and_thicknesses(
    rng: np.random.Generator,
) -> tuple[tuple[int, int], list[tuple[int, int, int, int]], list[int]] | None:
    n_rings = 4 if rng.random() < 0.6 else 3
    if n_rings == 4:
        thicknesses = [2, 1, 1, 1]
    else:
        patterns = ([2, 1, 1], [2, 2, 1], [3, 1, 1])
        thicknesses = list(patterns[int(rng.integers(len(patterns)))])

    inner_h = int(rng.integers(3, 7))
    inner_w = int(rng.integers(3, 7))
    boxes = [None] * n_rings
    boxes[-1] = (0, 0, inner_h - 1, inner_w - 1)

    for idx in range(n_rings - 2, -1, -1):
        t = thicknesses[idx]
        child = boxes[idx + 1]
        top = int(rng.integers(t + 3, t + 6))
        bottom = int(rng.integers(t + 3, t + 6))
        left = int(rng.integers(t + 3, t + 6))
        right = int(rng.integers(t + 3, t + 6))
        boxes[idx] = (
            int(child[0] - top),
            int(child[1] - left),
            int(child[2] + bottom),
            int(child[3] + right),
        )

    min_r = min(box[0] for box in boxes)
    min_c = min(box[1] for box in boxes)
    boxes = [
        (box[0] - min_r, box[1] - min_c, box[2] - min_r, box[3] - min_c)
        for box in boxes
    ]
    outer = boxes[0]
    outer_h = outer[2] - outer[0] + 1
    outer_w = outer[3] - outer[1] + 1
    if outer_h > 27 or outer_w > 27:
        return None

    extra_h_max = min(7, 30 - outer_h)
    extra_w_max = min(7, 30 - outer_w)
    if extra_h_max < 3 or extra_w_max < 3:
        return None

    extra_h = int(rng.integers(3, extra_h_max + 1))
    extra_w = int(rng.integers(3, extra_w_max + 1))
    top_margin = int(rng.integers(2, extra_h))
    left_margin = int(rng.integers(2, extra_w))
    bottom_margin = extra_h - top_margin
    right_margin = extra_w - left_margin

    shifted = [
        (
            box[0] + top_margin,
            box[1] + left_margin,
            box[2] + top_margin,
            box[3] + left_margin,
        )
        for box in boxes
    ]
    shape = (outer_h + top_margin + bottom_margin, outer_w + left_margin + right_margin)
    assert max(shape) <= 30
    return shape, shifted, thicknesses


def _strict_depth(r: int, c: int, boxes: list[tuple[int, int, int, int]]) -> int:
    depth = 0
    for r0, c0, r1, c1 in boxes:
        depth += int(r0 < r < r1 and c0 < c < c1)
    return depth


def _adjacent_to_blue(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == BLUE:
            return True
    return False


def _enumerate_control_candidates(
    grid: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> list[dict]:
    frames = _frame_grid(grid.shape, boxes, BG)
    h, w = grid.shape
    candidates = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BG:
                continue
            if _adjacent_to_blue(grid, r, c):
                continue
            candidates.append(
                {
                    "pos": (r, c),
                    "count": _ray_crossings(frames, r, c, BLUE),
                    "raw_count": _ray_crossings(grid, r, c, BLUE),
                    "depth": _strict_depth(r, c, boxes),
                }
            )
    return candidates


def _pick_controls(rng: np.random.Generator, candidates: list[dict]) -> list[dict] | None:
    buckets = [
        [cand for cand in candidates if cand["count"] == 0 and cand["depth"] == 0],
        [cand for cand in candidates if cand["count"] == 2 and cand["depth"] == 0],
        [cand for cand in candidates if cand["count"] == 4 and cand["depth"] == 1],
        [cand for cand in candidates if cand["count"] == 6 and cand["depth"] == 1],
        [cand for cand in candidates if cand["count"] >= 8 and cand["depth"] >= 2],
    ]
    if any(not bucket for bucket in buckets):
        return None

    for _ in range(80):
        picked: list[dict] = []
        used = set()
        ok = True
        for bucket in buckets:
            order = list(rng.permutation(len(bucket)))
            choice = None
            for idx in order:
                cand = bucket[int(idx)]
                if cand["pos"] in used:
                    continue
                r, c = cand["pos"]
                if any(abs(r - rr) + abs(c - cc) < 3 for rr, cc in used):
                    continue
                choice = cand
                break
            if choice is None:
                ok = False
                break
            picked.append(choice)
            used.add(choice["pos"])
        if ok:
            return picked
    return None


def _verify_instance(grid: np.ndarray) -> None:
    boxes = sorted(
        _detect_source_rectangles(grid),
        key=lambda b: (b[2] - b[0] + 1) * (b[3] - b[1] + 1),
        reverse=True,
    )
    controls = np.argwhere(grid == CTRL)
    assert len(controls) >= 5
    assert all(not _adjacent_to_blue(grid, int(r), int(c)) for r, c in controls)

    frames = _frame_grid(grid.shape, boxes, BG)
    counts = []
    raw_different = False
    depth_to_counts: dict[int, set[int]] = {}
    for r, c in controls:
        rr = int(r)
        cc = int(c)
        count = _ray_crossings(frames, rr, cc, BLUE)
        raw = _ray_crossings(grid, rr, cc, BLUE)
        depth = _strict_depth(rr, cc, boxes)
        counts.append(count)
        raw_different |= raw != count
        depth_to_counts.setdefault(depth, set()).add(count)

    assert {0, 2, 4, 6}.issubset(set(counts))
    assert any(count >= 8 for count in counts)
    assert raw_different, "at least one control must depend on hollowing"
    assert any(len(values) >= 2 for values in depth_to_counts.values())

    out = main(grid)
    for r, c in controls:
        rr = int(r)
        cc = int(c)
        assert int(out[rr, cc]) == _recolor_for_count(_ray_crossings(frames, rr, cc, BLUE))

    mask_controls = grid == CTRL
    expected_bg = frames.copy()
    expected_bg[mask_controls] = out[mask_controls]
    assert np.array_equal(out, expected_bg)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        sampled = _sample_bboxes_and_thicknesses(rng)
        if sampled is None:
            continue
        shape, boxes, thicknesses = sampled
        grid = np.full(shape, BG, dtype=int)
        for box, thickness in zip(boxes, thicknesses):
            _draw_ring(grid, box, thickness, BLUE, BG)

        candidates = _enumerate_control_candidates(grid, boxes)
        chosen = _pick_controls(rng, candidates)
        if chosen is None:
            continue

        for cand in chosen:
            r, c = cand["pos"]
            grid[r, c] = CTRL

        try:
            _verify_instance(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid nested-frame ray-count input")
