import random
from collections import Counter

import numpy as np


BG = 0
ARROW = 5
FRAME_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
MIN_FRAME_SIZE = 6
MAX_FRAME_SIZE = 10
PAIR_GAPS = [3, 5, 7]
SCENE_GAPS = [6, 8, 10, 12]
MIN_OVERLAP = 2
GEN_ATTEMPTS = 400

ARROW_PATTERNS = {
    "R": (3, 4, [(1, 0), (1, 1), (1, 2), (0, 3), (2, 3)]),
    "L": (3, 4, [(1, 1), (1, 2), (1, 3), (0, 0), (2, 0)]),
    "U": (4, 3, [(1, 1), (2, 1), (3, 1), (0, 0), (0, 2)]),
    "D": (4, 3, [(0, 1), (1, 1), (2, 1), (3, 0), (3, 2)]),
}


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_frame(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < MIN_FRAME_SIZE or w < MIN_FRAME_SIZE:
        return False
    if len(points) != 2 * (h + w) - 4:
        return False
    local = np.zeros((h, w), dtype=bool)
    for r, c in points:
        local[r - r0, c - c0] = True
    return bool(
        np.all(local[0, :])
        and np.all(local[-1, :])
        and np.all(local[:, 0])
        and np.all(local[:, -1])
        and not np.any(local[1:-1, 1:-1])
    )


def _detect_arrow(points: list[tuple[int, int]]) -> str:
    rows = Counter(r for r, _ in points)
    cols = Counter(c for _, c in points)

    row_hits = [r for r, n in rows.items() if n == 3]
    if len(row_hits) == 1:
        shaft_row = row_hits[0]
        shaft_cols = sorted(c for r, c in points if r == shaft_row)
        if shaft_cols == list(range(shaft_cols[0], shaft_cols[0] + 3)):
            tip_pts = [(r, c) for r, c in points if r != shaft_row]
            if len(tip_pts) == 2:
                tip_rows = sorted(r for r, _ in tip_pts)
                tip_cols = {c for _, c in tip_pts}
                if tip_rows == [shaft_row - 1, shaft_row + 1] and len(tip_cols) == 1:
                    tip_col = next(iter(tip_cols))
                    if tip_col == shaft_cols[-1] + 1:
                        return "R"
                    if tip_col == shaft_cols[0] - 1:
                        return "L"

    col_hits = [c for c, n in cols.items() if n == 3]
    if len(col_hits) == 1:
        shaft_col = col_hits[0]
        shaft_rows = sorted(r for r, c in points if c == shaft_col)
        if shaft_rows == list(range(shaft_rows[0], shaft_rows[0] + 3)):
            tip_pts = [(r, c) for r, c in points if c != shaft_col]
            if len(tip_pts) == 2:
                tip_cols = sorted(c for _, c in tip_pts)
                tip_rows = {r for r, _ in tip_pts}
                if tip_cols == [shaft_col - 1, shaft_col + 1] and len(tip_rows) == 1:
                    tip_row = next(iter(tip_rows))
                    if tip_row == shaft_rows[0] - 1:
                        return "U"
                    if tip_row == shaft_rows[-1] + 1:
                        return "D"

    raise AssertionError("failed to detect arrow direction")


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    frames: list[dict] = []
    colors = sorted(int(v) for v in np.unique(g) if int(v) not in {bg, ARROW})
    for color in colors:
        for comp in _components(g == color):
            if not _is_frame(comp):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            arrow_pts = [
                (r0 + 1 + rr, c0 + 1 + cc)
                for rr, cc in np.argwhere(g[r0 + 1 : r1, c0 + 1 : c1] == ARROW)
            ]
            assert len(arrow_pts) == 5
            frames.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "color": int(color),
                    "dir": _detect_arrow(arrow_pts),
                    "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                }
            )
    frames.sort(key=lambda item: item["bbox"])
    assert frames
    return frames


def _overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _target_candidates(specs: list[dict], idx: int) -> list[tuple[tuple[int, int], int]]:
    a = specs[idx]
    ar0, ac0, ar1, ac1 = a["bbox"]
    candidates: list[tuple[tuple[int, int], int]] = []
    for j, b in enumerate(specs):
        if j == idx:
            continue
        br0, bc0, br1, bc1 = b["bbox"]
        direction = a["dir"]
        if direction == "R" and bc0 > ac1 and _overlap_len(ar0, ar1, br0, br1) > 0:
            gap = bc0 - ac1 - 1
            offset = abs((ar0 + ar1) - (br0 + br1))
            candidates.append(((gap, offset), j))
        elif direction == "L" and bc1 < ac0 and _overlap_len(ar0, ar1, br0, br1) > 0:
            gap = ac0 - bc1 - 1
            offset = abs((ar0 + ar1) - (br0 + br1))
            candidates.append(((gap, offset), j))
        elif direction == "D" and br0 > ar1 and _overlap_len(ac0, ac1, bc0, bc1) > 0:
            gap = br0 - ar1 - 1
            offset = abs((ac0 + ac1) - (bc0 + bc1))
            candidates.append(((gap, offset), j))
        elif direction == "U" and br1 < ar0 and _overlap_len(ac0, ac1, bc0, bc1) > 0:
            gap = ar0 - br1 - 1
            offset = abs((ac0 + ac1) - (bc0 + bc1))
            candidates.append(((gap, offset), j))
    candidates.sort(key=lambda item: (item[0][0], item[0][1], item[1]))
    return candidates


def _analyze_specs(specs: list[dict]) -> tuple[list[int | None], set[tuple[int, int]], bool]:
    targets: list[int | None] = []
    unique = True
    for idx in range(len(specs)):
        candidates = _target_candidates(specs, idx)
        if len(candidates) >= 2 and candidates[0][0] == candidates[1][0]:
            unique = False
        targets.append(candidates[0][1] if candidates else None)

    pairs: set[tuple[int, int]] = set()
    for i, j in enumerate(targets):
        if j is None or j <= i:
            continue
        if targets[j] == i and specs[j]["dir"] == OPPOSITE[specs[i]["dir"]]:
            pairs.add((i, j))
    return targets, pairs, unique


def _shift_bbox(
    bbox: tuple[int, int, int, int], dr: int, dc: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _draw_arrow(grid: np.ndarray, bbox: tuple[int, int, int, int], direction: str) -> None:
    r0, c0, r1, c1 = bbox
    ih = r1 - r0 - 1
    iw = c1 - c0 - 1
    ph, pw, pts = ARROW_PATTERNS[direction]
    assert ih >= ph and iw >= pw
    sr = r0 + 1 + (ih - ph) // 2
    sc = c0 + 1 + (iw - pw) // 2
    for dr, dc in pts:
        grid[sr + dr, sc + dc] = ARROW


def _approach_steps(gap: int) -> tuple[int, int]:
    total = max(0, gap - 1)
    first = total // 2
    second = total - first
    return first, second


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    frames = _extract_frames(g)
    targets, pairs, _ = _analyze_specs(frames)
    del targets

    final_boxes = [frame["bbox"] for frame in frames]
    final_colors = [int(frame["color"]) for frame in frames]

    for i, j in pairs:
        fi = frames[i]
        fj = frames[j]
        bi = fi["bbox"]
        bj = fj["bbox"]

        if fi["dir"] in {"L", "R"}:
            if bi[1] <= bj[1]:
                left_idx, right_idx = i, j
            else:
                left_idx, right_idx = j, i
            left_box = frames[left_idx]["bbox"]
            right_box = frames[right_idx]["bbox"]
            gap = right_box[1] - left_box[3] - 1
            move_left, move_right = _approach_steps(gap)
            final_boxes[left_idx] = _shift_bbox(left_box, 0, move_left)
            final_boxes[right_idx] = _shift_bbox(right_box, 0, -move_right)
        else:
            if bi[0] <= bj[0]:
                top_idx, bottom_idx = i, j
            else:
                top_idx, bottom_idx = j, i
            top_box = frames[top_idx]["bbox"]
            bottom_box = frames[bottom_idx]["bbox"]
            gap = bottom_box[0] - top_box[2] - 1
            move_top, move_bottom = _approach_steps(gap)
            final_boxes[top_idx] = _shift_bbox(top_box, move_top, 0)
            final_boxes[bottom_idx] = _shift_bbox(bottom_box, -move_bottom, 0)

        if fi["area"] > fj["area"]:
            final_colors[j] = final_colors[i]
        elif fj["area"] > fi["area"]:
            final_colors[i] = final_colors[j]

    out = np.full_like(g, bg)
    for bbox, color in zip(final_boxes, final_colors):
        _draw_frame(out, bbox, int(color))
    return out


def _scene_bounds(specs: list[dict]) -> tuple[int, int, int, int]:
    rows0 = [spec["bbox"][0] for spec in specs]
    cols0 = [spec["bbox"][1] for spec in specs]
    rows1 = [spec["bbox"][2] for spec in specs]
    cols1 = [spec["bbox"][3] for spec in specs]
    return min(rows0), min(cols0), max(rows1), max(cols1)


def _separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], min_gap: int = 2
) -> bool:
    hgap = max(0, b[1] - a[3] - 1, a[1] - b[3] - 1)
    vgap = max(0, b[0] - a[2] - 1, a[0] - b[2] - 1)
    if hgap == 0 and vgap == 0:
        return False
    return hgap >= min_gap or vgap >= min_gap


def _fits(bbox: tuple[int, int, int, int], specs: list[dict]) -> bool:
    return all(_separated(bbox, spec["bbox"]) for spec in specs)


def _sample_dims() -> tuple[int, int]:
    return random.randint(MIN_FRAME_SIZE, MAX_FRAME_SIZE), random.randint(
        MIN_FRAME_SIZE, MAX_FRAME_SIZE
    )


def _sample_pair_dims(outcome: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if outcome == "equal":
        dims = _sample_dims()
        return dims, dims
    while True:
        d1 = _sample_dims()
        d2 = _sample_dims()
        a1 = d1[0] * d1[1]
        a2 = d2[0] * d2[1]
        if outcome == "first" and a1 > a2:
            return d1, d2
        if outcome == "second" and a2 > a1:
            return d1, d2


def _shift_specs(specs: list[dict], dr: int, dc: int) -> list[dict]:
    shifted = []
    for spec in specs:
        copied = dict(spec)
        copied["bbox"] = _shift_bbox(spec["bbox"], dr, dc)
        shifted.append(copied)
    return shifted


def _make_pair_local() -> tuple[list[dict], str]:
    orientation = random.choice(["H", "V"])
    outcome = random.choice(["first", "second", "equal"])
    d1, d2 = _sample_pair_dims(outcome)
    gap = random.choice(PAIR_GAPS)

    if orientation == "H":
        h1, w1 = d1
        h2, w2 = d2
        delta_low = -(h2 - MIN_OVERLAP)
        delta_high = h1 - MIN_OVERLAP
        r2 = random.randint(delta_low, delta_high)
        specs = [
            {"bbox": (0, 0, h1 - 1, w1 - 1), "dir": "R"},
            {"bbox": (r2, w1 + gap, r2 + h2 - 1, w1 + gap + w2 - 1), "dir": "L"},
        ]
    else:
        h1, w1 = d1
        h2, w2 = d2
        delta_low = -(w2 - MIN_OVERLAP)
        delta_high = w1 - MIN_OVERLAP
        c2 = random.randint(delta_low, delta_high)
        specs = [
            {"bbox": (0, 0, h1 - 1, w1 - 1), "dir": "D"},
            {"bbox": (h1 + gap, c2, h1 + gap + h2 - 1, c2 + w2 - 1), "dir": "U"},
        ]

    minr, minc, _, _ = _scene_bounds(specs)
    return _shift_specs(specs, -minr, -minc), orientation


def _place_pair(specs: list[dict]) -> tuple[list[dict], tuple[int, int]] | None:
    local_specs, _orientation = _make_pair_local()
    if not specs:
        shifted = _shift_specs(local_specs, 0, 0)
        return shifted, (0, 1)

    minr, minc, maxr, maxc = _scene_bounds(specs)
    local_minr, local_minc, local_maxr, local_maxc = _scene_bounds(local_specs)
    unit_h = local_maxr - local_minr + 1
    unit_w = local_maxc - local_minc + 1

    for _ in range(80):
        side = random.choice(["right", "left", "below", "above"])
        if side == "right":
            top = random.randint(minr - 6, maxr + 6)
            left = maxc + random.choice(SCENE_GAPS) + 1
        elif side == "left":
            top = random.randint(minr - 6, maxr + 6)
            left = minc - random.choice(SCENE_GAPS) - unit_w
        elif side == "below":
            top = maxr + random.choice(SCENE_GAPS) + 1
            left = random.randint(minc - 6, maxc + 6)
        else:
            top = minr - random.choice(SCENE_GAPS) - unit_h
            left = random.randint(minc - 6, maxc + 6)

        shifted = _shift_specs(local_specs, top - local_minr, left - local_minc)
        if all(_fits(spec["bbox"], specs) for spec in shifted):
            base = len(specs)
            return shifted, (base, base + 1)
    return None


def _place_one_way(specs: list[dict], target_idx: int) -> dict | None:
    target = specs[target_idx]
    tr0, tc0, tr1, tc1 = target["bbox"]
    h, w = _sample_dims()
    safe_sides = []
    if target["dir"] != "U":
        safe_sides.append("above")
    if target["dir"] != "D":
        safe_sides.append("below")
    if target["dir"] != "L":
        safe_sides.append("left")
    if target["dir"] != "R":
        safe_sides.append("right")
    random.shuffle(safe_sides)

    for side in safe_sides:
        for _ in range(40):
            gap = random.choice(PAIR_GAPS)
            if side == "above":
                c0_low = tc0 - w + MIN_OVERLAP
                c0_high = tc1 - MIN_OVERLAP + 1
                if c0_low > c0_high:
                    continue
                c0 = random.randint(c0_low, c0_high)
                r0 = tr0 - gap - h
                bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
                direction = "D"
            elif side == "below":
                c0_low = tc0 - w + MIN_OVERLAP
                c0_high = tc1 - MIN_OVERLAP + 1
                if c0_low > c0_high:
                    continue
                c0 = random.randint(c0_low, c0_high)
                r0 = tr1 + gap + 1
                bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
                direction = "U"
            elif side == "left":
                r0_low = tr0 - h + MIN_OVERLAP
                r0_high = tr1 - MIN_OVERLAP + 1
                if r0_low > r0_high:
                    continue
                r0 = random.randint(r0_low, r0_high)
                c0 = tc0 - gap - w
                bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
                direction = "R"
            else:
                r0_low = tr0 - h + MIN_OVERLAP
                r0_high = tr1 - MIN_OVERLAP + 1
                if r0_low > r0_high:
                    continue
                r0 = random.randint(r0_low, r0_high)
                c0 = tc1 + gap + 1
                bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
                direction = "L"

            if _fits(bbox, specs):
                return {"bbox": bbox, "dir": direction, "kind": "one_way", "target": target_idx}
    return None


def _interval_overlaps_any(start: int, end: int, intervals: list[tuple[int, int]]) -> bool:
    for a, b in intervals:
        if _overlap_len(start, end, a, b) > 0:
            return True
    return False


def _place_no_target(specs: list[dict]) -> dict | None:
    minr, minc, maxr, maxc = _scene_bounds(specs)
    h, w = _sample_dims()
    sides = ["left", "right", "above", "below"]
    random.shuffle(sides)

    for side in sides:
        if side == "left":
            blockers = [
                (spec["bbox"][0], spec["bbox"][2]) for spec in specs if spec["dir"] == "L"
            ]
            c0 = minc - random.choice(SCENE_GAPS) - w
            for _ in range(40):
                r0 = random.randint(minr - 10, maxr + 10)
                if not _interval_overlaps_any(r0, r0 + h - 1, blockers):
                    bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
                    if _fits(bbox, specs):
                        return {"bbox": bbox, "dir": "L", "kind": "no_target"}
        elif side == "right":
            blockers = [
                (spec["bbox"][0], spec["bbox"][2]) for spec in specs if spec["dir"] == "R"
            ]
            c0 = maxc + random.choice(SCENE_GAPS) + 1
            for _ in range(40):
                r0 = random.randint(minr - 10, maxr + 10)
                if not _interval_overlaps_any(r0, r0 + h - 1, blockers):
                    bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
                    if _fits(bbox, specs):
                        return {"bbox": bbox, "dir": "R", "kind": "no_target"}
        elif side == "above":
            blockers = [
                (spec["bbox"][1], spec["bbox"][3]) for spec in specs if spec["dir"] == "U"
            ]
            r0 = minr - random.choice(SCENE_GAPS) - h
            for _ in range(40):
                c0 = random.randint(minc - 10, maxc + 10)
                if not _interval_overlaps_any(c0, c0 + w - 1, blockers):
                    bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
                    if _fits(bbox, specs):
                        return {"bbox": bbox, "dir": "U", "kind": "no_target"}
        else:
            blockers = [
                (spec["bbox"][1], spec["bbox"][3]) for spec in specs if spec["dir"] == "D"
            ]
            r0 = maxr + random.choice(SCENE_GAPS) + 1
            for _ in range(40):
                c0 = random.randint(minc - 10, maxc + 10)
                if not _interval_overlaps_any(c0, c0 + w - 1, blockers):
                    bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
                    if _fits(bbox, specs):
                        return {"bbox": bbox, "dir": "D", "kind": "no_target"}
    return None


def _assign_colors(specs: list[dict]) -> None:
    palette = FRAME_COLORS[:]
    random.shuffle(palette)
    colors = palette[: len(specs)]
    if len(specs) >= 3 and random.random() < 0.3:
        i, j = random.sample(range(len(specs)), 2)
        colors[j] = colors[i]
    for spec, color in zip(specs, colors):
        spec["color"] = int(color)


def _render_specs(specs: list[dict]) -> np.ndarray:
    minr, minc, maxr, maxc = _scene_bounds(specs)
    top = random.randint(2, 5)
    left = random.randint(2, 5)
    bottom = random.randint(2, 5)
    right = random.randint(2, 5)
    h = (maxr - minr + 1) + top + bottom
    w = (maxc - minc + 1) + left + right
    grid = np.full((h, w), BG, dtype=int)
    dr = top - minr
    dc = left - minc
    for spec in specs:
        bbox = _shift_bbox(spec["bbox"], dr, dc)
        _draw_frame(grid, bbox, int(spec["color"]))
        _draw_arrow(grid, bbox, spec["dir"])
    return grid


def _gen_attempt() -> np.ndarray | None:
    specs: list[dict] = []
    expected_pairs: set[tuple[int, int]] = set()

    first_pair = _place_pair(specs)
    if first_pair is None:
        return None
    pair_specs, pair_ids = first_pair
    specs.extend(pair_specs)
    expected_pairs.add(pair_ids)

    if random.random() < 0.35:
        second_pair = _place_pair(specs)
        if second_pair is None:
            return None
        pair_specs, pair_ids = second_pair
        specs.extend(pair_specs)
        expected_pairs.add(pair_ids)

    single_mode = random.choice(["none", "no_target", "one_way", "both"])
    one_way_indices: list[int] = []
    no_target_indices: list[int] = []

    if single_mode in {"one_way", "both"}:
        pair_ring_choices = sorted({idx for pair in expected_pairs for idx in pair})
        one_way = _place_one_way(specs, random.choice(pair_ring_choices))
        if one_way is None:
            return None
        one_way_indices.append(len(specs))
        specs.append(one_way)

    if single_mode in {"no_target", "both"}:
        no_target = _place_no_target(specs)
        if no_target is None:
            return None
        no_target_indices.append(len(specs))
        specs.append(no_target)

    _assign_colors(specs)
    targets, actual_pairs, unique = _analyze_specs(specs)
    if not unique or actual_pairs != expected_pairs:
        return None

    for idx in one_way_indices:
        target = specs[idx]["target"]
        if targets[idx] != target or targets[target] == idx:
            return None
    for idx in no_target_indices:
        if targets[idx] is not None:
            return None

    grid = _render_specs(specs)
    if len(_extract_frames(grid)) != len(specs):
        return None
    out = main(grid)
    if out.shape != grid.shape or np.any(out == ARROW):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
