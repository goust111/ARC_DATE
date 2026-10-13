from __future__ import annotations

from collections import Counter, deque
import itertools
import random

import numpy as np


BACKGROUND = 0
MARKER = 7
PALETTE = [1, 2, 3, 4, 5, 6, 8, 9]

GRID_MIN = 20
GRID_MAX = 28
RECT_COUNT_CHOICES = [2, 3]
INTERIOR_MIN = 6
INTERIOR_MAX = 8
PATTERN_MIN = 3
PATTERN_MAX = 5
PATTERN_GAP_MIN = 1
PATTERN_GAP_MAX = 3
GROUP_GAP = 2
GEN_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_size(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    dr = 0
    if a[2] < b[0]:
        dr = int(b[0] - a[2])
    elif b[2] < a[0]:
        dr = int(a[0] - b[2])
    dc = 0
    if a[3] < b[1]:
        dc = int(b[1] - a[3])
    elif b[3] < a[1]:
        dc = int(a[1] - b[3])
    return dr + dc


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _pattern_records(grid: np.ndarray, bg: int) -> list[dict]:
    mask = (grid != bg) & (grid != MARKER)
    records: list[dict] = []
    for coords in _components(mask):
        if len(coords) <= 1:
            continue
        rr, cc = coords[:, 0], coords[:, 1]
        values = [int(grid[r, c]) for r, c in zip(rr, cc)]
        counts = Counter(values)
        if len(counts) != 3 or len(set(counts.values())) != 3:
            continue
        box = _bbox_coords(coords)
        h, w = _bbox_size(box)
        if not (PATTERN_MIN <= h <= PATTERN_MAX and PATTERN_MIN <= w <= PATTERN_MAX):
            continue
        records.append(
            {
                "coords": coords,
                "bbox": box,
                "counts": counts,
            }
        )
    return records


def _marker_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    marker_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == MARKER)]
    marker_set = set(marker_coords)
    rows = sorted({r for r, _c in marker_coords})
    cols = sorted({c for _r, c in marker_coords})
    candidates: list[dict] = []
    for r0, r1 in itertools.combinations(rows, 2):
        for c0, c1 in itertools.combinations(cols, 2):
            corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
            if not corners.issubset(marker_set):
                continue
            if int(np.count_nonzero(grid[r0 : r1 + 1, c0 : c1 + 1] == MARKER)) != 4:
                continue
            if r1 - r0 <= 1 or c1 - c0 <= 1:
                continue
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            seed_cells = np.argwhere((interior != bg) & (interior != MARKER))
            if seed_cells.shape[0] != 1:
                continue
            if np.any((interior != bg) & (interior == MARKER)):
                continue
            sr = r0 + 1 + int(seed_cells[0, 0])
            sc = c0 + 1 + int(seed_cells[0, 1])
            candidates.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "corners": corners,
                    "seed": (sr, sc),
                    "seed_color": int(grid[sr, sc]),
                }
            )
    return candidates


def _find_rectangles(grid: np.ndarray, bg: int) -> list[dict]:
    marker_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == MARKER)]
    assert marker_coords, "expected orange corner markers"
    assert len(marker_coords) % 4 == 0, "marker pixels must come in groups of four"
    expected = len(marker_coords) // 4
    assert expected in RECT_COUNT_CHOICES, "expected 2 or 3 marker rectangles"
    all_markers = set(marker_coords)
    candidates = _marker_candidates(grid, bg)
    valid: list[tuple[int, ...]] = []
    for combo in itertools.combinations(range(len(candidates)), expected):
        used: set[tuple[int, int]] = set()
        ok = True
        for idx in combo:
            corners = candidates[idx]["corners"]
            if used & corners:
                ok = False
                break
            used |= corners
        if ok and used == all_markers:
            valid.append(combo)
    assert len(valid) == 1, "marker rectangles must be uniquely determined"
    rects = [candidates[idx] for idx in valid[0]]
    rects.sort(key=lambda rec: rec["bbox"])
    return rects


def _associate_patterns(rectangles: list[dict], patterns: list[dict]) -> dict[int, dict]:
    assert len(patterns) == len(rectangles), "expected one 3-color pattern per rectangle"
    mapping: dict[int, dict] = {}
    for pat in patterns:
        distances = [_bbox_distance(pat["bbox"], rect["bbox"]) for rect in rectangles]
        best = min(distances)
        winners = [idx for idx, dist in enumerate(distances) if dist == best]
        assert len(winners) == 1, "pattern-to-rectangle nearest match must be unique"
        ridx = winners[0]
        assert ridx not in mapping, "two patterns cannot map to the same rectangle"
        mapping[ridx] = pat
    assert len(mapping) == len(rectangles)
    return mapping


def _scene(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    assert bg == BACKGROUND, "background must be black"
    rectangles = _find_rectangles(grid, bg)
    patterns = _pattern_records(grid, bg)
    mapping = _associate_patterns(rectangles, patterns)
    scene: list[dict] = []
    for ridx, rect in enumerate(rectangles):
        counts = mapping[ridx]["counts"]
        ordered = [color for color, _count in sorted(counts.items(), key=lambda kv: kv[1])]
        center_color, odd_color, even_color = ordered
        scene.append(
            {
                "bbox": rect["bbox"],
                "seed": rect["seed"],
                "seed_color": rect["seed_color"],
                "pattern_bbox": mapping[ridx]["bbox"],
                "center_color": int(center_color),
                "odd_color": int(odd_color),
                "even_color": int(even_color),
            }
        )
    return scene


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each orange-corner rectangle with seed-centered square rings whose colors come from a nearby pattern's frequency ranking.

    Input structure:
    - The grid has black background, 2-3 orange four-corner marker rectangles, and one non-black seed pixel inside each rectangle.
    - Near each rectangle, outside it and either above or to the left, sits one connected 3x3 to 5x5 pattern component.
    - Each such pattern uses exactly three non-black non-orange colors with pairwise distinct frequencies.

    Rule:
    - Match every external pattern to the unique nearest orange-corner rectangle by bbox distance.
    - Within each matched pattern, sort its three colors by frequency: rarest becomes the center-role color, middle becomes the odd-ring color, and most frequent becomes the even-ring color.
    - For every background pixel inside the matched rectangle, compute Chebyshev distance to that rectangle's seed pixel and fill odd distances with the odd-ring color and positive even distances with the even-ring color.
    - Keep the orange corner markers, the seed pixels, and the external patterns unchanged.

    Key insight:
    - The rectangle only supplies the fill region and the seed center; the actual ring palette is encoded indirectly by the nearby pattern's local color-frequency ordering.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for record in _scene(g):
        r0, c0, r1, c1 = record["bbox"]
        sr, sc = record["seed"]
        odd_color = record["odd_color"]
        even_color = record["even_color"]
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if out[r, c] != BACKGROUND:
                    continue
                dist = max(abs(r - sr), abs(c - sc))
                assert dist > 0
                out[r, c] = odd_color if dist % 2 == 1 else even_color
    return out


def _choose_counts(area: int) -> tuple[int, int, int]:
    for _ in range(200):
        cuts = sorted(random.sample(range(1, area), 2))
        a = cuts[0]
        b = cuts[1] - cuts[0]
        c = area - cuts[1]
        counts = tuple(sorted((a, b, c)))
        if counts[0] >= 1 and len(set(counts)) == 3:
            return counts
    raise RuntimeError("failed to sample distinct color frequencies")


def _make_pattern_patch(height: int, width: int, colors: tuple[int, int, int]) -> tuple[np.ndarray, dict[str, int]]:
    area = int(height * width)
    counts = _choose_counts(area)
    role_colors = list(colors)
    random.shuffle(role_colors)
    center_color, odd_color, even_color = [int(c) for c in role_colors]
    patch = np.zeros((height, width), dtype=int)
    cells = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(cells)
    cursor = 0
    for color, count in zip((center_color, odd_color, even_color), counts):
        for _ in range(count):
            r, c = cells[cursor]
            patch[r, c] = int(color)
            cursor += 1
    roles = {
        "center": center_color,
        "odd": odd_color,
        "even": even_color,
    }
    return patch, roles


def _sample_group_specs() -> list[dict]:
    rect_count = random.choice(RECT_COUNT_CHOICES)
    role_order = ["center", "odd", "even"]
    if rect_count == 3:
        seed_roles = random.sample(role_order, 3)
        orientations = ["above", "left", random.choice(["above", "left"])]
        random.shuffle(orientations)
    else:
        seed_roles = random.sample(role_order, 2)
        if random.random() < 0.7:
            orientations = ["above", "left"]
            random.shuffle(orientations)
        else:
            orientations = [random.choice(["above", "left"]) for _ in range(2)]

    triplets = random.sample(list(itertools.combinations(PALETTE, 3)), rect_count)
    specs: list[dict] = []
    for idx in range(rect_count):
        inner_h = random.randint(INTERIOR_MIN, INTERIOR_MAX)
        inner_w = random.randint(INTERIOR_MIN, INTERIOR_MAX)
        rect_h = inner_h + 2
        rect_w = inner_w + 2
        pat_h = random.randint(PATTERN_MIN, PATTERN_MAX)
        pat_w = random.randint(PATTERN_MIN, PATTERN_MAX)
        gap = random.randint(PATTERN_GAP_MIN, PATTERN_GAP_MAX)
        orientation = orientations[idx]
        if orientation == "above":
            group_h = pat_h + gap + rect_h
            group_w = rect_w
            offset = random.randint(0, rect_w - pat_w)
        else:
            group_h = rect_h
            group_w = pat_w + gap + rect_w
            offset = random.randint(0, rect_h - pat_h)
        specs.append(
            {
                "triplet": tuple(int(c) for c in triplets[idx]),
                "seed_role": seed_roles[idx],
                "inner_h": inner_h,
                "inner_w": inner_w,
                "rect_h": rect_h,
                "rect_w": rect_w,
                "pat_h": pat_h,
                "pat_w": pat_w,
                "gap": gap,
                "orientation": orientation,
                "group_h": group_h,
                "group_w": group_w,
                "offset": offset,
            }
        )
    return specs


def _derive_bboxes(spec: dict, group_r: int, group_c: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]]:
    if spec["orientation"] == "above":
        pr0 = group_r
        pc0 = group_c + spec["offset"]
        rr0 = group_r + spec["pat_h"] + spec["gap"]
        rc0 = group_c
    else:
        pr0 = group_r + spec["offset"]
        pc0 = group_c
        rr0 = group_r
        rc0 = group_c + spec["pat_w"] + spec["gap"]
    pattern_bbox = (pr0, pc0, pr0 + spec["pat_h"] - 1, pc0 + spec["pat_w"] - 1)
    rect_bbox = (rr0, rc0, rr0 + spec["rect_h"] - 1, rc0 + spec["rect_w"] - 1)
    group_bbox = (group_r, group_c, group_r + spec["group_h"] - 1, group_c + spec["group_w"] - 1)
    return rect_bbox, pattern_bbox, group_bbox


def _place_groups(height: int, width: int, specs: list[dict]) -> list[dict] | None:
    placed: list[dict] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    order = list(range(len(specs)))
    random.shuffle(order)
    for idx in order:
        spec = specs[idx]
        candidates: list[tuple[int, int, tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]]] = []
        for gr in range(height - spec["group_h"] + 1):
            for gc in range(width - spec["group_w"] + 1):
                rect_bbox, pattern_bbox, group_bbox = _derive_bboxes(spec, gr, gc)
                if (
                    rect_bbox[0] in used_rows
                    or rect_bbox[2] in used_rows
                    or rect_bbox[1] in used_cols
                    or rect_bbox[3] in used_cols
                ):
                    continue
                if any(_bbox_distance(group_bbox, old["group_bbox"]) < GROUP_GAP for old in placed):
                    continue
                candidates.append((gr, gc, rect_bbox, pattern_bbox, group_bbox))
        if not candidates:
            return None
        gr, gc, rect_bbox, pattern_bbox, group_bbox = random.choice(candidates)
        used_rows.update([rect_bbox[0], rect_bbox[2]])
        used_cols.update([rect_bbox[1], rect_bbox[3]])
        placed.append(
            {
                "index": idx,
                "group_r": gr,
                "group_c": gc,
                "rect_bbox": rect_bbox,
                "pattern_bbox": pattern_bbox,
                "group_bbox": group_bbox,
                **spec,
            }
        )
    placed.sort(key=lambda rec: rec["index"])
    return placed


def _seed_positions(rect_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect_bbox
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _render_input(height: int, width: int, placed: list[dict]) -> tuple[np.ndarray, list[dict]]:
    grid = np.zeros((height, width), dtype=int)
    records: list[dict] = []
    for rec in placed:
        pattern, roles = _make_pattern_patch(rec["pat_h"], rec["pat_w"], rec["triplet"])
        pr0, pc0, pr1, pc1 = rec["pattern_bbox"]
        grid[pr0 : pr1 + 1, pc0 : pc1 + 1] = pattern

        r0, c0, r1, c1 = rec["rect_bbox"]
        for rr, cc in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
            grid[rr, cc] = MARKER

        seed = random.choice(_seed_positions(rec["rect_bbox"]))
        seed_color = roles[rec["seed_role"]]
        grid[seed] = int(seed_color)

        records.append(
            {
                "rect_bbox": rec["rect_bbox"],
                "pattern_bbox": rec["pattern_bbox"],
                "seed": seed,
                "seed_role": rec["seed_role"],
                "center_color": roles["center"],
                "odd_color": roles["odd"],
                "even_color": roles["even"],
                "triplet": rec["triplet"],
                "pattern_counts": Counter(map(int, pattern.flatten())),
            }
        )
    return grid, records


def _expected_output(inp: np.ndarray, records: list[dict]) -> np.ndarray:
    out = inp.copy()
    for rec in records:
        r0, c0, r1, c1 = rec["rect_bbox"]
        sr, sc = rec["seed"]
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if out[r, c] != BACKGROUND:
                    continue
                dist = max(abs(r - sr), abs(c - sc))
                out[r, c] = rec["odd_color"] if dist % 2 == 1 else rec["even_color"]
    return out


def _semantic_checks(records: list[dict]) -> bool:
    triplets = [rec["triplet"] for rec in records]
    if len(set(triplets)) != len(triplets):
        return False
    pattern_boxes = [rec["pattern_bbox"] for rec in records]
    rect_boxes = [rec["rect_bbox"] for rec in records]
    for rec in records:
        counts = rec["pattern_counts"]
        if len(counts) != 3 or len(set(counts.values())) != 3:
            return False
        if rec["seed_role"] not in {"center", "odd", "even"}:
            return False
    for pat_idx, pat_box in enumerate(pattern_boxes):
        distances = [_bbox_distance(pat_box, rect_box) for rect_box in rect_boxes]
        best = min(distances)
        winners = [idx for idx, dist in enumerate(distances) if dist == best]
        if winners != [pat_idx]:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_group_specs()
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    placed = _place_groups(height, width, specs)
    if placed is None:
        return None
    inp, records = _render_input(height, width, placed)
    if not _semantic_checks(records):
        return None
    expected = _expected_output(inp, records)
    try:
        scene = _scene(inp)
    except AssertionError:
        return None
    if len(scene) != len(records):
        return None
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
