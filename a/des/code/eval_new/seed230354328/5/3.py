import random
from collections import Counter, deque

import numpy as np


BG = 0
FENCE = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400

SHAPES_BY_SIZE = {
    2: (
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
    ),
    3: (
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
    ),
    4: (
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 1), (1, 1), (2, 1), (2, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
    ),
    5: (
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
        ((0, 2), (1, 1), (1, 2), (2, 0), (2, 1)),
    ),
}


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _sorted_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted((int(r), int(c)) for r, c in coords)


def _project_coords(
    coords: list[tuple[int, int]],
    side: str,
    fence_index: int,
    window_pos: int,
) -> list[tuple[int, int]]:
    pts = _sorted_coords(coords)
    if side == "right":
        edge_col = max(c for _, c in pts)
        anchor_row = min(r for r, c in pts if c == edge_col)
        return [
            (window_pos + (r - anchor_row), fence_index + 1 + (edge_col - c))
            for r, c in pts
        ]
    if side == "left":
        edge_col = min(c for _, c in pts)
        anchor_row = min(r for r, c in pts if c == edge_col)
        return [
            (window_pos + (r - anchor_row), fence_index - 1 - (c - edge_col))
            for r, c in pts
        ]
    if side == "bottom":
        edge_row = max(r for r, _ in pts)
        anchor_col = min(c for r, c in pts if r == edge_row)
        return [
            (fence_index + 1 + (edge_row - r), window_pos + (c - anchor_col))
            for r, c in pts
        ]
    if side == "top":
        edge_row = min(r for r, _ in pts)
        anchor_col = min(c for r, c in pts if r == edge_row)
        return [
            (fence_index - 1 - (r - edge_row), window_pos + (c - anchor_col))
            for r, c in pts
        ]
    raise ValueError(f"unknown side: {side}")


def _in_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = point
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _line_segments(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in spans]


def _extract_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _mode_color(g)

    h_fences = [r for r in range(h) if np.all(g[r, :] != bg)]
    v_fences = [c for c in range(w) if np.all(g[:, c] != bg)]
    assert h_fences or v_fences, "expected at least one fence line"
    assert all(b - a >= 2 for a, b in zip(h_fences, h_fences[1:])), "horizontal fences must be separated"
    assert all(b - a >= 2 for a, b in zip(v_fences, v_fences[1:])), "vertical fences must be separated"

    fence_cells: list[int] = []
    for r in h_fences:
        fence_cells.extend(int(v) for v in g[r, :])
    for c in v_fences:
        fence_cells.extend(int(v) for v in g[:, c])
    fence_color = Counter(fence_cells).most_common(1)[0][0]
    assert fence_color != bg

    row_spans: list[tuple[int, int]] = []
    prev = -1
    for r in h_fences + [h]:
        if prev + 1 <= r - 1:
            row_spans.append((prev + 1, r - 1))
        prev = r
    col_spans: list[tuple[int, int]] = []
    prev = -1
    for c in v_fences + [w]:
        if prev + 1 <= c - 1:
            col_spans.append((prev + 1, c - 1))
        prev = c
    assert row_spans and col_spans

    h_windows: dict[int, list[tuple[int, int]]] = {}
    for r in h_fences:
        windows = [(r, c) for c in range(w) if g[r, c] not in (bg, fence_color)]
        assert 1 <= len(windows) <= 2, "each horizontal fence must have one or two windows"
        assert all(c not in v_fences for _, c in windows), "windows cannot sit on fence intersections"
        h_windows[r] = windows

    v_windows: dict[int, list[tuple[int, int]]] = {}
    for c in v_fences:
        windows = [(r, c) for r in range(h) if g[r, c] not in (bg, fence_color)]
        assert 1 <= len(windows) <= 2, "each vertical fence must have one or two windows"
        assert all(r not in h_fences for r, _ in windows), "windows cannot sit on fence intersections"
        v_windows[c] = windows

    regions: dict[tuple[int, int], dict] = {}
    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            bbox = (r0, c0, r1, c1)
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            comps = _components(sub != bg)
            if comps:
                assert len(comps) == 1, "each region may contain at most one seed"
                rel = comps[0]
                coords = [(r0 + rr, c0 + cc) for rr, cc in rel]
                colors = {int(g[r, c]) for r, c in coords}
                assert len(colors) == 1
                seed_color = next(iter(colors))
                assert seed_color not in (bg, fence_color)
                seed = {
                    "coords": _sorted_coords(coords),
                    "bbox": _bbox(coords),
                    "color": int(seed_color),
                    "size": len(coords),
                }
            else:
                seed = None
            regions[(i, j)] = {"bbox": bbox, "seed": seed}

    return {
        "grid": g,
        "bg": int(bg),
        "fence_color": int(fence_color),
        "h_fences": h_fences,
        "v_fences": v_fences,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "h_windows": h_windows,
        "v_windows": v_windows,
        "regions": regions,
    }


def _projection_tasks(structure: dict) -> list[dict]:
    row_spans = structure["row_spans"]
    col_spans = structure["col_spans"]
    h_fences = structure["h_fences"]
    v_fences = structure["v_fences"]
    h_windows = structure["h_windows"]
    v_windows = structure["v_windows"]
    regions = structure["regions"]

    tasks: list[dict] = []
    for (i, j), region in regions.items():
        seed = region["seed"]
        if seed is None:
            continue
        r0, c0, r1, c1 = region["bbox"]

        if i > 0:
            fence_r = h_fences[i - 1]
            for _, wc in h_windows[fence_r]:
                if c0 <= wc <= c1:
                    target = regions[(i - 1, j)]["bbox"]
                    tasks.append(
                        {
                            "seed": seed,
                            "side": "top",
                            "fence_index": fence_r,
                            "window_pos": wc,
                            "window_color": int(structure["grid"][fence_r, wc]),
                            "target_bbox": target,
                        }
                    )
        if i < len(row_spans) - 1:
            fence_r = h_fences[i]
            for _, wc in h_windows[fence_r]:
                if c0 <= wc <= c1:
                    target = regions[(i + 1, j)]["bbox"]
                    tasks.append(
                        {
                            "seed": seed,
                            "side": "bottom",
                            "fence_index": fence_r,
                            "window_pos": wc,
                            "window_color": int(structure["grid"][fence_r, wc]),
                            "target_bbox": target,
                        }
                    )
        if j > 0:
            fence_c = v_fences[j - 1]
            for wr, _ in v_windows[fence_c]:
                if r0 <= wr <= r1:
                    target = regions[(i, j - 1)]["bbox"]
                    tasks.append(
                        {
                            "seed": seed,
                            "side": "left",
                            "fence_index": fence_c,
                            "window_pos": wr,
                            "window_color": int(structure["grid"][wr, fence_c]),
                            "target_bbox": target,
                        }
                    )
        if j < len(col_spans) - 1:
            fence_c = v_fences[j]
            for wr, _ in v_windows[fence_c]:
                if r0 <= wr <= r1:
                    target = regions[(i, j + 1)]["bbox"]
                    tasks.append(
                        {
                            "seed": seed,
                            "side": "right",
                            "fence_index": fence_c,
                            "window_pos": wr,
                            "window_color": int(structure["grid"][wr, fence_c]),
                            "target_bbox": target,
                        }
                    )

    side_rank = {"top": 0, "left": 1, "bottom": 2, "right": 3}
    tasks.sort(
        key=lambda task: (
            task["seed"]["size"],
            task["seed"]["bbox"][0],
            task["seed"]["bbox"][1],
            side_rank[task["side"]],
            task["fence_index"],
            task["window_pos"],
        )
    )
    return tasks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each seed pattern through colored fence windows into neighboring regions.

    Input structure:
    - The background is black.
    - One-cell-thick grey full rows and full columns partition the grid into
      rectangular regions.
    - Each fence line contains one or two colored window pixels that replace grey
      cells on that line.
    - Some regions contain one small monochrome connected seed pattern, while the
      remaining regions stay empty.

    Rule:
    - Keep the fences, windows, and original seed patterns unchanged.
    - Every seed projects through each adjacent window on a bordering fence into
      the region on the other side of that fence.
    - Crossing a horizontal fence flips the seed vertically; crossing a vertical
      fence flips it horizontally. The projected copy takes the window's color.
    - The reflected copy is anchored by the window: the seed edge closest to the
      fence is placed flush to the opposite side of that fence, while offsets
      along the fence are preserved relative to the seed's boundary-facing edge.
      Larger seeds are painted later and therefore overwrite smaller projected
      copies where they overlap.

    Key insight:
    - Window pixels are both portals and color labels, while the fence orientation
      determines which mirror flip to apply to the transmitted seed.
    """
    structure = _extract_structure(np.asarray(grid, dtype=int))
    g = structure["grid"]
    out = g.copy()

    for task in _projection_tasks(structure):
        coords = _project_coords(
            task["seed"]["coords"],
            task["side"],
            task["fence_index"],
            task["window_pos"],
        )
        assert all(_in_bbox(pt, task["target_bbox"]) for pt in coords), "projection must stay inside the target region"
        assert all(int(g[r, c]) == structure["bg"] for r, c in coords), "projections may only land on empty cells"
        for r, c in coords:
            out[r, c] = task["window_color"]
    return out


def _sample_region_sizes(rng: random.Random, parts: int) -> list[int]:
    while True:
        sizes = [rng.randint(5, 6) for _ in range(parts)]
        if sum(sizes) + (parts - 1) <= 30:
            return sizes


def _build_spans(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int]]:
    spans: list[tuple[int, int]] = []
    fences: list[int] = []
    start = 0
    for index, size in enumerate(sizes):
        spans.append((start, start + size - 1))
        start += size
        if index != len(sizes) - 1:
            fences.append(start)
            start += 1
    return spans, fences


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _place_shape(
    rng: random.Random,
    shape: tuple[tuple[int, int], ...],
    bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    shape_h, shape_w = _shape_dims(shape)
    top_min = r0 + 1
    top_max = r1 - shape_h
    left_min = c0 + 1
    left_max = c1 - shape_w
    if top_min > top_max:
        top_min, top_max = r0, r1 - shape_h + 1
    if left_min > left_max:
        left_min, left_max = c0, c1 - shape_w + 1
    if top_min > top_max or left_min > left_max:
        raise ValueError("shape does not fit in region")
    top = rng.randint(top_min, top_max)
    left = rng.randint(left_min, left_max)
    return _sorted_coords([(top + dr, left + dc) for dr, dc in shape])


def _candidate_windows(
    coords: list[tuple[int, int]],
    side: str,
    fence_index: int,
    target_bbox: tuple[int, int, int, int],
    segment_span: tuple[int, int],
) -> list[tuple[int, frozenset[tuple[int, int]]]]:
    lo, hi = segment_span
    candidates: list[tuple[int, frozenset[tuple[int, int]]]] = []
    for pos in range(lo, hi + 1):
        projected = _project_coords(coords, side, fence_index, pos)
        if all(_in_bbox(pt, target_bbox) for pt in projected):
            candidates.append((pos, frozenset(projected)))
    return candidates


def _pick_overlap_pair(
    rng: random.Random,
    seed_infos: list[dict],
) -> dict[int, tuple[int, frozenset[tuple[int, int]]]] | None:
    pairs: list[tuple[int, int, int, int]] = []
    for a in range(len(seed_infos)):
        for b in range(a + 1, len(seed_infos)):
            if seed_infos[a]["size"] == seed_infos[b]["size"]:
                continue
            for ia, (_, proj_a) in enumerate(seed_infos[a]["candidates"]):
                for ib, (_, proj_b) in enumerate(seed_infos[b]["candidates"]):
                    if proj_a & proj_b:
                        pairs.append((a, b, ia, ib))
    if not pairs:
        return None
    a, b, ia, ib = rng.choice(pairs)
    return {
        a: seed_infos[a]["candidates"][ia],
        b: seed_infos[b]["candidates"][ib],
    }


def _pick_disjoint_windows(
    rng: random.Random,
    seed_infos: list[dict],
    preset: dict[int, tuple[int, frozenset[tuple[int, int]]]] | None = None,
) -> dict[int, tuple[int, frozenset[tuple[int, int]]]] | None:
    chosen = dict(preset or {})
    used: set[tuple[int, int]] = set()
    for _, proj in chosen.values():
        used |= set(proj)

    order = list(range(len(seed_infos)))
    rng.shuffle(order)
    order.sort(key=lambda idx: len(seed_infos[idx]["candidates"]))

    for idx in order:
        if idx in chosen:
            continue
        options = [
            cand
            for cand in seed_infos[idx]["candidates"]
            if not (set(cand[1]) & used)
        ]
        if not options:
            return None
        pos, proj = rng.choice(options)
        chosen[idx] = (pos, proj)
        used |= set(proj)
    return chosen


def _line_key_and_segment(seed_region: tuple[int, int], side: str, h_fences: list[int], v_fences: list[int]) -> tuple[tuple[str, int], int]:
    i, j = seed_region
    if side == "top":
        return ("h", h_fences[i - 1]), j
    if side == "bottom":
        return ("h", h_fences[i]), j
    if side == "left":
        return ("v", v_fences[j - 1]), i
    if side == "right":
        return ("v", v_fences[j]), i
    raise ValueError(side)


def _adjacent_segments(region: tuple[int, int], row_count: int, col_count: int, h_fences: list[int], v_fences: list[int]) -> list[tuple[tuple[str, int], int]]:
    i, j = region
    out: list[tuple[tuple[str, int], int]] = []
    if i > 0:
        out.append((("h", h_fences[i - 1]), j))
    if i < row_count - 1:
        out.append((("h", h_fences[i]), j))
    if j > 0:
        out.append((("v", v_fences[j - 1]), i))
    if j < col_count - 1:
        out.append((("v", v_fences[j]), i))
    return out


def _random_color(rng: random.Random, forbid: set[int]) -> int:
    choices = [color for color in PALETTE if color not in forbid]
    return int(rng.choice(choices))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    row_count = rng.choice((3, 4))
    col_count = rng.choice((3, 4))
    row_sizes = _sample_region_sizes(rng, row_count)
    col_sizes = _sample_region_sizes(rng, col_count)
    row_spans, h_fences = _build_spans(row_sizes)
    col_spans, v_fences = _build_spans(col_sizes)
    height = row_spans[-1][1] + 1
    width = col_spans[-1][1] + 1

    interior = [(i, j) for i in range(1, row_count - 1) for j in range(1, col_count - 1)]
    if not interior:
        return None
    hub = rng.choice(interior)
    force_overlap = rng.random() < 0.5

    side_to_region = {
        "top": (hub[0] - 1, hub[1]),
        "bottom": (hub[0] + 1, hub[1]),
        "left": (hub[0], hub[1] - 1),
        "right": (hub[0], hub[1] + 1),
    }

    chosen_sides = [rng.choice(("top", "bottom")), rng.choice(("left", "right"))]
    remaining = [side for side in ("top", "bottom", "left", "right") if side not in chosen_sides]
    rng.shuffle(remaining)
    chosen_sides.extend(remaining[: rng.randint(0, len(remaining))])
    chosen_sides = sorted(set(chosen_sides), key=("top", "bottom", "left", "right").index)

    seed_regions = [side_to_region[side] for side in chosen_sides]
    seed_colors = rng.sample(PALETTE, len(seed_regions))
    sizes = rng.sample(sorted(SHAPES_BY_SIZE), len(seed_regions))

    region_bboxes = {
        (i, j): (row_spans[i][0], col_spans[j][0], row_spans[i][1], col_spans[j][1])
        for i in range(row_count)
        for j in range(col_count)
    }

    seed_infos: list[dict] = []
    for side, region, color, size in zip(chosen_sides, seed_regions, seed_colors, sizes, strict=False):
        shape = rng.choice(SHAPES_BY_SIZE[size])
        coords = _place_shape(rng, shape, region_bboxes[region])
        target_bbox = region_bboxes[hub]
        line_key, segment_index = _line_key_and_segment(region, {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}[side], h_fences, v_fences)
        if line_key[0] == "h":
            fence_index = line_key[1]
            segment_span = col_spans[segment_index]
            proj_side = "bottom" if side == "top" else "top"
            candidates = _candidate_windows(coords, proj_side, fence_index, target_bbox, segment_span)
        else:
            fence_index = line_key[1]
            segment_span = row_spans[segment_index]
            proj_side = "right" if side == "left" else "left"
            candidates = _candidate_windows(coords, proj_side, fence_index, target_bbox, segment_span)
        if not candidates:
            return None
        seed_infos.append(
            {
                "source_side": side,
                "region": region,
                "coords": coords,
                "color": int(color),
                "size": int(size),
                "proj_side": proj_side,
                "line_key": line_key,
                "segment_index": int(segment_index),
                "target_bbox": target_bbox,
                "candidates": candidates,
            }
        )

    if force_overlap:
        preset = _pick_overlap_pair(rng, seed_infos)
        if preset is None:
            return None
        chosen = _pick_disjoint_windows(rng, seed_infos, preset=preset)
        if chosen is None:
            return None
    else:
        chosen = _pick_disjoint_windows(rng, seed_infos)
        if chosen is None:
            return None

    grid = np.full((height, width), BG, dtype=int)
    for fence_r in h_fences:
        grid[fence_r, :] = FENCE
    for fence_c in v_fences:
        grid[:, fence_c] = FENCE

    reserved_windows: dict[tuple[str, int], list[tuple[int, int, int]]] = {}
    allowed_segments: dict[tuple[str, int], set[int]] = {}
    blacklisted: dict[tuple[str, int], set[int]] = {}
    for fence_r in h_fences:
        key = ("h", fence_r)
        allowed_segments[key] = set(range(col_count))
        blacklisted[key] = set()
    for fence_c in v_fences:
        key = ("v", fence_c)
        allowed_segments[key] = set(range(row_count))
        blacklisted[key] = set()

    seed_region_set = set(seed_regions)
    for region in seed_region_set:
        for line_key, segment_index in _adjacent_segments(region, row_count, col_count, h_fences, v_fences):
            blacklisted[line_key].add(segment_index)

    for info_index, info in enumerate(seed_infos):
        line_key = info["line_key"]
        segment_index = info["segment_index"]
        blacklisted[line_key].discard(segment_index)
        pos, _ = chosen[info_index]
        if line_key[0] == "h":
            row = line_key[1]
            col = pos
            window_color = _random_color(rng, {info["color"]})
            reserved_windows.setdefault(line_key, []).append((row, col, window_color))
        else:
            row = pos
            col = line_key[1]
            window_color = _random_color(rng, {info["color"]})
            reserved_windows.setdefault(line_key, []).append((row, col, window_color))

    for line_key, items in reserved_windows.items():
        for r, c, color in items:
            grid[r, c] = color

    for info in seed_infos:
        for r, c in info["coords"]:
            grid[r, c] = info["color"]

    for line_key, blocked_segments in blacklisted.items():
        allowed_segments[line_key] -= blocked_segments

    occupied_windows = {line_key: {(r, c) for r, c, _ in reserved_windows.get(line_key, [])} for line_key in allowed_segments}

    for line_key, segments in allowed_segments.items():
        current = len(reserved_windows.get(line_key, []))
        target = rng.choice((1, 2))
        target = max(target, current)
        if current == target:
            continue
        span_list = _line_segments(col_spans if line_key[0] == "h" else row_spans)
        available_segments = list(segments)
        if not available_segments:
            return None
        while current < target:
            segment_index = rng.choice(available_segments)
            lo, hi = span_list[segment_index]
            if line_key[0] == "h":
                fence_r = line_key[1]
                candidates = [(fence_r, c) for c in range(lo, hi + 1) if (fence_r, c) not in occupied_windows[line_key]]
            else:
                fence_c = line_key[1]
                candidates = [(r, fence_c) for r in range(lo, hi + 1) if (r, fence_c) not in occupied_windows[line_key]]
            if not candidates:
                return None
            r, c = rng.choice(candidates)
            color = _random_color(rng, set())
            grid[r, c] = color
            reserved_windows.setdefault(line_key, []).append((r, c, color))
            occupied_windows[line_key].add((r, c))
            current += 1

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    if force_overlap:
        projected_cells = out.copy()
        projected_cells[grid != BG] = BG
        if np.count_nonzero(projected_cells) == sum(info["size"] for info in seed_infos):
            return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
