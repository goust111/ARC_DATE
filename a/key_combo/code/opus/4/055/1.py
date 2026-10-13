from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
LOOP_COLOR = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SPRITE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GEN_ATTEMPTS = 400
PATCH_ATTEMPTS = 120
PACK_ATTEMPTS = 120
SPRITE_ATTEMPTS = 80


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _trace_cycle(coords: np.ndarray) -> list[tuple[int, int]]:
    pts = {tuple(map(int, p)) for p in coords}
    for r, c in pts:
        degree = sum((nr, nc) in pts for nr, nc in _neighbors4(r, c))
        assert degree == 2, (r, c, degree)

    start = min(pts)
    neighbors = sorted((nr, nc) for nr, nc in _neighbors4(*start) if (nr, nc) in pts)
    assert len(neighbors) == 2

    path = [start]
    prev = start
    cur = neighbors[0]
    while cur != start:
        path.append(cur)
        nxt = [(nr, nc) for nr, nc in _neighbors4(*cur) if (nr, nc) in pts and (nr, nc) != prev]
        assert len(nxt) == 1, (cur, nxt)
        prev, cur = cur, nxt[0]
        assert len(path) <= len(pts)

    assert len(path) == len(pts)
    return path


def _concave_count(coords: np.ndarray) -> int:
    path = _trace_cycle(coords)
    counts = Counter()
    n = len(path)
    for i, (r, c) in enumerate(path):
        pr, pc = path[i - 1]
        nr, nc = path[(i + 1) % n]
        incoming = (r - pr, c - pc)
        outgoing = (nr - r, nc - c)
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        if cross:
            counts[1 if cross > 0 else -1] += 1
    assert counts[1] + counts[-1] >= 4
    assert abs(counts[1] - counts[-1]) == 4, counts
    return min(counts[1], counts[-1])


def _interior_mask(loop_mask: np.ndarray) -> np.ndarray:
    h, w = loop_mask.shape
    padded = np.pad(loop_mask.astype(bool), 1, constant_values=False)
    open_mask = ~padded
    seen = np.zeros_like(open_mask, dtype=bool)
    q = deque([(0, 0)])
    seen[0, 0] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h + 2 and 0 <= nc < w + 2 and open_mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return open_mask[1:-1, 1:-1] & ~seen[1:-1, 1:-1]


def _erode4(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if all(0 <= nr < h and 0 <= nc < w and mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                out[r, c] = True
    return out


def _sprite_crop_from_patch(patch: np.ndarray) -> np.ndarray:
    mask = (patch != BG) & (patch != LOOP_COLOR)
    comps = _components_from_mask(mask)
    assert len(comps) == 1
    return _crop(patch, _bbox_coords(comps[0]))


def _extract_loop_records(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    loop_comps = _components_from_mask(g == LOOP_COLOR)
    assert loop_comps

    sprite_seen = np.zeros_like(g, dtype=bool)
    records = []
    for coords in loop_comps:
        bbox = _bbox_coords(coords)
        local_grid = _crop(g, bbox)
        local_loop = local_grid == LOOP_COLOR
        interior = _interior_mask(local_loop)
        assert interior.any()

        sprite_mask = interior & (local_grid != BG) & (local_grid != LOOP_COLOR)
        sprite_comps = _components_from_mask(sprite_mask)
        assert len(sprite_comps) == 1
        sprite_coords = sprite_comps[0]
        sprite_vals = {int(local_grid[r, c]) for r, c in sprite_coords.tolist()}
        assert len(sprite_vals) == 1
        sprite_color = next(iter(sprite_vals))
        assert sprite_color not in (BG, LOOP_COLOR)
        assert 3 <= sprite_coords.shape[0] <= 7

        global_sprite = sprite_coords + np.array([bbox[0], bbox[1]])
        sprite_seen[global_sprite[:, 0], global_sprite[:, 1]] = True
        sprite_bbox = _bbox_coords(sprite_coords)
        records.append(
            {
                "bbox": bbox,
                "perimeter": int(coords.shape[0]),
                "concave": _concave_count(coords),
                "sprite_color": sprite_color,
                "sprite_area": int(sprite_coords.shape[0]),
                "sprite_crop": _crop(local_grid, sprite_bbox),
            }
        )

    assert np.array_equal(sprite_seen, (g != BG) & (g != LOOP_COLOR))
    return records


def _assemble_output(records: list[dict]) -> np.ndarray:
    keep = [rec for rec in records if rec["concave"] > 0]
    assert len(keep) >= 2
    counts = [rec["concave"] for rec in keep]
    assert len(set(counts)) == len(counts)
    keep.sort(key=lambda rec: rec["concave"])

    max_h = max(rec["sprite_crop"].shape[0] for rec in keep)
    total_w = sum(rec["sprite_crop"].shape[1] for rec in keep) + len(keep) - 1
    out = np.zeros((max_h, total_w), dtype=int)
    cursor = 0
    for rec in keep:
        crop = rec["sprite_crop"]
        h, w = crop.shape
        out[max_h - h : max_h, cursor : cursor + w] = crop
        cursor += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Keep only sprites inside concave grey loops, then sort and repack them by the loops' concave-corner counts.

    Input structure:
    - The grid has a black background, 4-6 disconnected grey one-pixel orthogonal loops, and one small monochrome 4-connected sprite inside each loop.
    - Every grey loop is a 4-connected closed cycle whose pixels all have degree 2, and loops do not touch or nest.
    - At least two loops are convex rectangles with zero concave corners, while at least two other loops have positive and pairwise distinct concave-corner counts.

    Rule:
    - Trace each grey loop and inspect turns along the cycle; straight pixels are ignored, and bend direction majority/minority determines convex versus concave turns.
    - Count the concave turns of each loop and discard every loop whose concave count is zero together with its interior sprite.
    - Crop each remaining sprite to its own tight bounding box, sort the kept sprites by their containing loop's concave count in ascending order, and place them left-to-right on a new black canvas.
    - Bottom-align the cropped sprites and separate consecutive sprites by exactly one black column.

    Key insight:
    - The same loop-tracing result serves twice: the minority turn count decides both which loops survive and the final left-to-right order of their cropped interior sprites.
    """
    records = _extract_loop_records(np.asarray(grid, dtype=int))
    return _assemble_output(records)


def _append_segment(path: list[tuple[int, int]], r0: int, c0: int, r1: int, c1: int) -> None:
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        cells = [(r0, c) for c in range(c0, c1 + step, step)]
    else:
        step = 1 if r1 >= r0 else -1
        cells = [(r, c0) for r in range(r0, r1 + step, step)]
    if path and cells and path[-1] == cells[0]:
        cells = cells[1:]
    path.extend(cells)


def _path_from_profiles(top: list[int], bottom: list[int]) -> list[tuple[int, int]]:
    width = len(top)
    path: list[tuple[int, int]] = []
    _append_segment(path, top[0], 0, top[0], 0)
    for c in range(width - 1):
        _append_segment(path, top[c], c, top[c], c + 1)
        _append_segment(path, top[c], c + 1, top[c + 1], c + 1)
    _append_segment(path, top[-1], width - 1, bottom[-1], width - 1)
    for c in range(width - 1, 0, -1):
        _append_segment(path, bottom[c], c, bottom[c], c - 1)
        _append_segment(path, bottom[c], c - 1, bottom[c - 1], c - 1)
    _append_segment(path, bottom[0], 0, top[0], 0)
    if path and path[-1] == path[0]:
        path.pop()
    return path


def _path_from_vertices(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    for (r0, c0), (r1, c1) in zip(vertices, vertices[1:]):
        _append_segment(path, r0, c0, r1, c1)
    if path and path[-1] == path[0]:
        path.pop()
    return path


def _random_connected_subset(candidate_mask: np.ndarray, area: int) -> np.ndarray | None:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(candidate_mask)]
    if len(cells) < area:
        return None
    for _ in range(SPRITE_ATTEMPTS):
        start = random.choice(cells)
        chosen = {start}
        while len(chosen) < area:
            frontier = []
            for r, c in chosen:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < candidate_mask.shape[0] and 0 <= nc < candidate_mask.shape[1]:
                        if candidate_mask[nr, nc] and (nr, nc) not in chosen:
                            frontier.append((nr, nc))
            if not frontier:
                break
            chosen.add(random.choice(frontier))
        if len(chosen) == area:
            coords = np.array(sorted(chosen), dtype=int)
            return coords
    return None


def _make_loop_patch(notch_count: int, color: int) -> tuple[np.ndarray, int] | None:
    for _ in range(PATCH_ATTEMPTS):
        min_h = 6 if notch_count == 0 else 7
        height = random.randint(min_h, 9)
        if notch_count == 0:
            min_w, max_w = 6, 9
        elif notch_count == 1:
            min_w, max_w = 8, 14
        else:
            min_w, max_w = 12, 14
        width = random.randint(min_w, max_w)

        top = [0] * width
        bottom = [height - 1] * width
        if notch_count > 0:
            intervals: list[tuple[int, int]] = []
            for _interval_attempt in range(60):
                intervals = []
                for _k in range(notch_count):
                    span = 3
                    start = random.randint(2, width - span - 2)
                    end = start + span - 1
                    if all(end + 2 < prev_start or start > prev_end + 2 for prev_start, prev_end in intervals):
                        intervals.append((start, end))
                if len(intervals) == notch_count:
                    break
            if len(intervals) != notch_count:
                continue
            max_depth = min(3, height - 4)
            if max_depth < 2:
                continue
            notches = [(start, end, random.randint(2, max_depth)) for start, end in sorted(intervals)]
        else:
            notches = []

        if notch_count == 0:
            vertices = [(0, 0), (0, width - 1), (height - 1, width - 1), (height - 1, 0), (0, 0)]
        else:
            vertices = [(0, 0)]
            for start, end, depth in notches:
                vertices.append((0, start))
                vertices.append((depth, start))
                vertices.append((depth, end))
                vertices.append((0, end))
            vertices.extend([(0, width - 1), (height - 1, width - 1), (height - 1, 0), (0, 0)])

        path = _path_from_vertices(vertices)
        coords = np.array(path, dtype=int)
        patch_mask = np.zeros((height, width), dtype=bool)
        patch_mask[coords[:, 0], coords[:, 1]] = True
        comps = _components_from_mask(patch_mask)
        if len(comps) != 1:
            continue
        if _concave_count(comps[0]) != 2 * notch_count:
            continue

        interior = _interior_mask(patch_mask)
        if len(_components_from_mask(interior)) != 1:
            continue
        candidate = _erode4(interior)
        if int(candidate.sum()) < 3:
            candidate = interior
        if int(candidate.sum()) < 3:
            continue

        sprite_area = random.randint(3, min(7, int(candidate.sum())))
        sprite_coords = _random_connected_subset(candidate, sprite_area)
        if sprite_coords is None:
            continue

        patch = np.zeros((height, width), dtype=int)
        patch[patch_mask] = LOOP_COLOR
        patch[sprite_coords[:, 0], sprite_coords[:, 1]] = int(color)

        for _ in range(random.randrange(4)):
            patch = np.rot90(patch)
        if random.random() < 0.5:
            patch = np.fliplr(patch)

        records = _extract_loop_records(patch)
        if len(records) != 1 or records[0]["concave"] != 2 * notch_count:
            continue
        return patch, 2 * notch_count
    return None


def _pack_patches(patches: list[np.ndarray]) -> list[tuple[int, int]] | None:
    count = len(patches)
    for _ in range(PACK_ATTEMPTS):
        row_count = 2 if count <= 4 else random.choice([2, 3])
        row_items = [[] for _ in range(row_count)]
        row_widths = [0] * row_count
        row_heights = [0] * row_count

        order = list(range(count))
        random.shuffle(order)
        for idx in order:
            h, w = patches[idx].shape
            ranked = sorted(range(row_count), key=lambda r: (row_widths[r], row_heights[r], random.random()))
            row = ranked[0]
            row_items[row].append(idx)
            row_widths[row] += w + (2 if len(row_items[row]) > 1 else 0)
            row_heights[row] = max(row_heights[row], h)

        req_w = max(row_widths)
        req_h = sum(row_heights) + 2 * (row_count - 1)
        if req_w > 26 or req_h > 26:
            continue

        grid_h = req_h + random.randint(2, min(4, 30 - req_h))
        grid_w = req_w + random.randint(2, min(4, 30 - req_w))
        if grid_h > 30 or grid_w > 30:
            continue

        positions: list[tuple[int, int] | None] = [None] * count
        extra_h = grid_h - req_h
        row_gap_extras = [0] * (row_count + 1)
        for _ in range(extra_h):
            row_gap_extras[random.randrange(row_count + 1)] += 1

        r_cursor = row_gap_extras[0]
        ok = True
        for row in range(row_count):
            items = row_items[row]
            if not items:
                r_cursor += row_heights[row] + 2 + row_gap_extras[row + 1]
                continue
            base_w = row_widths[row]
            extra_w = grid_w - base_w
            gaps = [0] * (len(items) + 1)
            for _ in range(extra_w):
                gaps[random.randrange(len(gaps))] += 1

            c_cursor = gaps[0]
            for i, idx in enumerate(items):
                patch = patches[idx]
                h, w = patch.shape
                r_off = random.randint(0, row_heights[row] - h)
                positions[idx] = (r_cursor + r_off, c_cursor)
                c_cursor += w
                if i + 1 < len(items):
                    c_cursor += 2 + gaps[i + 1]
            r_cursor += row_heights[row]
            if row + 1 < row_count:
                r_cursor += 2 + row_gap_extras[row + 1]

        if ok and all(pos is not None for pos in positions):
            return [(int(r), int(c)) for r, c in positions]
    return None


def _expected_from_meta(meta: list[dict]) -> np.ndarray:
    return _assemble_output(
        [
            {
                "concave": int(rec["concave"]),
                "sprite_crop": rec["sprite_crop"].copy(),
            }
            for rec in meta
        ]
    )


def _meta_records_for_grid(patches: list[np.ndarray], concaves: list[int], positions: list[tuple[int, int]]) -> list[dict]:
    meta = []
    for patch, concave, (r0, c0) in zip(patches, concaves, positions):
        h, w = patch.shape
        meta.append(
            {
                "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
                "concave": int(concave),
                "shape_area": int(h * w),
                "sprite_crop": _sprite_crop_from_patch(patch),
            }
        )
    return meta


def _passes_semantic_checks(grid: np.ndarray, meta: list[dict]) -> bool:
    try:
        actual = main(grid)
    except AssertionError:
        return False

    expected = _expected_from_meta(meta)
    if not np.array_equal(actual, expected):
        return False

    try:
        parsed = _extract_loop_records(grid)
    except AssertionError:
        return False

    parsed_counts = sorted(rec["concave"] for rec in parsed)
    meta_counts = sorted(int(rec["concave"]) for rec in meta)
    if parsed_counts != meta_counts:
        return False

    zeros = sum(rec["concave"] == 0 for rec in parsed)
    positives = [rec["concave"] for rec in parsed if rec["concave"] > 0]
    if not (4 <= len(parsed) <= 6 and zeros >= 2 and len(positives) >= 2 and len(set(positives)) == len(positives)):
        return False

    sprite_colors = set(int(rec["sprite_color"]) for rec in parsed)
    if len(sprite_colors) != len(parsed):
        return False

    selected = [rec for rec in meta if rec["concave"] > 0]
    conc_order = tuple(id(rec) for rec in sorted(selected, key=lambda rec: rec["concave"]))
    xpos_order = tuple(id(rec) for rec in sorted(selected, key=lambda rec: (rec["bbox"][1], rec["bbox"][0])))
    area_order = tuple(id(rec) for rec in sorted(selected, key=lambda rec: (rec["shape_area"], rec["bbox"][1], rec["bbox"][0])))
    if conc_order == xpos_order or conc_order == area_order:
        return False

    return True


def _generate_candidate() -> np.ndarray | None:
    total_loops = random.randint(4, 6)
    concave_loops = 2
    notch_counts = [1, 2]
    notch_counts += [0] * (total_loops - concave_loops)
    random.shuffle(notch_counts)

    colors = random.sample(SPRITE_COLORS, total_loops)
    patches = []
    concaves = []
    for notch_count, color in zip(notch_counts, colors):
        built = _make_loop_patch(notch_count, color)
        if built is None:
            return None
        patch, concave = built
        patches.append(patch)
        concaves.append(concave)

    positions = _pack_patches(patches)
    if positions is None:
        return None

    max_r = max(r + patch.shape[0] for patch, (r, _c) in zip(patches, positions))
    max_c = max(c + patch.shape[1] for patch, (_r, c) in zip(patches, positions))
    grid = np.zeros((max_r, max_c), dtype=int)
    for patch, (r0, c0) in zip(patches, positions):
        h, w = patch.shape
        view = grid[r0 : r0 + h, c0 : c0 + w]
        if np.any((view != BG) & (patch != BG)):
            return None
        view[patch != BG] = patch[patch != BG]

    meta = _meta_records_for_grid(patches, concaves, positions)
    if not _passes_semantic_checks(grid, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid loop-sorting ARC input")
