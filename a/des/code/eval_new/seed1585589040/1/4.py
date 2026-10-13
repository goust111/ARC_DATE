import random
from collections import Counter, deque

import numpy as np


BG = 0
RED = 2
GREEN = 3
YELLOW = 4

SYMMETRIES = ("horizontal", "vertical", "rot180")
MARK_COLORS = {"horizontal": RED, "vertical": GREEN, "rot180": YELLOW}

PATTERN_COLORS = (1, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 400
OBJECT_ATTEMPTS = 150
PACK_ATTEMPTS = 120


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _partner(r: int, c: int, h: int, w: int, kind: str) -> tuple[int, int]:
    if kind == "horizontal":
        return h - 1 - r, c
    if kind == "vertical":
        return r, w - 1 - c
    return h - 1 - r, w - 1 - c


def _orbits(h: int, w: int, kind: str) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            rr, cc = _partner(r, c, h, w, kind)
            if (rr, cc) == (r, c):
                orbit = ((r, c),)
            else:
                orbit = tuple(sorted(((r, c), (rr, cc))))
            seen.update(orbit)
            out.append(orbit)
    return out


def _mismatch_pairs(region: np.ndarray, kind: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = region.shape
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for r in range(h):
        for c in range(w):
            rr, cc = _partner(r, c, h, w, kind)
            if (rr, cc) == (r, c):
                continue
            pair = tuple(sorted(((r, c), (rr, cc))))
            if pair in seen:
                continue
            seen.add(pair)
            if int(region[pair[0]]) != int(region[pair[1]]):
                pairs.append(pair)
    return pairs


def _solve_region(region: np.ndarray) -> tuple[np.ndarray, str, list[tuple[int, int]], dict[str, int]]:
    region = np.asarray(region, dtype=int)
    scores = {kind: len(_mismatch_pairs(region, kind)) for kind in SYMMETRIES}
    best = min(SYMMETRIES, key=lambda kind: (scores[kind], SYMMETRIES.index(kind)))
    counts = Counter(int(v) for v in region.flat)

    marked: list[tuple[int, int]] = []
    out = region.copy()
    for p1, p2 in _mismatch_pairs(region, best):
        c1 = int(region[p1])
        c2 = int(region[p2])
        n1 = counts[c1]
        n2 = counts[c2]
        if n1 < n2:
            target = p1
        elif n2 < n1:
            target = p2
        else:
            target = min(p1, p2)
        out[target] = MARK_COLORS[best]
        marked.append(target)
    return out, best, marked, scores


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark the locally corrupted cell in each almost-symmetric rectangular pattern region.

    Input structure:
    - The background is black.
    - Several non-overlapping filled rectangles of non-black pixels appear on the grid.
    - Each rectangle is a multicolor pattern that was originally symmetric by a horizontal mirror, a vertical mirror, or a 180-degree rotation, but 1-3 pixels were altered.

    Rule:
    - Split the non-black pixels into separate rectangular regions.
    - For each region, compare horizontal mirroring, vertical mirroring, and 180-degree rotation; pick the symmetry with the fewest mismatched cell pairs as the intended one.
    - In every mismatched pair for that intended symmetry, mark the rarer color occurrence in the whole region: use red for horizontal-mirror corruption, green for vertical-mirror corruption, and yellow for 180-degree-rotation corruption.
    - Leave every other cell unchanged, including the uncorrupted partner of each mismatch.

    Key insight:
    - The task is not to restore symmetry, but to infer which symmetry the rectangle was closest to and then highlight only the locally inconsistent side of each broken pair.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    regions = []
    for comp in _components(g != BG):
        r0, c0, r1, c1 = _bbox(comp)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(sub != BG), "each foreground component must be a filled rectangle"
        regions.append((r0, c0, r1, c1))

    for r0, c0, r1, c1 in sorted(regions):
        solved, _kind, _marked, _scores = _solve_region(g[r0 : r1 + 1, c0 : c1 + 1])
        out[r0 : r1 + 1, c0 : c1 + 1] = solved
    return out


def _build_pristine_region(kind: str, h: int, w: int, colors: list[int], rng: random.Random) -> np.ndarray | None:
    region = np.full((h, w), colors[0], dtype=int)
    orbits = _orbits(h, w, kind)
    for orbit in orbits:
        color = rng.choice(colors)
        for r, c in orbit:
            region[r, c] = color

    if len(set(int(v) for v in region.flat)) < 2:
        return None

    other_scores = [len(_mismatch_pairs(region, other)) for other in SYMMETRIES if other != kind]
    if min(other_scores) == 0:
        return None
    return region


def _generate_object(kind: str, rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(OBJECT_ATTEMPTS):
        h = rng.randint(5, 8)
        w = rng.randint(5, 8)
        base_count = rng.choice((2, 3))
        base_colors = rng.sample(list(PATTERN_COLORS), base_count)
        pristine = _build_pristine_region(kind, h, w, base_colors, rng)
        if pristine is None:
            continue

        pristine_counts = Counter(int(v) for v in pristine.flat)
        heavy_orbits = []
        for orbit in _orbits(h, w, kind):
            if len(orbit) != 2:
                continue
            color = int(pristine[orbit[0]])
            if pristine_counts[color] >= 4:
                heavy_orbits.append((orbit, color))
        free_colors = [c for c in PATTERN_COLORS if c not in base_colors]
        max_tamper = min(3, len(heavy_orbits), len(free_colors))
        if max_tamper == 0:
            continue

        tamper_count = rng.randint(1, max_tamper)
        rng.shuffle(heavy_orbits)
        tampered = pristine.copy()
        tampered_positions: list[tuple[int, int]] = []
        for (orbit, _color), new_color in zip(heavy_orbits[:tamper_count], rng.sample(free_colors, tamper_count)):
            target = rng.choice(orbit)
            tampered[target] = new_color
            tampered_positions.append(target)

        solved, inferred, marks, scores = _solve_region(tampered)
        if inferred != kind:
            continue
        if any(scores[other] <= scores[kind] for other in SYMMETRIES if other != kind):
            continue
        if set(marks) != set(tampered_positions):
            continue

        expected = tampered.copy()
        for r, c in tampered_positions:
            expected[r, c] = MARK_COLORS[kind]
        if not np.array_equal(solved, expected):
            continue
        return tampered, expected
    return None


def _split_nonnegative(total: int, parts: int, rng: random.Random) -> list[int]:
    out = [0] * parts
    for _ in range(total):
        out[rng.randrange(parts)] += 1
    return out


def _pack_objects(objects: list[np.ndarray], rng: random.Random) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    indices = list(range(len(objects)))
    for _ in range(PACK_ATTEMPTS):
        rng.shuffle(indices)
        content_limit = rng.randint(max(obj.shape[1] for obj in objects), 24)
        rows: list[list[int]] = []
        current: list[int] = []
        current_width = 0
        for idx in indices:
            ow = objects[idx].shape[1]
            needed = ow if not current else current_width + 1 + ow
            if current and needed > content_limit:
                rows.append(current)
                current = [idx]
                current_width = ow
            else:
                current.append(idx)
                current_width = needed
        if current:
            rows.append(current)

        row_heights = [max(objects[idx].shape[0] for idx in row) for row in rows]
        row_base_widths = [sum(objects[idx].shape[1] for idx in row) + len(row) - 1 + 2 for row in rows]
        base_h = sum(row_heights) + len(rows) - 1 + 2
        base_w = max(row_base_widths)
        if base_h > 30 or base_w > 30:
            continue

        h = rng.randint(base_h, 30)
        w = rng.randint(base_w, 30)
        extra_y = _split_nonnegative(h - base_h, len(rows) + 1, rng)
        row_gaps = [1 + v for v in extra_y]

        positions = [(-1, -1)] * len(objects)
        y = row_gaps[0]
        for row_index, row in enumerate(rows):
            row_width = row_base_widths[row_index]
            extra_x = _split_nonnegative(w - row_width, len(row) + 1, rng)
            col_gaps = [1 + v for v in extra_x]
            x = col_gaps[0]
            row_h = row_heights[row_index]
            for item_index, idx in enumerate(row):
                obj = objects[idx]
                oh, ow = obj.shape
                dy = rng.randint(0, row_h - oh)
                positions[idx] = (y + dy, x)
                x += ow + col_gaps[item_index + 1]
            y += row_h
            if row_index + 1 < len(rows):
                y += row_gaps[row_index + 1]

        return (h, w), positions
    return None


def _generate_example(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    object_count = rng.choice((3, 4))
    kinds = ["horizontal", "vertical", "rot180"]
    if object_count == 4:
        kinds.append(rng.choice(kinds))
    rng.shuffle(kinds)

    inputs: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    for kind in kinds:
        built = _generate_object(kind, rng)
        if built is None:
            return None
        inp_obj, out_obj = built
        inputs.append(inp_obj)
        outputs.append(out_obj)

    packed = _pack_objects(inputs, rng)
    if packed is None:
        return None
    (h, w), positions = packed
    grid = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)

    for obj_in, obj_out, (r0, c0) in zip(inputs, outputs, positions):
        oh, ow = obj_in.shape
        grid[r0 : r0 + oh, c0 : c0 + ow] = obj_in
        expected[r0 : r0 + oh, c0 : c0 + ow] = obj_out

    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        built = _generate_example(rng)
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate valid input")
