import random
from collections import defaultdict

import numpy as np


BG = 0
BRIDGE = 8
PALETTE = (1, 2, 3, 4)
GEN_ATTEMPTS = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds((h, w), nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _crop_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _is_vertically_symmetric(coords: list[tuple[int, int]]) -> bool:
    mask = _crop_mask(coords)
    return bool(np.array_equal(mask, np.fliplr(mask)))


def _analyze_input(grid: np.ndarray) -> dict[int, dict]:
    info: dict[int, dict] = {}
    for color in PALETTE:
        comps = _components_of_color(grid, color)
        assert comps, f"missing color {color}"
        regions = [comp for comp in comps if len(comp) > 1]
        markers = [comp[0] for comp in comps if len(comp) == 1]
        assert len(regions) == 1, f"color {color} must have exactly one multi-pixel region"
        assert 2 <= len(markers) <= 3, f"color {color} must have 2-3 markers"
        region = regions[0]
        region_bbox = _bbox(region)
        marker_rows = [r for r, _c in markers]
        assert len(marker_rows) == len(set(marker_rows)), f"color {color} marker rows must be unique"
        for mr, mc in markers:
            r0, c0, r1, c1 = region_bbox
            assert not (r0 <= mr <= r1 and c0 <= mc <= c1), f"marker inside bbox for color {color}"
        info[color] = {
            "color": color,
            "region": region,
            "region_bbox": region_bbox,
            "markers": sorted(markers),
            "symmetric": _is_vertically_symmetric(region),
        }
    assert sum(item["symmetric"] for item in info.values()) == 2, "expected exactly two symmetric colors"
    return info


def _erase_to_surviving_markers(grid: np.ndarray, info: dict[int, dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for color, item in info.items():
        if item["symmetric"]:
            for r, c in item["region"]:
                out[r, c] = BG
        else:
            out[out == color] = BG
    return out


def _surviving_markers(info: dict[int, dict]) -> list[tuple[int, int, int]]:
    markers = []
    for color, item in info.items():
        if item["symmetric"]:
            for r, c in item["markers"]:
                markers.append((r, c, color))
    rows = [r for r, _c, _color in markers]
    assert len(rows) == len(set(rows)), "all surviving marker rows must be globally unique"
    return sorted(markers)


def _make_frames(markers: list[tuple[int, int, int]], h: int) -> list[dict]:
    frames = []
    for idx, (row, col, color) in enumerate(markers):
        start = 0 if idx == 0 else (markers[idx - 1][0] + row) // 2 + 1
        end = h - 1 if idx == len(markers) - 1 else (row + markers[idx + 1][0]) // 2
        assert start <= row <= end
        frames.append(
            {
                "row": int(row),
                "col": int(col),
                "color": int(color),
                "start": int(start),
                "end": int(end),
            }
        )
    return frames


def _claim(claims: dict[tuple[int, int], int], r: int, c: int, color: int) -> None:
    key = (int(r), int(c))
    if key in claims and claims[key] != color:
        raise AssertionError(f"conflicting frame paints at {key}")
    claims[key] = color


def _frame_only_grid(base: np.ndarray, frames: list[dict]) -> np.ndarray:
    h, w = base.shape
    claims: dict[tuple[int, int], int] = {}
    out = np.asarray(base, dtype=int).copy()

    for frame in frames:
        color = int(frame["color"])
        row = int(frame["row"])
        start = int(frame["start"])
        end = int(frame["end"])

        for rr in range(start, end + 1):
            _claim(claims, rr, 0, color)
            _claim(claims, rr, w - 1, color)
        for cc in range(w):
            _claim(claims, row, cc, color)
        if start == 0:
            for cc in range(w):
                _claim(claims, 0, cc, color)
        if end == h - 1:
            for cc in range(w):
                _claim(claims, h - 1, cc, color)

    for (r, c), color in claims.items():
        out[r, c] = color
    return out


def _bridge_candidate(frame_grid: np.ndarray, upper: dict, lower: dict) -> tuple[int, tuple[int, int, int, int]] | None:
    h, w = frame_grid.shape
    _ = h
    assert int(upper["row"]) < int(lower["row"])
    if w < 5:
        return None
    r0 = int(upper["row"]) + 1
    r1 = int(lower["row"]) - 1
    if r0 > r1:
        return None
    if np.any(frame_grid[r0 : r1 + 1, 1 : w - 1] != BG):
        return None
    return r1 - r0 + 1, (r0, 1, r1, w - 2)


def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: list[int], a: int, b: int) -> None:
    ra = _find(parent, a)
    rb = _find(parent, b)
    if ra != rb:
        parent[ra] = rb


def _paint_bridges(frame_grid: np.ndarray, frames: list[dict]) -> np.ndarray:
    out = frame_grid.copy()
    by_color: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for idx, frame in enumerate(frames):
        by_color[int(frame["color"])].append((idx, frame))

    for color_frames in by_color.values():
        if len(color_frames) < 2:
            continue
        local = sorted(color_frames, key=lambda item: item[1]["row"])
        parent = list(range(len(local)))
        edges: list[tuple[int, int, int, tuple[int, int, int, int]]] = []
        for i in range(len(local)):
            for j in range(i + 1, len(local)):
                candidate = _bridge_candidate(frame_grid, local[i][1], local[j][1])
                if candidate is None:
                    continue
                dist, rect = candidate
                edges.append((dist, i, j, rect))
        for _dist, i, j, rect in sorted(edges):
            if _find(parent, i) == _find(parent, j):
                continue
            _union(parent, i, j)
            r0, c0, r1, c1 = rect
            out[r0 : r1 + 1, c0 : c1 + 1] = BRIDGE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter colors by symmetric region shape, expand only the surviving markers into band frames, then bridge the shortest same-color gaps.

    Input structure:
    - The background is black.
    - Colors 1, 2, 3, and 4 each contain one 4-connected multi-pixel region plus 2-3 isolated single-pixel markers.
    - Exactly two colors have a multi-pixel region whose cropped mask is invariant under left-right reflection.

    Rule:
    - Test each color's multi-pixel region for vertical mirror symmetry after cropping it to its bounding box.
    - Erase every pixel of each asymmetric color, then erase the multi-pixel regions of the symmetric colors while keeping only their markers.
    - Sort the surviving markers by row, split the canvas into horizontal bands at row midpoints, draw each marker's full-width band frame, and finally fill the shortest clear same-color gaps between surviving frames with teal.

    Key insight:
    - The larger regions never appear in the answer; they only decide which marker colors are allowed to generate the frame network that is later connected.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    info = _analyze_input(g)
    marker_grid = _erase_to_surviving_markers(g, info)
    markers = _surviving_markers(info)
    frames = _make_frames(markers, g.shape[0])
    frame_grid = _frame_only_grid(marker_grid, frames)
    return _paint_bridges(frame_grid, frames)


def _connected_mask(mask: np.ndarray) -> bool:
    coords = list(map(tuple, np.argwhere(mask)))
    if not coords:
        return False
    seen = {coords[0]}
    stack = [coords[0]]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    coords = list(map(tuple, np.argwhere(mask)))
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_connected_cells(h: int, w: int, target: int) -> set[tuple[int, int]]:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    return cells


def _make_symmetric_region() -> np.ndarray | None:
    for _ in range(120):
        h = random.randint(3, 6)
        w = random.choice((3, 5))
        target = random.randint(5, min(12, h * w))
        half_w = (w + 1) // 2
        center_c = w // 2
        selected = {(random.randrange(h), center_c)}
        area = 1

        while True:
            frontier = []
            for r, c in selected:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < half_w and (nr, nc) not in selected:
                        frontier.append((nr, nc))
            choices = []
            for unit in frontier:
                weight = 1 if unit[1] == center_c else 2
                if area + weight <= target:
                    choices.append((unit, weight))
            if not choices:
                break
            unit, weight = random.choice(choices)
            selected.add(unit)
            area += weight
            if area == target:
                break

        if area != target:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in selected:
            mask[r, c] = True
            mask[r, w - 1 - c] = True
        mask = _crop_bool(mask)
        if 5 <= int(mask.sum()) <= 12 and _connected_mask(mask) and np.array_equal(mask, np.fliplr(mask)):
            return mask
    return None


def _make_asymmetric_region() -> np.ndarray | None:
    for _ in range(150):
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        target = random.randint(5, min(12, h * w))
        cells = _random_connected_cells(h, w, target)
        if len(cells) != target:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_bool(mask)
        if 5 <= int(mask.sum()) <= 12 and _connected_mask(mask) and not np.array_equal(mask, np.fliplr(mask)):
            return mask
    return None


def _place_patch(grid: np.ndarray, patch: np.ndarray, occupied: np.ndarray) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    spots = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + ph + 1)
            cc1 = min(w, c0 + pw + 1)
            if np.any(occupied[rr0:rr1, cc0:cc1]):
                continue
            spots.append((r0, c0))
    if not spots:
        return None
    return random.choice(spots)


def _survivor_sequence(s1: int, n1: int, s2: int, n2: int, mode: str) -> list[int]:
    if mode == "bridge_both":
        seq = [s1] * n1 + [s2] * n2
        if random.random() < 0.5:
            seq.reverse()
        return seq
    if mode == "bridge_one":
        if random.random() < 0.5:
            s1, s2, n1, n2 = s2, s1, n2, n1
        seq = [s2]
        seq.extend([s1] * n1)
        seq.extend([s2] * (n2 - 1))
        return seq
    pool = [s1] * n1 + [s2] * n2
    out = []
    last = None
    while pool:
        choices = [c for c in sorted(set(pool)) if c != last]
        if not choices:
            choices = sorted(set(pool))
        color = random.choice(choices)
        pool.remove(color)
        out.append(color)
        last = color
    return out


def _interleave_all_markers(counts: dict[int, int], survivors: tuple[int, int], mode: str) -> list[int]:
    s1, s2 = survivors
    seq = _survivor_sequence(s1, counts[s1], s2, counts[s2], mode)
    eliminated = []
    for color in PALETTE:
        if color not in survivors:
            eliminated.extend([color] * counts[color])
    for color in eliminated:
        pos = random.randint(0, len(seq))
        seq.insert(pos, color)
    return seq


def _safe_columns(grid: np.ndarray, row: int, bboxes: list[tuple[int, int, int, int]], taken: set[tuple[int, int]]) -> list[int]:
    h, w = grid.shape
    cols = []
    for c in range(1, w - 1):
        if int(grid[row, c]) != BG or (row, c) in taken:
            continue
        inside_bbox = any(r0 <= row <= r1 and c0 <= c <= c1 for r0, c0, r1, c1 in bboxes)
        if inside_bbox:
            continue
        ok = True
        for nr, nc in _neighbors8(row, c):
            if _in_bounds((h, w), nr, nc) and (int(grid[nr, nc]) != BG or (nr, nc) in taken):
                ok = False
                break
        if ok:
            cols.append(c)
    return cols


def _pick_rows_with_gap(valid_rows: list[int], count: int, min_gap: int) -> list[int] | None:
    valid = sorted(valid_rows)
    for _ in range(160):
        chosen: list[int] = []
        pool = valid[:]
        while pool and len(chosen) < count:
            row = random.choice(pool)
            chosen.append(row)
            pool = [r for r in pool if all(abs(r - x) >= min_gap for x in chosen)]
        if len(chosen) == count:
            return sorted(chosen)
    return None


def _expected_bridge_count(frames: list[dict], frame_grid: np.ndarray) -> int:
    total = 0
    by_color: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by_color[int(frame["color"])].append(frame)
    for items in by_color.values():
        parent = list(range(len(items)))
        edges = []
        ordered = sorted(items, key=lambda item: item["row"])
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                cand = _bridge_candidate(frame_grid, ordered[i], ordered[j])
                if cand is not None:
                    dist, _rect = cand
                    edges.append((dist, i, j))
        for _dist, i, j in sorted(edges):
            if _find(parent, i) != _find(parent, j):
                _union(parent, i, j)
                total += 1
    return total


def _validate_output(inp: np.ndarray, out: np.ndarray, mode: str) -> bool:
    info = _analyze_input(inp)
    marker_grid = _erase_to_surviving_markers(inp, info)
    if any(np.any(out == color) for color, item in info.items() if not item["symmetric"]):
        return False
    for color, item in info.items():
        if item["symmetric"] and any(out[r, c] == color for r, c in item["region"]):
            return False
    frames = _make_frames(_surviving_markers(info), inp.shape[0])
    frame_grid = _frame_only_grid(marker_grid, frames)
    if not np.array_equal(out[out != BRIDGE], frame_grid[out != BRIDGE]):
        return False
    bridge_count = _expected_bridge_count(frames, frame_grid)
    if bridge_count == 0 and np.any(out == BRIDGE):
        return False
    if bridge_count > 0 and not np.any(out == BRIDGE):
        return False
    if mode == "bridge_both" and bridge_count < 2:
        return False
    if mode == "bridge_one" and bridge_count < 1:
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(20, 26)
    w = random.randint(16, 24)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    survivors = tuple(sorted(random.sample(PALETTE, 2)))
    mode = random.choices(("bridge_both", "bridge_one", "mixed"), weights=(3, 3, 1), k=1)[0]

    region_masks: dict[int, np.ndarray] = {}
    for color in PALETTE:
        mask = _make_symmetric_region() if color in survivors else _make_asymmetric_region()
        if mask is None:
            return None
        region_masks[color] = mask

    placements: dict[int, tuple[int, int]] = {}
    bboxes: list[tuple[int, int, int, int]] = []
    for color in PALETTE:
        patch = np.where(region_masks[color], color, BG).astype(int)
        loc = _place_patch(grid, patch, occupied)
        if loc is None:
            return None
        r0, c0 = loc
        ph, pw = patch.shape
        mask = patch != BG
        grid[r0 : r0 + ph, c0 : c0 + pw][mask] = color
        occupied[r0 : r0 + ph, c0 : c0 + pw] |= mask
        placements[color] = (r0, c0)
        coords = [(r0 + r, c0 + c) for r, c in map(tuple, np.argwhere(mask))]
        bboxes.append(_bbox(coords))

    counts = {color: random.choice((2, 3)) for color in PALETTE}
    total_markers = sum(counts.values())
    valid_rows = [r for r in range(1, h - 1) if _safe_columns(grid, r, bboxes, set())]
    rows = _pick_rows_with_gap(valid_rows, total_markers, 2)
    if rows is None:
        return None

    order = _interleave_all_markers(counts, survivors, mode)
    assert len(order) == len(rows)
    taken: set[tuple[int, int]] = set()
    for row, color in zip(rows, order):
        cols = _safe_columns(grid, row, bboxes, taken)
        if not cols:
            return None
        col = random.choice(cols)
        grid[row, col] = color
        taken.add((row, col))

    info = _analyze_input(grid)
    if tuple(sorted(color for color, item in info.items() if item["symmetric"])) != survivors:
        return None

    out = main(grid)
    if not _validate_output(grid, out, mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
