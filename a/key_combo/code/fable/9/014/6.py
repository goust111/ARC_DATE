import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
DOT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_of_color(grid: np.ndarray, color: int, *, connectivity: int = 8) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    neighbors = _neighbors8 if connectivity == 8 else _neighbors4
    components = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != int(color):
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            r0, c0, r1, c1 = _bbox(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            components.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "coords": coords,
                }
            )

    components.sort(key=lambda comp: comp["bbox"])
    return components


def _scanline_fill_positions(mask: np.ndarray) -> list[tuple[int, int]]:
    fill: list[tuple[int, int]] = []
    for r in range(mask.shape[0]):
        cols = np.flatnonzero(mask[r])
        assert len(cols) % 2 == 0, "each outline row must contain an even number of outline pixels"
        for i in range(0, len(cols), 2):
            left, right = int(cols[i]), int(cols[i + 1])
            for c in range(left + 1, right):
                fill.append((r, c))
    return fill


def _scanline_area(mask: np.ndarray) -> int:
    return len(_scanline_fill_positions(mask))


def _outer_span_area(mask: np.ndarray) -> int:
    total = 0
    for r in range(mask.shape[0]):
        cols = np.flatnonzero(mask[r])
        if len(cols) >= 2:
            total += int(cols[-1]) - int(cols[0]) - 1
    return total


def _flood_interior_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    queue = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))

    return int(((~mask) & (~seen)).sum())


def _safe_fill_positions(mask: np.ndarray) -> list[tuple[int, int]]:
    safe: list[tuple[int, int]] = []
    for r, c in _scanline_fill_positions(mask):
        blocked = False
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                blocked = True
                break
        if not blocked:
            safe.append((r, c))
    return safe


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _make_rectangle(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _make_multi(kind: str, extra_rows: int) -> np.ndarray:
    if kind == "A":
        rows = [
            "11111111",
            "10011001",
            "10100101",
            *["10000001"] * (1 + extra_rows),
            "11111111",
        ]
    else:
        rows = [
            "11111111",
            "10100101",
            *["10000001"] * (1 + extra_rows),
            "10100101",
            "11111111",
        ]
    return _mask_from_rows(rows)


def _transform_mask(mask: np.ndarray, code: int) -> np.ndarray:
    if code == 0:
        return mask
    if code == 1:
        return np.fliplr(mask)
    if code == 2:
        return np.flipud(mask)
    if code == 3:
        return np.flipud(np.fliplr(mask))
    raise ValueError(code)


def _build_shape_library() -> list[dict]:
    library: list[dict] = []
    seen = set()

    bases = [
        ("rect", _make_rectangle(4, 6)),
        ("rect", _make_rectangle(5, 6)),
        ("rect", _make_rectangle(6, 6)),
        ("rect", _make_rectangle(5, 8)),
        ("rect", _make_rectangle(6, 8)),
    ]
    for kind in ("A", "B"):
        for extra in range(3):
            bases.append((f"multi_{kind}", _make_multi(kind, extra)))

    for family, base in bases:
        for code in range(4):
            mask = _transform_mask(base, code)
            key = mask.tobytes()
            if key in seen:
                continue
            seen.add(key)
            fill = _scanline_fill_positions(mask)
            library.append(
                {
                    "family": family,
                    "mask": mask,
                    "area": len(fill),
                    "fill": fill,
                    "safe_fill": _safe_fill_positions(mask),
                    "multi_span": any(np.count_nonzero(mask[r]) >= 4 for r in range(mask.shape[0])),
                    "outer_diff": _outer_span_area(mask) != len(fill),
                    "flood_diff": _flood_interior_count(mask) != len(fill),
                }
            )

    return library


SHAPE_LIBRARY = _build_shape_library()


def _inspect_shapes(grid: np.ndarray) -> list[dict]:
    return _components_of_color(grid, GREY, connectivity=8)


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each grey outline with the dot color whose global count matches its scanline-paired area.

    Input structure:
    - A black background contains several grey outline shapes and several colors of isolated single-pixel dots.
    - Outline shapes may be concave and may have rows containing 4 or more grey pixels, so one row can encode multiple fill spans.
    - Some dots can lie inside a shape's scanline-filled region as decoys, but they remain visible dots rather than becoming outline pixels.

    Rule:
    - For each grey outline separately, scan each row left to right and pair outline pixels as (1st,2nd), (3rd,4th), and so on.
    - Fill only the cells strictly between each paired outline-pixel pair, count how many cells this scanline rule selects, and find the unique dot color whose total number of dots in the whole grid equals that count.
    - Recolor the selected fill cells to that color while leaving the grey outline pixels and all pre-existing colored dots unchanged.

    Key insight:
    - The deciding signal is not which dot happens to sit inside a shape, and not a flood-filled interior; it is the row-wise paired scanline area matched against global dot counts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    outlines = _inspect_shapes(g)
    assert outlines, "expected at least one grey outline"

    dot_counts = Counter(int(v) for v in g.flatten() if int(v) not in (BLACK, GREY))
    count_to_colors: dict[int, list[int]] = {}
    for color, count in dot_counts.items():
        count_to_colors.setdefault(int(count), []).append(int(color))

    out = g.copy()
    for comp in outlines:
        fill = _scanline_fill_positions(comp["mask"])
        area = len(fill)
        assert area in count_to_colors and len(count_to_colors[area]) == 1, "scanline area must match one dot color count"
        fill_color = int(count_to_colors[area][0])
        r0, c0, _r1, _c1 = comp["bbox"]
        for rr, cc in fill:
            gr, gc = r0 + rr, c0 + cc
            if out[gr, gc] == BLACK:
                out[gr, gc] = fill_color
            else:
                assert out[gr, gc] != GREY, "fill cells must not overlap outline pixels"
    return out


def _candidate_dot_counts(used_counts: set[int], limit: int) -> list[int]:
    return [n for n in range(1, limit + 1) if n not in used_counts]


def _choose_shapes() -> list[dict]:
    n_shapes = random.choice([2, 3])
    mandatory_pool = [
        shape
        for shape in SHAPE_LIBRARY
        if shape["family"].startswith("multi_")
        and shape["multi_span"]
        and shape["outer_diff"]
        and len(shape["safe_fill"]) >= 2
    ]
    chosen = [random.choice(mandatory_pool)]
    used_areas = {chosen[0]["area"]}

    pool = [shape for shape in SHAPE_LIBRARY if shape["area"] not in used_areas]
    while len(chosen) < n_shapes:
        shape = random.choice(pool)
        chosen.append(shape)
        used_areas.add(shape["area"])
        pool = [cand for cand in pool if cand["area"] not in used_areas]

    return chosen


def _can_place_shape(rect_occ: np.ndarray, mask: np.ndarray, r0: int, c0: int, padding: int) -> bool:
    h, w = mask.shape
    hh, ww = rect_occ.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    rs = max(0, r0 - padding)
    cs = max(0, c0 - padding)
    re = min(hh, r0 + h + padding)
    ce = min(ww, c0 + w + padding)
    return not bool(rect_occ[rs:re, cs:ce].any())


def _stamp_shape(grid: np.ndarray, rect_occ: np.ndarray, shape: dict, r0: int, c0: int) -> None:
    mask = shape["mask"]
    h, w = mask.shape
    grid[r0 : r0 + h, c0 : c0 + w][mask] = GREY
    rect_occ[r0 : r0 + h, c0 : c0 + w] = True


def _place_shapes(grid: np.ndarray, shapes: list[dict]) -> list[tuple[int, int]] | None:
    rect_occ = np.zeros_like(grid, dtype=bool)
    placements: list[tuple[int, int]] = []
    order = sorted(range(len(shapes)), key=lambda i: shapes[i]["mask"].size, reverse=True)
    by_index = [None] * len(shapes)

    for i in order:
        mask = shapes[i]["mask"]
        h, w = mask.shape
        candidates = [(r, c) for r in range(grid.shape[0] - h + 1) for c in range(grid.shape[1] - w + 1)]
        random.shuffle(candidates)
        placed = None
        for r0, c0 in candidates:
            if _can_place_shape(rect_occ, mask, r0, c0, padding=2):
                _stamp_shape(grid, rect_occ, shapes[i], r0, c0)
                placed = (r0, c0)
                break
        if placed is None:
            return None
        by_index[i] = placed

    return [(int(r), int(c)) for r, c in by_index]


def _dot_can_go_outside(grid: np.ndarray, blocked: np.ndarray, r: int, c: int, forbidden: np.ndarray) -> bool:
    if forbidden[r, c] or blocked[r, c] or grid[r, c] != BLACK:
        return False
    for nr, nc in _neighbors8(r, c):
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] != BLACK:
            return False
    return True


def _dot_can_go_inside(grid: np.ndarray, blocked: np.ndarray, r: int, c: int) -> bool:
    if blocked[r, c] or grid[r, c] != BLACK:
        return False
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] != BLACK:
            return False
    for nr, nc in _neighbors8(r, c):
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and blocked[nr, nc]:
            return False
    return True


def _mark_dot_blocked(blocked: np.ndarray, r: int, c: int) -> None:
    blocked[r, c] = True
    for nr, nc in _neighbors8(r, c):
        if 0 <= nr < blocked.shape[0] and 0 <= nc < blocked.shape[1]:
            blocked[nr, nc] = True


def _place_dots(
    grid: np.ndarray,
    shapes: list[dict],
    placements: list[tuple[int, int]],
    color_counts: dict[int, int],
    requests: list[tuple[int, int]],
) -> bool:
    blocked = np.zeros_like(grid, dtype=bool)
    forbidden = np.zeros_like(grid, dtype=bool)
    placed_per_color = Counter()

    for shape, (r0, c0) in zip(shapes, placements):
        h, w = shape["mask"].shape
        forbidden[r0 : r0 + h, c0 : c0 + w] = True

    used_inside: dict[int, set[tuple[int, int]]] = {i: set() for i in range(len(shapes))}

    for host_idx, color in requests:
        shape = shapes[host_idx]
        r0, c0 = placements[host_idx]
        candidates = [(r0 + rr, c0 + cc) for rr, cc in shape["safe_fill"] if (rr, cc) not in used_inside[host_idx]]
        random.shuffle(candidates)
        chosen = None
        for gr, gc in candidates:
            if _dot_can_go_inside(grid, blocked, gr, gc):
                chosen = (gr, gc)
                break
        if chosen is None:
            return False
        gr, gc = chosen
        grid[gr, gc] = color
        _mark_dot_blocked(blocked, gr, gc)
        used_inside[host_idx].add((gr - r0, gc - c0))
        placed_per_color[color] += 1

    for color, target in color_counts.items():
        remaining = target - placed_per_color[color]
        for _ in range(remaining):
            candidates = [
                (r, c)
                for r in range(grid.shape[0])
                for c in range(grid.shape[1])
                if _dot_can_go_outside(grid, blocked, r, c, forbidden)
            ]
            if not candidates:
                return False
            gr, gc = random.choice(candidates)
            grid[gr, gc] = color
            _mark_dot_blocked(blocked, gr, gc)

    return True


def _expected_output(grid: np.ndarray, shapes: list[dict], placements: list[tuple[int, int]], area_to_color: dict[int, int]) -> np.ndarray:
    out = grid.copy()
    for shape, (r0, c0) in zip(shapes, placements):
        fill_color = area_to_color[shape["area"]]
        for rr, cc in shape["fill"]:
            gr, gc = r0 + rr, c0 + cc
            if out[gr, gc] == BLACK:
                out[gr, gc] = fill_color
    return out


def _validate_semantics(
    grid: np.ndarray,
    out: np.ndarray,
    shapes: list[dict],
    placements: list[tuple[int, int]],
    area_to_color: dict[int, int],
    wrong_decoy: tuple[int, int] | None,
) -> bool:
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False

    comps = _inspect_shapes(grid)
    if len(comps) != len(shapes):
        return False

    areas = []
    saw_multi = False
    saw_shortcut_breaker = False
    for comp in comps:
        area = _scanline_area(comp["mask"])
        areas.append(area)
        row_counts = [int(np.count_nonzero(comp["mask"][r])) for r in range(comp["mask"].shape[0])]
        if any(count >= 4 for count in row_counts):
            saw_multi = True
        if _outer_span_area(comp["mask"]) != area or _flood_interior_count(comp["mask"]) != area:
            saw_shortcut_breaker = True
    if len(set(areas)) != len(areas):
        return False
    if not saw_multi or not saw_shortcut_breaker:
        return False

    dot_counts = Counter(int(v) for v in grid.flatten() if int(v) not in (BLACK, GREY))
    if len(dot_counts) < len(shapes):
        return False
    if len(set(dot_counts.values())) != len(dot_counts):
        return False

    for color in dot_counts:
        pts = np.argwhere(grid == color)
        for r, c in pts:
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == int(color):
                    return False

    if sorted(areas) != sorted(area_to_color):
        return False
    if not np.array_equal(main(grid), out):
        return False

    if wrong_decoy is not None:
        host_idx, wrong_color = wrong_decoy
        host_shape = shapes[host_idx]
        shortcut = out.copy()
        expected_color = area_to_color[host_shape["area"]]
        for rr, cc in host_shape["fill"]:
            gr, gc = placements[host_idx][0] + rr, placements[host_idx][1] + cc
            if shortcut[gr, gc] == expected_color:
                shortcut[gr, gc] = wrong_color
        if np.array_equal(shortcut, out):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(24, 30)
    width = random.randint(24, 30)
    grid = np.zeros((height, width), dtype=int)

    shapes = _choose_shapes()
    placements = _place_shapes(grid, shapes)
    if placements is None:
        return None

    matched_colors = random.sample(DOT_COLORS, len(shapes))
    area_to_color = {shape["area"]: color for shape, color in zip(shapes, matched_colors)}
    color_counts = {color: shape["area"] for shape, color in zip(shapes, matched_colors)}

    spare_colors = [c for c in DOT_COLORS if c not in matched_colors]
    spare_count = random.randint(0, min(2, len(spare_colors)))
    used_counts = set(color_counts.values())
    free_counts = _candidate_dot_counts(used_counts, limit=7)
    random.shuffle(free_counts)
    for color, count in zip(random.sample(spare_colors, spare_count), free_counts):
        color_counts[color] = count
        used_counts.add(count)

    requests: list[tuple[int, int]] = []
    wrong_decoy: tuple[int, int] | None = None
    host_idx = 0
    if len(shapes) >= 2 and random.random() < 0.75:
        donor_idx = random.choice([i for i in range(len(shapes)) if i != host_idx])
        wrong_color = matched_colors[donor_idx]
        requests.append((host_idx, wrong_color))
        wrong_decoy = (host_idx, wrong_color)
    if random.random() < 0.45:
        requests.append((host_idx, matched_colors[host_idx]))

    if len(requests) > len(shapes[host_idx]["safe_fill"]):
        return None

    if not _place_dots(grid, shapes, placements, color_counts, requests):
        return None

    out = _expected_output(grid, shapes, placements, area_to_color)
    if not _validate_semantics(grid, out, shapes, placements, area_to_color, wrong_decoy):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
