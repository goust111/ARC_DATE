from __future__ import annotations

import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_BG = 0
GEN_GREY = 5
GEN_TEMPLATE = 9
GEN_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8]
GEN_DIRS = ["up", "right", "down", "left"]
GEN_STEPS = {"up": 1, "right": 2, "down": 3, "left": 4}
GEN_RETRY_LIMIT = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _normalize(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0 = min(rs)
    c0 = min(cs)
    norm = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, int(h), int(w)


def _centroid(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[float, float]:
    n = float(len(coords))
    return (
        sum(r for r, _ in coords) / n,
        sum(c for _, c in coords) / n,
    )


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _separator_rows(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    by_color: dict[int, list[int]] = {}
    for r in range(grid.shape[0]):
        if np.all(grid[r] == grid[r, 0]):
            color = int(grid[r, 0])
            by_color.setdefault(color, []).append(int(r))
    candidates = [(color, rows) for color, rows in by_color.items() if color != int(bg) and len(rows) == 2]
    assert len(candidates) == 1, f"expected one separator color with two full-width rows, got {candidates}"
    color, rows = candidates[0]
    rows = sorted(rows)
    return int(color), int(rows[0]), int(rows[1])


def _read_mapping(top: np.ndarray, bg: int, sep_color: int) -> dict[int, int]:
    if top.shape[0] == 0:
        raise AssertionError("expected a top palette region")
    counts = [int(np.count_nonzero((top[r] != int(bg)) & (top[r] != int(sep_color)))) for r in range(top.shape[0])]
    row = top[int(np.argmax(counts))]
    mapping: dict[int, int] = {}
    c = 0
    while c < top.shape[1] - 1:
        a = int(row[c])
        b = int(row[c + 1])
        if a not in (int(bg), int(sep_color)) and b not in (int(bg), int(sep_color)):
            mapping[a] = b
            c += 2
        else:
            c += 1
    assert mapping, "expected at least one palette pair"
    return mapping


def _template_color(middle: np.ndarray, bg: int) -> int:
    colors = [int(v) for v in np.unique(middle) if int(v) != int(bg)]
    assert len(colors) == 1, f"expected a single template color, got {colors}"
    return int(colors[0])


def _extract_holes(middle: np.ndarray, bg: int, template_color: int) -> list[dict]:
    holes = []
    h, w = middle.shape
    for comp in _components(middle == int(bg)):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        comp_set = set(comp)
        for r, c in comp:
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if (nr, nc) in comp_set:
                    continue
                assert int(middle[nr, nc]) == int(template_color), "hole must be enclosed by the template color"
        shape, sh, sw = _normalize(comp)
        holes.append(
            {
                "coords": comp,
                "shape": shape,
                "size": (sh, sw),
                "centroid": _centroid(comp),
            }
        )
    assert holes, "expected at least one enclosed hole"
    return holes


def _extract_fragments(bottom: np.ndarray, bg: int) -> list[dict]:
    frags = []
    for comp in _components(bottom != int(bg)):
        colors = {int(bottom[r, c]) for r, c in comp}
        assert len(colors) == 1, "each fragment should be monochrome"
        color = int(next(iter(colors)))
        shape, sh, sw = _normalize(comp)
        frags.append(
            {
                "coords": comp,
                "shape": shape,
                "size": (sh, sw),
                "centroid": _centroid(comp),
                "color": color,
            }
        )
    assert frags, "expected at least one bottom fragment"
    return frags


def _direction(hole_center: tuple[float, float], frag_center: tuple[float, float]) -> str:
    dr = float(frag_center[0] - hole_center[0])
    dc = float(frag_center[1] - hole_center[1])
    eps = 1e-9
    if abs(dc) <= eps and abs(dr) > eps:
        return "up" if dr < 0 else "down"
    if abs(dr) <= eps and abs(dc) > eps:
        return "left" if dc < 0 else "right"
    if abs(dr) >= abs(dc):
        return "up" if dr < 0 else "down"
    return "left" if dc < 0 else "right"


def _apply_mapping(color: int, mapping: dict[int, int], steps: int) -> int:
    out = int(color)
    for _ in range(int(steps)):
        out = int(mapping.get(out, out))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match bottom-band fragments to template holes and fill each hole with a direction-dependent mapped color.

    Input structure:
    - The background is the dominant color, and two full-width separator rows split the grid into a top palette band,
      a middle template band, and a bottom fragment band.
    - The top band contains adjacent color pairs; the left pixel of each pair maps to the right pixel.
    - The middle band contains one large template color region with enclosed background holes.
    - The bottom band contains monochrome connected fragments whose shapes exactly match the hole shapes by translation.

    Rule:
    - Read the palette pairs into a color-permutation dictionary.
    - Find each enclosed hole in the middle band and match it with the fragment of the same normalized shape in the bottom band.
    - Compare the matched hole and fragment centroids in their own band-local coordinates: above applies the permutation once,
      right twice, below three times, and left four times; then fill the hole with the transformed fragment color.
    - Erase the palette band, separator rows, and bottom fragments by returning only the filled middle band.

    Key insight:
    The shape match identifies which fragment belongs to which hole, but the final fill color is determined only after
    interpreting the fragment's relative direction as a number of palette applications.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    sep_color, r1, r2 = _separator_rows(g, bg)

    top = g[:r1]
    middle = g[r1 + 1 : r2].copy()
    bottom = g[r2 + 1 :]

    mapping = _read_mapping(top, bg, sep_color)
    template_color = _template_color(middle, bg)
    holes = _extract_holes(middle, bg, template_color)
    fragments = _extract_fragments(bottom, bg)

    hole_by_shape: dict[tuple[tuple[int, int], ...], list[dict]] = {}
    frag_by_shape: dict[tuple[tuple[int, int], ...], list[dict]] = {}
    for hole in holes:
        hole_by_shape.setdefault(hole["shape"], []).append(hole)
    for frag in fragments:
        frag_by_shape.setdefault(frag["shape"], []).append(frag)

    assert set(hole_by_shape) == set(frag_by_shape), "hole and fragment shape sets must match"

    out = middle.copy()
    for shape, hole_group in hole_by_shape.items():
        frag_group = frag_by_shape[shape]
        assert len(hole_group) == len(frag_group) == 1, "generator should provide one-to-one shape matches"
        hole = hole_group[0]
        frag = frag_group[0]
        direction = _direction(hole["centroid"], frag["centroid"])
        fill = _apply_mapping(int(frag["color"]), mapping, GEN_STEPS[direction])
        for r, c in hole["coords"]:
            out[int(r), int(c)] = int(fill)
    return out


def _random_shape() -> tuple[tuple[tuple[int, int], ...], int, int] | None:
    for _ in range(80):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        max_area = min(6, h * w - 1)
        if max_area < 3:
            continue
        area = random.randint(3, max_area)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((int(nr), int(nc)))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        shape, sh, sw = _normalize(tuple(cells))
        if sh < 2 or sw < 2 or len(shape) == sh * sw:
            continue
        return shape, sh, sw
    return None


def _absolute_cells(shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + r), int(c0 + c)) for r, c in shape]


def _can_place(cells: list[tuple[int, int]], blocked: np.ndarray) -> bool:
    h, w = blocked.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if bool(blocked[r, c]):
            return False
    return True


def _mark(cells: list[tuple[int, int]], blocked: np.ndarray) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _build_layout(
    band_h: int,
    band_w: int,
    template_bbox: tuple[int, int, int, int],
    shapes: list[tuple[tuple[tuple[int, int], ...], int, int]],
    directions: list[str],
) -> list[dict] | None:
    tr0, tc0, tr1, tc1 = template_bbox
    hole_blocked = np.zeros((band_h, band_w), dtype=bool)
    frag_blocked = np.zeros((band_h, band_w), dtype=bool)
    placed: list[dict] = []

    order = list(range(len(shapes)))
    random.shuffle(order)
    for idx in order:
        shape, sh, sw = shapes[idx]
        direction = directions[idx]
        deltas = list(range(2, 5))
        random.shuffle(deltas)
        placed_here = False
        for delta in deltas:
            r_min = tr0 + 1
            r_max = tr1 - sh
            c_min = tc0 + 1
            c_max = tc1 - sw
            if direction == "up":
                r_min = max(r_min, delta)
            elif direction == "down":
                r_max = min(r_max, band_h - sh - delta)
            elif direction == "left":
                c_min = max(c_min, delta)
            else:
                c_max = min(c_max, band_w - sw - delta)
            if r_min > r_max or c_min > c_max:
                continue

            candidates = [(r, c) for r in range(r_min, r_max + 1) for c in range(c_min, c_max + 1)]
            random.shuffle(candidates)
            for hr, hc in candidates:
                if direction == "up":
                    fr, fc = hr - delta, hc
                elif direction == "down":
                    fr, fc = hr + delta, hc
                elif direction == "left":
                    fr, fc = hr, hc - delta
                else:
                    fr, fc = hr, hc + delta
                hole_cells = _absolute_cells(shape, hr, hc)
                frag_cells = _absolute_cells(shape, fr, fc)
                if not _can_place(hole_cells, hole_blocked):
                    continue
                if not _can_place(frag_cells, frag_blocked):
                    continue
                _mark(hole_cells, hole_blocked)
                _mark(frag_cells, frag_blocked)
                placed.append(
                    {
                        "shape": shape,
                        "hole_origin": (int(hr), int(hc)),
                        "frag_origin": (int(fr), int(fc)),
                        "direction": direction,
                    }
                )
                placed_here = True
                break
            if placed_here:
                break
        if not placed_here:
            return None

    placed.sort(key=lambda item: (item["hole_origin"], item["frag_origin"]))
    return placed


def _palette_cycle() -> tuple[list[int], dict[int, int]]:
    cycle = random.sample(GEN_COLOR_POOL, 4)
    mapping = {int(cycle[i]): int(cycle[(i + 1) % 4]) for i in range(4)}
    return cycle, mapping


def _apply_direction_color(color: int, mapping: dict[int, int], direction: str) -> int:
    return _apply_mapping(int(color), mapping, GEN_STEPS[direction])


def _attempt_generate() -> tuple[np.ndarray, np.ndarray] | None:
    top_h = random.randint(1, 2)
    band_h = random.randint(9, 13)
    band_w = random.randint(15, 24)
    if top_h + 2 + 2 * band_h > 30:
        return None

    margin_top = random.randint(1, 2)
    margin_left = random.randint(1, 2)
    margin_bottom = random.randint(1, 2)
    margin_right = random.randint(1, 2)
    template_bbox = (
        margin_top,
        margin_left,
        band_h - 1 - margin_bottom,
        band_w - 1 - margin_right,
    )
    tr0, tc0, tr1, tc1 = template_bbox
    if tr1 - tr0 < 5 or tc1 - tc0 < 7:
        return None

    n_holes = random.randint(2, 4)
    directions = random.sample(GEN_DIRS, n_holes)
    shapes: list[tuple[tuple[tuple[int, int], ...], int, int]] = []
    seen_shapes: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(n_holes):
        shape_info = _random_shape()
        if shape_info is None:
            return None
        shape, sh, sw = shape_info
        if shape in seen_shapes:
            return None
        seen_shapes.add(shape)
        shapes.append((shape, sh, sw))

    placed = _build_layout(band_h, band_w, template_bbox, shapes, directions)
    if placed is None:
        return None

    cycle, mapping = _palette_cycle()
    piece_colors = [random.choice(cycle) for _ in range(n_holes)]
    random.shuffle(piece_colors)

    middle = np.full((band_h, band_w), GEN_BG, dtype=int)
    middle[tr0 : tr1 + 1, tc0 : tc1 + 1] = GEN_TEMPLATE
    bottom = np.full((band_h, band_w), GEN_BG, dtype=int)
    expected = middle.copy()

    for item, color in zip(placed, piece_colors, strict=False):
        shape = item["shape"]
        hole_cells = _absolute_cells(shape, *item["hole_origin"])
        frag_cells = _absolute_cells(shape, *item["frag_origin"])
        for r, c in hole_cells:
            middle[r, c] = GEN_BG
            expected[r, c] = _apply_direction_color(int(color), mapping, item["direction"])
        for r, c in frag_cells:
            bottom[r, c] = int(color)

    top = np.full((top_h, band_w), GEN_BG, dtype=int)
    legend_row = random.randrange(top_h)
    pairs = [(src, dst) for src, dst in mapping.items()]
    random.shuffle(pairs)
    need_w = 3 * len(pairs) - 1
    start_c = random.randint(0, band_w - need_w)
    c = start_c
    for src, dst in pairs:
        top[legend_row, c] = int(src)
        top[legend_row, c + 1] = int(dst)
        c += 3

    sep = np.full((1, band_w), GEN_GREY, dtype=int)
    grid = np.vstack([top, sep, middle, sep, bottom])
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        built = _attempt_generate()
        if built is not None:
            grid, _expected = built
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
