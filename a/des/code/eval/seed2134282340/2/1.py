from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BG = 0
SEPARATOR_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
GENERATE_RETRIES = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc2 = stack.pop()
            comp.append((int(cr), int(cc2)))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr = int(cr + dr)
                    nc = int(cc2 + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
        comps.append(comp)
    return comps


def _find_separator(grid: np.ndarray) -> int:
    h, w = grid.shape
    assert w % 2 == 1, "grid width must be odd"
    sep = w // 2
    sep_col = grid[:, sep]
    assert int(sep_col[0]) != BG, "separator must be non-background"
    assert bool(np.all(sep_col == sep_col[0])), "separator must be a solid vertical line"
    return int(sep)


def _direction_from_anchor(body: list[tuple[int, int]], anchor: tuple[int, int]) -> tuple[int, int]:
    rows = [r for r, _c in body]
    cols = [c for _r, c in body]
    ar, ac = anchor
    if ar < min(rows):
        return (-1, 0)
    if ar > max(rows):
        return (1, 0)
    if ac < min(cols):
        return (0, -1)
    if ac > max(cols):
        return (0, 1)
    raise AssertionError("anchor must lie outside the main-color body bbox")


def _seed_sort_key(cells: list[tuple[int, int]]) -> tuple[float, float, float]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return (float(min(rows)), float(sum(rows) / len(rows)), float(min(cols)))


def _extract_seed_specs(grid: np.ndarray) -> tuple[int, list[dict]]:
    sep = _find_separator(grid)
    left = np.asarray(grid[:, :sep], dtype=int)
    specs: list[dict] = []
    for comp in _components8(left != BG):
        vals = [int(left[r, c]) for r, c in comp]
        counts = Counter(vals)
        assert len(counts) == 2, "each seed must contain exactly one body color and one anchor color"
        main_color, main_count = counts.most_common(1)[0]
        assert int(main_count) >= 2, "main-color body must have at least two pixels"
        anchor_colors = [color for color, ct in counts.items() if int(color) != int(main_color) and int(ct) == 1]
        assert len(anchor_colors) == 1, "each seed must have a unique single-pixel anchor"
        anchor_color = int(anchor_colors[0])

        body = sorted((int(r), int(c)) for r, c in comp if int(left[r, c]) == int(main_color))
        anchor_cells = [(int(r), int(c)) for r, c in comp if int(left[r, c]) == int(anchor_color)]
        assert len(anchor_cells) == 1
        anchor = anchor_cells[0]

        specs.append(
            {
                "body": body,
                "color": int(main_color),
                "direction": _direction_from_anchor(body, anchor),
                "steps": int(len(body)),
                "sort_key": _seed_sort_key(comp),
            }
        )

    assert specs, "expected at least one seed on the left half"
    return sep, specs


def _pattern_cells(
    body: list[tuple[int, int]],
    direction: tuple[int, int],
    steps: int,
    sep: int,
    shape: tuple[int, int],
) -> list[tuple[int, int]] | None:
    h, w = shape
    dr, dc = direction
    base = [(int(r), int(2 * sep - c)) for r, c in body]
    cells: list[tuple[int, int]] = []
    for step in range(steps + 1):
        for r, c in base:
            rr = int(r + step * dr)
            cc = int(c + step * dc)
            if not (0 <= rr < h and sep < cc < w):
                return None
            cells.append((rr, cc))
    return cells


def _render_from_specs(grid: np.ndarray, sep: int, specs: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[:, sep + 1 :] = BG
    for spec in sorted(specs, key=lambda item: item["sort_key"]):
        cells = _pattern_cells(spec["body"], spec["direction"], spec["steps"], sep, out.shape)
        assert cells is not None, "generated pattern must stay inside the mirrored canvas"
        for r, c in cells:
            out[int(r), int(c)] = int(spec["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each left-half seed onto the right canvas, then grow it in the anchor-indicated
    direction one pixel at a time. The number of growth steps equals the main-color area.
    Lower seeds overwrite higher ones when the grown patterns overlap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    sep, specs = _extract_seed_specs(g)
    return _render_from_specs(g, sep, specs)


def _touches_body(body: set[tuple[int, int]], cell: tuple[int, int]) -> bool:
    cr, cc = cell
    for br, bc in body:
        if max(abs(int(br - cr)), abs(int(bc - cc))) == 1:
            return True
    return False


def _connected_body(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _build_template_options() -> dict[tuple[int, int], dict[int, list[tuple[tuple[tuple[int, int], ...], tuple[int, int]]]]]:
    by_dir: dict[tuple[int, int], dict[int, list[tuple[tuple[tuple[int, int], ...], tuple[int, int]]]]] = {
        direction: defaultdict(list) for direction in DIRECTIONS
    }
    for bits in range(1 << 9):
        body = {(idx // 3, idx % 3) for idx in range(9) if (bits >> idx) & 1}
        if not (2 <= len(body) <= 5):
            continue
        if not _connected_body(body):
            continue

        rows = [r for r, _c in body]
        cols = [c for _r, c in body]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        body_tuple = tuple(sorted((int(r), int(c)) for r, c in body))

        options = {
            (-1, 0): [(r0 - 1, c) for c in range(c0, c1 + 1) if r0 > 0],
            (1, 0): [(r1 + 1, c) for c in range(c0, c1 + 1) if r1 < 2],
            (0, -1): [(r, c0 - 1) for r in range(r0, r1 + 1) if c0 > 0],
            (0, 1): [(r, c1 + 1) for r in range(r0, r1 + 1) if c1 < 2],
        }

        for direction, anchors in options.items():
            for anchor in anchors:
                if anchor in body:
                    continue
                if not _touches_body(body, anchor):
                    continue
                by_dir[direction][len(body)].append((body_tuple, (int(anchor[0]), int(anchor[1]))))
    return by_dir


TEMPLATE_OPTIONS = _build_template_options()


def _body_is_horizontally_symmetric(body: tuple[tuple[int, int], ...]) -> bool:
    return set(body) == {(int(r), int(2 - c)) for r, c in body}


def _sample_directions(n_seeds: int) -> list[tuple[int, int]]:
    dirs = list(DIRECTIONS)
    if n_seeds == 4 and random.random() < 0.75:
        random.shuffle(dirs)
        return dirs
    if n_seeds == 3 and random.random() < 0.6:
        return random.sample(dirs, 3)
    out = [random.choice(dirs) for _ in range(n_seeds)]
    if n_seeds > 1 and len(set(out)) == 1:
        idx = random.randrange(n_seeds)
        out[idx] = random.choice([d for d in dirs if d != out[idx]])
    return out


def _choose_template(direction: tuple[int, int]) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    size_weights = {2: 2, 3: 3, 4: 3, 5: 2}
    sizes = [size for size in sorted(TEMPLATE_OPTIONS[direction]) if TEMPLATE_OPTIONS[direction][size]]
    chosen_size = random.choices(sizes, weights=[size_weights[size] for size in sizes], k=1)[0]
    return random.choice(TEMPLATE_OPTIONS[direction][chosen_size])


def _candidate_clear(occupied: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        for rr in range(max(0, int(r - 1)), min(h, int(r + 2))):
            for cc in range(max(0, int(c - 1)), min(w, int(c + 2))):
                if bool(occupied[rr, cc]):
                    return False
    return True


def _overlap_count(patterns: list[list[tuple[int, int]]]) -> int:
    seen: set[tuple[int, int]] = set()
    overlap = 0
    for cells in patterns:
        for cell in cells:
            if cell in seen:
                overlap += 1
            else:
                seen.add(cell)
    return overlap


def _gen_attempt(want_overlap: bool) -> np.ndarray | None:
    half_w = random.randint(8, 10) if want_overlap else random.randint(10, 14)
    height = random.randint(9, 12) if want_overlap else random.randint(11, 17)
    n_seeds = random.randint(3, 4) if want_overlap else random.randint(2, 3)
    width = 2 * half_w + 1

    directions = _sample_directions(n_seeds)
    colors = random.sample(SEED_COLORS, 2 * n_seeds)
    main_colors = colors[:n_seeds]
    anchor_colors = colors[n_seeds:]

    chosen_templates = []
    for direction in directions:
        body, anchor = _choose_template(direction)
        chosen_templates.append((direction, body, anchor))

    step_counts = [len(body) for _direction, body, _anchor in chosen_templates]
    if n_seeds > 1 and len(set(step_counts)) == 1:
        return None
    if not any(not _body_is_horizontally_symmetric(body) for _direction, body, _anchor in chosen_templates):
        return None

    occupied = np.zeros((height, half_w), dtype=bool)
    used_min_rows: set[int] = set()
    specs: list[dict] = [dict() for _ in range(n_seeds)]
    patterns: list[list[tuple[int, int]]] = []
    painted_cells: set[tuple[int, int]] = set()
    got_overlap = False

    order = sorted(range(n_seeds), key=lambda idx: len(chosen_templates[idx][1]), reverse=True)
    for place_idx, idx in enumerate(order):
        direction, body_local, anchor_local = chosen_templates[idx]
        placed = False
        for _ in range(160):
            top = random.randint(0, height - 3)
            left = random.randint(0, half_w - 3)

            body_abs = [(int(top + r), int(left + c)) for r, c in body_local]
            anchor_abs = (int(top + anchor_local[0]), int(left + anchor_local[1]))
            comp_cells = body_abs + [anchor_abs]
            if not _candidate_clear(occupied, comp_cells):
                continue

            min_row = min(r for r, _c in comp_cells)
            if min_row in used_min_rows:
                continue

            cells = _pattern_cells(body_abs, direction, len(body_abs), half_w, (height, width))
            if cells is None:
                continue
            cell_set = set(cells)
            overlap_cells = len(cell_set & painted_cells)
            if not want_overlap and overlap_cells != 0:
                continue
            if want_overlap and overlap_cells > max(6, len(cell_set) // 2):
                continue
            need_overlap_now = want_overlap and not got_overlap and place_idx == len(order) - 1
            if need_overlap_now and overlap_cells == 0:
                continue

            specs[idx] = {
                "body": sorted(body_abs),
                "color": int(main_colors[idx]),
                "anchor": anchor_abs,
                "anchor_color": int(anchor_colors[idx]),
                "direction": direction,
                "steps": int(len(body_abs)),
                "sort_key": _seed_sort_key(comp_cells),
            }
            patterns.append(cells)
            for r, c in comp_cells:
                occupied[int(r), int(c)] = True
            used_min_rows.add(int(min_row))
            painted_cells |= cell_set
            got_overlap = got_overlap or overlap_cells > 0
            placed = True
            break
        if not placed:
            return None

    overlap = _overlap_count(patterns)
    total_unique = len({cell for cells in patterns for cell in cells})
    if want_overlap:
        if not got_overlap or not (1 <= overlap <= max(6, total_unique // 3)):
            return None
    elif overlap != 0:
        return None

    if total_unique < 10 or total_unique > (height * half_w * 3) // 5:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[:, half_w] = SEPARATOR_COLOR
    for spec in specs:
        if not spec:
            return None
        for r, c in spec["body"]:
            grid[int(r), int(c)] = int(spec["color"])
        ar, ac = spec["anchor"]
        grid[int(ar), int(ac)] = int(spec["anchor_color"])

    expected = _render_from_specs(grid, half_w, specs)
    try:
        parsed_sep, parsed_specs = _extract_seed_specs(grid)
    except AssertionError:
        return None
    if parsed_sep != half_w or len(parsed_specs) != n_seeds:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    want_overlap = random.random() < 0.5
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(want_overlap)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATE_RETRIES} attempts")
