import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 300
RNG = np.random.default_rng()

SHAPES = [
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
    np.array([[0, 1], [1, 1], [0, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[0, 0, 1], [1, 1, 1]], dtype=int),
]


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(grid: np.ndarray, bg: int):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            norm = sorted((int(rr - r0), int(cc - c0)) for rr, cc in cells)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "shape": tuple(norm),
                    "area": len(cells),
                }
            )
    return comps


def _extract_panels(grid: np.ndarray):
    bg = _most_common_color(grid)
    h, w = grid.shape
    full_cols = {}
    for c in range(w):
        col = grid[:, c]
        if np.all(col == col[0]) and int(col[0]) != bg:
            full_cols.setdefault(int(col[0]), []).append(c)

    candidates = []
    for sep_color, cols in full_cols.items():
        runs = []
        start = cols[0]
        prev = cols[0]
        for c in cols[1:]:
            if c == prev + 1:
                prev = c
            else:
                runs.append((start, prev))
                start = prev = c
        runs.append((start, prev))
        if not runs:
            continue
        widths = []
        last = 0
        ok = True
        for a, b in runs:
            widths.append(a - last)
            last = b + 1
        widths.append(w - last)
        if min(widths) <= 0 or len(set(widths)) != 1:
            ok = False
        if ok and len(widths) >= 4 and len(widths) % 2 == 0:
            candidates.append((len(widths), sep_color, runs, widths[0]))

    assert candidates, "failed to detect equal-width panels"
    _, sep_color, runs, panel_w = max(candidates)

    panels = []
    start = 0
    for a, b in runs:
        panels.append(grid[:, start:a].copy())
        start = b + 1
    panels.append(grid[:, start:].copy())

    assert all(panel.shape == (h, panel_w) for panel in panels)
    return bg, sep_color, panel_w, panels


def _panel_objects(panel: np.ndarray, bg: int):
    objs = _find_components(panel, bg)
    objs.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["bbox"][2], obj["bbox"][3], obj["color"]))
    return objs


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, sep_color, panel_w, panels = _extract_panels(g)
    assert len(panels) >= 4 and len(panels) % 2 == 0, "expected example pairs plus one test pair"

    mapping = {}
    for pair_idx in range(0, len(panels) - 2, 2):
        inp = panels[pair_idx]
        out = panels[pair_idx + 1]
        in_objs = _panel_objects(inp, bg)
        out_objs = _panel_objects(out, bg)
        assert len(in_objs) == len(out_objs) >= 3, "each example panel pair should have matching objects"

        for src, dst in zip(in_objs, out_objs):
            assert src["bbox"] == dst["bbox"], "object positions should be preserved"
            assert src["shape"] == dst["shape"], "object shapes should be preserved"
            src_color = int(src["color"])
            dst_color = int(dst["color"])
            if src_color in mapping:
                assert mapping[src_color] == dst_color, "inconsistent color rule across examples"
            else:
                mapping[src_color] = dst_color

    test_in = panels[-2]
    test_out = panels[-1]
    assert np.all(test_out == bg), "test output panel should be blank"

    out = g.copy()
    base_col = g.shape[1] - panel_w
    for obj in _panel_objects(test_in, bg):
        color = mapping.get(int(obj["color"]), int(obj["color"]))
        for r, c in obj["coords"]:
            out[r, base_col + c] = color

    return out


def _random_shape():
    shape = SHAPES[int(RNG.integers(len(SHAPES)))]
    if RNG.random() < 0.5:
        shape = np.fliplr(shape)
    if RNG.random() < 0.5:
        shape = np.flipud(shape)
    if RNG.random() < 0.5:
        shape = shape.T
    return shape.copy()


def _paint_shape(panel: np.ndarray, shape: np.ndarray, top: int, left: int, color: int):
    h, w = shape.shape
    mask = shape.astype(bool)
    panel[top : top + h, left : left + w][mask] = color


def _can_place(occupied: np.ndarray, shape: np.ndarray, top: int, left: int):
    h, w = shape.shape
    if top < 0 or left < 0 or top + h > occupied.shape[0] or left + w > occupied.shape[1]:
        return False
    mask = np.zeros_like(occupied, dtype=bool)
    mask[top : top + h, left : left + w] |= shape.astype(bool)
    rows, cols = np.where(mask)
    for r, c in zip(rows, cols):
        r0 = max(0, r - 1)
        r1 = min(occupied.shape[0], r + 2)
        c0 = max(0, c - 1)
        c1 = min(occupied.shape[1], c + 2)
        if np.any(occupied[r0:r1, c0:c1]):
            return False
    return True


def _place_objects(panel_h: int, panel_w: int, count: int):
    occupied = np.zeros((panel_h, panel_w), dtype=bool)
    placed = []
    used_shapes = set()
    for _ in range(count):
        success = False
        for _ in range(200):
            shape = _random_shape()
            sig = tuple(map(tuple, shape.tolist()))
            if sig in used_shapes:
                continue
            h, w = shape.shape
            if h > panel_h - 2 or w > panel_w - 2:
                continue
            top = int(RNG.integers(1, panel_h - h))
            left = int(RNG.integers(1, panel_w - w))
            if _can_place(occupied, shape, top, left):
                occupied[top : top + h, left : left + w] |= shape.astype(bool)
                placed.append((shape, top, left))
                used_shapes.add(sig)
                success = True
                break
        if not success:
            return None
    placed.sort(key=lambda item: (item[1], item[2]))
    return placed


def _sample_mapping():
    domain_size = int(RNG.integers(3, 6))
    colors = list(RNG.choice(PALETTE, size=domain_size, replace=False))
    fixed_count = int(RNG.integers(1, domain_size))
    fixed = set(RNG.choice(colors, size=fixed_count, replace=False))
    rotating = [c for c in colors if c not in fixed]
    mapping = {c: c for c in colors}
    if rotating:
        shift = int(RNG.integers(1, len(rotating) + 1))
        if len(rotating) > 1:
            shift %= len(rotating)
            if shift == 0:
                shift = 1
        rotated = rotating[shift:] + rotating[:shift]
        for src, dst in zip(rotating, rotated):
            mapping[src] = dst
    if all(mapping[c] == c for c in colors):
        a, b = colors[:2]
        mapping[a], mapping[b] = b, a
    if all(mapping[c] != c for c in colors):
        mapping[colors[0]] = colors[0]
    return colors, mapping


def _render_panel(placements, colors, panel_h, panel_w):
    panel = np.full((panel_h, panel_w), BLACK, dtype=int)
    for (shape, top, left), color in zip(placements, colors):
        _paint_shape(panel, shape, top, left, color)
    return panel


def _assemble_grid(panels):
    h = panels[0].shape[0]
    sep = np.full((h, 1), GREY, dtype=int)
    pieces = []
    for idx, panel in enumerate(panels):
        if idx:
            pieces.append(sep)
        pieces.append(panel)
    return np.hstack(pieces)


def _gen_attempt():
    example_pairs = int(RNG.integers(1, 3))
    panel_h = int(RNG.integers(8, 13))
    panel_w = int(RNG.integers(8, 13))
    colors, mapping = _sample_mapping()

    panels = []
    covered = set()
    for pair_idx in range(example_pairs + 1):
        count = int(RNG.integers(3, min(6, len(colors) + 1)))
        placements = _place_objects(panel_h, panel_w, count)
        if placements is None:
            return None

        if pair_idx < example_pairs:
            if pair_idx == example_pairs - 1:
                chosen = list(colors)
                RNG.shuffle(chosen)
                chosen = chosen[:count]
            else:
                chosen = list(RNG.choice(colors, size=count, replace=False))
            covered.update(chosen)
            in_colors = chosen
            out_colors = [mapping[c] for c in in_colors]
            panels.append(_render_panel(placements, in_colors, panel_h, panel_w))
            panels.append(_render_panel(placements, out_colors, panel_h, panel_w))
        else:
            test_colors = list(RNG.choice(colors, size=count, replace=False))
            test_panel = _render_panel(placements, test_colors, panel_h, panel_w)
            blank_panel = np.full((panel_h, panel_w), BLACK, dtype=int)
            panels.append(test_panel)
            panels.append(blank_panel)

    if covered != set(colors):
        return None

    grid = _assemble_grid(panels)
    out = main(grid)
    if np.array_equal(out, grid):
        return None

    bg, _, _, split_panels = _extract_panels(grid)
    rendered = split_panels[-1]
    solved = _extract_panels(out)[3][-1]
    if not np.all(rendered == bg) or np.all(solved == bg):
        return None

    test_objs = _panel_objects(split_panels[-2], bg)
    solved_objs = _panel_objects(solved, bg)
    if len(test_objs) != len(solved_objs):
        return None
    changed = False
    unchanged = False
    for src, dst in zip(test_objs, solved_objs):
        if src["color"] == dst["color"]:
            unchanged = True
        else:
            changed = True
    if not (changed and unchanged):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
