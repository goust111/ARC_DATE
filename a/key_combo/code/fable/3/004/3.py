from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid.astype(int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r0, c0 in np.argwhere(mask):
        r0, c0 = int(r0), int(c0)
        if seen[r0, c0]:
            continue
        stack = [(r0, c0)]
        seen[r0, c0] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_patch(panel: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = panel[r0 : r1 + 1, c0 : c1 + 1].copy()
    mask = np.zeros(patch.shape, dtype=bool)
    rel = coords - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return patch, mask


def _panel_components(panel: np.ndarray, bg: int) -> list[dict]:
    comps = []
    for coords in _components(panel != int(bg)):
        comps.append({"coords": coords, "bbox": _bbox(coords), "area": int(coords.shape[0])})
    comps.sort(key=lambda d: (d["bbox"][1], d["bbox"][0], d["area"]))
    return comps


def _separator_indices(grid: np.ndarray, bg: int, axis: int) -> list[int]:
    indices: list[int] = []
    length = grid.shape[axis]
    for i in range(length):
        line = grid[i, :] if axis == 0 else grid[:, i]
        if bool(np.all(line == line[0])) and int(line[0]) != int(bg):
            indices.append(int(i))
    return indices


def _segments(length: int, separators: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = 0
    for sep in sorted(separators):
        if start <= sep - 1:
            out.append((start, sep - 1))
        start = sep + 1
    if start <= length - 1:
        out.append((start, length - 1))
    return out


def _panel_slices(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    h, w = grid.shape
    row_seps = _separator_indices(grid, bg, axis=0)
    col_seps = _separator_indices(grid, bg, axis=1)
    assert row_seps or col_seps, "expected visible separator lines"

    row_segments = _segments(h, row_seps)
    col_segments = _segments(w, col_seps)
    panels: list[np.ndarray] = []
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            panels.append(grid[r0 : r1 + 1, c0 : c1 + 1])
    assert len(panels) >= 2, "separators must create multiple panels"
    return panels


def _terminal(color: int, mapping: dict[int, int]) -> int:
    cur = int(color)
    seen: set[int] = set()
    while cur in mapping:
        assert cur not in seen, "replacement mapping must be acyclic"
        seen.add(cur)
        cur = int(mapping[cur])
    return cur


def _analyze_grid(
    grid: np.ndarray,
) -> tuple[int, dict[int, int], np.ndarray, np.ndarray, list[tuple[int, set[tuple[int, int]]]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    mapping: dict[int, int] = {}
    query: tuple[np.ndarray, np.ndarray] | None = None
    panel_edges: list[tuple[int, set[tuple[int, int]]]] = []

    for panel_index, panel in enumerate(_panel_slices(g, bg)):
        comps = _panel_components(panel, bg)
        if not comps:
            continue
        if len(comps) == 1:
            assert query is None, "expected exactly one query panel"
            query = _component_patch(panel, comps[0]["coords"])
            continue
        assert len(comps) == 2, "example panels must contain exactly two sprites"

        left, right = comps
        left_patch, left_mask = _component_patch(panel, left["coords"])
        right_patch, right_mask = _component_patch(panel, right["coords"])
        assert left_mask.shape == right_mask.shape, "example sprites must have matching bboxes"
        assert np.array_equal(left_mask, right_mask), "example sprites must have the same shape"

        edges: set[tuple[int, int]] = set()
        for rr, cc in np.argwhere(left_mask):
            rr, cc = int(rr), int(cc)
            src = int(left_patch[rr, cc])
            dst = int(right_patch[rr, cc])
            assert src != bg and dst != bg, "sprite pixels cannot use the background color"
            if src == dst:
                continue
            previous = mapping.get(src)
            assert previous is None or previous == dst, "a source color must have one replacement"
            mapping[src] = dst
            edges.add((src, dst))
        assert edges, "each example pair must provide at least one color replacement"
        panel_edges.append((int(panel_index), edges))

    assert query is not None, "expected one query sprite"
    assert mapping, "expected at least one replacement relation"
    for src in list(mapping):
        _terminal(src, mapping)
    return bg, mapping, query[0], query[1], panel_edges


def _apply_mapping(patch: np.ndarray, mask: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = patch.copy()
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        out[r, c] = _terminal(int(patch[r, c]), mapping)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Chain local example color replacements and recolor the query sprite to terminal colors.

    Input structure:
    - Separator lines divide the grid into panels.
    - Each example panel contains a left and right sprite with the same mask;
      corresponding cells give direct color replacement relations.
    - Exactly one query panel contains a single sprite.
    - The direct replacement graph is acyclic, and at least one query color
      needs replacements from different example panels to reach its terminal color.

    Rule:
    - Collect all direct color replacements from every example sprite pair.
    - For each non-background pixel in the query sprite crop, repeatedly follow
      the replacement relation until the color is no longer a source.
    - Return the cropped query sprite with only its sprite pixels recolored;
      background cells inside the crop stay unchanged.

    Key insight:
    - No single example panel gives the complete answer: the true recoloring is
      the transitive closure of all local color-replacement links.
    """
    _bg, mapping, query_patch, query_mask, _panel_edges = _analyze_grid(grid)
    return _apply_mapping(query_patch, query_mask, mapping)


def _grow_mask(h: int, w: int, target_area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    r, c = random.randrange(h), random.randrange(w)
    mask[r, c] = True
    while int(mask.sum()) < int(target_area):
        candidates: set[tuple[int, int]] = set()
        for rr, cc in np.argwhere(mask):
            for nr, nc in _neighbors4(int(rr), int(cc)):
                if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                    candidates.add((nr, nc))
        if not candidates:
            break
        nr, nc = random.choice(list(candidates))
        mask[nr, nc] = True
    return mask


def _random_mask(max_h: int, max_w: int, required_cells: int) -> np.ndarray:
    for _ in range(300):
        h = random.randint(2, int(max_h))
        w = random.randint(2, int(max_w))
        lo = max(int(required_cells), 3)
        if lo > h * w:
            continue
        target = random.randint(lo, h * w)
        mask = _grow_mask(h, w, target)
        if (
            int(mask.sum()) >= int(required_cells)
            and bool(mask[0, :].any())
            and bool(mask[-1, :].any())
            and bool(mask[:, 0].any())
            and bool(mask[:, -1].any())
        ):
            return mask

    for h in range(2, int(max_h) + 1):
        for w in range(2, int(max_w) + 1):
            if h * w >= int(required_cells):
                return np.ones((h, w), dtype=bool)
    raise RuntimeError("cannot build a large enough sprite mask")


def _colored_patch(mask: np.ndarray, required: list[int], choices: list[int], bg: int) -> np.ndarray:
    required_unique = list(dict.fromkeys(int(c) for c in required))
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    assert len(coords) >= len(required_unique)
    random.shuffle(coords)

    patch = np.full(mask.shape, int(bg), dtype=int)
    for color in required_unique:
        r, c = coords.pop()
        patch[r, c] = int(color)
    for r, c in coords:
        patch[r, c] = int(random.choice(choices))
    return patch


def _example_panel(panel_h: int, panel_w: int, edges: list[tuple[int, int]], bg: int) -> np.ndarray:
    max_w = max(2, min(4, (panel_w - 3) // 2))
    max_h = max(2, min(5, panel_h - 2))
    mask = _random_mask(max_h, max_w, len(edges))
    sh, sw = mask.shape

    gap_max = max(1, panel_w - 2 * sw - 2)
    gap = random.randint(1, gap_max)
    left_max = panel_w - 1 - 2 * sw - gap
    assert left_max >= 1
    r0 = random.randint(1, panel_h - sh - 1)
    c0 = random.randint(1, left_max)
    c1 = c0 + sw + gap

    sources = [src for src, _dst in edges]
    direct = dict(edges)
    left_patch = _colored_patch(mask, sources, sources, bg)
    right_patch = np.full(mask.shape, int(bg), dtype=int)
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        right_patch[r, c] = int(direct[int(left_patch[r, c])])

    panel = np.full((panel_h, panel_w), int(bg), dtype=int)
    panel[r0 : r0 + sh, c0 : c0 + sw][mask] = left_patch[mask]
    panel[r0 : r0 + sh, c1 : c1 + sw][mask] = right_patch[mask]
    return panel


def _query_panel(
    panel_h: int,
    panel_w: int,
    required: list[int],
    choices: list[int],
    bg: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_w = max(2, min(5, panel_w - 2))
    max_h = max(2, min(6, panel_h - 2))
    mask = _random_mask(max_h, max_w, len(required))
    sh, sw = mask.shape
    patch = _colored_patch(mask, required, choices, bg)

    r0 = random.randint(1, panel_h - sh - 1)
    c0 = random.randint(1, panel_w - sw - 1)
    panel = np.full((panel_h, panel_w), int(bg), dtype=int)
    panel[r0 : r0 + sh, c0 : c0 + sw][mask] = patch[mask]
    return panel, patch, mask


def _build_mapping(n_examples: int) -> tuple[dict[int, int], list[tuple[int, int]], list[int]]:
    order = random.sample(PALETTE, len(PALETTE))
    primary_len = random.choice([3, 4]) if n_examples == 3 else random.choice([4, 5])
    primary = order[:primary_len]
    edges = [(primary[i], primary[i + 1]) for i in range(primary_len - 1)]

    target_edges = n_examples + 1
    for src in order[primary_len:-1]:
        if len(edges) >= target_edges:
            break
        src_i = order.index(src)
        edges.append((src, random.choice(order[src_i + 1 :])))

    assert len(edges) >= target_edges
    edges = edges[:target_edges]
    mapping = {int(src): int(dst) for src, dst in edges}
    assert len(mapping) == len(edges)
    return mapping, edges, primary


def _edge_groups(edges: list[tuple[int, int]], n_examples: int) -> list[list[tuple[int, int]]]:
    groups: list[list[tuple[int, int]]] = [[] for _ in range(n_examples)]
    groups[0].append(edges[0])
    groups[1].append(edges[1])
    for edge in edges[2:]:
        empty = [i for i, group in enumerate(groups) if not group]
        group_i = random.choice(empty) if empty else random.randrange(n_examples)
        groups[group_i].append(edge)
    random.shuffle(groups)
    return groups


def _render_grid(
    panel_rows: int,
    panel_cols: int,
    panel_h: int,
    panel_w: int,
    panels: list[np.ndarray],
    bg: int,
) -> np.ndarray:
    h = panel_rows * panel_h + panel_rows - 1
    w = panel_cols * panel_w + panel_cols - 1
    grid = np.full((h, w), int(bg), dtype=int)

    for r in range(1, panel_rows):
        grid[r * panel_h + r - 1, :] = SEPARATOR
    for c in range(1, panel_cols):
        grid[:, c * panel_w + c - 1] = SEPARATOR

    for i, panel in enumerate(panels):
        pr = i // panel_cols
        pc = i % panel_cols
        r0 = pr * (panel_h + 1)
        c0 = pc * (panel_w + 1)
        grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel
    return grid


def _direct_once(patch: np.ndarray, mask: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = patch.copy()
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        out[r, c] = int(mapping.get(int(patch[r, c]), int(patch[r, c])))
    return out


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    mapping: dict[int, int],
    primary: list[int],
) -> bool:
    try:
        bg, recovered, query_patch, query_mask, panel_edges = _analyze_grid(grid)
        if recovered != mapping:
            return False
        if not np.array_equal(_apply_mapping(query_patch, query_mask, recovered), expected):
            return False

        direct = _direct_once(query_patch, query_mask, recovered)
        if np.array_equal(direct, expected):
            return False

        first = (int(primary[0]), int(primary[1]))
        second = (int(primary[1]), int(primary[2]))
        edge_panel: dict[tuple[int, int], int] = {}
        for panel_i, edges in panel_edges:
            for edge in edges:
                edge_panel[edge] = int(panel_i)
        if edge_panel.get(first) == edge_panel.get(second):
            return False
        if int(primary[0]) not in set(map(int, query_patch[query_mask].flatten())):
            return False
        if _terminal(int(primary[0]), recovered) == int(recovered[int(primary[0])]):
            return False

        fg_colors = set(map(int, expected[expected != bg].flatten()))
        if len(fg_colors) < 2:
            return False
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    n_examples = random.choice([3, 5])
    if n_examples == 3:
        panel_rows, panel_cols = 2, 2
        panel_h = random.randint(7, 10)
        panel_w = random.randint(9, 13)
    else:
        panel_rows, panel_cols = 2, 3
        panel_h = random.randint(7, 10)
        panel_w = 9

    mapping, edges, primary = _build_mapping(n_examples)
    groups = _edge_groups(edges, n_examples)
    example_panels = [_example_panel(panel_h, panel_w, group, BACKGROUND) for group in groups]

    extra_sources = [src for src, _dst in edges if src not in primary]
    assert extra_sources
    required_query = [primary[0], primary[1], primary[-1], random.choice(extra_sources)]
    query_choices = sorted({c for edge in edges for c in edge} | set(primary))
    query, query_patch, query_mask = _query_panel(
        panel_h,
        panel_w,
        required_query,
        query_choices,
        BACKGROUND,
    )
    expected = _apply_mapping(query_patch, query_mask, mapping)

    total_panels = n_examples + 1
    query_index = random.randrange(total_panels)
    panels: list[np.ndarray] = []
    example_i = 0
    for i in range(total_panels):
        if i == query_index:
            panels.append(query)
        else:
            panels.append(example_panels[example_i])
            example_i += 1

    grid = _render_grid(panel_rows, panel_cols, panel_h, panel_w, panels, BACKGROUND)
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if not _semantic_checks(grid, expected, mapping, primary):
        return None
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid color replacement chain input")
