from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PATTERN_COLORS = (1, 2, 3)
GREY = 5

GRID_SIZE = 19
PANEL_SIZE = 5
STRIDE = 6
PANEL_ORIGINS = (1, 7, 13)
SEPARATOR_IDXS = (0, 6, 12, 18)

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

MAX_GENERATION_ATTEMPTS = 400
MAX_PANEL_ATTEMPTS = 250


def _parse_template(rows: tuple[str, ...]) -> np.ndarray:
    coords = [(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "1"]
    return np.array(coords, dtype=int)


SOLID_BASES = [
    _parse_template(("111",)),
    _parse_template(("10", "11")),
    _parse_template(("11", "11")),
    _parse_template(("111", "010")),
    _parse_template(("10", "10", "11")),
    _parse_template(("010", "111", "010")),
    _parse_template(("100", "100", "111")),
]

CHAIN_BASES = [
    _parse_template(("100", "010", "001")),
    _parse_template(("110", "001")),
    _parse_template(("100", "011")),
    _parse_template(("110", "001", "001")),
    _parse_template(("1000", "0110", "0001")),
    _parse_template(("1100", "0010", "0011")),
    _parse_template(("1000", "0110", "0001", "0001")),
]


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _variants(base: np.ndarray) -> list[np.ndarray]:
    h = int(base[:, 0].max()) + 1
    w = int(base[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[base[:, 0], base[:, 1]] = True

    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        for candidate in (rotated, np.fliplr(rotated)):
            trimmed = _trim_mask(candidate)
            coords = tuple(tuple(map(int, rc)) for rc in np.argwhere(trimmed))
            if coords not in seen:
                seen.add(coords)
                out.append(np.array(coords, dtype=int))
    return out


SOLID_VARIANTS = [variant for base in SOLID_BASES for variant in _variants(base)]
CHAIN_VARIANTS = [variant for base in CHAIN_BASES for variant in _variants(base)]


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    dirs = DIR4 if int(connectivity) == 4 else DIR8
    visited = np.zeros_like(mask, dtype=bool)
    result: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in dirs:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            result.append(np.array(coords, dtype=int))
    return result


def _marker_bbox(mask: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(mask)
    height = int(coords[:, 0].max() - coords[:, 0].min() + 1)
    width = int(coords[:, 1].max() - coords[:, 1].min() + 1)
    return height, width


def _diagonal_touch(a: np.ndarray, b: np.ndarray) -> bool:
    b_set = {tuple(map(int, rc)) for rc in b}
    for r, c in a:
        for dr, dc in DIAGONALS:
            if (int(r + dr), int(c + dc)) in b_set:
                return True
    return False


def _classify_marker(mask: np.ndarray) -> str:
    comps8 = _components(mask, connectivity=8)
    assert len(comps8) == 1, "marker must be one 8-connected region"

    comps4 = _components(mask, connectivity=4)
    if len(comps4) == 1:
        return "solid"

    n = len(comps4)
    graph = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _diagonal_touch(comps4[i], comps4[j]):
                graph[i].add(j)
                graph[j].add(i)

    seen = {0}
    stack = [0]
    while stack:
        cur = stack.pop()
        for nxt in graph[cur]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)

    edge_count = sum(len(neigh) for neigh in graph) // 2
    assert len(seen) == n, "multi-lobe marker must connect through diagonal contacts"
    assert edge_count == n - 1, "multi-lobe marker must form a chain"
    assert max(len(neigh) for neigh in graph) <= 2, "multi-lobe marker must be non-branching"
    return "chain"


def _panel_output(panel: np.ndarray) -> tuple[np.ndarray, str]:
    marker_mask = panel == GREY
    pattern_mask = np.isin(panel, PATTERN_COLORS)
    pattern_colors = sorted(set(map(int, panel[pattern_mask])))
    assert len(pattern_colors) == 1, "each panel must contain exactly one pattern color"
    color = pattern_colors[0]

    marker_type = _classify_marker(marker_mask)
    content = np.where(marker_mask, BACKGROUND, panel)
    if marker_type == "chain":
        mirrored = np.fliplr(content)
    else:
        mirrored = np.flipud(content)

    out = mirrored.copy()
    out[marker_mask] = color
    return out, marker_type


def main(grid: np.ndarray) -> np.ndarray:
    """Decode each panel's grey topology into a mirror axis, then absorb the marker into the mirrored colored pattern.

    Input structure
    The grid is a 19x19 canvas split by one-cell grey grid lines into a 3x3
    array of 5x5 panels. Each panel contains black background, exactly one
    colored pixel pattern in blue/red/green, and exactly one grey marker region.

    Rule
    For each panel, analyze the marker's 4-connected decomposition inside its
    8-connected region. A single 4-connected marker is a solid type and means
    reflect the non-marker panel content across the horizontal midline; a
    multi-lobe diagonal chain means reflect the non-marker content across the
    vertical midline. After mirroring, recolor the original marker cells to the
    panel's pattern color. Grey separator lines stay grey.

    Key insight
    The marker's internal connectivity is not part of the final drawing; it is a
    hidden instruction that selects which mirror axis to apply before the marker
    is repainted away.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 19x19 3x3 panel lattice"

    out = g.copy()
    for idx in SEPARATOR_IDXS:
        assert np.all(g[idx, :] == GREY)
        assert np.all(g[:, idx] == GREY)

    for r0 in PANEL_ORIGINS:
        for c0 in PANEL_ORIGINS:
            panel = g[r0 : r0 + PANEL_SIZE, c0 : c0 + PANEL_SIZE]
            out[r0 : r0 + PANEL_SIZE, c0 : c0 + PANEL_SIZE] = _panel_output(panel)[0]

    return out


def _place_shape(variant_pool: list[np.ndarray]) -> np.ndarray:
    coords = random.choice(variant_pool)
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    top = random.randint(0, PANEL_SIZE - h)
    left = random.randint(0, PANEL_SIZE - w)
    return coords + np.array([top, left], dtype=int)


def _make_pattern(marker_mask: np.ndarray) -> np.ndarray | None:
    free_cells = [(r, c) for r in range(PANEL_SIZE) for c in range(PANEL_SIZE) if not marker_mask[r, c]]
    for _ in range(400):
        count = random.randint(3, 8)
        chosen = random.sample(free_cells, count)
        mask = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True

        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(np.fliplr(mask), np.flipud(mask)):
            continue
        if np.all(np.fliplr(mask)[marker_mask]) or np.all(np.flipud(mask)[marker_mask]):
            continue
        return mask
    return None


def _make_panel(marker_type: str) -> tuple[np.ndarray, dict] | None:
    variant_pool = CHAIN_VARIANTS if marker_type == "chain" else SOLID_VARIANTS
    for _ in range(MAX_PANEL_ATTEMPTS):
        marker_coords = _place_shape(variant_pool)
        marker_mask = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)
        marker_mask[marker_coords[:, 0], marker_coords[:, 1]] = True

        pattern_mask = _make_pattern(marker_mask)
        if pattern_mask is None:
            continue

        vertical_out = np.fliplr(pattern_mask)
        horizontal_out = np.flipud(pattern_mask)
        if np.array_equal(vertical_out, horizontal_out):
            continue

        color = random.choice(PATTERN_COLORS)
        panel = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
        panel[marker_mask] = GREY
        panel[pattern_mask] = color

        out_panel, classified = _panel_output(panel)
        if classified != marker_type:
            continue
        if marker_type == "chain" and np.all(vertical_out[marker_mask]):
            continue
        if marker_type == "solid" and np.all(horizontal_out[marker_mask]):
            continue
        if np.array_equal(panel, out_panel):
            continue

        return panel, {
            "type": marker_type,
            "color": color,
            "marker_area": int(marker_mask.sum()),
            "marker_bbox": _marker_bbox(marker_mask),
        }
    return None


def _assemble_grid(panels: list[np.ndarray]) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), GREY, dtype=int)
    panel_iter = iter(panels)
    for r0 in PANEL_ORIGINS:
        for c0 in PANEL_ORIGINS:
            grid[r0 : r0 + PANEL_SIZE, c0 : c0 + PANEL_SIZE] = next(panel_iter)
    return grid


def _semantic_check(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    for idx in SEPARATOR_IDXS:
        if not np.all(grid[idx, :] == GREY) or not np.all(grid[:, idx] == GREY):
            return False
        if not np.all(out[idx, :] == GREY) or not np.all(out[:, idx] == GREY):
            return False

    types: list[str] = []
    colors: list[int] = []
    areas: dict[str, set[int]] = {"chain": set(), "solid": set()}
    bboxes: dict[str, set[tuple[int, int]]] = {"chain": set(), "solid": set()}
    output_signatures = set()

    for r0 in PANEL_ORIGINS:
        for c0 in PANEL_ORIGINS:
            panel = grid[r0 : r0 + PANEL_SIZE, c0 : c0 + PANEL_SIZE]
            out_panel = out[r0 : r0 + PANEL_SIZE, c0 : c0 + PANEL_SIZE]
            marker_mask = panel == GREY
            pattern_mask = np.isin(panel, PATTERN_COLORS)

            if len(_components(marker_mask, 8)) != 1:
                return False
            if not 3 <= int(marker_mask.sum()) <= 5:
                return False
            if not 3 <= int(pattern_mask.sum()) <= 8:
                return False

            panel_colors = sorted(set(map(int, panel[pattern_mask])))
            if len(panel_colors) != 1:
                return False
            pattern_color = panel_colors[0]
            if np.array_equal(pattern_mask, np.fliplr(pattern_mask)):
                return False
            if np.array_equal(pattern_mask, np.flipud(pattern_mask)):
                return False
            if np.array_equal(np.fliplr(pattern_mask), np.flipud(pattern_mask)):
                return False

            expected_panel, marker_type = _panel_output(panel)
            if not np.array_equal(out_panel, expected_panel):
                return False
            if np.array_equal(panel, out_panel):
                return False
            if np.any(out_panel == GREY):
                return False

            types.append(marker_type)
            colors.append(pattern_color)
            areas[marker_type].add(int(marker_mask.sum()))
            bboxes[marker_type].add(_marker_bbox(marker_mask))
            output_signatures.add(out_panel.tobytes())

    if types.count("chain") < 2 or types.count("solid") < 2:
        return False
    if len(set(colors)) < 2:
        return False
    if not (set(colors[i] for i, t in enumerate(types) if t == "chain") & set(colors[i] for i, t in enumerate(types) if t == "solid")):
        return False
    if not (areas["chain"] & areas["solid"]):
        return False
    if not (bboxes["chain"] & bboxes["solid"]):
        return False
    if len(output_signatures) < 4:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    chain_count = random.randint(2, 7)
    panel_types = ["chain"] * chain_count + ["solid"] * (9 - chain_count)
    random.shuffle(panel_types)

    panels = []
    for marker_type in panel_types:
        built = _make_panel(marker_type)
        if built is None:
            return None
        panel, _meta = built
        panels.append(panel)

    grid = _assemble_grid(panels)
    return grid if _semantic_check(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
