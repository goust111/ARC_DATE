from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
SEPARATOR = 9
KEY_COLORS = (1, 2, 3, 4, 6, 7, 8)

PANEL_MIN = 6
PANEL_MAX = 9
AREA_MIN = 3
GENERATION_RETRIES = 300
PAIR_RETRIES = 120
TEMPLATE_RETRIES = 120
BRANCHES = ("cols_only", "rows_only", "both")
BOTH_LAYOUTS = ((2, 2), (2, 3), (3, 2))


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    h, w = map(int, mask.shape)
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(r, c, h, w):
            if mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == len(coords)


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        assert start < sep, "separator lines must leave nonempty regions"
        spans.append((start, sep - 1))
        start = sep + 1
    assert start < length, "separator lines must leave a final nonempty region"
    spans.append((start, length - 1))
    return spans


def _region_spans(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    sep_rows = [r for r in range(int(grid.shape[0])) if np.all(grid[r, :] == SEPARATOR)]
    sep_cols = [c for c in range(int(grid.shape[1])) if np.all(grid[:, c] == SEPARATOR)]
    assert sep_rows or sep_cols, "expected at least one maroon separator line"

    row_spans = _spans(int(grid.shape[0]), sep_rows)
    col_spans = _spans(int(grid.shape[1]), sep_cols)

    heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(heights) == 1 and len(widths) == 1, "subregions must be equal-sized"
    assert len(row_spans) * len(col_spans) >= 2, "expected multiple subregions"
    return row_spans, col_spans


def _extract_region(region: np.ndarray) -> dict[str, object]:
    colors = {int(v) for v in np.unique(region)}
    non_structural = colors - {BG, GREY}
    assert len(non_structural) == 1, "each region must contain exactly one key color"

    key_cells = np.argwhere((region != BG) & (region != GREY))
    assert len(key_cells) == 1, "each region must contain exactly one key pixel"
    key_pos = tuple(map(int, key_cells[0]))
    key_color = int(region[key_pos])

    template = region == GREY
    assert np.any(template), "each region must contain a grey template"
    assert _is_connected(template), "template must be 4-connected"

    return {"key_color": key_color, "key_pos": key_pos, "template": template}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap grey template masks between separator-defined partner regions that share the same key color.

    Input structure:
    - The background is black.
    - Full maroon separator rows and/or columns partition the grid into equal-sized rectangular regions.
    - Each region contains exactly one isolated non-black, non-grey key pixel.
    - Each region also contains one 4-connected grey template shape.

    Rule:
    - Split the grid into its separator-defined regions and read each region's key color and grey template mask.
    - Pair regions by equal key color; every key color appears in exactly two regions.
    - For each pair, erase both grey templates, then draw region A's template into region B at the same local coordinates using A's key color, and draw region B's template into region A using B's key color.
    - Preserve the key pixels, black background, and maroon separator lines.

    Key insight:
    - The repeated key colors are partner labels, and the important action is a cross-region exchange of template masks rather than recoloring each region's own template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    row_spans, col_spans = _region_spans(g)
    infos: list[dict[str, object]] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            info = _extract_region(g[r0 : r1 + 1, c0 : c1 + 1])
            info["bbox"] = (r0, c0, r1, c1)
            infos.append(info)

    groups: dict[int, list[int]] = {}
    for idx, info in enumerate(infos):
        groups.setdefault(int(info["key_color"]), []).append(idx)
    assert all(len(idxs) == 2 for idxs in groups.values()), "key colors must pair regions exactly"

    out = g.copy()
    out[out == GREY] = BG
    for idxs in groups.values():
        a, b = idxs
        for src, dst in ((a, b), (b, a)):
            r0, c0, r1, c1 = infos[dst]["bbox"]
            patch = out[r0 : r1 + 1, c0 : c1 + 1]
            patch[infos[src]["template"]] = int(infos[src]["key_color"])

    return out


def _choose_layout() -> tuple[int, int]:
    branch = random.choice(BRANCHES)
    if branch == "cols_only":
        return 1, 4
    if branch == "rows_only":
        return 4, 1
    return random.choice(BOTH_LAYOUTS)


def _choose_panel_size(rows: int, cols: int) -> tuple[int, int]:
    max_h = min(PANEL_MAX, (30 - (rows - 1)) // rows)
    max_w = min(PANEL_MAX, (30 - (cols - 1)) // cols)
    assert max_h >= PANEL_MIN and max_w >= PANEL_MIN
    return random.randint(PANEL_MIN, max_h), random.randint(PANEL_MIN, max_w)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    return r0, c0, r1, c1


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(map(tuple, np.argwhere(mask)))


def _key_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, mask.shape)
    candidates: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if mask[r, c]:
                continue
            if not np.any(mask[r - 1 : r + 2, c - 1 : c + 2]):
                candidates.append((r, c))
    return candidates


def _random_template_mask(h: int, w: int) -> np.ndarray | None:
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    max_area = min(10, max(AREA_MIN, ((h - 2) * (w - 2)) // 2))
    if max_area < AREA_MIN:
        return None

    for _ in range(TEMPLATE_RETRIES):
        target = random.randint(AREA_MIN, max_area)
        seed = random.choice(interior)
        cells = {seed}
        while len(cells) < target:
            frontier = {
                nbr
                for r, c in cells
                for nbr in _neighbors4(r, c, h, w)
                if 1 <= nbr[0] < h - 1 and 1 <= nbr[1] < w - 1 and nbr not in cells
            }
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        if len(cells) != target:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        r0, c0, r1, c1 = _bbox(mask)
        if r1 == r0 or c1 == c0:
            continue
        if not _key_candidates(mask):
            continue
        return mask
    return None


def _make_pair_specs(
    panel_h: int,
    panel_w: int,
    used_signatures: set[tuple[tuple[int, int], ...]],
) -> list[dict[str, object]] | None:
    for _ in range(PAIR_RETRIES):
        mask_a = _random_template_mask(panel_h, panel_w)
        mask_b = _random_template_mask(panel_h, panel_w)
        if mask_a is None or mask_b is None:
            continue

        sig_a = _mask_signature(mask_a)
        sig_b = _mask_signature(mask_b)
        if sig_a == sig_b or sig_a in used_signatures or sig_b in used_signatures:
            continue
        if _bbox(mask_a)[:2] == _bbox(mask_b)[:2]:
            continue

        cand_a = _key_candidates(mask_a)
        cand_b = _key_candidates(mask_b)
        random.shuffle(cand_a)
        random.shuffle(cand_b)
        for key_a in cand_a:
            for key_b in cand_b:
                if mask_a[key_b] or mask_b[key_a]:
                    continue
                return [
                    {"template": mask_a, "key_pos": key_a, "signature": sig_a},
                    {"template": mask_b, "key_pos": key_b, "signature": sig_b},
                ]
    return None


def _render_region(spec: dict[str, object]) -> np.ndarray:
    template = spec["template"]
    key_r, key_c = spec["key_pos"]
    region = np.full(template.shape, BG, dtype=int)
    region[template] = GREY
    region[key_r, key_c] = int(spec["key_color"])
    return region


def _assemble_grid(rows: int, cols: int, panel_h: int, panel_w: int, regions: list[np.ndarray]) -> np.ndarray:
    height = rows * panel_h + (rows - 1)
    width = cols * panel_w + (cols - 1)
    grid = np.full((height, width), BG, dtype=int)

    for rr in range(1, rows):
        grid[rr * (panel_h + 1) - 1, :] = SEPARATOR
    for cc in range(1, cols):
        grid[:, cc * (panel_w + 1) - 1] = SEPARATOR

    idx = 0
    for rr in range(rows):
        for cc in range(cols):
            r0 = rr * (panel_h + 1)
            c0 = cc * (panel_w + 1)
            grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = regions[idx]
            idx += 1
    return grid


def _gen_attempt() -> np.ndarray | None:
    rows, cols = _choose_layout()
    panel_h, panel_w = _choose_panel_size(rows, cols)
    region_count = rows * cols
    pair_count = region_count // 2
    colors = random.sample(KEY_COLORS, pair_count)

    specs: list[dict[str, object]] = []
    used_signatures: set[tuple[tuple[int, int], ...]] = set()
    for color in colors:
        pair_specs = _make_pair_specs(panel_h, panel_w, used_signatures)
        if pair_specs is None:
            return None
        for spec in pair_specs:
            used_signatures.add(spec["signature"])
            spec["key_color"] = color
            specs.append(spec)

    random.shuffle(specs)
    regions = [_render_region(spec) for spec in specs]
    grid = _assemble_grid(rows, cols, panel_h, panel_w, regions)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if np.any(out == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
