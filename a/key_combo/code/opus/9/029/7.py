import random
from collections import Counter

import numpy as np


BG = 0
SEP = 5
PALETTE = (1, 2, 3, 4, 6)
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors(connectivity: int):
    return NEIGHBORS4 if connectivity == 4 else NEIGHBORS8


def _find_components(grid: np.ndarray, colors: set[int], connectivity: int = 4) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color not in colors:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in _neighbors(connectivity):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append({"color": color, "coords": arr, "area": len(coords)})
    return comps


def _separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == SEP)]
    assert len(rows) == 1, "expected exactly one grey separator row"
    return rows[0]


def _decode_mapping(template: np.ndarray, prefix_rows: np.ndarray) -> dict[int, int]:
    template_colors = tuple(sorted(set(map(int, template))))
    assert len(template_colors) == 2 and BG not in template_colors and SEP not in template_colors
    mapping: dict[int, int] = {}
    for row in prefix_rows:
        nz = np.flatnonzero(row != BG)
        assert nz.size >= 2, "expected a non-trivial prefix row"
        assert np.array_equal(nz, np.arange(int(nz[-1]) + 1)), "prefix must start at column 0"
        prefix_len = int(nz[-1]) + 1
        assert 2 <= prefix_len <= 4
        assert np.all(row[prefix_len:] == BG), "suffix after prefix must be background"

        per_source: dict[int, int] = {}
        for c in range(prefix_len):
            src = int(row[c])
            dst = int(template[c])
            if src in per_source:
                assert per_source[src] == dst, "same source color must map consistently within a row"
            else:
                per_source[src] = dst

        novel = [(src, dst) for src, dst in per_source.items() if src != dst]
        assert len(novel) == 1, "each prefix row must define exactly one non-identity mapping"
        for src, dst in per_source.items():
            if src == dst:
                assert src in template_colors, "identity support colors must come from the template row"

        src, dst = novel[0]
        if src in mapping:
            assert mapping[src] == dst, "duplicate source colors must agree across rows"
        else:
            mapping[src] = dst

    assert mapping, "expected at least one decoded mapping"
    return mapping


def _apply_rule(grid: np.ndarray) -> tuple[np.ndarray, dict[int, int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sep_r = _separator_row(g)
    assert sep_r >= 2 and sep_r + 2 < g.shape[0], "separator must split non-empty upper and lower regions"

    template = g[0].copy()
    assert np.all(template != BG) and np.all(template != SEP), "template row must be fully colored"
    prefix_rows = g[1:sep_r]
    assert 1 <= prefix_rows.shape[0] <= 3, "expected 1-3 prefix rows"
    mapping = _decode_mapping(template, prefix_rows)

    lower = g[sep_r + 1 :].copy()
    lower_colors = set(map(int, np.unique(lower))) - {BG}
    comps = _find_components(lower, lower_colors, connectivity=4)
    assert comps, "expected lower connected components"

    out = g.copy()
    for comp in comps:
        color = int(comp["color"])
        value = mapping[color] if color in mapping else BG
        coords = comp["coords"]
        out[coords[:, 0] + sep_r + 1, coords[:, 1]] = value
    return out, mapping, comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode source-to-template color mappings from upper prefix rows and use them to recolor or erase lower components.

    Input structure:
    - A black-background grid is split by exactly one full-width grey separator row.
    - Above the separator, the first row is a full two-color alternating template row and the next 1-3 rows are left-prefix rows.
    - Each prefix row contains one novel source color mixed with unchanged template-color support cells, so it encodes exactly one non-identity source→target rule.
    - Below the separator are several detached monochrome 4-connected components on black background.

    Rule:
    - Read each prefix row against the template row column-by-column and extract its unique non-identity mapping from a lower source color to one template color.
    - Collect all such rules into a lookup table.
    - For every lower 4-connected component, if its color is a lookup-table source color, recolor the whole component to that mapped template color.
    - If a lower component's color is absent from the lookup table, erase that component to black. Keep the upper decoding area and grey separator unchanged.

    Key insight:
    - The upper prefixes are not incomplete rows to finish; they are a color codebook whose non-identity substitutions determine which lower-component colors survive and what they become.
    """
    out, _mapping, _comps = _apply_rule(grid)
    return out


def _random_shape(rng: random.Random, area: int) -> np.ndarray:
    for _ in range(200):
        cells = {(0, 0)}
        frontier = {(0, 0)}
        while len(cells) < area:
            base = rng.choice(tuple(frontier))
            nbrs = []
            for dr, dc in NEIGHBORS4:
                cand = (base[0] + dr, base[1] + dc)
                if cand not in cells:
                    nbrs.append(cand)
            if not nbrs:
                frontier = set(cells)
                continue
            cells.add(rng.choice(nbrs))
            frontier = set(cells)
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        norm = sorted((r - min(rs), c - min(cs)) for r, c in cells)
        arr = np.array(norm, dtype=int)
        h = int(arr[:, 0].max()) + 1
        w = int(arr[:, 1].max()) + 1
        if h <= 4 and w <= 4:
            if area == h * w and area <= 4:
                continue
            return arr
    raise RuntimeError("failed to build shape")


def _can_place(shape: np.ndarray, origin_r: int, origin_c: int, lower: np.ndarray) -> bool:
    h, w = lower.shape
    for dr, dc in shape:
        r = origin_r + int(dr)
        c = origin_c + int(dc)
        if not (0 <= r < h and 0 <= c < w):
            return False
        for nr, nc in [(r + x, c + y) for x, y in NEIGHBORS8] + [(r, c)]:
            if 0 <= nr < h and 0 <= nc < w and lower[nr, nc] != BG:
                return False
    return True


def _place_components(
    rng: random.Random,
    lower_h: int,
    width: int,
    comp_colors: list[int],
) -> np.ndarray | None:
    lower = np.zeros((lower_h, width), dtype=int)
    for color in comp_colors:
        placed = False
        area_choices = [3, 4, 5, 6, 7, 8]
        rng.shuffle(area_choices)
        for area in area_choices:
            shape = _random_shape(rng, area)
            shape_h = int(shape[:, 0].max()) + 1
            shape_w = int(shape[:, 1].max()) + 1
            origins = [
                (r, c)
                for r in range(lower_h - shape_h + 1)
                for c in range(width - shape_w + 1)
            ]
            rng.shuffle(origins)
            for origin_r, origin_c in origins:
                if not _can_place(shape, origin_r, origin_c, lower):
                    continue
                for dr, dc in shape:
                    lower[origin_r + int(dr), origin_c + int(dc)] = color
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return lower


def _build_prefix_row(template: np.ndarray, source_color: int, target_color: int, prefix_len: int) -> np.ndarray:
    row = np.zeros_like(template)
    for c in range(prefix_len):
        if int(template[c]) == target_color:
            row[c] = source_color
        else:
            row[c] = int(template[c])
    return row


def _expected_lower_out(lower: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = lower.copy()
    colors = set(map(int, np.unique(lower))) - {BG}
    for comp in _find_components(lower, colors, connectivity=4):
        fill = mapping[int(comp["color"])] if int(comp["color"]) in mapping else BG
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = fill
    return out


def _validate_generated(grid: np.ndarray, expected_mapping: dict[int, int]) -> None:
    out, mapping, comps = _apply_rule(grid)
    assert mapping == expected_mapping

    sep_r = _separator_row(grid)
    lower_in = grid[sep_r + 1 :]
    lower_out = out[sep_r + 1 :]
    assert np.array_equal(lower_out, _expected_lower_out(lower_in, mapping))
    assert np.array_equal(out[: sep_r + 1], grid[: sep_r + 1]), "upper region and separator must stay unchanged"
    assert not np.array_equal(lower_in, lower_out), "lower region must change"

    recolored = 0
    erased = 0
    present_out_targets = set()
    for comp in comps:
        coords = comp["coords"]
        before = int(comp["color"])
        after_vals = set(map(int, lower_out[coords[:, 0], coords[:, 1]]))
        assert len(after_vals) == 1
        after = after_vals.pop()
        if before in mapping:
            recolored += 1
            assert after == mapping[before]
            present_out_targets.add(after)
        else:
            erased += 1
            assert after == BG
    assert recolored >= 1 and erased >= 1

    if len(set(mapping.values())) == 2:
        assert present_out_targets == set(mapping.values()), "both target-color branches must be visible"


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = random.Random(random.randrange(1 << 30))
        width = rng.randint(10, 15)
        num_mappings = rng.choices([1, 2, 3], weights=[1, 4, 4], k=1)[0]
        template_colors = rng.sample(list(PALETTE), 2)
        target_a, target_b = template_colors
        source_pool = [c for c in PALETTE if c not in template_colors]
        mapped_sources = rng.sample(source_pool, num_mappings)

        if num_mappings == 1:
            targets = [rng.choice(template_colors)]
        else:
            targets = [target_a, target_b]
            while len(targets) < num_mappings:
                targets.append(rng.choice(template_colors))
            rng.shuffle(targets)

        template = np.array([template_colors[c % 2] for c in range(width)], dtype=int)
        prefix_rows = []
        expected_mapping: dict[int, int] = {}
        for source_color, target_color in zip(mapped_sources, targets):
            prefix_len = rng.randint(2, min(4, width))
            row = _build_prefix_row(template, source_color, target_color, prefix_len)
            prefix_rows.append(row)
            expected_mapping[source_color] = target_color

        unmapped_pool = [c for c in PALETTE if c not in mapped_sources]
        if any(c in template_colors for c in unmapped_pool) and rng.random() < 0.7:
            required_unmapped = rng.choice([c for c in unmapped_pool if c in template_colors])
        else:
            required_unmapped = rng.choice(unmapped_pool)

        num_components = rng.randint(max(4, num_mappings + 1), 6)
        comp_colors = list(mapped_sources) + [required_unmapped]
        while len(comp_colors) < num_components:
            pool = mapped_sources + unmapped_pool
            comp_colors.append(rng.choice(pool))
        rng.shuffle(comp_colors)

        lower_h = rng.randint(max(8, num_components + 3), 12)
        lower = _place_components(rng, lower_h, width, comp_colors)
        if lower is None:
            continue

        top_h = 1 + len(prefix_rows)
        height = top_h + 1 + lower_h
        if height > 30:
            continue

        grid = np.zeros((height, width), dtype=int)
        grid[0] = template
        for i, row in enumerate(prefix_rows, start=1):
            grid[i] = row
        grid[top_h] = SEP
        grid[top_h + 1 :] = lower

        if len(np.unique(grid)) < 5:
            continue
        _validate_generated(grid, expected_mapping)
        return grid

    raise RuntimeError("failed to generate a valid input")
