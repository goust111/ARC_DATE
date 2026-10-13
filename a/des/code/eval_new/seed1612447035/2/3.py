from __future__ import annotations

import random

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = tuple(range(1, 10))
MAX_GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    best = None
    for turns in range(4):
        rot = np.rot90(mask, turns)
        for variant in (rot, np.fliplr(rot)):
            key = (tuple(map(int, variant.shape)), variant.tobytes())
            if best is None or key < best:
                best = key
    assert best is not None
    return best


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return _canonical_shape_key(_tight_mask(coords))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.asarray(pts, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    comps.sort(key=lambda comp: comp["bbox"])
    return comps


def _parse_legend_1d(line: np.ndarray, bg: int) -> tuple[list[tuple[int, int, int]], list[int]] | None:
    vals = [int(x) for x in line.tolist()]
    n = len(vals)
    i = 0
    entries: list[tuple[int, int, int]] = []
    used_positions: list[int] = []
    while i < n:
        while i < n and vals[i] == bg:
            i += 1
        if i >= n:
            break
        if i + 3 >= n:
            return None
        a, b, gap, c = vals[i], vals[i + 1], vals[i + 2], vals[i + 3]
        if a == bg or b == bg or c == bg or gap != bg or a == b:
            return None
        entries.append((a, b, c))
        used_positions.extend([i, i + 1, i + 3])
        i += 4
    if not entries:
        return None
    return entries, used_positions


def _edge_line(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid[0, :]
    if edge == "bottom":
        return grid[-1, :]
    if edge == "left":
        return grid[:, 0]
    if edge == "right":
        return grid[:, -1]
    raise ValueError(edge)


def _edge_coord(edge: str, shape: tuple[int, int], pos: int) -> tuple[int, int]:
    h, w = shape
    if edge == "top":
        return 0, int(pos)
    if edge == "bottom":
        return h - 1, int(pos)
    if edge == "left":
        return int(pos), 0
    if edge == "right":
        return int(pos), w - 1
    raise ValueError(edge)


def _extract_legend(grid: np.ndarray, bg: int) -> tuple[dict[tuple[int, int], int], set[tuple[int, int]], str]:
    candidates = []
    for edge in ("top", "bottom", "left", "right"):
        parsed = _parse_legend_1d(_edge_line(grid, edge), bg)
        if parsed is None:
            continue
        entries, positions = parsed
        mapping: dict[tuple[int, int], int] = {}
        valid = True
        for a, b, c in entries:
            key = tuple(sorted((int(a), int(b))))
            prev = mapping.get(key)
            if prev is not None and prev != int(c):
                valid = False
                break
            mapping[key] = int(c)
        if not valid:
            continue
        coords = {_edge_coord(edge, grid.shape, pos) for pos in positions}
        candidates.append((len(entries), len(coords), edge, mapping, coords))
    assert candidates, "expected one legend-bearing edge"
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    _, _, edge, mapping, coords = candidates[0]
    return mapping, coords, edge


def _row_extents(coords: np.ndarray) -> dict[int, tuple[int, int]]:
    extents: dict[int, list[int]] = {}
    for r, c in coords:
        rr = int(r)
        extents.setdefault(rr, []).append(int(c))
    return {r: (min(cols), max(cols)) for r, cols in extents.items()}


def _col_extents(coords: np.ndarray) -> dict[int, tuple[int, int]]:
    extents: dict[int, list[int]] = {}
    for r, c in coords:
        cc = int(c)
        extents.setdefault(cc, []).append(int(r))
    return {c: (min(rows), max(rows)) for c, rows in extents.items()}


def _bridge_info(comp_a: dict, comp_b: dict) -> dict:
    coords_a = comp_a["coords"]
    coords_b = comp_b["coords"]
    area = int(coords_a.shape[0])
    assert area == int(coords_b.shape[0]), "paired components must have equal area"

    row_sum_a = int(coords_a[:, 0].sum())
    row_sum_b = int(coords_b[:, 0].sum())
    col_sum_a = int(coords_a[:, 1].sum())
    col_sum_b = int(coords_b[:, 1].sum())

    same_row = row_sum_a == row_sum_b
    same_col = col_sum_a == col_sum_b
    assert same_row ^ same_col, "pair must align in exactly one axis"

    if same_row:
        left, right = sorted(
            [comp_a, comp_b],
            key=lambda comp: (int(comp["coords"][:, 1].sum()), int(comp["bbox"][1])),
        )
        shared_row_sum = int(left["coords"][:, 0].sum())
        left_extents = _row_extents(left["coords"])
        right_extents = _row_extents(right["coords"])
        best = None
        for row in sorted(set(left_extents) & set(right_extents)):
            left_min, left_max = left_extents[row]
            right_min, _right_max = right_extents[row]
            gap = right_min - left_max - 1
            if gap <= 0:
                continue
            key = (gap, abs(int(row) * area - shared_row_sum), int(row))
            if best is None or key < best[0]:
                best = (key, int(row), int(left_max), int(right_min))
        assert best is not None, "horizontal pair must expose a visible gap"
        _, row, left_max, right_min = best
        line = [(row, c) for c in range(left_max + 1, right_min)]
        return {
            "orientation": "H",
            "line": line,
            "mid": line[(len(line) - 1) // 2],
            "start_color": int(left["color"]),
            "pair_key": tuple(sorted((int(left["color"]), int(right["color"])))),
        }

    top, bottom = sorted(
        [comp_a, comp_b],
        key=lambda comp: (int(comp["coords"][:, 0].sum()), int(comp["bbox"][0])),
    )
    shared_col_sum = int(top["coords"][:, 1].sum())
    top_extents = _col_extents(top["coords"])
    bottom_extents = _col_extents(bottom["coords"])
    best = None
    for col in sorted(set(top_extents) & set(bottom_extents)):
        top_min, top_max = top_extents[col]
        bottom_min, _bottom_max = bottom_extents[col]
        gap = bottom_min - top_max - 1
        if gap <= 0:
            continue
        key = (gap, abs(int(col) * area - shared_col_sum), int(col))
        if best is None or key < best[0]:
            best = (key, int(col), int(top_max), int(bottom_min))
    assert best is not None, "vertical pair must expose a visible gap"
    _, col, top_max, bottom_min = best
    line = [(r, col) for r in range(top_max + 1, bottom_min)]
    return {
        "orientation": "V",
        "line": line,
        "mid": line[(len(line) - 1) // 2],
        "start_color": int(top["color"]),
        "pair_key": tuple(sorted((int(top["color"]), int(bottom["color"])))),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Bridge each congruent color-mismatched pair and mark the bridge midpoint with the legend-defined mix color.

    Input structure:
    - bg = most common color.
    - Interior foreground objects are small monochrome 4-connected components, each with area <= 4.
    - Each canonical shape appears exactly twice, possibly rotated or reflected, and the two copies have different colors.
    - One grid edge contains legend blocks in the form A,B,bg,C along that edge, meaning colors A and B mix to color C.

    Rule:
    - Remove the legend from consideration and group components by rotation/flip-invariant shape.
    - For each matched pair, determine whether the two centroids share a row or a column, then connect the nearest visible edges along that axis through the background gap.
    - Color the whole bridge with the color of the leftmost or upper object, then replace the bridge midpoint (rounding toward that start when the length is even) with the legend result color for the pair's two colors.

    Key insight:
    - Pairing depends on shape congruence, but the midpoint color depends on an independent edge legend keyed only by the two object colors.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    legend_map, legend_coords, _legend_edge = _extract_legend(g, bg)

    work = g.copy()
    for r, c in legend_coords:
        work[r, c] = bg

    comps = _find_components(work, bg)
    assert comps, "expected at least one interior component"
    assert all(1 <= int(comp["area"]) <= 4 for comp in comps), "components must stay small"

    groups: dict[tuple[tuple[int, int], bytes], list[dict]] = {}
    for comp in comps:
        groups.setdefault(_shape_key(comp["coords"]), []).append(comp)

    out = g.copy()
    for matched in groups.values():
        assert len(matched) == 2, "each shape must appear exactly twice"
        assert int(matched[0]["color"]) != int(matched[1]["color"]), "paired shapes must have different colors"
        bridge = _bridge_info(matched[0], matched[1])
        mix = legend_map.get(bridge["pair_key"])
        assert mix is not None, "missing legend entry for a color pair"
        for r, c in bridge["line"]:
            assert int(work[r, c]) == bg, "bridge must run through empty background"
            out[r, c] = int(bridge["start_color"])
        mr, mc = bridge["mid"]
        out[mr, mc] = int(mix)
    return out


def _pattern(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


SHAPE_TEMPLATES = (
    _pattern(("##",)),
    _pattern(("###",)),
    _pattern(("#.", "##")),
    _pattern(("####",)),
    _pattern(("##", "##")),
    _pattern(("#..", "###")),
    _pattern(("###", ".#.")),
    _pattern(("##.", ".##")),
)


def _variant_masks(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen = set()
    for turns in range(4):
        rot = np.rot90(mask, turns)
        for variant in (rot, np.fliplr(rot)):
            key = (tuple(map(int, variant.shape)), variant.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(np.array(variant, dtype=bool))
    return variants


def _alignment_delta(mask_a: np.ndarray, mask_b: np.ndarray, orientation: str) -> int | None:
    coords_a = np.argwhere(mask_a)
    coords_b = np.argwhere(mask_b)
    area = int(coords_a.shape[0])
    assert area == int(coords_b.shape[0])
    if orientation == "H":
        diff = int(coords_a[:, 0].sum() - coords_b[:, 0].sum())
    else:
        diff = int(coords_a[:, 1].sum() - coords_b[:, 1].sum())
    if diff % area != 0:
        return None
    return diff // area


def _make_component(mask: np.ndarray, top: int, left: int, color: int) -> dict:
    coords = np.argwhere(mask).astype(int) + np.array([top, left], dtype=int)
    return {
        "color": int(color),
        "coords": coords,
        "bbox": _bbox(coords),
        "area": int(coords.shape[0]),
    }


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in coords}


def _expand_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _all_inside(cells: set[tuple[int, int]], bounds: dict) -> bool:
    return all(
        bounds["rmin"] <= r <= bounds["rmax"] and bounds["cmin"] <= c <= bounds["cmax"]
        for r, c in cells
    )


def _choose_variants(mask: np.ndarray, orientation: str) -> tuple[np.ndarray, np.ndarray]:
    variants = _variant_masks(mask)
    feasible_same = []
    feasible_diff = []
    for va in variants:
        for vb in variants:
            if _alignment_delta(va, vb, orientation) is None:
                continue
            if np.array_equal(va, vb):
                feasible_same.append((va, vb))
            else:
                feasible_diff.append((va, vb))
    if feasible_diff and random.random() < 0.6:
        return random.choice(feasible_diff)
    feasible = feasible_diff + feasible_same
    assert feasible
    return random.choice(feasible)


def _sample_orientations(n_pairs: int) -> list[str]:
    if n_pairs == 2:
        vals = ["H", "V"]
    else:
        vals = ["H", "H", "V"] if random.random() < 0.5 else ["H", "V", "V"]
    random.shuffle(vals)
    return vals


def _sample_parities(n_pairs: int) -> list[int]:
    if n_pairs == 2:
        vals = [0, 1]
    else:
        vals = [0, 0, 1] if random.random() < 0.5 else [0, 1, 1]
    random.shuffle(vals)
    return vals


def _place_pair(
    h: int,
    w: int,
    bounds: dict,
    forbidden: set[tuple[int, int]],
    orientation: str,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    color_a: int,
    color_b: int,
    mix_color: int,
    parity: int,
) -> dict | None:
    if orientation == "H":
        delta = _alignment_delta(mask_a, mask_b, "H")
        if delta is None:
            return None
        row_lo = max(bounds["rmin"], bounds["rmin"] - delta)
        row_hi = min(
            bounds["rmax"] - mask_a.shape[0] + 1,
            bounds["rmax"] - mask_b.shape[0] + 1 - delta,
        )
        gap_hi = bounds["cmax"] - bounds["cmin"] + 1 - mask_a.shape[1] - mask_b.shape[1]
        if row_lo > row_hi or gap_hi < 1:
            return None
        for _ in range(250):
            box_gap = random.randint(1, min(8, gap_hi))
            col_hi = bounds["cmax"] - (mask_a.shape[1] + box_gap + mask_b.shape[1]) + 1
            if col_hi < bounds["cmin"]:
                continue
            top_a = random.randint(row_lo, row_hi)
            top_b = top_a + delta
            left_a = random.randint(bounds["cmin"], col_hi)
            left_b = left_a + mask_a.shape[1] + box_gap
            comp_a = _make_component(mask_a, top_a, left_a, color_a)
            comp_b = _make_component(mask_b, top_b, left_b, color_b)
            try:
                bridge = _bridge_info(comp_a, comp_b)
            except AssertionError:
                continue
            if bridge["orientation"] != "H":
                continue
            if len(bridge["line"]) < 2 or len(bridge["line"]) > 8 or len(bridge["line"]) % 2 != parity:
                continue
            footprint = _coord_set(comp_a["coords"]) | _coord_set(comp_b["coords"]) | set(bridge["line"])
            if not _all_inside(footprint, bounds):
                continue
            expanded = _expand_cells(footprint, h, w)
            if expanded & forbidden:
                continue
            return {
                "a": comp_a,
                "b": comp_b,
                "mix": int(mix_color),
                "bridge": bridge,
                "expanded": expanded,
            }
        return None

    delta = _alignment_delta(mask_a, mask_b, "V")
    if delta is None:
        return None
    col_lo = max(bounds["cmin"], bounds["cmin"] - delta)
    col_hi = min(
        bounds["cmax"] - mask_a.shape[1] + 1,
        bounds["cmax"] - mask_b.shape[1] + 1 - delta,
    )
    gap_hi = bounds["rmax"] - bounds["rmin"] + 1 - mask_a.shape[0] - mask_b.shape[0]
    if col_lo > col_hi or gap_hi < 1:
        return None
    for _ in range(250):
        box_gap = random.randint(1, min(8, gap_hi))
        row_hi = bounds["rmax"] - (mask_a.shape[0] + box_gap + mask_b.shape[0]) + 1
        if row_hi < bounds["rmin"]:
            continue
        top_a = random.randint(bounds["rmin"], row_hi)
        top_b = top_a + mask_a.shape[0] + box_gap
        left_a = random.randint(col_lo, col_hi)
        left_b = left_a + delta
        comp_a = _make_component(mask_a, top_a, left_a, color_a)
        comp_b = _make_component(mask_b, top_b, left_b, color_b)
        try:
            bridge = _bridge_info(comp_a, comp_b)
        except AssertionError:
            continue
        if bridge["orientation"] != "V":
            continue
        if len(bridge["line"]) < 2 or len(bridge["line"]) > 8 or len(bridge["line"]) % 2 != parity:
            continue
        footprint = _coord_set(comp_a["coords"]) | _coord_set(comp_b["coords"]) | set(bridge["line"])
        if not _all_inside(footprint, bounds):
            continue
        expanded = _expand_cells(footprint, h, w)
        if expanded & forbidden:
            continue
        return {
            "a": comp_a,
            "b": comp_b,
            "mix": int(mix_color),
            "bridge": bridge,
            "expanded": expanded,
        }
    return None


def _draw_legend(grid: np.ndarray, edge: str, entries: list[tuple[int, int, int]]) -> set[tuple[int, int]]:
    length = grid.shape[1] if edge in ("top", "bottom") else grid.shape[0]
    total = 4 * len(entries) + (len(entries) - 1)
    start = random.randint(0, length - total)
    coords = set()
    idx = start
    for i, (a, b, c) in enumerate(entries):
        for offset, value in ((0, a), (1, b), (3, c)):
            rc = _edge_coord(edge, grid.shape, idx + offset)
            grid[rc] = int(value)
            coords.add(rc)
        idx += 4
        if i != len(entries) - 1:
            idx += 1
    return coords


def _expected_output(grid: np.ndarray, pairs: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for pair in pairs:
        bridge = pair["bridge"]
        for r, c in bridge["line"]:
            out[r, c] = int(bridge["start_color"])
        mr, mc = bridge["mid"]
        out[mr, mc] = int(pair["mix"])
    return out


def _sample_canvas(n_pairs: int, edge: str) -> tuple[int, int] | None:
    need = 4 * n_pairs + (n_pairs - 1)
    for _ in range(40):
        h = random.randint(16, 30)
        w = random.randint(16, 30)
        if edge in ("top", "bottom") and w < need:
            continue
        if edge in ("left", "right") and h < need:
            continue
        return h, w
    return None


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.choice([2, 3])
    legend_edge = random.choice(["top", "bottom", "left", "right"])
    canvas = _sample_canvas(n_pairs, legend_edge)
    if canvas is None:
        return None
    h, w = canvas
    bounds = {
        "rmin": 2 if legend_edge == "top" else 1,
        "rmax": h - 3 if legend_edge == "bottom" else h - 2,
        "cmin": 2 if legend_edge == "left" else 1,
        "cmax": w - 3 if legend_edge == "right" else w - 2,
    }
    if bounds["rmin"] > bounds["rmax"] or bounds["cmin"] > bounds["cmax"]:
        return None

    shape_indices = random.sample(range(len(SHAPE_TEMPLATES)), n_pairs)
    shape_colors = random.sample(COLOR_POOL, 2 * n_pairs)
    mix_colors = random.sample([c for c in COLOR_POOL if c not in shape_colors], n_pairs)
    orientations = _sample_orientations(n_pairs)
    parities = _sample_parities(n_pairs)

    grid = np.full((h, w), BG, dtype=int)
    forbidden: set[tuple[int, int]] = set()
    placed_pairs: list[dict] = []
    legend_entries: list[tuple[int, int, int]] = []

    for idx in range(n_pairs):
        base_mask = SHAPE_TEMPLATES[shape_indices[idx]]
        orientation = orientations[idx]
        parity = parities[idx]
        color_a, color_b = shape_colors[2 * idx : 2 * idx + 2]
        if random.random() < 0.5:
            color_a, color_b = color_b, color_a
        mask_a, mask_b = _choose_variants(base_mask, orientation)
        placed = _place_pair(
            h,
            w,
            bounds,
            forbidden,
            orientation,
            mask_a,
            mask_b,
            color_a,
            color_b,
            mix_colors[idx],
            parity,
        )
        if placed is None:
            return None
        placed_pairs.append(placed)
        forbidden |= placed["expanded"]
        for comp_key in ("a", "b"):
            comp = placed[comp_key]
            coords = comp["coords"]
            grid[coords[:, 0], coords[:, 1]] = int(comp["color"])
        legend_pair = (int(color_a), int(color_b))
        if random.random() < 0.5:
            legend_pair = legend_pair[::-1]
        legend_entries.append((legend_pair[0], legend_pair[1], int(mix_colors[idx])))

    random.shuffle(legend_entries)
    _draw_legend(grid, legend_edge, legend_entries)

    expected = _expected_output(grid, placed_pairs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bridging-pairs sample")
