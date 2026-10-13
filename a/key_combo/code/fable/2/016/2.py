from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


for _parent in Path(__file__).resolve().parents:
    _utils_dir = _parent / "new" / "solvers"
    if _utils_dir.is_dir():
        if str(_utils_dir) not in sys.path:
            sys.path.append(str(_utils_dir))
        break
else:
    raise ImportError("could not locate new/solvers for arc_utils import")

from arc_utils import Color, bbox_coords, colors_present, crop_bbox, find_components, get_bg, paste


OPS = ("rot90", "rot180", "flip_ud", "flip_lr")
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GRID_LIMIT = 30
GEN_ATTEMPTS = 300
OUTER_MARGIN = 1
PLACEMENT_GAP = 1
BG_COLOR = Color.BLACK


def _find_legend_divider(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for color in colors_present(grid, exclude={bg}):
        mask = grid == color
        for r in range(h):
            c = -1
            while c + 1 < w and mask[r, c + 1]:
                c += 1
            if r >= 2 and c >= 3 and mask[: r + 1, c].all():
                candidates.append((int(color), int(r), int(c)))
    assert len(candidates) == 1, "expected one L-shaped divider touching the top and left edges"
    return candidates[0]


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size, "shape mask must be non-empty"
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    trimmed = _trim_mask(mask)
    return tuple(tuple(map(int, rc)) for rc in np.argwhere(trimmed))


def _apply_op(mask: np.ndarray, op: str) -> np.ndarray:
    if op == "rot90":
        out = np.rot90(mask, -1)
    elif op == "rot180":
        out = np.rot90(mask, 2)
    elif op == "flip_ud":
        out = np.flipud(mask)
    elif op == "flip_lr":
        out = np.fliplr(mask)
    else:
        raise ValueError(f"unknown op: {op}")
    return _trim_mask(out)


def _matching_ops(src_mask: np.ndarray, dst_mask: np.ndarray) -> list[str]:
    target = _trim_mask(dst_mask)
    return [op for op in OPS if np.array_equal(_apply_op(src_mask, op), target)]


def _canonical_shape(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    base = _trim_mask(mask)
    variants = []
    cur = base
    for _ in range(4):
        variants.append(_mask_signature(cur))
        variants.append(_mask_signature(np.fliplr(cur)))
        cur = np.rot90(cur, -1)
    return min(set(variants))


def _legend_patch(legend: np.ndarray, comp: dict, color: int) -> np.ndarray:
    patch = crop_bbox(legend, comp["bbox"])
    return patch == int(color)


def _extract_legend_operations(
    grid: np.ndarray, bg: int
) -> tuple[int, int, int, dict[int, str]]:
    divider_color, divider_r, divider_c = _find_legend_divider(grid, bg)
    legend = grid[:divider_r, :divider_c]
    comps = find_components(legend, bg=bg, connectivity=4, monochromatic=True)
    by_color: dict[int, list[dict]] = {}
    for comp in comps:
        color = int(comp["color"])
        assert color != divider_color, "divider color must not appear inside the legend interior"
        by_color.setdefault(color, []).append(comp)
    assert by_color, "legend must contain at least one demo pair"

    color_to_op: dict[int, str] = {}
    for color, items in by_color.items():
        assert len(items) == 2, "each legend color must have exactly two components"
        items = sorted(items, key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
        left, right = items
        assert left["bbox"][1] < right["bbox"][1], "legend demo pair must be ordered left-to-right"
        src_mask = _legend_patch(legend, left, color)
        dst_mask = _legend_patch(legend, right, color)
        matches = _matching_ops(src_mask, dst_mask)
        assert len(matches) == 1, "legend demo must determine a unique operation"
        color_to_op[color] = matches[0]
    return divider_color, divider_r, divider_c, color_to_op


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each legend color's before-and-after demo pair to transform query sprites of the same color.

    Input structure:
    - A background grid contains a one-color L-shaped divider touching the top and left edges.
    - Inside the upper-left legend region, each non-divider color appears in exactly two components:
      the left one is the source shape and the right one is that same shape after one geometry change.
    - Outside the legend, monochrome query components reuse legend colors but can have unrelated shapes.

    Rule:
    - For each legend color, infer the unique operation shown by its left-to-right demo pair:
      a 90-degree clockwise turn, a 180-degree turn, a mirror across the horizontal axis,
      or a mirror across the vertical axis.
    - Apply that same operation to every query component of the same color in the query component's
      own bounding-box coordinate system, then place the transformed result so its new bounding box
      starts at the original bounding box's top-left corner.
    - Leave the divider, the legend region, and untouched background unchanged.

    Key insight:
    - The legend maps colors to operations, not shapes to colors; two queries with the same shape can
      transform differently if their colors point to different legend demos.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)

    divider_color, divider_r, divider_c, color_to_op = _extract_legend_operations(g, bg)

    query_grid = g.copy()
    query_grid[:divider_r, :divider_c] = bg
    query_grid[divider_r, : divider_c + 1] = bg
    query_grid[: divider_r + 1, divider_c] = bg

    query_comps = find_components(
        query_grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=sorted(color_to_op),
    )
    assert query_comps, "expected at least one query component"

    all_query_mask = query_grid != bg
    fixed_mask = (g != bg) & (~all_query_mask)
    planned_output = np.zeros(g.shape, dtype=bool)
    moves = []

    for comp in query_comps:
        color = int(comp["color"])
        assert color in color_to_op, "every query color must appear in the legend"
        r0, c0, _r1, _c1 = comp["bbox"]
        src_patch = crop_bbox(g, comp["bbox"]) == color
        dst_patch = _apply_op(src_patch, color_to_op[color])
        h, w = dst_patch.shape
        assert r0 + h <= g.shape[0] and c0 + w <= g.shape[1], "transformed query must stay in bounds"

        dst_mask = np.zeros(g.shape, dtype=bool)
        dst_mask[r0 : r0 + h, c0 : c0 + w] = dst_patch
        assert not np.any(dst_mask & fixed_mask), "query outputs must not overwrite divider or legend pixels"
        assert not np.any(dst_mask & planned_output), "transformed query outputs must not overlap each other"
        planned_output |= dst_mask
        moves.append((comp["coords"], int(r0), int(c0), color, dst_patch))

    out = g.copy()
    for coords, _r0, _c0, _color, _dst_patch in moves:
        out[coords[:, 0], coords[:, 1]] = bg
    for _coords, r0, c0, color, dst_patch in moves:
        patch = np.full(dst_patch.shape, bg, dtype=int)
        patch[dst_patch] = color
        paste(out, patch, r0, c0, transparent=bg)
    return out


def _coords_to_mask(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    h = max(rows) + 1
    w = max(cols) + 1
    mask = np.zeros((h, w), dtype=bool)
    rr, cc = np.array(coords, dtype=int).T
    mask[rr, cc] = True
    return mask


def _normalize_coords(coords: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    arr = np.array(sorted(coords), dtype=int)
    arr -= arr.min(axis=0)
    return tuple((int(r), int(c)) for r, c in arr)


def _random_polyomino(
    rng: np.random.Generator,
    *,
    size_range: tuple[int, int],
    max_height: int,
    max_width: int,
    require_non_square: bool,
) -> np.ndarray:
    lo, hi = size_range
    for _ in range(200):
        size = int(rng.integers(lo, hi + 1))
        cells = {(0, 0)}
        while len(cells) < size:
            frontier = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.add(nxt)
            frontier_list = tuple(frontier)
            cells.add(frontier_list[int(rng.integers(len(frontier_list)))])
        norm = _normalize_coords(cells)
        mask = _coords_to_mask(norm)
        if mask.shape[0] > max_height or mask.shape[1] > max_width:
            continue
        if require_non_square and mask.shape[0] == mask.shape[1]:
            continue
        return mask
    raise RuntimeError("failed to sample a bounded polyomino")


def _clear_demo(mask: np.ndarray) -> bool:
    src_sig = _mask_signature(mask)
    op_sigs = {_mask_signature(_apply_op(mask, op)) for op in OPS}
    return len(op_sigs) == len(OPS) and src_sig not in op_sigs


def _sample_clear_shape(
    rng: np.random.Generator,
    *,
    size_range: tuple[int, int],
    max_height: int,
    max_width: int,
    require_non_square: bool,
    forbidden_canons: set[tuple[tuple[int, int], ...]] | None = None,
) -> np.ndarray:
    forbidden_canons = set() if forbidden_canons is None else set(forbidden_canons)
    for _ in range(300):
        mask = _random_polyomino(
            rng,
            size_range=size_range,
            max_height=max_height,
            max_width=max_width,
            require_non_square=require_non_square,
        )
        if not _clear_demo(mask):
            continue
        if _canonical_shape(mask) in forbidden_canons:
            continue
        return mask
    raise RuntimeError("failed to sample a clear asymmetric shape")


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = np.full((h, w), BG_COLOR, dtype=int)
    patch[mask] = int(color)
    paste(grid, patch, r0, c0, transparent=BG_COLOR)


def _area_free(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, gap: int) -> bool:
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(blocked.shape[0], r0 + h + gap)
    cc1 = min(blocked.shape[1], c0 + w + gap)
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _block_area(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, gap: int) -> None:
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(blocked.shape[0], r0 + h + gap)
    cc1 = min(blocked.shape[1], c0 + w + gap)
    blocked[rr0:rr1, cc0:cc1] = True


def _candidate_positions(
    blocked: np.ndarray,
    divider_r: int,
    divider_c: int,
    foot_h: int,
    foot_w: int,
    region_hint: str,
) -> list[tuple[int, int]]:
    h, w = blocked.shape
    candidates: list[tuple[int, int]] = []
    regions = [region_hint, "top_right" if region_hint == "bottom" else "bottom"]
    seen: set[str] = set()
    for region in regions:
        if region in seen:
            continue
        seen.add(region)
        if region == "top_right":
            r_start = OUTER_MARGIN
            r_end = divider_r - foot_h - 1
            c_start = divider_c + 2
            c_end = w - foot_w - OUTER_MARGIN
        else:
            r_start = divider_r + 2
            r_end = h - foot_h - OUTER_MARGIN
            c_start = OUTER_MARGIN
            c_end = w - foot_w - OUTER_MARGIN
        for r0 in range(r_start, r_end + 1):
            for c0 in range(c_start, c_end + 1):
                if _area_free(blocked, r0, c0, foot_h, foot_w, PLACEMENT_GAP):
                    candidates.append((int(r0), int(c0)))
    return candidates


def _build_instance(rng: np.random.Generator) -> tuple[np.ndarray, dict] | None:
    work = np.full((GRID_LIMIT, GRID_LIMIT), BG_COLOR, dtype=int)
    palette = [int(c) for c in Color.NOT_BLACK]
    rng.shuffle(palette)
    query_colors = palette[:4]
    divider_color = palette[4]
    op_order = list(OPS)
    rng.shuffle(op_order)
    color_to_op = {color: op for color, op in zip(query_colors, op_order)}

    legend_shapes: dict[int, np.ndarray] = {}
    legend_canons: set[tuple[tuple[int, int], ...]] = set()
    for color, op in color_to_op.items():
        for _ in range(200):
            src_mask = _sample_clear_shape(
                rng,
                size_range=(4, 6),
                max_height=4,
                max_width=4,
                require_non_square=False,
                forbidden_canons=legend_canons,
            )
            if _matching_ops(src_mask, _apply_op(src_mask, op)) == [op]:
                legend_shapes[color] = src_mask
                legend_canons.add(_canonical_shape(src_mask))
                break
        else:
            return None

    pair_gap = 2
    band_gap = 1
    row = 1
    max_right = 0
    max_bottom = 0
    legend_meta = []
    for color in query_colors:
        src_mask = legend_shapes[color]
        dst_mask = _apply_op(src_mask, color_to_op[color])
        src_h, src_w = src_mask.shape
        dst_h, dst_w = dst_mask.shape
        band_h = max(src_h, dst_h)
        src_r = row + (band_h - src_h) // 2
        dst_r = row + (band_h - dst_h) // 2
        src_c = 1
        dst_c = src_c + src_w + pair_gap
        legend_meta.append((color, src_mask, src_r, src_c, dst_mask, dst_r, dst_c))
        max_right = max(max_right, src_c + src_w - 1, dst_c + dst_w - 1)
        max_bottom = max(max_bottom, src_r + src_h - 1, dst_r + dst_h - 1)
        row += band_h + band_gap

    divider_r = max_bottom + 2
    divider_c = max_right + 2
    if divider_r >= GRID_LIMIT - 5 or divider_c >= GRID_LIMIT - 8:
        return None

    work[divider_r, : divider_c + 1] = divider_color
    work[: divider_r + 1, divider_c] = divider_color
    for color, src_mask, src_r, src_c, dst_mask, dst_r, dst_c in legend_meta:
        _paint_mask(work, src_mask, src_r, src_c, color)
        _paint_mask(work, dst_mask, dst_r, dst_c, color)

    blocked = np.zeros_like(work, dtype=bool)
    blocked[:divider_r, :divider_c] = True
    blocked[divider_r, : divider_c + 1] = True
    blocked[: divider_r + 1, divider_c] = True

    shared_mask = _sample_clear_shape(
        rng,
        size_range=(5, 6),
        max_height=4,
        max_width=5,
        require_non_square=True,
        forbidden_canons=legend_canons,
    )
    if len({_mask_signature(_apply_op(shared_mask, op)) for op in OPS}) != len(OPS):
        return None

    query_specs = []
    shared_canon = _canonical_shape(shared_mask)
    for idx, color in enumerate(query_colors):
        region_hint = "top_right" if idx % 2 == 0 else "bottom"
        query_specs.append(
            {
                "color": color,
                "src_mask": shared_mask,
                "op": color_to_op[color],
                "region_hint": region_hint,
                "shared": True,
                "canon": shared_canon,
            }
        )
        if rng.random() < 0.75:
            extra_mask = _sample_clear_shape(
                rng,
                size_range=(4, 6),
                max_height=4,
                max_width=5,
                require_non_square=(color_to_op[color] == "rot90"),
                forbidden_canons=legend_canons,
            )
            query_specs.append(
                {
                    "color": color,
                    "src_mask": extra_mask,
                    "op": color_to_op[color],
                    "region_hint": "bottom" if region_hint == "top_right" else "top_right",
                    "shared": False,
                    "canon": _canonical_shape(extra_mask),
                }
            )

    rng.shuffle(query_specs)
    queries_meta = []
    for spec in query_specs:
        src_mask = spec["src_mask"]
        dst_mask = _apply_op(src_mask, spec["op"])
        foot_h = max(src_mask.shape[0], dst_mask.shape[0])
        foot_w = max(src_mask.shape[1], dst_mask.shape[1])
        candidates = _candidate_positions(blocked, divider_r, divider_c, foot_h, foot_w, spec["region_hint"])
        if not candidates:
            return None
        r0, c0 = candidates[int(rng.integers(len(candidates)))]
        _paint_mask(work, src_mask, r0, c0, spec["color"])
        _block_area(blocked, r0, c0, foot_h, foot_w, PLACEMENT_GAP)
        queries_meta.append(
            {
                "color": int(spec["color"]),
                "bbox": (int(r0), int(c0), int(r0 + src_mask.shape[0] - 1), int(c0 + src_mask.shape[1] - 1)),
                "src_mask": src_mask.copy(),
                "dst_mask": dst_mask.copy(),
                "shared": bool(spec["shared"]),
                "canon": spec["canon"],
            }
        )

    used = np.argwhere(work != BG_COLOR)
    max_r, max_c = used.max(axis=0)
    need_r = int(max_r)
    need_c = int(max_c)
    for item in queries_meta:
        r0, c0, _r1, _c1 = item["bbox"]
        need_r = max(need_r, int(r0 + item["dst_mask"].shape[0] - 1))
        need_c = max(need_c, int(c0 + item["dst_mask"].shape[1] - 1))
    extra_bottom = int(rng.integers(0, min(3, GRID_LIMIT - need_r - 1) + 1))
    extra_right = int(rng.integers(0, min(3, GRID_LIMIT - need_c - 1) + 1))
    final = work[: need_r + 1 + extra_bottom, : need_c + 1 + extra_right].copy()

    meta = {
        "divider_color": int(divider_color),
        "divider_r": int(divider_r),
        "divider_c": int(divider_c),
        "color_to_op": dict(color_to_op),
        "legend_canons": set(legend_canons),
        "queries": queries_meta,
        "shared_canon": shared_canon,
    }
    return final, meta


def _expected_from_meta(grid: np.ndarray, meta: dict) -> np.ndarray:
    out = grid.copy()
    for item in meta["queries"]:
        color = int(item["color"])
        r0, c0, r1, c1 = item["bbox"]
        view = out[r0 : r1 + 1, c0 : c1 + 1]
        view[item["src_mask"]] = BG_COLOR
        patch = np.full(item["dst_mask"].shape, BG_COLOR, dtype=int)
        patch[item["dst_mask"]] = color
        paste(out, patch, r0, c0, transparent=BG_COLOR)
    return out


def _validate_instance(grid: np.ndarray, meta: dict) -> None:
    assert grid.ndim == 2 and max(grid.shape) <= GRID_LIMIT
    assert meta["shared_canon"] not in meta["legend_canons"], "shared query shape must differ from legend shapes"

    divider_color, divider_r, divider_c, inferred = _extract_legend_operations(grid, BG_COLOR)
    assert divider_color == meta["divider_color"]
    assert divider_r == meta["divider_r"] and divider_c == meta["divider_c"]
    assert inferred == meta["color_to_op"], "legend must decode to the intended color->operation mapping"

    by_canon: dict[tuple[tuple[int, int], ...], set[int]] = {}
    distinguishes: dict[int, bool] = {color: False for color in meta["color_to_op"]}
    for item in meta["queries"]:
        by_canon.setdefault(item["canon"], set()).add(int(item["color"]))
        correct = _mask_signature(item["dst_mask"])
        wrong = {_mask_signature(_apply_op(item["src_mask"], op)) for op in OPS if op != meta["color_to_op"][item["color"]]}
        if correct not in wrong:
            distinguishes[int(item["color"])] = True
        assert item["canon"] not in meta["legend_canons"], "query shape should not be a legend-shape lookup decoy"
        if meta["color_to_op"][item["color"]] == "rot90":
            assert item["src_mask"].shape[0] != item["src_mask"].shape[1], "rot90 should be shown on a non-square query"
    assert any(len(colors) >= 2 for colors in by_canon.values()), "at least one query shape must repeat across colors"
    assert all(distinguishes.values()), "every color needs a query that distinguishes its operation from all others"

    expected = _expected_from_meta(grid, meta)
    actual = main(grid)
    assert np.array_equal(actual, expected), "main() must match the independently constructed target"


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        built = _build_instance(rng)
        if built is None:
            continue
        grid, meta = built
        _validate_instance(grid, meta)
        return grid
    raise RuntimeError("failed to generate a valid legend-demonstrates-transform input")
