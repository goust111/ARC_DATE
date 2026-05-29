# Variant of 31f7f899
# Dimension: D3 — 源对象：条形结构 (src)
# Change: 3-5个矩形单色垂直条，宽度1或3（奇数），高度1-11（奇数），关于基线上下对称... → 高度可变的多段条

from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg


# Design analysis for the D3 variant
# a) Concrete behavior:
#    The original task had one connected solid rectangle per color/slot. Here each bar keeps the
#    same odd width and the same vertical axis, but is split into multiple monochrome rectangles
#    separated by one-row background gaps. Every bar still crosses the baseline via a central
#    segment, so the scene still reads as "bars anchored to a line", but one bar may now look like
#    rows {-6,-4,-2,0,2,4,6} while another looks like {-3,-2,-1,0,1,2,3} over the same columns.
#    The key changed step is object extraction: the old solver treated each connected component as
#    one whole bar, which becomes wrong because a single slot now produces several disconnected
#    components of the same color. The new solver must first group components by color/slot, then
#    sort the grouped bars by total occupied height = number of filled rows, not by bbox height or
#    tallest segment. Example:
#      old logic on a 5-high solid bar: one component, bbox_h=5, rel rows {-2,-1,0,1,2}.
#      new input bar A: rows {-3,-2,-1,0,1,2,3} as segments [3,1,3], total=7, max segment=3.
#      new input bar B: rows {-6,-4,-2,0,2,4,6} as seven 1-row segments, total=7, max segment=1.
#    Both are multi-segment objects; the transformation redistributes complete row patterns.
#
# b) Complexity calibration:
#    To keep the difference visible but still inferable from few ARC examples, the generator uses a
#    small profile library. All bars are multi-segment, all widths are still globally 1 or 3, and
#    total occupied heights remain the familiar odd ladder {3,5,7,9,11}. The profiles are chosen so
#    that at least one pair reverses under "sort by total occupied height" vs "sort by tallest
#    segment" (specifically totals 5 and 7), making the new ranking rule observable instead of a
#    cosmetic gap insertion.
#
# c) Original main() assumptions that break:
#    - `for comp in find_components(... monochromatic=True ...)`: originally each bar was one
#      connected component. In the variant, one bar yields several components, so component != bar.
#    - `height = bbox_h(b)`: bbox height used to equal true bar height for solid rectangles. With
#      internal gaps this inflates the ranking key and can change order.
#    - `base_cols = ... if r == base_r`: harmless before because every bar crossed the baseline.
#      We preserve that property so anchor recovery remains stable, but we now compute the anchor
#      from all grouped coords instead of per-component coords.
#    - `rel = ... for r, c in pts`: originally copied one component. Now it must copy the union of
#      all segments belonging to the same color/slot, otherwise the output would lose pieces.
#
# d) Placement legality:
#    The generator keeps the original horizontal slot model: bars use one odd width shared by the
#    whole scene and slots are separated by at least one background column. Vertically, every
#    profile is precomputed with a known bbox height; the generator only uses profiles whose bbox
#    fits the chosen odd board size. Because output is produced by moving complete profiles between
#    the same non-overlapping slots, neither input nor output can create overlaps.
#
# e) Visual observability:
#    A human should see that colors stay tied to x-slots while the "gappy silhouettes" migrate.
#    The generator rejects samples where sorting by an alternate key (tallest segment height) would
#    yield the same output. This avoids degeneration into a dataset where the new D3 feature exists
#    visually but never affects the answer.
#
# Post-change solver assumptions:
# - each non-baseline color appears in exactly one slot, possibly split into multiple components;
# - all components of a color share the same odd-width x-range, so their median column is the slot
#   anchor;
# - every bar includes a central segment on the baseline row, preserving the original anchored look;
# - generator guarantees distinct total occupied heights, so sorting by total height is unambiguous.


PROFILE_SPECS: dict[int, tuple[int, tuple[int, ...]]] = {
    3: (1, (1,)),
    5: (3, (1,)),
    7: (1, (1, 1, 1)),
    9: (1, (3, 1)),
    11: (1, (3, 1, 1)),
}


def _find_base(grid: np.ndarray, bg: int) -> tuple[int, int]:
    best = None  # (count, row, color)
    H = int(grid.shape[0])
    for r in range(H):
        vals, counts = np.unique(grid[r, :], return_counts=True)
        for v, cnt in zip(vals, counts):
            v = int(v)
            cnt = int(cnt)
            if v == bg:
                continue
            if best is None or cnt > best[0]:
                best = (cnt, int(r), int(v))
    assert best is not None
    _cnt, base_r, base_col = best
    return int(base_r), int(base_col)


def _segment_lengths(row_offsets: list[int]) -> list[int]:
    if not row_offsets:
        return []
    rows = sorted(int(r) for r in row_offsets)
    lengths: list[int] = []
    run = 1
    for prev, cur in zip(rows, rows[1:]):
        if cur == prev + 1:
            run += 1
        else:
            lengths.append(int(run))
            run = 1
    lengths.append(int(run))
    return lengths


def _extract_objects(grid: np.ndarray) -> tuple[int, int, int, list[dict[str, object]]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    base_r, base_col = _find_base(g, bg)

    by_color: dict[int, list[np.ndarray]] = {}
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {bg, int(base_col)}:
            continue
        by_color.setdefault(col, []).append(np.array(comp["coords"], dtype=int))

    objects: list[dict[str, object]] = []
    for col, chunks in by_color.items():
        pts = np.concatenate(chunks, axis=0)
        coords = [(int(r), int(c)) for r, c in pts]
        cols_sorted = sorted(c for _r, c in coords)
        anchor = int(cols_sorted[len(cols_sorted) // 2])
        row_offsets = sorted({int(r) - int(base_r) for r, _c in coords})
        seg_lengths = _segment_lengths(row_offsets)
        rel = sorted((int(r) - int(base_r), int(c) - int(anchor)) for r, c in coords)
        objects.append(
            {
                "color": int(col),
                "anchor": int(anchor),
                "rel": rel,
                "size": int(len(rel)),
                "total_height": int(len(row_offsets)),
                "max_segment": int(max(seg_lengths)),
                "bbox_height": int(max(row_offsets) - min(row_offsets) + 1),
            }
        )

    return int(H), int(W), int(bg), objects


def _render_from_objects(
    H: int,
    W: int,
    bg: int,
    base_r: int,
    base_col: int,
    slots: list[dict[str, object]],
    shapes: list[dict[str, object]],
) -> np.ndarray:
    out = np.full((H, W), int(bg), dtype=int)
    out[int(base_r), :] = int(base_col)
    for slot, shape in zip(slots, shapes):
        slot_col = int(slot["color"])
        slot_anchor = int(slot["anchor"])
        for dr, dc in shape["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(slot_col)
    return out


def _solve_with_key(grid: np.ndarray, key: str = "total") -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    base_r, base_col = _find_base(g, bg)
    _H, _W, _bg, objects = _extract_objects(g)
    assert H == _H and W == _W and bg == _bg

    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    if key == "total":
        shapes = sorted(
            objects,
            key=lambda o: (
                int(o["total_height"]),
                int(o["max_segment"]),
                int(o["bbox_height"]),
                int(o["anchor"]),
                int(o["color"]),
            ),
        )
    elif key == "max_segment":
        shapes = sorted(
            objects,
            key=lambda o: (
                int(o["max_segment"]),
                int(o["total_height"]),
                int(o["bbox_height"]),
                int(o["anchor"]),
                int(o["color"]),
            ),
        )
    else:
        raise ValueError(f"unknown key: {key}")

    return _render_from_objects(H, W, bg, base_r, base_col, slots, shapes)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group disconnected same-color bar segments into one slot-object, then sort by total occupied rows.

    The fixed left-to-right slots keep their original x-positions and colors.
    What moves between slots is the full multi-segment silhouette of each bar.
    """
    return _solve_with_key(grid, key="total")


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _slot_lefts(n: int, k: int, width: int) -> list[int]:
    used = int(k * width + (k - 1))
    extra = int(n - used)
    if extra < 0:
        raise ValueError("bars do not fit")
    gaps = _rand_composition(extra, k + 1)
    lefts: list[int] = []
    c = int(gaps[0])
    for i in range(k):
        lefts.append(int(c))
        c += int(width)
        if i < k - 1:
            c += 1 + int(gaps[i + 1])
    return lefts


def _profile_rel(width: int, center_height: int, outer_heights: tuple[int, ...]) -> list[tuple[int, int]]:
    half_w = int(width // 2)
    rel: list[tuple[int, int]] = []

    def add_segment(r0: int, r1: int) -> None:
        for rr in range(int(r0), int(r1) + 1):
            for dc in range(-half_w, half_w + 1):
                rel.append((int(rr), int(dc)))

    center_half = int(center_height // 2)
    add_segment(-center_half, center_half)

    top_end = -center_half - 2
    bottom_start = center_half + 2
    for h in outer_heights:
        h = int(h)
        add_segment(top_end - h + 1, top_end)
        add_segment(bottom_start, bottom_start + h - 1)
        top_end -= h + 1
        bottom_start += h + 1

    return sorted(rel)


def _profile_dict(width: int, total_height: int) -> dict[str, object]:
    center_height, outer_heights = PROFILE_SPECS[int(total_height)]
    rel = _profile_rel(width, int(center_height), tuple(int(x) for x in outer_heights))
    row_offsets = sorted({int(dr) for dr, _dc in rel})
    seg_lengths = _segment_lengths(row_offsets)
    return {
        "rel": rel,
        "total_height": int(len(row_offsets)),
        "max_segment": int(max(seg_lengths)),
        "bbox_height": int(max(row_offsets) - min(row_offsets) + 1),
        "size": int(len(rel)),
    }


def _draw_scene(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    lefts: list[int],
    width: int,
    colors: list[int],
    profiles: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[base_r, :] = int(base_color)
    half_w = int(width // 2)
    for left, color, profile in zip(lefts, colors, profiles):
        anchor = int(left) + half_w
        for dr, dc in profile["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(anchor) + int(dc)
            grid[rr, cc] = int(color)
    return grid


def _selected_totals(k: int, n: int) -> list[int]:
    feasible = [t for t in PROFILE_SPECS if _profile_dict(1, t)["bbox_height"] <= int(n)]
    if 5 not in feasible or 7 not in feasible:
        raise ValueError("board too small for the distinguishing pair")
    chosen = {5, 7}
    others = [t for t in feasible if t not in chosen]
    if len(feasible) < int(k):
        raise ValueError("not enough feasible totals")
    extra = list(np.random.choice(others, size=int(k - 2), replace=False)) if k > 2 else []
    chosen.update(int(x) for x in extra)
    return sorted(int(t) for t in chosen)


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice([15, 17, 19]))
    base_r = int(n // 2)
    bg = 8
    base_color = 6

    max_count = min(5, len([t for t in PROFILE_SPECS if _profile_dict(1, t)["bbox_height"] <= n]))
    if max_count < 3:
        return None
    k = int(np.random.randint(3, max_count + 1))

    feasible_widths = [1]
    if n >= 4 * k - 1:
        feasible_widths.append(3)
    width = int(np.random.choice(feasible_widths))

    totals = _selected_totals(k, n)
    profiles_by_total = {int(t): _profile_dict(width, int(t)) for t in totals}
    if any(int(p["bbox_height"]) > n for p in profiles_by_total.values()):
        return None

    lefts = _slot_lefts(n, k, width)
    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 5, 7, 9], size=k, replace=False)]

    sorted_profiles = [profiles_by_total[int(t)] for t in totals]
    out = _draw_scene(n, base_r, base_color, bg, lefts, width, colors, sorted_profiles)

    perm = np.random.permutation(k)
    if np.array_equal(perm, np.arange(k)):
        a, b = np.random.choice(k, size=2, replace=False)
        perm[a], perm[b] = perm[b], perm[a]
    in_profiles = [sorted_profiles[int(i)] for i in perm]
    inp = _draw_scene(n, base_r, base_color, bg, lefts, width, colors, in_profiles)

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_solve_with_key(inp, key="max_segment"), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
