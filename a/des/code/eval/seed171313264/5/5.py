from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 300
PAIR_COUNT_CHOICES = (2, 3)
GRID_H_MIN = 14
GRID_H_MAX = 24
GRID_W_MIN = 18
GRID_W_MAX = 36


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    seen = {tuple(map(int, pts[0]))}
    q = deque([tuple(map(int, pts[0]))])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _spans_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _component_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            bbox = bbox_coords(arr)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": bbox,
                    "mask": _component_mask(arr, bbox),
                }
            )
    return components


def _find_legend_row(grid: np.ndarray, bg: int) -> int:
    rows = np.flatnonzero(np.any(grid != bg, axis=1))
    assert rows.size > 0, "grid must contain a legend row"
    return int(rows[-1])


def _pair_orientation(
    comp_a: dict[str, object], comp_b: dict[str, object], scene: np.ndarray, bg: int
) -> str | None:
    a0, a1, a2, a3 = comp_a["bbox"]  # type: ignore[index]
    b0, b1, b2, b3 = comp_b["bbox"]  # type: ignore[index]
    ha = int(a2 - a0 + 1)
    wa = int(a3 - a1 + 1)
    hb = int(b2 - b0 + 1)
    wb = int(b3 - b1 + 1)
    if (ha, wa) != (hb, wb):
        return None

    if int(a0) == int(b0) and int(a2) == int(b2):
        left, right = (comp_a, comp_b) if int(a1) < int(b1) else (comp_b, comp_a)
        l0, l1, l2, l3 = left["bbox"]  # type: ignore[index]
        r0, r1, r2, r3 = right["bbox"]  # type: ignore[index]
        if int(r1) - int(l1) == wa + 1:
            gap_col = int(l3) + 1
            if np.all(scene[int(l0) : int(l2) + 1, gap_col] == bg):
                return "h"

    if int(a1) == int(b1) and int(a3) == int(b3):
        top, bottom = (comp_a, comp_b) if int(a0) < int(b0) else (comp_b, comp_a)
        t0, t1, t2, t3 = top["bbox"]  # type: ignore[index]
        b0, b1, b2, b3 = bottom["bbox"]  # type: ignore[index]
        if int(b0) - int(t0) == ha + 1:
            gap_row = int(t2) + 1
            if np.all(scene[gap_row, int(t1) : int(t3) + 1] == bg):
                return "v"

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair equal-size neighboring shapes, subtract the smaller mask from the larger one,
    move the remainder to the smaller shape's position, and recolor by the bottom legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend_row = _find_legend_row(g, bg)
    legend_cols = np.flatnonzero(g[legend_row] != bg)
    legend_colors = [int(g[legend_row, c]) for c in legend_cols]
    assert legend_colors, "legend row must contain at least one color"

    scene = g.copy()
    scene[legend_row, :] = bg
    components = _find_components(scene, bg)
    assert len(components) == 2 * len(legend_colors), "scene must contain exactly two components per legend color"

    neighbors: list[list[int]] = [[] for _ in components]
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            if _pair_orientation(components[i], components[j], scene, bg) is not None:
                neighbors[i].append(j)
                neighbors[j].append(i)

    assert all(len(js) == 1 for js in neighbors), "each component must have exactly one valid partner"

    pairs: list[dict[str, object]] = []
    used: set[int] = set()
    for i, js in enumerate(neighbors):
        if i in used:
            continue
        j = js[0]
        assert i in neighbors[j], "pairing must be symmetric"
        used.add(i)
        used.add(j)

        comp_a = components[i]
        comp_b = components[j]
        area_a = int(comp_a["area"])
        area_b = int(comp_b["area"])
        assert area_a != area_b, "one component must be strictly larger than the other"
        ink = comp_a if area_a > area_b else comp_b
        mold = comp_b if area_a > area_b else comp_a

        ink_mask = np.asarray(ink["mask"], dtype=bool)
        mold_mask = np.asarray(mold["mask"], dtype=bool)
        leftover = ink_mask & ~mold_mask
        assert leftover.any(), "subtraction must leave at least one pixel"

        ia0, ia1, ia2, ia3 = ink["bbox"]  # type: ignore[index]
        ma0, ma1, ma2, ma3 = mold["bbox"]  # type: ignore[index]
        center_c = (ia1 + ia3 + ma1 + ma3) / 4.0
        center_r = (ia0 + ia2 + ma0 + ma2) / 4.0
        pairs.append(
            {
                "mold_bbox": mold["bbox"],
                "leftover": leftover,
                "center_c": center_c,
                "center_r": center_r,
            }
        )

    assert len(pairs) == len(legend_colors), "legend size must match pair count"
    pairs.sort(key=lambda item: (float(item["center_c"]), float(item["center_r"])))

    out = np.full_like(g, bg)
    out[legend_row, :] = g[legend_row, :]
    for idx, pair in enumerate(pairs):
        color = int(legend_colors[idx])
        r0, c0, _r1, _c1 = pair["mold_bbox"]  # type: ignore[index]
        leftover = np.asarray(pair["leftover"], dtype=bool)
        rr, cc = np.where(leftover)
        out[rr + int(r0), cc + int(c0)] = color
    return out


def _draw_random_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int], rng: np.random.Generator) -> None:
    r, c = start
    er, ec = end
    mask[r, c] = True
    while (r, c) != (er, ec):
        choices: list[tuple[int, int]] = []
        if r < er:
            choices.append((r + 1, c))
        elif r > er:
            choices.append((r - 1, c))
        if c < ec:
            choices.append((r, c + 1))
        elif c > ec:
            choices.append((r, c - 1))
        nr, nc = choices[int(rng.integers(len(choices)))]
        r, c = nr, nc
        mask[r, c] = True


def _grow_mask(mask: np.ndarray, target_area: int, rng: np.random.Generator) -> np.ndarray | None:
    h, w = mask.shape
    while int(mask.sum()) < target_area:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for nr, nc in neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        nr, nc = list(frontier)[int(rng.integers(len(frontier)))]
        mask[nr, nc] = True
    return mask


def _make_ink_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(80):
        mask = np.zeros((h, w), dtype=bool)
        hub = (int(rng.integers(h)), int(rng.integers(w)))
        anchors = [
            (0, int(rng.integers(w))),
            (h - 1, int(rng.integers(w))),
            (int(rng.integers(h)), 0),
            (int(rng.integers(h)), w - 1),
        ]
        for anchor in anchors:
            _draw_random_path(mask, hub, anchor, rng)
        base_area = int(mask.sum())
        min_target = max(base_area + 2, int(np.ceil(0.55 * h * w)))
        max_target = min(h * w - 1, max(min_target, int(np.ceil(0.82 * h * w))))
        if max_target < min_target:
            continue
        target = int(rng.integers(min_target, max_target + 1))
        grown = _grow_mask(mask, target, rng)
        if grown is None:
            continue
        if _spans_bbox(grown) and _mask_connected(grown) and int(grown.sum()) < h * w:
            return grown
    return None


def _removal_candidates(mask: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = mask.shape
    candidates: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        trial = mask.copy()
        trial[rr, cc] = False
        if not _spans_bbox(trial):
            continue
        if not _mask_connected(trial):
            continue
        degree = 0
        for nr, nc in neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                degree += 1
        candidates.append((degree, rr, cc))
    return candidates


def _shrink_connected_subset(mask: np.ndarray, target_area: int, rng: np.random.Generator) -> np.ndarray | None:
    current = mask.copy()
    while int(current.sum()) > target_area:
        candidates = _removal_candidates(current)
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        keep = max(1, (len(candidates) + 1) // 2)
        _degree, rr, cc = candidates[int(rng.integers(keep))]
        current[rr, cc] = False
    if int(current.sum()) != target_area:
        return None
    if not (_spans_bbox(current) and _mask_connected(current)):
        return None
    return current


def _make_pair_spec(orientation: str, rng: np.random.Generator) -> dict[str, object] | None:
    for _ in range(100):
        if orientation == "h":
            h = int(rng.integers(4, 7))
            w = int(rng.integers(3, 6))
        else:
            h = int(rng.integers(3, 6))
            w = int(rng.integers(4, 7))

        ink = _make_ink_mask(h, w, rng)
        if ink is None:
            continue

        ink_area = int(ink.sum())
        low = max(4, int(np.ceil(0.5 * ink_area)))
        high = ink_area - max(2, int(np.ceil(0.18 * ink_area)))
        if high < low:
            continue
        mold_area = int(rng.integers(low, high + 1))
        mold = _shrink_connected_subset(ink, mold_area, rng)
        if mold is None:
            continue

        leftover = ink & ~mold
        if int(leftover.sum()) < 2:
            continue

        return {
            "orientation": orientation,
            "h": h,
            "w": w,
            "ink_mask": ink,
            "mold_mask": mold,
            "extent_h": h if orientation == "h" else 2 * h + 1,
            "extent_w": 2 * w + 1 if orientation == "h" else w,
            "mold_first": bool(rng.integers(2)),
        }
    return None


def _split_nonnegative(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[int(rng.integers(parts))] += 1
    return values


def _sample_orientations(n_pairs: int, rng: np.random.Generator) -> list[str]:
    if n_pairs == 2:
        values = ["h", "v"]
    else:
        values = ["h", "v", "h" if bool(rng.integers(2)) else "v"]
    return [str(v) for v in rng.permutation(values)]


def _sample_legend_columns(width: int, count: int, rng: np.random.Generator) -> list[int]:
    min_left = 1
    min_right = 1
    min_gap = 1
    min_width = min_left + count + min_gap * (count - 1) + min_right
    if min_width > width:
        raise ValueError("legend does not fit")
    slack = width - min_width
    extras = _split_nonnegative(slack, count + 1, rng)
    cols: list[int] = []
    c = min_left + extras[0]
    for i in range(count):
        cols.append(c)
        c += 1
        if i < count - 1:
            c += min_gap + extras[i + 1]
    return cols


def _place_pair(
    grid: np.ndarray,
    spec: dict[str, object],
    top: int,
    left: int,
    ink_color: int,
    mold_color: int,
) -> dict[str, object]:
    h = int(spec["h"])
    w = int(spec["w"])
    orientation = str(spec["orientation"])
    mold_first = bool(spec["mold_first"])
    ink_mask = np.asarray(spec["ink_mask"], dtype=bool)
    mold_mask = np.asarray(spec["mold_mask"], dtype=bool)

    if orientation == "h":
        if mold_first:
            mold_pos = (top, left)
            ink_pos = (top, left + w + 1)
        else:
            ink_pos = (top, left)
            mold_pos = (top, left + w + 1)
    else:
        if mold_first:
            mold_pos = (top, left)
            ink_pos = (top + h + 1, left)
        else:
            ink_pos = (top, left)
            mold_pos = (top + h + 1, left)

    ir, ic = ink_pos
    mr, mc = mold_pos
    ink_rows, ink_cols = np.where(ink_mask)
    mold_rows, mold_cols = np.where(mold_mask)
    grid[ink_rows + ir, ink_cols + ic] = ink_color
    grid[mold_rows + mr, mold_cols + mc] = mold_color

    return {
        "ink_mask": ink_mask,
        "mold_mask": mold_mask,
        "mold_pos": mold_pos,
    }


def _render_expected(height: int, width: int, placed: list[dict[str, object]], legend_cols: list[int], legend_colors: list[int]) -> np.ndarray:
    out = np.full((height, width), BG, dtype=int)
    for col, color in zip(legend_cols, legend_colors):
        out[height - 1, int(col)] = int(color)
    for item, color in zip(placed, legend_colors):
        leftover = np.asarray(item["ink_mask"], dtype=bool) & ~np.asarray(item["mold_mask"], dtype=bool)
        mr, mc = item["mold_pos"]  # type: ignore[misc]
        rr, cc = np.where(leftover)
        out[rr + int(mr), cc + int(mc)] = int(color)
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_pairs = int(rng.choice(PAIR_COUNT_CHOICES))
    orientations = _sample_orientations(n_pairs, rng)
    specs: list[dict[str, object]] = []
    for orientation in orientations:
        spec = _make_pair_spec(orientation, rng)
        if spec is None:
            return None
        specs.append(spec)

    left_margin = int(rng.integers(1, 3))
    right_margin = int(rng.integers(1, 3))
    pair_gaps = [int(rng.integers(2, 5)) for _ in range(n_pairs - 1)]
    min_width = left_margin + right_margin + sum(int(spec["extent_w"]) for spec in specs) + sum(pair_gaps)
    if min_width > GRID_W_MAX:
        return None
    width_low = max(GRID_W_MIN, min_width)
    width = int(rng.integers(width_low, GRID_W_MAX + 1))
    slack = width - min_width
    extras = _split_nonnegative(slack, n_pairs + 1, rng)

    x_positions: list[int] = []
    cursor = left_margin + extras[0]
    for i, spec in enumerate(specs):
        x_positions.append(cursor)
        cursor += int(spec["extent_w"])
        if i < n_pairs - 1:
            cursor += pair_gaps[i] + extras[i + 1]

    max_extent_h = max(int(spec["extent_h"]) for spec in specs)
    height_low = max(GRID_H_MIN, max_extent_h + 2)
    height_high = max(height_low, GRID_H_MAX)
    if height_low > height_high:
        return None
    height = int(rng.integers(height_low, height_high + 1))
    usable_rows = height - 2
    if usable_rows < max_extent_h:
        return None

    all_colors = [int(v) for v in rng.permutation(PALETTE)]
    used_colors = all_colors[: 3 * n_pairs]
    pair_colors = used_colors[: 2 * n_pairs]
    legend_colors = used_colors[2 * n_pairs :]
    legend_cols = _sample_legend_columns(width, n_pairs, rng)

    grid = np.full((height, width), BG, dtype=int)
    placed: list[dict[str, object]] = []
    for i, spec in enumerate(specs):
        top = int(rng.integers(0, usable_rows - int(spec["extent_h"]) + 1))
        placed.append(
            _place_pair(
                grid,
                spec,
                top,
                x_positions[i],
                pair_colors[2 * i],
                pair_colors[2 * i + 1],
            )
        )

    for col, color in zip(legend_cols, legend_colors):
        grid[height - 1, int(col)] = int(color)

    expected = _render_expected(height, width, placed, legend_cols, legend_colors)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
