from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_RETRIES = 600
STAMP_COUNT_RANGE = (3, 6)
REGION_COUNT_RANGE = (2, 4)
REGION_SIZE_RANGE = (4, 8)
STAMP_H_MAX = 4
STAMP_W_MAX = 4


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _extract_stamps(grid: np.ndarray) -> list[dict]:
    mask = (grid != BLACK) & (grid != GREY)
    stamps: list[dict] = []
    for coords in _components(mask):
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "each stamp must be monochromatic"
        color = next(iter(colors))
        r0 = int(coords[:, 0].min())
        r1 = int(coords[:, 0].max())
        c0 = int(coords[:, 1].min())
        c1 = int(coords[:, 1].max())
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert bool((patch == color).all()), "each stamp must be a solid rectangle"
        stamps.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "h": int(r1 - r0 + 1),
                "w": int(c1 - c0 + 1),
                "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
            }
        )
    return stamps


def _nearest_rail(stamp: dict, rail_rows: list[int], rail_cols: list[int]) -> dict:
    r0, c0, r1, c1 = map(int, stamp["bbox"])
    candidates: list[tuple[int, str, int, str]] = []

    for rr in rail_rows:
        if rr < r0:
            candidates.append((int(r0 - rr - 1), "horizontal", int(rr), "top"))
        elif rr > r1:
            candidates.append((int(rr - r1 - 1), "horizontal", int(rr), "bottom"))

    for cc in rail_cols:
        if cc < c0:
            candidates.append((int(c0 - cc - 1), "vertical", int(cc), "left"))
        elif cc > c1:
            candidates.append((int(cc - c1 - 1), "vertical", int(cc), "right"))

    assert candidates, "each stamp must have a nearby rail"
    best_dist = min(d for d, _orient, _pos, _side in candidates)
    best = [item for item in candidates if item[0] == best_dist]
    assert len(best) == 1, "nearest rail must be unique"
    dist, orient, pos, side = best[0]
    return {"distance": dist, "orientation": orient, "position": pos, "side": side}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored stamp along its nearest grey rail by its own area, then print it onto the crossed rails.

    Input structure:
    - The background is black.
    - Grey pixels form complete horizontal or vertical one-cell-wide rails spanning an entire row or column.
    - The rails partition the canvas into rectangular regions.
    - Some regions contain exactly one monochromatic solid rectangle that does not touch any rail.

    Rule:
    - For every colored rectangle, find the unique nearest grey rail and whether that rail is horizontal or vertical.
    - Compute the stamp area as height times width.
    - Erase the original stamp.
    - If the nearest rail is horizontal, translate the whole stamp right by its area; if the nearest rail is vertical, translate it down by its area.
    - Draw the translated stamp in its new position, overwriting any grey rail pixels it covers while leaving all other rail pixels unchanged.

    Key insight:
    - The motion amount is not a side length or gap size: it is the stamp area, and the motion direction depends only on the rail orientation (horizontal -> right, vertical -> down).
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    h, w = g.shape
    rail_rows = [int(r) for r in range(h) if bool((g[r, :] == GREY).all())]
    rail_cols = [int(c) for c in range(w) if bool((g[:, c] == GREY).all())]
    assert rail_rows or rail_cols, "expected at least one rail"

    stamps = _extract_stamps(g)
    out = g.copy()
    out[(out != BLACK) & (out != GREY)] = BLACK

    for stamp in stamps:
        info = _nearest_rail(stamp, rail_rows, rail_cols)
        r0, c0, r1, c1 = map(int, stamp["bbox"])
        area = int(stamp["area"])
        if info["orientation"] == "horizontal":
            nr0, nc0 = r0, c0 + area
        else:
            nr0, nc0 = r0 + area, c0
        nr1 = nr0 + stamp["h"] - 1
        nc1 = nc0 + stamp["w"] - 1
        assert 0 <= nr0 <= nr1 < h and 0 <= nc0 <= nc1 < w, "translated stamp must stay in bounds"
        out[nr0 : nr1 + 1, nc0 : nc1 + 1] = int(stamp["color"])

    return out


def _sample_partition(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(200):
        nr = int(rng.integers(REGION_COUNT_RANGE[0], REGION_COUNT_RANGE[1] + 1))
        nc = int(rng.integers(REGION_COUNT_RANGE[0], REGION_COUNT_RANGE[1] + 1))
        row_sizes = rng.integers(REGION_SIZE_RANGE[0], REGION_SIZE_RANGE[1] + 1, size=nr)
        col_sizes = rng.integers(REGION_SIZE_RANGE[0], REGION_SIZE_RANGE[1] + 1, size=nc)
        if int(row_sizes.sum() + nr - 1) <= 30 and int(col_sizes.sum() + nc - 1) <= 30:
            return row_sizes.astype(int), col_sizes.astype(int)
    raise RuntimeError("failed to sample a valid partition")


def _build_layout(row_sizes: np.ndarray, col_sizes: np.ndarray) -> tuple[int, int, list[int], list[int], list[dict]]:
    h = int(row_sizes.sum() + len(row_sizes) - 1)
    w = int(col_sizes.sum() + len(col_sizes) - 1)

    rail_rows: list[int] = []
    rail_cols: list[int] = []
    regions: list[dict] = []

    row_starts: list[int] = []
    cur = 0
    for i, size in enumerate(row_sizes):
        row_starts.append(cur)
        cur += int(size)
        if i != len(row_sizes) - 1:
            rail_rows.append(cur)
            cur += 1

    col_starts: list[int] = []
    cur = 0
    for j, size in enumerate(col_sizes):
        col_starts.append(cur)
        cur += int(size)
        if j != len(col_sizes) - 1:
            rail_cols.append(cur)
            cur += 1

    for i, rs in enumerate(row_starts):
        for j, cs in enumerate(col_starts):
            r0 = int(rs)
            r1 = int(rs + row_sizes[i] - 1)
            c0 = int(cs)
            c1 = int(cs + col_sizes[j] - 1)
            regions.append({"ri": i, "ci": j, "bbox": (r0, c0, r1, c1)})

    return h, w, rail_rows, rail_cols, regions


def _has_side(region: dict, side: str, row_count: int, col_count: int) -> bool:
    if side == "top":
        return int(region["ri"]) > 0
    if side == "bottom":
        return int(region["ri"]) < row_count - 1
    if side == "left":
        return int(region["ci"]) > 0
    if side == "right":
        return int(region["ci"]) < col_count - 1
    raise ValueError(side)


def _rect_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _region_rail_distances(
    region: dict,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    row_count: int,
    col_count: int,
) -> dict[str, int]:
    rr0, cc0, rr1, cc1 = map(int, region["bbox"])
    dists: dict[str, int] = {}
    if _has_side(region, "top", row_count, col_count):
        dists["top"] = int(r0 - rr0)
    if _has_side(region, "bottom", row_count, col_count):
        dists["bottom"] = int(rr1 - r1)
    if _has_side(region, "left", row_count, col_count):
        dists["left"] = int(c0 - cc0)
    if _has_side(region, "right", row_count, col_count):
        dists["right"] = int(cc1 - c1)
    return dists


def _placement_candidates(
    region: dict,
    preferred_orientation: str | None,
    preferred_overlap: bool | None,
    h: int,
    w: int,
    rail_rows: list[int],
    rail_cols: list[int],
    row_count: int,
    col_count: int,
    occupied_final: list[tuple[int, int, int, int]],
    blocked_regions: list[tuple[int, int, int, int]],
) -> list[dict]:
    rr0, cc0, rr1, cc1 = map(int, region["bbox"])
    region_h = int(rr1 - rr0 + 1)
    region_w = int(cc1 - cc0 + 1)
    max_sh = min(STAMP_H_MAX, region_h - 2)
    max_sw = min(STAMP_W_MAX, region_w - 2)
    if max_sh <= 0 or max_sw <= 0:
        return []

    candidates: list[dict] = []
    for sh in range(1, max_sh + 1):
        for sw in range(1, max_sw + 1):
            for r0 in range(rr0 + 1, rr1 - sh + 1):
                r1 = r0 + sh - 1
                for c0 in range(cc0 + 1, cc1 - sw + 1):
                    c1 = c0 + sw - 1
                    dists = _region_rail_distances(region, r0, c0, r1, c1, row_count, col_count)
                    if not dists:
                        continue
                    best_dist = min(dists.values())
                    best_sides = [side for side, dist in dists.items() if dist == best_dist]
                    if len(best_sides) != 1:
                        continue
                    guide = best_sides[0]
                    orientation = "horizontal" if guide in ("top", "bottom") else "vertical"
                    if preferred_orientation is not None and orientation != preferred_orientation:
                        continue

                    area = int(sh * sw)
                    if orientation == "horizontal":
                        final_rect = (int(r0), int(c0 + area), int(r1), int(c1 + area))
                        overlaps = any(final_rect[1] <= cc <= final_rect[3] for cc in rail_cols)
                    else:
                        final_rect = (int(r0 + area), int(c0), int(r1 + area), int(c1))
                        overlaps = any(final_rect[0] <= rr <= final_rect[2] for rr in rail_rows)

                    fr0, fc0, fr1, fc1 = final_rect
                    if not (0 <= fr0 <= fr1 < h and 0 <= fc0 <= fc1 < w):
                        continue
                    if preferred_overlap is not None and overlaps != preferred_overlap:
                        continue
                    if any(_rect_intersects(final_rect, other) for other in occupied_final):
                        continue
                    if any(_rect_intersects(final_rect, other) for other in blocked_regions):
                        continue

                    candidates.append(
                        {
                            "guide": guide,
                            "orientation": orientation,
                            "original_bbox": (int(r0), int(c0), int(r1), int(c1)),
                            "final_bbox": final_rect,
                            "area": area,
                            "overlaps_rail": overlaps,
                        }
                    )
    return candidates


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    row_sizes, col_sizes = _sample_partition(rng)
    row_count = int(len(row_sizes))
    col_count = int(len(col_sizes))
    h, w, rail_rows, rail_cols, regions = _build_layout(row_sizes, col_sizes)

    max_stamps = min(len(regions), len(PALETTE), STAMP_COUNT_RANGE[1])
    if max_stamps < STAMP_COUNT_RANGE[0]:
        return None
    stamp_count = int(rng.integers(STAMP_COUNT_RANGE[0], max_stamps + 1))

    region_indices = rng.choice(len(regions), size=stamp_count, replace=False)
    selected = [regions[int(idx)] for idx in region_indices]
    rng.shuffle(selected)

    orientation_targets = ["horizontal"] * (stamp_count // 2) + ["vertical"] * (stamp_count - stamp_count // 2)
    overlap_targets = [True] * (stamp_count // 2) + [False] * (stamp_count - stamp_count // 2)
    rng.shuffle(orientation_targets)
    rng.shuffle(overlap_targets)

    placements: list[dict] = []
    final_rects: list[tuple[int, int, int, int]] = []
    region_boxes = [tuple(map(int, region["bbox"])) for region in selected]

    for idx, region in enumerate(selected):
        blocked = [box for j, box in enumerate(region_boxes) if j != idx]
        pref_orientation = orientation_targets[idx]
        pref_overlap = overlap_targets[idx]

        candidates = _placement_candidates(
            region,
            pref_orientation,
            pref_overlap,
            h,
            w,
            rail_rows,
            rail_cols,
            row_count,
            col_count,
            final_rects,
            blocked,
        )
        if not candidates:
            candidates = _placement_candidates(
                region,
                pref_orientation,
                None,
                h,
                w,
                rail_rows,
                rail_cols,
                row_count,
                col_count,
                final_rects,
                blocked,
            )
        if not candidates:
            candidates = _placement_candidates(
                region,
                None,
                pref_overlap,
                h,
                w,
                rail_rows,
                rail_cols,
                row_count,
                col_count,
                final_rects,
                blocked,
            )
        if not candidates:
            candidates = _placement_candidates(
                region,
                None,
                None,
                h,
                w,
                rail_rows,
                rail_cols,
                row_count,
                col_count,
                final_rects,
                blocked,
            )
        if not candidates:
            return None

        choice = candidates[int(rng.integers(len(candidates)))]
        placements.append(choice)
        final_rects.append(choice["final_bbox"])

    if len({int(item["area"]) for item in placements}) < 2:
        return None
    if {item["orientation"] for item in placements} != {"horizontal", "vertical"}:
        return None
    if {bool(item["overlaps_rail"]) for item in placements} != {False, True}:
        return None

    colors = rng.choice(PALETTE, size=stamp_count, replace=False)
    grid = np.full((h, w), BLACK, dtype=int)
    for rr in rail_rows:
        grid[rr, :] = GREY
    for cc in rail_cols:
        grid[:, cc] = GREY

    meta: list[dict] = []
    for color, placement in zip(colors, placements):
        r0, c0, r1, c1 = map(int, placement["original_bbox"])
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
        meta.append(
            {
                "color": int(color),
                "bbox": (r0, c0, r1, c1),
                "orientation": placement["orientation"],
                "guide": placement["guide"],
                "area": int(placement["area"]),
                "final_bbox": tuple(map(int, placement["final_bbox"])),
                "overlaps_rail": bool(placement["overlaps_rail"]),
            }
        )

    try:
        stamps = _extract_stamps(grid)
        if len(stamps) != stamp_count:
            return None
        seen = {(item["color"], item["bbox"]) for item in meta}
        for stamp in stamps:
            if (int(stamp["color"]), tuple(map(int, stamp["bbox"]))) not in seen:
                return None
            info = _nearest_rail(stamp, rail_rows, rail_cols)
            matched = next(item for item in meta if int(item["color"]) == int(stamp["color"]))
            if info["orientation"] != matched["orientation"] or info["side"] != matched["guide"]:
                return None
        out = main(grid)
        if out.shape != grid.shape:
            return None
        if not np.any(out != grid):
            return None
        output_mask = (out != BLACK) & (out != GREY)
        out_stamps = _components(output_mask)
        if len(out_stamps) != stamp_count:
            return None
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
