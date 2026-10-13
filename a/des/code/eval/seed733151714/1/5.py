from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
GREY = 5
DIR_ORDER = ("up", "down", "left", "right")
DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRY_LIMIT = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(cells, dtype=int))
    return comps


def _touches_edge(coords: np.ndarray, h: int, w: int) -> bool:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(
        np.any((rows == 0) | (rows == h - 1) | (cols == 0) | (cols == w - 1))
    )


def _direction_from_coords(
    coords: np.ndarray, seed: tuple[int, int]
) -> str | None:
    center = coords.mean(axis=0)
    dr = float(seed[0]) - float(center[0])
    dc = float(seed[1]) - float(center[1])
    if abs(dr) < 1e-9 and abs(dc) < 1e-9:
        return None
    if abs(dr) > abs(dc):
        return "down" if dr > 0 else "up"
    if abs(dc) > abs(dr):
        return "right" if dc > 0 else "left"
    return None


def _extract_regions(
    grid: np.ndarray,
) -> tuple[list[dict[str, object]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    region_map = np.full((h, w), -1, dtype=int)
    regions: list[dict[str, object]] = []

    for coords in _components(g != GREY):
        if _touches_edge(coords, h, w):
            continue

        values = [int(g[r, c]) for r, c in coords]
        seed_pixels = [(int(r), int(c), int(g[r, c])) for r, c in coords if int(g[r, c]) != BLACK]
        assert len(seed_pixels) == 1, "each enclosed region must contain exactly one colored seed"

        sr, sc, color = seed_pixels[0]
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        region_idx = len(regions)
        region_map[rows, cols] = region_idx
        regions.append(
            {
                "coords": coords,
                "seed": (sr, sc),
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "height": int(r1 - r0 + 1),
                "width": int(c1 - c0 + 1),
                "values": values,
            }
        )

    assert regions, "expected enclosed seed regions"
    colors = [int(region["color"]) for region in regions]
    assert len(colors) == len(set(colors)), "seed colors must be distinct"
    return regions, region_map


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow one branch from each seed away from its region centroid, using the region's orthogonal span as length,
    then bloom a 3x3 flower colored by the seed of the endpoint's region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    regions, region_map = _extract_regions(g)
    out = g.copy()

    for region_idx, region in enumerate(regions):
        coords = np.asarray(region["coords"], dtype=int)
        seed_r, seed_c = region["seed"]  # type: ignore[index]
        seed_color = int(region["color"])
        direction = _direction_from_coords(coords, (int(seed_r), int(seed_c)))
        assert direction is not None, "seed must not be exactly centered or diagonally ambiguous"

        length = int(region["height"]) if direction in {"left", "right"} else int(region["width"])
        dr, dc = DIR_VECS[direction]
        end_r = int(seed_r + dr * length)
        end_c = int(seed_c + dc * length)
        assert 0 <= end_r < g.shape[0] and 0 <= end_c < g.shape[1], "branch endpoint left the grid"

        target_region = int(region_map[end_r, end_c])
        assert target_region >= 0, "branch endpoint must land in an enclosed region"
        flower_color = int(regions[target_region]["color"])

        for step in range(1, length + 1):
            rr = int(seed_r + dr * step)
            cc = int(seed_c + dc * step)
            out[rr, cc] = seed_color

        for fr in (-1, 0, 1):
            for fc in (-1, 0, 1):
                rr = end_r + fr
                cc = end_c + fc
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                    out[rr, cc] = flower_color

        assert region_idx == int(region_map[int(seed_r), int(seed_c)])

    return out


def _build_layout() -> tuple[np.ndarray, list[dict[str, object]], np.ndarray]:
    margin = random.choice((1, 2))
    row_heights = [random.randint(3, 4), random.randint(7, 9)]
    col_widths = [random.randint(7, 9), random.randint(3, 4)]
    if random.random() < 0.5:
        row_heights.reverse()
    if random.random() < 0.5:
        col_widths.reverse()

    h = 2 * margin + 2 + sum(row_heights) + 1
    w = 2 * margin + 2 + sum(col_widths) + 1
    grid = np.full((h, w), BLACK, dtype=int)

    top = margin
    left = margin
    bottom = h - margin - 1
    right = w - margin - 1
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY
    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY

    split_r = top + 1 + row_heights[0]
    split_c = left + 1 + col_widths[0]
    grid[split_r, left : right + 1] = GREY
    grid[top : bottom + 1, split_c] = GREY

    boxes: list[dict[str, object]] = []
    region_map = np.full((h, w), -1, dtype=int)
    region_id = 0
    r = top + 1
    for row_h in row_heights:
        c = left + 1
        for col_w in col_widths:
            r0, c0 = int(r), int(c)
            r1, c1 = int(r + row_h - 1), int(c + col_w - 1)
            coords = np.array(
                [(rr, cc) for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)],
                dtype=int,
            )
            region_map[r0 : r1 + 1, c0 : c1 + 1] = region_id
            boxes.append(
                {
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "height": int(row_h),
                    "width": int(col_w),
                }
            )
            c += col_w + 1
            region_id += 1
        r += row_h + 1

    return grid, boxes, region_map


def _candidate_groups(
    boxes: list[dict[str, object]],
    region_map: np.ndarray,
    shape: tuple[int, int],
) -> list[dict[str, list[dict[str, object]]]]:
    h, w = shape
    frame_top = int(np.where(region_map >= 0)[0].min()) - 1
    frame_left = int(np.where(region_map >= 0)[1].min()) - 1
    frame_bottom = int(np.where(region_map >= 0)[0].max()) + 1
    frame_right = int(np.where(region_map >= 0)[1].max()) + 1

    groups: list[dict[str, list[dict[str, object]]]] = []
    for region_idx, box in enumerate(boxes):
        coords = np.asarray(box["coords"], dtype=int)
        by_dir = {direction: [] for direction in DIR_ORDER}
        length_h = int(box["height"])
        length_w = int(box["width"])
        for seed_r, seed_c in coords:
            direction = _direction_from_coords(coords, (int(seed_r), int(seed_c)))
            if direction is None:
                continue
            length = length_h if direction in {"left", "right"} else length_w
            dr, dc = DIR_VECS[direction]
            end_r = int(seed_r + dr * length)
            end_c = int(seed_c + dc * length)
            if not (
                frame_top + 1 <= end_r <= frame_bottom - 1
                and frame_left + 1 <= end_c <= frame_right - 1
            ):
                continue

            target = int(region_map[end_r, end_c])
            if target < 0:
                continue

            stem = {
                (int(seed_r + dr * step), int(seed_c + dc * step))
                for step in range(1, length + 1)
            }
            flower = {
                (int(end_r + fr), int(end_c + fc))
                for fr in (-1, 0, 1)
                for fc in (-1, 0, 1)
            }
            by_dir[direction].append(
                {
                    "region": region_idx,
                    "seed": (int(seed_r), int(seed_c)),
                    "dir": direction,
                    "length": int(length),
                    "endpoint": (int(end_r), int(end_c)),
                    "target": int(target),
                    "same": bool(target == region_idx),
                    "stem": stem,
                    "flower": flower,
                    "draw": stem | flower,
                }
            )

        for direction in DIR_ORDER:
            random.shuffle(by_dir[direction])
        groups.append(by_dir)
    return groups


def _choose_specs(
    groups: list[dict[str, list[dict[str, object]]]],
) -> list[dict[str, object]] | None:
    best_score: tuple[int, int, int] | None = None
    best_specs: list[dict[str, object]] | None = None

    perms = list(itertools.permutations(DIR_ORDER))
    random.shuffle(perms)
    for perm in perms:
        option_lists = [groups[i][perm[i]] for i in range(4)]
        if any(not options for options in option_lists):
            continue

        chosen: list[dict[str, object] | None] = [None, None, None, None]
        order = sorted(range(4), key=lambda i: len(option_lists[i]))

        def backtrack(pos: int) -> None:
            nonlocal best_score, best_specs
            if pos == 4:
                picks = [chosen[i] for i in range(4)]
                assert all(pick is not None for pick in picks)
                picks = [pick for pick in picks if pick is not None]
                if {bool(pick["same"]) for pick in picks} != {False, True}:
                    return

                total_overlap = 0
                flower_overlap = 0
                for i in range(4):
                    for j in range(i + 1, 4):
                        total_overlap += len(
                            set(picks[i]["draw"]) & set(picks[j]["draw"])
                        )
                        flower_overlap += len(
                            set(picks[i]["flower"]) & set(picks[j]["flower"])
                        )
                score = (flower_overlap * 10 + total_overlap, flower_overlap, total_overlap)
                if best_score is None or score < best_score:
                    best_score = score
                    best_specs = [dict(pick) for pick in picks]
                return

            idx = order[pos]
            for spec in option_lists[idx]:
                endpoint = tuple(spec["endpoint"])
                seed = tuple(spec["seed"])
                bad = False
                for other in chosen:
                    if other is None:
                        continue
                    if endpoint == tuple(other["endpoint"]):
                        bad = True
                        break
                    if seed in set(other["draw"]) or tuple(other["seed"]) in set(spec["draw"]):
                        bad = True
                        break
                if bad:
                    continue
                chosen[idx] = spec
                backtrack(pos + 1)
                chosen[idx] = None

        backtrack(0)

    return best_specs


def _render_expected(
    grid: np.ndarray,
    specs: list[dict[str, object]],
    region_colors: list[int],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for spec in specs:
        seed_r, seed_c = spec["seed"]  # type: ignore[index]
        direction = str(spec["dir"])
        length = int(spec["length"])
        seed_color = int(region_colors[int(spec["region"])])
        dr, dc = DIR_VECS[direction]
        for step in range(1, length + 1):
            rr = int(seed_r + dr * step)
            cc = int(seed_c + dc * step)
            out[rr, cc] = seed_color

        end_r, end_c = spec["endpoint"]  # type: ignore[index]
        flower_color = int(region_colors[int(spec["target"])])
        out[int(end_r) - 1 : int(end_r) + 2, int(end_c) - 1 : int(end_c) + 2] = flower_color
    return out


def _attempt_generate() -> np.ndarray | None:
    grid, boxes, region_map = _build_layout()
    groups = _candidate_groups(boxes, region_map, grid.shape)
    if any(not any(groups[i][direction] for direction in DIR_ORDER) for i in range(4)):
        return None

    specs = _choose_specs(groups)
    if specs is None:
        return None

    region_colors = random.sample(SEED_COLORS, 4)
    for region_idx, spec in enumerate(specs):
        sr, sc = spec["seed"]  # type: ignore[index]
        grid[int(sr), int(sc)] = int(region_colors[region_idx])

    expected = _render_expected(grid, specs, region_colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    for region_idx, spec in enumerate(specs):
        sr, sc = spec["seed"]  # type: ignore[index]
        if int(actual[int(sr), int(sc)]) != int(region_colors[region_idx]):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
