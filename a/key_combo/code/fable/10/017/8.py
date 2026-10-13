from __future__ import annotations

import math
import numpy as np


def _infer_colors(grid: np.ndarray) -> tuple[int, int, int]:
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 3, "expected exactly three colors: air, wall, and water"
    order = np.argsort(-counts, kind="stable")
    ranked = [(int(colors[i]), int(counts[i])) for i in order]
    bg = ranked[0][0]
    wall, water = ranked[1][0], ranked[2][0]
    assert ranked[1][1] > ranked[2][1], "wall color must be more common than droplet color"
    return bg, wall, water


def _find_open_row_and_spans(grid: np.ndarray, wall: int) -> tuple[int, list[tuple[int, int]]]:
    h, w = grid.shape
    for r in range(h - 1):
        wall_cols = [c for c in range(w) if int(grid[r, c]) == wall]
        if len(wall_cols) < 2:
            continue

        spans: list[tuple[int, int]] = []
        for left, right in zip(wall_cols, wall_cols[1:]):
            if right - left <= 1:
                continue
            cols = range(left + 1, right)
            if any(int(grid[r, c]) == wall for c in cols):
                continue
            if not any(np.any(grid[r + 1 :, c] != wall) for c in cols):
                continue
            if not all(np.any(grid[r + 1 :, c] == wall) for c in cols):
                continue
            spans.append((left + 1, right - 1))
        if spans:
            return r, spans
    raise AssertionError("failed to locate chamber opening row")


def _chamber_mask_and_depths(
    grid: np.ndarray,
    open_row: int,
    span: tuple[int, int],
    wall: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    h, _ = grid.shape
    c0, c1 = span
    mask = np.zeros((h, grid.shape[1]), dtype=bool)
    depths: list[int] = []
    for c in range(c0, c1 + 1):
        wall_rows = np.flatnonzero(grid[open_row + 1 :, c] == wall)
        assert wall_rows.size > 0, "every chamber column needs a floor"
        first_wall = int(open_row + 1 + wall_rows[0])
        assert first_wall > open_row + 1, "each chamber column must have positive depth"
        assert np.all(grid[open_row + 1 : first_wall, c] != wall), "chamber space must be vertically open"
        mask[open_row + 1 : first_wall, c] = True
        depths.append(first_wall - open_row - 1)
    return mask, tuple(depths)


def _fill_mask_for_count(mask: np.ndarray, open_row: int, count: int) -> tuple[np.ndarray, int | None]:
    if count == 0:
        return np.zeros_like(mask, dtype=bool), None

    cumulative = 0
    threshold = None
    for r in range(mask.shape[0] - 1, open_row, -1):
        row_count = int(mask[r].sum())
        if row_count == 0:
            continue
        cumulative += row_count
        if cumulative == count:
            threshold = r
            break
        assert cumulative < count, "droplet count must stop at a complete level"
    assert threshold is not None, "droplet count did not land on a flat level"
    row_ids = np.arange(mask.shape[0])[:, None]
    return mask & (row_ids >= threshold), threshold


def _level_options(mask: np.ndarray, open_row: int, *, allow_full: bool) -> list[dict[str, int]]:
    options = []
    cumulative = 0
    for r in range(mask.shape[0] - 1, open_row, -1):
        row_count = int(mask[r].sum())
        if row_count == 0:
            continue
        cumulative += row_count
        if not allow_full and r == open_row + 1:
            continue
        options.append(
            {
                "volume": cumulative,
                "threshold": r,
                "fill_rows": int(np.count_nonzero(mask.any(axis=1) & (np.arange(mask.shape[0]) >= r))),
            }
        )
    return options


def _geometry_info(grid: np.ndarray, wall: int) -> dict:
    g = np.asarray(grid, dtype=int)
    open_row, spans = _find_open_row_and_spans(g, wall)
    assert np.count_nonzero(g[:open_row, :] == wall) == 0, "walls should start at the opening row"
    chambers = []
    for span in spans:
        mask, depths = _chamber_mask_and_depths(g, open_row, span, wall)
        c0, c1 = span
        chambers.append(
            {
                "span": span,
                "mask": mask,
                "depths": depths,
                "options": _level_options(mask, open_row, allow_full=False),
                "width": c1 - c0 + 1,
            }
        )
    return {"grid": g, "open_row": open_row, "chambers": chambers}


def _scene_info(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, wall, water = _infer_colors(g)
    info = _geometry_info(g, wall)
    chambers = []
    for chamber in info["chambers"]:
        c0, c1 = chamber["span"]
        droplets = int(np.sum(g[: info["open_row"], c0 : c1 + 1] == water))
        fill_mask, threshold = _fill_mask_for_count(chamber["mask"], info["open_row"], droplets)
        chamber = dict(chamber)
        chamber["droplets"] = droplets
        chamber["fill_mask"] = fill_mask
        chamber["threshold"] = threshold
        chambers.append(chamber)
    return {
        "grid": g,
        "bg": bg,
        "wall": wall,
        "water": water,
        "open_row": info["open_row"],
        "chambers": chambers,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pour each chamber's droplet count into the chamber below and level the water over its stepped floor.

    Input structure:
    - The lower part of the canvas contains one open-top container made of one wall color.
    - Vertical wall columns divide the container into separate chambers, and each chamber's floor is a stepped profile.
    - Single-pixel droplets of one water color float above the openings; some chambers may receive none.

    Rule:
    - Assign each droplet to the chamber whose opening lies directly below its column.
    - For each chamber with n assigned droplets, fill exactly n interior cells of that chamber.
    - Fill from the chamber's lowest empty row upward, always completing an entire horizontal level before rising; remove all airborne droplets and keep the walls unchanged.

    Key insight:
    - The droplets encode volume, not height: you must count by chamber, conserve that many cells, and spread them across the irregular basin so the final water surface is flat.
    """
    info = _scene_info(np.asarray(grid, dtype=int))
    g = info["grid"]
    bg = info["bg"]
    water = info["water"]

    out = g.copy()
    out[out == water] = bg
    for chamber in info["chambers"]:
        out[chamber["fill_mask"]] = water
    return out


_GEN_BG = 0
_GEN_COLOR_POOL = tuple(range(1, 10))
_GEN_ATTEMPTS = 500


def _partition_total(rng: np.random.Generator, total: int, parts: int, *, lo: int, hi: int) -> list[int] | None:
    out: list[int] = []

    def rec(remaining: int, slots: int) -> bool:
        if slots == 0:
            return remaining == 0
        min_here = max(lo, remaining - (slots - 1) * hi)
        max_here = min(hi, remaining - (slots - 1) * lo)
        choices = list(range(min_here, max_here + 1))
        rng.shuffle(choices)
        for width in choices:
            out.append(width)
            if rec(remaining - width, slots - 1):
                return True
            out.pop()
        return False

    return out if rec(total, parts) else None


def _make_depth_profile(rng: np.random.Generator, width: int, max_depth: int) -> tuple[int, ...] | None:
    segment_count = int(rng.integers(2, min(4, width) + 1))
    segment_widths = _partition_total(rng, width, segment_count, lo=1, hi=width)
    if segment_widths is None:
        return None
    if max_depth < segment_count + 1:
        return None
    heights = rng.choice(np.arange(2, max_depth + 1), size=segment_count, replace=False).tolist()
    if rng.random() < 0.4:
        heights.sort()
    elif rng.random() < 0.8:
        heights.sort(reverse=True)
    else:
        rng.shuffle(heights)

    depths: list[int] = []
    for seg_w, height in zip(segment_widths, heights):
        depths.extend([int(height)] * int(seg_w))
    if len(set(depths)) < 2:
        return None
    return tuple(depths)


def _build_scene_grid(
    open_row: int,
    widths: list[int],
    profiles: list[tuple[int, ...]],
    wall: int,
    left_margin: int,
    right_margin: int,
    extra_floor_rows: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    max_depth = max(max(profile) for profile in profiles)
    h = open_row + max_depth + 2 + extra_floor_rows
    container_width = sum(widths) + len(widths) + 1
    w = left_margin + container_width + right_margin
    grid = np.full((h, w), _GEN_BG, dtype=int)

    spans: list[tuple[int, int]] = []
    cursor = left_margin
    grid[open_row:, cursor] = wall
    cursor += 1
    for width, depths in zip(widths, profiles):
        c0 = cursor
        c1 = cursor + width - 1
        spans.append((c0, c1))
        for offset, depth in enumerate(depths):
            floor_start = open_row + int(depth) + 1
            grid[floor_start:, c0 + offset] = wall
        cursor = c1 + 1
        grid[open_row:, cursor] = wall
        cursor += 1

    assert max(grid.shape) <= 30
    assert grid.shape[0] <= 30 and grid.shape[1] <= 30
    return grid, spans


def _choose_counts(
    rng: np.random.Generator,
    scene: dict,
    widths: list[int],
    open_row: int,
) -> dict[int, int] | None:
    chambers = scene["chambers"]
    pair_candidates = []
    for i in range(len(chambers)):
        for j in range(i + 1, len(chambers)):
            for oi in chambers[i]["options"]:
                for oj in chambers[j]["options"]:
                    if oi["volume"] != oj["volume"]:
                        continue
                    if oi["threshold"] == oj["threshold"]:
                        continue
                    if oi["volume"] > open_row * widths[i] or oj["volume"] > open_row * widths[j]:
                        continue
                    if oi["fill_rows"] < 2 and oj["fill_rows"] < 2:
                        continue
                    pair_candidates.append((i, j, int(oi["volume"])))
    if not pair_candidates:
        return None

    i, j, pair_volume = pair_candidates[int(rng.integers(0, len(pair_candidates)))]
    counts = {idx: None for idx in range(len(chambers))}
    counts[i] = pair_volume
    counts[j] = pair_volume

    unused = [idx for idx in range(len(chambers)) if counts[idx] is None]
    if unused:
        zero_idx = unused[int(rng.integers(0, len(unused)))]
        counts[zero_idx] = 0

    for idx, chamber in enumerate(chambers):
        if counts[idx] is not None:
            continue
        options = [opt for opt in chamber["options"] if opt["volume"] <= open_row * widths[idx]]
        if not options:
            return None
        if rng.random() < 0.35:
            counts[idx] = 0
        else:
            choice = options[int(rng.integers(0, len(options)))]
            counts[idx] = int(choice["volume"])

    final_counts = {idx: int(value) for idx, value in counts.items()}
    if sum(final_counts.values()) == 0:
        return None
    positive = [idx for idx, value in final_counts.items() if value > 0]
    if len(positive) < 2:
        return None
    if not any(final_counts[idx] == 0 for idx in final_counts):
        return None
    cluster_candidates = [
        idx
        for idx in positive
        if math.ceil(final_counts[idx] / open_row) < widths[idx]
    ]
    if not cluster_candidates:
        return None
    leveled_irregular = False
    for idx in positive:
        fill_mask, _ = _fill_mask_for_count(chambers[idx]["mask"], open_row, final_counts[idx])
        if len(set(chambers[idx]["depths"])) >= 2 and np.count_nonzero(fill_mask.any(axis=1)) >= 2:
            leveled_irregular = True
            break
    if not leveled_irregular:
        return None
    return final_counts


def _place_droplets(
    rng: np.random.Generator,
    grid: np.ndarray,
    spans: list[tuple[int, int]],
    counts: dict[int, int],
    water: int,
    open_row: int,
) -> bool:
    cluster_candidates = [
        idx
        for idx, count in counts.items()
        if count > 0 and math.ceil(count / open_row) < (spans[idx][1] - spans[idx][0] + 1)
    ]
    cluster_idx = cluster_candidates[int(rng.integers(0, len(cluster_candidates)))]

    for idx, (c0, c1) in enumerate(spans):
        count = counts[idx]
        if count == 0:
            continue
        width = c1 - c0 + 1
        all_cols = list(range(c0, c1 + 1))
        if idx == cluster_idx:
            min_focus = math.ceil(count / open_row)
            max_focus = min(width - 1, min_focus + 1)
            focus_size = int(rng.integers(min_focus, max_focus + 1))
            if rng.random() < 0.5:
                focus_cols = all_cols[:focus_size]
            else:
                focus_cols = all_cols[-focus_size:]
        else:
            focus_cols = all_cols
        slots = [(r, c) for c in focus_cols for r in range(open_row)]
        if count > len(slots):
            return False
        chosen = rng.choice(len(slots), size=count, replace=False)
        for slot_idx in np.atleast_1d(chosen):
            r, c = slots[int(slot_idx)]
            grid[r, c] = water
    return True


def _build_expected_output(scene: dict) -> np.ndarray:
    g = scene["grid"]
    bg = scene["bg"]
    water = scene["water"]
    out = g.copy()
    out[out == water] = bg
    for chamber in scene["chambers"]:
        out[chamber["fill_mask"]] = water
    return out


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        scene = _scene_info(grid)
    except AssertionError:
        return False

    chambers = scene["chambers"]
    open_row = scene["open_row"]
    water = scene["water"]

    if len(chambers) < 2:
        return False
    if int(np.count_nonzero(grid[:open_row, :] == water)) == 0:
        return False
    if int(np.count_nonzero(grid[open_row:, :] == water)) != 0:
        return False
    if not any(ch["droplets"] == 0 for ch in chambers):
        return False
    if not any(ch["droplets"] > 0 for ch in chambers):
        return False
    if not any(len(set(ch["depths"])) >= 2 for ch in chambers):
        return False

    same_volume_diff_level = False
    for i in range(len(chambers)):
        for j in range(i + 1, len(chambers)):
            if (
                chambers[i]["droplets"] > 0
                and chambers[i]["droplets"] == chambers[j]["droplets"]
                and chambers[i]["threshold"] != chambers[j]["threshold"]
            ):
                same_volume_diff_level = True
    if not same_volume_diff_level:
        return False

    leveled_irregular = False
    clustered_spread = False
    for ch in chambers:
        fill_rows = np.flatnonzero(ch["fill_mask"].any(axis=1))
        if fill_rows.size >= 2 and len(set(ch["depths"])) >= 2:
            leveled_irregular = True
        if ch["droplets"] > 0:
            c0, c1 = ch["span"]
            droplet_cols = np.flatnonzero(np.any(grid[:open_row, c0 : c1 + 1] == water, axis=0))
            water_cols = np.flatnonzero(np.any(ch["fill_mask"][:, c0 : c1 + 1], axis=0))
            if droplet_cols.size > 0 and water_cols.size > droplet_cols.size:
                clustered_spread = True
    if not leveled_irregular or not clustered_spread:
        return False

    expected = _build_expected_output(scene)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    wall, water = rng.choice(_GEN_COLOR_POOL, size=2, replace=False)
    chamber_count = int(rng.choice(np.array([3, 3, 3, 4, 4, 2])))
    widths = [int(rng.integers(3, 7)) for _ in range(chamber_count)]
    open_row = int(rng.integers(4, 9))
    max_depth = int(rng.integers(5, 10))
    profiles = []
    for width in widths:
        profile = _make_depth_profile(rng, width, max_depth)
        if profile is None:
            return None
        profiles.append(profile)

    container_width = sum(widths) + chamber_count + 1
    if container_width > 28:
        return None
    max_side_budget = 30 - container_width
    if max_side_budget < 2:
        return None
    left_margin = int(rng.integers(1, min(3, max_side_budget - 1) + 1))
    right_limit = min(3, 30 - container_width - left_margin)
    if right_limit < 1:
        return None
    right_margin = int(rng.integers(1, right_limit + 1))

    tallest = max(max(profile) for profile in profiles)
    max_extra_floor = min(2, 30 - (open_row + tallest + 2))
    if max_extra_floor < 0:
        return None
    extra_floor_rows = int(rng.integers(0, max_extra_floor + 1))
    grid, spans = _build_scene_grid(
        open_row,
        widths,
        profiles,
        int(wall),
        left_margin,
        right_margin,
        extra_floor_rows,
    )
    if max(grid.shape) > 30:
        return None

    scene = _geometry_info(grid, int(wall))
    counts = _choose_counts(rng, scene, widths, open_row)
    if counts is None:
        return None

    for idx, chamber in enumerate(scene["chambers"]):
        desired = counts[idx]
        fill_mask, threshold = _fill_mask_for_count(chamber["mask"], open_row, desired)
        chamber["droplets"] = desired
        chamber["fill_mask"] = fill_mask
        chamber["threshold"] = threshold

    if not _place_droplets(rng, grid, spans, counts, int(water), open_row):
        return None
    if not _validate_semantics(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid input for puzzle 8")
