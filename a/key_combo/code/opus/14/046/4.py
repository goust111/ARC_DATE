import random
from typing import Any

import numpy as np


BG = 0
BASELINE = 5
SLOT_COLORS = [2, 1, 3, 4]  # left-to-right: red, blue, green, yellow
MATCHABLE_COLORS = set(SLOT_COLORS)
UNMATCHED_COLOR = 7


def _find_base_row(grid: np.ndarray) -> int:
    full_rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == BASELINE)]
    if full_rows:
        return int(full_rows[-1])
    return int(np.argmax(np.sum(grid == BASELINE, axis=1)))


def _detect_slots(grid: np.ndarray, base_r: int) -> list[dict[str, int]]:
    slots: list[dict[str, int]] = []
    if base_r == 0:
        return slots
    for c in range(grid.shape[1]):
        color = int(grid[base_r - 1, c])
        if color not in MATCHABLE_COLORS:
            continue
        h = 0
        r = base_r - 1
        while r >= 0 and int(grid[r, c]) == color:
            h += 1
            r -= 1
        if h >= 2:
            slots.append({"col": int(c), "color": color, "height": int(h)})
    slots.sort(key=lambda slot: slot["col"])
    assert len(slots) == 4, f"expected 4 slots, found {len(slots)}"
    assert [slot["color"] for slot in slots] == SLOT_COLORS
    return slots


def _render_bars(shape: tuple[int, int], base_r: int, slot_cols: list[int], heights: list[int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    grid[base_r, :] = BASELINE
    for col, color, height in zip(slot_cols, SLOT_COLORS, heights):
        grid[base_r - height : base_r, col] = color
    return grid


def _loose_pixels(grid: np.ndarray, base_r: int, slot_cols: set[int]) -> list[tuple[int, int, int]]:
    pixels: list[tuple[int, int, int]] = []
    for r in range(base_r):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color in (BG, BASELINE) or c in slot_cols:
                continue
            pixels.append((int(r), int(c), color))
    return pixels


def _stop_column(
    bars_grid: np.ndarray,
    r: int,
    start_c: int,
    color: int,
    target_col: int,
) -> int | None:
    if color not in MATCHABLE_COLORS:
        return None

    current = int(start_c)
    while True:
        if current == target_col + 1 and int(bars_grid[r, target_col]) == color:
            return current
        next_c = current - 1
        if next_c < 0:
            return current
        if int(bars_grid[r, next_c]) != BG:
            return current
        current = next_c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the four baseline bars by height, then slide loose pixels left toward the color-matched sorted slots.

    Input structure:
    - Black background with one full grey baseline row near the bottom.
    - Four width-1 colored pillars stand on the baseline, with fixed slot colors red, blue, green, yellow from left to right.
    - Additional isolated colored pixels appear above the baseline in blank columns between or to the right of the pillars; orange pixels have no matching pillar.

    Rule:
    - Measure the four pillar heights, sort those heights from shortest to tallest, and redraw the pillars into the same four slot columns while preserving each slot's original color.
    - For each non-orange loose pixel, keep its row and slide it left one cell at a time across the sorted layout.
    - Stop when the next step would enter any pillar cell or the baseline, or when the pixel reaches the cell immediately right of its same-color target pillar in the sorted layout.
    - Delete orange loose pixels because they match no pillar.

    Key insight:
    - Slot colors stay fixed but pillar heights are reassigned by rank, so the post-sort geometry changes which rows are blocked before a loose pixel can reach its color target.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    base_r = _find_base_row(g)
    slots = _detect_slots(g, base_r)
    slot_cols = [slot["col"] for slot in slots]
    sorted_heights = sorted(slot["height"] for slot in slots)
    bars_grid = _render_bars(g.shape, base_r, slot_cols, sorted_heights)

    color_to_col = {color: col for color, col in zip(SLOT_COLORS, slot_cols)}
    out = bars_grid.copy()
    for r, c, color in _loose_pixels(g, base_r, set(slot_cols)):
        stop_c = _stop_column(bars_grid, r, c, color, color_to_col.get(color, -99))
        if stop_c is not None:
            out[r, stop_c] = color
    return out


def _search_candidates(input_heights: list[int], output_heights: list[int]) -> list[dict[str, int | str]]:
    candidates: list[dict[str, int | str]] = []
    max_level = max(max(input_heights), max(output_heights))
    for target in range(4):
        for zone in range(target, 4):
            for level in range(1, max_level + 1):
                if input_heights[target] < level:
                    continue
                if any(input_heights[k] >= level for k in range(target + 1, zone + 1)):
                    continue

                stop = None
                for k in range(zone, -1, -1):
                    if output_heights[k] >= level:
                        stop = k
                        break
                if stop is None:
                    continue

                candidates.append(
                    {
                        "target": target,
                        "zone": zone,
                        "level": level,
                        "stop": int(stop),
                        "color": SLOT_COLORS[target],
                        "kind": "reach" if stop == target else "blocked",
                    }
                )
    return candidates


def _sample_geometry() -> tuple[int, int, int, list[int]]:
    height = random.randint(10, 13)
    left_margin = random.randint(1, 3)
    gaps = [random.randint(3, 5) for _ in range(3)]
    right_margin = random.randint(4, 7)
    slot_cols = [left_margin]
    for gap in gaps:
        slot_cols.append(slot_cols[-1] + gap)
    width = slot_cols[-1] + right_margin + 1
    return height, width, height - 1, slot_cols


def _zone_columns(slot_cols: list[int], width: int) -> dict[int, list[int]]:
    return {
        0: list(range(slot_cols[0] + 1, slot_cols[1])),
        1: list(range(slot_cols[1] + 1, slot_cols[2])),
        2: list(range(slot_cols[2] + 1, slot_cols[3])),
        3: list(range(slot_cols[3] + 1, width)),
    }


def _pick_core_candidates(catalog: list[dict[str, int | str]]) -> list[dict[str, int | str]] | None:
    reach = [c for c in catalog if c["kind"] == "reach"]
    blocked = [c for c in catalog if c["kind"] == "blocked"]
    skipped = [c for c in blocked if int(c["stop"]) > int(c["target"]) + 1]
    if not reach or not blocked or not skipped:
        return None

    random.shuffle(reach)
    random.shuffle(blocked)
    random.shuffle(skipped)
    for a in reach:
        for b in blocked:
            if int(a["level"]) == int(b["level"]):
                continue
            for c in skipped:
                levels = {int(a["level"]), int(b["level"]), int(c["level"])}
                if len(levels) != 3:
                    continue
                return [dict(a), dict(b), dict(c)]
    return None


def _choose_extra_candidate(
    catalog: list[dict[str, int | str]],
    used_levels: set[int],
    prefer_target: int | None,
) -> dict[str, int | str] | None:
    candidates = [c for c in catalog if int(c["level"]) not in used_levels]
    if prefer_target is not None:
        preferred = [c for c in candidates if int(c["target"]) == prefer_target]
        if preferred:
            return dict(random.choice(preferred))
    if not candidates:
        return None
    return dict(random.choice(candidates))


def _place_pixels(
    pixel_specs: list[dict[str, int | str]],
    slot_cols: list[int],
    width: int,
    base_r: int,
) -> bool:
    zones = _zone_columns(slot_cols, width)
    used: list[tuple[int, int]] = []

    for spec in pixel_specs:
        r = base_r - int(spec["level"])
        final_col = slot_cols[int(spec["stop"])] + 1 if spec["color"] != UNMATCHED_COLOR else -1
        zone = int(spec["zone"])
        allowed = [c for c in zones[zone] if c > final_col]
        random.shuffle(allowed)
        placed = False
        for c in allowed:
            if any(pr == r and pc == c for pr, pc in used):
                continue
            if any(pc == c and abs(pr - r) == 1 for pr, pc in used):
                continue
            spec["row"] = r
            spec["col"] = int(c)
            used.append((r, int(c)))
            placed = True
            break
        if not placed:
            return False
    return True


def _expected_from_specs(
    shape: tuple[int, int],
    base_r: int,
    slot_cols: list[int],
    output_heights: list[int],
    pixel_specs: list[dict[str, int | str]],
) -> np.ndarray:
    out = _render_bars(shape, base_r, slot_cols, output_heights)
    for spec in pixel_specs:
        if spec["color"] == UNMATCHED_COLOR:
            continue
        r = int(spec["row"])
        c = slot_cols[int(spec["stop"])] + 1
        out[r, c] = int(spec["color"])
    return out


def _validate_structure(grid: np.ndarray) -> dict[str, Any] | None:
    try:
        base_r = _find_base_row(grid)
        if base_r != grid.shape[0] - 1:
            return None
        slots = _detect_slots(grid, base_r)
    except Exception:
        return None

    slot_cols = [slot["col"] for slot in slots]
    heights = [slot["height"] for slot in slots]
    if heights == sorted(heights):
        return None
    if len(set(heights)) != 4:
        return None

    loose = _loose_pixels(grid, base_r, set(slot_cols))
    if not 3 <= len(loose) <= 5:
        return None
    if sum(color == UNMATCHED_COLOR for _r, _c, color in loose) > 1:
        return None
    if any(color not in MATCHABLE_COLORS | {UNMATCHED_COLOR} for _r, _c, color in loose):
        return None
    if any(c <= slot_cols[0] for _r, c, _color in loose):
        return None

    loose_set = {(r, c) for r, c, _color in loose}
    for r, c in loose_set:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) in loose_set:
                return None

    return {"base_r": base_r, "slot_cols": slot_cols, "heights": heights, "loose": loose}


def _validate_semantics(
    grid: np.ndarray,
    slot_cols: list[int],
    input_heights: list[int],
    output_heights: list[int],
    pixel_specs: list[dict[str, int | str]],
) -> bool:
    info = _validate_structure(grid)
    if info is None:
        return False
    if info["slot_cols"] != slot_cols or info["heights"] != input_heights:
        return False

    has_reach = False
    has_blocked = False
    has_skipped = False
    for spec in pixel_specs:
        color = int(spec["color"])
        if color == UNMATCHED_COLOR:
            continue

        target = int(spec["target"])
        zone = int(spec["zone"])
        level = int(spec["level"])
        stop = int(spec["stop"])
        if input_heights[target] < level:
            return False
        if any(input_heights[k] >= level for k in range(target + 1, zone + 1)):
            return False

        inferred = None
        for k in range(zone, -1, -1):
            if output_heights[k] >= level:
                inferred = k
                break
        if inferred != stop:
            return False

        if stop == target:
            has_reach = True
        else:
            has_blocked = True
        if stop > target + 1:
            has_skipped = True

    if not (has_reach and has_blocked and has_skipped):
        return False

    expected = _expected_from_specs(grid.shape, grid.shape[0] - 1, slot_cols, output_heights, pixel_specs)
    return np.array_equal(main(grid), expected)


def _gen_attempt() -> np.ndarray | None:
    height, width, base_r, slot_cols = _sample_geometry()
    all_heights = random.sample(range(2, 8), 4)
    output_heights = sorted(all_heights)

    perm = list(np.random.permutation(4))
    if perm == [0, 1, 2, 3]:
        return None
    input_heights = [output_heights[i] for i in perm]

    catalog = _search_candidates(input_heights, output_heights)
    core = _pick_core_candidates(catalog)
    if core is None:
        return None

    pixel_specs = core
    used_levels = {int(spec["level"]) for spec in pixel_specs}

    extra_matched = random.randint(0, 2)
    if random.random() < 0.6 and extra_matched == 2:
        extra_matched = 1
    repeat_target = int(random.choice(pixel_specs)["target"]) if random.random() < 0.55 else None
    for _ in range(extra_matched):
        extra = _choose_extra_candidate(catalog, used_levels, repeat_target)
        if extra is None:
            break
        pixel_specs.append(extra)
        used_levels.add(int(extra["level"]))

    add_orange = random.random() < 0.6 and len(pixel_specs) <= 4
    if add_orange:
        available_levels = [level for level in range(1, max(output_heights) + 1) if level not in used_levels]
        if not available_levels:
            return None
        pixel_specs.append(
            {
                "color": UNMATCHED_COLOR,
                "level": random.choice(available_levels),
                "zone": random.randint(0, 3),
                "target": -1,
                "stop": -1,
                "kind": "delete",
            }
        )

    if not 3 <= len(pixel_specs) <= 5:
        return None

    if not _place_pixels(pixel_specs, slot_cols, width, base_r):
        return None

    grid = _render_bars((height, width), base_r, slot_cols, input_heights)
    for spec in pixel_specs:
        grid[int(spec["row"]), int(spec["col"])] = int(spec["color"])

    if np.array_equal(grid, _expected_from_specs(grid.shape, base_r, slot_cols, output_heights, pixel_specs)):
        return None
    if not _validate_semantics(grid, slot_cols, input_heights, output_heights, pixel_specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
