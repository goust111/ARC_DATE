from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import find_components, get_bg


BG = 0
BRACKET = 1
RED = 2
GREEN = 3

GRID_MIN = 12
GRID_MAX = 16
BRACKET_WIDTH_MIN = 4
BRACKET_WIDTH_MAX = 6
GEN_ATTEMPTS = 500


def _component_cells(component: dict) -> set[tuple[int, int]]:
    return set(map(tuple, component["coords"]))


def _is_top_bracket(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    if r0 != 0 or r1 != 1 or c1 - c0 + 1 < BRACKET_WIDTH_MIN:
        return False
    expected = {(0, c) for c in range(c0, c1 + 1)}
    expected.add((1, c0))
    expected.add((1, c1))
    return _component_cells(component) == expected


def _is_bottom_bracket(component: dict, height: int) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    if r0 != height - 2 or r1 != height - 1 or c1 - c0 + 1 < BRACKET_WIDTH_MIN:
        return False
    expected = {(height - 1, c) for c in range(c0, c1 + 1)}
    expected.add((height - 2, c0))
    expected.add((height - 2, c1))
    return _component_cells(component) == expected


def _find_brackets(grid: np.ndarray) -> tuple[dict, dict]:
    height = grid.shape[0]
    components = find_components(
        grid,
        bg=BG,
        colors=[BRACKET],
        connectivity=4,
        monochromatic=True,
    )
    top = [component for component in components if _is_top_bracket(component)]
    bottom = [component for component in components if _is_bottom_bracket(component, height)]
    assert len(top) == 1 and len(bottom) == 1, "expected one top bracket and one bottom bracket"
    return top[0], bottom[0]


def _seed_order(seed: tuple[int, int, int]) -> tuple[int, int, int]:
    r, c, color = seed
    return int(c), int(r), int(color)


def _ray_mask(shape: tuple[int, int], seed: tuple[int, int, int], blockers: np.ndarray) -> np.ndarray:
    height, width = shape
    r, c, color = map(int, seed)
    mask = np.zeros(shape, dtype=bool)

    if color == RED:
        top = r
        while top - 1 >= 0 and not blockers[top - 1, c]:
            top -= 1
        bottom = r
        while bottom + 1 < height and not blockers[bottom + 1, c]:
            bottom += 1
        mask[top : bottom + 1, c] = True
        return mask

    assert color == GREEN, "only red and green seeds are valid"
    left = c
    while left - 1 >= 0 and not blockers[r, left - 1]:
        left -= 1
    right = c
    while right + 1 < width and not blockers[r, right + 1]:
        right += 1
    mask[r, left : right + 1] = True
    return mask


def _sort_channel_fragments(
    fragments: list[dict],
    height: int,
    direction: int,
) -> list[dict]:
    if direction == 1:
        key = lambda fragment: (
            height - 1 - int(fragment["coords"][:, 0].max()),
            -int(fragment["coords"][:, 0].max()),
            int(fragment["coords"][:, 1].min()),
            int(fragment["seed_index"]),
        )
    else:
        key = lambda fragment: (
            int(fragment["coords"][:, 0].min()),
            int(fragment["coords"][:, 1].min()),
            int(fragment["coords"][:, 0].max()),
            int(fragment["seed_index"]),
        )
    return sorted(fragments, key=key)


def _settle_channel(
    shape: tuple[int, int],
    occupied: np.ndarray,
    fragments: list[dict],
    *,
    direction: int,
) -> list[dict]:
    height, _width = shape
    settled: list[dict] = []
    for fragment in _sort_channel_fragments(fragments, height, direction):
        coords = fragment["coords"]
        shift = 0
        while True:
            candidate = coords + np.array([shift + direction, 0])
            rows = candidate[:, 0]
            cols = candidate[:, 1]
            if int(rows.min()) < 0 or int(rows.max()) >= height:
                break
            if np.any(occupied[rows, cols]):
                break
            shift += direction

        final_coords = coords + np.array([shift, 0])
        assert not np.any(occupied[final_coords[:, 0], final_coords[:, 1]])
        occupied[final_coords[:, 0], final_coords[:, 1]] = True
        settled.append(
            {
                **fragment,
                "final_coords": final_coords,
                "shift": int(shift),
                "moved": bool(shift != 0),
            }
        )
    return settled


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    assert g.ndim == 2
    assert max(height, width) <= 30
    assert set(map(int, np.unique(g))) <= {BG, BRACKET, RED, GREEN}

    bg = int(get_bg(g))
    assert bg == BG, "background must be black"
    top_bracket, bottom_bracket = _find_brackets(g)

    blockers = g == BRACKET
    seeds = [(int(r), int(c), int(g[r, c])) for r, c in np.argwhere(np.isin(g, [RED, GREEN]))]
    assert 2 <= len(seeds) <= 4, "expected 2 to 4 seeds"
    ordered_seeds = sorted(seeds, key=_seed_order)

    stripe_entries: list[dict] = []
    raw_overlap_count = 0
    for seed_index, seed in enumerate(ordered_seeds):
        raw_mask = _ray_mask(g.shape, seed, blockers)
        overlap = np.zeros_like(raw_mask, dtype=bool)
        for entry in stripe_entries:
            overlap |= entry["visible_mask"] & raw_mask
        raw_overlap_count += int(np.count_nonzero(overlap))
        for entry in stripe_entries:
            entry["visible_mask"][raw_mask] = False
        stripe_entries.append(
            {
                "seed_index": int(seed_index),
                "seed": seed,
                "color": int(seed[2]),
                "raw_mask": raw_mask,
                "visible_mask": raw_mask.copy(),
            }
        )

    top_c0, top_c1 = int(top_bracket["bbox"][1]), int(top_bracket["bbox"][3])
    bottom_c0, bottom_c1 = int(bottom_bracket["bbox"][1]), int(bottom_bracket["bbox"][3])
    assert top_c1 < bottom_c0 or bottom_c1 < top_c0, "channel spans must not overlap"

    spans = {
        "top": (top_c0, top_c1, 1),
        "bottom": (bottom_c0, bottom_c1, -1),
    }

    ray_render = np.full_like(g, BG)
    ray_render[blockers] = BRACKET
    fragments_by_channel: dict[str, list[dict]] = {"top": [], "bottom": []}
    fixed_pieces: list[tuple[int, np.ndarray]] = []

    for entry in stripe_entries:
        visible_mask = entry["visible_mask"]
        assert np.any(visible_mask), "a seed may not be fully overwritten"
        ray_render[visible_mask] = entry["color"]

        moving_mask = np.zeros_like(visible_mask, dtype=bool)
        for channel, (c0, c1, direction) in spans.items():
            channel_mask = np.zeros_like(visible_mask, dtype=bool)
            channel_mask[:, c0 : c1 + 1] = True
            fragment_mask = visible_mask & channel_mask
            if not np.any(fragment_mask):
                continue
            coords = np.argwhere(fragment_mask)
            fragments_by_channel[channel].append(
                {
                    "channel": channel,
                    "direction": int(direction),
                    "seed_index": int(entry["seed_index"]),
                    "color": int(entry["color"]),
                    "coords": coords,
                }
            )
            moving_mask |= fragment_mask

        fixed_mask = visible_mask & ~moving_mask
        if np.any(fixed_mask):
            fixed_pieces.append((int(entry["color"]), np.argwhere(fixed_mask)))

    occupied = blockers.copy()
    for _color, coords in fixed_pieces:
        assert not np.any(occupied[coords[:, 0], coords[:, 1]])
        occupied[coords[:, 0], coords[:, 1]] = True

    settled_fragments: list[dict] = []
    for channel, (_c0, _c1, direction) in spans.items():
        channel_occupied = occupied.copy()
        settled = _settle_channel(
            g.shape,
            channel_occupied,
            fragments_by_channel[channel],
            direction=direction,
        )
        occupied = channel_occupied
        settled_fragments.extend(settled)

    out = np.full_like(g, BG)
    out[blockers] = BRACKET
    out[g == RED] = BG
    out[g == GREEN] = BG
    for color, coords in fixed_pieces:
        assert not np.any(out[coords[:, 0], coords[:, 1]] == BRACKET)
        out[coords[:, 0], coords[:, 1]] = color
    for fragment in settled_fragments:
        coords = fragment["final_coords"]
        assert not np.any(out[coords[:, 0], coords[:, 1]] == BRACKET)
        assert not np.any(np.isin(out[coords[:, 0], coords[:, 1]], [RED, GREEN]))
        out[coords[:, 0], coords[:, 1]] = fragment["color"]

    details = {
        "seeds": ordered_seeds,
        "raw_overlap_count": int(raw_overlap_count),
        "ray_render": ray_render,
        "fragments_by_channel": fragments_by_channel,
        "settled_fragments": settled_fragments,
        "top_span": (top_c0, top_c1),
        "bottom_span": (bottom_c0, bottom_c1),
        "fixed_pixel_count": int(sum(coords.shape[0] for _color, coords in fixed_pieces)),
    }
    return out, details


def main(grid: np.ndarray) -> np.ndarray:
    """Grow red and green seeds into bracket-blocked rays, then stack each channel fragment under bracket-directed gravity.

    Input structure:
    - The grid is black background with exactly two blue bracket frames: one at the top edge open downward and one at the bottom edge open upward.
    - Two to four seeds appear in the interior: red seeds and green seeds.
    - A red seed lies in a bracket column interval and defines a vertical ray; a green seed defines a horizontal ray that may cross one or both bracket intervals.

    Rule:
    - Sort seeds from left to right, then from top to bottom.
    - Extend each red seed vertically and each green seed horizontally until the next step would hit a blue bracket cell; later rays overwrite earlier ones at crossings.
    - For each visible ray, split off the cells that lie inside the top bracket's column span and drop that fragment downward, and split off the cells that lie inside the bottom bracket's column span and drop that fragment upward.
    - Treat every such fragment as a rigid object, settle fragments in each channel from nearer the gravity target edge to farther, stop on the grid edge or already settled fragments, and leave every ray cell outside both channel spans fixed.

    Key insight:
    - The seed color determines the ray shape, but the two bracket column spans independently carve movable fragments out of those rays and impose opposite vertical gravities on them.
    """
    return _simulate(np.asarray(grid, dtype=int))[0]


def _add_top_bracket(grid: np.ndarray, c0: int, width: int) -> None:
    grid[0, c0 : c0 + width] = BRACKET
    grid[1, c0] = BRACKET
    grid[1, c0 + width - 1] = BRACKET


def _add_bottom_bracket(grid: np.ndarray, c0: int, width: int) -> None:
    h = grid.shape[0]
    grid[h - 1, c0 : c0 + width] = BRACKET
    grid[h - 2, c0] = BRACKET
    grid[h - 2, c0 + width - 1] = BRACKET


def _sample_brackets(width: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    left_w = random.randint(BRACKET_WIDTH_MIN, min(BRACKET_WIDTH_MAX, width - BRACKET_WIDTH_MIN - 2))
    right_w = random.randint(BRACKET_WIDTH_MIN, min(BRACKET_WIDTH_MAX, width - left_w - 2))
    free = width - left_w - right_w
    if free < 2:
        return None
    gap = random.randint(2, free)
    margin_left = random.randint(0, free - gap)
    left_c0 = margin_left
    right_c0 = left_c0 + left_w + gap
    left_span = (left_c0, left_w)
    right_span = (right_c0, right_w)
    if random.choice([False, True]):
        return left_span, right_span
    return right_span, left_span


def _sample_distinct(rows: list[int], count: int, min_gap: int = 1) -> list[int] | None:
    rows = rows.copy()
    random.shuffle(rows)
    chosen: list[int] = []
    for value in rows:
        if all(abs(value - other) >= min_gap for other in chosen):
            chosen.append(value)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _zone_values(width: int, top_span: tuple[int, int], bottom_span: tuple[int, int]) -> list[list[int]]:
    top_c0, top_c1 = top_span
    bottom_c0, bottom_c1 = bottom_span
    left_c1 = min(top_c0, bottom_c0) - 1
    right_c0 = max(top_c1, bottom_c1) + 1
    middle_lo = min(top_c1, bottom_c1) + 1
    middle_hi = max(top_c0, bottom_c0) - 1
    zones = [
        [c for c in range(0, left_c1 + 1)],
        [c for c in range(middle_lo, middle_hi + 1)],
        [c for c in range(right_c0, width)],
    ]
    return [zone for zone in zones if zone]


def _remove_seed(grid: np.ndarray, seed: tuple[int, int, int]) -> np.ndarray:
    out = grid.copy()
    r, c, _color = seed
    out[r, c] = BG
    return out


def _build_candidate() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    bracket_sample = _sample_brackets(width)
    if bracket_sample is None:
        return None
    (top_c0, top_w), (bottom_c0, bottom_w) = bracket_sample
    top_span = (top_c0, top_c0 + top_w - 1)
    bottom_span = (bottom_c0, bottom_c0 + bottom_w - 1)

    red_count, green_count = random.choice([(2, 1), (2, 2), (3, 1)])
    seed_count = red_count + green_count
    rows = _sample_distinct(list(range(2, height - 2)), seed_count, min_gap=1 if green_count == 1 else 2)
    if rows is None:
        return None
    random.shuffle(rows)

    top_choices = list(range(top_span[0], top_span[1] + 1))
    bottom_choices = list(range(bottom_span[0], bottom_span[1] + 1))
    if len(top_choices) < 1 or len(bottom_choices) < 1:
        return None

    red_columns = [random.choice(top_choices), random.choice(bottom_choices)]
    if red_count == 3:
        extra_pool = top_choices + bottom_choices
        random.shuffle(extra_pool)
        for column in extra_pool:
            if red_columns.count(column) == 0:
                red_columns.append(column)
                break
        else:
            return None

    zones = _zone_values(width, top_span, bottom_span)
    if not zones:
        return None

    green_columns: list[int] = []
    available_zones = zones.copy()
    random.shuffle(available_zones)
    for _ in range(green_count):
        if available_zones:
            zone = available_zones.pop(0)
        else:
            zone = [c for c in range(width)]
        random.shuffle(zone)
        for column in zone:
            if column not in green_columns:
                green_columns.append(column)
                break
        else:
            return None

    seeds: list[tuple[int, int, int]] = []
    for idx, column in enumerate(red_columns):
        seeds.append((rows[idx], int(column), RED))
    for idx, column in enumerate(green_columns):
        seeds.append((rows[red_count + idx], int(column), GREEN))

    grid = np.full((height, width), BG, dtype=int)
    _add_top_bracket(grid, top_c0, top_w)
    _add_bottom_bracket(grid, bottom_c0, bottom_w)
    for r, c, color in seeds:
        if grid[r, c] != BG:
            return None
        grid[r, c] = color
    return grid


def _validate_candidate(grid: np.ndarray) -> bool:
    try:
        out, details = _simulate(grid)
    except Exception:
        return False

    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, details["ray_render"]):
        return False
    if details["raw_overlap_count"] == 0:
        return False
    if details["fixed_pixel_count"] == 0:
        return False

    top_fragments = details["fragments_by_channel"]["top"]
    bottom_fragments = details["fragments_by_channel"]["bottom"]
    if len(top_fragments) < 2 or len(bottom_fragments) < 2:
        return False

    moved_by_channel = {"top": False, "bottom": False}
    max_shift = 0
    for fragment in details["settled_fragments"]:
        moved_by_channel[fragment["channel"]] |= fragment["moved"]
        max_shift = max(max_shift, abs(int(fragment["shift"])))
    if not moved_by_channel["top"] or not moved_by_channel["bottom"]:
        return False
    if max_shift < 2:
        return False

    if RED not in set(map(int, np.unique(out))) or GREEN not in set(map(int, np.unique(out))):
        return False

    seed_channel_sets: dict[int, set[str]] = {}
    for fragment in details["settled_fragments"]:
        seed_channel_sets.setdefault(int(fragment["seed_index"]), set()).add(fragment["channel"])
    if not any(channels == {"top", "bottom"} for channels in seed_channel_sets.values()):
        return False

    for seed in details["seeds"]:
        reduced = _remove_seed(grid, seed)
        try:
            alt_out = main(reduced)
        except Exception:
            return False
        if np.array_equal(alt_out, out):
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None and _validate_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid ray-growth bracket-gravity puzzle")
