from __future__ import annotations

import random
from collections import deque
from itertools import permutations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

RANK_TO_DIR = {
    1: (0, 1),
    2: (1, 0),
    3: (1, 1),
}
SIDES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 400
MAX_LAYOUT_TRIES = 220


def _neighbors4(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4((cr, cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _analyze_bar(coords: list[tuple[int, int]]) -> dict:
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    area = len(coords)

    if len(rows) == 1:
        ordered_cols = sorted(c for _, c in coords)
        assert ordered_cols == list(range(ordered_cols[0], ordered_cols[0] + area)), "horizontal bar must be gapless"
        return {
            "orientation": "H",
            "length": area,
            "start": (rows[0], ordered_cols[0]),
            "cells": tuple(sorted(coords)),
        }

    if len(cols) == 1:
        ordered_rows = sorted(r for r, _ in coords)
        assert ordered_rows == list(range(ordered_rows[0], ordered_rows[0] + area)), "vertical bar must be gapless"
        return {
            "orientation": "V",
            "length": area,
            "start": (ordered_rows[0], cols[0]),
            "cells": tuple(sorted(coords)),
        }

    raise AssertionError("each gray component must be a straight one-cell-thick bar")


def _bar_cells(start_r: int, start_c: int, length: int, orientation: str) -> tuple[tuple[int, int], ...]:
    if orientation == "H":
        return tuple((start_r, start_c + dc) for dc in range(length))
    return tuple((start_r + dr, start_c) for dr in range(length))


def _indicator_from_bar(
    start_r: int,
    start_c: int,
    length: int,
    orientation: str,
    side: str,
    attach: int,
) -> tuple[int, int]:
    if orientation == "H":
        if side == "top":
            return start_r - 1, start_c + attach
        if side == "bottom":
            return start_r + 1, start_c + attach
        if side == "left":
            return start_r, start_c - 1
        if side == "right":
            return start_r, start_c + length
    else:
        if side == "left":
            return start_r + attach, start_c - 1
        if side == "right":
            return start_r + attach, start_c + 1
        if side == "top":
            return start_r - 1, start_c
        if side == "bottom":
            return start_r + length, start_c
    raise AssertionError("invalid side/orientation combination")


def _point_reflect(center: tuple[int, int], cell: tuple[int, int]) -> tuple[int, int]:
    cr, cc = center
    r, c = cell
    return 2 * cr - r, 2 * cc - c


def _all_adjacent_indicators(cells: tuple[tuple[int, int], ...], orientation: str) -> list[tuple[int, int]]:
    ordered = sorted(cells)
    if orientation == "H":
        row = ordered[0][0]
        cols = [c for _, c in ordered]
        out = [(row - 1, c) for c in cols] + [(row + 1, c) for c in cols]
        out.extend([(row, cols[0] - 1), (row, cols[-1] + 1)])
    else:
        col = ordered[0][1]
        rows = [r for r, _ in ordered]
        out = [(r, col - 1) for r in rows] + [(r, col + 1) for r in rows]
        out.extend([(rows[0] - 1, col), (rows[-1] + 1, col)])
    dedup: list[tuple[int, int]] = []
    seen = set()
    for cell in out:
        if cell not in seen:
            seen.add(cell)
            dedup.append(cell)
    return dedup


def _object_from_bar_and_indicator(
    grid_shape: tuple[int, int],
    bar_cells: tuple[tuple[int, int], ...],
    orientation: str,
    length: int,
    rank: int,
    indicator: tuple[int, int],
    side: str | None = None,
    start: tuple[int, int] | None = None,
    attach: int | None = None,
) -> dict | None:
    h, w = grid_shape
    shadow_dir = RANK_TO_DIR[rank]
    shadow_cells = tuple((r + shadow_dir[0], c + shadow_dir[1]) for r, c in bar_cells)
    mirror_bar = tuple(_point_reflect(indicator, cell) for cell in bar_cells)
    mirror_shadow = tuple(_point_reflect(indicator, cell) for cell in shadow_cells)
    all_cells = list(bar_cells) + [indicator] + list(shadow_cells) + list(mirror_bar) + list(mirror_shadow)
    if not all(0 <= r < h and 0 <= c < w for r, c in all_cells):
        return None
    return {
        "cells": frozenset(bar_cells),
        "orientation": orientation,
        "length": length,
        "rank": rank,
        "indicator": indicator,
        "shadow_dir": shadow_dir,
        "shadow_cells": frozenset(shadow_cells),
        "mirror_bar": frozenset(mirror_bar),
        "mirror_shadow": frozenset(mirror_shadow),
        "input_cells": frozenset(set(bar_cells) | {indicator}),
        "side": side,
        "start": start,
        "attach": attach,
    }


def _render_input(grid_shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    for obj in objects:
        for r, c in obj["cells"]:
            out[r, c] = GREY
        ir, ic = obj["indicator"]
        out[ir, ic] = RED
    return out


def _render_output(grid_shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    out = _render_input(grid_shape, objects)
    indicators = {obj["indicator"] for obj in objects}

    for obj in objects:
        for r, c in obj["shadow_cells"]:
            if (r, c) in indicators or out[r, c] == GREY:
                continue
            out[r, c] = BLUE

    for obj in sorted(objects, key=lambda item: item["rank"]):
        for r, c in obj["mirror_shadow"]:
            if (r, c) in indicators:
                continue
            out[r, c] = YELLOW
        for r, c in obj["mirror_bar"]:
            if (r, c) in indicators:
                continue
            out[r, c] = GREEN

    for ir, ic in indicators:
        out[ir, ic] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rank the three gray bars by length, add rank-directed blue shadows, then point-mirror each bar-shadow pair across its red indicator.

    Input structure:
    - A black background contains exactly three gray one-cell-thick straight bars and exactly three red indicator pixels.
    - Each gray object is a single horizontal or vertical 4-connected bar, and the three bar lengths are all different.
    - Each red pixel is 4-neighbor-adjacent to exactly one gray bar and serves as that bar's mirror center.

    Rule:
    - Sort the three gray bars by length from longest to shortest.
    - Give rank 1 a blue shadow shifted one cell right, rank 2 a blue shadow shifted one cell down, and rank 3 a blue shadow shifted one cell down-right; gray bar pixels stay gray when a shadow overlaps them.
    - Point-reflect each bar together with its shadow through its attached red indicator pixel; recolor mirrored bar pixels green and mirrored shadow pixels yellow.
    - Mirrored pixels overwrite earlier gray/blue pixels, but the red indicator pixels remain red.

    Key insight:
    - Length ranking does not choose colors directly; it chooses the shadow direction, while the red indicator independently fixes the 180-degree mirror center that relocates the whole annotated bar.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = set(int(v) for v in np.unique(g))
    assert colors <= {BLACK, RED, GREY}, "input should only contain black, red, and gray"

    bar_components = _components_of_color(g, GREY)
    indicator_components = _components_of_color(g, RED)
    assert len(bar_components) == 3, "expected exactly three gray bars"
    assert len(indicator_components) == 3, "expected exactly three red indicators"
    assert all(len(comp) == 1 for comp in indicator_components), "each indicator must be a singleton"

    analyzed = []
    indicator_to_bar: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(bar_components):
        info = _analyze_bar(comp)
        analyzed.append(info)
        bar_cells = set(info["cells"])
        for red_comp in indicator_components:
            indicator = red_comp[0]
            if any(nbr in bar_cells for nbr in _neighbors4(indicator)):
                assert indicator not in indicator_to_bar, "indicator touches multiple bars"
                indicator_to_bar[indicator] = idx

    assert len(indicator_to_bar) == 3, "each indicator must touch exactly one bar"
    bar_to_indicator: dict[int, tuple[int, int]] = {}
    for indicator, idx in indicator_to_bar.items():
        assert idx not in bar_to_indicator, "bar must have exactly one indicator"
        bar_to_indicator[idx] = indicator
    assert len(bar_to_indicator) == 3, "each bar must have exactly one indicator"

    lengths = [info["length"] for info in analyzed]
    assert len(set(lengths)) == 3, "bar lengths must be distinct"

    objects = []
    sorted_indices = sorted(range(3), key=lambda i: analyzed[i]["length"], reverse=True)
    rank_by_index = {idx: rank for rank, idx in enumerate(sorted_indices, start=1)}
    for idx, info in enumerate(analyzed):
        obj = _object_from_bar_and_indicator(
            g.shape,
            info["cells"],
            info["orientation"],
            info["length"],
            rank_by_index[idx],
            bar_to_indicator[idx],
        )
        assert obj is not None, "all transformed pixels must stay in bounds"
        objects.append(obj)

    return _render_output(g.shape, objects)


def _enumerate_candidates(
    grid_shape: tuple[int, int],
    length: int,
    orientation: str,
    side: str,
    rank: int,
) -> list[dict]:
    h, w = grid_shape
    candidates: list[dict] = []

    if orientation == "H":
        row_range = range(2, h - 2)
        col_range = range(2, w - length - 1)
        attach_range = range(length) if side in ("top", "bottom") else range(1)
    else:
        row_range = range(2, h - length - 1)
        col_range = range(2, w - 2)
        attach_range = range(length) if side in ("left", "right") else range(1)

    for r in row_range:
        for c in col_range:
            bar_cells = _bar_cells(r, c, length, orientation)
            for attach in attach_range:
                indicator = _indicator_from_bar(r, c, length, orientation, side, attach)
                obj = _object_from_bar_and_indicator(
                    grid_shape,
                    bar_cells,
                    orientation,
                    length,
                    rank,
                    indicator,
                    side=side,
                    start=(r, c),
                    attach=attach,
                )
                if obj is not None:
                    candidates.append(obj)
    random.shuffle(candidates)
    return candidates


def _touches4(set_a: frozenset[tuple[int, int]], set_b: frozenset[tuple[int, int]]) -> bool:
    if len(set_a) > len(set_b):
        set_a, set_b = set_b, set_a
    for cell in set_a:
        if any(nbr in set_b for nbr in _neighbors4(cell)):
            return True
    return False


def _input_compatible(candidate: dict, chosen: list[dict]) -> bool:
    for other in chosen:
        if candidate["input_cells"] & other["input_cells"]:
            return False
        if _touches4(candidate["input_cells"], other["input_cells"]):
            return False
    return True


def _cross_overlap_cells(objects: list[dict]) -> set[tuple[int, int]]:
    base_union = set()
    for obj in objects:
        base_union |= set(obj["cells"])
        base_union |= set(obj["shadow_cells"])
    overlap = set()
    for obj in objects:
        overlap |= (set(obj["mirror_bar"]) | set(obj["mirror_shadow"])) & base_union
    return overlap


def _mirror_pairwise_clean(objects: list[dict]) -> bool:
    mirror_sets = [set(obj["mirror_bar"]) | set(obj["mirror_shadow"]) for obj in objects]
    for i in range(len(mirror_sets)):
        for j in range(i + 1, len(mirror_sets)):
            if mirror_sets[i] & mirror_sets[j]:
                return False
    return True


def _indicators_clear(objects: list[dict]) -> bool:
    indicators = {obj["indicator"] for obj in objects}
    for obj in objects:
        if indicators & set(obj["shadow_cells"]):
            return False
        if indicators & set(obj["mirror_bar"]):
            return False
        if indicators & set(obj["mirror_shadow"]):
            return False
    return True


def _direction_uniqueness(grid_shape: tuple[int, int], objects: list[dict], expected: np.ndarray) -> bool:
    h, w = grid_shape
    ordered = sorted(objects, key=lambda item: item["rank"])
    true_dirs = tuple(obj["shadow_dir"] for obj in ordered)
    for perm in permutations((RANK_TO_DIR[1], RANK_TO_DIR[2], RANK_TO_DIR[3])):
        if perm == true_dirs:
            continue
        alt_objects = []
        for obj, alt_dir in zip(ordered, perm):
            alt = dict(obj)
            alt["shadow_dir"] = alt_dir
            alt["shadow_cells"] = frozenset((r + alt_dir[0], c + alt_dir[1]) for r, c in obj["cells"])
            alt["mirror_shadow"] = frozenset(_point_reflect(obj["indicator"], cell) for cell in alt["shadow_cells"])
            all_cells = set(alt["cells"]) | set(alt["shadow_cells"]) | set(alt["mirror_bar"]) | set(alt["mirror_shadow"]) | {alt["indicator"]}
            if not all(0 <= r < h and 0 <= c < w for r, c in all_cells):
                alt_objects = []
                break
            alt_objects.append(alt)
        if not alt_objects:
            continue
        if np.array_equal(_render_output(grid_shape, alt_objects), expected):
            return False
    return True


def _indicator_uniqueness(grid_shape: tuple[int, int], objects: list[dict], expected: np.ndarray) -> bool:
    for idx, obj in enumerate(objects):
        candidates = _all_adjacent_indicators(tuple(sorted(obj["cells"])), obj["orientation"])
        for alt_indicator in candidates:
            if alt_indicator == obj["indicator"]:
                continue
            alt_obj = _object_from_bar_and_indicator(
                grid_shape,
                tuple(sorted(obj["cells"])),
                obj["orientation"],
                obj["length"],
                obj["rank"],
                alt_indicator,
                side=obj["side"],
                start=obj["start"],
                attach=obj["attach"],
            )
            if alt_obj is None:
                continue
            alt_objects = list(objects)
            alt_objects[idx] = alt_obj
            if np.array_equal(_render_output(grid_shape, alt_objects), expected):
                return False
    return True


def _semantic_checks(grid_shape: tuple[int, int], objects: list[dict], want_overlap: bool) -> bool:
    if not _mirror_pairwise_clean(objects):
        return False
    if not _indicators_clear(objects):
        return False

    overlap_cells = _cross_overlap_cells(objects)
    if want_overlap:
        if not (1 <= len(overlap_cells) <= 3):
            return False
    elif overlap_cells:
        return False

    expected = _render_output(grid_shape, objects)
    if np.count_nonzero(expected == BLUE) < 3 or np.count_nonzero(expected == YELLOW) < 3:
        return False
    colors = set(int(v) for v in np.unique(expected))
    if not {RED, GREY, BLUE, GREEN, YELLOW} <= colors:
        return False
    if not _direction_uniqueness(grid_shape, objects, expected):
        return False
    if not _indicator_uniqueness(grid_shape, objects, expected):
        return False

    inp = _render_input(grid_shape, objects)
    try:
        solved = main(inp)
    except AssertionError:
        return False
    return np.array_equal(solved, expected)


def _sample_specs() -> tuple[bool, tuple[int, int], list[tuple[int, int, str, str]]]:
    want_overlap = random.random() < 0.35
    if want_overlap:
        h = random.randint(13, 16)
        w = random.randint(15, 18)
    else:
        h = random.randint(14, 18)
        w = random.randint(16, 20)

    lengths = random.sample(range(3, 8), 3)
    count_h = random.choice((1, 2))
    orientations = ["H"] * count_h + ["V"] * (3 - count_h)
    random.shuffle(orientations)
    sides = random.sample(SIDES, 3)

    sorted_indices = sorted(range(3), key=lambda i: lengths[i], reverse=True)
    ranks = [0, 0, 0]
    for rank, idx in enumerate(sorted_indices, start=1):
        ranks[idx] = rank

    specs = [(lengths[i], ranks[i], orientations[i], sides[i]) for i in range(3)]
    return want_overlap, (h, w), specs


def _try_layout(grid_shape: tuple[int, int], specs: list[tuple[int, int, str, str]], want_overlap: bool) -> list[dict] | None:
    candidate_lists = [
        _enumerate_candidates(grid_shape, length, orientation, side, rank)
        for length, rank, orientation, side in specs
    ]
    if any(not candidates for candidates in candidate_lists):
        return None

    order = sorted(range(3), key=lambda idx: len(candidate_lists[idx]))
    for _ in range(MAX_LAYOUT_TRIES):
        chosen_by_index: dict[int, dict] = {}
        chosen_list: list[dict] = []
        failed = False
        for idx in order:
            valid = [cand for cand in candidate_lists[idx] if _input_compatible(cand, chosen_list)]
            if not valid:
                failed = True
                break
            pick = random.choice(valid)
            chosen_by_index[idx] = pick
            chosen_list.append(pick)
        if failed:
            continue
        objects = [chosen_by_index[i] for i in range(3)]
        if _semantic_checks(grid_shape, objects, want_overlap):
            return objects
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        want_overlap, grid_shape, specs = _sample_specs()
        objects = _try_layout(grid_shape, specs, want_overlap)
        if objects is not None:
            return _render_input(grid_shape, objects)
        if want_overlap:
            objects = _try_layout(grid_shape, specs, False)
            if objects is not None:
                return _render_input(grid_shape, objects)
    raise RuntimeError("failed to generate a valid input")
