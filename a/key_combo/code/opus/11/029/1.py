from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
CUP = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


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
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        _in_bounds((h, w), nr, nc)
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _side_cells(
    bbox: tuple[int, int, int, int], side: str, *, include_corners: bool = False
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        cols = range(c0, c1 + 1) if include_corners else range(c0 + 1, c1)
        return [(r0, c) for c in cols]
    if side == "bottom":
        cols = range(c0, c1 + 1) if include_corners else range(c0 + 1, c1)
        return [(r1, c) for c in cols]
    if side == "left":
        rows = range(r0, r1 + 1) if include_corners else range(r0 + 1, r1)
        return [(r, c0) for r in rows]
    rows = range(r0, r1 + 1) if include_corners else range(r0 + 1, r1)
    return [(r, c1) for r in rows]


def _is_contiguous_gap(cells: list[tuple[int, int]], side: str) -> bool:
    if not cells:
        return False
    if side in ("top", "bottom"):
        idxs = sorted(c for _, c in cells)
    else:
        idxs = sorted(r for r, _ in cells)
    return idxs == list(range(idxs[0], idxs[-1] + 1))


def _detect_opening(
    grid: np.ndarray, bbox: tuple[int, int, int, int]
) -> tuple[str, list[tuple[int, int]]]:
    perimeter = set(_perimeter_cells(bbox))
    candidates: list[tuple[str, list[tuple[int, int]]]] = []
    for side in ("top", "bottom", "left", "right"):
        gap = [cell for cell in _side_cells(bbox, side) if int(grid[cell]) != CUP]
        if not (1 <= len(gap) <= 2):
            continue
        if not _is_contiguous_gap(gap, side):
            continue
        gap_set = set(gap)
        ok = True
        for cell in perimeter:
            if cell in gap_set:
                if int(grid[cell]) == CUP:
                    ok = False
                    break
            elif int(grid[cell]) != CUP:
                ok = False
                break
        if ok:
            candidates.append((side, sorted(gap)))
    assert len(candidates) == 1, "expected exactly one cup opening"
    return candidates[0]


def _unique_frequency_color(interior: np.ndarray) -> tuple[int, int, dict[int, int]]:
    counts = Counter(
        int(v) for v in interior.ravel() if int(v) not in (BG, CUP)
    )
    assert len(counts) >= 2, "cup interior must contain at least two colors"
    freq_hist = Counter(counts.values())
    winners = [color for color, cnt in counts.items() if freq_hist[cnt] == 1]
    assert len(winners) == 1, "cup must have exactly one unique-frequency color"
    color = winners[0]
    return color, counts[color], dict(counts)


def _analyze_cups(grid: np.ndarray) -> list[dict]:
    cups: list[dict] = []
    for comp in _components_of_color(grid, CUP):
        bbox = _bbox(comp)
        r0, c0, r1, c1 = bbox
        assert r1 - r0 + 1 >= 5 and c1 - c0 + 1 >= 5
        side, gap_cells = _detect_opening(grid, bbox)
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        assert interior.size > 0
        assert not np.any(interior == CUP), "cup interior must not contain wall color"
        ray_color, ray_len, color_counts = _unique_frequency_color(interior)
        cups.append(
            {
                "bbox": bbox,
                "opening_side": side,
                "gap_cells": gap_cells,
                "direction": SIDE_TO_DIR[side],
                "ray_color": ray_color,
                "ray_len": ray_len,
                "color_counts": color_counts,
            }
        )
    cups.sort(key=lambda info: info["bbox"])
    assert cups, "expected at least one cup"
    return cups


def _ray_cells(
    shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int], length: int
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r, c = start
    dr, dc = direction
    for _ in range(length):
        if not _in_bounds(shape, r, c):
            break
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count colors inside each grey cup, select the unique-frequency color, and emit an outward ray of that color whose length equals its count.

    Input structure:
    - The grid has a black background and 2-3 separate grey cup outlines.
    - Each cup is a one-pixel-thick rectangular border with one contiguous 1-2 cell opening on exactly one side.
    - Inside each cup are scattered non-black, non-grey pixels from multiple colors.

    Rule:
    - For each grey cup, use its bounding box to identify the rectangular interior and the opened side.
    - Count the non-black, non-grey colors inside that interior and find the one whose frequency is unique among the present colors.
    - Let N be that color's count, then recolor every opening cell and extend outward from it in the opening direction for exactly N cells using that same color.
    - Keep every original wall pixel and every interior pixel unchanged.

    Key insight:
    - The cup opening provides only the ray direction; the interior frequency pattern independently determines both the ray color and its exact length.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for cup in _analyze_cups(g):
        for start in cup["gap_cells"]:
            for rr, cc in _ray_cells(g.shape, start, cup["direction"], cup["ray_len"]):
                out[rr, cc] = cup["ray_color"]
    return out


def _expand_zone(
    shape: tuple[int, int], cells: set[tuple[int, int]], radius: int
) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = r + dr
                cc = c + dc
                if _in_bounds(shape, rr, cc):
                    expanded.add((rr, cc))
    return expanded


def _all_bbox_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _pick_count_pattern(area: int, relation: str) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for unique_count in range(1, min(5, area) + 1):
        for repeated_count in range(1, area + 1):
            if unique_count == repeated_count:
                continue
            if unique_count + 2 * repeated_count > area:
                continue
            if relation == "smallest" and not (unique_count < repeated_count):
                continue
            if relation == "largest" and not (unique_count > repeated_count):
                continue
            candidates.append((unique_count, repeated_count))
    assert candidates, f"no valid count pattern for area={area} relation={relation}"
    return random.choice(candidates)


def _max_length_to_edge(
    shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]
) -> int:
    r, c = start
    dr, dc = direction
    length = 0
    while _in_bounds(shape, r, c):
        length += 1
        r += dr
        c += dc
    return length


def _build_gap_cells(
    bbox: tuple[int, int, int, int], side: str, gap_len: int
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side in ("top", "bottom"):
        options = list(range(c0 + 1, c1))
        start = random.randint(0, len(options) - gap_len)
        cols = options[start : start + gap_len]
        row = r0 if side == "top" else r1
        return [(row, c) for c in cols]
    options = list(range(r0 + 1, r1))
    start = random.randint(0, len(options) - gap_len)
    rows = options[start : start + gap_len]
    col = c0 if side == "left" else c1
    return [(r, col) for r in rows]


def _sample_cup_spec(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    side: str,
    gap_len: int,
    relation: str,
) -> dict | None:
    h, w = shape
    for _ in range(300):
        box_h = random.choice((5, 6))
        box_w = random.choice((5, 6))
        area = (box_h - 2) * (box_w - 2)
        unique_count, repeated_count = _pick_count_pattern(area, relation)
        ray_len = unique_count
        dr, dc = SIDE_TO_DIR[side]

        min_r0, max_r0 = 1, h - box_h - 1
        min_c0, max_c0 = 1, w - box_w - 1
        if side == "top":
            min_r0 = max(min_r0, ray_len - 1)
        elif side == "bottom":
            max_r0 = min(max_r0, h - box_h - ray_len)
        elif side == "left":
            min_c0 = max(min_c0, ray_len - 1)
        else:
            max_c0 = min(max_c0, w - box_w - ray_len)
        if min_r0 > max_r0 or min_c0 > max_c0:
            continue

        r0 = random.randint(min_r0, max_r0)
        c0 = random.randint(min_c0, max_c0)
        bbox = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        gap_cells = _build_gap_cells(bbox, side, gap_len)
        ray_sets = [
            _ray_cells(shape, start, (dr, dc), ray_len)
            for start in gap_cells
        ]
        if any(len(ray) != ray_len for ray in ray_sets):
            continue

        bbox_cells = _all_bbox_cells(bbox)
        ray_cells = {cell for ray in ray_sets for cell in ray}
        footprint = bbox_cells | ray_cells
        reserve = _expand_zone(shape, bbox_cells, 2) | _expand_zone(shape, ray_cells, 1)
        if not reserve.isdisjoint(blocked):
            continue

        colors = random.sample(PALETTE, 3)
        unique_color = random.choice(colors)
        repeated_colors = [color for color in colors if color != unique_color]
        counts = {
            unique_color: unique_count,
            repeated_colors[0]: repeated_count,
            repeated_colors[1]: repeated_count,
        }

        interior_cells = [
            (r, c)
            for r in range(r0 + 1, r0 + box_h - 1)
            for c in range(c0 + 1, c0 + box_w - 1)
        ]
        total_colored = sum(counts.values())
        if total_colored > len(interior_cells):
            continue
        chosen = random.sample(interior_cells, total_colored)
        painted: dict[tuple[int, int], int] = {}
        idx = 0
        for color, cnt in counts.items():
            for cell in chosen[idx : idx + cnt]:
                painted[cell] = color
            idx += cnt

        alt_color = repeated_colors[0]
        alt_len = ray_len - 1 if ray_len > 1 else 2
        max_len = min(_max_length_to_edge(shape, cell, (dr, dc)) for cell in gap_cells)
        if not (1 <= alt_len <= max_len):
            continue

        return {
            "bbox": bbox,
            "opening_side": side,
            "direction": (dr, dc),
            "gap_cells": sorted(gap_cells),
            "gap_len": gap_len,
            "bbox_cells": bbox_cells,
            "ray_cells": ray_cells,
            "reserve": reserve,
            "interior_painted": painted,
            "counts": counts,
            "ray_color": unique_color,
            "ray_len": ray_len,
            "alt_color": alt_color,
            "alt_len": alt_len,
            "relation": relation,
            "footprint": footprint,
        }
    return None


def _render_input(shape: tuple[int, int], cups: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for cup in cups:
        r0, c0, r1, c1 = cup["bbox"]
        for r, c in _perimeter_cells(cup["bbox"]):
            grid[r, c] = CUP
        for r, c in cup["gap_cells"]:
            grid[r, c] = BG
        for (r, c), color in cup["interior_painted"].items():
            grid[r, c] = color
        assert not np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == CUP)
    return grid


def _render_expected(inp: np.ndarray, cups: list[dict]) -> np.ndarray:
    out = inp.copy()
    for cup in cups:
        for start in cup["gap_cells"]:
            for r, c in _ray_cells(inp.shape, start, cup["direction"], cup["ray_len"]):
                out[r, c] = cup["ray_color"]
    return out


def _render_variant(
    inp: np.ndarray,
    cups: list[dict],
    cup_index: int,
    *,
    color: int | None = None,
    length: int | None = None,
    direction: tuple[int, int] | None = None,
) -> np.ndarray:
    out = inp.copy()
    for idx, cup in enumerate(cups):
        use_color = cup["ray_color"] if idx != cup_index or color is None else color
        use_len = cup["ray_len"] if idx != cup_index or length is None else length
        use_dir = cup["direction"] if idx != cup_index or direction is None else direction
        for start in cup["gap_cells"]:
            for r, c in _ray_cells(inp.shape, start, use_dir, use_len):
                out[r, c] = use_color
    return out


def _validate_generated(inp: np.ndarray, cups: list[dict], expected: np.ndarray) -> None:
    detected = _analyze_cups(inp)
    assert len(detected) == len(cups)
    for spec, found in zip(sorted(cups, key=lambda x: x["bbox"]), detected):
        assert found["bbox"] == spec["bbox"]
        assert found["opening_side"] == spec["opening_side"]
        assert found["gap_cells"] == spec["gap_cells"]
        assert found["ray_color"] == spec["ray_color"]
        assert found["ray_len"] == spec["ray_len"]
        assert found["color_counts"] == spec["counts"]
        freq_hist = Counter(spec["counts"].values())
        assert freq_hist[spec["ray_len"]] == 1
        assert spec["relation"] == ("smallest" if spec["ray_len"] < spec["counts"][spec["alt_color"]] else "largest")
        assert spec["gap_len"] in (1, 2)
    assert np.array_equal(main(inp), expected)
    assert not np.array_equal(inp, expected)

    all_ray_cells = [frozenset(cup["ray_cells"]) for cup in cups]
    assert len(set(all_ray_cells)) == len(all_ray_cells)
    for i, cup in enumerate(cups):
        wrong_color = _render_variant(inp, cups, i, color=cup["alt_color"])
        wrong_length = _render_variant(inp, cups, i, length=cup["alt_len"])
        wrong_direction = _render_variant(inp, cups, i, direction=(-cup["direction"][0], -cup["direction"][1]))
        assert not np.array_equal(wrong_color, expected)
        assert not np.array_equal(wrong_length, expected)
        assert not np.array_equal(wrong_direction, expected)


def _attempt_generate() -> np.ndarray | None:
    shape = (random.randint(18, 24), random.randint(18, 24))
    num_cups = 2 if random.random() < 0.65 else 3

    sides = random.sample(("top", "bottom", "left", "right"), num_cups)
    if num_cups == 2 and random.random() < 0.7:
        gap_lens = [1, 2]
        relations = ["smallest", "largest"]
    else:
        gap_lens = [1, 2] + [random.choice((1, 2))] * (num_cups - 2)
        relations = ["smallest", "largest"] + [random.choice(("smallest", "largest"))] * (num_cups - 2)
    random.shuffle(gap_lens)
    random.shuffle(relations)

    blocked: set[tuple[int, int]] = set()
    cups: list[dict] = []
    for side, gap_len, relation in zip(sides, gap_lens, relations):
        cup = _sample_cup_spec(shape, blocked, side, gap_len, relation)
        if cup is None:
            return None
        blocked |= cup["reserve"]
        cups.append(cup)

    inp = _render_input(shape, cups)
    expected = _render_expected(inp, cups)
    try:
        _validate_generated(inp, cups, expected)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(250):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cup-ray ARC input")
