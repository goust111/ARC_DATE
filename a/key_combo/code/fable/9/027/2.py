from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import find_components, get_bg


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in coords}


def _bbox_of_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _translate_cells(
    cells: set[tuple[int, int]], dr: int, dc: int
) -> set[tuple[int, int]]:
    return {(r + int(dr), c + int(dc)) for r, c in cells}


def _touches_edge(cells: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in cells)


def _solid_completion_offsets(
    grid: np.ndarray,
    source: set[tuple[int, int]],
    anchor: set[tuple[int, int]],
    bg: int,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    offsets: list[tuple[int, int]] = []

    for dr in range(1 - h, h):
        for dc in range(1 - w, w):
            moved = _translate_cells(source, dr, dc)
            if any(r < 0 or r >= h or c < 0 or c >= w for r, c in moved):
                continue
            if moved & anchor:
                continue
            if any(int(grid[r, c]) != int(bg) and (r, c) not in source for r, c in moved):
                continue

            union = moved | anchor
            r0, c0, r1, c1 = _bbox_of_cells(union)
            rect_h = r1 - r0 + 1
            rect_w = c1 - c0 + 1
            if rect_h < 2 or rect_w < 2:
                continue
            if len(union) == rect_h * rect_w:
                offsets.append((int(dr), int(dc)))

    return offsets


def _find_crack_pairs(grid: np.ndarray, bg: int) -> dict[int, dict]:
    components = find_components(grid, bg=bg, connectivity=4, monochromatic=True)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in components:
        by_color[int(comp["color"])].append(comp)

    found: dict[int, dict] = {}
    for color, comps in by_color.items():
        edge_parts = [
            comp
            for comp in comps
            if int(comp["area"]) >= 2 and _touches_edge(_coord_set(comp["coords"]), grid.shape)
        ]
        inner_parts = [
            comp
            for comp in comps
            if int(comp["area"]) >= 2 and not _touches_edge(_coord_set(comp["coords"]), grid.shape)
        ]

        candidates: list[dict] = []
        for src in edge_parts:
            source = _coord_set(src["coords"])
            for anc in inner_parts:
                anchor = _coord_set(anc["coords"])
                offsets = _solid_completion_offsets(grid, source, anchor, bg)
                if len(offsets) == 1:
                    candidates.append(
                        {
                            "color": int(color),
                            "source": source,
                            "anchor": anchor,
                            "vector": offsets[0],
                        }
                    )

        if len(candidates) == 1:
            found[int(color)] = candidates[0]

    assert len(found) == 2, "expected exactly two crack colors"
    return found


def _find_passengers(
    grid: np.ndarray, bg: int, crack_colors: set[int]
) -> list[dict]:
    passengers: list[dict] = []
    components = find_components(grid, bg=bg, connectivity=4, monochromatic=False)
    for comp in components:
        coords = comp["coords"]
        values = grid[coords[:, 0], coords[:, 1]].astype(int)
        colors, counts = np.unique(values, return_counts=True)
        if len(colors) != 2:
            continue

        marker_candidates = [
            int(color)
            for color, count in zip(colors, counts)
            if int(color) in crack_colors and int(count) == 1
        ]
        if len(marker_candidates) != 1:
            continue

        marker_color = marker_candidates[0]
        body_colors = [int(color) for color in colors if int(color) != marker_color]
        if len(body_colors) != 1:
            continue
        body_color = body_colors[0]
        if body_color in crack_colors:
            continue
        if int(counts[np.where(colors == body_color)[0][0]]) < 3:
            continue

        marker_coords = coords[values == marker_color]
        marker_r, marker_c = map(int, marker_coords[0])
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if not (r0 < marker_r < r1 and c0 < marker_c < c1):
            continue

        passengers.append(
            {
                "coords": coords.copy(),
                "values": values.copy(),
                "bbox": (r0, c0, r1, c1),
                "marker_color": int(marker_color),
            }
        )

    return passengers


def _apply_completions(grid: np.ndarray, cracks: dict[int, dict], bg: int) -> np.ndarray:
    out = grid.copy()
    for info in cracks.values():
        source = info["source"]
        color = int(info["color"])
        dr, dc = map(int, info["vector"])
        for r, c in source:
            out[r, c] = int(bg)
        for r, c in source:
            rr, cc = r + dr, c + dc
            assert int(out[rr, cc]) == int(bg)
            out[rr, cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each split rectangle to measure a hidden translation vector, then dispatch that vector to keyed passenger objects.

    Input structure:
    - The dominant color is the background.
    - Exactly two crack colors each appear as one edge-touching fragment plus one interior anchor fragment; translating the edge fragment by a unique offset merges it with the anchor into a solid axis-aligned rectangle.
    - Several passenger components use their own body color and contain exactly one embedded singleton marker whose color is one of the crack colors.

    Rule:
    - For each crack color, find the unique translation that moves the edge fragment onto background cells so its union with the same-colored anchor becomes a filled rectangle; erase the fragment at its old location and paint it at the completed location.
    - Move every passenger component rigidly, including its embedded marker, by the translation vector belonging to that marker color, and clear the passenger's original cells to background.
    - Leave all other pixels unchanged.

    Key insight:
    - The rectangle completions are measuring devices rather than final goals: each completed crack color reveals a reusable vector, and the marker inside each passenger is the key that selects which hidden vector to apply.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    cracks = _find_crack_pairs(g, bg)
    out = _apply_completions(g, cracks, bg)

    passengers = _find_passengers(g, bg, set(cracks))
    for passenger in passengers:
        dr, dc = cracks[int(passenger["marker_color"])]["vector"]
        coords = passenger["coords"]
        values = passenger["values"]
        out[coords[:, 0], coords[:, 1]] = int(bg)
        for (r, c), value in zip(coords, values):
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]
            assert int(out[rr, cc]) == int(bg)
            out[rr, cc] = int(value)

    return out


GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (18, 26)
RECT_H_RANGE = (3, 6)
RECT_W_RANGE = (4, 7)
PASSENGER_COUNT_RANGE = (2, 4)
PASSENGER_H_RANGE = (3, 5)
PASSENGER_W_RANGE = (3, 5)
BYSTANDER_COUNT_RANGE = (0, 2)
RECT_MARGIN = 2
OBJECT_GAP = 1
MAX_GENERATION_ATTEMPTS = 500
ALL_COLORS = tuple(range(10))


def _footprint_cells(top: int, left: int, height: int, width: int) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(int(top), int(top) + int(height))
        for c in range(int(left), int(left) + int(width))
    }


def _entries_from_patch(
    patch: np.ndarray, top: int, left: int, bg: int
) -> list[tuple[int, int, int]]:
    entries: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(patch != bg):
        entries.append((int(top) + int(r), int(left) + int(c), int(patch[r, c])))
    return entries


def _can_place(cells: set[tuple[int, int]], forbidden: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w and (r, c) not in forbidden for r, c in cells)


def _reserve(
    forbidden: set[tuple[int, int]],
    cells: set[tuple[int, int]],
    shape: tuple[int, int],
    gap: int = OBJECT_GAP,
) -> None:
    h, w = shape
    for r, c in cells:
        for rr in range(r - gap, r + gap + 1):
            for cc in range(c - gap, c + gap + 1):
                if 0 <= rr < h and 0 <= cc < w:
                    forbidden.add((rr, cc))


def _paint_entries(grid: np.ndarray, entries: list[tuple[int, int, int]]) -> None:
    for r, c, color in entries:
        grid[int(r), int(c)] = int(color)


def _gen_split_patches(rect_h: int, rect_w: int, color: int, bg: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(80):
        rot = random.randrange(4)
        base_h, base_w = (rect_h, rect_w) if rot % 2 == 0 else (rect_w, rect_h)
        widths = [random.randint(1, base_w - 1) for _ in range(base_h)]
        if len(set(widths)) == 1 and random.random() < 0.75:
            continue

        source_area = sum(base_w - width for width in widths)
        anchor_area = sum(widths)
        if min(source_area, anchor_area) < 4:
            continue

        source = np.full((base_h, base_w), bg, dtype=int)
        anchor = np.full((base_h, base_w), bg, dtype=int)
        for r, width in enumerate(widths):
            anchor[r, base_w - width : base_w] = int(color)
            source[r, : base_w - width] = int(color)

        source = np.rot90(source, k=rot)
        anchor = np.rot90(anchor, k=rot)
        return anchor, source
    return None


def _choose_rect_position(
    shape: tuple[int, int],
    rect_h: int,
    rect_w: int,
    forbidden_in: set[tuple[int, int]],
    forbidden_out: set[tuple[int, int]],
) -> tuple[int, int] | None:
    h, w = shape
    rows = list(range(RECT_MARGIN, h - rect_h - RECT_MARGIN + 1))
    cols = list(range(RECT_MARGIN, w - rect_w - RECT_MARGIN + 1))
    if not rows or not cols:
        return None
    random.shuffle(rows)
    random.shuffle(cols)
    for top in rows:
        for left in cols:
            footprint = _footprint_cells(top, left, rect_h, rect_w)
            if _can_place(footprint, forbidden_in, shape) and _can_place(footprint, forbidden_out, shape):
                return int(top), int(left)
    return None


def _choose_source_placement(
    final_cells: set[tuple[int, int]],
    shape: tuple[int, int],
    forbidden: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], tuple[int, int]] | None:
    h, w = shape
    r0, c0, r1, c1 = _bbox_of_cells(final_cells)
    edges = ["top", "bottom", "left", "right"]
    random.shuffle(edges)

    for edge in edges:
        shifts: list[tuple[int, int]] = []
        if edge == "top":
            dr = -r0
            for dc in range(-c0, w - c1):
                shifts.append((dr, dc))
        elif edge == "bottom":
            dr = (h - 1) - r1
            for dc in range(-c0, w - c1):
                shifts.append((dr, dc))
        elif edge == "left":
            dc = -c0
            for dr in range(-r0, h - r1):
                shifts.append((dr, dc))
        else:
            dc = (w - 1) - c1
            for dr in range(-r0, h - r1):
                shifts.append((dr, dc))

        random.shuffle(shifts)
        for dr, dc in shifts:
            if abs(dr) + abs(dc) < 3:
                continue
            moved = _translate_cells(final_cells, dr, dc)
            mr0, mc0, mr1, mc1 = _bbox_of_cells(moved)
            footprint = _footprint_cells(mr0, mc0, mr1 - mr0 + 1, mc1 - mc0 + 1)
            if _can_place(footprint, forbidden, shape):
                return moved, (-int(dr), -int(dc))
    return None


def _gen_passenger_patch(body_color: int, marker_color: int) -> np.ndarray:
    height = random.randint(*PASSENGER_H_RANGE)
    width = random.randint(*PASSENGER_W_RANGE)
    patch = np.full((height, width), body_color, dtype=int)
    marker_r = random.randint(1, height - 2)
    marker_c = random.randint(1, width - 2)
    patch[marker_r, marker_c] = int(marker_color)
    return patch


def _choose_patch_position(
    shape: tuple[int, int],
    patch: np.ndarray,
    vector: tuple[int, int],
    forbidden_in: set[tuple[int, int]],
    forbidden_out: set[tuple[int, int]],
) -> tuple[int, int] | None:
    h, w = shape
    ph, pw = patch.shape
    dr, dc = map(int, vector)
    candidates: list[tuple[int, int]] = []

    for top in range(0, h - ph + 1):
        dest_top = top + dr
        if dest_top < 0 or dest_top + ph > h:
            continue
        for left in range(0, w - pw + 1):
            dest_left = left + dc
            if dest_left < 0 or dest_left + pw > w:
                continue
            src_foot = _footprint_cells(top, left, ph, pw)
            dst_foot = _footprint_cells(dest_top, dest_left, ph, pw)
            if _can_place(src_foot, forbidden_in, shape) and _can_place(dst_foot, forbidden_out, shape):
                candidates.append((int(top), int(left)))

    if not candidates:
        return None
    return random.choice(candidates)


def _gen_bystander_patch(color: int, bg: int) -> np.ndarray:
    kind = random.choice(["l", "step", "bar"])
    if kind == "l":
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        patch = np.full((h, w), bg, dtype=int)
        patch[:, 0] = color
        patch[-1, :] = color
        return patch
    if kind == "step":
        patch = np.full((3, 3), bg, dtype=int)
        patch[0, :2] = color
        patch[1, 1:] = color
        patch[2, 2] = color
        return patch
    h = random.randint(2, 4)
    w = random.randint(2, 4)
    patch = np.full((h, w), color, dtype=int)
    patch[0, 0] = bg
    return patch


def _expected_vectors(cracks: list[dict]) -> dict[int, tuple[int, int]]:
    return {int(item["color"]): tuple(map(int, item["vector"])) for item in cracks}


def _make_wrong_uniform_output(
    inp: np.ndarray,
    cracks: dict[int, dict],
    passengers: list[dict],
    bg: int,
    chosen_color: int,
) -> np.ndarray | None:
    out = _apply_completions(inp, cracks, bg)
    dr, dc = cracks[int(chosen_color)]["vector"]
    for passenger in passengers:
        coords = passenger["coords"]
        values = passenger["values"]
        out[coords[:, 0], coords[:, 1]] = int(bg)
        for (r, c), value in zip(coords, values):
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if not (0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]):
                return None
            out[rr, cc] = int(value)
    return out


def _validate_semantics(
    inp: np.ndarray,
    out: np.ndarray,
    expected_vectors: dict[int, tuple[int, int]],
) -> bool:
    try:
        produced = main(inp)
    except Exception:
        return False
    if not np.array_equal(produced, out):
        return False

    bg = int(get_bg(inp))
    cracks = _find_crack_pairs(inp, bg)
    actual_vectors = {color: tuple(map(int, info["vector"])) for color, info in cracks.items()}
    if actual_vectors != expected_vectors:
        return False
    if len(set(actual_vectors.values())) != 2:
        return False

    passengers = _find_passengers(inp, bg, set(cracks))
    if len(passengers) < 2:
        return False
    used = {int(passenger["marker_color"]) for passenger in passengers}
    if used != set(cracks):
        return False

    if np.array_equal(_apply_completions(inp, cracks, bg), out):
        return False
    for color in cracks:
        wrong = _make_wrong_uniform_output(inp, cracks, passengers, bg, color)
        if wrong is not None and np.array_equal(wrong, out):
            return False

    if np.count_nonzero(inp != bg) >= inp.size // 2:
        return False
    if np.array_equal(inp, out):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    shape = (h, w)

    bg = random.choice(ALL_COLORS)
    palette = [color for color in ALL_COLORS if color != bg]
    random.shuffle(palette)

    passenger_count = random.randint(*PASSENGER_COUNT_RANGE)
    bystander_count = random.randint(*BYSTANDER_COUNT_RANGE)
    needed = 2 + passenger_count + bystander_count
    if len(palette) < needed:
        return None

    crack_colors = palette[:2]
    passenger_body_colors = palette[2 : 2 + passenger_count]
    bystander_colors = palette[2 + passenger_count : needed]

    inp = np.full(shape, bg, dtype=int)
    out = np.full(shape, bg, dtype=int)
    forbidden_in: set[tuple[int, int]] = set()
    forbidden_out: set[tuple[int, int]] = set()
    cracks: list[dict] = []

    for crack_color in crack_colors:
        rect_h = random.randint(*RECT_H_RANGE)
        rect_w = random.randint(*RECT_W_RANGE)
        split = _gen_split_patches(rect_h, rect_w, crack_color, bg)
        if split is None:
            return None
        anchor_patch, source_patch = split

        rect_pos = _choose_rect_position(shape, rect_h, rect_w, forbidden_in, forbidden_out)
        if rect_pos is None:
            return None
        top, left = rect_pos
        rect_foot = _footprint_cells(top, left, rect_h, rect_w)
        _reserve(forbidden_in, rect_foot, shape)
        _reserve(forbidden_out, rect_foot, shape)

        _paint_entries(out, [(r, c, crack_color) for r, c in rect_foot])
        _paint_entries(inp, _entries_from_patch(anchor_patch, top, left, bg))

        source_final = {
            (top + int(r), left + int(c))
            for r, c in np.argwhere(source_patch == crack_color)
        }
        placed = _choose_source_placement(source_final, shape, forbidden_in)
        if placed is None:
            return None
        source_input, vector = placed
        _reserve(forbidden_in, source_input, shape)
        _paint_entries(inp, [(r, c, crack_color) for r, c in source_input])

        cracks.append(
            {
                "color": int(crack_color),
                "vector": tuple(map(int, vector)),
            }
        )

    expected_vectors = _expected_vectors(cracks)
    if len(set(expected_vectors.values())) != 2:
        return None

    marker_colors = [crack_colors[i % 2] for i in range(passenger_count)]
    random.shuffle(marker_colors)
    if set(marker_colors) != set(crack_colors):
        return None

    passenger_specs = list(zip(passenger_body_colors, marker_colors))
    random.shuffle(passenger_specs)
    for body_color, marker_color in passenger_specs:
        patch = _gen_passenger_patch(body_color, marker_color)
        pos = _choose_patch_position(
            shape,
            patch,
            expected_vectors[int(marker_color)],
            forbidden_in,
            forbidden_out,
        )
        if pos is None:
            return None
        top, left = pos
        ph, pw = patch.shape
        dr, dc = expected_vectors[int(marker_color)]
        dst_top = top + dr
        dst_left = left + dc
        src_foot = _footprint_cells(top, left, ph, pw)
        dst_foot = _footprint_cells(dst_top, dst_left, ph, pw)
        _reserve(forbidden_in, src_foot, shape)
        _reserve(forbidden_out, dst_foot, shape)
        _paint_entries(inp, _entries_from_patch(patch, top, left, bg))
        _paint_entries(out, _entries_from_patch(patch, dst_top, dst_left, bg))

    for color in bystander_colors:
        patch = _gen_bystander_patch(color, bg)
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for top in range(0, h - ph + 1):
            for left in range(0, w - pw + 1):
                foot = _footprint_cells(top, left, ph, pw)
                if _can_place(foot, forbidden_in, shape) and _can_place(foot, forbidden_out, shape):
                    candidates.append((top, left))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        foot = _footprint_cells(top, left, ph, pw)
        _reserve(forbidden_in, foot, shape)
        _reserve(forbidden_out, foot, shape)
        entries = _entries_from_patch(patch, top, left, bg)
        _paint_entries(inp, entries)
        _paint_entries(out, entries)

    if not _validate_semantics(inp, out, expected_vectors):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
