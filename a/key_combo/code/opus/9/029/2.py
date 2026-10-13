from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, colors_present, find_components


Coord = tuple[int, int]

_GEN_RETRIES = 400
_GEN_SIZES = (11, 13, 15)
_GEN_INPUT_COLORS = (Color.BLUE, Color.RED, Color.GREEN)
_GEN_OUTPUT_SEQUENCE = (Color.BLUE, Color.RED, Color.GREEN)

_COUNT_PATTERNS = {
    2: [(6, 2), (5, 2), (7, 3), (6, 4), (5, 3)],
    3: [(7, 5, 2), (6, 4, 1), (7, 4, 2), (6, 3, 2), (5, 3, 1)],
}

_COUNT_TO_STRUCTURES = {
    1: [["center"]],
    2: [["pair_tb"], ["pair_lr"]],
    3: [["pair_tb", "center"], ["pair_lr", "center"]],
    4: [["quartet"], ["pair_tb", "pair_lr"]],
    5: [["quartet", "center"]],
    6: [["quartet", "pair_tb"], ["quartet", "pair_lr"]],
    7: [["quartet", "pair_tb", "center"], ["quartet", "pair_lr", "center"]],
}

_PRIORITY = {"center": 0, "pair_tb": 1, "pair_lr": 1, "quartet": 2}


def _mask(lines: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in lines], dtype=bool)


_FREE_TEMPLATES = [
    _mask(("###",)),
    _mask(("#.", "##")),
    _mask((".#", "##")),
    _mask(("###", ".#.")),
    _mask((".#.", "###")),
    _mask(("##.", ".##")),
    _mask(("#..", "###")),
    _mask(("##", "##")),
    _mask((".#.", "###", ".#.")),
    _mask(("###", "##.")),
]

_VERTICAL_TEMPLATES = [
    _mask(("###",)),
    _mask((".#.", "###")),
    _mask(("###", ".#.")),
    _mask((".#.", "###", ".#.")),
    _mask(("..#..", ".###.")),
    _mask((".###.", "..#..")),
    _mask(("#####",)),
    _mask(("..#..", "#####")),
]

_CENTER_TEMPLATES = [
    _mask(("###",)),
    _mask((".#.", ".#.", ".#.")),
    _mask((".#.", "###", ".#.")),
    _mask(("..#..", "#####", "..#..")),
    _mask((".#.", ".#.", "###", ".#.", ".#.")),
]


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _neighbors4_of(cell: Coord) -> list[Coord]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _mask_coords(mask: np.ndarray) -> list[Coord]:
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _normalize_coords(coords: set[Coord]) -> set[Coord]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return {(r - r0, c - c0) for r, c in coords}


def _rot90_coords(coords: set[Coord]) -> set[Coord]:
    max_r = max(r for r, _ in coords)
    return {(c, max_r - r) for r, c in coords}


def _flip_coords(coords: set[Coord]) -> set[Coord]:
    max_c = max(c for _, c in coords)
    return {(r, max_c - c) for r, c in coords}


def _unique_variants(mask: np.ndarray) -> list[np.ndarray]:
    base = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    seen: set[tuple[Coord, ...]] = set()
    out: list[np.ndarray] = []
    variants: list[set[Coord]] = []
    cur = base
    for _ in range(4):
        variants.append(cur)
        variants.append(_flip_coords(cur))
        cur = _rot90_coords(cur)
    for coords in variants:
        norm = _normalize_coords(coords)
        key = tuple(sorted(norm))
        if key in seen:
            continue
        seen.add(key)
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        arr = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            arr[r, c] = True
        out.append(arr)
    return out


_FREE_VARIANTS = [variant for tpl in _FREE_TEMPLATES for variant in _unique_variants(tpl)]
_VERTICAL_VARIANTS = []
for tpl in _VERTICAL_TEMPLATES:
    _VERTICAL_VARIANTS.append(tpl.copy())
    flipped = np.flipud(tpl)
    if not np.array_equal(flipped, tpl):
        _VERTICAL_VARIANTS.append(flipped.copy())
_CENTER_VARIANTS = []
for tpl in _CENTER_TEMPLATES:
    for variant in _unique_variants(tpl):
        if np.array_equal(variant, np.flipud(variant)) and np.array_equal(variant, np.fliplr(variant)):
            _CENTER_VARIANTS.append(variant)


def _articulation_points(mask: np.ndarray) -> list[Coord]:
    cells = _mask_coords(mask)
    cell_set = set(cells)
    if len(cells) <= 2:
        return []
    points: list[Coord] = []
    for cell in cells:
        remain = cell_set - {cell}
        start = next(iter(remain))
        stack = [start]
        seen = {start}
        while stack:
            cur = stack.pop()
            for nxt in _neighbors4_of(cur):
                if nxt in remain and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        if seen != remain:
            points.append(cell)
    return points


def _orbit(shape: tuple[int, int], r: int, c: int) -> tuple[Coord, ...]:
    h, w = map(int, shape)
    coords = {
        (int(r), int(c)),
        (h - 1 - int(r), int(c)),
        (int(r), w - 1 - int(c)),
        (h - 1 - int(r), w - 1 - int(c)),
    }
    return tuple(sorted(coords))


def _reflection_orbits(shape: tuple[int, int]) -> list[tuple[Coord, ...]]:
    h, w = map(int, shape)
    seen: set[frozenset[Coord]] = set()
    out: list[tuple[Coord, ...]] = []
    for r in range(h):
        for c in range(w):
            orb = frozenset(_orbit(shape, r, c))
            if orb in seen:
                continue
            seen.add(orb)
            out.append(tuple(sorted(orb)))
    return out


def _dominant_orbit_color(g: np.ndarray, orbit: tuple[Coord, ...], removed: int) -> int | None:
    vals = [int(g[r, c]) for r, c in orbit if int(g[r, c]) != int(removed)]
    if not vals:
        return None
    counts = Counter(vals).most_common()
    if len(counts) >= 2 and counts[0][1] == counts[1][1]:
        return None
    return int(counts[0][0])


def _infer_marker_color(g: np.ndarray) -> int:
    candidates: list[int] = []
    orbits = _reflection_orbits(g.shape)
    for color in colors_present(g):
        if not np.any(g == int(color)):
            continue
        ok = True
        for orb in orbits:
            if _dominant_orbit_color(g, orb, int(color)) is None:
                ok = False
                break
        if ok:
            candidates.append(int(color))
    assert len(candidates) == 1, "expected exactly one marker color"
    return candidates[0]


def _complete_grid(g: np.ndarray) -> np.ndarray:
    marker = _infer_marker_color(g)
    out = np.asarray(g, dtype=int).copy()
    for orb in _reflection_orbits(out.shape):
        fill = _dominant_orbit_color(out, orb, marker)
        assert fill is not None
        for r, c in orb:
            out[r, c] = fill
    return out


def _component_counts(g: np.ndarray) -> dict[int, int]:
    cols = colors_present(g, exclude={Color.BLACK, Color.YELLOW})
    comps = find_components(g, bg=Color.BLACK, connectivity=4, monochromatic=True, colors=cols)
    out = {int(color): 0 for color in cols}
    for comp in comps:
        out[int(comp["color"])] += 1
    return out


def _recolor_from_component_counts(g: np.ndarray) -> np.ndarray:
    counts = _component_counts(g)
    assert 2 <= len(counts) <= 3, "expected 2-3 foreground colors"
    values = list(counts.values())
    assert len(set(values)) == len(values), "component counts must be distinct"
    ordered = [color for color, _n in sorted(counts.items(), key=lambda kv: -kv[1])]
    mapping = {src: _GEN_OUTPUT_SEQUENCE[i] for i, src in enumerate(ordered)}
    out = np.zeros_like(g)
    for src, dst in mapping.items():
        out[g == int(src)] = int(dst)
    return out


def _raw_input_ranking_output(g: np.ndarray) -> np.ndarray | None:
    counts = _component_counts(g)
    if len(counts) < 2:
        return None
    if len(set(counts.values())) != len(counts):
        return None
    ordered = [color for color, _n in sorted(counts.items(), key=lambda kv: -kv[1])]
    mapping = {src: _GEN_OUTPUT_SEQUENCE[i] for i, src in enumerate(ordered)}
    out = np.zeros_like(g)
    for src, dst in mapping.items():
        out[g == int(src)] = int(dst)
    return out


def _is_symmetric(g: np.ndarray) -> bool:
    return np.array_equal(g, np.flipud(g)) and np.array_equal(g, np.fliplr(g))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the yellow holes by double-axis reflection, count each restored color's 4-connected components, then recolor by descending component count.

    Input structure:
    - A square black-background grid contains 2-3 foreground colors chosen from blue, red, and green.
    - The underlying foreground pattern is symmetric across the horizontal and vertical center lines, but some foreground pixels have been replaced by yellow marker cells.
    - Each yellow cell belongs to a reflection orbit whose remaining visible cells reveal the original orbit color, and at least one orbit is a bridge whose removal changes a foreground color's component count.

    Rule:
    - Identify the unique marker color such that removing it leaves every four-way reflection orbit with exactly one surviving orbit color.
    - Fill every yellow cell with the majority color from its own reflection orbit to recover the symmetric pattern.
    - On the restored grid, count the 4-connected components of each foreground color, sort the colors from most components to fewest, then recolor them in that order to blue, red, and green; keep black unchanged.

    Key insight:
    - The yellow pixels are missing symmetric cells, and restoring them is mandatory because the component-count ranking that drives the final recoloring is defined only on the completed symmetric pattern.
    """
    g = np.asarray(grid, dtype=int)
    completed = _complete_grid(g)
    return _recolor_from_component_counts(completed)


def _cells_from_mask(mask: np.ndarray, r0: int, c0: int) -> set[Coord]:
    return {(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(mask)}


def _quartet_cells(n: int, mask: np.ndarray, r0: int, c0: int) -> set[Coord]:
    cells = set()
    for r, c in _cells_from_mask(mask, r0, c0):
        cells.update(_orbit((n, n), r, c))
    return cells


def _pair_tb_cells(n: int, mask: np.ndarray, r0: int) -> set[Coord]:
    mid = n // 2
    c0 = mid - mask.shape[1] // 2
    top = _cells_from_mask(mask, r0, c0)
    return top | {(n - 1 - r, c) for r, c in top}


def _pair_lr_cells(n: int, mask: np.ndarray, c0: int) -> set[Coord]:
    mid = n // 2
    r0 = mid - mask.shape[0] // 2
    left = _cells_from_mask(mask, r0, c0)
    return left | {(r, n - 1 - c) for r, c in left}


def _center_cells(n: int, mask: np.ndarray) -> set[Coord]:
    mid = n // 2
    r0 = mid - mask.shape[0] // 2
    c0 = mid - mask.shape[1] // 2
    return _cells_from_mask(mask, r0, c0)


def _can_place(cells: set[Coord], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    if not cells:
        return False
    if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
        return False
    if len(cells) != len(set(cells)):
        return False
    blocked = _dilate8(occupied)
    return all(not blocked[r, c] for r, c in cells)


def _paint_cells(g: np.ndarray, cells: set[Coord], color: int) -> None:
    for r, c in cells:
        g[r, c] = int(color)


def _choose_quartet(n: int, occupied: np.ndarray, bridge_required: bool) -> tuple[set[Coord], tuple[Coord, ...] | None] | None:
    mid = n // 2
    variants = _FREE_VARIANTS[:]
    random.shuffle(variants)
    for mask in variants:
        arts = _articulation_points(mask)
        if bridge_required and not arts:
            continue
        h, w = mask.shape
        rows = list(range(0, mid - h + 1))
        cols = list(range(0, mid - w + 1))
        if not rows or not cols:
            continue
        random.shuffle(rows)
        random.shuffle(cols)
        for r0 in rows:
            for c0 in cols:
                cells = _quartet_cells(n, mask, r0, c0)
                if not _can_place(cells, occupied):
                    continue
                bridge_orbit = None
                if bridge_required:
                    ar, ac = random.choice(arts)
                    bridge_orbit = _orbit((n, n), r0 + ar, c0 + ac)
                return cells, bridge_orbit
    return None


def _choose_pair_tb(n: int, occupied: np.ndarray) -> set[Coord] | None:
    mid = n // 2
    variants = _VERTICAL_VARIANTS[:]
    random.shuffle(variants)
    for mask in variants:
        h, _w = mask.shape
        rows = list(range(0, mid - h + 1))
        random.shuffle(rows)
        for r0 in rows:
            cells = _pair_tb_cells(n, mask, r0)
            if _can_place(cells, occupied):
                return cells
    return None


def _choose_pair_lr(n: int, occupied: np.ndarray) -> set[Coord] | None:
    mid = n // 2
    variants = [tpl.T for tpl in _VERTICAL_VARIANTS]
    random.shuffle(variants)
    for mask in variants:
        _h, w = mask.shape
        cols = list(range(0, mid - w + 1))
        random.shuffle(cols)
        for c0 in cols:
            cells = _pair_lr_cells(n, mask, c0)
            if _can_place(cells, occupied):
                return cells
    return None


def _choose_center(n: int, occupied: np.ndarray) -> set[Coord] | None:
    variants = _CENTER_VARIANTS[:]
    random.shuffle(variants)
    for mask in variants:
        cells = _center_cells(n, mask)
        if _can_place(cells, occupied):
            return cells
    return None


def _structure_items(color_totals: dict[int, int]) -> list[dict]:
    highest_color = max(color_totals, key=color_totals.get)
    items: list[dict] = []
    for color, total in color_totals.items():
        choices = _COUNT_TO_STRUCTURES[int(total)]
        if color == highest_color:
            quartet_choices = [choice for choice in choices if "quartet" in choice]
            parts = random.choice(quartet_choices if quartet_choices else choices)
        else:
            parts = random.choice(choices)
        for kind in parts:
            items.append(
                {
                    "kind": kind,
                    "color": int(color),
                    "bridge_required": color == highest_color and kind == "quartet",
                }
            )
    random.shuffle(items)
    items.sort(key=lambda item: _PRIORITY[item["kind"]])
    return items


def _build_completed_grid() -> tuple[np.ndarray, tuple[Coord, ...]]:
    for _ in range(_GEN_RETRIES):
        n = random.choice(_GEN_SIZES)
        grid = np.zeros((n, n), dtype=int)
        occupied = np.zeros((n, n), dtype=bool)

        n_colors = random.choice((2, 3))
        colors = random.sample(list(_GEN_INPUT_COLORS), n_colors)
        totals_desc = random.choice(_COUNT_PATTERNS[n_colors])
        shuffled_totals = list(totals_desc)
        random.shuffle(shuffled_totals)
        color_totals = {color: total for color, total in zip(colors, shuffled_totals)}

        bridge_orbit: tuple[Coord, ...] | None = None
        ok = True
        for item in _structure_items(color_totals):
            kind = item["kind"]
            if kind == "quartet":
                picked = _choose_quartet(n, occupied, item["bridge_required"])
                if picked is None:
                    ok = False
                    break
                cells, maybe_bridge = picked
                if item["bridge_required"]:
                    bridge_orbit = maybe_bridge
            elif kind == "pair_tb":
                cells = _choose_pair_tb(n, occupied)
                if cells is None:
                    ok = False
                    break
            elif kind == "pair_lr":
                cells = _choose_pair_lr(n, occupied)
                if cells is None:
                    ok = False
                    break
            else:
                cells = _choose_center(n, occupied)
                if cells is None:
                    ok = False
                    break

            _paint_cells(grid, cells, item["color"])
            for r, c in cells:
                occupied[r, c] = True

        if not ok or bridge_orbit is None:
            continue
        if not _is_symmetric(grid):
            continue
        if colors_present(grid, exclude={Color.BLACK}) != sorted(colors):
            continue
        comps = find_components(grid, bg=Color.BLACK, connectivity=4, monochromatic=True, colors=colors)
        if not comps:
            continue
        if any(not (3 <= int(comp["area"]) <= 8) for comp in comps):
            continue
        counts = _component_counts(grid)
        if counts != color_totals:
            continue
        if len(set(counts.values())) != len(counts):
            continue
        out = _recolor_from_component_counts(grid)
        if np.array_equal(out, grid):
            continue
        return grid, bridge_orbit
    raise RuntimeError("failed to build a completed symmetric grid")


def _foreground_orbits(grid: np.ndarray) -> list[tuple[Coord, ...]]:
    out = []
    for orb in _reflection_orbits(grid.shape):
        r, c = orb[0]
        if int(grid[r, c]) != Color.BLACK:
            out.append(orb)
    return out


def _apply_markers(completed: np.ndarray, bridge_orbit: tuple[Coord, ...]) -> np.ndarray:
    inp = completed.copy()
    bridge_pick = random.randint(1, len(bridge_orbit) - 1)
    for r, c in random.sample(list(bridge_orbit), bridge_pick):
        inp[r, c] = Color.YELLOW

    extras = [orb for orb in _foreground_orbits(completed) if set(orb) != set(bridge_orbit) and len(orb) > 1]
    random.shuffle(extras)
    for orb in extras[: random.randint(0, 2)]:
        take = random.randint(1, len(orb) - 1)
        for r, c in random.sample(list(orb), take):
            inp[r, c] = Color.YELLOW
    return inp


def _validate_generated(inp: np.ndarray) -> bool:
    completed = _complete_grid(inp)
    out = main(inp)

    if not _is_symmetric(completed):
        return False
    fg = colors_present(completed, exclude={Color.BLACK})
    if not (2 <= len(fg) <= 3 and set(fg) <= set(_GEN_INPUT_COLORS)):
        return False

    comps = find_components(completed, bg=Color.BLACK, connectivity=4, monochromatic=True, colors=fg)
    if any(not (3 <= int(comp["area"]) <= 8) for comp in comps):
        return False

    counts_after = _component_counts(completed)
    if len(set(counts_after.values())) != len(counts_after):
        return False

    counts_before = _component_counts(inp)
    if not any(counts_before.get(color, 0) != count for color, count in counts_after.items()):
        return False

    naive = _raw_input_ranking_output(inp)
    if naive is None or np.array_equal(naive, out):
        return False

    if np.array_equal(completed, out):
        return False

    if not np.array_equal(out, _recolor_from_component_counts(completed)):
        return False

    marker = _infer_marker_color(inp)
    if marker != Color.YELLOW:
        return False

    for orb in _reflection_orbits(inp.shape):
        if all(int(inp[r, c]) == Color.YELLOW for r, c in orb):
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        completed, bridge_orbit = _build_completed_grid()
        inp = _apply_markers(completed, bridge_orbit)
        if _validate_generated(inp):
            return inp
    raise RuntimeError("failed to generate a valid input")
