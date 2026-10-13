from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


def _load_arc_utils():
    candidates = [
        Path("/home/lixinyu/snap/new/solvers"),
        Path(__file__).resolve().parents[6] / "new" / "solvers",
    ]
    for directory in candidates:
        if directory.is_dir():
            path = str(directory)
            if path not in sys.path:
                sys.path.insert(0, path)
            break
    from arc_utils import Color, colors_present, get_bg, in_bounds

    return Color, colors_present, get_bg, in_bounds


Color, colors_present, get_bg, in_bounds = _load_arc_utils()

ORIENTATIONS: dict[str, tuple[int, int]] = {
    "h": (0, 1),
    "v": (1, 0),
    "d1": (1, 1),
    "d2": (1, -1),
}
GEN_PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_GRID_ROWS = (15, 28)
GEN_GRID_COLS = (15, 28)
GEN_OBJECT_COUNT_CHOICES = (2, 3, 4)
GEN_OBJECT_COUNT_WEIGHTS = (0.35, 0.40, 0.25)
GEN_LINE_COUNT_CHOICES = (2, 3, 4)
GEN_LINE_COUNT_WEIGHTS = (0.38, 0.38, 0.24)
GEN_MAX_SIDE_PIXELS = 5
GEN_MIN_TOTAL_PIXELS = 4
GEN_CASE_ATTEMPTS = 400
GEN_OBJECT_ATTEMPTS = 120


def _anchor_patch(anchor: tuple[int, int]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(ar + dr, ac + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _step_for(orientation: str, sign: int) -> tuple[int, int]:
    dr, dc = ORIENTATIONS[orientation]
    if sign < 0:
        dr, dc = -dr, -dc
    return dr, dc


def _advance(anchor: tuple[int, int], orientation: str, signed_t: int) -> tuple[int, int]:
    ar, ac = anchor
    dr, dc = ORIENTATIONS[orientation]
    return ar + dr * signed_t, ac + dc * signed_t


def _side_capacity(shape: tuple[int, int], anchor: tuple[int, int], orientation: str, sign: int) -> int:
    ar, ac = anchor
    dr, dc = _step_for(orientation, sign)
    count = 0
    k = 2
    while in_bounds(shape, ar + dr * k, ac + dc * k):
        count += 1
        k += 1
    return count


def _classify_relative(anchor: tuple[int, int], cell: tuple[int, int]) -> tuple[str, int] | None:
    ar, ac = anchor
    r, c = cell
    if max(abs(r - ar), abs(c - ac)) <= 1:
        return None
    dr = r - ar
    dc = c - ac
    if dr == 0:
        return "h", dc
    if dc == 0:
        return "v", dr
    if dr == dc:
        return "d1", dr
    if dr == -dc:
        return "d2", dr
    return None


def _groups_for_anchor(
    coords: np.ndarray, anchor: tuple[int, int], shape: tuple[int, int]
) -> dict[str, list[int]] | None:
    ar, ac = anchor
    if not (1 <= ar < shape[0] - 1 and 1 <= ac < shape[1] - 1):
        return None
    groups: dict[str, list[int]] = {}
    for raw_r, raw_c in coords:
        item = _classify_relative(anchor, (int(raw_r), int(raw_c)))
        if item is None:
            return None
        orientation, signed_t = item
        groups.setdefault(orientation, []).append(int(signed_t))
    if not 2 <= len(groups) <= 4:
        return None
    return groups


def _solve_color(grid: np.ndarray, color: int) -> tuple[tuple[int, int], dict[str, list[int]]]:
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0
    h, w = grid.shape
    matches: list[tuple[tuple[int, int], dict[str, list[int]]]] = []
    for ar in range(1, h - 1):
        for ac in range(1, w - 1):
            groups = _groups_for_anchor(coords, (ar, ac), grid.shape)
            if groups is not None:
                matches.append(((ar, ac), groups))
    assert len(matches) == 1, f"expected a unique hidden anchor for color {color}, got {len(matches)}"
    return matches[0]


def _paint_anchor(out: np.ndarray, anchor: tuple[int, int], color: int) -> None:
    ar, ac = anchor
    out[ar - 1 : ar + 2, ac - 1 : ac + 2] = int(color)


def _paint_compressed_rays(
    out: np.ndarray, anchor: tuple[int, int], color: int, groups: dict[str, list[int]]
) -> None:
    for orientation, signed_ts in groups.items():
        pos_count = sum(1 for t in signed_ts if t > 0)
        neg_count = sum(1 for t in signed_ts if t < 0)
        for k in range(2, pos_count + 2):
            r, c = _advance(anchor, orientation, k)
            out[r, c] = int(color)
        for k in range(2, neg_count + 2):
            r, c = _advance(anchor, orientation, -k)
            out[r, c] = int(color)


def _count_sides(groups: dict[str, list[int]]) -> dict[str, dict[int, int]]:
    return {
        orientation: {
            -1: sum(1 for t in signed_ts if t < 0),
            1: sum(1 for t in signed_ts if t > 0),
        }
        for orientation, signed_ts in groups.items()
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer each color's missing 3x3 anchor from converging line pixels, then pack the pixels inward against that anchor.

    Input structure:
    - A black background contains several foreground colors.
    - For each color, all pixels lie on 2 to 4 horizontal, vertical, or diagonal lines that pass through one shared hidden grid point.
    - That shared point and its full 3x3 neighborhood contain no pixel of that color, so the anchor block is absent from the input.

    Rule:
    - For each color independently, find the unique grid point whose four canonical lines can cover all pixels of that color.
    - Draw a solid 3x3 block of that color centered on the inferred point.
    - On every used line, compress the color's pixels separately on each side of the block into the nearest possible consecutive cells just outside the block, and erase the original scattered positions.

    Key insight:
    - The target anchor never appears directly; it must be reconstructed from the common intersection of multiple sparse line clues before the ray-compression step can be applied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    out = np.full_like(g, bg)
    for color in colors_present(g, exclude={bg}):
        anchor, groups = _solve_color(g, int(color))
        _paint_anchor(out, anchor, int(color))
        _paint_compressed_rays(out, anchor, int(color), groups)
    return out


def _sample_signed_positions(capacity: int, count: int) -> list[int]:
    return sorted(random.sample(range(2, capacity + 2), count))


def _sample_object(
    shape: tuple[int, int], color: int, reserved: set[tuple[int, int]]
) -> dict | None:
    h, w = shape
    for _ in range(GEN_OBJECT_ATTEMPTS):
        anchor = (random.randint(1, h - 2), random.randint(1, w - 2))
        anchor_cells = _anchor_patch(anchor)
        if anchor_cells & reserved:
            continue

        line_count = random.choices(GEN_LINE_COUNT_CHOICES, weights=GEN_LINE_COUNT_WEIGHTS, k=1)[0]
        orientations = random.sample(list(ORIENTATIONS), line_count)
        sides: dict[str, dict[int, list[int]]] = {}
        union_cells = set(anchor_cells)
        total_pixels = 0
        both_sides_used = False

        for orientation in orientations:
            capacities = {
                sign: _side_capacity(shape, anchor, orientation, sign) for sign in (-1, 1)
            }
            usable_sides = [sign for sign, capacity in capacities.items() if capacity > 0]
            if not usable_sides:
                break

            if len(usable_sides) == 2 and random.random() < 0.42:
                chosen_sides = [-1, 1]
                both_sides_used = True
            else:
                chosen_sides = [random.choice(usable_sides)]

            side_map: dict[int, list[int]] = {}
            for sign in chosen_sides:
                capacity = capacities[sign]
                max_count = min(GEN_MAX_SIDE_PIXELS, capacity)
                if max_count <= 0:
                    break
                count = random.randint(1, max_count)
                mags = _sample_signed_positions(capacity, count)
                side_map[sign] = mags
                total_pixels += count
                for mag in mags:
                    union_cells.add(_advance(anchor, orientation, sign * mag))
                for k in range(2, count + 2):
                    union_cells.add(_advance(anchor, orientation, sign * k))
            else:
                sides[orientation] = side_map
                continue
            break

        if len(sides) != line_count:
            continue
        if total_pixels < max(GEN_MIN_TOTAL_PIXELS, line_count + 1):
            continue
        if not both_sides_used and random.random() < 0.35:
            continue
        if union_cells & reserved:
            continue

        return {
            "color": int(color),
            "anchor": anchor,
            "sides": sides,
            "reserved": union_cells,
        }
    return None


def _render_case(shape: tuple[int, int], specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, Color.BLACK, dtype=int)
    out = np.full(shape, Color.BLACK, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        anchor = spec["anchor"]
        _paint_anchor(out, anchor, color)
        for orientation, side_map in spec["sides"].items():
            for sign, mags in side_map.items():
                for mag in mags:
                    r, c = _advance(anchor, orientation, sign * mag)
                    inp[r, c] = color
                for k in range(2, len(mags) + 2):
                    r, c = _advance(anchor, orientation, sign * k)
                    out[r, c] = color
    return inp, out


def _spec_counts(spec: dict) -> dict[str, dict[int, int]]:
    result: dict[str, dict[int, int]] = {}
    for orientation, side_map in spec["sides"].items():
        result[orientation] = {-1: len(side_map.get(-1, [])), 1: len(side_map.get(1, []))}
    return result


def _generate_case(return_meta: bool = False):
    for _ in range(GEN_CASE_ATTEMPTS):
        shape = (
            random.randint(*GEN_GRID_ROWS),
            random.randint(*GEN_GRID_COLS),
        )
        object_count = random.choices(
            GEN_OBJECT_COUNT_CHOICES, weights=GEN_OBJECT_COUNT_WEIGHTS, k=1
        )[0]
        colors = random.sample(GEN_PALETTE, object_count)

        reserved: set[tuple[int, int]] = set()
        specs: list[dict] = []
        ok = True
        for color in colors:
            spec = _sample_object(shape, color, reserved)
            if spec is None:
                ok = False
                break
            specs.append(spec)
            reserved |= spec["reserved"]
        if not ok:
            continue

        inp, expected = _render_case(shape, specs)
        try:
            solved = main(inp)
            if not np.array_equal(solved, expected):
                continue
            for spec in specs:
                anchor, groups = _solve_color(inp, int(spec["color"]))
                if anchor != spec["anchor"]:
                    raise AssertionError("anchor mismatch")
                if _count_sides(groups) != _spec_counts(spec):
                    raise AssertionError("side-count mismatch")
        except Exception:
            continue

        if return_meta:
            return inp, expected, specs
        return inp
    raise RuntimeError("failed to generate a valid invisible-anchor case")


def generate_input() -> np.ndarray:
    return _generate_case(return_meta=False)
