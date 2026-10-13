from __future__ import annotations

import random
from collections import Counter

import numpy as np


# Noise has a fixed visible role; triangle and background colors are randomized
# from the rest of the ARC palette.  No numeric property of a color is used.
NOISE_COLOR = 5
_CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")
_GEN_ATTEMPTS = 500
_SEEN_OUTPUT_INPUT: dict[tuple[tuple[int, int], bytes], bytes] = {}


def _background(grid: np.ndarray) -> int:
    """Return the dominant color, following the shared ARC toolbox convention."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _corner_cell(corner: str, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    return {
        "top_left": (0, 0),
        "top_right": (0, w - 1),
        "bottom_left": (h - 1, 0),
        "bottom_right": (h - 1, w - 1),
    }[corner]


def _to_local(corner: str, r: int, c: int, shape: tuple[int, int]) -> tuple[int, int]:
    """Convert canonical (row, column) coordinates to distances from a corner."""
    h, w = shape
    if corner == "top_left":
        return r, c
    if corner == "top_right":
        return r, w - 1 - c
    if corner == "bottom_left":
        return h - 1 - r, c
    return h - 1 - r, w - 1 - c


def _from_local(corner: str, u: int, v: int, shape: tuple[int, int]) -> tuple[int, int]:
    """Map corner-relative distances (u, v) back to (row, column)."""
    h, w = shape
    if corner == "top_left":
        return u, v
    if corner == "top_right":
        return u, w - 1 - v
    if corner == "bottom_left":
        return h - 1 - u, v
    return h - 1 - u, w - 1 - v


def _outline_cells(
    corner: str, diagonal_sum: int, shape: tuple[int, int]
) -> set[tuple[int, int]]:
    """Return both legs and the hypotenuse of one corner triangle."""
    s = int(diagonal_sum)
    cells = {
        _from_local(corner, 0, d, shape) for d in range(s + 1)
    }
    cells |= {
        _from_local(corner, d, 0, shape) for d in range(s + 1)
    }
    cells |= {
        _from_local(corner, d, s - d, shape) for d in range(s + 1)
    }
    return cells


def _parse_triangles(grid: np.ndarray, bg: int) -> list[dict]:
    """Read every corner L and its same-colored, corner-relative residue line."""
    h, w = grid.shape
    specs: list[dict] = []
    corner_colors: set[int] = set()

    for corner in _CORNERS:
        cr, cc = _corner_cell(corner, grid.shape)
        color = int(grid[cr, cc])
        if color == bg:
            continue
        assert color not in corner_colors, "corner L colors must be distinct"
        corner_colors.add(color)

        global_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        local_coords = [_to_local(corner, r, c, grid.shape) for r, c in global_coords]
        residues = [(u, v) for u, v in local_coords if u > 0 and v > 0]
        assert len(residues) >= 2, "each L needs at least two interior residue samples"

        sums = {u + v for u, v in residues}
        assert len(sums) == 1, "same-colored residues must have one coordinate sum"
        diagonal_sum = int(next(iter(sums)))
        assert 5 <= diagonal_sum < min(h, w), "the inferred triangle must fit the grid"

        vertical = sorted(u for u, v in local_coords if v == 0)
        horizontal = sorted(v for u, v in local_coords if u == 0)
        assert vertical == list(range(len(vertical))), "vertical L arm must be contiguous from its corner"
        assert horizontal == list(range(len(horizontal))), "horizontal L arm must be contiguous from its corner"
        assert len(vertical) == len(horizontal) >= 2, "the two visible L arms must have equal positive length"
        arm_length = len(vertical)
        assert arm_length <= diagonal_sum - 1, "visible arms must stop before their final endpoints"

        allowed = {
            (u, v)
            for u, v in local_coords
            if (u == 0 and v < arm_length)
            or (v == 0 and u < arm_length)
            or (u > 0 and v > 0 and u + v == diagonal_sum)
        }
        assert allowed == set(local_coords), "triangle color may occur only in its L or residue samples"
        assert len({u for u, _v in residues}) >= 2, "residues must visibly sample different diagonal positions"

        specs.append(
            {
                "corner": corner,
                "color": color,
                "sum": diagonal_sum,
                "arm_length": arm_length,
                "residues": set(residues),
            }
        )

    assert specs, "expected at least one corner L with same-colored residues"
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete every corner L into the triangle outline encoded by its same-colored diagonal residues, and erase off-color noise.

    Input structure:
    - A dominant background contains one or more single-pixel-thick L shapes rooted
      at distinct grid corners; every L has a distinct color and two equal short arms.
    - Interior pixels of each L color are sparse samples whose two distances from
      that L's corner have one constant sum. Isolated colors used by no corner are noise.

    Rule:
    - For each corner independently, read the constant s from its same-colored
      interior samples (so the leg length in pixels is n = s + 1).
    - Extend both corner-relative boundary legs through distance s and paint every
      cell u + v = s as the complete hypotenuse. Remove every non-background pixel
      whose color does not belong to a corner L.

    Key insight:
    - Coordinates are measured from each L's own corner: the sparse points are not
      debris but samples of one hidden anti-diagonal, and its constant sum fixes the
      only triangle size compatible with that corner, color, and orientation.
    """
    g = np.asarray(grid)
    assert g.ndim == 2, "expected a two-dimensional ARC grid"
    assert 0 < g.shape[0] <= 30 and 0 < g.shape[1] <= 30, "ARC dimensions must not exceed 30"
    bg = _background(g)
    specs = _parse_triangles(g, bg)

    out = np.full_like(g, bg)
    owner: dict[tuple[int, int], int] = {}
    for spec in specs:
        color = int(spec["color"])
        for cell in _outline_cells(spec["corner"], spec["sum"], g.shape):
            assert cell not in owner, "completed triangle outlines must not overlap"
            owner[cell] = color
            out[cell] = color
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _well_separated(
    candidate: set[tuple[int, int]], existing: set[tuple[int, int]], padding: int = 2
) -> bool:
    if not existing:
        return True
    for r, c in candidate:
        for rr, cc in existing:
            if max(abs(r - rr), abs(c - cc)) <= padding:
                return False
    return True


def _sample_offsets(diagonal_sum: int) -> tuple[int, ...]:
    """Canonical sparse samples determined only by the visible final geometry."""
    s = int(diagonal_sum)
    offsets = {s // 3, (2 * s) // 3}
    if s % 2 == 0:
        offsets.add(s // 2)
    offsets = {u for u in offsets if 0 < u < s}
    result = tuple(sorted(offsets))
    assert len(result) >= 2 and max(result) - min(result) >= 2
    return result


def _arm_length(diagonal_sum: int) -> int:
    """Canonical visible L length, kept strictly shorter than the inferred legs."""
    return 2 + int(diagonal_sum) % 3


def _safe_noise_candidates(
    expected: np.ndarray, bg: int, specs: list[dict]
) -> list[tuple[int, int]]:
    """Choose deterministic, scattered middle cells away from all triangle outlines."""
    h, w = expected.shape
    outline = {tuple(map(int, rc)) for rc in np.argwhere(expected != bg)}
    candidates: list[tuple[int, int]] = []
    for r in range(2, h - 2):
        for c in range(2, w - 2):
            if int(expected[r, c]) != bg:
                continue
            if any(max(abs(r - rr), abs(c - cc)) <= 2 for rr, cc in outline):
                continue
            candidates.append((r, c))

    # Sort spatially so generator construction order cannot change the input
    # corresponding to an otherwise identical completed output.
    spatial_specs = sorted(specs, key=lambda s: _CORNERS.index(str(s["corner"])))
    phase = h * 37 + w * 19 + sum(
        (i + 1) * int(s["sum"]) for i, s in enumerate(spatial_specs)
    )
    candidates.sort(key=lambda p: (((p[0] * 43 + p[1] * 71 + phase) ** 2) % 1009, p))
    return candidates


def _canonical_input(expected: np.ndarray, bg: int, specs: list[dict]) -> np.ndarray | None:
    """Derive exactly one valid residue/noise input from a completed output."""
    inp = np.full_like(expected, bg)
    for spec in specs:
        corner = str(spec["corner"])
        color = int(spec["color"])
        s = int(spec["sum"])
        arm = _arm_length(s)
        for d in range(arm):
            inp[_from_local(corner, d, 0, inp.shape)] = color
            inp[_from_local(corner, 0, d, inp.shape)] = color
        for u in _sample_offsets(s):
            inp[_from_local(corner, u, s - u, inp.shape)] = color

    candidates = _safe_noise_candidates(expected, bg, specs)
    noise_count = 1 + (expected.shape[0] + expected.shape[1] + sum(int(s["sum"]) for s in specs)) % 3
    chosen: list[tuple[int, int]] = []
    for cell in candidates:
        if all(max(abs(cell[0] - r), abs(cell[1] - c)) >= 4 for r, c in chosen):
            chosen.append(cell)
            if len(chosen) == noise_count:
                break
    if len(chosen) != noise_count:
        return None
    for cell in chosen:
        inp[cell] = NOISE_COLOR
    return inp


def _validate_generated(inp: np.ndarray, expected: np.ndarray, bg: int, specs: list[dict]) -> bool:
    """Check the described semantics and counterfactual dependencies, not just closure."""
    try:
        parsed = _parse_triangles(inp, bg)
        if len(parsed) != len(specs):
            return False
        if not np.array_equal(main(inp), expected):
            return False

        corner_colors = {int(spec["color"]) for spec in specs}
        noise = [
            tuple(map(int, rc))
            for rc in np.argwhere((inp != bg) & ~np.isin(inp, list(corner_colors)))
        ]
        if not noise or any(int(inp[cell]) != NOISE_COLOR or int(expected[cell]) != bg for cell in noise):
            return False

        for spec in parsed:
            corner = str(spec["corner"])
            color = int(spec["color"])
            s = int(spec["sum"])
            arm = int(spec["arm_length"])
            if set(u for u, _v in spec["residues"]) != set(_sample_offsets(s)):
                return False

            # Both completion operations must contribute pixels absent from the input.
            new_leg = {
                _from_local(corner, d, 0, inp.shape) for d in range(arm, s + 1)
            } | {
                _from_local(corner, 0, d, inp.shape) for d in range(arm, s + 1)
            }
            full_diagonal = {
                _from_local(corner, u, s - u, inp.shape) for u in range(s + 1)
            }
            if len(new_leg) < 4 or not all(int(inp[p]) == bg and int(expected[p]) == color for p in new_leg):
                return False
            missing_diagonal = {p for p in full_diagonal if int(inp[p]) == bg}
            if len(missing_diagonal) < 2 or not all(int(expected[p]) == color for p in missing_diagonal):
                return False

            # Without this L's residue samples its size is not inferable.
            no_residue = inp.copy()
            no_residue[no_residue == color] = bg
            for d in range(arm):
                no_residue[_from_local(corner, d, 0, inp.shape)] = color
                no_residue[_from_local(corner, 0, d, inp.shape)] = color
            try:
                main(no_residue)
            except AssertionError:
                pass
            else:
                return False

            # Without the corner L, same-colored samples are merely off-color noise.
            no_l = inp.copy()
            for d in range(arm):
                no_l[_from_local(corner, d, 0, inp.shape)] = bg
                no_l[_from_local(corner, 0, d, inp.shape)] = bg
            try:
                counterfactual = main(no_l)
            except AssertionError:
                counterfactual = None
            if counterfactual is not None and np.array_equal(counterfactual, expected):
                return False
    except (AssertionError, IndexError, ValueError):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    object_count = random.randint(2, 4)

    palette = [c for c in range(10) if c != NOISE_COLOR]
    bg = random.choice(palette)
    object_palette = [c for c in palette if c != bg]
    colors = random.sample(object_palette, object_count)
    corners = random.sample(list(_CORNERS), object_count)
    random.shuffle(corners)

    occupied: set[tuple[int, int]] = set()
    specs: list[dict] = []
    max_sum = min(13, h - 3, w - 3)
    if max_sum < 6:
        return None

    for corner, color in zip(corners, colors):
        choices = list(range(6, max_sum + 1))
        random.shuffle(choices)
        placed = False
        for s in choices:
            cells = _outline_cells(corner, s, (h, w))
            if not _well_separated(cells, occupied, padding=2):
                continue
            specs.append({"corner": corner, "color": color, "sum": s})
            occupied |= cells
            placed = True
            break
        if not placed:
            return None

    expected = np.full((h, w), bg, dtype=int)
    for spec in specs:
        for cell in _outline_cells(spec["corner"], int(spec["sum"]), expected.shape):
            expected[cell] = int(spec["color"])

    if int(np.count_nonzero(expected == bg)) <= expected.size // 2:
        return None
    inp = _canonical_input(expected, bg, specs)
    if inp is None or not _validate_generated(inp, expected, bg, specs):
        return None

    # The canonical construction makes the generator injective with respect to
    # outputs: seeing the same output again must imply byte-identical input.
    output_key = (expected.shape, expected.tobytes())
    old_input = _SEEN_OUTPUT_INPUT.get(output_key)
    if old_input is not None and old_input != inp.tobytes():
        return None
    _SEEN_OUTPUT_INPUT[output_key] = inp.tobytes()
    return inp


def generate_input() -> np.ndarray:
    """Generate a varied, non-overlapping valid input for the corner-triangle task."""
    for _ in range(_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed to construct a valid triangle-residue grid")
