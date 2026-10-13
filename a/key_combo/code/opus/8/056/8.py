from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

DIRS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

SIDE_ALLOWED_DIRS = {
    "left": ((-1, 0), (1, 0), (0, 1), (-1, 1), (1, 1)),
    "right": ((-1, 0), (1, 0), (0, -1), (-1, -1), (1, -1)),
    "up": ((0, -1), (0, 1), (1, 0), (1, -1), (1, 1)),
    "down": ((0, -1), (0, 1), (-1, 0), (-1, -1), (-1, 1)),
}


@dataclass(frozen=True)
class RaySpec:
    direction: tuple[int, int]
    radii: tuple[int, ...]


@dataclass(frozen=True)
class ObjectSpec:
    color: int
    side: str
    helper_shift: int
    rays: tuple[RaySpec, ...]


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _anchor_rel() -> set[tuple[int, int]]:
    return {(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _helper_rel(side: str, shift: int) -> set[tuple[int, int]]:
    if side == "left":
        return {(shift - 1 + dr, dc) for dr in (0, 1) for dc in (-3, -2)}
    if side == "right":
        return {(shift - 1 + dr, dc) for dr in (0, 1) for dc in (2, 3)}
    if side == "up":
        return {(dr, shift - 1 + dc) for dr in (-3, -2) for dc in (0, 1)}
    return {(dr, shift - 1 + dc) for dr in (2, 3) for dc in (0, 1)}


def _compressed_rel(spec: ObjectSpec) -> set[tuple[int, int]]:
    cells = set(_anchor_rel())
    for ray in spec.rays:
        dr, dc = ray.direction
        for i in range(len(ray.radii)):
            cells.add(((i + 2) * dr, (i + 2) * dc))
    return cells


def _input_rel(spec: ObjectSpec) -> set[tuple[int, int]]:
    cells = set(_anchor_rel())
    cells |= _helper_rel(spec.side, spec.helper_shift)
    for ray in spec.rays:
        dr, dc = ray.direction
        for radius in ray.radii:
            cells.add((radius * dr, radius * dc))
    return cells


def _mirror_rel(cells: Iterable[tuple[int, int]], side: str) -> set[tuple[int, int]]:
    out = set()
    for dr, dc in cells:
        if side == "left":
            out.add((dr, -3 - dc))
        elif side == "right":
            out.add((dr, 3 - dc))
        elif side == "up":
            out.add((-3 - dr, dc))
        else:
            out.add((3 - dr, dc))
    return out


def _abs_cells(center: tuple[int, int], rel_cells: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    cr, cc = center
    return {(cr + dr, cc + dc) for dr, dc in rel_cells}


def _bbox(cells: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(cells)
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _find_anchor(mask: np.ndarray) -> tuple[int, int, int, int]:
    h, w = mask.shape
    anchors = []
    for r in range(h - 2):
        for c in range(w - 2):
            if bool(mask[r : r + 3, c : c + 3].all()):
                anchors.append((r, c, r + 2, c + 2))
    assert len(anchors) == 1, "each color must contain exactly one solid 3x3 anchor block"
    return anchors[0]


def _find_helper_side(grid: np.ndarray, anchor: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = anchor
    candidates = []
    for shift in (0, 1):
        if c0 >= 2 and np.all(grid[r0 + shift : r0 + shift + 2, c0 - 2 : c0] == GREY):
            candidates.append("left")
        if c1 + 2 < grid.shape[1] and np.all(grid[r0 + shift : r0 + shift + 2, c1 + 1 : c1 + 3] == GREY):
            candidates.append("right")
        if r0 >= 2 and np.all(grid[r0 - 2 : r0, c0 + shift : c0 + shift + 2] == GREY):
            candidates.append("up")
        if r1 + 2 < grid.shape[0] and np.all(grid[r1 + 1 : r1 + 3, c0 + shift : c0 + shift + 2] == GREY):
            candidates.append("down")
    assert len(candidates) == 1, "each anchor must have exactly one adjacent grey 2x2 helper"
    return candidates[0]


def _collect_compressed_cells(grid: np.ndarray, color: int, anchor: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = anchor
    cr, cc = r0 + 1, c0 + 1
    rays: dict[tuple[int, int], list[int]] = {}

    for raw_r, raw_c in np.argwhere(grid == color):
        r, c = int(raw_r), int(raw_c)
        if r0 <= r <= r1 and c0 <= c <= c1:
            continue
        dr = r - cr
        dc = c - cc
        sdr = _sgn(dr)
        sdc = _sgn(dc)
        assert (sdr, sdc) != (0, 0)
        assert dr == 0 or dc == 0 or abs(dr) == abs(dc), "loose pixels must lie on 8-neighbor rays"
        radius = max(abs(dr), abs(dc))
        assert radius >= 2, "loose pixels must lie outside the 3x3 anchor"
        rays.setdefault((sdr, sdc), []).append(radius)

    assert len(rays) >= 2, "each color needs at least two ray directions"
    cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    for (dr, dc), radii in rays.items():
        radii = sorted(radii)
        assert len(radii) == len(set(radii)), "each ray can contain at most one pixel per distance"
        for i in range(len(radii)):
            cells.add((cr + (i + 2) * dr, cc + (i + 2) * dc))
    return cells


def _mirror_abs(cells: Iterable[tuple[int, int]], side: str, anchor: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = anchor
    out = set()
    for r, c in cells:
        if side == "left":
            out.add((r, 2 * c0 - 1 - c))
        elif side == "right":
            out.add((r, 2 * c1 + 1 - c))
        elif side == "up":
            out.add((2 * r0 - 1 - r, c))
        else:
            out.add((2 * r1 + 1 - r, c))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each color's ray pixels to its 3x3 anchor, then mirror that dense pattern across the helper-indicated anchor edge in grey.

    Input structure:
    - The background is black.
    - There are exactly two non-grey foreground colors; each color has one solid 3x3 anchor block and several same-colored single pixels lying on horizontal, vertical, or 45-degree rays from the anchor center.
    - Each anchor has one grey 2x2 helper square orthogonally adjacent to one of its four sides.

    Rule:
    - For each foreground color, keep the 3x3 anchor fixed and group all same-colored loose pixels by ray direction from the anchor center.
    - Compress every ray inward so its pixels occupy consecutive cells starting at the first cell just outside the anchor on that ray.
    - Use the helper square to choose the mirror axis: reflect the anchor plus all compressed pixels across the contacted anchor edge and draw the reflected copy in grey, then remove the helper squares.
    - Keep the original-colored compressed pattern unchanged; if a grey mirror would land on an original-colored cell, the original color stays visible.

    Key insight:
    - The sparse pixels do not mirror in place: they must first collapse into a compact pattern around the anchor, and only that compact pattern is mirrored using the anchor-helper contact edge as the axis.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    colors = sorted(int(v) for v in np.unique(g) if int(v) not in (BG, GREY))
    assert len(colors) == 2, "expected exactly two non-grey foreground colors"

    tasks = []
    for color in colors:
        anchor = _find_anchor(g == color)
        side = _find_helper_side(g, anchor)
        compressed = _collect_compressed_cells(g, color, anchor)
        center = (anchor[0] + 1, anchor[1] + 1)
        tasks.append((center, color, anchor, side, compressed))
        for r, c in compressed:
            assert _in_bounds(g.shape, r, c)
            assert out[r, c] in (BG, color)
            out[r, c] = color

    for _center, _color, anchor, side, compressed in sorted(tasks, key=lambda x: x[0]):
        mirrored = _mirror_abs(compressed, side, anchor)
        for r, c in mirrored:
            assert _in_bounds(g.shape, r, c), "mirrored copy must remain inside the grid"
            if out[r, c] in (BG, GREY):
                out[r, c] = GREY

    return out


def _choose_dirs(side: str, need_cardinal: bool, need_diagonal: bool) -> tuple[tuple[int, int], ...]:
    allowed = list(SIDE_ALLOWED_DIRS[side])
    cardinals = [d for d in allowed if 0 in d]
    diagonals = [d for d in allowed if 0 not in d]

    chosen: list[tuple[int, int]] = []
    if need_cardinal:
        chosen.append(random.choice(cardinals))
    if need_diagonal:
        diag = random.choice(diagonals)
        if diag not in chosen:
            chosen.append(diag)

    target_count = random.randint(2, len(allowed))
    random.shuffle(allowed)
    for direction in allowed:
        if direction not in chosen:
            chosen.append(direction)
        if len(chosen) >= target_count:
            break
    return tuple(chosen)


def _choose_radii(count: int) -> tuple[int, ...]:
    max_radius = min(8, count + random.randint(2, 5))
    if max_radius < count + 1:
        max_radius = count + 1
    return tuple(sorted(random.sample(range(2, max_radius + 1), count)))


def _build_spec(color: int, side: str, need_cardinal: bool, need_diagonal: bool) -> ObjectSpec:
    for _ in range(80):
        directions = _choose_dirs(side, need_cardinal, need_diagonal)
        counts = [random.randint(1, 3) for _ in directions]
        if all(count == 1 for count in counts):
            counts[random.randrange(len(counts))] = random.randint(2, 3)

        rays = []
        for direction, count in zip(directions, counts, strict=False):
            rays.append(RaySpec(direction=direction, radii=_choose_radii(count)))
        spec = ObjectSpec(
            color=color,
            side=side,
            helper_shift=random.randint(0, 1),
            rays=tuple(rays),
        )
        try:
            _validate_object_spec(spec)
            return spec
        except AssertionError:
            continue
    raise AssertionError("unable to build a non-degenerate object spec")


def _validate_object_spec(spec: ObjectSpec) -> None:
    assert spec.side in SIDE_ALLOWED_DIRS
    assert spec.helper_shift in (0, 1)
    dirs = [ray.direction for ray in spec.rays]
    assert len(dirs) == len(set(dirs))
    assert len(dirs) >= 2
    assert set(dirs).issubset(set(SIDE_ALLOWED_DIRS[spec.side]))
    assert any(0 in direction for direction in dirs)
    assert any(0 not in direction for direction in dirs)
    assert any(len(ray.radii) >= 2 for ray in spec.rays)
    assert any(tuple(ray.radii) != tuple(range(2, 2 + len(ray.radii))) for ray in spec.rays)
    for ray in spec.rays:
        assert tuple(sorted(ray.radii)) == ray.radii
        assert len(ray.radii) == len(set(ray.radii))
        assert all(radius >= 2 for radius in ray.radii)


def _valid_centers(shape: tuple[int, int], spec: ObjectSpec) -> list[tuple[int, int]]:
    rel_all = _input_rel(spec) | _compressed_rel(spec) | _mirror_rel(_compressed_rel(spec), spec.side)
    r0, c0, r1, c1 = _bbox(rel_all)
    centers = []
    for cr in range(-r0, shape[0] - r1):
        for cc in range(-c0, shape[1] - c1):
            centers.append((cr, cc))
    random.shuffle(centers)
    return centers


def _render_from_specs(
    shape: tuple[int, int],
    placements: list[tuple[ObjectSpec, tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros(shape, dtype=int)
    out = np.zeros(shape, dtype=int)

    for spec, center in placements:
        for r, c in _abs_cells(center, _anchor_rel()):
            inp[r, c] = spec.color
            out[r, c] = spec.color
        for r, c in _abs_cells(center, _helper_rel(spec.side, spec.helper_shift)):
            inp[r, c] = GREY
        for ray in spec.rays:
            dr, dc = ray.direction
            for radius in ray.radii:
                r, c = center[0] + radius * dr, center[1] + radius * dc
                inp[r, c] = spec.color
            for i in range(len(ray.radii)):
                r, c = center[0] + (i + 2) * dr, center[1] + (i + 2) * dc
                out[r, c] = spec.color

    for spec, center in sorted(placements, key=lambda x: x[1]):
        mirrored = _abs_cells(center, _mirror_rel(_compressed_rel(spec), spec.side))
        for r, c in mirrored:
            if out[r, c] == BG:
                out[r, c] = GREY
    return inp, out


def _semantic_grid_checks(
    inp: np.ndarray,
    out: np.ndarray,
    placements: list[tuple[ObjectSpec, tuple[int, int]]],
) -> None:
    assert inp.ndim == 2 and out.ndim == 2
    assert max(inp.shape) <= 30 and max(out.shape) <= 30
    assert sorted(int(v) for v in np.unique(inp) if int(v) not in (BG, GREY)) == sorted(
        spec.color for spec, _ in placements
    )
    assert int(np.sum(inp == GREY)) == 8
    assert int(np.sum(out == GREY)) >= 18

    input_sets = []
    output_sets = []
    for spec, center in placements:
        anchor = _abs_cells(center, _anchor_rel())
        helper = _abs_cells(center, _helper_rel(spec.side, spec.helper_shift))
        sparse = set()
        compressed = _abs_cells(center, _compressed_rel(spec))
        mirrored = _abs_cells(center, _mirror_rel(_compressed_rel(spec), spec.side))

        for ray in spec.rays:
            dr, dc = ray.direction
            for radius in ray.radii:
                sparse.add((center[0] + radius * dr, center[1] + radius * dc))

        assert anchor.isdisjoint(helper)
        assert helper.isdisjoint(sparse)
        assert len({ray.direction for ray in spec.rays}) >= 2
        assert all(_in_bounds(inp.shape, r, c) for r, c in anchor | helper | sparse | compressed | mirrored)
        input_sets.append(anchor | helper | sparse)
        output_sets.append(compressed | mirrored)

    assert input_sets[0].isdisjoint(input_sets[1])
    assert output_sets[0].isdisjoint(output_sets[1])
    assert np.array_equal(main(inp), out)


def _generate_attempt() -> np.ndarray | None:
    try:
        shape = (random.randint(16, 24), random.randint(16, 24))
        colors = random.sample(COLOR_POOL, 2)
        sides = random.sample(tuple(SIDE_ALLOWED_DIRS), 2)

        specs = [
            _build_spec(colors[0], sides[0], need_cardinal=True, need_diagonal=True),
            _build_spec(colors[1], sides[1], need_cardinal=True, need_diagonal=True),
        ]

        center_options = [_valid_centers(shape, spec) for spec in specs]
        if not center_options[0] or not center_options[1]:
            return None

        placements: list[tuple[ObjectSpec, tuple[int, int]]] = []
        used_input: set[tuple[int, int]] = set()
        used_output: set[tuple[int, int]] = set()

        for idx, spec in enumerate(specs):
            found = None
            for center in center_options[idx]:
                output_rel = _compressed_rel(spec) | _mirror_rel(_compressed_rel(spec), spec.side)
                input_cells = _abs_cells(center, _input_rel(spec))
                output_cells = _abs_cells(center, output_rel)
                if input_cells & used_input:
                    continue
                if output_cells & used_output:
                    continue
                if any(max(abs(center[0] - prev[1][0]), abs(center[1] - prev[1][1])) < 7 for prev in placements):
                    continue
                found = center
                break
            if found is None:
                return None
            placements.append((spec, found))
            used_input |= _abs_cells(found, _input_rel(spec))
            used_output |= _abs_cells(found, _compressed_rel(spec) | _mirror_rel(_compressed_rel(spec), spec.side))

        inp, out = _render_from_specs(shape, placements)
        _semantic_grid_checks(inp, out, placements)
        return inp
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
