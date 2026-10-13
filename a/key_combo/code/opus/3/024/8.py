from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


for _parent in Path(__file__).resolve().parents:
    _tools = _parent / "new" / "solvers"
    if (_tools / "arc_utils.py").exists():
        sys.path.insert(0, str(_tools))
        break

from arc_utils import Color, bbox_coords, components_mask, in_bounds, paste


LINE_COLORS = (Color.BLUE, Color.GREEN, Color.YELLOW)
DIR_UNITS = ((0, 1), (1, 0), (1, 1), (1, -1))
GAP_CHOICES = (2, 3, 4)
MAX_EXTRAPOLATION = 8
GEN_ATTEMPTS = 300
SPRITE_OFFSETS = ((-3, -3), (-3, 1), (1, -3), (1, 1))
SPRITE_TEMPLATES = (
    np.array(
        [
            [1, 1, 1],
            [1, 0, 0],
            [1, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=int,
    ),
)


def _sorted_row_major(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted((int(r), int(c)) for r, c in points)


def _ray_from(last_marker: tuple[int, int], step: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = map(int, step)
    r, c = map(int, last_marker)
    out: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not in_bounds(shape, r, c):
            return out
        out.append((r, c))


def _extract_groups(grid: np.ndarray) -> list[dict]:
    groups: list[dict] = []
    for color in LINE_COLORS:
        if not np.any(grid == color):
            continue
        comps = components_mask(grid == color, connectivity=4)
        singletons: list[tuple[int, int]] = []
        sprite_coords = None
        for comp in comps:
            pts = _sorted_row_major([(int(r), int(c)) for r, c in comp])
            if len(pts) == 1:
                singletons.append(pts[0])
            else:
                assert sprite_coords is None, f"color {color} should have one non-marker component"
                sprite_coords = comp

        assert len(singletons) == 3, f"color {color} should have exactly three marker pixels"
        assert sprite_coords is not None, f"color {color} missing sprite component"

        markers = _sorted_row_major(singletons)
        step = (
            markers[1][0] - markers[0][0],
            markers[1][1] - markers[0][1],
        )
        assert step != (0, 0), "arithmetic step must be non-zero"
        assert markers[2] == (markers[1][0] + step[0], markers[1][1] + step[1]), (
            f"color {color} markers must form an arithmetic line"
        )
        assert step[0] >= 0 and (step[0] == 0 or step[1] in (0, step[0], -step[0])) and not (
            step[0] == 0 and step[1] <= 0
        ), f"unsupported step vector {step}"

        r0, c0, r1, c1 = bbox_coords(sprite_coords)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "sprite bbox must be 3x3"
        sprite = np.where(grid[r0 : r1 + 1, c0 : c1 + 1] == color, color, Color.BLACK)
        sprite_cells = {(int(r), int(c)) for r, c in sprite_coords}
        groups.append(
            {
                "color": int(color),
                "markers": markers,
                "step": (int(step[0]), int(step[1])),
                "sprite": sprite.astype(int),
                "sprite_cells": sprite_cells,
                "sprite_top_left": (int(r0), int(c0)),
            }
        )

    groups.sort(key=lambda g: g["markers"][0])
    assert 2 <= len(groups) <= 3, "expected 2-3 line/sprite groups"
    return groups


def _render(grid: np.ndarray, groups: list[dict], *, rotate: bool) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for group in groups:
        for r, c in group["sprite_cells"]:
            out[r, c] = Color.BLACK

    for group in groups:
        ray = _ray_from(group["markers"][-1], group["step"], out.shape)
        assert ray, "each group must extrapolate at least one step"
        for r, c in ray:
            out[r, c] = Color.RED

        quarter_turns = len(ray) % 4 if rotate else 0
        rotated = np.rot90(group["sprite"], -quarter_turns)
        end_r, end_c = ray[-1]
        paste(out, rotated, end_r - 1, end_c - 1, transparent=Color.BLACK, clip=True)

    return out.astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate each arithmetic line to encode a rotation, then move its sprite to the ray endpoint.

    Input structure:
    - The grid is black except for 2-3 color groups using blue, green, and yellow.
    - Each group has three equally spaced collinear marker pixels plus one separate 3x3 same-color sprite.
    - The markers are ordered in row-major order; the sprite sits beside the last marker and does not block the extrapolation path.

    Rule:
    - For each group, read the marker step vector from the row-major-sorted marker triple.
    - Starting after the last marker, keep stepping by that vector, painting red pixels until the next step would leave the grid; let N be the number of red pixels.
        - Rotate the 3x3 sprite clockwise by (N mod 4) quarter-turns, erase the original sprite, and place the rotated sprite so its center aligns with the final red pixel, clipping any out-of-bounds part.

    Key insight:
    - The extrapolated ray length is not only a drawing instruction: its step count modulo four uniquely encodes the sprite rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _render(g, _extract_groups(g), rotate=True)


def _sprite_cells(template: np.ndarray, top_left: tuple[int, int], color: int) -> tuple[np.ndarray, set[tuple[int, int]]]:
    patch = np.where(template != 0, color, Color.BLACK).astype(int)
    r0, c0 = map(int, top_left)
    cells = {
        (r0 + int(r), c0 + int(c))
        for r, c in np.argwhere(template != 0)
    }
    return patch, cells


def _touches4(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r1, c1 in a:
        for r2, c2 in b:
            if abs(r1 - r2) + abs(c1 - c2) == 1:
                return True
    return False


def _enumerate_geometries(shape: tuple[int, int], template: np.ndarray, target_mod: int) -> list[dict]:
    h, w = map(int, shape)
    out: list[dict] = []
    rotated_mask = np.rot90(template, -target_mod)

    for gap in GAP_CHOICES:
        for dr0, dc0 in DIR_UNITS:
            step = (gap * dr0, gap * dc0)
            for r0 in range(h):
                for c0 in range(w):
                    markers = [
                        (r0, c0),
                        (r0 + step[0], c0 + step[1]),
                        (r0 + 2 * step[0], c0 + 2 * step[1]),
                    ]
                    if not all(in_bounds(shape, r, c) for r, c in markers):
                        continue

                    ray = _ray_from(markers[-1], step, shape)
                    if not ray or len(ray) > MAX_EXTRAPOLATION or len(ray) % 4 != target_mod:
                        continue

                    last = markers[-1]
                    for off_r, off_c in SPRITE_OFFSETS:
                        top_left = (last[0] + off_r, last[1] + off_c)
                        sr, sc = top_left
                        if sr < 0 or sc < 0 or sr + 3 > h or sc + 3 > w:
                            continue

                        input_cells = {
                            (sr + int(r), sc + int(c))
                            for r, c in np.argwhere(template != 0)
                        }
                        marker_cells = set(markers)
                        path_cells = set(ray)
                        if input_cells & marker_cells:
                            continue
                        if input_cells & path_cells:
                            continue
                        if _touches4(input_cells, marker_cells):
                            continue

                        end_r, end_c = ray[-1]
                        output_sprite = {
                            (end_r - 1 + int(r), end_c - 1 + int(c))
                            for r, c in np.argwhere(rotated_mask != 0)
                            if in_bounds(shape, end_r - 1 + int(r), end_c - 1 + int(c))
                        }
                        output_cells = marker_cells | path_cells | output_sprite
                        out.append(
                            {
                                "markers": markers,
                                "step": step,
                                "ray": ray,
                                "n_steps": len(ray),
                                "sprite_top_left": top_left,
                                "input_cells": marker_cells | input_cells,
                                "marker_cells": marker_cells,
                                "path_cells": path_cells,
                                "output_cells": output_cells,
                                "direction": (dr0, dc0),
                            }
                        )

    random.shuffle(out)
    return out


def _build_group(color: int, geometry: dict, template: np.ndarray) -> dict:
    sprite, sprite_cells = _sprite_cells(template, geometry["sprite_top_left"], color)
    return {
        "color": int(color),
        "markers": [(int(r), int(c)) for r, c in geometry["markers"]],
        "step": (int(geometry["step"][0]), int(geometry["step"][1])),
        "sprite": sprite,
        "sprite_cells": sprite_cells,
        "sprite_top_left": tuple(map(int, geometry["sprite_top_left"])),
    }


def _choose_geometry(
    geometries: list[dict],
    input_occ: set[tuple[int, int]],
    path_occ: set[tuple[int, int]],
    output_occ: set[tuple[int, int]],
    used_directions: set[tuple[int, int]],
) -> dict | None:
    valid: list[tuple[int, dict]] = []
    for geometry in geometries:
        if geometry["input_cells"] & input_occ:
            continue
        if geometry["path_cells"] & input_occ:
            continue
        if geometry["input_cells"] & path_occ:
            continue
        if geometry["output_cells"] & output_occ:
            continue

        score = 0
        if geometry["direction"] not in used_directions:
            score += 2
        score += len({r for r, _ in geometry["marker_cells"]})
        valid.append((score, geometry))

    if not valid:
        return None

    valid.sort(key=lambda item: item[0], reverse=True)
    best_score = valid[0][0]
    best = [geometry for score, geometry in valid if score == best_score]
    return random.choice(best)


def _metadata_render(shape: tuple[int, int], groups: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for group in groups:
        color = group["color"]
        for r, c in group["markers"]:
            grid[r, c] = color
        sr, sc = group["sprite_top_left"]
        paste(grid, group["sprite"], sr, sc, transparent=Color.BLACK)
    return _render(grid, groups, rotate=True)


def _validate_generated_grid(grid: np.ndarray, groups: list[dict]) -> None:
    extracted = _extract_groups(grid)
    assert len(extracted) == len(groups), "group count mismatch"

    expected = _metadata_render(grid.shape, groups)
    actual = main(grid)
    assert np.array_equal(actual, expected), "main() must match metadata-derived output"

    no_rotation = _render(grid, extracted, rotate=False)
    assert not np.array_equal(actual, no_rotation), "generator must reject no-rotation-degenerate cases"

    mods = []
    all_input = {(int(r), int(c)) for r, c in np.argwhere(grid != Color.BLACK)}
    for group in groups:
        ray = _ray_from(group["markers"][-1], group["step"], grid.shape)
        assert ray, "each group needs a non-empty extrapolation"
        mods.append(len(ray) % 4)
        assert set(ray) & set(group["sprite_cells"]) == set(), "sprite may not block its own ray"
        other_pixels = all_input - set(group["markers"]) - set(group["sprite_cells"])
        assert set(ray).isdisjoint(other_pixels), "ray must not pass through another group's input pixels"

    assert len(set(mods)) >= 2, "at least two groups must have different rotation codes"


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    n_groups = 3 if random.random() < 0.65 else 2
    colors = random.sample(list(LINE_COLORS), n_groups)
    mods = random.sample([0, 1, 2, 3], n_groups)

    template = random.choice(SPRITE_TEMPLATES).copy()
    template = np.rot90(template, -random.randrange(4))
    geometries_by_mod = {mod: _enumerate_geometries((h, w), template, mod) for mod in mods}
    if any(not geometries_by_mod[mod] for mod in mods):
        return None

    input_occ: set[tuple[int, int]] = set()
    path_occ: set[tuple[int, int]] = set()
    output_occ: set[tuple[int, int]] = set()
    used_directions: set[tuple[int, int]] = set()
    groups: list[dict] = []

    group_order = list(range(n_groups))
    random.shuffle(group_order)
    planned = [(colors[i], mods[i]) for i in group_order]
    planned.sort(key=lambda item: item[1] == 0)

    for color, mod in planned:
        geometry = _choose_geometry(geometries_by_mod[mod], input_occ, path_occ, output_occ, used_directions)
        if geometry is None:
            return None

        group = _build_group(color, geometry, template)
        groups.append(group)
        input_occ |= geometry["input_cells"]
        path_occ |= geometry["path_cells"]
        output_occ |= geometry["output_cells"]
        used_directions.add(geometry["direction"])

    grid = np.zeros((h, w), dtype=int)
    for group in groups:
        color = group["color"]
        for r, c in group["markers"]:
            grid[r, c] = color
        sr, sc = group["sprite_top_left"]
        paste(grid, group["sprite"], sr, sc, transparent=Color.BLACK)

    _validate_generated_grid(grid, groups)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arithmetic-line rotation input")
