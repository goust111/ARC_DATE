from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, find_components, neighbors8


TARGET_PROFILES = (
    {
        "target": (2, 2),
        "pieces": (
            {"profile": (1, 1), "offset": 0},
            {"profile": (1, 1), "offset": 0},
        ),
        "same_area_decoys": ((3, 1), (1, 3)),
    },
    {
        "target": (3, 1),
        "pieces": (
            {"profile": (2,), "offset": 0},
            {"profile": (1, 1), "offset": 0},
        ),
        "same_area_decoys": ((2, 2), (1, 3)),
    },
    {
        "target": (1, 3),
        "pieces": (
            {"profile": (1, 1), "offset": 0},
            {"profile": (2,), "offset": 1},
        ),
        "same_area_decoys": ((2, 2), (3, 1)),
    },
    {
        "target": (2, 2, 1),
        "pieces": (
            {"profile": (1, 1, 1), "offset": 0},
            {"profile": (1, 1), "offset": 0},
        ),
        "same_area_decoys": ((3, 2), (1, 1, 3), (2, 3)),
    },
    {
        "target": (1, 2, 2),
        "pieces": (
            {"profile": (1, 1, 1), "offset": 0},
            {"profile": (1, 1), "offset": 1},
        ),
        "same_area_decoys": ((2, 3), (3, 1, 1), (3, 2)),
    },
)

ALL_PROFILES = tuple(
    {
        (2,),
        (3,),
        (1, 1),
        (2, 1),
        (1, 2),
        (2, 2),
        (3, 1),
        (1, 3),
        (2, 1, 1),
        (1, 1, 2),
        (2, 2, 1),
        (1, 2, 2),
        (3, 2),
        (2, 3),
        (1, 1, 3),
        (3, 1, 1),
    }
)


def _profile_height(profile: tuple[int, ...]) -> int:
    return max(int(h) for h in profile)


def _profile_area(profile: tuple[int, ...]) -> int:
    return sum(int(h) for h in profile)


def _coords_from_profile(profile: tuple[int, ...], bottom: int, left: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for dc, height in enumerate(profile):
        for r in range(int(bottom) - int(height) + 1, int(bottom) + 1):
            coords.append((r, int(left) + dc))
    return coords


def _normalized_mask(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = bbox_coords(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _apply_gravity(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    out[g == Color.TEAL] = Color.TEAL

    h, w = g.shape
    for c in range(w):
        falling = [int(v) for v in g[:, c] if int(v) not in (Color.BLACK, Color.TEAL)]
        free_rows = [r for r in range(h) if int(g[r, c]) != Color.TEAL]
        if len(falling) > len(free_rows):
            raise AssertionError("column has no room for falling pixels")
        for r, color in zip(free_rows[-len(falling) :], falling):
            out[r, c] = color
    return out


def _expected_output(grid: np.ndarray) -> np.ndarray:
    settled = _apply_gravity(grid)
    template_components = find_components(settled, bg=Color.BLACK, connectivity=4, colors=[Color.TEAL])
    assert len(template_components) == 1, "expected exactly one teal template"
    template_shape = _normalized_mask(template_components[0]["coords"])

    out = settled.copy()
    for color in (Color.BLUE, Color.RED):
        for comp in find_components(settled, bg=Color.BLACK, connectivity=4, colors=[color]):
            if _normalized_mask(comp["coords"]) == template_shape:
                coords = comp["coords"]
                out[coords[:, 0], coords[:, 1]] = Color.TEAL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let all non-template colored pixels fall by column, then recolor any gravity-formed component whose shape matches the teal template.

    Input structure:
    - Black background with several small blue/red 4-connected objects in the upper part of the grid.
    - Exactly one teal 4-connected template object sits on the bottom edge and does not move.
    - Objects are separated in the input, but same-color pieces can share columns at different heights.

    Rule:
    - Apply independent column gravity to every non-black, non-teal pixel, preserving each column's top-to-bottom color order and treating teal as a fixed obstacle.
    - Recompute 4-connected monochrome components after the fall.
    - Translation-normalize the teal template shape and every non-teal component shape.
    - Recolor every non-teal component whose normalized shape exactly matches the teal template to teal; leave all other pixels unchanged.

    Key insight:
    - The sought matching shape is not necessarily present in the input: gravity can merge separated same-color objects into a new component whose normalized mask matches the bottom template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    template_components = find_components(g, bg=Color.BLACK, connectivity=4, colors=[Color.TEAL])
    assert len(template_components) == 1, "expected exactly one teal template"
    return _expected_output(g)


def _random_composition(total: int, parts: int) -> list[int]:
    buckets = [0] * parts
    for _ in range(total):
        buckets[random.randrange(parts)] += 1
    return buckets


def _choose_group_columns(widths: dict[str, int], total_width: int) -> dict[str, int] | None:
    order = list(widths)
    random.shuffle(order)
    min_used = sum(widths[name] for name in order) + (len(order) - 1)
    if min_used > total_width:
        return None
    slack = total_width - min_used
    extras = _random_composition(slack, len(order) + 1)
    starts: dict[str, int] = {}
    c = extras[0]
    for idx, name in enumerate(order):
        starts[name] = c
        c += widths[name]
        if idx != len(order) - 1:
            c += 1 + extras[idx + 1]
    return starts


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != Color.BLACK:
            return False
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != Color.BLACK:
                return False
    return True


def _paint(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = int(color)


def _place_target_pieces(grid: np.ndarray, left: int, pieces: tuple[dict, ...], upper_limit: int) -> bool:
    bottom_ranges = []
    for piece in pieces:
        piece_height = _profile_height(piece["profile"])
        bottoms = list(range(piece_height + 1, upper_limit + 1))
        if not bottoms:
            return False
        bottom_ranges.append(bottoms)

    candidates: list[tuple[int, int]] = []
    for b0 in bottom_ranges[0]:
        coords0 = _coords_from_profile(pieces[0]["profile"], b0, left + pieces[0]["offset"])
        if not _can_place(grid, coords0):
            continue
        for b1 in bottom_ranges[1]:
            coords1 = _coords_from_profile(pieces[1]["profile"], b1, left + pieces[1]["offset"])
            if abs(b0 - b1) < 3:
                continue
            if not _can_place(grid, coords1):
                continue
            trial = grid.copy()
            _paint(trial, coords0, Color.BLUE)
            if _can_place(trial, coords1):
                candidates.append((b0, b1))
    if not candidates:
        return False

    b0, b1 = random.choice(candidates)
    for piece, bottom in zip(pieces, (b0, b1)):
        coords = _coords_from_profile(piece["profile"], bottom, left + piece["offset"])
        _paint(grid, coords, Color.BLUE)
    return True


def _place_single_object(
    grid: np.ndarray,
    profile: tuple[int, ...],
    left: int,
    color: int,
    upper_limit: int,
) -> bool:
    height = _profile_height(profile)
    bottoms = list(range(height + 1, upper_limit + 1))
    random.shuffle(bottoms)
    for bottom in bottoms:
        coords = _coords_from_profile(profile, bottom, left)
        if _can_place(grid, coords):
            _paint(grid, coords, color)
            return True
    return False


def _build_input(
    height: int,
    width: int,
    target_case: dict,
    same_profile: tuple[int, ...],
    other_profile: tuple[int, ...],
) -> np.ndarray | None:
    template_profile = target_case["target"]
    widths = {
        "target": len(template_profile),
        "same": len(same_profile),
        "other": len(other_profile),
        "template": len(template_profile),
    }
    starts = _choose_group_columns(widths, width)
    if starts is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    upper_limit = height // 2 + random.randint(1, 2)

    working = grid.copy()
    if not _place_target_pieces(working, starts["target"], target_case["pieces"], upper_limit):
        return None

    if not _place_single_object(working, same_profile, starts["same"], Color.BLUE, upper_limit):
        return None
    if not _place_single_object(working, other_profile, starts["other"], Color.RED, upper_limit):
        return None

    template_coords = _coords_from_profile(template_profile, height - 1, starts["template"])
    if not _can_place(working, template_coords):
        return None
    _paint(working, template_coords, Color.TEAL)

    return working


def _swap_red_blue(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    blue = out == Color.BLUE
    red = out == Color.RED
    out[blue] = Color.RED
    out[red] = Color.BLUE
    return out


def _validate_semantics(grid: np.ndarray, target_profile: tuple[int, ...], target_color: int) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    template_components = find_components(grid, bg=Color.BLACK, connectivity=4, colors=[Color.TEAL])
    if len(template_components) != 1:
        return False
    template_comp = template_components[0]
    if template_comp["bbox"][2] != grid.shape[0] - 1:
        return False
    template_shape = _normalized_mask(template_comp["coords"])
    if template_shape != _normalized_mask(np.array(_coords_from_profile(target_profile, _profile_height(target_profile) - 1, 0))):
        return False

    initial_mono = find_components(
        grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[Color.BLUE, Color.RED, Color.TEAL],
    )
    initial_non_teal = [comp for comp in initial_mono if int(comp["color"]) != Color.TEAL]
    if len(initial_non_teal) < 4:
        return False
    if not all(2 <= int(comp["area"]) <= 5 for comp in initial_non_teal):
        return False

    initial_touching = find_components(grid, bg=Color.BLACK, connectivity=8, monochromatic=False)
    if len(initial_touching) != len(initial_non_teal) + 1:
        return False

    settled = _apply_gravity(grid)
    settled_non_teal = []
    matches = []
    for color in (Color.BLUE, Color.RED):
        comps = find_components(settled, bg=Color.BLACK, connectivity=4, colors=[color])
        settled_non_teal.extend(comps)
        matches.extend(
            comp for comp in comps if _normalized_mask(comp["coords"]) == template_shape
        )
    if len(matches) != 1:
        return False
    if int(matches[0]["color"]) != int(target_color):
        return False

    initial_target = [comp for comp in initial_non_teal if int(comp["color"]) == int(target_color)]
    settled_target = [comp for comp in settled_non_teal if int(comp["color"]) == int(target_color)]
    if len(initial_target) < 3 or len(settled_target) >= len(initial_target):
        return False
    if any(_normalized_mask(comp["coords"]) == template_shape for comp in initial_target):
        return False
    if max(int(comp["area"]) for comp in initial_target) >= int(matches[0]["area"]):
        return False
    if not any(_normalized_mask(comp["coords"]) != template_shape for comp in settled_target):
        return False

    expected = _expected_output(grid)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return False
    if np.array_equal(solved, grid):
        return False
    return True


def _choose_profiles(case: dict) -> tuple[tuple[int, ...], tuple[int, ...]]:
    target = case["target"]
    if random.random() < 0.75:
        same_profile = random.choice(case["same_area_decoys"])
    else:
        same_profile = random.choice([p for p in ALL_PROFILES if p != target])
    other_profile = random.choice([p for p in ALL_PROFILES if p not in {target, same_profile}])
    return tuple(same_profile), tuple(other_profile)


def _gen_attempt() -> np.ndarray | None:
    case = random.choice(TARGET_PROFILES)
    target_color = random.choice((Color.BLUE, Color.RED))
    same_profile, other_profile = _choose_profiles(case)

    min_width = (
        len(case["target"])
        + len(case["target"])
        + len(same_profile)
        + len(other_profile)
        + 3
    )
    width = random.randint(max(12, min_width), 16)
    height = random.randint(11, 15)

    grid = _build_input(height, width, case, same_profile, other_profile)
    if grid is None:
        return None
    if target_color == Color.RED:
        grid = _swap_red_blue(grid)

    if not _validate_semantics(grid, case["target"], target_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
