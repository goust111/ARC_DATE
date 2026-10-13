from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, get_bg, in_bounds, paste, random_free_location


BG = Color.BLACK
GRAY = Color.GREY
CARDINAL_OFFSETS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
NEIGHBOR_OFFSETS = [
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
]

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
BARE_COUNT_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 300

FOOTPRINT = np.ones((3, 3), dtype=int)
MARKER_COLOR_POOL = [color for color in Color.NOT_BLACK if color != GRAY]


def _patch_index(offset: tuple[int, int]) -> tuple[int, int]:
    dr, dc = offset
    return dr + 1, dc + 1


def _cardinal_index(offset: tuple[int, int]) -> int:
    return CARDINAL_OFFSETS.index(tuple(offset))


def _rotate_offset_cw(offset: tuple[int, int], k: int) -> tuple[int, int]:
    dr, dc = map(int, offset)
    for _ in range(k % 4):
        dr, dc = dc, -dr
    return dr, dc


def _rotate_patch_cw(patch: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(np.asarray(patch, dtype=int), -(k % 4))


def _gray_neighbor_offset(grid: np.ndarray, r: int, c: int, bg: int) -> tuple[int, int] | None:
    color = int(grid[r, c])
    if color in (bg, GRAY):
        return None
    hits = []
    for dr, dc in CARDINAL_OFFSETS:
        rr, cc = r + dr, c + dc
        if in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == GRAY:
            hits.append((dr, dc))
    if len(hits) != 1:
        return None
    return hits[0]


def _extract_roles(grid: np.ndarray) -> dict[int, dict[str, list[dict]]]:
    bg = int(get_bg(grid))
    roles: dict[int, dict[str, list[dict]]] = {}
    h, w = grid.shape

    for r in range(h):
        for c in range(w):
            gray_off = _gray_neighbor_offset(grid, r, c, bg)
            if gray_off is None:
                continue
            assert 0 < r < h - 1 and 0 < c < w - 1, "markers must have a full 3x3 neighborhood"

            patch = grid[r - 1 : r + 2, c - 1 : c + 2].copy()
            assert int(np.count_nonzero(patch == GRAY)) == 1, "each marker neighborhood must contain one gray cell"

            payload = (patch != bg) & (patch != GRAY)
            payload[1, 1] = False
            bucket = roles.setdefault(int(grid[r, c]), {"exemplars": [], "bares": []})
            record = {"center": (r, c), "gray_off": gray_off, "patch": patch}
            if bool(payload.any()):
                bucket["exemplars"].append(record)
            else:
                bucket["bares"].append(record)

    assert roles, "expected at least one marker color"
    for color, data in roles.items():
        assert len(data["exemplars"]) == 1, f"marker color {color} must have exactly one exemplar"
    return roles


def _stamp_from_template(
    out: np.ndarray,
    center: tuple[int, int],
    template_patch: np.ndarray,
    source_gray_off: tuple[int, int],
    target_gray_off: tuple[int, int],
    bg: int,
) -> None:
    k = (_cardinal_index(target_gray_off) - _cardinal_index(source_gray_off)) % 4
    rotated = _rotate_patch_cw(template_patch, k)

    marker_r, marker_c = center
    sat_pr, sat_pc = _patch_index(target_gray_off)
    assert int(rotated[1, 1]) == int(out[marker_r, marker_c]), "template center must stay on the marker"
    assert int(rotated[sat_pr, sat_pc]) == GRAY, "rotated gray compass must align with the satellite"

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr, dc) == (0, 0):
                continue
            color = int(rotated[dr + 1, dc + 1])
            if color in (bg, GRAY):
                continue
            rr, cc = marker_r + dr, marker_c + dc
            assert in_bounds(out.shape, rr, cc)
            assert int(out[rr, cc]) in (bg, color), "stamped decorations must not overwrite other objects"
            if int(out[rr, cc]) == bg:
                out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each bare marker with the rotated 3x3 decoration from the exemplar of the same color.

    Input structure:
    - The background is dominant, and gray cells act only as exemplar compass cells or bare-marker satellites.
    - Each marker color has exactly one decorated exemplar whose 3x3 neighborhood contains one gray cardinal neighbor.
    - The remaining markers of that color are bare except for their own single gray cardinal satellite.

    Rule:
    - Group markers by color and use the unique decorated exemplar of each color as that color's template.
    - Rotate the exemplar's full 3x3 neighborhood so its gray compass cell points toward a bare marker's gray satellite.
    - Copy the rotated non-background, non-gray decoration cells into the bare marker's 3x3 neighborhood, filling only background cells and preserving the marker and satellite.

    Key insight:
    - The marker color chooses which exemplar template to use, while the local gray satellite chooses how much that template must rotate before stamping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    roles = _extract_roles(g)
    out = g.copy()

    for marker_color, data in roles.items():
        exemplar = data["exemplars"][0]
        source_gray_off = exemplar["gray_off"]
        template_patch = exemplar["patch"]
        assert int(template_patch[1, 1]) == marker_color

        for bare in data["bares"]:
            center = bare["center"]
            sat_r = center[0] + bare["gray_off"][0]
            sat_c = center[1] + bare["gray_off"][1]
            assert int(g[sat_r, sat_c]) == GRAY, "bare markers must carry a gray satellite"
            _stamp_from_template(
                out,
                center,
                template_patch,
                source_gray_off,
                bare["gray_off"],
                bg,
            )

    return out


def _allowed_template_offsets(gray_off: tuple[int, int]) -> list[tuple[int, int]]:
    forbidden = {gray_off}
    for dr, dc in CARDINAL_OFFSETS:
        rr, cc = gray_off[0] + dr, gray_off[1] + dc
        if -1 <= rr <= 1 and -1 <= cc <= 1 and (rr, cc) != (0, 0):
            forbidden.add((rr, cc))
    return [off for off in NEIGHBOR_OFFSETS if off not in forbidden]


def _build_template_patch(
    marker_color: int,
    gray_off: tuple[int, int],
    decoration: dict[tuple[int, int], int],
) -> np.ndarray:
    patch = np.full((3, 3), BG, dtype=int)
    patch[1, 1] = int(marker_color)
    patch[_patch_index(gray_off)] = GRAY
    for off, color in decoration.items():
        patch[_patch_index(off)] = int(color)
    return patch


def _build_bare_patch(marker_color: int, gray_off: tuple[int, int]) -> np.ndarray:
    patch = np.full((3, 3), BG, dtype=int)
    patch[1, 1] = int(marker_color)
    patch[_patch_index(gray_off)] = GRAY
    return patch


def _sample_template(marker_color: int, blocked_colors: set[int]) -> tuple[np.ndarray, tuple[int, int]]:
    gray_off = random.choice(CARDINAL_OFFSETS)
    allowed_offsets = _allowed_template_offsets(gray_off)
    n_cells = random.randint(2, min(4, len(allowed_offsets)))
    offsets = random.sample(allowed_offsets, n_cells)

    palette_pool = [color for color in Color.NOT_BLACK if color not in blocked_colors]
    n_palette = random.randint(1, min(2, len(palette_pool)))
    palette = random.sample(palette_pool, n_palette)

    assigned = list(palette)
    while len(assigned) < n_cells:
        assigned.append(random.choice(palette))
    random.shuffle(assigned)

    decoration = {off: color for off, color in zip(offsets, assigned)}
    return _build_template_patch(marker_color, gray_off, decoration), gray_off


def _sample_bare_directions(exemplar_gray_off: tuple[int, int], count: int) -> list[tuple[int, int]]:
    first = random.choice([off for off in CARDINAL_OFFSETS if off != exemplar_gray_off])
    second = random.choice([off for off in CARDINAL_OFFSETS if off != first])
    directions = [first, second]
    while len(directions) < count:
        directions.append(random.choice(CARDINAL_OFFSETS))
    random.shuffle(directions)
    return directions[:count]


def _oracle_output(
    grid: np.ndarray,
    templates: dict[int, dict[str, object]],
    bare_specs: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    for spec in bare_specs:
        marker_color = int(spec["marker_color"])
        template = templates[marker_color]
        _stamp_from_template(
            out,
            spec["center"],
            template["patch"],
            template["gray_off"],
            spec["gray_off"],
            BG,
        )
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    marker_colors = random.sample(MARKER_COLOR_POOL, 2)
    blocked_colors = {BG, GRAY, *marker_colors}

    templates: dict[int, dict[str, object]] = {}
    objects: list[dict[str, object]] = []
    for marker_color in marker_colors:
        template_patch, gray_off = _sample_template(marker_color, blocked_colors)
        templates[marker_color] = {"patch": template_patch, "gray_off": gray_off}
        objects.append({"kind": "exemplar", "patch": template_patch, "marker_color": marker_color})

        bare_count = random.randint(*BARE_COUNT_RANGE)
        for sat_off in _sample_bare_directions(gray_off, bare_count):
            objects.append(
                {
                    "kind": "bare",
                    "patch": _build_bare_patch(marker_color, sat_off),
                    "marker_color": marker_color,
                    "gray_off": sat_off,
                }
            )

    random.shuffle(objects)
    grid = np.full((height, width), BG, dtype=int)
    occupied = np.full((height, width), BG, dtype=int)
    bare_specs: list[dict[str, object]] = []

    for obj in objects:
        try:
            r0, c0 = random_free_location(
                occupied,
                FOOTPRINT,
                transparent=BG,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None

        paste(occupied, FOOTPRINT, r0, c0)
        paste(grid, obj["patch"], r0, c0, transparent=BG)
        center = (r0 + 1, c0 + 1)

        if obj["kind"] == "bare":
            bare_specs.append(
                {
                    "center": center,
                    "gray_off": obj["gray_off"],
                    "marker_color": obj["marker_color"],
                }
            )

    expected = _oracle_output(grid, templates, bare_specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid compass-stamp input")
