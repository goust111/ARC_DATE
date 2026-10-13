from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS) not in sys.path:
    sys.path.append(str(_UTILS))

from arc_utils import Color, components_mask, find_components, paste


BG = Color.BLACK
SEP = Color.GREY
COLOR_POOL = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
MAX_ATTEMPTS = 400


def _separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEP)]
    assert len(rows) == 1, "expected exactly one full-width grey separator row"
    return int(rows[0])


def _component_patch(grid: np.ndarray, component: dict, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = map(int, component["bbox"])
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    color = int(component["color"])
    for r, c in component["coords"]:
        patch[int(r) - r0, int(c) - c0] = color
    return patch, (r0, c0, r1, c1)


def _enclosed_hole_area(region: np.ndarray, bg: int) -> int:
    total = 0
    for coords in components_mask(region == int(bg), connectivity=4):
        rows = coords[:, 0]
        cols = coords[:, 1]
        if (
            int(rows.min()) > 0
            and int(rows.max()) < region.shape[0] - 1
            and int(cols.min()) > 0
            and int(cols.max()) < region.shape[1] - 1
        ):
            total += int(coords.shape[0])
    return total


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only hole-containing upper objects, mirror them across the separator, and slide the originals by their hole area.

    Input structure:
    - The grid has black background and one full-width grey horizontal separator row.
    - Above the separator are separated monochrome rectangular objects; some are hollow frames with enclosed black holes and some are solid blocks.
    - Below the separator is otherwise background.

    Rule:
    - Detect the upper 4-connected monochrome objects and measure, inside each object's bounding box, the total area of any 4-connected black region that does not touch the bbox border.
    - Erase every object with no such enclosed hole.
    - For each remaining object, mirror it across the grey separator into the lower half with the same color and same columns.
    - Also move the original upper object horizontally by K cells, where K is its enclosed-hole area; move right if it stays in bounds, otherwise move left by K.
    - Process the kept objects from left to right by original column so later ones overwrite earlier ones on overlap; leave the separator unchanged.

    Key insight:
    - The enclosed black hole serves two roles at once: it selects which objects matter and its area directly determines how far each selected object slides.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    sep_r = _separator_row(g)
    work = g.copy()
    work[sep_r, :] = BG

    objects = []
    for component in find_components(work, bg=BG, connectivity=4, monochromatic=True):
        r0, c0, r1, c1 = map(int, component["bbox"])
        if r1 >= sep_r:
            continue
        region = work[r0 : r1 + 1, c0 : c1 + 1]
        hole_area = _enclosed_hole_area(region, BG)
        if hole_area <= 0:
            continue
        patch, bbox = _component_patch(work, component, BG)
        objects.append(
            {
                "bbox": bbox,
                "patch": patch,
                "hole_area": int(hole_area),
                "sort_key": (int(c0), int(r0)),
            }
        )

    out = np.full((h, w), BG, dtype=int)
    out[sep_r, :] = SEP

    for obj in sorted(objects, key=lambda item: item["sort_key"]):
        patch = obj["patch"]
        r0, c0, r1, c1 = obj["bbox"]
        k = int(obj["hole_area"])

        shift = k if c1 + k < w else -k
        if c0 + shift < 0 or c1 + shift >= w:
            shift = 0

        paste(out, patch, r0, c0 + shift, transparent=BG)
        mirror_r0 = 2 * sep_r - r1
        paste(out, patch[::-1, :], mirror_r0, c0, transparent=BG)

    return out


def _frame_from_area(area: int) -> tuple[int, int]:
    options = [
        (h, w)
        for h in range(3, 7)
        for w in range(3, 7)
        if (h - 2) * (w - 2) == area
    ]
    return random.choice(options)


def _make_frame(color: int, h: int, w: int) -> np.ndarray:
    patch = np.full((h, w), int(color), dtype=int)
    patch[1:-1, 1:-1] = BG
    return patch


def _make_solid(color: int, max_h: int) -> np.ndarray:
    h = random.randint(3, min(6, max_h))
    w = random.randint(3, 6)
    return np.full((h, w), int(color), dtype=int)


def _rect_clear(mask: np.ndarray, r: int, c: int, h: int, w: int, pad: int) -> bool:
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(mask.shape[0], r + h + pad)
    c1 = min(mask.shape[1], c + w + pad)
    return not bool(mask[r0:r1, c0:c1].any())


def _mark_rect(mask: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    mask[r : r + h, c : c + w] = True


def _candidate_positions(
    place_h: int,
    width: int,
    patch_h: int,
    patch_w: int,
    shift: int,
    branch: str,
    orig_occ: np.ndarray,
    shifted_occ: np.ndarray,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r in range(0, place_h - patch_h + 1):
        for c in range(0, width - patch_w + 1):
            if not _rect_clear(orig_occ, r, c, patch_h, patch_w, pad=1):
                continue
            right_fits = c + patch_w - 1 + shift < width
            left_fits = c - shift >= 0
            if branch == "right":
                if not right_fits:
                    continue
                shifted_c = c + shift
            else:
                if right_fits or not left_fits:
                    continue
                shifted_c = c - shift
            if not _rect_clear(shifted_occ, r, shifted_c, patch_h, patch_w, pad=0):
                continue
            candidates.append((r, c))
    return candidates


def _choose_hole_areas(count: int, width: int, max_h: int) -> list[int] | None:
    area_options = []
    for area in (1, 2, 3, 4, 6, 8, 9, 12, 16):
        feasible = False
        for h in range(3, min(6, max_h) + 1):
            for w in range(3, 7):
                if (h - 2) * (w - 2) == area and width - w >= area:
                    feasible = True
        if feasible:
            area_options.append(area)
    if len(area_options) < count:
        return None
    return random.sample(area_options, count)


def _build_attempt() -> np.ndarray | None:
    top_h = random.randint(7, 12)
    height = 2 * top_h + 1
    width = random.randint(18, 28)
    place_h = top_h - 1

    n_objects = random.randint(3, 5)
    n_hollow = random.randint(1, min(3, n_objects))
    hole_areas = _choose_hole_areas(n_hollow, width, place_h)
    if hole_areas is None:
        return None

    colors = random.sample(COLOR_POOL, n_objects)
    hollow_colors = colors[:n_hollow]
    solid_colors = colors[n_hollow:]

    branches = [random.choice(["left", "right"]) for _ in range(n_hollow)]
    if n_hollow >= 2:
        branches[0] = "left"
        branches[1] = "right"
        random.shuffle(branches)

    hollow_specs = []
    for color, area, branch in zip(hollow_colors, hole_areas, branches):
        frame_h, frame_w = _frame_from_area(area)
        if frame_h > place_h or width - frame_w < area:
            return None
        hollow_specs.append(
            {
                "color": int(color),
                "area": int(area),
                "branch": branch,
                "patch": _make_frame(int(color), frame_h, frame_w),
            }
        )

    hollow_specs.sort(
        key=lambda spec: (
            -spec["area"],
            -spec["patch"].shape[0] * spec["patch"].shape[1],
            random.random(),
        )
    )

    orig_occ = np.zeros((place_h, width), dtype=bool)
    shifted_occ = np.zeros((place_h, width), dtype=bool)
    placed: list[tuple[np.ndarray, int, int]] = []

    for spec in hollow_specs:
        patch = spec["patch"]
        ph, pw = patch.shape
        k = int(spec["area"])
        candidates = _candidate_positions(place_h, width, ph, pw, k, spec["branch"], orig_occ, shifted_occ)
        if not candidates:
            return None
        r, c = random.choice(candidates)
        _mark_rect(orig_occ, r, c, ph, pw)
        shifted_c = c + k if spec["branch"] == "right" else c - k
        _mark_rect(shifted_occ, r, shifted_c, ph, pw)
        placed.append((patch, r, c))

    solids: list[np.ndarray] = []
    for color in solid_colors:
        solids.append(_make_solid(int(color), place_h))

    solids.sort(key=lambda patch: (-(patch.shape[0] * patch.shape[1]), random.random()))
    for patch in solids:
        ph, pw = patch.shape
        candidates = [
            (r, c)
            for r in range(0, place_h - ph + 1)
            for c in range(0, width - pw + 1)
            if _rect_clear(orig_occ, r, c, ph, pw, pad=1)
        ]
        if not candidates:
            return None
        r, c = random.choice(candidates)
        _mark_rect(orig_occ, r, c, ph, pw)
        placed.append((patch, r, c))

    grid = np.full((height, width), BG, dtype=int)
    grid[top_h, :] = SEP
    for patch, r, c in placed:
        paste(grid, patch, r, c, transparent=BG)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
