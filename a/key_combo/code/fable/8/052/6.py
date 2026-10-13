from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

DIR_BY_COLOR = {
    RED: (0, 1),      # right
    BLUE: (0, -1),    # left
    GREEN: (1, 0),    # down
    YELLOW: (-1, 0),  # up
}

MOTION_COLORS = tuple(DIR_BY_COLOR)

# Generator bounds.
GRID_SIDE_RANGE = (18, 28)
OBJECT_COUNT_RANGE = (3, 5)
SOLID_AREA_RANGE = (6, 20)
MAX_SHAPE_SIDE = 8
MAX_OBJECT_GEN_ATTEMPTS = 120
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("empty mask")
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _interior_holes(mask: np.ndarray) -> list[np.ndarray]:
    holes: list[np.ndarray] = []
    for comp in _components_from_mask(~mask):
        touches_edge = np.any(
            (comp[:, 0] == 0)
            | (comp[:, 1] == 0)
            | (comp[:, 0] == mask.shape[0] - 1)
            | (comp[:, 1] == mask.shape[1] - 1)
        )
        if not bool(touches_edge):
            holes.append(comp)
    return holes


def _solid_area(mask: np.ndarray) -> int:
    holes = _interior_holes(mask)
    return int(mask.sum() + sum(int(len(hole)) for hole in holes))


def _describe_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    objects: list[dict] = []
    for color in MOTION_COLORS:
        for coords in _components_from_mask(g == color):
            r0, c0, r1, c1 = _bbox(coords)
            submask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            submask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            objects.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "visible_area": int(len(coords)),
                    "solid_area": _solid_area(submask),
                    "direction": DIR_BY_COLOR[int(color)],
                }
            )
    return objects


def _render_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each colored object by the rank of its hole-filled area.

    Input structure:
    - The grid has a black background and several separated 4-connected objects.
    - Every object is monochrome, using red, blue, green, or yellow as a direction code.
    - Some objects are hollow and enclose black holes; others are solid.

    Rule:
    - For each object, compute its solid area: visible cells plus any enclosed-hole cells.
    - Rank all objects globally from smallest solid area to largest; all solid areas are distinct.
    - Move each whole object by its rank distance: red right, blue left, green down, yellow up.
    - Clear the canvas to black, then redraw all moved objects with their original shapes and colors.

    Key insight:
    - The displacement splits into two visible clues: color gives the direction, while
      the movement length comes from the object's global rank after mentally filling holes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _describe_objects(g)
    assert objects, "expected at least one motion object"
    solid_areas = [obj["solid_area"] for obj in objects]
    assert len(set(solid_areas)) == len(solid_areas), "solid areas must be unique"

    ranked = sorted(objects, key=lambda obj: obj["solid_area"])
    for rank, obj in enumerate(ranked, start=1):
        obj["rank"] = int(rank)

    out = np.zeros_like(g)
    occupied = np.zeros(g.shape, dtype=bool)
    h, w = g.shape
    for obj in objects:
        dr, dc = obj["direction"]
        rank = obj["rank"]
        shifted = obj["coords"] + np.array([rank * dr, rank * dc], dtype=int)
        assert np.all(shifted[:, 0] >= 0) and np.all(shifted[:, 0] < h)
        assert np.all(shifted[:, 1] >= 0) and np.all(shifted[:, 1] < w)
        assert not bool(np.any(occupied[shifted[:, 0], shifted[:, 1]])), "moved objects must not overlap"
        out[shifted[:, 0], shifted[:, 1]] = obj["color"]
        occupied[shifted[:, 0], shifted[:, 1]] = True

    return out


def _random_orientation(mask: np.ndarray) -> np.ndarray:
    out = mask
    if random.random() < 0.5:
        out = np.fliplr(out)
    turns = random.randint(0, 3)
    if turns:
        out = np.rot90(out, turns)
    return np.array(out, dtype=bool, copy=True)


def _random_polyomino(area: int) -> np.ndarray | None:
    for _ in range(MAX_OBJECT_GEN_ATTEMPTS):
        cells = {(0, 0)}
        frontier = set(_neighbors4(0, 0))
        while len(cells) < area:
            if not frontier:
                break
            choice = random.choice(tuple(frontier))
            frontier.remove(choice)
            cells.add(choice)
            for nbr in _neighbors4(*choice):
                if nbr not in cells:
                    frontier.add(nbr)
        if len(cells) != area:
            continue

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        h = max(rows) - min(rows) + 1
        w = max(cols) - min(cols) + 1
        if max(h, w) > MAX_SHAPE_SIDE:
            continue

        mask = np.zeros((h, w), dtype=bool)
        r_shift = -min(rows)
        c_shift = -min(cols)
        for r, c in cells:
            mask[r + r_shift, c + c_shift] = True
        mask = _random_orientation(mask)
        if len(_components_from_mask(mask)) != 1:
            continue
        if _interior_holes(mask):
            continue
        bbox_area = mask.shape[0] * mask.shape[1]
        if area >= 8 and bbox_area == area:
            continue
        return mask
    return None


def _random_hollow_mask() -> np.ndarray | None:
    for _ in range(MAX_OBJECT_GEN_ATTEMPTS):
        outer_h = random.randint(4, 6)
        outer_w = random.randint(4, 7)
        hole_h = random.randint(1, outer_h - 2)
        hole_w = random.randint(1, outer_w - 2)
        if hole_h * hole_w < 2:
            continue
        hr = random.randint(1, outer_h - hole_h - 1)
        hc = random.randint(1, outer_w - hole_w - 1)

        mask = np.ones((outer_h, outer_w), dtype=bool)
        mask[hr : hr + hole_h, hc : hc + hole_w] = False

        if random.random() < 0.7:
            side = random.choice(("top", "bottom", "left", "right"))
            if side in ("top", "bottom"):
                ext_h = random.randint(1, 2)
                ext_w = random.randint(1, outer_w)
                ext_c = random.randint(0, outer_w - ext_w)
                new_mask = np.zeros((outer_h + ext_h, outer_w), dtype=bool)
                if side == "top":
                    new_mask[ext_h:, :] = mask
                    new_mask[:ext_h, ext_c : ext_c + ext_w] = True
                else:
                    new_mask[:outer_h, :] = mask
                    new_mask[outer_h:, ext_c : ext_c + ext_w] = True
            else:
                ext_w = random.randint(1, 2)
                ext_h = random.randint(1, outer_h)
                ext_r = random.randint(0, outer_h - ext_h)
                new_mask = np.zeros((outer_h, outer_w + ext_w), dtype=bool)
                if side == "left":
                    new_mask[:, ext_w:] = mask
                    new_mask[ext_r : ext_r + ext_h, :ext_w] = True
                else:
                    new_mask[:, :outer_w] = mask
                    new_mask[ext_r : ext_r + ext_h, outer_w:] = True
            mask = new_mask

        mask = _random_orientation(mask)
        mask = _crop_mask(mask)
        holes = _interior_holes(mask)
        if len(holes) != 1:
            continue
        if max(mask.shape) > MAX_SHAPE_SIDE:
            continue
        return mask
    return None


def _sample_colors(count: int) -> list[int]:
    colors = [
        random.choice((RED, BLUE)),
        random.choice((GREEN, YELLOW)),
    ]
    while len(colors) < count:
        colors.append(random.choice(MOTION_COLORS))
    if count >= 4 and random.random() < 0.55:
        colors[random.randrange(count)] = random.choice(colors)
    random.shuffle(colors)
    return colors


def _bbox_free(forbidden: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    return not bool(np.any(forbidden[r0 : r0 + h, c0 : c0 + w]))


def _mark_forbidden(forbidden: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(forbidden.shape[0], r0 + h + 1)
    cc1 = min(forbidden.shape[1], c0 + w + 1)
    forbidden[rr0:rr1, cc0:cc1] = True


def _place_specs(specs: list[dict], shape: tuple[int, int]) -> dict[int, tuple[int, int, int, int]] | None:
    h, w = shape
    out_forbidden = np.zeros(shape, dtype=bool)
    in_forbidden = np.zeros(shape, dtype=bool)
    placements: dict[int, tuple[int, int, int, int]] = {}

    order = sorted(
        range(len(specs)),
        key=lambda idx: specs[idx]["mask"].shape[0] * specs[idx]["mask"].shape[1],
        reverse=True,
    )
    for idx in order:
        spec = specs[idx]
        mask = spec["mask"]
        mh, mw = mask.shape
        dr, dc = spec["direction"]
        rank = spec["rank"]
        candidates: list[tuple[int, int, int, int]] = []
        for out_r in range(h - mh + 1):
            in_r = out_r - rank * dr
            if in_r < 0 or in_r + mh > h:
                continue
            for out_c in range(w - mw + 1):
                in_c = out_c - rank * dc
                if in_c < 0 or in_c + mw > w:
                    continue
                if not _bbox_free(out_forbidden, out_r, out_c, mh, mw):
                    continue
                if not _bbox_free(in_forbidden, in_r, in_c, mh, mw):
                    continue
                candidates.append((in_r, in_c, out_r, out_c))
        if not candidates:
            return None
        placement = random.choice(candidates)
        placements[idx] = placement
        _mark_forbidden(in_forbidden, placement[0], placement[1], mh, mw)
        _mark_forbidden(out_forbidden, placement[2], placement[3], mh, mw)
    return placements


def _render_scene(shape: tuple[int, int], specs: list[dict], placements: dict[int, tuple[int, int, int, int]], *, use_output: bool) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for idx, spec in enumerate(specs):
        r0, c0, out_r, out_c = placements[idx]
        top_r, top_c = (out_r, out_c) if use_output else (r0, c0)
        patch = _render_patch(spec["mask"], spec["color"])
        h, w = patch.shape
        region = grid[top_r : top_r + h, top_c : top_c + w]
        region[patch != 0] = patch[patch != 0]
    return grid


def _generate_specs() -> list[dict] | None:
    count = random.randint(*OBJECT_COUNT_RANGE)
    hollow_count = random.choices([0, 1, 2], weights=[1, 5, 4], k=1)[0]
    hollow_count = min(hollow_count, count - 1)
    hollow_indices = set(random.sample(range(count), hollow_count))
    colors = _sample_colors(count)

    specs: list[dict] = []
    for idx in range(count):
        is_hollow = idx in hollow_indices
        mask = _random_hollow_mask() if is_hollow else _random_polyomino(random.randint(*SOLID_AREA_RANGE))
        if mask is None:
            return None
        visible_area = int(mask.sum())
        solid_area = _solid_area(mask)
        hole_area = int(solid_area - visible_area)
        specs.append(
            {
                "mask": mask,
                "color": int(colors[idx]),
                "direction": DIR_BY_COLOR[int(colors[idx])],
                "visible_area": visible_area,
                "solid_area": solid_area,
                "hole_area": hole_area,
                "is_hollow": bool(is_hollow),
            }
        )

    solid_areas = [spec["solid_area"] for spec in specs]
    if len(set(solid_areas)) != len(solid_areas):
        return None

    visible_areas = [spec["visible_area"] for spec in specs]
    if len(set(visible_areas)) != len(visible_areas):
        return None

    if hollow_count > 0:
        solid_order = tuple(np.argsort(solid_areas))
        visible_order = tuple(np.argsort(visible_areas))
        if solid_order == visible_order:
            return None
        if not any(spec["hole_area"] >= 3 for spec in specs if spec["is_hollow"]):
            return None

    if not any(spec["direction"][0] == 0 for spec in specs):
        return None
    if not any(spec["direction"][1] == 0 for spec in specs):
        return None

    ranked_indices = sorted(range(count), key=lambda idx: specs[idx]["solid_area"])
    for rank, idx in enumerate(ranked_indices, start=1):
        specs[idx]["rank"] = int(rank)
    return specs


def _gen_attempt() -> np.ndarray | None:
    specs = _generate_specs()
    if specs is None:
        return None

    shape = (
        random.randint(*GRID_SIDE_RANGE),
        random.randint(*GRID_SIDE_RANGE),
    )
    placements = _place_specs(specs, shape)
    if placements is None:
        return None

    inp = _render_scene(shape, specs, placements, use_output=False)
    expected = _render_scene(shape, specs, placements, use_output=True)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    described = _describe_objects(inp)
    if len(described) != len(specs):
        return None
    if len({obj["solid_area"] for obj in described}) != len(specs):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
