from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _background(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r0, c0 in np.argwhere(mask):
        r0, c0 = int(r0), int(c0)
        if seen[r0, c0]:
            continue
        stack = [(r0, c0)]
        seen[r0, c0] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_rectangular_frame(grid: np.ndarray, coords: np.ndarray, color: int) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    perimeter = 2 * h + 2 * w - 4
    if int(coords.shape[0]) != perimeter:
        return None
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.all(sub[border] == color):
        return None
    if np.any(sub[~border] == color):
        return None
    return r0, c0, r1, c1


def _find_fenced_slots(grid: np.ndarray, bg: int) -> list[dict]:
    slots: list[dict] = []
    for color in set(map(int, grid.ravel())) - {int(bg)}:
        for coords in _components(grid == color):
            box = _is_rectangular_frame(grid, coords, color)
            if box is None:
                continue
            r0, c0, r1, c1 = box
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            seeds = np.argwhere((interior != bg) & (interior != color))
            if int(seeds.shape[0]) != 1:
                continue
            sr, sc = map(int, seeds[0])
            seed_color = int(interior[sr, sc])
            slots.append(
                {
                    "bbox": box,
                    "seed_color": seed_color,
                    "seed_abs": (r0 + 1 + sr, c0 + 1 + sc),
                    "seed_rel": (sr, sc),
                    "cavity_shape": (r1 - r0 - 1, c1 - c0 - 1),
                }
            )
    slots.sort(key=lambda s: (s["bbox"][0], s["bbox"][1]))
    return slots


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _matching_rotations(mask: np.ndarray, cavity_shape: tuple[int, int], seed_rel: tuple[int, int]) -> list[np.ndarray]:
    target = np.ones(cavity_shape, dtype=bool)
    target[seed_rel] = False
    matches: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        if rotated.shape == target.shape and np.array_equal(rotated, target):
            matches.append(rotated)
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Place each colored outside fragment into the matching seeded rectangular fence after rotation.

    Input structure:
    - The background is the most common color.
    - Several one-pixel-thick closed rectangular fences share one fence color.
    - Each fence interior contains exactly one single-pixel seed, and seed colors are distinct.
    - Outside the fences, each seed color also forms one connected fragment whose area is the matching cavity area minus the seed cell.

    Rule:
    - For every fence, read the seed color and find the outside fragment of that same color.
    - Rotate that fragment by 0, 90, 180, or 270 degrees so its missing cell coincides with the seed and all other cavity cells are covered.
    - Erase the original outside fragment and draw the rotated fragment into the fenced cavity, leaving the fence and seed in place.

    Key insight: The seed color is the match key, and the seed position is the anchor that selects the fragment rotation by complementing the cavity.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background(g)
    slots = _find_fenced_slots(g, bg)
    assert slots, "expected at least one seeded rectangular fence"

    out = g.copy()
    used_fragments: set[tuple[int, tuple[tuple[int, int], ...]]] = set()

    for slot in slots:
        color = int(slot["seed_color"])
        cavity_h, cavity_w = map(int, slot["cavity_shape"])
        required_area = cavity_h * cavity_w - 1
        seed_abs = tuple(map(int, slot["seed_abs"]))
        seed_rel = tuple(map(int, slot["seed_rel"]))

        candidates = []
        for coords in _components(g == color):
            if int(coords.shape[0]) != required_area:
                continue
            coord_set = {tuple(map(int, p)) for p in coords}
            if seed_abs in coord_set:
                continue
            mask = _mask_from_coords(coords)
            matches = _matching_rotations(mask, (cavity_h, cavity_w), seed_rel)
            if matches:
                candidates.append((coords, matches))

        assert len(candidates) == 1, "each seed color must identify one fitting outside fragment"
        frag_coords, rotations = candidates[0]
        assert len(rotations) == 1, "fragment rotation must be unique"
        key = (color, tuple(sorted(tuple(map(int, p)) for p in frag_coords)))
        assert key not in used_fragments, "fragment reused"
        used_fragments.add(key)

        out[frag_coords[:, 0], frag_coords[:, 1]] = bg
        r0, c0, _r1, _c1 = slot["bbox"]
        interior = out[r0 + 1 : r0 + 1 + cavity_h, c0 + 1 : c0 + 1 + cavity_w]
        interior[rotations[0]] = color

    return out


def _target_mask(cavity_shape: tuple[int, int], seed_rel: tuple[int, int]) -> np.ndarray:
    mask = np.ones(cavity_shape, dtype=bool)
    mask[seed_rel] = False
    return mask


def _sample_slot(rng: random.Random) -> dict | None:
    for _ in range(80):
        h = rng.randint(2, 5)
        w = rng.randint(2, 5)
        seed_rel = (rng.randrange(h), rng.randrange(w))
        target = _target_mask((h, w), seed_rel)
        rot = rng.randrange(4)
        fragment = np.rot90(target, rot)
        if len(_matching_rotations(fragment, (h, w), seed_rel)) == 1:
            return {
                "cavity_shape": (h, w),
                "seed_rel": seed_rel,
                "target_mask": target,
                "fragment_mask": fragment,
                "rotation": rot,
            }
    return None


def _frame_patch(spec: dict, bg: int, fence_color: int, seed_color: int) -> np.ndarray:
    h, w = spec["cavity_shape"]
    sr, sc = spec["seed_rel"]
    patch = np.full((h + 2, w + 2), bg, dtype=int)
    patch[0, :] = fence_color
    patch[-1, :] = fence_color
    patch[:, 0] = fence_color
    patch[:, -1] = fence_color
    patch[sr + 1, sc + 1] = seed_color
    return patch


def _fragment_patch(spec: dict, bg: int, seed_color: int) -> np.ndarray:
    mask = spec["fragment_mask"]
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = seed_color
    return patch


def _place_patch(
    canvas: np.ndarray,
    reserved: np.ndarray,
    patch: np.ndarray,
    bg: int,
    rng: random.Random,
    padding: int = 1,
) -> tuple[int, int] | None:
    h, w = canvas.shape
    ph, pw = patch.shape
    if ph > h or pw > w:
        return None
    rows = list(range(1, h - ph)) if h - ph >= 2 else list(range(0, h - ph + 1))
    cols = list(range(1, w - pw)) if w - pw >= 2 else list(range(0, w - pw + 1))
    positions = [(r, c) for r in rows for c in cols]
    rng.shuffle(positions)
    paint = patch != bg
    for r, c in positions:
        rr0, rr1 = max(0, r - padding), min(h, r + ph + padding)
        cc0, cc1 = max(0, c - padding), min(w, c + pw + padding)
        if np.any(reserved[rr0:rr1, cc0:cc1]):
            continue
        view = canvas[r : r + ph, c : c + pw]
        if np.any(view[paint] != bg):
            continue
        view[paint] = patch[paint]
        reserved[rr0:rr1, cc0:cc1] = True
        return r, c
    return None


def _expected_output(inp: np.ndarray, specs: list[dict], bg: int) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        coords = spec["fragment_coords"]
        out[coords[:, 0], coords[:, 1]] = bg
    for spec in specs:
        fr, fc = spec["frame_top"]
        h, w = spec["cavity_shape"]
        target = spec["target_mask"]
        color = spec["seed_color"]
        interior = out[fr + 1 : fr + 1 + h, fc + 1 : fc + 1 + w]
        interior[target] = color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_slots = rng.randint(2, 4)
    min_dim = 16 + 3 * (n_slots - 2)
    h = rng.randint(min_dim, 30)
    w = rng.randint(min_dim, 30)

    colors = list(range(10))
    rng.shuffle(colors)
    bg = colors.pop()
    fence_color = colors.pop()
    seed_colors = [colors.pop() for _ in range(n_slots)]

    specs: list[dict] = []
    for seed_color in seed_colors:
        spec = _sample_slot(rng)
        if spec is None:
            return None
        spec["seed_color"] = seed_color
        specs.append(spec)

    canvas = np.full((h, w), bg, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    frame_order = specs[:]
    rng.shuffle(frame_order)
    for spec in frame_order:
        patch = _frame_patch(spec, bg, fence_color, spec["seed_color"])
        top_left = _place_patch(canvas, reserved, patch, bg, rng)
        if top_left is None:
            return None
        spec["frame_top"] = top_left

    fragment_order = specs[:]
    rng.shuffle(fragment_order)
    fragment_order.sort(key=lambda s: -int(s["fragment_mask"].size))
    for spec in fragment_order:
        patch = _fragment_patch(spec, bg, spec["seed_color"])
        top_left = _place_patch(canvas, reserved, patch, bg, rng)
        if top_left is None:
            return None
        tr, tc = top_left
        coords = np.argwhere(spec["fragment_mask"])
        coords[:, 0] += tr
        coords[:, 1] += tc
        spec["fragment_coords"] = coords

    if _background(canvas) != bg:
        return None

    expected = _expected_output(canvas, specs, bg)
    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seeded-fragment fence input")
