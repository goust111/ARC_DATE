from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
FOREGROUND_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []

    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue

        stack = [(sr, sc)]
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in NEIGHBORS8:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))

        out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _bbox_is_full(mask: np.ndarray) -> bool:
    return bool(
        mask.size
        and mask[0, :].any()
        and mask[-1, :].any()
        and mask[:, 0].any()
        and mask[:, -1].any()
    )


def _is_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask)) == 1


def _extract_objects(g: np.ndarray) -> list[dict[str, object]]:
    colors = set(map(int, np.unique(g))) - {BLACK}
    assert len(colors) == 1, "expected one non-background object color"
    fg = next(iter(colors))
    assert fg != RED, "input object color must differ from the red fill color"

    objects: list[dict[str, object]] = []
    for coords in _components(g == fg):
        box = _bbox(coords)
        mask = _crop(g == fg, box)
        assert _bbox_is_full(mask), "missing pixel must not change the object's bbox"
        objects.append({"bbox": box, "mask": mask})

    assert len(objects) >= 2, "expected multiple damaged copies"
    areas = {int(obj["mask"].sum()) for obj in objects}
    assert len(areas) == 1, "all damaged copies must have the same area"
    return sorted(objects, key=lambda obj: obj["bbox"])  # visible reading order only


def _solve_unique_output(g: np.ndarray, objects: list[dict[str, object]]) -> np.ndarray:
    masks = [np.asarray(obj["mask"], dtype=bool) for obj in objects]
    area = int(masks[0].sum())
    target_area = area + 1
    rotated = [[np.rot90(mask, k) for k in range(4)] for mask in masks]
    dimensions = sorted({rot.shape for obj_rots in rotated for rot in obj_rots})
    unique_outputs: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}

    def add_candidate(union: np.ndarray, picked: list[tuple[int, np.ndarray]]) -> None:
        if int(union.sum()) != target_area:
            return

        holes: list[tuple[int, int]] = []
        for obj, (k, aligned) in zip(objects, picked):
            missing = union & ~aligned
            if int(missing.sum()) != 1:
                return

            original_missing = np.rot90(missing, -k)
            local = np.argwhere(original_missing)
            if local.shape[0] != 1:
                return

            r0, c0, _r1, _c1 = obj["bbox"]
            rr = int(r0) + int(local[0, 0])
            cc = int(c0) + int(local[0, 1])
            if int(g[rr, cc]) != BLACK:
                return
            holes.append((rr, cc))

        out = g.copy()
        for rr, cc in holes:
            out[rr, cc] = RED
        unique_outputs[(out.shape, out.tobytes())] = out

    for dim in dimensions:
        choices: list[list[tuple[int, np.ndarray]]] = []
        for obj_rots in rotated:
            obj_choices = [(k, rot) for k, rot in enumerate(obj_rots) if rot.shape == dim]
            if not obj_choices:
                break
            choices.append(obj_choices)
        if len(choices) != len(objects):
            continue

        def search(i: int, union: np.ndarray, picked: list[tuple[int, np.ndarray]]) -> None:
            if len(unique_outputs) > 1:
                return
            if i == len(choices):
                add_candidate(union, picked)
                return
            for choice in choices[i]:
                new_union = union | choice[1]
                if int(new_union.sum()) <= target_area:
                    search(i + 1, new_union, picked + [choice])

        search(0, np.zeros(dim, dtype=bool), [])
        if len(unique_outputs) > 1:
            break

    assert len(unique_outputs) == 1, "expected exactly one recoverable consensus template"
    return next(iter(unique_outputs.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete every rotated damaged copy from the hidden cross-copy consensus template.

    Input structure:
    - A black background contains several same-color, non-red, 8-connected objects.
    - The objects are copies of one asymmetric template, each rotated by 0/90/180/270 degrees.
    - Every copy is missing exactly one template pixel, and that missing pixel does not shrink its bbox.

    Rule:
    - Try the four rotations of each object and keep only the alignment whose union is one pixel larger than every damaged copy.
    - In that shared alignment, the union is the hidden complete template and each aligned copy differs from it at exactly one pixel.
    - Rotate each missing pixel back into its own object's bbox and color that input position red; keep all other cells unchanged.

    Key insight:
    - The complete template never appears directly; only the cross-copy union after solving the relative rotations reveals each object's missing pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"
    objects = _extract_objects(g)
    return _solve_unique_output(g, objects)


def _random_path_mask(h: int, w: int, target_area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    center = (random.randrange(1, h - 1), random.randrange(1, w - 1))
    anchors = [
        (0, random.randrange(w)),
        (h - 1, random.randrange(w)),
        (random.randrange(h), 0),
        (random.randrange(h), w - 1),
    ]

    for ar, ac in anchors:
        r, c = ar, ac
        mask[r, c] = True
        while (r, c) != center:
            dr = 0 if r == center[0] else (1 if center[0] > r else -1)
            dc = 0 if c == center[1] else (1 if center[1] > c else -1)
            moves: list[tuple[int, int]] = []
            if dr:
                moves.append((dr, 0))
            if dc:
                moves.append((0, dc))
            if dr and dc:
                moves.extend([(dr, dc), (dr, dc)])
            step_r, step_c = random.choice(moves)
            r += step_r
            c += step_c
            mask[r, c] = True

    while int(mask.sum()) < target_area:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for dr, dc in NEIGHBORS8:
                rr = int(r) + dr
                cc = int(c) + dc
                if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                    frontier.add((rr, cc))
        if not frontier:
            break
        rr, cc = random.choice(tuple(frontier))
        mask[rr, cc] = True

    return mask


def _rotationally_asymmetric(mask: np.ndarray) -> bool:
    for k in range(1, 4):
        rot = np.rot90(mask, k)
        if rot.shape == mask.shape and np.array_equal(rot, mask):
            return False
    return True


def _missing_candidates(template: np.ndarray) -> list[tuple[int, int]]:
    h, w = template.shape
    out: list[tuple[int, int]] = []
    for r, c in np.argwhere(template):
        r = int(r)
        c = int(c)
        if r in (0, h - 1) or c in (0, w - 1):
            continue

        damaged = template.copy()
        damaged[r, c] = False
        if not _bbox_is_full(damaged) or not _is_connected(damaged):
            continue

        neighbor_count = 0
        for dr, dc in NEIGHBORS8:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and template[rr, cc]:
                neighbor_count += 1
        if 2 <= neighbor_count < 8:
            out.append((r, c))
    return out


def _make_template(copy_count: int) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    dims = ((4, 5), (5, 4), (5, 5), (5, 6), (6, 5), (4, 6), (6, 4))
    for _ in range(500):
        h, w = random.choice(dims)
        low = max(h + w, int(round(h * w * 0.45)))
        high = min(h * w - 3, int(round(h * w * 0.72)))
        if low > high:
            continue
        target_area = random.randint(low, high)
        template = _random_path_mask(h, w, target_area)
        if int(template.sum()) != target_area:
            continue
        if not _bbox_is_full(template) or not _is_connected(template):
            continue
        if not _rotationally_asymmetric(template):
            continue
        candidates = _missing_candidates(template)
        if len(candidates) >= min(4, copy_count):
            return template, candidates
    return None


def _damaged_copy(
    template: np.ndarray, rotation: int, missing: tuple[int, int]
) -> tuple[np.ndarray, tuple[int, int]]:
    full = np.rot90(template, rotation)
    missing_mask = np.zeros(template.shape, dtype=bool)
    missing_mask[missing] = True
    rotated_missing = np.rot90(missing_mask, rotation)
    local = np.argwhere(rotated_missing)
    assert local.shape[0] == 1

    damaged = full.copy()
    mr = int(local[0, 0])
    mc = int(local[0, 1])
    damaged[mr, mc] = False
    assert _bbox_is_full(damaged) and _is_connected(damaged)
    return damaged, (mr, mc)


def _place_copies(
    items: list[tuple[np.ndarray, tuple[int, int]]], shape: tuple[int, int], color: int
) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    h, w = shape
    grid = np.full((h, w), BLACK, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(items)
    order = list(range(len(items)))
    random.shuffle(order)

    for idx in order:
        patch, _missing = items[idx]
        ph, pw = patch.shape
        locations: list[tuple[int, int]] = []
        for r in range(1, h - ph):
            for c in range(1, w - pw):
                r0 = max(0, r - 1)
                c0 = max(0, c - 1)
                r1 = min(h, r + ph + 1)
                c1 = min(w, c + pw + 1)
                if not reserved[r0:r1, c0:c1].any():
                    locations.append((r, c))
        if not locations:
            return None

        r, c = random.choice(locations)
        positions[idx] = (r, c)
        grid[r : r + ph, c : c + pw][patch] = color
        r0 = max(0, r - 1)
        c0 = max(0, c - 1)
        r1 = min(h, r + ph + 1)
        c1 = min(w, c + pw + 1)
        reserved[r0:r1, c0:c1] = True

    return grid, [pos for pos in positions if pos is not None]


def _gen_attempt() -> np.ndarray | None:
    copy_count = random.randint(4, 6)
    made = _make_template(copy_count)
    if made is None:
        return None
    template, candidates = made

    rotations = [0, 1, 2, 3]
    while len(rotations) < copy_count:
        rotations.append(random.randrange(4))
    random.shuffle(rotations)

    distinct_missing = random.randint(3, min(4, copy_count, len(candidates)))
    missing_choices = random.sample(candidates, distinct_missing)
    while len(missing_choices) < copy_count:
        missing_choices.append(random.choice(candidates))
    random.shuffle(missing_choices)
    if len(set(missing_choices)) < 3:
        return None

    items = [
        _damaged_copy(template, rotation, missing)
        for rotation, missing in zip(rotations, missing_choices)
    ]

    color = random.choice(FOREGROUND_COLORS)
    for _ in range(30):
        height = random.randint(18, 30)
        width = random.randint(18, 30)
        placed = _place_copies(items, (height, width), color)
        if placed is None:
            continue
        inp, positions = placed
        if len(_components(inp == color)) != copy_count:
            continue

        expected = inp.copy()
        for (patch, local_missing), (r, c) in zip(items, positions):
            mr, mc = local_missing
            assert not patch[mr, mc]
            expected[r + mr, c + mc] = RED

        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid consensus-template input")
