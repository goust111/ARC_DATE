from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg


Pixel = tuple[int, int, int]


def _array_key(arr: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    arr = np.asarray(arr, dtype=int)
    return tuple(arr.shape), tuple(int(x) for x in arr.ravel())


def _rotate_coord(r: int, c: int, pivot2: tuple[int, int], turns: int) -> tuple[int, int]:
    turns %= 4
    if turns == 0:
        return int(r), int(c)

    pr2, pc2 = map(int, pivot2)
    r = int(r)
    c = int(c)

    if turns == 1:
        nr_num = pr2 + pc2 - 2 * c
        nc_num = pc2 - pr2 + 2 * r
        assert nr_num % 2 == 0 and nc_num % 2 == 0
        return nr_num // 2, nc_num // 2

    if turns == 2:
        return pr2 - r, pc2 - c

    nr_num = pr2 - pc2 + 2 * c
    nc_num = pr2 + pc2 - 2 * r
    assert nr_num % 2 == 0 and nc_num % 2 == 0
    return nr_num // 2, nc_num // 2


def _rotate_pixels(
    pixels: tuple[Pixel, ...], pivot2: tuple[int, int], turns: int
) -> tuple[Pixel, ...]:
    rotated = []
    for r, c, color in pixels:
        nr, nc = _rotate_coord(r, c, pivot2, turns)
        rotated.append((int(nr), int(nc), int(color)))
    assert len({(r, c) for r, c, _color in rotated}) == len(rotated)
    return tuple(sorted(rotated))


def _extract_objects(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = components_mask(g != bg, connectivity=8)

    objects = []
    for coords in comps:
        b = bbox_coords(coords)
        r0, c0, _r1, _c1 = b
        crop = crop_bbox(g, b)
        pixels = tuple(
            sorted((int(r), int(c), int(g[int(r), int(c)])) for r, c in coords.tolist())
        )
        objects.append(
            {
                "bbox": b,
                "crop": crop,
                "crop_keys": {
                    turns: _array_key(np.rot90(crop, turns))
                    for turns in range(4)
                },
                "pixels": pixels,
                "n": len(pixels),
                "sum_r": sum(r for r, _c, _color in pixels),
                "sum_c": sum(c for _r, c, _color in pixels),
                "rel_pixels": tuple(
                    sorted((r - r0, c - c0, color) for r, c, color in pixels)
                ),
            }
        )

    objects.sort(key=lambda obj: obj["bbox"])
    return bg, objects


def _infer_pivot2(a: dict, b: dict, turns: int) -> tuple[int, int] | None:
    n = int(a["n"])
    if n != int(b["n"]):
        return None

    sr0, sc0 = int(a["sum_r"]), int(a["sum_c"])
    sr1, sc1 = int(b["sum_r"]), int(b["sum_c"])

    if turns == 1:
        pr2_num = sr0 + sr1 + sc0 - sc1
        pc2_num = -sr0 + sr1 + sc0 + sc1
    elif turns == 3:
        pr2_num = sr0 + sr1 - sc0 + sc1
        pc2_num = sr0 + sc0 - sr1 + sc1
    else:
        raise ValueError("only quarter-turn relations are used here")

    if pr2_num % n != 0 or pc2_num % n != 0:
        return None

    pivot2 = (pr2_num // n, pc2_num // n)
    if (pivot2[0] ^ pivot2[1]) & 1:
        return None
    return pivot2


def _pair_relations(objects: list[dict]) -> list[dict]:
    relations = []
    for i, obj_a in enumerate(objects):
        for j, obj_b in enumerate(objects):
            if i == j or obj_a["n"] != obj_b["n"]:
                continue
            for turns in (1, 3):
                if obj_a["crop_keys"][turns] != obj_b["crop_keys"][0]:
                    continue
                pivot2 = _infer_pivot2(obj_a, obj_b, turns)
                if pivot2 is None:
                    continue
                if _rotate_pixels(obj_a["pixels"], pivot2, turns) != obj_b["pixels"]:
                    continue
                relations.append(
                    {
                        "src": i,
                        "dst": j,
                        "turns": turns,
                        "pivot2": pivot2,
                    }
                )
    return relations


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, objects = _extract_objects(g)
    assert objects, "expected at least one sprite"

    relations = _pair_relations(objects)
    assert relations, "expected at least one valid quarter-turn relation"

    pivot_counts = Counter(tuple(rel["pivot2"]) for rel in relations)
    best_support = max(pivot_counts.values())
    best_pivots = [pivot2 for pivot2, count in pivot_counts.items() if count == best_support]
    assert len(best_pivots) == 1, "expected a unique common pivot"
    pivot2 = best_pivots[0]

    orbits: dict[tuple, dict] = {}
    for obj in objects:
        variants = tuple(_rotate_pixels(obj["pixels"], pivot2, turns) for turns in range(4))
        assert len(set(variants)) == 4, "orbit members must be four distinct sprites"
        key = tuple(sorted(variants))
        entry = orbits.setdefault(
            key,
            {
                "variants": tuple(sorted(variants)),
                "present": set(),
            },
        )
        entry["present"].add(obj["pixels"])

    incomplete = [entry for entry in orbits.values() if len(entry["present"]) == 3]
    assert len(incomplete) == 1, "expected exactly one incomplete orbit"
    for entry in orbits.values():
        assert len(entry["present"]) in {3, 4}

    missing_variants = [
        variant
        for variant in incomplete[0]["variants"]
        if variant not in incomplete[0]["present"]
    ]
    assert len(missing_variants) == 1

    return {
        "bg": bg,
        "objects": objects,
        "relations": relations,
        "pivot2": pivot2,
        "pivot_counts": pivot_counts,
        "best_support": best_support,
        "orbits": orbits,
        "missing_pixels": missing_variants[0],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the missing quarter-turn copy of the unique sprite orbit that has only three visible members.

    Input structure:
    - A black background contains several small multicolor sprites that do not touch.
    - Each sprite belongs to a four-member orbit obtained by rotating one sprite around the same hidden pivot by 90 degrees at a time.
    - Exactly one orbit is incomplete and shows only three of its four rotated copies.

    Rule:
    - Group sprites by exact pixelwise 90-degree rotational equivalence around a single common pivot, even when that pivot lies on a half-cell.
    - Infer the pivot from matching quarter-turn sprite pairs, then enumerate the four rotated placements for every orbit.
    - Find the unique orbit with one missing member and paint that missing rotated sprite into the grid.

    Key insight:
    - The pivot is never drawn, so the task is solved by recovering a shared quarter-turn center from visible sprite pairs and using that invisible center to complete the one damaged orbit.
    """
    g = np.asarray(grid, dtype=int)
    analysis = _analyze(g)

    out = g.copy()
    for r, c, color in analysis["missing_pixels"]:
        assert out[int(r), int(c)] == analysis["bg"]
        out[int(r), int(c)] = int(color)
    return out


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, mask.shape)
    cells = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            if (
                (r > 0 and mask[r - 1, c])
                or (r + 1 < h and mask[r + 1, c])
                or (c > 0 and mask[r, c - 1])
                or (c + 1 < w and mask[r, c + 1])
            ):
                cells.append((r, c))
    return cells


def _grow_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seed = (random.randrange(h), random.randrange(w))
    mask[seed] = True
    while int(mask.sum()) < area:
        mask[random.choice(_frontier(mask))] = True
    return mask


def _canonical_patch_key(patch: np.ndarray) -> tuple:
    return min(_array_key(np.rot90(patch, turns)) for turns in range(4))


def _generate_patch(used_keys: set[tuple]) -> np.ndarray:
    for _ in range(400):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        area_hi = min(h * w, 7)
        area_lo = min(area_hi, max(3, (h * w) // 2))
        area = random.randint(area_lo, area_hi)
        mask = _grow_mask(h, w, area)
        b = bbox_coords(np.argwhere(mask))
        mask = crop_bbox(mask.astype(int), b).astype(bool)
        h, w = map(int, mask.shape)

        patch = np.zeros((h, w), dtype=int)
        cell_count = int(mask.sum())
        palette_size = random.randint(2, min(4, cell_count))
        palette = random.sample(range(1, 10), palette_size)
        coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        random.shuffle(coords)
        for idx, (r, c) in enumerate(coords):
            if idx < palette_size:
                color = palette[idx]
            else:
                color = random.choice(palette)
            patch[r, c] = int(color)

        rot_keys = [_array_key(np.rot90(patch, turns)) for turns in range(4)]
        if len(set(rot_keys)) != 4:
            continue

        key = _canonical_patch_key(patch)
        if key in used_keys:
            continue
        return patch

    raise RuntimeError("failed to generate a distinct asymmetric patch")


def _coords_only(pixels: tuple[Pixel, ...]) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c, _color in pixels}


def _expand_forbidden(forbidden: set[tuple[int, int]], pixels: tuple[Pixel, ...]) -> None:
    for r, c, _color in pixels:
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                forbidden.add((rr + dr, cc + dc))


def _fits_canvas(pixels: tuple[Pixel, ...], shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    return all(0 <= r < h and 0 <= c < w for r, c, _color in pixels)


def _make_base_pixels(patch: np.ndarray, r0: int, c0: int) -> tuple[Pixel, ...]:
    pixels = []
    for r in range(patch.shape[0]):
        for c in range(patch.shape[1]):
            color = int(patch[r, c])
            if color != 0:
                pixels.append((int(r0 + r), int(c0 + c), color))
    return tuple(sorted(pixels))


def _place_group(
    shape: tuple[int, int],
    pivot2: tuple[int, int],
    patch: np.ndarray,
    forbidden: set[tuple[int, int]],
) -> list[tuple[Pixel, ...]] | None:
    h, w = map(int, shape)
    ph, pw = map(int, patch.shape)

    for _ in range(600):
        r0 = random.randint(0, h - ph)
        c0 = random.randint(0, w - pw)
        base = _make_base_pixels(patch, r0, c0)
        if any(2 * r == pivot2[0] and 2 * c == pivot2[1] for r, c, _color in base):
            continue
        members = [_rotate_pixels(base, pivot2, turns) for turns in range(4)]
        if len(set(members)) != 4:
            continue
        if not all(_fits_canvas(member, shape) for member in members):
            continue

        local_forbidden = set(forbidden)
        ok = True
        for member in members:
            coords = _coords_only(member)
            if len(coords) != len(member):
                ok = False
                break
            if any((r, c) in local_forbidden for r, c in coords):
                ok = False
                break
            _expand_forbidden(local_forbidden, member)
        if not ok:
            continue

        return members
    return None


def _paint(grid: np.ndarray, pixels: tuple[Pixel, ...]) -> None:
    for r, c, color in pixels:
        grid[int(r), int(c)] = int(color)


def _semantic_ok(inp: np.ndarray, expected: np.ndarray, group_count: int) -> bool:
    try:
        analysis = _analyze(inp)
    except AssertionError:
        return False

    object_count = len(analysis["objects"])
    if object_count != 4 * group_count - 1:
        return False

    counts = sorted(analysis["pivot_counts"].values(), reverse=True)
    if not counts or counts[0] < 8:
        return False
    if len(counts) > 1 and counts[0] == counts[1]:
        return False

    if not np.array_equal(main(inp), expected):
        return False

    out_objects = components_mask(expected != int(get_bg(expected)), connectivity=8)
    if len(out_objects) != 4 * group_count:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(500):
        h = random.randint(20, 28)
        w = random.randint(20, 28)
        parity = random.choice([0, 1])
        pr2 = 2 * random.randint(5, h - 6) + parity
        pc2 = 2 * random.randint(5, w - 6) + parity
        pivot2 = (int(pr2), int(pc2))

        group_count = random.choice([2, 3])
        used_keys: set[tuple] = set()
        forbidden: set[tuple[int, int]] = set()
        groups: list[list[tuple[Pixel, ...]]] = []

        ok = True
        for _group_idx in range(group_count):
            placed = None
            for _patch_try in range(120):
                patch = _generate_patch(used_keys)
                placed = _place_group((h, w), pivot2, patch, forbidden)
                if placed is not None:
                    used_keys.add(_canonical_patch_key(patch))
                    break
            if placed is None:
                ok = False
                break
            for member in placed:
                _expand_forbidden(forbidden, member)
            groups.append(placed)

        if not ok:
            continue

        missing_group_idx = random.randrange(group_count)
        missing_member_idx = random.randrange(4)

        inp = np.zeros((h, w), dtype=int)
        expected = np.zeros((h, w), dtype=int)

        for group_idx, members in enumerate(groups):
            for member_idx, member in enumerate(members):
                _paint(expected, member)
                if group_idx == missing_group_idx and member_idx == missing_member_idx:
                    continue
                _paint(inp, member)

        if np.array_equal(inp, expected):
            continue
        if not _semantic_ok(inp, expected, group_count):
            continue
        return inp

    raise RuntimeError("failed to generate a valid invisible-pivot orbit input")
