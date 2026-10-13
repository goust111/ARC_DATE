"""ARC task: infer invisible mirror axes from paired monochrome sprites."""

from __future__ import annotations

import random
from collections import deque

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (n, 2) arrays of (row, column)."""
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[np.ndarray] = []
    height, width = mask.shape
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue = deque([(sr, sc)])
        cells: list[Coord] = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    result = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    result[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return result


def _overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def main(grid: np.ndarray) -> np.ndarray:
    """Reflect singleton sprites across invisible axes defined by true mirror pairs.

    Input structure:
    - A black grid contains asymmetric, monochrome, 4-connected sprites. Colors
      occurring in two separated components form a side-by-side row-band pair or
      an above-below column-band pair; colors occurring once are singleton sprites.

    Rule:
    - A side-by-side pair defines its integer vertical axis only when its right
      sprite is the left-right reflection of its left sprite. An above-below pair
      analogously defines a horizontal axis. A 180-degree rotation pair defines no
      axis. Assign each singleton to the unique pair whose governed row or column
      interval overlaps it, and add its same-colored reflection only for a true axis.

    Key insight:
    - The paired sprites encode an unseen mirror: mirror pairs copy their governed
      singleton, while visually plausible 180-degree rotation pairs are distractors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    assert np.any(g == 0), "expected black background"

    groups: list[dict] = []
    singletons: list[tuple[int, np.ndarray, BBox]] = []
    for color in (int(x) for x in np.unique(g) if int(x) != 0):
        comps = _components(g == color)
        assert len(comps) in (1, 2), "each color must occur in one sprite or one pair"
        if len(comps) == 1:
            singletons.append((color, comps[0], _bbox(comps[0])))
            continue

        a, b = comps
        ba, bb = _bbox(a), _bbox(b)
        ma, mb = _mask(a), _mask(b)
        if ba[0] == bb[0] and ba[2] == bb[2] and (ba[3] < bb[1] or bb[3] < ba[1]):
            if bb[1] < ba[1]:
                a, b, ba, bb, ma, mb = b, a, bb, ba, mb, ma
            assert ma.shape == mb.shape
            axis2 = ba[1] + bb[3]
            assert axis2 % 2 == 0, "vertical axis must lie on an integer column"
            if np.array_equal(mb, np.fliplr(ma)):
                kind = "mirror"
            else:
                assert np.array_equal(mb, np.rot90(ma, 2)), "pair must be a mirror or 180-degree rotation"
                kind = "rotation"
            groups.append({"orientation": "vertical", "band": (ba[0], ba[2]),
                           "axis": axis2 // 2, "kind": kind})
        elif ba[1] == bb[1] and ba[3] == bb[3] and (ba[2] < bb[0] or bb[2] < ba[0]):
            if bb[0] < ba[0]:
                a, b, ba, bb, ma, mb = b, a, bb, ba, mb, ma
            assert ma.shape == mb.shape
            axis2 = ba[0] + bb[2]
            assert axis2 % 2 == 0, "horizontal axis must lie on an integer row"
            if np.array_equal(mb, np.flipud(ma)):
                kind = "mirror"
            else:
                assert np.array_equal(mb, np.rot90(ma, 2)), "pair must be a mirror or 180-degree rotation"
                kind = "rotation"
            groups.append({"orientation": "horizontal", "band": (ba[1], ba[3]),
                           "axis": axis2 // 2, "kind": kind})
        else:
            raise AssertionError("paired sprites must share exactly one governed band")

    assert groups and singletons, "expected paired groups and singleton sprites"
    out = g.copy()
    height, width = g.shape
    for color, coords, box in singletons:
        candidates = []
        for group in groups:
            lo, hi = group["band"]
            if group["orientation"] == "vertical" and _overlap(box[0], box[2], lo, hi):
                candidates.append(group)
            elif group["orientation"] == "horizontal" and _overlap(box[1], box[3], lo, hi):
                candidates.append(group)
        assert len(candidates) == 1, "each singleton must have one unambiguous governing pair"
        group = candidates[0]
        if group["kind"] == "rotation":
            continue
        if group["orientation"] == "vertical":
            reflected = [(int(r), 2 * int(group["axis"]) - int(c)) for r, c in coords]
        else:
            reflected = [(2 * int(group["axis"]) - int(r), int(c)) for r, c in coords]
        for r, c in reflected:
            assert 0 <= r < height and 0 <= c < width, "reflected sprite must remain in bounds"
            assert int(out[r, c]) in (0, color), "reflected sprite must not cover another object"
            out[r, c] = color
    return out


def _connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask)) == 1


def _asymmetric_mask(height: int, width: int) -> np.ndarray:
    """Sample a connected full-bbox mask with four distinct D2 orientations."""
    for _ in range(200):
        mask = np.random.random((height, width)) < random.uniform(0.45, 0.72)
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if not (3 <= int(mask.sum()) < height * width) or not _connected(mask):
            continue
        variants = (mask, np.fliplr(mask), np.flipud(mask), np.rot90(mask, 2))
        if len({v.tobytes() for v in variants}) == 4:
            return mask
    raise RuntimeError("could not sample an asymmetric sprite")


def _paint(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> np.ndarray:
    coords = np.argwhere(mask) + np.array([r0, c0])
    assert np.all(grid[coords[:, 0], coords[:, 1]] == 0)
    grid[coords[:, 0], coords[:, 1]] = color
    return coords


def _reflect_coords(coords: np.ndarray, orientation: str, axis: int) -> np.ndarray:
    result = coords.copy()
    if orientation == "vertical":
        result[:, 1] = 2 * axis - result[:, 1]
    else:
        result[:, 0] = 2 * axis - result[:, 0]
    return result


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, records: list[dict]) -> bool:
    """Check rule meaning, not merely solver/generator agreement."""
    if len(records) != 3:
        return False
    if {(r["orientation"], r["kind"]) for r in records if r["kind"] == "mirror"} != {
        ("vertical", "mirror"), ("horizontal", "mirror")
    }:
        return False
    if sum(r["kind"] == "rotation" for r in records) != 1:
        return False

    changed = expected != inp
    for record in records:
        for singleton, _color in record["singletons"]:
            target = _reflect_coords(singleton, record["orientation"], record["axis"])
            if np.any(inp[target[:, 0], target[:, 1]] != 0):
                return False
            target_set = {(int(r), int(c)) for r, c in target}
            actual_set = {(int(r), int(c)) for r, c in np.argwhere(changed)}
            if record["kind"] == "mirror" and not target_set <= actual_set:
                return False
            if record["kind"] == "rotation" and target_set & actual_set:
                return False

    # Independently re-check unique governance from interval geometry.
    for record in records:
        for singleton, _color in record["singletons"]:
            box = _bbox(singleton)
            matches = 0
            for other in records:
                lo, hi = other["band"]
                interval = (box[0], box[2]) if other["orientation"] == "vertical" else (box[1], box[3])
                matches += int(_overlap(interval[0], interval[1], lo, hi))
            if matches != 1:
                return False
    return bool(changed.any())


def _gen_attempt() -> np.ndarray | None:
    height, width = random.randint(29, 30), random.randint(29, 30)
    row_offset = random.randint(0, height - 29)
    col_offset = random.randint(0, width - 29)
    zones = [(row_offset + 10 * i, col_offset + 10 * i) for i in range(3)]
    random.shuffle(zones)

    fake_orientation = random.choice(("vertical", "horizontal"))
    specifications = [("vertical", "mirror"), ("horizontal", "mirror"),
                      (fake_orientation, "rotation")]
    singleton_counts = [random.randint(1, 2) for _ in range(3)]
    colors = random.sample(range(1, 10), 3 + sum(singleton_counts))
    color_index = 0
    inp = np.zeros((height, width), dtype=int)
    records: list[dict] = []

    for index, ((orientation, kind), (zr, zc)) in enumerate(zip(specifications, zones)):
        pair_color = colors[color_index]
        color_index += 1
        singleton_colors = colors[color_index:color_index + singleton_counts[index]]
        color_index += singleton_counts[index]
        axis = (zc + 4) if orientation == "vertical" else (zr + 4)
        singleton_items: list[tuple[np.ndarray, int]] = []

        if orientation == "vertical":
            source = _asymmetric_mask(4, 2)
            pair_r = random.randint(zr + 1, zr + 3)
            _paint(inp, source, pair_r, zc, pair_color)
            transform = np.fliplr(source) if kind == "mirror" else np.rot90(source, 2)
            _paint(inp, transform, pair_r, zc + 7, pair_color)
            singleton_c = zc + random.choice((2, 5))
            row_offsets = [random.randint(0, 2)] if len(singleton_colors) == 1 else [0, 2]
            for singleton_color, row_delta in zip(singleton_colors, row_offsets):
                singleton_mask = _asymmetric_mask(2, 2)
                singleton = _paint(inp, singleton_mask, pair_r + row_delta, singleton_c, singleton_color)
                singleton_items.append((singleton, singleton_color))
            band = (pair_r, pair_r + 3)
        else:
            source = _asymmetric_mask(2, 4)
            pair_c = random.randint(zc + 1, zc + 3)
            _paint(inp, source, zr, pair_c, pair_color)
            transform = np.flipud(source) if kind == "mirror" else np.rot90(source, 2)
            _paint(inp, transform, zr + 7, pair_c, pair_color)
            singleton_r = zr + random.choice((2, 5))
            col_offsets = [random.randint(0, 2)] if len(singleton_colors) == 1 else [0, 2]
            for singleton_color, col_delta in zip(singleton_colors, col_offsets):
                singleton_mask = _asymmetric_mask(2, 2)
                singleton = _paint(inp, singleton_mask, singleton_r, pair_c + col_delta, singleton_color)
                singleton_items.append((singleton, singleton_color))
            band = (pair_c, pair_c + 3)

        for singleton, _singleton_color in singleton_items:
            reflected = _reflect_coords(singleton, orientation, axis)
            if np.any(inp[reflected[:, 0], reflected[:, 1]] != 0):
                return None
        records.append({"orientation": orientation, "kind": kind, "axis": axis,
                        "band": band, "singletons": singleton_items})

    expected = inp.copy()
    for record in records:
        if record["kind"] == "mirror":
            for singleton, singleton_color in record["singletons"]:
                reflected = _reflect_coords(singleton, record["orientation"], record["axis"])
                expected[reflected[:, 0], reflected[:, 1]] = singleton_color

    if not _semantic_checks(inp, expected, records):
        return None
    try:
        actual = main(inp)
    except (AssertionError, ValueError):
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    """Generate a diverse, nondegenerate instance covering both mirrors and a fake."""
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid invisible-mirror instance")
