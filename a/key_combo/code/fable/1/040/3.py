from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GRAY = 5
MAX_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(np.array(coords, dtype=int))
    return components


def _background_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _compress_runs(mask: np.ndarray) -> np.ndarray:
    row_keep = [0]
    for r in range(1, mask.shape[0]):
        if not np.array_equal(mask[r], mask[r - 1]):
            row_keep.append(r)
    compressed = mask[row_keep, :]

    col_keep = [0]
    for c in range(1, compressed.shape[1]):
        if not np.array_equal(compressed[:, c], compressed[:, c - 1]):
            col_keep.append(c)
    return compressed[:, col_keep]


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask)


def _signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return _mask_key(_compress_runs(mask))


def _rotation_signatures(mask: np.ndarray) -> set[tuple[tuple[int, ...], ...]]:
    return {_signature(np.rot90(mask, turns)) for turns in range(4)}


def _dihedral_keys(mask: np.ndarray) -> set[tuple[tuple[int, ...], ...]]:
    keys = set()
    for turns in range(4):
        rotated = np.rot90(mask, turns)
        keys.add(_mask_key(rotated))
        keys.add(_mask_key(np.fliplr(rotated)))
    return keys


def _adjacent_duplicates(mask: np.ndarray) -> bool:
    for r in range(1, mask.shape[0]):
        if np.array_equal(mask[r], mask[r - 1]):
            return True
    for c in range(1, mask.shape[1]):
        if np.array_equal(mask[:, c], mask[:, c - 1]):
            return True
    return False


def _mirror(mask: np.ndarray, axis: str) -> np.ndarray:
    return np.fliplr(mask) if axis == "h" else np.flipud(mask)


def _stretch_mask(mask: np.ndarray, row_factors: list[int], col_factors: list[int]) -> np.ndarray:
    return np.repeat(np.repeat(mask, row_factors, axis=0), col_factors, axis=1)


def _sample_factors(length: int, active: bool, max_dim: int = 8) -> list[int]:
    if not active:
        return [1] * length
    for _ in range(100):
        factors = [1] * length
        picks = random.sample(range(length), random.randint(1, min(2, length)))
        for idx in picks:
            factors[idx] = random.choice((2, 3)) if length <= 4 else 2
        if sum(factors) <= max_dim:
            return factors
    raise RuntimeError("failed to sample valid stretch factors")


def _random_connected_mask() -> np.ndarray:
    for _ in range(300):
        h = random.randint(4, 6)
        w = random.randint(4, 6)
        target = random.randint(max(6, (h * w) // 3), min(h * w - 2, (2 * h * w) // 3))

        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target:
            continue

        mask = np.zeros((h, w), dtype=bool)
        coords = np.array(sorted(cells), dtype=int)
        mask[coords[:, 0], coords[:, 1]] = True
        mask = _component_mask(np.argwhere(mask))

        if min(mask.shape) < 3 or max(mask.shape) < 4:
            continue
        if _adjacent_duplicates(mask):
            continue
        density = float(mask.mean())
        if not (0.28 <= density <= 0.72):
            continue
        if len(_dihedral_keys(mask)) != 8:
            continue
        if not _rotation_signatures(mask).isdisjoint(_rotation_signatures(np.fliplr(mask))):
            continue
        return mask
    raise RuntimeError("failed to generate a chiral prototype")


def _build_candidate(proto_mask: np.ndarray, mode: str) -> np.ndarray:
    rotated = np.rot90(proto_mask, random.randrange(4))
    row_factors = _sample_factors(rotated.shape[0], mode in {"row", "both"})
    col_factors = _sample_factors(rotated.shape[1], mode in {"col", "both"})
    return _stretch_mask(rotated, row_factors, col_factors)


def _place_objects(height: int, width: int, objects: list[dict]) -> bool:
    placed_boxes: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(objects)), key=lambda i: objects[i]["mask"].size, reverse=True)
    for idx in order:
        mask = objects[idx]["mask"]
        h, w = mask.shape
        r_lo = 1
        c_lo = 1
        r_hi = height - h - 1
        c_hi = width - w - 1
        if r_hi < r_lo or c_hi < c_lo:
            return False
        positions = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            r1 = r0 + h - 1
            c1 = c0 + w - 1
            if any(
                not (r1 < pr0 - 1 or r0 > pr1 + 1 or c1 < pc0 - 1 or c0 > pc1 + 1)
                for pr0, pc0, pr1, pc1 in placed_boxes
            ):
                continue
            objects[idx]["pos"] = (r0, c0)
            placed_boxes.append((r0, c0, r1, c1))
            placed = True
            break
        if not placed:
            return False
    return True


def _render_scene(height: int, width: int, objects: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), BLACK, dtype=int)
    out = inp.copy()
    for obj in objects:
        r0, c0 = obj["pos"]
        mask = obj["mask"]
        h, w = mask.shape
        inp[r0 : r0 + h, c0 : c0 + w][mask] = obj["color"]
        target = obj["target_color"] if obj["kind"] == "candidate" else obj["color"]
        out[r0 : r0 + h, c0 : c0 + w][mask] = target
    return inp, out


def _validate_scene(inp: np.ndarray, expected: np.ndarray) -> None:
    bg = _background_color(inp)
    assert bg == BLACK

    fg_colors = [int(c) for c in np.unique(inp) if int(c) != bg]
    assert set(fg_colors) == {BLUE, RED, GRAY}

    comps_by_color = {color: _components_mask(inp == color) for color in fg_colors}
    prototype_colors = [color for color, comps in comps_by_color.items() if len(comps) == 1]
    candidate_colors = [color for color, comps in comps_by_color.items() if len(comps) > 1]
    assert len(prototype_colors) == 2
    assert candidate_colors == [GRAY]

    proto_rotations = {
        color: _rotation_signatures(_component_mask(comps_by_color[color][0]))
        for color in prototype_colors
    }
    assert proto_rotations[prototype_colors[0]].isdisjoint(proto_rotations[prototype_colors[1]])

    match_counts = {color: 0 for color in prototype_colors}
    for coords in comps_by_color[GRAY]:
        sig = _signature(_component_mask(coords))
        hits = [color for color, sigs in proto_rotations.items() if sig in sigs]
        assert len(hits) == 1
        match_color = hits[0]
        match_counts[match_color] += 1
        assert np.all(expected[coords[:, 0], coords[:, 1]] == match_color)

    assert all(count > 0 for count in match_counts.values())
    for color in prototype_colors:
        coords = comps_by_color[color][0]
        assert np.all(expected[coords[:, 0], coords[:, 1]] == color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray candidate by the color of the chirally matching prototype.

    Input structure:
    - A black background contains exactly three foreground colors.
    - Two colors each form one prototype object, and those two prototypes are mirror images without any self symmetry.
    - The third color forms several candidate objects obtained from one prototype by rotation and row/column run stretching.

    Rule:
    - Split the grid into monochrome 4-connected objects and crop each object to its bounding box.
    - Collapse adjacent identical rows and adjacent identical columns inside every cropped object.
    - For each gray candidate, compare its compressed mask with the two prototype compressed masks using rotations only, never reflections.
    - Recolor the whole candidate with the unique prototype color whose rotation class matches.

    Key insight:
    - Run compression removes the geometric stretching, but chirality remains; because the two prototypes are mirror partners, only rotation-only matching can distinguish them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    fg_colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert len(fg_colors) == 3

    comps_by_color = {color: _components_mask(g == color) for color in fg_colors}
    prototype_colors = [color for color, comps in comps_by_color.items() if len(comps) == 1]
    candidate_colors = [color for color, comps in comps_by_color.items() if len(comps) > 1]
    assert len(prototype_colors) == 2 and len(candidate_colors) == 1

    proto_rotations = {
        color: _rotation_signatures(_component_mask(comps_by_color[color][0]))
        for color in prototype_colors
    }

    out = g.copy()
    candidate_color = candidate_colors[0]
    for coords in comps_by_color[candidate_color]:
        sig = _signature(_component_mask(coords))
        hits = [color for color, sigs in proto_rotations.items() if sig in sigs]
        assert len(hits) == 1
        out[coords[:, 0], coords[:, 1]] = hits[0]
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        base = _random_connected_mask()
        red_proto = np.rot90(base, random.randrange(4))
        blue_proto = _mirror(red_proto, random.choice(("h", "v")))

        if not _rotation_signatures(red_proto).isdisjoint(_rotation_signatures(blue_proto)):
            continue

        candidate_count = random.randint(4, 7)
        candidate_labels = [RED, BLUE] + [random.choice((RED, BLUE)) for _ in range(candidate_count - 2)]
        random.shuffle(candidate_labels)

        modes = ["none", "row", "col", "both"]
        extra_modes = [random.choice(modes) for _ in range(candidate_count - len(modes))]
        candidate_modes = modes + extra_modes
        random.shuffle(candidate_modes)

        objects = [
            {"kind": "prototype", "mask": red_proto, "color": RED, "target_color": RED},
            {"kind": "prototype", "mask": blue_proto, "color": BLUE, "target_color": BLUE},
        ]

        ok = True
        for label, mode in zip(candidate_labels, candidate_modes):
            proto = red_proto if label == RED else blue_proto
            candidate_mask = _build_candidate(proto, mode)
            if max(candidate_mask.shape) > 8:
                ok = False
                break
            objects.append(
                {
                    "kind": "candidate",
                    "mask": candidate_mask,
                    "color": GRAY,
                    "target_color": label,
                }
            )
        if not ok:
            continue

        total_fg = sum(int(obj["mask"].sum()) for obj in objects)
        total_boxes = sum((obj["mask"].shape[0] + 2) * (obj["mask"].shape[1] + 2) for obj in objects)
        side_lo = max(
            16,
            max(max(obj["mask"].shape) + 2 for obj in objects),
            int(np.ceil(np.sqrt(total_fg * 2.2))),
            int(np.ceil(np.sqrt(total_boxes))),
        )
        if side_lo > 30:
            continue

        for _ in range(20):
            height = random.randint(side_lo, min(30, side_lo + 5))
            width = random.randint(side_lo, min(30, side_lo + 5))
            scene_objects = [dict(obj) for obj in objects]
            if not _place_objects(height, width, scene_objects):
                continue
            inp, expected = _render_scene(height, width, scene_objects)
            try:
                _validate_scene(inp, expected)
            except AssertionError:
                continue
            if not np.array_equal(main(inp), expected):
                continue
            return inp
    raise RuntimeError("failed to generate valid chirality-classification input")
