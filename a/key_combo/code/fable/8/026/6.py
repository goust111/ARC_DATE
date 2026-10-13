from __future__ import annotations

import random

import numpy as np


BG = 0
ANCHOR_COLOR = 8
SPRITE_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
SIZE_OPTIONS = (15, 20, 25, 30)
GEN_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def _bbox_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))

    return comps


def _symmetric_bbox(box: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = box
    return h - 1 - r1, w - 1 - c1, h - 1 - r0, w - 1 - c0


def _extract_objects(grid: np.ndarray) -> list[dict]:
    comps = _components_mask(grid != BG)
    objects: list[dict] = []

    for coords in comps:
        box = _bbox_coords(coords)
        area = int(coords.shape[0])
        h = _bbox_h(box)
        w = _bbox_w(box)
        points = {tuple(map(int, rc)) for rc in coords}

        anchor = None
        obj_box = None

        if area == 9 and (h, w) == (3, 3):
            obj_box = box
        elif area == 10 and (h, w) in {(3, 4), (4, 3)}:
            candidates = []
            for r0 in range(box[0], box[2] - 1):
                for c0 in range(box[1], box[3] - 1):
                    cells = {(r, c) for r in range(r0, r0 + 3) for c in range(c0, c0 + 3)}
                    if cells <= points:
                        candidates.append((r0, c0, r0 + 2, c0 + 2))
            assert len(candidates) == 1, "anchored component must contain one solid 3x3 object"
            obj_box = candidates[0]
            obj_cells = {(r, c) for r in range(obj_box[0], obj_box[2] + 1) for c in range(obj_box[1], obj_box[3] + 1)}
            extras = list(points - obj_cells)
            assert len(extras) == 1, "anchored component must have one extra anchor pixel"
            anchor = extras[0]
        else:
            raise AssertionError("unexpected foreground component geometry")

        sprite = _crop_bbox(grid, obj_box)
        assert sprite.shape == (3, 3)
        assert np.all(sprite != BG)
        objects.append({"bbox": obj_box, "anchor": anchor, "sprite": sprite})

    return sorted(objects, key=lambda obj: obj["bbox"])


def _pair_objects(objects: list[dict], shape: tuple[int, int]) -> list[tuple[dict, dict]]:
    by_bbox = {obj["bbox"]: obj for obj in objects}
    used: set[tuple[int, int, int, int]] = set()
    pairs: list[tuple[dict, dict]] = []

    for box in sorted(by_bbox):
        if box in used:
            continue
        partner_box = _symmetric_bbox(box, shape)
        assert partner_box in by_bbox and partner_box != box, "every object must have a distinct symmetric partner"
        pairs.append((by_bbox[box], by_bbox[partner_box]))
        used.add(box)
        used.add(partner_box)

    return pairs


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Restore global point symmetry by rotating each unanchored mismatched sprite in place.

    Input structure:
    - A black background contains several filled 3x3 multi-color sprites.
    - Sprite bboxes come in point-symmetric pairs with respect to the whole-grid center.
    - Some mismatched pairs include exactly one extra singleton anchor pixel 4-neighbor-adjacent
      to one sprite of the pair.

    Rule:
    - For every symmetric pair, compare one sprite with the 180-degree rotation of its partner.
    - If the pair already matches, keep both sprites unchanged.
    - If the pair mismatches, keep the anchored sprite fixed and rotate its partner by 90, 180,
      or 270 degrees in place so the two sprites become 180-degree counterparts; then erase all
      anchor pixels.

    Key insight:
    - The anchor only decides which sprite is fixed; the needed rotation is inferred from the
      global goal that every paired bbox must become point-symmetric as a pattern.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2

    objects = _extract_objects(grid)
    pairs = _pair_objects(objects, grid.shape)

    out = grid.copy()

    for obj in objects:
        if obj["anchor"] is not None:
            out[obj["anchor"]] = BG

    for left, right in pairs:
        left_sprite = left["sprite"]
        right_sprite = right["sprite"]
        if np.array_equal(right_sprite, np.rot90(left_sprite, 2)):
            continue

        anchored = [obj for obj in (left, right) if obj["anchor"] is not None]
        assert len(anchored) == 1, "every mismatched pair must have exactly one anchor"
        anchored_obj = anchored[0]
        moving_obj = right if anchored_obj is left else left

        target = np.rot90(anchored_obj["sprite"], 2)
        matches = [k for k in (1, 2, 3) if np.array_equal(np.rot90(moving_obj["sprite"], k), target)]
        assert len(matches) == 1, "moving sprite must need a unique non-zero quarter-turn"

        repaired = np.rot90(moving_obj["sprite"], matches[0]).copy()
        r0, c0, r1, c1 = moving_obj["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = repaired

    return out


def _distinct_rotations(sprite: np.ndarray) -> bool:
    seen = {np.rot90(sprite, k).tobytes() for k in range(4)}
    return len(seen) == 4


def _random_sprite() -> np.ndarray:
    for _ in range(200):
        n_colors = random.randint(3, 4)
        palette = random.sample(SPRITE_COLORS, n_colors)
        cells = list(palette)
        while len(cells) < 9:
            cells.append(random.choice(palette))
        random.shuffle(cells)
        sprite = np.array(cells, dtype=int).reshape(3, 3)
        if len(set(cells)) >= 3 and _distinct_rotations(sprite):
            return sprite
    raise RuntimeError("failed to build a non-degenerate sprite")


def _candidate_pairs(shape: tuple[int, int]) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    h, w = shape
    rows = list(range(1, h - 2, 5))
    cols = list(range(1, w - 2, 5))
    pairs = []
    for r0 in rows:
        for c0 in cols:
            box = (r0, c0, r0 + 2, c0 + 2)
            partner = _symmetric_bbox(box, shape)
            if box < partner:
                pairs.append((box, partner))
    return pairs


def _anchor_options(box: tuple[int, int, int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = box
    opts = []
    if r0 - 1 >= 0:
        opts.extend((r0 - 1, c) for c in range(c0, c1 + 1))
    if r1 + 1 < h:
        opts.extend((r1 + 1, c) for c in range(c0, c1 + 1))
    if c0 - 1 >= 0:
        opts.extend((r, c0 - 1) for r in range(r0, r1 + 1))
    if c1 + 1 < w:
        opts.extend((r, c1 + 1) for r in range(r0, r1 + 1))
    return opts


def _generate_case() -> np.ndarray:
    shape = (random.choice(SIZE_OPTIONS), random.choice(SIZE_OPTIONS))
    pair_choices = _candidate_pairs(shape)
    max_pairs = min(6, len(pair_choices))
    n_pairs = random.randint(2, max_pairs)
    chosen_pairs = random.sample(pair_choices, n_pairs)

    mismatch_count = random.randint(1, max(1, n_pairs - 1))
    mismatch_indices = set(random.sample(range(n_pairs), mismatch_count))

    grid = np.full(shape, BG, dtype=int)

    for idx, (box_a, box_b) in enumerate(chosen_pairs):
        sprite_a = _random_sprite()
        if idx in mismatch_indices:
            anchor_on_a = random.choice([True, False])
            anchor_box = box_a if anchor_on_a else box_b
            fixed_sprite = sprite_a
            moved_target = np.rot90(fixed_sprite, 2)
            wrong_k = random.choice((1, 2, 3))
            moved_sprite = np.rot90(moved_target, wrong_k).copy()

            if anchor_on_a:
                patch_a, patch_b = fixed_sprite, moved_sprite
            else:
                patch_a, patch_b = moved_sprite, fixed_sprite

            anchor_r, anchor_c = random.choice(_anchor_options(anchor_box, shape))
            grid[anchor_r, anchor_c] = ANCHOR_COLOR
        else:
            patch_a = sprite_a
            patch_b = np.rot90(sprite_a, 2).copy()

        for box, patch in ((box_a, patch_a), (box_b, patch_b)):
            r0, c0, r1, c1 = box
            grid[r0 : r1 + 1, c0 : c1 + 1] = patch

    return grid


def _validate_example(inp: np.ndarray, out: np.ndarray) -> None:
    assert isinstance(inp, np.ndarray) and isinstance(out, np.ndarray)
    assert inp.ndim == 2 and out.ndim == 2
    assert max(inp.shape) <= 30 and max(out.shape) <= 30

    inp_objects = _extract_objects(inp)
    out_objects = _extract_objects(out)
    assert len(inp_objects) == len(out_objects) >= 4

    inp_map = {obj["bbox"]: obj for obj in inp_objects}
    out_map = {obj["bbox"]: obj for obj in out_objects}
    assert set(inp_map) == set(out_map)
    assert all(obj["anchor"] is None for obj in out_objects)

    anchor_cells = [obj["anchor"] for obj in inp_objects if obj["anchor"] is not None]
    assert anchor_cells
    anchor_values = {int(inp[r, c]) for r, c in anchor_cells}
    assert anchor_values == {ANCHOR_COLOR}
    assert not np.any(out == ANCHOR_COLOR)
    for r, c in anchor_cells:
        assert int(out[r, c]) == BG

    mismatch_pairs = 0
    matched_pairs = 0
    for left, right in _pair_objects(inp_objects, inp.shape):
        out_left = out_map[left["bbox"]]
        out_right = out_map[right["bbox"]]

        input_matches = np.array_equal(right["sprite"], np.rot90(left["sprite"], 2))
        anchors = [obj for obj in (left, right) if obj["anchor"] is not None]
        assert np.array_equal(out_right["sprite"], np.rot90(out_left["sprite"], 2))

        if input_matches:
            matched_pairs += 1
            assert len(anchors) == 0
            assert np.array_equal(out_left["sprite"], left["sprite"])
            assert np.array_equal(out_right["sprite"], right["sprite"])
        else:
            mismatch_pairs += 1
            assert len(anchors) == 1
            anchored = anchors[0]
            moving = right if anchored is left else left
            out_anchored = out_map[anchored["bbox"]]
            out_moving = out_map[moving["bbox"]]
            assert np.array_equal(out_anchored["sprite"], anchored["sprite"])
            assert np.array_equal(out_moving["sprite"], np.rot90(anchored["sprite"], 2))
            assert not np.array_equal(out_moving["sprite"], moving["sprite"])

    assert mismatch_pairs >= 1
    assert matched_pairs >= 1


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_case()
        out = main(grid)
        try:
            _validate_example(grid, out)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
