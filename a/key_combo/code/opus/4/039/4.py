from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
SPRITE_COLORS = (1, 2, 3, 4)
SLOT_NAMES = ("tl", "tr", "bl", "br")
ROTATION_K = {0: 0, 1: 3, 2: 2, 3: 1}  # np.rot90 k values for clockwise 0/90/180/270
GEN_MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_sprites(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                }
            )

    comps.sort(key=lambda comp: comp["bbox"])
    return comps


def _count_holes_in_patch(patch: np.ndarray) -> int:
    zero_mask = patch == BLACK
    h, w = patch.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0

    for r in range(h):
        for c in range(w):
            if not zero_mask[r, c] or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            touches_border = False
            while q:
                rr, cc = q.popleft()
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and zero_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            if not touches_border:
                holes += 1

    return holes


def _rotate_patch_for_holes(patch: np.ndarray, hole_count: int) -> np.ndarray:
    return np.rot90(patch, ROTATION_K[int(hole_count)]).copy()


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for sprite in _find_sprites(out):
        r0, c0, r1, c1 = sprite["bbox"]
        assert r1 - r0 == c1 - c0, "each sprite bbox must be square"
        patch = _crop_bbox(out, sprite["bbox"])
        hole_count = _count_holes_in_patch(patch)
        assert 0 <= hole_count <= 3, "expected 0..3 holes per sprite"
        out[r0 : r1 + 1, c0 : c1 + 1] = _rotate_patch_for_holes(patch, hole_count)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each square sprite in place by an angle encoded by its own hole count.

    Input structure:
    - The grid is black background with 3 or 4 separated single-color sprites.
    - Each sprite is one monochromatic 4-connected component with a square 5x5 to 7x7 bounding box.
    - Inside a sprite bbox, black cells may form 0, 1, 2, or 3 enclosed 4-connected holes.

    Rule:
    - Find every non-black 4-connected sprite.
    - Within each sprite's own bounding box, treat the colored pixels as walls and count the black 4-connected regions that do not touch the bbox border.
    - Rotate that entire square bbox in place according to the hole count: 0 -> no rotation, 1 -> 90 degrees clockwise, 2 -> 180 degrees, 3 -> 270 degrees clockwise.
    - Leave the black background outside all sprite bounding boxes unchanged.

    Key insight:
    - The sprite's topology is the hidden instruction: you must count enclosed holes before you know which geometric rotation to apply.
    """
    return _apply_rule(np.asarray(grid, dtype=int))


def _make_base_patch(size: int, hole_count: int) -> np.ndarray:
    patch = np.zeros((size, size), dtype=int)
    patch[0, :] = 1
    patch[-1, :] = 1
    patch[:, 0] = 1
    patch[:, -1] = 1

    if hole_count == 0:
        gap_side = random.randrange(4)
        gap_pos = random.randint(1, size - 2)
        if gap_side == 0:
            patch[0, gap_pos] = 0
        elif gap_side == 1:
            patch[gap_pos, -1] = 0
        elif gap_side == 2:
            patch[-1, gap_pos] = 0
        else:
            patch[gap_pos, 0] = 0
    elif hole_count == 1:
        patch[1 : size - 2, 1] = 1
        patch[1, 1 : min(size - 1, 3)] = 1
    elif hole_count == 2:
        divider = 2
        patch[:, divider] = 1
        patch[1, 1:divider] = 1
    elif hole_count == 3:
        divider = 2
        patch[:, divider] = 1
        patch[size // 2, : divider + 1] = 1
        patch[1, divider + 1 : size - 1] = 1
    else:
        raise ValueError("hole count must be 0..3")

    return patch


def _random_patch(size: int, hole_count: int) -> np.ndarray:
    patch = _make_base_patch(size, hole_count)
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    if random.random() < 0.5:
        patch = np.flipud(patch)
    patch = np.rot90(patch, random.randrange(4)).copy()

    assert _count_holes_in_patch(patch) == hole_count
    assert len(_find_sprites(patch)) == 1
    if hole_count > 0:
        assert not np.array_equal(patch, _rotate_patch_for_holes(patch, hole_count))
    return patch


def _choose_scene_spec() -> list[dict]:
    sprite_count = 4 if random.random() < 0.65 else 3
    hole_counts = random.sample((0, 1, 2, 3), sprite_count)
    colors = random.sample(SPRITE_COLORS, sprite_count)
    slots = random.sample(SLOT_NAMES, sprite_count)

    specs = []
    for color, hole_count, slot in zip(colors, hole_counts, slots):
        size = random.randint(5, 7)
        patch = _random_patch(size, hole_count)
        specs.append(
            {
                "color": color,
                "holes": hole_count,
                "size": size,
                "slot": slot,
                "input_patch": np.where(patch == 1, color, BLACK),
            }
        )
    return specs


def _slot_band_max(specs: list[dict], slot_group: set[str]) -> int:
    sizes = [spec["size"] for spec in specs if spec["slot"] in slot_group]
    return max(sizes) if sizes else 0


def _distribute_extra(total_extra: int, buckets: list[str]) -> dict[str, int]:
    extra = {name: 0 for name in buckets}
    for _ in range(total_extra):
        extra[random.choice(buckets)] += 1
    return extra


def _place_specs(specs: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    top_max = _slot_band_max(specs, {"tl", "tr"})
    bottom_max = _slot_band_max(specs, {"bl", "br"})
    left_max = _slot_band_max(specs, {"tl", "bl"})
    right_max = _slot_band_max(specs, {"tr", "br"})

    row_gap = random.randint(2, 4)
    col_gap = random.randint(2, 4)
    outer = random.randint(1, 2)

    top_band = top_max + random.randint(0, 3)
    bottom_band = bottom_max + random.randint(0, 3)
    left_band = left_max + random.randint(0, 3)
    right_band = right_max + random.randint(0, 3)

    min_h = outer + top_band + row_gap + bottom_band + outer
    min_w = outer + left_band + col_gap + right_band + outer

    target_h = random.randint(max(18, min_h), min(28, max(18, min_h + 4)))
    target_w = random.randint(max(18, min_w), min(28, max(18, min_w + 4)))

    row_extra = _distribute_extra(target_h - min_h, ["top", "bottom", "gap", "outer_top", "outer_bottom"])
    col_extra = _distribute_extra(target_w - min_w, ["left", "right", "gap", "outer_left", "outer_right"])

    top_band += row_extra["top"]
    bottom_band += row_extra["bottom"]
    row_gap += row_extra["gap"]
    outer_top = outer + row_extra["outer_top"]
    outer_bottom = outer + row_extra["outer_bottom"]

    left_band += col_extra["left"]
    right_band += col_extra["right"]
    col_gap += col_extra["gap"]
    outer_left = outer + col_extra["outer_left"]
    outer_right = outer + col_extra["outer_right"]

    h = outer_top + top_band + row_gap + bottom_band + outer_bottom
    w = outer_left + left_band + col_gap + right_band + outer_right
    assert 18 <= h <= 30 and 18 <= w <= 30

    row_ranges = {
        "t": (outer_top, outer_top + top_band - 1),
        "b": (outer_top + top_band + row_gap, outer_top + top_band + row_gap + bottom_band - 1),
    }
    col_ranges = {
        "l": (outer_left, outer_left + left_band - 1),
        "r": (outer_left + left_band + col_gap, outer_left + left_band + col_gap + right_band - 1),
    }

    grid = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)
    placed_specs: list[dict] = []

    for spec in specs:
        size = spec["size"]
        row_key = "t" if spec["slot"][0] == "t" else "b"
        col_key = "l" if spec["slot"][1] == "l" else "r"
        rr0, rr1 = row_ranges[row_key]
        cc0, cc1 = col_ranges[col_key]
        top = random.randint(rr0, rr1 - size + 1)
        left = random.randint(cc0, cc1 - size + 1)

        bottom = top + size
        right = left + size
        input_patch = spec["input_patch"]
        output_patch = _rotate_patch_for_holes(input_patch, spec["holes"])

        grid[top:bottom, left:right] = input_patch
        expected[top:bottom, left:right] = output_patch

        placed = dict(spec)
        placed["bbox"] = (top, left, bottom - 1, right - 1)
        placed["output_patch"] = output_patch
        placed_specs.append(placed)

    return grid, expected, placed_specs


def _validate_scene(grid: np.ndarray, expected: np.ndarray, specs: list[dict]) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(grid, expected):
        return False

    comps = _find_sprites(grid)
    if len(comps) != len(specs):
        return False

    colors = [comp["color"] for comp in comps]
    if len(set(colors)) != len(colors):
        return False

    spec_by_color = {spec["color"]: spec for spec in specs}
    found_holes = set()
    for comp in comps:
        if comp["color"] not in spec_by_color:
            return False

        spec = spec_by_color[comp["color"]]
        bbox = comp["bbox"]
        size = bbox[2] - bbox[0] + 1
        if size != bbox[3] - bbox[1] + 1 or not (5 <= size <= 7):
            return False
        if bbox != spec["bbox"]:
            return False

        patch = _crop_bbox(grid, bbox)
        holes = _count_holes_in_patch(patch)
        if holes != spec["holes"]:
            return False
        found_holes.add(holes)

        if not np.array_equal(patch, spec["input_patch"]):
            return False
        if not np.array_equal(_rotate_patch_for_holes(patch, holes), spec["output_patch"]):
            return False
        if holes > 0 and np.array_equal(patch, spec["output_patch"]):
            return False

    if len(found_holes) != len(specs):
        return False
    if not np.array_equal(_apply_rule(grid), expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        specs = _choose_scene_spec()
        grid, expected, placed_specs = _place_specs(specs)
        if _validate_scene(grid, expected, placed_specs):
            return grid
    raise RuntimeError("failed to generate a valid hole-count rotation scene")
