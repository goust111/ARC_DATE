from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


BACKGROUND = 0
FILL_COLOR = 2
SPRITE_COLORS = (1, 3, 4, 8, 9)
INDICATOR_COLORS = (2, 6, 7)
SIDES = ("top", "bottom", "left", "right")


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
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
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox(coords)


def _indicator_side(
    sprite_coords: np.ndarray,
    sprite_bbox: tuple[int, int, int, int],
    indicator_coord: tuple[int, int],
) -> str | None:
    r0, c0, r1, c1 = sprite_bbox
    ir, ic = indicator_coord
    coord_set = {tuple(map(int, rc)) for rc in sprite_coords}
    if ir == r0 - 1 and c0 <= ic <= c1 and (r0, ic) in coord_set:
        return "top"
    if ir == r1 + 1 and c0 <= ic <= c1 and (r1, ic) in coord_set:
        return "bottom"
    if ic == c0 - 1 and r0 <= ir <= r1 and (ir, c0) in coord_set:
        return "left"
    if ic == c1 + 1 and r0 <= ir <= r1 and (ir, c1) in coord_set:
        return "right"
    return None


def _extract_entities(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    colors = [int(v) for v in np.unique(g) if int(v) != BACKGROUND]
    components: list[dict[str, object]] = []
    for color in colors:
        for coords in _component_coords(g == color):
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )

    sprites = [comp for comp in components if int(comp["area"]) > 1]
    indicators = [comp for comp in components if int(comp["area"]) == 1]
    if not sprites or not indicators:
        raise ValueError("expected both sprites and indicator pixels")

    used_indicator_ids: set[int] = set()
    entities: list[dict[str, object]] = []
    for sprite in sprites:
        sprite_coords = np.asarray(sprite["coords"], dtype=int)
        sprite_bbox = tuple(map(int, sprite["bbox"]))
        candidates: list[tuple[dict[str, object], str]] = []
        for indicator in indicators:
            coord = tuple(map(int, np.asarray(indicator["coords"], dtype=int)[0]))
            side = _indicator_side(sprite_coords, sprite_bbox, coord)
            if side is not None:
                candidates.append((indicator, side))
        if len(candidates) != 1:
            raise ValueError("each sprite must have exactly one uniquely matched indicator")
        indicator, side = candidates[0]
        indicator_id = id(indicator)
        if indicator_id in used_indicator_ids:
            raise ValueError("indicator matched multiple sprites")
        used_indicator_ids.add(indicator_id)
        r0, c0, r1, c1 = sprite_bbox
        entities.append(
            {
                "sprite_color": int(sprite["color"]),
                "indicator_color": int(indicator["color"]),
                "bbox": sprite_bbox,
                "coords": sprite_coords,
                "side": side,
                "indicator_coord": tuple(map(int, np.asarray(indicator["coords"], dtype=int)[0])),
                "crop": g[r0 : r1 + 1, c0 : c1 + 1].copy(),
            }
        )

    if len(used_indicator_ids) != len(indicators):
        raise ValueError("all indicators must be used exactly once")
    entities.sort(key=lambda item: (int(item["bbox"][0]), int(item["bbox"][1])))
    return entities


def _mirror_placement(
    bbox: tuple[int, int, int, int], side: str
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if side == "top":
        return r0 - h, c0
    if side == "bottom":
        return r1 + 1, c0
    if side == "left":
        return r0, c0 - w
    if side == "right":
        return r0, c1 + 1
    raise ValueError(f"unknown side: {side}")


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each incomplete sprite's bbox holes, then mirror the completed bbox toward its adjacent indicator and recolor the mirrored copy.

    Input structure:
    - A black background contains 2-3 incomplete monochrome 4-connected sprites, each with its own tight bounding box.
    - Every sprite's bbox contains 1-3 black missing cells, while the remaining sprite cells keep one unique sprite color.
    - Exactly one single-pixel indicator of a different color sits just outside one side of each bbox and 4-touches that side.

    Rule:
    - For every sprite, compute the bbox of the sprite pixels only and fill black cells inside that bbox with red.
    - Use the indicator position to choose the mirrored side: top/bottom indicators reflect across a horizontal bbox edge, left/right indicators across a vertical bbox edge.
    - Mirror the completed bbox region, recolor all non-black pixels of that mirrored copy to the indicator color, and place it flush against the indicated side so it covers the indicator pixel.
    - Process sprites in bbox top-left order; later processed content overwrites earlier content on overlaps.

    Key insight:
    - The missing interior cells must be restored before mirroring, because the recovered bbox region is the geometry that gets duplicated and recolored.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    entities = _extract_entities(g)

    for entity in entities:
        r0, c0, r1, c1 = map(int, entity["bbox"])
        crop = np.asarray(entity["crop"], dtype=int)
        completed = np.where(crop == BACKGROUND, FILL_COLOR, crop)
        _paste(out, completed, r0, c0)

        if entity["side"] in ("top", "bottom"):
            mirrored = np.flipud(completed)
        else:
            mirrored = np.fliplr(completed)
        mirrored = mirrored.copy()
        mirrored[mirrored != BACKGROUND] = int(entity["indicator_color"])

        mr0, mc0 = _mirror_placement((r0, c0, r1, c1), str(entity["side"]))
        mh, mw = mirrored.shape
        if mr0 < 0 or mc0 < 0 or mr0 + mh > out.shape[0] or mc0 + mw > out.shape[1]:
            raise ValueError("mirrored copy would leave the canvas")
        _paste(out, mirrored, mr0, mc0)

    return out


def _is_connected(mask: np.ndarray) -> bool:
    comps = _component_coords(mask)
    return len(comps) == 1


def _make_sprite_mask(h: int, w: int, hole_count: int, rng: np.random.Generator) -> np.ndarray:
    all_cells = np.array([(r, c) for r in range(h) for c in range(w)], dtype=int)
    for _ in range(300):
        chosen = rng.choice(len(all_cells), size=hole_count, replace=False)
        mask = np.ones((h, w), dtype=bool)
        for idx in chosen:
            r, c = map(int, all_cells[idx])
            mask[r, c] = False
        remaining = int(mask.sum())
        if not (4 <= remaining <= 8):
            continue
        if not _is_connected(mask):
            continue
        if _mask_bbox(mask) != (0, 0, h - 1, w - 1):
            continue
        return mask
    raise RuntimeError("failed to build a valid incomplete sprite")


def _side_anchor_candidates(mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    if side == "top":
        return [(0, c) for c in range(w) if mask[0, c]]
    if side == "bottom":
        return [(h - 1, c) for c in range(w) if mask[h - 1, c]]
    if side == "left":
        return [(r, 0) for r in range(h) if mask[r, 0]]
    if side == "right":
        return [(r, w - 1) for r in range(h) if mask[r, w - 1]]
    raise ValueError(f"unknown side: {side}")


def _zone_shape(h: int, w: int, side: str) -> tuple[int, int]:
    if side in ("top", "bottom"):
        return 2 * h, w
    return h, 2 * w


def _origin_from_zone(zone_r: int, zone_c: int, h: int, w: int, side: str) -> tuple[int, int]:
    if side == "top":
        return zone_r + h, zone_c
    if side == "bottom":
        return zone_r, zone_c
    if side == "left":
        return zone_r, zone_c + w
    if side == "right":
        return zone_r, zone_c
    raise ValueError(f"unknown side: {side}")


def _mirror_bbox_from_origin(r0: int, c0: int, h: int, w: int, side: str) -> tuple[int, int, int, int]:
    if side == "top":
        return r0 - h, c0, r0 - 1, c0 + w - 1
    if side == "bottom":
        return r0 + h, c0, r0 + 2 * h - 1, c0 + w - 1
    if side == "left":
        return r0, c0 - w, r0 + h - 1, c0 - 1
    if side == "right":
        return r0, c0 + w, r0 + h - 1, c0 + 2 * w - 1
    raise ValueError(f"unknown side: {side}")


def _expanded_slices(
    r0: int, c0: int, h: int, w: int, pad: int, canvas_shape: tuple[int, int]
) -> tuple[slice, slice]:
    hh, ww = canvas_shape
    return (
        slice(max(0, r0 - pad), min(hh, r0 + h + pad)),
        slice(max(0, c0 - pad), min(ww, c0 + w + pad)),
    )


def _build_candidate_specs(rng: np.random.Generator, num_objects: int) -> list[dict[str, object]]:
    side_choices = list(rng.choice(SIDES, size=num_objects, replace=False))
    sprite_colors = list(rng.choice(SPRITE_COLORS, size=num_objects, replace=False))
    indicator_colors = list(rng.choice(INDICATOR_COLORS, size=num_objects, replace=False))
    dims = [(2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (2, 5), (5, 2)]
    specs: list[dict[str, object]] = []
    for idx in range(num_objects):
        h, w = dims[int(rng.integers(len(dims)))]
        legal_holes = [k for k in (1, 2, 3) if 4 <= h * w - k <= 8]
        hole_count = int(rng.choice(legal_holes))
        mask = _make_sprite_mask(h, w, hole_count, rng)
        side = str(side_choices[idx])
        anchors = _side_anchor_candidates(mask, side)
        local_anchor = anchors[int(rng.integers(len(anchors)))]
        zone_h, zone_w = _zone_shape(h, w, side)
        specs.append(
            {
                "h": h,
                "w": w,
                "mask": mask,
                "hole_count": hole_count,
                "side": side,
                "sprite_color": int(sprite_colors[idx]),
                "indicator_color": int(indicator_colors[idx]),
                "local_anchor": local_anchor,
                "zone_h": zone_h,
                "zone_w": zone_w,
            }
        )
    specs.sort(key=lambda spec: (-(int(spec["zone_h"]) * int(spec["zone_w"])), -int(spec["h"]) - int(spec["w"])))
    return specs


def _place_specs(specs: list[dict[str, object]], h: int, w: int, rng: np.random.Generator) -> bool:
    occupied = np.zeros((h, w), dtype=bool)
    for spec in specs:
        zone_h = int(spec["zone_h"])
        zone_w = int(spec["zone_w"])
        candidates = [(r, c) for r in range(h - zone_h + 1) for c in range(w - zone_w + 1)]
        rng.shuffle(candidates)
        placed = False
        for zone_r, zone_c in candidates:
            rr, cc = _expanded_slices(zone_r, zone_c, zone_h, zone_w, 1, (h, w))
            if occupied[rr, cc].any():
                continue
            occupied[zone_r : zone_r + zone_h, zone_c : zone_c + zone_w] = True
            r0, c0 = _origin_from_zone(zone_r, zone_c, int(spec["h"]), int(spec["w"]), str(spec["side"]))
            spec["origin"] = (r0, c0)
            placed = True
            break
        if not placed:
            return False
    return True


def _render_input(specs: list[dict[str, object]], h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for spec in specs:
        r0, c0 = map(int, spec["origin"])
        mask = np.asarray(spec["mask"], dtype=bool)
        sprite_color = int(spec["sprite_color"])
        for rr, cc in np.argwhere(mask):
            grid[r0 + int(rr), c0 + int(cc)] = sprite_color

        ar, ac = map(int, spec["local_anchor"])
        if spec["side"] == "top":
            indicator = (r0 - 1, c0 + ac)
        elif spec["side"] == "bottom":
            indicator = (r0 + ar + 1, c0 + ac)
        elif spec["side"] == "left":
            indicator = (r0 + ar, c0 - 1)
        else:
            indicator = (r0 + ar, c0 + ac + 1)
        grid[indicator] = int(spec["indicator_color"])
    return grid


def _validate_generated_example(grid: np.ndarray) -> None:
    entities = _extract_entities(grid)
    if len(entities) not in (2, 3):
        raise AssertionError("generator must create 2-3 sprites")
    sprite_colors = [int(entity["sprite_color"]) for entity in entities]
    indicator_colors = [int(entity["indicator_color"]) for entity in entities]
    if len(set(sprite_colors)) != len(sprite_colors):
        raise AssertionError("sprite colors must be distinct")
    if len(set(indicator_colors)) != len(indicator_colors):
        raise AssertionError("indicator colors must be distinct")

    seen_mirror = np.zeros(grid.shape, dtype=bool)
    out = main(grid)
    for entity in entities:
        r0, c0, r1, c1 = map(int, entity["bbox"])
        crop = np.asarray(entity["crop"], dtype=int)
        zero_count = int((crop == BACKGROUND).sum())
        if zero_count < 1 or zero_count > 3:
            raise AssertionError("each sprite bbox must contain 1-3 holes")
        if int((crop != BACKGROUND).sum()) < 4 or int((crop != BACKGROUND).sum()) > 8:
            raise AssertionError("each sprite must contain 4-8 visible pixels")

        completed = np.where(crop == BACKGROUND, FILL_COLOR, crop)
        if not np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], completed):
            raise AssertionError("completed original bbox does not match the rule")

        mr0, mc0 = _mirror_placement((r0, c0, r1, c1), str(entity["side"]))
        mh, mw = completed.shape
        mirror_region = out[mr0 : mr0 + mh, mc0 : mc0 + mw]
        expected_mirror = np.full((mh, mw), int(entity["indicator_color"]), dtype=int)
        if not np.array_equal(mirror_region, expected_mirror):
            raise AssertionError("mirrored copy does not match the recolored completion")

        overlap = seen_mirror[mr0 : mr0 + mh, mc0 : mc0 + mw]
        if overlap.any():
            raise AssertionError("generator should keep mirrored zones separated for clarity")
        seen_mirror[mr0 : mr0 + mh, mc0 : mc0 + mw] = True

    if np.array_equal(grid, out):
        raise AssertionError("input and output must differ")


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        h = int(rng.integers(14, 19))
        w = int(rng.integers(14, 19))
        num_objects = int(rng.integers(2, 4))
        specs = _build_candidate_specs(rng, num_objects)
        if not _place_specs(specs, h, w, rng):
            continue
        grid = _render_input(specs, h, w)
        try:
            _validate_generated_example(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input after 400 attempts")


def _run_self_checks() -> None:
    top_case = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 6, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )
    top_expected = np.array(
        [
            [0, 6, 6, 6, 0, 0],
            [0, 6, 6, 6, 0, 0],
            [0, 1, 2, 1, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )
    assert np.array_equal(main(top_case), top_expected)

    bottom_case = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 3, 3, 3, 0],
            [0, 0, 3, 0, 3, 0],
            [0, 0, 7, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )
    bottom_expected = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 3, 3, 3, 0],
            [0, 0, 3, 2, 3, 0],
            [0, 0, 7, 7, 7, 0],
            [0, 0, 7, 7, 7, 0],
        ],
        dtype=int,
    )
    assert np.array_equal(main(bottom_case), bottom_expected)

    left_case = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 6, 4, 4, 0],
            [0, 0, 0, 0, 4, 0],
            [0, 0, 0, 4, 4, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )
    left_expected = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 6, 6, 4, 4, 0],
            [0, 6, 6, 2, 4, 0],
            [0, 6, 6, 4, 4, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )
    assert np.array_equal(main(left_case), left_expected)

    right_case = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 8, 8, 0, 0, 0],
            [0, 8, 8, 6, 0, 0],
            [0, 8, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )
    right_expected = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 8, 8, 6, 6, 0],
            [0, 8, 8, 6, 6, 0],
            [0, 8, 2, 6, 6, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )
    assert np.array_equal(main(right_case), right_expected)

    overlap_case = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 6, 0, 3, 3, 3, 0],
            [0, 1, 1, 1, 0, 7, 3, 0, 3, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )
    overlap_expected = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 2, 7, 7, 7, 3, 3, 3, 0],
            [0, 1, 1, 7, 7, 7, 3, 2, 3, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )
    assert np.array_equal(main(overlap_case), overlap_expected)

    for _ in range(25):
        _validate_generated_example(generate_input())


if __name__ == "__main__":
    _run_self_checks()
