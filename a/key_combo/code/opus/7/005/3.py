from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, crop_bbox, find_components, get_bg, label_mask


GREY = Color.GREY
SPRITE_COLORS = (Color.BLUE, Color.RED, Color.GREEN)
ORANGE = Color.ORANGE
DIRS = ("up", "down", "left", "right")

GRID_MIN = 13
GRID_MAX = 19
GAP_MIN = 4
GAP_MAX = 7
MAX_GENERATION_ATTEMPTS = 500


SHAPE_LIBRARY = (
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
)


def _has_solid_2x2(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for r in range(h - 1):
        for c in range(w - 1):
            if bool(mask[r : r + 2, c : c + 2].all()):
                return True
    return False


def _shape_spec(coords: tuple[tuple[int, int], ...]) -> dict:
    arr = np.array(coords, dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    arr = arr - np.array([r0, c0], dtype=int)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    mask = np.zeros((h, w), dtype=bool)
    mask[arr[:, 0], arr[:, 1]] = True
    assert not _has_solid_2x2(mask)
    return {
        "coords": tuple(sorted((int(r), int(c)) for r, c in arr)),
        "mask": mask,
        "h": h,
        "w": w,
        "area": int(mask.sum()),
        "top_cols": tuple(int(c) for c in np.where(mask[0])[0]),
        "bottom_cols": tuple(int(c) for c in np.where(mask[-1])[0]),
        "left_rows": tuple(int(r) for r in np.where(mask[:, 0])[0]),
        "right_rows": tuple(int(r) for r in np.where(mask[:, -1])[0]),
    }


SHAPES = tuple(_shape_spec(coords) for coords in SHAPE_LIBRARY)


def _signature_from_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    patch[rel[:, 0], rel[:, 1]] = True
    return tuple(sorted((int(r), int(c)) for r, c in np.argwhere(patch)))


def _coords_from_top_left(top: tuple[int, int], rel_coords: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    r0, c0 = map(int, top)
    return [(r0 + int(r), c0 + int(c)) for r, c in rel_coords]


def _component_mask_from_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    arr = np.array(coords, dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = arr - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _is_connected(coords: list[tuple[int, int]]) -> bool:
    mask = _component_mask_from_coords(coords)
    _labels, count = label_mask(mask, connectivity=4)
    return count == 1


def _extract_sprites(grid: np.ndarray) -> list[dict]:
    sprites = find_components(
        grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        colors=SPRITE_COLORS,
        return_format="dict",
    )
    out = []
    for comp in sprites:
        coords = np.array(comp["coords"], dtype=int)
        sig = _signature_from_coords(coords)
        r0, c0, r1, c1 = map(int, comp["bbox"])
        out.append(
            {
                "color": int(comp["color"]),
                "coords": tuple((int(r), int(c)) for r, c in coords),
                "bbox": (r0, c0, r1, c1),
                "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
                "signature": sig,
            }
        )
    return out


def _parse_anchor_component(comp: dict) -> dict:
    coords = [tuple(map(int, rc)) for rc in comp["coords"]]
    coord_set = set(coords)
    r0, c0, r1, c1 = map(int, comp["bbox"])
    candidates = []
    for ar in range(r0, r1):
        for ac in range(c0, c1):
            anchor_cells = {(ar, ac), (ar + 1, ac), (ar, ac + 1), (ar + 1, ac + 1)}
            if not anchor_cells.issubset(coord_set):
                continue
            remaining = [rc for rc in coords if rc not in anchor_cells]
            if len(remaining) < 3 or len(remaining) > 6:
                continue
            if not _is_connected(remaining):
                continue
            rem_set = set(remaining)
            side_hits = []
            if all(r < ar for r, _c in remaining) and any((ar - 1, ac + dc) in rem_set for dc in (0, 1)):
                side_hits.append("up")
            if all(r > ar + 1 for r, _c in remaining) and any((ar + 2, ac + dc) in rem_set for dc in (0, 1)):
                side_hits.append("down")
            if all(c < ac for _r, c in remaining) and any((ar + dr, ac - 1) in rem_set for dr in (0, 1)):
                side_hits.append("left")
            if all(c > ac + 1 for _r, c in remaining) and any((ar + dr, ac + 2) in rem_set for dr in (0, 1)):
                side_hits.append("right")
            if len(side_hits) != 1:
                continue
            signature = _signature_from_coords(np.array(remaining, dtype=int))
            candidates.append(
                {
                    "anchor_bbox": (ar, ac, ar + 1, ac + 1),
                    "anchor_center": (ar + 0.5, ac + 0.5),
                    "indicator_side": side_hits[0],
                    "indicator_coords": tuple(sorted(remaining)),
                    "indicator_signature": signature,
                    "all_grey_coords": tuple(sorted(coords)),
                }
            )
    assert len(candidates) == 1, "each grey component must contain one parseable 2x2 anchor"
    return candidates[0]


def _extract_anchors(grid: np.ndarray) -> list[dict]:
    comps = find_components(
        grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        colors=(GREY,),
        return_format="dict",
    )
    return [_parse_anchor_component(comp) for comp in comps]


def _movement_direction(sprite: dict, anchor: dict) -> str:
    sr, sc = sprite["center"]
    ar, ac = anchor["anchor_center"]
    dr = ar - sr
    dc = ac - sc
    assert abs(dr) != abs(dc), "anchor direction must be uniquely horizontal or vertical"
    if abs(dc) > abs(dr):
        return "right" if dc > 0 else "left"
    return "down" if dr > 0 else "up"


def _slide_sprite(sprite: dict, anchor: dict, direction: str) -> tuple[list[tuple[int, int]], str]:
    coords = list(sprite["coords"])
    ar, ac, _ar1, _ac1 = anchor["anchor_bbox"]
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    if direction == "right":
        dc = ac - 1 - max(cols)
        moved = [(r, c + dc) for r, c in coords]
        return moved, "left"
    if direction == "left":
        dc = ac + 2 - min(cols)
        moved = [(r, c + dc) for r, c in coords]
        return moved, "right"
    if direction == "down":
        dr = ar - 1 - max(rows)
        moved = [(r + dr, c) for r, c in coords]
        return moved, "top"
    dr = ar + 2 - min(rows)
    moved = [(r + dr, c) for r, c in coords]
    return moved, "bottom"


def _mirror_across_anchor(moved: list[tuple[int, int]], anchor: dict, contact_edge: str) -> list[tuple[int, int]]:
    ar0, ac0, ar1, ac1 = anchor["anchor_bbox"]
    mirrored = []
    for r, c in moved:
        if contact_edge == "left":
            mirrored.append((r, 2 * ac0 - 1 - c))
        elif contact_edge == "right":
            mirrored.append((r, 2 * ac1 + 1 - c))
        elif contact_edge == "top":
            mirrored.append((2 * ar0 - 1 - r, c))
        else:
            mirrored.append((2 * ar1 + 1 - r, c))
    return mirrored


def _touches_anchor_edge(moved: list[tuple[int, int]], anchor: dict, contact_edge: str) -> bool:
    ar0, ac0, ar1, ac1 = anchor["anchor_bbox"]
    moved_set = set(moved)
    if contact_edge == "left":
        return any((r, ac0 - 1) in moved_set for r in (ar0, ar1))
    if contact_edge == "right":
        return any((r, ac1 + 1) in moved_set for r in (ar0, ar1))
    if contact_edge == "top":
        return any((ar0 - 1, c) in moved_set for c in (ac0, ac1))
    return any((ar1 + 1, c) in moved_set for c in (ac0, ac1))


def _all_in_bounds(shape: tuple[int, int], coords: list[tuple[int, int]]) -> bool:
    h, w = map(int, shape)
    return all(0 <= r < h and 0 <= c < w for r, c in coords)


def _render_output(shape: tuple[int, int], assignments: list[tuple[dict, dict]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    occupied: dict[tuple[int, int], int] = {}
    for sprite, anchor in assignments:
        direction = _movement_direction(sprite, anchor)
        moved, edge = _slide_sprite(sprite, anchor, direction)
        mirrored = _mirror_across_anchor(moved, anchor, edge)
        assert _touches_anchor_edge(moved, anchor, edge)
        assert _all_in_bounds(shape, moved)
        assert _all_in_bounds(shape, mirrored)
        assert not (set(moved) & set(mirrored))
        for r, c in moved:
            assert (r, c) not in occupied
            occupied[(r, c)] = int(sprite["color"])
            out[r, c] = int(sprite["color"])
        for r, c in mirrored:
            assert (r, c) not in occupied
            occupied[(r, c)] = ORANGE
            out[r, c] = ORANGE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored sprite to the anchor whose gray side-shape has the same normalized mask, then slide and mirror it across the contact edge.

    Input structure:
    - The background is black.
    - There are 2 or 3 colored 4-connected sprites in blue, red, and/or green; each sprite has 3 to 6 cells and a distinct normalized bbox mask.
    - There are equally many gray structures, each formed by one solid 2x2 anchor square plus one side-adjacent gray indicator shape whose normalized bbox mask matches exactly one sprite.

    Rule:
    - Compute the normalized bbox mask of every colored sprite and every gray indicator shape.
    - Pair each sprite with the unique anchor whose indicator mask is identical.
    - For each paired sprite, determine the unique dominant cardinal direction from the sprite center to the anchor center, slide the sprite along that axis until it becomes 4-neighbor adjacent to the anchor's facing edge, then reflect the slid sprite across that contact edge.
    - Remove the original sprites, anchors, and indicators; keep only the slid sprite in its original color and the reflected copy in orange on a black background.

    Key insight:
    - The gray side-shapes are not decoration: their normalized masks are the only reliable key that tells which anchor each sprite should approach, and that pairing fixes both the slide target and the mirror axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == Color.BLACK

    sprites = _extract_sprites(g)
    anchors = _extract_anchors(g)
    assert 2 <= len(sprites) <= 3
    assert len(sprites) == len(anchors)

    sprite_by_sig = {sprite["signature"]: sprite for sprite in sprites}
    assert len(sprite_by_sig) == len(sprites), "sprite normalized masks must be unique"

    assignments = []
    seen_sigs = set()
    for anchor in anchors:
        sig = anchor["indicator_signature"]
        assert sig in sprite_by_sig, "every indicator must match a sprite"
        assert sig not in seen_sigs, "indicator matches must be one-to-one"
        seen_sigs.add(sig)
        assignments.append((sprite_by_sig[sig], anchor))

    return _render_output(g.shape, assignments)


def _indicator_positions(shape: dict, anchor_top_left: tuple[int, int], side: str) -> list[tuple[int, int]]:
    ar, ac = map(int, anchor_top_left)
    h = int(shape["h"])
    w = int(shape["w"])
    positions = []
    if side == "up":
        for shift in range(-w + 1, 2):
            ir = ar - h
            ic = ac + shift
            coords = _coords_from_top_left((ir, ic), shape["coords"])
            if any(r == ar - 1 and ac <= c <= ac + 1 for r, c in coords):
                positions.append((ir, ic))
    elif side == "down":
        for shift in range(-w + 1, 2):
            ir = ar + 2
            ic = ac + shift
            coords = _coords_from_top_left((ir, ic), shape["coords"])
            if any(r == ar + 2 and ac <= c <= ac + 1 for r, c in coords):
                positions.append((ir, ic))
    elif side == "left":
        for shift in range(-h + 1, 2):
            ir = ar + shift
            ic = ac - w
            coords = _coords_from_top_left((ir, ic), shape["coords"])
            if any(c == ac - 1 and ar <= r <= ar + 1 for r, c in coords):
                positions.append((ir, ic))
    else:
        for shift in range(-h + 1, 2):
            ir = ar + shift
            ic = ac + 2
            coords = _coords_from_top_left((ir, ic), shape["coords"])
            if any(c == ac + 2 and ar <= r <= ar + 1 for r, c in coords):
                positions.append((ir, ic))
    return positions


def _build_pair_candidate(grid_shape: tuple[int, int], shape: dict, direction: str, indicator_side: str, gap: int) -> dict | None:
    h, w = map(int, grid_shape)
    ah = aw = 2
    for _ in range(60):
        ar = random.randint(1, h - ah - 1)
        ac = random.randint(1, w - aw - 1)
        anchor = {"anchor_bbox": (ar, ac, ar + 1, ac + 1), "anchor_center": (ar + 0.5, ac + 0.5)}
        indicator_tops = _indicator_positions(shape, (ar, ac), indicator_side)
        random.shuffle(indicator_tops)
        if not indicator_tops:
            continue

        if direction == "right":
            contact_rows = shape["right_rows"]
            final_c = ac - shape["w"]
            mirror_max_c = ac + shape["w"] - 1
            if final_c < 0 or mirror_max_c >= w:
                continue
            row_pairs = [(edge_row, target_row) for edge_row in contact_rows for target_row in (ar, ar + 1)]
            random.shuffle(row_pairs)
            for edge_row, target_row in row_pairs:
                final_r = target_row - edge_row
                input_c = final_c - gap
                input_r = final_r
                if input_c < 0 or input_r < 0 or input_r + shape["h"] > h:
                    continue
                break
            else:
                continue
        elif direction == "left":
            contact_rows = shape["left_rows"]
            final_c = ac + 2
            mirror_min_c = ac - shape["w"]
            if final_c + shape["w"] > w or mirror_min_c < 0:
                continue
            row_pairs = [(edge_row, target_row) for edge_row in contact_rows for target_row in (ar, ar + 1)]
            random.shuffle(row_pairs)
            for edge_row, target_row in row_pairs:
                final_r = target_row - edge_row
                input_c = final_c + gap
                input_r = final_r
                if input_c + shape["w"] > w or input_r < 0 or input_r + shape["h"] > h:
                    continue
                break
            else:
                continue
        elif direction == "down":
            contact_cols = shape["bottom_cols"]
            final_r = ar - shape["h"]
            mirror_max_r = ar + shape["h"] - 1
            if final_r < 0 or mirror_max_r >= h:
                continue
            col_pairs = [(edge_col, target_col) for edge_col in contact_cols for target_col in (ac, ac + 1)]
            random.shuffle(col_pairs)
            for edge_col, target_col in col_pairs:
                final_c = target_col - edge_col
                input_r = final_r - gap
                input_c = final_c
                if input_r < 0 or input_c < 0 or input_c + shape["w"] > w:
                    continue
                break
            else:
                continue
        else:
            contact_cols = shape["top_cols"]
            final_r = ar + 2
            mirror_min_r = ar - shape["h"]
            if final_r + shape["h"] > h or mirror_min_r < 0:
                continue
            col_pairs = [(edge_col, target_col) for edge_col in contact_cols for target_col in (ac, ac + 1)]
            random.shuffle(col_pairs)
            for edge_col, target_col in col_pairs:
                final_c = target_col - edge_col
                input_r = final_r + gap
                input_c = final_c
                if input_r + shape["h"] > h or input_c < 0 or input_c + shape["w"] > w:
                    continue
                break
            else:
                continue

        sprite_top = (input_r, input_c)
        final_top = (final_r, final_c)
        sprite_coords = _coords_from_top_left(sprite_top, shape["coords"])
        moved_coords = _coords_from_top_left(final_top, shape["coords"])
        ar0, ac0, ar1, ac1 = anchor["anchor_bbox"]
        grey_anchor = [(ar0, ac0), (ar0, ac1), (ar1, ac0), (ar1, ac1)]
        indicator_top = indicator_tops[0]
        indicator_coords = _coords_from_top_left(indicator_top, shape["coords"])
        grey_coords = grey_anchor + indicator_coords

        sprite_comp = {
            "coords": tuple(sprite_coords),
            "bbox": bbox_coords(np.array(sprite_coords, dtype=int)),
            "center": (
                (min(r for r, _c in sprite_coords) + max(r for r, _c in sprite_coords)) / 2.0,
                (min(c for _r, c in sprite_coords) + max(c for _r, c in sprite_coords)) / 2.0,
            ),
            "signature": shape["coords"],
        }
        full_anchor = {
            **anchor,
            "indicator_side": indicator_side,
            "indicator_coords": tuple(indicator_coords),
            "indicator_signature": shape["coords"],
            "all_grey_coords": tuple(grey_coords),
        }
        actual_direction = _movement_direction(sprite_comp, full_anchor)
        if actual_direction != direction:
            continue
        moved, edge = _slide_sprite(sprite_comp, full_anchor, direction)
        mirrored = _mirror_across_anchor(moved, full_anchor, edge)
        if set(grey_coords) & set(sprite_coords):
            continue
        if set(grey_coords) & set(moved):
            continue
        if not _all_in_bounds(grid_shape, sprite_coords + indicator_coords + grey_anchor):
            continue
        if not _all_in_bounds(grid_shape, moved):
            continue
        if not _all_in_bounds(grid_shape, mirrored):
            continue
        return {
            "shape": shape,
            "direction": direction,
            "indicator_side": indicator_side,
            "gap": gap,
            "sprite_top": sprite_top,
            "indicator_top": indicator_top,
            "anchor_top_left": (ar, ac),
            "sprite_coords": sprite_coords,
            "moved_coords": moved,
            "mirrored_coords": mirrored,
            "grey_coords": grey_coords,
        }
    return None


def _coords_with_padding(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...], shape: tuple[int, int], pad: int) -> set[tuple[int, int]]:
    h, w = map(int, shape)
    out = set()
    for r, c in coords:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _sweep_coords(sprite_coords: list[tuple[int, int]], moved_coords: list[tuple[int, int]]) -> set[tuple[int, int]]:
    if sprite_coords == moved_coords:
        return set(sprite_coords)
    if all(r0 == r1 for (r0, _c0), (r1, _c1) in zip(sprite_coords, moved_coords)):
        delta = moved_coords[0][1] - sprite_coords[0][1]
        step = 1 if delta > 0 else -1
        out = set()
        for offset in range(0, delta + step, step):
            for r, c in sprite_coords:
                out.add((r, c + offset))
        return out
    delta = moved_coords[0][0] - sprite_coords[0][0]
    step = 1 if delta > 0 else -1
    out = set()
    for offset in range(0, delta + step, step):
        for r, c in sprite_coords:
            out.add((r + offset, c))
    return out


def _nearest_anchor_mismatch(sprites: list[dict], anchors: list[dict], matches: dict[tuple[tuple[int, int], ...], int]) -> bool:
    for sprite in sprites:
        sr, sc = sprite["center"]
        dists = []
        for idx, anchor in enumerate(anchors):
            ar, ac = anchor["anchor_center"]
            dists.append(((sr - ar) ** 2 + (sc - ac) ** 2, idx))
        dists.sort()
        if matches[sprite["signature"]] != dists[0][1]:
            return True
    return False


def _order_mismatch(sprites: list[dict], anchors: list[dict], matches: dict[tuple[tuple[int, int], ...], int], axis: int) -> bool:
    sprite_order = sorted(sprites, key=lambda item: (item["center"][axis], item["center"][1 - axis]))
    anchor_order = sorted(range(len(anchors)), key=lambda idx: (anchors[idx]["anchor_center"][axis], anchors[idx]["anchor_center"][1 - axis]))
    rank_by_anchor = {anchor_idx: rank for rank, anchor_idx in enumerate(anchor_order)}
    for rank, sprite in enumerate(sprite_order):
        if rank_by_anchor[matches[sprite["signature"]]] != rank:
            return True
    return False


def _validate_instance(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if get_bg(grid) != Color.BLACK:
        return False
    sprites = _extract_sprites(grid)
    anchors = _extract_anchors(grid)
    if not (2 <= len(sprites) <= 3 and len(sprites) == len(anchors)):
        return False
    if len({sprite["color"] for sprite in sprites}) != len(sprites):
        return False
    if len({sprite["signature"] for sprite in sprites}) != len(sprites):
        return False
    if len({anchor["indicator_signature"] for anchor in anchors}) != len(anchors):
        return False

    sprite_by_sig = {sprite["signature"]: sprite for sprite in sprites}
    matches = {}
    assignments = []
    for idx, anchor in enumerate(anchors):
        sig = anchor["indicator_signature"]
        if sig not in sprite_by_sig or sig in matches:
            return False
        matches[sig] = idx
        assignments.append((sprite_by_sig[sig], anchor))

    occupied_grey = set()
    for anchor in anchors:
        occupied_grey |= set(anchor["all_grey_coords"])

    final_occupied = set()
    for sprite, anchor in assignments:
        direction = _movement_direction(sprite, anchor)
        moved, edge = _slide_sprite(sprite, anchor, direction)
        mirrored = _mirror_across_anchor(moved, anchor, edge)
        if not _all_in_bounds(grid.shape, moved + mirrored):
            return False
        if set(moved) & occupied_grey:
            return False
        if set(moved) & set(mirrored):
            return False
        if not _touches_anchor_edge(moved, anchor, edge):
            return False
        sweep = _sweep_coords(list(sprite["coords"]), moved)
        other_static = set()
        for other_sprite in sprites:
            if other_sprite is sprite:
                continue
            other_static |= set(other_sprite["coords"])
        for other_anchor in anchors:
            if other_anchor is anchor:
                other_static |= set(other_anchor["indicator_coords"])
            else:
                other_static |= set(other_anchor["all_grey_coords"])
        if sweep & other_static:
            return False
        if final_occupied & set(moved):
            return False
        if final_occupied & set(mirrored):
            return False
        final_occupied |= set(moved) | set(mirrored)

    if len({anchor["indicator_side"] for anchor in anchors}) < 2:
        return False
    if not (
        _nearest_anchor_mismatch(sprites, anchors, matches)
        or _order_mismatch(sprites, anchors, matches, axis=0)
        or _order_mismatch(sprites, anchors, matches, axis=1)
    ):
        return False

    out = main(grid)
    if not isinstance(out, np.ndarray) or out.ndim != 2 or out.shape != grid.shape:
        return False
    if not np.array_equal(out, _render_output(grid.shape, assignments)):
        return False
    if len(np.unique(out)) < 3:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    k = random.choice((2, 2, 3))
    shapes = list(random.sample(list(SHAPES), k))
    colors = random.sample(list(SPRITE_COLORS), k)

    occupied_input = set()
    reserved_input = set()
    occupied_output = set()
    occupied_sweeps = set()
    pairs = []

    for idx in range(k):
        shape = shapes[idx]
        color = colors[idx]
        placed = None
        for _ in range(120):
            direction = random.choice(DIRS)
            indicator_choices = [side for side in DIRS if side != direction]
            indicator_side = random.choice(indicator_choices)
            gap = random.randint(GAP_MIN, GAP_MAX)
            candidate = _build_pair_candidate((h, w), shape, direction, indicator_side, gap)
            if candidate is None:
                continue

            sprite_coords = candidate["sprite_coords"]
            grey_coords = candidate["grey_coords"]
            moved_coords = candidate["moved_coords"]
            mirrored_coords = candidate["mirrored_coords"]
            sweep = _sweep_coords(sprite_coords, moved_coords)

            if set(sprite_coords) & occupied_sweeps:
                continue
            if set(grey_coords) & occupied_sweeps:
                continue
            if sweep & occupied_input:
                continue
            if set(sprite_coords) & reserved_input:
                continue
            if set(grey_coords) & reserved_input:
                continue
            if set(moved_coords) & occupied_output:
                continue
            if set(mirrored_coords) & occupied_output:
                continue

            placed = {
                **candidate,
                "color": color,
            }
            break
        if placed is None:
            return None

        sprite_coords = placed["sprite_coords"]
        grey_coords = placed["grey_coords"]
        moved_coords = placed["moved_coords"]
        mirrored_coords = placed["mirrored_coords"]
        sweep = _sweep_coords(sprite_coords, moved_coords)

        occupied_input |= set(sprite_coords) | set(grey_coords)
        reserved_input |= _coords_with_padding(sprite_coords + grey_coords, (h, w), pad=1)
        occupied_output |= set(moved_coords) | set(mirrored_coords)
        occupied_sweeps |= sweep
        pairs.append(placed)

    grid = np.zeros((h, w), dtype=int)
    for pair in pairs:
        for r, c in pair["sprite_coords"]:
            grid[r, c] = pair["color"]
        for r, c in pair["grey_coords"]:
            grid[r, c] = GREY

    return grid if _validate_instance(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shape-match slide-mirror ARC input")
