from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
PORTAL_COLORS = (2, 3, 4, 6, 7, 8, 9)
SIDE_TO_INDEX = {"top": 0, "right": 1, "bottom": 2, "left": 3}
PAIR_COUNT_OPTIONS = (1, 2, 3)
MAX_GENERATION_ATTEMPTS = 400
PAIR_ATTEMPTS = 120
ROOM_PLACEMENT_ATTEMPTS = 200


def _prefix(mask: np.ndarray) -> np.ndarray:
    return np.pad(mask.astype(int), ((1, 0), (1, 0)), constant_values=0).cumsum(0).cumsum(1)


def _rect_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    return int(ps[r1 + 1, c1 + 1] - ps[r0, c1 + 1] - ps[r1 + 1, c0] + ps[r0, c0])


def _border_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    total = _rect_sum(ps, r0, c0, r1, c1)
    inner = _rect_sum(ps, r0 + 1, c0 + 1, r1 - 1, c1 - 1)
    return total - inner


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def _portal_side(r0: int, c0: int, r1: int, c1: int, r: int, c: int) -> str:
    hits = []
    if r == r0:
        hits.append("top")
    if r == r1:
        hits.append("bottom")
    if c == c0:
        hits.append("left")
    if c == c1:
        hits.append("right")
    assert len(hits) == 1, "portal pixels must not be on corners"
    return hits[0]


def _inner_anchor(side: str, r: int, c: int) -> tuple[int, int]:
    if side == "top":
        return r + 1, c
    if side == "bottom":
        return r - 1, c
    if side == "left":
        return r, c + 1
    return r, c - 1


def _rotate_offset(dr: int, dc: int, turns_cw: int) -> tuple[int, int]:
    for _ in range(turns_cw % 4):
        dr, dc = dc, -dr
    return int(dr), int(dc)


def _rotation_turns(src_side: str, dst_side: str) -> int:
    return (SIDE_TO_INDEX[dst_side] + 2 - SIDE_TO_INDEX[src_side]) % 4


def _find_rooms(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    ps_blue = _prefix(g == BLUE)
    ps_black = _prefix(g == BLACK)
    ps_grey = _prefix(g == GREY)
    ps_bad_interior = _prefix((g != BLACK) & (g != GREY))

    rooms = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if int(g[r0, c0]) != BLUE:
                continue
            for r1 in range(r0 + 2, h):
                if int(g[r1, c0]) != BLUE:
                    continue
                for c1 in range(c0 + 2, w):
                    if int(g[r0, c1]) != BLUE or int(g[r1, c1]) != BLUE:
                        continue

                    if _rect_sum(ps_bad_interior, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != 0:
                        continue
                    if _border_sum(ps_black, r0, c0, r1, c1) != 0:
                        continue
                    if _border_sum(ps_grey, r0, c0, r1, c1) != 0:
                        continue

                    perimeter = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
                    blue_on_border = _border_sum(ps_blue, r0, c0, r1, c1)
                    if perimeter - blue_on_border != 2:
                        continue

                    portals = []
                    for pr, pc in _border_coords(r0, c0, r1, c1):
                        color = int(g[pr, pc])
                        if color != BLUE:
                            side = _portal_side(r0, c0, r1, c1, pr, pc)
                            portals.append(
                                {
                                    "color": color,
                                    "pixel": (int(pr), int(pc)),
                                    "side": side,
                                    "anchor": _inner_anchor(side, int(pr), int(pc)),
                                }
                            )
                    if len(portals) != 2:
                        continue

                    interior = g[r0 + 1 : r1, c0 + 1 : c1]
                    grey_pts = np.argwhere(interior == GREY)
                    pattern_coords = [(int(r0 + 1 + r), int(c0 + 1 + c)) for r, c in grey_pts]
                    rooms.append(
                        {
                            "bbox": (int(r0), int(c0), int(r1), int(c1)),
                            "interior_bbox": (int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1)),
                            "portals": portals,
                            "pattern_coords": pattern_coords,
                            "has_pattern": bool(pattern_coords),
                        }
                    )

    rooms.sort(key=lambda room: room["bbox"])
    assert rooms, "expected at least one room"
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Teleport grey room-interior patterns through same-colored border portals.

    Each portal pair links one patterned room to one empty room. The source-side
    portal fixes the pattern's outgoing direction; the target-side portal fixes
    the incoming direction. The pattern is rotated so the source direction points
    toward the opposite of the target side, then translated by matching the
    source and target inner-adjacent anchor cells. Cells that land outside the
    target interior are cropped away, and surviving cells are recolored to the
    portal-pair color.
    """
    g = np.asarray(grid, dtype=int)
    rooms = _find_rooms(g)

    portal_pairs: dict[int, list[tuple[dict, dict]]] = {}
    for room in rooms:
        for portal in room["portals"]:
            portal_pairs.setdefault(int(portal["color"]), []).append((room, portal))

    out = g.copy()
    for color, endpoints in portal_pairs.items():
        assert len(endpoints) == 2, "every portal color must appear exactly twice"
        (room_a, portal_a), (room_b, portal_b) = endpoints
        assert room_a["has_pattern"] != room_b["has_pattern"], "each portal pair needs one source room"

        if room_a["has_pattern"]:
            src_room, src_portal = room_a, portal_a
            dst_room, dst_portal = room_b, portal_b
        else:
            src_room, src_portal = room_b, portal_b
            dst_room, dst_portal = room_a, portal_a

        turns = _rotation_turns(str(src_portal["side"]), str(dst_portal["side"]))
        src_anchor_r, src_anchor_c = map(int, src_portal["anchor"])
        dst_anchor_r, dst_anchor_c = map(int, dst_portal["anchor"])
        ir0, ic0, ir1, ic1 = map(int, dst_room["interior_bbox"])

        for sr, sc in src_room["pattern_coords"]:
            rr, cc = _rotate_offset(int(sr) - src_anchor_r, int(sc) - src_anchor_c, turns)
            tr = dst_anchor_r + rr
            tc = dst_anchor_c + cc
            if ir0 <= tr <= ir1 and ic0 <= tc <= ic1:
                out[tr, tc] = int(color)

    return out


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _is_rotation_distinct(mask: np.ndarray) -> bool:
    tight = _tight_mask(mask)
    for k in range(1, 4):
        rotated = np.rot90(tight, k)
        if rotated.shape == tight.shape and np.array_equal(rotated, tight):
            return False
    return True


def _random_connected_mask(h: int, w: int, want_crop: bool) -> np.ndarray:
    area = h * w
    if want_crop:
        lo = max(4, area // 3)
        hi = max(lo, min(area - 1, (2 * area) // 3))
    else:
        lo = max(3, area // 5)
        hi = max(lo, min(area - 2, area // 2))

    for _ in range(80):
        target = random.randint(lo, hi)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = {start}
        while len(cells) < target:
            options = []
            for r, c in frontier:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        options.append((nr, nc))
            if not options:
                break
            nr, nc = random.choice(options)
            cells.add((nr, nc))
            frontier.add((nr, nc))
        if len(cells) != target:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not _is_rotation_distinct(mask):
            continue
        return mask

    raise AssertionError("failed to sample a non-symmetric connected pattern")


def _portal_candidates(inner_h: int, inner_w: int) -> list[dict]:
    spots = []
    for c in range(inner_w):
        spots.append({"side": "top", "border_rel": (0, c + 1), "anchor_in": (0, c)})
        spots.append({"side": "bottom", "border_rel": (inner_h + 1, c + 1), "anchor_in": (inner_h - 1, c)})
    for r in range(inner_h):
        spots.append({"side": "left", "border_rel": (r + 1, 0), "anchor_in": (r, 0)})
        spots.append({"side": "right", "border_rel": (r + 1, inner_w + 1), "anchor_in": (r, inner_w - 1)})
    return spots


def _choose_source_portals(inner_h: int, inner_w: int) -> list[dict]:
    spots = _portal_candidates(inner_h, inner_w)
    for _ in range(60):
        pick = random.sample(spots, 2)
        dist = abs(int(pick[0]["border_rel"][0]) - int(pick[1]["border_rel"][0])) + abs(
            int(pick[0]["border_rel"][1]) - int(pick[1]["border_rel"][1])
        )
        if dist >= 2:
            return pick
    raise AssertionError("failed to choose source portal positions")


def _target_side_for(src_side: str, turns: int) -> str:
    idx = (SIDE_TO_INDEX[src_side] + turns + 2) % 4
    for side, side_idx in SIDE_TO_INDEX.items():
        if side_idx == idx:
            return side
    raise AssertionError("unknown side index")


def _choose_target_portals(inner_h: int, inner_w: int, target_sides: list[str]) -> list[dict] | None:
    spots = _portal_candidates(inner_h, inner_w)
    chosen = []
    used = set()
    for side in target_sides:
        choices = [spot for spot in spots if spot["side"] == side and tuple(spot["border_rel"]) not in used]
        if not choices:
            return None
        spot = random.choice(choices)
        chosen.append(spot)
        used.add(tuple(spot["border_rel"]))
    return chosen


def _teleported_cells(
    pattern_mask: np.ndarray,
    src_anchor_in: tuple[int, int],
    src_side: str,
    dst_anchor_in: tuple[int, int],
    dst_side: str,
    dst_shape: tuple[int, int],
) -> tuple[set[tuple[int, int]], bool]:
    turns = _rotation_turns(src_side, dst_side)
    dst_h, dst_w = dst_shape
    cells: set[tuple[int, int]] = set()
    cropped = False
    for sr, sc in np.argwhere(pattern_mask):
        dr, dc = _rotate_offset(int(sr) - int(src_anchor_in[0]), int(sc) - int(src_anchor_in[1]), turns)
        tr = int(dst_anchor_in[0]) + dr
        tc = int(dst_anchor_in[1]) + dc
        if 0 <= tr < dst_h and 0 <= tc < dst_w:
            cells.add((tr, tc))
        else:
            cropped = True
    return cells, cropped


def _build_pair(colors: tuple[int, int]) -> dict:
    want_crop = bool(random.getrandbits(1))
    for _ in range(PAIR_ATTEMPTS):
        if want_crop:
            src_h = random.randint(5, 7)
            src_w = random.randint(5, 7)
            dst_h = random.randint(3, 6)
            dst_w = random.randint(3, 6)
        else:
            src_h = random.randint(4, 6)
            src_w = random.randint(4, 6)
            dst_h = random.randint(5, 8)
            dst_w = random.randint(5, 8)

        pattern = _random_connected_mask(src_h, src_w, want_crop=want_crop)
        src_portals = _choose_source_portals(src_h, src_w)
        desired_turns = [random.randrange(4), random.randrange(4)]
        target_sides = [_target_side_for(str(portal["side"]), int(turns)) for portal, turns in zip(src_portals, desired_turns, strict=False)]
        dst_portals = _choose_target_portals(dst_h, dst_w, target_sides)
        if dst_portals is None:
            continue

        placements = []
        occupied: set[tuple[int, int]] = set()
        cropped_any = False
        valid = True
        for color, src_portal, dst_portal in zip(colors, src_portals, dst_portals, strict=False):
            cells, cropped = _teleported_cells(
                pattern,
                tuple(src_portal["anchor_in"]),
                str(src_portal["side"]),
                tuple(dst_portal["anchor_in"]),
                str(dst_portal["side"]),
                (dst_h, dst_w),
            )
            if len(cells) < 2 or cells & occupied:
                valid = False
                break
            occupied |= cells
            cropped_any |= bool(cropped)
            placements.append({"color": int(color), "cells": cells})
        if not valid or cropped_any != want_crop:
            continue

        return {
            "source_inner": (int(src_h), int(src_w)),
            "target_inner": (int(dst_h), int(dst_w)),
            "pattern": pattern,
            "colors": tuple(int(c) for c in colors),
            "src_portals": src_portals,
            "dst_portals": dst_portals,
            "placements": placements,
        }

    raise AssertionError("failed to build a valid portal pair")


def _room_patch(inner_h: int, inner_w: int, portals: list[dict], colors: tuple[int, int], pattern: np.ndarray | None) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), BLACK, dtype=int)
    patch[0, :] = BLUE
    patch[-1, :] = BLUE
    patch[:, 0] = BLUE
    patch[:, -1] = BLUE
    if pattern is not None:
        patch[1:-1, 1:-1][pattern] = GREY
    for portal, color in zip(portals, colors, strict=False):
        pr, pc = map(int, portal["border_rel"])
        patch[pr, pc] = int(color)
    return patch


def _sample_canvas(room_shapes: list[tuple[int, int]]) -> tuple[int, int]:
    total_area = sum(h * w for h, w in room_shapes)
    max_h = max(h for h, _ in room_shapes)
    max_w = max(w for _, w in room_shapes)
    base = int(np.ceil(np.sqrt(total_area * random.uniform(2.2, 3.0))))
    h = random.randint(max(max_h + 2, base - 2), max(max_h + 3, base + 6))
    w = random.randint(max(max_w + 2, base - 1), max(max_w + 3, base + 8))
    return int(h), int(w)


def _place_rooms(room_patches: list[np.ndarray]) -> list[tuple[int, int]] | None:
    room_shapes = [tuple(map(int, patch.shape)) for patch in room_patches]
    h, w = _sample_canvas(room_shapes)
    order = sorted(range(len(room_patches)), key=lambda idx: room_patches[idx].size, reverse=True)
    positions: list[tuple[int, int] | None] = [None] * len(room_patches)
    placed_boxes: list[tuple[int, int, int, int]] = []

    for idx in order:
        ph, pw = map(int, room_patches[idx].shape)
        options = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        random.shuffle(options)
        placed = False
        for r0, c0 in options[:ROOM_PLACEMENT_ATTEMPTS]:
            box = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
            ok = True
            for er0, ec0, er1, ec1 in placed_boxes:
                if not (box[2] + 1 < er0 or er1 + 1 < box[0] or box[3] + 1 < ec0 or ec1 + 1 < box[1]):
                    ok = False
                    break
            if ok:
                positions[idx] = (int(r0), int(c0))
                placed_boxes.append(box)
                placed = True
                break
        if not placed:
            return None

    return [(int(r), int(c)) for r, c in positions]


def _build_case() -> np.ndarray:
    pair_count = random.choice(PAIR_COUNT_OPTIONS)
    colors = random.sample(PORTAL_COLORS, 2 * pair_count)
    pairs = []
    for pair_idx in range(pair_count):
        pair_colors = (int(colors[2 * pair_idx]), int(colors[2 * pair_idx + 1]))
        pairs.append(_build_pair(pair_colors))

    room_patches = []
    for pair in pairs:
        src_h, src_w = map(int, pair["source_inner"])
        dst_h, dst_w = map(int, pair["target_inner"])
        room_patches.append(_room_patch(src_h, src_w, pair["src_portals"], pair["colors"], pair["pattern"]))
        room_patches.append(_room_patch(dst_h, dst_w, pair["dst_portals"], pair["colors"], None))

    positions = _place_rooms(room_patches)
    if positions is None:
        raise AssertionError("failed to place room patches")

    h = max(r + patch.shape[0] for (r, _), patch in zip(positions, room_patches, strict=False))
    w = max(c + patch.shape[1] for (_, c), patch in zip(positions, room_patches, strict=False))
    grid = np.full((int(h), int(w)), BLACK, dtype=int)
    for (r0, c0), patch in zip(positions, room_patches, strict=False):
        ph, pw = patch.shape
        grid[int(r0) : int(r0) + ph, int(c0) : int(c0) + pw] = patch

    rooms = _find_rooms(grid)
    assert len(rooms) == 2 * pair_count, "generated grid must have the expected room count"
    out = main(grid)
    assert np.any(out != grid), "teleporting should change the grid"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            return _build_case()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid ARC input")
