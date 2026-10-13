from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREEN = 3
GRAY = 5
SPRITE_COLORS = [1, 2, 4, 6, 7, 8, 9]
GEN_MAX_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _frame_border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for coords in _components_mask(grid == GRAY):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        local = np.zeros((h, w), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        if np.array_equal(local, _frame_border_mask(h, w)):
            frames.append((r0, c0, r1, c1))
    return sorted(frames)


def _extract_sprite(inner: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    patch = inner[r0 : r1 + 1, c0 : c1 + 1].copy()
    local_mask = np.zeros(patch.shape, dtype=bool)
    local_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    patch[~local_mask] = BLACK
    colors = Counter(int(v) for v in patch[patch != BLACK])
    anchor_colors = [color for color, count in colors.items() if count == 1]
    assert len(anchor_colors) == 1, "each sprite must have exactly one singleton anchor color"
    anchor_color = int(anchor_colors[0])
    anchor_pos = tuple(map(int, np.argwhere(patch == anchor_color)[0]))
    cells = np.argwhere(patch != BLACK)
    return {
        "patch": patch,
        "anchor_color": anchor_color,
        "anchor_pos": anchor_pos,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "area": int(coords.shape[0]),
        "cells": [tuple(map(int, cell)) for cell in cells],
    }


def _parse_puzzle(grid: np.ndarray) -> tuple[dict, dict]:
    frames = _find_frames(grid)
    assert len(frames) == 2, "expected exactly two gray frames"
    source = None
    target = None

    for bbox in frames:
        r0, c0, r1, c1 = bbox
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        comps = _components_mask(inner != BLACK)
        assert comps, "frame interior must contain objects"
        areas = [int(comp.shape[0]) for comp in comps]
        if all(area == 1 for area in areas):
            assert target is None, "expected one target frame"
            markers = []
            for comp in comps:
                rr, cc = map(int, comp[0])
                markers.append({"pos": (rr, cc), "color": int(inner[rr, cc])})
            target = {"bbox": bbox, "inner_shape": inner.shape, "markers": markers}
        else:
            assert source is None, "expected one source frame"
            sprites = [_extract_sprite(inner, comp) for comp in comps]
            source = {"bbox": bbox, "inner_shape": inner.shape, "sprites": sprites}

    assert source is not None and target is not None, "could not classify source and target frames"

    sprite_colors = {int(sprite["anchor_color"]) for sprite in source["sprites"]}
    marker_colors = {int(marker["color"]) for marker in target["markers"]}
    assert sprite_colors == marker_colors, "marker colors must match source anchor colors"
    assert len(source["sprites"]) == len(target["markers"]), "marker count must equal sprite count"
    return source, target


def _stamp_into_inner(inner: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ih, iw = inner.shape
    ph, pw = patch.shape
    for pr in range(ph):
        for pc in range(pw):
            color = int(patch[pr, pc])
            if color == BLACK:
                continue
            rr = top + pr
            cc = left + pc
            if 0 <= rr < ih and 0 <= cc < iw:
                inner[rr, cc] = color


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _chebyshev_path_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    r, c = int(sr), int(sc)
    step_r = _sign(er - sr)
    step_c = _sign(ec - sc)
    cells: list[tuple[int, int]] = []
    while r != er and c != ec:
        r += step_r
        c += step_c
        if (r, c) != (er, ec):
            cells.append((r, c))
    while r != er:
        r += step_r
        if (r, c) != (er, ec):
            cells.append((r, c))
    while c != ec:
        c += step_c
        if (r, c) != (er, ec):
            cells.append((r, c))
    return cells


def _render_target(
    outer_shape: tuple[int, int],
    sprites_by_color: dict[int, dict],
    markers: list[dict],
) -> np.ndarray:
    oh, ow = map(int, outer_shape)
    out = np.zeros((oh, ow), dtype=int)
    out[0, :] = GRAY
    out[-1, :] = GRAY
    out[:, 0] = GRAY
    out[:, -1] = GRAY
    inner = out[1:-1, 1:-1]
    anchor_positions: list[tuple[int, int]] = []

    for marker in markers:
        sprite = sprites_by_color[int(marker["color"])]
        mr, mc = map(int, marker["pos"])
        ar, ac = map(int, sprite["anchor_pos"])
        _stamp_into_inner(inner, sprite["patch"], mr - ar, mc - ac)
        anchor_positions.append((mr + 1, mc + 1))

    anchor_positions.sort()
    for start, end in zip(anchor_positions, anchor_positions[1:]):
        for rr, cc in _chebyshev_path_cells(start, end):
            if int(out[rr, cc]) == BLACK:
                out[rr, cc] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp source sprites into the target frame by anchor color, then connect stamped anchors with green Chebyshev paths.

    Input structure:
    - A black background contains exactly two non-touching hollow gray rectangles.
    - The source rectangle encloses 2-3 multicolor 4-connected sprites; each sprite has exactly one singleton anchor color.
    - The target rectangle encloses the same number of isolated single-pixel markers, each colored with one sprite anchor color.

    Rule:
    - Identify the source frame as the one containing multi-pixel sprites and the target frame as the one containing singleton markers.
    - For each source sprite, use its singleton color as the anchor key and anchor position.
    - Clear the target interior to black, stamp each matched sprite so its anchor lands on the marker of the same color, then sort stamped anchors in row-major order.
    - For each consecutive anchor pair, draw the intermediate cells of the diagonal-then-straight Chebyshev-shortest path in green, but only on black cells.

    Key insight:
    - The singleton anchor color serves two roles at once: it chooses which sprite to stamp and it fixes the absolute endpoints of the later path geometry.
    """
    g = np.asarray(grid, dtype=int)
    source, target = _parse_puzzle(g)
    sprites_by_color = {
        int(sprite["anchor_color"]): sprite
        for sprite in source["sprites"]
    }
    target_bbox = target["bbox"]
    outer_shape = (target_bbox[2] - target_bbox[0] + 1, target_bbox[3] - target_bbox[1] + 1)
    return _render_target(outer_shape, sprites_by_color, target["markers"])


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seed = (random.randrange(h), random.randrange(w))
    mask[seed] = True
    frontier = {seed}
    while int(mask.sum()) < area:
        candidates: set[tuple[int, int]] = set()
        for r, c in frontier:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    candidates.add((nr, nc))
        if not candidates:
            break
        pick = random.choice(list(candidates))
        mask[pick] = True
        frontier.add(pick)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return mask
    if rows.size < h:
        mask[random.choice([0, h - 1]), random.randrange(w)] = True
    if cols.size < w:
        mask[random.randrange(h), random.choice([0, w - 1])] = True
    while int(mask.sum()) < area:
        boundary = []
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    boundary.append((nr, nc))
        if not boundary:
            break
        mask[random.choice(boundary)] = True
    return mask


def _make_sprite(anchor_color: int, body_colors: list[int]) -> dict:
    for _ in range(200):
        h = random.randint(3, 4)
        w = random.randint(4, 5)
        area = random.randint(max(6, h + w - 1), min(h * w - 1, 10))
        mask = _grow_connected_mask(h, w, area)
        if int(mask.sum()) != area:
            continue
        if len(_components_mask(mask)) != 1:
            continue
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if np.all(mask):
            continue
        cells = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
        anchor = random.choice(cells)
        patch = np.zeros((h, w), dtype=int)
        use_accent = len(body_colors) > 1 and area >= 6 and random.random() < 0.6
        accent_cells: set[tuple[int, int]] = set()
        if use_accent:
            others = [cell for cell in cells if cell != anchor]
            accent_count = min(len(others), random.randint(2, min(3, len(others))))
            accent_cells = set(random.sample(others, accent_count))
        for cell in cells:
            if cell == anchor:
                patch[cell] = int(anchor_color)
            elif cell in accent_cells:
                patch[cell] = int(body_colors[1])
            else:
                patch[cell] = int(body_colors[0])
        counts = Counter(int(v) for v in patch[patch != BLACK])
        if counts[int(anchor_color)] != 1:
            continue
        if sum(count == 1 for count in counts.values()) != 1:
            continue
        return {
            "patch": patch,
            "anchor_color": int(anchor_color),
            "anchor_pos": anchor,
            "cells": cells,
            "h": h,
            "w": w,
        }
    raise RuntimeError("failed to build sprite")


def _translated_cells(sprite: dict, anchor: tuple[int, int]) -> set[tuple[int, int]]:
    ar, ac = map(int, sprite["anchor_pos"])
    top = int(anchor[0] - ar)
    left = int(anchor[1] - ac)
    return {(top + r, left + c) for r, c in sprite["cells"]}


def _row_major_gt(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return (int(a[0]), int(a[1])) > (int(b[0]), int(b[1]))


def _relation_ok(prev: tuple[int, int], cur: tuple[int, int], relation: str) -> bool:
    dr = int(cur[0] - prev[0])
    dc = int(cur[1] - prev[1])
    if not _row_major_gt(cur, prev):
        return False
    if relation == "same_row":
        return dr == 0 and dc >= 2
    if relation == "same_col":
        return dc == 0 and dr >= 2
    if relation == "diag":
        return dr > 0 and abs(dr) == abs(dc) and abs(dc) >= 2
    if relation == "general":
        return dr > 0 and dc != 0 and abs(dr) != abs(dc)
    raise ValueError(f"unknown relation {relation}")


def _feasible_anchor_positions(sprite: dict, inner_h: int, inner_w: int) -> list[tuple[int, int]]:
    ar, ac = map(int, sprite["anchor_pos"])
    r_min = ar
    r_max = inner_h - (int(sprite["h"]) - ar)
    c_min = ac
    c_max = inner_w - (int(sprite["w"]) - ac)
    if r_min > r_max or c_min > c_max:
        return []
    return [(r, c) for r in range(r_min, r_max + 1) for c in range(c_min, c_max + 1)]


def _anchors_far_enough(anchors: list[tuple[int, int]], candidate: tuple[int, int]) -> bool:
    for anchor in anchors:
        if max(abs(int(anchor[0] - candidate[0])), abs(int(anchor[1] - candidate[1]))) < 2:
            return False
    return True


def _place_target_sprites(
    sprites: list[dict],
    inner_h: int,
    inner_w: int,
    relations: list[str],
    require_occupied_path: bool,
) -> list[dict] | None:
    for _ in range(400):
        order = sprites[:]
        random.shuffle(order)
        feasible = [_feasible_anchor_positions(sprite, inner_h, inner_w) for sprite in order]
        if any(not positions for positions in feasible):
            continue
        anchors: list[tuple[int, int]] = []
        occupied: list[set[tuple[int, int]]] = []
        placed: list[dict] = []
        first_choices = feasible[0][:]
        random.shuffle(first_choices)
        if not first_choices:
            continue
        start_ok = False
        for first in first_choices:
            anchors = [first]
            occupied = [_translated_cells(order[0], first)]
            placed = [{"sprite": order[0], "anchor": first}]
            start_ok = True
            break
        if not start_ok:
            continue

        success = True
        for idx in range(1, len(order)):
            relation = relations[idx - 1]
            candidates = [pos for pos in feasible[idx] if _relation_ok(anchors[-1], pos, relation)]
            random.shuffle(candidates)
            choice = None
            for candidate in candidates:
                cells = _translated_cells(order[idx], candidate)
                if not _anchors_far_enough(anchors, candidate):
                    continue
                if any(cells & other for other in occupied):
                    continue
                choice = candidate
                occupied.append(cells)
                anchors.append(candidate)
                placed.append({"sprite": order[idx], "anchor": candidate})
                break
            if choice is None:
                success = False
                break
        if not success:
            continue

        markers = [
            {"pos": tuple(map(int, item["anchor"])), "color": int(item["sprite"]["anchor_color"])}
            for item in placed
        ]
        sprites_by_color = {
            int(item["sprite"]["anchor_color"]): item["sprite"]
            for item in placed
        }
        stamped = _render_target((inner_h + 2, inner_w + 2), sprites_by_color, markers)
        sorted_anchors = sorted((anchor[0] + 1, anchor[1] + 1) for anchor in anchors)
        if require_occupied_path:
            touches_sprite = False
            for start, end in zip(sorted_anchors, sorted_anchors[1:]):
                for rr, cc in _chebyshev_path_cells(start, end):
                    if int(stamped[rr, cc]) not in (BLACK, GREEN):
                        touches_sprite = True
                        break
                if touches_sprite:
                    break
            if not touches_sprite:
                continue
        return markers
    return None


def _can_place_source(sprite: dict, top: int, left: int, occupied: set[tuple[int, int]]) -> bool:
    for r, c in sprite["cells"]:
        rr = int(top + r)
        cc = int(left + c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (rr + dr, cc + dc) in occupied:
                    return False
    return True


def _place_source_sprites(sprites: list[dict], inner_h: int, inner_w: int) -> list[dict] | None:
    for _ in range(300):
        order = sprites[:]
        random.shuffle(order)
        occupied: set[tuple[int, int]] = set()
        placements: list[dict] = []
        success = True
        for sprite in order:
            top_choices = list(range(1, inner_h - int(sprite["h"])))
            left_choices = list(range(1, inner_w - int(sprite["w"])))
            random.shuffle(top_choices)
            random.shuffle(left_choices)
            found = None
            for top in top_choices:
                for left in left_choices:
                    if _can_place_source(sprite, top, left, occupied):
                        found = (top, left)
                        break
                if found is not None:
                    break
            if found is None:
                success = False
                break
            top, left = found
            placements.append({"sprite": sprite, "top": int(top), "left": int(left)})
            for r, c in sprite["cells"]:
                occupied.add((int(top + r), int(left + c)))
        if success:
            return placements
    return None


def _draw_frame(outer_h: int, outer_w: int) -> np.ndarray:
    frame = np.zeros((outer_h, outer_w), dtype=int)
    frame[0, :] = GRAY
    frame[-1, :] = GRAY
    frame[:, 0] = GRAY
    frame[:, -1] = GRAY
    return frame


def _compose_input(
    grid_h: int,
    grid_w: int,
    src_box: tuple[int, int],
    dst_box: tuple[int, int],
    src_origin: tuple[int, int],
    dst_origin: tuple[int, int],
    source_placements: list[dict],
    target_markers: list[dict],
) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    src_h, src_w = map(int, src_box)
    dst_h, dst_w = map(int, dst_box)
    sr, sc = map(int, src_origin)
    dr, dc = map(int, dst_origin)
    grid[sr : sr + src_h, sc : sc + src_w] = _draw_frame(src_h, src_w)
    grid[dr : dr + dst_h, dc : dc + dst_w] = _draw_frame(dst_h, dst_w)

    src_inner = grid[sr + 1 : sr + src_h - 1, sc + 1 : sc + src_w - 1]
    for item in source_placements:
        _stamp_into_inner(src_inner, item["sprite"]["patch"], int(item["top"]), int(item["left"]))

    dst_inner = grid[dr + 1 : dr + dst_h - 1, dc + 1 : dc + dst_w - 1]
    for marker in target_markers:
        mr, mc = map(int, marker["pos"])
        dst_inner[mr, mc] = int(marker["color"])
    return grid


def _choose_profile() -> dict:
    profiles = [
        {"n": 2, "relations": ["general"], "touch": False},
        {"n": 2, "relations": ["same_row"], "touch": False},
        {"n": 2, "relations": ["same_col"], "touch": False},
        {"n": 2, "relations": ["diag"], "touch": False},
        {"n": 3, "relations": ["general", "same_row"], "touch": False},
        {"n": 3, "relations": ["same_col", "general"], "touch": False},
        {"n": 3, "relations": ["diag", "general"], "touch": False},
        {"n": 3, "relations": ["general", "diag"], "touch": True},
    ]
    return random.choice(profiles)


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, profile: dict) -> bool:
    source, target = _parse_puzzle(inp)
    if len(source["sprites"]) != int(profile["n"]):
        return False
    if len(target["markers"]) != int(profile["n"]):
        return False
    if not np.array_equal(main(inp), expected):
        return False
    actual = main(inp)
    if not np.any(actual == GREEN):
        return False
    anchors = sorted((marker["pos"][0] + 1, marker["pos"][1] + 1) for marker in target["markers"])
    for relation, start, end in zip(profile["relations"], anchors, anchors[1:]):
        if not _relation_ok((start[0] - 1, start[1] - 1), (end[0] - 1, end[1] - 1), relation):
            return False
    if profile["touch"]:
        sprites_by_color = {int(sprite["anchor_color"]): sprite for sprite in source["sprites"]}
        stamped = _render_target(expected.shape, sprites_by_color, target["markers"])
        touched = False
        for start, end in zip(anchors, anchors[1:]):
            for rr, cc in _chebyshev_path_cells(start, end):
                if int(stamped[rr, cc]) not in (BLACK, GREEN):
                    touched = True
                    break
            if touched:
                break
        if not touched:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        profile = _choose_profile()
        n = int(profile["n"])
        color_pool = SPRITE_COLORS[:]
        random.shuffle(color_pool)
        anchor_colors = color_pool[:n]
        remaining = color_pool[n:]

        sprites: list[dict] = []
        for anchor_color in anchor_colors:
            body_count = 2 if remaining and random.random() < 0.5 else 1
            body_colors = random.sample(remaining, body_count)
            sprites.append(_make_sprite(anchor_color, body_colors))

        src_inner_h = random.randint(max(8, max(sprite["h"] for sprite in sprites) + 3), 11)
        src_inner_w = random.randint(max(9, max(sprite["w"] for sprite in sprites) + 5), 13)
        dst_inner_h = random.randint(max(8, max(sprite["h"] for sprite in sprites) + 3), 12)
        dst_inner_w = random.randint(max(9, max(sprite["w"] for sprite in sprites) + 4), 14)

        target_markers = _place_target_sprites(
            sprites,
            dst_inner_h,
            dst_inner_w,
            list(profile["relations"]),
            bool(profile["touch"]),
        )
        if target_markers is None:
            continue

        source_placements = _place_source_sprites(sprites, src_inner_h, src_inner_w)
        if source_placements is None:
            continue

        src_outer = (src_inner_h + 2, src_inner_w + 2)
        dst_outer = (dst_inner_h + 2, dst_inner_w + 2)
        gap = random.randint(2, 4)
        margin_a = random.randint(0, 2)
        margin_b = random.randint(0, 2)
        margin_c = random.randint(0, 2)
        margin_d = random.randint(0, 2)
        if random.random() < 0.5:
            grid_h = max(src_outer[0], dst_outer[0]) + margin_a + margin_b
            grid_w = src_outer[1] + dst_outer[1] + gap + margin_c + margin_d
            if grid_h > 30 or grid_w > 30:
                continue
            if random.random() < 0.5:
                src_origin = (random.randint(0, grid_h - src_outer[0]), margin_c)
                dst_origin = (random.randint(0, grid_h - dst_outer[0]), margin_c + src_outer[1] + gap)
            else:
                dst_origin = (random.randint(0, grid_h - dst_outer[0]), margin_c)
                src_origin = (random.randint(0, grid_h - src_outer[0]), margin_c + dst_outer[1] + gap)
        else:
            grid_h = src_outer[0] + dst_outer[0] + gap + margin_a + margin_b
            grid_w = max(src_outer[1], dst_outer[1]) + margin_c + margin_d
            if grid_h > 30 or grid_w > 30:
                continue
            if random.random() < 0.5:
                src_origin = (margin_a, random.randint(0, grid_w - src_outer[1]))
                dst_origin = (margin_a + src_outer[0] + gap, random.randint(0, grid_w - dst_outer[1]))
            else:
                dst_origin = (margin_a, random.randint(0, grid_w - dst_outer[1]))
                src_origin = (margin_a + dst_outer[0] + gap, random.randint(0, grid_w - src_outer[1]))

        inp = _compose_input(
            grid_h,
            grid_w,
            src_outer,
            dst_outer,
            src_origin,
            dst_origin,
            source_placements,
            target_markers,
        )
        sprites_by_color = {int(sprite["anchor_color"]): sprite for sprite in sprites}
        expected = _render_target(dst_outer, sprites_by_color, target_markers)
        if _semantic_checks(inp, expected, profile):
            return inp
    raise RuntimeError("failed to generate a valid input")
