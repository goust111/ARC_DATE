from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
PINK = 6
ORANGE = 7

SIDES = ("top", "right", "bottom", "left")
COLOR_TO_SIDE = {BLUE: "top", RED: "right", GREEN: "bottom", YELLOW: "left"}
SIDE_TO_COLOR = {side: color for color, side in COLOR_TO_SIDE.items()}
TARGET_ANCHORS = {"top": (0, 1), "right": (1, 2), "bottom": (2, 1), "left": (1, 0)}
FRAME_COLORS = {BLUE, RED, GREEN, YELLOW}
SPRITE_COLORS = {GRAY, PINK, ORANGE}
GEN_RETRIES = 400


def _anchor_coord(sprite: np.ndarray) -> tuple[int, int]:
    locs = np.argwhere(sprite == GRAY)
    assert locs.shape == (1, 2), "sprite must contain exactly one gray anchor"
    return int(locs[0, 0]), int(locs[0, 1])


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    q = deque([tuple(map(int, coords[0]))])
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    H, W = grid.shape
    candidates = []
    for r0 in range(H - 4):
        for c0 in range(W - 4):
            if int(grid[r0, c0]) != BLUE:
                continue
            for r1 in range(r0 + 4, H):
                if int(grid[r1, c0]) != GREEN:
                    continue
                for c1 in range(c0 + 4, W):
                    if int(grid[r0, c1]) != BLUE or int(grid[r1, c1]) != GREEN:
                        continue
                    if not np.all(grid[r0, c0 : c1 + 1] == BLUE):
                        continue
                    if not np.all(grid[r1, c0 : c1 + 1] == GREEN):
                        continue
                    if not np.all(grid[r0 + 1 : r1, c0] == YELLOW):
                        continue
                    if not np.all(grid[r0 + 1 : r1, c1] == RED):
                        continue
                    candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, "expected exactly one four-colored frame"
    return candidates[0]


def _extract_template(grid: np.ndarray, frame: tuple[int, int, int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = frame
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = np.isin(interior, list(SPRITE_COLORS))
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected one sprite template inside the frame"
    sr0, sc0 = coords.min(axis=0)
    sr1, sc1 = coords.max(axis=0)
    assert (int(sr1 - sr0 + 1), int(sc1 - sc0 + 1)) == (3, 3), "template bbox must be 3x3"
    patch = interior[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
    anchor = _anchor_coord(patch)
    assert anchor in set(TARGET_ANCHORS.values()), "anchor must lie on a side-center cell"
    return patch, (r0 + 1 + int(sr0), c0 + 1 + int(sc0))


def _rotate_to_side(sprite: np.ndarray, side: str) -> np.ndarray:
    target = TARGET_ANCHORS[side]
    matches = []
    for k in range(4):
        rot = np.rot90(sprite, -k)
        if _anchor_coord(rot) == target:
            matches.append(rot)
    assert len(matches) == 1, "sprite orientation must be uniquely determined by anchor side"
    return matches[0]


def _stamp_cells(sprite: np.ndarray, attach: tuple[int, int]) -> dict[tuple[int, int], int]:
    ar, ac = _anchor_coord(sprite)
    out = {}
    for pr, pc in np.argwhere((sprite != BG) & (sprite != GRAY)):
        rr = int(attach[0] + pr - ar)
        cc = int(attach[1] + pc - ac)
        out[(rr, cc)] = int(sprite[pr, pc])
    return out


def _stamp_inside_frame(
    sprite: np.ndarray, attach: tuple[int, int], frame: tuple[int, int, int, int]
) -> bool:
    r0, c0, r1, c1 = frame
    for rr, cc in _stamp_cells(sprite, attach):
        if not (r0 + 1 <= rr <= r1 - 1 and c0 + 1 <= cc <= c1 - 1):
            return False
    return True


def _extract_markers(grid: np.ndarray, frame: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = frame
    markers = []
    side_projections = {side: set() for side in SIDES}
    for r, c in np.argwhere(np.isin(grid, list(FRAME_COLORS))):
        r = int(r)
        c = int(c)
        on_frame = (
            (r == r0 and c0 <= c <= c1)
            or (r == r1 and c0 <= c <= c1)
            or (c == c0 and r0 < r < r1)
            or (c == c1 and r0 < r < r1)
        )
        if on_frame:
            continue
        color = int(grid[r, c])
        side = COLOR_TO_SIDE[color]
        if side == "top":
            assert r < r0 and c0 <= c <= c1
            attach = (r0, c)
            distance = r0 - r
        elif side == "right":
            assert c > c1 and r0 <= r <= r1
            attach = (r, c1)
            distance = c - c1
        elif side == "bottom":
            assert r > r1 and c0 <= c <= c1
            attach = (r1, c)
            distance = r - r1
        else:
            assert c < c0 and r0 <= r <= r1
            attach = (r, c0)
            distance = c0 - c
        side_projections[side].add(attach)
        markers.append(
            {
                "row": r,
                "col": c,
                "color": color,
                "side": side,
                "attach": attach,
                "distance": int(distance),
            }
        )
    assert 3 <= len(markers) <= 6, "expected 3 to 6 markers"
    assert len({m["distance"] for m in markers}) == len(markers), "marker distances must be unique"
    for side in SIDES:
        expected = [m for m in markers if m["side"] == side]
        assert len(side_projections[side]) == len(expected), "same-side projections must be unique"
    return markers


def _render(
    grid: np.ndarray,
    frame: tuple[int, int, int, int],
    template: np.ndarray,
    template_origin: tuple[int, int],
    markers: list[dict],
    *,
    sort_near_last: bool = True,
    rotate: bool = True,
    erase_template: bool = True,
    strict: bool = True,
) -> np.ndarray:
    out = np.array(grid, copy=True)
    tr0, tc0 = template_origin
    if erase_template:
        out[tr0 : tr0 + 3, tc0 : tc0 + 3] = BG

    ordered = sorted(markers, key=lambda m: m["distance"], reverse=sort_near_last)
    for marker in ordered:
        side = marker["side"]
        sprite = _rotate_to_side(template, side) if rotate else template
        if strict:
            assert _stamp_inside_frame(sprite, marker["attach"], frame), "stamp must stay inside frame"
        for (rr, cc), color in _stamp_cells(sprite, marker["attach"]).items():
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp rotated copies of the inner sprite onto frame-edge anchors chosen by same-colored outer markers, with nearer markers drawn on top.

    Input structure:
    - The background is black.
    - One single-pixel rectangular frame appears with blue top, red right, green bottom, and yellow left edges.
    - A multicolor 3x3 sprite with one gray anchor sits inside the frame near the upper-left, and 3 to 6 single-pixel outer markers use the four frame-edge colors.

    Rule:
    - Find the inner 3x3 sprite and its gray anchor.
    - For each outer marker, use its color to choose the matching frame edge and project to the attachment point on that edge by shared row or shared column.
    - Rotate the sprite so the gray anchor faces outward from that edge, align the anchor to the attachment point, erase the original sprite, and stamp only the non-gray sprite pixels.
    - If stamped non-gray pixels overlap, draw the marker farther from its matched edge first and the nearer marker last.

    Key insight:
    - The frame encodes three signals at once: marker color chooses the edge, that edge fixes the sprite rotation, and the marker-to-edge distance fixes the overlap order.
    """
    g = np.asarray(grid, dtype=int)
    frame = _find_frame(g)
    template, template_origin = _extract_template(g, frame)
    markers = _extract_markers(g, frame)
    return _render(g, frame, template, template_origin, markers, strict=True)


def _random_template() -> np.ndarray:
    anchor = TARGET_ANCHORS["top"]
    positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    while True:
        count = random.randint(4, 6)
        chosen = set(random.sample(positions, count))
        if (1, 1) not in chosen:
            continue
        if not any(r == 2 for r, _ in chosen):
            continue
        mask = np.zeros((3, 3), dtype=bool)
        mask[anchor] = True
        for rc in chosen:
            mask[rc] = True
        if not _connected(mask):
            continue
        if not mask.any(axis=0).all() or not mask.any(axis=1).all():
            continue

        sprite = np.zeros((3, 3), dtype=int)
        sprite[anchor] = GRAY
        colors = [PINK] * len(chosen)
        for i in random.sample(range(len(chosen)), random.randint(1, len(chosen) - 1)):
            colors[i] = ORANGE
        for (r, c), color in zip(sorted(chosen), colors):
            sprite[r, c] = color

        visible_rots = {np.where(np.rot90(sprite, k) == GRAY, BG, np.rot90(sprite, k)).tobytes() for k in range(4)}
        if len(visible_rots) != 4:
            continue
        return np.rot90(sprite, random.randrange(4)).astype(int)


def _draw_frame(grid: np.ndarray, frame: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame
    grid[r0, c0 : c1 + 1] = BLUE
    grid[r1, c0 : c1 + 1] = GREEN
    grid[r0 + 1 : r1, c0] = YELLOW
    grid[r0 + 1 : r1, c1] = RED


def _eligible_attaches(template: np.ndarray, frame: tuple[int, int, int, int]) -> dict[str, list[tuple[int, int]]]:
    r0, c0, r1, c1 = frame
    out = {}
    for side in SIDES:
        sprite = _rotate_to_side(template, side)
        attach_list = []
        if side in ("top", "bottom"):
            edge_r = r0 if side == "top" else r1
            for c in range(c0, c1 + 1):
                attach = (edge_r, c)
                if _stamp_inside_frame(sprite, attach, frame):
                    attach_list.append(attach)
        else:
            edge_c = c0 if side == "left" else c1
            for r in range(r0, r1 + 1):
                attach = (r, edge_c)
                if _stamp_inside_frame(sprite, attach, frame):
                    attach_list.append(attach)
        out[side] = attach_list
    return out


def _marker_side_sequence(count: int, require_overlap: bool) -> list[str]:
    while True:
        distinct = 4 if count >= 4 and random.random() < 0.75 else 3
        seq = random.sample(list(SIDES), distinct)
        while len(seq) < count:
            choices = [side for side in SIDES if seq.count(side) < 2]
            seq.append(random.choice(choices))
        random.shuffle(seq)
        if require_overlap and all(seq.count(side) == 1 for side in SIDES):
            continue
        return seq


def _build_stamp_layout(
    template: np.ndarray,
    frame: tuple[int, int, int, int],
    side_seq: list[str],
    require_overlap: bool,
) -> list[dict] | None:
    eligible = _eligible_attaches(template, frame)
    counts = Counter(side_seq)
    if any(len(eligible[side]) < count for side, count in counts.items()):
        return None

    for _ in range(300):
        chosen = []
        for side in side_seq:
            pool = [attach for attach in eligible[side] if attach not in [item["attach"] for item in chosen if item["side"] == side]]
            if not pool:
                break
            attach = random.choice(pool)
            sprite = _rotate_to_side(template, side)
            chosen.append({"side": side, "attach": attach, "sprite": sprite, "cells": _stamp_cells(sprite, attach)})
        if len(chosen) != len(side_seq):
            continue

        overlap = False
        conflict = False
        painted = {}
        for item in chosen:
            for coord, color in item["cells"].items():
                if coord in painted:
                    overlap = True
                    if painted[coord] != color:
                        conflict = True
                else:
                    painted[coord] = color
        if require_overlap:
            if conflict:
                return chosen
        else:
            if not overlap:
                return chosen
    return None


def _place_template(grid: np.ndarray, frame: tuple[int, int, int, int], template: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = frame
    max_roff = min(2, r1 - r0 - 4)
    max_coff = min(2, c1 - c0 - 4)
    tr0 = r0 + 1 + random.randint(0, max_roff)
    tc0 = c0 + 1 + random.randint(0, max_coff)
    patch = grid[tr0 : tr0 + 3, tc0 : tc0 + 3]
    assert np.all(patch == BG)
    mask = template != BG
    patch[mask] = template[mask]
    return tr0, tc0


def _materialize_markers(
    grid_shape: tuple[int, int],
    frame: tuple[int, int, int, int],
    stamps: list[dict],
) -> list[dict] | None:
    H, W = grid_shape
    r0, c0, r1, c1 = frame
    distances = list(range(1, len(stamps) + 1))
    random.shuffle(distances)
    markers = []
    used = set()
    for stamp, dist in zip(stamps, distances):
        side = stamp["side"]
        ar, ac = stamp["attach"]
        if side == "top":
            mr, mc = r0 - dist, ac
        elif side == "right":
            mr, mc = ar, c1 + dist
        elif side == "bottom":
            mr, mc = r1 + dist, ac
        else:
            mr, mc = ar, c0 - dist
        if not (0 <= mr < H and 0 <= mc < W):
            return None
        if (mr, mc) in used:
            return None
        used.add((mr, mc))
        markers.append(
            {
                "row": mr,
                "col": mc,
                "color": SIDE_TO_COLOR[side],
                "side": side,
                "attach": stamp["attach"],
                "distance": dist,
            }
        )
    return markers


def _anti_shortcut_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    frame: tuple[int, int, int, int],
    template: np.ndarray,
    template_origin: tuple[int, int],
    markers: list[dict],
    require_overlap: bool,
) -> bool:
    if len({m["side"] for m in markers}) < 3:
        return False
    if np.array_equal(inp, expected):
        return False
    if np.array_equal(_render(inp, frame, template, template_origin, markers, rotate=False, strict=False), expected):
        return False
    if np.array_equal(_render(inp, frame, template, template_origin, markers, erase_template=False), expected):
        return False
    if require_overlap:
        if np.array_equal(_render(inp, frame, template, template_origin, markers, sort_near_last=False), expected):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    require_overlap = random.random() < 0.7
    marker_count = random.choices([3, 4, 5, 6], weights=[1, 3, 3, 2])[0]
    if require_overlap and marker_count == 3:
        marker_count = 4

    H = random.randint(22, 28)
    W = random.randint(22, 28)
    frame_h = random.randint(8, min(12, H - 2 * marker_count - 1))
    frame_w = random.randint(8, min(12, W - 2 * marker_count - 1))
    r0 = random.randint(marker_count, H - frame_h - marker_count)
    c0 = random.randint(marker_count, W - frame_w - marker_count)
    frame = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)

    grid = np.zeros((H, W), dtype=int)
    _draw_frame(grid, frame)

    template = _random_template()
    template_origin = _place_template(grid, frame, template)

    side_seq = _marker_side_sequence(marker_count, require_overlap=require_overlap)
    layout = _build_stamp_layout(template, frame, side_seq, require_overlap=require_overlap)
    if layout is None:
        return None

    markers = _materialize_markers(grid.shape, frame, layout)
    if markers is None:
        return None
    for marker in markers:
        if int(grid[marker["row"], marker["col"]]) != BG:
            return None
        grid[marker["row"], marker["col"]] = marker["color"]

    expected = _render(grid, frame, template, template_origin, markers)
    if not np.array_equal(main(grid), expected):
        return None
    if not _anti_shortcut_checks(grid, expected, frame, template, template_origin, markers, require_overlap):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
