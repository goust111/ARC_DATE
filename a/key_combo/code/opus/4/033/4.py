from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask


BG = Color.BLACK
BODY_COLORS = (Color.BLUE, Color.GREEN, Color.GREY)
MARKER_COLORS = (Color.RED, Color.YELLOW, Color.ORANGE)

GRID_MIN = 18
GRID_MAX = 24
RECT_SIDE_MIN = 3
RECT_SIDE_MAX = 6
GAP_MIN = 2
GAP_MAX = 4
GEN_MAX_ATTEMPTS = 400


def _object_info(grid: np.ndarray) -> list[dict]:
    """Return each marked rectangle as a single non-black connected component."""
    objects = []
    for coords in components_mask(grid != BG, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch != BG), "each object must occupy a solid rectangle bbox"
        body_colors = [int(v) for v in np.unique(patch) if int(v) in BODY_COLORS]
        assert len(body_colors) == 1, "each object must contain exactly one body color"
        objects.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "body_color": body_colors[0],
                "patch": patch,
            }
        )
    return objects


def _majority_color(values: list[int]) -> int:
    counts = Counter(values)
    assert counts, "expected at least one marker vote"
    ordered = counts.most_common()
    if len(ordered) > 1:
        assert ordered[0][1] > ordered[1][1], "marker votes must have a strict majority"
    return int(ordered[0][0])


def _pair_info(grid: np.ndarray, objects: list[dict]) -> list[dict]:
    """Pair same-color rectangles, find their gap, and compute the bridge vote."""
    groups: dict[int, list[dict]] = {}
    for obj in objects:
        groups.setdefault(int(obj["body_color"]), []).append(obj)

    pairs = []
    for body_color, group in groups.items():
        assert len(group) == 2, "each body color must appear on exactly two rectangles"
        a, b = group
        ar0, ac0, ar1, ac1 = a["bbox"]
        br0, bc0, br1, bc1 = b["bbox"]

        if ar0 == br0 and ar1 == br1:
            left, right = sorted(group, key=lambda item: item["bbox"][1])
            lr0, lc0, lr1, lc1 = left["bbox"]
            rr0, rc0, rr1, rc1 = right["bbox"]
            gap_width = rc0 - lc1 - 1
            assert GAP_MIN <= gap_width <= GAP_MAX
            gap_bbox = (lr0, lc1 + 1, lr1, rc0 - 1)
            left_edge = grid[lr0 : lr1 + 1, lc1]
            right_edge = grid[rr0 : rr1 + 1, rc0]
            orientation = "horizontal"
        else:
            assert ac0 == bc0 and ac1 == bc1, "same-color rectangles must align horizontally or vertically"
            top, bottom = sorted(group, key=lambda item: item["bbox"][0])
            tr0, tc0, tr1, tc1 = top["bbox"]
            br0, bc0, br1, bc1 = bottom["bbox"]
            gap_height = br0 - tr1 - 1
            assert GAP_MIN <= gap_height <= GAP_MAX
            gap_bbox = (tr1 + 1, tc0, br0 - 1, tc1)
            left_edge = grid[tr1, tc0 : tc1 + 1]
            right_edge = grid[br0, bc0 : bc1 + 1]
            orientation = "vertical"

        gr0, gc0, gr1, gc1 = gap_bbox
        assert np.all(grid[gr0 : gr1 + 1, gc0 : gc1 + 1] == BG), "gap must be black before bridging"

        votes = [int(v) for v in np.concatenate((left_edge, right_edge)) if int(v) in MARKER_COLORS]
        winner = _majority_color(votes)
        center = ((gr0 + gr1) / 2.0, (gc0 + gc1) / 2.0)
        pairs.append(
            {
                "body_color": body_color,
                "gap_bbox": gap_bbox,
                "bridge_color": winner,
                "orientation": orientation,
                "center": center,
            }
        )

    return sorted(pairs, key=lambda item: (item["center"][0], item["center"][1]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the gap between each same-color rectangle pair with the majority inward-edge marker color, then erase the markers.

    Input structure:
    - The grid is mostly black background.
    - There are 2-3 color-distinct pairs of filled axis-aligned rectangles using blue, green, or grey body colors.
    - In each pair, the two rectangles share the same row span or column span, are separated by a 2-4 cell black gap, and only their gap-facing edge may contain red/yellow/orange marker pixels.

    Rule:
    - Treat each non-black connected component as one marked rectangle and recover its full bbox.
    - Pair rectangles by shared body color, then determine whether they face each other horizontally or vertically.
    - Collect the red/yellow/orange marker pixels on the two inward-facing edges, choose the strict majority color, and fill the whole black gap rectangle with that color.
    - Finally restore every rectangle bbox to its body color, removing the marker pixels.

    Key insight:
    - The bridge geometry comes from the two aligned rectangles, but the bridge color is decided only by a majority vote of the marker pixels on the two edges that face the gap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _object_info(g)
    pairs = _pair_info(g, objects)

    out = g.copy()
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(obj["body_color"])

    for pair in pairs:
        r0, c0, r1, c1 = pair["gap_bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(pair["bridge_color"])

    return out


def _sample_winners(num_pairs: int) -> list[int]:
    if num_pairs == 2:
        return random.sample(MARKER_COLORS, 2)
    winners = list(MARKER_COLORS)
    random.shuffle(winners)
    return winners


def _base_vote_counts(winner: int, mode: str) -> tuple[Counter, Counter]:
    losers = [color for color in MARKER_COLORS if color != winner]
    a, b = random.sample(losers, 2)
    if mode == "left_misleading":
        left = Counter({a: 2, winner: 1})
        right = Counter({winner: 2, b: 1})
    elif mode == "right_misleading":
        left = Counter({winner: 2, a: 1})
        right = Counter({b: 2, winner: 1})
    else:
        left = Counter({winner: 2, a: 1})
        right = Counter({b: 1})
    return left, right


def _pad_vote_counts(left: Counter, right: Counter, edge_len: int, winner: int) -> tuple[Counter, Counter]:
    if sum(left.values()) > edge_len or sum(right.values()) > edge_len:
        raise AssertionError("edge is too short for vote layout")
    combined = left + right
    assert _majority_color(list(combined.elements())) == winner
    return left, right


def _counts_to_edge_slots(edge_len: int, counts: Counter) -> list[int | None]:
    slots = [None] * edge_len
    positions = list(range(edge_len))
    random.shuffle(positions)
    idx = 0
    for color, n in counts.items():
        for _ in range(n):
            slots[positions[idx]] = int(color)
            idx += 1
    return slots


def _build_votes(edge_len: int, winner: int, mode: str) -> dict:
    left_counts, right_counts = _base_vote_counts(winner, mode)
    left_counts, right_counts = _pad_vote_counts(left_counts, right_counts, edge_len, winner)
    combined = left_counts + right_counts
    assert sum(combined.values()) >= 3

    return {
        "left_slots": _counts_to_edge_slots(edge_len, left_counts),
        "right_slots": _counts_to_edge_slots(edge_len, right_counts),
        "left_winner": _majority_color(list(left_counts.elements())),
        "right_winner": _majority_color(list(right_counts.elements())),
        "winner": _majority_color(list(combined.elements())),
        "combined_counts": combined,
    }


def _orientation_plan(num_pairs: int) -> list[str]:
    orientations = ["horizontal", "vertical"]
    if num_pairs == 3:
        orientations.append(random.choice(orientations))
    random.shuffle(orientations)
    return orientations


def _vote_modes(num_pairs: int) -> list[str]:
    modes = ["left_misleading", "right_misleading"]
    if num_pairs == 3:
        modes.append("plain")
    random.shuffle(modes)
    return modes


def _pair_footprint(pair: dict) -> tuple[int, int]:
    if pair["orientation"] == "horizontal":
        return pair["rect_h"], 2 * pair["rect_w"] + pair["gap"]
    return 2 * pair["rect_h"] + pair["gap"], pair["rect_w"]


def _try_place_pairs(grid_shape: tuple[int, int], pairs: list[dict]) -> bool:
    h, w = grid_shape
    occupancy = np.zeros((h, w), dtype=bool)
    for pair in sorted(pairs, key=lambda item: _pair_footprint(item)[0] * _pair_footprint(item)[1], reverse=True):
        ph, pw = _pair_footprint(pair)
        candidates = [
            (r, c)
            for r in range(1, h - ph)
            for c in range(1, w - pw)
            if not occupancy[max(0, r - 1) : min(h, r + ph + 1), max(0, c - 1) : min(w, c + pw + 1)].any()
        ]
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        pair["anchor"] = (r0, c0)
        occupancy[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)] = True
    return True


def _paint_pair(grid: np.ndarray, pair: dict) -> None:
    r0, c0 = pair["anchor"]
    body = int(pair["body_color"])
    rect_h = int(pair["rect_h"])
    rect_w = int(pair["rect_w"])
    gap = int(pair["gap"])
    if pair["orientation"] == "horizontal":
        left_c0 = c0
        right_c0 = c0 + rect_w + gap
        grid[r0 : r0 + rect_h, left_c0 : left_c0 + rect_w] = body
        grid[r0 : r0 + rect_h, right_c0 : right_c0 + rect_w] = body
        for idx, color in enumerate(pair["votes"]["left_slots"]):
            if color is not None:
                grid[r0 + idx, left_c0 + rect_w - 1] = int(color)
        for idx, color in enumerate(pair["votes"]["right_slots"]):
            if color is not None:
                grid[r0 + idx, right_c0] = int(color)
    else:
        top_r0 = r0
        bottom_r0 = r0 + rect_h + gap
        grid[top_r0 : top_r0 + rect_h, c0 : c0 + rect_w] = body
        grid[bottom_r0 : bottom_r0 + rect_h, c0 : c0 + rect_w] = body
        for idx, color in enumerate(pair["votes"]["left_slots"]):
            if color is not None:
                grid[top_r0 + rect_h - 1, c0 + idx] = int(color)
        for idx, color in enumerate(pair["votes"]["right_slots"]):
            if color is not None:
                grid[bottom_r0, c0 + idx] = int(color)


def _expected_output(shape: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for pair in pairs:
        r0, c0 = pair["anchor"]
        rect_h = int(pair["rect_h"])
        rect_w = int(pair["rect_w"])
        gap = int(pair["gap"])
        body = int(pair["body_color"])
        if pair["orientation"] == "horizontal":
            left_c0 = c0
            right_c0 = c0 + rect_w + gap
            out[r0 : r0 + rect_h, left_c0 : left_c0 + rect_w] = body
            out[r0 : r0 + rect_h, right_c0 : right_c0 + rect_w] = body
            out[r0 : r0 + rect_h, left_c0 + rect_w : right_c0] = int(pair["votes"]["winner"])
        else:
            top_r0 = r0
            bottom_r0 = r0 + rect_h + gap
            out[top_r0 : top_r0 + rect_h, c0 : c0 + rect_w] = body
            out[bottom_r0 : bottom_r0 + rect_h, c0 : c0 + rect_w] = body
            out[top_r0 + rect_h : bottom_r0, c0 : c0 + rect_w] = int(pair["votes"]["winner"])
    return out


def _semantic_checks(grid: np.ndarray, expected: np.ndarray, pairs: list[dict]) -> bool:
    if not np.array_equal(main(grid), expected):
        return False
    if len({pair["body_color"] for pair in pairs}) != len(pairs):
        return False
    if len({pair["votes"]["winner"] for pair in pairs}) < min(2, len(pairs)):
        return False
    if not any(pair["votes"]["winner"] != pair["votes"]["left_winner"] for pair in pairs):
        return False
    if not any(pair["votes"]["winner"] != pair["votes"]["right_winner"] for pair in pairs):
        return False
    if not {"horizontal", "vertical"}.issubset({pair["orientation"] for pair in pairs}):
        return False
    detected = _pair_info(grid, _object_info(grid))
    if len(detected) != len(pairs):
        return False
    detected_map = {int(item["body_color"]): item for item in detected}
    for pair in pairs:
        got = detected_map.get(int(pair["body_color"]))
        if got is None:
            return False
        if int(got["bridge_color"]) != int(pair["votes"]["winner"]):
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    grid_h = random.randint(GRID_MIN, GRID_MAX)
    grid_w = random.randint(GRID_MIN, GRID_MAX)
    num_pairs = random.choice((2, 3))
    body_colors = random.sample(BODY_COLORS, num_pairs)
    winners = _sample_winners(num_pairs)
    orientations = _orientation_plan(num_pairs)
    modes = _vote_modes(num_pairs)

    pairs = []
    for i in range(num_pairs):
        rect_h = random.randint(RECT_SIDE_MIN, RECT_SIDE_MAX)
        rect_w = random.randint(RECT_SIDE_MIN, RECT_SIDE_MAX)
        gap = random.randint(GAP_MIN, GAP_MAX)
        edge_len = rect_h if orientations[i] == "horizontal" else rect_w
        votes = _build_votes(edge_len, winners[i], modes[i])
        pairs.append(
            {
                "body_color": body_colors[i],
                "orientation": orientations[i],
                "mode": modes[i],
                "rect_h": rect_h,
                "rect_w": rect_w,
                "gap": gap,
                "votes": votes,
            }
        )

    if not _try_place_pairs((grid_h, grid_w), pairs):
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for pair in pairs:
        _paint_pair(grid, pair)

    expected = _expected_output(grid.shape, pairs)
    if not _semantic_checks(grid, expected, pairs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
