from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7

DASH_COLORS = (BLUE, GREEN, ORANGE)
SHELL_COLORS = (RED, YELLOW, PINK)
MOTIF_CENTER_COLOR = GREY

GEN_HEIGHT_RANGE = (15, 21)
GEN_WIDTH_RANGE = (18, 27)
GEN_MOTIF_COUNT_CHOICES = (2, 3)
GEN_SEGMENT_COUNT_CHOICES = (3, 4)
GEN_COLORED_LENGTH_CHOICES = (3, 5)
GEN_ATTEMPTS = 400

_ALL_SHELL_OFFSETS = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)

MOTIF_LIBRARY = (
    ((-1, 0), (0, -1), (0, 1), (1, 0)),
    ((-1, -1), (-1, 1), (1, -1), (1, 1)),
    ((-1, -1), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 1)),
    ((-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 0), (1, 1)),
    _ALL_SHELL_OFFSETS,
)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _shell_mask_from_offsets(offsets: tuple[tuple[int, int], ...]) -> np.ndarray:
    mask = np.zeros((3, 3), dtype=bool)
    for dr, dc in offsets:
        mask[1 + dr, 1 + dc] = True
    return mask


def _is_valid_motif_mask(mask: np.ndarray) -> bool:
    shell_mask = np.asarray(mask, dtype=bool).copy()
    if shell_mask.shape != (3, 3):
        return False
    shell_mask[1, 1] = False
    count = int(shell_mask.sum())
    if count < 4:
        return False
    rotational = np.array_equal(shell_mask, np.rot90(shell_mask, 1))
    mirrored = np.array_equal(shell_mask, np.fliplr(shell_mask)) and np.array_equal(shell_mask, np.flipud(shell_mask))
    return rotational or mirrored


def _find_complete_motifs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    motifs: list[dict] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != MOTIF_CENTER_COLOR:
                continue
            patch = g[r - 1 : r + 2, c - 1 : c + 2]
            colors = {
                int(v)
                for v in patch.flat
                if int(v) not in (BLACK, MOTIF_CENTER_COLOR)
            }
            if len(colors) != 1:
                continue
            shell_color = next(iter(colors))
            shell_mask = patch == shell_color
            shell_mask[1, 1] = False
            if not _is_valid_motif_mask(shell_mask):
                continue
            allowed = (patch == BLACK) | (patch == MOTIF_CENTER_COLOR) | (patch == shell_color)
            if not bool(np.all(allowed)):
                continue
            motifs.append(
                {
                    "center": (r, c),
                    "shell_color": int(shell_color),
                    "shell_mask": shell_mask.copy(),
                }
            )
    motifs.sort(key=lambda item: (item["center"][0], item["center"][1]))
    return motifs


def _find_dashed_segments(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    segments: list[dict] = []
    for r in range(h):
        c = 0
        while c < w:
            color = int(g[r, c])
            if color == BLACK:
                c += 1
                continue
            if c >= 2 and int(g[r, c - 1]) == BLACK and int(g[r, c - 2]) == color:
                c += 1
                continue
            cols = [c]
            end_col = c
            while end_col + 2 < w and int(g[r, end_col + 1]) == BLACK and int(g[r, end_col + 2]) == color:
                cols.append(end_col + 2)
                end_col += 2
            if len(cols) >= 3:
                gap_cols = [left + 1 for left in cols[:-1]]
                segments.append(
                    {
                        "row": r,
                        "color": color,
                        "cols": tuple(cols),
                        "gap_cols": tuple(gap_cols),
                        "mid_coord": (r, cols[len(cols) // 2]),
                    }
                )
                c = end_col + 1
            else:
                c += 1
    segments.sort(key=lambda item: (item["row"], item["cols"][0]))
    return segments


def _nearest_motif(segment: dict, motifs: list[dict]) -> dict:
    distances = [_manhattan(segment["mid_coord"], motif["center"]) for motif in motifs]
    best = min(distances)
    assert distances.count(best) == 1, "each dashed segment must have a unique nearest motif"
    return motifs[distances.index(best)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each horizontal dashed segment's one-cell gaps using the shell color of its nearest complete symmetric motif, then erase the motifs.

    Input structure:
    - A black background contains several horizontal dashed segments made of one repeated non-black color, with one black cell between adjacent same-color pixels.
    - The grid also contains 2-3 complete 3x3 motifs: each motif has a grey center and a same-color shell arranged symmetrically inside the surrounding 3x3 patch.
    - Motifs and dashed segments are spatially separated, and each dashed segment has a unique nearest motif by Manhattan distance from its median dash pixel to motif center.

    Rule:
    - Detect every complete 3x3 motif whose shell has at least four same-color pixels and is either 90-degree rotationally symmetric or symmetric across both horizontal and vertical axes.
    - Detect every horizontal dashed segment with at least three same-color pixels laid out at alternating columns.
    - For each segment, locate its median colored pixel, find the uniquely nearest motif center by Manhattan distance, and use that motif's shell color to fill every one-cell black gap inside the segment.
    - After all segments are filled, erase every complete motif by turning its entire 3x3 footprint black.

    Key insight:
    - The complete motifs are disposable remote color instructions: they are identified only to supply shell colors to nearby dashed segments and then disappear from the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert bg == BLACK, "expected black background"

    motifs = _find_complete_motifs(g)
    segments = _find_dashed_segments(g)
    assert motifs, "expected at least one complete motif"
    assert segments, "expected at least one dashed segment"

    out = g.copy()
    for segment in segments:
        motif = _nearest_motif(segment, motifs)
        fill_color = int(motif["shell_color"])
        r = int(segment["row"])
        for gap_c in segment["gap_cols"]:
            assert int(g[r, gap_c]) == BLACK
            assert int(g[r, gap_c - 1]) == int(segment["color"])
            assert int(g[r, gap_c + 1]) == int(segment["color"])
            out[r, gap_c] = fill_color

    for motif in motifs:
        r, c = motif["center"]
        out[r - 1 : r + 2, c - 1 : c + 2] = BLACK
    return out


def _paint_motif(grid: np.ndarray, center: tuple[int, int], shell_color: int, offsets: tuple[tuple[int, int], ...]) -> None:
    r, c = center
    grid[r, c] = MOTIF_CENTER_COLOR
    for dr, dc in offsets:
        grid[r + dr, c + dc] = int(shell_color)


def _reserve_box(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    h, w = mask.shape
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(h - 1, int(r1))
    cc1 = min(w - 1, int(c1))
    if rr0 <= rr1 and cc0 <= cc1:
        mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _choose_distinct_dash_colors(count: int) -> list[int]:
    while True:
        colors = [random.choice(DASH_COLORS) for _ in range(count)]
        if len(set(colors)) >= min(2, count):
            return colors


def _place_motifs(height: int, width: int, motif_count: int) -> tuple[list[dict], np.ndarray] | None:
    shell_colors = random.sample(list(SHELL_COLORS), motif_count)
    masks = random.sample(list(MOTIF_LIBRARY), motif_count)
    centers: list[tuple[int, int]] = []
    reserved = np.zeros((height, width), dtype=bool)

    candidate_centers = [
        (r, c)
        for r in range(2, height - 2)
        for c in range(2, width - 2)
    ]
    random.shuffle(candidate_centers)
    for shell_color, offsets in zip(shell_colors, masks):
        placed = False
        for r, c in candidate_centers:
            if reserved[r - 1 : r + 2, c - 1 : c + 2].any():
                continue
            centers.append((r, c))
            _reserve_box(reserved, r - 2, c - 2, r + 2, c + 2)
            placed = True
            break
        if not placed:
            return None

    motifs = []
    for center, shell_color, offsets in zip(centers, shell_colors, masks):
        motifs.append(
            {
                "center": center,
                "shell_color": int(shell_color),
                "offsets": tuple(offsets),
            }
        )
    motifs.sort(key=lambda item: (item["center"][0], item["center"][1]))
    return motifs, reserved


def _assignment_vector(motif_count: int, segment_count: int) -> list[int]:
    assignments = list(range(motif_count))
    while len(assignments) < segment_count:
        assignments.append(random.randrange(motif_count))
    random.shuffle(assignments)
    return assignments


def _segment_candidates(
    height: int,
    width: int,
    motifs: list[dict],
    reserved: np.ndarray,
    target_motif_index: int,
    colored_len: int,
) -> list[dict]:
    candidates: list[dict] = []
    span = 2 * colored_len - 1
    target_center = motifs[target_motif_index]["center"]
    for row in range(height):
        for start in range(0, width - span + 1):
            end = start + span - 1
            pad_r0 = row - 1
            pad_r1 = row + 1
            pad_c0 = start - 1
            pad_c1 = end + 1
            rr0 = max(0, pad_r0)
            rr1 = min(height - 1, pad_r1)
            cc0 = max(0, pad_c0)
            cc1 = min(width - 1, pad_c1)
            if reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            cols = tuple(start + 2 * i for i in range(colored_len))
            mid_coord = (row, cols[len(cols) // 2])
            distances = [_manhattan(mid_coord, motif["center"]) for motif in motifs]
            best = min(distances)
            if distances.count(best) != 1:
                continue
            nearest_index = distances.index(best)
            if nearest_index != target_motif_index:
                continue
            ordered = sorted(distances)
            if len(ordered) > 1 and ordered[1] - ordered[0] < 2:
                continue
            mid_bias = abs(mid_coord[0] - target_center[0]) + abs(mid_coord[1] - target_center[1])
            candidates.append(
                {
                    "row": row,
                    "start": start,
                    "cols": cols,
                    "gap_cols": tuple(col + 1 for col in cols[:-1]),
                    "mid_coord": mid_coord,
                    "distance": best,
                    "bias": mid_bias,
                }
            )
    random.shuffle(candidates)
    candidates.sort(key=lambda item: (item["distance"], item["bias"], item["row"], item["start"]))
    return candidates


def _place_segments(height: int, width: int, motifs: list[dict], reserved: np.ndarray) -> list[dict] | None:
    motif_count = len(motifs)
    segment_count = random.choice(GEN_SEGMENT_COUNT_CHOICES)
    assignments = _assignment_vector(motif_count, segment_count)
    dash_colors = _choose_distinct_dash_colors(segment_count)
    colored_lengths = [random.choice(GEN_COLORED_LENGTH_CHOICES) for _ in range(segment_count)]
    if len(set(colored_lengths)) == 1 and segment_count >= 2:
        colored_lengths[random.randrange(segment_count)] = 3 if colored_lengths[0] == 5 else 5

    segments: list[dict] = []
    local_reserved = reserved.copy()
    for target_motif_index, dash_color, colored_len in zip(assignments, dash_colors, colored_lengths):
        candidates = _segment_candidates(height, width, motifs, local_reserved, target_motif_index, colored_len)
        if not candidates:
            return None
        chosen = random.choice(candidates[: min(12, len(candidates))])
        segment = {
            "row": int(chosen["row"]),
            "color": int(dash_color),
            "cols": tuple(chosen["cols"]),
            "gap_cols": tuple(chosen["gap_cols"]),
            "mid_coord": tuple(chosen["mid_coord"]),
            "target_motif_index": int(target_motif_index),
        }
        segments.append(segment)
        _reserve_box(local_reserved, segment["row"] - 1, segment["cols"][0] - 1, segment["row"] + 1, segment["cols"][-1] + 1)

    if len({segment["target_motif_index"] for segment in segments}) != motif_count:
        return None
    return segments


def _render_input(height: int, width: int, motifs: list[dict], segments: list[dict]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for motif in motifs:
        _paint_motif(grid, motif["center"], motif["shell_color"], motif["offsets"])
    for segment in segments:
        row = int(segment["row"])
        for col in segment["cols"]:
            grid[row, col] = int(segment["color"])
    return grid


def _render_expected(height: int, width: int, motifs: list[dict], segments: list[dict]) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    for segment in segments:
        row = int(segment["row"])
        fill_color = int(motifs[segment["target_motif_index"]]["shell_color"])
        for col in segment["cols"]:
            out[row, col] = int(segment["color"])
        for col in segment["gap_cols"]:
            out[row, col] = fill_color
    return out


def _validate_instance(inp: np.ndarray, expected: np.ndarray, motifs: list[dict], segments: list[dict]) -> bool:
    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False

    found_motifs = _find_complete_motifs(inp)
    found_segments = _find_dashed_segments(inp)
    if len(found_motifs) != len(motifs) or len(found_segments) != len(segments):
        return False

    motif_by_center = {tuple(motif["center"]): motif for motif in found_motifs}
    for motif in motifs:
        center = tuple(motif["center"])
        if center not in motif_by_center:
            return False
        found = motif_by_center[center]
        if int(found["shell_color"]) != int(motif["shell_color"]):
            return False
        if not np.array_equal(found["shell_mask"], _shell_mask_from_offsets(motif["offsets"])):
            return False
        r, c = center
        if np.any(expected[r - 1 : r + 2, c - 1 : c + 2] != BLACK):
            return False

    found_by_key = {
        (int(segment["row"]), tuple(segment["cols"]), int(segment["color"])): segment
        for segment in found_segments
    }
    for segment in segments:
        key = (int(segment["row"]), tuple(segment["cols"]), int(segment["color"]))
        if key not in found_by_key:
            return False
        found = found_by_key[key]
        motif = _nearest_motif(found, found_motifs)
        expected_color = int(motifs[segment["target_motif_index"]]["shell_color"])
        if int(motif["shell_color"]) != expected_color:
            return False
        row = int(segment["row"])
        for col in segment["cols"]:
            if int(expected[row, col]) != int(segment["color"]):
                return False
        for col in segment["gap_cols"]:
            if int(expected[row, col]) != expected_color:
                return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        height = random.randint(*GEN_HEIGHT_RANGE)
        width = random.randint(*GEN_WIDTH_RANGE)
        motif_count = random.choice(GEN_MOTIF_COUNT_CHOICES)

        motif_data = _place_motifs(height, width, motif_count)
        if motif_data is None:
            continue
        motifs, reserved = motif_data

        segments = _place_segments(height, width, motifs, reserved)
        if segments is None:
            continue

        inp = _render_input(height, width, motifs, segments)
        expected = _render_expected(height, width, motifs, segments)

        if _validate_instance(inp, expected, motifs, segments):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
