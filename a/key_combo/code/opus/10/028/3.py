from __future__ import annotations

import itertools
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
MAROON = 9

FRAME_COLORS = [RED, GREEN, BLUE, YELLOW, PINK]
DIRECTION_TO_SEED = {
    "up": ORANGE,
    "left": RED,
    "down": MAROON,
    "right": BLUE,
}
SEED_TO_DIRECTION = {color: direction for direction, color in DIRECTION_TO_SEED.items()}

GEN_FRAME_COUNT_CHOICES = [4, 5]
GEN_DIRECTIONAL_COUNT_RANGE = range(1, 7)
GEN_IGNORED_COUNT_RANGE = range(2, 5)
GEN_INTERIOR_HEIGHT_RANGE = range(4, 6)
GEN_INTERIOR_WIDTH_RANGE = range(4, 6)
GEN_ROW_GAP_RANGE = range(2, 4)
GEN_COL_GAP_RANGE = range(2, 4)
GEN_MARGIN_RANGE = range(0, 3)
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_for_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == int(color)
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _ring_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if len(coords) < 8:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None
    boundary = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    pts = set(coords)
    if pts != boundary:
        return None
    return r0, c0, r1, c1


def _marker_partition(
    marker_coords: list[tuple[int, int]],
    seed: tuple[int, int],
    direction: str,
) -> tuple[int, int, int]:
    sr, sc = seed
    directional = 0
    aligned_ignored = 0
    opposite_ignored = 0
    for r, c in marker_coords:
        if direction == "up":
            if r < sr:
                directional += 1
            elif r == sr:
                aligned_ignored += 1
            else:
                opposite_ignored += 1
        elif direction == "down":
            if r > sr:
                directional += 1
            elif r == sr:
                aligned_ignored += 1
            else:
                opposite_ignored += 1
        elif direction == "left":
            if c < sc:
                directional += 1
            elif c == sc:
                aligned_ignored += 1
            else:
                opposite_ignored += 1
        else:
            if c > sc:
                directional += 1
            elif c == sc:
                aligned_ignored += 1
            else:
                opposite_ignored += 1
    return directional, aligned_ignored, opposite_ignored


def _extract_frames(grid: np.ndarray, *, strict: bool) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    frames: list[dict] = []
    marker_cover = np.zeros(g.shape, dtype=int)
    for color in sorted(set(map(int, g.ravel())) - {bg}):
        for comp in _components_for_color(g, color):
            ring = _ring_bbox(comp)
            if ring is None:
                continue
            r0, c0, r1, c1 = ring
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            seed_local = np.argwhere((inner != bg) & (inner != GREY))
            if strict:
                assert seed_local.shape[0] == 1, "each frame interior must contain exactly one seed"
            if seed_local.shape[0] != 1:
                continue

            seed_r = int(seed_local[0, 0] + r0 + 1)
            seed_c = int(seed_local[0, 1] + c0 + 1)
            seed_color = int(g[seed_r, seed_c])
            if strict:
                assert seed_color in SEED_TO_DIRECTION, "seed color must encode a known direction"
                assert seed_color != color, "seed color must differ from the frame color"
            if seed_color not in SEED_TO_DIRECTION or seed_color == color:
                continue

            marker_local = np.argwhere(inner == GREY)
            marker_coords = [(int(r + r0 + 1), int(c + c0 + 1)) for r, c in marker_local]
            if strict:
                assert len(marker_coords) >= 2, "each frame must contain at least two gray markers"
            direction = SEED_TO_DIRECTION[seed_color]
            directional, aligned_ignored, opposite_ignored = _marker_partition(
                marker_coords,
                (seed_r, seed_c),
                direction,
            )
            if strict:
                assert directional >= 1, "each frame needs at least one directional marker"

            local_marker_mask = inner == GREY
            marker_cover[r0 + 1 : r1, c0 + 1 : c1] += local_marker_mask.astype(int)
            frames.append(
                {
                    "frame_color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "seed": (seed_r, seed_c),
                    "seed_color": seed_color,
                    "direction": direction,
                    "directional_count": directional,
                    "total_markers": len(marker_coords),
                    "aligned_ignored": aligned_ignored,
                    "opposite_ignored": opposite_ignored,
                }
            )

    if strict:
        assert frames, "expected at least one hollow frame"
        counts = [frame["directional_count"] for frame in frames]
        assert len(counts) == len(set(counts)), "directional counts must be unique"
        assert len({frame["frame_color"] for frame in frames}) == len(frames), "frame colors must be unique"
        if np.any(g == GREY):
            assert np.all(marker_cover[g == GREY] == 1), "every gray marker must belong to exactly one frame"
    return frames


def _expected_output(frame_specs: list[dict], bg: int = BLACK) -> np.ndarray:
    ordered = sorted(frame_specs, key=lambda item: item["directional_count"])
    width = max(item["directional_count"] for item in ordered)
    out = np.full((len(ordered), width), bg, dtype=int)
    for row, item in enumerate(ordered):
        out[row, : item["directional_count"]] = int(item["frame_color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Summarize each frame with a row whose length is its seed-directed gray-marker count.

    Input structure:
    - Black background with several non-overlapping hollow rectangular frames in distinct colors.
    - Each frame interior contains gray marker pixels and exactly one non-gray seed pixel.
    - Seed colors encode directions: orange=up, red=left, maroon=down, blue=right.

    Rule:
    - Detect every hollow frame and find its unique interior seed pixel.
    - Count only the gray markers strictly on the encoded side of that seed within the same frame.
    - Sort the frames by this directional count from smallest to largest.
    - Output one row per frame, filled with the frame color for the first K cells and black elsewhere.

    Key insight:
    The seed color does not mark a destination; it chooses the half-space inside the frame whose gray markers are valid, and that filtered count controls both row order and row length.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames = _extract_frames(g, strict=True)
    return _expected_output(frames, bg=bg)


def _local_partitions(
    ih: int,
    iw: int,
    seed_r: int,
    seed_c: int,
    direction: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    all_cells = [(r, c) for r in range(ih) for c in range(iw) if (r, c) != (seed_r, seed_c)]
    if direction == "up":
        counted = [(r, c) for r, c in all_cells if r < seed_r]
        aligned = [(r, c) for r, c in all_cells if r == seed_r]
        opposite = [(r, c) for r, c in all_cells if r > seed_r]
    elif direction == "down":
        counted = [(r, c) for r, c in all_cells if r > seed_r]
        aligned = [(r, c) for r, c in all_cells if r == seed_r]
        opposite = [(r, c) for r, c in all_cells if r < seed_r]
    elif direction == "left":
        counted = [(r, c) for r, c in all_cells if c < seed_c]
        aligned = [(r, c) for r, c in all_cells if c == seed_c]
        opposite = [(r, c) for r, c in all_cells if c > seed_c]
    else:
        counted = [(r, c) for r, c in all_cells if c > seed_c]
        aligned = [(r, c) for r, c in all_cells if c == seed_c]
        opposite = [(r, c) for r, c in all_cells if c < seed_c]
    excluded = aligned + opposite
    return counted, aligned, opposite, excluded


def _build_frame_patch(
    frame_color: int,
    direction: str,
    directional_count: int,
    ignored_count: int,
) -> tuple[np.ndarray, dict]:
    seed_color = DIRECTION_TO_SEED[direction]
    for _ in range(200):
        ih = random.choice(list(GEN_INTERIOR_HEIGHT_RANGE))
        iw = random.choice(list(GEN_INTERIOR_WIDTH_RANGE))
        candidates = []
        for seed_r in range(ih):
            for seed_c in range(iw):
                counted, aligned, opposite, excluded = _local_partitions(ih, iw, seed_r, seed_c, direction)
                if (
                    len(counted) >= directional_count
                    and len(aligned) >= 1
                    and len(opposite) >= 1
                    and len(excluded) >= ignored_count
                ):
                    candidates.append((seed_r, seed_c, counted, aligned, opposite, excluded))
        if not candidates:
            continue

        seed_r, seed_c, counted, aligned, opposite, excluded = random.choice(candidates)
        directional_markers = random.sample(counted, directional_count)
        ignored_markers = [random.choice(aligned), random.choice(opposite)]
        remaining_excluded = [cell for cell in excluded if cell not in set(ignored_markers)]
        extra_needed = ignored_count - 2
        if len(remaining_excluded) < extra_needed:
            continue
        ignored_markers.extend(random.sample(remaining_excluded, extra_needed))

        patch = np.full((ih + 2, iw + 2), BLACK, dtype=int)
        patch[0, :] = frame_color
        patch[-1, :] = frame_color
        patch[:, 0] = frame_color
        patch[:, -1] = frame_color
        patch[seed_r + 1, seed_c + 1] = seed_color
        for r, c in directional_markers + ignored_markers:
            patch[r + 1, c + 1] = GREY

        return patch, {
            "frame_color": frame_color,
            "direction": direction,
            "directional_count": directional_count,
            "total_markers": directional_count + ignored_count,
        }
    raise RuntimeError("failed to build frame patch")


def _sample_directions(n_frames: int) -> list[str]:
    directions = ["up", "left", "down", "right"]
    if n_frames == 5:
        directions.append(random.choice(directions))
    random.shuffle(directions)
    return directions


def _sample_counts(n_frames: int) -> tuple[list[int], list[int]]:
    for _ in range(300):
        directional = random.sample(list(GEN_DIRECTIONAL_COUNT_RANGE), n_frames)
        ignored = [random.choice(list(GEN_IGNORED_COUNT_RANGE)) for _ in range(n_frames)]
        totals = [a + b for a, b in zip(directional, ignored)]
        if len(set(totals)) != n_frames:
            continue
        if len(set(ignored)) < 2:
            continue
        dir_order = sorted(range(n_frames), key=lambda i: directional[i])
        total_order = sorted(range(n_frames), key=lambda i: totals[i])
        if dir_order != total_order:
            return directional, ignored
    raise RuntimeError("failed to sample distinguishing counts")


def _assign_frame_colors(directions: list[str]) -> list[int]:
    n_frames = len(directions)
    valid_assignments = [
        list(colors)
        for colors in itertools.permutations(FRAME_COLORS, n_frames)
        if all(colors[i] != DIRECTION_TO_SEED[directions[i]] for i in range(n_frames))
    ]
    if valid_assignments:
        return random.choice(valid_assignments)
    raise RuntimeError("failed to assign frame colors")


def _compose_scene(patches: list[np.ndarray]) -> np.ndarray | None:
    slots = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    chosen_slots = random.sample(slots, len(patches))
    random.shuffle(chosen_slots)
    slot_map = {slot: patch for slot, patch in zip(chosen_slots, patches)}

    row_heights = [
        max((slot_map[(row, col)].shape[0] for col in range(3) if (row, col) in slot_map), default=0)
        for row in range(2)
    ]
    col_widths = [
        max((slot_map[(row, col)].shape[1] for row in range(2) if (row, col) in slot_map), default=0)
        for col in range(3)
    ]

    row_gap = random.choice(list(GEN_ROW_GAP_RANGE))
    col_gap = random.choice(list(GEN_COL_GAP_RANGE))
    top = random.choice(list(GEN_MARGIN_RANGE))
    left = random.choice(list(GEN_MARGIN_RANGE))
    bottom = random.choice(list(GEN_MARGIN_RANGE))
    right = random.choice(list(GEN_MARGIN_RANGE))

    height = top + bottom + sum(row_heights) + row_gap
    width = left + right + sum(col_widths) + 2 * col_gap
    if height > 30 or width > 30:
        return None

    row_starts = [top, top + row_heights[0] + row_gap]
    col_starts = [left]
    for col in range(1, 3):
        col_starts.append(col_starts[-1] + col_widths[col - 1] + col_gap)

    grid = np.full((height, width), BLACK, dtype=int)
    for (row, col), patch in slot_map.items():
        slot_h = row_heights[row]
        slot_w = col_widths[col]
        slack_r = slot_h - patch.shape[0]
        slack_c = slot_w - patch.shape[1]
        r0 = row_starts[row] + random.randint(0, slack_r)
        c0 = col_starts[col] + random.randint(0, slack_c)
        non_bg = patch != BLACK
        grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][non_bg] = patch[non_bg]
    return grid


def _validate_scene(grid: np.ndarray, specs: list[dict]) -> bool:
    try:
        extracted = _extract_frames(grid, strict=True)
    except AssertionError:
        return False

    if len(extracted) != len(specs):
        return False

    by_color = {item["frame_color"]: item for item in extracted}
    spec_by_color = {item["frame_color"]: item for item in specs}
    if set(by_color) != set(spec_by_color):
        return False

    for color, spec in spec_by_color.items():
        frame = by_color[color]
        if frame["direction"] != spec["direction"]:
            return False
        if frame["directional_count"] != spec["directional_count"]:
            return False
        if frame["total_markers"] != spec["total_markers"]:
            return False
        if frame["aligned_ignored"] < 1 or frame["opposite_ignored"] < 1:
            return False
        if frame["total_markers"] <= frame["directional_count"]:
            return False

    dir_order = [item["frame_color"] for item in sorted(specs, key=lambda item: item["directional_count"])]
    total_order = [item["frame_color"] for item in sorted(specs, key=lambda item: item["total_markers"])]
    if dir_order == total_order:
        return False

    return np.array_equal(main(grid), _expected_output(specs, bg=BLACK))


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.choice(GEN_FRAME_COUNT_CHOICES)
    directions = _sample_directions(n_frames)
    frame_colors = _assign_frame_colors(directions)
    directional_counts, ignored_counts = _sample_counts(n_frames)

    patches: list[np.ndarray] = []
    specs: list[dict] = []
    for frame_color, direction, directional_count, ignored_count in zip(
        frame_colors,
        directions,
        directional_counts,
        ignored_counts,
    ):
        patch, spec = _build_frame_patch(frame_color, direction, directional_count, ignored_count)
        patches.append(patch)
        specs.append(spec)

    grid = _compose_scene(patches)
    if grid is None:
        return None
    return grid if _validate_scene(grid, specs) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
