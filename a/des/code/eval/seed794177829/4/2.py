from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
ARROW = 9
TRACK_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)
MAX_GEN_ATTEMPTS = 500


@dataclass(frozen=True)
class Track:
    color: int
    orient: str  # "h" or "v"
    fixed: int   # row if horizontal, col if vertical
    lo: int
    hi: int
    marker_side: str  # "left"/"right" for h, "top"/"bottom" for v

    @property
    def length(self) -> int:
        return int(self.hi - self.lo + 1)


def _marker_cells(track: Track) -> list[tuple[int, int]]:
    if track.orient == "h" and track.marker_side == "left":
        r = int(track.fixed)
        c = int(track.lo)
        return [(r - 1, c - 2), (r - 1, c - 1), (r, c - 2), (r, c - 1)]
    if track.orient == "h" and track.marker_side == "right":
        r = int(track.fixed)
        c = int(track.hi)
        return [(r, c + 1), (r, c + 2), (r + 1, c + 1), (r + 1, c + 2)]
    if track.orient == "v" and track.marker_side == "top":
        r = int(track.lo)
        c = int(track.fixed)
        return [(r - 2, c - 1), (r - 2, c), (r - 1, c - 1), (r - 1, c)]
    assert track.orient == "v" and track.marker_side == "bottom"
    r = int(track.hi)
    c = int(track.fixed)
    return [(r + 1, c), (r + 1, c + 1), (r + 2, c), (r + 2, c + 1)]


def _marker_bbox(track: Track) -> tuple[int, int, int, int]:
    cells = _marker_cells(track)
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _line_cells(track: Track) -> list[tuple[int, int]]:
    if track.orient == "h":
        return [(int(track.fixed), c) for c in range(int(track.lo), int(track.hi) + 1)]
    return [(r, int(track.fixed)) for r in range(int(track.lo), int(track.hi) + 1)]


def _track_cell_set(track: Track) -> set[tuple[int, int]]:
    return set(_line_cells(track))


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    r, c = map(int, cell)
    return 0 <= r < h and 0 <= c < w


def _find_arrow_direction(grid: np.ndarray) -> str:
    pts = np.argwhere(grid == ARROW)
    assert int(pts.shape[0]) == 3, "expected exactly one 3-pixel maroon arrow"
    on_left = bool((pts[:, 1] == 0).any())
    on_top = bool((pts[:, 0] == 0).any())
    assert on_left ^ on_top, "arrow must be on the left edge or the top edge"
    return "right" if on_left else "down"


def _find_marker_blocks(grid: np.ndarray, color: int) -> list[tuple[int, int, int, int]]:
    h, w = map(int, grid.shape)
    out: list[tuple[int, int, int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            if bool((grid[r : r + 2, c : c + 2] == int(color)).all()):
                out.append((int(r), int(c), int(r + 1), int(c + 1)))
    return out


def _parse_track(grid: np.ndarray, color: int) -> Track:
    h, w = map(int, grid.shape)
    blocks = _find_marker_blocks(grid, int(color))
    assert len(blocks) == 1, f"color {color} must have exactly one 2x2 endpoint block"
    r0, c0, r1, c1 = blocks[0]

    sides: list[tuple[str, tuple[int, int]]] = []
    right_neighbors = [(r0, c1 + 1), (r1, c1 + 1)]
    if c1 + 1 < w:
        matched = [cell for cell in right_neighbors if int(grid[cell]) == int(color)]
        if len(matched) == 1:
            sides.append(("left", matched[0]))
    left_neighbors = [(r0, c0 - 1), (r1, c0 - 1)]
    if c0 - 1 >= 0:
        matched = [cell for cell in left_neighbors if int(grid[cell]) == int(color)]
        if len(matched) == 1:
            sides.append(("right", matched[0]))
    bottom_neighbors = [(r1 + 1, c0), (r1 + 1, c1)]
    if r1 + 1 < h:
        matched = [cell for cell in bottom_neighbors if int(grid[cell]) == int(color)]
        if len(matched) == 1:
            sides.append(("top", matched[0]))
    top_neighbors = [(r0 - 1, c0), (r0 - 1, c1)]
    if r0 - 1 >= 0:
        matched = [cell for cell in top_neighbors if int(grid[cell]) == int(color)]
        if len(matched) == 1:
            sides.append(("bottom", matched[0]))

    assert len(sides) == 1, f"color {color} must attach to its marker on exactly one side"
    marker_side, start = sides[0]
    sr, sc = start

    if marker_side == "left":
        hi = int(sc)
        while hi + 1 < w and int(grid[sr, hi + 1]) not in (BG, ARROW):
            hi += 1
        return Track(int(color), "h", int(sr), int(sc), int(hi), "left")

    if marker_side == "right":
        lo = int(sc)
        while lo - 1 >= 0 and int(grid[sr, lo - 1]) not in (BG, ARROW):
            lo -= 1
        return Track(int(color), "h", int(sr), int(lo), int(sc), "right")

    if marker_side == "top":
        hi = int(sr)
        while hi + 1 < h and int(grid[hi + 1, sc]) not in (BG, ARROW):
            hi += 1
        return Track(int(color), "v", int(sc), int(sr), int(hi), "top")

    lo = int(sr)
    while lo - 1 >= 0 and int(grid[lo - 1, sc]) not in (BG, ARROW):
        lo -= 1
    return Track(int(color), "v", int(sc), int(lo), int(sr), "bottom")


def _extract_tracks(grid: np.ndarray) -> list[Track]:
    colors = [int(c) for c in np.unique(grid) if int(c) not in (BG, ARROW)]
    tracks: list[Track] = []
    for color in colors:
        blocks = _find_marker_blocks(grid, int(color))
        if blocks:
            tracks.append(_parse_track(grid, int(color)))
    assert tracks, "expected at least one track"
    lengths = [track.length for track in tracks]
    assert len(set(lengths)) == len(lengths), "track lengths must be unique"
    return tracks


def _choose_nub_cell(
    base_grid: np.ndarray,
    track: Track,
    anchor: tuple[int, int],
    taken: set[tuple[int, int]] | None = None,
) -> tuple[int, int] | None:
    taken = taken or set()
    h, w = map(int, base_grid.shape)
    r, c = map(int, anchor)
    if track.orient == "h":
        candidates = [(r - 1, c), (r + 1, c)]
    else:
        candidates = [(r, c - 1), (r, c + 1)]
    for nr, nc in candidates:
        if 0 <= nr < h and 0 <= nc < w and int(base_grid[nr, nc]) == BG and (int(nr), int(nc)) not in taken:
            return int(nr), int(nc)
    return None


def _extract_fragments(grid: np.ndarray, tracks: list[Track]) -> list[dict]:
    fragments: list[dict] = []
    by_color = {track.color: track for track in tracks}
    line_sets = {track.color: _track_cell_set(track) for track in tracks}
    marker_sets = {track.color: set(_marker_cells(track)) for track in tracks}

    for color, track in by_color.items():
        same_color = [(int(r), int(c)) for r, c in np.argwhere(grid == int(color))]
        for cell in same_color:
            if cell in line_sets[color] or cell in marker_sets[color]:
                continue
            neighbors = []
            r, c = cell
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in line_sets[color]:
                    neighbors.append((int(nr), int(nc)))
            if len(neighbors) == 1:
                fragments.append(
                    {
                        "color": int(color),
                        "track_color": int(color),
                        "nub": (int(r), int(c)),
                        "anchor": neighbors[0],
                        "track": track,
                    }
                )
    return sorted(fragments, key=lambda item: (item["color"], item["anchor"][0], item["anchor"][1]))


def _build_crossings(tracks: list[Track]) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    horizontals = [track for track in tracks if track.orient == "h"]
    verticals = [track for track in tracks if track.orient == "v"]
    h_map = {track.color: [] for track in horizontals}
    v_map = {track.color: [] for track in verticals}

    for h_track in horizontals:
        for v_track in verticals:
            if int(h_track.lo) <= int(v_track.fixed) <= int(h_track.hi) and int(v_track.lo) <= int(h_track.fixed) <= int(v_track.hi):
                h_map[h_track.color].append((int(v_track.fixed), int(v_track.color)))
                v_map[v_track.color].append((int(h_track.fixed), int(h_track.color)))

    for color in h_map:
        h_map[color].sort()
    for color in v_map:
        v_map[color].sort()
    return h_map, v_map


def _simulate_fragment(
    start_track: Track,
    anchor: tuple[int, int],
    tracks_by_color: dict[int, Track],
    h_crossings: dict[int, list[tuple[int, int]]],
    v_crossings: dict[int, list[tuple[int, int]]],
) -> tuple[Track, tuple[int, int]]:
    current = start_track
    row, col = map(int, anchor)

    while True:
        if current.orient == "h":
            turned = False
            for cross_col, next_color in h_crossings.get(current.color, []):
                if int(cross_col) <= int(col):
                    continue
                nxt = tracks_by_color[int(next_color)]
                if int(nxt.length) < int(current.length) and int(nxt.hi) > int(current.fixed):
                    current = nxt
                    row = int(row)
                    col = int(cross_col)
                    turned = True
                    break
            if not turned:
                return current, (int(current.fixed), int(current.hi))
            continue

        turned = False
        for cross_row, next_color in v_crossings.get(current.color, []):
            if int(cross_row) <= int(row):
                continue
            nxt = tracks_by_color[int(next_color)]
            if int(nxt.length) < int(current.length) and int(nxt.hi) > int(current.fixed):
                current = nxt
                row = int(cross_row)
                col = int(col)
                turned = True
                break
        if not turned:
            return current, (int(current.hi), int(current.fixed))


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    tracks = _extract_tracks(g)
    direction = _find_arrow_direction(g)
    required_orient = "h" if direction == "right" else "v"

    fragments = _extract_fragments(g, tracks)
    tracks_by_color = {track.color: track for track in tracks}
    h_crossings, v_crossings = _build_crossings(tracks)

    for fragment in fragments:
        assert fragment["track"].orient == required_orient, "initial fragments must align with the arrow direction"

    out = g.copy()
    for fragment in fragments:
        nr, nc = fragment["nub"]
        out[int(nr), int(nc)] = BG

    taken_final_nubs: set[tuple[int, int]] = set()
    for fragment in fragments:
        final_track, final_anchor = _simulate_fragment(
            fragment["track"], fragment["anchor"], tracks_by_color, h_crossings, v_crossings
        )
        nub = _choose_nub_cell(out, final_track, final_anchor, taken=taken_final_nubs)
        assert nub is not None, "every final fragment position must have room for a visible nub"
        out[nub] = int(fragment["color"])
        taken_final_nubs.add(nub)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate colored fragments moving on straight tracks with priority-based turns.

    Visible encoding used by this implementation:
    - Each track color owns exactly one straight 1-cell-wide segment and one 2x2 same-color endpoint block.
    - Crossings are drawn by painting the shorter track last, so a longer track may be visually interrupted at
      the intersection cell but still continues geometrically through it.
    - Because a same-color single pixel on top of a same-color line is invisible in ARC, each fragment is encoded
      as a same-color nub attached to one track cell from the side; the nub moves while the line itself stays fixed.
    - A left-edge maroon L means motion starts to the right; a top-edge maroon L means motion starts downward.
    """
    return _solve(grid)


def _sample_spaced_positions(
    rng: random.Random, count: int, lo: int, hi: int, gap: int
) -> list[int] | None:
    values = list(range(int(lo), int(hi) + 1))
    if len(values) < count:
        return None
    for _ in range(200):
        picks = sorted(rng.sample(values, count))
        if all(int(b - a) >= int(gap) for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _random_lengths(
    rng: random.Random,
    n_h: int,
    n_v: int,
    feasible_h: list[int],
    feasible_v: list[int],
) -> tuple[list[int], list[int]] | None:
    if len(feasible_h) < n_h or len(feasible_v) < n_v:
        return None
    for _ in range(200):
        h_lengths = rng.sample(feasible_h, n_h)
        remaining_v = [value for value in feasible_v if value not in h_lengths]
        if len(remaining_v) < n_v:
            continue
        v_lengths = rng.sample(remaining_v, n_v)
        if max(h_lengths) > min(v_lengths) and min(h_lengths) < max(v_lengths):
            rng.shuffle(h_lengths)
            rng.shuffle(v_lengths)
            return h_lengths, v_lengths
    return None


def _feasible_lo_values(length: int, min_cross: int, max_cross: int, limit_hi: int) -> list[int]:
    # Keep one non-intersection line cell on both sides of the crossing band.
    lower = max(2, int(max_cross - length + 2))
    upper = min(int(min_cross - 1), int(limit_hi - length + 1))
    if lower > upper:
        return []
    return list(range(int(lower), int(upper) + 1))


def _build_static_grid(
    shape: tuple[int, int], tracks: list[Track], direction: str, rng: random.Random
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for track in sorted(tracks, key=lambda item: item.length, reverse=True):
        for r, c in _line_cells(track):
            grid[int(r), int(c)] = int(track.color)
    for track in tracks:
        for r, c in _marker_cells(track):
            grid[int(r), int(c)] = int(track.color)

    h, w = map(int, shape)
    if direction == "right":
        arrow_r = 1 if h <= 14 else rng.randint(1, h - 6)
        arrow_cells = [(arrow_r, 0), (arrow_r + 1, 0), (arrow_r + 1, 1)]
    else:
        arrow_c = 1 if w <= 14 else rng.randint(1, w - 6)
        arrow_cells = [(0, arrow_c), (0, arrow_c + 1), (1, arrow_c + 1)]
    for r, c in arrow_cells:
        grid[int(r), int(c)] = ARROW
    return grid


def _turn_points_for_track(
    track: Track,
    tracks_by_color: dict[int, Track],
    h_crossings: dict[int, list[tuple[int, int]]],
    v_crossings: dict[int, list[tuple[int, int]]],
) -> list[int]:
    out: list[int] = []
    if track.orient == "h":
        for cross_col, other_color in h_crossings.get(track.color, []):
            other = tracks_by_color[int(other_color)]
            if int(other.length) < int(track.length) and int(other.hi) > int(track.fixed):
                out.append(int(cross_col))
        return out

    for cross_row, other_color in v_crossings.get(track.color, []):
        other = tracks_by_color[int(other_color)]
        if int(other.length) < int(track.length) and int(other.hi) > int(track.fixed):
            out.append(int(cross_row))
    return out


def _positions_on_track(track: Track) -> list[tuple[int, int]]:
    return _line_cells(track)


def _axis_value(track: Track, cell: tuple[int, int]) -> int:
    return int(cell[1] if track.orient == "h" else cell[0])


def _candidate_anchors(
    base_grid: np.ndarray,
    track: Track,
    forbidden_axis: set[int],
    lo_limit: int,
    hi_limit: int,
    taken_nubs: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    out: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for cell in _positions_on_track(track):
        axis = _axis_value(track, cell)
        if axis in forbidden_axis:
            continue
        if axis < int(lo_limit) or axis > int(hi_limit):
            continue
        nub = _choose_nub_cell(base_grid, track, cell, taken=taken_nubs)
        if nub is not None:
            out.append((cell, nub))
    return out


def _pick_fragments(rng: random.Random, base_grid: np.ndarray, tracks: list[Track], direction: str) -> list[dict] | None:
    initial_orient = "h" if direction == "right" else "v"
    usable_tracks = [track for track in tracks if track.orient == initial_orient]
    if len(usable_tracks) < 2:
        return None

    tracks_by_color = {track.color: track for track in tracks}
    h_crossings, v_crossings = _build_crossings(tracks)
    intersections = {
        track.color: set(_turn_points_for_track(track, tracks_by_color, h_crossings, v_crossings))
        for track in usable_tracks
    }
    all_cross_axes = {
        track.color: {axis for axis, _ in (h_crossings.get(track.color, []) if track.orient == "h" else v_crossings.get(track.color, []))}
        for track in usable_tracks
    }

    turn_choices: list[tuple[Track, list[tuple[tuple[int, int], tuple[int, int]]]]] = []
    no_turn_choices: list[tuple[Track, list[tuple[tuple[int, int], tuple[int, int]]]]] = []

    for track in usable_tracks:
        turn_axes = sorted(intersections[track.color])
        forbidden = set(all_cross_axes[track.color])
        hi_safe = int(track.hi - 1)
        if hi_safe < int(track.lo):
            continue

        if turn_axes:
            turning = _candidate_anchors(
                base_grid,
                track,
                forbidden,
                int(track.lo),
                int(turn_axes[0] - 1),
                taken_nubs=set(),
            )
            if turning:
                turn_choices.append((track, turning))

            no_turning = _candidate_anchors(
                base_grid,
                track,
                forbidden,
                int(turn_axes[-1] + 1),
                int(hi_safe),
                taken_nubs=set(),
            )
            if no_turning:
                no_turn_choices.append((track, no_turning))
        else:
            only_straight = _candidate_anchors(
                base_grid, track, forbidden, int(track.lo), int(hi_safe), taken_nubs=set()
            )
            if only_straight:
                no_turn_choices.append((track, only_straight))

    if not turn_choices or not no_turn_choices:
        return None

    fragments: list[dict] = []
    used_colors: set[int] = set()
    taken_initial_nubs: set[tuple[int, int]] = set()

    turn_track, turn_candidates = rng.choice(turn_choices)
    turn_candidates = [pair for pair in turn_candidates if pair[1] not in taken_initial_nubs]
    if not turn_candidates:
        return None
    turn_anchor, turn_nub = rng.choice(turn_candidates)
    fragments.append(
        {"color": turn_track.color, "track": turn_track, "anchor": turn_anchor, "nub": turn_nub}
    )
    used_colors.add(turn_track.color)
    taken_initial_nubs.add(turn_nub)

    filtered_no_turn = [
        (track, [pair for pair in candidates if pair[1] not in taken_initial_nubs])
        for track, candidates in no_turn_choices
        if track.color not in used_colors
    ]
    filtered_no_turn = [(track, candidates) for track, candidates in filtered_no_turn if candidates]
    if not filtered_no_turn:
        return None
    straight_track, straight_candidates = rng.choice(filtered_no_turn)
    straight_anchor, straight_nub = rng.choice(straight_candidates)
    fragments.append(
        {"color": straight_track.color, "track": straight_track, "anchor": straight_anchor, "nub": straight_nub}
    )
    used_colors.add(straight_track.color)
    taken_initial_nubs.add(straight_nub)

    extras = [track for track in usable_tracks if track.color not in used_colors]
    rng.shuffle(extras)
    extra_target = rng.randint(0, min(2, len(extras)))
    for track in extras:
        if len(fragments) >= 2 + extra_target:
            break
        forbidden = set(all_cross_axes[track.color])
        candidates = _candidate_anchors(
            base_grid, track, forbidden, int(track.lo), int(track.hi - 1), taken_nubs=taken_initial_nubs
        )
        if not candidates:
            continue
        anchor, nub = rng.choice(candidates)
        fragments.append({"color": track.color, "track": track, "anchor": anchor, "nub": nub})
        used_colors.add(track.color)
        taken_initial_nubs.add(nub)

    # Validate final placements as well.
    taken_final_nubs: set[tuple[int, int]] = set()
    final_anchors: set[tuple[int, int]] = set()
    for fragment in fragments:
        final_track, final_anchor = _simulate_fragment(
            fragment["track"], fragment["anchor"], tracks_by_color, h_crossings, v_crossings
        )
        if final_anchor in final_anchors:
            return None
        final_nub = _choose_nub_cell(base_grid, final_track, final_anchor, taken=taken_final_nubs)
        if final_nub is None:
            return None
        final_anchors.add(final_anchor)
        taken_final_nubs.add(final_nub)

    return fragments


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 27)
    w = rng.randint(18, 27)
    direction = rng.choice(["right", "down"])
    n_h, n_v = rng.choice([(2, 2), (3, 2), (2, 3), (3, 3)])

    rows = _sample_spaced_positions(rng, n_h, 3, h - 4, 3)
    cols = _sample_spaced_positions(rng, n_v, 3, w - 4, 3)
    if rows is None or cols is None:
        return None

    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    feasible_h = [
        length
        for length in range(max(6, max_col - min_col + 1), w - 3)
        if _feasible_lo_values(length, min_col, max_col, w - 3)
    ]
    feasible_v = [
        length
        for length in range(max(6, max_row - min_row + 1), h - 3)
        if _feasible_lo_values(length, min_row, max_row, h - 3)
    ]

    sampled = _random_lengths(rng, n_h, n_v, feasible_h, feasible_v)
    if sampled is None:
        return None
    h_lengths, v_lengths = sampled

    colors = list(rng.sample(TRACK_COLORS, n_h + n_v))
    tracks: list[Track] = []

    for row, length, color in zip(rows, h_lengths, colors[:n_h]):
        lo_choices = _feasible_lo_values(length, min_col, max_col, w - 3)
        if not lo_choices:
            return None
        lo = rng.choice(lo_choices)
        hi = int(lo + length - 1)
        tracks.append(
            Track(
                int(color),
                "h",
                int(row),
                int(lo),
                int(hi),
                rng.choice(["left", "right"]),
            )
        )

    for col, length, color in zip(cols, v_lengths, colors[n_h:]):
        lo_choices = _feasible_lo_values(length, min_row, max_row, h - 3)
        if not lo_choices:
            return None
        lo = rng.choice(lo_choices)
        hi = int(lo + length - 1)
        tracks.append(
            Track(
                int(color),
                "v",
                int(col),
                int(lo),
                int(hi),
                rng.choice(["top", "bottom"]),
            )
        )

    # Static geometry validation.
    occupied_markers: set[tuple[int, int]] = set()
    line_cells_all: set[tuple[int, int]] = set()
    for track in tracks:
        marker_cells = _marker_cells(track)
        if any(not _in_bounds((h, w), cell) for cell in marker_cells):
            return None
        for cell in marker_cells:
            if cell in occupied_markers:
                return None
            occupied_markers.add(cell)
        for cell in _line_cells(track):
            if not _in_bounds((h, w), cell):
                return None
            line_cells_all.add(cell)

    if occupied_markers & line_cells_all:
        return None

    base_grid = _build_static_grid((h, w), tracks, direction, rng)
    if len(_extract_tracks(base_grid)) != len(tracks):
        return None

    fragments = _pick_fragments(rng, base_grid, tracks, direction)
    if fragments is None:
        return None

    grid = base_grid.copy()
    for fragment in fragments:
        nr, nc = fragment["nub"]
        if int(grid[nr, nc]) != BG:
            return None
        grid[nr, nc] = int(fragment["color"])

    # Final self-check: the generated instance must round-trip through main().
    try:
        out = _solve(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid magnetic-track ARC sample")
