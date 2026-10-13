from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
MAROON = 9
TRACK_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)


@dataclass(frozen=True)
class Track:
    color: int
    orient: str
    index: int
    marker: tuple[int, int]
    inward: tuple[int, int]


def _detect_tracks(g: np.ndarray) -> list[Track]:
    h, w = g.shape
    tracks: list[Track] = []

    for r in range(h):
        if int(g[r, 0]) == MAROON:
            vals = [int(g[r, c]) for c in range(1, w)]
            if vals and len(set(vals)) == 1 and vals[0] not in (BG, MAROON):
                tracks.append(Track(vals[0], "h", r, (r, 0), (0, 1)))
        if int(g[r, w - 1]) == MAROON:
            vals = [int(g[r, c]) for c in range(w - 1)]
            if vals and len(set(vals)) == 1 and vals[0] not in (BG, MAROON):
                tracks.append(Track(vals[0], "h", r, (r, w - 1), (0, -1)))

    for c in range(w):
        if int(g[0, c]) == MAROON:
            vals = [int(g[r, c]) for r in range(1, h)]
            if vals and len(set(vals)) == 1 and vals[0] not in (BG, MAROON):
                tracks.append(Track(vals[0], "v", c, (0, c), (1, 0)))
        if int(g[h - 1, c]) == MAROON:
            vals = [int(g[r, c]) for r in range(h - 1)]
            if vals and len(set(vals)) == 1 and vals[0] not in (BG, MAROON):
                tracks.append(Track(vals[0], "v", c, (h - 1, c), (-1, 0)))

    by_color: dict[int, Track] = {}
    for tr in tracks:
        by_color.setdefault(tr.color, tr)
    return list(by_color.values())


def _on_track(tr: Track, r: int, c: int) -> bool:
    if tr.orient == "h":
        return r == tr.index
    return c == tr.index


def _projection_distance(tr: Track, p: tuple[int, int]) -> int:
    r, c = p
    mr, mc = tr.marker
    if tr.orient == "h":
        return abs(c - mc)
    return abs(r - mr)


def _perpendicular_distance(tr: Track, p: tuple[int, int]) -> int:
    r, c = p
    if tr.orient == "h":
        return abs(r - tr.index)
    return abs(c - tr.index)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collect colored fragments by sliding them onto same-colored magnetic tracks.

    Input structure:
    - The background is black. One or more full-width or full-height one-pixel tracks use distinct non-black, non-maroon colors.
    - Each track has one maroon endpoint marker on the border; the remaining cells on that row or column are the track color.
    - Single-pixel fragments of each track color are scattered away from the matching track.

    Rule:
    - Match each fragment to the track of the same color.
    - Project the fragment perpendicularly onto that row or column, then slide it along the track toward the maroon endpoint.
    - For each track, order arriving fragments by their original projected distance from the maroon endpoint, nearest first, and stack them in consecutive cells inward from the marker.
    - Remove the original off-track fragments while keeping all tracks and maroon markers fixed.

    Key insight: The color gives the rail pairing, but the stack order is controlled by the fragment's projected coordinate on the rail, not by its original row or column.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    tracks = _detect_tracks(g)

    moves: list[tuple[Track, list[tuple[int, int]]]] = []
    for tr in tracks:
        pieces = [
            (int(r), int(c))
            for r, c in np.argwhere(g == tr.color)
            if not _on_track(tr, int(r), int(c))
        ]
        if pieces:
            moves.append((tr, pieces))

    for _tr, pieces in moves:
        for r, c in pieces:
            out[r, c] = BG

    for tr, pieces in moves:
        pieces.sort(
            key=lambda p: (
                _projection_distance(tr, p),
                _perpendicular_distance(tr, p),
                p[0],
                p[1],
            )
        )
        mr, mc = tr.marker
        dr, dc = tr.inward
        for i, _p in enumerate(pieces, start=1):
            rr = mr + dr * i
            cc = mc + dc * i
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                out[rr, cc] = tr.color

    return out


def _sample_spaced(values: list[int], count: int, gap: int) -> list[int] | None:
    for _ in range(200):
        shuffled = values[:]
        random.shuffle(shuffled)
        chosen: list[int] = []
        for v in shuffled:
            if all(abs(v - u) >= gap for u in chosen):
                chosen.append(v)
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _clear_for_fragment(g: np.ndarray, r: int, c: int) -> bool:
    h, w = g.shape
    if int(g[r, c]) != BG:
        return False
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            if int(g[rr, cc]) != BG:
                return False
    return True


def _paint_tracks(
    h: int,
    w: int,
    orient: str,
    track_count: int,
) -> tuple[np.ndarray, list[Track]] | None:
    g = np.full((h, w), BG, dtype=int)
    axis_len = h if orient == "h" else w
    candidates = list(range(2, axis_len - 2))
    positions = _sample_spaced(candidates, track_count, gap=3)
    if positions is None:
        return None

    colors = random.sample(TRACK_COLORS, track_count)
    tracks: list[Track] = []
    for color, pos in zip(colors, positions):
        if orient == "h":
            marker_on_low_end = bool(random.getrandbits(1))
            g[pos, :] = color
            if marker_on_low_end:
                g[pos, 0] = MAROON
                tracks.append(Track(color, "h", pos, (pos, 0), (0, 1)))
            else:
                g[pos, w - 1] = MAROON
                tracks.append(Track(color, "h", pos, (pos, w - 1), (0, -1)))
        else:
            marker_on_low_end = bool(random.getrandbits(1))
            g[:, pos] = color
            if marker_on_low_end:
                g[0, pos] = MAROON
                tracks.append(Track(color, "v", pos, (0, pos), (1, 0)))
            else:
                g[h - 1, pos] = MAROON
                tracks.append(Track(color, "v", pos, (h - 1, pos), (-1, 0)))
    return g, tracks


def _fragment_candidates_for_axis(
    tr: Track,
    h: int,
    w: int,
    all_track_indices: set[int],
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    if tr.orient == "h":
        axis_positions = list(range(1, w)) if tr.marker[1] == 0 else list(range(0, w - 1))
        for c in axis_positions:
            for r in range(h):
                if any(abs(r - idx) < 2 for idx in all_track_indices):
                    continue
                cells.append((r, c))
    else:
        axis_positions = list(range(1, h)) if tr.marker[0] == 0 else list(range(0, h - 1))
        for r in axis_positions:
            for c in range(w):
                if any(abs(c - idx) < 2 for idx in all_track_indices):
                    continue
                cells.append((r, c))
    random.shuffle(cells)
    return cells


def _place_fragments(g: np.ndarray, tracks: list[Track]) -> dict[int, list[tuple[int, int]]] | None:
    h, w = g.shape
    all_indices = {tr.index for tr in tracks}
    placed_by_color: dict[int, list[tuple[int, int]]] = {}

    for tr in tracks:
        max_count = min(5, (w - 2 if tr.orient == "h" else h - 2))
        count = random.randint(2, max_count)
        used_distances: set[int] = set()
        placed: list[tuple[int, int]] = []

        for r, c in _fragment_candidates_for_axis(tr, h, w, all_indices):
            dist = _projection_distance(tr, (r, c))
            if dist in used_distances:
                continue
            if not _clear_for_fragment(g, r, c):
                continue
            g[r, c] = tr.color
            placed.append((r, c))
            used_distances.add(dist)
            if len(placed) == count:
                break

        if len(placed) != count:
            return None
        placed_by_color[tr.color] = placed

    return placed_by_color


def _validate_generated(g: np.ndarray, tracks: list[Track], fragments: dict[int, list[tuple[int, int]]]) -> bool:
    out = main(g)
    if out.shape != g.shape or max(out.shape) > 30:
        return False

    detected = _detect_tracks(g)
    if len(detected) != len(tracks):
        return False
    if {tr.color for tr in detected} != {tr.color for tr in tracks}:
        return False

    for tr in tracks:
        for r, c in fragments[tr.color]:
            if int(out[r, c]) != BG:
                return False
        color_cells = [(int(r), int(c)) for r, c in np.argwhere(out == tr.color)]
        if any(not _on_track(tr, r, c) for r, c in color_cells):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    orient = random.choice(["h", "v"])
    h = random.randint(11, 26)
    w = random.randint(11, 26)

    axis_len = h if orient == "h" else w
    max_tracks = min(4, max(2, (axis_len - 4 + 2) // 3))
    track_count = random.randint(2, max_tracks)

    painted = _paint_tracks(h, w, orient, track_count)
    if painted is None:
        return None
    g, tracks = painted

    fragments = _place_fragments(g, tracks)
    if fragments is None:
        return None
    if not _validate_generated(g, tracks, fragments):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(500):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("failed to generate a valid magnetic-track input")
