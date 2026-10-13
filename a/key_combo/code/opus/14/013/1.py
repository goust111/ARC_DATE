from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
SEGMENT_COLOR_POOL = (BLUE, GREEN, YELLOW, PINK)

MAX_GEN_ATTEMPTS = 500
_GEN_COUNTER = 0

_SCENARIOS = (
    {"selected_count": 2, "marker_count": 2, "require_multi": True, "require_same": False, "require_cross": True},
    {"selected_count": 2, "marker_count": 2, "require_multi": False, "require_same": True, "require_cross": True},
    {"selected_count": 1, "marker_count": 2, "require_multi": False, "require_same": True, "require_cross": False},
    {"selected_count": 1, "marker_count": 2, "require_multi": False, "require_same": False, "require_cross": True},
    {"selected_count": 2, "marker_count": 3, "require_multi": False, "require_same": True, "require_cross": False},
    {"selected_count": 2, "marker_count": 3, "require_multi": True, "require_same": True, "require_cross": True},
)

_SEGMENT_CACHE: dict[tuple[int, int, int], list[tuple[tuple[int, int], ...]]] = {}


@dataclass(frozen=True)
class Segment:
    color: int
    coords: tuple[tuple[int, int], ...]
    diag: int
    residue: int
    center: tuple[int, int]
    anti_sums: frozenset[int]


@dataclass(frozen=True)
class MarkerCandidate:
    pos: tuple[int, int]
    hits: frozenset[int]
    modes: tuple[tuple[int, str], ...]


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _make_segment(color: int, cells: tuple[tuple[int, int], ...]) -> Segment:
    coords = tuple(sorted((int(r), int(c)) for r, c in cells))
    diag = coords[0][0] - coords[0][1]
    mean_r = sum(r for r, _ in coords) / len(coords)
    mean_c = sum(c for _, c in coords) / len(coords)
    center = min(
        coords,
        key=lambda rc: (abs(rc[0] - mean_r) + abs(rc[1] - mean_c), rc[0], rc[1]),
    )
    return Segment(
        color=int(color),
        coords=coords,
        diag=int(diag),
        residue=int(diag % 3),
        center=center,
        anti_sums=frozenset(r + c for r, c in coords),
    )


def _extract_segments(grid: np.ndarray) -> list[Segment]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    segments: list[Segment] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in (BG, RED) or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr in range(max(0, rr - 1), min(h, rr + 2)):
                    for nc in range(max(0, cc - 1), min(w, cc + 2)):
                        if (nr, nc) == (rr, cc) or visited[nr, nc]:
                            continue
                        if int(g[nr, nc]) == color:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
            segments.append(_make_segment(color, tuple(cells)))
    return segments


def _marker_coords(grid: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(np.asarray(grid, dtype=int) == RED)]


def _activated_mask(shape: tuple[int, int], markers: list[tuple[int, int]]) -> np.ndarray:
    rows, cols = np.indices(shape)
    activated = np.zeros(shape, dtype=bool)
    for r, c in markers:
        activated |= ((rows - cols) == (r - c)) | ((rows + cols) == (r + c))
    return activated


def _nearest_marker_distance(point: tuple[int, int], markers: list[tuple[int, int]]) -> int:
    return min(abs(point[0] - r) + abs(point[1] - c) for r, c in markers)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use red diagonal rays to select which short NW-SE segments survive, then fill only the selected segments' modulo-3 diagonal classes across the black background.
    
    Input structure:
    - The grid is mostly black, with 2-3 single red marker pixels and several short colored NW-SE diagonal segments.
    - Every colored segment is a contiguous run of 2-4 pixels on one fixed `row-col` diagonal.
    - The red markers are isolated from the segments; they act only as diagonal selectors.
    
    Rule:
    - Through each red marker, activate the full NW-SE and NE-SW diagonals out to the grid border.
    - A colored segment is selected iff at least one of its pixels lies on any activated diagonal cell.
    - For every selected segment, compute `d = (row-col) mod 3` from that segment and paint every originally black cell with the same modulo-3 offset using the segment's color.
    - Erase every unselected segment to black, keep red markers unchanged, and if multiple selected segments share the same modulo-3 class, let the one closer to a red marker win that class.
    
    Key insight:
    - The red diagonals are not paint strokes: they are gates that decide which source segments stay active, and only those active segments expand into full modulo-3 diagonal fill classes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    markers = _marker_coords(g)
    if not markers:
        return g.copy()

    segments = _extract_segments(g)
    activated = _activated_mask(g.shape, markers)
    selected = [seg for seg in segments if any(activated[r, c] for r, c in seg.coords)]

    grouped: dict[int, list[Segment]] = {}
    for seg in selected:
        grouped.setdefault(seg.residue, []).append(seg)

    winners: dict[int, Segment] = {}
    for residue, group in grouped.items():
        winners[residue] = min(
            group,
            key=lambda seg: (_nearest_marker_distance(seg.center, markers), seg.center[0], seg.center[1]),
        )

    out = g.copy()
    for seg in segments:
        if seg not in selected:
            for r, c in seg.coords:
                out[r, c] = BG

    rows, cols = np.indices(g.shape)
    for residue, winner in winners.items():
        fill_mask = ((rows - cols) % 3 == residue) & (g == BG)
        out[fill_mask] = winner.color
        if len(grouped[residue]) > 1:
            for seg in grouped[residue]:
                for r, c in seg.coords:
                    out[r, c] = winner.color

    out[g == RED] = RED
    return out


def _segment_options(h: int, w: int, residue: int) -> list[tuple[tuple[int, int], ...]]:
    key = (h, w, residue)
    if key in _SEGMENT_CACHE:
        return _SEGMENT_CACHE[key]

    options: list[tuple[tuple[int, int], ...]] = []
    for diag in range(-(w - 1), h):
        if diag % 3 != residue:
            continue
        diag_cells = [(r, r - diag) for r in range(h) if 0 <= r - diag < w]
        if len(diag_cells) < 2:
            continue
        for length in (2, 3, 4):
            if len(diag_cells) < length:
                continue
            for start in range(len(diag_cells) - length + 1):
                seg = tuple(diag_cells[start : start + length])
                options.append(seg)
    _SEGMENT_CACHE[key] = options
    return options


def _mark_buffer(mask: np.ndarray, cells: tuple[tuple[int, int], ...]) -> None:
    h, w = mask.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                mask[rr, cc] = True


def _place_segments(rng: np.random.Generator, h: int, w: int) -> list[Segment] | None:
    residues = [0, 1, 2]
    order = list(map(int, rng.permutation(residues)))
    reserved = np.zeros((h, w), dtype=bool)
    chosen_cells: dict[int, tuple[tuple[int, int], ...]] = {}

    for residue in order:
        options = _segment_options(h, w, residue)
        valid = [seg for seg in options if all(not reserved[r, c] for r, c in seg)]
        if not valid:
            return None
        seg = valid[int(rng.integers(len(valid)))]
        chosen_cells[residue] = seg
        _mark_buffer(reserved, seg)

    colors = list(map(int, rng.choice(SEGMENT_COLOR_POOL, size=3, replace=False)))
    rng.shuffle(colors)
    residue_to_color = {residue: colors[i] for i, residue in enumerate(residues)}
    segment_order = list(map(int, rng.permutation(residues)))
    return [_make_segment(residue_to_color[residue], chosen_cells[residue]) for residue in segment_order]


def _enumerate_marker_candidates(h: int, w: int, segments: list[Segment]) -> list[MarkerCandidate]:
    blocked = np.zeros((h, w), dtype=bool)
    for seg in segments:
        _mark_buffer(blocked, seg.coords)

    candidates: list[MarkerCandidate] = []
    for r in range(h):
        for c in range(w):
            if blocked[r, c]:
                continue
            diff = r - c
            anti = r + c
            hits = []
            modes = []
            for idx, seg in enumerate(segments):
                if diff == seg.diag:
                    hits.append(idx)
                    modes.append((idx, "same"))
                elif anti in seg.anti_sums:
                    hits.append(idx)
                    modes.append((idx, "cross"))
            if hits:
                candidates.append(
                    MarkerCandidate(
                        pos=(r, c),
                        hits=frozenset(hits),
                        modes=tuple(sorted(modes)),
                    )
                )
    return candidates


def _marker_set_conditions(chosen: list[MarkerCandidate]) -> tuple[bool, bool, bool]:
    has_multi = any(len(cand.hits) > 1 for cand in chosen)
    has_same = any(mode == "same" for cand in chosen for _idx, mode in cand.modes)
    has_cross = any(mode == "cross" for cand in chosen for _idx, mode in cand.modes)
    return has_multi, has_same, has_cross


def _search_markers(
    rng: np.random.Generator,
    candidates: list[MarkerCandidate],
    target: frozenset[int],
    marker_count: int,
    require_multi: bool,
    require_same: bool,
    require_cross: bool,
) -> list[tuple[int, int]] | None:
    relevant = [cand for cand in candidates if cand.hits <= target]
    if not relevant:
        return None

    order = list(range(len(relevant)))
    rng.shuffle(order)
    relevant = [relevant[i] for i in order]

    def rec(start: int, chosen: list[MarkerCandidate], covered: frozenset[int]) -> list[tuple[int, int]] | None:
        if len(chosen) == marker_count:
            has_multi, has_same, has_cross = _marker_set_conditions(chosen)
            if (
                covered == target
                and (has_multi or not require_multi)
                and (has_same or not require_same)
                and (has_cross or not require_cross)
            ):
                return [cand.pos for cand in chosen]
            return None

        possible = set(covered)
        for cand in relevant[start:]:
            if all(_chebyshev(cand.pos, prev.pos) > 1 for prev in chosen):
                possible.update(cand.hits)
        if not target.issubset(possible):
            return None

        for i in range(start, len(relevant)):
            cand = relevant[i]
            if any(_chebyshev(cand.pos, prev.pos) <= 1 for prev in chosen):
                continue
            found = rec(i + 1, chosen + [cand], covered | cand.hits)
            if found is not None:
                return found
        return None

    return rec(0, [], frozenset())


def _build_grid(h: int, w: int, segments: list[Segment], markers: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for seg in segments:
        for r, c in seg.coords:
            grid[r, c] = seg.color
    for r, c in markers:
        grid[r, c] = RED
    return grid


def _scenario_ok(grid: np.ndarray, scenario: dict[str, int | bool]) -> bool:
    markers = _marker_coords(grid)
    segments = _extract_segments(grid)
    candidate_map = {
        cand.pos: cand for cand in _enumerate_marker_candidates(grid.shape[0], grid.shape[1], segments)
    }
    chosen = [candidate_map[pos] for pos in markers]
    covered = frozenset().union(*(cand.hits for cand in chosen))
    if len(covered) != int(scenario["selected_count"]):
        return False
    has_multi, has_same, has_cross = _marker_set_conditions(chosen)
    return (
        (has_multi or not bool(scenario["require_multi"]))
        and (has_same or not bool(scenario["require_same"]))
        and (has_cross or not bool(scenario["require_cross"]))
    )


def _segments_are_isolated(segments: list[Segment], markers: list[tuple[int, int]]) -> bool:
    all_segment_cells = [cell for seg in segments for cell in seg.coords]
    for i, cell_a in enumerate(all_segment_cells):
        for cell_b in all_segment_cells[i + 1 :]:
            if cell_a == cell_b:
                return False
    for i, seg_a in enumerate(segments):
        for seg_b in segments[i + 1 :]:
            if any(_chebyshev(a, b) <= 1 for a in seg_a.coords for b in seg_b.coords):
                return False
    for i, a in enumerate(markers):
        for b in markers[i + 1 :]:
            if _chebyshev(a, b) <= 1:
                return False
    for marker in markers:
        for cell in all_segment_cells:
            if _chebyshev(marker, cell) <= 1:
                return False
    return True


def _semantic_check(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    markers = _marker_coords(g)
    if len(markers) not in (2, 3):
        return False

    segments = _extract_segments(g)
    if len(segments) != 3:
        return False
    if len({seg.color for seg in segments}) != 3:
        return False
    if {seg.residue for seg in segments} != {0, 1, 2}:
        return False
    if not _segments_are_isolated(segments, markers):
        return False

    for seg in segments:
        if len(seg.coords) not in (2, 3, 4):
            return False
        if len({r - c for r, c in seg.coords}) != 1:
            return False
        coords = sorted(seg.coords)
        for (r0, c0), (r1, c1) in zip(coords, coords[1:]):
            if (r1 - r0, c1 - c0) != (1, 1):
                return False

    activated = _activated_mask(g.shape, markers)
    selected = [seg for seg in segments if any(activated[r, c] for r, c in seg.coords)]
    if len(selected) not in (1, 2):
        return False
    if len(selected) == len(segments):
        return False

    out = main(g)
    if np.array_equal(out, g):
        return False
    if not np.array_equal(out[g == RED], g[g == RED]):
        return False

    rows, cols = np.indices(g.shape)
    selected_by_residue = {seg.residue: seg.color for seg in selected}
    for seg in segments:
        expected = seg.color if seg in selected else BG
        if any(int(out[r, c]) != expected for r, c in seg.coords):
            return False

    for residue, color in selected_by_residue.items():
        fill_mask = ((rows - cols) % 3 == residue) & (g == BG)
        if not np.any(fill_mask):
            return False
        if not np.all(out[fill_mask] == color):
            return False

    for residue in {0, 1, 2} - set(selected_by_residue):
        empty_mask = ((rows - cols) % 3 == residue) & (g == BG)
        if not np.all(out[empty_mask] == BG):
            return False

    return True


def _attempt_generate(rng: np.random.Generator, scenario: dict[str, int | bool]) -> np.ndarray | None:
    h = int(rng.integers(13, 19))
    w = int(rng.integers(13, 19))
    segments = _place_segments(rng, h, w)
    if segments is None:
        return None

    candidates = _enumerate_marker_candidates(h, w, segments)
    if not candidates:
        return None

    segment_indices = list(range(len(segments)))
    target_sets = [
        frozenset(combo)
        for combo in itertools.combinations(segment_indices, int(scenario["selected_count"]))
    ]
    rng.shuffle(target_sets)

    for target in target_sets:
        markers = _search_markers(
            rng=rng,
            candidates=candidates,
            target=target,
            marker_count=int(scenario["marker_count"]),
            require_multi=bool(scenario["require_multi"]),
            require_same=bool(scenario["require_same"]),
            require_cross=bool(scenario["require_cross"]),
        )
        if markers is None:
            continue
        grid = _build_grid(h, w, segments, markers)
        if _semantic_check(grid) and _scenario_ok(grid, scenario):
            return grid
    return None


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    primary = _SCENARIOS[_GEN_COUNTER % len(_SCENARIOS)]
    _GEN_COUNTER += 1

    scenario_order = [primary] + [sc for sc in _SCENARIOS if sc is not primary]
    for scenario in scenario_order:
        for _ in range(MAX_GEN_ATTEMPTS // len(scenario_order)):
            rng = np.random.default_rng()
            grid = _attempt_generate(rng, scenario)
            if grid is not None:
                return grid
    raise RuntimeError("failed to generate a valid selective diagonal fill input")
