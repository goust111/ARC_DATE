from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, find_components, get_bg


Coord = tuple[int, int]


@dataclass(frozen=True)
class Container:
    idx: int
    r0: int
    c0: int
    r1: int
    c1: int

    @property
    def height(self) -> int:
        return int(self.r1 - self.r0 + 1)

    @property
    def width(self) -> int:
        return int(self.c1 - self.c0 + 1)


@dataclass(frozen=True)
class HitSpec:
    container_idx: int
    entry_side: str
    distance: int
    stripe_cells: tuple[Coord, ...]
    corridor_cells: tuple[Coord, ...]


@dataclass(frozen=True)
class MarkerCandidate:
    orientation: str
    anchor: Coord
    cells: tuple[Coord, Coord]
    hits: tuple[HitSpec, ...]
    stripe_cells: tuple[Coord, ...]
    corridor_cells: tuple[Coord, ...]

    @property
    def dead_sides(self) -> int:
        return 2 - len(self.hits)

    @property
    def dual_hit(self) -> bool:
        return len(self.hits) == 2

    @property
    def target_indices(self) -> tuple[int, ...]:
        return tuple(sorted({hit.container_idx for hit in self.hits}))


GEN_MAX_ATTEMPTS = 400
GEN_SELECT_ATTEMPTS = 180
GEN_PLACE_ATTEMPTS = 140
GEN_HEIGHT_RANGE = (18, 28)
GEN_WIDTH_RANGE = (18, 28)
GEN_CONTAINER_H_RANGE = (3, 7)
GEN_CONTAINER_W_RANGE = (3, 7)
GEN_CONTAINER_BORDER_MARGIN = 3
GEN_PALETTE = [1, 2, 3, 4, 5, 6, 8, 9]
GEN_MODES = ("cross", "dual", "mixed")


def _expand8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    return out


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _stripe_cells(
    container: Container,
    orientation: str,
    entry_side: str,
    distance: int,
) -> tuple[Coord, ...]:
    d = int(distance)
    if orientation == "h":
        assert 1 <= d <= container.width
        if entry_side == "left":
            cols = range(container.c0, container.c0 + d)
        else:
            assert entry_side == "right"
            cols = range(container.c1 - d + 1, container.c1 + 1)
        return tuple((r, c) for r in range(container.r0, container.r1 + 1) for c in cols)

    assert orientation == "v"
    assert 1 <= d <= container.height
    if entry_side == "top":
        rows = range(container.r0, container.r0 + d)
    else:
        assert entry_side == "bottom"
        rows = range(container.r1 - d + 1, container.r1 + 1)
    return tuple((r, c) for r in rows for c in range(container.c0, container.c1 + 1))


def _build_hit(
    containers: list[Container],
    orientation: str,
    container_idx: int,
    entry_side: str,
    distance: int,
    corridor_cells: list[Coord],
) -> HitSpec:
    container = containers[int(container_idx)]
    stripe_cells = _stripe_cells(container, orientation, entry_side, distance)
    return HitSpec(
        container_idx=int(container_idx),
        entry_side=str(entry_side),
        distance=int(distance),
        stripe_cells=stripe_cells,
        corridor_cells=tuple((int(r), int(c)) for r, c in corridor_cells),
    )


def _trace_ray(
    grid: np.ndarray,
    bg: int,
    containers: list[Container],
    container_lookup: np.ndarray,
    start: Coord,
    direction: Coord,
    orientation: str,
) -> HitSpec | None:
    dr, dc = map(int, direction)
    r, c = map(int, start)
    distance = 0
    corridor_cells: list[Coord] = []
    if dr == 0 and dc == 1:
        entry_side = "left"
    elif dr == 0 and dc == -1:
        entry_side = "right"
    elif dr == 1 and dc == 0:
        entry_side = "top"
    else:
        assert dr == -1 and dc == 0
        entry_side = "bottom"

    while _in_bounds(grid.shape, r, c):
        value = int(grid[r, c])
        if value == int(bg):
            corridor_cells.append((r, c))
            distance += 1
            r += dr
            c += dc
            continue
        if value == Color.ORANGE:
            container_idx = int(container_lookup[r, c])
            assert container_idx >= 0, "orange cell must belong to a known rectangle"
            return _build_hit(containers, orientation, container_idx, entry_side, distance, corridor_cells)
        raise AssertionError("ray path may cross only background before reaching an orange rectangle")
    return None


def _parse_containers(grid: np.ndarray, bg: int) -> tuple[list[Container], np.ndarray]:
    comps = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[Color.ORANGE],
        return_format="dict",
    )
    comps = sorted(comps, key=lambda comp: tuple(map(int, comp["bbox"])))
    assert comps, "expected at least one orange rectangle"

    lookup = np.full(grid.shape, -1, dtype=int)
    containers: list[Container] = []
    for idx, comp in enumerate(comps):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        assert int(comp["area"]) == bbox_h(comp["bbox"]) * bbox_w(comp["bbox"])
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == Color.ORANGE), "orange object must be solid"
        lookup[r0 : r1 + 1, c0 : c1 + 1] = idx
        containers.append(Container(idx=idx, r0=r0, c0=c0, r1=r1, c1=c1))
    return containers, lookup


def _parse_markers(grid: np.ndarray, bg: int) -> list[tuple[str, Coord, Coord, int]]:
    comps = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    markers: list[tuple[str, Coord, Coord, int]] = []
    for comp in comps:
        color = int(comp["color"])
        if color in (int(bg), Color.ORANGE):
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        assert int(comp["area"]) == 2, "every non-orange object must be a two-cell marker"
        if r0 == r1 and c1 == c0 + 1:
            markers.append(("h", (r0, c0), (r0, c0 + 1), color))
        elif c0 == c1 and r1 == r0 + 1:
            markers.append(("v", (r0, c0), (r0 + 1, c0), color))
        else:
            raise AssertionError("marker must be a 1x2 or 2x1 domino")
    markers.sort(key=lambda item: (item[1], item[2], item[3]))
    assert markers, "expected at least one colored domino marker"
    return markers


def _collect_marker_hits(
    grid: np.ndarray,
    bg: int,
    containers: list[Container],
    container_lookup: np.ndarray,
    orientation: str,
    first_cell: Coord,
    second_cell: Coord,
) -> tuple[HitSpec, ...]:
    if orientation == "h":
        starts = [((first_cell[0], first_cell[1] - 1), (0, -1)), ((second_cell[0], second_cell[1] + 1), (0, 1))]
    else:
        starts = [((first_cell[0] - 1, first_cell[1]), (-1, 0)), ((second_cell[0] + 1, second_cell[1]), (1, 0))]

    hits: list[HitSpec] = []
    for start, direction in starts:
        hit = _trace_ray(grid, bg, containers, container_lookup, start, direction, orientation)
        if hit is not None:
            hits.append(hit)
    assert hits, "every marker must hit at least one container"
    return tuple(hits)


def _paint_hits(
    out: np.ndarray,
    layer: np.ndarray,
    hits: tuple[HitSpec, ...],
    color: int,
) -> None:
    value = int(color)
    for hit in hits:
        for r, c in hit.stripe_cells:
            existing = int(layer[r, c])
            assert existing in (-1, value), "same-orientation stripe overlap must not conflict"
            layer[r, c] = value
            out[r, c] = value


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each domino marker into hidden-distance stripes inside the first orange rectangles reached along its axis.

    Input structure:
    - The grid has a black background, several solid orange rectangular containers, and several isolated colored domino markers.
    - Every marker is exactly 1x2 or 2x1, and its orientation defines the only two ray directions to inspect.
    - Objects do not touch, so any successful marker-to-container corridor consists only of background cells.

    Rule:
    - Erase every marker to background.
    - Each marker sends one ray from each end along its long axis; a ray only counts if it reaches an orange rectangle before the grid edge.
    - If the strict background gap from the emitting endpoint to the struck container edge is d, then paint d full columns from the hit edge for horizontal rays or d full rows from the hit edge for vertical rays, using the marker color.
    - Apply all horizontal-ray column stripes first and all vertical-ray row stripes second, so vertical stripes overwrite crossings.

    Key insight:
    - The transferred quantity is not marker shape or size but the invisible endpoint-to-container gap length, and the border-facing dead ray is only a distractor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    containers, container_lookup = _parse_containers(g, bg)
    markers = _parse_markers(g, bg)

    base = g.copy()
    for _orientation, first_cell, second_cell, _color in markers:
        base[first_cell] = bg
        base[second_cell] = bg

    horizontal = base.copy()
    h_layer = np.full(g.shape, -1, dtype=int)
    vertical_hits: list[tuple[tuple[HitSpec, ...], int]] = []

    for orientation, first_cell, second_cell, color in markers:
        hits = _collect_marker_hits(g, bg, containers, container_lookup, orientation, first_cell, second_cell)
        if orientation == "h":
            _paint_hits(horizontal, h_layer, hits, color)
        else:
            vertical_hits.append((hits, color))

    out = horizontal.copy()
    v_layer = np.full(g.shape, -1, dtype=int)
    for hits, color in vertical_hits:
        _paint_hits(out, v_layer, hits, color)
    return out


def _render_container_grid(shape: tuple[int, int], containers: list[Container]) -> np.ndarray:
    grid = np.full(shape, Color.BLACK, dtype=int)
    for container in containers:
        grid[container.r0 : container.r1 + 1, container.c0 : container.c1 + 1] = Color.ORANGE
    return grid


def _place_containers(shape: tuple[int, int], count: int) -> list[Container] | None:
    height, width = map(int, shape)
    occupied = np.zeros(shape, dtype=bool)
    containers: list[Container] = []
    for idx in range(int(count)):
        placed = False
        for _ in range(GEN_PLACE_ATTEMPTS):
            h = random.randint(*GEN_CONTAINER_H_RANGE)
            w = random.randint(*GEN_CONTAINER_W_RANGE)
            if h >= height - 2 * GEN_CONTAINER_BORDER_MARGIN or w >= width - 2 * GEN_CONTAINER_BORDER_MARGIN:
                continue
            r0 = random.randint(GEN_CONTAINER_BORDER_MARGIN, height - GEN_CONTAINER_BORDER_MARGIN - h)
            c0 = random.randint(GEN_CONTAINER_BORDER_MARGIN, width - GEN_CONTAINER_BORDER_MARGIN - w)
            r1 = r0 + h - 1
            c1 = c0 + w - 1
            blocked = _expand8(occupied)
            if blocked[r0 : r1 + 1, c0 : c1 + 1].any():
                continue
            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            containers.append(Container(idx=idx, r0=r0, c0=c0, r1=r1, c1=c1))
            placed = True
            break
        if not placed:
            return None
    return containers


def _enumerate_candidates(containers: list[Container], shape: tuple[int, int]) -> list[MarkerCandidate]:
    grid = _render_container_grid(shape, containers)
    container_lookup = np.full(shape, -1, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    for container in containers:
        container_lookup[container.r0 : container.r1 + 1, container.c0 : container.c1 + 1] = container.idx
        occupied[container.r0 : container.r1 + 1, container.c0 : container.c1 + 1] = True
    blocked = _expand8(occupied)

    candidates: list[MarkerCandidate] = []
    invalid = object()

    def trace_for_candidate(
        start: Coord,
        direction: Coord,
        orientation: str,
    ) -> tuple[str, HitSpec | None, tuple[Coord, ...]]:
        dr, dc = map(int, direction)
        r, c = map(int, start)
        distance = 0
        corridor_cells: list[Coord] = []
        if dr == 0 and dc == 1:
            entry_side = "left"
        elif dr == 0 and dc == -1:
            entry_side = "right"
        elif dr == 1 and dc == 0:
            entry_side = "top"
        else:
            entry_side = "bottom"

        while _in_bounds(shape, r, c):
            value = int(grid[r, c])
            if value == Color.BLACK:
                corridor_cells.append((r, c))
                distance += 1
                r += dr
                c += dc
                continue
            if value == Color.ORANGE:
                try:
                    return (
                        "hit",
                        _build_hit(
                            containers,
                            orientation,
                            int(container_lookup[r, c]),
                            entry_side,
                            distance,
                            corridor_cells,
                        ),
                        tuple(corridor_cells),
                    )
                except AssertionError:
                    return ("invalid", None, tuple(corridor_cells))
            return ("invalid", None, tuple(corridor_cells))
        return ("border", None, tuple(corridor_cells))

    for r in range(shape[0]):
        for c in range(shape[1] - 1):
            cells = ((r, c), (r, c + 1))
            if any(blocked[rr, cc] for rr, cc in cells):
                continue
            hits = []
            status_left, hit_left, corridor_left = trace_for_candidate((r, c - 1), (0, -1), "h")
            status_right, hit_right, corridor_right = trace_for_candidate((r, c + 2), (0, 1), "h")
            if status_left == "invalid" or status_right == "invalid":
                continue
            if hit_left is not None:
                hits.append(hit_left)
            if hit_right is not None:
                hits.append(hit_right)
            if not hits:
                continue
            stripe_cells = tuple(sorted({cell for hit in hits for cell in hit.stripe_cells}))
            corridor_cells = tuple(sorted(set(corridor_left) | set(corridor_right)))
            candidates.append(
                MarkerCandidate(
                    orientation="h",
                    anchor=(r, c),
                    cells=cells,
                    hits=tuple(hits),
                    stripe_cells=stripe_cells,
                    corridor_cells=corridor_cells,
                )
            )

    for r in range(shape[0] - 1):
        for c in range(shape[1]):
            cells = ((r, c), (r + 1, c))
            if any(blocked[rr, cc] for rr, cc in cells):
                continue
            hits = []
            status_up, hit_up, corridor_up = trace_for_candidate((r - 1, c), (-1, 0), "v")
            status_down, hit_down, corridor_down = trace_for_candidate((r + 2, c), (1, 0), "v")
            if status_up == "invalid" or status_down == "invalid":
                continue
            if hit_up is not None:
                hits.append(hit_up)
            if hit_down is not None:
                hits.append(hit_down)
            if not hits:
                continue
            stripe_cells = tuple(sorted({cell for hit in hits for cell in hit.stripe_cells}))
            corridor_cells = tuple(sorted(set(corridor_up) | set(corridor_down)))
            candidates.append(
                MarkerCandidate(
                    orientation="v",
                    anchor=(r, c),
                    cells=cells,
                    hits=tuple(hits),
                    stripe_cells=stripe_cells,
                    corridor_cells=corridor_cells,
                )
            )

    return candidates


def _empty_state(shape: tuple[int, int]) -> dict[str, object]:
    return {
        "marker_occ": np.zeros(shape, dtype=bool),
        "blocked": np.zeros(shape, dtype=bool),
        "reserved": set(),
        "hpaint": np.zeros(shape, dtype=bool),
        "vpaint": np.zeros(shape, dtype=bool),
        "chosen": [],
    }


def _candidate_fits(candidate: MarkerCandidate, state: dict[str, object]) -> bool:
    marker_occ = state["marker_occ"]
    blocked = state["blocked"]
    reserved: set[Coord] = state["reserved"]
    hpaint = state["hpaint"]
    vpaint = state["vpaint"]

    assert isinstance(marker_occ, np.ndarray)
    assert isinstance(blocked, np.ndarray)
    assert isinstance(hpaint, np.ndarray)
    assert isinstance(vpaint, np.ndarray)

    for r, c in candidate.cells:
        if bool(blocked[r, c]) or (r, c) in reserved:
            return False

    for r, c in candidate.corridor_cells:
        if bool(marker_occ[r, c]):
            return False

    paint_mask = hpaint if candidate.orientation == "h" else vpaint
    for r, c in candidate.stripe_cells:
        if bool(paint_mask[r, c]):
            return False

    return True


def _add_candidate(candidate: MarkerCandidate, state: dict[str, object]) -> None:
    marker_occ = state["marker_occ"]
    reserved: set[Coord] = state["reserved"]
    blocked = state["blocked"]
    hpaint = state["hpaint"]
    vpaint = state["vpaint"]
    chosen: list[MarkerCandidate] = state["chosen"]

    assert isinstance(marker_occ, np.ndarray)
    assert isinstance(blocked, np.ndarray)
    assert isinstance(hpaint, np.ndarray)
    assert isinstance(vpaint, np.ndarray)

    for r, c in candidate.cells:
        marker_occ[r, c] = True
    blocked[:] = _expand8(marker_occ)
    reserved.update(candidate.corridor_cells)
    paint_mask = hpaint if candidate.orientation == "h" else vpaint
    for r, c in candidate.stripe_cells:
        paint_mask[r, c] = True
    chosen.append(candidate)


def _selection_features(chosen: list[MarkerCandidate], containers: list[Container]) -> dict[str, object]:
    orientations = {candidate.orientation for candidate in chosen}
    dead_sides = sum(candidate.dead_sides for candidate in chosen)
    dual_hits = sum(1 for candidate in chosen if candidate.dual_hit)
    affected = {hit.container_idx for candidate in chosen for hit in candidate.hits}
    h_targets = {hit.container_idx for candidate in chosen if candidate.orientation == "h" for hit in candidate.hits}
    v_targets = {hit.container_idx for candidate in chosen if candidate.orientation == "v" for hit in candidate.hits}
    cross_targets = h_targets & v_targets
    return {
        "orientations": orientations,
        "dead_sides": dead_sides,
        "dual_hits": dual_hits,
        "affected": affected,
        "cross_targets": cross_targets,
        "untouched": len(containers) - len(affected),
    }


def _seed_for_mode(
    mode: str,
    candidates: list[MarkerCandidate],
    shape: tuple[int, int],
) -> list[MarkerCandidate] | None:
    h_candidates = [candidate for candidate in candidates if candidate.orientation == "h"]
    v_candidates = [candidate for candidate in candidates if candidate.orientation == "v"]
    if not h_candidates or not v_candidates:
        return None

    if mode == "cross":
        pairs: list[tuple[MarkerCandidate, MarkerCandidate]] = []
        for h_candidate in h_candidates:
            h_targets = set(h_candidate.target_indices)
            for v_candidate in v_candidates:
                if not (h_targets & set(v_candidate.target_indices)):
                    continue
                state = _empty_state(shape)
                if not _candidate_fits(h_candidate, state):
                    continue
                _add_candidate(h_candidate, state)
                if not _candidate_fits(v_candidate, state):
                    continue
                pairs.append((h_candidate, v_candidate))
        if not pairs:
            return None
        seed = random.choice(pairs)
        return [seed[0], seed[1]]

    if mode == "dual":
        dual_candidates = [candidate for candidate in candidates if candidate.dual_hit]
        if not dual_candidates:
            return None
        return [random.choice(dual_candidates)]

    dead_candidates = [candidate for candidate in candidates if candidate.dead_sides >= 1]
    if not dead_candidates:
        return None
    return [random.choice(dead_candidates)]


def _choose_markers(
    containers: list[Container],
    candidates: list[MarkerCandidate],
    mode: str,
    shape: tuple[int, int],
) -> list[MarkerCandidate] | None:
    if len(candidates) < 2:
        return None

    target_min = 3 if len(containers) >= 3 else 2
    target_max = min(5, len(candidates))
    if target_max < target_min:
        return None

    for _ in range(GEN_SELECT_ATTEMPTS):
        state = _empty_state(shape)
        seed = _seed_for_mode(mode, candidates, shape)
        if seed is None:
            return None
        valid_seed = True
        for candidate in seed:
            if not _candidate_fits(candidate, state):
                valid_seed = False
                break
            _add_candidate(candidate, state)
        if not valid_seed:
            continue

        target_count = random.randint(target_min, target_max)
        pool = candidates[:]
        random.shuffle(pool)

        while len(state["chosen"]) < target_count:
            chosen_list: list[MarkerCandidate] = state["chosen"]
            features = _selection_features(chosen_list, containers)
            need_h = "h" not in features["orientations"]
            need_v = "v" not in features["orientations"]

            options = [
                candidate
                for candidate in pool
                if candidate not in chosen_list and _candidate_fits(candidate, state)
            ]
            if not options:
                break

            if need_h:
                preferred = [candidate for candidate in options if candidate.orientation == "h"]
            elif need_v:
                preferred = [candidate for candidate in options if candidate.orientation == "v"]
            elif mode == "mixed" and features["untouched"] <= 0:
                preferred = [candidate for candidate in options if any(hit.container_idx not in features["affected"] for hit in candidate.hits)]
            elif mode == "dual" and features["dual_hits"] == 0:
                preferred = [candidate for candidate in options if candidate.dual_hit]
            elif features["dead_sides"] == 0:
                preferred = [candidate for candidate in options if candidate.dead_sides >= 1]
            else:
                preferred = options

            candidate = random.choice(preferred or options)
            _add_candidate(candidate, state)

        chosen = list(state["chosen"])
        features = _selection_features(chosen, containers)
        if len(chosen) < 2:
            continue
        if features["orientations"] != {"h", "v"}:
            continue
        if int(features["dead_sides"]) < 1:
            continue
        if len(features["affected"]) < 2:
            continue
        if mode == "cross" and not features["cross_targets"]:
            continue
        if mode == "dual" and int(features["dual_hits"]) < 1:
            continue
        if mode == "mixed" and int(features["untouched"]) < 1:
            continue
        return chosen

    return None


def _render_input(shape: tuple[int, int], containers: list[Container], chosen: list[MarkerCandidate]) -> np.ndarray:
    grid = _render_container_grid(shape, containers)
    colors = random.sample(GEN_PALETTE, len(chosen))
    for candidate, color in zip(chosen, colors):
        for r, c in candidate.cells:
            grid[r, c] = int(color)
    return grid


def _generate_attempt() -> np.ndarray | None:
    mode = random.choice(GEN_MODES)
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    container_count = random.randint(3, 5) if mode == "mixed" else random.randint(2, 4)
    containers = _place_containers((height, width), container_count)
    if containers is None:
        return None

    candidates = _enumerate_candidates(containers, (height, width))
    chosen = _choose_markers(containers, candidates, mode, (height, width))
    if chosen is None:
        return None

    grid = _render_input((height, width), containers, chosen)
    out = main(grid)
    if not isinstance(out, np.ndarray) or out.ndim != 2 or max(out.shape) > 30:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-distance-stripes input")
