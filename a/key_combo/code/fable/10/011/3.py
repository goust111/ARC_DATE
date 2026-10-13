from __future__ import annotations

from collections import Counter
from importlib.machinery import SourceFileLoader
from math import ceil
from pathlib import Path

import numpy as np

try:
    _ARC_UTILS = SourceFileLoader(
        "_arc_utils_shared",
        "/home/lixinyu/snap/new/solvers/arc_utils.py",
    ).load_module()
    get_bg = _ARC_UTILS.get_bg
    find_components = _ARC_UTILS.find_components
except Exception:  # pragma: no cover - fallback only
    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])

    def _neighbors(mask: np.ndarray, r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if connectivity == 4 and abs(dr) + abs(dc) != 1:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]:
                    out.append((rr, cc))
        return out

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors=None,
        min_area: int = 1,
        return_format: str = "dict",
    ) -> list[dict]:
        g = np.asarray(grid, dtype=int)
        if bg is None:
            bg = get_bg(g)
        assert monochromatic and return_format == "dict"
        if colors is None:
            colors = sorted(set(map(int, g.ravel())) - {int(bg)})
        comps: list[dict] = []
        for color in colors:
            mask = g == int(color)
            seen = np.zeros_like(mask, dtype=bool)
            for r, c in np.argwhere(mask):
                r = int(r)
                c = int(c)
                if seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                coords: list[tuple[int, int]] = []
                while stack:
                    rr, cc = stack.pop()
                    coords.append((rr, cc))
                    for nr, nc in _neighbors(mask, rr, cc, connectivity):
                        if not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                if len(coords) < min_area:
                    continue
                arr = np.array(coords, dtype=int)
                r0, c0 = arr.min(axis=0)
                r1, c1 = arr.max(axis=0)
                comps.append(
                    {
                        "color": int(color),
                        "coords": arr,
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "area": int(arr.shape[0]),
                    }
                )
        return comps


RNG = np.random.default_rng()
PALETTE = tuple(range(10))
CHAIN_LEN_RANGE = (4, 6)
DISTRACTOR_RANGE = (1, 3)
CELL_SIZE_CHOICES = (6, 7)
LINE_LENGTH_BY_CELL = {6: (3, 4), 7: (3, 5)}
MAX_ATTEMPTS = 400
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))
FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))


def _coord_neighbors(coord: tuple[int, int], deltas: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    r, c = coord
    return [(r + dr, c + dc) for dr, dc in deltas]


def _component_endpoints(coords: np.ndarray) -> list[tuple[int, int]]:
    cells = {tuple(map(int, rc)) for rc in coords}
    degrees: dict[tuple[int, int], int] = {}
    for cell in cells:
        degrees[cell] = sum((nbr in cells) for nbr in _coord_neighbors(cell, EIGHT))
    assert all(deg in (1, 2) for deg in degrees.values()), "line must be a one-cell-thick simple path"
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "line must have exactly two endpoints"
    return endpoints


def _parse_lines_and_markers(grid: np.ndarray) -> tuple[int, dict[int, dict], dict[int, int], list[int]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True)

    lines: dict[int, dict] = {}
    markers_by_target: dict[int, list[tuple[int, int]]] = {}

    for comp in comps:
        color = int(comp["color"])
        area = int(comp["area"])
        if area == 1:
            coord = tuple(map(int, comp["coords"][0]))
            markers_by_target.setdefault(color, []).append(coord)
            continue
        assert area >= 3, "line segments must be length >= 3"
        assert color not in lines, "each line color must define exactly one segment"
        lines[color] = {
            "color": color,
            "coords": comp["coords"],
            "bbox": comp["bbox"],
            "area": area,
            "endpoints": _component_endpoints(comp["coords"]),
            "out_targets": [],
            "marker_coords": [],
        }

    assert len(lines) >= 3, "expected several colored line objects"

    indegree = {color: 0 for color in lines}
    for target_color, marker_coords in markers_by_target.items():
        assert target_color in lines, "every marker color must name a real line"
        for marker in marker_coords:
            source_candidates: list[tuple[int, int]] = []
            for source_color, line in lines.items():
                endpoint_hits = [
                    idx
                    for idx, endpoint in enumerate(line["endpoints"])
                    if abs(endpoint[0] - marker[0]) + abs(endpoint[1] - marker[1]) == 1
                ]
                if endpoint_hits:
                    assert len(endpoint_hits) == 1
                    source_candidates.append((source_color, endpoint_hits[0]))
            assert len(source_candidates) == 1, "each marker must attach to exactly one line endpoint"
            source_color, endpoint_idx = source_candidates[0]
            assert source_color != target_color, "markers may not point to their own line"
            line = lines[source_color]
            line["out_targets"].append(target_color)
            line["marker_coords"].append(marker)
            line.setdefault("marker_endpoints", []).append(endpoint_idx)
            indegree[target_color] += 1

    for line in lines.values():
        out_targets = list(map(int, line["out_targets"]))
        assert len(out_targets) <= 2, "a line can expose at most two endpoint markers"
        assert len(set(out_targets)) == len(out_targets), "a line cannot repeat the same target marker"
        if len(out_targets) == 2:
            marker_endpoints = line.get("marker_endpoints", [])
            assert len(set(marker_endpoints)) == 2, "two markers must occupy the two distinct endpoints"

    distractors = [
        color
        for color, line in lines.items()
        if indegree[color] == 0 and len(line["out_targets"]) == 0
    ]
    return bg, lines, indegree, distractors


def _follow_chain(lines: dict[int, dict], indegree: dict[int, int]) -> list[int]:
    head_candidates = [
        color for color, line in lines.items() if indegree[color] == 0 and len(line["out_targets"]) > 0
    ]
    tail_candidates = [
        color for color, line in lines.items() if indegree[color] > 0 and len(line["out_targets"]) == 0
    ]
    assert len(head_candidates) == 1, "there must be a unique tagged but unreferenced head"
    assert len(tail_candidates) == 1, "there must be a unique referenced but untagged tail"

    active = {
        color
        for color, line in lines.items()
        if indegree[color] > 0 or len(line["out_targets"]) > 0
    }
    head = head_candidates[0]
    order: list[int] = []
    visited: set[int] = set()
    current = head

    while True:
        assert current not in visited, "chain may not revisit a line"
        order.append(current)
        visited.add(current)
        unseen = [target for target in lines[current]["out_targets"] if target not in visited]
        if len(lines[current]["out_targets"]) == 0:
            break
        assert len(unseen) == 1, "from each visited line exactly one marker must lead forward"
        current = unseen[0]

    assert visited == active, "chain traversal must cover every non-distractor line exactly once"
    return order


def _analyze(grid: np.ndarray) -> dict:
    bg, lines, indegree, distractors = _parse_lines_and_markers(grid)
    order = _follow_chain(lines, indegree)
    return {
        "bg": bg,
        "lines": lines,
        "indegree": indegree,
        "distractors": distractors,
        "order": order,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Lock onto the unique unreferenced tagged head, follow the endpoint marker chain, skip fake back-pointers, and compress the chain into color stripes.

    Input structure:
    - A dominant background surrounds several one-cell-thick colored line segments, each
      with a unique body color.
    - Chain lines may carry one or two single-pixel endpoint tags of other colors; each tag
      color matches the body color of a referenced line segment.
    - Some extra isolated line segments are distractors: they carry no endpoint tags and
      their body colors are never referenced by any tag.

    Rule:
    - Treat each single-pixel tag as an outgoing pointer from the endpoint it touches to the
      line whose body uses that same color.
    - Identify the unique chain head as the only line that has at least one outgoing tag but
      is never referenced by any tag.
    - Starting from that head, repeatedly follow the unique tag that points to an unvisited
      line, ignoring any companion tag that points back to an already visited line, until the
      unique referenced but untagged tail is reached.
    - Output an N x 3 grid, where N is the number of visited chain lines, and fill row k with
      the body color of the k-th line in traversal order.

    Key insight:
    - Geometry is only bait: the solution comes from reconstructing a color-coded pointer graph,
      using the combined out-edge/no-in-edge test to find the head and the visited set to reject
      fake back-pointers.
    """
    info = _analyze(np.asarray(grid, dtype=int))
    order = info["order"]
    out = np.zeros((len(order), 3), dtype=int)
    for r, color in enumerate(order):
        out[r, :] = int(color)
    return out


def _segment_candidates(cell_size: int) -> list[dict]:
    lo, hi = LINE_LENGTH_BY_CELL[cell_size]
    candidates: list[dict] = []
    for length in range(lo, hi + 1):
        for dr, dc in DIRS:
            if dc == -1:
                c_starts = range(length - 1, cell_size - 1)
            else:
                c_starts = range(1, cell_size - 1)
            r_starts = range(1, cell_size - 1)
            for r0 in r_starts:
                for c0 in c_starts:
                    cells = [(r0 + i * dr, c0 + i * dc) for i in range(length)]
                    if not all(0 <= r < cell_size and 0 <= c < cell_size for r, c in cells):
                        continue
                    if not all(1 <= r <= cell_size - 2 and 1 <= c <= cell_size - 2 for r, c in cells):
                        continue
                    endpoints = [cells[0], cells[-1]]
                    by_endpoint: list[list[tuple[int, int]]] = []
                    ok = True
                    cell_set = set(cells)
                    for endpoint in endpoints:
                        picks = []
                        for nr, nc in _coord_neighbors(endpoint, FOUR):
                            if not (0 <= nr < cell_size and 0 <= nc < cell_size):
                                continue
                            if (nr, nc) in cell_set:
                                continue
                            adjacent_line = [
                                cell
                                for cell in cell_set
                                if abs(cell[0] - nr) + abs(cell[1] - nc) == 1
                            ]
                            if adjacent_line != [endpoint]:
                                continue
                            picks.append((nr, nc))
                        if not picks:
                            ok = False
                            break
                        by_endpoint.append(picks)
                    if ok:
                        candidates.append(
                            {
                                "cells": cells,
                                "endpoints": endpoints,
                                "marker_slots": by_endpoint,
                            }
                        )
    assert candidates, "expected compartment line candidates"
    return candidates


_CANDIDATE_CACHE = {size: _segment_candidates(size) for size in CELL_SIZE_CHOICES}


def _choose_layout(n_objects: int) -> tuple[int, int]:
    possible = [(2, ceil(n_objects / 2)), (3, ceil(n_objects / 3))]
    possible = [(rows, cols) for cols, rows in possible if rows * cols >= n_objects and rows <= 3]
    return possible[RNG.integers(0, len(possible))]


def _place_attempt() -> tuple[np.ndarray, list[int]] | None:
    chain_len = int(RNG.integers(CHAIN_LEN_RANGE[0], CHAIN_LEN_RANGE[1] + 1))
    max_distractors = min(DISTRACTOR_RANGE[1], 9 - chain_len)
    min_distractors = min(DISTRACTOR_RANGE[0], max_distractors)
    distractor_count = int(RNG.integers(min_distractors, max_distractors + 1))
    bg = int(RNG.choice(PALETTE))
    usable_colors = [color for color in PALETTE if color != bg]
    RNG.shuffle(usable_colors)
    chain_colors = usable_colors[:chain_len]
    distractor_colors = usable_colors[chain_len : chain_len + distractor_count]

    internal = list(range(1, chain_len - 1))
    fake_count = int(RNG.integers(1, len(internal) + 1))
    fake_sources = set(map(int, RNG.choice(internal, size=fake_count, replace=False)))

    outgoing: dict[int, list[int]] = {color: [] for color in chain_colors}
    for idx, color in enumerate(chain_colors[:-1]):
        outgoing[color].append(chain_colors[idx + 1])
        if idx in fake_sources:
            fake_target = chain_colors[int(RNG.integers(1, idx + 1))]
            outgoing[color].append(fake_target)
        RNG.shuffle(outgoing[color])
    outgoing[chain_colors[-1]] = []

    objects = chain_colors + distractor_colors
    rows, cols = _choose_layout(len(objects))
    cell_size = int(RNG.choice(CELL_SIZE_CHOICES))
    grid = np.full((rows * cell_size, cols * cell_size), bg, dtype=int)

    compartments = [(r, c) for r in range(rows) for c in range(cols)]
    RNG.shuffle(compartments)
    assignment = {color: compartments[idx] for idx, color in enumerate(objects)}
    templates = {color: _CANDIDATE_CACHE[cell_size][int(RNG.integers(len(_CANDIDATE_CACHE[cell_size])))] for color in objects}

    placed_meta: dict[int, dict] = {}
    for color in objects:
        cr, cc = assignment[color]
        top = cr * cell_size
        left = cc * cell_size
        tpl = templates[color]
        global_cells = [(top + r, left + c) for r, c in tpl["cells"]]
        for r, c in global_cells:
            grid[r, c] = int(color)
        placed_meta[color] = {
            "cells": global_cells,
            "endpoints": [(top + r, left + c) for r, c in tpl["endpoints"]],
            "marker_slots": [
                [(top + r, left + c) for r, c in slots] for slots in tpl["marker_slots"]
            ],
            "compartment": assignment[color],
        }

    for source_color in chain_colors[:-1]:
        targets = outgoing[source_color]
        source = placed_meta[source_color]
        endpoint_order = [0, 1]
        RNG.shuffle(endpoint_order)
        if len(targets) == 1:
            endpoint_order = endpoint_order[:1]
        else:
            assert len(targets) == 2
        for endpoint_idx, target_color in zip(endpoint_order, targets):
            slots = source["marker_slots"][endpoint_idx]
            marker_r, marker_c = slots[int(RNG.integers(0, len(slots)))]
            assert grid[marker_r, marker_c] == bg
            grid[marker_r, marker_c] = int(target_color)

    return grid, chain_colors


def _ordering_signature(info: dict, colors: list[int], key) -> list[int]:
    return sorted(colors, key=lambda color: key(info["lines"][color]))


def _validate_generated(grid: np.ndarray, expected_chain: list[int]) -> bool:
    info = _analyze(grid)
    if info["order"] != list(map(int, expected_chain)):
        return False
    if len(info["distractors"]) < 1:
        return False
    if not any(len(info["lines"][color]["out_targets"]) == 2 for color in expected_chain[:-1]):
        return False

    colors = list(expected_chain)
    top_order = _ordering_signature(info, colors, lambda line: (line["bbox"][0], line["bbox"][1]))
    left_order = _ordering_signature(info, colors, lambda line: (line["bbox"][1], line["bbox"][0]))
    length_order = _ordering_signature(info, colors, lambda line: (line["area"], line["bbox"][0], line["bbox"][1]))
    rev_length_order = list(reversed(length_order))
    if colors in (top_order, left_order, length_order, rev_length_order):
        return False

    markerful = [color for color in colors if len(info["lines"][color]["out_targets"]) == 2]
    if markerful:
        endpoint_patterns = {
            tuple(sorted(info["lines"][color].get("marker_endpoints", []))) for color in markerful
        }
        if endpoint_patterns != {(0, 1)}:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        attempt = _place_attempt()
        if attempt is None:
            continue
        grid, expected_chain = attempt
        try:
            if _validate_generated(grid, expected_chain):
                out = main(grid)
                assert out.shape == (len(expected_chain), 3)
                assert np.array_equal(out[:, 0], np.array(expected_chain, dtype=int))
                return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid fake-pointer chain puzzle")


if __name__ == "__main__":
    path = Path(__file__)
    sample = generate_input()
    print(path.name, sample.shape, main(sample).shape)
