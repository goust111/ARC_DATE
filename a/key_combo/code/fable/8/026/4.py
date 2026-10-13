from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

try:
    _ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
    if str(_ARC_UTILS_DIR) not in sys.path:
        sys.path.append(str(_ARC_UTILS_DIR))
    from arc_utils import Color, get_bg
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GRAY = 5
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9

    def get_bg(grid: np.ndarray) -> int:
        values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
        return int(values[np.argmax(counts)])


Coord = tuple[int, int]

BG_COLOR = Color.BLACK
CHECKPOINT_COLOR = Color.GRAY
PAIR_COLORS = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]

GRID_HEIGHT_RANGE = (10, 24)
GRID_WIDTH_RANGE = (10, 24)
PAIR_COUNT_CHOICES = [3, 3, 4, 4]
MIN_SPAN = 2
MAX_SPAN = 9
PAIR_PLACEMENT_RETRIES = 150
CHECKPOINT_SET_RETRIES = 180
GENERATION_RETRIES = 700


def _segment_cells(a: Coord, b: Coord) -> list[Coord]:
    r0, c0 = map(int, a)
    r1, c1 = map(int, b)
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("segment must be horizontal or vertical")


def _l_path(end_a: Coord, end_b: Coord, orientation: int) -> frozenset[Coord]:
    (r0, c0), (r1, c1) = end_a, end_b
    if int(orientation) == 0:
        bend = (r0, c1)
    else:
        bend = (r1, c0)
    cells = set(_segment_cells((r0, c0), bend))
    cells.update(_segment_cells(bend, (r1, c1)))
    return frozenset(cells)


def _make_pair_spec(color: int, endpoints: tuple[Coord, Coord]) -> dict:
    end_a, end_b = endpoints
    path0 = _l_path(end_a, end_b, 0)
    path1 = _l_path(end_a, end_b, 1)
    return {
        "color": int(color),
        "endpoints": (tuple(map(int, end_a)), tuple(map(int, end_b))),
        "paths": (path0, path1),
    }


def _parse_pair_specs(grid: np.ndarray, bg: int) -> list[dict]:
    colors = [int(c) for c in np.unique(grid) if int(c) not in {int(bg), CHECKPOINT_COLOR}]
    assert colors, "expected at least one endpoint pair"

    pair_specs = []
    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.shape == (2, 2), "each endpoint color must appear exactly twice"
        endpoints = tuple(sorted((tuple(map(int, coords[0])), tuple(map(int, coords[1])))))
        (r0, c0), (r1, c1) = endpoints
        assert r0 != r1 and c0 != c1, "paired endpoints must be on different rows and columns"
        pair_specs.append(_make_pair_spec(color, endpoints))

    return sorted(pair_specs, key=lambda spec: spec["color"])


def _enumerate_reports(pair_specs: list[dict]) -> list[dict]:
    reports = []
    for bits in itertools.product((0, 1), repeat=len(pair_specs)):
        union: set[Coord] = set()
        disjoint = True
        for spec, bit in zip(pair_specs, bits):
            path = spec["paths"][bit]
            if union & path:
                disjoint = False
            union.update(path)
        reports.append(
            {
                "bits": tuple(bits),
                "disjoint": disjoint,
                "union": frozenset(union),
            }
        )
    return reports


def _render_solution(shape: tuple[int, int], pair_specs: list[dict], bits: tuple[int, ...], bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for spec, bit in zip(pair_specs, bits):
        color = int(spec["color"])
        for r, c in spec["paths"][bit]:
            out[r, c] = color
    return out


def _has_orientation_variety(reports: list[dict], n_pairs: int) -> bool:
    if not reports:
        return False
    for idx in range(n_pairs):
        values = {int(report["bits"][idx]) for report in reports}
        if values != {0, 1}:
            return False
    return True


def _semantic_summary_from_reports(
    pair_specs: list[dict],
    checkpoints: frozenset[Coord],
    all_reports: list[dict],
) -> dict:
    disjoint_reports = [report for report in all_reports if report["disjoint"]]
    cover_reports = [report for report in all_reports if checkpoints <= report["union"]]
    valid_reports = [report for report in disjoint_reports if checkpoints <= report["union"]]
    solution_bits = valid_reports[0]["bits"] if len(valid_reports) == 1 else None

    endpoint_cells = {endpoint for spec in pair_specs for endpoint in spec["endpoints"]}
    checkpoint_sources = set()
    if solution_bits is not None:
        for idx, spec in enumerate(pair_specs):
            chosen = spec["paths"][solution_bits[idx]]
            if any(cell in chosen for cell in checkpoints):
                checkpoint_sources.add(idx)

    all_checkpoint_essential = True
    for checkpoint in checkpoints:
        reduced = checkpoints - {checkpoint}
        reduced_valid = [report for report in disjoint_reports if reduced <= report["union"]]
        if len(reduced_valid) <= 1:
            all_checkpoint_essential = False
            break

    return {
        "all_reports": all_reports,
        "disjoint_reports": disjoint_reports,
        "cover_reports": cover_reports,
        "valid_reports": valid_reports,
        "solution_bits": solution_bits,
        "checkpoint_count": len(checkpoints),
        "checkpoints_touch_pairs": len(checkpoint_sources),
        "all_checkpoint_essential": all_checkpoint_essential,
        "disjoint_variety": _has_orientation_variety(disjoint_reports, len(pair_specs)),
        "cover_variety": _has_orientation_variety(cover_reports, len(pair_specs)),
        "endpoint_overlap_free": all(checkpoint not in endpoint_cells for checkpoint in checkpoints),
    }


def _semantic_summary_from_parts(pair_specs: list[dict], checkpoints: frozenset[Coord]) -> dict:
    return _semantic_summary_from_reports(pair_specs, checkpoints, _enumerate_reports(pair_specs))


def _semantic_summary(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    pair_specs = _parse_pair_specs(g, bg)
    checkpoints = frozenset(tuple(map(int, cell)) for cell in np.argwhere(g == CHECKPOINT_COLOR))
    summary = _semantic_summary_from_parts(pair_specs, checkpoints)
    summary["pair_count"] = len(pair_specs)
    summary["bg"] = bg
    return summary


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color endpoint pair by the uniquely valid non-overlapping L path that covers every gray checkpoint.

    Input structure:
    - A black background contains several endpoint pairs, each pair made of exactly two singleton pixels of the same non-gray color.
    - For every pair, the two endpoints lie on different rows and different columns, so they define two candidate single-bend orthogonal paths.
    - Additional gray singleton pixels act as checkpoints and lie on candidate paths.

    Rule:
    - For each color pair, choose exactly one of its two possible right-angle paths and draw the full inclusive path in that color.
    - The chosen paths must be pairwise cell-disjoint, and every gray checkpoint must lie on one chosen path and be recolored by that path.
    - The input is guaranteed to admit exactly one bend-choice combination satisfying both global conditions; output the completed colored paths on the original background.

    Key insight:
    - No pair can be settled in isolation: the correct bend assignment appears only after jointly enforcing checkpoint coverage and path disjointness across all pairs.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    pair_specs = _parse_pair_specs(g, bg)
    checkpoints = frozenset(tuple(map(int, cell)) for cell in np.argwhere(g == CHECKPOINT_COLOR))
    reports = _enumerate_reports(pair_specs)
    valid_reports = [report for report in reports if report["disjoint"] and checkpoints <= report["union"]]
    assert len(valid_reports) == 1, "expected exactly one globally valid bend assignment"

    return _render_solution(g.shape, pair_specs, valid_reports[0]["bits"], bg)


def _sample_endpoint_pair(height: int, width: int, used_endpoints: set[Coord]) -> tuple[Coord, Coord] | None:
    row_candidates = list(range(height))
    col_candidates = list(range(width))
    for _ in range(PAIR_PLACEMENT_RETRIES):
        r0, r1 = sorted(random.sample(row_candidates, 2))
        c0, c1 = sorted(random.sample(col_candidates, 2))
        if r1 - r0 < MIN_SPAN or c1 - c0 < MIN_SPAN:
            continue
        if r1 - r0 > MAX_SPAN or c1 - c0 > MAX_SPAN:
            continue
        if random.random() < 0.5:
            endpoints = ((r0, c0), (r1, c1))
        else:
            endpoints = ((r0, c1), (r1, c0))
        if endpoints[0] in used_endpoints or endpoints[1] in used_endpoints:
            continue
        return endpoints
    return None


def _build_random_pair_specs(height: int, width: int, pair_count: int) -> list[dict] | None:
    colors = random.sample(PAIR_COLORS, pair_count)
    used_endpoints: set[Coord] = set()
    pair_specs = []
    for color in colors:
        endpoints = _sample_endpoint_pair(height, width, used_endpoints)
        if endpoints is None:
            return None
        used_endpoints.update(endpoints)
        pair_specs.append(_make_pair_spec(color, endpoints))
    return pair_specs


def _candidate_checkpoint_cells(pair_specs: list[dict], bits: tuple[int, ...]) -> list[Coord]:
    endpoints = {endpoint for spec in pair_specs for endpoint in spec["endpoints"]}
    cells = []
    for spec, bit in zip(pair_specs, bits):
        for cell in sorted(spec["paths"][bit]):
            if cell not in endpoints:
                cells.append(cell)
    informative = set(cells)
    return sorted(informative)


def _pick_checkpoints(pair_specs: list[dict]) -> tuple[tuple[int, ...], frozenset[Coord]] | None:
    all_reports = _enumerate_reports(pair_specs)
    disjoint_reports = [report for report in all_reports if report["disjoint"]]
    if len(disjoint_reports) < 2:
        return None

    candidate_targets = disjoint_reports[:]
    random.shuffle(candidate_targets)
    for target in candidate_targets:
        target_bits = target["bits"]
        candidates = _candidate_checkpoint_cells(pair_specs, target_bits)
        if len(candidates) < 2:
            continue

        useful = [
            cell
            for cell in candidates
            if any(cell not in report["union"] for report in all_reports if report["bits"] != target_bits)
        ]
        if len(useful) < 2:
            continue

        max_size = min(6, len(useful))
        sizes = list(range(2, max_size + 1))
        random.shuffle(sizes)
        for size in sizes:
            for _ in range(CHECKPOINT_SET_RETRIES):
                checkpoints = frozenset(random.sample(useful, size))
                summary = _semantic_summary_from_reports(pair_specs, checkpoints, all_reports)
                if summary["solution_bits"] != target_bits:
                    continue
                if len(summary["valid_reports"]) != 1:
                    continue
                if len(summary["cover_reports"]) < 2:
                    continue
                if not summary["all_checkpoint_essential"]:
                    continue
                if not summary["endpoint_overlap_free"]:
                    continue
                if summary["checkpoints_touch_pairs"] < min(2, len(pair_specs)):
                    continue
                return target_bits, checkpoints
    return None


def _grid_from_instance(height: int, width: int, pair_specs: list[dict], checkpoints: frozenset[Coord]) -> np.ndarray:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    for spec in pair_specs:
        for r, c in spec["endpoints"]:
            grid[r, c] = int(spec["color"])
    for r, c in checkpoints:
        grid[r, c] = CHECKPOINT_COLOR
    return grid


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    pair_count = random.choice(PAIR_COUNT_CHOICES)

    pair_specs = _build_random_pair_specs(height, width, pair_count)
    if pair_specs is None:
        return None

    chosen = _pick_checkpoints(pair_specs)
    if chosen is None:
        return None
    target_bits, checkpoints = chosen

    grid = _grid_from_instance(height, width, pair_specs, checkpoints)
    expected = _render_solution(grid.shape, pair_specs, target_bits, BG_COLOR)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    summary = _semantic_summary(grid)
    if len(summary["valid_reports"]) != 1:
        return None
    if summary["solution_bits"] != target_bits:
        return None
    if len(summary["disjoint_reports"]) < 2 or len(summary["cover_reports"]) < 2:
        return None
    if not summary["all_checkpoint_essential"]:
        return None
    if summary["checkpoint_count"] < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid unique-checkpoint L-path instance")
