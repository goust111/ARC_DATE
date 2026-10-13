from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color

BLOCK = 3
SIDE_CHOICES = tuple(range(12, 31, 3))
MIN_ACTIVE_BLOCKS = 4
CORNER_CASE_PROB = 0.45
_RECENT_OUTPUT_SIGNATURES: set[tuple[int, tuple[tuple[int, int], ...]]] = set()


def _step_with_reflection(
    r: int, c: int, dr: int, dc: int, side: int
) -> tuple[int, int, int, int, bool, bool]:
    nr = r + dr
    nc = c + dc
    hit_vertical = nr < 0 or nr >= side
    hit_horizontal = nc < 0 or nc >= side

    if hit_vertical:
        dr = -dr
        nr = r + dr
    if hit_horizontal:
        dc = -dc
        nc = c + dc

    return nr, nc, dr, dc, hit_vertical, hit_horizontal


def _trace_ray(
    side: int, start: tuple[int, int]
) -> tuple[set[tuple[int, int]], dict[str, int | bool]]:
    r, c = map(int, start)
    dr, dc = -1, 1
    visited = {(r, c)}
    stats = {
        "closed": False,
        "horizontal_bounces": 0,
        "vertical_bounces": 0,
        "corner_bounces": 0,
        "steps": 0,
    }

    limit = side**4
    for step in range(1, limit + 1):
        r, c, dr, dc, hit_vertical, hit_horizontal = _step_with_reflection(
            r, c, dr, dc, side
        )
        visited.add((r, c))
        if hit_vertical:
            stats["vertical_bounces"] += 1
        if hit_horizontal:
            stats["horizontal_bounces"] += 1
        if hit_vertical and hit_horizontal:
            stats["corner_bounces"] += 1
        if (r, c) == start:
            stats["closed"] = True
            stats["steps"] = step
            return visited, stats

    stats["steps"] = limit
    return visited, stats


def _active_blocks_from_pixels(
    pixels: set[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    blocks = {
        (r // BLOCK, c // BLOCK)
        for r, c in pixels
        if r % BLOCK == 1 and c % BLOCK == 1
    }
    return tuple(sorted(blocks))


def _render_output(
    side: int, active_blocks: tuple[tuple[int, int], ...]
) -> np.ndarray:
    out = np.full((side, side), Color.BLACK, dtype=int)
    for br, bc in active_blocks:
        r0 = br * BLOCK
        c0 = bc * BLOCK
        out[r0 : r0 + BLOCK, c0 : c0 + BLOCK] = Color.GREEN
    return out


def _validate_input_grid(grid: np.ndarray) -> tuple[int, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    side_h, side_w = g.shape
    assert side_h == side_w, "expected a square grid"
    assert side_h % BLOCK == 0, "grid side must be a multiple of 3"
    assert side_h <= 30, "grid side must not exceed 30"
    assert np.all((g == Color.BLACK) | (g == Color.BLUE)), "input must be black with one blue pixel"

    blue = np.argwhere(g == Color.BLUE)
    assert blue.shape[0] == 1, "expected exactly one blue start pixel"
    start = tuple(map(int, blue[0]))
    assert start[0] % BLOCK == 1 and start[1] % BLOCK == 1, "start pixel must be at a 3x3 block center"
    return side_h, start


def _catalog_entry(side: int, start: tuple[int, int]) -> dict[str, object]:
    pixels, stats = _trace_ray(side, start)
    blocks = _active_blocks_from_pixels(pixels)
    return {
        "side": side,
        "start": start,
        "pixels": pixels,
        "active_blocks": blocks,
        "active_count": len(blocks),
        "corner_case": bool(stats["corner_bounces"]),
        "stats": stats,
        "signature": (side, blocks),
    }


def _build_catalog() -> list[dict[str, object]]:
    catalog_by_signature: dict[tuple[int, tuple[tuple[int, int], ...]], dict[str, object]] = {}

    for side in SIDE_CHOICES:
        macro_side = side // BLOCK
        for br in range(macro_side):
            for bc in range(macro_side):
                start = (br * BLOCK + 1, bc * BLOCK + 1)
                entry = _catalog_entry(side, start)
                if entry["active_count"] < MIN_ACTIVE_BLOCKS:
                    continue
                signature = entry["signature"]
                if signature not in catalog_by_signature:
                    catalog_by_signature[signature] = entry

    catalog = sorted(
        catalog_by_signature.values(),
        key=lambda entry: (
            int(entry["side"]),
            int(entry["active_count"]),
            int(entry["start"][0]),
            int(entry["start"][1]),
        ),
    )
    assert catalog, "failed to build a non-empty generator catalog"
    return catalog


_GENERATOR_CATALOG = _build_catalog()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert a hidden diagonal bounce path into solid green 3x3 macro blocks.

    Input structure:
    - The grid is a square black canvas whose side length is a multiple of 3.
    - Exactly one blue pixel appears, and it sits at the center cell of a 3x3 macro block.
    - The blue pixel is the ray start; no other visible objects are present.

    Rule:
    - Start from the blue pixel and move one cell at a time in the up-right diagonal direction.
    - When the next step would leave the canvas, reflect that axis at the boundary; a corner hit flips both axes.
    - Continue until the ray returns to the start cell, collecting every visited pixel on the hidden trajectory.
    - For each 3x3 macro block, fill the whole block green iff its center cell lies on the visited trajectory; otherwise keep it black.
    - Do not draw the ray itself and do not preserve the blue start marker.

    Key insight:
    - The visible answer is determined only by which macro-block centers the invisible bouncing ray passes through.
    """
    side, start = _validate_input_grid(grid)
    pixels, _stats = _trace_ray(side, start)
    active_blocks = _active_blocks_from_pixels(pixels)
    return _render_output(side, active_blocks)


def _validate_example(
    grid: np.ndarray,
    expected_signature: tuple[int, tuple[tuple[int, int], ...]] | None = None,
    require_corner_case: bool | None = None,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    side, start = _validate_input_grid(grid)
    pixels, stats = _trace_ray(side, start)
    assert bool(stats["closed"]), "trajectory must close by returning to the start"
    assert int(stats["horizontal_bounces"]) > 0, "trajectory must hit a left/right wall"
    assert int(stats["vertical_bounces"]) > 0, "trajectory must hit a top/bottom wall"

    active_blocks = _active_blocks_from_pixels(pixels)
    assert len(active_blocks) >= MIN_ACTIVE_BLOCKS, "generator must avoid underspecified tiny outputs"

    active_rows = {br for br, _bc in active_blocks}
    active_cols = {bc for _br, bc in active_blocks}
    assert len(active_rows) >= 2 and len(active_cols) >= 2, "activated blocks must span both axes"

    if require_corner_case is not None:
        assert bool(stats["corner_bounces"]) == require_corner_case, "corner-branch requirement not met"

    expected = _render_output(side, active_blocks)
    actual = main(grid)
    assert np.array_equal(actual, expected), "main() must match the semantic oracle"

    signature = (side, active_blocks)
    if expected_signature is not None:
        assert signature == expected_signature, "generated example does not match its catalog signature"
    return signature


def _choose_catalog_entry(rng: np.random.Generator) -> dict[str, object]:
    available = [
        entry
        for entry in _GENERATOR_CATALOG
        if entry["signature"] not in _RECENT_OUTPUT_SIGNATURES
    ]
    if not available:
        _RECENT_OUTPUT_SIGNATURES.clear()
        available = list(_GENERATOR_CATALOG)

    want_corner_case = bool(rng.random() < CORNER_CASE_PROB)
    filtered = [
        entry for entry in available if bool(entry["corner_case"]) == want_corner_case
    ]
    if filtered:
        available = filtered

    sizes = sorted({int(entry["side"]) for entry in available})
    side = int(rng.choice(sizes))
    side_entries = [entry for entry in available if int(entry["side"]) == side]
    return side_entries[int(rng.integers(0, len(side_entries)))]


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(200):
        entry = _choose_catalog_entry(rng)
        side = int(entry["side"])
        start = tuple(map(int, entry["start"]))

        grid = np.full((side, side), Color.BLACK, dtype=int)
        grid[start] = Color.BLUE

        signature = _validate_example(
            grid,
            expected_signature=entry["signature"],
            require_corner_case=bool(entry["corner_case"]),
        )
        _RECENT_OUTPUT_SIGNATURES.add(signature)
        return grid

    raise RuntimeError("failed to generate a valid ARC example")
