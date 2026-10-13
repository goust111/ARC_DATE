from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 7
MARKER = 9
OUTLINE_COLORS = (1, 2, 3, 5)
GRID_SHAPE = (16, 16)
GEN_RETRIES = 500


def _neighbors(coords: tuple[int, int], connectivity: int) -> Iterable[tuple[int, int]]:
    r, c = coords
    if connectivity == 4:
        yield r - 1, c
        yield r + 1, c
        yield r, c - 1
        yield r, c + 1
        return
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors((rr, cc), connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _diamond_outline_coords(cr: int, cc: int, radius: int) -> list[tuple[int, int]]:
    if radius <= 0:
        return [(int(cr), int(cc))]
    pts: list[tuple[int, int]] = []
    for dr in range(-radius, radius + 1):
        dc = radius - abs(dr)
        rr = cr + dr
        if dc == 0:
            pts.append((rr, cc))
        else:
            pts.append((rr, cc - dc))
            pts.append((rr, cc + dc))
    return pts


def _diamond_interior_coords(cr: int, cc: int, radius: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for dr in range(-(radius - 1), radius):
        span = radius - 1 - abs(dr)
        rr = cr + dr
        for dc in range(-span, span + 1):
            pts.append((rr, cc + dc))
    return pts


def _paint_diamond(grid: np.ndarray, cr: int, cc: int, radius: int, color: int) -> None:
    for r, c in _diamond_outline_coords(int(cr), int(cc), int(radius)):
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            grid[r, c] = int(color)


def _extract_diamonds(grid: np.ndarray) -> list[dict]:
    specs: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if c not in (BG, MARKER)):
        for coords in _components(grid == color, connectivity=8):
            assert coords.size > 0
            r0, c0, r1, c1 = _bbox(coords)
            assert r1 - r0 == c1 - c0, "diamond bbox must be square"
            assert (r1 - r0) % 2 == 0, "diamond bbox side must be odd"
            cr = (r0 + r1) // 2
            cc = (c0 + c1) // 2
            radius = (r1 - r0) // 2
            expected = set(_diamond_outline_coords(cr, cc, radius))
            actual = {tuple(map(int, rc)) for rc in coords}
            assert actual == expected, "component must be a full unclipped diamond outline"
            specs.append({"color": int(color), "center": (cr, cc), "radius": int(radius)})
    assert specs, "expected at least one diamond outline"
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count maroon marker pixels inside each diamond outline and use parity to shrink or expand that outline.

    Input structure:
    - A 16x16 orange-background grid contains several one-pixel-thick Manhattan diamond outlines.
    - Each outline is monochromatic, unclipped, and centered on an integer grid cell.
    - Maroon marker pixels may appear strictly inside the diamonds and nowhere on the outlines.

    Rule:
    - Identify every diamond's center and radius from its outline component.
    - Count the maroon markers with Manhattan distance strictly smaller than that radius.
    - Replace each outline by the same-color outline of radius +1 when the count is even, or radius -1 when the count is odd.
    - Remove all maroon markers; if a shrunk radius reaches 0, keep only the center pixel.

    Key insight:
    - The markers do not survive into the output; only the parity of each diamond-local marker count determines whether that particular outline grows or contracts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    diamonds = _extract_diamonds(g)
    out = np.full_like(g, BG)
    markers = np.argwhere(g == MARKER)

    for spec in diamonds:
        cr, cc = spec["center"]
        radius = spec["radius"]
        inside = 0
        for mr, mc in markers:
            if abs(int(mr) - cr) + abs(int(mc) - cc) < radius:
                inside += 1
        new_radius = radius + 1 if inside % 2 == 0 else radius - 1
        _paint_diamond(out, cr, cc, new_radius, spec["color"])

    return out


def _sample_markers(cr: int, cc: int, radius: int, count: int) -> list[tuple[int, int]] | None:
    interior = _diamond_interior_coords(cr, cc, radius)
    random.shuffle(interior)
    chosen: list[tuple[int, int]] = []

    def rec(start: int) -> bool:
        if len(chosen) == count:
            return True
        remaining = count - len(chosen)
        if len(interior) - start < remaining:
            return False
        for idx in range(start, len(interior)):
            cell = interior[idx]
            if any(abs(cell[0] - rr) + abs(cell[1] - cc2) == 1 for rr, cc2 in chosen):
                continue
            chosen.append(cell)
            if rec(idx + 1):
                return True
            chosen.pop()
        return False

    return chosen if rec(0) else None


def _spec_output_radius(spec: dict) -> int:
    radius = int(spec["radius"])
    count = int(spec["count"])
    return radius + 1 if count % 2 == 0 else radius - 1


def _center_candidates(spec: dict, placed: list[dict]) -> list[tuple[int, int]]:
    radius = int(spec["radius"])
    out_radius = _spec_output_radius(spec)
    margin = max(radius, out_radius)
    if margin < 0:
        margin = 0
    candidates: list[tuple[int, int]] = []
    h, w = GRID_SHAPE
    for cr in range(margin, h - margin):
        for cc in range(margin, w - margin):
            ok = True
            for other in placed:
                d = abs(cr - other["center"][0]) + abs(cc - other["center"][1])
                req_in = radius + other["radius"] + 2
                req_out = out_radius + _spec_output_radius(other) + 2
                if d < max(req_in, req_out):
                    ok = False
                    break
            if ok:
                candidates.append((cr, cc))
    random.shuffle(candidates)
    return candidates


def _allowed_counts(radius: int, category: str) -> list[int]:
    if category == "zero":
        return [0]
    if category == "odd":
        return [n for n in (1, 3) if radius >= 3]
    if category == "even_positive":
        return [n for n in (2, 4)]
    raise ValueError(f"unknown category {category}")


def _build_specs() -> list[dict]:
    mode = random.choices(
        ["balanced3", "balanced4", "large3"],
        weights=[4, 3, 2],
        k=1,
    )[0]
    if mode == "balanced3":
        dup_radius = random.choice([3, 4])
        specs = [
            {"radius": dup_radius, "category": "odd"},
            {"radius": dup_radius, "category": "even_positive"},
            {"radius": random.choice([2, 3, 4]), "category": "zero"},
        ]
    elif mode == "balanced4":
        pair_radius = random.choice([3, 4])
        specs = [
            {"radius": pair_radius, "category": "odd"},
            {"radius": pair_radius, "category": "even_positive"},
            {"radius": 2, "category": "zero"},
            {"radius": random.choice([2, 3]), "category": "even_positive"},
        ]
    else:
        specs = [
            {"radius": 5, "category": "zero"},
            {"radius": random.choice([3, 4]), "category": "odd"},
            {"radius": random.choice([2, 3]), "category": "even_positive"},
        ]

    for spec in specs:
        choices = _allowed_counts(spec["radius"], spec["category"])
        spec["count"] = random.choice(choices)

    colors = random.sample(list(OUTLINE_COLORS), len(specs))
    random.shuffle(specs)
    for spec, color in zip(specs, colors):
        spec["color"] = color
    return specs


def _render_specs(specs: list[dict]) -> np.ndarray:
    grid = np.full(GRID_SHAPE, BG, dtype=int)
    for spec in specs:
        cr, cc = spec["center"]
        _paint_diamond(grid, cr, cc, spec["radius"], spec["color"])
        for mr, mc in spec["markers"]:
            grid[mr, mc] = MARKER
    return grid


def _same_specs(a: list[dict], b: list[dict]) -> bool:
    key = lambda spec: (spec["color"], spec["center"], spec["radius"])
    return sorted((key(spec) for spec in a)) == sorted((key(spec) for spec in b))


def _expected_output(specs: list[dict]) -> np.ndarray:
    out = np.full(GRID_SHAPE, BG, dtype=int)
    for spec in specs:
        cr, cc = spec["center"]
        _paint_diamond(out, cr, cc, _spec_output_radius(spec), spec["color"])
    return out


def _validate_specs(specs: list[dict], grid: np.ndarray) -> None:
    assert grid.shape == GRID_SHAPE
    assert np.all(np.isin(grid, [BG, MARKER, *OUTLINE_COLORS]))
    counts = [int(spec["count"]) for spec in specs]
    assert any(count % 2 == 0 for count in counts)
    assert any(count % 2 == 1 for count in counts)
    assert any(count == 0 for count in counts)
    assert any(count in (2, 4) for count in counts)
    assert any(count in (1, 3) for count in counts)

    for spec in specs:
        if spec["radius"] == 2:
            assert spec["count"] % 2 == 0
        assert len(spec["markers"]) == spec["count"]
        for mr, mc in spec["markers"]:
            assert abs(mr - spec["center"][0]) + abs(mc - spec["center"][1]) < spec["radius"]
        for i, (r0, c0) in enumerate(spec["markers"]):
            for r1, c1 in spec["markers"][i + 1 :]:
                assert abs(r0 - r1) + abs(c0 - c1) != 1

    for i, spec in enumerate(specs):
        for other in specs[i + 1 :]:
            d = abs(spec["center"][0] - other["center"][0]) + abs(spec["center"][1] - other["center"][1])
            req_in = spec["radius"] + other["radius"] + 2
            req_out = _spec_output_radius(spec) + _spec_output_radius(other) + 2
            assert d >= req_in
            assert d >= req_out

    extracted = _extract_diamonds(grid)
    assert _same_specs(specs, extracted), "solver-side extraction must match generator specs"

    marker_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == MARKER)}
    assert len(marker_cells) == sum(spec["count"] for spec in specs)
    for marker in marker_cells:
        owners = [
            spec
            for spec in specs
            if abs(marker[0] - spec["center"][0]) + abs(marker[1] - spec["center"][1]) < spec["radius"]
        ]
        assert len(owners) == 1, "every marker must belong to exactly one diamond interior"

    expected = _expected_output(specs)
    actual = main(grid)
    assert np.array_equal(actual, expected), "main() must implement the spec exactly"
    assert not np.any(actual == MARKER)


def _gen_attempt() -> np.ndarray | None:
    specs = _build_specs()
    ordered = sorted(specs, key=lambda spec: (max(spec["radius"], _spec_output_radius(spec)), spec["radius"]), reverse=True)
    placed: list[dict] = []

    for spec in ordered:
        markers_count = spec["count"]
        success = False
        for cr, cc in _center_candidates(spec, placed):
            markers = _sample_markers(cr, cc, spec["radius"], markers_count)
            if markers is None:
                continue
            spec["center"] = (cr, cc)
            spec["markers"] = markers
            placed.append(spec)
            success = True
            break
        if not success:
            return None

    grid = _render_specs(specs)
    try:
        _validate_specs(specs, grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_RETRIES} attempts")
