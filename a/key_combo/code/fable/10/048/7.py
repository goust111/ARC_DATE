from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

MAX_GENERATION_ATTEMPTS = 400
BAR_LENGTH_VALUES = tuple(range(2, 8))
BAR_SCENARIOS = ("h_only", "v_only", "mixed")
SIDES = ("left", "right", "top", "bottom")


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def halo_mask(shape: tuple[int, int], coords: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    h, w = shape
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = True
    return out


def checkerboard_mask(
    shape: tuple[int, int], bbox: tuple[int, int, int, int]
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    out = np.zeros(shape, dtype=bool)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if ((r - r0) + (c - c0)) % 2 == 0:
                out[r, c] = True
    return out


def lattice_info(
    grid: np.ndarray,
) -> tuple[int, int, tuple[int, int, int, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    counts = Counter(map(int, g.ravel()))
    candidates = [
        color
        for color, count in counts.items()
        if color not in (bg, GREY) and count > 1
    ]
    assert len(candidates) == 1, "expected exactly one repeated non-grey foreground color"
    lattice_color = int(candidates[0])

    coords = [tuple(map(int, rc)) for rc in np.argwhere(g == lattice_color)]
    r0, c0, r1, c1 = bbox_from_coords(coords)
    bbox = (r0, c0, r1, c1)
    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    assert patch.shape[0] % 2 == 1 and patch.shape[1] % 2 == 1
    assert int(patch[0, 0]) == lattice_color

    expected = np.full(patch.shape, bg, dtype=int)
    expected[(np.indices(patch.shape).sum(axis=0) % 2) == 0] = lattice_color
    assert np.array_equal(patch, expected), "lattice must be a full checkerboard rectangle"

    lattice_mask = checkerboard_mask(g.shape, bbox)
    assert np.array_equal(g[lattice_mask], np.full(np.count_nonzero(lattice_mask), lattice_color))
    return bg, lattice_color, bbox, lattice_mask


def parse_bars(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bars: list[dict] = []
    for comp in component_coords(g == GREY):
        r0, c0, r1, c1 = bbox_from_coords(comp)
        if r0 == r1:
            cols = sorted(c for _, c in comp)
            assert cols == list(range(c0, c1 + 1))
            orientation = "h"
            endpoints = [(r0, c0), (r0, c1)]
            allowed = {
                (r0 - 1, c0),
                (r0 + 1, c0),
                (r0 - 1, c1),
                (r0 + 1, c1),
            }
        else:
            assert c0 == c1
            rows = sorted(r for r, _ in comp)
            assert rows == list(range(r0, r1 + 1))
            orientation = "v"
            endpoints = [(r0, c0), (r1, c0)]
            allowed = {
                (r0, c0 - 1),
                (r0, c0 + 1),
                (r1, c0 - 1),
                (r1, c0 + 1),
            }

        adjacent: set[tuple[int, int]] = set()
        for r, c in comp:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                    if g[nr, nc] not in (bg, GREY):
                        adjacent.add((nr, nc))
        assert len(adjacent) == 1, "each grey bar must have exactly one attached colored pixel"
        attachment = next(iter(adjacent))
        assert attachment in allowed, "attachment must touch an endpoint of the grey bar"

        bars.append(
            {
                "orientation": orientation,
                "length": len(comp),
                "bbox": (r0, c0, r1, c1),
                "attachment": attachment,
                "color": int(g[attachment]),
                "cells": comp,
                "endpoints": endpoints,
            }
        )
    assert bars, "expected at least one grey ruler bar"
    return bars


def axis_hits(
    axis_start: int, axis_end: int, period: int, residue: int
) -> tuple[int, ...]:
    return tuple(
        idx
        for idx in range(axis_start, axis_end + 1)
        if idx % period == residue % period
    )


def axis_signature_is_unique(
    axis_start: int, axis_end: int, period: int, residue: int
) -> bool:
    target = axis_hits(axis_start, axis_end, period, residue)
    if len(target) < 2:
        return False
    for alt_period in BAR_LENGTH_VALUES:
        for alt_residue in range(alt_period):
            if alt_period == period and alt_residue == residue % period:
                continue
            if axis_hits(axis_start, axis_end, alt_period, alt_residue) == target:
                return False
    return True


def bar_effect_mask(
    shape: tuple[int, int],
    lattice_mask: np.ndarray,
    lattice_bbox: tuple[int, int, int, int],
    orientation: str,
    period: int,
    attachment: tuple[int, int],
) -> np.ndarray:
    r0, c0, r1, c1 = lattice_bbox
    out = np.zeros(shape, dtype=bool)
    if orientation == "h":
        cols = axis_hits(c0, c1, period, attachment[1])
        out[:, list(cols)] = True
    else:
        rows = axis_hits(r0, r1, period, attachment[0])
        out[list(rows), :] = True
    return out & lattice_mask


def render_output_from_parsed(
    shape: tuple[int, int],
    bg: int,
    lattice_color: int,
    lattice_mask: np.ndarray,
    bars: list[dict],
    lattice_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    out[lattice_mask] = lattice_color
    painted = np.zeros(shape, dtype=bool)
    for bar in bars:
        effect = bar_effect_mask(
            shape,
            lattice_mask,
            lattice_bbox,
            bar["orientation"],
            bar["length"],
            bar["attachment"],
        )
        assert np.any(effect), "every bar must recolor at least one lattice foreground pixel"
        assert not np.any(painted & effect), "bar effects must be disjoint for determinism"
        painted |= effect
        out[effect] = bar["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor the lattice foreground on ruler-selected periodic rows or columns, then erase the remote controls.

    Input structure:
    - One side of the grid contains a rectangular checkerboard lattice made from
      the background color and a single repeated foreground color.
    - Outside that lattice, the blank area contains one or more straight grey
      ruler bars, each with exactly one colored pixel attached to one endpoint.

    Rule:
    - Interpret each horizontal grey bar as a column controller: its length is
      the period and the attached pixel's column gives the phase.
    - Interpret each vertical grey bar as a row controller: its length is the
      period and the attached pixel's row gives the phase.
    - Inside the lattice only, recolor foreground lattice cells lying on the
      selected congruent columns or rows to the attached pixel's color.
    - Leave lattice background cells unchanged and erase every grey bar plus its
      attached control pixel back to the background color.

    Key insight:
    - The lattice itself carries no period information; each remote grey bar
      encodes axis, period, phase, and repaint color through its orientation,
      length, endpoint alignment, and attached pixel color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, lattice_color, lattice_bbox, lattice_mask = lattice_info(g)
    bars = parse_bars(g, bg)
    return render_output_from_parsed(g.shape, bg, lattice_color, lattice_mask, bars, lattice_bbox)


def choose_layout() -> tuple[tuple[int, int], tuple[int, int, int, int], tuple[int, int, int, int]]:
    side = random.choice(SIDES)
    gap = random.randint(2, 4)
    lattice_h = random.choice([5, 7, 9, 11, 13])
    lattice_w = random.choice([5, 7, 9, 11, 13])

    if side in ("left", "right"):
        stripe_w = random.randint(6, 10)
        extra_top = random.randint(0, 3)
        extra_bottom = random.randint(0, 3)
        height = lattice_h + extra_top + extra_bottom
        width = lattice_w + gap + stripe_w
        if side == "left":
            lattice_bbox = (extra_top, 0, extra_top + lattice_h - 1, lattice_w - 1)
            stripe_bbox = (0, lattice_w + gap, height - 1, width - 1)
        else:
            lattice_bbox = (extra_top, width - lattice_w, extra_top + lattice_h - 1, width - 1)
            stripe_bbox = (0, 0, height - 1, width - lattice_w - gap - 1)
    else:
        stripe_h = random.randint(6, 10)
        extra_left = random.randint(0, 3)
        extra_right = random.randint(0, 3)
        height = lattice_h + gap + stripe_h
        width = lattice_w + extra_left + extra_right
        if side == "top":
            lattice_bbox = (0, extra_left, lattice_h - 1, extra_left + lattice_w - 1)
            stripe_bbox = (lattice_h + gap, 0, height - 1, width - 1)
        else:
            lattice_bbox = (height - lattice_h, extra_left, height - 1, extra_left + lattice_w - 1)
            stripe_bbox = (0, 0, height - lattice_h - gap - 1, width - 1)

    assert max(height, width) <= 30
    return (height, width), lattice_bbox, stripe_bbox


def add_lattice(grid: np.ndarray, lattice_bbox: tuple[int, int, int, int], color: int) -> np.ndarray:
    mask = checkerboard_mask(grid.shape, lattice_bbox)
    grid[mask] = color
    return mask


def structure_coords(
    shape: tuple[int, int],
    stripe_bbox: tuple[int, int, int, int],
    orientation: str,
    length: int,
    anchor_a: int,
    anchor_b: int,
    attach_side: int,
    attach_end: str,
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    sr0, sc0, sr1, sc1 = stripe_bbox
    if orientation == "h":
        row = anchor_a
        c0 = anchor_b
        c1 = c0 + length - 1
        if not (sr0 <= row <= sr1 and sc0 <= c0 <= c1 <= sc1):
            return None
        endpoint_c = c0 if attach_end == "start" else c1
        attachment = (row + attach_side, endpoint_c)
        if not (sr0 <= attachment[0] <= sr1 and sc0 <= attachment[1] <= sc1):
            return None
        bar = [(row, c) for c in range(c0, c1 + 1)]
        return bar, attachment

    col = anchor_a
    r0 = anchor_b
    r1 = r0 + length - 1
    if not (sc0 <= col <= sc1 and sr0 <= r0 <= r1 <= sr1):
        return None
    endpoint_r = r0 if attach_end == "start" else r1
    attachment = (endpoint_r, col + attach_side)
    if not (sr0 <= attachment[0] <= sr1 and sc0 <= attachment[1] <= sc1):
        return None
    bar = [(r, col) for r in range(r0, r1 + 1)]
    return bar, attachment


def candidate_bars(
    shape: tuple[int, int],
    stripe_bbox: tuple[int, int, int, int],
    lattice_mask: np.ndarray,
    lattice_bbox: tuple[int, int, int, int],
    reserved: np.ndarray,
    painted: np.ndarray,
    color: int,
    orientations: tuple[str, ...],
) -> list[dict]:
    sr0, sc0, sr1, sc1 = stripe_bbox
    candidates: list[dict] = []
    for orientation in orientations:
        for length in BAR_LENGTH_VALUES:
            if orientation == "h" and length > (sc1 - sc0 + 1):
                continue
            if orientation == "v" and length > (sr1 - sr0 + 1):
                continue

            if orientation == "h":
                axis_start, axis_end = lattice_bbox[1], lattice_bbox[3]
                row_values = range(sr0, sr1 + 1)
                col_values = range(sc0, sc1 - length + 2)
                for row in row_values:
                    for c0 in col_values:
                        for attach_side in (-1, 1):
                            for attach_end in ("start", "end"):
                                built = structure_coords(
                                    shape,
                                    stripe_bbox,
                                    "h",
                                    length,
                                    row,
                                    c0,
                                    attach_side,
                                    attach_end,
                                )
                                if built is None:
                                    continue
                                bar_cells, attachment = built
                                if not axis_signature_is_unique(
                                    axis_start, axis_end, length, attachment[1]
                                ):
                                    continue
                                effect = bar_effect_mask(
                                    shape, lattice_mask, lattice_bbox, "h", length, attachment
                                )
                                if not np.any(effect) or np.any(effect & painted):
                                    continue
                                footprint = bar_cells + [attachment]
                                footprint_mask = np.zeros(shape, dtype=bool)
                                for r, c in footprint:
                                    footprint_mask[r, c] = True
                                if np.any(footprint_mask & reserved):
                                    continue
                                candidates.append(
                                    {
                                        "orientation": "h",
                                        "length": length,
                                        "bar_cells": bar_cells,
                                        "attachment": attachment,
                                        "color": color,
                                        "effect": effect,
                                        "halo": halo_mask(shape, footprint),
                                    }
                                )
            else:
                axis_start, axis_end = lattice_bbox[0], lattice_bbox[2]
                col_values = range(sc0, sc1 + 1)
                row_values = range(sr0, sr1 - length + 2)
                for col in col_values:
                    for r0 in row_values:
                        for attach_side in (-1, 1):
                            for attach_end in ("start", "end"):
                                built = structure_coords(
                                    shape,
                                    stripe_bbox,
                                    "v",
                                    length,
                                    col,
                                    r0,
                                    attach_side,
                                    attach_end,
                                )
                                if built is None:
                                    continue
                                bar_cells, attachment = built
                                if not axis_signature_is_unique(
                                    axis_start, axis_end, length, attachment[0]
                                ):
                                    continue
                                effect = bar_effect_mask(
                                    shape, lattice_mask, lattice_bbox, "v", length, attachment
                                )
                                if not np.any(effect) or np.any(effect & painted):
                                    continue
                                footprint = bar_cells + [attachment]
                                footprint_mask = np.zeros(shape, dtype=bool)
                                for r, c in footprint:
                                    footprint_mask[r, c] = True
                                if np.any(footprint_mask & reserved):
                                    continue
                                candidates.append(
                                    {
                                        "orientation": "v",
                                        "length": length,
                                        "bar_cells": bar_cells,
                                        "attachment": attachment,
                                        "color": color,
                                        "effect": effect,
                                        "halo": halo_mask(shape, footprint),
                                    }
                                )
    return candidates


def scenario_plan() -> tuple[str, list[str]]:
    scenario = random.choice(BAR_SCENARIOS)
    if scenario == "mixed":
        count = random.randint(2, 4)
        orientations = ["h", "v"]
        while len(orientations) < count:
            orientations.append(random.choice(["h", "v"]))
        random.shuffle(orientations)
        return scenario, orientations
    count = random.randint(1, 3)
    orientation = "h" if scenario == "h_only" else "v"
    return scenario, [orientation] * count


def semantic_checks(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        bg, lattice_color, lattice_bbox, lattice_mask = lattice_info(grid)
        bars = parse_bars(grid, bg)
        rebuilt = render_output_from_parsed(
            grid.shape, bg, lattice_color, lattice_mask, bars, lattice_bbox
        )
    except Exception:
        return False
    if not np.array_equal(rebuilt, expected):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    shape, lattice_bbox, stripe_bbox = choose_layout()
    bg = BG
    lattice_color = random.choice(PALETTE)
    attachment_colors = [c for c in PALETTE if c != lattice_color]

    grid = np.full(shape, bg, dtype=int)
    lattice_mask = add_lattice(grid, lattice_bbox, lattice_color)

    _scenario, orientation_plan = scenario_plan()
    if len(orientation_plan) > len(attachment_colors):
        return None
    colors = random.sample(attachment_colors, len(orientation_plan))

    reserved = np.zeros(shape, dtype=bool)
    painted = np.zeros(shape, dtype=bool)
    bars: list[dict] = []

    for orientation, color in zip(orientation_plan, colors):
        candidates = candidate_bars(
            shape,
            stripe_bbox,
            lattice_mask,
            lattice_bbox,
            reserved,
            painted,
            color,
            (orientation,),
        )
        if not candidates:
            return None
        chosen = random.choice(candidates)
        for r, c in chosen["bar_cells"]:
            grid[r, c] = GREY
        ar, ac = chosen["attachment"]
        grid[ar, ac] = color
        reserved |= chosen["halo"]
        painted |= chosen["effect"]
        bars.append(
            {
                "orientation": chosen["orientation"],
                "length": chosen["length"],
                "attachment": chosen["attachment"],
                "color": color,
            }
        )

    expected = np.full(shape, bg, dtype=int)
    expected[lattice_mask] = lattice_color
    for bar in bars:
        effect = bar_effect_mask(
            shape,
            lattice_mask,
            lattice_bbox,
            bar["orientation"],
            bar["length"],
            bar["attachment"],
        )
        expected[effect] = bar["color"]

    if not semantic_checks(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
