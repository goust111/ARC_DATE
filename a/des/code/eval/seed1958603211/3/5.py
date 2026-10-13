from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")

GEN_ATTEMPTS = 400
LAYOUT_ATTEMPTS = 120
GRID_MIN = 14
GRID_MAX = 30
OUTER_SIZE_MIN = 5
OUTER_SIZE_MAX = 8  # exclusive
GAP_MIN = 2
GAP_MAX = 5  # exclusive
MARGIN_MIN = 1
MARGIN_MAX = 4  # exclusive


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _growth_side(seed_r: int, seed_c: int, r0: int, c0: int, r1: int, c1: int) -> str:
    distances = {
        "top": int(seed_r - r0),
        "bottom": int(r1 - seed_r),
        "left": int(seed_c - c0),
        "right": int(c1 - seed_c),
    }
    best = min(distances.values())
    matches = [side for side, dist in distances.items() if dist == best]
    assert len(matches) == 1, "seed direction must be unambiguous"
    return matches[0]


def _objects(grid: np.ndarray) -> list[dict[str, int | str]]:
    colors = sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, GREY))
    found: list[dict[str, int | str]] = []
    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.size > 0
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        assert height >= 3 and width >= 3

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = _border_mask(height, width)
        assert np.all(sub[border] == color), "each object must be a rectangular ring"

        interior = sub[1:-1, 1:-1]
        interior_colors = {int(v) for v in np.unique(interior)}
        assert interior_colors <= {BG, GREY, color}

        seed_cells = np.argwhere(interior == color)
        assert len(seed_cells) == 1, "each ring must contain exactly one seed"
        seed_dr, seed_dc = map(int, seed_cells[0])
        seed_r = int(r0 + 1 + seed_dr)
        seed_c = int(c0 + 1 + seed_dc)

        found.append(
            {
                "color": color,
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "seed_r": seed_r,
                "seed_c": seed_c,
                "grey_count": int(np.count_nonzero(interior == GREY)),
                "side": _growth_side(seed_r, seed_c, r0, c0, r1, c1),
            }
        )
    return found


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the seed's biased position to choose a frame edge, then grow a same-color strip
    outward from that edge by a length equal to the number of grey markers inside.
    Grey markers are removed in the output.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for obj in _objects(g):
        color = int(obj["color"])
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        r1 = int(obj["r1"])
        c1 = int(obj["c1"])
        count = int(obj["grey_count"])
        side = str(obj["side"])

        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == GREY] = BG

        if count == 0:
            continue

        if side == "top":
            assert r0 - count >= 0
            out[r0 - count : r0, c0 : c1 + 1] = color
        elif side == "bottom":
            assert r1 + count < out.shape[0]
            out[r1 + 1 : r1 + 1 + count, c0 : c1 + 1] = color
        elif side == "left":
            assert c0 - count >= 0
            out[r0 : r1 + 1, c0 - count : c0] = color
        else:
            assert c1 + count < out.shape[1]
            out[r0 : r1 + 1, c1 + 1 : c1 + 1 + count] = color

    return out


def _seed_candidates(inner_h: int, inner_w: int, side: str) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for ir in range(inner_h):
        for ic in range(inner_w):
            distances = {
                "top": ir + 1,
                "bottom": inner_h - ir,
                "left": ic + 1,
                "right": inner_w - ic,
            }
            best = min(distances.values())
            if best > 2:
                continue
            matches = [name for name, value in distances.items() if value == best]
            if matches == [side]:
                candidates.append((ir, ic))
    return candidates


def _sample_marker_count(rng: np.random.Generator, capacity: int) -> int:
    if capacity <= 0:
        return 0
    limit = min(4, capacity)
    if rng.random() < 0.2:
        return 0
    return int(rng.integers(1, limit + 1))


def _finish_spec(
    side: str,
    color: int,
    outer_h: int,
    outer_w: int,
    seed: tuple[int, int],
    grey_positions: list[tuple[int, int]],
) -> dict[str, object]:
    count = len(grey_positions)
    if side in ("top", "bottom"):
        foot_h = outer_h + count
        foot_w = outer_w
        ring_dr = count if side == "top" else 0
        ring_dc = 0
    else:
        foot_h = outer_h
        foot_w = outer_w + count
        ring_dr = 0
        ring_dc = count if side == "left" else 0
    return {
        "side": side,
        "color": int(color),
        "outer_h": int(outer_h),
        "outer_w": int(outer_w),
        "seed": seed,
        "grey_positions": grey_positions,
        "grey_count": count,
        "foot_h": int(foot_h),
        "foot_w": int(foot_w),
        "ring_dr": int(ring_dr),
        "ring_dc": int(ring_dc),
    }


def _sample_specs(rng: np.random.Generator) -> list[dict[str, object]] | None:
    count = int(rng.integers(2, 5))
    colors = [int(v) for v in rng.choice(np.array(PALETTE, dtype=int), size=count, replace=False)]
    directions = [str(v) for v in rng.permutation(np.array(SIDES, dtype=object))[:count]]

    specs: list[dict[str, object]] = []
    total_markers = 0
    fallback_choices: list[tuple[int, list[tuple[int, int]]]] = []

    for color, side in zip(colors, directions, strict=False):
        outer_h = int(rng.integers(OUTER_SIZE_MIN, OUTER_SIZE_MAX))
        outer_w = int(rng.integers(OUTER_SIZE_MIN, OUTER_SIZE_MAX))
        inner_h = outer_h - 2
        inner_w = outer_w - 2

        seeds = _seed_candidates(inner_h, inner_w, side)
        if not seeds:
            return None
        seed = seeds[int(rng.integers(0, len(seeds)))]

        free_cells = [(r, c) for r in range(inner_h) for c in range(inner_w) if (r, c) != seed]
        grey_count = _sample_marker_count(rng, len(free_cells))
        grey_positions: list[tuple[int, int]] = []
        if grey_count:
            picks = rng.choice(len(free_cells), size=grey_count, replace=False)
            grey_positions = [free_cells[int(i)] for i in np.atleast_1d(picks)]
        else:
            fallback_choices.append((len(specs), free_cells))

        total_markers += grey_count
        specs.append(_finish_spec(side, color, outer_h, outer_w, seed, grey_positions))

    if total_markers == 0:
        if not fallback_choices:
            return None
        spec_index, free_cells = fallback_choices[int(rng.integers(0, len(fallback_choices)))]
        grey_cell = free_cells[int(rng.integers(0, len(free_cells)))]
        spec = specs[spec_index]
        specs[spec_index] = _finish_spec(
            str(spec["side"]),
            int(spec["color"]),
            int(spec["outer_h"]),
            int(spec["outer_w"]),
            tuple(spec["seed"]),
            [grey_cell],
        )

    return specs


def _layout_specs(
    rng: np.random.Generator, specs: list[dict[str, object]]
) -> tuple[int, int, list[tuple[int, int]]] | None:
    n = len(specs)
    for _ in range(LAYOUT_ATTEMPTS):
        order = [int(v) for v in rng.permutation(n)]
        if n == 2:
            row_count = 1 if rng.random() < 0.5 else 2
        else:
            row_count = 2 if rng.random() < 0.7 else 1

        if row_count == 1:
            rows = [order]
        else:
            cut = int(rng.integers(1, n))
            rows = [order[:cut], order[cut:]]
            if not rows[0] or not rows[1]:
                continue

        row_widths: list[int] = []
        row_heights: list[int] = []
        row_gaps: list[list[int]] = []
        for row in rows:
            gaps = [int(rng.integers(GAP_MIN, GAP_MAX)) for _ in range(max(0, len(row) - 1))]
            width = sum(int(specs[idx]["foot_w"]) for idx in row) + sum(gaps)
            height = max(int(specs[idx]["foot_h"]) for idx in row)
            row_widths.append(width)
            row_heights.append(height)
            row_gaps.append(gaps)

        vertical_gap = int(rng.integers(GAP_MIN, GAP_MAX)) if len(rows) == 2 else 0
        top = int(rng.integers(MARGIN_MIN, MARGIN_MAX))
        bottom = int(rng.integers(MARGIN_MIN, MARGIN_MAX))
        left = int(rng.integers(MARGIN_MIN, MARGIN_MAX))
        right = int(rng.integers(MARGIN_MIN, MARGIN_MAX))

        height = top + bottom + sum(row_heights) + vertical_gap
        width = left + right + max(row_widths)
        if not (GRID_MIN <= height <= GRID_MAX and GRID_MIN <= width <= GRID_MAX):
            continue

        placements: list[tuple[int, int] | None] = [None] * n
        row_y = top
        for row_index, row in enumerate(rows):
            slack_x = width - left - right - row_widths[row_index]
            row_x = left + (0 if slack_x == 0 else int(rng.integers(0, slack_x + 1)))
            cur_x = row_x
            for pos, idx in enumerate(row):
                foot_h = int(specs[idx]["foot_h"])
                y_slack = row_heights[row_index] - foot_h
                foot_y = row_y + (0 if y_slack == 0 else int(rng.integers(0, y_slack + 1)))
                placements[idx] = (foot_y, cur_x)
                if pos < len(row) - 1:
                    cur_x += int(specs[idx]["foot_w"]) + row_gaps[row_index][pos]
            row_y += row_heights[row_index]
            if row_index < len(rows) - 1:
                row_y += vertical_gap

        assert all(p is not None for p in placements)
        return height, width, [(int(r), int(c)) for r, c in placements]

    return None


def _draw_ring(canvas: np.ndarray, r0: int, c0: int, height: int, width: int, color: int) -> None:
    canvas[r0, c0 : c0 + width] = color
    canvas[r0 + height - 1, c0 : c0 + width] = color
    canvas[r0 : r0 + height, c0] = color
    canvas[r0 : r0 + height, c0 + width - 1] = color


def _render(
    height: int, width: int, specs: list[dict[str, object]], placements: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), BG, dtype=int)
    out = inp.copy()

    for spec, (foot_r, foot_c) in zip(specs, placements, strict=False):
        color = int(spec["color"])
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        ring_r = int(foot_r + int(spec["ring_dr"]))
        ring_c = int(foot_c + int(spec["ring_dc"]))
        grey_positions = list(spec["grey_positions"])
        grey_count = int(spec["grey_count"])
        side = str(spec["side"])
        seed_r, seed_c = map(int, spec["seed"])

        _draw_ring(inp, ring_r, ring_c, outer_h, outer_w, color)
        _draw_ring(out, ring_r, ring_c, outer_h, outer_w, color)

        inp[ring_r + 1 + seed_r, ring_c + 1 + seed_c] = color
        out[ring_r + 1 + seed_r, ring_c + 1 + seed_c] = color

        for gr, gc in grey_positions:
            inp[ring_r + 1 + int(gr), ring_c + 1 + int(gc)] = GREY

        if grey_count == 0:
            continue
        if side == "top":
            out[ring_r - grey_count : ring_r, ring_c : ring_c + outer_w] = color
        elif side == "bottom":
            out[ring_r + outer_h : ring_r + outer_h + grey_count, ring_c : ring_c + outer_w] = color
        elif side == "left":
            out[ring_r : ring_r + outer_h, ring_c - grey_count : ring_c] = color
        else:
            out[ring_r : ring_r + outer_h, ring_c + outer_w : ring_c + outer_w + grey_count] = color

    return inp, out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    specs = _sample_specs(rng)
    if specs is None:
        return None

    layout = _layout_specs(rng, specs)
    if layout is None:
        return None
    height, width, placements = layout

    inp, expected = _render(height, width, specs, placements)
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
