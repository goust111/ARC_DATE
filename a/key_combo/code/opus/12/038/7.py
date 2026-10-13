from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


def _compress_runs(values: list[int]) -> list[int]:
    out: list[int] = []
    for value in values:
        if not out or out[-1] != value:
            out.append(int(value))
    return out


def _row_majorities(grid: np.ndarray) -> list[int]:
    g = np.asarray(grid, dtype=int)
    color_rows: dict[int, float] = {}
    for color in np.unique(g):
        rows = np.argwhere(g == int(color))[:, 0]
        color_rows[int(color)] = float(rows.mean())

    majors: list[int] = []
    for r in range(g.shape[0]):
        values, counts = np.unique(g[r, :], return_counts=True)
        max_count = int(counts.max())
        winners = [int(v) for v, c in zip(values, counts) if int(c) == max_count]
        if len(winners) == 1:
            majors.append(winners[0])
            continue
        distances = [(abs(color_rows[color] - r), color) for color in winners]
        best = min(dist for dist, _ in distances)
        closest = [color for dist, color in distances if dist == best]
        assert len(closest) == 1, "row-majority tie must be resolvable"
        majors.append(int(closest[0]))
    return majors


def _anchor_cells(anchor: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = anchor
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _find_anchor(mask: np.ndarray) -> tuple[int, int, int, int]:
    h, w = mask.shape
    anchors: list[tuple[int, int, int, int]] = []
    for r in range(h - 2):
        for c in range(w - 2):
            if not bool(mask[r : r + 3, c : c + 3].all()):
                continue
            r0 = max(0, r - 1)
            c0 = max(0, c - 1)
            r1 = min(h, r + 4)
            c1 = min(w, c + 4)
            ring = mask[r0:r1, c0:c1].copy()
            ring[r - r0 : r - r0 + 3, c - c0 : c - c0 + 3] = False
            if bool(ring.any()):
                continue
            anchors.append((r, c, r + 2, c + 2))
    assert len(anchors) == 1, f"expected one isolated 3x3 anchor, got {len(anchors)}"
    return anchors[0]


def _ray_terminal(
    coord: tuple[int, int],
    anchor: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    r, c = coord
    r0, c0, r1, c1 = anchor
    dr = 1 if r < r0 else -1 if r > r1 else 0
    dc = 1 if c < c0 else -1 if c > c1 else 0
    assert (dr, dc) != (0, 0), "ray pixel cannot be inside the anchor"

    cr, cc = r, c
    for _ in range(max(shape) * 2):
        nr, nc = cr + dr, cc + dc
        if r0 <= nr <= r1 and c0 <= nc <= c1:
            return (cr, cc), (dr, dc)
        assert 0 <= nr < shape[0] and 0 <= nc < shape[1], "ray must reach the anchor"
        cr, cc = nr, nc
    raise AssertionError("ray did not reach anchor")


def _group_rays(
    coords: list[tuple[int, int]],
    anchor: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> dict[tuple[tuple[int, int], tuple[int, int]], list[int]]:
    rays: dict[tuple[tuple[int, int], tuple[int, int]], list[int]] = defaultdict(list)
    for coord in coords:
        terminal, step = _ray_terminal(coord, anchor, shape)
        dist = max(abs(coord[0] - terminal[0]), abs(coord[1] - terminal[1]))
        rays[(terminal, step)].append(int(dist))
    for distances in rays.values():
        assert len(distances) == len(set(distances)), "duplicate ray distance on one terminal"
        distances.sort()
    return rays


def _scene_info(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    row_majorities = _row_majorities(g)
    band_colors = _compress_runs(row_majorities)
    assert len(band_colors) >= 3, "expected at least three horizontal bands"
    first_color = int(band_colors[0])
    last_color = int(band_colors[-1])
    assert first_color != last_color, "first and last bands must differ"

    anchors: dict[int, tuple[int, int, int, int]] = {}
    foreground_masks: dict[int, np.ndarray] = {}
    rays: dict[int, dict[tuple[tuple[int, int], tuple[int, int]], list[int]]] = {}
    for color in (first_color, last_color):
        mask = (g == color) & (np.array(row_majorities, dtype=int)[:, None] != color)
        anchor = _find_anchor(mask)
        anchor_mask = np.zeros_like(mask, dtype=bool)
        r0, c0, r1, c1 = anchor
        anchor_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        loose = [
            (int(r), int(c))
            for r, c in np.argwhere(mask & ~anchor_mask)
        ]
        anchors[color] = anchor
        foreground_masks[color] = mask
        rays[color] = _group_rays(loose, anchor, g.shape)

    return {
        "row_majorities": row_majorities,
        "band_colors": band_colors,
        "live_color": first_color,
        "dead_color": last_color,
        "anchors": anchors,
        "foreground_masks": foreground_masks,
        "rays": rays,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Use the row-band order to decide which anchor survives and which matching object is erased.
    Input structure:
    - The whole grid is covered by 3-5 horizontal color bands with no black background; each row still has a clear majority color.
    - Exactly two colors also form isolated solid 3x3 anchor blocks placed on rows whose majority color is different.
    - Each such anchor color has extra same-colored pixels scattered on horizontal, vertical, or 45-degree rays aimed at its anchor.
    Rule:
    - Read the top-to-bottom row-majority band sequence and take its first color as the surviving anchor color and its last color as the erased anchor color.
    - For the surviving color, keep the 3x3 anchor fixed, erase the loose ray pixels, and repack each ray inward into consecutive cells adjacent to the anchor along that same ray.
    - For the erased color, remove both the 3x3 anchor and all of its ray pixels, restoring each cleared cell to the majority color of its row.
    Key insight:
    - The background bands are an instruction list: the first band color says which anchored ray pattern must shrink inward, while the last band color says which same-colored anchored object must disappear.
    """
    g = np.asarray(grid, dtype=int)
    info = _scene_info(g)
    row_majorities = info["row_majorities"]
    live_color = int(info["live_color"])
    dead_color = int(info["dead_color"])
    out = g.copy()

    for color in (live_color, dead_color):
        for r, c in np.argwhere(info["foreground_masks"][color]):
            out[int(r), int(c)] = int(row_majorities[int(r)])

    live_anchor = info["anchors"][live_color]
    for r, c in _anchor_cells(live_anchor):
        out[r, c] = live_color

    for (terminal, step), distances in info["rays"][live_color].items():
        tr, tc = terminal
        dr, dc = step
        for i, _dist in enumerate(distances):
            rr = tr - i * dr
            cc = tc - i * dc
            out[rr, cc] = live_color

    return out


GRID_H_RANGE = (15, 22)
GRID_W_RANGE = (13, 19)
MIN_BANDS = 3
MAX_BANDS = 5
MIN_BAND_THICKNESS = 3
PALETTE = tuple(range(1, 10))
ANCHOR_COUNT = 2
ANCHOR_SIZE = 3
ANCHOR_MARGIN = 1
MIN_TERMINALS = 3
MAX_TERMINALS = 6
MAX_PIXELS_PER_RAY = 3
MAX_ATTEMPTS = 400


def _random_partition(total: int, parts: int, min_each: int) -> list[int]:
    cuts = sorted(random.sample(range(min_each, total - min_each * (parts - 1) + 1), 1)) if parts == 2 else None
    if parts == 1:
        return [total]
    remaining = total - min_each * parts
    slots = [0] * parts
    for _ in range(remaining):
        slots[random.randrange(parts)] += 1
    values = [min_each + add for add in slots]
    random.shuffle(values)
    return values


def _sample_thicknesses(height: int, n_bands: int) -> list[int]:
    assert height >= n_bands * MIN_BAND_THICKNESS
    while True:
        thicknesses = _random_partition(height, n_bands, MIN_BAND_THICKNESS)
        middles = thicknesses[1:-1]
        if middles and max(middles) >= ANCHOR_SIZE:
            return thicknesses


def _band_rows(thicknesses: list[int], colors: list[int]) -> tuple[list[int], list[tuple[int, int, int]]]:
    rows: list[int] = []
    spans: list[tuple[int, int, int]] = []
    start = 0
    for thickness, color in zip(thicknesses, colors):
        rows.extend([int(color)] * int(thickness))
        spans.append((start, start + thickness - 1, int(color)))
        start += thickness
    return rows, spans


def _rect_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    row_gap = max(0, max(a[0], b[0]) - min(a[2], b[2]) - 1)
    col_gap = max(0, max(a[1], b[1]) - min(a[3], b[3]) - 1)
    if row_gap == 0 and col_gap == 0:
        return 0
    return max(row_gap, col_gap)


def _place_anchors(
    row_colors: list[int],
    spans: list[tuple[int, int, int]],
    width: int,
) -> list[tuple[int, int, int, int]]:
    eligible = [(r0, r1, color) for r0, r1, color in spans[1:-1] if r1 - r0 + 1 >= ANCHOR_SIZE]
    assert eligible, "need at least one middle band wide enough for an anchor"
    for _ in range(200):
        anchors: list[tuple[int, int, int, int]] = []
        for _k in range(ANCHOR_COUNT):
            band_r0, band_r1, _bg_color = random.choice(eligible)
            top = random.randint(band_r0, band_r1 - ANCHOR_SIZE + 1)
            left = random.randint(ANCHOR_MARGIN, width - ANCHOR_SIZE - ANCHOR_MARGIN)
            anchors.append((top, left, top + ANCHOR_SIZE - 1, left + ANCHOR_SIZE - 1))
        if _rect_gap(anchors[0], anchors[1]) >= 3:
            return anchors
    raise RuntimeError("failed to place separated anchors")


def _terminals(anchor: tuple[int, int, int, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    r0, c0, r1, c1 = anchor
    terms: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for c in range(c0, c1 + 1):
        terms.append(((r0 - 1, c), (-1, 0)))
        terms.append(((r1 + 1, c), (1, 0)))
    for r in range(r0, r1 + 1):
        terms.append(((r, c0 - 1), (0, -1)))
        terms.append(((r, c1 + 1), (0, 1)))
    terms.extend(
        [
            ((r0 - 1, c0 - 1), (-1, -1)),
            ((r0 - 1, c1 + 1), (-1, 1)),
            ((r1 + 1, c0 - 1), (1, -1)),
            ((r1 + 1, c1 + 1), (1, 1)),
        ]
    )
    return terms


def _available_distances(
    terminal: tuple[int, int],
    outward: tuple[int, int],
    row_colors: list[int],
    own_color: int,
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> list[int]:
    h, w = shape
    tr, tc = terminal
    dr, dc = outward
    distances: list[int] = []
    dist = 1
    while True:
        rr = tr + dist * dr
        cc = tc + dist * dc
        if not (0 <= rr < h and 0 <= cc < w):
            break
        if row_colors[rr] == own_color:
            break
        if (rr, cc) not in occupied:
            distances.append(dist)
        dist += 1
    return distances


def _sample_rays(
    anchor: tuple[int, int, int, int],
    own_color: int,
    row_colors: list[int],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    force_diagonal: bool,
) -> list[dict]:
    candidates = []
    for terminal, outward in _terminals(anchor):
        tr, tc = terminal
        if not (0 <= tr < shape[0] and 0 <= tc < shape[1]):
            continue
        distances = _available_distances(terminal, outward, row_colors, own_color, shape, occupied)
        if distances:
            candidates.append(
                {
                    "terminal": terminal,
                    "outward": outward,
                    "distances": distances,
                    "diagonal": abs(outward[0]) == 1 and abs(outward[1]) == 1,
                }
            )
    if len(candidates) < MIN_TERMINALS:
        raise RuntimeError("not enough usable rays")

    max_terms = min(MAX_TERMINALS, len(candidates))
    used_count = random.randint(MIN_TERMINALS, max_terms)
    chosen: list[dict] = []
    pool = candidates[:]
    if force_diagonal:
        diagonals = [cand for cand in pool if cand["diagonal"]]
        if diagonals:
            first = random.choice(diagonals)
            chosen.append(first)
            pool.remove(first)
    remaining = used_count - len(chosen)
    chosen.extend(random.sample(pool, remaining))

    rays: list[dict] = []
    for cand in chosen:
        usable = cand["distances"]
        max_pixels = min(MAX_PIXELS_PER_RAY, len(usable))
        n_pixels = 1
        if max_pixels >= 2 and random.random() < 0.55:
            n_pixels = random.randint(2, max_pixels)
        chosen_distances = sorted(random.sample(usable, n_pixels))
        ray = {
            "terminal": cand["terminal"],
            "outward": cand["outward"],
            "distances": chosen_distances,
        }
        rays.append(ray)
        tr, tc = cand["terminal"]
        dr, dc = cand["outward"]
        for dist in chosen_distances:
            occupied.add((tr + dist * dr, tc + dist * dc))
    return rays


def _paint_input(
    base: np.ndarray,
    colors: list[int],
    anchors: list[tuple[int, int, int, int]],
    ray_specs: list[list[dict]],
) -> np.ndarray:
    grid = base.copy()
    for color, anchor in zip(colors, anchors):
        r0, c0, r1, c1 = anchor
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    for color, rays in zip(colors, ray_specs):
        for ray in rays:
            tr, tc = ray["terminal"]
            dr, dc = ray["outward"]
            for dist in ray["distances"]:
                grid[tr + dist * dr, tc + dist * dc] = int(color)
    return grid


def _paint_expected(
    base: np.ndarray,
    live_color: int,
    live_anchor: tuple[int, int, int, int],
    live_rays: list[dict],
) -> np.ndarray:
    out = base.copy()
    r0, c0, r1, c1 = live_anchor
    out[r0 : r1 + 1, c0 : c1 + 1] = int(live_color)
    for ray in live_rays:
        tr, tc = ray["terminal"]
        dr, dc = ray["outward"]
        for i in range(len(ray["distances"])):
            out[tr + i * dr, tc + i * dc] = int(live_color)
    return out


def _raw_row_majorities_match(grid: np.ndarray, expected_rows: list[int]) -> bool:
    for r, expected in enumerate(expected_rows):
        counts = Counter(map(int, grid[r, :]))
        max_count = max(counts.values())
        winners = [color for color, count in counts.items() if count == max_count]
        if len(winners) != 1 or int(winners[0]) != int(expected):
            return False
    return True


def _attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    n_bands = random.randint(MIN_BANDS, min(MAX_BANDS, len(PALETTE)))
    band_colors = random.sample(list(PALETTE), n_bands)
    thicknesses = _sample_thicknesses(height, n_bands)
    row_colors, spans = _band_rows(thicknesses, band_colors)

    base = np.repeat(np.array(row_colors, dtype=int)[:, None], width, axis=1)
    anchor_boxes = _place_anchors(row_colors, spans, width)
    anchor_colors = [band_colors[0], band_colors[-1]]
    occupied = set()
    for anchor in anchor_boxes:
        occupied.update(_anchor_cells(anchor))

    force_live_diagonal = random.random() < 0.6
    force_dead_diagonal = random.random() < 0.6
    try:
        live_rays = _sample_rays(
            anchor_boxes[0],
            anchor_colors[0],
            row_colors,
            base.shape,
            occupied,
            force_live_diagonal,
        )
        dead_rays = _sample_rays(
            anchor_boxes[1],
            anchor_colors[1],
            row_colors,
            base.shape,
            occupied,
            force_dead_diagonal,
        )
    except RuntimeError:
        return None

    input_grid = _paint_input(base, anchor_colors, anchor_boxes, [live_rays, dead_rays])
    expected = _paint_expected(base, anchor_colors[0], anchor_boxes[0], live_rays)

    if not _raw_row_majorities_match(input_grid, row_colors):
        return None
    if np.array_equal(input_grid, expected):
        return None
    if not np.array_equal(main(input_grid), expected):
        return None

    info = _scene_info(input_grid)
    if info["band_colors"] != band_colors:
        return None
    if int((expected == anchor_colors[1]).sum()) >= int((input_grid == anchor_colors[1]).sum()):
        return None
    if sum(len(ray["distances"]) for ray in live_rays) == 0 or sum(len(ray["distances"]) for ray in dead_rays) == 0:
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
