import numpy as np


AZURE = 8
GREY = 5
ARROW_COLORS = (2, 3, 4, 6, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRIES = 400
PLACEMENT_RETRIES = 200


def _in_bounds(r: int, c: int, height: int, width: int) -> bool:
    return 0 <= r < height and 0 <= c < width


def _perpendiculars(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = direction
    return ((-dc, dr), (dc, -dr))


def _find_color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if _in_bounds(nr, nc, height, width) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            components.append(np.array(coords, dtype=int))
    return components


def _parse_arrow(coords: np.ndarray) -> dict | None:
    pts = {tuple(map(int, rc)) for rc in coords}
    if len(pts) != 4:
        return None

    matches: list[dict] = []
    for head in pts:
        hr, hc = head
        for direction in DIRS4:
            dr, dc = direction
            shaft = [(hr - 2 * dr, hc - 2 * dc), (hr - dr, hc - dc), (hr, hc)]
            shaft_set = set(shaft)
            if not shaft_set <= pts:
                continue
            extra = pts - shaft_set
            if len(extra) != 1:
                continue
            tail = shaft[0]
            wing = next(iter(extra))
            tail_side = (int(wing[0] - tail[0]), int(wing[1] - tail[1]))
            if tail_side not in _perpendiculars(direction):
                continue
            matches.append(
                {
                    "head": (int(hr), int(hc)),
                    "direction": (int(dr), int(dc)),
                    "tail": (int(tail[0]), int(tail[1])),
                    "tail_side": (int(tail_side[0]), int(tail_side[1])),
                    "coords": coords.copy(),
                }
            )

    if len(matches) != 1:
        return None
    return matches[0]


def _detect_arrows(grid: np.ndarray) -> list[dict]:
    arrows: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) not in (AZURE, GREY)):
        for comp in _find_color_components(grid, color):
            arrow = _parse_arrow(comp)
            assert arrow is not None, f"invalid arrow component for color {color}"
            arrow["color"] = int(color)
            arrows.append(arrow)
    assert arrows, "expected at least one arrow"
    return arrows


def _rank_by_color(arrows: list[dict]) -> dict[int, int]:
    totals: dict[int, int] = {}
    for arrow in arrows:
        color = int(arrow["color"])
        totals[color] = totals.get(color, 0) + int(len(arrow["coords"]))
    values = list(totals.values())
    assert len(values) == len(set(values)), "arrow-color pixel totals must be unique"
    ordered = sorted(totals.items(), key=lambda item: -item[1])
    return {int(color): idx + 1 for idx, (color, _total) in enumerate(ordered)}


def _bundle_offsets(rank: int, tail_side: tuple[int, int]) -> list[tuple[int, int]]:
    offsets = [(0, 0)]
    step = 1
    tr, tc = tail_side
    while len(offsets) < rank:
        offsets.append((-step * tr, -step * tc))
        if len(offsets) < rank:
            offsets.append((step * tr, step * tc))
        step += 1
    return offsets


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape

    arrows = _detect_arrows(g)
    rank_by_color = _rank_by_color(arrows)
    arrows = sorted(
        arrows,
        key=lambda arrow: (
            int(arrow["head"][0]),
            int(arrow["head"][1]),
            int(arrow["direction"][0]),
            int(arrow["direction"][1]),
            int(arrow["tail_side"][0]),
            int(arrow["tail_side"][1]),
        ),
    )

    out = g.copy()
    original_non_bg = g != AZURE
    stats = {"lane_reasons": [], "arrows": []}

    for arrow in arrows:
        head_r, head_c = map(int, arrow["head"])
        dr, dc = map(int, arrow["direction"])
        color = int(arrow["color"])
        rank = int(rank_by_color[color])
        arrow_stats = {
            "color": color,
            "rank": rank,
            "head": (head_r, head_c),
            "lane_paints": [],
            "lane_reasons": [],
        }

        for off_r, off_c in _bundle_offsets(rank, arrow["tail_side"]):
            r = head_r + int(off_r) + dr
            c = head_c + int(off_c) + dc
            painted = 0
            while _in_bounds(r, c, height, width) and int(out[r, c]) == AZURE:
                out[r, c] = color
                painted += 1
                r += dr
                c += dc

            if not _in_bounds(r, c, height, width):
                reason = "edge"
            elif bool(original_non_bg[r, c]):
                reason = "wall" if int(g[r, c]) == GREY else "object"
            else:
                reason = "ray"

            arrow_stats["lane_paints"].append(int(painted))
            arrow_stats["lane_reasons"].append(reason)
            stats["lane_reasons"].append(reason)

        stats["arrows"].append(arrow_stats)

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rank arrow colors by total pixel count, then let each arrow emit that many
    adjacent rays in its pointing direction until a non-background cell or the
    grid edge blocks the ray.
    """
    out, _stats = _simulate(grid)
    return out


def _make_arrow(
    head: tuple[int, int],
    direction: tuple[int, int],
    tail_side: tuple[int, int],
    color: int,
    rank: int,
) -> dict:
    hr, hc = head
    dr, dc = direction
    tail = (int(hr - 2 * dr), int(hc - 2 * dc))
    cells = [
        (int(hr), int(hc)),
        (int(hr - dr), int(hc - dc)),
        tail,
        (int(tail[0] + tail_side[0]), int(tail[1] + tail_side[1])),
    ]
    return {
        "color": int(color),
        "rank": int(rank),
        "head": (int(hr), int(hc)),
        "direction": (int(dr), int(dc)),
        "tail_side": (int(tail_side[0]), int(tail_side[1])),
        "cells": cells,
    }


def _distance_to_boundary(start: tuple[int, int], direction: tuple[int, int], height: int, width: int) -> int:
    r, c = start
    dr, dc = direction
    dist = 0
    while _in_bounds(r, c, height, width):
        dist += 1
        r += dr
        c += dc
    return int(dist)


def _cells_clear(occupied: np.ndarray, cells: list[tuple[int, int]], margin: int = 1) -> bool:
    height, width = occupied.shape
    for r, c in cells:
        if not _in_bounds(r, c, height, width):
            return False
        r0 = max(0, int(r) - margin)
        r1 = min(height - 1, int(r) + margin)
        c0 = max(0, int(c) - margin)
        c1 = min(width - 1, int(c) + margin)
        if bool(np.any(occupied[r0 : r1 + 1, c0 : c1 + 1])):
            return False
    return True


def _stamp_cells(grid: np.ndarray, occupied: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)
        occupied[int(r), int(c)] = True


def _stamp_rect(grid: np.ndarray, occupied: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _wall_cells(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return [(int(r), int(c)) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _arrow_bundle_fits(arrow: dict, height: int, width: int) -> bool:
    head_r, head_c = arrow["head"]
    for off_r, off_c in _bundle_offsets(int(arrow["rank"]), arrow["tail_side"]):
        if not _in_bounds(int(head_r + off_r), int(head_c + off_c), height, width):
            return False
        start_r = int(head_r + off_r + arrow["direction"][0])
        start_c = int(head_c + off_c + arrow["direction"][1])
        if not _in_bounds(start_r, start_c, height, width):
            return False
    return True


def _place_random_arrow(
    rng: np.random.Generator,
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
    rank: int,
    min_corridor: int,
) -> dict | None:
    height, width = grid.shape
    for _ in range(PLACEMENT_RETRIES):
        direction = DIRS4[int(rng.integers(len(DIRS4)))]
        tail_side = _perpendiculars(direction)[int(rng.integers(2))]
        head = (int(rng.integers(height)), int(rng.integers(width)))
        arrow = _make_arrow(head, direction, tail_side, color, rank)
        if not _cells_clear(occupied, arrow["cells"], margin=1):
            continue
        if not _arrow_bundle_fits(arrow, height, width):
            continue
        center_start = (int(head[0] + direction[0]), int(head[1] + direction[1]))
        if _distance_to_boundary(center_start, direction, height, width) < min_corridor:
            continue
        _stamp_cells(grid, occupied, arrow["cells"], color)
        return arrow
    return None


def _place_special_shooter(
    rng: np.random.Generator,
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
    rank: int,
    mode: str,
) -> dict | None:
    min_corridor = 7 if mode in {"wall", "object"} else 5
    return _place_random_arrow(rng, grid, occupied, color, rank, min_corridor=min_corridor)


def _place_blocking_wall(
    rng: np.random.Generator,
    grid: np.ndarray,
    occupied: np.ndarray,
    shooter: dict,
) -> bool:
    height, width = grid.shape
    head_r, head_c = shooter["head"]
    dr, dc = shooter["direction"]
    offsets = _bundle_offsets(int(shooter["rank"]), shooter["tail_side"])
    lane = offsets[int(rng.integers(len(offsets)))]
    start = (int(head_r + lane[0] + dr), int(head_c + lane[1] + dc))
    max_dist = _distance_to_boundary(start, shooter["direction"], height, width)
    if max_dist < 4:
        return False

    for _ in range(PLACEMENT_RETRIES):
        dist = int(rng.integers(3, min(max_dist, 8)))
        hit = (int(head_r + lane[0] + dr * dist), int(head_c + lane[1] + dc * dist))
        thick = int(rng.integers(1, 3))
        span = int(rng.integers(1, 4))

        if dr == 0:
            c0 = int(hit[1]) if dc > 0 else int(hit[1] - thick + 1)
            c1 = c0 + thick - 1
            r0 = int(hit[0] - rng.integers(span))
            r1 = r0 + span - 1
        else:
            r0 = int(hit[0]) if dr > 0 else int(hit[0] - thick + 1)
            r1 = r0 + thick - 1
            c0 = int(hit[1] - rng.integers(span))
            c1 = c0 + span - 1

        if not (0 <= r0 <= r1 < height and 0 <= c0 <= c1 < width):
            continue
        rect = (r0, c0, r1, c1)
        if not _cells_clear(occupied, _wall_cells(rect), margin=1):
            continue
        _stamp_rect(grid, occupied, rect, GREY)
        return True
    return False


def _place_blocker_arrow(
    rng: np.random.Generator,
    grid: np.ndarray,
    occupied: np.ndarray,
    shooter: dict,
    color: int,
    rank: int,
) -> dict | None:
    height, width = grid.shape
    head_r, head_c = shooter["head"]
    dr, dc = shooter["direction"]
    lane = (0, 0)
    start = (int(head_r + dr), int(head_c + dc))
    max_dist = _distance_to_boundary(start, shooter["direction"], height, width)
    if max_dist < 4:
        return None

    blocker_dirs = ((0, 1), (0, -1)) if dr != 0 else ((1, 0), (-1, 0))
    for _ in range(PLACEMENT_RETRIES):
        dist = int(rng.integers(3, min(max_dist, 8)))
        hit = (int(head_r + lane[0] + dr * dist), int(head_c + lane[1] + dc * dist))
        direction = blocker_dirs[int(rng.integers(2))]
        tail_side = _perpendiculars(direction)[int(rng.integers(2))]
        blocker_head = (int(hit[0] + direction[0]), int(hit[1] + direction[1]))
        arrow = _make_arrow(blocker_head, direction, tail_side, color, rank)
        if not _cells_clear(occupied, arrow["cells"], margin=1):
            continue
        if not _arrow_bundle_fits(arrow, height, width):
            continue
        _stamp_cells(grid, occupied, arrow["cells"], color)
        return arrow
    return None


def _place_random_wall(rng: np.random.Generator, grid: np.ndarray, occupied: np.ndarray) -> bool:
    height, width = grid.shape
    for _ in range(PLACEMENT_RETRIES):
        wall_h = int(rng.integers(1, 5))
        wall_w = int(rng.integers(1, 5))
        r0 = int(rng.integers(0, height - wall_h + 1))
        c0 = int(rng.integers(0, width - wall_w + 1))
        rect = (r0, c0, r0 + wall_h - 1, c0 + wall_w - 1)
        if not _cells_clear(occupied, _wall_cells(rect), margin=1):
            continue
        _stamp_rect(grid, occupied, rect, GREY)
        return True
    return False


def _validate_generated(grid: np.ndarray, mode: str) -> bool:
    try:
        out, stats = _simulate(grid)
    except AssertionError:
        return False

    reasons = set(stats["lane_reasons"])
    if "ray" in reasons:
        return False
    if "edge" not in reasons:
        return False
    if mode == "wall" and "wall" not in reasons:
        return False
    if mode == "object" and "object" not in reasons:
        return False

    if int(np.count_nonzero(out != grid)) < 8:
        return False
    if not any(arrow["rank"] > 1 and sum(p > 0 for p in arrow["lane_paints"]) >= 2 for arrow in stats["arrows"]):
        return False

    grey_components = _find_color_components(grid, GREY)
    if not grey_components:
        return False
    for comp in grey_components:
        rows = comp[:, 0]
        cols = comp[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        if int(comp.shape[0]) != (r1 - r0 + 1) * (c1 - c0 + 1):
            return False

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    num_colors = int(rng.choice([2, 3, 4]))
    shift = 0 if num_colors == 4 else int(rng.integers(0, 2))
    counts = np.arange(1 + shift, num_colors + 1 + shift)
    colors = [int(c) for c in rng.choice(ARROW_COLORS, size=num_colors, replace=False)]
    rng.shuffle(counts)
    rank_by_color = {
        int(color): 1 + int(np.sum(counts > count))
        for color, count in zip(colors, counts, strict=True)
    }

    total_arrows = int(np.sum(counts))
    min_side = 18 if total_arrows <= 10 else 22
    height = int(rng.integers(min_side, min_side + 9))
    width = int(rng.integers(min_side, min_side + 9))
    grid = np.full((height, width), AZURE, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    specs: list[dict] = []
    for color, count in zip(colors, counts, strict=True):
        for _ in range(int(count)):
            specs.append({"color": int(color), "rank": int(rank_by_color[int(color)])})
    rng.shuffle(specs)

    widest_color = max(rank_by_color, key=lambda color: rank_by_color[color])
    special_idx = next(i for i, spec in enumerate(specs) if int(spec["color"]) == int(widest_color))
    special_spec = specs.pop(special_idx)

    mode = str(rng.choice(["edge", "wall", "object"]))
    blocker_spec = None
    if mode == "object":
        blocker_idx = next((i for i, spec in enumerate(specs) if int(spec["color"]) != int(special_spec["color"])), None)
        if blocker_idx is None:
            blocker_idx = 0
        blocker_spec = specs.pop(int(blocker_idx))

    shooter = _place_special_shooter(
        rng,
        grid,
        occupied,
        color=int(special_spec["color"]),
        rank=int(special_spec["rank"]),
        mode=mode,
    )
    if shooter is None:
        return None

    if mode == "wall" and not _place_blocking_wall(rng, grid, occupied, shooter):
        return None

    if mode == "object":
        assert blocker_spec is not None
        blocker = _place_blocker_arrow(
            rng,
            grid,
            occupied,
            shooter=shooter,
            color=int(blocker_spec["color"]),
            rank=int(blocker_spec["rank"]),
        )
        if blocker is None:
            return None

    for spec in specs:
        arrow = _place_random_arrow(
            rng,
            grid,
            occupied,
            color=int(spec["color"]),
            rank=int(spec["rank"]),
            min_corridor=3,
        )
        if arrow is None:
            return None

    wall_target = int(rng.integers(1, 5))
    existing_walls = len(_find_color_components(grid, GREY))
    while existing_walls < wall_target:
        if not _place_random_wall(rng, grid, occupied):
            break
        existing_walls += 1

    if not _validate_generated(grid, mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
