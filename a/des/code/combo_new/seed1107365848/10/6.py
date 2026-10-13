import random
from typing import Optional

import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ARROW_SHAPES = {
    "up": {
        "head": (0, 1),
        "tail": (2, 1),
        "grey": {(0, 1), (1, 1), (2, 0), (2, 2)},
    },
    "down": {
        "head": (2, 1),
        "tail": (0, 1),
        "grey": {(0, 0), (0, 2), (1, 1), (2, 1)},
    },
    "left": {
        "head": (1, 0),
        "tail": (1, 2),
        "grey": {(0, 2), (1, 0), (1, 1), (2, 2)},
    },
    "right": {
        "head": (1, 2),
        "tail": (1, 0),
        "grey": {(0, 0), (1, 1), (1, 2), (2, 0)},
    },
}


BACKGROUND = 8
GREY = 5
TAIL_PALETTE = (1, 2, 3, 4, 6, 7, 9)
GRID_MIN = 16
GRID_MAX = 29
MAX_ATTEMPTS = 400
CHAIN_LENGTH_MIN = 1
CHAIN_LENGTH_MAX = 3
TOTAL_ARROWS_MIN = 2
TOTAL_ARROWS_MAX = 6
STEP_MIN = 5


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _match_arrow(patch: np.ndarray, bg: int, grey: int) -> Optional[dict[str, object]]:
    if patch.shape != (3, 3):
        return None

    matches: list[dict[str, object]] = []
    for name, spec in ARROW_SHAPES.items():
        tail_r, tail_c = spec["tail"]
        ok = True
        tail_color = None
        for r in range(3):
            for c in range(3):
                value = int(patch[r, c])
                pos = (r, c)
                if pos in spec["grey"]:
                    if value != grey:
                        ok = False
                        break
                elif pos == (tail_r, tail_c):
                    if value in (bg, grey):
                        ok = False
                        break
                    tail_color = value
                elif value != bg:
                    ok = False
                    break
            if not ok:
                break
        if ok and tail_color is not None:
            matches.append({"direction": name, "tail_color": int(tail_color)})

    if len(matches) != 1:
        return None
    return matches[0]


def _detect_arrows(grid: np.ndarray) -> tuple[int, int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _components(g != bg)
    assert comps, "expected at least one arrow"

    non_bg_colors = sorted(int(x) for x in np.unique(g) if int(x) != bg)
    candidates: list[tuple[int, list[dict[str, object]]]] = []
    for grey in non_bg_colors:
        arrows: list[dict[str, object]] = []
        valid = True
        for comp in comps:
            if len(comp) != 5:
                valid = False
                break
            r0, c0, r1, c1 = _bbox(comp)
            if (r1 - r0, c1 - c0) != (2, 2):
                valid = False
                break
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            info = _match_arrow(patch, bg=bg, grey=grey)
            if info is None:
                valid = False
                break

            direction = str(info["direction"])
            head_r, head_c = ARROW_SHAPES[direction]["head"]
            arrow = {
                "direction": direction,
                "dir": DIRS[direction],
                "tail_color": int(info["tail_color"]),
                "bbox": (r0, c0, r1, c1),
                "bbox_cells": {
                    (rr, cc)
                    for rr in range(r0, r1 + 1)
                    for cc in range(c0, c1 + 1)
                },
                "head": (r0 + head_r, c0 + head_c),
            }
            arrows.append(arrow)
        if valid:
            candidates.append((grey, arrows))

    assert len(candidates) == 1, "arrow framework color must be uniquely identifiable"
    grey, arrows = candidates[0]
    arrows.sort(key=lambda arrow: tuple(arrow["bbox"]))  # deterministic processing
    return bg, grey, arrows


def _trace_segments(
    arrows: list[dict[str, object]],
    shape: tuple[int, int],
) -> tuple[list[Optional[int]], list[list[tuple[int, int]]]]:
    bbox_owner: dict[tuple[int, int], int] = {}
    for idx, arrow in enumerate(arrows):
        for cell in arrow["bbox_cells"]:
            assert cell not in bbox_owner, "arrow boxes must not overlap"
            bbox_owner[cell] = idx

    targets: list[Optional[int]] = []
    segments: list[list[tuple[int, int]]] = []
    for idx, arrow in enumerate(arrows):
        dr, dc = arrow["dir"]
        r, c = arrow["head"]
        segment: list[tuple[int, int]] = []
        while True:
            nr = int(r) + int(dr)
            nc = int(c) + int(dc)
            if not _in_bounds(shape, nr, nc):
                targets.append(None)
                segments.append(segment)
                break

            owner = bbox_owner.get((nr, nc))
            if owner is not None:
                assert owner != idx, "ray cannot re-enter its own arrow box"
                targets.append(owner)
                segments.append(segment)
                break

            segment.append((nr, nc))
            r, c = nr, nc

    return targets, segments


def _assert_acyclic(targets: list[Optional[int]]) -> None:
    state = [0] * len(targets)
    for start in range(len(targets)):
        cur = start
        trail: list[int] = []
        while cur is not None and state[cur] == 0:
            state[cur] = 1
            trail.append(cur)
            cur = targets[cur]
        assert cur is None or state[cur] == 2, "ray relay graph must reach the boundary"
        for node in trail:
            state[node] = 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each arrow's tail color along its pointing direction, relaying at later arrows without repainting the arrow sprites themselves.

    Input structure:
    - The background is the dominant color.
    - Every foreground object is a 3x3 arrow sprite whose frame is mostly grey.
    - Each arrow has one narrow head pixel on its pointing side and one colored tail
      pixel on the opposite side, flanked by two grey wing pixels.

    Rule:
    - Keep every arrow pixel unchanged.
    - For each arrow, start from the cell immediately beyond its head and paint a
      straight ray in the arrow's direction using that arrow's tail color.
    - Stop the ray at the grid boundary, or just before it would enter another
      arrow's 3x3 box; that encountered arrow then serves as the next relay in the
      same chain with its own direction and tail color.

    Key insight:
    - The head side is the narrow side of the 3x3 sprite, so arrow orientation comes
      from shape asymmetry rather than from the tail color; once the arrows are read,
      the output is the union of all box-clipped straight relay segments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg, _grey, arrows = _detect_arrows(g)

    targets, segments = _trace_segments(arrows, g.shape)
    _assert_acyclic(targets)

    out = g.copy()
    claims: dict[tuple[int, int], set[int]] = {}
    for arrow, segment in zip(arrows, segments):
        color = int(arrow["tail_color"])
        for cell in segment:
            claims.setdefault(cell, set()).add(color)

    for cell, colors in claims.items():
        assert int(g[cell]) == bg, "ray cells must be background in the input"
        assert len(colors) == 1, "independent ray segments must not conflict"
        out[cell] = next(iter(colors))

    return out


def _bbox_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(rr, cc) for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)}


def _moat_cells(center: tuple[int, int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    r, c = center
    return {
        (rr, cc)
        for rr in range(max(0, r - 2), min(h, r + 3))
        for cc in range(max(0, c - 2), min(w, c + 3))
    }


def _center_valid(
    center: tuple[int, int],
    shape: tuple[int, int],
    used_bbox: set[tuple[int, int]],
    used_moat: set[tuple[int, int]],
    used_rays: set[tuple[int, int]],
) -> bool:
    h, w = shape
    r, c = center
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    bbox_cells = _bbox_cells(center)
    if bbox_cells & used_bbox:
        return False
    if bbox_cells & used_rays:
        return False
    if _moat_cells(center, shape) & used_moat:
        return False
    return True


def _ray_to_target(
    center: tuple[int, int],
    direction: tuple[int, int],
    target_center: tuple[int, int],
) -> list[tuple[int, int]]:
    dr, dc = direction
    target_box = _bbox_cells(target_center)
    r = center[0] + dr
    c = center[1] + dc
    cells: list[tuple[int, int]] = []
    while True:
        nr = r + dr
        nc = c + dc
        if (nr, nc) in target_box:
            return cells
        cells.append((nr, nc))
        r, c = nr, nc


def _ray_to_boundary(
    center: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    dr, dc = direction
    r = center[0] + dr
    c = center[1] + dc
    cells: list[tuple[int, int]] = []
    while True:
        nr = r + dr
        nc = c + dc
        if not _in_bounds(shape, nr, nc):
            return cells
        cells.append((nr, nc))
        r, c = nr, nc


def _max_step(center: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> int:
    h, w = shape
    r, c = center
    dr, dc = direction
    if dr == -1:
        return r - 1
    if dr == 1:
        return h - 2 - r
    if dc == -1:
        return c - 1
    return w - 2 - c


def _opposite(direction: tuple[int, int]) -> tuple[int, int]:
    return (-direction[0], -direction[1])


def _dir_name(direction: tuple[int, int]) -> str:
    for name, vec in DIRS.items():
        if vec == direction:
            return name
    raise AssertionError("unknown direction")


def _extend_chain(
    shape: tuple[int, int],
    length: int,
    centers: list[tuple[int, int]],
    records: list[dict[str, object]],
    used_bbox: set[tuple[int, int]],
    used_moat: set[tuple[int, int]],
    used_rays: set[tuple[int, int]],
    prev_step: Optional[tuple[int, int]],
) -> Optional[tuple[list[dict[str, object]], set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]]:
    current = centers[-1]
    directions = list(DIRS.values())
    random.shuffle(directions)

    if len(centers) == length:
        for direction in directions:
            if prev_step is not None and direction == _opposite(prev_step):
                continue
            ray = _ray_to_boundary(current, direction, shape)
            if not ray:
                continue
            ray_set = set(ray)
            if ray_set & used_bbox:
                continue
            if ray_set & used_rays:
                continue
            record = {"center": current, "direction": _dir_name(direction)}
            return records + [record], used_bbox, used_moat, used_rays | ray_set
        return None

    for direction in directions:
        if prev_step is not None and direction == _opposite(prev_step):
            continue
        max_step = _max_step(current, direction, shape)
        distances = list(range(STEP_MIN, max_step + 1))
        random.shuffle(distances)
        for step in distances:
            next_center = (
                current[0] + direction[0] * step,
                current[1] + direction[1] * step,
            )
            if not _center_valid(next_center, shape, used_bbox, used_moat, used_rays):
                continue
            ray = _ray_to_target(current, direction, next_center)
            if not ray:
                continue
            ray_set = set(ray)
            if ray_set & used_bbox:
                continue
            if ray_set & used_rays:
                continue

            next_bbox = _bbox_cells(next_center)
            result = _extend_chain(
                shape=shape,
                length=length,
                centers=centers + [next_center],
                records=records + [{"center": current, "direction": _dir_name(direction)}],
                used_bbox=used_bbox | next_bbox,
                used_moat=used_moat | _moat_cells(next_center, shape),
                used_rays=used_rays | ray_set,
                prev_step=direction,
            )
            if result is not None:
                return result

    return None


def _place_chain(
    shape: tuple[int, int],
    length: int,
    used_bbox: set[tuple[int, int]],
    used_moat: set[tuple[int, int]],
    used_rays: set[tuple[int, int]],
) -> Optional[tuple[list[dict[str, object]], set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]]:
    h, w = shape
    centers = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(centers)

    for start in centers:
        if not _center_valid(start, shape, used_bbox, used_moat, used_rays):
            continue
        start_bbox = _bbox_cells(start)
        result = _extend_chain(
            shape=shape,
            length=length,
            centers=[start],
            records=[],
            used_bbox=used_bbox | start_bbox,
            used_moat=used_moat | _moat_cells(start, shape),
            used_rays=used_rays,
            prev_step=None,
        )
        if result is not None:
            return result
    return None


def _partition_lengths(total: int) -> list[int]:
    valid_counts = [
        count
        for count in range(1, 4)
        if total >= count + 1 and total <= CHAIN_LENGTH_MAX * count
    ]
    count = random.choice(valid_counts)
    lengths = [2] + [1] * (count - 1)
    remaining = total - sum(lengths)
    while remaining > 0:
        choices = [idx for idx, length in enumerate(lengths) if length < CHAIN_LENGTH_MAX]
        idx = random.choice(choices)
        lengths[idx] += 1
        remaining -= 1
    tail = lengths[1:]
    random.shuffle(tail)
    lengths[1:] = tail
    return lengths


def _draw_arrow(grid: np.ndarray, center: tuple[int, int], direction: str, tail_color: int) -> None:
    r, c = center
    r0 = r - 1
    c0 = c - 1
    spec = ARROW_SHAPES[direction]
    for rr, cc in spec["grey"]:
        grid[r0 + rr, c0 + cc] = GREY
    tr, tc = spec["tail"]
    grid[r0 + tr, c0 + tc] = tail_color


def _gen_attempt() -> Optional[np.ndarray]:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    total_arrows = random.randint(TOTAL_ARROWS_MIN, TOTAL_ARROWS_MAX)
    lengths = _partition_lengths(total_arrows)

    used_bbox: set[tuple[int, int]] = set()
    used_moat: set[tuple[int, int]] = set()
    used_rays: set[tuple[int, int]] = set()
    records: list[dict[str, object]] = []

    for length in lengths:
        placed = _place_chain(
            shape=(h, w),
            length=length,
            used_bbox=used_bbox,
            used_moat=used_moat,
            used_rays=used_rays,
        )
        if placed is None:
            return None
        chain_records, used_bbox, used_moat, used_rays = placed
        records.extend(chain_records)

    if len(records) > len(TAIL_PALETTE):
        return None

    colors = random.sample(TAIL_PALETTE, len(records))
    random.shuffle(records)
    grid = np.full((h, w), BACKGROUND, dtype=int)
    for record, color in zip(records, colors):
        center = tuple(record["center"])
        direction = str(record["direction"])
        _draw_arrow(grid, center, direction, color)

    try:
        out = main(grid)
    except AssertionError:
        return None

    changed = int(np.count_nonzero(out != grid))
    if changed < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid relay-arrow sample")
