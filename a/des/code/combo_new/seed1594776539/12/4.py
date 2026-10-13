import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("top", "bottom", "left", "right")
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_RETRIES = 400
SHAPE_RETRIES = 80
PLACE_RETRIES = 120


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return {(r - min_r, c - min_c) for r, c in coords}


def _is_gap_boundary_cell(
    r: int,
    c: int,
    inner_h: int,
    inner_w: int,
    side: str,
    gap_inner_start: int,
) -> bool:
    if side == "left":
        return c == 0 and r in (gap_inner_start, gap_inner_start + 1)
    if side == "right":
        return c == inner_w - 1 and r in (gap_inner_start, gap_inner_start + 1)
    if side == "top":
        return r == 0 and c in (gap_inner_start, gap_inner_start + 1)
    return r == inner_h - 1 and c in (gap_inner_start, gap_inner_start + 1)


def _shape_cell_allowed(
    r: int,
    c: int,
    inner_h: int,
    inner_w: int,
    side: str,
    gap_inner_start: int,
) -> bool:
    if r in (0, inner_h - 1) or c in (0, inner_w - 1):
        return _is_gap_boundary_cell(r, c, inner_h, inner_w, side, gap_inner_start)
    return True


def _reflected_normalized(
    coords: set[tuple[int, int]],
    vertical_axis: bool,
) -> set[tuple[int, int]]:
    norm = _normalize(coords)
    max_r = max(r for r, _ in norm)
    max_c = max(c for _, c in norm)
    if vertical_axis:
        refl = {(r, max_c - c) for r, c in norm}
    else:
        refl = {(max_r - r, c) for r, c in norm}
    return _normalize(refl)


def _frame_gap_local(local: np.ndarray) -> tuple[str, int] | None:
    h, w = local.shape
    if h < 5 or w < 5:
        return None

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(local & ~border):
        return None

    missing = np.argwhere(border & ~local)
    if len(missing) != 2:
        return None

    rs = missing[:, 0]
    cs = missing[:, 1]
    if np.all(rs == 0):
        c0, c1 = sorted(int(x) for x in cs)
        if 0 < c0 and c1 == c0 + 1 and c1 < w - 1:
            return "top", c0
    if np.all(rs == h - 1):
        c0, c1 = sorted(int(x) for x in cs)
        if 0 < c0 and c1 == c0 + 1 and c1 < w - 1:
            return "bottom", c0
    if np.all(cs == 0):
        r0, r1 = sorted(int(x) for x in rs)
        if 0 < r0 and r1 == r0 + 1 and r1 < h - 1:
            return "left", r0
    if np.all(cs == w - 1):
        r0, r1 = sorted(int(x) for x in rs)
        if 0 < r0 and r1 == r0 + 1 and r1 < h - 1:
            return "right", r0
    return None


def _find_containers(grid: np.ndarray) -> list[dict]:
    containers: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if c not in (BG, GREY)):
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in comp:
                local[r - r0, c - c0] = True
            gap = _frame_gap_local(local)
            if gap is None:
                continue
            if not np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == GREY):
                continue
            containers.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "side": gap[0],
                }
            )
    containers.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pour each grey pattern out of its gapped frame by mirroring it across the gap side and recoloring it.

    Input structure:
    - The background is black.
    - The grid contains several non-overlapping rectangular containers made from 1-pixel monochrome borders.
    - Each border has exactly one 2-pixel gap on one side, away from the corners.
    - Inside each container there is a small 4-connected grey pattern; different containers use different border colors.

    Rule:
    - Keep every rectangular border exactly as it is, including its 2-pixel gap.
    - Remove the grey pattern from the inside of each container, turning those cells black.
    - Detect which side contains the gap; that side is the pouring direction.
    - Mirror the grey pattern across that container side: left/right gaps cause a horizontal flip, top/bottom gaps cause a vertical flip.
    - Stamp the mirrored pattern outside the container in the reflected positions, recoloring it with the container border color.
    - If reflected pixels fall beyond the grid boundary, drop those out-of-bounds cells.

    Key insight:
    - The missing border segment identifies both where the pattern exits and which mirror axis to use, while the emitted copy changes from grey to the frame color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    h, w = g.shape
    out = g.copy()

    for container in _find_containers(g):
        color = int(container["color"])
        r0, c0, r1, c1 = container["bbox"]
        side = container["side"]

        pattern = [
            (r, c)
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
            if g[r, c] == GREY
        ]

        for r, c in pattern:
            out[r, c] = BG

        for r, c in pattern:
            if side == "left":
                rr, cc = r, 2 * c0 - c
            elif side == "right":
                rr, cc = r, 2 * c1 - c
            elif side == "top":
                rr, cc = 2 * r0 - r, c
            else:
                rr, cc = 2 * r1 - r, c
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = color

    return out


def _make_shape(inner_h: int, inner_w: int, side: str, gap_inner_start: int) -> set[tuple[int, int]] | None:
    anchors = []
    if side == "left":
        anchors = [(gap_inner_start, 0), (gap_inner_start + 1, 0)]
    elif side == "right":
        anchors = [(gap_inner_start, inner_w - 1), (gap_inner_start + 1, inner_w - 1)]
    elif side == "top":
        anchors = [(0, gap_inner_start), (0, gap_inner_start + 1)]
    else:
        anchors = [(inner_h - 1, gap_inner_start), (inner_h - 1, gap_inner_start + 1)]

    for _ in range(SHAPE_RETRIES):
        target_area = random.randint(3, 6)
        shape: set[tuple[int, int]] = {random.choice(anchors)}

        while len(shape) < target_area:
            frontier = set()
            for r, c in shape:
                for nr, nc in _neighbors4(r, c, inner_h, inner_w):
                    if (nr, nc) in shape:
                        continue
                    if _shape_cell_allowed(nr, nc, inner_h, inner_w, side, gap_inner_start):
                        frontier.add((nr, nc))
            if not frontier:
                break
            inward = [
                cell
                for cell in frontier
                if not _is_gap_boundary_cell(
                    cell[0], cell[1], inner_h, inner_w, side, gap_inner_start
                )
            ]
            shape.add(random.choice(inward or list(frontier)))

        if len(shape) != target_area:
            continue
        if all(
            _is_gap_boundary_cell(r, c, inner_h, inner_w, side, gap_inner_start)
            for r, c in shape
        ):
            continue

        reflected = _reflected_normalized(shape, vertical_axis=side in ("left", "right"))
        if reflected == _normalize(shape):
            continue

        return shape
    return None


def _frame_patch(outer_h: int, outer_w: int, side: str, gap_inner_start: int, color: int) -> np.ndarray:
    patch = np.zeros((outer_h, outer_w), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    if side == "top":
        patch[0, gap_inner_start + 1 : gap_inner_start + 3] = BG
    elif side == "bottom":
        patch[-1, gap_inner_start + 1 : gap_inner_start + 3] = BG
    elif side == "left":
        patch[gap_inner_start + 1 : gap_inner_start + 3, 0] = BG
    else:
        patch[gap_inner_start + 1 : gap_inner_start + 3, -1] = BG
    return patch


def _projected_coords(
    top: int,
    left: int,
    outer_h: int,
    outer_w: int,
    side: str,
    shape: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    r0, c0 = top, left
    r1, c1 = top + outer_h - 1, left + outer_w - 1
    for ir, ic in shape:
        r = top + 1 + ir
        c = left + 1 + ic
        if side == "left":
            rr, cc = r, 2 * c0 - c
        elif side == "right":
            rr, cc = r, 2 * c1 - c
        elif side == "top":
            rr, cc = 2 * r0 - r, c
        else:
            rr, cc = 2 * r1 - r, c
        coords.append((rr, cc))
    return coords


def _mark_rect(mask: np.ndarray, rect: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = rect
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _rect_free(mask: np.ndarray, rect: tuple[int, int, int, int], pad: int = 1) -> bool:
    h, w = mask.shape
    r0, c0, r1, c1 = rect
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _candidate_axis_values(limit: int, want_edge: bool, toward_end: bool) -> list[int]:
    if limit <= 0:
        return [0]
    values = list(range(limit + 1))
    if not want_edge:
        return values
    edge_span = min(2, limit)
    if toward_end:
        return list(range(limit - edge_span, limit + 1))
    return list(range(0, edge_span + 1))


def _build_specs(k: int) -> list[dict] | None:
    colors = random.sample(COLOR_POOL, k)
    sides = random.sample(list(SIDES), k)
    specs: list[dict] = []
    for color, side in zip(colors, sides):
        inner_h = random.randint(4, 7)
        inner_w = random.randint(4, 7)
        if side in ("left", "right"):
            gap_inner_start = random.randint(1, inner_h - 3)
        else:
            gap_inner_start = random.randint(1, inner_w - 3)
        shape = _make_shape(inner_h, inner_w, side, gap_inner_start)
        if shape is None:
            return None
        specs.append(
            {
                "color": color,
                "side": side,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
                "gap_inner_start": gap_inner_start,
                "shape": shape,
                "want_clip": random.random() < 0.35,
            }
        )
    specs.sort(
        key=lambda spec: (
            -(spec["outer_h"] * spec["outer_w"]),
            spec["side"],
        )
    )
    return specs


def _place_specs(grid: np.ndarray, specs: list[dict]) -> bool:
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)

    for spec in specs:
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        side = spec["side"]
        want_clip = bool(spec["want_clip"])

        top_edge = side == "top"
        bottom_edge = side == "bottom"
        left_edge = side == "left"
        right_edge = side == "right"

        row_values = _candidate_axis_values(h - outer_h, want_clip and top_edge, want_clip and bottom_edge)
        col_values = _candidate_axis_values(w - outer_w, want_clip and left_edge, want_clip and right_edge)
        attempts = [(random.choice(row_values), random.choice(col_values)) for _ in range(PLACE_RETRIES)]

        placed = False
        for top, left in attempts:
            frame_rect = (top, left, top + outer_h - 1, left + outer_w - 1)
            if not _rect_free(occupied, frame_rect, pad=1):
                continue

            projected = _projected_coords(
                top,
                left,
                outer_h,
                outer_w,
                side,
                spec["shape"],
            )
            visible = [(r, c) for r, c in projected if 0 <= r < h and 0 <= c < w]
            clipped = len(visible) < len(projected)
            if want_clip != clipped:
                continue
            if not visible:
                continue

            if visible:
                vr0, vc0, vr1, vc1 = _bbox(visible)
                proj_rect = (vr0, vc0, vr1, vc1)
                if not _rect_free(occupied, proj_rect, pad=1):
                    continue
            else:
                proj_rect = None

            patch = _frame_patch(
                outer_h,
                outer_w,
                side,
                int(spec["gap_inner_start"]),
                int(spec["color"]),
            )
            grid[top : top + outer_h, left : left + outer_w] = patch
            for ir, ic in spec["shape"]:
                grid[top + 1 + ir, left + 1 + ic] = GREY

            _mark_rect(occupied, frame_rect, pad=1)
            if proj_rect is not None:
                _mark_rect(occupied, proj_rect, pad=1)
            placed = True
            break

        if not placed:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    k = random.choice([2, 3, 4])
    min_side = 10 + 3 * k
    h = random.randint(min_side, 30)
    w = random.randint(min_side, 30)
    specs = _build_specs(k)
    if specs is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    if not _place_specs(grid, specs):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if np.any(out == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
