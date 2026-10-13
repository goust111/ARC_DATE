import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

MARKER_COLORS = (RED, GREEN, YELLOW, ORANGE)
PATTERN_COLORS = (BLUE, 6, 8, 9)
CORNERS = ("tl", "tr", "bl", "br")

GRID_RANGE = (25, 30)
OUTER_SIZE_OPTIONS = (5, 7)
EXTRA_MARKER_RANGE = (0, 4)
MAX_GEN_ATTEMPTS = 400


def _rot_ccw_from_side(side: str) -> int:
    if side == "top":
        return 0
    if side == "right":
        return 3
    if side == "bottom":
        return 2
    if side == "left":
        return 1
    raise ValueError(side)


def _template_candidate(g: np.ndarray, r0: int, c0: int, size: int):
    mid = size // 2
    side_pos = {
        "top": (r0, c0 + mid),
        "right": (r0 + mid, c0 + size - 1),
        "bottom": (r0 + size - 1, c0 + mid),
        "left": (r0 + mid, c0),
    }
    side_set = set(side_pos.values())
    side_colors = {side: int(g[pos]) for side, pos in side_pos.items()}
    if len(set(side_colors.values())) != 4:
        return None
    if any(color in (BLACK, GREY) for color in side_colors.values()):
        return None

    for rr in range(r0, r0 + size):
        for cc in range(c0, c0 + size):
            if rr not in (r0, r0 + size - 1) and cc not in (c0, c0 + size - 1):
                continue
            value = int(g[rr, cc])
            if (rr, cc) in side_set:
                if value in (BLACK, GREY):
                    return None
            elif value != GREY:
                return None

    inner = g[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1]
    if not np.any(inner != BLACK):
        return None

    h, w = g.shape
    corner_dist = min(
        r0 + c0,
        r0 + (w - (c0 + size)),
        (h - (r0 + size)) + c0,
        (h - (r0 + size)) + (w - (c0 + size)),
    )
    return {
        "bbox": (r0, c0, r0 + size - 1, c0 + size - 1),
        "template": inner.copy(),
        "rotation_by_color": {
            side_colors["top"]: _rot_ccw_from_side("top"),
            side_colors["right"]: _rot_ccw_from_side("right"),
            side_colors["bottom"]: _rot_ccw_from_side("bottom"),
            side_colors["left"]: _rot_ccw_from_side("left"),
        },
        "score": (corner_dist, -size, r0, c0),
    }


def _find_template(g: np.ndarray):
    h, w = g.shape
    candidates = []
    for size in range(5, min(h, w) + 1, 2):
        for r0 in range(h - size + 1):
            for c0 in range(w - size + 1):
                candidate = _template_candidate(g, r0, c0, size)
                if candidate is not None:
                    candidates.append(candidate)
    if not candidates:
        raise ValueError("template area not found")
    candidates.sort(key=lambda item: item["score"])
    return candidates[0]


def _stamp(out: np.ndarray, center_r: int, center_c: int, template: np.ndarray) -> None:
    th, tw = template.shape
    anchor_r = th // 2
    anchor_c = tw // 2
    for rr, cc in np.argwhere(template != BLACK):
        tr = center_r + int(rr) - anchor_r
        tc = center_c + int(cc) - anchor_c
        if 0 <= tr < out.shape[0] and 0 <= tc < out.shape[1]:
            out[tr, tc] = int(template[rr, cc])


def _apply_stamps(
    grid: np.ndarray,
    template_bbox: tuple[int, int, int, int],
    template: np.ndarray,
    rotation_by_color: dict[int, int],
) -> np.ndarray:
    r0, c0, r1, c1 = template_bbox
    markers = []
    for rr in range(grid.shape[0]):
        for cc in range(grid.shape[1]):
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                continue
            color = int(grid[rr, cc])
            if color in rotation_by_color:
                markers.append((rr, cc, color))

    out = grid.copy()
    for rr, cc, color in sorted(markers):
        rotated = np.rot90(template, rotation_by_color[color])
        _stamp(out, rr, cc, rotated)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the framed template, read its border color→rotation mapping, and stamp
    rotated copies centered on every matching marker pixel.
    """
    g = np.asarray(grid, dtype=int)
    info = _find_template(g)
    return _apply_stamps(g, info["bbox"], info["template"], info["rotation_by_color"])


def _random_asymmetric_mask(size: int) -> np.ndarray | None:
    center = size // 2
    target = random.randint(max(3, size), size * size - 1)
    mask = np.zeros((size, size), dtype=bool)
    mask[center, center] = True

    while int(mask.sum()) < target:
        frontier = []
        for rr in range(size):
            for cc in range(size):
                if mask[rr, cc]:
                    continue
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < size and 0 <= nc < size and mask[nr, nc]:
                            frontier.append((rr, cc))
                            dr = dc = 2
                            break
                    else:
                        continue
                    break
        if not frontier:
            return None
        rr, cc = random.choice(frontier)
        mask[rr, cc] = True

    for turns in (1, 2, 3):
        if np.array_equal(mask, np.rot90(mask, turns)):
            return None
    return mask


def _make_template(inner_size: int) -> np.ndarray | None:
    for _ in range(60):
        mask = _random_asymmetric_mask(inner_size)
        if mask is None:
            continue
        color = random.choice(PATTERN_COLORS)
        template = np.zeros((inner_size, inner_size), dtype=int)
        template[mask] = color
        if template[inner_size // 2, inner_size // 2] == BLACK:
            continue
        return template
    return None


def _panel_origin(h: int, w: int, outer: int, corner: str) -> tuple[int, int]:
    margin_r = random.randint(0, 1)
    margin_c = random.randint(0, 1)
    if corner == "tl":
        return margin_r, margin_c
    if corner == "tr":
        return margin_r, w - outer - margin_c
    if corner == "bl":
        return h - outer - margin_r, margin_c
    if corner == "br":
        return h - outer - margin_r, w - outer - margin_c
    raise ValueError(corner)


def _draw_panel(grid: np.ndarray, r0: int, c0: int, template: np.ndarray) -> dict[int, int]:
    outer = template.shape[0] + 2
    grid[r0 : r0 + outer, c0 : c0 + outer] = BLACK
    grid[r0, c0 : c0 + outer] = GREY
    grid[r0 + outer - 1, c0 : c0 + outer] = GREY
    grid[r0 : r0 + outer, c0] = GREY
    grid[r0 : r0 + outer, c0 + outer - 1] = GREY

    mid = outer // 2
    side_pos = {
        "top": (r0, c0 + mid, RED),
        "right": (r0 + mid, c0 + outer - 1, GREEN),
        "bottom": (r0 + outer - 1, c0 + mid, YELLOW),
        "left": (r0 + mid, c0, ORANGE),
    }
    rotation_by_color = {}
    for side, (rr, cc, color) in side_pos.items():
        grid[rr, cc] = color
        rotation_by_color[color] = _rot_ccw_from_side(side)

    grid[r0 + 1 : r0 + outer - 1, c0 + 1 : c0 + outer - 1] = template
    return rotation_by_color


def _rotated_templates(template: np.ndarray, rotation_by_color: dict[int, int]) -> dict[int, np.ndarray]:
    return {color: np.rot90(template, turns) for color, turns in rotation_by_color.items()}


def _abs_cells(center: tuple[int, int], template: np.ndarray) -> set[tuple[int, int]]:
    cr, cc = center
    anchor_r = template.shape[0] // 2
    anchor_c = template.shape[1] // 2
    cells = set()
    for rr, xx in np.argwhere(template != BLACK):
        cells.add((cr + int(rr) - anchor_r, cc + int(xx) - anchor_c))
    return cells


def _intersects_bbox(cells: set[tuple[int, int]], bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return any(r0 <= rr <= r1 and c0 <= cc <= c1 for rr, cc in cells)


def _candidate_centers(
    h: int,
    w: int,
    panel_bbox: tuple[int, int, int, int],
    rotated: dict[int, np.ndarray],
) -> dict[int, list[tuple[int, int, set[tuple[int, int]]]]]:
    inner = next(iter(rotated.values())).shape[0]
    half = inner // 2
    out = {color: [] for color in rotated}
    for color, template in rotated.items():
        for rr in range(half, h - half):
            for cc in range(half, w - half):
                if panel_bbox[0] <= rr <= panel_bbox[2] and panel_bbox[1] <= cc <= panel_bbox[3]:
                    continue
                cells = _abs_cells((rr, cc), template)
                if _intersects_bbox(cells, panel_bbox):
                    continue
                out[color].append((rr, cc, cells))
    return out


def _choose_markers(
    candidates: dict[int, list[tuple[int, int, set[tuple[int, int]]]]],
    specs: list[int],
    force_overlap: bool,
) -> list[tuple[int, int, int]]:
    chosen = []
    used_centers = set()
    used_cells = set()
    overlap_made = False

    for idx, color in enumerate(specs):
        pool = [item for item in candidates[color] if (item[0], item[1]) not in used_centers]
        if not force_overlap:
            pool = [item for item in pool if item[2].isdisjoint(used_cells)]
            if not pool:
                raise ValueError("no non-overlapping marker placement")
            rr, cc, cells = random.choice(pool)
        else:
            if idx == 0:
                if not pool:
                    raise ValueError("no first marker placement")
                rr, cc, cells = random.choice(pool)
            elif not overlap_made:
                overlap_pool = [item for item in pool if item[2] & used_cells]
                if not overlap_pool:
                    raise ValueError("failed to force overlap")
                rr, cc, cells = random.choice(overlap_pool)
                overlap_made = True
            else:
                if not pool:
                    raise ValueError("no later marker placement")
                rr, cc, cells = random.choice(pool)

        chosen.append((rr, cc, color))
        used_centers.add((rr, cc))
        used_cells |= cells

    if force_overlap and not overlap_made:
        raise ValueError("overlap branch missing")
    return chosen


def _expected_output(
    grid: np.ndarray,
    template_bbox: tuple[int, int, int, int],
    template: np.ndarray,
    markers: list[tuple[int, int, int]],
    rotation_by_color: dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    for rr, cc, color in sorted(markers):
        rotated = np.rot90(template, rotation_by_color[color])
        _stamp(out, rr, cc, rotated)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    outer = random.choice(OUTER_SIZE_OPTIONS)
    inner = outer - 2
    corner = random.choice(CORNERS)
    template = _make_template(inner)
    if template is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    panel_r0, panel_c0 = _panel_origin(h, w, outer, corner)
    rotation_by_color = _draw_panel(grid, panel_r0, panel_c0, template)
    panel_bbox = (panel_r0, panel_c0, panel_r0 + outer - 1, panel_c0 + outer - 1)

    rotated = _rotated_templates(template, rotation_by_color)
    candidates = _candidate_centers(h, w, panel_bbox, rotated)

    extra = random.randint(*EXTRA_MARKER_RANGE)
    specs = list(MARKER_COLORS) + [random.choice(MARKER_COLORS) for _ in range(extra)]
    random.shuffle(specs)
    force_overlap = random.random() < 0.5

    try:
        markers = _choose_markers(candidates, specs, force_overlap)
    except ValueError:
        return None

    for rr, cc, color in markers:
        grid[rr, cc] = color

    expected = _expected_output(grid, panel_bbox, template, markers, rotation_by_color)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
