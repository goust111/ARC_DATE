from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
YELLOW = 4
BODY_COLORS = tuple(c for c in range(10) if c not in (BG, YELLOW))
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PERP_DIRS = {
    "up": ((0, -1), (0, 1)),
    "down": ((0, -1), (0, 1)),
    "left": ((-1, 0), (1, 0)),
    "right": ((-1, 0), (1, 0)),
}
VEC_TO_NAME = {vec: name for name, vec in DIRS.items()}
MAX_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bg_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel())
    return int(counts.most_common(1)[0][0])


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(sorted(comp))
    return comps


def _bbox(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(cells)
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _trim_cells(
    body: set[tuple[int, int]],
    arrow: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    all_cells = body | arrow
    r0, c0, _r1, _c1 = _bbox(all_cells)
    body_n = {(r - r0, c - c0) for r, c in body}
    arrow_n = {(r - r0, c - c0) for r, c in arrow}
    return body_n, arrow_n


def _mask_connected(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    return len(_components_from_mask(mask)) == 1


def _decode_arrow(
    yellow_cells: list[tuple[int, int]] | set[tuple[int, int]],
    body_cells: list[tuple[int, int]] | set[tuple[int, int]],
) -> dict | None:
    yellow = set(yellow_cells)
    body = set(body_cells)
    if len(yellow) != 3:
        return None

    degrees = {
        cell: sum((cell[0] + dr, cell[1] + dc) in yellow for dr, dc in DIRS.values())
        for cell in yellow
    }
    centers = [cell for cell, deg in degrees.items() if deg == 2]
    if len(centers) != 1:
        return None
    center = centers[0]

    arms = [(r - center[0], c - center[1]) for r, c in yellow if (r, c) != center]
    if len(arms) != 2 or any(abs(dr) + abs(dc) != 1 for dr, dc in arms):
        return None
    if arms[0][0] == arms[1][0] or arms[0][1] == arms[1][1]:
        return None

    def forward_body(vec: tuple[int, int]) -> bool:
        return (center[0] + 2 * vec[0], center[1] + 2 * vec[1]) in body

    def support_score(vec: tuple[int, int]) -> int:
        arm = (center[0] + vec[0], center[1] + vec[1])
        score = 4 if forward_body(vec) else 0
        for nr, nc in _neighbors4(*arm):
            if (nr, nc) == center or (nr, nc) in yellow:
                continue
            if (nr, nc) in body:
                score += 1
        return score

    forward = [forward_body(vec) for vec in arms]
    if forward.count(True) == 1:
        tip_vec = arms[forward.index(False)]
    else:
        scores = [support_score(vec) for vec in arms]
        if scores[0] != scores[1]:
            tip_vec = arms[0] if scores[0] < scores[1] else arms[1]
        else:
            tip_vec = sorted(arms)[0]

    return {
        "center": center,
        "direction": VEC_TO_NAME[tip_vec],
        "tip_vec": tip_vec,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each marked region repeatedly toward the grid edge using its internal arrow.

    Input structure:
    - The dominant background is black.
    - Each foreground object is one 4-connected region made of a unique body color plus exactly three yellow arrow pixels.
    - The yellow arrow is an L-shaped 3-pixel marker embedded in the region; one arm is the shaft and the perpendicular arm is the tip.

    Rule:
    - For each foreground component, locate its three yellow cells and infer the tip direction among up, down, left, and right.
    - Remove the yellow arrow from the region shape, compute that body shape's span along the tip direction from its bounding box, and translate the full body by that span repeatedly.
    - Paint every in-bounds translated copy in yellow while leaving the original component unchanged; if a copy overlaps another original region, the yellow copy stays on top.

    Key insight:
    - The arrow supplies only a direction code; the repeated translation distance comes from the marked region's own directional span, so the body tiles itself toward the boundary at a fixed period.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    out = g.copy()

    for comp in _components_from_mask(g != bg):
        yellow = [(r, c) for r, c in comp if int(g[r, c]) == YELLOW]
        body = {(r, c) for r, c in comp if int(g[r, c]) != YELLOW}
        if len(yellow) != 3 or not body:
            continue
        if len({int(g[r, c]) for r, c in body}) != 1:
            continue

        arrow = _decode_arrow(yellow, body)
        if arrow is None:
            continue
        direction = arrow["direction"]
        dr, dc = DIRS[direction]

        br0, bc0, br1, bc1 = _bbox(body)
        span = (br1 - br0 + 1) if direction in {"up", "down"} else (bc1 - bc0 + 1)
        step = 1
        while True:
            shifted = [(r + dr * span * step, c + dc * span * step) for r, c in body]
            if not all(0 <= rr < h and 0 <= cc < w for rr, cc in shifted):
                break
            for rr, cc in shifted:
                out[rr, cc] = YELLOW
            step += 1

    return out


def _shape_boundary(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    boundary: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if any(not _in_bounds((h, w), nr, nc) or not mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                boundary.append((r, c))
    return boundary


def _protected_halo(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if _in_bounds(shape, rr, cc):
                    out.add((rr, cc))
    return out


def _make_local_object(direction: str, rng: np.random.Generator) -> dict | None:
    d = DIRS[direction]
    for _ in range(120):
        height = int(rng.integers(6, 9))
        width = int(rng.integers(6, 9))
        shaft = tuple(int(v) for v in PERP_DIRS[direction][int(rng.integers(0, 2))])

        row_candidates: list[int] = []
        col_candidates: list[int] = []
        for r in range(height):
            for c in range(width):
                pts = [
                    (r, c),
                    (r + d[0], c + d[1]),
                    (r + shaft[0], c + shaft[1]),
                    (r + 2 * shaft[0], c + 2 * shaft[1]),
                    (r + 2 * d[0], c + 2 * d[1]),
                ]
                if all(_in_bounds((height, width), rr, cc) for rr, cc in pts):
                    row_candidates.append(r)
                    col_candidates.append(c)
        if not row_candidates:
            continue

        if direction == "right":
            center_col = width - 3
            row_options = [r for r in range(height) if _in_bounds((height, width), r + shaft[0], center_col + shaft[1])
                           and _in_bounds((height, width), r + 2 * shaft[0], center_col + 2 * shaft[1])]
            if not row_options:
                continue
            center = (int(rng.choice(row_options)), center_col)
        elif direction == "left":
            center_col = 2
            row_options = [r for r in range(height) if _in_bounds((height, width), r + shaft[0], center_col + shaft[1])
                           and _in_bounds((height, width), r + 2 * shaft[0], center_col + 2 * shaft[1])]
            if not row_options:
                continue
            center = (int(rng.choice(row_options)), center_col)
        elif direction == "down":
            center_row = height - 3
            col_options = [c for c in range(width) if _in_bounds((height, width), center_row + shaft[0], c + shaft[1])
                           and _in_bounds((height, width), center_row + 2 * shaft[0], c + 2 * shaft[1])]
            if not col_options:
                continue
            center = (center_row, int(rng.choice(col_options)))
        else:
            center_row = 2
            col_options = [c for c in range(width) if _in_bounds((height, width), center_row + shaft[0], c + shaft[1])
                           and _in_bounds((height, width), center_row + 2 * shaft[0], c + 2 * shaft[1])]
            if not col_options:
                continue
            center = (center_row, int(rng.choice(col_options)))

        tip = (center[0] + d[0], center[1] + d[1])
        shaft_cell = (center[0] + shaft[0], center[1] + shaft[1])
        shaft_support = (center[0] + 2 * shaft[0], center[1] + 2 * shaft[1])
        tip_notch = (center[0] + 2 * d[0], center[1] + 2 * d[1])
        arrow = {center, tip, shaft_cell}

        mask = np.ones((height, width), dtype=bool)
        mask[tip_notch] = False
        for r, c in arrow:
            mask[r, c] = False

        protected = _protected_halo(arrow | {tip_notch, shaft_support}, mask.shape)
        max_removals = max(2, (height * width) // 6)
        removals = int(rng.integers(0, max_removals + 1))
        for _ in range(removals):
            candidates = [cell for cell in _shape_boundary(mask) if cell not in protected]
            if not candidates:
                break
            accepted = False
            for cell in rng.permutation(np.array(candidates, dtype=int)).tolist():
                rr, cc = int(cell[0]), int(cell[1])
                if not mask[rr, cc]:
                    continue
                trial = mask.copy()
                trial[rr, cc] = False
                if trial.sum() < 12 or not _mask_connected(trial):
                    continue
                mask = trial
                accepted = True
                break
            if not accepted:
                break

        body = {(r, c) for r in range(height) for c in range(width) if mask[r, c]}
        if len(body) < 12:
            continue
        body, arrow = _trim_cells(body, arrow)
        all_cells = body | arrow
        h2 = max(r for r, _ in all_cells) + 1
        w2 = max(c for _, c in all_cells) + 1
        if len(_components_from_mask(_cells_to_mask(body, (h2, w2)))) != 1:
            continue
        if any(r in (0, h2 - 1) or c in (0, w2 - 1) for r, c in arrow):
            continue
        parsed = _decode_arrow(arrow, body)
        if parsed is None or parsed["direction"] != direction:
            continue

        body_bbox = _bbox(body)
        span = (body_bbox[2] - body_bbox[0] + 1) if direction in {"up", "down"} else (body_bbox[3] - body_bbox[1] + 1)
        return {
            "body": body,
            "arrow": arrow,
            "component": body | arrow,
            "direction": direction,
            "span": int(span),
            "shape": (h2, w2),
        }
    return None


def _cells_to_mask(cells: set[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _copy_count(
    body: set[tuple[int, int]],
    direction: str,
    span: int,
    origin: tuple[int, int],
    shape: tuple[int, int],
) -> int:
    dr, dc = DIRS[direction]
    body_r0, body_c0, body_r1, body_c1 = _bbox(body)
    if direction == "right":
        return max(0, (shape[1] - 1 - (origin[1] + body_c1)) // span)
    if direction == "left":
        return max(0, (origin[1] + body_c0) // span)
    if direction == "down":
        return max(0, (shape[0] - 1 - (origin[0] + body_r1)) // span)
    return max(0, (origin[0] + body_r0) // span)


def _translate(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _halo(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if _in_bounds(shape, rr, cc):
                    out.add((rr, cc))
    return out


def _place_object(
    obj: dict,
    blocked: set[tuple[int, int]],
    grid_shape: tuple[int, int],
    rng: np.random.Generator,
) -> dict | None:
    comp = obj["component"]
    comp_r0, comp_c0, comp_r1, comp_c1 = _bbox(comp)
    assert comp_r0 == 0 and comp_c0 == 0

    row_range = range(0, grid_shape[0] - comp_r1)
    col_range = range(0, grid_shape[1] - comp_c1)
    candidates: list[dict] = []
    for top in row_range:
        for left in col_range:
            copies = _copy_count(obj["body"], obj["direction"], obj["span"], (top, left), grid_shape)
            if copies < 1 or copies > 3:
                continue
            reserve = _translate(comp, top, left)
            dr, dc = DIRS[obj["direction"]]
            for step in range(1, copies + 1):
                reserve |= _translate(obj["body"], top + dr * obj["span"] * step, left + dc * obj["span"] * step)
            if reserve & blocked:
                continue
            candidates.append(
                {
                    "top": top,
                    "left": left,
                    "copies": int(copies),
                    "reserve": reserve,
                }
            )

    if not candidates:
        return None
    cand = candidates[int(rng.integers(0, len(candidates)))]
    top = int(cand["top"])
    left = int(cand["left"])
    dr, dc = DIRS[obj["direction"]]
    body_global = _translate(obj["body"], top, left)
    arrow_global = _translate(obj["arrow"], top, left)
    projected = [
        _translate(obj["body"], top + dr * obj["span"] * step, left + dc * obj["span"] * step)
        for step in range(1, int(cand["copies"]) + 1)
    ]
    return {
        **obj,
        "top": top,
        "left": left,
        "copies": int(cand["copies"]),
        "body_global": body_global,
        "arrow_global": arrow_global,
        "projected": projected,
        "reserve_global": cand["reserve"],
    }


def _render_input(shape: tuple[int, int], placed: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for rec in placed:
        color = int(rec["color"])
        for r, c in rec["body_global"]:
            grid[r, c] = color
        for r, c in rec["arrow_global"]:
            grid[r, c] = YELLOW
    return grid


def _generated_pair_is_valid(grid: np.ndarray, placed: list[dict]) -> bool:
    if _bg_color(grid) != BG:
        return False
    solved = main(grid)
    if np.array_equal(solved, grid):
        return False
    if int(np.count_nonzero(grid == YELLOW)) != 3 * len(placed):
        return False
    if int(np.count_nonzero(solved == YELLOW)) <= int(np.count_nonzero(grid == YELLOW)):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(18, 31))
    w = int(rng.integers(18, 31))
    area = h * w
    max_objects = 1 if area < 420 else 2 if area < 560 else 3 if area < 760 else 4
    object_count = int(rng.integers(1, max_objects + 1))
    colors = [int(c) for c in rng.choice(BODY_COLORS, size=object_count, replace=False)]
    directions = list(rng.permutation(np.array(list(DIRS), dtype=object))[:object_count])

    blocked: set[tuple[int, int]] = set()
    placed: list[dict] = []
    for color, direction in zip(colors, directions):
        placed_rec = None
        for _ in range(80):
            local = _make_local_object(str(direction), rng)
            if local is None:
                continue
            rec = _place_object({**local, "color": int(color)}, blocked, (h, w), rng)
            if rec is None:
                continue
            placed_rec = rec
            blocked |= _halo(rec["reserve_global"], (h, w))
            placed.append(rec)
            break
        if placed_rec is None:
            return None

    grid = _render_input((h, w), placed)
    if not _generated_pair_is_valid(grid, placed):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
