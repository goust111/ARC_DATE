from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ARROW_TEMPLATES = {
    frozenset({(0, 1), (0, 0), (0, 2), (1, 1), (2, 1)}): "up",
    frozenset({(0, 1), (1, 1), (2, 1), (2, 0), (2, 2)}): "down",
    frozenset({(1, 0), (1, 1), (1, 2), (0, 0), (2, 0)}): "left",
    frozenset({(1, 0), (1, 1), (1, 2), (0, 2), (2, 2)}): "right",
}
STEP_BY_DIR = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# --------------- solver ---------------


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(len(cells)),
                }
            )
    return comps


def _arrow_direction(coords: np.ndarray) -> str | None:
    if len(coords) != 5:
        return None
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    if r1 - r0 != 2 or c1 - c0 != 2:
        return None
    norm = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    return ARROW_TEMPLATES.get(norm)


def _boundary_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    out = [(r0, c) for c in range(c0, c1 + 1)]
    out.extend((r, c1) for r in range(r0 + 1, r1))
    out.extend((r1, c) for c in range(c1, c0 - 1, -1))
    out.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return out


def _edge_coords(r0: int, c0: int, r1: int, c1: int, direction: str) -> list[tuple[int, int]]:
    if direction == "up":
        return [(r0, c) for c in range(c0, c1 + 1)]
    if direction == "down":
        return [(r1, c) for c in range(c0, c1 + 1)]
    if direction == "left":
        return [(r, c0) for r in range(r0, r1 + 1)]
    return [(r, c1) for r in range(r0, r1 + 1)]


def _validate_frame(
    grid: np.ndarray,
    bg: int,
    arrow_comp: dict,
    direction: str,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> dict | None:
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
        return None

    arrow_coords = arrow_comp["coords"]
    ar0, ac0, ar1, ac1 = arrow_comp["bbox"]
    if not (r0 < ar0 and ar1 < r1 and c0 < ac0 and ac1 < c1):
        return None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    if int(np.count_nonzero(inner != bg)) != 5:
        return None

    arrow_set = {(int(r), int(c)) for r, c in arrow_coords}
    inner_non_bg = {
        (int(r0 + 1 + rr), int(c0 + 1 + cc))
        for rr, cc in np.argwhere(inner != bg)
    }
    if inner_non_bg != arrow_set:
        return None

    arrow_color = int(arrow_comp["color"])
    frame_color: int | None = None
    marker_count = 0
    markers: set[tuple[int, int]] = set()
    for br, bc in _boundary_coords(r0, c0, r1, c1):
        value = int(grid[br, bc])
        if value == bg:
            return None
        if value == arrow_color:
            marker_count += 1
            markers.add((br, bc))
            continue
        if frame_color is None:
            frame_color = value
        elif value != frame_color:
            return None

    if frame_color is None or not (1 <= marker_count <= 6):
        return None

    if not any(coord not in markers for coord in _edge_coords(r0, c0, r1, c1, direction)):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "frame_color": int(frame_color),
        "arrow_color": arrow_color,
        "direction": direction,
        "length": marker_count,
        "markers": markers,
    }


def _find_frame_for_arrow(grid: np.ndarray, bg: int, arrow_comp: dict, direction: str) -> dict:
    h, w = grid.shape
    ar0, ac0, ar1, ac1 = arrow_comp["bbox"]
    candidates: list[dict] = []

    for r0 in range(ar0 - 1, -1, -1):
        for r1 in range(ar1 + 1, h):
            for c0 in range(ac0 - 1, -1, -1):
                for c1 in range(ac1 + 1, w):
                    frame = _validate_frame(grid, bg, arrow_comp, direction, r0, c0, r1, c1)
                    if frame is not None:
                        candidates.append(frame)

    assert candidates, "expected an enclosing hollow frame for each arrow"
    candidates.sort(
        key=lambda item: (
            (item["bbox"][2] - item["bbox"][0] + 1) * (item["bbox"][3] - item["bbox"][1] + 1),
            item["bbox"],
        )
    )
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit outward rays from the arrow-facing edge of each hollow frame.

    The arrow direction selects the emitting edge, and the number of border
    pixels recolored to the arrow color gives the ray length. Rays start from
    all non-marker pixels on that edge and stop early on non-background cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _bg_color(g)

    components = _find_components(g, bg)
    arrows = []
    for comp in components:
        direction = _arrow_direction(comp["coords"])
        if direction is not None:
            arrows.append((comp, direction))

    assert arrows, "expected at least one 5-pixel arrow"

    frames = []
    for comp, direction in arrows:
        frame = _find_frame_for_arrow(g, bg, comp, direction)
        frames.append(frame)

    boxes = [frame["bbox"] for frame in frames]
    assert len(set(boxes)) == len(boxes), "each frame should contain exactly one arrow"

    out = g.copy()
    frames.sort(key=lambda item: item["bbox"])
    h, w = g.shape
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        dr, dc = STEP_BY_DIR[frame["direction"]]
        for sr, sc in _edge_coords(r0, c0, r1, c1, frame["direction"]):
            if (sr, sc) in frame["markers"]:
                continue
            rr = sr
            cc = sc
            for _ in range(frame["length"]):
                rr += dr
                cc += dc
                if not (0 <= rr < h and 0 <= cc < w):
                    break
                if int(out[rr, cc]) != bg:
                    break
                out[rr, cc] = int(frame["frame_color"])

    return out


# --------------- generator ---------------

PALETTE = tuple(range(10))
DIRS = ("up", "down", "left", "right")
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
FRAME_COUNT_RANGE = (1, 4)
FRAME_SIDE_RANGE = (7, 10)
RAY_LEN_RANGE = (1, 6)
PLACE_ATTEMPTS = 120
GEN_ATTEMPTS = 400


def _arrow_patch(color: int, direction: str) -> np.ndarray:
    patch = np.zeros((3, 3), dtype=int)
    for r, c in next(template for template, name in ARROW_TEMPLATES.items() if name == direction):
        patch[int(r), int(c)] = int(color)
    return patch


def _sample_count_and_size() -> tuple[int, int, int]:
    count = random.randint(*FRAME_COUNT_RANGE)
    min_dim = max(GRID_H_RANGE[0], 6 * count + 8)
    h = random.randint(min(min_dim, GRID_H_RANGE[1]), GRID_H_RANGE[1])
    min_w = max(GRID_W_RANGE[0], 6 * count + 8)
    w = random.randint(min(min_w, GRID_W_RANGE[1]), GRID_W_RANGE[1])
    return count, h, w


def _perimeter_cycle(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    return _boundary_coords(r0, c0, r1, c1)


def _sample_marker_positions(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    direction: str,
    length: int,
) -> set[tuple[int, int]] | None:
    cycle = _perimeter_cycle(r0, c0, r1, c1)
    edge = set(_edge_coords(r0, c0, r1, c1, direction))
    size = len(cycle)

    for _ in range(100):
        picks = sorted(random.sample(range(size), length))
        ok = True
        for i, idx in enumerate(picks):
            prev_idx = picks[i - 1]
            if i > 0 and idx - prev_idx == 1:
                ok = False
                break
        if ok and size > 1 and picks[0] == 0 and picks[-1] == size - 1:
            ok = False
        if not ok:
            continue
        markers = {cycle[idx] for idx in picks}
        if any(coord not in markers for coord in edge):
            return markers
    return None


def _place_one_frame(
    grid: np.ndarray,
    blocked: np.ndarray,
    frame_color: int,
    arrow_color: int,
    direction: str,
    length: int,
    prefer_truncation: bool,
) -> dict | None:
    h, w = grid.shape
    outer_h = random.randint(*FRAME_SIDE_RANGE)
    outer_w = random.randint(*FRAME_SIDE_RANGE)

    r_choices = list(range(1, h - outer_h))
    c_choices = list(range(1, w - outer_w))
    if not r_choices or not c_choices:
        return None

    if prefer_truncation:
        if direction == "up":
            r_choices = [r for r in r_choices if r <= length]
        elif direction == "down":
            r_choices = [r for r in r_choices if h - (r + outer_h) <= length]
        elif direction == "left":
            c_choices = [c for c in c_choices if c <= length]
        else:
            c_choices = [c for c in c_choices if w - (c + outer_w) <= length]
        if not r_choices or not c_choices:
            return None

    for _ in range(PLACE_ATTEMPTS):
        r0 = random.choice(r_choices)
        c0 = random.choice(c_choices)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        if blocked[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
            continue

        top_choices = list(range(r0 + 2, r1 - 3))
        left_choices = list(range(c0 + 2, c1 - 3))
        if not top_choices or not left_choices:
            continue
        ar0 = random.choice(top_choices)
        ac0 = random.choice(left_choices)
        markers = _sample_marker_positions(r0, c0, r1, c1, direction, length)
        if markers is None:
            continue

        grid[r0 : r1 + 1, c0 : c1 + 1] = grid[r0 : r1 + 1, c0 : c1 + 1]
        grid[r0, c0 : c1 + 1] = frame_color
        grid[r1, c0 : c1 + 1] = frame_color
        grid[r0 : r1 + 1, c0] = frame_color
        grid[r0 : r1 + 1, c1] = frame_color
        for mr, mc in markers:
            grid[mr, mc] = arrow_color
        patch = _arrow_patch(arrow_color, direction)
        mask = patch != 0
        view = grid[ar0 : ar0 + 3, ac0 : ac0 + 3]
        view[mask] = arrow_color

        blocked[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True
        return {
            "bbox": (r0, c0, r1, c1),
            "frame_color": int(frame_color),
            "arrow_color": int(arrow_color),
            "direction": direction,
            "length": int(length),
            "markers": markers,
        }
    return None


def _expected_output(inp: np.ndarray, bg: int, specs: list[dict]) -> tuple[np.ndarray, bool, bool]:
    out = inp.copy()
    h, w = inp.shape
    painted: set[tuple[int, int]] = set()
    truncated_any = False
    changed = False

    for spec in sorted(specs, key=lambda item: item["bbox"]):
        r0, c0, r1, c1 = spec["bbox"]
        dr, dc = STEP_BY_DIR[spec["direction"]]
        for sr, sc in _edge_coords(r0, c0, r1, c1, spec["direction"]):
            if (sr, sc) in spec["markers"]:
                continue
            rr = sr
            cc = sc
            advanced = 0
            for _ in range(spec["length"]):
                rr += dr
                cc += dc
                if not (0 <= rr < h and 0 <= cc < w):
                    truncated_any = True
                    break
                if int(inp[rr, cc]) != bg:
                    truncated_any = True
                    break
                if (rr, cc) in painted:
                    return out, truncated_any, False
                painted.add((rr, cc))
                out[rr, cc] = int(spec["frame_color"])
                advanced += 1
                changed = True
            if advanced < spec["length"]:
                truncated_any = True

    return out, truncated_any, changed


def _gen_attempt() -> np.ndarray | None:
    frame_count, h, w = _sample_count_and_size()
    bg = random.choice(PALETTE)
    palette = [color for color in PALETTE if color != bg]
    colors = random.sample(palette, 2 * frame_count)
    frame_colors = colors[:frame_count]
    arrow_colors = colors[frame_count:]
    directions = [random.choice(DIRS) for _ in range(frame_count)]
    lengths = [random.randint(*RAY_LEN_RANGE) for _ in range(frame_count)]
    want_truncation = random.random() < 0.4

    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    specs: list[dict] = []

    order = list(range(frame_count))
    random.shuffle(order)
    for idx in order:
        spec = _place_one_frame(
            grid,
            blocked,
            frame_colors[idx],
            arrow_colors[idx],
            directions[idx],
            lengths[idx],
            prefer_truncation=want_truncation and idx == order[0],
        )
        if spec is None:
            return None
        specs.append(spec)

    expected, truncated_any, changed = _expected_output(grid, bg, specs)
    if not changed:
        return None
    if want_truncation and not truncated_any:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
