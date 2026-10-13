import random
from collections import Counter, deque

import numpy as np


BG_COLOR = 0
GRID_H_RANGE = (16, 30)
GRID_W_RANGE = (16, 30)
GROUP_COUNT_CHOICES = (1, 2, 2, 3)
LAYER_RANGE = (2, 4)
INNER_H_RANGE_BY_GROUPS = {
    1: (1, 7),
    2: (1, 5),
    3: (1, 4),
}
INNER_W_RANGE_BY_GROUPS = {
    1: (1, 8),
    2: (1, 6),
    3: (1, 5),
}
PLACEMENT_MARGIN = 1
PLACEMENT_PAD = 2
GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 200


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_hollow_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False
    for r, c in coords:
        if r not in (r0, r1) and c not in (c0, c1):
            return False
    return True


def _collect_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for coords in _components(grid == color):
            if _is_hollow_frame(coords):
                frames.append(
                    {
                        "color": color,
                        "coords": coords,
                        "bbox": _bbox(coords),
                    }
                )
    return frames


def _group_frames(frames: list[dict]) -> list[list[dict]]:
    frame_map = {
        (int(frame["color"]), *map(int, frame["bbox"])): frame for frame in frames
    }
    innermost: list[dict] = []
    for frame in frames:
        color = int(frame["color"])
        r0, c0, r1, c1 = map(int, frame["bbox"])
        child_key = (color, r0 + 2, c0 + 2, r1 - 2, c1 - 2)
        if child_key not in frame_map:
            innermost.append(frame)

    groups: list[list[dict]] = []
    used: set[tuple[int, int, int, int, int]] = set()
    for frame in sorted(innermost, key=lambda item: item["bbox"]):
        key = (int(frame["color"]), *map(int, frame["bbox"]))
        if key in used:
            continue
        chain: list[dict] = []
        current = frame
        while current is not None:
            current_key = (int(current["color"]), *map(int, current["bbox"]))
            if current_key in used:
                break
            used.add(current_key)
            chain.append(current)
            color = int(current["color"])
            r0, c0, r1, c1 = map(int, current["bbox"])
            parent_key = (color, r0 - 2, c0 - 2, r1 + 2, c1 + 2)
            current = frame_map.get(parent_key)
        groups.append(chain)
    return groups


def _inner_marker_count(grid: np.ndarray, frame: dict, bg: int) -> int:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    patch = grid[r0 + 1 : r1, c0 + 1 : c1]
    return int(np.count_nonzero(patch != bg))


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the markers in the innermost cavity of each nested frame group and keep
    exactly that many frame layers from the inside outward.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    frames = _collect_frames(g, bg)
    groups = _group_frames(frames)

    out = np.full_like(g, bg)
    for group in groups:
        keep_layers = min(_inner_marker_count(g, group[0], bg), len(group))
        for frame in group[:keep_layers]:
            _paint_frame(out, frame["bbox"], int(frame["color"]))
    return out


def _sample_group_spec(rng: random.Random, n_groups: int) -> dict:
    layers = rng.randint(LAYER_RANGE[0], LAYER_RANGE[1])
    keep_layers = rng.randint(1, layers)

    inner_h_lo, inner_h_hi = INNER_H_RANGE_BY_GROUPS[n_groups]
    inner_w_lo, inner_w_hi = INNER_W_RANGE_BY_GROUPS[n_groups]
    while True:
        inner_h = rng.randint(inner_h_lo, inner_h_hi)
        inner_w = rng.randint(inner_w_lo, inner_w_hi)
        if inner_h * inner_w >= keep_layers:
            break

    outer_h = inner_h + 4 * layers - 2
    outer_w = inner_w + 4 * layers - 2
    return {
        "layers": layers,
        "keep_layers": keep_layers,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
    }


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    h_grid, w_grid = occupied.shape
    rr0 = max(0, r0 - PLACEMENT_PAD)
    cc0 = max(0, c0 - PLACEMENT_PAD)
    rr1 = min(h_grid, r0 + h + PLACEMENT_PAD)
    cc1 = min(w_grid, c0 + w + PLACEMENT_PAD)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_group(
    grid: np.ndarray,
    occupied: np.ndarray,
    spec: dict,
    frame_color: int,
    marker_color: int,
    rng: random.Random,
) -> dict | None:
    h_grid, w_grid = grid.shape
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    max_r0 = h_grid - outer_h - PLACEMENT_MARGIN
    max_c0 = w_grid - outer_w - PLACEMENT_MARGIN
    if max_r0 < PLACEMENT_MARGIN or max_c0 < PLACEMENT_MARGIN:
        return None

    for _ in range(PLACEMENT_RETRIES):
        r0 = rng.randint(PLACEMENT_MARGIN, max_r0)
        c0 = rng.randint(PLACEMENT_MARGIN, max_c0)
        if not _can_place(occupied, r0, c0, outer_h, outer_w):
            continue

        layers = int(spec["layers"])
        for layer in range(layers):
            inset = 2 * layer
            bbox = (
                r0 + inset,
                c0 + inset,
                r0 + outer_h - 1 - inset,
                c0 + outer_w - 1 - inset,
            )
            _paint_frame(grid, bbox, frame_color)

        inner_r0 = r0 + 2 * (layers - 1) + 1
        inner_c0 = c0 + 2 * (layers - 1) + 1
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        inner_cells = [
            (inner_r0 + dr, inner_c0 + dc)
            for dr in range(inner_h)
            for dc in range(inner_w)
        ]
        marker_cells = rng.sample(inner_cells, int(spec["keep_layers"]))
        for rr, cc in marker_cells:
            grid[rr, cc] = marker_color

        occupied[r0 : r0 + outer_h, c0 : c0 + outer_w] = True
        return {
            "bbox": (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1),
            "layers": layers,
            "keep_layers": int(spec["keep_layers"]),
        }
    return None


def _validate_generated(grid: np.ndarray, expected_groups: list[dict], bg: int) -> bool:
    frames = _collect_frames(grid, bg)
    groups = _group_frames(frames)
    if len(groups) != len(expected_groups):
        return False

    actual = []
    for group in groups:
        inner = group[0]
        actual.append(
            (
                tuple(map(int, inner["bbox"])),
                len(group),
                _inner_marker_count(grid, inner, bg),
            )
        )

    expected = []
    for spec in expected_groups:
        r0, c0, r1, c1 = map(int, spec["bbox"])
        layers = int(spec["layers"])
        inner_bbox = (
            r0 + 2 * (layers - 1),
            c0 + 2 * (layers - 1),
            r1 - 2 * (layers - 1),
            c1 - 2 * (layers - 1),
        )
        expected.append((inner_bbox, layers, int(spec["keep_layers"])))

    return sorted(actual) == sorted(expected)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_groups = rng.choice(GROUP_COUNT_CHOICES)
    grid_h = rng.randint(GRID_H_RANGE[0], GRID_H_RANGE[1])
    grid_w = rng.randint(GRID_W_RANGE[0], GRID_W_RANGE[1])

    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    frame_color = rng.choice([1, 3, 4, 5, 6, 7, 8, 9])
    marker_color = rng.choice([c for c in range(1, 10) if c != frame_color])

    expected_groups: list[dict] = []
    for _ in range(n_groups):
        spec = _sample_group_spec(rng, n_groups)
        placed = _place_group(grid, occupied, spec, frame_color, marker_color, rng)
        if placed is None:
            return None
        expected_groups.append(placed)

    if not _validate_generated(grid, expected_groups, BG_COLOR):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
