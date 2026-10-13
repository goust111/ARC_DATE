from __future__ import annotations

from collections import deque
import random

import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    max_count = int(counts.max())
    candidates = values[counts == max_count]
    assert len(candidates) == 1, "background color must be uniquely dominant"
    return int(candidates[0])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(len(coords)),
                }
            )

    return components


def _solid_rectangle(component: dict, grid: np.ndarray) -> tuple[int, int, int, int, int]:
    r0, c0, r1, c1 = component["bbox"]
    color = int(component["color"])
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = sub.shape
    assert int(component["area"]) == h * w, "each object must fill its bbox"
    assert bool(np.all(sub == color)), "each object must be a solid rectangle"
    return r0, c0, h, w, color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each solid rectangle according to its aspect ratio.

    Wide rectangles collapse into their leftmost column, tall rectangles collapse
    into their topmost row, and squares remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _dominant_color(g)
    components = _find_components(g, bg)
    assert components, "expected at least one foreground rectangle"

    out = np.full_like(g, bg)
    for component in components:
        r0, c0, h, w, color = _solid_rectangle(component, g)
        if w > h:
            out[r0 : r0 + h, c0] = color
        elif h > w:
            out[r0, c0 : c0 + w] = color
        else:
            out[r0 : r0 + h, c0 : c0 + w] = color

    return out


# --------------- generator ---------------

BG_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
OBJECT_COUNT_MIN = 3
OBJECT_COUNT_MAX = 6
MIN_GRID_SIDE = 12
MAX_GRID_SIDE = 26
PLACEMENT_GAP = 1
MAX_ATTEMPTS = 400


def _sample_kind_sequence() -> list[str]:
    n_objects = random.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX)
    kinds = ["wide", "tall", "square"]
    for _ in range(n_objects - 3):
        kinds.append(random.choice(["wide", "tall", "square"]))
    random.shuffle(kinds)
    return kinds


def _sample_dims(kind: str) -> tuple[int, int]:
    if kind == "wide":
        height = random.randint(2, 4)
        width = random.randint(height + 1, min(8, height + 4))
        return height, width
    if kind == "tall":
        width = random.randint(2, 4)
        height = random.randint(width + 1, min(8, width + 4))
        return height, width
    side = random.randint(2, 5)
    return side, side


def _choose_grid_shape(specs: list[dict]) -> tuple[int, int] | None:
    total_area = sum(int(spec["h"]) * int(spec["w"]) for spec in specs)
    min_h = max(MIN_GRID_SIDE, max(int(spec["h"]) for spec in specs) + 2)
    min_w = max(MIN_GRID_SIDE, max(int(spec["w"]) for spec in specs) + 2)

    candidates = [
        (h, w)
        for h in range(min_h, MAX_GRID_SIDE + 1)
        for w in range(min_w, MAX_GRID_SIDE + 1)
        if h * w >= total_area * 3
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def _separated(r0: int, c0: int, h: int, w: int, placed: list[dict]) -> bool:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    for obj in placed:
        or0 = int(obj["r0"])
        oc0 = int(obj["c0"])
        or1 = or0 + int(obj["h"]) - 1
        oc1 = oc0 + int(obj["w"]) - 1
        if not (
            r1 + PLACEMENT_GAP < or0
            or or1 + PLACEMENT_GAP < r0
            or c1 + PLACEMENT_GAP < oc0
            or oc1 + PLACEMENT_GAP < c0
        ):
            return False
    return True


def _place_objects(specs: list[dict], grid_h: int, grid_w: int) -> list[dict] | None:
    placed: list[dict] = []
    for spec in sorted(specs, key=lambda item: int(item["h"]) * int(item["w"]), reverse=True):
        h = int(spec["h"])
        w = int(spec["w"])
        positions = [(r, c) for r in range(grid_h - h + 1) for c in range(grid_w - w + 1)]
        random.shuffle(positions)

        chosen = None
        for r0, c0 in positions:
            if _separated(r0, c0, h, w, placed):
                chosen = (r0, c0)
                break
        if chosen is None:
            return None

        r0, c0 = chosen
        placed.append(
            {
                "r0": r0,
                "c0": c0,
                "h": h,
                "w": w,
                "color": int(spec["color"]),
            }
        )
    return placed


def _render_input(grid_h: int, grid_w: int, objects: list[dict]) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    for obj in objects:
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        h = int(obj["h"])
        w = int(obj["w"])
        grid[r0 : r0 + h, c0 : c0 + w] = int(obj["color"])
    return grid


def _render_output(grid_h: int, grid_w: int, objects: list[dict]) -> np.ndarray:
    out = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    for obj in objects:
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        h = int(obj["h"])
        w = int(obj["w"])
        color = int(obj["color"])
        if w > h:
            out[r0 : r0 + h, c0] = color
        elif h > w:
            out[r0, c0 : c0 + w] = color
        else:
            out[r0 : r0 + h, c0 : c0 + w] = color
    return out


def _has_unique_background(grid: np.ndarray) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    idx = int(np.argmax(counts))
    return int(values[idx]) == BG_COLOR and int((counts == counts[idx]).sum()) == 1


def _gen_attempt() -> np.ndarray | None:
    kinds = _sample_kind_sequence()
    colors = random.sample(PALETTE, len(kinds))
    specs = []
    for kind, color in zip(kinds, colors):
        h, w = _sample_dims(kind)
        specs.append({"h": h, "w": w, "color": color})

    shape = _choose_grid_shape(specs)
    if shape is None:
        return None
    grid_h, grid_w = shape

    objects = _place_objects(specs, grid_h, grid_w)
    if objects is None:
        return None

    inp = _render_input(grid_h, grid_w, objects)
    if not _has_unique_background(inp):
        return None

    expected = _render_output(grid_h, grid_w, objects)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
