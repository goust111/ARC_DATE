from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3

OBJECT_COLORS = (4, 5, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 400

MARKER_BY_KIND = {
    "solid": RED,
    "partial": GREEN,
    "hollow": BLUE,
}


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    return components


def _is_hollow_frame(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    if not np.all(mask[0, :]) or not np.all(mask[-1, :]):
        return False
    if not np.all(mask[:, 0]) or not np.all(mask[:, -1]):
        return False
    return not np.any(mask[1:-1, 1:-1])


def _kind_from_mask(mask: np.ndarray) -> str:
    if np.all(mask):
        return "solid"
    if _is_hollow_frame(mask):
        return "hollow"
    return "partial"


def _component_kind(grid: np.ndarray, component: dict) -> str:
    r0, c0, r1, c1 = component["bbox"]
    color = int(component["color"])
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = patch == color
    assert np.all(mask | (patch == BG)), "expected each bbox to contain only object cells and background"
    return _kind_from_mask(mask)


def _marker_job(component: dict, kind: str) -> tuple[int, int, int]:
    r0, c0, r1, c1 = component["bbox"]
    if kind == "solid":
        return int(r1) + 1, int(c1) + 1, RED
    if kind == "hollow":
        return int(r0) - 1, int(c0) - 1, BLUE
    assert kind == "partial"
    return int(r0) - 1, int(c1) + 1, GREEN


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark each object outside its bounding box according to its hollowness.

    Solid objects get a red marker at the outside bottom-right corner.
    Fully hollow rectangular frames get a blue marker at the outside top-left corner.
    All other non-solid objects get a green marker at the outside top-right corner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    h, w = g.shape
    components = _find_components(g)
    assert components, "expected at least one foreground object"

    claimed: set[tuple[int, int]] = set()
    for component in components:
        kind = _component_kind(g, component)
        mr, mc, color = _marker_job(component, kind)
        assert 0 <= mr < h and 0 <= mc < w, "marker must stay in bounds"
        assert int(g[mr, mc]) == BG, "expected marker cells to be empty in the input"
        assert (mr, mc) not in claimed, "expected unique marker locations"
        claimed.add((mr, mc))
        out[mr, mc] = color

    return out


def _solid_patch(color: int) -> np.ndarray:
    h = random.randint(1, 5)
    w = random.randint(1, 5)
    return np.full((h, w), color, dtype=int)


def _hollow_patch(color: int) -> np.ndarray:
    h = random.randint(3, 7)
    w = random.randint(3, 7)
    patch = np.full((h, w), BG, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _partial_holes_patch(color: int) -> np.ndarray:
    h = random.randint(4, 7)
    w = random.randint(4, 7)
    patch = np.full((h, w), color, dtype=int)
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    max_holes = len(interior) - 1
    hole_cap = max(1, min(max_holes, len(interior) // 2 + 1))
    for r, c in random.sample(interior, random.randint(1, hole_cap)):
        patch[r, c] = BG
    return patch


def _partial_bite_patch(color: int) -> np.ndarray:
    h = random.randint(4, 7)
    w = random.randint(4, 7)
    patch = np.full((h, w), color, dtype=int)
    side = random.choice(("top", "bottom", "left", "right"))

    if side in ("top", "bottom"):
        depth = random.randint(1, h - 2)
        span = random.randint(1, w - 2)
        start = random.randint(0, w - span)
        rows = slice(0, depth) if side == "top" else slice(h - depth, h)
        patch[rows, start : start + span] = BG
    else:
        depth = random.randint(1, w - 2)
        span = random.randint(1, h - 2)
        start = random.randint(0, h - span)
        cols = slice(0, depth) if side == "left" else slice(w - depth, w)
        patch[start : start + span, cols] = BG

    return patch


def _partial_patch(color: int) -> np.ndarray:
    for _ in range(40):
        mode = random.choice(("holes", "bite", "mixed"))
        if mode == "holes":
            patch = _partial_holes_patch(color)
        elif mode == "bite":
            patch = _partial_bite_patch(color)
        else:
            patch = _partial_bite_patch(color)
            h, w = patch.shape
            interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1) if patch[r, c] == color]
            if interior:
                rr, cc = random.choice(interior)
                patch[rr, cc] = BG

        if _kind_from_mask(patch == color) == "partial":
            return patch
    raise RuntimeError("failed to build a partial-hollow object")


def _make_patch(kind: str, color: int) -> np.ndarray:
    if kind == "solid":
        return _solid_patch(color)
    if kind == "hollow":
        return _hollow_patch(color)
    assert kind == "partial"
    return _partial_patch(color)


def _sample_kinds(n_objects: int) -> list[str]:
    kinds = ["solid", "partial", "hollow"]
    while len(kinds) < n_objects:
        kinds.append(random.choice(("solid", "partial", "hollow")))
    random.shuffle(kinds)
    return kinds


def _stamp_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != BG
    window = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    window[mask] = patch[mask]


def _try_place(height: int, width: int, objects: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.full((height, width), BG, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)
    placements: list[dict] = []

    for obj in sorted(objects, key=lambda item: item["patch"].shape[0] * item["patch"].shape[1], reverse=True):
        patch = obj["patch"]
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, height - ph):
            for c0 in range(1, width - pw):
                if np.any(reserved[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1]):
                    continue
                candidates.append((r0, c0))

        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        _stamp_patch(grid, patch, r0, c0)
        reserved[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1] = True
        placements.append(
            {
                "kind": obj["kind"],
                "color": obj["color"],
                "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
            }
        )

    return grid, placements


def _expected_output(grid: np.ndarray, placements: list[dict]) -> np.ndarray:
    out = grid.copy()
    used: set[tuple[int, int]] = set()
    h, w = grid.shape

    for placement in placements:
        mr, mc, color = _marker_job(placement, placement["kind"])
        if not (0 <= mr < h and 0 <= mc < w):
            raise AssertionError("planned marker went out of bounds")
        if int(grid[mr, mc]) != BG or (mr, mc) in used:
            raise AssertionError("planned marker placement collided")
        used.add((mr, mc))
        out[mr, mc] = color

    return out


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(3, 5)
    min_side = 12 + 2 * max(0, n_objects - 3)
    height = random.randint(min_side, min_side + 8)
    width = random.randint(min_side, min_side + 8)

    kinds = _sample_kinds(n_objects)
    colors = random.sample(OBJECT_COLORS, n_objects)
    objects = [{"kind": kind, "color": color, "patch": _make_patch(kind, color)} for kind, color in zip(kinds, colors)]

    placed = _try_place(height, width, objects)
    if placed is None:
        return None
    grid, placements = placed

    components = _find_components(grid)
    if len(components) != n_objects:
        return None

    expected = _expected_output(grid, placements)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
