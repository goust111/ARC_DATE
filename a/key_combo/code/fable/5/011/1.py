from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


RED = 2
_KINDS = ("H", "V", "D1", "D2")


def _background(grid: np.ndarray) -> int:
    """Return the dominant color, following the shared ARC toolbox convention."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N, 2) arrays of (row, column)."""
    h, w = mask.shape
    unseen = mask.astype(bool).copy()
    result: list[np.ndarray] = []
    for sr, sc in np.argwhere(unseen):
        sr, sc = int(sr), int(sc)
        if not unseen[sr, sc]:
            continue
        unseen[sr, sc] = False
        queue = deque([(sr, sc)])
        cells: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < h and 0 <= cc < w and unseen[rr, cc]:
                    unseen[rr, cc] = False
                    queue.append((rr, cc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(coords: np.ndarray) -> tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, r0, c0


def _reflect(mask: np.ndarray, kind: str) -> np.ndarray:
    if kind == "H":
        return mask[::-1, :]
    if kind == "V":
        return mask[:, ::-1]
    if kind == "D1":
        return mask.T
    if kind == "D2":
        return np.rot90(mask, 2).T
    raise ValueError(f"unknown reflection kind: {kind}")


def _axes_of(mask: np.ndarray) -> list[str]:
    axes = []
    for kind in _KINDS:
        if kind.startswith("D") and mask.shape[0] != mask.shape[1]:
            continue
        if np.array_equal(mask, _reflect(mask, kind)):
            axes.append(kind)
    return axes


def _side_values(coords: np.ndarray, shape: tuple[int, int], kind: str) -> np.ndarray:
    """Signed side of an axis, in the object's own cropped coordinate system."""
    h, w = shape
    r = coords[:, 0]
    c = coords[:, 1]
    if kind == "H":
        return 2 * r - (h - 1)
    if kind == "V":
        return 2 * c - (w - 1)
    if kind == "D1":
        return r - c
    return r + c - (h - 1)


def _direction(kind: str, side: int) -> tuple[int, int]:
    if kind == "H":
        return (side, 0)
    if kind == "V":
        return (0, side)
    if kind == "D1":
        return (side, -side)
    return (side, side)


def _analyze_mask(mask: np.ndarray) -> dict:
    """Infer the unique completion axis, missing side, completed mask and motion."""
    candidates: list[dict] = []
    for kind in _KINDS:
        if kind.startswith("D") and mask.shape[0] != mask.shape[1]:
            continue
        mirrored = _reflect(mask, kind)
        missing = mirrored & ~mask
        if not missing.any():
            continue
        completed = mask | mirrored
        # The repaired object itself must have exactly one reflection axis.
        if _axes_of(completed) != [kind]:
            continue
        if len(_components(completed)) != 1:
            continue
        missing_coords = np.argwhere(missing)
        values = _side_values(missing_coords, mask.shape, kind)
        if np.all(values > 0):
            side = 1
        elif np.all(values < 0):
            side = -1
        else:
            continue
        candidates.append(
            {
                "kind": kind,
                "side": side,
                "direction": _direction(kind, side),
                "completed": completed,
                "missing": missing,
            }
        )
    assert len(candidates) == 1, "each incomplete object must have one unique one-sided reflection completion"
    return candidates[0]


def _edge_distance(coords: set[tuple[int, int]], direction: tuple[int, int], shape: tuple[int, int]) -> int:
    h, w = shape
    dr, dc = direction
    limits = []
    for r, c in coords:
        row_limit = h - 1 - r if dr > 0 else r if dr < 0 else max(h, w)
        col_limit = w - 1 - c if dc > 0 else c if dc < 0 else max(h, w)
        limits.append(min(row_limit, col_limit))
    return int(min(limits))


def _shift(coords: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in coords}


def _settle(
    objects: list[dict],
    shape: tuple[int, int],
    order: list[int] | None = None,
    *,
    require_unique_order: bool = True,
) -> tuple[list[set[tuple[int, int]]], list[str], list[int]]:
    h, w = shape
    distances = [
        _edge_distance(obj["coords"], obj["direction"], shape)
        for obj in objects
    ]
    if require_unique_order:
        assert len(set(distances)) == len(distances), "edge distances must uniquely determine settling order"
    if order is None:
        order = sorted(range(len(objects)), key=lambda i: distances[i])

    occupied: set[tuple[int, int]] = set()
    final: list[set[tuple[int, int]]] = [set() for _ in objects]
    reasons = [""] * len(objects)
    steps = [0] * len(objects)
    for index in order:
        current = set(objects[index]["coords"])
        assert not (current & occupied), "a settled object may not cover an object that has not begun sliding"
        dr, dc = objects[index]["direction"]
        while True:
            nxt = _shift(current, dr, dc)
            if any(not (0 <= r < h and 0 <= c < w) for r, c in nxt):
                reason = "edge"
                break
            if nxt & occupied:
                reason = "object"
                break
            current = nxt
            steps[index] += 1
        final[index] = current
        reasons[index] = reason
        occupied |= current
    return final, reasons, steps


def _parse_objects(grid: np.ndarray, bg: int) -> tuple[list[dict], list[tuple[int, int]]]:
    components = _components(grid == RED)
    objects: list[dict] = []
    noise: list[tuple[int, int]] = []
    for coords in components:
        if len(coords) == 1:
            noise.append(tuple(map(int, coords[0])))
            continue
        mask, r0, c0 = _crop_mask(coords)
        info = _analyze_mask(mask)
        local = np.argwhere(info["completed"])
        info = dict(info)
        info["coords"] = {(int(r + r0), int(c + c0)) for r, c in local}
        info["input_coords"] = {tuple(map(int, p)) for p in coords}
        info["origin"] = (r0, c0)
        objects.append(info)
    return objects, noise


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each one-sided incomplete red reflection object, then slide it toward its missing side until it reaches an edge or a settled object.

    Input structure:
    - A dominant background surrounds several 4-connected red objects and isolated
      red singleton noise pixels.
    - Each non-singleton is missing pixels from exactly one side of exactly one
      horizontal, vertical, main-diagonal, or anti-diagonal reflection axis.

    Rule:
    - Erase singleton noise and repair every object by reflecting its present half
      across its unique axis. The missing side selects the perpendicular cardinal
      or diagonal motion direction.
    - Settle objects in increasing distance to their forward grid edge. Move each
      repaired object one cell at a time until its next step leaves the grid or
      overlaps an already settled object; erase all original positions.

    Key insight:
    - The absent pixels are both the reflected content to restore and a directional
      code. They cannot be treated like the true noise singletons, and each object
      has its own axis-relative direction rather than sharing one gravity vector.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 0 < g.shape[0] <= 30 and 0 < g.shape[1] <= 30
    bg = _background(g)
    assert set(map(int, np.unique(g))) <= {bg, RED}, "only background and red are expected"
    objects, _noise = _parse_objects(g, bg)
    assert objects, "expected at least one non-singleton incomplete object"

    final, _reasons, _steps = _settle(objects, g.shape)
    out = np.full_like(g, bg)
    for cells in final:
        for r, c in cells:
            out[r, c] = RED
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask)) == 1


def _tight(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _horizontal_complete() -> np.ndarray:
    for _ in range(100):
        h = random.choice((5, 7))
        w = random.choice([n for n in range(5, 9) if n != h])
        mid = h // 2
        depths = [random.randint(0, mid) for _ in range(w)]
        # Two full-depth columns keep the bbox stable after deleting a tip.
        for c in random.sample(range(w), 2):
            depths[c] = mid
        mask = np.zeros((h, w), dtype=bool)
        for c, depth in enumerate(depths):
            mask[mid - depth : mid + depth + 1, c] = True
        if _axes_of(mask) == ["H"]:
            return mask
    raise RuntimeError("failed to make a horizontal object")


def _diagonal_complete() -> np.ndarray:
    for _ in range(200):
        n = random.choice((5, 6, 7))
        mask = np.zeros((n, n), dtype=bool)
        # A one-cell band makes the diagonal scaffold 4-connected.
        for r in range(n):
            for c in range(n):
                if abs(r - c) <= 1:
                    mask[r, c] = True
        # Symmetric paired protrusions make the silhouette non-degenerate.
        for _j in range(random.randint(2, n + 1)):
            r = random.randrange(n)
            c = random.randrange(n)
            if abs(r - c) in (2, 3):
                mask[r, c] = True
                mask[c, r] = True
        if _connected(mask) and _tight(mask) and _axes_of(mask) == ["D1"]:
            return mask
    raise RuntimeError("failed to make a diagonal object")


def _make_object(kind: str, forced_side: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (incomplete, complete, inferred metadata) for the requested axis."""
    for _ in range(300):
        if kind == "H":
            complete = _horizontal_complete()
        elif kind == "V":
            complete = _horizontal_complete().T
        elif kind == "D1":
            complete = _diagonal_complete()
        else:
            complete = np.fliplr(_diagonal_complete())

        side = random.choice((-1, 1)) if forced_side is None else int(forced_side)
        assert side in (-1, 1)
        coords = np.argwhere(complete)
        values = _side_values(coords, complete.shape, kind)
        removable = coords[values * side > 0]
        if len(removable) < 2:
            continue
        max_missing = min(3, max(1, len(removable) // 3))
        k = random.randint(1, max_missing)
        for _j in range(80):
            chosen = removable[np.random.choice(len(removable), size=k, replace=False)]
            incomplete = complete.copy()
            incomplete[chosen[:, 0], chosen[:, 1]] = False
            if not (_connected(incomplete) and _tight(incomplete)):
                continue
            try:
                info = _analyze_mask(incomplete)
            except AssertionError:
                continue
            if (
                info["kind"] == kind
                and info["side"] == side
                and np.array_equal(info["completed"], complete)
                and int(info["missing"].sum()) == k
            ):
                return incomplete, complete, info
    raise RuntimeError(f"failed to make a uniquely incomplete {kind} object")


def _expanded_overlap(
    occupied: set[tuple[int, int]],
    cells: set[tuple[int, int]],
    padding: int,
) -> bool:
    if not occupied:
        return False
    expanded = {
        (r + dr, c + dc)
        for r, c in occupied
        for dr in range(-padding, padding + 1)
        for dc in range(-padding, padding + 1)
    }
    return bool(expanded & cells)


def _place_objects(
    shape: tuple[int, int],
    specs: list[tuple[np.ndarray, np.ndarray, dict]],
) -> list[dict] | None:
    h, w = shape
    occupied: set[tuple[int, int]] = set()
    placed: list[dict] = []
    for incomplete, complete, info in sorted(specs, key=lambda x: int(x[1].sum()), reverse=True):
        ph, pw = complete.shape
        positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        random.shuffle(positions)
        chosen = None
        local_complete = np.argwhere(complete)
        for r0, c0 in positions:
            cells = {(int(r0 + r), int(c0 + c)) for r, c in local_complete}
            if not _expanded_overlap(occupied, cells, padding=2):
                chosen = (r0, c0, cells)
                break
        if chosen is None:
            return None
        r0, c0, cells = chosen
        occupied |= cells
        placed.append(
            {
                "kind": info["kind"],
                "direction": info["direction"],
                "side": info["side"],
                "coords": cells,
                "input_coords": {
                    (int(r0 + r), int(c0 + c)) for r, c in np.argwhere(incomplete)
                },
                "missing_count": int(info["missing"].sum()),
            }
        )
    return placed


def _render_union(shape: tuple[int, int], bg: int, groups: list[set[tuple[int, int]]]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for cells in groups:
        for r, c in cells:
            out[r, c] = RED
    return out


def _scatter_noise(
    grid: np.ndarray,
    forbidden_input: set[tuple[int, int]],
    forbidden_output: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    noise: list[tuple[int, int]] = []
    candidates = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(candidates)
    for r, c in candidates:
        if (r, c) in forbidden_output:
            continue
        if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in forbidden_input):
            continue
        if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in noise):
            continue
        grid[r, c] = RED
        noise.append((r, c))
        if len(noise) == count:
            return noise
    return None


def _semantic_checks(grid: np.ndarray, intended: list[dict], noise: list[tuple[int, int]]) -> bool:
    bg = _background(grid)
    if int(np.count_nonzero(grid == bg)) <= grid.size // 2:
        return False
    try:
        parsed, parsed_noise = _parse_objects(grid, bg)
        n_objects = len(intended)
        if len(parsed) != n_objects or len(parsed_noise) != len(noise):
            return False
        if {obj["kind"] for obj in parsed} != set(_KINDS):
            return False
        if len({obj["direction"] for obj in parsed}) != n_objects:
            return False
        final, reasons, steps = _settle(parsed, grid.shape)
        if "edge" not in reasons or "object" not in reasons or min(steps) < 2:
            return False
        actual = main(grid)
    except (AssertionError, RuntimeError, ValueError):
        return False

    final_union = set().union(*final)
    if any(cell in final_union for cell in noise):
        return False
    if np.array_equal(actual, grid):
        return False

    # Every inferred completion adds visible pixels at the destination.
    for obj, cells in zip(parsed, final):
        if not (obj["coords"] - obj["input_coords"]):
            return False
        if not cells <= final_union:
            return False

    # Prove that completion cannot be replaced by sliding the damaged masks.
    damaged = [dict(obj, coords=set(obj["input_coords"])) for obj in parsed]
    try:
        damaged_final, _r, _s = _settle(damaged, grid.shape)
    except AssertionError:
        damaged_final = []
    if damaged_final and set().union(*damaged_final) == final_union:
        return False

    # Prove that the prescribed near-edge-first ordering affects this scene.
    distances = [_edge_distance(obj["coords"], obj["direction"], grid.shape) for obj in parsed]
    forward = sorted(range(len(parsed)), key=lambda i: distances[i])
    try:
        reverse_final, _r, _s = _settle(
            parsed, grid.shape, list(reversed(forward)), require_unique_order=False
        )
    except AssertionError:
        reverse_final = []
    if reverse_final and set().union(*reverse_final) == final_union:
        return False

    # Intended and independently parsed object metadata must agree as multisets.
    intended_sig = sorted((x["kind"], x["direction"], len(x["coords"])) for x in intended)
    parsed_sig = sorted((x["kind"], x["direction"], len(x["coords"])) for x in parsed)
    return intended_sig == parsed_sig


def _generation_attempt() -> np.ndarray | None:
    n_objects = random.randint(4, 6)
    min_side = 23 + 2 * (n_objects - 4)
    h = random.randint(min_side, 30)
    w = random.randint(min_side, 30)
    bg = random.choice([0, 1, 3, 4, 5, 6, 7, 8, 9])
    try:
        # Start with all four axis types, then add at most one object per axis.
        # Its opposite missing side supplies the other direction for that axis.
        base = [_make_object(kind) for kind in _KINDS]
        sides = {spec[2]["kind"]: spec[2]["side"] for spec in base}
        extra_kinds = random.sample(list(_KINDS), n_objects - 4)
        extras = [_make_object(kind, -sides[kind]) for kind in extra_kinds]
        specs = base + extras
    except RuntimeError:
        return None
    placed = _place_objects((h, w), specs)
    if placed is None:
        return None

    try:
        final, reasons, steps = _settle(placed, (h, w))
    except AssertionError:
        return None
    if "edge" not in reasons or "object" not in reasons or min(steps) < 2:
        return None

    grid = np.full((h, w), bg, dtype=int)
    input_cells = set().union(*(obj["input_coords"] for obj in placed))
    for r, c in input_cells:
        grid[r, c] = RED
    final_cells = set().union(*final)
    noise_count = random.randint(2, 5)
    noise = _scatter_noise(grid, input_cells, final_cells, noise_count)
    if noise is None:
        return None
    if not _semantic_checks(grid, placed, noise):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate instance covering every axis and both stop causes."""
    for _ in range(1200):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid incomplete-symmetry sliding puzzle")
