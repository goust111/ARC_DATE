import random
from collections import deque

import numpy as np


BG = 0
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SHIFT_BY_DIRECTION = {
    "up": (-2, 0),
    "down": (2, 0),
    "left": (0, -2),
    "right": (0, 2),
}
AXIS_ORDER = {"vertical": 0, "horizontal": 1}
DIRECTION_ORDER = {"left": 0, "right": 1, "up": 2, "down": 3}

GEN_GRID_MIN = 16
GEN_GRID_MAX = 28
GEN_OBJECT_MIN = 2
GEN_OBJECT_MAX = 5
GEN_OBJECT_RETRIES = 120
GEN_PLACEMENT_RETRIES = 160
GEN_ATTEMPTS = 300


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    bbox = _bbox_from_coords(coords)
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, bbox


def _find_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            cells = []

            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            coords = np.array(cells, dtype=int)
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                }
            )

    return objects


def _erode4(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return mask.copy()
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    return (
        mask
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )


def _defect_candidate(mask: np.ndarray, axis: str) -> dict | None:
    if axis == "horizontal":
        mirrored = np.flipud(mask)
        missing = mirrored & ~mask
        if not np.any(missing):
            return None
        rows = np.where(missing)[0]
        center = (mask.shape[0] - 1) / 2.0
        if np.all(rows < center):
            direction = "up"
        elif np.all(rows > center):
            direction = "down"
        else:
            return None
    else:
        mirrored = np.fliplr(mask)
        missing = mirrored & ~mask
        if not np.any(missing):
            return None
        cols = np.where(missing)[1]
        center = (mask.shape[1] - 1) / 2.0
        if np.all(cols < center):
            direction = "left"
        elif np.all(cols > center):
            direction = "right"
        else:
            return None

    completed = mask | missing
    if axis == "horizontal":
        if not np.array_equal(completed, np.flipud(completed)):
            return None
    else:
        if not np.array_equal(completed, np.fliplr(completed)):
            return None

    return {
        "axis": axis,
        "direction": direction,
        "missing": missing,
        "missing_count": int(missing.sum()),
        "completed": completed,
    }


def _analyze_mask(mask: np.ndarray) -> dict:
    horizontal_symmetric = bool(np.array_equal(mask, np.flipud(mask)))
    vertical_symmetric = bool(np.array_equal(mask, np.fliplr(mask)))

    candidates = []
    for axis in ("vertical", "horizontal"):
        candidate = _defect_candidate(mask, axis)
        if candidate is not None:
            candidates.append(candidate)

    return {
        "perfect": horizontal_symmetric or vertical_symmetric,
        "horizontal_symmetric": horizontal_symmetric,
        "vertical_symmetric": vertical_symmetric,
        "candidates": candidates,
    }


def _choose_defect_candidate(candidates: list[dict]) -> dict:
    assert candidates, "expected at least one defect-completion candidate"
    return sorted(
        candidates,
        key=lambda cand: (
            int(cand["missing_count"]),
            AXIS_ORDER[str(cand["axis"])],
            DIRECTION_ORDER[str(cand["direction"])],
        ),
    )[0]


def _transform_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int], dict]:
    analysis = _analyze_mask(mask)
    if bool(analysis["perfect"]):
        return _erode4(mask), (0, 0), analysis

    candidate = _choose_defect_candidate(list(analysis["candidates"]))
    return (
        np.array(candidate["completed"], copy=True),
        SHIFT_BY_DIRECTION[str(candidate["direction"])],
        analysis,
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Perfectly symmetric objects are eroded by one 4-neighbor layer.
    Objects with a one-sided symmetry defect are completed and shifted
    two cells toward the missing side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    out = np.zeros_like(g)

    for obj in _find_objects(g):
        mask, (r0, c0, _r1, _c1) = _mask_from_coords(obj["coords"])
        transformed, (dr, dc), _analysis = _transform_mask(mask)
        rows, cols = np.where(transformed)

        if rows.size == 0:
            continue

        abs_rows = rows + int(r0) + int(dr)
        abs_cols = cols + int(c0) + int(dc)

        assert int(abs_rows.min()) >= 0 and int(abs_rows.max()) < h
        assert int(abs_cols.min()) >= 0 and int(abs_cols.max()) < w
        assert not np.any(out[abs_rows, abs_cols] != BG)

        out[abs_rows, abs_cols] = int(obj["color"])

    return out


def _mask_is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False

    start = tuple(map(int, pts[0]))
    q = deque([start])
    seen = {start}

    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if not (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]):
                continue
            if not bool(mask[nr, nc]) or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))

    return len(seen) == int(mask.sum())


def _make_vertical_symmetric_mask(rng: random.Random) -> np.ndarray:
    for _ in range(GEN_OBJECT_RETRIES):
        height = rng.randint(4, 8)
        odd_axis = rng.random() < 0.55
        max_reach = rng.randint(1, 4)
        min_reach = 0 if odd_axis else 1

        reaches = []
        current = rng.randint(min_reach, max_reach)
        for row in range(height):
            if row > 0:
                low = max(min_reach, current - 1)
                high = min(max_reach, current + 1)
                current = rng.randint(low, high)
            reaches.append(current)

        reaches[rng.randrange(height)] = max_reach
        if len(set(reaches)) == 1 and max_reach > min_reach:
            idx = rng.randrange(height)
            if reaches[idx] == max_reach:
                reaches[idx] = max(min_reach, reaches[idx] - 1)
            else:
                reaches[idx] = min(max_reach, reaches[idx] + 1)
            reaches[rng.randrange(height)] = max_reach

        width = 2 * max_reach + (1 if odd_axis else 0)
        mask = np.zeros((height, width), dtype=bool)
        for r, reach in enumerate(reaches):
            if odd_axis:
                left = max_reach - reach
                right = max_reach + reach
            else:
                left = max_reach - reach
                right = max_reach + reach - 1
            mask[r, left : right + 1] = True

        if int(mask.sum()) < 6:
            continue
        if not _mask_is_connected(mask):
            continue
        return mask

    raise RuntimeError("failed to build a symmetric mask")


def _carve_vertical_defect(mask: np.ndarray, side: str, rng: random.Random) -> np.ndarray | None:
    work = np.array(mask, copy=True)
    h, w = work.shape
    edge_col = 0 if side == "left" else w - 1

    candidates = []
    edge_rows = []
    for r in range(h):
        cols = np.flatnonzero(work[r])
        if cols.size <= 1:
            continue
        candidates.append(int(r))
        if (side == "left" and int(cols[0]) == edge_col) or (
            side == "right" and int(cols[-1]) == edge_col
        ):
            edge_rows.append(int(r))

    if not candidates:
        return None

    protected = rng.choice(edge_rows) if edge_rows else None
    eligible = [r for r in sorted(candidates) if r != protected]
    if not eligible:
        return None

    band_len = rng.randint(1, min(3, len(eligible)))
    start = rng.randint(0, len(eligible) - band_len)
    selected = eligible[start : start + band_len]
    changed = False

    for r in selected:
        cols = np.flatnonzero(work[r])
        max_trim = min(2, int(cols.size) - 1)
        if max_trim <= 0:
            continue
        trim = rng.randint(1, max_trim)
        if side == "left":
            work[r, cols[:trim]] = False
        else:
            work[r, cols[-trim:]] = False
        changed = True

    if not changed:
        return None
    if side == "left" and not np.any(work[:, 0]):
        return None
    if side == "right" and not np.any(work[:, -1]):
        return None
    if not _mask_is_connected(work):
        return None
    return work


def _make_object_spec(kind: str, rng: random.Random) -> dict | None:
    for _ in range(GEN_OBJECT_RETRIES):
        axis = rng.choice(("vertical", "horizontal"))
        base = _make_vertical_symmetric_mask(rng)
        full_mask = base if axis == "vertical" else base.T

        expected_direction = None
        if kind == "perfect":
            input_mask = full_mask
        else:
            expected_direction = rng.choice(
                ("left", "right") if axis == "vertical" else ("up", "down")
            )
            if axis == "vertical":
                input_mask = _carve_vertical_defect(full_mask, expected_direction, rng)
            else:
                temp_side = "left" if expected_direction == "up" else "right"
                temp = _carve_vertical_defect(full_mask.T, temp_side, rng)
                input_mask = None if temp is None else temp.T

        if input_mask is None or not _mask_is_connected(input_mask):
            continue

        transformed, shift, analysis = _transform_mask(input_mask)
        if kind == "perfect":
            if not bool(analysis["perfect"]):
                continue
        else:
            if bool(analysis["perfect"]):
                continue
            candidates = list(analysis["candidates"])
            best = _choose_defect_candidate(candidates)
            if str(best["direction"]) != str(expected_direction):
                continue
            best_count = int(best["missing_count"])
            if sum(int(cand["missing_count"]) == best_count for cand in candidates) != 1:
                continue

        input_coords = [tuple(map(int, rc)) for rc in np.argwhere(input_mask)]
        output_coords = [
            (int(r + shift[0]), int(c + shift[1]))
            for r, c in np.argwhere(transformed)
        ]

        return {
            "kind": kind,
            "axis": axis,
            "color": None,
            "input_mask": input_mask,
            "input_coords": input_coords,
            "output_coords": output_coords,
        }

    return None


def _buffered(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    buffered = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                buffered.add((int(r + dr), int(c + dc)))
    return buffered


def _place_specs(specs: list[dict], h: int, w: int, rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    input_blocked: set[tuple[int, int]] = set()
    output_blocked: set[tuple[int, int]] = set()

    ordered = sorted(
        specs,
        key=lambda spec: (
            -len(spec["input_coords"]),
            -len(spec["output_coords"]),
        ),
    )

    for spec in ordered:
        mask_h, mask_w = spec["input_mask"].shape
        output_coords = list(spec["output_coords"])

        out_min_r = min((r for r, _ in output_coords), default=0)
        out_max_r = max((r for r, _ in output_coords), default=mask_h - 1)
        out_min_c = min((c for _, c in output_coords), default=0)
        out_max_c = max((c for _, c in output_coords), default=mask_w - 1)

        base_r_min = max(0, -out_min_r)
        base_r_max = min(h - mask_h, h - 1 - out_max_r)
        base_c_min = max(0, -out_min_c)
        base_c_max = min(w - mask_w, w - 1 - out_max_c)

        if base_r_min > base_r_max or base_c_min > base_c_max:
            return None

        positions = [
            (r, c)
            for r in range(base_r_min, base_r_max + 1)
            for c in range(base_c_min, base_c_max + 1)
        ]
        rng.shuffle(positions)

        placed = False
        for base_r, base_c in positions[:GEN_PLACEMENT_RETRIES]:
            abs_input = {
                (int(base_r + r), int(base_c + c)) for r, c in spec["input_coords"]
            }
            abs_output = {
                (int(base_r + r), int(base_c + c)) for r, c in spec["output_coords"]
            }

            if abs_input & input_blocked:
                continue
            if abs_output & output_blocked:
                continue

            if any(grid[r, c] != BG for r, c in abs_input):
                continue

            for r, c in abs_input:
                grid[r, c] = int(spec["color"])

            input_blocked.update(_buffered(abs_input))
            output_blocked.update(_buffered(abs_output))
            placed = True
            break

        if not placed:
            return None

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    obj_count = rng.randint(GEN_OBJECT_MIN, GEN_OBJECT_MAX)
    kinds = [
        "perfect" if rng.random() < 0.5 else "defective" for _ in range(obj_count)
    ]
    if all(kind == "perfect" for kind in kinds):
        kinds[rng.randrange(obj_count)] = "defective"
    if all(kind == "defective" for kind in kinds):
        kinds[rng.randrange(obj_count)] = "perfect"

    specs = []
    for kind in kinds:
        spec = _make_object_spec(kind, rng)
        if spec is None:
            return None
        specs.append(spec)

    colors = rng.sample(list(range(1, 10)), k=obj_count)
    for spec, color in zip(specs, colors):
        spec["color"] = int(color)

    h = rng.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    w = rng.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    grid = _place_specs(specs, h, w, rng)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if len(_find_objects(grid)) != obj_count:
        return None
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    """
    Build a black-background scene with several well-separated monochrome objects.
    Each sample includes both perfect and defective symmetry cases.
    """
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
