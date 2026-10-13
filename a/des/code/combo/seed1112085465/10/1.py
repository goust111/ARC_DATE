import random
from collections import deque

import numpy as np


BG = 0
FRAME_COLOR = 5
LABEL_COLORS = tuple(c for c in range(10) if c not in {BG, FRAME_COLOR})

GENERATION_ATTEMPTS = 400
GRID_MIN = 20
GRID_MAX = 30
CONTAINER_COUNT_RANGE = (2, 4)
FRAME_SIDE_RANGE = (4, 8)
PLACEMENT_ATTEMPTS = 250
GAP_ATTEMPTS = 80

_RNG = random.Random()


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _normalize(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.asarray(coords, dtype=int)
    mins = coords.min(axis=0)
    shifted = coords - mins
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _shape_to_patch(shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    height = max(r for r, _ in shape) + 1
    width = max(c for _, c in shape) + 1
    patch = np.full((height, width), BG, dtype=int)
    for r, c in shape:
        patch[r, c] = int(color)
    return patch


def _expand_bbox(
    bbox: tuple[int, int, int, int], pad: int, h: int, w: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(h - 1, r1 + pad),
        min(w - 1, c1 + pad),
    )


def _mark(blocked: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _region_free(blocked: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return not np.any(blocked[r0 : r1 + 1, c0 : c1 + 1])


def _extract_containers(grid: np.ndarray) -> list[dict]:
    grey_comps = _components(grid == FRAME_COLOR)
    assert grey_comps, "expected at least one grey container"

    containers = []
    for coords in grey_comps:
        r0, c0, r1, c1 = _bbox(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= 3 and width >= 3, "container bbox too small"

        local = np.zeros((height, width), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        border = np.zeros((height, width), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        assert not np.any(local & ~border), "grey pixels must lie on the border"
        gap_mask = border & ~local
        assert np.any(gap_mask), "each container needs a gap"
        assert len(_components(gap_mask)) == 1, "gap shape must be connected"

        gap_local = np.argwhere(gap_mask)
        assert np.all(grid[r0 + gap_local[:, 0], c0 + gap_local[:, 1]] == BG), (
            "gap cells must stay black"
        )
        containers.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap_shape": _normalize(gap_local),
            }
        )
    return containers


def _extract_labels(grid: np.ndarray) -> list[dict]:
    colors = sorted(set(map(int, np.unique(grid))) - {BG, FRAME_COLOR})
    labels = []
    for color in colors:
        for coords in _components(grid == color):
            labels.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "shape": _normalize(coords),
                }
            )
    assert labels, "expected at least one colored label"
    return labels


def _paint_ring(
    out: np.ndarray,
    source: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
) -> None:
    h, w = source.shape
    r0, c0, r1, c1 = bbox
    pr0, pc0, pr1, pc1 = _expand_bbox(bbox, 1, h, w)

    ring = np.ones((pr1 - pr0 + 1, pc1 - pc0 + 1), dtype=bool)
    ring[r0 - pr0 : r1 - pr0 + 1, c0 - pc0 : c1 - pc0 + 1] = False

    src = source[pr0 : pr1 + 1, pc0 : pc1 + 1]
    dst = out[pr0 : pr1 + 1, pc0 : pc1 + 1]
    dst[ring & (src == BG)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored label to the unique grey frame gap with the same shape.
    Paint the label color on the one-cell outer ring around that frame and erase
    the label itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    containers = _extract_containers(g)
    labels = _extract_labels(g)
    assert len(containers) == len(labels), "containers and labels must pair one-to-one"

    by_shape: dict[tuple[tuple[int, int], ...], dict] = {}
    for container in containers:
        key = container["gap_shape"]
        assert key not in by_shape, "gap shapes must be unique"
        by_shape[key] = container

    out = g.copy()
    for label in labels:
        key = label["shape"]
        assert key in by_shape, "every label must match a container gap shape"
        container = by_shape.pop(key)
        out[label["coords"][:, 0], label["coords"][:, 1]] = BG
        _paint_ring(out, g, container["bbox"], label["color"])

    assert not by_shape, "every container must receive exactly one label"
    return out


def _make_side_gap(height: int, width: int, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        length = rng.randint(2, min(4, width))
        row = 0 if rng.random() < 0.5 else height - 1
        start = rng.randint(0, width - length)
        cells = [(row, start + dc) for dc in range(length)]
    else:
        length = rng.randint(2, min(4, height))
        col = 0 if rng.random() < 0.5 else width - 1
        start = rng.randint(0, height - length)
        cells = [(start + dr, col) for dr in range(length)]
    return np.array(cells, dtype=int)


def _make_corner_gap(height: int, width: int, rng: random.Random) -> np.ndarray:
    vlen = rng.randint(2, min(4, height))
    hlen = rng.randint(2, min(4, width))
    corner = rng.choice(["tl", "tr", "bl", "br"])

    if corner == "tl":
        cells = {(r, 0) for r in range(vlen)} | {(0, c) for c in range(hlen)}
    elif corner == "tr":
        cells = {(r, width - 1) for r in range(vlen)} | {
            (0, width - 1 - c) for c in range(hlen)
        }
    elif corner == "bl":
        cells = {(height - 1 - r, 0) for r in range(vlen)} | {
            (height - 1, c) for c in range(hlen)
        }
    else:
        cells = {(height - 1 - r, width - 1) for r in range(vlen)} | {
            (height - 1, width - 1 - c) for c in range(hlen)
        }
    return np.array(sorted(cells), dtype=int)


def _choose_gap(
    height: int,
    width: int,
    used_shapes: set[tuple[tuple[int, int], ...]],
    preferred_branch: str,
    rng: random.Random,
):
    order = [preferred_branch, "corner" if preferred_branch == "side" else "side"]
    for branch in order:
        for _ in range(GAP_ATTEMPTS):
            coords = (
                _make_side_gap(height, width, rng)
                if branch == "side"
                else _make_corner_gap(height, width, rng)
            )
            key = _normalize(coords)
            if key not in used_shapes:
                return coords, key
    return None


def _branch_schedule(count: int, rng: random.Random) -> list[str]:
    branches = ["side", "corner"] if count >= 2 else [rng.choice(["side", "corner"])]
    while len(branches) < count:
        branches.append(rng.choice(["side", "corner"]))
    rng.shuffle(branches)
    return branches


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(CONTAINER_COUNT_RANGE[0], CONTAINER_COUNT_RANGE[1])
    height = rng.randint(max(GRID_MIN, 14 + 3 * count), GRID_MAX)
    width = rng.randint(max(GRID_MIN, 14 + 3 * count), GRID_MAX)

    colors = rng.sample(list(LABEL_COLORS), count)
    branches = _branch_schedule(count, rng)
    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    specs = []

    for color, branch in zip(colors, branches):
        placed = None
        for _ in range(GAP_ATTEMPTS):
            fh = rng.randint(FRAME_SIDE_RANGE[0], FRAME_SIDE_RANGE[1])
            fw = rng.randint(FRAME_SIDE_RANGE[0], FRAME_SIDE_RANGE[1])
            gap = _choose_gap(fh, fw, used_shapes, branch, rng)
            if gap is None:
                continue
            gap_cells, shape = gap
            placed = {
                "frame_size": (fh, fw),
                "gap_cells": gap_cells,
                "shape": shape,
                "color": int(color),
            }
            used_shapes.add(shape)
            break
        if placed is None:
            return None
        specs.append(placed)

    grid = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    for spec in specs:
        fh, fw = spec["frame_size"]
        frame_placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = rng.randint(1, height - fh - 1)
            c0 = rng.randint(1, width - fw - 1)
            bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            reserve = _expand_bbox(bbox, 2, height, width)
            if not _region_free(blocked, reserve):
                continue

            grid[r0, c0 : c0 + fw] = FRAME_COLOR
            grid[r0 + fh - 1, c0 : c0 + fw] = FRAME_COLOR
            grid[r0 : r0 + fh, c0] = FRAME_COLOR
            grid[r0 : r0 + fh, c0 + fw - 1] = FRAME_COLOR

            gap_abs = spec["gap_cells"] + np.array([[r0, c0]], dtype=int)
            grid[gap_abs[:, 0], gap_abs[:, 1]] = BG
            _mark(blocked, reserve)
            spec["frame_bbox"] = bbox
            frame_placed = True
            break
        if not frame_placed:
            return None

    for spec in specs:
        patch = _shape_to_patch(spec["shape"], spec["color"])
        ph, pw = patch.shape
        label_placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = rng.randint(0, height - ph)
            c0 = rng.randint(0, width - pw)
            bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            reserve = _expand_bbox(bbox, 1, height, width)
            if not _region_free(blocked, reserve):
                continue

            region = grid[r0 : r0 + ph, c0 : c0 + pw]
            if np.any((patch != BG) & (region != BG)):
                continue

            region[patch != BG] = patch[patch != BG]
            _mark(blocked, reserve)
            label_placed = True
            break
        if not label_placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
