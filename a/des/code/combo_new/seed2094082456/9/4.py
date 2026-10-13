from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
MARKER_COLOR = 9
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8]
DIRS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_ATTEMPTS = 600


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    return len(_components(mask)) == 1 if cells.size else False


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _marker_candidates(mask: np.ndarray, direction: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    if direction == "up":
        return [(0, c) for c in range(1, w - 1) if bool(mask[0, c])]
    if direction == "down":
        return [(h - 1, c) for c in range(1, w - 1) if bool(mask[h - 1, c])]
    if direction == "left":
        return [(r, 0) for r in range(1, h - 1) if bool(mask[r, 0])]
    return [(r, w - 1) for r in range(1, h - 1) if bool(mask[r, w - 1])]


def _extract_objects(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    bg = get_bg(grid)
    comps = _components(grid != bg)
    assert comps, "expected at least one foreground object"

    objects: list[dict[str, object]] = []
    for comp in comps:
        vals = grid[comp[:, 0], comp[:, 1]]
        marker_cells = comp[vals == MARKER_COLOR]
        assert len(marker_cells) == 1, "each object must contain exactly one maroon marker"

        object_colors = {int(v) for v in vals if int(v) != MARKER_COLOR}
        assert len(object_colors) == 1, "each mixed component must have one non-marker color"
        color = next(iter(object_colors))

        r0, c0, r1, c1 = _bbox(comp)
        mr, mc = map(int, marker_cells[0])
        side_hits = []
        if mr == r0:
            side_hits.append("up")
        if mr == r1:
            side_hits.append("down")
        if mc == c0:
            side_hits.append("left")
        if mc == c1:
            side_hits.append("right")
        assert len(side_hits) == 1, "marker must sit on exactly one bbox side"

        rel = comp - np.array([[r0, c0]])
        objects.append(
            {
                "color": int(color),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "direction": side_hits[0],
                "marker": (int(mr), int(mc)),
                "rel": rel.astype(int),
            }
        )

    objects.sort(key=lambda obj: tuple(obj["bbox"]))  # type: ignore[arg-type]
    return int(bg), objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each maroon-pointed object and stamp its copies straight ahead until blocked.

    Input structure:
    - A dominant background surrounds several 4-connected solid objects in distinct colors.
    - Each object contains exactly one maroon pixel that replaces one edge pixel of that object.
    - The maroon pixel lies on exactly one side of the object's bounding box, revealing one of the
      four cardinal projection directions.

    Rule:
    - Treat each non-background connected component as one object-plus-marker unit and identify its
      unique maroon cell.
    - Restore the maroon cell to the object's own color.
    - Repeatedly copy the repaired object by one bounding-box height upward or downward, or one
      bounding-box width leftward or rightward, according to the marked side.
    - Keep stamping copies until the next full copy would leave the grid or touch any non-background
      pixel, then stop; the original object stays in place and all maroon markers disappear.

    Key insight:
    - The maroon pixel is not a separate object: it is a replaced border cell whose bbox-side
      position secretly encodes a projection direction, after which the whole object repeats as a
      fixed-size stamp.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, objects = _extract_objects(g)
    out = g.copy()

    for obj in objects:
        mr, mc = obj["marker"]  # type: ignore[assignment]
        out[int(mr), int(mc)] = int(obj["color"])

    h, w = g.shape
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]  # type: ignore[assignment]
        rel = obj["rel"]  # type: ignore[assignment]
        color = int(obj["color"])
        dr, dc = DIRS[str(obj["direction"])]
        step_r = dr * (int(r1) - int(r0) + 1)
        step_c = dc * (int(c1) - int(c0) + 1)

        shift = 1
        while True:
            coords = rel + np.array([[int(r0 + shift * step_r), int(c0 + shift * step_c)]])
            rows = coords[:, 0]
            cols = coords[:, 1]
            if (
                np.any(rows < 0)
                or np.any(rows >= h)
                or np.any(cols < 0)
                or np.any(cols >= w)
            ):
                break
            if np.any(out[rows, cols] != bg):
                break
            out[rows, cols] = color
            shift += 1

    return out


def _mark_with_buffer(target: np.ndarray, coords: np.ndarray) -> None:
    h, w = target.shape
    for r, c in coords:
        for rr in range(max(0, int(r) - 1), min(h, int(r) + 2)):
            for cc in range(max(0, int(c) - 1), min(w, int(c) + 2)):
                target[rr, cc] = True


def _make_shape(direction: str) -> tuple[np.ndarray, tuple[int, int]] | None:
    for _ in range(120):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        mask = np.ones((h, w), dtype=bool)

        corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
        for rr, cc in random.sample(corners, random.choice([0, 0, 1, 1, 2])):
            mask[rr, cc] = False

        removable = []
        if direction != "up":
            removable.extend((0, c) for c in range(1, w - 1))
        if direction != "down":
            removable.extend((h - 1, c) for c in range(1, w - 1))
        if direction != "left":
            removable.extend((r, 0) for r in range(1, h - 1))
        if direction != "right":
            removable.extend((r, w - 1) for r in range(1, h - 1))
        random.shuffle(removable)
        if removable and random.random() < 0.35:
            for rr, cc in removable:
                if not bool(mask[rr, cc]):
                    continue
                trial = mask.copy()
                trial[rr, cc] = False
                if _is_connected(trial):
                    mask = trial
                    break

        coords = np.argwhere(mask)
        if len(coords) < max(6, h + w - 2):
            continue
        if _bbox(coords) != (0, 0, h - 1, w - 1):
            continue
        if not _is_connected(mask):
            continue

        markers = _marker_candidates(mask, direction)
        if markers:
            return mask, random.choice(markers)
    return None


def _simulate_copies(
    base_cells: np.ndarray,
    bbox_shape: tuple[int, int],
    direction: str,
    blocked_mask: np.ndarray,
    original_mask: np.ndarray,
) -> tuple[list[np.ndarray], str]:
    h, w = blocked_mask.shape
    local = blocked_mask.copy()
    local[base_cells[:, 0], base_cells[:, 1]] = True
    dr, dc = DIRS[direction]
    step_r = dr * bbox_shape[0]
    step_c = dc * bbox_shape[1]
    copies: list[np.ndarray] = []

    shift = 1
    while True:
        coords = base_cells + np.array([[shift * step_r, shift * step_c]])
        rows = coords[:, 0]
        cols = coords[:, 1]
        if (
            np.any(rows < 0)
            or np.any(rows >= h)
            or np.any(cols < 0)
            or np.any(cols >= w)
        ):
            return copies, "boundary"

        overlap = local[rows, cols]
        if np.any(overlap):
            if np.any(original_mask[rows[overlap], cols[overlap]]):
                return copies, "object"
            return copies, "copy"

        copies.append(coords)
        local[rows, cols] = True
        shift += 1


def _find_placement(
    shape: np.ndarray,
    direction: str,
    stop_mode: str,
    blocked_mask: np.ndarray,
    original_mask: np.ndarray,
    original_buffer: np.ndarray,
) -> tuple[tuple[int, int], list[np.ndarray]] | None:
    rel = np.argwhere(shape)
    sh, sw = shape.shape
    h, w = blocked_mask.shape
    candidates: list[tuple[tuple[int, int], list[np.ndarray]]] = []

    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            cells = rel + np.array([[r0, c0]])
            rows = cells[:, 0]
            cols = cells[:, 1]
            if np.any(blocked_mask[rows, cols]):
                continue
            if np.any(original_buffer[rows, cols]):
                continue

            copies, reason = _simulate_copies(cells, shape.shape, direction, blocked_mask, original_mask)
            if not (1 <= len(copies) <= 4):
                continue
            if reason != stop_mode:
                continue
            candidates.append(((int(r0), int(c0)), copies))

    if not candidates:
        return None
    return random.choice(candidates)


def _render_expected(shape_specs: list[dict[str, object]], grid_shape: tuple[int, int]) -> np.ndarray:
    out = np.full(grid_shape, BG, dtype=int)
    for spec in shape_specs:
        color = int(spec["color"])
        rel = spec["rel"]  # type: ignore[assignment]
        r0, c0 = spec["top_left"]  # type: ignore[assignment]
        base = rel + np.array([[int(r0), int(c0)]])
        out[base[:, 0], base[:, 1]] = color
        for copy_coords in spec["copies"]:  # type: ignore[assignment]
            out[copy_coords[:, 0], copy_coords[:, 1]] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(15, 28)
    w = random.randint(15, 28)
    n_objects = random.randint(2, 4)

    directions = random.sample(list(DIRS), n_objects)
    colors = random.sample(COLOR_POOL, n_objects)

    stop_modes = ["boundary"]
    required_object_idx = random.randint(1, n_objects - 1)
    for idx in range(1, n_objects):
        if idx == required_object_idx:
            stop_modes.append("object")
        else:
            stop_modes.append(random.choice(["boundary", "object"]))

    blocked_mask = np.zeros((h, w), dtype=bool)
    original_mask = np.zeros((h, w), dtype=bool)
    original_buffer = np.zeros((h, w), dtype=bool)
    specs: list[dict[str, object]] = []

    for color, direction, stop_mode in zip(colors, directions, stop_modes):
        built = _make_shape(direction)
        if built is None:
            return None
        shape, marker_rel = built

        placement = _find_placement(shape, direction, stop_mode, blocked_mask, original_mask, original_buffer)
        if placement is None:
            return None
        top_left, copies = placement

        rel = np.argwhere(shape)
        base = rel + np.array([[top_left[0], top_left[1]]])
        blocked_mask[base[:, 0], base[:, 1]] = True
        original_mask[base[:, 0], base[:, 1]] = True
        _mark_with_buffer(original_buffer, base)
        for copy_coords in copies:
            blocked_mask[copy_coords[:, 0], copy_coords[:, 1]] = True

        specs.append(
            {
                "color": int(color),
                "direction": direction,
                "top_left": top_left,
                "rel": rel.astype(int),
                "marker_rel": marker_rel,
                "copies": copies,
            }
        )

    inp = np.full((h, w), BG, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        rel = spec["rel"]  # type: ignore[assignment]
        r0, c0 = spec["top_left"]  # type: ignore[assignment]
        coords = rel + np.array([[int(r0), int(c0)]])
        inp[coords[:, 0], coords[:, 1]] = color
        mr, mc = spec["marker_rel"]  # type: ignore[assignment]
        inp[int(r0) + int(mr), int(c0) + int(mc)] = MARKER_COLOR

    expected = _render_expected(specs, inp.shape)
    if int(np.unique(inp, return_counts=True)[0][np.argmax(np.unique(inp, return_counts=True)[1])]) != BG:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None

    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(inp, solved):
        return None
    if int((inp == MARKER_COLOR).sum()) != n_objects:
        return None
    if len(np.unique(inp)) < 4:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid directional-stamping input")
