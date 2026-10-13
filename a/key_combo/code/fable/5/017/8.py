from __future__ import annotations

import random
from collections import deque

import numpy as np


# Coordinate convention follows arc_utils.py: grid[r, c], and bboxes are
# inclusive (r0, c0, r1, c1).  The small local helpers keep this puzzle
# directly loadable through SourceFileLoader without modifying sys.path.


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)[::-1]
    assert len(order) and (len(order) == 1 or counts[order[0]] > counts[order[1]])
    return int(values[order[0]])


def _canonical(shape: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    return (min(h, w), max(h, w))


def _monochromatic_components(grid: np.ndarray, bg: int) -> list[dict]:
    """Return 4-connected components, separated by visible color."""
    height, width = map(int, grid.shape)
    components: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        unseen = grid == color
        while bool(unseen.any()):
            start = tuple(map(int, np.argwhere(unseen)[0]))
            unseen[start] = False
            queue = deque([start])
            cells: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < height and 0 <= nc < width and bool(unseen[nr, nc]):
                        unseen[nr, nc] = False
                        queue.append((nr, nc))

            coords = np.asarray(cells, dtype=int)
            r0, c0 = map(int, coords.min(axis=0))
            r1, c1 = map(int, coords.max(axis=0))
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "sprite": grid[r0 : r1 + 1, c0 : c1 + 1].copy(),
                }
            )
    return components


def _decode(grid: np.ndarray) -> dict:
    """Classify every foreground component as one frame or one solid block."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size
    bg = _background(g)
    frames: list[dict] = []
    blocks: list[dict] = []

    for component in _monochromatic_components(g, bg):
        color = int(component["color"])
        sprite = component["sprite"]
        h, w = map(int, sprite.shape)
        solid = bool(np.all(sprite == color))
        frame = False
        if h >= 3 and w >= 3:
            border = np.zeros((h, w), dtype=bool)
            border[[0, -1], :] = True
            border[:, [0, -1]] = True
            frame = bool(np.all(sprite[border] == color) and np.all(sprite[~border] == bg))
        assert solid != frame, "each object must be exactly one solid block or hollow frame"

        r0, c0, _r1, _c1 = component["bbox"]
        item = {
            **component,
            "position": (int(r0), int(c0)),
            "shape": (h, w),
        }
        if frame:
            item["inner_shape"] = (h - 2, w - 2)
            item["key"] = _canonical(item["inner_shape"])
            frames.append(item)
        else:
            item["key"] = _canonical((h, w))
            blocks.append(item)

    assert len(frames) >= 2 and len(frames) == len(blocks)
    frame_by_key = {item["key"]: item for item in frames}
    block_by_key = {item["key"]: item for item in blocks}
    assert len(frame_by_key) == len(frames), "frame cavity sizes must be unique up to rotation"
    assert len(block_by_key) == len(blocks), "block sizes must be unique up to rotation"
    assert set(frame_by_key) == set(block_by_key), "every cavity must have exactly one fitting block"

    areas = [int(a * b) for a, b in frame_by_key]
    assert len(set(areas)) == len(areas), "cavity areas must give an unambiguous stack order"
    return {
        "bg": bg,
        "frames": frames,
        "blocks": blocks,
        "frame_by_key": frame_by_key,
        "block_by_key": block_by_key,
    }


def _assembled_units(decoded: dict) -> list[dict]:
    bg = int(decoded["bg"])
    units: list[dict] = []
    for key, frame in decoded["frame_by_key"].items():
        block = decoded["block_by_key"][key]
        inner_shape = tuple(map(int, frame["inner_shape"]))
        oriented = block["sprite"].copy()
        if oriented.shape != inner_shape:
            oriented = np.rot90(oriented)
        assert oriented.shape == inner_shape

        outer_h, outer_w = map(int, frame["shape"])
        unit = np.full((outer_h, outer_w), bg, dtype=int)
        frame_color = int(frame["color"])
        unit[[0, -1], :] = frame_color
        unit[:, [0, -1]] = frame_color
        unit[1:-1, 1:-1] = oriented
        units.append(
            {
                "key": key,
                "area": int(inner_shape[0] * inner_shape[1]),
                "frame_position": frame["position"],
                "grid": unit,
            }
        )
    return units


def _stack_units(units: list[dict], bg: int) -> np.ndarray:
    assert units
    out_h = sum(int(item["grid"].shape[0]) for item in units) + len(units) - 1
    out_w = max(int(item["grid"].shape[1]) for item in units)
    out = np.full((out_h, out_w), int(bg), dtype=int)
    row = 0
    for item in units:
        patch = item["grid"]
        h, w = map(int, patch.shape)
        out[row : row + h, :w] = patch
        row += h + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit each solid rectangle into its uniquely size-matched hollow frame, then stack the assemblies by decreasing cavity area.

    Input structure:
    - A uniquely dominant background separates equally many monochromatic, one-cell-thick hollow rectangular frames and solid rectangular blocks.
    - Cavity dimensions and block dimensions form a one-to-one match when 90-degree rotation is allowed; match keys and cavity areas are unique.
    - Colors do not encode partners: a block can share a color with a differently sized frame.

    Rule:
    - Match every block to the frame whose cavity has exactly the same height and width, allowing a 90-degree rotation of the block.
    - Keep the frame color and place the oriented block, in its own color, throughout that frame's cavity.
    - Sort the completed units by cavity area from largest to smallest, stack them top-to-bottom left-aligned with one background row between neighbors, and pad the right with background to the widest frame.

    Key insight:
    - Unordered geometric dimensions, not color or input position, determine the pairing; rotation, area ordering, and assembly are three independent decisions.
    """
    decoded = _decode(np.asarray(grid, dtype=int))
    units = _assembled_units(decoded)
    units.sort(key=lambda item: int(item["area"]), reverse=True)
    return _stack_units(units, int(decoded["bg"]))


_SHAPE_POOL = (
    (2, 3),  # area 6
    (2, 4),  # area 8
    (3, 3),  # area 9
    (2, 5),  # area 10
    (3, 4),  # area 12
    (2, 7),  # area 14, deliberately elongated
    (3, 5),  # area 15
    (4, 4),  # area 16
    (3, 6),  # area 18
    (4, 5),  # area 20
    (5, 5),  # area 25
)


def _make_frame(inner_shape: tuple[int, int], color: int, bg: int) -> np.ndarray:
    h, w = map(int, inner_shape)
    patch = np.full((h + 2, w + 2), int(bg), dtype=int)
    patch[[0, -1], :] = int(color)
    patch[:, [0, -1]] = int(color)
    return patch


def _sample_spec() -> dict | None:
    n = random.choices((4, 5), weights=(3, 1), k=1)[0]

    # This forced pair is an area-vs-perimeter counterexample: area 15 or 16
    # is larger than area 14, while its perimeter is smaller.
    compact = random.choice(((3, 5), (4, 4)))
    remaining = [shape for shape in _SHAPE_POOL if shape not in {(2, 7), compact}]
    canonical_shapes = [(2, 7), compact] + random.sample(remaining, n - 2)
    random.shuffle(canonical_shapes)

    inner_shapes: list[tuple[int, int]] = []
    for h, w in canonical_shapes:
        inner_shapes.append((w, h) if h != w and random.random() < 0.5 else (h, w))

    areas = [h * w for h, w in canonical_shapes]
    perimeters = [2 * (h + w) for h, w in canonical_shapes]
    assert len(set(areas)) == n
    # Both row and column extent must contain a strict inversion relative to area,
    # ruling out height-only and width-only stacking shortcuts.
    height_inversion = any(
        areas[i] > areas[j] and inner_shapes[i][0] < inner_shapes[j][0]
        for i in range(n) for j in range(n)
    )
    width_inversion = any(
        areas[i] > areas[j] and inner_shapes[i][1] < inner_shapes[j][1]
        for i in range(n) for j in range(n)
    )
    perimeter_inversion = any(
        areas[i] > areas[j] and perimeters[i] < perimeters[j]
        for i in range(n) for j in range(n)
    )
    if not (height_inversion and width_inversion and perimeter_inversion):
        return None

    output_h = sum(h + 2 for h, _w in inner_shapes) + n - 1
    output_w = max(w + 2 for _h, w in inner_shapes)
    if output_h > 30 or output_w > 30:
        return None

    bg = random.randrange(10)
    frame_colors = random.sample([color for color in range(10) if color != bg], n)
    shift = random.randrange(1, n)
    block_colors = [frame_colors[(i + shift) % n] for i in range(n)]

    nonsquare = [i for i, (h, w) in enumerate(inner_shapes) if h != w]
    if len(nonsquare) < 2:
        return None
    needs_rotation = [False] * n
    for i in nonsquare:
        needs_rotation[i] = random.random() < 0.5
    false_i, true_i = random.sample(nonsquare, 2)
    needs_rotation[false_i] = False
    needs_rotation[true_i] = True

    block_shapes = [
        (w, h) if needs_rotation[i] else (h, w)
        for i, (h, w) in enumerate(inner_shapes)
    ]
    return {
        "n": n,
        "bg": bg,
        "canonical_shapes": canonical_shapes,
        "inner_shapes": inner_shapes,
        "block_shapes": block_shapes,
        "frame_colors": frame_colors,
        "block_colors": block_colors,
        "needs_rotation": needs_rotation,
    }


def _place_objects(spec: dict) -> tuple[np.ndarray, dict] | None:
    n, bg = int(spec["n"]), int(spec["bg"])
    low = 22 if n == 4 else 25
    height = random.randint(low, 30)
    width = random.randint(low, 30)
    grid = np.full((height, width), bg, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    objects: list[dict] = []
    for i in range(n):
        objects.append(
            {
                "kind": "frame",
                "index": i,
                "patch": _make_frame(spec["inner_shapes"][i], spec["frame_colors"][i], bg),
            }
        )
        objects.append(
            {
                "kind": "block",
                "index": i,
                "patch": np.full(spec["block_shapes"][i], spec["block_colors"][i], dtype=int),
            }
        )

    # Largest bounding boxes first makes random packing reliable; positions remain random.
    random.shuffle(objects)
    objects.sort(key=lambda item: int(np.prod(item["patch"].shape)), reverse=True)
    placements = {"frame": {}, "block": {}}
    for item in objects:
        patch = item["patch"]
        ph, pw = map(int, patch.shape)
        candidates: list[tuple[int, int]] = []
        for r in range(1, height - ph):
            for c in range(1, width - pw):
                # Reserve complete bounding boxes, including frame cavities, with
                # one visible background cell around every pair of objects.
                if not bool(occupied[r - 1 : r + ph + 1, c - 1 : c + pw + 1].any()):
                    candidates.append((r, c))
        if not candidates:
            return None
        r, c = random.choice(candidates)
        grid[r : r + ph, c : c + pw] = patch
        occupied[r : r + ph, c : c + pw] = True
        placements[item["kind"]][int(item["index"])] = (int(r), int(c))

    values, counts = np.unique(grid, return_counts=True)
    if int(values[int(np.argmax(counts))]) != bg or int(np.sum(counts == counts.max())) != 1:
        return None
    return grid, placements


def _expected_from_spec(spec: dict, order: list[int] | None = None) -> np.ndarray:
    bg, n = int(spec["bg"]), int(spec["n"])
    if order is None:
        order = sorted(range(n), key=lambda i: int(np.prod(spec["inner_shapes"][i])), reverse=True)
    units: list[dict] = []
    for i in order:
        h, w = map(int, spec["inner_shapes"][i])
        unit = _make_frame((h, w), int(spec["frame_colors"][i]), bg)
        unit[1:-1, 1:-1] = int(spec["block_colors"][i])
        units.append({"grid": unit})
    return _stack_units(units, bg)


def _semantic_validation(grid: np.ndarray, spec: dict, placements: dict) -> bool:
    """Check rule dependencies and common wrong strategies, not just solver closure."""
    n = int(spec["n"])
    areas = [int(np.prod(shape)) for shape in spec["inner_shapes"]]
    descending = sorted(range(n), key=lambda i: areas[i], reverse=True)
    frame_order = sorted(range(n), key=lambda i: placements["frame"][i])
    block_order = sorted(range(n), key=lambda i: placements["block"][i])

    # Input reading order must not accidentally supply the requested output order
    # or the correct positional pairing.
    if frame_order == descending or block_order == descending:
        return False
    positional_pairing = all(
        _canonical(spec["inner_shapes"][fi]) == _canonical(spec["block_shapes"][bi])
        for fi, bi in zip(frame_order, block_order)
    )
    if positional_pairing:
        return False

    # Color pairing is a full derangement.  At least one same-color block is too
    # large in one dimension even after rotation, so color cannot be a fit key.
    if any(spec["frame_colors"][i] == spec["block_colors"][i] for i in range(n)):
        return False
    has_unfittable_color_decoy = False
    for frame_i in range(n):
        block_i = spec["block_colors"].index(spec["frame_colors"][frame_i])
        fh, fw = _canonical(spec["inner_shapes"][frame_i])
        bh, bw = _canonical(spec["block_shapes"][block_i])
        if bh > fh or bw > fw:
            has_unfittable_color_decoy = True
    if not has_unfittable_color_decoy:
        return False

    if not any(spec["needs_rotation"]) or all(spec["needs_rotation"]):
        return False
    if not any(
        spec["needs_rotation"][i]
        and tuple(spec["block_shapes"][i]) != tuple(spec["inner_shapes"][i])
        for i in range(n)
    ):
        return False

    expected = _expected_from_spec(spec)
    wrong_unsorted = _expected_from_spec(spec, frame_order)
    if np.array_equal(expected, wrong_unsorted):
        return False

    try:
        decoded = _decode(grid)
        actual = main(grid)
    except (AssertionError, ValueError, IndexError):
        return False
    if len(decoded["frames"]) != n or len(decoded["blocks"]) != n:
        return False
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return False
    return max(grid.shape) <= 30 and max(actual.shape) <= 30


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate instance satisfying all puzzle invariants."""
    for _ in range(1000):
        spec = _sample_spec()
        if spec is None:
            continue
        placed = _place_objects(spec)
        if placed is None:
            continue
        grid, placements = placed
        if _semantic_validation(grid, spec, placements):
            return grid
    raise RuntimeError("failed to generate a valid size-fit assembly instance")
