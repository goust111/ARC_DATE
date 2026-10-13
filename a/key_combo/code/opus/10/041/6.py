import random
from collections import Counter, deque

import numpy as np


BG = 0
MARKER_COLORS = (1, 2, 3, 4)
FRAME_COLOR = 5
TEMPLATE_COLOR = 6
TRANSLATE_COLOR = 7
ERASE_COLOR = 8
RECOLOR_COLOR = 9
OP_COLORS = (TRANSLATE_COLOR, ERASE_COLOR, RECOLOR_COLOR)

SHAPE_LIBRARY = {
    "h3": {"kind": "h", "length": 3},
    "v4": {"kind": "v", "length": 4},
    "h5": {"kind": "h", "length": 5},
}

CORNER_ORDER = ("tl", "tr", "bl", "br")
CORNER_ANCHOR_OFFSETS = {
    "tl": (1, 1),
    "tr": (1, -1),
    "bl": (-1, 1),
    "br": (-1, -1),
}
CORNER_TRANSLATIONS = {
    "tl": (-2, -2),
    "tr": (-2, 2),
    "bl": (2, -2),
    "br": (2, 2),
}

GEN_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    colors: tuple[int, ...] | list[int] | None = None,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g) if bg is None else int(bg)
    if colors is None:
        palette = sorted(set(map(int, g.flatten())) - {bg})
    else:
        palette = [int(c) for c in colors if int(c) != bg]

    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for color in palette:
        cells = np.argwhere((g == color) & ~seen)
        for r, c in cells:
            r = int(r)
            c = int(c)
            if seen[r, c] or int(g[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if not in_bounds(g.shape, nr, nc):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def normalized_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = bbox_coords(coords)
    shifted = coords - np.array([r0, c0], dtype=int)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def shape_offsets(shape_id: str, corner: str) -> list[tuple[int, int]]:
    spec = SHAPE_LIBRARY[shape_id]
    length = int(spec["length"])
    if spec["kind"] == "h":
        if corner in {"tl", "bl"}:
            return [(0, dc) for dc in range(length)]
        return [(0, -dc) for dc in range(length)]
    if corner in {"tl", "tr"}:
        return [(dr, 0) for dr in range(length)]
    return [(-dr, 0) for dr in range(length)]


def corner_anchor(frame_bbox: tuple[int, int, int, int], corner: str) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    dr, dc = CORNER_ANCHOR_OFFSETS[corner]
    return (r0 + 1 if dr > 0 else r1 - 1, c0 + 1 if dc > 0 else c1 - 1)


def component_from_anchor(grid: np.ndarray, anchor: tuple[int, int]) -> dict:
    g = np.asarray(grid, dtype=int)
    sr, sc = map(int, anchor)
    color = int(g[sr, sc])
    assert color not in {BG, FRAME_COLOR, TEMPLATE_COLOR, *OP_COLORS}
    q = deque([(sr, sc)])
    seen = {anchor}
    coords = []
    while q:
        r, c = q.popleft()
        coords.append((r, c))
        for nr, nc in neighbors4(r, c):
            if not in_bounds(g.shape, nr, nc):
                continue
            if (nr, nc) in seen or int(g[nr, nc]) != color:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    arr = np.asarray(coords, dtype=int)
    return {
        "color": color,
        "coords": arr,
        "bbox": bbox_coords(arr),
        "area": int(arr.shape[0]),
    }


def rectangular_frame(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    candidates = []
    for comp in find_components(g, bg=BG, colors=[FRAME_COLOR]):
        r0, c0, r1, c1 = comp["bbox"]
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = np.zeros_like(g, dtype=bool)
        border[r0 : r1 + 1, c0] = True
        border[r0 : r1 + 1, c1] = True
        border[r0, c0 : c1 + 1] = True
        border[r1, c0 : c1 + 1] = True
        mask = np.zeros_like(g, dtype=bool)
        coords = comp["coords"]
        mask[coords[:, 0], coords[:, 1]] = True
        if np.array_equal(mask, border):
            candidates.append(comp)
    assert len(candidates) == 1, "expected exactly one grey rectangular frame"
    return candidates[0]


def parse_templates(grid: np.ndarray) -> dict[tuple[tuple[int, int], ...], int]:
    g = np.asarray(grid, dtype=int)
    templates: dict[tuple[tuple[int, int], ...], int] = {}
    for comp in find_components(g, bg=BG, colors=[TEMPLATE_COLOR]):
        r0, c0, r1, c1 = comp["bbox"]
        coord_set = {tuple(map(int, p)) for p in comp["coords"]}
        candidates = []
        if c1 + 1 < g.shape[1]:
            for r in range(r0, r1 + 1):
                if (r, c1) not in coord_set:
                    continue
                val = int(g[r, c1 + 1])
                if val in OP_COLORS:
                    candidates.append(val)
        assert len(candidates) == 1, "each template must have one right-side indicator"
        sig = normalized_shape(comp["coords"])
        assert sig not in templates, "template shapes must be unique"
        templates[sig] = candidates[0]
    assert set(templates.values()).issubset(set(OP_COLORS))
    assert templates, "expected at least one template"
    return templates


def parse_corner_markers(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> dict[str, dict]:
    markers: dict[str, dict] = {}
    seen: set[tuple[int, int]] = set()
    for corner in CORNER_ORDER:
        comp = component_from_anchor(grid, corner_anchor(frame_bbox, corner))
        coords_set = {tuple(map(int, p)) for p in comp["coords"]}
        assert seen.isdisjoint(coords_set), "corner markers must be distinct components"
        seen |= coords_set
        markers[corner] = comp
    return markers


def expected_from_scene(inp: np.ndarray, scene: dict) -> np.ndarray:
    out = np.asarray(inp, dtype=int).copy()
    for item in scene["markers"]:
        coords = item["coords"]
        op = item["op"]
        if op in {TRANSLATE_COLOR, ERASE_COLOR}:
            out[coords[:, 0], coords[:, 1]] = BG
        elif op == RECOLOR_COLOR:
            out[coords[:, 0], coords[:, 1]] = RECOLOR_COLOR
    for item in scene["markers"]:
        if item["op"] != TRANSLATE_COLOR:
            continue
        delta = np.array(item["delta"], dtype=int)
        moved = item["coords"] + delta
        out[moved[:, 0], moved[:, 1]] = item["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each interior-corner marker to a magenta template shape, then translate, erase, or recolor it according to the template's corner instruction.

    Input structure:
    - The background is black, with one hollow grey rectangular frame and four small monochrome marker components anchored at the frame's interior corners.
    - Outside the frame are magenta template components, each paired with one single indicator pixel immediately to its right in orange, teal, or maroon.
    - Marker colors may repeat; template matching depends only on 4-connected shape after translation normalization.

    Rule:
    - Read every magenta template's normalized shape and its adjacent indicator color.
    - For each interior-corner marker, match its normalized shape to exactly one template and use that template's indicator to choose an operation.
    - Orange moves the whole marker rigidly from the interior corner to the corresponding exterior corner, teal erases it, and maroon recolors it in place to maroon; the grey frame and all templates stay unchanged.

    Key insight:
    - Shape matching is the branch selector: the matched template does not repaint all copies uniformly, it decides which of three different spatial actions applies to that specific corner marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frame = rectangular_frame(g)
    frame_bbox = frame["bbox"]
    templates = parse_templates(g)
    markers = parse_corner_markers(g, frame_bbox)

    actions: dict[str, tuple[dict, int]] = {}
    for corner, comp in markers.items():
        sig = normalized_shape(comp["coords"])
        matches = [op for shape, op in templates.items() if shape == sig]
        assert len(matches) == 1, "each corner marker must match exactly one template"
        actions[corner] = (comp, matches[0])

    out = g.copy()
    for corner, (comp, op) in actions.items():
        coords = comp["coords"]
        if op in {TRANSLATE_COLOR, ERASE_COLOR}:
            out[coords[:, 0], coords[:, 1]] = BG
        else:
            out[coords[:, 0], coords[:, 1]] = RECOLOR_COLOR

    for corner, (comp, op) in actions.items():
        if op != TRANSLATE_COLOR:
            continue
        dr, dc = CORNER_TRANSLATIONS[corner]
        moved = comp["coords"] + np.array([dr, dc], dtype=int)
        assert np.all(moved[:, 0] >= 0) and np.all(moved[:, 0] < g.shape[0])
        assert np.all(moved[:, 1] >= 0) and np.all(moved[:, 1] < g.shape[1])
        existing = out[moved[:, 0], moved[:, 1]]
        assert np.all(existing == BG), "translated marker must land on empty background"
        out[moved[:, 0], moved[:, 1]] = int(comp["color"])

    return out


def build_template_patch(shape_id: str, op: int) -> np.ndarray:
    spec = SHAPE_LIBRARY[shape_id]
    length = int(spec["length"])
    if spec["kind"] == "h":
        patch = np.full((1, length + 1), BG, dtype=int)
        patch[0, :length] = TEMPLATE_COLOR
        patch[0, length] = op
        return patch

    patch = np.full((length, 2), BG, dtype=int)
    patch[:, 0] = TEMPLATE_COLOR
    patch[random.randrange(length), 1] = op
    return patch


def can_place_patch(
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    padding: int,
) -> bool:
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(H, r0 + h + padding)
    cc1 = min(W, c0 + w + padding)
    if occupied[rr0:rr1, cc0:cc1].any():
        return False
    return True


def paste_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    target = grid[r0 : r0 + h, c0 : c0 + w]
    assert np.all(target[patch != BG] == BG)
    target[patch != BG] = patch[patch != BG]
    occupied[r0 : r0 + h, c0 : c0 + w] |= patch != BG


def sample_frame() -> tuple[int, int, int, int, int, int]:
    for _ in range(100):
        frame_h = random.randint(10, 14)
        frame_w = random.randint(10, 14)
        top = random.randint(3, 6)
        bottom = random.randint(3, 6)
        left = random.randint(3, 6)
        right = random.randint(3, 6)
        H = frame_h + top + bottom
        W = frame_w + left + right
        if H <= 30 and W <= 30:
            return H, W, top, left, frame_h, frame_w
    raise RuntimeError("failed to sample frame")


def sample_marker_colors() -> list[int]:
    while True:
        colors = [random.choice(MARKER_COLORS) for _ in range(4)]
        uniq = len(set(colors))
        if 1 < uniq < 4:
            return colors


def reserve_frame_region(
    occupied: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    *,
    extra: int,
) -> None:
    r0, c0, r1, c1 = frame_bbox
    rr0 = max(0, r0 - extra)
    cc0 = max(0, c0 - extra)
    rr1 = min(occupied.shape[0], r1 + extra + 1)
    cc1 = min(occupied.shape[1], c1 + extra + 1)
    occupied[rr0:rr1, cc0:cc1] = True


def place_templates(
    grid: np.ndarray,
    occupied: np.ndarray,
    shape_to_op: dict[str, int],
    frame_bbox: tuple[int, int, int, int],
) -> None:
    temp_occupied = occupied.copy()
    reserve_frame_region(temp_occupied, frame_bbox, extra=2)
    items = list(shape_to_op.items())
    random.shuffle(items)
    for shape_id, op in items:
        patch = build_template_patch(shape_id, op)
        h, w = patch.shape
        candidates = [(r, c) for r in range(grid.shape[0] - h + 1) for c in range(grid.shape[1] - w + 1)]
        random.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            if not can_place_patch(grid, temp_occupied, patch, r0, c0, padding=1):
                continue
            paste_patch(grid, temp_occupied, patch, r0, c0)
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place template")


def draw_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR


def build_scene() -> tuple[np.ndarray, dict]:
    H, W, r0, c0, frame_h, frame_w = sample_frame()
    grid = np.full((H, W), BG, dtype=int)
    frame_bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
    draw_frame(grid, frame_bbox)

    shape_ids = list(SHAPE_LIBRARY)
    ops = list(OP_COLORS)
    random.shuffle(ops)
    shape_to_op = dict(zip(shape_ids, ops))

    repeated_shape = random.choice(shape_ids)
    corner_shapes = shape_ids + [repeated_shape]
    random.shuffle(corner_shapes)

    marker_colors = sample_marker_colors()
    markers = []
    for corner, shape_id, color in zip(CORNER_ORDER, corner_shapes, marker_colors):
        anchor = corner_anchor(frame_bbox, corner)
        coords = np.asarray(
            [(anchor[0] + dr, anchor[1] + dc) for dr, dc in shape_offsets(shape_id, corner)],
            dtype=int,
        )
        assert np.all(coords[:, 0] > r0) and np.all(coords[:, 0] < frame_bbox[2])
        assert np.all(coords[:, 1] > c0) and np.all(coords[:, 1] < frame_bbox[3])
        grid[coords[:, 0], coords[:, 1]] = color
        moved = coords + np.array(CORNER_TRANSLATIONS[corner], dtype=int)
        assert np.all(moved[:, 0] >= 0) and np.all(moved[:, 0] < H)
        assert np.all(moved[:, 1] >= 0) and np.all(moved[:, 1] < W)
        assert np.all(grid[moved[:, 0], moved[:, 1]] == BG)
        markers.append(
            {
                "corner": corner,
                "shape_id": shape_id,
                "coords": coords,
                "color": int(color),
                "op": int(shape_to_op[shape_id]),
                "delta": CORNER_TRANSLATIONS[corner],
            }
        )

    occupied = grid != BG
    place_templates(grid, occupied, shape_to_op, frame_bbox)

    scene = {
        "frame_bbox": frame_bbox,
        "shape_to_op": shape_to_op,
        "markers": markers,
    }
    return grid, scene


def validate_scene(inp: np.ndarray, scene: dict) -> None:
    frame = rectangular_frame(inp)
    assert frame["bbox"] == scene["frame_bbox"]

    templates = parse_templates(inp)
    assert len(templates) == 3
    assert set(templates.values()) == set(OP_COLORS)

    markers = parse_corner_markers(inp, scene["frame_bbox"])
    matched_ops = []
    for item in scene["markers"]:
        comp = markers[item["corner"]]
        assert int(comp["color"]) == item["color"]
        assert {tuple(map(int, p)) for p in comp["coords"]} == {
            tuple(map(int, p)) for p in item["coords"]
        }
        sig = normalized_shape(comp["coords"])
        assert templates[sig] == item["op"]
        matched_ops.append(item["op"])

    counts = Counter(matched_ops)
    assert set(counts) == set(OP_COLORS)
    assert sorted(counts.values()) == [1, 1, 2]

    expected = expected_from_scene(inp, scene)
    actual = main(inp)
    assert np.array_equal(actual, expected)

    for item in scene["markers"]:
        coords = item["coords"]
        moved = coords + np.array(item["delta"], dtype=int)
        if item["op"] == TRANSLATE_COLOR:
            assert np.all(actual[coords[:, 0], coords[:, 1]] == BG)
            assert np.all(actual[moved[:, 0], moved[:, 1]] == item["color"])
        elif item["op"] == ERASE_COLOR:
            assert np.all(actual[coords[:, 0], coords[:, 1]] == BG)
        else:
            assert np.all(actual[coords[:, 0], coords[:, 1]] == RECOLOR_COLOR)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            inp, scene = build_scene()
            validate_scene(inp, scene)
            return inp
        except (AssertionError, RuntimeError):
            continue
    raise RuntimeError("failed to generate a valid puzzle input")
