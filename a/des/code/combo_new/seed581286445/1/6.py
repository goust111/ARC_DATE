import numpy as np


BG = 0
YELLOW = 4
PALETTE = tuple(color for color in range(10) if color not in (BG, YELLOW))
CORNERS = ("tl", "tr", "bl", "br")
RNG = np.random.default_rng()
GEN_RETRIES = 400
PLACEMENT_RETRIES = 200


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _input_box(obj: dict) -> tuple[int, int, int, int]:
    r0 = int(obj["r0"])
    c0 = int(obj["c0"])
    r1 = r0 + int(obj["h"]) - 1
    c1 = c0 + int(obj["w"]) - 1
    return (r0 - 1, c0 - 1, r1 + 1, c1 + 1)


def _output_box(obj: dict) -> tuple[int, int, int, int]:
    r0 = int(obj["r0"])
    c0 = int(obj["c0"])
    r1 = r0 + int(obj["h"]) - 1
    c1 = c0 + int(obj["w"]) - 1
    layers = int(obj["layers"])
    return (r0 - layers, c0 - layers, r1 + layers, c1 + layers)


def _side_positions(obj: dict) -> list[tuple[int, int]]:
    r0 = int(obj["r0"])
    c0 = int(obj["c0"])
    h = int(obj["h"])
    w = int(obj["w"])
    r1 = r0 + h - 1
    c1 = c0 + w - 1

    positions: list[tuple[int, int]] = []
    positions.extend((r0 - 1, c) for c in range(c0, c1 + 1))
    positions.extend((r1 + 1, c) for c in range(c0, c1 + 1))
    positions.extend((r, c0 - 1) for r in range(r0, r1 + 1))
    positions.extend((r, c1 + 1) for r in range(r0, r1 + 1))
    return positions


def _corner_position(obj: dict, corner: str | None = None) -> tuple[int, int]:
    name = str(obj["corner"] if corner is None else corner)
    r0 = int(obj["r0"])
    c0 = int(obj["c0"])
    r1 = r0 + int(obj["h"]) - 1
    c1 = c0 + int(obj["w"]) - 1
    if name == "tl":
        return (r0 - 1, c0 - 1)
    if name == "tr":
        return (r0 - 1, c1 + 1)
    if name == "bl":
        return (r1 + 1, c0 - 1)
    return (r1 + 1, c1 + 1)


def _paint_ring_border(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    for c in range(c0, c1 + 1):
        if int(grid[r0, c]) == BG:
            grid[r0, c] = int(color)
        if int(grid[r1, c]) == BG:
            grid[r1, c] = int(color)
    for r in range(r0 + 1, r1):
        if int(grid[r, c0]) == BG:
            grid[r, c0] = int(color)
        if int(grid[r, c1]) == BG:
            grid[r, c1] = int(color)


def _find_color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(cells)
    return components


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    rectangles: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, YELLOW)):
        for cells in _find_color_components(grid, color):
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            r0 = min(rows)
            r1 = max(rows)
            c0 = min(cols)
            c1 = max(cols)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if area < 2 or area != len(cells):
                continue
            if not bool((grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)).all()):
                continue
            rectangles.append(
                {
                    "rect_color": int(color),
                    "r0": int(r0),
                    "c0": int(c0),
                    "h": int(r1 - r0 + 1),
                    "w": int(c1 - c0 + 1),
                }
            )
    rectangles.sort(key=lambda obj: (int(obj["r0"]), int(obj["c0"]), int(obj["rect_color"])))
    return rectangles


def _annotate_rectangles(grid: np.ndarray, rectangles: list[dict]) -> list[dict]:
    h, w = grid.shape
    annotated: list[dict] = []
    for base in rectangles:
        obj = dict(base)
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        r1 = r0 + int(obj["h"]) - 1
        c1 = c0 + int(obj["w"]) - 1

        side_positions = [
            (r, c)
            for r, c in (
                [(r0 - 1, c) for c in range(c0, c1 + 1)]
                + [(r1 + 1, c) for c in range(c0, c1 + 1)]
                + [(r, c0 - 1) for r in range(r0, r1 + 1)]
                + [(r, c1 + 1) for r in range(r0, r1 + 1)]
            )
            if 0 <= r < h and 0 <= c < w
        ]
        marker_count = sum(int(grid[r, c]) == YELLOW for r, c in side_positions)
        assert 1 <= marker_count <= 4, "each rectangle must have one to four yellow markers"
        obj["layers"] = int(marker_count)

        corner_hits: list[tuple[str, int]] = []
        for name in CORNERS:
            rr, cc = _corner_position(obj, name)
            if 0 <= rr < h and 0 <= cc < w:
                value = int(grid[rr, cc])
                if value not in (BG, YELLOW, int(obj["rect_color"])):
                    corner_hits.append((name, value))
        assert len(corner_hits) == 1, "each rectangle must have exactly one diagonal color marker"
        obj["corner"] = str(corner_hits[0][0])
        obj["ring_color"] = int(corner_hits[0][1])
        annotated.append(obj)
    return annotated


def _render_output(objects: list[dict], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    ordered = sorted(objects, key=lambda obj: (int(obj["r0"]), int(obj["c0"]), int(obj["rect_color"])))
    for obj in ordered:
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        h = int(obj["h"])
        w = int(obj["w"])
        out[r0 : r0 + h, c0 : c0 + w] = int(obj["rect_color"])
    for obj in ordered:
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        h = int(obj["h"])
        w = int(obj["w"])
        ring_color = int(obj["ring_color"])
        for layer in range(1, int(obj["layers"]) + 1):
            rr0 = r0 - layer
            cc0 = c0 - layer
            rr1 = r0 + h - 1 + layer
            cc1 = c0 + w - 1 + layer
            _paint_ring_border(out, rr0, cc0, rr1, cc1, ring_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow colored rectangular rings around each solid rectangle based on nearby yellow counts and diagonal color markers.

    Input structure:
    - The background is black.
    - Several disjoint solid colored rectangles appear on the grid.
    - In the side-adjacent moat around each rectangle, one to four yellow singleton markers sit on the top, bottom, left, or right side.
    - Exactly one of the four outward diagonal corner cells around each rectangle contains a non-background color marker that gives the ring color.

    Rule:
    - Detect every solid rectangle and count the yellow markers on its side-adjacent moat; that count is the number of growth layers.
    - Read the unique diagonal marker color for that rectangle.
    - Remove all yellow markers and diagonal marker pixels, keep the original rectangles, and draw that many one-pixel-thick concentric rectangular rings in the diagonal marker color.
    - A ring pixel is only painted onto background; existing rectangles or previously painted rings are never overwritten.

    Key insight:
    The puzzle splits the growth specification into two orthogonal clues: side yellow markers encode how many layers to add, while the single diagonal marker encodes which color to use.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rectangles = _extract_rectangles(g)
    assert rectangles, "expected at least one solid rectangle"
    objects = _annotate_rectangles(g, rectangles)

    rect_mask = np.zeros_like(g, dtype=bool)
    for obj in objects:
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        h = int(obj["h"])
        w = int(obj["w"])
        rect_mask[r0 : r0 + h, c0 : c0 + w] = True

    out = g.copy()
    out[~rect_mask] = BG

    # Raster-order painting gives a stable resolution when ring fronts meet.
    for obj in objects:
        ring_color = int(obj["ring_color"])
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        h = int(obj["h"])
        w = int(obj["w"])
        for layer in range(1, int(obj["layers"]) + 1):
            rr0 = r0 - layer
            cc0 = c0 - layer
            rr1 = r0 + h - 1 + layer
            cc1 = c0 + w - 1 + layer
            _paint_ring_border(out, rr0, cc0, rr1, cc1, ring_color)
    return out


def _make_object(rect_color: int, ring_color: int, h: int, w: int, layers: int) -> dict:
    return {
        "rect_color": int(rect_color),
        "ring_color": int(ring_color),
        "h": int(h),
        "w": int(w),
        "layers": int(layers),
        "corner": str(RNG.choice(CORNERS)),
    }


def _assign_markers(obj: dict) -> None:
    positions = _side_positions(obj)
    count = int(obj["layers"])
    picks = RNG.choice(len(positions), size=count, replace=False)
    obj["markers"] = [positions[int(idx)] for idx in np.atleast_1d(picks)]


def _place_interacting_pair(obj_a: dict, obj_b: dict, height: int, width: int) -> bool:
    if int(obj_a["layers"]) + int(obj_b["layers"]) < 3:
        return False

    for _ in range(PLACEMENT_RETRIES):
        if bool(RNG.integers(0, 2)):
            gap_max = int(obj_a["layers"] + obj_b["layers"])
            if gap_max < 3:
                continue
            gap = int(RNG.integers(3, gap_max + 1))

            r0_min = max(int(obj_a["layers"]), int(obj_b["layers"]))
            r0_max = height - max(int(obj_a["h"] + obj_a["layers"]), int(obj_b["h"] + obj_b["layers"]))
            c0_min = int(obj_a["layers"])
            c0_max = width - (
                int(obj_a["w"]) + gap + int(obj_b["w"]) + int(obj_b["layers"]) - 1
            )
            if r0_min > r0_max or c0_min > c0_max:
                continue

            r0 = int(RNG.integers(r0_min, r0_max + 1))
            c0 = int(RNG.integers(c0_min, c0_max + 1))
            obj_a["r0"] = r0
            obj_a["c0"] = c0
            obj_b["r0"] = r0
            obj_b["c0"] = c0 + int(obj_a["w"]) - 1 + gap
        else:
            gap_max = int(obj_a["layers"] + obj_b["layers"])
            if gap_max < 3:
                continue
            gap = int(RNG.integers(3, gap_max + 1))

            c0_min = max(int(obj_a["layers"]), int(obj_b["layers"]))
            c0_max = width - max(int(obj_a["w"] + obj_a["layers"]), int(obj_b["w"] + obj_b["layers"]))
            r0_min = int(obj_a["layers"])
            r0_max = height - (
                int(obj_a["h"]) + gap + int(obj_b["h"]) + int(obj_b["layers"]) - 1
            )
            if r0_min > r0_max or c0_min > c0_max:
                continue

            c0 = int(RNG.integers(c0_min, c0_max + 1))
            r0 = int(RNG.integers(r0_min, r0_max + 1))
            obj_a["r0"] = r0
            obj_a["c0"] = c0
            obj_b["r0"] = r0 + int(obj_a["h"]) - 1 + gap
            obj_b["c0"] = c0

        if _boxes_overlap(_input_box(obj_a), _input_box(obj_b)):
            continue
        if not _boxes_overlap(_output_box(obj_a), _output_box(obj_b)):
            continue
        return True
    return False


def _place_separated_object(obj: dict, placed: list[dict], height: int, width: int) -> bool:
    r0_min = int(obj["layers"])
    r0_max = height - int(obj["h"] + obj["layers"])
    c0_min = int(obj["layers"])
    c0_max = width - int(obj["w"] + obj["layers"])
    if r0_min > r0_max or c0_min > c0_max:
        return False

    for _ in range(PLACEMENT_RETRIES):
        obj["r0"] = int(RNG.integers(r0_min, r0_max + 1))
        obj["c0"] = int(RNG.integers(c0_min, c0_max + 1))
        in_box = _input_box(obj)
        out_box = _output_box(obj)
        ok = True
        for other in placed:
            if _boxes_overlap(in_box, _input_box(other)) or _boxes_overlap(out_box, _output_box(other)):
                ok = False
                break
        if ok:
            return True
    return False


def _build_objects(height: int, width: int, n_objects: int) -> list[dict] | None:
    max_h = min(5, max(2, height // max(3, n_objects)))
    max_w = min(6, max(2, width // max(3, n_objects)))
    if max_h < 2 or max_w < 2:
        return None

    palette = list(PALETTE)
    RNG.shuffle(palette)
    rect_colors = palette[:n_objects]
    ring_pool = [color for color in palette if color not in rect_colors]

    objects = [
        _make_object(
            rect_color=rect_colors[idx],
            ring_color=ring_pool[idx],
            h=int(RNG.integers(2, max_h + 1)),
            w=int(RNG.integers(2, max_w + 1)),
            layers=int(RNG.integers(1, 5)),
        )
        for idx in range(n_objects)
    ]

    interaction = n_objects >= 2 and RNG.random() < 0.35
    if interaction:
        if int(objects[0]["layers"]) + int(objects[1]["layers"]) < 3:
            interaction = False
        else:
            shared_color = int(RNG.choice(ring_pool))
            objects[0]["ring_color"] = shared_color
            objects[1]["ring_color"] = shared_color
            spare_colors = [color for color in ring_pool if color != shared_color]
            RNG.shuffle(spare_colors)
            for idx in range(2, n_objects):
                objects[idx]["ring_color"] = int(spare_colors[(idx - 2) % len(spare_colors)])

    placed: list[dict] = []
    start_idx = 0
    if interaction:
        if not _place_interacting_pair(objects[0], objects[1], height, width):
            return None
        placed.extend([objects[0], objects[1]])
        start_idx = 2

    for idx in range(start_idx, n_objects):
        if not _place_separated_object(objects[idx], placed, height, width):
            return None
        placed.append(objects[idx])

    for obj in objects:
        _assign_markers(obj)
    return objects


def _render_input(objects: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for obj in objects:
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        h = int(obj["h"])
        w = int(obj["w"])
        grid[r0 : r0 + h, c0 : c0 + w] = int(obj["rect_color"])
        for r, c in obj["markers"]:
            grid[int(r), int(c)] = YELLOW
        rr, cc = _corner_position(obj)
        grid[int(rr), int(cc)] = int(obj["ring_color"])
    return grid


def _layout_is_valid(objects: list[dict], shape: tuple[int, int]) -> bool:
    height, width = shape
    seen_inputs: list[tuple[int, int, int, int]] = []
    for obj in objects:
        out_box = _output_box(obj)
        if out_box[0] < 0 or out_box[1] < 0 or out_box[2] >= height or out_box[3] >= width:
            return False
        in_box = _input_box(obj)
        for other_box in seen_inputs:
            if _boxes_overlap(in_box, other_box):
                return False
        seen_inputs.append(in_box)

        marker_positions = list(obj["markers"])
        if len(marker_positions) != int(obj["layers"]) or len(set(marker_positions)) != len(marker_positions):
            return False
        if any(pos not in _side_positions(obj) for pos in marker_positions):
            return False
        if _corner_position(obj) in marker_positions:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_objects = int(RNG.integers(1, 5))
    min_size = 12 + 2 * n_objects
    height = int(RNG.integers(min_size, 31))
    width = int(RNG.integers(min_size, 31))

    objects = _build_objects(height, width, n_objects)
    if objects is None or not _layout_is_valid(objects, (height, width)):
        return None

    grid = _render_input(objects, (height, width))
    expected = _render_output(objects, (height, width))
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
