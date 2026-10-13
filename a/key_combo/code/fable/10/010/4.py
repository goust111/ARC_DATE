from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


_NEIGHBORS_8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

_PALETTE = list(range(10))
_FG_PALETTE = list(range(1, 10))

_GRID_H_RANGE = (14, 22)
_GRID_W_RANGE = (15, 24)
_OBJECT_COUNT_RANGE = (3, 5)
_OBJECT_W_RANGE = (1, 4)
_OBJECT_H_RANGE = (1, 4)
_GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    mask = grid == int(color)
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in _NEIGHBORS_8:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and mask[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.asarray(cells, dtype=int))
    return components


def _is_full_width_bar(coords: np.ndarray, h: int, w: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return coords.shape[0] == w and r0 == r1 and c0 == 0 and c1 == w - 1 and r0 in {0, h - 1}


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 5 and w >= 5

    top_color = int(g[0, 0])
    bottom_color = int(g[h - 1, 0])
    assert top_color != bottom_color
    assert bool(np.all(g[0, :] == top_color))
    assert bool(np.all(g[h - 1, :] == bottom_color))

    bg = _get_bg(g)
    port_colors = sorted(set(map(int, np.unique(g))) - {bg, top_color, bottom_color})
    assert len(port_colors) == 1
    port_color = int(port_colors[0])

    port_components = _components_of_color(g, port_color)
    assert port_components
    ports = []
    for comp in port_components:
        assert comp.shape[0] == 1, "ports must be isolated singleton pixels"
        ports.append((int(comp[0, 0]), int(comp[0, 1])))

    objects = []
    for color in (top_color, bottom_color):
        for coords in _components_of_color(g, color):
            if _is_full_width_bar(coords, h, w):
                continue
            r0, c0, r1, c1 = _bbox(coords)
            assert 0 < r0 <= r1 < h - 1
            objects.append(
                {
                    "coords": coords.copy(),
                    "bbox": (r0, c0, r1, c1),
                    "color": int(color),
                }
            )

    assert objects
    objects.sort(key=lambda obj: (obj["bbox"][1], obj["bbox"][0]))
    for left, right in zip(objects, objects[1:]):
        assert left["bbox"][3] < right["bbox"][1], "object column bands must be disjoint"

    return {
        "grid": g,
        "h": h,
        "w": w,
        "bg": bg,
        "top_color": top_color,
        "bottom_color": bottom_color,
        "port_color": port_color,
        "ports": ports,
        "objects": objects,
    }


def _simulate_object(
    coords: np.ndarray,
    color: int,
    band_ports: list[tuple[int, int, int]],
    h: int,
    top_color: int,
    bottom_color: int,
) -> tuple[np.ndarray, int, set[int]]:
    current = coords.copy()
    current_color = int(color)
    consumed: set[int] = set()

    while True:
        r0, _c0, r1, _c1 = _bbox(current)
        if current_color == top_color:
            ahead = [(row, idx) for row, _col, idx in band_ports if idx not in consumed and row < r0]
            if ahead:
                port_row, port_idx = max(ahead)
                current = current + np.array([port_row - r0, 0], dtype=int)
                consumed.add(int(port_idx))
                current_color = bottom_color
                continue
            current = current + np.array([1 - r0, 0], dtype=int)
            return current, current_color, consumed

        ahead = [(row, idx) for row, _col, idx in band_ports if idx not in consumed and row > r1]
        if ahead:
            port_row, port_idx = min(ahead)
            current = current + np.array([port_row - r1, 0], dtype=int)
            consumed.add(int(port_idx))
            current_color = top_color
            continue
        current = current + np.array([(h - 2) - r1, 0], dtype=int)
        return current, current_color, consumed


def _solve_from_scene(scene: dict) -> np.ndarray:
    h = int(scene["h"])
    w = int(scene["w"])
    bg = int(scene["bg"])
    top_color = int(scene["top_color"])
    bottom_color = int(scene["bottom_color"])
    port_color = int(scene["port_color"])
    ports = list(scene["ports"])

    out = np.full((h, w), bg, dtype=int)
    out[0, :] = top_color
    out[h - 1, :] = bottom_color

    consumed_ports: set[int] = set()
    for obj in scene["objects"]:
        r0, c0, r1, c1 = obj["bbox"]
        band_ports = [(r, c, idx) for idx, (r, c) in enumerate(ports) if c0 <= c <= c1]
        final_coords, final_color, used = _simulate_object(
            obj["coords"],
            obj["color"],
            band_ports,
            h,
            top_color,
            bottom_color,
        )
        consumed_ports |= used
        assert bool(np.all((0 <= final_coords[:, 0]) & (final_coords[:, 0] < h)))
        assert bool(np.all((0 <= final_coords[:, 1]) & (final_coords[:, 1] < w)))
        assert bool(np.all(out[final_coords[:, 0], final_coords[:, 1]] == bg))
        out[final_coords[:, 0], final_coords[:, 1]] = int(final_color)

    for idx, (r, c) in enumerate(ports):
        if idx not in consumed_ports:
            assert out[r, c] == bg
            out[r, c] = port_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each object toward its matching border bar while ports flip its color and reverse its travel.

    Input structure:
    - The top row is a full-width bar in color A and the bottom row is a full-width bar in color B.
    - Interior foreground objects are monochrome 8-connected components in color A or B, and their column bands do not overlap.
    - Color C marks isolated port pixels inside the interior; ports are consumed if an object's moving bbox edge reaches them.

    Rule:
    - Each object moves vertically toward the border bar whose color matches the object's current color.
    - If an unused port lies ahead anywhere inside that object's column band, the object slides until its leading bbox edge reaches the port row, then the port disappears, the object switches to the other bar color, and its travel direction reverses.
    - Repeating this process until no unused port remains ahead makes the object dock against the inner side of its current target bar; unused ports stay visible and the two border bars never change.

    Key insight:
    - Ports are one-shot switch points, so the final color and docking side depend on the consumed port history inside the object's own column band rather than on the object's starting color alone.
    """
    return _solve_from_scene(_parse_scene(np.asarray(grid, dtype=int)))


def _corner_path(h: int, w: int) -> set[tuple[int, int]]:
    if h == 1 and w == 1:
        return {(0, 0)}
    r0 = random.choice([0, h - 1])
    c0 = random.choice([0, w - 1])
    r1 = h - 1 - r0
    c1 = w - 1 - c0
    steps = max(h, w)
    cells: set[tuple[int, int]] = set()
    for k in range(steps):
        t = 0.0 if steps == 1 else k / (steps - 1)
        rr = int(round(r0 + (r1 - r0) * t))
        cc = int(round(c0 + (c1 - c0) * t))
        cells.add((rr, cc))
    return cells


def _random_connected_mask(h: int, w: int) -> np.ndarray:
    total = h * w
    min_area = max(1, int(np.ceil(total * 0.45)))
    area = random.randint(min_area, total)
    cells = _corner_path(h, w)
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in _NEIGHBORS_8:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        cells.add(random.choice(frontier))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _sample_port_rows(top: int, obj_h: int, direction: int, count: int, h: int) -> list[int] | None:
    def rec(curr_top: int, curr_dir: int, remaining: int, chosen: tuple[int, ...]) -> list[int] | None:
        if remaining == 0:
            return []
        curr_bottom = curr_top + obj_h - 1
        if curr_dir < 0:
            candidates = list(range(1, curr_top))
        else:
            candidates = list(range(curr_bottom + 1, h - 1))
        random.shuffle(candidates)
        for port_row in candidates:
            if any(abs(port_row - prev) <= 1 for prev in chosen):
                continue
            next_top = port_row if curr_dir < 0 else port_row - obj_h + 1
            tail = rec(next_top, -curr_dir, remaining - 1, chosen + (port_row,))
            if tail is not None:
                return [int(port_row), *tail]
        return None

    return rec(top, direction, count, ())


def _sample_band_layout(width: int, object_count: int) -> list[tuple[int, int]] | None:
    widths = [random.randint(*_OBJECT_W_RANGE) for _ in range(object_count)]
    min_gap_total = object_count + 1
    extra = width - sum(widths) - min_gap_total
    if extra < 0:
        return None
    gaps = [1] * (object_count + 1)
    for _ in range(extra):
        gaps[random.randrange(len(gaps))] += 1
    bands = []
    cursor = gaps[0]
    for idx, band_w in enumerate(widths):
        bands.append((cursor, cursor + band_w - 1))
        cursor += band_w + gaps[idx + 1]
    return bands


def _make_object_descriptor(
    grid_h: int,
    band: tuple[int, int],
    start_side: str,
    port_count: int,
    top_color: int,
    bottom_color: int,
) -> dict | None:
    band_w = band[1] - band[0] + 1
    for _ in range(80):
        max_h = min(_OBJECT_H_RANGE[1], grid_h - 4)
        if port_count >= 2:
            max_h = min(max_h, 3)
        obj_h = random.randint(_OBJECT_H_RANGE[0], max_h)
        direction = -1 if start_side == "top" else 1
        start_top = random.randint(1, grid_h - 1 - obj_h - 1)
        port_rows = _sample_port_rows(start_top, obj_h, direction, port_count, grid_h)
        if port_rows is None:
            continue
        ports = [
            {
                "row": int(row),
                "col": random.randint(band[0], band[1]),
            }
            for row in port_rows
        ]
        color = top_color if start_side == "top" else bottom_color
        return {
            "band": band,
            "mask": _random_connected_mask(obj_h, band_w),
            "top": int(start_top),
            "color": int(color),
            "ports": ports,
        }
    return None


def _render_input(scene: dict) -> np.ndarray:
    h = int(scene["h"])
    w = int(scene["w"])
    bg = int(scene["bg"])
    top_color = int(scene["top_color"])
    bottom_color = int(scene["bottom_color"])
    port_color = int(scene["port_color"])

    grid = np.full((h, w), bg, dtype=int)
    grid[0, :] = top_color
    grid[h - 1, :] = bottom_color

    for obj in scene["objects"]:
        r0 = int(obj["top"])
        c0 = int(obj["band"][0])
        rr, cc = np.argwhere(obj["mask"]).T
        grid[r0 + rr, c0 + cc] = int(obj["color"])
        for port in obj["ports"]:
            grid[int(port["row"]), int(port["col"])] = port_color

    for r, c in scene["distractor_ports"]:
        grid[int(r), int(c)] = port_color
    return grid


def _expected_from_descriptors(scene: dict) -> np.ndarray:
    h = int(scene["h"])
    w = int(scene["w"])
    bg = int(scene["bg"])
    top_color = int(scene["top_color"])
    bottom_color = int(scene["bottom_color"])
    port_color = int(scene["port_color"])

    out = np.full((h, w), bg, dtype=int)
    out[0, :] = top_color
    out[h - 1, :] = bottom_color

    unused_ports = {(int(r), int(c)) for r, c in scene["distractor_ports"]}
    for obj in scene["objects"]:
        ports = list(obj["ports"])
        current_top = int(obj["top"])
        current_color = int(obj["color"])
        used = set()
        while True:
            bottom = current_top + obj["mask"].shape[0] - 1
            if current_color == top_color:
                ahead = [(port["row"], idx) for idx, port in enumerate(ports) if idx not in used and port["row"] < current_top]
                if ahead:
                    row, idx = max(ahead)
                    current_top = int(row)
                    used.add(int(idx))
                    current_color = bottom_color
                    continue
                current_top = 1
                break
            ahead = [(port["row"], idx) for idx, port in enumerate(ports) if idx not in used and port["row"] > bottom]
            if ahead:
                row, idx = min(ahead)
                current_top = int(row) - obj["mask"].shape[0] + 1
                used.add(int(idx))
                current_color = top_color
                continue
            current_top = h - obj["mask"].shape[0] - 1
            break

        rr, cc = np.argwhere(obj["mask"]).T
        coords_r = current_top + rr
        coords_c = int(obj["band"][0]) + cc
        assert bool(np.all(out[coords_r, coords_c] == bg))
        out[coords_r, coords_c] = current_color
        for idx, port in enumerate(ports):
            if idx not in used:
                unused_ports.add((int(port["row"]), int(port["col"])))

    for r, c in unused_ports:
        assert out[r, c] == bg
        out[r, c] = port_color
    return out


def _pick_distractor_ports(scene: dict, bands: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    h = int(scene["h"])
    w = int(scene["w"])
    occupied_ports = {(int(port["row"]), int(port["col"])) for obj in scene["objects"] for port in obj["ports"]}
    band_cols = {c for c0, c1 in bands for c in range(c0, c1 + 1)}
    free_cols = [c for c in range(w) if c not in band_cols]
    candidates = []
    for r in range(1, h - 1):
        for c in free_cols:
            if any(abs(r - rr) <= 1 and abs(c - cc) <= 1 for rr, cc in occupied_ports):
                continue
            candidates.append((r, c))
    if len(candidates) < 1:
        return None
    random.shuffle(candidates)
    distractors = [candidates[0]]
    for r, c in candidates[1:]:
        if len(distractors) >= random.randint(1, min(3, len(candidates))):
            break
        if all(abs(r - rr) > 1 or abs(c - cc) > 1 for rr, cc in distractors):
            distractors.append((r, c))
    return distractors


def _scene_semantics_ok(scene: dict, grid: np.ndarray, expected: np.ndarray) -> bool:
    port_counts = [len(obj["ports"]) for obj in scene["objects"]]
    if not any(count == 0 for count in port_counts):
        return False
    if not any(count == 1 for count in port_counts):
        return False
    if not any(count >= 2 for count in port_counts):
        return False

    start_colors = [obj["color"] for obj in scene["objects"]]
    if len(set(start_colors)) != 2:
        return False

    parity_by_color: dict[int, set[int]] = {}
    for obj in scene["objects"]:
        parity_by_color.setdefault(int(obj["color"]), set()).add(len(obj["ports"]) % 2)
    if not any(len(parities) >= 2 for parities in parity_by_color.values()):
        return False

    if np.array_equal(grid, expected):
        return False
    input_ports = int(np.count_nonzero(grid == scene["port_color"]))
    output_ports = int(np.count_nonzero(expected == scene["port_color"]))
    if not (0 < output_ports < input_ports):
        return False

    try:
        if not np.array_equal(main(grid), expected):
            return False
    except AssertionError:
        return False

    mut_obj = next((obj for obj in scene["objects"] if obj["ports"]), None)
    if mut_obj is None:
        return False
    mutated = grid.copy()
    first_port = mut_obj["ports"][0]
    mutated[int(first_port["row"]), int(first_port["col"])] = scene["bg"]
    if np.array_equal(main(mutated), expected):
        return False

    mut_obj2 = next((obj for obj in scene["objects"] if len(obj["ports"]) >= 2), None)
    if mut_obj2 is None:
        return False
    mutated2 = grid.copy()
    second_port = mut_obj2["ports"][1]
    mutated2[int(second_port["row"]), int(second_port["col"])] = scene["bg"]
    if np.array_equal(main(mutated2), expected):
        return False

    parsed = _parse_scene(grid)
    if len(parsed["objects"]) != len(scene["objects"]):
        return False
    return True


def _generate_scene() -> dict | None:
    h = random.randint(*_GRID_H_RANGE)
    w = random.randint(*_GRID_W_RANGE)
    bg = random.choice(_PALETTE)
    top_color, bottom_color, port_color = random.sample([c for c in _FG_PALETTE if c != bg], 3)

    object_count = random.randint(*_OBJECT_COUNT_RANGE)
    bands = _sample_band_layout(w, object_count)
    if bands is None:
        return None

    anchor = random.choice(["top", "bottom"])
    other = "bottom" if anchor == "top" else "top"
    specs = [
        (anchor, 0),
        (anchor, 1),
        (other, random.choice([2, 3])),
    ]
    if object_count >= 4:
        specs.append((other, random.choice([0, 1])))
    while len(specs) < object_count:
        specs.append((random.choice(["top", "bottom"]), random.randint(0, 3)))
    random.shuffle(specs)

    objects = []
    for band, (start_side, port_count) in zip(bands, specs):
        obj = _make_object_descriptor(h, band, start_side, port_count, top_color, bottom_color)
        if obj is None:
            return None
        objects.append(obj)

    scene = {
        "h": h,
        "w": w,
        "bg": bg,
        "top_color": top_color,
        "bottom_color": bottom_color,
        "port_color": port_color,
        "objects": objects,
        "distractor_ports": [],
    }
    distractors = _pick_distractor_ports(scene, bands)
    if distractors is None:
        return None
    scene["distractor_ports"] = distractors
    return scene


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        scene = _generate_scene()
        if scene is None:
            continue
        grid = _render_input(scene)
        expected = _expected_from_descriptors(scene)
        if _scene_semantics_ok(scene, grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid port-switch sliding scene")
