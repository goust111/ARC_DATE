from __future__ import annotations

import random

import numpy as np


BG = 0
NON_BG_COLORS = tuple(range(1, 10))

GRID_HEIGHT_RANGE = (18, 30)
GRID_WIDTH_RANGE = (18, 30)
STRUCTURE_COUNT_CHOICES = (1, 2)
SINGLE_STRUCTURE_FRAME_RANGE = (1, 4)
MULTI_STRUCTURE_FRAME_RANGE = (1, 3)
CORE_HEIGHT_RANGE = (1, 5)
CORE_WIDTH_RANGE = (1, 5)
GAP_RANGE = (1, 4)
PLACEMENT_PAD = 2
GENERATION_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[counts.argmax()])


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            pts = np.asarray(coords, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": pts,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )
    return components


def _is_solid_rect(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    return int(comp["area"]) == area and bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color))


def _is_rect_frame(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    color = int(comp["color"])
    sub = grid[r0 : r1 + 1, c0 : c1 + 1] == color
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(sub, expected))


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)


def _parse_objects(grid: np.ndarray, bg: int) -> list[dict]:
    objects: list[dict] = []
    for comp in _components_by_color(grid, bg):
        if _is_rect_frame(grid, comp):
            kind = "frame"
        elif _is_solid_rect(grid, comp):
            kind = "solid"
        else:
            raise AssertionError("all non-background components must be rectangles or rectangular frames")
        comp = dict(comp)
        comp["kind"] = kind
        objects.append(comp)
    return objects


def _parent_frame_indices(objects: list[dict]) -> list[int | None]:
    frames = [i for i, obj in enumerate(objects) if obj["kind"] == "frame"]
    parents: list[int | None] = [None] * len(objects)

    for i, obj in enumerate(objects):
        containing = [j for j in frames if j != i and _strictly_contains(objects[j]["bbox"], obj["bbox"])]
        if containing:
            parents[i] = min(containing, key=lambda j: _bbox_area(objects[j]["bbox"]))
    return parents


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify nested hollow rectangular frames and a solid core rectangle.
    Fill each gap outward from the core, switching the fill color whenever a
    frame border is crossed. Frame pixels stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    objects = _parse_objects(g, bg)
    parents = _parent_frame_indices(objects)

    cores = [i for i, obj in enumerate(objects) if obj["kind"] == "solid" and parents[i] is not None]
    assert cores, "expected at least one solid core inside a frame"

    out = g.copy()
    bg_mask = g == bg
    for core_idx in cores:
        chain = [core_idx]
        while parents[chain[-1]] is not None:
            chain.append(parents[chain[-1]])

        for inner_idx, outer_idx in zip(chain, chain[1:], strict=False):
            inner = objects[inner_idx]
            outer = objects[outer_idx]
            ir0, ic0, ir1, ic1 = inner["bbox"]
            or0, oc0, or1, oc1 = outer["bbox"]

            gap = np.zeros_like(g, dtype=bool)
            gap[or0 + 1 : or1, oc0 + 1 : oc1] = True
            gap[ir0 : ir1 + 1, ic0 : ic1 + 1] = False
            out[gap & bg_mask] = int(inner["color"])

    return out


def _draw_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)


def _build_structure_patch(core_color: int, frame_colors: list[int]) -> np.ndarray:
    core_h = random.randint(*CORE_HEIGHT_RANGE)
    core_w = random.randint(*CORE_WIDTH_RANGE)
    boxes: list[tuple[str, int, tuple[int, int, int, int]]] = [
        ("solid", int(core_color), (0, 0, core_h - 1, core_w - 1))
    ]
    current_bbox = boxes[0][2]

    for color in frame_colors:
        top_gap = random.randint(*GAP_RANGE)
        bottom_gap = random.randint(*GAP_RANGE)
        left_gap = random.randint(*GAP_RANGE)
        right_gap = random.randint(*GAP_RANGE)
        r0, c0, r1, c1 = current_bbox
        current_bbox = (
            r0 - top_gap - 1,
            c0 - left_gap - 1,
            r1 + bottom_gap + 1,
            c1 + right_gap + 1,
        )
        boxes.append(("frame", int(color), current_bbox))

    min_r = min(bbox[0] for _, _, bbox in boxes)
    min_c = min(bbox[1] for _, _, bbox in boxes)
    max_r = max(bbox[2] for _, _, bbox in boxes)
    max_c = max(bbox[3] for _, _, bbox in boxes)
    patch = np.full((max_r - min_r + 1, max_c - min_c + 1), BG, dtype=int)

    for kind, color, bbox in boxes:
        shifted = (bbox[0] - min_r, bbox[1] - min_c, bbox[2] - min_r, bbox[3] - min_c)
        if kind == "solid":
            _draw_rect(patch, shifted, color)
        else:
            _draw_frame(patch, shifted, color)
    return patch


def _candidate_locations(occupied: np.ndarray, patch_shape: tuple[int, int], pad: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    ph, pw = patch_shape
    locations: list[tuple[int, int]] = []
    if ph > h or pw > w:
        return locations

    for r0 in range(h - ph + 1):
        r1 = r0 + ph
        rr0 = max(0, r0 - pad)
        rr1 = min(h, r1 + pad)
        for c0 in range(w - pw + 1):
            c1 = c0 + pw
            cc0 = max(0, c0 - pad)
            cc1 = min(w, c1 + pad)
            if not occupied[rr0:rr1, cc0:cc1].any():
                locations.append((r0, c0))
    return locations


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    ph, pw = patch.shape
    mask = patch != BG
    grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]


def _sample_frame_counts(structure_count: int) -> list[int]:
    frame_counts: list[int] = []
    for _ in range(structure_count):
        if structure_count == 1:
            frame_counts.append(random.randint(*SINGLE_STRUCTURE_FRAME_RANGE))
        else:
            frame_counts.append(random.randint(*MULTI_STRUCTURE_FRAME_RANGE))
    return frame_counts


def _build_candidate_grid() -> np.ndarray | None:
    rows = random.randint(*GRID_HEIGHT_RANGE)
    cols = random.randint(*GRID_WIDTH_RANGE)
    structure_count = random.choice(STRUCTURE_COUNT_CHOICES)
    frame_counts = _sample_frame_counts(structure_count)
    colors_needed = sum(count + 1 for count in frame_counts)
    if colors_needed > len(NON_BG_COLORS):
        return None

    sampled_colors = random.sample(NON_BG_COLORS, colors_needed)
    patches: list[np.ndarray] = []
    cursor = 0
    for frame_count in frame_counts:
        chain_colors = sampled_colors[cursor : cursor + frame_count + 1]
        cursor += frame_count + 1
        patch = _build_structure_patch(chain_colors[0], chain_colors[1:])
        if patch.shape[0] > rows or patch.shape[1] > cols:
            return None
        patches.append(patch)

    patches.sort(key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True)
    grid = np.full((rows, cols), BG, dtype=int)
    occupied = np.zeros((rows, cols), dtype=bool)
    for patch in patches:
        locations = _candidate_locations(occupied, patch.shape, PLACEMENT_PAD)
        if not locations:
            return None
        r0, c0 = random.choice(locations)
        _paste_patch(grid, patch, r0, c0)
        occupied[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] |= patch != BG
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_candidate_grid()
        if grid is None:
            continue
        out = main(grid)
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
