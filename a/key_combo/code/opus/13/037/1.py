from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
FG_COLORS = (1, 2, 3, 4)
GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            components.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return components


def _mask_from_component(component: dict) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = component["coords"] - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, component["bbox"]


def _hole_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    bg_mask = ~mask
    reachable = np.zeros_like(bg_mask, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if bg_mask[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg_mask[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bg_mask[nr, nc] and not reachable[nr, nc]:
                reachable[nr, nc] = True
                q.append((nr, nc))

    return bg_mask & ~reachable


def _translate(coords: np.ndarray, dc: int) -> np.ndarray:
    moved = coords.copy()
    moved[:, 1] += int(dc)
    return moved


def _place_open_objects(
    out: np.ndarray, open_components: list[dict], bg: int
) -> list[dict]:
    h, w = out.shape
    placements: list[dict] = []
    ordered = sorted(
        open_components,
        key=lambda comp: (-comp["bbox"][3], -comp["bbox"][1], comp["bbox"][0], comp["bbox"][2]),
    )

    for comp in ordered:
        r0, c0, r1, c1 = comp["bbox"]
        width = c1 - c0 + 1
        placed = None
        for new_c0 in range(w - width, -1, -1):
            shifted = _translate(comp["coords"], new_c0 - c0)
            if np.any(shifted[:, 0] < 0) or np.any(shifted[:, 0] >= h) or np.any(shifted[:, 1] < 0) or np.any(shifted[:, 1] >= w):
                continue
            if np.any(out[shifted[:, 0], shifted[:, 1]] != bg):
                continue
            out[shifted[:, 0], shifted[:, 1]] = comp["color"]
            placed = {
                "input_bbox": comp["bbox"],
                "output_bbox": _bbox(shifted),
                "color": comp["color"],
                "coords": shifted,
                "moved": int(new_c0 != c0),
                "retreated": int(new_c0 + width - 1 < w - 1),
            }
            placements.append(placed)
            break
        if placed is None:
            raise AssertionError("open object could not be placed")
    return placements


def _solve_with_details(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    out = g.copy()
    closed_components: list[dict] = []
    open_components: list[dict] = []

    for comp in _find_components(g, bg):
        mask, bbox = _mask_from_component(comp)
        holes = _hole_mask(mask)
        if np.any(holes):
            out[comp["coords"][:, 0], comp["coords"][:, 1]] = bg
            r0, c0, _, _ = bbox
            hole_coords = np.argwhere(holes) + np.array([r0, c0], dtype=int)
            out[hole_coords[:, 0], hole_coords[:, 1]] = comp["color"]
            closed_components.append(
                {
                    "color": comp["color"],
                    "bbox": bbox,
                    "hole_coords": hole_coords,
                    "input_coords": comp["coords"],
                }
            )
        else:
            out[comp["coords"][:, 0], comp["coords"][:, 1]] = bg
            open_components.append(comp)

    placements = _place_open_objects(out, open_components, bg)
    return out, {"bg": bg, "closed": closed_components, "open": placements, "open_input": open_components}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn hole-containing frames into solid fills and slide hole-free sprites to the right edge.

    Input structure:
    - The grid has a black background and several monochromatic 4-connected foreground objects.
    - Some objects are one-cell-thick hollow rectangular frames whose pixels enclose at least one background hole.
    - The remaining objects are hole-free open sprites, possibly overlapping in row span but separated in the input.

    Rule:
    - Classify each foreground object by topology: objects with an enclosed background hole are closed; all others are open.
    - For each closed object, erase its original pixels and fill every enclosed background hole with that object's own color.
    - For each open object, erase it from its original location and slide it horizontally right so its bounding-box right edge reaches the grid boundary; if that target overlaps already settled content, keep shifting it left until it fits.
    - Process open objects in descending order of their original bounding-box right edge, preserving shape and row position.

    Key insight:
    - The deciding feature is not color, size, or outline style but whether the object truly traps a background region.
    """
    out, _ = _solve_with_details(grid)
    return out


def _normalize(coords: list[tuple[int, int]]) -> np.ndarray:
    arr = np.array(coords, dtype=int)
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    return arr


def _transform_shape(coords: list[tuple[int, int]]) -> np.ndarray:
    arr = _normalize(coords)
    if random.random() < 0.5:
        arr[:, 1] *= -1
        arr = _normalize([tuple(x) for x in arr])
    k = random.randrange(4)
    for _ in range(k):
        arr = np.stack([arr[:, 1], -arr[:, 0]], axis=1)
        arr = _normalize([tuple(x) for x in arr])
    return arr


def _coords_to_patch(coords: np.ndarray, color: int) -> np.ndarray:
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    patch = np.zeros((h, w), dtype=int)
    patch[coords[:, 0], coords[:, 1]] = int(color)
    return patch


OPEN_SHAPES = {
    "solid_2x2": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "solid_2x3": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    "solid_3x2": [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)],
    "l4": [(0, 0), (1, 0), (2, 0), (2, 1)],
    "t5": [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
    "plus5": [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
    "zigzag4": [(0, 0), (0, 1), (1, 1), (1, 2)],
    "stairs5": [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
    "c7": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1), (2, 2)],
    "u7": [(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)],
}


def _make_open_patch(kind: str, color: int) -> np.ndarray:
    coords = _transform_shape(OPEN_SHAPES[kind])
    patch = _coords_to_patch(coords, color)
    mask = patch != BG
    assert np.count_nonzero(mask) == len(OPEN_SHAPES[kind])
    assert not np.any(_hole_mask(mask))
    return patch


def _make_frame_patch(height: int, width: int, color: int) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _can_place(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, col_limit: int) -> bool:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > grid.shape[0] or c0 + w > grid.shape[1]:
        return False
    coords = np.argwhere(patch != BG)
    abs_coords = coords + np.array([r0, c0], dtype=int)
    if int(abs_coords[:, 1].max()) > col_limit:
        return False
    for rr, cc in abs_coords:
        if int(grid[rr, cc]) != BG:
            return False
        for nr, nc in _neighbors4(int(rr), int(cc)):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) != BG:
                return False
    return True


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> np.ndarray:
    h, w = patch.shape
    sub = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    sub[mask] = patch[mask]
    return grid


def _assign_colors(n_closed: int, n_open: int) -> tuple[list[int], list[int]]:
    palette = list(FG_COLORS)
    random.shuffle(palette)
    shared = palette[0]
    remaining = palette[: random.randint(2, 4)]
    closed_colors = [shared]
    open_colors = [shared]
    while len(closed_colors) < n_closed:
        closed_colors.append(random.choice(remaining))
    while len(open_colors) < n_open:
        open_colors.append(random.choice(remaining))
    random.shuffle(closed_colors)
    random.shuffle(open_colors)
    return closed_colors, open_colors


def _build_open_patches(colors: list[int]) -> list[np.ndarray]:
    patches: list[np.ndarray] = []
    required = ["c7", random.choice(["solid_2x2", "solid_2x3", "solid_3x2"])]
    optional_pool = ["u7", "l4", "t5", "plus5", "zigzag4", "stairs5", "solid_2x2", "solid_2x3", "solid_3x2"]
    kinds = required[: len(colors)]
    while len(kinds) < len(colors):
        kinds.append(random.choice(optional_pool))
    random.shuffle(kinds)
    for color, kind in zip(colors, kinds):
        patches.append(_make_open_patch(kind, color))
    return patches


def _build_closed_patches(colors: list[int]) -> list[np.ndarray]:
    patches: list[np.ndarray] = []
    for color in colors:
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        patches.append(_make_frame_patch(h, w, color))
    return patches


def _choose_counts() -> tuple[int, int]:
    total = random.randint(4, 6)
    n_open = random.randint(2, min(4, total - 1))
    n_closed = total - n_open
    return n_closed, n_open


def _place_closed_objects(grid: np.ndarray, patches: list[np.ndarray], col_limit: int) -> bool:
    order = sorted(range(len(patches)), key=lambda i: patches[i].size, reverse=True)
    for idx in order:
        patch = patches[idx]
        h, w = patch.shape
        placed = False
        for _ in range(60):
            r0 = random.randint(0, grid.shape[0] - h)
            c0 = random.randint(0, max(0, col_limit - w + 1))
            if _can_place(grid, patch, r0, c0, col_limit + w - 1):
                _paste(grid, patch, r0, c0)
                placed = True
                break
        if not placed:
            return False
    return True


def _lane_candidates(height: int, patch_h: int, anchor: int) -> list[int]:
    candidates = []
    for delta in (0, -1, 1, -2, 2, -3, 3):
        r0 = anchor + delta
        if 0 <= r0 <= height - patch_h:
            candidates.append(r0)
    if not candidates:
        candidates = list(range(0, height - patch_h + 1))
    random.shuffle(candidates)
    return candidates


def _place_open_objects_initial(grid: np.ndarray, patches: list[np.ndarray], col_limit: int) -> bool:
    h = grid.shape[0]
    if col_limit < 6:
        return False
    lane_count = 1 if len(patches) <= 2 else 2
    lane_anchors = [random.randint(0, max(0, h - 3)) for _ in range(lane_count)]
    edge_pool = list(range(3, col_limit + 1))
    random.shuffle(edge_pool)
    edge_pool = sorted(edge_pool[: len(patches) * 5], reverse=True)
    used_edges: set[int] = set()

    assignments = list(range(len(patches)))
    random.shuffle(assignments)
    for n, patch_idx in enumerate(assignments):
        patch = patches[patch_idx]
        ph, pw = patch.shape
        lane = lane_anchors[0 if n < 2 else random.randrange(lane_count)]
        candidates: list[tuple[int, int]] = []
        for edge in edge_pool:
            if edge in used_edges or edge < pw - 1:
                continue
            c0 = edge - pw + 1
            if c0 < 0:
                continue
            for r0 in _lane_candidates(h, ph, lane):
                candidates.append((r0, c0))
        random.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            if _can_place(grid, patch, r0, c0, col_limit):
                _paste(grid, patch, r0, c0)
                used_edges.add(c0 + pw - 1)
                placed = True
                break
        if not placed:
            return False
    return True


def _sample_quality(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    components = _find_components(grid, BG)
    if not 4 <= len(components) <= 6:
        return False

    details_out, details = _solve_with_details(grid)
    closed = details["closed"]
    open_outputs = details["open"]
    if not closed or len(open_outputs) < 2:
        return False
    if not any(item["moved"] for item in open_outputs):
        return False
    if not any(item["retreated"] for item in open_outputs):
        return False
    if not any(item["output_bbox"][3] == grid.shape[1] - 1 for item in open_outputs):
        return False

    transformed_pixels = np.count_nonzero(grid != details_out)
    if transformed_pixels < 8:
        return False

    hole_areas = [len(item["hole_coords"]) for item in closed]
    if max(hole_areas) < 2:
        return False

    colors = {int(x) for x in grid.ravel() if int(x) != BG}
    if len(colors) < 2:
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(13, 17)
    w = random.randint(18, 24)
    right_blank = random.randint(8, 10)
    col_limit = w - right_blank - 1
    if col_limit < 7:
        return None

    n_closed, n_open = _choose_counts()
    closed_colors, open_colors = _assign_colors(n_closed, n_open)
    grid = np.zeros((h, w), dtype=int)

    closed_patches = _build_closed_patches(closed_colors)
    open_patches = _build_open_patches(open_colors)

    if not _place_closed_objects(grid, closed_patches, col_limit):
        return None
    if not _place_open_objects_initial(grid, open_patches, col_limit):
        return None
    if np.any(grid[:, w - right_blank :] != BG):
        return None
    if not _sample_quality(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
