from __future__ import annotations

import numpy as np


BG = 0
PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_ATTEMPTS = 500
WORK_SIZE = 30


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    mask = grid == color
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _frame_from_component(coords: list[tuple[int, int]], color: int) -> dict[str, object] | None:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3 or len(coords) != 2 * h + 2 * w - 4:
        return None
    coord_set = set(coords)
    for cc in range(c0, c1 + 1):
        if (r0, cc) not in coord_set or (r1, cc) not in coord_set:
            return None
    for rr in range(r0, r1 + 1):
        if (rr, c0) not in coord_set or (rr, c1) not in coord_set:
            return None
    return {"color": int(color), "bbox": (int(r0), int(c0), int(r1), int(c1))}


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in sorted(set(map(int, grid.flat)) - {int(bg)}):
        for coords in _components_of_color(grid, int(color)):
            frame = _frame_from_component(coords, int(color))
            if frame is not None:
                frames.append(frame)
    return frames


def _extract_pattern_patch(
    grid: np.ndarray, frame_bbox: tuple[int, int, int, int], frame_color: int, bg: int
) -> np.ndarray | None:
    r0, c0, r1, c1 = frame_bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = (inner != bg) & (inner != frame_color)
    if not mask.any():
        return None
    coords = np.argwhere(mask)
    pr0, pc0 = coords.min(axis=0)
    pr1, pc1 = coords.max(axis=0)
    patch = np.full((pr1 - pr0 + 1, pc1 - pc0 + 1), bg, dtype=int)
    patch_mask = mask[pr0 : pr1 + 1, pc0 : pc1 + 1]
    patch_vals = inner[pr0 : pr1 + 1, pc0 : pc1 + 1]
    patch[patch_mask] = patch_vals[patch_mask]
    return patch


def _shrink_once(patch: np.ndarray, bg: int) -> np.ndarray:
    h, w = patch.shape
    out = np.full(((h + 1) // 2, (w + 1) // 2), bg, dtype=int)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            block = patch[2 * r : min(h, 2 * r + 2), 2 * c : min(w, 2 * c + 2)]
            vals = block[block != bg]
            if vals.size == 0:
                continue
            colors, counts = np.unique(vals, return_counts=True)
            out[r, c] = int(colors[np.argmax(counts)])
    return out


def _paste_on_bg(out: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = out.shape
    for r in range(patch.shape[0]):
        rr = top + r
        if not (0 <= rr < h):
            continue
        for c in range(patch.shape[1]):
            cc = left + c
            if not (0 <= cc < w):
                continue
            value = int(patch[r, c])
            if value != bg and int(out[rr, cc]) == bg:
                out[rr, cc] = value


def _structure_chains(frames: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    keyed = {
        (int(frame["color"]), tuple(frame["bbox"])): idx
        for idx, frame in enumerate(frames)
    }
    outer_of: dict[int, int] = {}
    inner_of: dict[int, int] = {}
    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        key = (int(frame["color"]), (r0 - 2, c0 - 2, r1 + 2, c1 + 2))
        if key in keyed:
            outer = keyed[key]
            outer_of[idx] = int(outer)
            inner_of[int(outer)] = idx

    innermost = [idx for idx in range(len(frames)) if idx not in inner_of]
    chains: list[list[dict[str, object]]] = []
    seen: set[int] = set()
    for start in innermost:
        if start in seen:
            continue
        chain: list[dict[str, object]] = []
        cur = start
        while True:
            chain.append(frames[cur])
            seen.add(cur)
            if cur not in outer_of:
                break
            cur = outer_of[cur]
        chains.append(chain)
    return chains


def main(grid: np.ndarray) -> np.ndarray:
    """Replicate the innermost sprite outward once per nested frame layer.

    Input structure:
    - Background is black.
    - Each structure contains one to three same-colored hollow rectangular frames nested with a one-cell black gap between consecutive layers.
    - The innermost frame encloses a smaller monochrome pattern drawn in a different color.

    Rule:
    - Keep every frame and the original inner pattern unchanged.
    - Let L be the number of nested frames in a structure.
    - Crop the inner pattern to its tight bounding box. A copy that moves one layer outward is shrunk by 2x2 block pooling: an output cell takes the pattern color when any source cell in that block is colored.
    - For each gap between layer k and layer k+1, place the k-times-shrunk copy with its top-left corner at the top-left black cell just inside layer k+1; only black cells are painted, so frames remain intact.
    - Place the L-times-shrunk copy once more just outside the outermost frame at its lower-right corner, clipping any part that falls outside the grid.

    Key insight:
    - The frame depth is a hidden counter: it determines both how many extra copies appear and how many times the inner pattern must be shrunk before each outward placement.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(BG if np.any(g == BG) else np.bincount(g.ravel()).argmax())
    out = g.copy()

    frames = _find_frames(g, bg)
    for chain in _structure_chains(frames):
        inner = chain[0]
        patch = _extract_pattern_patch(g, inner["bbox"], int(inner["color"]), bg)
        if patch is None:
            continue

        scaled = patch.copy()
        for layer_idx in range(1, len(chain)):
            scaled = _shrink_once(scaled, bg)
            outer_bbox = chain[layer_idx]["bbox"]
            _paste_on_bg(out, scaled, outer_bbox[0] + 1, outer_bbox[1] + 1, bg)

        scaled = _shrink_once(scaled, bg)
        outermost = chain[-1]["bbox"]
        _paste_on_bg(out, scaled, outermost[2] + 1, outermost[3] + 1, bg)

    return out


def _grow_connected_cells(
    rng: np.random.Generator, height: int, width: int, area: int
) -> set[tuple[int, int]]:
    seed = (int(rng.integers(0, height)), int(rng.integers(0, width)))
    cells = {seed}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(frontier[int(rng.integers(0, len(frontier)))])
    return cells


def _is_border_only(mask: np.ndarray) -> bool:
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return False
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(mask, border)


def _make_pattern_patch(rng: np.random.Generator, color: int, bg: int) -> np.ndarray | None:
    for _ in range(80):
        h = int(rng.integers(2, 5))
        w = int(rng.integers(2, 5))
        area = int(rng.integers(max(2, (h * w + 1) // 2), h * w + 1))
        cells = _grow_connected_cells(rng, h, w, area)
        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        for r, c in cells:
            patch[r - r0, c - c0] = color
        if patch.shape[0] < 2 or patch.shape[1] < 2:
            continue
        if _is_border_only(patch != bg):
            continue
        return patch
    return None


def _draw_frame(patch: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    patch[r0, c0 : c1 + 1] = color
    patch[r1, c0 : c1 + 1] = color
    patch[r0 : r1 + 1, c0] = color
    patch[r0 : r1 + 1, c1] = color


def _build_structure(
    rng: np.random.Generator, frame_color: int, pattern_color: int, layers: int, bg: int
) -> dict[str, object] | None:
    pattern = _make_pattern_patch(rng, pattern_color, bg)
    if pattern is None:
        return None

    room_h = pattern.shape[0] + int(rng.integers(1, 4))
    room_w = pattern.shape[1] + int(rng.integers(1, 4))
    inner_h = room_h + 2
    inner_w = room_w + 2
    outer_h = inner_h + 4 * (layers - 1)
    outer_w = inner_w + 4 * (layers - 1)

    patch = np.full((outer_h, outer_w), bg, dtype=int)
    for depth in range(layers):
        inset = 2 * depth
        bbox = (inset, inset, outer_h - 1 - inset, outer_w - 1 - inset)
        _draw_frame(patch, bbox, frame_color)

    inner_inset = 2 * (layers - 1)
    room_top = inner_inset + 1
    room_left = inner_inset + 1
    pr = room_top + int(rng.integers(0, room_h - pattern.shape[0] + 1))
    pc = room_left + int(rng.integers(0, room_w - pattern.shape[1] + 1))
    pattern_mask = pattern != bg
    patch[pr : pr + pattern.shape[0], pc : pc + pattern.shape[1]][pattern_mask] = pattern[pattern_mask]

    outside = pattern.copy()
    for _ in range(layers):
        outside = _shrink_once(outside, bg)

    reserve_h = outer_h + outside.shape[0]
    reserve_w = outer_w + outside.shape[1]
    if reserve_h > WORK_SIZE or reserve_w > WORK_SIZE:
        return None

    return {
        "patch": patch,
        "input_shape": patch.shape,
        "reserve_shape": (reserve_h, reserve_w),
    }


def _rects_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int
) -> bool:
    return not (
        a[2] + padding < b[0]
        or b[2] + padding < a[0]
        or a[3] + padding < b[1]
        or b[3] + padding < a[1]
    )


def _place_structures(
    rng: np.random.Generator, structures: list[dict[str, object]], bg: int
) -> np.ndarray | None:
    grid = np.full((WORK_SIZE, WORK_SIZE), bg, dtype=int)
    occupied: list[tuple[int, int, int, int]] = []
    ordered = sorted(
        structures,
        key=lambda item: item["reserve_shape"][0] * item["reserve_shape"][1],
        reverse=True,
    )

    for structure in ordered:
        reserve_h, reserve_w = structure["reserve_shape"]
        input_h, input_w = structure["input_shape"]
        max_r = WORK_SIZE - reserve_h
        max_c = WORK_SIZE - reserve_w
        if max_r < 0 or max_c < 0:
            return None

        placed = False
        for _ in range(200):
            top = int(rng.integers(0, max_r + 1))
            left = int(rng.integers(0, max_c + 1))
            reserve_rect = (top, left, top + reserve_h - 1, left + reserve_w - 1)
            if any(_rects_overlap(reserve_rect, other, padding=1) for other in occupied):
                continue
            patch = structure["patch"]
            grid[top : top + input_h, left : left + input_w] = patch
            occupied.append(reserve_rect)
            placed = True
            break
        if not placed:
            return None

    return grid


def _crop_to_input_extent(rng: np.random.Generator, grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != bg)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top = max(0, int(r0) - int(rng.integers(0, 3)))
    left = max(0, int(c0) - int(rng.integers(0, 3)))
    bottom = min(WORK_SIZE - 1, int(r1) + int(rng.integers(0, 4)))
    right = min(WORK_SIZE - 1, int(c1) + int(rng.integers(0, 4)))
    return grid[top : bottom + 1, left : right + 1]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    bg = int(BG)
    n_structures = int(rng.integers(1, 4))
    layers_list = [int(rng.integers(1, 4)) for _ in range(n_structures)]
    color_count = 2 * n_structures
    if color_count > len(PALETTE):
        return None
    colors = [int(v) for v in rng.choice(PALETTE, size=color_count, replace=False)]

    structures: list[dict[str, object]] = []
    for idx, layers in enumerate(layers_list):
        structure = _build_structure(
            rng,
            frame_color=colors[2 * idx],
            pattern_color=colors[2 * idx + 1],
            layers=layers,
            bg=bg,
        )
        if structure is None:
            return None
        structures.append(structure)

    full = _place_structures(rng, structures, bg)
    if full is None:
        return None

    cropped = _crop_to_input_extent(rng, full, bg)
    if max(cropped.shape) > 30:
        return None
    out = main(cropped)
    if out.shape != cropped.shape:
        return None
    if np.array_equal(out, cropped):
        return None
    return cropped


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    bg = int(BG)
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        out = main(grid)
        if grid.ndim == 2 and out.ndim == 2 and max(grid.shape) <= 30 and max(out.shape) <= 30:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
