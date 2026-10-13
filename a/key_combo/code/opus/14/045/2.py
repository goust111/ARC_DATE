import hashlib
from collections import deque

import numpy as np


COLOR_POOL = (1, 2, 3, 6, 7)
MAX_GENERATION_ATTEMPTS = 600
_SEEN_OUTPUT_SIGNATURES: set[str] = set()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                }
            )
    return components


def _has_enclosed_black_region(region: np.ndarray) -> bool:
    h, w = region.shape
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if int(region[r, c]) != 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_border = r in (0, h - 1) or c in (0, w - 1)
            while q:
                cr, cc = q.popleft()
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(region[nr, nc]) == 0:
                        seen[nr, nc] = True
                        touches_border |= nr in (0, h - 1) or nc in (0, w - 1)
                        q.append((nr, nc))
            if not touches_border:
                return True
    return False


def _is_frame_candidate(region: np.ndarray, color: int) -> bool:
    h, w = region.shape
    if h != w or h not in (5, 6):
        return False
    if not np.all(region[0, :] == color):
        return False
    if not np.all(region[-1, :] == color):
        return False
    if not np.all(region[:, 0] == color):
        return False
    if not np.all(region[:, -1] == color):
        return False
    return _has_enclosed_black_region(region)


def _extract_pattern(region: np.ndarray, color: int) -> np.ndarray:
    inner = region[1:-1, 1:-1]
    pattern = (inner == color).astype(int)
    n = pattern.shape[0]
    assert inner.shape == (n, n)
    assert n in (3, 4)
    return pattern


def _build_output(pattern: np.ndarray, color: int) -> np.ndarray:
    n = pattern.shape[0]
    sprite = pattern * int(color)
    out = np.zeros((n * n, n * n), dtype=int)
    for r in range(n):
        for c in range(n):
            if int(pattern[r, c]) == 1:
                out[r * n : (r + 1) * n, c * n : (c + 1) * n] = sprite
    return out


def _select_frame(grid: np.ndarray) -> tuple[dict, np.ndarray]:
    candidates = []
    for comp in _find_components(grid):
        r0, c0, r1, c1 = comp["bbox"]
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        color = int(comp["color"])
        if _is_frame_candidate(region, color):
            candidates.append((comp, region))
    assert len(candidates) == 1, "expected exactly one hollow frame"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Extract the unique hole-bearing frame's interior pattern and self-stamp it into a larger block grid.

    Input structure:
    - The grid has a black background and 3-5 separated single-color 4-connected foreground objects.
    - Exactly one object is a complete 1-pixel rectangular frame whose square interior is 3x3 or 4x4.
    - Inside that frame, some cells are the frame color and some are black, forming a binary seed pattern.

    Rule:
    - Find the unique foreground object whose bounding box is a complete square frame and whose bbox contains a 4-connected black region that does not touch the bbox border.
    - Crop its interior square and convert it to a binary pattern P where the frame color is 1 and black is 0.
    - Create a black output of size (N*N) x (N*N), where N is the side length of P.
    - For each foreground cell of P, copy the whole pattern P into the corresponding NxN output block using the frame color; background cells of P leave their blocks black.

    Key insight:
    - The frame's interior pattern serves two roles at once: it is both the sprite being copied and the mask deciding which blocks receive a copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comp, region = _select_frame(g)
    color = int(comp["color"])
    pattern = _extract_pattern(region, color)
    return _build_output(pattern, color)


def _connected_pattern(rng: np.random.Generator, n: int) -> np.ndarray:
    for _ in range(200):
        target = int(rng.integers(2, n * n - 1))
        boundary_cells = [(r, c) for r in range(n) for c in range(n) if r in (0, n - 1) or c in (0, n - 1)]
        start = boundary_cells[int(rng.integers(len(boundary_cells)))]
        fg = {start}
        frontier = set()
        for nr, nc in _neighbors4(*start):
            if 0 <= nr < n and 0 <= nc < n:
                frontier.add((nr, nc))
        while len(fg) < target and frontier:
            choices = list(frontier)
            cell = choices[int(rng.integers(len(choices)))]
            frontier.remove(cell)
            fg.add(cell)
            for nr, nc in _neighbors4(*cell):
                if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in fg:
                    frontier.add((nr, nc))
        if len(fg) != target:
            continue
        pattern = np.zeros((n, n), dtype=int)
        for r, c in fg:
            pattern[r, c] = 1
        zeros = n * n - target
        rows_with_fg = sum(int(pattern[r, :].any()) for r in range(n))
        cols_with_fg = sum(int(pattern[:, c].any()) for c in range(n))
        mixed_rows = sum(0 < int(pattern[r, :].sum()) < n for r in range(n))
        mixed_cols = sum(0 < int(pattern[:, c].sum()) < n for c in range(n))
        if zeros < 2:
            continue
        if rows_with_fg < 2 or cols_with_fg < 2:
            continue
        if mixed_rows == 0 or mixed_cols == 0:
            continue
        return pattern
    raise RuntimeError("failed to generate a non-degenerate connected pattern")


def _make_frame_patch(color: int, pattern: np.ndarray) -> np.ndarray:
    n = pattern.shape[0]
    patch = np.zeros((n + 2, n + 2), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    patch[1:-1, 1:-1][pattern == 1] = color
    return patch


def _make_open_frame_patch(color: int, rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    side = int(rng.integers(4))
    if side == 0:
        gap_c = int(rng.integers(1, w - 1))
        patch[0, gap_c] = 0
    elif side == 1:
        gap_c = int(rng.integers(1, w - 1))
        patch[-1, gap_c] = 0
    elif side == 2:
        gap_r = int(rng.integers(1, h - 1))
        patch[gap_r, 0] = 0
    else:
        gap_r = int(rng.integers(1, h - 1))
        patch[gap_r, -1] = 0
    return patch


def _make_filled_rect_patch(color: int, rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(2, 6))
    w = int(rng.integers(2, 6))
    return np.full((h, w), color, dtype=int)


def _make_l_patch(color: int, rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(3, 6))
    w = int(rng.integers(3, 6))
    thickness = int(rng.integers(1, min(h, w)))
    patch = np.zeros((h, w), dtype=int)
    orientation = int(rng.integers(4))
    if orientation == 0:
        patch[:, :thickness] = color
        patch[-thickness:, :] = color
    elif orientation == 1:
        patch[:, -thickness:] = color
        patch[-thickness:, :] = color
    elif orientation == 2:
        patch[:, :thickness] = color
        patch[:thickness, :] = color
    else:
        patch[:, -thickness:] = color
        patch[:thickness, :] = color
    return patch


def _make_bar_patch(color: int, rng: np.random.Generator) -> np.ndarray:
    if int(rng.integers(2)) == 0:
        return np.full((1, int(rng.integers(3, 7))), color, dtype=int)
    return np.full((int(rng.integers(3, 7)), 1), color, dtype=int)


def _rectangles_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_patches(
    rng: np.random.Generator, shape: tuple[int, int], patches: list[dict]
) -> list[dict] | None:
    h, w = shape
    placed = []
    occupied_boxes: list[tuple[int, int, int, int]] = []
    ordered = sorted(patches, key=lambda item: item["patch"].shape[0] * item["patch"].shape[1], reverse=True)
    for item in ordered:
        patch = item["patch"]
        ph, pw = patch.shape
        positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        rng.shuffle(positions)
        found = None
        for r0, c0 in positions:
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if any(_rectangles_conflict(box, other, gap=1) for other in occupied_boxes):
                continue
            found = box
            placed.append({**item, "origin": (r0, c0), "bbox": box})
            occupied_boxes.append(box)
            break
        if found is None:
            return None
    return placed


def _render(shape: tuple[int, int], placed: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for item in placed:
        patch = item["patch"]
        r0, c0 = item["origin"]
        ph, pw = patch.shape
        view = grid[r0 : r0 + ph, c0 : c0 + pw]
        mask = patch != 0
        assert not np.any(view[mask] != 0)
        view[mask] = patch[mask]
    return grid


def _has_open_frame_distractor(grid: np.ndarray, selected_bbox: tuple[int, int, int, int]) -> bool:
    for comp in _find_components(grid):
        if comp["bbox"] == selected_bbox:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        color = int(comp["color"])
        if region.shape[0] < 4 or region.shape[1] < 4:
            continue
        border_count = (
            int(np.sum(region[0, :] == color))
            + int(np.sum(region[-1, :] == color))
            + int(np.sum(region[1:-1, 0] == color))
            + int(np.sum(region[1:-1, -1] == color))
        )
        border_total = 2 * region.shape[0] + 2 * region.shape[1] - 4
        if border_count >= border_total - 1 and not _has_enclosed_black_region(region):
            return True
    return False


def _output_signature(grid: np.ndarray) -> str:
    out = main(grid)
    return hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()


def _validate_constructed_sample(
    grid: np.ndarray, pattern: np.ndarray, frame_color: int, frame_outer: int
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    components = _find_components(grid)
    if not (3 <= len(components) <= 5):
        return False
    colors = [int(comp["color"]) for comp in components]
    if len(colors) != len(set(colors)):
        return False
    if not set(colors).issubset(set(COLOR_POOL)):
        return False
    comp, region = _select_frame(grid)
    recovered = _extract_pattern(region, int(comp["color"]))
    if int(comp["color"]) != frame_color:
        return False
    if region.shape != (frame_outer, frame_outer):
        return False
    if not np.array_equal(recovered, pattern):
        return False
    if not np.array_equal(main(grid), _build_output(pattern, frame_color)):
        return False
    if not _has_open_frame_distractor(grid, comp["bbox"]):
        return False
    frame_area = frame_outer * frame_outer
    other_box_areas = [
        (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
        for bbox in [c["bbox"] for c in components if c["bbox"] != comp["bbox"]]
    ]
    if not any(area >= frame_area for area in other_box_areas):
        return False
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) == 0:
                continue
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != 0 and int(grid[nr, nc]) != int(grid[r, c]):
                    return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    grid_h = int(rng.integers(13, 17))
    grid_w = int(rng.integers(13, 17))
    object_count = int(rng.integers(3, 6))
    colors = list(rng.choice(np.array(COLOR_POOL), size=object_count, replace=False))

    n = 3 if rng.random() < 0.5 else 4
    frame_color = int(colors[0])
    pattern = _connected_pattern(rng, n)
    frame_patch = _make_frame_patch(frame_color, pattern)
    frame_outer = n + 2

    patches = [
        {"kind": "frame", "patch": frame_patch, "color": frame_color},
        {
            "kind": "open_frame",
            "patch": _make_open_frame_patch(int(colors[1]), rng, frame_outer, frame_outer),
            "color": int(colors[1]),
        },
    ]

    for color in colors[2:]:
        kind_roll = float(rng.random())
        if kind_roll < 0.4:
            patch = _make_filled_rect_patch(int(color), rng)
        elif kind_roll < 0.75:
            patch = _make_l_patch(int(color), rng)
        else:
            patch = _make_bar_patch(int(color), rng)
        patches.append({"kind": "solid", "patch": patch, "color": int(color)})

    placed = _place_patches(rng, (grid_h, grid_w), patches)
    if placed is None:
        return None

    grid = _render((grid_h, grid_w), placed)
    if not _validate_constructed_sample(grid, pattern, frame_color, frame_outer):
        return None
    sig = _output_signature(grid)
    if sig in _SEEN_OUTPUT_SIGNATURES:
        return None
    _SEEN_OUTPUT_SIGNATURES.add(sig)
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
