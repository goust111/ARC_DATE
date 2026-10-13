import importlib.util
from collections import deque
from pathlib import Path

import numpy as np

try:
    # Prefer direct import when running inside existing ARC environments.
    from arc_utils import bbox_coords, components_mask, find_components, get_bg
except Exception:
    # Fallback for direct `SourceFileLoader` execution from arbitrary paths.
    _utils_module = None
    for _parent in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        _candidate = _parent / "new" / "solvers" / "arc_utils.py"
        if _candidate.exists():
            _spec = importlib.util.spec_from_file_location("_arc_utils", str(_candidate))
            if _spec is not None and _spec.loader is not None:
                _utils = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_utils)
                _utils_module = _utils
                break
    if _utils_module is None:
        raise ImportError("arc_utils not found for loading helper functions")

    bbox_coords = _utils_module.bbox_coords
    components_mask = _utils_module.components_mask
    find_components = _utils_module.find_components
    get_bg = _utils_module.get_bg


def _flood_from_edge(grid: np.ndarray, bg: int) -> np.ndarray:
    """
    Return a bool mask marking background cells connected to canvas edges with 4-neighborhood.
    """
    h, w = map(int, grid.shape)
    mask = np.zeros((h, w), dtype=bool)
    q = deque()

    def _push(rr: int, cc: int) -> None:
        if 0 <= rr < h and 0 <= cc < w and (not mask[rr, cc]) and int(grid[rr, cc]) == bg:
            mask[rr, cc] = True
            q.append((rr, cc))

    for c in range(w):
        _push(0, c)
        _push(h - 1, c)
    for r in range(1, h - 1):
        _push(r, 0)
        _push(r, w - 1)

    while q:
        r, c = q.popleft()
        _push(r - 1, c)
        _push(r + 1, c)
        _push(r, c - 1)
        _push(r, c + 1)

    return mask


def _frame_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue

        coords = comp["coords"]
        rows = coords[:, 0]
        cols = coords[:, 1]
        area = int(coords.shape[0])

        on_border = (rows == r0) | (rows == r1) | (cols == c0) | (cols == c1)
        if not bool(np.all(on_border)):
            continue

        if area < 4:
            continue

        max_perimeter = 2 * (h + w) - 4
        if area > max_perimeter:
            continue

        frames.append(
            {
                "color": int(comp["color"]),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "area": int(area),
                "interior_area": int((h - 2) * (w - 2)),
            }
        )

    return frames


def _contains(frame: dict, comp_bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = frame["bbox"]
    cr0, cc0, cr1, cc1 = comp_bbox
    return r0 < cr0 and cr1 < r1 and c0 < cc0 and cc1 < c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each background region not leaked from the canvas edge with the color of its innermost enclosing frame.

    Input structure
    - One dominant background color and one or more single-pixel-thick monochrome rectangular frame borders.
    - Frames may be nested; some may contain background-color gaps on their borders.
    - No other foreground colors/objects exist besides these frame pixels.

    Rule
    - Flood-fill background from all canvas-edge background cells through 4-connectivity; keep all reached background unchanged.
    - Detect background connected components that are not reached from the edge.
    - For each unreached component, choose the smallest-area frame whose interior strictly contains that component (the innermost enclosing frame).
    - Recolor the whole component with that frame's color; leave all original non-background pixels unchanged.

    Key insight
    - Leakage through frame gaps is handled by edge connectivity first; only sealed background pockets are recolored.
    - The color source is determined by the innermost enclosing frame along the nesting chain, so a broken frame yields no independent ownership.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    frames = _frame_candidates(g, bg)
    assert frames, "expected at least one rectangular frame candidate"

    edge_reachable = _flood_from_edge(g, bg)
    bg_mask = g == bg
    out = g.copy()

    for coords in components_mask(bg_mask, connectivity=4):
        if len(coords) == 0:
            continue
        if bool(np.any(edge_reachable[coords[:, 0], coords[:, 1]])):
            continue

        comp_bbox = bbox_coords(coords)
        containers = [f for f in frames if _contains(f, comp_bbox)]
        assert containers, f"unreached background component not inside any frame: {comp_bbox}"

        owner = min(containers, key=lambda f: int(f["interior_area"]))
        out[coords[:, 0], coords[:, 1]] = owner["color"]

    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _apply_gaps(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, bg: int, n_gaps: int) -> None:
    perimeter = set()
    for c in range(c0, c1 + 1):
        perimeter.add((r0, c))
        perimeter.add((r1, c))
    for r in range(r0 + 1, r1):
        perimeter.add((r, c0))
        perimeter.add((r, c1))

    perimeter = list(perimeter)
    if not perimeter:
        return

    n_gaps = int(min(len(perimeter), max(1, n_gaps)))
    for idx in np.random.choice(len(perimeter), size=n_gaps, replace=False):
        rr, cc = perimeter[int(idx)]
        grid[int(rr), int(cc)] = bg


def _gen_nested_frames(h: int, w: int, n: int, colors: list[int]) -> list[tuple[int, int, int, int, int]]:
    outer_h = np.random.randint(max(6, min(10, h - 2)) + 1)
    outer_w = np.random.randint(max(6, min(10, w - 2)) + 1)
    r0 = int(np.random.randint(1, h - outer_h))
    c0 = int(np.random.randint(1, w - outer_w))
    frames = [(r0, c0, r0 + outer_h - 1, c0 + outer_w - 1, colors[0])]

    while len(frames) < n:
        pr0, pc0, pr1, pc1, pcolor = frames[-1]
        ph = pr1 - pr0 + 1
        pw = pc1 - pc0 + 1
        max_in_h = ph - 4
        max_in_w = pw - 4
        if max_in_h < 3 or max_in_w < 3:
            break

        h_new = int(np.random.randint(3, max_in_h + 1))
        w_new = int(np.random.randint(3, max_in_w + 1))
        r0_new = int(np.random.randint(pr0 + 2, pr1 - h_new))
        c0_new = int(np.random.randint(pc0 + 2, pc1 - w_new))
        r1_new = r0_new + h_new - 1
        c1_new = c0_new + w_new - 1
        color = colors[len(frames)]
        frames.append((r0_new, c0_new, r1_new, c1_new, color))

    return frames


def generate_input() -> np.ndarray:
    """
    Random ARC input generator with nested frames and optional frame gaps.

    It creates scenes where leaked regions (edge-connected background) stay at background color
    while sealed background pockets are recolored by the innermost enclosing frame color.
    """
    for _ in range(300):
        h = int(np.random.randint(12, 31))
        w = int(np.random.randint(12, 31))
        bg = int(np.random.choice([0, 1, 2, 3, 4]))
        while True:
            frame_count = int(np.random.randint(1, 5))
            palette = [c for c in range(10) if c != bg]
            np.random.shuffle(palette)
            colors = palette[:frame_count]
            frames = _gen_nested_frames(h, w, frame_count, colors)
            if len(frames) >= 1 and len(frames) <= frame_count:
                break

        grid = np.full((h, w), bg, dtype=int)
        for r0, c0, r1, c1, color in frames:
            _draw_frame(grid, r0, c0, r1, c1, color)

        broken = [np.random.random() < 0.52 for _ in frames]
        if all(broken):
            broken[np.random.randint(len(broken))] = False

        for i, (r0, c0, r1, c1, color) in enumerate(frames):
            if not broken[i]:
                continue
            side_len = 2 * (r1 - r0 + 1 + c1 - c0 + 1) - 4
            max_gap = min(3, max(1, side_len // 8), side_len)
            n_gaps = int(np.random.randint(1, max_gap + 1))
            _apply_gaps(grid, r0, c0, r1, c1, bg, n_gaps)

        try:
            if np.array_equal(main(grid), grid):
                # all regions leaked, too trivial for this rule
                continue
            return grid
        except Exception:
            continue

    raise RuntimeError("failed to generate a valid 6.py input")
