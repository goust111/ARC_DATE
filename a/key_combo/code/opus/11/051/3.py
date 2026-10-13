import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
MARKER = 7
PALETTE_COLORS = (1, 2, 3, 4)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _frame_mask_from_bbox(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _extract_frames(grid: np.ndarray) -> list[dict]:
    mask = np.isin(grid, (FRAME, MARKER))
    comps = _components(mask)
    frames = []
    for pts in comps:
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must have a non-empty interior"
        expected = _frame_mask_from_bbox(grid.shape, (r0, c0, r1, c1))
        got = np.zeros(grid.shape, dtype=bool)
        got[pts[:, 0], pts[:, 1]] = True
        assert np.array_equal(expected, got), "component must be a one-pixel rectangular frame"
        marker_pts = [(int(r), int(c)) for r, c in np.argwhere((grid == MARKER) & expected)]
        frames.append({"bbox": (r0, c0, r1, c1), "mask": expected, "markers": marker_pts})
    assert frames, "expected at least one nested frame"
    return frames


def _depth_map(frames: list[dict]) -> dict[tuple[int, int, int, int], int]:
    bboxes = [frame["bbox"] for frame in frames]
    depth_of = {}
    for bbox in bboxes:
        r0, c0, r1, c1 = bbox
        depth = 1
        for other in bboxes:
            if other == bbox:
                continue
            or0, oc0, or1, oc1 = other
            if or0 < r0 and oc0 < c0 and or1 > r1 and oc1 > c1:
                depth += 1
        depth_of[bbox] = depth
    assert sorted(depth_of.values()) == list(range(1, len(frames) + 1)), "frames must form a strict nesting chain"
    return depth_of


def _read_palette(grid: np.ndarray) -> list[int]:
    palette = []
    i = 0
    h, w = grid.shape
    while i < h and i < w and grid[i, i] != BG:
        palette.append(int(grid[i, i]))
        i += 1
    assert palette, "expected a diagonal palette from the top-left corner"
    return palette


def _marker_side(bbox: tuple[int, int, int, int], marker: tuple[int, int]) -> str:
    r0, c0, r1, c1 = bbox
    r, c = marker
    is_corner = (r in (r0, r1)) and (c in (c0, c1))
    assert not is_corner, "corner markers are ambiguous"
    if r == r0:
        return "top"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    if c == c1:
        return "right"
    raise AssertionError("marker must lie on its frame border")


def _ray_dir(side: str) -> tuple[int, int]:
    return {
        "top": (1, 0),
        "bottom": (-1, 0),
        "left": (0, 1),
        "right": (0, -1),
    }[side]


def _trace_ray(grid: np.ndarray, start: tuple[int, int], side: str, length: int) -> list[tuple[int, int]]:
    dr, dc = _ray_dir(side)
    r, c = start
    pts = []
    for step in range(1, length + 1):
        nr = r + dr * step
        nc = c + dc * step
        if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
            break
        if grid[nr, nc] != BG:
            break
        pts.append((nr, nc))
    return pts


def main(grid: np.ndarray) -> np.ndarray:
    """Color nested frames from a diagonal palette and extend inward rays whose length equals frame depth.

    Input structure:
    - The background is black.
    - Several one-pixel-thick gray rectangular frames are strictly nested with black gaps between them.
    - A distinct-color diagonal marker sequence starts at the grid's top-left corner, and orange pixels mark edge positions on the frames.

    Rule:
    - Read the diagonal palette from the top-left corner down-right until the diagonal first hits black.
    - Determine frame depth by nesting order, recolor every non-orange pixel of depth-d frame with palette[d].
    - From each orange marker, extend an inward ray perpendicular to that side for d steps, painting only black cells with the same palette[d] color; keep the palette markers and orange markers unchanged.

    Key insight:
    - The same hidden variable, nesting depth, simultaneously selects each frame's band color and each marker ray's length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette = _read_palette(g)
    frames = _extract_frames(g)
    depth_of = _depth_map(frames)
    assert len(palette) == len(frames), "palette length must equal nesting depth count"

    out = g.copy()
    for frame in frames:
        bbox = frame["bbox"]
        depth = depth_of[bbox]
        color = palette[depth - 1]
        mask = frame["mask"]
        out[mask & (g == FRAME)] = color

    for frame in frames:
        bbox = frame["bbox"]
        depth = depth_of[bbox]
        color = palette[depth - 1]
        for marker in frame["markers"]:
            side = _marker_side(bbox, marker)
            for rr, cc in _trace_ray(out, marker, side, depth):
                out[rr, cc] = color

    return out


def _build_nested_bboxes(depth: int) -> list[tuple[int, int, int, int]] | None:
    if depth >= 4:
        inner_h = random.randint(4, 6)
        inner_w = random.randint(4, 6)
    elif depth >= 3:
        if random.random() < 0.5:
            inner_h = random.randint(4, depth + 1)
            inner_w = random.randint(5, 9)
        else:
            inner_h = random.randint(5, 9)
            inner_w = random.randint(4, depth + 1)
    else:
        inner_h = random.randint(5, 9)
        inner_w = random.randint(5, 9)
    bboxes = [(0, 0, inner_h - 1, inner_w - 1)]
    for _ in range(depth - 1):
        r0, c0, r1, c1 = bboxes[-1]
        max_inset = 3 if depth >= 4 else 5
        top = random.randint(3, max_inset)
        left = random.randint(3, max_inset)
        bottom = random.randint(3, max_inset)
        right = random.randint(3, max_inset)
        bboxes.append((r0 - top, c0 - left, r1 + bottom, c1 + right))
    bboxes = list(reversed(bboxes))

    outer = bboxes[0]
    outer_h = outer[2] - outer[0] + 1
    outer_w = outer[3] - outer[1] + 1
    shift_r = depth + 1 + random.randint(0, 2)
    shift_c = depth + 1 + random.randint(0, 2)
    base_h = shift_r + outer_h
    base_w = shift_c + outer_w
    if base_h > 29 or base_w > 29:
        return None

    shifted = []
    base_r0, base_c0 = outer[0], outer[1]
    for r0, c0, r1, c1 in bboxes:
        shifted.append((r0 - base_r0 + shift_r, c0 - base_c0 + shift_c, r1 - base_r0 + shift_r, c1 - base_c0 + shift_c))
    return shifted


def _paint_input_grid(bboxes: list[tuple[int, int, int, int]], palette: list[int]) -> np.ndarray:
    base_h = max(b[2] for b in bboxes) + 1
    base_w = max(b[3] for b in bboxes) + 1
    height = base_h + random.randint(1, max(1, 30 - base_h))
    width = base_w + random.randint(1, max(1, 30 - base_w))
    grid = np.zeros((height, width), dtype=int)
    for i, color in enumerate(palette):
        grid[i, i] = color
    for bbox in bboxes:
        r0, c0, r1, c1 = bbox
        grid[r0, c0 : c1 + 1] = FRAME
        grid[r1, c0 : c1 + 1] = FRAME
        grid[r0 : r1 + 1, c0] = FRAME
        grid[r0 : r1 + 1, c1] = FRAME
    return grid


def _candidate_markers(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    depth: int,
    reserved_rays: set[tuple[int, int]],
) -> dict[str, list[tuple[tuple[int, int], list[tuple[int, int]]]]]:
    r0, c0, r1, c1 = bbox
    side_cells = {
        "top": [(r0, c) for c in range(c0 + 1, c1)],
        "bottom": [(r1, c) for c in range(c0 + 1, c1)],
        "left": [(r, c0) for r in range(r0 + 1, r1)],
        "right": [(r, c1) for r in range(r0 + 1, r1)],
    }
    candidates = {"full": [], "truncated": []}
    for side, cells in side_cells.items():
        random.shuffle(cells)
        for marker in cells:
            path = _trace_ray(grid, marker, side, depth)
            if not path:
                continue
            if any(pt in reserved_rays for pt in path):
                continue
            bucket = "full" if len(path) == depth else "truncated"
            candidates[bucket].append((marker, path))
    return candidates


def _choose_marker_specs(grid: np.ndarray, bboxes: list[tuple[int, int, int, int]]) -> list[dict] | None:
    depth = len(bboxes)
    reserved_rays: set[tuple[int, int]] = set()
    reserved_markers: set[tuple[int, int]] = set()
    chosen = []
    want_truncation = depth >= 3 and random.random() < 0.65
    saw_truncation = False
    saw_full = False

    for frame_depth, bbox in enumerate(bboxes, start=1):
        candidates = _candidate_markers(grid, bbox, frame_depth, reserved_rays)
        options_full = candidates["full"][:]
        options_trunc = candidates["truncated"][:]
        random.shuffle(options_full)
        random.shuffle(options_trunc)

        count = random.randint(1, 3)
        frame_markers = []
        used_side_positions = set()

        pools = []
        if want_truncation and not saw_truncation and options_trunc:
            pools.append("truncated")
        pools.extend(random.choices(["full", "truncated"], weights=[3, 2], k=count + 2))
        pools.append("full")

        for pref in pools:
            if len(frame_markers) >= count:
                break
            source = options_trunc if pref == "truncated" else options_full
            for idx, (marker, path) in enumerate(list(source)):
                if marker in reserved_markers or marker in used_side_positions:
                    continue
                if any(abs(marker[0] - mr) <= 1 and abs(marker[1] - mc) <= 1 for mr, mc in used_side_positions):
                    continue
                frame_markers.append({"marker": marker, "path": path, "kind": pref})
                used_side_positions.add(marker)
                reserved_markers.add(marker)
                reserved_rays.update(path)
                del source[idx]
                if pref == "truncated":
                    saw_truncation = True
                else:
                    saw_full = True
                break

        if not frame_markers:
            return None
        chosen.append({"bbox": bbox, "depth": frame_depth, "markers": frame_markers})

    if want_truncation and not saw_truncation:
        return None
    if not saw_full:
        return None
    return chosen


def _validate_structure(inp: np.ndarray, out: np.ndarray):
    palette = _read_palette(inp)
    frames = _extract_frames(inp)
    depth_of = _depth_map(frames)
    assert len(frames) == len(palette)

    for i, color in enumerate(palette):
        assert inp[i, i] == color
        for nr in range(max(0, i - 1), min(inp.shape[0], i + 2)):
            for nc in range(max(0, i - 1), min(inp.shape[1], i + 2)):
                if (nr, nc) == (i, i):
                    continue
                assert inp[nr, nc] != FRAME and inp[nr, nc] != MARKER, "palette must be isolated from frames"

    ray_lengths = []
    for frame in frames:
        bbox = frame["bbox"]
        depth = depth_of[bbox]
        color = palette[depth - 1]
        recolored = np.count_nonzero(frame["mask"] & (inp == FRAME))
        assert recolored > 0, "each frame must keep at least one recolorable border pixel"
        for marker in frame["markers"]:
            side = _marker_side(bbox, marker)
            path = _trace_ray(inp, marker, side, depth)
            assert path, "every marker must emit a visible ray"
            ray_lengths.append((depth, len(path)))
            for rr, cc in path:
                assert out[rr, cc] == color
        border_vals = out[frame["mask"] & (inp != MARKER)]
        assert np.all(border_vals == color), "frame border must be recolored by depth palette"

    assert any(length == depth for depth, length in ray_lengths), "need at least one full-length ray"
    if len(palette) >= 3:
        assert any(length < depth for depth, length in ray_lengths), "deep examples should expose truncation"

    wrong = inp.copy()
    wrong_palette = palette[::-1]
    for frame in frames:
        bbox = frame["bbox"]
        depth = depth_of[bbox]
        wrong[frame["mask"] & (inp == FRAME)] = wrong_palette[depth - 1]
    assert not np.array_equal(wrong, out), "frame colors must matter to the output"
    assert not np.array_equal(inp, out), "transform must be non-trivial"


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        depth = random.randint(2, 4)
        palette = random.sample(PALETTE_COLORS, depth)
        bboxes = _build_nested_bboxes(depth)
        if bboxes is None:
            continue
        grid = _paint_input_grid(bboxes, palette)
        chosen = _choose_marker_specs(grid, bboxes)
        if chosen is None:
            continue
        inp = grid.copy()
        for frame in chosen:
            for spec in frame["markers"]:
                r, c = spec["marker"]
                inp[r, c] = MARKER
        try:
            out = main(inp)
            _validate_structure(inp, out)
        except AssertionError:
            continue
        return inp
    raise RuntimeError("failed to generate a valid nested-ray ARC input")
