from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

STRUCT4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
STRUCT8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)

AREA_TO_DIMS = {
    6: [(2, 3), (3, 2)],
    9: [(3, 3)],
    12: [(3, 4), (4, 3), (2, 6), (6, 2)],
    16: [(4, 4), (2, 8), (8, 2)],
}

GRID_MIN = 18
GRID_MAX = 24
FRAME_COUNTS = (3, 4)
FRAME_PADDING = 2
FRAME_BORDER = 1
SOURCE_PADDING = 1
MAX_GENERATION_ATTEMPTS = 500


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_coords(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    steps = STRUCT4 if int(connectivity) == 4 else STRUCT8
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _frame_border_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    grey_comps = _component_coords(grid == GREY, connectivity=8)
    for coords in grey_comps:
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        border_mask = _frame_border_mask(grid.shape, bbox)
        comp_mask = np.zeros(grid.shape, dtype=bool)
        comp_mask[coords[:, 0], coords[:, 1]] = True
        if not np.array_equal(comp_mask, border_mask):
            continue

        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        markers = np.argwhere(inner == RED)
        if markers.shape[0] != 1:
            continue
        if not np.all((inner == BLACK) | (inner == RED)):
            continue

        control = (r0 + 1 + int(markers[0, 0]), c0 + 1 + int(markers[0, 1]))
        frames.append(
            {
                "bbox": bbox,
                "control": control,
                "area": int(inner.shape[0] * inner.shape[1]),
                "inner_shape": (int(inner.shape[0]), int(inner.shape[1])),
            }
        )
    return frames


def _expected_output_from_frame(grid: np.ndarray, frame: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[out == BLUE] = BLACK

    r0, c0, r1, c1 = frame["bbox"]
    cr, cc = frame["control"]
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            dist = max(abs(r - cr), abs(c - cc))
            out[r, c] = BLUE if dist % 2 == 0 else YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the frame whose interior area matches the source-pixel count, then fill that frame with concentric parity rings around its control pixel.

    Input structure:
    - A black background contains 3 or 4 non-overlapping grey one-pixel-thick hollow rectangles.
    - Each frame encloses exactly one red control pixel and otherwise only black interior cells.
    - Separate blue source pixels lie outside every frame and are isolated from one another.

    Rule:
    - Count all blue source pixels.
    - Compute each grey frame's interior area, including the red control pixel cell.
    - Select the unique frame whose interior area equals the blue-pixel count.
    - Inside that frame, compute each interior cell's Chebyshev distance from the red control pixel.
    - Recolor even-distance cells blue and odd-distance cells yellow, covering the control pixel itself.
    - Erase the original blue source pixels outside the frames and leave all non-selected frames unchanged.

    Key insight:
    - Three independent cues must be combined: blue pixels determine which frame to use, the chosen frame's red pixel fixes the ring center, and Chebyshev-distance parity determines the two fill colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    frames = _extract_frames(g)
    assert len(frames) >= 3, "expected 3 or more valid frames"

    source_count = int(np.count_nonzero(g == BLUE))
    matches = [frame for frame in frames if frame["area"] == source_count]
    assert len(matches) == 1, "expected a unique area match"
    return _expected_output_from_frame(g, matches[0])


def _control_candidates(inner_h: int, inner_w: int) -> list[tuple[int, int]]:
    center_r = inner_h // 2
    center_c = inner_w // 2
    candidates: list[tuple[int, int]] = []
    edge_biased: list[tuple[int, int]] = []

    for r in range(inner_h):
        for c in range(inner_w):
            if inner_h % 2 == 1 and inner_w % 2 == 1 and r == center_r and c == center_c:
                continue
            candidates.append((r, c))
            if min(r, c, inner_h - 1 - r, inner_w - 1 - c) <= 1:
                edge_biased.append((r, c))
    return edge_biased or candidates


def _free_positions(
    occupied: np.ndarray, patch_h: int, patch_w: int, *, border: int, padding: int
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(border, h - border - patch_h + 1):
        for c0 in range(border, w - border - patch_w + 1):
            r1 = r0 + patch_h - 1
            c1 = c0 + patch_w - 1
            rr0 = max(0, r0 - padding)
            cc0 = max(0, c0 - padding)
            rr1 = min(h - 1, r1 + padding)
            cc1 = min(w - 1, c1 + padding)
            if not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                positions.append((r0, c0))
    return positions


def _mark_rect(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _forbidden_source_cells(grid: np.ndarray) -> np.ndarray:
    forbidden = grid != BLACK
    h, w = grid.shape
    extra = forbidden.copy()
    for r in range(h):
        for c in range(w):
            if not forbidden[r, c]:
                continue
            for dr, dc in STRUCT8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    extra[nr, nc] = True
    return extra


def _choose_sources(grid: np.ndarray, count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    blocked = _forbidden_source_cells(grid)
    h, w = grid.shape
    candidates = [(r, c) for r in range(h) for c in range(w) if not blocked[r, c]]
    rng.shuffle(candidates)

    selected: list[tuple[int, int]] = []
    banned = blocked.copy()
    for r, c in candidates:
        if banned[r, c]:
            continue
        selected.append((r, c))
        for dr, dc in STRUCT8 + ((0, 0),):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                banned[nr, nc] = True
        if len(selected) == count:
            return selected
    return None


def _sample_frame_specs(rng: random.Random) -> tuple[list[dict], int]:
    frame_count = rng.choice(FRAME_COUNTS)
    areas = rng.sample(list(AREA_TO_DIMS), frame_count)
    target_area = rng.choice(areas)
    specs = []
    for area in areas:
        inner_h, inner_w = rng.choice(AREA_TO_DIMS[area])
        specs.append({"area": area, "inner_h": inner_h, "inner_w": inner_w})
    rng.shuffle(specs)
    return specs, int(target_area)


def _build_instance(rng: random.Random) -> tuple[np.ndarray, dict] | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    specs, target_area = _sample_frame_specs(rng)
    specs = sorted(specs, key=lambda spec: (spec["inner_h"] + 2) * (spec["inner_w"] + 2), reverse=True)

    frames_meta: list[dict] = []
    for spec in specs:
        outer_h = spec["inner_h"] + 2
        outer_w = spec["inner_w"] + 2
        positions = _free_positions(
            occupied, outer_h, outer_w, border=FRAME_BORDER, padding=FRAME_PADDING
        )
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _mark_rect(occupied, r0, c0, outer_h, outer_w)

        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY

        rel_r, rel_c = rng.choice(_control_candidates(spec["inner_h"], spec["inner_w"]))
        control = (r0 + 1 + rel_r, c0 + 1 + rel_c)
        grid[control] = RED
        frames_meta.append(
            {
                "bbox": (r0, c0, r1, c1),
                "control": control,
                "area": spec["area"],
                "inner_shape": (spec["inner_h"], spec["inner_w"]),
            }
        )

    targets = [frame for frame in frames_meta if frame["area"] == target_area]
    if len(targets) != 1:
        return None
    target_frame = targets[0]

    source_points = _choose_sources(grid, target_area, rng)
    if source_points is None:
        return None
    for r, c in source_points:
        grid[r, c] = BLUE

    meta = {
        "frames": frames_meta,
        "target": target_frame,
        "sources": source_points,
    }
    if not _validate_instance(grid, meta):
        return None
    return grid, meta


def _validate_instance(grid: np.ndarray, meta: dict) -> bool:
    frames = _extract_frames(grid)
    if len(frames) != len(meta["frames"]):
        return False

    areas = [frame["area"] for frame in frames]
    if len(set(areas)) != len(areas):
        return False

    source_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == BLUE)]
    if len(source_coords) != meta["target"]["area"]:
        return False

    frame_boxes = {tuple(frame["bbox"]) for frame in frames}
    meta_boxes = {tuple(frame["bbox"]) for frame in meta["frames"]}
    if frame_boxes != meta_boxes:
        return False

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        cr, cc = frame["control"]
        inner_h, inner_w = frame["inner_shape"]
        if inner_h % 2 == 1 and inner_w % 2 == 1:
            if (cr, cc) == (r0 + 1 + inner_h // 2, c0 + 1 + inner_w // 2):
                return False

    for r, c in source_coords:
        if any(r0 <= r <= r1 and c0 <= c <= c1 for r0, c0, r1, c1 in frame_boxes):
            return False
        for dr, dc in STRUCT4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == BLUE:
                return False

    matches = [frame for frame in frames if frame["area"] == len(source_coords)]
    if len(matches) != 1:
        return False
    if tuple(matches[0]["bbox"]) != tuple(meta["target"]["bbox"]):
        return False

    expected = _expected_output_from_frame(grid, meta["target"])
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    if np.count_nonzero(actual == BLUE) <= 1:
        return False
    if not np.any(actual == YELLOW):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_instance(rng)
        if built is not None:
            return built[0]
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
