from __future__ import annotations

from collections import deque
import random

import numpy as np


COUNT_TO_COLOR = {1: 4, 2: 7, 3: 6, 4: 9}
FRAME_COLORS = (1, 2, 3)
OBJECT_COLOR = 5
BACKGROUND = 0

GRID_SIDE_RANGE = (18, 26)
FRAME_COUNT_CHOICES = (2, 3)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 200
OBJECT_PLACEMENT_RETRIES = 200
SHAPE_RETRIES = 120

BASE_OBJECT_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            queue = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _boundary_from_filled(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    boundary = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                boundary[rr, cc] = True
                break
    return boundary


def _interior_from_frame(frame_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = frame_mask.shape
    exterior = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def push(rr: int, cc: int) -> None:
        if 0 <= rr < h and 0 <= cc < w and (not frame_mask[rr, cc]) and (not exterior[rr, cc]):
            exterior[rr, cc] = True
            queue.append((rr, cc))

    for r in range(h):
        push(r, 0)
        push(r, w - 1)
    for c in range(w):
        push(0, c)
        push(h - 1, c)

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            push(nr, nc)

    empty = ~frame_mask
    interior = empty & (~exterior)
    return interior, exterior


def _diagonal_halo_for_frame(
    grid: np.ndarray, frame_mask: np.ndarray, interior_mask: np.ndarray, bbox: tuple[int, int, int, int]
) -> set[tuple[int, int]]:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    halo: set[tuple[int, int]] = set()
    for r in range(max(0, r0 - 1), min(h, r1 + 2)):
        for c in range(max(0, c0 - 1), min(w, c1 + 2)):
            if grid[r, c] != BACKGROUND or interior_mask[r, c] or frame_mask[r, c]:
                continue
            has_diag = False
            has_ortho = False
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and frame_mask[nr, nc]:
                    has_ortho = True
                    break
            if has_ortho:
                continue
            for dr in (-1, 1):
                for dc in (-1, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and frame_mask[nr, nc]:
                        has_diag = True
                        break
                if has_diag:
                    break
            if has_diag:
                halo.add((r, c))
    return halo


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    for color in sorted(set(int(v) for v in np.unique(g)) - {BACKGROUND, OBJECT_COLOR}):
        for coords in _components(g == color, connectivity=4):
            if coords.shape[0] < 8:
                continue
            r0, c0, r1, c1 = _bbox_coords(coords)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local[coords[:, 0] - r0, coords[:, 1] - c0] = True

            if any(
                all(
                    0 <= nr < local.shape[0] and 0 <= nc < local.shape[1] and local[nr, nc]
                    for nr, nc in _neighbors4(int(r), int(c))
                )
                for r, c in np.argwhere(local)
            ):
                continue

            interior_local, _ = _interior_from_frame(local)
            if not interior_local.any():
                continue

            interior = np.zeros_like(g, dtype=bool)
            interior[r0 : r1 + 1, c0 : c1 + 1] = interior_local
            frame_mask = np.zeros_like(g, dtype=bool)
            frame_mask[coords[:, 0], coords[:, 1]] = True
            halo = _diagonal_halo_for_frame(g, frame_mask, interior, (r0, c0, r1, c1))
            grey_components = _components((g == OBJECT_COLOR) & interior, connectivity=4)
            if not grey_components:
                continue
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "frame_mask": frame_mask,
                    "interior": interior,
                    "halo": halo,
                    "count": len(grey_components),
                }
            )
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each frame's diagonal-only exterior halo by counting the gray objects sealed inside it.

    Input structure:
    - The grid has black background, plus 2 to 3 differently colored closed frames.
    - Each frame is a one-cell-thick enclosing boundary, and its interior contains several separate gray 4-connected objects.
    - Frames do not overlap or touch, and each frame's halo candidates are black pixels outside the frame.

    Rule:
    - Detect every closed frame and its enclosed interior region.
    - Count the number K of gray 4-connected objects strictly inside that frame.
    - Map K to the halo color by 1->4, 2->7, 3->6, 4->9.
    - Recolor every black pixel that is diagonally adjacent to that frame but not orthogonally adjacent to it, while leaving the frame, its gray objects, and all interior background unchanged.

    Key insight:
    - The painted positions come from local diagonal-only adjacency to each enclosing frame, but the paint color is an indirect code determined by the number of interior gray objects rather than by the frame color itself.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    painted = np.zeros(g.shape, dtype=bool)

    for frame in _extract_frames(g):
        count = int(frame["count"])
        assert count in COUNT_TO_COLOR, "expected 1 to 4 gray objects inside each frame"
        fill_color = COUNT_TO_COLOR[count]
        for r, c in frame["halo"]:
            assert not painted[r, c], "frame halos must be disjoint"
            out[r, c] = fill_color
            painted[r, c] = True

    return out


def _normalize_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in shape)
    c0 = min(c for _, c in shape)
    return tuple(sorted((r - r0, c - c0) for r, c in shape))


def _shape_transforms(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    pts = np.array(shape, dtype=int)
    out: set[tuple[tuple[int, int], ...]] = set()
    for flip in (False, True):
        arr = pts.copy()
        if flip:
            arr[:, 1] *= -1
        for _ in range(4):
            arr = np.stack([arr[:, 1], -arr[:, 0]], axis=1)
            out.add(_normalize_shape(tuple((int(r), int(c)) for r, c in arr)))
    return sorted(out)


ALL_OBJECT_SHAPES = tuple(
    tuple(_shape_transforms(shape))
    for shape in BASE_OBJECT_SHAPES
)


def _build_filled_shape(rng: random.Random, irregular: bool) -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        base_h = rng.randint(6, 9)
        base_w = rng.randint(6, 10)
        cells = {(r, c) for r in range(base_h) for c in range(base_w)}
        if irregular:
            side_count = rng.randint(1, 3)
            for side in rng.sample(("top", "bottom", "left", "right"), side_count):
                if side in {"top", "bottom"}:
                    tab_w = rng.randint(3, max(3, base_w - 1))
                    tab_d = rng.randint(2, 3)
                    start = rng.randint(0, base_w - tab_w)
                    rows = range(-tab_d, 0) if side == "top" else range(base_h, base_h + tab_d)
                    for r in rows:
                        for c in range(start, start + tab_w):
                            cells.add((r, c))
                else:
                    tab_h = rng.randint(3, max(3, base_h - 1))
                    tab_d = rng.randint(2, 3)
                    start = rng.randint(0, base_h - tab_h)
                    cols = range(-tab_d, 0) if side == "left" else range(base_w, base_w + tab_d)
                    for r in range(start, start + tab_h):
                        for c in cols:
                            cells.add((r, c))

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r_shift = 1 - min(rows)
        c_shift = 1 - min(cols)
        h = max(rows) - min(rows) + 3
        w = max(cols) - min(cols) + 3
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r + r_shift, c + c_shift] = True

        boundary = _boundary_from_filled(mask)
        interior = mask & (~boundary)
        if interior.sum() < 12:
            continue
        return mask
    return None


def _place_objects(interior: np.ndarray, count: int, rng: random.Random) -> np.ndarray | None:
    h, w = interior.shape
    occupied = np.zeros((h, w), dtype=bool)
    for _idx in range(count):
        placed = False
        for _ in range(OBJECT_PLACEMENT_RETRIES):
            shape_family = rng.choice(ALL_OBJECT_SHAPES)
            shape = rng.choice(shape_family)
            sh = 1 + max(r for r, _ in shape)
            sw = 1 + max(c for _, c in shape)
            if sh > h or sw > w:
                continue
            top = rng.randint(0, h - sh)
            left = rng.randint(0, w - sw)
            cells = [(top + r, left + c) for r, c in shape]
            if any(not interior[r, c] for r, c in cells):
                continue
            blocked = False
            for r, c in cells:
                for nr in range(r - 1, r + 2):
                    for nc in range(c - 1, c + 2):
                        if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                            blocked = True
                            break
                    if blocked:
                        break
                if blocked:
                    break
            if blocked:
                continue
            for r, c in cells:
                occupied[r, c] = True
            placed = True
            break
        if not placed:
            return None
    if len(_components(occupied, connectivity=4)) != count:
        return None
    return occupied


def _make_frame_scene(color: int, count: int, irregular: bool, rng: random.Random) -> dict | None:
    filled = _build_filled_shape(rng, irregular=irregular)
    if filled is None:
        return None
    boundary = _boundary_from_filled(filled)
    interior = filled & (~boundary)
    objects = _place_objects(interior, count=count, rng=rng)
    if objects is None:
        return None

    scene = np.zeros(filled.shape, dtype=int)
    scene[boundary] = color
    scene[objects] = OBJECT_COLOR

    interior_bg = interior & (~objects)
    halo = _diagonal_halo_for_frame(scene, boundary, interior, _bbox_coords(np.argwhere(boundary)))
    if not halo:
        return None

    bbox = _bbox_coords(np.argwhere(boundary))
    bbox_corner_halo = {
        (bbox[0] - 1, bbox[1] - 1),
        (bbox[0] - 1, bbox[3] + 1),
        (bbox[2] + 1, bbox[1] - 1),
        (bbox[2] + 1, bbox[3] + 1),
    }
    bbox_corner_halo = {(r, c) for r, c in bbox_corner_halo if 0 <= r < scene.shape[0] and 0 <= c < scene.shape[1]}

    return {
        "grid": scene,
        "count": count,
        "color": color,
        "halo": halo,
        "bbox": bbox,
        "irregular": halo != bbox_corner_halo,
        "interior_bg_count": int(interior_bg.sum()),
    }


def _rects_far_enough(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_scenes(grid_shape: tuple[int, int], scenes: list[dict], rng: random.Random) -> np.ndarray | None:
    h, w = grid_shape
    grid = np.zeros((h, w), dtype=int)
    placed_rects: list[tuple[int, int, int, int]] = []

    for scene in sorted(scenes, key=lambda item: item["grid"].shape[0] * item["grid"].shape[1], reverse=True):
        patch = scene["grid"]
        ph, pw = patch.shape
        if ph > h or pw > w:
            return None
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = rng.randint(0, h - ph)
            left = rng.randint(0, w - pw)
            rect = (top, left, top + ph - 1, left + pw - 1)
            if any(not _rects_far_enough(rect, other, gap=1) for other in placed_rects):
                continue
            mask = patch != BACKGROUND
            grid[top : top + ph, left : left + pw][mask] = patch[mask]
            placed_rects.append(rect)
            placed = True
            break
        if not placed:
            return None
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    frames = _extract_frames(grid)
    if len(frames) not in FRAME_COUNT_CHOICES:
        return False
    counts = [int(frame["count"]) for frame in frames]
    if len(set(counts)) != len(counts):
        return False
    if any(count not in COUNT_TO_COLOR for count in counts):
        return False
    if len({int(frame["color"]) for frame in frames}) != len(frames):
        return False
    if not any(frame["halo"] != {
        (frame["bbox"][0] - 1, frame["bbox"][1] - 1),
        (frame["bbox"][0] - 1, frame["bbox"][3] + 1),
        (frame["bbox"][2] + 1, frame["bbox"][1] - 1),
        (frame["bbox"][2] + 1, frame["bbox"][3] + 1),
    } for frame in frames):
        return False

    seen: set[tuple[int, int]] = set()
    for frame in frames:
        if not frame["halo"]:
            return False
        if any(cell in seen for cell in frame["halo"]):
            return False
        seen |= frame["halo"]

    out = main(grid)
    diff = out != grid
    if not diff.any():
        return False

    expected_changed = np.zeros(grid.shape, dtype=bool)
    for frame in frames:
        fill_color = COUNT_TO_COLOR[int(frame["count"])]
        for r, c in frame["halo"]:
            if out[r, c] != fill_color:
                return False
            expected_changed[r, c] = True
        interior_bg = frame["interior"] & (grid == BACKGROUND)
        if np.any(out[interior_bg] != BACKGROUND):
            return False
        if np.any(out[frame["frame_mask"]] != frame["color"]):
            return False

    if np.any(diff & (~expected_changed)):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_SIDE_RANGE)
    w = rng.randint(*GRID_SIDE_RANGE)
    frame_count = rng.choice(FRAME_COUNT_CHOICES)
    frame_colors = rng.sample(list(FRAME_COLORS), frame_count)
    counts = rng.sample([1, 2, 3, 4], frame_count)

    irregular_flags = [True] + [rng.random() < 0.6 for _ in range(frame_count - 1)]
    rng.shuffle(irregular_flags)

    scenes: list[dict] = []
    for color, count, irregular in zip(frame_colors, counts, irregular_flags):
        scene = _make_frame_scene(color=color, count=count, irregular=irregular, rng=rng)
        if scene is None:
            return None
        scenes.append(scene)

    grid = _place_scenes((h, w), scenes, rng=rng)
    if grid is None:
        return None
    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    seed = random.randrange(1 << 30)
    rng = random.Random(seed)
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
