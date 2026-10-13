from __future__ import annotations

import itertools
import numpy as np

GREEN = 3
MAROON = 9
SIDES = ("up", "right", "down", "left")
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
STAMP_COLORS = np.array([0, 1, 2, 4, 5, 6, 7, 8], dtype=int)
STAMP_LIBRARY = (
    np.array([[0, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[0, 0, 0, 1], [1, 1, 1, 1]], dtype=bool),
    np.array([[0, 0, 1, 0], [1, 1, 1, 1]], dtype=bool),
)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _density_profile(mask: np.ndarray, side: str) -> tuple[float, ...]:
    h, w = mask.shape
    row_density = tuple(float(v) / float(w) for v in mask.sum(axis=1))
    col_density = tuple(float(v) / float(h) for v in mask.sum(axis=0))
    if side == "up":
        return row_density
    if side == "down":
        return row_density[::-1]
    if side == "left":
        return col_density
    return col_density[::-1]


def _open_side(mask: np.ndarray) -> str:
    profiles = {side: _density_profile(mask, side) for side in SIDES}
    return min(SIDES, key=lambda side: (profiles[side], SIDES.index(side)))


def _rotate_to_side(mask: np.ndarray, target_side: str) -> np.ndarray:
    rotations = [_tight_mask(np.rot90(mask, k)) for k in range(4)]
    for rotated in rotations:
        if _open_side(rotated) == target_side:
            return rotated
    return rotations[0]


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    pts = np.argwhere(mask)
    grid[top + pts[:, 0], left + pts[:, 1]] = int(color)


def _draw_frame(grid: np.ndarray, frame: dict) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    grid[r0 : r1 + 1, c0] = MAROON
    grid[r0 : r1 + 1, c1] = MAROON
    grid[r0, c0 : c1 + 1] = MAROON
    grid[r1, c0 : c1 + 1] = MAROON
    gap_r, gap_c = frame["gap_cell"]
    grid[gap_r, gap_c] = GREEN


def _parse_frame(coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 5 or w < 5:
        return None
    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if np.any(local[1:-1, 1:-1]):
        return None
    if not (local[0, 0] and local[0, -1] and local[-1, 0] and local[-1, -1]):
        return None

    side_slices = {
        "up": local[0, 1:-1],
        "down": local[-1, 1:-1],
        "left": local[1:-1, 0],
        "right": local[1:-1, -1],
    }
    missing = {side: np.flatnonzero(~edge) for side, edge in side_slices.items()}
    total_missing = sum(int(len(v)) for v in missing.values())
    if total_missing != 1:
        return None

    gap_side = next(side for side, idxs in missing.items() if len(idxs) == 1)
    gap_offset = int(missing[gap_side][0] + 1)
    if gap_side in ("up", "down"):
        gap_cell = (int(r0 if gap_side == "up" else r1), int(c0 + gap_offset))
    else:
        gap_cell = (int(r0 + gap_offset), int(c0 if gap_side == "left" else c1))

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "inner": (int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1)),
        "gap_side": gap_side,
        "gap_cell": gap_cell,
        "perimeter": coords.copy(),
        "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
    }


def _detect_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for coords in _components(grid == MAROON):
        frame = _parse_frame(coords)
        if frame is not None:
            frames.append(frame)
    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def _extract_stamps(grid: np.ndarray) -> list[dict]:
    colors = sorted(int(v) for v in np.unique(grid) if int(v) not in (GREEN, MAROON))
    stamps = []
    for color in colors:
        for coords in _components(grid == int(color)):
            r0, c0, r1, c1 = _bbox(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            stamps.append(
                {
                    "color": int(color),
                    "coords": coords.copy(),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                }
            )
    stamps.sort(key=lambda stamp: (stamp["bbox"], stamp["color"]))
    return stamps


def _distance_between_sets(a: np.ndarray, b: np.ndarray) -> int:
    best = 10**9
    for r, c in a:
        d = np.abs(b[:, 0] - int(r)) + np.abs(b[:, 1] - int(c))
        cur = int(d.min())
        if cur < best:
            best = cur
            if best == 0:
                return 0
    return int(best)


def _assign_stamps(frames: list[dict], stamps: list[dict]) -> dict[int, int]:
    n = len(frames)
    distances = np.zeros((n, n), dtype=int)
    for si, stamp in enumerate(stamps):
        for fi, frame in enumerate(frames):
            distances[si, fi] = _distance_between_sets(stamp["coords"], frame["perimeter"])

    nearest = distances.argmin(axis=1)
    if len(set(int(v) for v in nearest)) == n:
        strict = all(int(np.count_nonzero(distances[si] == distances[si, nearest[si]])) == 1 for si in range(n))
        if strict:
            return {int(fi): int(si) for si, fi in enumerate(nearest)}

    best_key = None
    best_perm = None
    for perm in itertools.permutations(range(n)):
        costs = tuple(int(distances[int(perm[fi]), fi]) for fi in range(n))
        key = (sum(costs), costs)
        if best_key is None or key < best_key:
            best_key = key
            best_perm = perm
    assert best_perm is not None
    return {int(fi): int(best_perm[fi]) for fi in range(n)}


def _placement_for_frame(frame: dict, mask: np.ndarray) -> tuple[int, int]:
    ir0, ic0, ir1, ic1 = frame["inner"]
    inner_h = int(ir1 - ir0 + 1)
    inner_w = int(ic1 - ic0 + 1)
    h, w = mask.shape
    if h > inner_h or w > inner_w:
        raise ValueError("stamp does not fit inside frame")
    if frame["gap_side"] == "up":
        return int(ir1 - h + 1), int(ic0 + (inner_w - w) // 2)
    if frame["gap_side"] == "down":
        return int(ir0), int(ic0 + (inner_w - w) // 2)
    if frame["gap_side"] == "left":
        return int(ir0 + (inner_h - h) // 2), int(ic1 - w + 1)
    return int(ir0 + (inner_h - h) // 2), int(ic0)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frames = _detect_frames(g)
    stamps = _extract_stamps(g)
    if not frames or len(frames) != len(stamps):
        return g.copy()

    try:
        assignment = _assign_stamps(frames, stamps)
    except Exception:
        return g.copy()

    out = g.copy()
    for stamp in stamps:
        out[stamp["coords"][:, 0], stamp["coords"][:, 1]] = GREEN

    for fi, frame in enumerate(frames):
        stamp = stamps[assignment[fi]]
        rotated = _rotate_to_side(stamp["mask"], frame["gap_side"])
        try:
            top, left = _placement_for_frame(frame, rotated)
        except ValueError:
            return g.copy()
        _paint_mask(out, rotated, top, left, stamp["color"])

    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_box(box: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    return (int(box[0] - pad), int(box[1] - pad), int(box[2] + pad), int(box[3] + pad))


def _make_frame_specs(rng: np.random.Generator, count: int) -> list[dict]:
    colors = rng.choice(STAMP_COLORS, size=count, replace=False)
    specs = []
    for idx in range(count):
        base_mask = STAMP_LIBRARY[int(rng.integers(len(STAMP_LIBRARY)))]
        gap_side = SIDES[int(rng.integers(4))]
        final_mask = _rotate_to_side(base_mask, gap_side)
        in_rot = int(rng.integers(4))
        input_mask = _tight_mask(np.rot90(base_mask, in_rot))
        if _open_side(input_mask) == gap_side and rng.random() < 0.75:
            alt = [k for k in range(4) if _open_side(_tight_mask(np.rot90(base_mask, k))) != gap_side]
            in_rot = int(alt[int(rng.integers(len(alt)))])
            input_mask = _tight_mask(np.rot90(base_mask, in_rot))

        min_inner_h = max(int(final_mask.shape[0]), 3)
        min_inner_w = max(int(final_mask.shape[1]), 3)
        inner_h = int(rng.integers(min_inner_h, min_inner_h + 4))
        inner_w = int(rng.integers(min_inner_w, min_inner_w + 5))
        outer_h = int(inner_h + 2)
        outer_w = int(inner_w + 2)

        if gap_side in ("up", "down"):
            gap_offset = int(rng.integers(1, outer_w - 1))
        else:
            gap_offset = int(rng.integers(1, outer_h - 1))

        specs.append(
            {
                "color": int(colors[idx]),
                "gap_side": gap_side,
                "gap_offset": gap_offset,
                "input_mask": input_mask,
                "final_mask": final_mask,
                "outer_h": outer_h,
                "outer_w": outer_w,
            }
        )
    return specs


def _place_frames(rng: np.random.Generator, shape: tuple[int, int], specs: list[dict]) -> list[dict] | None:
    h, w = shape
    frames: list[dict] = []
    for spec in specs:
        placed = False
        outer_h = spec["outer_h"]
        outer_w = spec["outer_w"]
        if h - outer_h - 1 < 1 or w - outer_w - 1 < 1:
            return None
        for _ in range(250):
            r0 = int(rng.integers(1, h - outer_h))
            c0 = int(rng.integers(1, w - outer_w))
            bbox = (int(r0), int(c0), int(r0 + outer_h - 1), int(c0 + outer_w - 1))
            grown = _expand_box(bbox, 3)
            if any(_boxes_overlap(grown, _expand_box(frame["bbox"], 3)) for frame in frames):
                continue
            gap_side = spec["gap_side"]
            gap_offset = spec["gap_offset"]
            if gap_side in ("up", "down"):
                gap_cell = (int(r0 if gap_side == "up" else r0 + outer_h - 1), int(c0 + gap_offset))
            else:
                gap_cell = (int(r0 + gap_offset), int(c0 if gap_side == "left" else c0 + outer_w - 1))
            frame = {
                "bbox": bbox,
                "inner": (int(r0 + 1), int(c0 + 1), int(r0 + outer_h - 2), int(c0 + outer_w - 2)),
                "gap_side": gap_side,
                "gap_cell": gap_cell,
                "perimeter": None,
                "center": ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
            }
            spec["bbox"] = bbox
            frames.append(frame)
            placed = True
            break
        if not placed:
            return None

    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def _frame_area_mask(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _candidate_positions(
    grid: np.ndarray,
    blocked: np.ndarray,
    frames: list[dict],
    frame_index: int,
    mask: np.ndarray,
) -> list[tuple[int, float, int, int]]:
    h, w = grid.shape
    ph, pw = mask.shape
    local_pts = np.argwhere(mask)
    frame = frames[frame_index]
    candidates: list[tuple[int, float, int, int]] = []
    for top in range(h - ph + 1):
        for left in range(w - pw + 1):
            if np.any(blocked[top : top + ph, left : left + pw]):
                continue
            if np.any(grid[top : top + ph, left : left + pw] != GREEN):
                continue
            placed_pts = local_pts + np.array([top, left], dtype=int)
            distances = [_distance_between_sets(placed_pts, other["perimeter"]) for other in frames]
            order = np.argsort(distances)
            if int(order[0]) != frame_index:
                continue
            if len(order) > 1 and int(distances[order[0]]) == int(distances[order[1]]):
                continue
            d0 = int(distances[frame_index])
            if not (1 <= d0 <= 5):
                continue
            center_r = top + (ph - 1) / 2.0
            center_c = left + (pw - 1) / 2.0
            spread = abs(center_r - frame["center"][0]) + abs(center_c - frame["center"][1])
            candidates.append((d0, spread, int(top), int(left)))
    candidates.sort()
    return candidates


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    frame_count = int(rng.integers(2, 5))
    size_ranges = {2: (18, 25), 3: (21, 28), 4: (24, 31)}
    low, high = size_ranges[frame_count]
    h = int(rng.integers(low, high))
    w = int(rng.integers(low, high))

    specs = _make_frame_specs(rng, frame_count)
    frames = _place_frames(rng, (h, w), specs)
    if frames is None:
        return None

    input_grid = np.full((h, w), GREEN, dtype=int)
    output_grid = np.full((h, w), GREEN, dtype=int)

    for frame in frames:
        _draw_frame(input_grid, frame)
        _draw_frame(output_grid, frame)
    frames = _detect_frames(input_grid)
    if len(frames) != frame_count:
        return None

    by_bbox = {frame["bbox"]: frame for frame in frames}
    frame_index_by_bbox = {frame["bbox"]: idx for idx, frame in enumerate(frames)}
    for spec in specs:
        spec["frame"] = by_bbox[spec["bbox"]]
        spec["frame_index"] = frame_index_by_bbox[spec["bbox"]]

    for spec in specs:
        final_mask = spec["final_mask"]
        frame = spec["frame"]
        top, left = _placement_for_frame(frame, final_mask)
        _paint_mask(output_grid, final_mask, top, left, spec["color"])

    blocked = _frame_area_mask((h, w), frames)
    remaining = set(range(frame_count))
    placed_any = False
    while remaining:
        option_map: dict[int, list[tuple[int, float, int, int]]] = {}
        for idx in remaining:
            candidates = _candidate_positions(
                input_grid,
                blocked,
                frames,
                specs[idx]["frame_index"],
                specs[idx]["input_mask"],
            )
            if not candidates:
                return None
            option_map[idx] = candidates
        idx = min(remaining, key=lambda item: len(option_map[item]))
        choices = option_map[idx][: min(12, len(option_map[idx]))]
        choice = choices[int(rng.integers(len(choices)))]
        _, _, top, left = choice
        _paint_mask(input_grid, specs[idx]["input_mask"], top, left, specs[idx]["color"])
        remaining.remove(idx)
        placed_any = True

    if not placed_any:
        return None
    if np.array_equal(input_grid, output_grid):
        return None
    if not np.array_equal(main(input_grid), output_grid):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
