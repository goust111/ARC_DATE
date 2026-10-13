from __future__ import annotations

import random
from typing import Any

import numpy as np


BLACK = 0
GREY = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
USABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

MIN_STAMPS = 2
MAX_STAMPS = 4
MIN_CANVAS = 8
MAX_CANVAS = 12
MAX_SHAPE_SIDE = 4
MIN_SHAPE_CELLS = 3
MAX_SHAPE_CELLS = 6
GEN_RETRIES = 400
TARGET_RETRIES = 40
SOURCE_RETRIES = 160
SHAPE_RETRIES = 120
MAX_CANDIDATES_PER_STAMP = 12
MAX_CANDIDATE_PRODUCT = 64
STAMP_COUNT_CHOICES = [2, 2, 2, 3, 3, 4]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        pts: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            pts.append((cr, cc))
            for dr, dc in DIR4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _rotate_stamp(mask: np.ndarray, locator: tuple[int, int], k: int) -> tuple[np.ndarray, tuple[int, int]]:
    rot_mask = np.rot90(mask, k)
    loc_mask = np.zeros(mask.shape, dtype=bool)
    loc_mask[locator] = True
    rot_loc = np.rot90(loc_mask, k)
    lr, lc = np.argwhere(rot_loc)[0]
    return rot_mask, (int(lr), int(lc))


def _stamp_signature(mask: np.ndarray, locator: tuple[int, int]) -> tuple[Any, ...]:
    return tuple(
        (
            tuple(rot_mask.shape),
            rot_mask.tobytes(),
            rot_loc[0],
            rot_loc[1],
        )
        for rot_mask, rot_loc in (_rotate_stamp(mask, locator, k) for k in range(4))
    )


def _unique_rotation_count(mask: np.ndarray, locator: tuple[int, int]) -> int:
    sigs = set()
    for k in range(4):
        rot_mask, rot_loc = _rotate_stamp(mask, locator, k)
        sigs.add((tuple(rot_mask.shape), rot_mask.tobytes(), rot_loc))
    return len(sigs)


def _find_canvas(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey_mask = grid == GREY
    candidates: list[tuple[int, int, int, int]] = []
    for pts in _components(grey_mask):
        r0, c0, r1, c1 = _bbox(pts)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        expected = 2 * (h + w) - 4
        if int(pts.shape[0]) != expected:
            continue
        sub = grey_mask[r0 : r1 + 1, c0 : c1 + 1]
        frame = np.zeros_like(sub, dtype=bool)
        frame[0, :] = True
        frame[-1, :] = True
        frame[:, 0] = True
        frame[:, -1] = True
        if np.array_equal(sub, frame):
            candidates.append((r0, c0, r1, c1))
    assert candidates, "expected a grey rectangular frame"
    return max(candidates, key=lambda box: (box[2] - box[0] + 1) * (box[3] - box[1] + 1))


def _parse_anchors(grid: np.ndarray, frame: tuple[int, int, int, int]) -> dict[int, tuple[int, int]]:
    r0, c0, r1, c1 = frame
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    anchor_mask = (inner != BLACK) & (inner != GREY)
    anchors: dict[int, tuple[int, int]] = {}
    for pts in _components(anchor_mask):
        assert int(pts.shape[0]) == 1, "anchors must be isolated single pixels"
        rr, cc = map(int, pts[0])
        color = int(inner[rr, cc])
        assert color not in anchors, "anchor colors must be unique"
        anchors[color] = (r0 + 1 + rr, c0 + 1 + cc)
    assert anchors, "expected at least one anchor"
    return anchors


def _parse_stamps(
    grid: np.ndarray, frame: tuple[int, int, int, int], anchor_colors: set[int]
) -> list[dict[str, Any]]:
    r0, c0, r1, c1 = frame
    outside = (grid != BLACK) & (grid != GREY)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    stamps: list[dict[str, Any]] = []
    for pts in _components(outside):
        colors = [int(grid[int(r), int(c)]) for r, c in pts]
        uniq, counts = np.unique(np.array(colors, dtype=int), return_counts=True)
        uniq_list = [int(x) for x in uniq]
        count_map = {int(color): int(count) for color, count in zip(uniq, counts)}
        assert len(uniq_list) == 2, "each stamp must use one main color and one locator color"

        locator_candidates = [color for color in uniq_list if color in anchor_colors]
        if len(locator_candidates) == 1:
            locator_color = locator_candidates[0]
        else:
            locator_candidates = [color for color in uniq_list if count_map[color] == 1]
            assert len(locator_candidates) == 1, "could not identify locator color"
            locator_color = locator_candidates[0]
        main_colors = [color for color in uniq_list if color != locator_color]
        assert len(main_colors) == 1
        main_color = main_colors[0]
        assert count_map[main_color] >= 2, "stamp body must have at least two pixels"
        assert count_map[locator_color] == 1, "locator must be unique within its stamp"

        sr0, sc0, sr1, sc1 = _bbox(pts)
        mask = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=bool)
        locator = None
        for r, c in pts:
            rr = int(r) - sr0
            cc = int(c) - sc0
            mask[rr, cc] = True
            if int(grid[int(r), int(c)]) == locator_color:
                locator = (rr, cc)
        assert locator is not None
        stamps.append(
            {
                "main_color": int(main_color),
                "locator_color": int(locator_color),
                "mask": mask,
                "locator": locator,
            }
        )

    stamps.sort(key=lambda stamp: stamp["locator_color"])
    return stamps


def _placement_candidates(
    stamp: dict[str, Any], anchor: tuple[int, int], inner_box: tuple[int, int, int, int]
) -> list[dict[str, Any]]:
    ir0, ic0, ir1, ic1 = inner_box
    inner_w = ic1 - ic0 + 1
    seen: dict[tuple[int, ...], dict[str, Any]] = {}
    for rotation in range(4):
        rot_mask, rot_loc = _rotate_stamp(stamp["mask"], stamp["locator"], rotation)
        body = np.argwhere(rot_mask)
        top = anchor[0] - rot_loc[0]
        left = anchor[1] - rot_loc[1]
        abs_coords = body + np.array([top, left], dtype=int)
        if (
            int(abs_coords[:, 0].min()) < ir0
            or int(abs_coords[:, 0].max()) > ir1
            or int(abs_coords[:, 1].min()) < ic0
            or int(abs_coords[:, 1].max()) > ic1
        ):
            continue
        rel_rows = abs_coords[:, 0] - ir0
        rel_cols = abs_coords[:, 1] - ic0
        linear = [int(r * inner_w + c) for r, c in zip(rel_rows, rel_cols)]
        key = tuple(sorted(linear))
        if key in seen:
            continue
        bits = 0
        for idx in key:
            bits |= 1 << idx
        seen[key] = {
            "rotation": rotation,
            "coords": abs_coords.astype(int),
            "bits": bits,
            "main_color": int(stamp["main_color"]),
            "locator_color": int(stamp["locator_color"]),
        }
    return list(seen.values())


def _search_solutions(
    stamps: list[dict[str, Any]],
    anchors: dict[int, tuple[int, int]],
    inner_box: tuple[int, int, int, int],
    max_solutions: int,
) -> tuple[list[list[dict[str, Any]]], list[int]]:
    prepared: list[dict[str, Any]] = []
    for idx, stamp in enumerate(stamps):
        locator_color = int(stamp["locator_color"])
        assert locator_color in anchors, "every stamp locator needs a matching anchor"
        cands = _placement_candidates(stamp, anchors[locator_color], inner_box)
        assert cands, "every stamp must have at least one legal placement"
        prepared.append({"index": idx, "cands": cands})

    order = sorted(range(len(prepared)), key=lambda i: len(prepared[i]["cands"]))
    candidate_counts = [len(item["cands"]) for item in prepared]
    chosen: list[dict[str, Any] | None] = [None] * len(stamps)
    occupied = 0
    solutions: list[list[dict[str, Any]]] = []

    def dfs(depth: int) -> None:
        nonlocal occupied
        if len(solutions) >= max_solutions:
            return
        if depth == len(order):
            solutions.append([item for item in chosen if item is not None])
            return
        slot = prepared[order[depth]]
        for cand in slot["cands"]:
            if occupied & int(cand["bits"]):
                continue
            prev = occupied
            occupied |= int(cand["bits"])
            chosen[slot["index"]] = cand
            dfs(depth + 1)
            chosen[slot["index"]] = None
            occupied = prev

    dfs(0)
    return solutions, candidate_counts


def _solve_details(grid: np.ndarray, max_solutions: int = 2) -> dict[str, Any]:
    frame = _find_canvas(grid)
    anchors = _parse_anchors(grid, frame)
    stamps = _parse_stamps(grid, frame, set(anchors))
    assert len(stamps) == len(anchors), "stamp count must match anchor count"
    assert {stamp["locator_color"] for stamp in stamps} == set(anchors), "anchor colors must match stamps"
    inner_box = (frame[0] + 1, frame[1] + 1, frame[2] - 1, frame[3] - 1)
    solutions, candidate_counts = _search_solutions(stamps, anchors, inner_box, max_solutions=max_solutions)
    return {
        "frame": frame,
        "anchors": anchors,
        "stamps": stamps,
        "solutions": solutions,
        "candidate_counts": candidate_counts,
    }


def _render_output(
    shape: tuple[int, int], frame: tuple[int, int, int, int], solution: list[dict[str, Any]]
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    r0, c0, r1, c1 = frame
    out[r0, c0 : c1 + 1] = GREY
    out[r1, c0 : c1 + 1] = GREY
    out[r0 : r1 + 1, c0] = GREY
    out[r0 : r1 + 1, c1] = GREY
    for placement in solution:
        coords = placement["coords"]
        out[coords[:, 0], coords[:, 1]] = int(placement["main_color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Place each outside stamp onto the grey-framed canvas by matching its locator color
    to the corresponding anchor and choosing the unique rotation that keeps all stamps
    inside the canvas without overlap.
    """
    g = np.asarray(grid, dtype=int)
    details = _solve_details(g, max_solutions=2)
    assert len(details["solutions"]) == 1, "expected a unique stamp arrangement"
    return _render_output(g.shape, details["frame"], details["solutions"][0])


def _random_connected_mask(rng: random.Random) -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        h = rng.randint(2, MAX_SHAPE_SIDE)
        w = rng.randint(2, MAX_SHAPE_SIDE)
        target = rng.randint(MIN_SHAPE_CELLS, min(MAX_SHAPE_CELLS, h * w))
        cells = {(rng.randrange(h), rng.randrange(w))}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if int(mask.sum()) < MIN_SHAPE_CELLS:
            continue
        if len(_components(mask)) != 1:
            continue
        return mask
    return None


def _make_random_stamp(rng: random.Random, main_color: int, locator_color: int) -> dict[str, Any] | None:
    require_four = rng.random() < 0.4
    for _ in range(SHAPE_RETRIES):
        mask = _random_connected_mask(rng)
        if mask is None:
            return None
        cells = [tuple(map(int, pt)) for pt in np.argwhere(mask)]
        rng.shuffle(cells)
        for locator in cells:
            uniq = _unique_rotation_count(mask, locator)
            if uniq < 2:
                continue
            if require_four and uniq < 4:
                continue
            return {
                "mask": mask,
                "locator": locator,
                "main_color": int(main_color),
                "locator_color": int(locator_color),
                "target_rotation": rng.randrange(4),
            }
    return None


def _mark_with_gap(mask: np.ndarray, coords: np.ndarray) -> None:
    h, w = mask.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for nr in range(max(0, rr - 1), min(h, rr + 2)):
            for nc in range(max(0, cc - 1), min(w, cc + 2)):
                mask[nr, nc] = True


def _place_targets(
    rng: random.Random,
    stamps: list[dict[str, Any]],
    canvas_h: int,
    canvas_w: int,
    require_ambiguity: bool,
) -> list[dict[str, Any]] | None:
    special_idx = rng.randrange(len(stamps)) if require_ambiguity else None
    regular = [i for i in range(len(stamps)) if i != special_idx]
    regular.sort(key=lambda i: int(stamps[i]["mask"].sum()), reverse=True)
    order = regular + ([special_idx] if special_idx is not None else [])
    inner_box = (0, 0, canvas_h - 1, canvas_w - 1)
    blocked = np.zeros((canvas_h, canvas_w), dtype=bool)
    occupied = np.zeros((canvas_h, canvas_w), dtype=bool)
    placements: list[dict[str, Any] | None] = [None] * len(stamps)

    for idx in order:
        stamp = stamps[idx]
        rot_mask, rot_loc = _rotate_stamp(stamp["mask"], stamp["locator"], int(stamp["target_rotation"]))
        body = np.argwhere(rot_mask)
        rh, rw = rot_mask.shape
        options: list[dict[str, Any]] = []
        for top in range(canvas_h - rh + 1):
            for left in range(canvas_w - rw + 1):
                coords = body + np.array([top, left], dtype=int)
                if np.any(blocked[coords[:, 0], coords[:, 1]]):
                    continue
                anchor = (top + rot_loc[0], left + rot_loc[1])
                candidates = _placement_candidates(stamp, anchor, inner_box)
                intended_key = tuple(sorted(int(r) * canvas_w + int(c) for r, c in coords))
                matching = [cand for cand in candidates if tuple(sorted(int(r) * canvas_w + int(c) for r, c in cand["coords"])) == intended_key]
                if len(matching) != 1:
                    continue
                if idx != special_idx:
                    if len(candidates) != 1:
                        continue
                else:
                    if len(candidates) != 2:
                        continue
                    other = [cand for cand in candidates if cand is not matching[0]]
                    if len(other) != 1:
                        continue
                    if not np.any(occupied[other[0]["coords"][:, 0], other[0]["coords"][:, 1]]):
                        continue
                options.append(
                    {
                        "top": int(top),
                        "left": int(left),
                        "coords": coords.astype(int),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                    }
                )
        if not options:
            return None
        chosen = rng.choice(options)
        top = int(chosen["top"])
        left = int(chosen["left"])
        coords = chosen["coords"]
        anchor = chosen["anchor"]
        _mark_with_gap(blocked, coords)
        occupied[coords[:, 0], coords[:, 1]] = True
        placements[idx] = {
            "top": int(top),
            "left": int(left),
            "coords": coords,
            "anchor": (int(anchor[0]), int(anchor[1])),
            "rotation": int(stamp["target_rotation"]),
            "main_color": int(stamp["main_color"]),
            "locator_color": int(stamp["locator_color"]),
        }

    assert all(item is not None for item in placements)
    return [item for item in placements if item is not None]


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _stamp_patch(stamp: dict[str, Any]) -> np.ndarray:
    patch = np.zeros(stamp["mask"].shape, dtype=int)
    patch[stamp["mask"]] = int(stamp["main_color"])
    lr, lc = stamp["locator"]
    patch[lr, lc] = int(stamp["locator_color"])
    return patch


def _place_source_stamps(
    rng: random.Random,
    grid: np.ndarray,
    frame: tuple[int, int, int, int],
    stamps: list[dict[str, Any]],
) -> bool:
    rows, cols = grid.shape
    frame_rect = frame
    placed: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(stamps)), key=lambda i: stamps[i]["mask"].shape[0] * stamps[i]["mask"].shape[1], reverse=True)

    for idx in order:
        patch = _stamp_patch(stamps[idx])
        ph, pw = patch.shape
        options: list[tuple[int, int]] = []
        for top in range(rows - ph + 1):
            for left in range(cols - pw + 1):
                rect = (top, left, top + ph - 1, left + pw - 1)
                if _rects_overlap(rect, frame_rect, pad=1):
                    continue
                if any(_rects_overlap(rect, other, pad=1) for other in placed):
                    continue
                options.append((top, left))
        if not options:
            return False
        top, left = rng.choice(options)
        patch_mask = patch != BLACK
        grid[top : top + ph, left : left + pw][patch_mask] = patch[patch_mask]
        placed.append((top, left, top + ph - 1, left + pw - 1))

    return True


def _render_expected(
    shape: tuple[int, int], frame: tuple[int, int, int, int], placements: list[dict[str, Any]]
) -> np.ndarray:
    solution = []
    for placement in placements:
        solution.append(
            {
                "coords": placement["coords"],
                "main_color": placement["main_color"],
            }
        )
    return _render_output(shape, frame, solution)


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    n_stamps = rng.choice(STAMP_COUNT_CHOICES)
    colors = USABLE_COLORS[:]
    rng.shuffle(colors)
    anchor_colors = colors[:n_stamps]
    main_colors = colors[n_stamps : 2 * n_stamps]
    stamps: list[dict[str, Any]] = []
    for main_color, locator_color in zip(main_colors, anchor_colors):
        stamp = _make_random_stamp(rng, main_color, locator_color)
        if stamp is None:
            return None
        stamps.append(stamp)

    require_ambiguity = rng.random() < (0.35 if n_stamps < 4 else 0.2)
    rotated_shapes = [_rotate_stamp(stamp["mask"], stamp["locator"], int(stamp["target_rotation"]))[0] for stamp in stamps]
    min_canvas_h = max(MIN_CANVAS, max(int(shape.shape[0]) for shape in rotated_shapes))
    min_canvas_w = max(MIN_CANVAS, max(int(shape.shape[1]) for shape in rotated_shapes))
    max_canvas_h = min(MAX_CANVAS, min_canvas_h + 3)
    max_canvas_w = min(MAX_CANVAS, min_canvas_w + 3)
    if min_canvas_h > max_canvas_h or min_canvas_w > max_canvas_w:
        return None
    for _ in range(TARGET_RETRIES):
        canvas_h = rng.randint(min_canvas_h, max_canvas_h)
        canvas_w = rng.randint(min_canvas_w, max_canvas_w)
        placements_local = _place_targets(rng, stamps, canvas_h, canvas_w, require_ambiguity)
        if placements_local is None:
            continue

        frame_h = canvas_h + 2
        frame_w = canvas_w + 2
        rows = frame_h + rng.randint(8, 12 + n_stamps)
        cols = frame_w + rng.randint(8, 12 + n_stamps)
        min_free = sum((stamp["mask"].shape[0] + 2) * (stamp["mask"].shape[1] + 2) for stamp in stamps)
        if rows * cols - frame_h * frame_w < min_free:
            continue
        if rows <= frame_h + 1 or cols <= frame_w + 1:
            continue

        frame_top = rng.randint(1, rows - frame_h - 1)
        frame_left = rng.randint(1, cols - frame_w - 1)
        frame = (frame_top, frame_left, frame_top + frame_h - 1, frame_left + frame_w - 1)
        grid = np.zeros((rows, cols), dtype=int)
        grid[frame[0], frame[1] : frame[3] + 1] = GREY
        grid[frame[2], frame[1] : frame[3] + 1] = GREY
        grid[frame[0] : frame[2] + 1, frame[1]] = GREY
        grid[frame[0] : frame[2] + 1, frame[3]] = GREY

        placements_global: list[dict[str, Any]] = []
        anchor_map: dict[int, tuple[int, int]] = {}
        for placement in placements_local:
            shifted = placement["coords"] + np.array([frame_top + 1, frame_left + 1], dtype=int)
            anchor_r = frame_top + 1 + placement["anchor"][0]
            anchor_c = frame_left + 1 + placement["anchor"][1]
            grid[anchor_r, anchor_c] = int(placement["locator_color"])
            anchor_map[int(placement["locator_color"])] = (anchor_r, anchor_c)
            placements_global.append(
                {
                    "coords": shifted.astype(int),
                    "anchor": (anchor_r, anchor_c),
                    "rotation": placement["rotation"],
                    "main_color": placement["main_color"],
                    "locator_color": placement["locator_color"],
                }
            )

        inner_box = (frame[0] + 1, frame[1] + 1, frame[2] - 1, frame[3] - 1)
        candidate_counts = [
            len(_placement_candidates(stamp, anchor_map[int(stamp["locator_color"])], inner_box))
            for stamp in stamps
        ]
        candidate_product = 1
        for count in candidate_counts:
            candidate_product *= count
        if max(candidate_counts) > MAX_CANDIDATES_PER_STAMP:
            continue
        if candidate_product > MAX_CANDIDATE_PRODUCT:
            continue
        if require_ambiguity and max(candidate_counts) != 2:
            continue
        if not require_ambiguity and max(candidate_counts) != 1:
            continue

        if not _place_source_stamps(rng, grid, frame, stamps):
            continue

        details = _solve_details(grid, max_solutions=2)
        if len(details["solutions"]) != 1:
            continue
        expected = _render_expected(grid.shape, frame, placements_global)
        if not np.array_equal(expected, _render_output(grid.shape, details["frame"], details["solutions"][0])):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-and-canvas instance")
