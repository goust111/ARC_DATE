from __future__ import annotations

from collections import deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _background(grid: np.ndarray) -> int:
    """Return the most frequent (visible background) color."""
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    """Return boolean-mask components as arrays of (row, column) coordinates."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    result: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue = deque([(sr, sc)])
        cells: list[Coord] = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for dr, dc in steps:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_cells(box: BBox) -> set[Coord]:
    r0, c0, r1, c1 = box
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells.update((r1, c) for c in range(c0, c1 + 1))
    cells.update((r, c0) for r in range(r0 + 1, r1))
    cells.update((r, c1) for r in range(r0 + 1, r1))
    return cells


def _frame_records(grid: np.ndarray, bg: int) -> list[dict]:
    """Locate full frames and one-gap frame-like objects without using color order."""
    records: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for coords in _components(grid == color, connectivity=8):
            if len(coords) < 7:
                continue
            box = _bbox(coords)
            r0, c0, r1, c1 = box
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            border = _border_cells(box)
            hits = sum(int(grid[r, c]) == color for r, c in border)
            # The second case lets the solver recognize and exclude the interiors
            # of the deliberately broken-frame decoys from the marker field.
            if hits < len(border) - 1:
                continue
            records.append(
                {
                    "color": color,
                    "bbox": box,
                    "complete": hits == len(border),
                    "border_hits": hits,
                }
            )
    return records


def _outside_all_frames(rc: Coord, records: list[dict]) -> bool:
    r, c = rc
    return all(not (r0 <= r <= r1 and c0 <= c <= c1) for r0, c0, r1, c1 in (x["bbox"] for x in records))


def _is_visibly_isolated(grid: np.ndarray, rc: Coord, bg: int) -> bool:
    r, c = rc
    h, w = grid.shape
    for nr in range(max(0, r - 1), min(h, r + 2)):
        for nc in range(max(0, c - 1), min(w, c + 2)):
            if (nr, nc) != (r, c) and int(grid[nr, nc]) != bg:
                return False
    return True


def _vote_count(grid: np.ndarray, color: int, records: list[dict], bg: int) -> int:
    votes = 0
    for comp in _components(grid == int(color), connectivity=8):
        if len(comp) != 1:
            continue
        rc = tuple(map(int, comp[0]))
        if _outside_all_frames(rc, records) and _is_visibly_isolated(grid, rc, bg):
            votes += 1
    return votes


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the interior of the complete rectangular frame whose outside singleton votes equal its interior area.

    Input structure:
    - A dominant background contains several differently colored one-cell rectangular frames with multicolor interiors.
    - Some frame-like objects have a border gap, and isolated outside pixels reuse the corresponding frame colors.

    Rule:
    - Reject every frame whose four border sides are not complete. For each remaining frame, count visibly isolated same-color pixels lying outside all frame boxes.
    - Select the unique complete frame whose vote count equals its strict interior area, then return that interior unchanged and without its border.

    Key insight:
    - Completeness is a mandatory first filter; the answer is found by equating two separate quantities, outside singleton count and inside geometric area, rather than by choosing the largest frame or most common vote color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _background(g)
    records = _frame_records(g, bg)
    complete = [record for record in records if record["complete"]]
    assert complete, "expected at least one complete rectangular frame"

    matches = []
    for record in complete:
        r0, c0, r1, c1 = record["bbox"]
        area = (r1 - r0 - 1) * (c1 - c0 - 1)
        votes = _vote_count(g, int(record["color"]), records, bg)
        if votes == area:
            matches.append(record)

    assert len(matches) == 1, "expected exactly one complete frame with count equal to area"
    r0, c0, r1, c1 = matches[0]["bbox"]
    return g[r0 + 1 : r1, c0 + 1 : c1].copy()


# ------------------------------ generator ------------------------------

_TARGET_DIMS = ((2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (3, 3))
_LARGE_DIMS = ((3, 4), (4, 3), (3, 5), (5, 3), (4, 4))
_OTHER_DIMS = ((2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (3, 4), (4, 3))
_MAX_ATTEMPTS = 500
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _random_interior(rng: random.Random, h: int, w: int, colors: list[int]) -> np.ndarray:
    """Make a sparse, visibly multicolor pattern that cannot mimic a frame."""
    out = np.zeros((h, w), dtype=int)
    cell_count = h * w
    n_colored = rng.randint(max(2, cell_count // 3), max(2, cell_count - 1))
    chosen = rng.sample(range(cell_count), n_colored)
    motif_colors = rng.sample(colors, k=min(3, len(colors)))
    for i, index in enumerate(chosen):
        out.flat[index] = motif_colors[i % 2] if i < 2 else rng.choice(motif_colors)
    return out


def _make_plans(rng: random.Random) -> list[dict]:
    n_frames = rng.choice((3, 4))
    frame_colors = rng.sample(range(1, 10), n_frames)
    interior_colors = [c for c in range(1, 10) if c not in frame_colors]

    target_dims = rng.choice(_TARGET_DIMS)
    target_area = target_dims[0] * target_dims[1]
    large_dims = rng.choice([dims for dims in _LARGE_DIMS if dims[0] * dims[1] > target_area])
    large_area = large_dims[0] * large_dims[1]
    large_vote_options = [v for v in range(target_area + 1, target_area + 5) if v != large_area]
    large_votes = rng.choice(large_vote_options)
    broken_dims = rng.choice(_OTHER_DIMS)
    broken_area = broken_dims[0] * broken_dims[1]

    specs = [
        ("target", target_dims, target_area, True),
        ("larger_complete", large_dims, large_votes, True),
        ("broken_equal", broken_dims, broken_area, False),
    ]
    if n_frames == 4:
        extra_dims = rng.choice(_OTHER_DIMS)
        extra_area = extra_dims[0] * extra_dims[1]
        extra_votes = rng.choice([v for v in range(2, 12) if v != extra_area])
        specs.append(("extra_complete", extra_dims, extra_votes, True))

    plans = []
    used_patterns: set[tuple[tuple[int, int], bytes]] = set()
    for color, (role, (ih, iw), votes, complete) in zip(frame_colors, specs):
        for _ in range(50):
            interior = _random_interior(rng, ih, iw, interior_colors)
            signature = (interior.shape, interior.tobytes())
            if signature not in used_patterns:
                used_patterns.add(signature)
                break
        plans.append(
            {
                "role": role,
                "color": int(color),
                "ih": ih,
                "iw": iw,
                "area": ih * iw,
                "votes": int(votes),
                "complete": bool(complete),
                "interior": interior,
            }
        )
    rng.shuffle(plans)
    return plans


def _boxes_too_close(a: BBox, b: BBox, margin: int = 2) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _place_frames(rng: random.Random, grid: np.ndarray, plans: list[dict]) -> bool:
    h, w = grid.shape
    boxes: list[BBox] = []
    for plan in sorted(plans, key=lambda x: (x["ih"] + 2) * (x["iw"] + 2), reverse=True):
        oh, ow = plan["ih"] + 2, plan["iw"] + 2
        candidates = [
            (r, c)
            for r in range(1, h - oh)
            for c in range(1, w - ow)
            if all(not _boxes_too_close((r, c, r + oh - 1, c + ow - 1), old) for old in boxes)
        ]
        if not candidates:
            return False
        r0, c0 = rng.choice(candidates)
        r1, c1 = r0 + oh - 1, c0 + ow - 1
        box = (r0, c0, r1, c1)
        color = plan["color"]
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        grid[r0 + 1 : r1, c0 + 1 : c1] = plan["interior"]

        gap = None
        if not plan["complete"]:
            side = rng.choice(("top", "bottom", "left", "right"))
            if side in ("top", "bottom"):
                gap = (r0 if side == "top" else r1, rng.randint(c0 + 1, c1 - 1))
            else:
                gap = (rng.randint(r0 + 1, r1 - 1), c0 if side == "left" else c1)
            grid[gap] = 0
        plan["bbox"] = box
        plan["gap"] = gap
        boxes.append(box)
    return True


def _marker_candidates(grid: np.ndarray, boxes: list[BBox]) -> list[Coord]:
    bg = 0
    h, w = grid.shape
    candidates = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != bg:
                continue
            if any(r0 <= r <= r1 and c0 <= c <= c1 for r0, c0, r1, c1 in boxes):
                continue
            if _is_visibly_isolated(grid, (r, c), bg):
                candidates.append((r, c))
    return candidates


def _place_markers(rng: random.Random, grid: np.ndarray, plans: list[dict]) -> bool:
    boxes = [plan["bbox"] for plan in plans]
    ballot = [plan["color"] for plan in plans for _ in range(plan["votes"])]
    rng.shuffle(ballot)
    positions: dict[int, list[Coord]] = {plan["color"]: [] for plan in plans}
    for color in ballot:
        candidates = _marker_candidates(grid, boxes)
        if not candidates:
            return False
        rc = rng.choice(candidates)
        grid[rc] = color
        positions[color].append(rc)
    for plan in plans:
        plan["markers"] = positions[plan["color"]]
    return True


def _semantic_validation(grid: np.ndarray, plans: list[dict]) -> bool:
    """Validate the stated semantics and the anti-shortcut counterexamples."""
    if grid.ndim != 2 or max(grid.shape) > 30 or _background(grid) != 0:
        return False
    if len({plan["color"] for plan in plans}) != len(plans):
        return False

    records = _frame_records(grid, 0)
    expected_boxes = {plan["bbox"] for plan in plans}
    actual_boxes = {record["bbox"] for record in records}
    if actual_boxes != expected_boxes or len(records) != len(plans):
        return False

    record_by_box = {record["bbox"]: record for record in records}
    for plan in plans:
        record = record_by_box[plan["bbox"]]
        if bool(record["complete"]) != plan["complete"]:
            return False
        if _vote_count(grid, plan["color"], records, 0) != plan["votes"]:
            return False
        if len(plan["markers"]) != plan["votes"]:
            return False
        if not plan["complete"] and record["border_hits"] != len(_border_cells(plan["bbox"])) - 1:
            return False

    target = next(plan for plan in plans if plan["role"] == "target")
    broken = next(plan for plan in plans if plan["role"] == "broken_equal")
    complete = [plan for plan in plans if plan["complete"]]
    equality_matches = [plan for plan in complete if plan["votes"] == plan["area"]]
    if equality_matches != [target] or broken["votes"] != broken["area"]:
        return False
    if max(complete, key=lambda x: x["area"]) is target:
        return False
    if max(complete, key=lambda x: x["votes"]) is target:
        return False

    # Distinct interiors and a direct, metadata-independent solver comparison
    # rule out fixed outputs and accidental selection of a decoy.
    patterns = {(plan["interior"].shape, plan["interior"].tobytes()) for plan in plans}
    if len(patterns) != len(plans):
        return False
    try:
        actual = main(grid)
    except (AssertionError, ValueError):
        return False
    return np.array_equal(actual, target["interior"])


def _generate_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    grid = np.zeros((rng.randint(24, 30), rng.randint(24, 30)), dtype=int)
    plans = _make_plans(rng)
    if not _place_frames(rng, grid, plans):
        return None
    if not _place_markers(rng, grid, plans):
        return None
    if not _semantic_validation(grid, plans):
        return None
    target = next(plan for plan in plans if plan["role"] == "target")
    return grid, target["interior"]


def generate_input() -> np.ndarray:
    """Generate a diverse, uniquely solvable scene with explicit shortcut counterexamples."""
    rng = random.Random()
    for _ in range(_MAX_ATTEMPTS):
        result = _generate_attempt(rng)
        if result is None:
            continue
        grid, expected = result
        signature = (expected.shape, expected.tobytes())
        if signature in _SEEN_OUTPUTS:
            continue
        _SEEN_OUTPUTS.add(signature)
        return grid
    raise RuntimeError("failed to generate a valid count-equals-area frame puzzle")
