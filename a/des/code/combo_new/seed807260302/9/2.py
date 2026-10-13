import random
from collections import Counter

import numpy as np


BG = 8
FRAME = 5
PIECE_COLORS = [1, 2, 3, 4, 6, 7, 9]
MAX_NOTCH_DEPTH = 2
WALL_THICKNESS = MAX_NOTCH_DEPTH + 1
GEN_RETRIES = 300
SPEC_RETRIES = 200
SCATTER_RETRIES = 200


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _mask_from_coords(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0)


def _mask_coords(mask: np.ndarray, top: int = 0, left: int = 0) -> list[tuple[int, int]]:
    pts = np.argwhere(mask)
    return [(int(top + r), int(left + c)) for r, c in pts]


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _rect_contains(rect: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = rect
    return r0 <= r <= r1 and c0 <= c <= c1


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _rect_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    return np.pad(mask.astype(int), ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)


def _rect_sum(pref: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(pref[r1 + 1, c1 + 1] - pref[r0, c1 + 1] - pref[r1 + 1, c0] + pref[r0, c0])


def _largest_true_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    h, w = mask.shape
    pref = _prefix_sum(mask)
    best_rect = None
    best_score = None
    for r0 in range(h):
        for r1 in range(r0, h):
            for c0 in range(w):
                for c1 in range(c0, w):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if _rect_sum(pref, r0, c0, r1, c1) != area:
                        continue
                    height = r1 - r0 + 1
                    width = c1 - c0 + 1
                    score = (area, min(height, width), max(height, width), -r0, -c0)
                    if best_score is None or score > best_score:
                        best_score = score
                        best_rect = (r0, c0, r1, c1)
    assert best_rect is not None
    return best_rect


def _canonical_rotation_key(mask: np.ndarray) -> bytes:
    keys = []
    for k in range(4):
        rot = np.rot90(mask, k)
        keys.append(f"{rot.shape[0]}x{rot.shape[1]}:".encode() + rot.astype(np.uint8).tobytes())
    return min(keys)


def _extract_piece_components(grid: np.ndarray, bg: int, frame_coords: set[tuple[int, int]]) -> list[dict]:
    non_frame = (grid != bg)
    for r, c in frame_coords:
        non_frame[r, c] = False
    pieces = []
    for pts in _components(non_frame):
        color = int(grid[pts[0]])
        mask, (r0, c0) = _mask_from_coords(pts)
        pieces.append(
            {
                "coords": pts,
                "color": color,
                "mask": mask,
                "bbox": (r0, c0, r0 + mask.shape[0] - 1, c0 + mask.shape[1] - 1),
                "area": int(mask.sum()),
            }
        )
    return pieces


def _scene_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    best_frame = None
    best_size = -1
    for color in [int(v) for v in np.unique(g) if int(v) != bg]:
        for pts in _components(g == color):
            if len(pts) > best_size:
                best_size = len(pts)
                best_frame = {"color": color, "coords": pts}
    assert best_frame is not None

    frame_coords = set(best_frame["coords"])
    fr0, fc0, fr1, fc1 = _bbox(frame_coords)
    sub = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    bg_rect = _largest_true_rectangle(sub == bg)
    ir0 = fr0 + bg_rect[0]
    ic0 = fc0 + bg_rect[1]
    ir1 = fr0 + bg_rect[2]
    ic1 = fc0 + bg_rect[3]
    core_rect = (ir0, ic0, ir1, ic1)

    notch_mask = np.zeros_like(g, dtype=bool)
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if g[r, c] != bg:
                continue
            if _rect_contains(core_rect, r, c):
                continue
            notch_mask[r, c] = True

    notch_components = []
    for idx, pts in enumerate(_components(notch_mask)):
        notch_components.append({"id": idx, "coords": set(pts), "area": len(pts)})

    pieces = _extract_piece_components(g, bg, frame_coords)
    return {
        "grid": g,
        "bg": bg,
        "frame_color": int(best_frame["color"]),
        "frame_coords": frame_coords,
        "outer_bbox": (fr0, fc0, fr1, fc1),
        "core_rect": core_rect,
        "notches": notch_components,
        "pieces": pieces,
    }


def _piece_candidates(struct: dict, piece: dict) -> list[dict]:
    g = struct["grid"]
    bg = struct["bg"]
    fr0, fc0, fr1, fc1 = struct["outer_bbox"]
    core_rect = struct["core_rect"]
    notch_lookup = {frozenset(n["coords"]): n["id"] for n in struct["notches"]}
    candidates = []
    seen = set()

    for k in range(4):
        rmask = np.rot90(piece["mask"], k)
        h, w = rmask.shape
        rel_coords = np.argwhere(rmask)
        for top in range(fr0, fr1 - h + 2):
            for left in range(fc0, fc1 - w + 2):
                placed = []
                outside = []
                ok = True
                for rr, cc in rel_coords:
                    r = int(top + rr)
                    c = int(left + cc)
                    if g[r, c] != bg:
                        ok = False
                        break
                    placed.append((r, c))
                    if not _rect_contains(core_rect, r, c):
                        outside.append((r, c))
                if not ok or not outside:
                    continue
                outside_key = frozenset(outside)
                notch_id = notch_lookup.get(outside_key)
                if notch_id is None:
                    continue
                sig = (k, top, left, notch_id)
                if sig in seen:
                    continue
                seen.add(sig)
                candidates.append(
                    {
                        "piece_color": piece["color"],
                        "piece_area": piece["area"],
                        "rotation": k,
                        "top": top,
                        "left": left,
                        "notch_id": notch_id,
                        "coords": set(placed),
                    }
                )
    return candidates


def _assign_candidates(all_candidates: list[list[dict]], notch_count: int) -> list[dict]:
    piece_count = len(all_candidates)
    assert piece_count == notch_count
    order = sorted(range(piece_count), key=lambda i: len(all_candidates[i]))
    chosen: list[dict | None] = [None] * piece_count
    used: set[int] = set()

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return len(used) == notch_count
        idx = order[pos]
        for cand in all_candidates[idx]:
            notch_id = int(cand["notch_id"])
            if notch_id in used:
                continue
            chosen[idx] = cand
            used.add(notch_id)
            if dfs(pos + 1):
                return True
            used.remove(notch_id)
            chosen[idx] = None
        return False

    ok = dfs(0)
    assert ok
    return [cand for cand in chosen if cand is not None]


def _solve(grid: np.ndarray) -> np.ndarray:
    struct = _scene_structure(grid)
    g = struct["grid"]
    bg = struct["bg"]
    pieces = struct["pieces"]
    notches = struct["notches"]
    assert pieces
    assert len(pieces) == len(notches)

    all_candidates = [_piece_candidates(struct, piece) for piece in pieces]
    assert all(all_candidates)
    placed = _assign_candidates(all_candidates, len(notches))

    out = g.copy()
    piece_mask = np.zeros_like(out, dtype=bool)
    for piece in pieces:
        for r, c in piece["coords"]:
            piece_mask[r, c] = True
    out[piece_mask] = bg

    placed.sort(key=lambda item: (-int(item["piece_area"]), int(item["piece_color"])))
    for item in placed:
        for r, c in item["coords"]:
            out[r, c] = int(item["piece_color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered monochrome fragments back into the grey frame by matching jagged notch contours.

    Input structure:
    - The background is azure.
    - A large grey rectangular frame contains a rectangular inner cavity, while extra background cells along the inner wall form several irregular notch shapes.
    - Outside the frame, several disconnected single-color fragments are scattered; each fragment is mostly a rectangle with one jagged edge.

    Rule:
    - Treat the largest grey component as the template frame and the largest background rectangle inside it as the core cavity.
    - Each background component inside the frame but outside that core cavity is a wall notch; find the scattered fragment that can be rotated and translated so its pixels outside the core cavity match that notch exactly.
    - Move every matched fragment into the frame, erase its original outside location to background, and color the inserted cells with the fragment's original color.
    - If inserted fragments overlap inside the cavity, paste them in descending area order so larger fragments stay underneath smaller ones.

    Key insight: The frame can be decomposed into a large rectangular cavity plus smaller notch components, so solving reduces to rotation-based shape matching where a fragment's protruding edge must coincide exactly with one notch.
    """
    return _solve(np.asarray(grid, dtype=int))


def _make_profile(rng: random.Random, span: int) -> tuple[int, ...]:
    for _ in range(50):
        profile = [rng.randint(1, MAX_NOTCH_DEPTH) for _ in range(span)]
        if len(set(profile)) >= 2:
            return tuple(profile)
    return tuple([1] * (span - 1) + [2])


def _piece_target(
    side: str,
    core_rect: tuple[int, int, int, int],
    offset: int,
    span: int,
    body_depth: int,
    profile: tuple[int, ...],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[int, int, int, int]]:
    ir0, ic0, ir1, ic1 = core_rect
    coords: set[tuple[int, int]] = set()
    notch: set[tuple[int, int]] = set()

    if side == "top":
        for j, depth in enumerate(profile):
            c = ic0 + offset + j
            for r in range(ir0 - depth, ir0):
                notch.add((r, c))
        for r in range(ir0, ir0 + body_depth):
            for c in range(ic0 + offset, ic0 + offset + span):
                coords.add((r, c))
        body_rect = (ir0, ic0 + offset, ir0 + body_depth - 1, ic0 + offset + span - 1)
    elif side == "bottom":
        for j, depth in enumerate(profile):
            c = ic0 + offset + j
            for r in range(ir1 + 1, ir1 + depth + 1):
                notch.add((r, c))
        for r in range(ir1 - body_depth + 1, ir1 + 1):
            for c in range(ic0 + offset, ic0 + offset + span):
                coords.add((r, c))
        body_rect = (ir1 - body_depth + 1, ic0 + offset, ir1, ic0 + offset + span - 1)
    elif side == "left":
        for j, depth in enumerate(profile):
            r = ir0 + offset + j
            for c in range(ic0 - depth, ic0):
                notch.add((r, c))
        for r in range(ir0 + offset, ir0 + offset + span):
            for c in range(ic0, ic0 + body_depth):
                coords.add((r, c))
        body_rect = (ir0 + offset, ic0, ir0 + offset + span - 1, ic0 + body_depth - 1)
    else:
        assert side == "right"
        for j, depth in enumerate(profile):
            r = ir0 + offset + j
            for c in range(ic1 + 1, ic1 + depth + 1):
                notch.add((r, c))
        for r in range(ir0 + offset, ir0 + offset + span):
            for c in range(ic1 - body_depth + 1, ic1 + 1):
                coords.add((r, c))
        body_rect = (ir0 + offset, ic1 - body_depth + 1, ir0 + offset + span - 1, ic1)

    coords |= notch
    return coords, notch, body_rect


def _shape_signature(coords: set[tuple[int, int]]) -> tuple[bytes, int]:
    mask, _ = _mask_from_coords(coords)
    return _canonical_rotation_key(mask), int(mask.sum())


def _pairwise_overlap(specs: list[dict]) -> bool:
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            if specs[i]["target_coords"] & specs[j]["target_coords"]:
                return True
    return False


def _build_specs(
    rng: random.Random,
    core_rect: tuple[int, int, int, int],
    want_overlap: bool,
) -> list[dict] | None:
    cavity_h = core_rect[2] - core_rect[0] + 1
    cavity_w = core_rect[3] - core_rect[1] + 1
    side_pool = ["top", "bottom", "left", "right"]

    if want_overlap:
        n = rng.choice([2, 3, 4])
    else:
        n = rng.choice([1, 2, 3, 4])
    sides = rng.sample(side_pool, n)

    specs = []
    sigs: set[bytes] = set()
    areas: set[int] = set()
    colors = rng.sample(PIECE_COLORS, n)

    for side, color in zip(sides, colors):
        axis_len = cavity_w if side in ("top", "bottom") else cavity_h
        depth_cap = cavity_h - 1 if side in ("top", "bottom") else cavity_w - 1
        max_span = min(axis_len - 2, 8)
        if max_span < 4 or depth_cap < 2:
            return None
        span = rng.randint(4, max_span)
        max_offset = axis_len - span - 1
        if max_offset < 1:
            return None
        offset = rng.randint(1, max_offset)
        body_depth = rng.randint(2, min(5, depth_cap))
        profile = _make_profile(rng, span)

        target_coords, notch_coords, body_rect = _piece_target(side, core_rect, offset, span, body_depth, profile)
        key, area = _shape_signature(target_coords)
        if key in sigs or area in areas:
            return None
        sigs.add(key)
        areas.add(area)
        specs.append(
            {
                "side": side,
                "color": color,
                "offset": offset,
                "span": span,
                "body_depth": body_depth,
                "profile": profile,
                "target_coords": target_coords,
                "notch_coords": notch_coords,
                "body_rect": body_rect,
                "area": area,
            }
        )

    overlap = _pairwise_overlap(specs)
    if want_overlap != overlap:
        return None
    return specs


def _scatter_piece(
    rng: random.Random,
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    mask: np.ndarray,
    outer_bbox: tuple[int, int, int, int],
) -> bool:
    h, w = grid.shape
    rotated = np.rot90(mask, rng.randrange(4))
    ph, pw = rotated.shape
    coords_local = np.argwhere(rotated)
    candidates = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    rng.shuffle(candidates)

    fr0, fc0, fr1, fc1 = outer_bbox
    padded_frame = (max(0, fr0 - 1), max(0, fc0 - 1), min(h - 1, fr1 + 1), min(w - 1, fc1 + 1))

    for top, left in candidates:
        placed = []
        ok = True
        for rr, cc in coords_local:
            r = int(top + rr)
            c = int(left + cc)
            if blocked[r, c]:
                ok = False
                break
            if _rect_contains(padded_frame, r, c):
                ok = False
                break
            placed.append((r, c))
        if not ok:
            continue
        for r, c in placed:
            grid[r, c] = color
        for r, c in placed:
            for nr in range(max(0, r - 1), min(h, r + 2)):
                for nc in range(max(0, c - 1), min(w, c + 2)):
                    blocked[nr, nc] = True
        return True
    return False


def _expected_output(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    piece_colors = {int(spec["color"]) for spec in specs}
    for color in piece_colors:
        out[out == color] = BG
    for spec in sorted(specs, key=lambda item: (-int(item["area"]), int(item["color"]))):
        for r, c in spec["target_coords"]:
            out[r, c] = int(spec["color"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    cavity_h = rng.randint(8, 14)
    cavity_w = rng.randint(8, 14)
    outer_h = cavity_h + 2 * WALL_THICKNESS
    outer_w = cavity_w + 2 * WALL_THICKNESS

    min_h = max(outer_h + 6, 20)
    min_w = max(outer_w + 8, 20)
    if min_h > 30 or min_w > 30:
        return None
    h = rng.randint(min_h, 30)
    w = rng.randint(min_w, 30)

    fr0 = rng.randint(2, h - outer_h - 2)
    fc0 = rng.randint(2, w - outer_w - 2)
    fr1 = fr0 + outer_h - 1
    fc1 = fc0 + outer_w - 1
    core_rect = (
        fr0 + WALL_THICKNESS,
        fc0 + WALL_THICKNESS,
        fr1 - WALL_THICKNESS,
        fc1 - WALL_THICKNESS,
    )

    want_overlap = rng.random() < 0.5
    specs = None
    for _ in range(SPEC_RETRIES):
        specs = _build_specs(rng, core_rect, want_overlap)
        if specs is not None:
            break
    if specs is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    grid[fr0 : fr1 + 1, fc0 : fc1 + 1] = FRAME
    ir0, ic0, ir1, ic1 = core_rect
    grid[ir0 : ir1 + 1, ic0 : ic1 + 1] = BG
    for spec in specs:
        for r, c in spec["notch_coords"]:
            grid[r, c] = BG

    blocked = np.zeros((h, w), dtype=bool)
    for r in range(max(0, fr0 - 1), min(h, fr1 + 2)):
        for c in range(max(0, fc0 - 1), min(w, fc1 + 2)):
            blocked[r, c] = True

    for spec in specs:
        mask, _ = _mask_from_coords(spec["target_coords"])
        if not _scatter_piece(rng, grid, blocked, int(spec["color"]), mask, (fr0, fc0, fr1, fc1)):
            return None

    expected = _expected_output(grid, specs)
    solved = _solve(grid)
    if not np.array_equal(expected, solved):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
