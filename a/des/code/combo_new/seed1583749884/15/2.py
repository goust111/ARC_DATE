from __future__ import annotations

import random

import numpy as np


DIR_VECTORS = {
    "E": (0, 1),
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
}
OPPOSITE = {"E": "W", "N": "S", "S": "N", "W": "E"}


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    seen = np.zeros((height, width), dtype=bool)
    comps: list[dict] = []

    for r in range(height):
        for c in range(width):
            color = int(g[r, c])
            if color == int(bg) or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = int(cr + dr), int(cc + dc)
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            rows = [r0 for r0, _ in coords]
            cols = [c0 for _, c0 in coords]
            comps.append(
                {
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "color": color,
                    "coords": coords,
                    "coord_set": frozenset(coords),
                }
            )

    comps.sort(key=lambda comp: comp["bbox"])
    return comps


def _is_contiguous(indices: list[int]) -> bool:
    if not indices:
        return False
    return indices[-1] - indices[0] + 1 == len(indices)


def _profile_candidates(
    profile: list[int],
    side: str,
    start_coord: int,
    edge_coord: int,
) -> list[dict]:
    unique = sorted(set(profile))
    if len(unique) != 2 or unique[1] - unique[0] != 1:
        return []

    if side == "E":
        role_map = {"notch": unique[0], "bump": unique[1]}
    elif side == "W":
        role_map = {"notch": unique[1], "bump": unique[0]}
    elif side == "N":
        role_map = {"notch": unique[1], "bump": unique[0]}
    else:
        role_map = {"notch": unique[0], "bump": unique[1]}

    candidates: list[dict] = []
    all_indices = list(range(len(profile)))
    for kind, value in role_map.items():
        segment = [idx for idx, prof in enumerate(profile) if prof == value]
        remainder = [idx for idx in all_indices if idx not in segment]
        if 0 < len(segment) < len(profile) and _is_contiguous(segment) and not _is_contiguous(remainder):
            candidates.append(
                {
                    "dir": side,
                    "edge": int(edge_coord),
                    "kind": kind,
                    "segment": (int(start_coord + segment[0]), int(start_coord + segment[-1])),
                }
            )
    return candidates


def _describe_component(component: dict) -> dict:
    r0, c0, r1, c1 = component["bbox"]
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)

    left = [width + c1 + 1] * height
    right = [c0 - 1] * height
    top = [height + r1 + 1] * width
    bottom = [r0 - 1] * width

    for r, c in component["coords"]:
        rr = int(r - r0)
        cc = int(c - c0)
        left[rr] = min(left[rr], int(c))
        right[rr] = max(right[rr], int(c))
        top[cc] = min(top[cc], int(r))
        bottom[cc] = max(bottom[cc], int(r))

    assert all(val <= c1 for val in left) and all(val >= c0 for val in right)
    assert all(val <= r1 for val in top) and all(val >= r0 for val in bottom)

    candidates: list[dict] = []
    candidates.extend(_profile_candidates(right, "E", r0, c1))
    candidates.extend(_profile_candidates(left, "W", r0, c0))
    candidates.extend(_profile_candidates(top, "N", c0, r0))
    candidates.extend(_profile_candidates(bottom, "S", c0, r1))
    assert len(candidates) == 1, "each object must have one unique notch or bump side"

    description = dict(component)
    description.update(candidates[0])
    return description


def _candidate_move(
    notch: dict,
    bump: dict,
    occupied_without_notch: frozenset[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[int, int, int] | None:
    if OPPOSITE[notch["dir"]] != bump["dir"]:
        return None
    if notch["segment"] != bump["segment"]:
        return None

    shift = int(bump["edge"] - notch["edge"])
    if notch["dir"] == "E" and shift <= 0:
        return None
    if notch["dir"] == "W" and shift >= 0:
        return None
    if notch["dir"] == "S" and shift <= 0:
        return None
    if notch["dir"] == "N" and shift >= 0:
        return None

    dr, dc = DIR_VECTORS[notch["dir"]]
    dr *= abs(shift)
    dc *= abs(shift)
    distance = abs(shift)
    height, width = shape

    for step in range(1, distance + 1):
        step_r = 0 if dr == 0 else int(np.sign(dr) * step)
        step_c = 0 if dc == 0 else int(np.sign(dc) * step)
        moved = {(r + step_r, c + step_c) for r, c in notch["coord_set"]}
        if any(r < 0 or r >= height or c < 0 or c >= width for r, c in moved):
            return None
        if moved & occupied_without_notch:
            return None

    return int(dr), int(dc), int(distance)


def _candidate_edges(descriptions: list[dict], shape: tuple[int, int]) -> tuple[list[dict], list[dict], dict[int, list[tuple[int, int, int, int]]]]:
    notches = [desc for desc in descriptions if desc["kind"] == "notch"]
    bumps = [desc for desc in descriptions if desc["kind"] == "bump"]
    all_occupied = frozenset(coord for desc in descriptions for coord in desc["coord_set"])

    edges: dict[int, list[tuple[int, int, int, int]]] = {}
    for ni, notch in enumerate(notches):
        occupied_without_notch = all_occupied - notch["coord_set"]
        options: list[tuple[int, int, int, int]] = []
        for bi, bump in enumerate(bumps):
            move = _candidate_move(notch, bump, occupied_without_notch, shape)
            if move is None:
                continue
            dr, dc, distance = move
            options.append((bi, dr, dc, distance))
        options.sort(key=lambda item: (item[3], bumps[item[0]]["bbox"]))
        edges[ni] = options

    return notches, bumps, edges


def _solve_matching(edges: dict[int, list[tuple[int, int, int, int]]], bumps: list[dict]) -> dict[int, tuple[int, int, int]]:
    inbound = {bi: 0 for bi in range(len(bumps))}
    for options in edges.values():
        for bi, _dr, _dc, _dist in options:
            inbound[bi] += 1

    if all(len(options) == 1 for options in edges.values()) and all(count == 1 for count in inbound.values()):
        return {ni: (options[0][0], options[0][1], options[0][2]) for ni, options in edges.items()}

    order = sorted(edges, key=lambda ni: (len(edges[ni]),))
    best_score: int | None = None
    best_assignment: dict[int, tuple[int, int, int]] | None = None
    used_bumps: set[int] = set()
    current: dict[int, tuple[int, int, int]] = {}

    def search(pos: int, total_distance: int) -> None:
        nonlocal best_assignment, best_score
        if best_score is not None and total_distance >= best_score:
            return
        if pos == len(order):
            best_score = total_distance
            best_assignment = dict(current)
            return

        notch_index = order[pos]
        for bump_index, dr, dc, distance in edges[notch_index]:
            if bump_index in used_bumps:
                continue
            used_bumps.add(bump_index)
            current[notch_index] = (bump_index, dr, dc)
            search(pos + 1, total_distance + distance)
            used_bumps.remove(bump_index)
            del current[notch_index]

    search(0, 0)
    assert best_assignment is not None and len(best_assignment) == len(edges), "expected a complete matching"
    return best_assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each notched object along its unique socket direction until it interlocks with the matching protruding object.

    Input structure:
    - The background is a single dominant color.
    - Every foreground object is 4-connected and monochromatic.
    - Each object has exactly one modified side: either a one-cell-deep notch or a one-cell protrusion on one cardinal side.
    - Objects come in pairs: a notched object is aligned with a different object's complementary protrusion on the opposite side.

    Rule:
    - Detect the unique notch/protrusion side and the aligned segment for every object.
    - Pair each notched object with the protruding object whose complementary segment lies on the opposite side.
    - Translate the notched object straight along its notch direction until the protrusion exactly fills the notch.
    - Clear the notched object's original cells to background and keep all colors unchanged after docking.

    Key insight:
    - The pairing signal is geometric: the row/column interval of the notch matches the interval of the protrusion, and the notch object only moves along that side's axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _background_color(g)
    components = _components(g, bg)
    assert components and len(components) % 2 == 0, "expected an even number of foreground objects"

    descriptions = [_describe_component(component) for component in components]
    notches, bumps, edges = _candidate_edges(descriptions, g.shape)
    assert len(notches) == len(bumps) and len(notches) * 2 == len(descriptions)
    assert all(edges.values()), "each notched object must have at least one matching protrusion"

    assignment = _solve_matching(edges, bumps)
    out = g.copy()

    for notch in notches:
        for r, c in notch["coord_set"]:
            out[r, c] = int(bg)

    placed: list[tuple[int, frozenset[tuple[int, int]]]] = []
    occupied_final: set[tuple[int, int]] = set()
    fixed_occupied = set(coord for bump in bumps for coord in bump["coord_set"])

    for notch_index, notch in enumerate(notches):
        _bump_index, dr, dc = assignment[notch_index]
        moved = frozenset((r + dr, c + dc) for r, c in notch["coord_set"])
        assert not (moved & occupied_final), "moved notch objects must remain disjoint"
        assert not (moved & fixed_occupied), "moved object cannot overwrite a fixed object"
        occupied_final.update(moved)
        placed.append((int(notch["color"]), moved))

    for color, coords in placed:
        for r, c in coords:
            out[r, c] = color

    return out


GEN_BG = 8
GEN_BORDER = 1
GEN_BODY_HEIGHT = (4, 7)
GEN_BODY_WIDTH = (4, 6)
GEN_GAP = (2, 6)
GEN_GRID_HEIGHT = (16, 30)
GEN_GRID_WIDTH = (16, 30)
GEN_PAIR_COUNT = (1, 3)
GEN_PADDING = 2
GEN_RETRIES = 500


def _paste(canvas: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    mask = patch != int(bg)
    canvas[top : top + patch.shape[0], left : left + patch.shape[1]][mask] = patch[mask]


def _make_pair_patch() -> tuple[np.ndarray, np.ndarray]:
    body_h = random.randint(*GEN_BODY_HEIGHT)
    notch_w = random.randint(*GEN_BODY_WIDTH)
    bump_w = random.randint(*GEN_BODY_WIDTH)
    segment_len = random.randint(1, body_h - 2)
    segment_top = random.randint(1, body_h - segment_len - 1)
    gap = random.randint(*GEN_GAP)
    color_a, color_b = random.sample([c for c in range(10) if c != GEN_BG], 2)

    notch = np.full((body_h, notch_w), color_a, dtype=int)
    notch[segment_top : segment_top + segment_len, notch_w - 1] = GEN_BG

    bump = np.full((body_h, bump_w + 1), GEN_BG, dtype=int)
    bump[:, 1:] = color_b
    bump[segment_top : segment_top + segment_len, 0] = color_b

    patch_w = gap + notch_w + bump_w
    inp = np.full((body_h, patch_w), GEN_BG, dtype=int)
    out = np.full((body_h, patch_w), GEN_BG, dtype=int)

    _paste(inp, notch, 0, 0, GEN_BG)
    _paste(out, notch, 0, gap, GEN_BG)
    _paste(inp, bump, 0, gap + notch_w - 1, GEN_BG)
    _paste(out, bump, 0, gap + notch_w - 1, GEN_BG)

    rot_k = random.randrange(4)
    return np.rot90(inp, rot_k), np.rot90(out, rot_k)


def _boxes_separated(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], padding: int) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return (
        ar1 + padding < br0
        or br1 + padding < ar0
        or ac1 + padding < bc0
        or bc1 + padding < ac0
    )


def _place_pairs(pair_patches: list[tuple[np.ndarray, np.ndarray]], height: int, width: int) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.full((height, width), GEN_BG, dtype=int)
    out = np.full((height, width), GEN_BG, dtype=int)
    boxes: list[tuple[int, int, int, int]] = []

    ordered = sorted(pair_patches, key=lambda pair: pair[0].shape[0] * pair[0].shape[1], reverse=True)
    for inp_patch, out_patch in ordered:
        ph, pw = inp_patch.shape
        if ph + 2 * GEN_BORDER > height or pw + 2 * GEN_BORDER > width:
            return None

        candidates = [
            (r, c)
            for r in range(GEN_BORDER, height - ph - GEN_BORDER + 1)
            for c in range(GEN_BORDER, width - pw - GEN_BORDER + 1)
        ]
        random.shuffle(candidates)

        placed = False
        for top, left in candidates:
            box = (top, left, top + ph - 1, left + pw - 1)
            if all(_boxes_separated(box, other, GEN_PADDING) for other in boxes):
                boxes.append(box)
                _paste(inp, inp_patch, top, left, GEN_BG)
                _paste(out, out_patch, top, left, GEN_BG)
                placed = True
                break
        if not placed:
            return None

    return inp, out


def _bg_is_dominant(grid: np.ndarray) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    bg_index = int(np.where(values == GEN_BG)[0][0])
    bg_count = int(counts[bg_index])
    others = [int(count) for idx, count in enumerate(counts) if idx != bg_index]
    return not others or bg_count > max(others)


def _has_unique_pairing(grid: np.ndarray) -> bool:
    try:
        bg = _background_color(grid)
        descriptions = [_describe_component(component) for component in _components(grid, bg)]
        notches, bumps, edges = _candidate_edges(descriptions, grid.shape)
    except AssertionError:
        return False

    if len(notches) == 0 or len(notches) != len(bumps):
        return False
    if any(len(options) != 1 for options in edges.values()):
        return False

    inbound = [0] * len(bumps)
    for options in edges.values():
        inbound[options[0][0]] += 1
    return all(count == 1 for count in inbound)


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*GEN_PAIR_COUNT)
    pair_patches = [_make_pair_patch() for _ in range(pair_count)]
    height = random.randint(*GEN_GRID_HEIGHT)
    width = random.randint(*GEN_GRID_WIDTH)

    placed = _place_pairs(pair_patches, height, width)
    if placed is None:
        return None
    inp, expected = placed

    if not _bg_is_dominant(inp) or not _has_unique_pairing(inp):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
