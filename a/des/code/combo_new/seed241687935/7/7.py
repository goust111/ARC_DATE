from __future__ import annotations

import random

import numpy as np


BLACK = 0
MARKER_PALETTE = (1, 2, 3, 4, 6, 8, 9)
GREY = 5
ORANGE = 7
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 500


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "mask must contain at least one foreground cell"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return np.asarray(mask[r0:r1, c0:c1], dtype=bool)


def _mask_from_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[r - r0, c - c0] = True
    return out


def _rotations(mask: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(mask, k) for k in range(4)]


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((tuple(map(int, rot.shape)), rot.tobytes()) for rot in _rotations(mask))


def _distinct_rotation_count(mask: np.ndarray) -> int:
    return len({(tuple(map(int, rot.shape)), rot.tobytes()) for rot in _rotations(mask)})


def _has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for comp in _components(~mask):
        if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            return True
    return False


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _marker_candidates(mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    if side == "top":
        if h < 2:
            return out
        for c in range(w):
            if mask[0, c] and not mask[1, c]:
                out.append((1, c))
    elif side == "bottom":
        if h < 2:
            return out
        for c in range(w):
            if mask[h - 1, c] and not mask[h - 2, c]:
                out.append((h - 2, c))
    elif side == "left":
        if w < 2:
            return out
        for r in range(h):
            if mask[r, 0] and not mask[r, 1]:
                out.append((r, 1))
    else:
        assert side == "right"
        if w < 2:
            return out
        for r in range(h):
            if mask[r, w - 1] and not mask[r, w - 2]:
                out.append((r, w - 2))
    return out


def _random_connected_shape(side: str) -> tuple[np.ndarray, tuple[int, int]]:
    for _ in range(400):
        h = random.randint(2, 5)
        w = random.randint(2, 5)
        max_area = min(8, h * w - 1)
        if max_area < 3:
            continue
        target_area = random.randint(3, max_area)
        if side == "top":
            start = (0, random.randrange(w))
        elif side == "bottom":
            start = (h - 1, random.randrange(w))
        elif side == "left":
            start = (random.randrange(h), 0)
        else:
            start = (random.randrange(h), w - 1)

        cells = {start}
        while len(cells) < target_area:
            frontier: list[tuple[tuple[int, int], int]] = []
            seen = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        if side == "top":
                            weight = nr + 1
                        elif side == "bottom":
                            weight = h - nr
                        elif side == "left":
                            weight = nc + 1
                        else:
                            weight = w - nc
                        frontier.append(((nr, nc), int(weight)))
            if not frontier:
                break
            choices = [item for item, _ in frontier]
            weights = [weight for _, weight in frontier]
            cells.add(random.choices(choices, weights=weights, k=1)[0])

        if len(cells) != target_area:
            continue

        raw = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            raw[r, c] = True
        mask = _crop_mask(raw)

        if mask.all():
            continue
        if _has_hole(mask):
            continue
        marker_options = _marker_candidates(mask, side)
        if not marker_options:
            continue
        if _distinct_rotation_count(mask) < 2 and random.random() < 0.85:
            continue
        return mask, random.choice(marker_options)

    raise AssertionError("failed to generate a cavity shape")


def _match_piece_masks(piece_masks: list[np.ndarray], cavity_masks: list[np.ndarray]) -> dict[int, int]:
    candidates: list[list[int]] = []
    for piece_mask in piece_masks:
        matches: list[int] = []
        for cavity_idx, cavity_mask in enumerate(cavity_masks):
            for rot in _rotations(piece_mask):
                if np.array_equal(rot, cavity_mask):
                    matches.append(int(cavity_idx))
                    break
        assert matches, "every orange piece must match at least one cavity"
        candidates.append(matches)

    order = sorted(range(len(piece_masks)), key=lambda idx: (len(candidates[idx]), piece_masks[idx].size))
    assignment: dict[int, int] = {}
    used: set[int] = set()

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        piece_idx = order[pos]
        for cavity_idx in candidates[piece_idx]:
            if cavity_idx in used:
                continue
            used.add(cavity_idx)
            assignment[piece_idx] = cavity_idx
            if backtrack(pos + 1):
                return True
            used.remove(cavity_idx)
            del assignment[piece_idx]
        return False

    assert backtrack(0), "piece-to-cavity matching must be solvable"
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate scattered orange fragments into the central mold's edge notches and recolor them by the notch markers.

    Input structure:
    - Background is black.
    - A large central mold is a mostly solid grey rectangle whose edge contains several black notch cavities.
    - Each cavity has exactly one colored marker pixel on an adjacent grey mold cell near the cavity entrance.
    - Outside the mold, there are orange 4-connected fragments, one per cavity.

    Rule:
    - Find the mold's bounding rectangle and extract each black cavity that touches that rectangle's border.
    - For each orange fragment, try the four 90-degree rotations and find the unique cavity with the same shape.
    - Fill the matched cavity with that cavity's marker color.
    - Restore every marker pixel to grey and erase the original orange fragments back to black.

    Key insight:
    - The orange pieces only supply shape; the visible marker next to each cavity determines the final fill color after the rotated piece is inserted.
    """
    g = np.asarray(grid, dtype=int)
    mold_mask = (g != BLACK) & (g != ORANGE)
    assert mold_mask.any(), "expected a central mold"

    mold_coords = np.argwhere(mold_mask)
    r0, c0 = mold_coords.min(axis=0)
    r1, c1 = mold_coords.max(axis=0) + 1
    mold = g[r0:r1, c0:c1]
    mh, mw = mold.shape

    cavity_components = [
        comp
        for comp in _components(mold == BLACK)
        if any(rr in (0, mh - 1) or cc in (0, mw - 1) for rr, cc in comp)
    ]
    assert cavity_components, "expected edge cavities inside the mold"

    cavities: list[dict[str, object]] = []
    for comp in cavity_components:
        marker_positions: set[tuple[int, int]] = set()
        for rr, cc in comp:
            for dr, dc in DIRS4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < mh and 0 <= nc < mw:
                    value = int(mold[nr, nc])
                    if value not in (BLACK, GREY):
                        marker_positions.add((int(nr), int(nc)))
        assert len(marker_positions) == 1, "each cavity must have exactly one adjacent colored marker"
        marker_r, marker_c = next(iter(marker_positions))
        cavities.append(
            {
                "cells": comp,
                "mask": _mask_from_coords(comp),
                "marker_color": int(mold[marker_r, marker_c]),
            }
        )

    piece_components = _components(g == ORANGE)
    assert len(piece_components) == len(cavities), "piece count must equal cavity count"
    piece_masks = [_mask_from_coords(comp) for comp in piece_components]
    cavity_masks = [np.asarray(cavity["mask"], dtype=bool) for cavity in cavities]
    _match_piece_masks(piece_masks, cavity_masks)

    out = np.zeros_like(g)
    out[r0:r1, c0:c1] = GREY
    for cavity in cavities:
        color = int(cavity["marker_color"])
        for rr, cc in cavity["cells"]:
            out[r0 + rr, c0 + cc] = color
    return out


def _required_span(spans: list[int]) -> int:
    if not spans:
        return 8
    return sum(int(span) for span in spans) + 2 * (len(spans) + 1)


def _place_cavities(specs: list[dict[str, object]], mold_h: int, mold_w: int) -> list[dict[str, object]]:
    occupied = np.zeros((mold_h, mold_w), dtype=bool)
    placed: list[dict[str, object] | None] = [None] * len(specs)
    order = sorted(range(len(specs)), key=lambda idx: -int(np.sum(specs[idx]["mask"])))

    for idx in order:
        spec = specs[idx]
        mask = np.asarray(spec["mask"], dtype=bool)
        marker_rel = tuple(spec["marker_rel"])
        side = str(spec["side"])
        h, w = mask.shape

        if side == "top":
            anchors = [(0, c) for c in range(1, mold_w - w)]
        elif side == "bottom":
            anchors = [(mold_h - h, c) for c in range(1, mold_w - w)]
        elif side == "left":
            anchors = [(r, 0) for r in range(1, mold_h - h)]
        else:
            assert side == "right"
            anchors = [(r, mold_w - w) for r in range(1, mold_h - h)]

        random.shuffle(anchors)
        blocked = _dilate8(occupied)
        success = False
        for anchor_r, anchor_c in anchors:
            cells = [
                (int(anchor_r + rr), int(anchor_c + cc))
                for rr, cc in np.argwhere(mask)
            ]
            marker = (int(anchor_r + marker_rel[0]), int(anchor_c + marker_rel[1]))
            if any(blocked[rr, cc] for rr, cc in cells):
                continue
            if blocked[marker]:
                continue
            if marker[0] in (0, mold_h - 1) or marker[1] in (0, mold_w - 1):
                continue
            for rr, cc in cells:
                occupied[rr, cc] = True
            occupied[marker] = True
            placed[idx] = {
                "side": side,
                "mask": mask,
                "marker_color": int(spec["marker_color"]),
                "cells": cells,
                "marker": marker,
            }
            success = True
            break
        assert success, "failed to place all cavities"

    return [item for item in placed if item is not None]


def _validate_mold(mold: np.ndarray, expected_count: int) -> None:
    assert len(_components(mold != BLACK)) == 1, "mold must remain 4-connected"
    black_components = [
        comp
        for comp in _components(mold == BLACK)
        if any(
            r in (0, mold.shape[0] - 1) or c in (0, mold.shape[1] - 1)
            for r, c in comp
        )
    ]
    assert len(black_components) == expected_count, "black cavities must stay separated"
    for comp in black_components:
        markers = set()
        for r, c in comp:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < mold.shape[0] and 0 <= nc < mold.shape[1]:
                    val = int(mold[nr, nc])
                    if val not in (BLACK, GREY):
                        markers.add((int(nr), int(nc)))
        assert len(markers) == 1, "each cavity must keep one adjacent marker"


def _centered_offset(total_pad: int) -> int:
    center = total_pad // 2
    low = max(2, center - 2)
    high = min(total_pad - 2, center + 2)
    assert low <= high, "need at least a small border around the mold"
    return random.randint(low, high)


def _can_place_mask(blocked: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = mask.shape
    return not bool(np.any(mask & blocked[top : top + h, left : left + w]))


def _build_instance() -> np.ndarray:
    cavity_count = random.randint(2, 4)
    marker_colors = random.sample(MARKER_PALETTE, cavity_count)
    specs: list[dict[str, object]] = []
    used_keys: set[tuple[tuple[int, int], bytes]] = set()

    for marker_color in marker_colors:
        side = random.choice(SIDES)
        for _ in range(120):
            mask, marker_rel = _random_connected_shape(side)
            key = _canonical_key(mask)
            if key in used_keys:
                continue
            used_keys.add(key)
            specs.append(
                {
                    "side": side,
                    "mask": mask,
                    "marker_rel": marker_rel,
                    "marker_color": int(marker_color),
                    "piece_mask": np.rot90(mask, random.randrange(4)),
                }
            )
            break
        else:
            raise AssertionError("failed to collect unique cavity shapes")

    top_spans = [int(np.asarray(spec["mask"]).shape[1]) for spec in specs if spec["side"] == "top"]
    bottom_spans = [int(np.asarray(spec["mask"]).shape[1]) for spec in specs if spec["side"] == "bottom"]
    left_spans = [int(np.asarray(spec["mask"]).shape[0]) for spec in specs if spec["side"] == "left"]
    right_spans = [int(np.asarray(spec["mask"]).shape[0]) for spec in specs if spec["side"] == "right"]
    top_depth = max([int(np.asarray(spec["mask"]).shape[0]) for spec in specs if spec["side"] == "top"] + [0])
    bottom_depth = max([int(np.asarray(spec["mask"]).shape[0]) for spec in specs if spec["side"] == "bottom"] + [0])
    left_depth = max([int(np.asarray(spec["mask"]).shape[1]) for spec in specs if spec["side"] == "left"] + [0])
    right_depth = max([int(np.asarray(spec["mask"]).shape[1]) for spec in specs if spec["side"] == "right"] + [0])

    min_h = max(_required_span(left_spans), _required_span(right_spans), top_depth + bottom_depth + 5, 10)
    min_w = max(_required_span(top_spans), _required_span(bottom_spans), left_depth + right_depth + 5, 10)
    assert min_h <= 18 and min_w <= 18, "mold became too large"

    mold_h = random.randint(min_h, min(18, min_h + 4))
    mold_w = random.randint(min_w, min(18, min_w + 4))
    placed = _place_cavities(specs, mold_h, mold_w)

    mold = np.full((mold_h, mold_w), GREY, dtype=int)
    for item in placed:
        for r, c in item["cells"]:
            mold[r, c] = BLACK
        marker_r, marker_c = item["marker"]
        mold[marker_r, marker_c] = int(item["marker_color"])
    _validate_mold(mold, cavity_count)

    extra_h = random.randint(6, min(12, 30 - mold_h))
    extra_w = random.randint(6, min(12, 30 - mold_w))
    grid_h = mold_h + extra_h
    grid_w = mold_w + extra_w
    top = _centered_offset(grid_h - mold_h)
    left = _centered_offset(grid_w - mold_w)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[top : top + mold_h, left : left + mold_w] = mold

    expected = np.zeros_like(grid)
    expected[top : top + mold_h, left : left + mold_w] = GREY
    for item in placed:
        color = int(item["marker_color"])
        for r, c in item["cells"]:
            expected[top + r, left + c] = color

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    occupied[top : top + mold_h, left : left + mold_w] = True

    piece_order = sorted(range(cavity_count), key=lambda idx: -int(np.sum(np.asarray(specs[idx]["piece_mask"], dtype=bool))))
    for idx in piece_order:
        piece_mask = np.asarray(specs[idx]["piece_mask"], dtype=bool)
        ph, pw = piece_mask.shape
        blocked = _dilate8(occupied)
        candidates = [
            (r, c)
            for r in range(grid_h - ph + 1)
            for c in range(grid_w - pw + 1)
            if _can_place_mask(blocked, piece_mask, r, c)
        ]
        assert candidates, "failed to place all orange pieces"
        place_r, place_c = random.choice(candidates)
        for rr, cc in np.argwhere(piece_mask):
            grid[place_r + rr, place_c + cc] = ORANGE
            occupied[place_r + rr, place_c + cc] = True

    produced = main(grid)
    assert np.array_equal(produced, expected), "generator must agree with solver"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            return _build_instance()
        except AssertionError:
            continue
    raise AssertionError("failed to generate a valid puzzle instance")
