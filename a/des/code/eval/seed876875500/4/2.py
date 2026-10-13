from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
RED = 2
GRAY = 5
ORANGE = 7
PATTERN_COLORS = (1, 3, 4, 6, 8, 9)

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
ARROW_HEADS = {
    (-1, 0): [(-1, -1), (-1, 1)],
    (1, 0): [(1, -1), (1, 1)],
    (0, -1): [(-1, -1), (1, -1)],
    (0, 1): [(-1, 1), (1, 1)],
}

MIN_GROUPS = 1
MAX_GROUPS = 3
MAX_ATTEMPTS = 400
WORK_SIZE = 34
PLACEMENT_GAP = 2
OUTER_MARGIN_MIN = 1
OUTER_MARGIN_MAX = 3


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIRS4 if connectivity == 4 else DIRS8
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if not _in_bounds(h, w, nr, nc):
                        continue
                    if not bool(mask[nr, nc]) or bool(seen[nr, nc]):
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    if isinstance(points, np.ndarray):
        arr = points
    else:
        arr = np.array(points, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    assert len(pts) > 0, "expected non-empty mask"
    return _bbox(pts)


def _rects_too_close(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _rotate_clockwise(arr: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(arr, -int(quarter_turns) % 4)


def _rotate_vec_ccw(dr: int, dc: int, quarter_turns: int) -> tuple[int, int]:
    for _ in range(int(quarter_turns) % 4):
        dr, dc = -dc, dr
    return dr, dc


def _find_frames(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    frames: list[dict] = []
    border_owner: dict[tuple[int, int], int] = {}

    for comp in _components(grid == GRAY, connectivity=4):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        if len(comp) != 2 * h + 2 * w - 4:
            continue
        if any(r not in (r0, r1) and c not in (c0, c1) for r, c in comp):
            continue
        if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != BLACK):
            continue

        idx = len(frames)
        inner = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
        frames.append({"outer": (r0, c0, r1, c1), "inner": inner})
        for cell in comp:
            assert cell not in border_owner, "frame borders must be disjoint"
            border_owner[cell] = idx

    return frames, border_owner


def _parse_arrow(comp: list[tuple[int, int]]) -> dict:
    cells = set(comp)
    matches: list[dict] = []
    for br, bc in comp:
        for direction, offsets in ARROW_HEADS.items():
            expected = {(br, bc)}
            for dr, dc in offsets:
                expected.add((br + dr, bc + dc))
            if expected == cells:
                matches.append({"body": (br, bc), "dir": direction})
    assert len(matches) == 1, f"ambiguous red arrow: {sorted(comp)}"
    return matches[0]


def _find_arrows(grid: np.ndarray) -> list[dict]:
    arrows = []
    for comp in _components(grid == RED, connectivity=8):
        assert len(comp) == 3, "every red component must be a 3-pixel arrow"
        arrows.append(_parse_arrow(comp))
    return arrows


def _find_source_components(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    special = np.isin(grid, [BLACK, RED, GRAY, ORANGE])
    mask = ~special
    owner = np.full(grid.shape, -1, dtype=int)
    comps: list[dict] = []

    for comp in _components(mask, connectivity=4):
        idx = len(comps)
        for r, c in comp:
            owner[r, c] = idx
        r0, c0, r1, c1 = _bbox(comp)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        comps.append(
            {
                "cells": set(comp),
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "color": int(grid[comp[0][0], comp[0][1]]),
            }
        )

    return comps, owner


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow each red arrow to the first gray frame, count orange markers on the ray,
    rotate the adjacent source pattern clockwise by that count, and paint it into
    the frame interior.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    frames, frame_owner = _find_frames(g)
    assert frames, "expected at least one gray target frame"

    arrows = _find_arrows(g)
    assert arrows, "expected at least one red arrow"

    sources, source_owner = _find_source_components(g)
    assert sources, "expected at least one source pattern"

    for arrow in arrows:
        br, bc = arrow["body"]
        dr, dc = arrow["dir"]

        adjacent_sources = set()
        for sdr, sdc in DIRS4:
            nr, nc = br + sdr, bc + sdc
            if _in_bounds(h, w, nr, nc) and int(source_owner[nr, nc]) >= 0:
                adjacent_sources.add(int(source_owner[nr, nc]))
        assert len(adjacent_sources) == 1, "arrow body must touch exactly one source pattern"
        source = sources[adjacent_sources.pop()]

        behind = (br - dr, bc - dc)
        assert behind in source["cells"], "source must lie immediately behind the arrow body"

        orange_count = 0
        target_frame = None
        rr, cc = br + dr, bc + dc
        while _in_bounds(h, w, rr, cc):
            val = int(g[rr, cc])
            if val == BLACK:
                pass
            elif val == ORANGE:
                orange_count += 1
            elif val == GRAY:
                assert (rr, cc) in frame_owner, "encountered gray pixel that is not a frame border"
                target_frame = frames[frame_owner[(rr, cc)]]
                break
            else:
                raise AssertionError(f"unexpected blocker {val} on arrow path")
            rr += dr
            cc += dc
        assert target_frame is not None, "every arrow path must hit a frame"

        sr0, sc0, sr1, sc1 = source["bbox"]
        source_patch = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
        rotated = _rotate_clockwise(source_patch, orange_count % 4)

        tr0, tc0, tr1, tc1 = target_frame["inner"]
        th = tr1 - tr0 + 1
        tw = tc1 - tc0 + 1
        ph, pw = rotated.shape
        assert ph <= th and pw <= tw, "frame interior must fit the rotated pattern"
        assert (th - ph) % 2 == 0 and (tw - pw) % 2 == 0, "centering offset must be integral"
        roff = (th - ph) // 2
        coff = (tw - pw) // 2

        dest = out[tr0 + roff : tr0 + roff + ph, tc0 + coff : tc0 + coff + pw]
        if not (np.all(dest == BLACK) or np.array_equal(dest, rotated)):
            raise AssertionError("multiple arrows cannot paint conflicting payloads into the same frame")
        out[tr0 + roff : tr0 + roff + ph, tc0 + coff : tc0 + coff + pw] = rotated

    return out


def _random_manhattan_path(
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = end
    out = [(r, c)]
    while (r, c) != (tr, tc):
        moves = []
        if r < tr:
            moves.append((1, 0))
        elif r > tr:
            moves.append((-1, 0))
        if c < tc:
            moves.append((0, 1))
        elif c > tc:
            moves.append((0, -1))
        dr, dc = random.choice(moves)
        r += dr
        c += dc
        out.append((r, c))
    return out


def _random_shape(size: int, anchor_row: int) -> np.ndarray:
    assert 0 <= anchor_row < size
    target_min = size + 2
    center = (size // 2, size // 2)

    for _ in range(200):
        required = {
            center,
            (anchor_row, size - 1),
            (random.randrange(size), 0),
            (0, random.randrange(size)),
            (size - 1, random.randrange(size)),
        }
        cells = {center}
        for pt in required:
            cells.update(_random_manhattan_path(center, pt))

        target_area = random.randint(max(target_min, len(cells)), size * size - 1)
        while len(cells) < target_area:
            frontier = []
            for r, c in sorted(cells):
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        mask = np.zeros((size, size), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if not mask[:, 0].any() or not mask[:, -1].any() or not mask[0, :].any() or not mask[-1, :].any():
            continue
        if not bool(mask[anchor_row, size - 1]):
            continue
        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        return mask

    raise RuntimeError("failed to generate a non-symmetric source shape")


def _spaced_positions(path_len: int, count: int) -> list[int]:
    if count == 0:
        return []
    candidates = [
        comb
        for comb in itertools.combinations(range(path_len), count)
        if all(b - a > 1 for a, b in zip(comb, comb[1:]))
    ]
    assert candidates, "path must be long enough for isolated orange markers"
    return list(random.choice(candidates))


def _draw_frame(patch: np.ndarray, r0: int, c0: int, size: int) -> None:
    r1 = r0 + size + 1
    c1 = c0 + size + 1
    patch[r0, c0 : c1 + 1] = GRAY
    patch[r1, c0 : c1 + 1] = GRAY
    patch[r0 : r1 + 1, c0] = GRAY
    patch[r0 : r1 + 1, c1] = GRAY


def _make_group(color: int) -> dict:
    for _ in range(200):
        size = random.randint(3, 5)
        anchor_row = random.randint(1, size - 2)
        orange_count = random.randint(0, 3)
        path_len = max(1, 2 * orange_count - 1) + random.randint(0, 3)
        orange_positions = _spaced_positions(path_len, orange_count)
        shape = _random_shape(size, anchor_row)

        source_patch = np.full((size, size), BLACK, dtype=int)
        source_patch[shape] = color

        frame_left = size + path_len + 1
        height = size + 2
        width = 2 * size + path_len + 3
        patch = np.full((height, width), BLACK, dtype=int)
        patch[1 : 1 + size, :size] = source_patch

        body_r = 1 + anchor_row
        body_c = size
        patch[body_r, body_c] = RED
        patch[body_r - 1, body_c + 1] = RED
        patch[body_r + 1, body_c + 1] = RED
        for pos in orange_positions:
            patch[body_r, size + 1 + pos] = ORANGE
        _draw_frame(patch, 0, frame_left, size)

        source_rect = np.zeros_like(patch, dtype=bool)
        source_rect[1 : 1 + size, :size] = True
        frame_inner = np.zeros_like(patch, dtype=bool)
        frame_inner[1 : 1 + size, frame_left + 1 : frame_left + 1 + size] = True

        rot_k = random.randrange(4)
        patch = np.rot90(patch, rot_k)
        source_rect = np.rot90(source_rect, rot_k)
        frame_inner = np.rot90(frame_inner, rot_k)

        sr0, sc0, sr1, sc1 = _bbox_from_mask(source_rect)
        fr0, fc0, fr1, fc1 = _bbox_from_mask(frame_inner)
        final_source_patch = patch[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
        direction = _rotate_vec_ccw(0, 1, rot_k)

        return {
            "patch": patch,
            "source_patch": final_source_patch,
            "source_bbox": (sr0, sc0, sr1, sc1),
            "frame_inner_bbox": (fr0, fc0, fr1, fc1),
            "orange_count": orange_count,
            "direction": direction,
        }

    raise RuntimeError("failed to build a valid group")


def _place_groups(groups: list[dict]) -> tuple[np.ndarray, list[dict]]:
    work = np.full((WORK_SIZE, WORK_SIZE), BLACK, dtype=int)
    placed_rects: list[tuple[int, int, int, int]] = []
    placed_groups: list[dict] = []

    order = list(range(len(groups)))
    order.sort(key=lambda i: groups[i]["patch"].shape[0] * groups[i]["patch"].shape[1], reverse=True)

    for idx in order:
        group = groups[idx]
        patch = group["patch"]
        ph, pw = patch.shape
        positions = [
            (r0, c0)
            for r0 in range(1, WORK_SIZE - ph)
            for c0 in range(1, WORK_SIZE - pw)
        ]
        random.shuffle(positions)

        chosen = None
        for r0, c0 in positions:
            rect = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if any(_rects_too_close(rect, other, PLACEMENT_GAP) for other in placed_rects):
                continue
            chosen = (r0, c0, rect)
            break
        assert chosen is not None, "failed to place a group without interference"

        r0, c0, rect = chosen
        non_bg = patch != BLACK
        work[r0 : r0 + ph, c0 : c0 + pw][non_bg] = patch[non_bg]
        placed_rects.append(rect)

        sr0, sc0, sr1, sc1 = group["source_bbox"]
        fr0, fc0, fr1, fc1 = group["frame_inner_bbox"]
        placed_groups.append(
            {
                "source_patch": group["source_patch"],
                "orange_count": group["orange_count"],
                "source_bbox": (r0 + sr0, c0 + sc0, r0 + sr1, c0 + sc1),
                "frame_inner_bbox": (r0 + fr0, c0 + fc0, r0 + fr1, c0 + fc1),
                "direction": group["direction"],
            }
        )

    occupied = np.argwhere(work != BLACK)
    assert len(occupied) > 0, "placement produced an empty canvas"
    r0, c0, r1, c1 = _bbox(occupied)
    pad_top = random.randint(OUTER_MARGIN_MIN, OUTER_MARGIN_MAX)
    pad_bottom = random.randint(OUTER_MARGIN_MIN, OUTER_MARGIN_MAX)
    pad_left = random.randint(OUTER_MARGIN_MIN, OUTER_MARGIN_MAX)
    pad_right = random.randint(OUTER_MARGIN_MIN, OUTER_MARGIN_MAX)
    r0 = max(0, r0 - pad_top)
    c0 = max(0, c0 - pad_left)
    r1 = min(WORK_SIZE - 1, r1 + pad_bottom)
    c1 = min(WORK_SIZE - 1, c1 + pad_right)

    cropped = work[r0 : r1 + 1, c0 : c1 + 1].copy()
    adjusted = []
    for group in placed_groups:
        sr0, sc0, sr1, sc1 = group["source_bbox"]
        fr0, fc0, fr1, fc1 = group["frame_inner_bbox"]
        adjusted.append(
            {
                "source_patch": group["source_patch"],
                "orange_count": group["orange_count"],
                "source_bbox": (sr0 - r0, sc0 - c0, sr1 - r0, sc1 - c0),
                "frame_inner_bbox": (fr0 - r0, fc0 - c0, fr1 - r0, fc1 - c0),
                "direction": group["direction"],
            }
        )

    return cropped, adjusted


def _expected_from_metadata(grid: np.ndarray, placed_groups: list[dict]) -> np.ndarray:
    out = grid.copy()
    for group in placed_groups:
        fr0, fc0, fr1, fc1 = group["frame_inner_bbox"]
        rotated = _rotate_clockwise(group["source_patch"], group["orange_count"] % 4)
        assert rotated.shape == (fr1 - fr0 + 1, fc1 - fc0 + 1), "frame interior must match source patch size"
        out[fr0 : fr1 + 1, fc0 : fc1 + 1] = rotated
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            group_count = random.randint(MIN_GROUPS, MAX_GROUPS)
            colors = random.sample(list(PATTERN_COLORS), group_count)
            groups = [_make_group(color) for color in colors]
            grid, metadata = _place_groups(groups)
            expected = _expected_from_metadata(grid, metadata)
            actual = main(grid)
            if np.array_equal(actual, expected):
                return grid
        except (AssertionError, RuntimeError):
            continue
    raise RuntimeError("failed to generate a valid ARC input")
