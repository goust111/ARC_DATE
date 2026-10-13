from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import itertools
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, paste


BG = Color.BLACK
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SIDES = ("up", "down", "left", "right")
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


@dataclass(frozen=True)
class Piece:
    color: int
    crop: np.ndarray
    body_bbox: tuple[int, int, int, int]
    tenon_side: str | None
    tenon_profile: tuple[int, ...]
    mortise_side: str | None
    mortise_profile: tuple[int, ...]

    @property
    def body_h(self) -> int:
        r0, _c0, r1, _c1 = self.body_bbox
        return int(r1 - r0 + 1)

    @property
    def body_w(self) -> int:
        _r0, c0, _r1, c1 = self.body_bbox
        return int(c1 - c0 + 1)

    @property
    def norm_tenon(self) -> tuple[int, ...]:
        return _normalize_profile(self.tenon_profile)

    @property
    def norm_mortise(self) -> tuple[int, ...]:
        return _normalize_profile(self.mortise_profile)


def _normalize_profile(profile: tuple[int, ...]) -> tuple[int, ...]:
    if not profile:
        return ()
    ordered = tuple(sorted(int(v) for v in profile))
    base = ordered[0]
    return tuple(int(v - base) for v in ordered)


def _mirror_profile(profile: tuple[int, ...]) -> tuple[int, ...]:
    span = int(max(profile))
    return _normalize_profile(tuple(span - v for v in profile))


def _profile_fit(normalized: tuple[int, ...], side_len: int) -> bool:
    return bool(normalized) and len(normalized) < side_len and max(normalized) < side_len


def _choose_actual_profile(normalized: tuple[int, ...], side_len: int) -> tuple[int, ...]:
    max_start = side_len - 1 - max(normalized)
    start = random.randint(0, max_start)
    return tuple(int(start + v) for v in normalized)


def _extract_outside(outside_mask: np.ndarray, full_mask: np.ndarray, body_bbox: tuple[int, int, int, int]) -> tuple[str | None, tuple[int, ...]] | None:
    r0, c0, r1, c1 = body_bbox
    coords = np.argwhere(outside_mask)
    if coords.size == 0:
        return None, ()
    valid: list[tuple[str, tuple[int, ...]]] = []

    if r0 > 0 and np.all(coords[:, 0] == r0 - 1) and np.all((c0 <= coords[:, 1]) & (coords[:, 1] <= c1)):
        profile = tuple(sorted(int(c - c0) for _r, c in coords))
        if len(profile) < (c1 - c0 + 1) and all(full_mask[r0, c] for _r, c in coords):
            valid.append(("up", profile))
    if r1 + 1 < full_mask.shape[0] and np.all(coords[:, 0] == r1 + 1) and np.all((c0 <= coords[:, 1]) & (coords[:, 1] <= c1)):
        profile = tuple(sorted(int(c - c0) for _r, c in coords))
        if len(profile) < (c1 - c0 + 1) and all(full_mask[r1, c] for _r, c in coords):
            valid.append(("down", profile))
    if c0 > 0 and np.all(coords[:, 1] == c0 - 1) and np.all((r0 <= coords[:, 0]) & (coords[:, 0] <= r1)):
        profile = tuple(sorted(int(r - r0) for r, _c in coords))
        if len(profile) < (r1 - r0 + 1) and all(full_mask[r, c0] for r, _c in coords):
            valid.append(("left", profile))
    if c1 + 1 < full_mask.shape[1] and np.all(coords[:, 1] == c1 + 1) and np.all((r0 <= coords[:, 0]) & (coords[:, 0] <= r1)):
        profile = tuple(sorted(int(r - r0) for r, _c in coords))
        if len(profile) < (r1 - r0 + 1) and all(full_mask[r, c1] for r, _c in coords):
            valid.append(("right", profile))

    if len(valid) != 1:
        return None
    return valid[0]


def _extract_holes(body_mask: np.ndarray) -> tuple[str | None, tuple[int, ...]] | None:
    coords = np.argwhere(~body_mask)
    if coords.size == 0:
        return None, ()
    h, w = body_mask.shape
    valid: list[tuple[str, tuple[int, ...]]] = []

    if np.all(coords[:, 0] == 0):
        profile = tuple(sorted(int(c) for _r, c in coords))
        if len(profile) < w:
            valid.append(("up", profile))
    if np.all(coords[:, 0] == h - 1):
        profile = tuple(sorted(int(c) for _r, c in coords))
        if len(profile) < w:
            valid.append(("down", profile))
    if np.all(coords[:, 1] == 0):
        profile = tuple(sorted(int(r) for r, _c in coords))
        if len(profile) < h:
            valid.append(("left", profile))
    if np.all(coords[:, 1] == w - 1):
        profile = tuple(sorted(int(r) for r, _c in coords))
        if len(profile) < h:
            valid.append(("right", profile))

    if len(valid) != 1:
        return None
    return valid[0]


def _piece_candidates(crop: np.ndarray) -> list[Piece]:
    colors = set(map(int, np.unique(crop))) - {BG}
    assert len(colors) == 1, "each sprite must be monochromatic"
    color = colors.pop()
    mask = crop != BG
    h, w = crop.shape
    candidates: list[Piece] = []
    seen = set()

    for trim_t, trim_b, trim_l, trim_r in itertools.product((0, 1), repeat=4):
        r0, r1 = trim_t, h - 1 - trim_b
        c0, c1 = trim_l, w - 1 - trim_r
        if r0 > r1 or c0 > c1:
            continue
        body_h = r1 - r0 + 1
        body_w = c1 - c0 + 1
        if min(body_h, body_w) < 3:
            continue

        body_mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        outside_mask = mask.copy()
        outside_mask[r0 : r1 + 1, c0 : c1 + 1] = False
        tenon = _extract_outside(outside_mask, mask, (r0, c0, r1, c1))
        if tenon is None:
            continue
        mortise = _extract_holes(body_mask)
        if mortise is None:
            continue

        tenon_side, tenon_profile = tenon
        mortise_side, mortise_profile = mortise

        if tenon_side is not None and mortise_side is not None and tenon_side == mortise_side:
            continue

        body_filled = body_mask.copy()
        if mortise_side is not None:
            if mortise_side == "up":
                body_filled[0, list(mortise_profile)] = True
            elif mortise_side == "down":
                body_filled[-1, list(mortise_profile)] = True
            elif mortise_side == "left":
                body_filled[list(mortise_profile), 0] = True
            else:
                body_filled[list(mortise_profile), -1] = True
        if not body_filled.all():
            continue

        piece = Piece(
            color=color,
            crop=np.asarray(crop, dtype=int).copy(),
            body_bbox=(int(r0), int(c0), int(r1), int(c1)),
            tenon_side=tenon_side,
            tenon_profile=tenon_profile,
            mortise_side=mortise_side,
            mortise_profile=mortise_profile,
        )
        key = (piece.body_bbox, piece.tenon_side, piece.tenon_profile, piece.mortise_side, piece.mortise_profile)
        if key not in seen:
            seen.add(key)
            candidates.append(piece)

    assert candidates, "failed to identify any valid body rectangle"
    return candidates


def _candidate_lists_from_grid(grid: np.ndarray) -> list[list[Piece]]:
    candidate_lists = []
    for comp in find_components(np.asarray(grid, dtype=int), bg=BG, connectivity=4, monochromatic=False):
        crop = crop_bbox(grid, comp["bbox"])
        candidate_lists.append(_piece_candidates(crop))
    assert len(candidate_lists) >= 3, "expected a chain of at least three sprites"
    return candidate_lists


def _match(curr: Piece, nxt: Piece) -> bool:
    if curr.tenon_side is None or nxt.mortise_side is None:
        return False
    return nxt.mortise_side == OPPOSITE[curr.tenon_side] and curr.norm_tenon == nxt.norm_mortise


def _next_body_pos(curr: Piece, curr_pos: tuple[int, int], nxt: Piece) -> tuple[int, int]:
    assert curr.tenon_side is not None and nxt.mortise_side is not None
    shift = int(min(curr.tenon_profile) - min(nxt.mortise_profile))
    r, c = curr_pos
    if curr.tenon_side == "right":
        return int(r + shift), int(c + curr.body_w)
    if curr.tenon_side == "left":
        return int(r + shift), int(c - nxt.body_w)
    if curr.tenon_side == "down":
        return int(r + curr.body_h), int(c + shift)
    return int(r - nxt.body_h), int(c + shift)


def _piece_cells(piece: Piece, body_pos: tuple[int, int]) -> set[tuple[int, int]]:
    body_r, body_c = body_pos
    r0, c0, _r1, _c1 = piece.body_bbox
    crop_r = body_r - r0
    crop_c = body_c - c0
    cells = set()
    for rr, cc in np.argwhere(piece.crop != BG):
        cells.add((int(crop_r + rr), int(crop_c + cc)))
    return cells


def _render_assembly(pieces: list[Piece], positions: dict[int, tuple[int, int]]) -> np.ndarray:
    placed_boxes = []
    for idx, piece in enumerate(pieces):
        body_r, body_c = positions[idx]
        r0, c0, _r1, _c1 = piece.body_bbox
        top = body_r - r0
        left = body_c - c0
        placed_boxes.append((top, left, top + piece.crop.shape[0] - 1, left + piece.crop.shape[1] - 1))

    min_r = min(box[0] for box in placed_boxes)
    min_c = min(box[1] for box in placed_boxes)
    max_r = max(box[2] for box in placed_boxes)
    max_c = max(box[3] for box in placed_boxes)
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), BG, dtype=int)

    for idx, piece in enumerate(pieces):
        body_r, body_c = positions[idx]
        r0, c0, _r1, _c1 = piece.body_bbox
        top = body_r - r0 - min_r
        left = body_c - c0 - min_c
        paste(out, piece.crop, int(top), int(left), transparent=BG)
    return out


def _enumerate_valid_assemblies(pieces: list[Piece]) -> list[tuple[list[int], dict[int, tuple[int, int]]]]:
    heads = [i for i, piece in enumerate(pieces) if piece.mortise_side is None]
    tails = [i for i, piece in enumerate(pieces) if piece.tenon_side is None]
    if len(heads) != 1 or len(tails) != 1:
        return []

    head = heads[0]
    results: list[tuple[list[int], dict[int, tuple[int, int]]]] = []

    def dfs(order: list[int], positions: dict[int, tuple[int, int]], occupied: set[tuple[int, int]]) -> None:
        curr_idx = order[-1]
        curr_piece = pieces[curr_idx]
        if curr_piece.tenon_side is None:
            if len(order) == len(pieces):
                results.append((order.copy(), positions.copy()))
            return

        candidates = [idx for idx in range(len(pieces)) if idx not in positions and _match(curr_piece, pieces[idx])]
        for nxt_idx in candidates:
            nxt_piece = pieces[nxt_idx]
            nxt_pos = _next_body_pos(curr_piece, positions[curr_idx], nxt_piece)
            nxt_cells = _piece_cells(nxt_piece, nxt_pos)
            if occupied & nxt_cells:
                continue
            positions[nxt_idx] = nxt_pos
            order.append(nxt_idx)
            dfs(order, positions, occupied | nxt_cells)
            order.pop()
            del positions[nxt_idx]

    start_pos = {head: (0, 0)}
    dfs([head], start_pos, _piece_cells(pieces[head], (0, 0)))
    return results


def _enumerate_candidate_assemblies(candidate_lists: list[list[Piece]]) -> list[tuple[list[Piece], list[int], dict[int, tuple[int, int]]]]:
    results: list[tuple[list[Piece], list[int], dict[int, tuple[int, int]]]] = []
    for combo in itertools.product(*candidate_lists):
        pieces = list(combo)
        assemblies = _enumerate_valid_assemblies(pieces)
        for order, positions in assemblies:
            results.append((pieces, order, positions))
    return results


def main(grid: np.ndarray) -> np.ndarray:
    """Assemble the scattered sprites into the unique tenon-mortise fold chain and crop the finished chain.

    Input structure:
    - Black background with several disconnected monochrome sprites scattered at arbitrary positions.
    - Each sprite has a solid rectangular body, plus at most one outward one-cell-deep tenon strip and at most one inward one-cell-deep mortise strip on a body edge.
    - Exactly one sprite has no mortise and is the chain head; exactly one sprite has no tenon and is the chain tail.
    - Colors and input positions do not encode the order, and some mortises may be mirror-shaped distractors for another tenon.

    Rule:
    - Recover each sprite's body rectangle, the edge carrying its tenon, and the edge carrying its mortise.
    - Start from the sprite without a mortise. Repeatedly find the unique unused sprite whose mortise lies on the opposite side and whose missing strip matches the current tenon by translation along that edge.
    - Place that next sprite on the side pointed to by the tenon so the tenon cells fill the mortise cells; keep every other colored cell unchanged.
    - Continue until the tail is placed, then return the minimal bounding box of the assembled chain.

    Key insight:
    - Neither order nor offsets come from position, size, or color; the only reliable signal is exact tenon-versus-mortise complement on the specified side, with mirrors rejected because they are not translationally compatible.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    candidate_lists = _candidate_lists_from_grid(g)
    assemblies = _enumerate_candidate_assemblies(candidate_lists)
    assert len(assemblies) == 1, "the chain must be uniquely determined"
    pieces, _order, positions = assemblies[0]
    out = _render_assembly(pieces, positions)
    assert out.ndim == 2 and max(out.shape) <= 30
    return out


PROFILE_CATALOG = tuple(
    pattern
    for span in range(3, 5)
    for size in range(2, min(4, span + 1) + 1)
    for pattern in itertools.combinations(range(span + 1), size)
    if pattern[0] == 0 and pattern[-1] == span and _mirror_profile(pattern) != pattern
)


def _random_directions(link_count: int) -> list[str]:
    for _ in range(200):
        dirs = [random.choice(SIDES)]
        for _idx in range(1, link_count):
            choices = [side for side in SIDES if side != OPPOSITE[dirs[-1]]]
            dirs.append(random.choice(choices))
        if len(set(dirs)) == 1:
            continue
        repeated = any(dirs[i] == dirs[j] for i in range(link_count) for j in range(i + 1, link_count))
        if repeated:
            return dirs
    raise RuntimeError("failed to sample a useful direction chain")


def _side_len(h: int, w: int, side: str) -> int:
    return h if side in ("left", "right") else w


def _choose_link_profiles(body_shapes: list[tuple[int, int]], directions: list[str]) -> list[tuple[int, ...]] | None:
    link_count = len(directions)
    same_pairs = [(i, j) for i in range(link_count) for j in range(i + 1, link_count) if directions[i] == directions[j]]
    random.shuffle(same_pairs)
    for mirror_a, mirror_b in same_pairs:
        used: set[tuple[int, ...]] = set()
        profiles: list[tuple[int, ...] | None] = [None] * link_count

        len_a0 = _side_len(*body_shapes[mirror_a], directions[mirror_a])
        len_a1 = _side_len(*body_shapes[mirror_a + 1], OPPOSITE[directions[mirror_a]])
        len_b0 = _side_len(*body_shapes[mirror_b], directions[mirror_b])
        len_b1 = _side_len(*body_shapes[mirror_b + 1], OPPOSITE[directions[mirror_b]])
        seeds = [p for p in PROFILE_CATALOG if all(_profile_fit(p, L) for L in (len_a0, len_a1)) and all(_profile_fit(_mirror_profile(p), L) for L in (len_b0, len_b1))]
        random.shuffle(seeds)
        for seed in seeds:
            profiles[mirror_a] = seed
            profiles[mirror_b] = _mirror_profile(seed)
            used.update((seed, _mirror_profile(seed)))

            ok = True
            for idx in range(link_count):
                if profiles[idx] is not None:
                    continue
                side = directions[idx]
                curr_len = _side_len(*body_shapes[idx], side)
                next_len = _side_len(*body_shapes[idx + 1], OPPOSITE[side])
                choices = [p for p in PROFILE_CATALOG if p not in used and _profile_fit(p, curr_len) and _profile_fit(p, next_len)]
                if not choices:
                    ok = False
                    break
                pick = random.choice(choices)
                profiles[idx] = pick
                used.add(pick)
            if ok:
                return [tuple(p) for p in profiles]
    return None


def _build_piece(
    color: int,
    body_h: int,
    body_w: int,
    mortise_side: str | None,
    mortise_profile: tuple[int, ...],
    tenon_side: str | None,
    tenon_profile: tuple[int, ...],
) -> np.ndarray:
    top_pad = 1 if tenon_side == "up" else 0
    left_pad = 1 if tenon_side == "left" else 0
    bottom_pad = 1 if tenon_side == "down" else 0
    right_pad = 1 if tenon_side == "right" else 0
    grid = np.full((body_h + top_pad + bottom_pad, body_w + left_pad + right_pad), BG, dtype=int)

    body_r0 = top_pad
    body_c0 = left_pad
    grid[body_r0 : body_r0 + body_h, body_c0 : body_c0 + body_w] = int(color)

    if mortise_side == "up":
        grid[body_r0, body_c0 + np.array(mortise_profile, dtype=int)] = BG
    elif mortise_side == "down":
        grid[body_r0 + body_h - 1, body_c0 + np.array(mortise_profile, dtype=int)] = BG
    elif mortise_side == "left":
        grid[body_r0 + np.array(mortise_profile, dtype=int), body_c0] = BG
    elif mortise_side == "right":
        grid[body_r0 + np.array(mortise_profile, dtype=int), body_c0 + body_w - 1] = BG

    if tenon_side == "up":
        grid[body_r0 - 1, body_c0 + np.array(tenon_profile, dtype=int)] = int(color)
    elif tenon_side == "down":
        grid[body_r0 + body_h, body_c0 + np.array(tenon_profile, dtype=int)] = int(color)
    elif tenon_side == "left":
        grid[body_r0 + np.array(tenon_profile, dtype=int), body_c0 - 1] = int(color)
    elif tenon_side == "right":
        grid[body_r0 + np.array(tenon_profile, dtype=int), body_c0 + body_w] = int(color)

    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=False)
    if len(comps) != 1:
        raise ValueError("generated piece is disconnected")
    return grid


def _piece_from_spec(
    crop: np.ndarray,
    color: int,
    body_h: int,
    body_w: int,
    mortise_side: str | None,
    mortise_profile: tuple[int, ...],
    tenon_side: str | None,
    tenon_profile: tuple[int, ...],
) -> Piece:
    r0 = 1 if tenon_side == "up" else 0
    c0 = 1 if tenon_side == "left" else 0
    return Piece(
        color=int(color),
        crop=np.asarray(crop, dtype=int).copy(),
        body_bbox=(int(r0), int(c0), int(r0 + body_h - 1), int(c0 + body_w - 1)),
        tenon_side=tenon_side,
        tenon_profile=tuple(int(v) for v in tenon_profile),
        mortise_side=mortise_side,
        mortise_profile=tuple(int(v) for v in mortise_profile),
    )


def _sample_piece_specs() -> tuple[list[Piece], np.ndarray] | None:
    n = random.choice((4, 5))
    body_shapes = [(random.randint(4, 6), random.randint(4, 6)) for _ in range(n)]
    directions = _random_directions(n - 1)
    profiles = _choose_link_profiles(body_shapes, directions)
    if profiles is None:
        return None

    colors = [random.choice(COLOR_POOL) for _ in range(n)]
    pieces: list[Piece] = []
    for idx in range(n):
        body_h, body_w = body_shapes[idx]
        mortise_side = None if idx == 0 else OPPOSITE[directions[idx - 1]]
        tenon_side = None if idx == n - 1 else directions[idx]
        mortise_profile = ()
        tenon_profile = ()
        if mortise_side is not None:
            mortise_profile = _choose_actual_profile(profiles[idx - 1], _side_len(body_h, body_w, mortise_side))
        if tenon_side is not None:
            tenon_profile = _choose_actual_profile(profiles[idx], _side_len(body_h, body_w, tenon_side))
        try:
            crop = _build_piece(colors[idx], body_h, body_w, mortise_side, mortise_profile, tenon_side, tenon_profile)
            piece = _piece_from_spec(crop, colors[idx], body_h, body_w, mortise_side, mortise_profile, tenon_side, tenon_profile)
            parsed = _piece_candidates(crop)
            if not any(
                cand.body_bbox == piece.body_bbox
                and cand.tenon_side == piece.tenon_side
                and cand.tenon_profile == piece.tenon_profile
                and cand.mortise_side == piece.mortise_side
                and cand.mortise_profile == piece.mortise_profile
                for cand in parsed
            ):
                return None
            pieces.append(piece)
        except (AssertionError, ValueError):
            return None

    assemblies = _enumerate_valid_assemblies(pieces)
    if len(assemblies) != 1:
        return None
    _order, positions = assemblies[0]
    out = _render_assembly(pieces, positions)
    if max(out.shape) > 30:
        return None
    return pieces, out


def _scatter_input(pieces: list[Piece]) -> np.ndarray | None:
    max_h = max(piece.crop.shape[0] for piece in pieces)
    max_w = max(piece.crop.shape[1] for piece in pieces)
    H = random.randint(max(18, max_h + 6), 30)
    W = random.randint(max(18, max_w + 6), 30)
    grid = np.full((H, W), BG, dtype=int)
    blocked = np.zeros((H, W), dtype=bool)

    order = list(range(len(pieces)))
    random.shuffle(order)
    for idx in order:
        piece = pieces[idx]
        ph, pw = piece.crop.shape
        anchors = [(r, c) for r in range(H - ph + 1) for c in range(W - pw + 1)]
        random.shuffle(anchors)
        placed = False
        for r, c in anchors:
            rr0 = max(0, r - 1)
            cc0 = max(0, c - 1)
            rr1 = min(H, r + ph + 1)
            cc1 = min(W, c + pw + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            paste(grid, piece.crop, r, c, transparent=BG)
            blocked[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        sampled = _sample_piece_specs()
        if sampled is None:
            continue
        pieces, expected = sampled
        grid = _scatter_input(pieces)
        if grid is None:
            continue
        candidate_lists = _candidate_lists_from_grid(grid)
        assemblies = _enumerate_candidate_assemblies(candidate_lists)
        if len(assemblies) != 1:
            continue
        solved = main(grid)
        if np.array_equal(solved, expected):
            return grid
    raise RuntimeError("failed to generate a valid tenon-mortise chain input")
