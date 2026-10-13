from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
SPRITE = 2
FRAME = 4

CORNER_SIGNS = {
    "tl": (-1, -1),
    "tr": (-1, 1),
    "bl": (1, -1),
    "br": (1, 1),
}
SIGN_TO_CORNER = {sign: corner for corner, sign in CORNER_SIGNS.items()}
FRAME_KINDS = tuple(CORNER_SIGNS)
FLIP_CASES = (
    (False, False),
    (True, False),
    (False, True),
    (True, True),
)

BASE_SPRITES = {
    3: (np.array([[0, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),),
    4: (np.array([[0, 0, 0, 1], [0, 0, 1, 1], [1, 1, 1, 0]], dtype=bool),),
    5: (
        np.array(
            [[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0]],
            dtype=bool,
        ),
        np.array(
            [[1, 1, 1, 1, 1], [0, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 0, 0]],
            dtype=bool,
        ),
    ),
    6: (
        np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [0, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            dtype=bool,
        ),
    ),
}

GEN_RETRIES = 400
GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 28)
NOISE_RANGE = (1, 4)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    assert connectivity == 4
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def square_corners(r0: int, c0: int, side: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (r0, c0),
        "tr": (r0, c0 + side),
        "bl": (r0 + side, c0),
        "br": (r0 + side, c0 + side),
    }


def dominant_quadrant(mask: np.ndarray) -> str:
    coords = np.argwhere(mask)
    assert coords.size > 0
    h, w = mask.shape
    counts: Counter[tuple[int, int]] = Counter()
    for r, c in coords:
        vr = -1 if 2 * int(r) + 1 < h else (1 if 2 * int(r) + 1 > h else 0)
        hc = -1 if 2 * int(c) + 1 < w else (1 if 2 * int(c) + 1 > w else 0)
        if vr == 0 or hc == 0:
            continue
        counts[(vr, hc)] += 1
    if not counts:
        mean_r, mean_c = coords.mean(axis=0)
        fallback = (-1 if 2 * mean_r + 1 < h else 1, -1 if 2 * mean_c + 1 < w else 1)
        return SIGN_TO_CORNER[fallback]
    best = max(counts.values())
    winners = [sign for sign, value in counts.items() if value == best]
    if len(winners) > 1:
        mean_r, mean_c = coords.mean(axis=0)
        fallback = (-1 if 2 * mean_r + 1 < h else 1, -1 if 2 * mean_c + 1 < w else 1)
        if fallback in winners:
            return SIGN_TO_CORNER[fallback]
        winners = sorted(winners)
    return SIGN_TO_CORNER[winners[0]]


def mirror_to_kind(mask: np.ndarray, kind: str) -> np.ndarray:
    out = mask.copy()
    vr, hc = CORNER_SIGNS[kind]
    if vr > 0:
        out = out[::-1, :]
    if hc > 0:
        out = out[:, ::-1]
    return out


def is_nonsymmetric(mask: np.ndarray) -> bool:
    return (
        not np.array_equal(mask, mask[::-1, :])
        and not np.array_equal(mask, mask[:, ::-1])
        and not np.array_equal(mask, mask[::-1, ::-1])
    )


def sprite_side(mask: np.ndarray) -> int:
    return int(max(mask.shape))


def extract_red_sprites(grid: np.ndarray) -> tuple[dict[int, np.ndarray], list[tuple[int, int]]]:
    sprites: dict[int, np.ndarray] = {}
    isolated: list[tuple[int, int]] = []
    for comp in components_mask(grid == SPRITE, connectivity=4):
        if comp.shape[0] == 1:
            isolated.append(tuple(map(int, comp[0])))
            continue
        r0, c0, r1, c1 = bbox_coords(comp)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        shifted = comp - np.array([[r0, c0]])
        mask[shifted[:, 0], shifted[:, 1]] = True
        side = sprite_side(mask)
        assert side not in sprites
        sprites[side] = mask
    return sprites, isolated


def build_sprite(side: int, dominant_kind: str) -> np.ndarray:
    candidates: list[np.ndarray] = []
    for variant in BASE_SPRITES[side]:
        bases = [variant]
        if variant.shape[0] != variant.shape[1]:
            bases.append(variant.T)
        for base in bases:
            for flip_h in (False, True):
                for flip_v in (False, True):
                    mask = base.copy()
                    if flip_h:
                        mask = mask[:, ::-1]
                    if flip_v:
                        mask = mask[::-1, :]
                    if dominant_quadrant(mask) != dominant_kind:
                        continue
                    if not (3 <= int(mask.sum()) <= 10):
                        continue
                    if sprite_side(mask) != side:
                        continue
                    if not is_nonsymmetric(mask):
                        continue
                    if len(components_mask(mask, connectivity=4)) != 1:
                        continue
                    candidates.append(mask.copy())
    assert candidates
    return random.choice(candidates)


def orient_for_missing(mask: np.ndarray, missing_kind: str) -> np.ndarray:
    sprite_kind = dominant_quadrant(mask)
    s_v, s_h = CORNER_SIGNS[sprite_kind]
    m_v, m_h = CORNER_SIGNS[missing_kind]
    out = mask.copy()
    if s_h != m_h:
        out = out[:, ::-1]
    if s_v != m_v:
        out = out[::-1, :]
    return out


def stamp_origin(missing: tuple[int, int], missing_kind: str, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    mr, mc = map(int, missing)
    vr, hc = CORNER_SIGNS[missing_kind]
    r0 = mr - h + 1 if vr < 0 else mr
    c0 = mc - w + 1 if hc < 0 else mc
    return r0, c0


def stamp_rect(missing: tuple[int, int], missing_kind: str, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0 = stamp_origin(missing, missing_kind, shape)
    h, w = map(int, shape)
    return r0, c0, r0 + h - 1, c0 + w - 1


def paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    target = grid[r0 : r0 + h, c0 : c0 + w]
    target[mask] = int(color)


def paint_frame_markers(grid: np.ndarray, frame: dict) -> None:
    for r, c in frame["markers"]:
        grid[r, c] = FRAME


def frame_candidates(grid: np.ndarray) -> list[dict]:
    coords = [tuple(map(int, p)) for p in np.argwhere(grid == FRAME)]
    coord_set = set(coords)
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    seen = set()
    out = []
    for r0 in rows:
        for r1 in rows:
            if r1 <= r0:
                continue
            side = r1 - r0
            for c0 in cols:
                c1 = c0 + side
                if c1 not in cols:
                    continue
                corners = square_corners(r0, c0, side)
                present = tuple(sorted(pos for pos in corners.values() if pos in coord_set))
                if len(present) != 3:
                    continue
                missing_kind = next(kind for kind, pos in corners.items() if pos not in present)
                key = (side, missing_kind, present)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "side": side,
                        "missing_kind": missing_kind,
                        "missing": corners[missing_kind],
                        "markers": present,
                    }
                )
    return out


def find_frames(grid: np.ndarray) -> list[dict]:
    coords = [tuple(map(int, p)) for p in np.argwhere(grid == FRAME)]
    assert coords and len(coords) % 3 == 0
    candidates = frame_candidates(grid)
    by_marker: dict[tuple[int, int], list[int]] = {}
    for idx, cand in enumerate(candidates):
        for marker in cand["markers"]:
            by_marker.setdefault(marker, []).append(idx)

    target_count = len(coords) // 3
    remaining_all = set(coords)
    solutions: list[list[dict]] = []

    def backtrack(remaining: set[tuple[int, int]], used_sides: set[int], chosen: list[dict]) -> None:
        if len(solutions) > 1:
            return
        if not remaining:
            if len(chosen) == target_count:
                solutions.append(chosen.copy())
            return
        if len(chosen) >= target_count:
            return
        start = min(remaining)
        for idx in by_marker.get(start, []):
            cand = candidates[idx]
            markers = set(cand["markers"])
            if cand["side"] in used_sides or not markers.issubset(remaining):
                continue
            chosen.append(cand)
            backtrack(remaining - markers, used_sides | {cand["side"]}, chosen)
            chosen.pop()

    backtrack(remaining_all, set(), [])
    assert len(solutions) == 1
    return sorted(solutions[0], key=lambda frame: frame["side"])


def build_output_from_parts(shape: tuple[int, int], frames: list[dict], sprites: dict[int, np.ndarray]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for frame in frames:
        paint_frame_markers(out, frame)
    for frame in frames:
        sprite = orient_for_missing(sprites[frame["side"]], frame["missing_kind"])
        r0, c0 = stamp_origin(frame["missing"], frame["missing_kind"], sprite.shape)
        h, w = sprite.shape
        assert 0 <= r0 and 0 <= c0 and r0 + h <= shape[0] and c0 + w <= shape[1]
        target = out[r0 : r0 + h, c0 : c0 + w]
        assert not np.any(target[sprite] == FRAME)
        target[sprite] = SPRITE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase isolated red distractors, then stamp each remaining sprite into its matched three-corner frame with the flip chosen by missing-corner side and sprite centroid.

    Input structure:
    - A black background contains yellow singleton markers arranged as 2-4 separate three-corner square frames.
    - Red pixels form the same number of multi-pixel 4-connected sprites plus extra isolated red single-pixel distractors.
    - Each real sprite has a unique bounding-box side length that matches exactly one yellow frame side length.

    Rule:
    - Find all red 4-connected components and erase every area-1 component.
    - Match each remaining sprite to the yellow frame with the same square side length, then detect the frame's missing corner.
    - Compute the sprite's dominant centroid quadrant inside its own bbox; flip horizontally when that quadrant is horizontally opposite the missing corner, and flip vertically when it is vertically opposite.
    - Stamp the flipped sprite at the missing corner so the bbox corner facing the frame lands on the missing point, and keep only the yellow markers plus these stamped red results.

    Key insight:
    - The yellow frame fixes where the sprite must go, but the exact reflection is only determined after first filtering out isolated red noise and then combining missing-corner direction with the sprite's internal mass bias.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    frames = find_frames(g)
    sprites, _isolated = extract_red_sprites(g)
    frame_sides = {frame["side"] for frame in frames}
    assert frame_sides == set(sprites)
    return build_output_from_parts(g.shape, frames, sprites)


def mark_rect(mask: np.ndarray, rect: tuple[int, int, int, int], pad: int = 0) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = rect
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def frame_reserved_area(item: dict) -> int:
    h, w = item["sprite"].shape
    return (item["side"] + 1) ** 2 + h * w


def quadrant_for_flip(missing_kind: str, flip_v: bool, flip_h: bool) -> str:
    m_v, m_h = CORNER_SIGNS[missing_kind]
    return SIGN_TO_CORNER[(-m_v if flip_v else m_v, -m_h if flip_h else m_h)]


def candidate_frame_placements(h: int, w: int, item: dict) -> list[dict]:
    side = item["side"]
    sprite_shape = item["sprite"].shape
    missing_kind = item["missing_kind"]
    candidates = []
    for r0 in range(h - side):
        for c0 in range(w - side):
            corners = square_corners(r0, c0, side)
            missing = corners[missing_kind]
            s_rect = stamp_rect(missing, missing_kind, sprite_shape)
            sr0, sc0, sr1, sc1 = s_rect
            if sr0 < 0 or sc0 < 0 or sr1 >= h or sc1 >= w:
                continue
            candidates.append(
                {
                    "frame_r0": r0,
                    "frame_c0": c0,
                    "missing": missing,
                    "markers": tuple(pos for kind, pos in corners.items() if kind != missing_kind),
                    "frame_rect": (r0, c0, r0 + side, c0 + side),
                    "stamp_rect": s_rect,
                }
            )
    random.shuffle(candidates)
    return candidates


def place_frames(meta_items: list[dict], h: int, w: int) -> bool:
    guard = np.zeros((h, w), dtype=bool)
    for item in sorted(meta_items, key=frame_reserved_area, reverse=True):
        chosen = None
        for cand in candidate_frame_placements(h, w, item):
            test = np.zeros_like(guard)
            mark_rect(test, cand["frame_rect"], pad=1)
            mark_rect(test, cand["stamp_rect"], pad=1)
            if np.any(test & guard):
                continue
            chosen = cand
            guard |= test
            break
        if chosen is None:
            return False
        item.update(chosen)
    return True


def candidate_bbox_positions(guard: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = guard.shape
    ph, pw = map(int, shape)
    out = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            if guard[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)].any():
                continue
            out.append((r0, c0))
    random.shuffle(out)
    return out


def render_input(meta: dict) -> np.ndarray:
    h, w = meta["shape"]
    grid = np.full((h, w), BG, dtype=int)
    for item in meta["items"]:
        paint_frame_markers(grid, item)
    for item in meta["items"]:
        r0, c0 = item["input_pos"]
        paint_mask(grid, item["sprite"], r0, c0, SPRITE)
    for r, c in meta["noise"]:
        grid[r, c] = SPRITE
    return grid


def build_expected_output(meta: dict) -> np.ndarray:
    frames = [
        {
            "side": item["side"],
            "missing_kind": item["missing_kind"],
            "missing": item["missing"],
            "markers": tuple(item["markers"]),
        }
        for item in meta["items"]
    ]
    sprites = {item["side"]: item["sprite"] for item in meta["items"]}
    return build_output_from_parts(meta["shape"], frames, sprites)


def validate_example(grid: np.ndarray, meta: dict | None = None) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert set(np.unique(g)).issubset({BG, SPRITE, FRAME})

    frames = find_frames(g)
    assert 2 <= len(frames) <= 4
    frame_sides = [frame["side"] for frame in frames]
    assert len(frame_sides) == len(set(frame_sides))

    sprites, isolated = extract_red_sprites(g)
    assert len(sprites) == len(frames)
    assert len(isolated) >= 1
    assert set(sprites) == set(frame_sides)
    missing_points = {frame["missing"] for frame in frames}
    assert all(pixel not in missing_points for pixel in isolated)

    flip_cases = set()
    for frame in frames:
        sprite = sprites[frame["side"]]
        assert 3 <= int(sprite.sum()) <= 10
        assert is_nonsymmetric(sprite)
        kind = dominant_quadrant(sprite)
        s_v, s_h = CORNER_SIGNS[kind]
        m_v, m_h = CORNER_SIGNS[frame["missing_kind"]]
        flip_case = (s_v != m_v, s_h != m_h)
        flip_cases.add(flip_case)
        if flip_case != (False, False):
            assert not np.array_equal(orient_for_missing(sprite, frame["missing_kind"]), sprite)

    assert len(flip_cases) >= min(2, len(frames))

    out = main(g)
    assert out.shape == g.shape
    assert np.array_equal(out == FRAME, g == FRAME)
    out_sprites, out_isolated = extract_red_sprites(out)
    assert not out_isolated
    assert set(out_sprites) == set(frame_sides)

    if meta is not None:
        expected = build_expected_output(meta)
        assert np.array_equal(out, expected)
        parsed_frames = {(frame["side"], frame["missing_kind"], frame["missing"]) for frame in frames}
        meta_frames = {(item["side"], item["missing_kind"], item["missing"]) for item in meta["items"]}
        assert parsed_frames == meta_frames


def _gen_attempt() -> dict | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n = random.randint(2, 4)
    sides = random.sample([3, 4, 5, 6], n)
    missing_kinds = random.sample(FRAME_KINDS, n)
    flip_cases = random.sample(FLIP_CASES, n)

    meta_items = []
    for side, missing_kind, (flip_v, flip_h) in zip(sides, missing_kinds, flip_cases):
        dominant_kind = quadrant_for_flip(missing_kind, flip_v, flip_h)
        sprite = build_sprite(side, dominant_kind)
        meta_items.append(
            {
                "side": side,
                "missing_kind": missing_kind,
                "flip_case": (flip_v, flip_h),
                "dominant_kind": dominant_kind,
                "sprite": sprite,
            }
        )

    if not place_frames(meta_items, h, w):
        return None

    guard = np.zeros((h, w), dtype=bool)
    for item in meta_items:
        mark_rect(guard, item["frame_rect"], pad=1)
        mark_rect(guard, item["stamp_rect"], pad=1)

    for item in sorted(meta_items, key=lambda entry: int(entry["sprite"].sum()), reverse=True):
        positions = candidate_bbox_positions(guard, item["sprite"].shape)
        if not positions:
            return None
        item["input_pos"] = positions[0]
        r0, c0 = item["input_pos"]
        mark_rect(guard, (r0, c0, r0 + item["sprite"].shape[0] - 1, c0 + item["sprite"].shape[1] - 1), pad=1)

    noise_forbidden = {item["missing"] for item in meta_items}
    available = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if not guard[r, c] and (r, c) not in noise_forbidden
    ]
    random.shuffle(available)
    if not available:
        return None
    noise_count = min(random.randint(*NOISE_RANGE), len(available))
    noise = []
    for r, c in available:
        if len(noise) >= noise_count:
            break
        if guard[r, c]:
            continue
        noise.append((r, c))
        mark_rect(guard, (r, c, r, c), pad=1)
    if not noise:
        return None

    meta = {"shape": (h, w), "items": meta_items, "noise": noise}
    grid = render_input(meta)
    if np.array_equal(grid, build_expected_output(meta)):
        return None
    try:
        validate_example(grid, meta)
    except AssertionError:
        return None
    return meta


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        meta = _gen_attempt()
        if meta is not None:
            return render_input(meta)
    raise RuntimeError("failed to generate a valid input for puzzle 4")
