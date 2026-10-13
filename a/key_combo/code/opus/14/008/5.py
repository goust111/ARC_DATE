from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
SPRITE = 2
MARKER = 4

CORNER_SIGN = {
    "tl": (-1, -1),
    "tr": (-1, 1),
    "bl": (1, -1),
    "br": (1, 1),
}
SIGN_TO_CORNER = {v: k for k, v in CORNER_SIGN.items()}
CORNER_ORDER = ("tl", "tr", "bl", "br")


def _dominant_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((factor, factor), dtype=int)).astype(bool)


def _frame_corners(r0: int, c0: int, side: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (r0, c0),
        "tr": (r0, c0 + side),
        "bl": (r0 + side, c0),
        "br": (r0 + side, c0 + side),
    }


def _three_corner_frames(grid: np.ndarray, marker_color: int = MARKER) -> list[dict]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == int(marker_color))]
    coord_set = set(coords)
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    frames: list[dict] = []

    for r0 in rows:
        for r1 in rows:
            if r1 <= r0:
                continue
            side = r1 - r0
            for c0 in cols:
                c1 = c0 + side
                if c1 not in cols:
                    continue
                corners = _frame_corners(r0, c0, side)
                present = [name for name, pos in corners.items() if pos in coord_set]
                if len(present) != 3:
                    continue
                missing_kind = next(name for name in CORNER_ORDER if name not in present)
                sign = CORNER_SIGN[missing_kind]
                missing = corners[missing_kind]
                paste_r = missing[0] - side + 1 if sign[0] < 0 else missing[0]
                paste_c = missing[1] - side + 1 if sign[1] < 0 else missing[1]
                frames.append(
                    {
                        "side": int(side),
                        "kind": missing_kind,
                        "missing": missing,
                        "markers": tuple(corners[name] for name in present),
                        "paste": (int(paste_r), int(paste_c)),
                    }
                )
    return frames


def _missing_matches_bbox(kind: str, missing: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    if kind == "tl":
        return missing == (r1, c1)
    if kind == "tr":
        return missing == (r1, c0)
    if kind == "bl":
        return missing == (r0, c1)
    if kind == "br":
        return missing == (r0, c0)
    raise AssertionError(kind)


def _oriented_scaled_mask(source_mask: np.ndarray, source_kind: str, target_kind: str, scale: int) -> np.ndarray:
    out = _scale_mask(source_mask, scale)
    source_sign = CORNER_SIGN[source_kind]
    target_sign = CORNER_SIGN[target_kind]
    if source_sign[0] != target_sign[0]:
        out = out[::-1, :]
    if source_sign[1] != target_sign[1]:
        out = out[:, ::-1]
    return out


def _paste_red(out: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    h, w = mask.shape
    target = out[r0 : r0 + h, c0 : c0 + w]
    assert target.shape == mask.shape
    assert np.all((target[mask] == BG) | (target[mask] == SPRITE))
    target[mask] = SPRITE


def main(grid: np.ndarray) -> np.ndarray:
    """Scale and flip the source sprite into every three-corner marker frame according to frame size and missing-corner position.

    Input structure:
    - The background is black.
    - One red 4-connected sprite occupies a tight square bbox of side K where K is 3 or 4.
    - Yellow singleton pixels appear in groups of three, each group marking three corners of a square frame with one missing corner.
    - Exactly one yellow frame has side K and its missing corner touches the corresponding corner of the source sprite bbox.

    Rule:
    - Extract the red source sprite mask from its KxK bbox and identify the source yellow frame whose missing corner anchors that bbox.
    - For every other yellow frame of side L, compute the integer scale factor S = L / K, enlarge the source mask by pixel replication, and flip it horizontally and/or vertically according to how the target missing corner differs from the source missing corner.
    - Paste the scaled and oriented red mask outside the target frame so that the bbox corner corresponding to the target missing corner touches the missing-corner position.
    - Keep the original yellow markers and the original source sprite unchanged.

    Key insight:
    - The same three-corner square encodes two independent parameters at once: the absent corner gives the flip orientation, while the square side length gives the sprite scaling factor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sprite_parts = _components(g == SPRITE)
    assert len(sprite_parts) == 1
    sprite_coords = sprite_parts[0]
    sprite_bbox = _bbox_coords(sprite_coords)
    r0, c0, r1, c1 = sprite_bbox
    k = r1 - r0 + 1
    assert k == c1 - c0 + 1
    sprite_mask = _crop_bbox(g == SPRITE, sprite_bbox)

    frames = _three_corner_frames(g, MARKER)
    marker_coords = sorted(map(tuple, np.argwhere(g == MARKER)))
    used_markers = sorted(marker for frame in frames for marker in frame["markers"])
    assert used_markers == marker_coords

    source_frames = [
        frame
        for frame in frames
        if frame["side"] == k and _missing_matches_bbox(frame["kind"], frame["missing"], sprite_bbox)
    ]
    assert len(source_frames) == 1
    source = source_frames[0]

    out = g.copy()
    for frame in frames:
        if frame is source:
            continue
        side = int(frame["side"])
        assert side % k == 0
        scale = side // k
        stamp = _oriented_scaled_mask(sprite_mask, source["kind"], frame["kind"], scale)
        pr, pc = frame["paste"]
        assert 0 <= pr and 0 <= pc
        assert pr + side <= g.shape[0] and pc + side <= g.shape[1]
        _paste_red(out, stamp, pr, pc)
    return out


GEN_GRID_RANGE = (24, 30)
GEN_K_CHOICES = (3, 4)
GEN_TARGET_COUNT = (2, 3)
GEN_MAX_ATTEMPTS = 500
GEN_MARKER_CLEARANCE = 1


def _all_dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    cur = mask
    for _ in range(4):
        variants.append(cur)
        variants.append(np.fliplr(cur))
        cur = np.rot90(cur)
    return variants


def _valid_source_mask(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    if len(_components(mask)) != 1:
        return False
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_coords(coords)
    if (r0, c0) != (0, 0) or r1 != mask.shape[0] - 1 or c1 != mask.shape[1] - 1:
        return False
    if mask.all():
        return False
    for variant in _all_dihedral_variants(mask)[1:]:
        if np.array_equal(mask, variant):
            return False
    return True


def _random_source_mask(k: int) -> np.ndarray | None:
    cells = [(r, c) for r in range(k) for c in range(k)]
    min_area = max(4, k + 1)
    max_area = k * k - 1
    for _ in range(150):
        area = random.randint(min_area, max_area)
        mask = np.zeros((k, k), dtype=bool)
        for r, c in random.sample(cells, area):
            mask[r, c] = True
        if _valid_source_mask(mask):
            return mask
    return None


def _kind_by_relation(source_kind: str, relation: str) -> str:
    sr, sc = CORNER_SIGN[source_kind]
    if relation == "same":
        return source_kind
    if relation == "h":
        return SIGN_TO_CORNER[(sr, -sc)]
    if relation == "v":
        return SIGN_TO_CORNER[(-sr, sc)]
    if relation == "hv":
        return SIGN_TO_CORNER[(-sr, -sc)]
    raise AssertionError(relation)


def _target_relations(count: int) -> list[str]:
    if count == 2:
        return list(
            random.choice(
                [
                    ("h", "v"),
                    ("h", "hv"),
                    ("v", "hv"),
                    ("same", "h"),
                    ("same", "v"),
                    ("same", "hv"),
                ]
            )
        )
    return list(
        random.choice(
            [
                ("same", "h", "v"),
                ("same", "h", "hv"),
                ("same", "v", "hv"),
                ("h", "v", "hv"),
            ]
        )
    )


def _make_frame(kind: str, side: int, r0: int, c0: int) -> dict:
    corners = _frame_corners(r0, c0, side)
    missing = corners[kind]
    sign = CORNER_SIGN[kind]
    paste_r = missing[0] - side + 1 if sign[0] < 0 else missing[0]
    paste_c = missing[1] - side + 1 if sign[1] < 0 else missing[1]
    return {
        "kind": kind,
        "side": int(side),
        "missing": missing,
        "markers": tuple(corners[name] for name in CORNER_ORDER if name != kind),
        "paste": (int(paste_r), int(paste_c)),
    }


def _candidate_frames(shape: tuple[int, int], kind: str, side: int) -> list[dict]:
    h, w = shape
    frames = []
    for r0 in range(h - side):
        for c0 in range(w - side):
            frame = _make_frame(kind, side, r0, c0)
            pr, pc = frame["paste"]
            if 0 <= pr and 0 <= pc and pr + side <= h and pc + side <= w:
                frames.append(frame)
    random.shuffle(frames)
    return frames


def _control_keys(marker_coords: set[tuple[int, int]]) -> set[tuple[int, str, tuple[int, int], tuple[tuple[int, int], ...]]]:
    rows = sorted({r for r, _ in marker_coords})
    cols = sorted({c for _, c in marker_coords})
    keys = set()
    for r0 in rows:
        for r1 in rows:
            if r1 <= r0:
                continue
            side = r1 - r0
            for c0 in cols:
                c1 = c0 + side
                if c1 not in cols:
                    continue
                corners = _frame_corners(r0, c0, side)
                present = [kind for kind, pos in corners.items() if pos in marker_coords]
                if len(present) != 3:
                    continue
                missing = next(kind for kind in CORNER_ORDER if kind not in present)
                keys.add((side, missing, corners[missing], tuple(sorted(corners[k] for k in present))))
    return keys


def _clean_controls(placed: list[dict]) -> bool:
    markers = {pos for frame in placed for pos in frame["markers"]}
    actual = _control_keys(markers)
    intended = {
        (frame["side"], frame["kind"], frame["missing"], tuple(sorted(frame["markers"])))
        for frame in placed
    }
    return actual == intended


def _can_place(
    frame: dict,
    placed: list[dict],
    paste_used: np.ndarray,
    marker_used: np.ndarray,
    marker_forbidden: np.ndarray,
) -> bool:
    side = int(frame["side"])
    pr, pc = frame["paste"]
    if paste_used[pr : pr + side, pc : pc + side].any():
        return False
    if marker_used[pr : pr + side, pc : pc + side].any():
        return False
    for r, c in frame["markers"]:
        if marker_forbidden[r, c] or paste_used[r, c]:
            return False
    return _clean_controls(placed + [frame])


def _mark_frame(frame: dict, paste_used: np.ndarray, marker_used: np.ndarray, marker_forbidden: np.ndarray) -> None:
    side = int(frame["side"])
    pr, pc = frame["paste"]
    paste_used[pr : pr + side, pc : pc + side] = True
    h, w = paste_used.shape
    for r, c in frame["markers"]:
        marker_used[r, c] = True
        marker_forbidden[
            max(0, r - GEN_MARKER_CLEARANCE) : min(h, r + GEN_MARKER_CLEARANCE + 1),
            max(0, c - GEN_MARKER_CLEARANCE) : min(w, c + GEN_MARKER_CLEARANCE + 1),
        ] = True


def _place_frames(shape: tuple[int, int], specs: list[dict]) -> list[dict] | None:
    paste_used = np.zeros(shape, dtype=bool)
    marker_used = np.zeros(shape, dtype=bool)
    marker_forbidden = np.zeros(shape, dtype=bool)
    placed: list[dict] = []
    for spec in sorted(specs, key=lambda x: (-x["side"], x["is_source"])):
        candidates = _candidate_frames(shape, spec["kind"], spec["side"])
        for frame in candidates:
            if _can_place(frame, placed, paste_used, marker_used, marker_forbidden):
                frame["is_source"] = spec["is_source"]
                placed.append(frame)
                _mark_frame(frame, paste_used, marker_used, marker_forbidden)
                break
        else:
            return None
    return placed


def _render_input_output(shape: tuple[int, int], placed: list[dict], source_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.zeros(shape, dtype=int)
    out = np.zeros(shape, dtype=int)

    source_frame = next(frame for frame in placed if frame["is_source"])
    for frame in placed:
        for r, c in frame["markers"]:
            inp[r, c] = MARKER
            out[r, c] = MARKER

    source_side = int(source_frame["side"])
    source_stamp = source_mask.copy()
    assert source_stamp.shape == (source_side, source_side)
    sr, sc = source_frame["paste"]
    _paste_red(inp, source_stamp, sr, sc)
    _paste_red(out, source_stamp, sr, sc)

    for frame in placed:
        if frame["is_source"]:
            continue
        scale = int(frame["side"]) // source_side
        stamp = _oriented_scaled_mask(source_mask, source_frame["kind"], frame["kind"], scale)
        pr, pc = frame["paste"]
        _paste_red(out, stamp, pr, pc)

    return inp, out


def _sample_specs(k: int) -> list[dict]:
    source_kind = random.choice(CORNER_ORDER)
    target_count = random.randint(*GEN_TARGET_COUNT)
    relations = _target_relations(target_count)
    scales = [2, 3]
    if target_count == 3:
        scales.append(random.choice([2, 3]))
    random.shuffle(scales)
    specs = [{"kind": source_kind, "side": k, "is_source": True}]
    for relation, scale in zip(relations, scales):
        specs.append(
            {
                "kind": _kind_by_relation(source_kind, relation),
                "side": k * scale,
                "is_source": False,
            }
        )
    return specs


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_GRID_RANGE)
    w = random.randint(*GEN_GRID_RANGE)
    k = random.choice(GEN_K_CHOICES)
    source_mask = _random_source_mask(k)
    if source_mask is None:
        return None

    specs = _sample_specs(k)
    placed = _place_frames((h, w), specs)
    if placed is None:
        return None

    inp, out = _render_input_output((h, w), placed, source_mask)
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 5")
