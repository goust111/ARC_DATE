import itertools
import random
from collections import deque

import numpy as np


BG = 0
COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIR_TO_CANON_K = (0, 1, 2, 3)


def _components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or int(grid[sr, sc]) == bg:
                continue
            color = int(grid[sr, sc])
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            coords = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != bg:
                        assert int(grid[nr, nc]) == color, "each piece must be solid-colored"
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
            comps.append({"color": color, "bbox": (int(r0), int(c0), int(r1), int(c1)), "patch": patch})
    return comps


def _profile(values: np.ndarray, *, true_cells: bool) -> tuple[int, ...]:
    vals = values.astype(bool)
    idx = np.flatnonzero(vals if true_cells else ~vals)
    return tuple(int(x) for x in idx)


def _proper_profile(profile: tuple[int, ...], length: int) -> bool:
    return 0 < len(profile) < length


def _parse_canonical(patch: np.ndarray) -> dict | None:
    mask = patch != BG
    colors = sorted(set(int(x) for x in patch[mask]))
    if len(colors) != 1:
        return None
    color = colors[0]
    h, w = mask.shape
    if h < 4 or w < 3:
        return None

    # Canonical orientation: chain grows to the right.  A base has only a
    # right-edge notch; a moving piece has a left tab and maybe a right notch.
    if np.all(mask[:, :-1]):
        notch = _profile(mask[:, -1], true_cells=False)
        if _proper_profile(notch, h):
            return {"kind": "base", "color": color, "patch": patch, "height": h, "body_w": w, "tab": None, "notch": notch}

    if np.all(mask[:, 1:]):
        tab = _profile(mask[:, 0], true_cells=True)
        if _proper_profile(tab, h):
            return {"kind": "last", "color": color, "patch": patch, "height": h, "body_w": w - 1, "tab": tab, "notch": None}

    if w >= 3 and np.all(mask[:, 1:-1]):
        tab = _profile(mask[:, 0], true_cells=True)
        notch = _profile(mask[:, -1], true_cells=False)
        if _proper_profile(tab, h) and _proper_profile(notch, h):
            return {"kind": "middle", "color": color, "patch": patch, "height": h, "body_w": w - 1, "tab": tab, "notch": notch}

    return None


def _overlay(out: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != BG
    h, w = patch.shape
    region = out[r0 : r0 + h, c0 : c0 + w]
    assert region.shape == patch.shape, "patch placement out of bounds"
    assert not np.any((region != BG) & mask), "assembled pieces must not overlap"
    region[mask] = patch[mask]


def _solve_as_direction(grid: np.ndarray, k: int) -> np.ndarray:
    comps = _components(grid, BG)
    assert len(comps) >= 4, "expected a base and at least three moving pieces"

    parsed = []
    for comp in comps:
        canon_patch = np.rot90(comp["patch"], k).copy()
        item = _parse_canonical(canon_patch)
        assert item is not None, "component is not a valid tab/notch piece"
        parsed.append(item)

    heights = {item["height"] for item in parsed}
    assert len(heights) == 1, "all pieces must share the same assembly thickness"

    bases = [item for item in parsed if item["kind"] == "base"]
    assert len(bases) == 1, "there must be exactly one notched base"
    base = bases[0]

    remaining = [item for item in parsed if item is not base]
    assert all(item["tab"] is not None for item in remaining), "every moving piece needs a tab"
    assert sum(1 for item in remaining if item["notch"] is None) == 1, "there must be exactly one terminal piece"
    assert base["body_w"] > max(item["body_w"] for item in remaining), "the fixed base is the broadest body"

    total_w = base["body_w"] + sum(item["body_w"] for item in remaining)
    h = base["height"]
    out = np.zeros((h, total_w), dtype=int)
    _overlay(out, base["patch"], 0, 0)

    cursor = base["body_w"]
    open_notch = base["notch"]
    chain = []
    while remaining:
        matches = [item for item in remaining if item["tab"] == open_notch]
        assert len(matches) == 1, "each open notch must have one unique complementary tab"
        piece = matches[0]
        chain.append(piece)
        remaining = [item for item in remaining if item is not piece]
        _overlay(out, piece["patch"], 0, cursor - 1)
        cursor += piece["body_w"]
        if piece["notch"] is None:
            assert not remaining, "terminal piece can only appear at the end of the chain"
            break
        open_notch = piece["notch"]

    assert not remaining and cursor == total_w, "assembly chain must consume all pieces"
    assert np.all(out != BG), "assembled result must be a seamless solid rectangle"
    return np.rot90(out, -k).copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble tabbed and notched pieces into one seamless solid rectangle.

    Input structure:
    - The background is black.  One broad solid-colored base rectangle has a
      single edge notch, and several separated smaller solid-colored moving
      pieces each have one protruding tab; all but the terminal moving piece
      also have their own notch on the opposite chain edge.
    - Piece colors and scattered distances are decoys: some pieces may share
      colors or sit closer to an open notch while having the wrong tab profile.

    Rule:
    - Try the four cardinal chain directions without rotating individual
      pieces relative to each other.
    - Starting from the unique base notch, repeatedly translate the only piece
      whose tab profile exactly complements the current notch.  A placed
      piece's own notch becomes the next open interface.
    - Preserve all piece colors and return the tight assembled rectangle once
      the terminal tab-only piece closes the last gap.

    Key insight:
    - The order is not labeled by color, distance, or size; the unique visible
      key is the exact shape profile shared by each notch and the next tab.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected a 2D ARC grid"

    candidates = []
    for k in DIR_TO_CANON_K:
        try:
            candidates.append(_solve_as_direction(g, k))
        except AssertionError:
            pass
    assert len(candidates) == 1, "the tab/notch chain must determine one output"
    return candidates[0]


# ---------------------------- generator ----------------------------

GEN_ATTEMPTS = 500
PLACE_ATTEMPTS = 80
PAD = 1


def _transform_coords(coords: list[tuple[int, int]], shape: tuple[int, int], k: int) -> list[tuple[int, int]]:
    k %= 4
    h, w = shape
    out = []
    for r, c in coords:
        if k == 0:
            out.append((r, c))
        elif k == 1:
            out.append((w - 1 - c, r))
        elif k == 2:
            out.append((h - 1 - r, w - 1 - c))
        else:
            out.append((c, h - 1 - r))
    return out


def _mean_point(coords: list[tuple[int, int]], loc: tuple[int, int]) -> tuple[float, float]:
    rr = [loc[0] + r for r, _ in coords]
    cc = [loc[1] + c for _, c in coords]
    return (sum(rr) / len(rr), sum(cc) / len(cc))


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _all_profiles(h: int) -> list[tuple[int, ...]]:
    profiles = []
    for size in range(1, h):
        for combo in itertools.combinations(range(h), size):
            profiles.append(tuple(int(x) for x in combo))
    random.shuffle(profiles)
    return profiles


def _choose_profiles(h: int, n_active: int) -> list[tuple[int, ...]]:
    same_size = random.randint(2, min(3, h - 2))
    noncontig = [
        p for p in itertools.combinations(range(h), same_size)
        if p[-1] - p[0] + 1 != len(p)
    ]
    p0 = tuple(random.choice(noncontig))
    p2_options = [p for p in itertools.combinations(range(h), same_size) if tuple(p) != p0]
    p2 = tuple(random.choice(p2_options))

    profiles: list[tuple[int, ...] | None] = [None] * n_active
    profiles[0] = p0
    profiles[2] = p2
    used = {p0, p2}
    for p in _all_profiles(h):
        if p in used:
            continue
        for i in range(n_active):
            if profiles[i] is None:
                profiles[i] = p
                used.add(p)
                break
        if all(x is not None for x in profiles):
            break
    assert all(x is not None for x in profiles)
    return [tuple(x) for x in profiles if x is not None]


def _choose_colors(n_active: int) -> tuple[int, list[int]]:
    dup = random.choice(COLORS)
    base_color = random.choice([c for c in COLORS if c != dup])
    piece_colors = [0] * n_active
    piece_colors[0] = dup
    piece_colors[2] = dup
    for i in range(n_active):
        if piece_colors[i]:
            continue
        banned = {base_color}
        if i > 0:
            banned.add(piece_colors[i - 1])
        if i + 1 < n_active and piece_colors[i + 1]:
            banned.add(piece_colors[i + 1])
        if i in (1, 3):
            banned.add(dup)
        choices = [c for c in COLORS if c not in banned]
        piece_colors[i] = random.choice(choices)
    return base_color, piece_colors


def _make_canonical_patches() -> tuple[list[np.ndarray], np.ndarray, list[dict]]:
    thickness = random.randint(5, 8)
    n_active = random.randint(3, 5)
    body_ws = [random.randint(2, 4) for _ in range(n_active)]
    base_w = random.randint(max(body_ws) + 1, 7)
    profiles = _choose_profiles(thickness, n_active)
    base_color, piece_colors = _choose_colors(n_active)

    patches = []
    meta = []
    base = np.full((thickness, base_w), base_color, dtype=int)
    base[list(profiles[0]), -1] = BG
    patches.append(base)
    meta.append({"role": "base", "profile": profiles[0], "body_w": base_w})

    for i in range(n_active):
        body_w = body_ws[i]
        patch = np.zeros((thickness, body_w + 1), dtype=int)
        patch[:, 1:] = piece_colors[i]
        patch[list(profiles[i]), 0] = piece_colors[i]
        if i < n_active - 1:
            patch[list(profiles[i + 1]), -1] = BG
        patches.append(patch)
        meta.append({"role": "piece", "profile": profiles[i], "body_w": body_w})

    total_w = base_w + sum(body_ws)
    expected = np.zeros((thickness, total_w), dtype=int)
    _overlay(expected, base, 0, 0)
    cursor = base_w
    for patch, body_w in zip(patches[1:], body_ws):
        _overlay(expected, patch, 0, cursor - 1)
        cursor += body_w

    assert np.all(expected != BG)
    return patches, expected, meta


def _forbidden_mask(occupied: np.ndarray, padding: int = PAD) -> np.ndarray:
    if padding <= 0:
        return occupied.copy()
    h, w = occupied.shape
    forbidden = np.zeros_like(occupied, dtype=bool)
    for dr in range(-padding, padding + 1):
        for dc in range(-padding, padding + 1):
            src_r0 = max(0, -dr)
            src_r1 = min(h, h - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(w, w - dc)
            dst_r0 = src_r0 + dr
            dst_r1 = src_r1 + dr
            dst_c0 = src_c0 + dc
            dst_c1 = src_c1 + dc
            forbidden[dst_r0:dst_r1, dst_c0:dst_c1] |= occupied[src_r0:src_r1, src_c0:src_c1]
    return forbidden


def _valid_locations(occupied: np.ndarray, patch: np.ndarray) -> list[tuple[int, int]]:
    gh, gw = occupied.shape
    ph, pw = patch.shape
    forbidden = _forbidden_mask(occupied, PAD)
    mask = patch != BG
    locs = []
    for r in range(gh - ph + 1):
        for c in range(gw - pw + 1):
            if not np.any(forbidden[r : r + ph, c : c + pw] & mask):
                locs.append((r, c))
    random.shuffle(locs)
    return locs


def _paste_piece(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, loc: tuple[int, int]) -> None:
    r0, c0 = loc
    mask = patch != BG
    region = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    region[mask] = patch[mask]
    occ_region = occupied[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    occ_region[mask] = True


def _port_center(meta_item: dict, canon_patch: np.ndarray, loc: tuple[int, int], k: int, *, base_notch: bool) -> tuple[float, float]:
    if base_notch:
        coords = [(r, meta_item["body_w"] - 1) for r in meta_item["profile"]]
    else:
        coords = [(r, 0) for r in meta_item["profile"]]
    actual_coords = _transform_coords(coords, canon_patch.shape, -k)
    return _mean_point(actual_coords, loc)


def _scatter_with_decoys(canon_patches: list[np.ndarray], meta: list[dict], k: int) -> np.ndarray | None:
    actual_patches = [np.rot90(p, -k).copy() for p in canon_patches]

    for _ in range(PLACE_ATTEMPTS):
        gh = random.randint(22, 30)
        gw = random.randint(22, 30)
        grid = np.zeros((gh, gw), dtype=int)
        occupied = np.zeros((gh, gw), dtype=bool)
        locs: list[tuple[int, int] | None] = [None] * len(actual_patches)

        base_locs = _valid_locations(occupied, actual_patches[0])
        if not base_locs:
            continue
        locs[0] = random.choice(base_locs)
        _paste_piece(grid, occupied, actual_patches[0], locs[0])
        base_center = _port_center(meta[0], canon_patches[0], locs[0], k, base_notch=True)

        wrong_i = 3  # third moving piece: same color and same tab size as the first, wrong shape.
        wrong_candidates = _valid_locations(occupied, actual_patches[wrong_i])
        if not wrong_candidates:
            continue
        wrong_candidates.sort(key=lambda loc: _dist(base_center, _port_center(meta[wrong_i], canon_patches[wrong_i], loc, k, base_notch=False)))
        locs[wrong_i] = random.choice(wrong_candidates[: min(12, len(wrong_candidates))])
        wrong_center = _port_center(meta[wrong_i], canon_patches[wrong_i], locs[wrong_i], k, base_notch=False)
        wrong_dist = _dist(base_center, wrong_center)
        _paste_piece(grid, occupied, actual_patches[wrong_i], locs[wrong_i])

        true_i = 1
        true_candidates = []
        for loc in _valid_locations(occupied, actual_patches[true_i]):
            d = _dist(base_center, _port_center(meta[true_i], canon_patches[true_i], loc, k, base_notch=False))
            if d > wrong_dist + 2:
                true_candidates.append(loc)
        if not true_candidates:
            continue
        locs[true_i] = random.choice(true_candidates)
        true_center = _port_center(meta[true_i], canon_patches[true_i], locs[true_i], k, base_notch=False)
        _paste_piece(grid, occupied, actual_patches[true_i], locs[true_i])

        rest = [i for i in range(1, len(actual_patches)) if i not in (true_i, wrong_i)]
        random.shuffle(rest)
        ok = True
        for i in rest:
            candidates = _valid_locations(occupied, actual_patches[i])
            if not candidates:
                ok = False
                break
            locs[i] = random.choice(candidates)
            _paste_piece(grid, occupied, actual_patches[i], locs[i])
        if not ok:
            continue

        assert locs[true_i] is not None and locs[wrong_i] is not None
        true_dist = _dist(base_center, true_center)
        same_color = int(canon_patches[true_i][canon_patches[true_i] != BG][0]) == int(canon_patches[wrong_i][canon_patches[wrong_i] != BG][0])
        same_size = len(meta[true_i]["profile"]) == len(meta[wrong_i]["profile"])
        wrong_shape = meta[true_i]["profile"] != meta[wrong_i]["profile"]
        if same_color and same_size and wrong_shape and wrong_dist < true_dist:
            return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    canon_patches, canon_expected, meta = _make_canonical_patches()
    if max(canon_expected.shape) > 30:
        return None
    k = random.choice(DIR_TO_CANON_K)
    expected = np.rot90(canon_expected, -k).copy()
    if max(expected.shape) > 30:
        return None
    inp = _scatter_with_decoys(canon_patches, meta, k)
    if inp is None or max(inp.shape) > 30:
        return None
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tab/notch assembly-chain input")
