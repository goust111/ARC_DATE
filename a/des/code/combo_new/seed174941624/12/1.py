from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
MAX_GENERATION_RETRIES = 300
FRAME_COLORS = [1, 2, 5, 6, 7, 8, 9]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _quadrant_bounds(frame: dict, name: str) -> tuple[int, int, int, int]:
    q = int(frame["q"])
    r0 = int(frame["r0"]) + 1
    c0 = int(frame["c0"]) + 1
    if name == "tl":
        return r0, c0, r0 + q, c0 + q
    if name == "tr":
        return r0, c0 + q, r0 + q, c0 + 2 * q
    if name == "bl":
        return r0 + q, c0, r0 + 2 * q, c0 + q
    return r0 + q, c0 + q, r0 + 2 * q, c0 + 2 * q


def _extract_quadrant(grid: np.ndarray, frame: dict, name: str) -> np.ndarray:
    r0, c0, r1, c1 = _quadrant_bounds(frame, name)
    return grid[r0:r1, c0:c1].copy()


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        for comp in _components(grid == int(color)):
            r0, c0, r1, c1 = _bbox(comp)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h != w or h < 6 or h % 2 != 0:
                continue
            q = (h - 2) // 2
            if q < 2:
                continue
            comp_set = {tuple(map(int, pt)) for pt in comp.tolist()}
            if any(r not in (r0, r1) and c not in (c0, c1) for r, c in comp_set):
                continue
            border: list[tuple[int, int]] = []
            for cc in range(c0, c1 + 1):
                border.append((r0, cc))
                border.append((r1, cc))
            for rr in range(r0 + 1, r1):
                border.append((rr, c0))
                border.append((rr, c1))
            border = list(dict.fromkeys(border))
            missing = [pos for pos in border if pos not in comp_set]
            if len(missing) != 1:
                continue
            gap_r, gap_c = missing[0]
            if gap_c in (c0, c1) or gap_r not in (r0, r1):
                continue
            if gap_r == r0:
                missing_quadrant = "tl" if gap_c <= c0 + q else "tr"
            else:
                missing_quadrant = "bl" if gap_c <= c0 + q else "br"
            frames.append(
                {
                    "color": int(color),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "q": int(q),
                    "gap": (int(gap_r), int(gap_c)),
                    "missing": missing_quadrant,
                }
            )
    frames.sort(key=lambda d: (int(d["r0"]), int(d["c0"])))
    return frames


def _candidate_patches(grid: np.ndarray, frames: list[dict], bg: int) -> list[dict]:
    mask = grid != int(bg)
    for frame in frames:
        mask[int(frame["r0"]) : int(frame["r1"]) + 1, int(frame["c0"]) : int(frame["c1"]) + 1] = False
    patches: list[dict] = []
    for comp in _components(mask):
        r0, c0, r1, c1 = _bbox(comp)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        patches.append({"bbox": (int(r0), int(c0), int(r1), int(c1)), "patch": patch})
    patches.sort(key=lambda d: d["bbox"])
    return patches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each damaged framed motif by selecting and rotating the matching outside patch.

    Input structure:
    - The background is a dominant color.
    - One or more 1-pixel-thick square frames contain a 2x2 arrangement of same-sized colored quadrants, with exactly one quadrant replaced by background.
    - Each damaged frame has a single missing frame pixel on its top or bottom edge, and several detached square candidate patches of the same size sit elsewhere on the grid.

    Rule:
    - Detect every frame and use the gap position to identify whether the missing quadrant is top-left, top-right, bottom-left, or bottom-right.
    - Read the quadrant diagonally opposite the missing one, then find the detached patch whose pixels exactly match that opposite quadrant.
    - Rotate that patch by 180 degrees, paste it into the missing quadrant, repair the frame gap with the frame color, and erase every detached candidate patch to background.

    Key insight:
    - The border gap tells you which quadrant is absent, and the opposite quadrant identifies the correct patch because the hidden quadrant is its 180-degree rotation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames = _find_frames(g, bg)
    assert frames, "expected at least one valid frame"

    opposite = {"tl": "br", "tr": "bl", "bl": "tr", "br": "tl"}
    patches = _candidate_patches(g, frames, bg)
    assert patches, "expected detached candidate patches"

    out = g.copy()
    for patch in patches:
        r0, c0, r1, c1 = patch["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)

    used: set[int] = set()
    for frame in frames:
        diagonal_name = opposite[str(frame["missing"])]
        target = _extract_quadrant(g, frame, diagonal_name)
        matches = [
            idx
            for idx, patch in enumerate(patches)
            if idx not in used
            and patch["patch"].shape == target.shape
            and np.array_equal(patch["patch"], target)
        ]
        assert len(matches) == 1, "every frame must have exactly one matching patch"
        fill = np.rot90(patches[matches[0]]["patch"], 2)
        used.add(int(matches[0]))
        r0, c0, r1, c1 = _quadrant_bounds(frame, str(frame["missing"]))
        out[r0:r1, c0:c1] = fill
        gap_r, gap_c = frame["gap"]
        out[int(gap_r), int(gap_c)] = int(frame["color"])

    return out


def _random_patch(
    rng: np.random.Generator, q: int, palette: list[int], blocked: set[bytes]
) -> np.ndarray:
    max_colors = min(4, len(palette))
    for _ in range(200):
        n_colors = int(rng.integers(2, max_colors + 1))
        colors = [int(v) for v in rng.choice(np.asarray(palette), size=n_colors, replace=False)]
        patch = rng.choice(np.asarray(colors, dtype=int), size=(q, q)).astype(int)
        for color in colors:
            rr = int(rng.integers(0, q))
            cc = int(rng.integers(0, q))
            patch[rr, cc] = int(color)
        if len(set(int(v) for v in patch.flat)) < 2:
            continue
        if any(np.any(patch == int(BG)) for _ in [0]):
            continue
        rot_keys = {np.rot90(patch, k).tobytes() for k in range(4)}
        if len(rot_keys) < 4:
            continue
        if rot_keys & blocked:
            continue
        return patch
    raise RuntimeError("failed to sample a distinctive patch")


def _gap_column(rng: np.random.Generator, q: int, side: str) -> int:
    if side == "left":
        return int(rng.integers(1, q + 1))
    return int(rng.integers(q + 1, 2 * q + 1))


def _build_frame(
    rng: np.random.Generator,
    q: int,
    frame_color: int,
    base_patch: np.ndarray,
    missing: str,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    quadrants = {
        "tl": base_patch,
        "tr": np.rot90(base_patch, -1),
        "br": np.rot90(base_patch, 2),
        "bl": np.rot90(base_patch, 1),
    }
    size = 2 * q + 2
    frame = np.full((size, size), int(BG), dtype=int)
    frame[0, :] = int(frame_color)
    frame[-1, :] = int(frame_color)
    frame[:, 0] = int(frame_color)
    frame[:, -1] = int(frame_color)

    for name, patch in quadrants.items():
        if name == missing:
            continue
        if name == "tl":
            frame[1 : 1 + q, 1 : 1 + q] = patch
        elif name == "tr":
            frame[1 : 1 + q, 1 + q : 1 + 2 * q] = patch
        elif name == "bl":
            frame[1 + q : 1 + 2 * q, 1 : 1 + q] = patch
        else:
            frame[1 + q : 1 + 2 * q, 1 + q : 1 + 2 * q] = patch

    if missing == "tl":
        gap = (0, _gap_column(rng, q, "left"))
    elif missing == "tr":
        gap = (0, _gap_column(rng, q, "right"))
    elif missing == "bl":
        gap = (size - 1, _gap_column(rng, q, "left"))
    else:
        gap = (size - 1, _gap_column(rng, q, "right"))
    frame[gap] = int(BG)

    target_name = {"tl": "br", "tr": "bl", "bl": "tr", "br": "tl"}[missing]
    return frame, quadrants[target_name].copy(), (int(gap[0]), int(gap[1]))


def _can_place(occ: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> bool:
    H, W = occ.shape
    if r < 0 or c < 0 or r + h > H or c + w > W:
        return False
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(H, r + h + pad)
    c1 = min(W, c + w + pad)
    return not bool(occ[r0:r1, c0:c1].any())


def _place_objects(
    rng: np.random.Generator, objects: list[np.ndarray], h: int, w: int
) -> list[tuple[int, int]] | None:
    occ = np.zeros((h, w), dtype=bool)
    positions: list[tuple[int, int]] = []
    for obj in objects:
        oh, ow = map(int, obj.shape)
        placed = False
        for _ in range(400):
            r = int(rng.integers(0, h - oh + 1))
            c = int(rng.integers(0, w - ow + 1))
            if not _can_place(occ, r, c, oh, ow, pad=1):
                continue
            occ[r : r + oh, c : c + ow] = True
            positions.append((int(r), int(c)))
            placed = True
            break
        if not placed:
            return None
    return positions


def _expected_output(
    grid: np.ndarray,
    frame_specs: list[dict],
    patch_specs: list[dict],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for patch in patch_specs:
        r0, c0 = patch["pos"]
        q = int(patch["patch"].shape[0])
        out[r0 : r0 + q, c0 : c0 + q] = int(BG)
    for frame in frame_specs:
        r0, c0 = frame["pos"]
        q = int(frame["q"])
        missing = str(frame["missing"])
        fill = np.rot90(frame["target"], 2)
        if missing == "tl":
            out[r0 + 1 : r0 + 1 + q, c0 + 1 : c0 + 1 + q] = fill
        elif missing == "tr":
            out[r0 + 1 : r0 + 1 + q, c0 + 1 + q : c0 + 1 + 2 * q] = fill
        elif missing == "bl":
            out[r0 + 1 + q : r0 + 1 + 2 * q, c0 + 1 : c0 + 1 + q] = fill
        else:
            out[r0 + 1 + q : r0 + 1 + 2 * q, c0 + 1 + q : c0 + 1 + 2 * q] = fill
        gap_r, gap_c = frame["gap_abs"]
        out[int(gap_r), int(gap_c)] = int(frame["color"])
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    q = int(rng.integers(2, 5))
    max_frames = 2 if q == 4 else 3
    n_frames = int(rng.integers(1, max_frames + 1))
    n_distractors = int(rng.integers(1, n_frames + 3))

    frame_color = int(rng.choice(np.asarray(FRAME_COLORS, dtype=int)))
    palette = [c for c in range(10) if c not in (BG, frame_color)]
    size = 2 * q + 2

    blocked_exact: set[bytes] = set()
    blocked_rotations: set[bytes] = set()
    frame_objects: list[np.ndarray] = []
    frame_specs: list[dict] = []
    directions = ["tl", "tr", "bl", "br"]

    for _ in range(n_frames):
        base = _random_patch(rng, q, palette, blocked_rotations)
        missing = str(rng.choice(np.asarray(directions)))
        frame, target, gap_rel = _build_frame(rng, q, frame_color, base, missing)
        key = target.tobytes()
        if key in blocked_exact:
            return None
        blocked_exact.add(key)
        blocked_rotations.update(np.rot90(target, k).tobytes() for k in range(4))
        frame_objects.append(frame)
        frame_specs.append(
            {
                "grid": frame,
                "q": int(q),
                "missing": missing,
                "target": target,
                "gap_rel": gap_rel,
                "color": int(frame_color),
            }
        )

    patch_arrays: list[np.ndarray] = [spec["target"].copy() for spec in frame_specs]
    for _ in range(n_distractors):
        patch_arrays.append(_random_patch(rng, q, palette, blocked_rotations))
    rng.shuffle(patch_arrays)

    objects = frame_objects + patch_arrays
    total_area = n_frames * size * size + len(patch_arrays) * q * q
    min_side = max(size + 2, int(np.ceil(np.sqrt(total_area * 2.4))))
    min_side = min(min_side, 30)

    for _ in range(40):
        h = int(rng.integers(min_side, 31))
        w = int(rng.integers(min_side, 31))
        positions = _place_objects(rng, objects, h, w)
        if positions is None:
            continue

        grid = np.full((h, w), int(BG), dtype=int)
        for obj, (r0, c0) in zip(objects, positions):
            oh, ow = map(int, obj.shape)
            grid[r0 : r0 + oh, c0 : c0 + ow] = obj

        for idx, spec in enumerate(frame_specs):
            r0, c0 = positions[idx]
            gap_r, gap_c = spec["gap_rel"]
            spec["pos"] = (int(r0), int(c0))
            spec["gap_abs"] = (int(r0 + gap_r), int(c0 + gap_c))

        patch_specs: list[dict] = []
        for idx, patch in enumerate(patch_arrays, start=n_frames):
            r0, c0 = positions[idx]
            patch_specs.append({"patch": patch, "pos": (int(r0), int(c0))})

        expected = _expected_output(grid, frame_specs, patch_specs)
        solved = main(grid)
        if np.array_equal(solved, expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_RETRIES):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
