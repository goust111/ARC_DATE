from __future__ import annotations

from collections import Counter
import random

import numpy as np
from scipy.ndimage import label


STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        lab, n = label(grid == int(color), structure=STRUCT4)
        for comp_id in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(comp_id))
            if coords.size == 0:
                continue
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return comps


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return mask[:0, :0].copy()
    return _crop_bbox(mask.astype(bool), _bbox_coords(coords)).astype(bool)


def _is_frame_component(mask: np.ndarray) -> bool:
    if mask.shape[0] < 5 or mask.shape[1] < 5:
        return False
    if not mask[0].all() or not mask[-1].all() or not mask[:, 0].all() or not mask[:, -1].all():
        return False
    return bool((~mask[1:-1, 1:-1]).any())


def _placement_for_side(frame_shape: tuple[int, int], key_shape: tuple[int, int], side: str) -> tuple[int, int] | None:
    fh, fw = frame_shape
    kh, kw = key_shape
    inner_h = fh - 2
    inner_w = fw - 2
    if side == "U":
        if inner_w != kw or inner_h <= kh:
            return None
        return 1, 1
    if side == "D":
        if inner_w != kw or inner_h <= kh:
            return None
        return fh - 1 - kh, 1
    if side == "L":
        if inner_h != kh or inner_w <= kw:
            return None
        return 1, 1
    if side == "R":
        if inner_h != kh or inner_w <= kw:
            return None
        return 1, fw - 1 - kw
    raise ValueError(side)


def _match_frame_side(frame_mask: np.ndarray, key_mask: np.ndarray, side: str) -> tuple[int, int] | None:
    placement = _placement_for_side(frame_mask.shape, key_mask.shape, side)
    if placement is None:
        return None
    r0, c0 = placement
    kh, kw = key_mask.shape
    strip = frame_mask[r0 : r0 + kh, c0 : c0 + kw]
    if strip.shape != key_mask.shape:
        return None
    if not np.array_equal(key_mask, ~strip):
        return None
    return int(r0), int(c0)


def _enumerate_candidates(frames: list[dict], keys: list[dict]) -> list[list[dict]]:
    all_candidates: list[list[dict]] = []
    for key_idx, key in enumerate(keys):
        found: dict[tuple, dict] = {}
        for rot in range(4):
            rot_mask = _trim_mask(np.rot90(key["mask"], rot))
            for frame_idx, frame in enumerate(frames):
                for side in ("U", "D", "L", "R"):
                    rel = _match_frame_side(frame["mask"], rot_mask, side)
                    if rel is None:
                        continue
                    abs_r = int(frame["bbox"][0] + rel[0])
                    abs_c = int(frame["bbox"][1] + rel[1])
                    sig = (frame_idx, side, abs_r, abs_c, rot_mask.shape, rot_mask.tobytes())
                    if sig in found:
                        continue
                    found[sig] = {
                        "key": int(key_idx),
                        "frame": int(frame_idx),
                        "side": side,
                        "rot": int(rot),
                        "abs_r": int(abs_r),
                        "abs_c": int(abs_c),
                        "mask": rot_mask.copy(),
                    }
        cands = sorted(found.values(), key=lambda cand: (cand["frame"], cand["side"], cand["abs_r"], cand["abs_c"], cand["rot"]))
        all_candidates.append(cands)
    return all_candidates


def _solve_assignment(candidates: list[list[dict]], frame_count: int) -> dict[int, dict] | None:
    if any(len(cands) == 0 for cands in candidates):
        return None
    order = sorted(range(len(candidates)), key=lambda idx: (len(candidates[idx]), idx))
    used_frames: set[int] = set()
    assignment: dict[int, dict] = {}

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return len(used_frames) == frame_count
        key_idx = order[pos]
        for cand in candidates[key_idx]:
            frame_idx = int(cand["frame"])
            if frame_idx in used_frames:
                continue
            used_frames.add(frame_idx)
            assignment[key_idx] = cand
            if dfs(pos + 1):
                return True
            del assignment[key_idx]
            used_frames.remove(frame_idx)
        return False

    if not dfs(0):
        return None
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each scattered key into the unique hollow frame whose toothed inner wall it complements.

    Input structure:
    - bg is the dominant color.
    - Several monochrome hollow rectangular lock frames sit on the canvas; each frame keeps a full outer border and has teeth protruding inward from exactly one inner wall.
    - The same number of monochrome key sprites lie outside the frames.  Keys are solid connected masks whose bounding boxes share one common rectangle up to 90-degree rotation.
    - Key colors may equal the wrong frame colors, so color alone is unreliable.

    Rule:
    - Separate frame components from key components.
    - For each key, test all 0/90/180/270 rotations against every frame side and find the unique placement where the rotated key exactly fills the complement of that frame's toothed interior strip.
    - Erase the key from its original location and paste it into that lock interior with its own color.
    - Leave frames, teeth, and all other cells unchanged.

    Key insight:
    - Matching depends on geometric profile complement: within the docking strip, frame teeth and rotated key cells must partition the strip into a perfect solid rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _find_components(g, bg)

    frames: list[dict] = []
    keys: list[dict] = []
    for comp in comps:
        bbox = comp["bbox"]
        crop = _crop_bbox(g, bbox)
        mask = crop == int(comp["color"])
        item = {
            "color": int(comp["color"]),
            "bbox": bbox,
            "coords": comp["coords"],
            "mask": mask.astype(bool),
        }
        if _is_frame_component(mask):
            frames.append(item)
        else:
            keys.append(item)

    assert frames and len(frames) == len(keys), "expected equal counts of frames and keys"

    candidates = _enumerate_candidates(frames, keys)
    assignment = _solve_assignment(candidates, len(frames))
    assert assignment is not None, "expected a unique docking assignment"

    out = g.copy()
    for key in keys:
        out[key["coords"][:, 0], key["coords"][:, 1]] = int(bg)

    for key_idx in sorted(assignment):
        cand = assignment[key_idx]
        color = int(keys[key_idx]["color"])
        patch = np.full(cand["mask"].shape, int(bg), dtype=int)
        patch[cand["mask"]] = color
        _paste(out, patch, cand["abs_r"], cand["abs_c"], transparent=bg)

    return out


def _make_profile(thickness: int, span: int, rng: random.Random) -> tuple[int, ...]:
    for _ in range(200):
        vals = [rng.randrange(thickness)]
        for _ in range(span - 1):
            step = rng.choice((-1, 0, 1))
            vals.append(max(0, min(thickness - 1, vals[-1] + step)))
        vals[rng.randrange(span)] = 0
        vals[rng.randrange(span)] = rng.randrange(1, thickness)
        if max(vals) == 0 or len(set(vals)) < 2:
            continue
        if sum(vals[i] != vals[i + 1] for i in range(span - 1)) == 0:
            continue
        return tuple(int(v) for v in vals)
    raise RuntimeError("failed to sample a tooth profile")


def _build_lock_and_key(
    side: str,
    profile: tuple[int, ...],
    thickness: int,
    extra: int,
    frame_color: int,
    bg: int,
) -> tuple[np.ndarray, np.ndarray]:
    span = len(profile)
    if side in ("U", "D"):
        inner_h = int(thickness + extra)
        inner_w = int(span)
        frame = np.full((inner_h + 2, inner_w + 2), int(bg), dtype=int)
        frame[0, :] = frame[-1, :] = frame[:, 0] = frame[:, -1] = int(frame_color)
        strip = np.zeros((thickness, span), dtype=bool)
        for c, depth in enumerate(profile):
            if side == "U":
                strip[: int(depth), int(c)] = True
            else:
                strip[thickness - int(depth) :, int(c)] = True
        if side == "U":
            frame[1 : 1 + thickness, 1 : 1 + span][strip] = int(frame_color)
        else:
            frame[1 + inner_h - thickness : 1 + inner_h, 1 : 1 + span][strip] = int(frame_color)
        key_mask = ~strip
        return frame, key_mask

    inner_h = int(span)
    inner_w = int(thickness + extra)
    frame = np.full((inner_h + 2, inner_w + 2), int(bg), dtype=int)
    frame[0, :] = frame[-1, :] = frame[:, 0] = frame[:, -1] = int(frame_color)
    strip = np.zeros((span, thickness), dtype=bool)
    for r, depth in enumerate(profile):
        if side == "L":
            strip[int(r), : int(depth)] = True
        else:
            strip[int(r), thickness - int(depth) :] = True
    if side == "L":
        frame[1 : 1 + span, 1 : 1 + thickness][strip] = int(frame_color)
    else:
        frame[1 : 1 + span, 1 + inner_w - thickness : 1 + inner_w][strip] = int(frame_color)
    key_mask = ~strip
    return frame, key_mask


def _pick_sides(count: int, rng: random.Random) -> list[str]:
    vertical = rng.choice(["U", "D"])
    horizontal = rng.choice(["L", "R"])
    sides = [vertical, horizontal]
    if count == 3:
        extras = ["U", "D", "L", "R"]
        rng.shuffle(extras)
        for side in extras:
            if side not in sides or rng.random() < 0.5:
                sides.append(side)
                break
    rng.shuffle(sides)
    return sides


def _derangement(values: list[int], rng: random.Random) -> list[int]:
    if len(values) == 2:
        return [int(values[1]), int(values[0])]
    while True:
        perm = list(values)
        rng.shuffle(perm)
        if all(int(a) != int(b) for a, b in zip(values, perm)):
            return [int(v) for v in perm]


def _random_place_bbox(
    occ: np.ndarray,
    h: int,
    w: int,
    rng: random.Random,
    *,
    padding: int = 1,
    border: int = 1,
) -> tuple[int, int] | None:
    H, W = occ.shape
    if h + 2 * border > H or w + 2 * border > W:
        return None
    candidates = [
        (r0, c0)
        for r0 in range(border, H - border - h + 1)
        for c0 in range(border, W - border - w + 1)
    ]
    rng.shuffle(candidates)
    for r0, c0 in candidates:
        rr0 = max(0, r0 - padding)
        cc0 = max(0, c0 - padding)
        rr1 = min(H, r0 + h + padding)
        cc1 = min(W, c0 + w + padding)
        if occ[rr0:rr1, cc0:cc1].any():
            continue
        occ[r0 : r0 + h, c0 : c0 + w] = True
        return int(r0), int(c0)
    return None


def _sample_shape_family(
    count: int,
    thickness: int,
    span: int,
    sides: list[str],
    frame_colors: list[int],
    key_colors: list[int],
    bg: int,
    rng: random.Random,
) -> list[dict] | None:
    for _ in range(300):
        family: list[dict] = []
        for idx in range(count):
            profile = _make_profile(thickness, span, rng)
            extra = rng.randint(1, 3)
            frame_patch, key_mask = _build_lock_and_key(
                sides[idx],
                profile,
                thickness,
                extra,
                frame_colors[idx],
                bg,
            )
            family.append(
                {
                    "side": sides[idx],
                    "profile": profile,
                    "frame_color": int(frame_colors[idx]),
                    "key_color": int(key_colors[idx]),
                    "frame_patch": frame_patch,
                    "frame_mask": frame_patch == int(frame_colors[idx]),
                    "key_mask": key_mask.astype(bool),
                }
            )

        pseudo_frames = [
            {"bbox": (0, 0, item["frame_mask"].shape[0] - 1, item["frame_mask"].shape[1] - 1), "mask": item["frame_mask"]}
            for item in family
        ]
        pseudo_keys = [{"mask": item["key_mask"]} for item in family]
        cands = _enumerate_candidates(pseudo_frames, pseudo_keys)
        assignment = _solve_assignment(cands, count)
        if assignment is None:
            continue
        if any(len(lst) != 1 for lst in cands):
            continue
        return family
    return None


def _draw_output_key(frame_bbox: tuple[int, int, int, int], frame_mask: np.ndarray, key_mask: np.ndarray, side: str) -> tuple[int, int]:
    rel = _match_frame_side(frame_mask, key_mask, side)
    assert rel is not None
    return int(frame_bbox[0] + rel[0]), int(frame_bbox[1] + rel[1])


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = 0
    count = rng.choice([2, 3])
    thickness = rng.choice([3, 4])
    span = rng.randint(thickness + 2, 7)
    sides = _pick_sides(count, rng)

    palette = list(range(1, 10))
    rng.shuffle(palette)
    frame_colors = [int(v) for v in palette[:count]]
    key_colors = _derangement(frame_colors, rng)

    family = _sample_shape_family(count, thickness, span, sides, frame_colors, key_colors, bg, rng)
    if family is None:
        return None

    frame_dims = [item["frame_patch"].shape for item in family]
    key_dims = [item["key_mask"].shape for item in family]
    max_h = max(max(h for h, _ in frame_dims), max(h for h, _ in key_dims))
    max_w = max(max(w for _, w in frame_dims), max(w for _, w in key_dims))

    H = rng.randint(max(22, max_h + 4), 30)
    W = rng.randint(max(22, max_w + 4), 30)
    inp = np.full((H, W), int(bg), dtype=int)
    out = np.full((H, W), int(bg), dtype=int)
    occ = np.zeros((H, W), dtype=bool)

    rng.shuffle(family)
    for item in family:
        fh, fw = item["frame_patch"].shape
        frame_pos = _random_place_bbox(occ, fh, fw, rng, padding=2, border=1)
        if frame_pos is None:
            return None
        fr, fc = frame_pos
        frame_bbox = (fr, fc, fr + fh - 1, fc + fw - 1)
        item["frame_bbox"] = frame_bbox
        _paste(inp, item["frame_patch"], fr, fc, transparent=bg)
        _paste(out, item["frame_patch"], fr, fc, transparent=bg)

        abs_r, abs_c = _draw_output_key(frame_bbox, item["frame_mask"], item["key_mask"], item["side"])
        item["dock_abs"] = (abs_r, abs_c)
        out_patch = np.full(item["key_mask"].shape, int(bg), dtype=int)
        out_patch[item["key_mask"]] = int(item["key_color"])
        _paste(out, out_patch, abs_r, abs_c, transparent=bg)

    rotations = [rng.randrange(4) for _ in family]
    if all(rot == 0 for rot in rotations):
        rotations[rng.randrange(len(rotations))] = rng.choice([1, 2, 3])

    for item, rot in zip(family, rotations):
        key_mask = _trim_mask(np.rot90(item["key_mask"], int(rot)))
        patch = np.full(key_mask.shape, int(bg), dtype=int)
        patch[key_mask] = int(item["key_color"])
        kh, kw = patch.shape
        key_pos = _random_place_bbox(occ, kh, kw, rng, padding=2, border=1)
        if key_pos is None:
            return None
        kr, kc = key_pos
        item["input_key_mask"] = key_mask
        item["input_key_bbox"] = (kr, kc, kr + kh - 1, kc + kw - 1)
        _paste(inp, patch, kr, kc, transparent=bg)

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None

    comps = _find_components(inp, bg)
    frames = []
    keys = []
    for comp in comps:
        crop = _crop_bbox(inp, comp["bbox"])
        mask = crop == int(comp["color"])
        item = {"bbox": comp["bbox"], "mask": mask.astype(bool)}
        if _is_frame_component(mask):
            frames.append(item)
        else:
            keys.append(item)
    cands = _enumerate_candidates(frames, keys)
    if any(len(lst) != 1 for lst in cands):
        return None
    if _solve_assignment(cands, len(frames)) is None:
        return None

    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 400 attempts")
