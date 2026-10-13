from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
OPENING_TO_CW = {"up": 0, "right": 1, "down": 2, "left": 3}
COLOR_POOL = tuple(range(1, 10))
OPENINGS = tuple(OPENING_TO_CW)
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            out.append(pts)
    return out


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _component_mask(pts: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(pts)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        mask[r - r0, c - c0] = True
    return mask


def _rotate_mask(mask: np.ndarray, cw_steps: int) -> np.ndarray:
    return np.rot90(mask.astype(bool), -int(cw_steps))


def _u_frame_info(pts: list[tuple[int, int]]) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(pts)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    sub = np.zeros((h, w), dtype=bool)
    for r, c in pts:
        sub[r - r0, c - c0] = True

    if np.any(sub[1:-1, 1:-1]):
        return None
    if not (sub[0, 0] and sub[0, -1] and sub[-1, 0] and sub[-1, -1]):
        return None

    sides = {
        "up": sub[0, 1:-1],
        "down": sub[-1, 1:-1],
        "left": sub[1:-1, 0],
        "right": sub[1:-1, -1],
    }

    missing: list[str] = []
    for name, strip in sides.items():
        if np.all(~strip):
            missing.append(name)
        elif not np.all(strip):
            return None

    if len(missing) != 1:
        return None

    return {
        "opening": missing[0],
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "interior_bbox": (int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1)),
    }


def _center_start(space: int, size: int, bias: int) -> int:
    diff = int(space - size)
    assert diff >= 0
    base = diff // 2
    if diff % 2 == 0:
        return base
    return base + 1 if bias > 0 else base


def _place_fragment(
    out: np.ndarray,
    color: int,
    frame_info: dict[str, object],
    fragment_mask: np.ndarray,
) -> None:
    opening = str(frame_info["opening"])
    ir0, ic0, ir1, ic1 = frame_info["interior_bbox"]  # type: ignore[misc]
    ih = int(ir1 - ir0 + 1)
    iw = int(ic1 - ic0 + 1)
    ph, pw = fragment_mask.shape
    assert ph <= ih and pw <= iw

    row_bias = 1 if opening == "up" else (-1 if opening == "down" else 0)
    col_bias = 1 if opening == "left" else (-1 if opening == "right" else 0)
    dr = _center_start(ih, ph, row_bias)
    dc = _center_start(iw, pw, col_bias)

    window = out[ir0 + dr : ir0 + dr + ph, ic0 + dc : ic0 + dc + pw]
    assert np.all(window[fragment_mask] == BG)
    window[fragment_mask] = int(color)


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        comps = _components(grid == color)
        assert len(comps) == 2, f"color {color} must form one frame and one fragment"

        frame_candidates: list[tuple[int, dict[str, object]]] = []
        for i, pts in enumerate(comps):
            info = _u_frame_info(pts)
            if info is not None:
                frame_candidates.append((i, info))
        assert len(frame_candidates) == 1, f"color {color} must have exactly one U-frame"

        frame_idx, frame_info = frame_candidates[0]
        frag_idx = 1 - frame_idx
        frag_pts = comps[frag_idx]
        frag_mask = _component_mask(frag_pts)

        fr0, fc0, fr1, fc1 = frame_info["bbox"]  # type: ignore[misc]
        sr0, sc0, sr1, sc1 = _bbox(frag_pts)
        assert sr1 < fr0 or sr0 > fr1 or sc1 < fc0 or sc0 > fc1, "fragment must be outside its frame"

        objects.append(
            {
                "color": int(color),
                "frame_info": frame_info,
                "fragment_pts": frag_pts,
                "fragment_mask": frag_mask,
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """Match each fragment to the same-colored U-frame, rotate from canonical up to the frame opening, and place it centered inside."""
    g = np.asarray(grid, dtype=int)
    objects = _extract_objects(g)

    out = g.copy()
    for obj in objects:
        for r, c in obj["fragment_pts"]:  # type: ignore[index]
            out[r, c] = BG

    for obj in objects:
        color = int(obj["color"])
        frame_info = obj["frame_info"]
        fragment_mask = np.asarray(obj["fragment_mask"], dtype=bool)
        opening = str(frame_info["opening"])  # type: ignore[index]
        rotated = _rotate_mask(fragment_mask, OPENING_TO_CW[opening])
        _place_fragment(out, color, frame_info, rotated)

    return out


def _make_up_fragment(rng: random.Random) -> np.ndarray:
    wing = rng.randint(1, 2)
    stem_len = rng.randint(1, 3)
    stem_width = rng.randint(1, 2 if wing == 2 else 1)

    width = 2 * wing + 1
    height = wing + 1 + stem_len
    center = width // 2
    mask = np.zeros((height, width), dtype=bool)

    for r in range(wing + 1):
        half = r
        mask[r, center - half : center + half + 1] = True
    left = center - (stem_width - 1) // 2
    right = left + stem_width
    for r in range(wing + 1, height):
        mask[r, left:right] = True

    if wing == 2 and rng.random() < 0.5:
        mask[wing, center - 2] = False
        mask[wing, center + 2] = True

    return mask


def _make_frame_patch(interior_h: int, interior_w: int, opening: str, color: int) -> np.ndarray:
    patch = np.zeros((interior_h + 2, interior_w + 2), dtype=int)
    patch[:, 0] = color
    patch[:, -1] = color
    patch[0, :] = color
    patch[-1, :] = color
    if opening == "up":
        patch[0, 1:-1] = BG
    elif opening == "down":
        patch[-1, 1:-1] = BG
    elif opening == "left":
        patch[1:-1, 0] = BG
    elif opening == "right":
        patch[1:-1, -1] = BG
    else:
        raise ValueError(opening)
    return patch


def _transparent_patch(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, int(color), BG)


def _can_place(occupied: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> bool:
    H, W = occupied.shape
    if r < 1 or c < 1 or r + h > H - 1 or c + w > W - 1:
        return False
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(H, r + h + pad)
    c1 = min(W, c + w + pad)
    return not np.any(occupied[r0:r1, c0:c1])


def _paint(grid: np.ndarray, patch: np.ndarray, r: int, c: int) -> None:
    h, w = patch.shape
    sub = grid[r : r + h, c : c + w]
    mask = patch != BG
    sub[mask] = patch[mask]


def _choose_position(
    occupied: np.ndarray,
    h: int,
    w: int,
    rng: random.Random,
    pad: int = 1,
) -> tuple[int, int] | None:
    H, W = occupied.shape
    candidates: list[tuple[int, int]] = []
    for r in range(1, H - h):
        for c in range(1, W - w):
            if _can_place(occupied, r, c, h, w, pad=pad):
                candidates.append((r, c))
    if not candidates:
        return None
    return rng.choice(candidates)


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    object_count = rng.randint(2, 4)
    colors = rng.sample(COLOR_POOL, object_count)

    specs: list[dict[str, object]] = []
    reserve_areas = 0
    max_h = 0
    max_w = 0

    for color in colors:
        opening = rng.choice(OPENINGS)
        fragment_up = _make_up_fragment(rng)
        fragment_rot = _rotate_mask(fragment_up, OPENING_TO_CW[opening])
        fh, fw = fragment_rot.shape

        slack_h = rng.randint(0, 3)
        slack_w = rng.randint(0, 3)
        if opening in ("up", "down") and rng.random() < 0.5:
            slack_h = 1 + 2 * rng.randint(0, 1)
        if opening in ("left", "right") and rng.random() < 0.5:
            slack_w = 1 + 2 * rng.randint(0, 1)

        interior_h = fh + slack_h
        interior_w = fw + slack_w
        frame_patch = _make_frame_patch(interior_h, interior_w, opening, color)
        fragment_patch = _transparent_patch(fragment_up, color)

        frame_h, frame_w = frame_patch.shape
        frag_h, frag_w = fragment_patch.shape
        reserve_areas += frame_h * frame_w + frag_h * frag_w
        max_h = max(max_h, frame_h, frag_h)
        max_w = max(max_w, frame_w, frag_w)

        specs.append(
            {
                "color": int(color),
                "opening": opening,
                "fragment_up": fragment_up,
                "frame_patch": frame_patch,
                "fragment_patch": fragment_patch,
                "frame_size": (frame_h, frame_w),
                "fragment_size": (frag_h, frag_w),
            }
        )

    side = int(np.ceil(np.sqrt(reserve_areas * 1.8)))
    base_h = max(max_h + 4, side + rng.randint(0, 4))
    base_w = max(max_w + 4, side + rng.randint(0, 4))
    if rng.random() < 0.5:
        base_h += rng.randint(0, 5)
    else:
        base_w += rng.randint(0, 5)

    for extra in range(6):
        H = base_h + rng.randint(0, 2) + extra
        W = base_w + rng.randint(0, 2) + extra
        grid = np.zeros((H, W), dtype=int)
        occupied = np.zeros((H, W), dtype=bool)

        placements: list[dict[str, object]] = []
        order = sorted(
            range(object_count),
            key=lambda i: specs[i]["frame_size"][0] * specs[i]["frame_size"][1],  # type: ignore[index]
            reverse=True,
        )

        ok = True
        for idx in order:
            spec = specs[idx]
            frame_h, frame_w = spec["frame_size"]  # type: ignore[misc]
            pos = _choose_position(occupied, frame_h, frame_w, rng, pad=1)
            if pos is None:
                ok = False
                break
            fr, fc = pos
            _paint(grid, spec["frame_patch"], fr, fc)  # type: ignore[arg-type]
            occupied[fr : fr + frame_h, fc : fc + frame_w] = True
            placements.append({"idx": idx, "frame_pos": (fr, fc)})

        if not ok:
            continue

        rng.shuffle(placements)
        for item in placements:
            spec = specs[item["idx"]]  # type: ignore[index]
            frag_h, frag_w = spec["fragment_size"]  # type: ignore[misc]
            pos = _choose_position(occupied, frag_h, frag_w, rng, pad=1)
            if pos is None:
                ok = False
                break
            sr, sc = pos
            _paint(grid, spec["fragment_patch"], sr, sc)  # type: ignore[arg-type]
            occupied[sr : sr + frag_h, sc : sc + frag_w] = True
            item["fragment_pos"] = (sr, sc)

        if not ok:
            continue

        expected = grid.copy()
        for item in placements:
            spec = specs[item["idx"]]  # type: ignore[index]
            sr, sc = item["fragment_pos"]  # type: ignore[index]
            frag_h, frag_w = spec["fragment_size"]  # type: ignore[misc]
            expected[sr : sr + frag_h, sc : sc + frag_w][spec["fragment_patch"] != BG] = BG  # type: ignore[index]

            fr, fc = item["frame_pos"]  # type: ignore[index]
            frame_h, frame_w = spec["frame_size"]  # type: ignore[misc]
            frame_info = {
                "opening": spec["opening"],
                "bbox": (fr, fc, fr + frame_h - 1, fc + frame_w - 1),
                "interior_bbox": (fr + 1, fc + 1, fr + frame_h - 2, fc + frame_w - 2),
            }
            rotated = _rotate_mask(spec["fragment_up"], OPENING_TO_CW[spec["opening"]])  # type: ignore[index]
            _place_fragment(expected, spec["color"], frame_info, rotated)  # type: ignore[arg-type]

        try:
            if np.array_equal(main(grid), expected):
                return grid, expected
        except Exception:
            continue

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            grid, _expected = case
            return grid
    raise RuntimeError("failed to generate a valid input")
