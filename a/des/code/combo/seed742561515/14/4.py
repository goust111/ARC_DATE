import random
from collections import deque

import numpy as np


BG = 0
AZURE = 8
MAIN_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
CORNERS = ("tl", "tr", "br", "bl")
SIDES = ("top", "right", "bottom", "left")
ROTATION_TURNS = {
    "tl": 0,
    "tr": 1,
    "br": 2,
    "bl": 3,
}
GEN_RETRIES = 400
_RNG = random.Random()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[dict[str, np.ndarray | tuple[int, int, int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict[str, np.ndarray | tuple[int, int, int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []

            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            coords = np.array(pts, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            comps.append({"coords": coords, "bbox": (r0, c0, r1, c1)})

    return comps


def _hole_bbox(patch: np.ndarray) -> tuple[int, int, int, int]:
    hole = patch == BG
    assert hole.any(), "each object must contain a hole"
    assert not hole[0, :].any()
    assert not hole[-1, :].any()
    assert not hole[:, 0].any()
    assert not hole[:, -1].any()

    coords = np.argwhere(hole)
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    sub = hole[r0 : r1 + 1, c0 : c1 + 1]
    assert sub.all(), "hole must be a solid rectangle"
    assert int(hole.sum()) == int(sub.size), "object must have exactly one rectangular hole"

    filled = patch.copy()
    filled[r0 : r1 + 1, c0 : c1 + 1] = 1
    assert not (filled == BG).any(), "outer object must be solid outside the hole"
    return r0, c0, r1, c1


def _hole_corner(patch: np.ndarray) -> str:
    hr0, hc0, hr1, hc1 = _hole_bbox(patch)
    h, w = patch.shape
    top = hr0
    left = hc0
    bottom = h - 1 - hr1
    right = w - 1 - hc1

    assert top != bottom, "hole must be vertically off-center"
    assert left != right, "hole must be horizontally off-center"

    vertical = "t" if top < bottom else "b"
    horizontal = "l" if left < right else "r"
    return vertical + horizontal


def _rotate_cw(patch: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(patch, (-turns) % 4)


def _paste_nonzero(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    assert 0 <= r0 and 0 <= c0 and r0 + h <= dst.shape[0] and c0 + w <= dst.shape[1]
    region = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    assert not (region[mask] != BG).any(), "objects must stay disjoint after rotation"
    region[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each hollow rectangular ring in place according to the corner its hole leans toward.

    Hole toward:
    - top-left -> 0 degrees
    - top-right -> 90 degrees clockwise
    - bottom-right -> 180 degrees
    - bottom-left -> 270 degrees clockwise
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)

    for comp in _components(g != BG):
        r0, c0, r1, c1 = comp["bbox"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        corner = _hole_corner(patch)
        turns = ROTATION_TURNS[corner]
        rotated = _rotate_cw(patch, turns)
        rh, rw = rotated.shape
        h, w = patch.shape

        assert (h - rh) % 2 == 0 and (w - rw) % 2 == 0, "90-degree rotations need a well-defined center"
        nr0 = r0 + (h - rh) // 2
        nc0 = c0 + (w - rw) // 2
        _paste_nonzero(out, rotated, nr0, nc0)

    return out


def _sample_patch(rng: random.Random, main_color: int, corner: str, azure_side: str) -> dict[str, object] | None:
    for _ in range(200):
        hole_h = rng.randint(2, 4)
        hole_w = rng.randint(2, 4)

        near_v = rng.randint(1, 3)
        far_v = rng.randint(near_v + 1, 4)
        near_h = rng.randint(1, 3)
        far_h = rng.randint(near_h + 1, 4)

        top, bottom = (near_v, far_v) if corner[0] == "t" else (far_v, near_v)
        left, right = (near_h, far_h) if corner[1] == "l" else (far_h, near_h)
        h = top + hole_h + bottom
        w = left + hole_w + right

        if h > 10 or w > 10:
            continue
        if h % 2 != w % 2:
            continue

        patch = np.full((h, w), main_color, dtype=int)
        patch[top : top + hole_h, left : left + hole_w] = BG

        if azure_side == "top":
            patch[0, :] = AZURE
        elif azure_side == "right":
            patch[:, -1] = AZURE
        elif azure_side == "bottom":
            patch[-1, :] = AZURE
        else:
            patch[:, 0] = AZURE

        return {
            "patch": patch,
            "turns": ROTATION_TURNS[corner],
            "span": max(h, w),
            "corner": corner,
            "azure_side": azure_side,
        }

    return None


def _sample_corners(rng: random.Random, n_obj: int) -> list[str]:
    corners = list(CORNERS)
    rng.shuffle(corners)
    return corners[:n_obj]


def _layout_objects(rng: random.Random, specs: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray] | None:
    reserves = [int(spec["span"]) + 2 for spec in specs]
    max_reserve = max(reserves)
    total_area = sum(reserve * reserve for reserve in reserves)

    for _ in range(200):
        h = rng.randint(max(16, max_reserve), 30)
        w = rng.randint(max(16, max_reserve), 30)
        if h * w < int(total_area * 1.35):
            continue

        occupied = np.zeros((h, w), dtype=bool)
        placed: list[dict[str, object] | None] = [None] * len(specs)
        order = sorted(range(len(specs)), key=lambda idx: reserves[idx], reverse=True)

        ok = True
        for idx in order:
            reserve = reserves[idx]
            span = int(specs[idx]["span"])
            patch = np.asarray(specs[idx]["patch"], dtype=int)
            rotated = _rotate_cw(patch, int(specs[idx]["turns"]))
            ph, pw = patch.shape
            rh, rw = rotated.shape

            placed_here = False
            for _ in range(400):
                box_r0 = rng.randint(0, h - reserve)
                box_c0 = rng.randint(0, w - reserve)
                if occupied[box_r0 : box_r0 + reserve, box_c0 : box_c0 + reserve].any():
                    continue

                inner_r0 = box_r0 + 1
                inner_c0 = box_c0 + 1
                in_r0 = inner_r0 + (span - ph) // 2
                in_c0 = inner_c0 + (span - pw) // 2
                out_r0 = inner_r0 + (span - rh) // 2
                out_c0 = inner_c0 + (span - rw) // 2

                placed[idx] = {
                    "input_patch": patch,
                    "output_patch": rotated,
                    "input_pos": (in_r0, in_c0),
                    "output_pos": (out_r0, out_c0),
                }
                occupied[box_r0 : box_r0 + reserve, box_c0 : box_c0 + reserve] = True
                placed_here = True
                break

            if not placed_here:
                ok = False
                break

        if not ok:
            continue

        inp = np.zeros((h, w), dtype=int)
        out = np.zeros((h, w), dtype=int)
        try:
            for item in placed:
                assert item is not None
                in_r0, in_c0 = item["input_pos"]
                out_r0, out_c0 = item["output_pos"]
                _paste_nonzero(inp, np.asarray(item["input_patch"], dtype=int), int(in_r0), int(in_c0))
                _paste_nonzero(out, np.asarray(item["output_patch"], dtype=int), int(out_r0), int(out_c0))
        except AssertionError:
            continue

        if np.array_equal(inp, out):
            continue

        return inp, out

    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_obj = rng.randint(2, 4)
    colors = list(MAIN_COLORS)
    rng.shuffle(colors)
    corners = _sample_corners(rng, n_obj)

    specs: list[dict[str, object]] = []
    for i in range(n_obj):
        spec = _sample_patch(rng, colors[i], corners[i], rng.choice(SIDES))
        if spec is None:
            return None
        specs.append(spec)

    layout = _layout_objects(rng, specs)
    if layout is None:
        return None
    inp, expected = layout

    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
