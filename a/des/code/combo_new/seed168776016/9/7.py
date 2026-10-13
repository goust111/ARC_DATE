import random

import numpy as np


BG = 0
FRAME = 1
ANCHOR = 2
OBJECT_COLORS = [3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 300
CORNERS = ("tl", "tr", "bl", "br")


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        pts.append((r, c1))
    return pts


def _extract_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frame_mask = (grid == FRAME) | (grid == ANCHOR)
    frames: list[dict[str, object]] = []
    for comp in _components(frame_mask):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = set(_border_coords(r0, c0, r1, c1))
        pts = set(comp)
        if pts != border:
            continue
        anchor_pts = [(r, c) for r, c in comp if int(grid[r, c]) == ANCHOR]
        if len(anchor_pts) != 1:
            continue
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                "anchor": anchor_pts[0],
                "inner_area": int((r1 - r0 - 1) * (c1 - c0 - 1)),
            }
        )
    return frames


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    object_mask = (grid != BG) & (grid != FRAME) & (grid != ANCHOR)
    objects: list[dict[str, object]] = []
    for comp in _components(object_mask):
        r0, c0, r1, c1 = _bbox(comp)
        rel_coords = []
        colors = []
        for r, c in comp:
            rel_coords.append((int(r - r0), int(c - c0)))
            colors.append(int(grid[r, c]))
        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": (r1 - r0 + 1, c1 - c0 + 1),
                "count": len(comp),
                "rel_coords": rel_coords,
                "colors": colors,
                "abs_coords": comp,
            }
        )
    return objects


def _anchor_corner(frame: dict[str, object]) -> str:
    r0, c0, r1, c1 = frame["bbox"]
    ar, ac = frame["anchor"]
    inner_corners = [
        ("tl", (r0 + 1, c0 + 1)),
        ("tr", (r0 + 1, c1 - 1)),
        ("bl", (r1 - 1, c0 + 1)),
        ("br", (r1 - 1, c1 - 1)),
    ]
    best_name = None
    best_dist = None
    for name, (cr, cc) in inner_corners:
        dist = abs(ar - cr) + abs(ac - cc)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_name = name
    assert best_name is not None
    return best_name


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each outside object into the uniquely area-matched container and align it by the red border anchor.

    Input structure:
    - Black background.
    - Several Blue hollow rectangular frames, each with exactly one Red anchor pixel on its border and a Black interior.
    - Several 4-connected colored objects outside the frames, using colors other than Black, Blue, and Red.

    Rule:
    - Detect each frame, compute its interior area, and detect each outside object with its pixel count.
    - Pair objects to frames by exact equality: object pixel count = frame interior area, with a unique match for every item.
    - For each pair, find which interior corner is closest to the frame's Red anchor, then place the object's bounding-box corner of the same type onto that interior corner without rotation or reflection.
    - Erase the original outside objects and restore every Red anchor pixel back to Blue.

    Key insight: The matching clue is numerical area equality, while the red border pixel is a second instruction that chooses which corner controls the in-frame placement.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = _extract_frames(g)
    objects = _extract_objects(g)
    assert frames, "expected at least one frame"
    assert objects, "expected at least one movable object"

    frame_by_area: dict[int, dict[str, object]] = {}
    for frame in frames:
        area = int(frame["inner_area"])
        assert area not in frame_by_area, "frame areas must be unique"
        frame_by_area[area] = frame

    object_by_count: dict[int, dict[str, object]] = {}
    for obj in objects:
        count = int(obj["count"])
        assert count not in object_by_count, "object sizes must be unique"
        object_by_count[count] = obj

    assert set(frame_by_area) == set(object_by_count), "objects and frames must match by area"

    for frame in frames:
        ar, ac = frame["anchor"]
        out[ar, ac] = FRAME

    for obj in objects:
        for r, c in obj["abs_coords"]:
            out[r, c] = BG

    for area, frame in frame_by_area.items():
        obj = object_by_count[area]
        ir0, ic0, ir1, ic1 = frame["inner_bbox"]
        oh, ow = obj["shape"]
        corner = _anchor_corner(frame)

        if corner == "tl":
            base_r, base_c = ir0, ic0
        elif corner == "tr":
            base_r, base_c = ir0, ic1 - ow + 1
        elif corner == "bl":
            base_r, base_c = ir1 - oh + 1, ic0
        else:
            base_r, base_c = ir1 - oh + 1, ic1 - ow + 1

        for (rr, cc), color in zip(obj["rel_coords"], obj["colors"]):
            tr = int(base_r + rr)
            tc = int(base_c + cc)
            assert ir0 <= tr <= ir1 and ic0 <= tc <= ic1, "object must fit inside its frame"
            out[tr, tc] = int(color)

    return out


def _area_options() -> dict[int, list[tuple[int, int]]]:
    opts: dict[int, list[tuple[int, int]]] = {}
    for h in range(2, 6):
        for w in range(2, 6):
            opts.setdefault(h * w, []).append((h, w))
    return opts


AREA_OPTIONS = _area_options()


def _choose_anchor_local(frame_h: int, frame_w: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return random.choice([(0, 0), (0, 1), (1, 0)])
    if corner == "tr":
        return random.choice([(0, frame_w - 1), (0, frame_w - 2), (1, frame_w - 1)])
    if corner == "bl":
        return random.choice([(frame_h - 1, 0), (frame_h - 1, 1), (frame_h - 2, 0)])
    return random.choice(
        [(frame_h - 1, frame_w - 1), (frame_h - 1, frame_w - 2), (frame_h - 2, frame_w - 1)]
    )


def _can_place(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = occ.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _place_rect(occ: np.ndarray, h: int, w: int, pad: int = 1) -> tuple[int, int] | None:
    hh, ww = occ.shape
    spots = [(r, c) for r in range(hh - h + 1) for c in range(ww - w + 1)]
    random.shuffle(spots)
    for r0, c0 in spots:
        if _can_place(occ, r0, c0, h, w, pad=pad):
            occ[r0 : r0 + h, c0 : c0 + w] = True
            return r0, c0
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    n_items = random.choice([2, 3, 4])
    areas = random.sample(sorted(AREA_OPTIONS), n_items)
    colors = random.sample(OBJECT_COLORS, n_items)
    corners = list(CORNERS)
    random.shuffle(corners)

    specs = []
    for idx, area in enumerate(areas):
        inner_h, inner_w = random.choice(AREA_OPTIONS[area])
        specs.append(
            {
                "area": area,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "frame_h": inner_h + 2,
                "frame_w": inner_w + 2,
                "color": colors[idx],
                "corner": corners[idx],
            }
        )

    occ = np.zeros((h, w), dtype=bool)

    for spec in sorted(specs, key=lambda x: int(x["frame_h"]) * int(x["frame_w"]), reverse=True):
        pos = _place_rect(occ, int(spec["frame_h"]), int(spec["frame_w"]), pad=1)
        if pos is None:
            return None
        spec["frame_pos"] = pos

    for spec in sorted(specs, key=lambda x: int(x["inner_h"]) * int(x["inner_w"]), reverse=True):
        pos = _place_rect(occ, int(spec["inner_h"]), int(spec["inner_w"]), pad=1)
        if pos is None:
            return None
        spec["object_pos"] = pos

    grid = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)

    for spec in specs:
        fr, fc = spec["frame_pos"]
        fh, fw = int(spec["frame_h"]), int(spec["frame_w"])
        color = int(spec["color"])

        grid[fr, fc : fc + fw] = FRAME
        grid[fr + fh - 1, fc : fc + fw] = FRAME
        grid[fr : fr + fh, fc] = FRAME
        grid[fr : fr + fh, fc + fw - 1] = FRAME

        expected[fr, fc : fc + fw] = FRAME
        expected[fr + fh - 1, fc : fc + fw] = FRAME
        expected[fr : fr + fh, fc] = FRAME
        expected[fr : fr + fh, fc + fw - 1] = FRAME
        expected[fr + 1 : fr + fh - 1, fc + 1 : fc + fw - 1] = color

        ar, ac = _choose_anchor_local(fh, fw, str(spec["corner"]))
        grid[fr + ar, fc + ac] = ANCHOR

        or0, oc0 = spec["object_pos"]
        ih, iw = int(spec["inner_h"]), int(spec["inner_w"])
        grid[or0 : or0 + ih, oc0 : oc0 + iw] = color

    if len(_extract_frames(grid)) != n_items:
        return None
    if len(_extract_objects(grid)) != n_items:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
