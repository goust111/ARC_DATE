import random

import numpy as np


BG = 0
FRAME_COLOR = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ["top", "bottom", "left", "right"]
KEY_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(pts)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _tight_mask_from_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _canonical_rot(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    best = None
    for k in range(4):
        rot = np.rot90(mask, k)
        key = ((int(rot.shape[0]), int(rot.shape[1])), rot.tobytes())
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _distance_to_bbox_edge(h: int, w: int) -> np.ndarray:
    rr = np.arange(h)[:, None]
    cc = np.arange(w)[None, :]
    return np.minimum(np.minimum(rr, h - 1 - rr), np.minimum(cc, w - 1 - cc))


def _parse_frame_component(component_mask: np.ndarray, subgrid: np.ndarray) -> dict | None:
    h, w = component_mask.shape
    if h < 7 or w < 7:
        return None
    if not (
        component_mask[0, 0]
        and component_mask[0, -1]
        and component_mask[-1, 0]
        and component_mask[-1, -1]
    ):
        return None
    if np.any(subgrid[~component_mask] != BG):
        return None

    d = _distance_to_bbox_edge(h, w)
    valids = []
    max_t = min(h, w) // 2 - 1
    for t in range(2, max_t + 1):
        border_mask = d < t
        inner_boundary = d == (t - 1)
        interior = d >= t
        notch_zone = d < (t - 1)

        if not interior.any():
            continue
        if np.any(component_mask & ~border_mask):
            continue
        if np.any(component_mask & interior):
            continue
        if not np.all(component_mask[inner_boundary]):
            continue
        if np.any(subgrid[interior] != BG):
            continue

        notch_mask = notch_zone & (~component_mask)
        if not notch_mask.any():
            continue
        if np.any(subgrid[notch_mask] != BG):
            continue

        if len(_components(notch_mask)) != 1:
            continue

        pts = [(int(r), int(c)) for r, c in np.argwhere(notch_mask)]
        if any((r in (0, h - 1)) and (c in (0, w - 1)) for r, c in pts):
            continue

        touches = set()
        for r, c in pts:
            if r == 0:
                touches.add("top")
            if r == h - 1:
                touches.add("bottom")
            if c == 0:
                touches.add("left")
            if c == w - 1:
                touches.add("right")
        if len(touches) != 1:
            continue

        valids.append(
            {
                "thickness": t,
                "side": next(iter(touches)),
                "border_mask": border_mask,
                "notch_mask": notch_mask,
            }
        )

    if len(valids) != 1:
        return None
    return valids[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored key to the congruent notch in a grey door frame,
    fill the notch, recolor the whole frame to the key color, and erase the keys.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    frames = []
    frame_pixels = np.zeros((h, w), dtype=bool)
    for color in sorted(int(c) for c in np.unique(g) if int(c) != BG):
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            comp_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in comp:
                comp_mask[r - r0, c - c0] = True
            info = _parse_frame_component(comp_mask, g[r0 : r1 + 1, c0 : c1 + 1])
            if info is None:
                continue
            for r, c in comp:
                frame_pixels[r, c] = True
            notch_coords = [(int(r), int(c)) for r, c in np.argwhere(info["notch_mask"])]
            notch_shape = _tight_mask_from_coords(notch_coords)
            frames.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "border_mask": info["border_mask"],
                    "shape_key": _canonical_rot(notch_shape),
                }
            )

    assert frames, "expected at least one valid door frame"

    keys = {}
    key_components = []
    for comp in _components((g != BG) & (~frame_pixels)):
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each key must be monochrome"
        color = next(iter(colors))
        key_mask = _tight_mask_from_coords(comp)
        key_sig = _canonical_rot(key_mask)
        assert key_sig not in keys, "key shapes must be unique up to rotation"
        keys[key_sig] = color
        key_components.append(comp)

    assert len(keys) == len(frames), "frame count and key count must match"

    out = g.copy()
    for comp in key_components:
        for r, c in comp:
            out[r, c] = BG

    for frame in frames:
        color = keys[frame["shape_key"]]
        r0, c0, r1, c1 = frame["bbox"]
        sub = out[r0 : r1 + 1, c0 : c1 + 1]
        sub[frame["border_mask"]] = color
        sub[~frame["border_mask"]] = BG

    return out


def _random_connected_shape(height: int, width: int, side: str) -> np.ndarray | None:
    boundary = []
    if side == "top":
        boundary = [(0, c) for c in range(width)]
    elif side == "bottom":
        boundary = [(height - 1, c) for c in range(width)]
    elif side == "left":
        boundary = [(r, 0) for r in range(height)]
    else:
        boundary = [(r, width - 1) for r in range(height)]

    total = height * width
    area_lo = 2 if total == 2 else 3
    area_hi = min(6, total)
    if area_lo > area_hi:
        return None

    for _ in range(120):
        target = random.randint(area_lo, area_hi)
        start = random.choice(boundary)
        cells = {start}
        frontier = set()
        for dr, dc in DIR4:
            nr = start[0] + dr
            nc = start[1] + dc
            if 0 <= nr < height and 0 <= nc < width:
                frontier.add((nr, nc))
        while len(cells) < target and frontier:
            pick = random.choice(tuple(frontier))
            frontier.remove(pick)
            if pick in cells:
                continue
            cells.add(pick)
            for dr, dc in DIR4:
                nr = pick[0] + dr
                nc = pick[1] + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if len(cells) != target:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _tight_mask(mask)

        if side == "top" and not mask[0, :].any():
            continue
        if side == "bottom" and not mask[-1, :].any():
            continue
        if side == "left" and not mask[:, 0].any():
            continue
        if side == "right" and not mask[:, -1].any():
            continue
        return mask
    return None


def _build_frame_spec(used_shapes: set[tuple[tuple[int, int], bytes]], key_color: int) -> dict | None:
    for _ in range(200):
        side = random.choice(SIDES)
        thickness = random.choice([3, 4])
        span = random.randint(2, 5)
        if side in ("top", "bottom"):
            notch = _random_connected_shape(thickness - 1, span, side)
        else:
            notch = _random_connected_shape(span, thickness - 1, side)
        if notch is None:
            continue

        shape_key = _canonical_rot(notch)
        if shape_key in used_shapes:
            continue

        nh, nw = notch.shape
        min_h = max(2 * thickness + 3, 9)
        min_w = max(2 * thickness + 3, 9)
        if side in ("left", "right"):
            min_h = max(min_h, 2 * thickness + nh)
        else:
            min_w = max(min_w, 2 * thickness + nw)

        max_h = max(min_h, 14)
        max_w = max(min_w, 14)
        if min_h > max_h or min_w > max_w:
            continue

        outer_h = random.randint(min_h, max_h)
        outer_w = random.randint(min_w, max_w)
        d = _distance_to_bbox_edge(outer_h, outer_w)
        border_mask = d < thickness
        patch = np.zeros((outer_h, outer_w), dtype=int)
        patch[border_mask] = FRAME_COLOR

        if side in ("top", "bottom"):
            c0_lo = thickness
            c0_hi = outer_w - thickness - nw
            if c0_lo > c0_hi:
                continue
            c0 = random.randint(c0_lo, c0_hi)
            if side == "top":
                sub = patch[:nh, c0 : c0 + nw]
            else:
                sub = patch[outer_h - nh :, c0 : c0 + nw]
            sub[notch] = BG
        else:
            r0_lo = thickness
            r0_hi = outer_h - thickness - nh
            if r0_lo > r0_hi:
                continue
            r0 = random.randint(r0_lo, r0_hi)
            if side == "left":
                sub = patch[r0 : r0 + nh, :nw]
            else:
                sub = patch[r0 : r0 + nh, outer_w - nw :]
            sub[notch] = BG

        if len(_components(patch == FRAME_COLOR)) != 1:
            continue

        parsed = _parse_frame_component(patch == FRAME_COLOR, patch)
        if parsed is None:
            continue

        key_rot = random.randrange(4)
        key_mask = np.rot90(notch, key_rot)
        key_patch = np.zeros(key_mask.shape, dtype=int)
        key_patch[key_mask] = key_color

        used_shapes.add(shape_key)
        return {
            "shape_key": shape_key,
            "key_color": key_color,
            "frame_patch": patch,
            "frame_mask": patch != BG,
            "frame_border_mask": border_mask,
            "key_patch": key_patch,
            "key_mask": key_mask,
        }
    return None


def _pack_objects(objects: list[dict]) -> tuple[np.ndarray, dict[str, tuple[int, int]]] | None:
    max_h = max(int(obj["patch"].shape[0]) for obj in objects)
    max_w = max(int(obj["patch"].shape[1]) for obj in objects)
    total_area = sum(int(obj["patch"].shape[0] * obj["patch"].shape[1]) for obj in objects)

    for _ in range(120):
        base = int(np.ceil(np.sqrt(total_area * 2.1)))
        low_h = max(24, max_h + 2, base - 2)
        low_w = max(24, max_w + 2, base - 2)
        high_h = max(low_h, min(40, low_h + 8))
        high_w = max(low_w, min(40, low_w + 8))
        if low_h > high_h or low_w > high_w:
            continue

        h = random.randint(low_h, high_h)
        w = random.randint(low_w, high_w)
        reserved = np.zeros((h, w), dtype=bool)
        placements: dict[str, tuple[int, int]] = {}
        ok = True

        for obj in sorted(objects, key=lambda item: item["patch"].shape[0] * item["patch"].shape[1], reverse=True):
            ph, pw = obj["patch"].shape
            placed = False
            for _ in range(500):
                r0 = random.randint(0, h - ph)
                c0 = random.randint(0, w - pw)
                rr0 = max(0, r0 - 1)
                rr1 = min(h, r0 + ph + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(w, c0 + pw + 1)
                if reserved[rr0:rr1, cc0:cc1].any():
                    continue
                reserved[rr0:rr1, cc0:cc1] = True
                placements[obj["id"]] = (r0, c0)
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        grid = np.zeros((h, w), dtype=int)
        for obj in objects:
            r0, c0 = placements[obj["id"]]
            ph, pw = obj["patch"].shape
            sub = grid[r0 : r0 + ph, c0 : c0 + pw]
            sub[obj["mask"]] = obj["patch"][obj["mask"]]
        return grid, placements
    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        frame_count = random.randint(2, 4)
        colors = random.sample(KEY_COLORS, frame_count)
        used_shapes: set[tuple[tuple[int, int], bytes]] = set()
        specs = []
        ok = True
        for color in colors:
            spec = _build_frame_spec(used_shapes, color)
            if spec is None:
                ok = False
                break
            specs.append(spec)
        if not ok:
            continue

        objects = []
        for idx, spec in enumerate(specs):
            objects.append(
                {
                    "id": f"frame_{idx}",
                    "patch": spec["frame_patch"],
                    "mask": spec["frame_mask"],
                }
            )
            objects.append(
                {
                    "id": f"key_{idx}",
                    "patch": spec["key_patch"],
                    "mask": spec["key_patch"] != BG,
                }
            )

        packed = _pack_objects(objects)
        if packed is None:
            continue
        grid, placements = packed

        expected = np.zeros_like(grid)
        for idx, spec in enumerate(specs):
            r0, c0 = placements[f"frame_{idx}"]
            ph, pw = spec["frame_patch"].shape
            sub = expected[r0 : r0 + ph, c0 : c0 + pw]
            sub[spec["frame_border_mask"]] = spec["key_color"]

        if not np.array_equal(main(grid), expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
