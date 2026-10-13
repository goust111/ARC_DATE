import numpy as np
from collections import deque


BG = 8
RNG = np.random.default_rng()
MAX_ATTEMPTS = 400

SMALL_DIMS = [(2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (3, 3)]
MEDIUM_DIMS = [(4, 6), (6, 4), (5, 5), (4, 7), (7, 4), (5, 6), (6, 5), (5, 7), (7, 5), (6, 6)]
LARGE_DIMS = [(8, 9), (9, 8), (8, 10), (10, 8), (9, 9), (9, 10), (10, 9), (10, 10), (9, 11), (11, 9)]
FG_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            bbox = _bbox_from_coords(arr)
            r0, c0, r1, c1 = bbox
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox,
                    "bbox_area": (r1 - r0 + 1) * (c1 - c0 + 1),
                }
            )
    return comps


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _scaled_patch(patch: np.ndarray, tier: str) -> np.ndarray:
    if tier == "large":
        return np.array(patch[::2, ::2], dtype=int)
    if tier == "small":
        return np.repeat(np.repeat(patch, 2, axis=0), 2, axis=1)
    return np.array(patch, dtype=int)


def _centered_top_left(bbox: tuple[int, int, int, int], new_h: int, new_w: int) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    top = (r0 + r1 - (new_h - 1)) // 2
    left = (c0 + c1 - (new_w - 1)) // 2
    return int(top), int(left)


def _paste_clipped(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = dst.shape
    ph, pw = patch.shape
    sr0 = max(0, -top)
    sc0 = max(0, -left)
    dr0 = max(0, top)
    dc0 = max(0, left)
    rh = min(ph - sr0, h - dr0)
    cw = min(pw - sc0, w - dc0)
    if rh > 0 and cw > 0:
        dst[dr0 : dr0 + rh, dc0 : dc0 + cw] = patch[sr0 : sr0 + rh, sc0 : sc0 + cw]


def _tier_splits(areas: list[int]) -> tuple[int, int]:
    assert len(areas) >= 3
    scores = []
    for i in range(len(areas) - 1):
        a = areas[i]
        b = areas[i + 1]
        score = np.log((b + 1.0) / (a + 1.0)) + 0.01 * (b - a)
        scores.append((float(score), i))
    gap_ids = sorted(idx for _, idx in sorted(scores, reverse=True)[:2])
    assert gap_ids[0] + 1 < gap_ids[1] + 1
    return gap_ids[0] + 1, gap_ids[1] + 1


def _classify_objects(objs: list[dict]) -> dict[int, str]:
    ordered = sorted(range(len(objs)), key=lambda i: objs[i]["bbox_area"])
    areas = [objs[i]["bbox_area"] for i in ordered]
    s1, s2 = _tier_splits(areas)
    labels = {}
    for rank, obj_idx in enumerate(ordered):
        if rank < s1:
            labels[obj_idx] = "small"
        elif rank < s2:
            labels[obj_idx] = "medium"
        else:
            labels[obj_idx] = "large"
    return labels


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = BG if np.any(g == BG) else int(np.bincount(g.ravel()).argmax())
    objs = _find_components(g, bg)
    assert len(objs) >= 3

    labels = _classify_objects(objs)
    out = np.full(g.shape, bg, dtype=int)
    for idx, obj in enumerate(objs):
        patch = _crop(g, obj["bbox"])
        scaled = _scaled_patch(patch, labels[idx])
        top, left = _centered_top_left(obj["bbox"], scaled.shape[0], scaled.shape[1])
        _paste_clipped(out, scaled, top, left)
    return out


def _make_mask(h: int, w: int) -> np.ndarray:
    kind = RNG.choice(["solid", "frame", "plus", "tee", "ell"])
    mask = np.zeros((h, w), dtype=bool)
    if kind == "solid":
        mask[:] = True
    elif kind == "frame" and h >= 3 and w >= 3:
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
    elif kind == "plus":
        mask[:, w // 2] = True
        mask[h // 2, :] = True
        if h >= 5 and RNG.random() < 0.5:
            mask[:, min(w - 1, w // 2 + 1)] = True
        if w >= 5 and RNG.random() < 0.5:
            mask[min(h - 1, h // 2 + 1), :] = True
    elif kind == "tee":
        mask[0, :] = True
        stem = w // 2
        mask[:, stem] = True
        if h >= 5 and RNG.random() < 0.5:
            mask[:, min(w - 1, stem + 1)] = True
    else:
        mask[:, 0] = True
        mask[-1, :] = True
        if RNG.random() < 0.5:
            mask[:, min(w - 1, 1)] = True
        if RNG.random() < 0.5:
            mask[max(0, h - 2), :] = True
    return mask


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _place_objects(specs: list[dict], h: int, w: int) -> list[dict] | None:
    placed = []
    src_rects = []
    dst_rects = []
    order = sorted(specs, key=lambda x: max(x["h"] * x["w"], x["out_h"] * x["out_w"]), reverse=True)
    for spec in order:
        for _ in range(250):
            if spec["clip"]:
                side = RNG.choice(["top", "bottom", "left", "right"])
                if side == "top":
                    r0 = 0
                    c0 = int(RNG.integers(0, w - spec["w"] + 1))
                elif side == "bottom":
                    r0 = h - spec["h"]
                    c0 = int(RNG.integers(0, w - spec["w"] + 1))
                elif side == "left":
                    r0 = int(RNG.integers(0, h - spec["h"] + 1))
                    c0 = 0
                else:
                    r0 = int(RNG.integers(0, h - spec["h"] + 1))
                    c0 = w - spec["w"]
            else:
                r0 = int(RNG.integers(0, h - spec["h"] + 1))
                c0 = int(RNG.integers(0, w - spec["w"] + 1))

            src = (r0, c0, r0 + spec["h"] - 1, c0 + spec["w"] - 1)
            bbox = src
            top, left = _centered_top_left(bbox, spec["out_h"], spec["out_w"])
            dr0 = max(0, top)
            dc0 = max(0, left)
            dr1 = min(h - 1, top + spec["out_h"] - 1)
            dc1 = min(w - 1, left + spec["out_w"] - 1)
            dst = (dr0, dc0, dr1, dc1)
            if any(_rects_overlap(src, other, margin=1) for other in src_rects):
                continue
            if any(_rects_overlap(dst, other, margin=1) for other in dst_rects):
                continue
            spec = dict(spec)
            spec["r0"] = r0
            spec["c0"] = c0
            spec["src"] = src
            spec["dst"] = dst
            placed.append(spec)
            src_rects.append(src)
            dst_rects.append(dst)
            break
        else:
            return None
    return placed


def _tier_counts_from_labels(labels: dict[int, str]) -> tuple[int, int, int]:
    small = sum(v == "small" for v in labels.values())
    medium = sum(v == "medium" for v in labels.values())
    large = sum(v == "large" for v in labels.values())
    return small, medium, large


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(24, 33))
    w = int(RNG.integers(24, 33))
    counts = {
        "small": int(RNG.integers(1, 3)),
        "medium": int(RNG.integers(1, 3)),
        "large": int(RNG.integers(1, 3)),
    }
    total = sum(counts.values())
    if total > len(FG_COLORS):
        return None

    clip_small = bool(RNG.random() < 0.5)
    colors = list(RNG.choice(FG_COLORS, size=total, replace=False))
    specs = []
    color_idx = 0
    for tier, dims_pool in (("small", SMALL_DIMS), ("medium", MEDIUM_DIMS), ("large", LARGE_DIMS)):
        for i in range(counts[tier]):
            hh, ww = dims_pool[int(RNG.integers(0, len(dims_pool)))]
            out_h, out_w = _scaled_patch(np.zeros((hh, ww), dtype=int), tier).shape
            specs.append(
                {
                    "tier": tier,
                    "color": int(colors[color_idx]),
                    "h": int(hh),
                    "w": int(ww),
                    "mask": _make_mask(int(hh), int(ww)),
                    "out_h": int(out_h),
                    "out_w": int(out_w),
                    "clip": clip_small and tier == "small" and i == 0,
                }
            )
            color_idx += 1

    placed = _place_objects(specs, h, w)
    if placed is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    for spec in placed:
        patch = np.full((spec["h"], spec["w"]), BG, dtype=int)
        patch[spec["mask"]] = spec["color"]
        r0 = spec["r0"]
        c0 = spec["c0"]
        grid[r0 : r0 + spec["h"], c0 : c0 + spec["w"]] = np.where(patch != BG, patch, grid[r0 : r0 + spec["h"], c0 : c0 + spec["w"]])

    objs = _find_components(grid, BG)
    if len(objs) != total:
        return None
    labels = _classify_objects(objs)
    if _tier_counts_from_labels(labels) != (counts["small"], counts["medium"], counts["large"]):
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
