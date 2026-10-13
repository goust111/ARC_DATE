import random
from collections import Counter, deque

import numpy as np


BG = 8
NON_BG_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
GEN_RETRIES = 400


def _bbox(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _mono_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != bg):
        for coords in _components_mask(grid == color):
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return comps


def _shape_mask_from_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return int(mask.shape[0]), int(mask.shape[1]), mask.astype(np.uint8).tobytes()


def _frame_holes(grid: np.ndarray, color: int, bbox: tuple[int, int, int, int], bg: int):
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    # A valid frame must occupy the full outer perimeter of its bbox.
    if np.any(grid[r0, c0 : c1 + 1] != color):
        return None
    if np.any(grid[r1, c0 : c1 + 1] != color):
        return None
    if np.any(grid[r0 : r1 + 1, c0] != color):
        return None
    if np.any(grid[r0 : r1 + 1, c1] != color):
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    holes = []
    for comp in _components_mask(sub == bg):
        if any(rr == 0 or rr == sub.shape[0] - 1 or cc == 0 or cc == sub.shape[1] - 1 for rr, cc in comp):
            continue
        global_coords = [(r0 + rr, c0 + cc) for rr, cc in comp]
        holes.append(global_coords)
    return holes if holes else None


def _draw_nested_ring(out: np.ndarray, top: int, left: int, side: int, color: int, bg: int):
    h, w = out.shape
    for layer in range((side + 1) // 2):
        val = color if layer % 2 == 0 else bg
        r0, c0 = top + layer, left + layer
        r1, c1 = top + side - 1 - layer, left + side - 1 - layer
        if not (0 <= r0 < h and 0 <= c0 < w and 0 <= r1 < h and 0 <= c1 < w):
            continue
        out[r0, c0 : c1 + 1] = val
        out[r1, c0 : c1 + 1] = val
        out[r0 : r1 + 1, c0] = val
        out[r0 : r1 + 1, c1] = val


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = Counter(g.flatten().tolist()).most_common(1)[0][0]
    out = g.copy()

    comps = _mono_components(g, bg)
    frames = []
    patterns = []
    for comp in comps:
        holes = _frame_holes(g, int(comp["color"]), comp["bbox"], bg)
        if holes is None:
            patterns.append(comp)
        else:
            frames.append(
                {
                    "color": int(comp["color"]),
                    "bbox": comp["bbox"],
                    "coords": comp["coords"],
                    "holes": holes,
                }
            )

    patterns_by_shape: dict[tuple[int, int, bytes], list[dict]] = {}
    for comp in patterns:
        mask = _shape_mask_from_coords(comp["coords"])
        patterns_by_shape.setdefault(_shape_key(mask), []).append(comp)

    for frame in sorted(frames, key=lambda d: d["bbox"]):
        filled = 0
        for hole in frame["holes"]:
            key = _shape_key(_shape_mask_from_coords(hole))
            matches = patterns_by_shape.get(key, [])
            assert matches, "each hole must have a matching outside pattern"
            pat = matches.pop()
            fill_color = int(pat["color"])
            for r, c in pat["coords"]:
                out[r, c] = bg
            for r, c in hole:
                out[r, c] = fill_color
            filled += 1

        r0, c0, r1, c1 = frame["bbox"]
        side = 2 * filled - 1
        _draw_nested_ring(out, r1 + 1, c1 + 1, side, frame["color"], bg)

    return out


def _can_place(occ: np.ndarray, top: int, left: int, patch: np.ndarray, pad: int = 1) -> bool:
    h, w = occ.shape
    ph, pw = patch.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + ph + pad)
    c1 = min(w, left + pw + pad)
    return not occ[r0:r1, c0:c1].any()


def _stamp(grid: np.ndarray, top: int, left: int, patch: np.ndarray, occ: np.ndarray):
    ph, pw = patch.shape
    mask = patch != BG
    view = grid[top : top + ph, left : left + pw]
    view[mask] = patch[mask]
    occ[top : top + ph, left : left + pw] |= mask


def _reserve(occ: np.ndarray, top: int, left: int, height: int, width: int):
    occ[top : top + height, left : left + width] = True


def _build_frame_patch(rng: random.Random, n_holes: int, color: int, used_shapes: set):
    for _ in range(200):
        if n_holes == 1:
            hh = rng.randint(1, 4)
            ww = rng.randint(1, 4)
            shapes = [(hh, ww)]
            if any(s in used_shapes for s in shapes):
                continue
            patch = np.full((hh + 2, ww + 2), BG, dtype=int)
            patch[0, :] = color
            patch[-1, :] = color
            patch[:, 0] = color
            patch[:, -1] = color
            holes = [((1, 1), (hh, ww))]
        elif n_holes == 2:
            if rng.random() < 0.5:
                hh = rng.randint(1, 4)
                w1, w2 = rng.sample([1, 2, 3, 4], 2)
                shapes = [(hh, w1), (hh, w2)]
                if any(s in used_shapes for s in shapes):
                    continue
                patch = np.full((hh + 2, w1 + w2 + 3), BG, dtype=int)
                patch[0, :] = color
                patch[-1, :] = color
                patch[:, 0] = color
                patch[:, -1] = color
                patch[1 : 1 + hh, 1 + w1] = color
                holes = [((1, 1), (hh, w1)), ((1, w1 + 2), (hh, w2))]
            else:
                ww = rng.randint(1, 4)
                h1, h2 = rng.sample([1, 2, 3, 4], 2)
                shapes = [(h1, ww), (h2, ww)]
                if any(s in used_shapes for s in shapes):
                    continue
                patch = np.full((h1 + h2 + 3, ww + 2), BG, dtype=int)
                patch[0, :] = color
                patch[-1, :] = color
                patch[:, 0] = color
                patch[:, -1] = color
                patch[1 + h1, 1 : 1 + ww] = color
                holes = [((1, 1), (h1, ww)), ((h1 + 2, 1), (h2, ww))]
        else:
            case = rng.choice(["vhl", "vhr", "hvt", "hvb"])
            if case in {"vhl", "vhr"}:
                w_split = rng.randint(1, 4)
                w_full = rng.randint(1, 4)
                h1 = rng.randint(1, 3)
                h2 = rng.randint(1, 3)
                h_full = h1 + h2 + 1
                if case == "vhl":
                    shapes = [(h1, w_split), (h2, w_split), (h_full, w_full)]
                    if len(set(shapes)) != 3 or any(s in used_shapes for s in shapes):
                        continue
                    patch = np.full((h_full + 2, w_split + w_full + 3), BG, dtype=int)
                    patch[0, :] = color
                    patch[-1, :] = color
                    patch[:, 0] = color
                    patch[:, -1] = color
                    patch[1 : 1 + h_full, 1 + w_split] = color
                    patch[1 + h1, 1 : 1 + w_split] = color
                    holes = [
                        ((1, 1), (h1, w_split)),
                        ((h1 + 2, 1), (h2, w_split)),
                        ((1, w_split + 2), (h_full, w_full)),
                    ]
                else:
                    shapes = [(h_full, w_full), (h1, w_split), (h2, w_split)]
                    if len(set(shapes)) != 3 or any(s in used_shapes for s in shapes):
                        continue
                    patch = np.full((h_full + 2, w_full + w_split + 3), BG, dtype=int)
                    patch[0, :] = color
                    patch[-1, :] = color
                    patch[:, 0] = color
                    patch[:, -1] = color
                    patch[1 : 1 + h_full, 1 + w_full] = color
                    patch[1 + h1, w_full + 2 : w_full + 2 + w_split] = color
                    holes = [
                        ((1, 1), (h_full, w_full)),
                        ((1, w_full + 2), (h1, w_split)),
                        ((h1 + 2, w_full + 2), (h2, w_split)),
                    ]
            else:
                h_split = rng.randint(1, 4)
                h_full = rng.randint(1, 4)
                w1 = rng.randint(1, 3)
                w2 = rng.randint(1, 3)
                w_full = w1 + w2 + 1
                if case == "hvt":
                    shapes = [(h_split, w1), (h_split, w2), (h_full, w_full)]
                    if len(set(shapes)) != 3 or any(s in used_shapes for s in shapes):
                        continue
                    patch = np.full((h_split + h_full + 3, w_full + 2), BG, dtype=int)
                    patch[0, :] = color
                    patch[-1, :] = color
                    patch[:, 0] = color
                    patch[:, -1] = color
                    patch[1 + h_split, 1 : 1 + w_full] = color
                    patch[1 : 1 + h_split, 1 + w1] = color
                    holes = [
                        ((1, 1), (h_split, w1)),
                        ((1, w1 + 2), (h_split, w2)),
                        ((h_split + 2, 1), (h_full, w_full)),
                    ]
                else:
                    shapes = [(h_full, w_full), (h_split, w1), (h_split, w2)]
                    if len(set(shapes)) != 3 or any(s in used_shapes for s in shapes):
                        continue
                    patch = np.full((h_full + h_split + 3, w_full + 2), BG, dtype=int)
                    patch[0, :] = color
                    patch[-1, :] = color
                    patch[:, 0] = color
                    patch[:, -1] = color
                    patch[1 + h_full, 1 : 1 + w_full] = color
                    patch[h_full + 2 : h_full + 2 + h_split, 1 + w1] = color
                    holes = [
                        ((1, 1), (h_full, w_full)),
                        ((h_full + 2, 1), (h_split, w1)),
                        ((h_full + 2, w1 + 2), (h_split, w2)),
                    ]

        hole_specs = []
        for (top, left), (hh, ww) in holes:
            coords = [(top + r, left + c) for r in range(hh) for c in range(ww)]
            hole_specs.append({"dims": (hh, ww), "coords": coords})
        return {"patch": patch, "holes": hole_specs, "used": {h["dims"] for h in hole_specs}}
    return None


def _place_patterns(
    rng: random.Random,
    grid: np.ndarray,
    occ: np.ndarray,
    hole_records: list[dict],
):
    h, w = grid.shape
    pattern_records = []
    rng.shuffle(hole_records)
    for hole in hole_records:
        hh, ww = hole["dims"]
        color = rng.choice([c for c in NON_BG_COLORS if c != hole["frame_color"]])
        patch = np.full((hh, ww), color, dtype=int)
        placed = False
        positions = [(r, c) for r in range(h - hh + 1) for c in range(w - ww + 1)]
        rng.shuffle(positions)
        for top, left in positions:
            if not _can_place(occ, top, left, patch, pad=1):
                continue
            _stamp(grid, top, left, patch, occ)
            pattern_records.append(
                {
                    "coords": [(top + r, left + c) for r in range(hh) for c in range(ww)],
                    "color": color,
                    "dims": (hh, ww),
                }
            )
            hole["fill_color"] = color
            placed = True
            break
        if not placed:
            return None
    return pattern_records


def _expected_from_plan(inp: np.ndarray, frames: list[dict], patterns: list[dict]) -> np.ndarray:
    out = inp.copy()
    pattern_by_dims = {}
    for pat in patterns:
        pattern_by_dims[pat["dims"]] = pat

    for frame in frames:
        for hole in frame["holes"]:
            pat = pattern_by_dims[hole["dims"]]
            for r, c in pat["coords"]:
                out[r, c] = BG
            for r, c in hole["global_coords"]:
                out[r, c] = pat["color"]
        side = 2 * len(frame["holes"]) - 1
        r0, c0, r1, c1 = frame["bbox"]
        _draw_nested_ring(out, r1 + 1, c1 + 1, side, frame["color"], BG)
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        h = rng.randint(24, 34)
        w = rng.randint(24, 34)
        grid = np.full((h, w), BG, dtype=int)
        occ = np.zeros((h, w), dtype=bool)

        n_frames = rng.randint(1, 3)
        frame_colors = rng.sample(NON_BG_COLORS, n_frames)
        used_shapes = set()
        frames = []
        all_holes = []

        frame_specs = []
        ok = True
        for i in range(n_frames):
            n_holes = rng.choice([1, 2, 3])
            spec = _build_frame_patch(rng, n_holes, frame_colors[i], used_shapes)
            if spec is None:
                ok = False
                break
            used_shapes |= spec["used"]
            frame_specs.append((frame_colors[i], spec))
        if not ok:
            continue

        order = list(range(n_frames))
        rng.shuffle(order)
        for idx in order:
            color, spec = frame_specs[idx]
            patch = spec["patch"]
            ph, pw = patch.shape
            side = 2 * len(spec["holes"]) - 1
            reserve_h = ph + side
            reserve_w = pw + side
            positions = [(r, c) for r in range(h - reserve_h + 1) for c in range(w - reserve_w + 1)]
            rng.shuffle(positions)
            placed = False
            for top, left in positions:
                reserve_patch = np.full((reserve_h, reserve_w), color, dtype=int)
                if not _can_place(occ, top, left, reserve_patch, pad=1):
                    continue
                _stamp(grid, top, left, patch, occ)
                _reserve(occ, top + ph, left + pw, side, side)
                bbox = (top, left, top + ph - 1, left + pw - 1)
                frame_rec = {"color": color, "bbox": bbox, "holes": []}
                for hole in spec["holes"]:
                    global_coords = [(top + r, left + c) for r, c in hole["coords"]]
                    hole_rec = {
                        "dims": hole["dims"],
                        "global_coords": global_coords,
                        "frame_color": color,
                    }
                    frame_rec["holes"].append(hole_rec)
                    all_holes.append(hole_rec)
                frames.append(frame_rec)
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        patterns = _place_patterns(rng, grid, occ, all_holes)
        if patterns is None:
            continue

        expected = _expected_from_plan(grid, frames, patterns)
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid

    raise RuntimeError("generate_input: exhausted retries")
