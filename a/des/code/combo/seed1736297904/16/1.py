import random
from typing import Dict, List, Optional, Tuple

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GAP_TO_ROT_K = {"up": 0, "right": 3, "down": 2, "left": 1}


def _bbox(coords: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _color_components(grid: np.ndarray) -> List[Dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[Dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: List[Tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            r0, c0, r1, c1 = _bbox(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "height": r1 - r0 + 1,
                    "width": c1 - c0 + 1,
                }
            )
    return comps


def _classify_frame_component(grid: np.ndarray, comp: Dict[str, object]) -> Optional[Dict[str, object]]:
    color = int(comp["color"])
    coords = comp["coords"]
    r0, c0, r1, c1 = comp["bbox"]
    h = int(comp["height"])
    w = int(comp["width"])
    if h < 3 or w < 3:
        return None

    comp_set = set(coords)
    perimeter = set()
    for cc in range(c0, c1 + 1):
        perimeter.add((r0, cc))
        perimeter.add((r1, cc))
    for rr in range(r0, r1 + 1):
        perimeter.add((rr, c0))
        perimeter.add((rr, c1))
    if not comp_set.issubset(perimeter):
        return None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1] if h > 2 and w > 2 else np.zeros((0, 0), dtype=int)
    inner_other_colors = {int(v) for v in np.unique(inner) if int(v) not in (BG, color)}
    missing = perimeter - comp_set

    if not missing:
        if h < 5 or w < 5 or len(inner_other_colors) != 1:
            return None
        return {
            "kind": "stamp",
            "color": color,
            "pattern_color": next(iter(inner_other_colors)),
            "bbox": (r0, c0, r1, c1),
            "height": h,
            "width": w,
        }

    if len(missing) != 1 or h < 7 or w < 7 or np.any(inner != BG):
        return None

    gap = next(iter(missing))
    centers = {
        (r0, c0 + w // 2): "up",
        (r1, c0 + w // 2): "down",
        (r0 + h // 2, c0): "left",
        (r0 + h // 2, c1): "right",
    }
    gap_dir = centers.get(gap)
    if gap_dir is None:
        return None

    return {
        "kind": "container",
        "color": color,
        "gap_dir": gap_dir,
        "bbox": (r0, c0, r1, c1),
        "height": h,
        "width": w,
    }


def _extract_pattern(stamp: Dict[str, object], grid: np.ndarray) -> Tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = stamp["bbox"]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    pattern_color = int(stamp["pattern_color"])
    mask = patch == pattern_color
    coords = np.argwhere(mask)
    assert len(coords) > 0
    pr0, pc0 = coords.min(axis=0)
    pr1, pc1 = coords.max(axis=0)
    ph = int(pr1 - pr0 + 1)
    pw = int(pc1 - pc0 + 1)
    assert (ph, pw) == (3, 3)
    pattern = np.where(mask[pr0 : pr1 + 1, pc0 : pc1 + 1], pattern_color, BG)
    return pattern, int(pr0), int(pc0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each gapped frame to the same-colored stamp, rotate the stamp's 3x3
    inner pattern from the gap direction, remove the stamp, and place the
    rotated pattern inside the container cavity.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    containers: List[Dict[str, object]] = []
    stamps: List[Dict[str, object]] = []
    for comp in _color_components(g):
        info = _classify_frame_component(g, comp)
        if info is None:
            continue
        if info["kind"] == "container":
            containers.append(info)
        else:
            stamps.append(info)

    used_stamps = set()
    for container in containers:
        need_shape = (int(container["height"]) - 2, int(container["width"]) - 2)
        matches = [
            (idx, stamp)
            for idx, stamp in enumerate(stamps)
            if idx not in used_stamps
            and int(stamp["color"]) == int(container["color"])
            and (int(stamp["height"]), int(stamp["width"])) == need_shape
        ]
        assert len(matches) == 1
        stamp_idx, stamp = matches[0]
        used_stamps.add(stamp_idx)

        sr0, sc0, sr1, sc1 = stamp["bbox"]
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = BG

        pattern, rel_r, rel_c = _extract_pattern(stamp, g)
        rotated = np.rot90(pattern, GAP_TO_ROT_K[str(container["gap_dir"])])

        cr0, cc0, _cr1, _cc1 = container["bbox"]
        target_r = int(cr0) + 1 + rel_r
        target_c = int(cc0) + 1 + rel_c
        target = out[target_r : target_r + 3, target_c : target_c + 3]
        mask = rotated != BG
        target[mask] = rotated[mask]

    assert len(used_stamps) == len(stamps) == len(containers)
    return out


PAIR_COUNT_CHOICES = [1, 2, 3]
MAX_CANVAS = 30
PLACEMENT_PAD = 1
GEN_RETRIES = 300
PLACEMENT_RETRIES = 200
PATTERN_RETRIES = 100


def _rotations_all_distinct(mask: np.ndarray) -> bool:
    seen = set()
    for k in range(4):
        seen.add(np.rot90(mask, k).tobytes())
    return len(seen) == 4


def _random_pattern(rng: random.Random) -> np.ndarray:
    for _ in range(PATTERN_RETRIES):
        area = rng.randint(4, 6)
        cells = [(r, c) for r in range(3) for c in range(3)]
        start = rng.choice(cells)
        chosen = {start}
        frontier = set()
        for dr, dc in DIRS4:
            nr, nc = start[0] + dr, start[1] + dc
            if 0 <= nr < 3 and 0 <= nc < 3:
                frontier.add((nr, nc))
        while len(chosen) < area and frontier:
            nxt = rng.choice(tuple(frontier))
            frontier.remove(nxt)
            chosen.add(nxt)
            for dr, dc in DIRS4:
                nr, nc = nxt[0] + dr, nxt[1] + dc
                if 0 <= nr < 3 and 0 <= nc < 3 and (nr, nc) not in chosen:
                    frontier.add((nr, nc))
        if len(chosen) != area:
            continue
        mask = np.zeros((3, 3), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        if not (mask[0].any() and mask[2].any() and mask[:, 0].any() and mask[:, 2].any()):
            continue
        if not _rotations_all_distinct(mask):
            continue
        return mask
    raise RuntimeError("failed to generate asymmetric 3x3 pattern")


def _draw_border(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int, gap_dir: Optional[str]) -> None:
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color
    if gap_dir is None:
        return
    if gap_dir == "up":
        grid[r0, c0 + w // 2] = BG
    elif gap_dir == "down":
        grid[r0 + h - 1, c0 + w // 2] = BG
    elif gap_dir == "left":
        grid[r0 + h // 2, c0] = BG
    else:
        grid[r0 + h // 2, c0 + w - 1] = BG


def _boxes_clear(existing: List[Tuple[int, int, int, int]], cand: Tuple[int, int, int, int], pad: int) -> bool:
    r0, c0, r1, c1 = cand
    for er0, ec0, er1, ec1 in existing:
        if not (r1 + pad < er0 or er1 + pad < r0 or c1 + pad < ec0 or ec1 + pad < c0):
            return False
    return True


def _render_input(specs: List[Dict[str, object]], shape: Tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        cr0, cc0, ch, cw = spec["container"]
        sr0, sc0, sh, sw = spec["stamp"]
        _draw_border(grid, int(cr0), int(cc0), int(ch), int(cw), int(spec["frame_color"]), str(spec["gap_dir"]))
        _draw_border(grid, int(sr0), int(sc0), int(sh), int(sw), int(spec["frame_color"]), None)
        pr, pc = spec["pattern_rel"]
        mask = spec["pattern_mask"]
        pat_color = int(spec["pattern_color"])
        window = grid[int(sr0) + int(pr) : int(sr0) + int(pr) + 3, int(sc0) + int(pc) : int(sc0) + int(pc) + 3]
        window[mask] = pat_color
    return grid


def _expected_output(inp: np.ndarray, specs: List[Dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        sr0, sc0, sh, sw = spec["stamp"]
        out[int(sr0) : int(sr0) + int(sh), int(sc0) : int(sc0) + int(sw)] = BG
    for spec in specs:
        cr0, cc0, _ch, _cw = spec["container"]
        pr, pc = spec["pattern_rel"]
        pattern = np.where(spec["pattern_mask"], int(spec["pattern_color"]), BG)
        rotated = np.rot90(pattern, GAP_TO_ROT_K[str(spec["gap_dir"])])
        window = out[int(cr0) + 1 + int(pr) : int(cr0) + 4 + int(pr), int(cc0) + 1 + int(pc) : int(cc0) + 4 + int(pc)]
        mask = rotated != BG
        window[mask] = rotated[mask]
    return out


def _shift_specs(specs: List[Dict[str, object]], dr: int, dc: int) -> List[Dict[str, object]]:
    shifted = []
    for spec in specs:
        cr0, cc0, ch, cw = spec["container"]
        sr0, sc0, sh, sw = spec["stamp"]
        shifted.append(
            {
                "frame_color": int(spec["frame_color"]),
                "pattern_color": int(spec["pattern_color"]),
                "gap_dir": str(spec["gap_dir"]),
                "pattern_mask": np.array(spec["pattern_mask"], copy=True),
                "pattern_rel": tuple(spec["pattern_rel"]),
                "container": (int(cr0) - dr, int(cc0) - dc, int(ch), int(cw)),
                "stamp": (int(sr0) - dr, int(sc0) - dc, int(sh), int(sw)),
            }
        )
    return shifted


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    n_pairs = rng.choice(PAIR_COUNT_CHOICES)
    colors = rng.sample(list(range(1, 10)), 2 * n_pairs)
    frame_colors = colors[:n_pairs]
    pattern_colors = colors[n_pairs:]
    directions = rng.sample(["up", "right", "down", "left"], n_pairs)
    size_choices = [7, 9] if n_pairs == 3 else [7, 9, 11]

    specs: List[Dict[str, object]] = []
    for i in range(n_pairs):
        ch = rng.choice(size_choices)
        cw = rng.choice(size_choices)
        sh, sw = ch - 2, cw - 2
        pr = rng.randint(1, sh - 4)
        pc = rng.randint(1, sw - 4)
        specs.append(
            {
                "frame_color": frame_colors[i],
                "pattern_color": pattern_colors[i],
                "gap_dir": directions[i],
                "pattern_mask": _random_pattern(rng),
                "pattern_rel": (pr, pc),
                "container": (0, 0, ch, cw),
                "stamp": (0, 0, sh, sw),
            }
        )

    placed_boxes: List[Tuple[int, int, int, int]] = []
    place_order = []
    for pair_idx, spec in enumerate(specs):
        ch, cw = int(spec["container"][2]), int(spec["container"][3])
        sh, sw = int(spec["stamp"][2]), int(spec["stamp"][3])
        place_order.append((ch * cw, pair_idx, "container", ch, cw))
        place_order.append((sh * sw, pair_idx, "stamp", sh, sw))
    place_order.sort(reverse=True)

    placed = []
    for _area, pair_idx, kind, h, w in place_order:
        ok = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = rng.randint(1, MAX_CANVAS - h - 1)
            c0 = rng.randint(1, MAX_CANVAS - w - 1)
            cand = (r0, c0, r0 + h - 1, c0 + w - 1)
            if not _boxes_clear(placed_boxes, cand, PLACEMENT_PAD):
                continue
            placed_boxes.append(cand)
            if kind == "container":
                specs[pair_idx]["container"] = (r0, c0, h, w)
            else:
                specs[pair_idx]["stamp"] = (r0, c0, h, w)
            placed.append(cand)
            ok = True
            break
        if not ok:
            return None

    min_r = min(r0 for r0, _c0, _r1, _c1 in placed)
    min_c = min(c0 for _r0, c0, _r1, _c1 in placed)
    max_r = max(r1 for _r0, _c0, r1, _c1 in placed)
    max_c = max(c1 for _r0, _c0, _r1, c1 in placed)

    top = rng.randint(0, min_r - 1)
    left = rng.randint(0, min_c - 1)
    bottom = rng.randint(max_r + 1, MAX_CANVAS - 1)
    right = rng.randint(max_c + 1, MAX_CANVAS - 1)

    master = _render_input(specs, (MAX_CANVAS, MAX_CANVAS))
    inp = master[top : bottom + 1, left : right + 1]
    shifted_specs = _shift_specs(specs, top, left)
    expected = _expected_output(inp, shifted_specs)

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        inp = _gen_attempt(rng)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
