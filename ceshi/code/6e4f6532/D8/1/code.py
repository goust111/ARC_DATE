from __future__ import annotations

from collections import Counter, defaultdict
import importlib.util
from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import label

ROOT = Path("/home/lixinyu/snap/new")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import get_bg, paste


def _load_base_module():
    solver_path = ROOT / "arc2_solver_select" / "6e4f6532.py"
    spec = importlib.util.spec_from_file_location("arc_6e4f6532_base", solver_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load base solver from {solver_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE = _load_base_module()


def apply_transform(arr: np.ndarray, flip_ud: bool = False, rot_k: int = 0) -> np.ndarray:
    out = np.flipud(arr) if flip_ud else arr
    return np.rot90(out, int(rot_k))


def transform_coords(coords, shape, flip_ud: bool = False, rot_k: int = 0):
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    out = apply_transform(mask, flip_ud=flip_ud, rot_k=rot_k)
    return set(map(tuple, np.argwhere(out)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    D8 variant: one train may dock onto multiple equal-sized stations, creating copies.

    Design notes used to drive both solver and generator:
    - The original task hinges on a uniqueness assumption: a train's coupler count
      identifies exactly one station group. This variant changes only that part.
    - To keep the rule visually recoverable, the generator creates exactly one
      repeated station size and keeps source trains unique by coupler count. The
      repeated groups all share the same orientation cue, so every copy lands with
      the same transformed sprite.
    - The solver therefore replaces the old "exactly one target group" assertion
      with "one or more target groups of this size", computes the single allowed
      orientation, and pastes the transformed train at every matching station.
    """
    inp = grid
    grid = inp.copy()
    H, W = grid.shape
    bg = get_bg(grid)

    def find_edge_stripes():
        stripe = np.zeros((H, W), dtype=bool)
        colors = set()
        top_h = 0
        for r in range(H):
            row = grid[r, :]
            if (row == row[0]).all() and row[0] != bg:
                top_h += 1
            else:
                break
        if top_h:
            stripe[:top_h, :] = True
            colors.add(int(grid[0, 0]))
        bot_h = 0
        for r in range(H - 1, -1, -1):
            row = grid[r, :]
            if (row == row[0]).all() and row[0] != bg:
                bot_h += 1
            else:
                break
        if bot_h:
            stripe[H - bot_h :, :] = True
            colors.add(int(grid[H - 1, 0]))
        inner_r0, inner_r1 = top_h, H - bot_h
        left_w = 0
        for c in range(W):
            col = grid[inner_r0:inner_r1, c]
            if col.size == 0:
                break
            if (col == col[0]).all() and col[0] != bg:
                left_w += 1
            else:
                break
        if left_w:
            stripe[inner_r0:inner_r1, :left_w] = True
            colors.add(int(grid[inner_r0, 0]))
        right_w = 0
        for c in range(W - 1, -1, -1):
            col = grid[inner_r0:inner_r1, c]
            if col.size == 0:
                break
            if (col == col[0]).all() and col[0] != bg:
                right_w += 1
            else:
                break
        if right_w:
            stripe[inner_r0:inner_r1, W - right_w :] = True
            colors.add(int(grid[inner_r0, W - 1]))
        return stripe, colors, (top_h, bot_h, left_w, right_w)

    def find_internal_stripes(edge_info):
        top_h, bot_h, left_w, right_w = edge_info
        r0, r1, c0, c1 = top_h, H - bot_h, left_w, W - right_w
        stripe = np.zeros((H, W), dtype=bool)
        colors = set()
        c = c0
        while c < c1:
            col = grid[r0:r1, c]
            if col.size and (col == col[0]).all() and col[0] != bg:
                color = int(col[0])
                c2 = c
                while c2 + 1 < c1:
                    col2 = grid[r0:r1, c2 + 1]
                    if col2.size and (col2 == col2[0]).all() and int(col2[0]) == color:
                        c2 += 1
                    else:
                        break
                stripe[r0:r1, c : c2 + 1] = True
                colors.add(color)
                c = c2 + 1
            else:
                c += 1
        r = r0
        while r < r1:
            row = grid[r, c0:c1]
            if row.size and (row == row[0]).all() and row[0] != bg:
                color = int(row[0])
                r2 = r
                while r2 + 1 < r1:
                    row2 = grid[r2 + 1, c0:c1]
                    if row2.size and (row2 == row2[0]).all() and int(row2[0]) == color:
                        r2 += 1
                    else:
                        break
                stripe[r : r2 + 1, c0:c1] = True
                colors.add(color)
                r = r2 + 1
            else:
                r += 1
        return stripe, colors

    def station_groups():
        sm = np.zeros((H, W), dtype=bool)
        for r in range(H):
            for c in range(W):
                if grid[r, c] != 9:
                    continue
                ok = all(
                    grid[r + dr, c + dc] != 8
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if (dr != 0 or dc != 0) and 0 <= r + dr < H and 0 <= c + dc < W
                )
                if ok:
                    sm[r, c] = True
        lab, n = label(sm, structure=np.ones((3, 3), dtype=int))
        groups = []
        for k in range(1, n + 1):
            coords = np.argwhere(lab == k)
            groups.append([(int(r), int(c)) for r, c in coords])
        return groups

    def regions_from_stripes(stripe_mask):
        non = ~stripe_mask
        lab, n = label(non)
        regs = []
        for k in range(1, n + 1):
            coords = np.argwhere(lab == k)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            if (r1 - r0 + 1) >= 5 and (c1 - c0 + 1) >= 5:
                regs.append({"r0": int(r0), "r1": int(r1), "c0": int(c0), "c1": int(c1)})
        return regs

    def boundary_dirs_for_region(reg, stripe_colors):
        dirs = {}
        r0, r1, c0, c1 = reg["r0"], reg["r1"], reg["c0"], reg["c1"]
        if r0 - 1 >= 0:
            mc = Counter(grid[r0 - 1, c0:c1 + 1].tolist()).most_common(1)[0][0]
            if mc in stripe_colors:
                dirs[int(mc)] = 0
        if r1 + 1 < H:
            mc = Counter(grid[r1 + 1, c0:c1 + 1].tolist()).most_common(1)[0][0]
            if mc in stripe_colors:
                dirs[int(mc)] = 2
        if c0 - 1 >= 0:
            mc = Counter(grid[r0:r1 + 1, c0 - 1].tolist()).most_common(1)[0][0]
            if mc in stripe_colors:
                dirs[int(mc)] = 3
        if c1 + 1 < W:
            mc = Counter(grid[r0:r1 + 1, c1 + 1].tolist()).most_common(1)[0][0]
            if mc in stripe_colors:
                dirs[int(mc)] = 1
        return dirs

    def xform_coords(coords, shape, flip_ud, rot_k):
        result = transform_coords(coords, shape, flip_ud, rot_k)
        new_shape = apply_transform(np.zeros(shape), flip_ud, rot_k).shape
        return list(result), new_shape

    def choose_best_op(train, divider_dirs):
        bbox0 = train["bbox"]
        H0, W0 = bbox0.shape
        markers_by_color = defaultdict(list)
        for v, r, c in train["marker_pixels"]:
            if v in divider_dirs:
                markers_by_color[v].append((r, c))
        assert markers_by_color, "Each train must expose divider-colored markers"
        assert train["body9"], "Each train must carry at least one coupler"
        ref_r = sum(r for r, _ in train["body9"]) / len(train["body9"])
        ref_c = sum(c for _, c in train["body9"]) / len(train["body9"])
        best, best_score = (False, 0), -1
        for flip_ud in (False, True):
            for k in range(4):
                ref_t = list(
                    transform_coords(
                        [(int(round(ref_r)), int(round(ref_c)))], (H0, W0), flip_ud, k
                    )
                )
                rr_ref, cc_ref = ref_t[0]
                score = 0
                for color, pts in markers_by_color.items():
                    mr = sum(r for r, _ in pts) / len(pts)
                    mc = sum(c for _, c in pts) / len(pts)
                    m_t = list(
                        transform_coords(
                            [(int(round(mr)), int(round(mc)))], (H0, W0), flip_ud, k
                        )
                    )
                    rr_m, cc_m = m_t[0]
                    dr, dc = rr_m - rr_ref, cc_m - cc_ref
                    dirc = (0 if dr < 0 else 2) if abs(dr) >= abs(dc) else (3 if dc < 0 else 1)
                    if dirc == divider_dirs[color]:
                        score += 1
                if score > best_score:
                    best_score, best = score, (flip_ud, k)
        return best

    edge_mask, edge_colors, edge_info = find_edge_stripes()
    internal_mask, internal_colors = find_internal_stripes(edge_info)
    stripe_mask = edge_mask | internal_mask
    stripe_colors = edge_colors | internal_colors
    regions = regions_from_stripes(stripe_mask)

    groups = station_groups()
    groups_by_size = defaultdict(list)
    for g in groups:
        groups_by_size[len(g)].append(g)
    station_set = set(pt for g in groups for pt in g)

    comp_mask = (grid != bg) & (~stripe_mask)
    lab, n = label(comp_mask)
    trains = []
    for k in range(1, n + 1):
        coords = np.argwhere(lab == k)
        vals = [int(grid[r, c]) for r, c in coords]
        if 8 not in vals:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        bbox = grid[r0:r1 + 1, c0:c1 + 1].copy()
        abs_coords = [(int(r), int(c)) for r, c in coords]
        body9, marker_pixels = [], []
        for r, c in abs_coords:
            v = int(grid[r, c])
            if v == 9 and (r, c) not in station_set:
                body9.append((int(r - r0), int(c - c0)))
            if v in stripe_colors:
                marker_pixels.append((v, int(r - r0), int(c - c0)))
        trains.append(
            {
                "abs_coords": abs_coords,
                "bbox": bbox,
                "body9": body9,
                "marker_pixels": marker_pixels,
            }
        )

    out = grid.copy()
    for t in trains:
        for r, c in t["abs_coords"]:
            out[r, c] = bg

    for t in trains:
        n9 = len(t["body9"])
        assert n9 > 0, "Each train must carry at least one coupler"
        candidate_groups = groups_by_size.get(n9)
        assert candidate_groups, f"Expected at least one station group for coupler count {n9}"

        placements = []
        ops = set()
        for g in candidate_groups:
            matching_regions = [
                rr
                for rr in regions
                if all(rr["r0"] <= r <= rr["r1"] and rr["c0"] <= c <= rr["c1"] for r, c in g)
            ]
            assert len(matching_regions) == 1, "Each station group must belong to exactly one region"
            reg = matching_regions[0]
            divider_dirs = boundary_dirs_for_region(reg, stripe_colors)
            op = choose_best_op(t, divider_dirs)
            placements.append((g, reg, op))
            ops.add(op)

        assert len(ops) == 1, "All duplicated targets for one train must agree on orientation"
        op = placements[0][2]
        tbbox = apply_transform(t["bbox"], *op)
        t_body9, (H1, W1) = xform_coords(t["body9"], t["bbox"].shape, *op)
        body9_set = set(t_body9)

        for g, reg, _ in placements:
            gset = set(g)
            matching_offsets = {
                (sr - br, sc - bc)
                for br, bc in body9_set
                for sr, sc in gset
                if all((r + sr - br, c + sc - bc) in gset for r, c in body9_set)
            }
            assert len(matching_offsets) == 1, "Couplers must align to a unique station-group offset"
            off_r, off_c = next(iter(matching_offsets))
            ok = True
            for r in range(H1):
                for c in range(W1):
                    v = int(tbbox[r, c])
                    if v == bg or v == 9:
                        continue
                    rr, cc = r + off_r, c + off_c
                    if not (reg["r0"] <= rr <= reg["r1"] and reg["c0"] <= cc <= reg["c1"]):
                        ok = False
                        break
                    if stripe_mask[rr, cc] or (out[rr, cc] != bg and out[rr, cc] != 9):
                        ok = False
                        break
                if not ok:
                    break
            assert ok, "Docked train copy must fit entirely within its target region"
            for r in range(H1):
                for c in range(W1):
                    v = int(tbbox[r, c])
                    if v != bg and v != 9:
                        out[r + off_r, c + off_c] = v
    return out


def _train_specs():
    n_trains = random.choice(_BASE.TRAIN_COUNTS)
    sizes = random.sample(_BASE.STATION_SIZES, n_trains)
    repeat_idx = random.randrange(n_trains)
    specs = []
    for i, size in enumerate(sizes):
        specs.append({"size": size, "copies": 2 if i == repeat_idx else 1})
    specs.sort(key=lambda spec: spec["copies"], reverse=True)
    return specs


def _mask_for_patch(shape, patch, r0, c0, bg):
    mask = np.zeros(shape, dtype=bool)
    ph, pw = patch.shape
    mask[r0 : r0 + ph, c0 : c0 + pw] = patch != bg
    return mask


def _commit_patch(grid, occupied, patch, bg, location):
    r0, c0 = location
    paste(grid, patch, r0, c0, transparent=bg)
    return occupied | _mask_for_patch(grid.shape, patch, r0, c0, bg)


def _place_train_family(output, bg, occupied, regions, size, copies):
    H, W = output.shape
    for _ in range(_BASE.DOCK_RETRIES * 3):
        reg = random.choice(regions)
        result = _BASE._build_docked_patch(bg, reg["dirs"], size)
        if result is None:
            continue
        patch, coupler_rel = result
        current_occupied = occupied.copy()
        chosen = []
        for _ in range(copies):
            locs = _BASE._patch_locations(
                reg,
                patch,
                bg,
                _BASE._dilate(current_occupied),
                coupler_rel,
                True,
                0,
            )
            if not locs:
                break
            loc = random.choice(locs)
            chosen.append(loc)
            current_occupied = current_occupied | _mask_for_patch((H, W), patch, loc[0], loc[1], bg)
        if len(chosen) != copies:
            continue

        stations = []
        for loc in chosen:
            output_occ = _mask_for_patch((H, W), patch, loc[0], loc[1], bg)
            paste(output, patch, loc[0], loc[1], transparent=bg)
            occupied |= output_occ
            stations.append([(loc[0] + r, loc[1] + c) for r, c in coupler_rel])
        return {"patch": patch, "stations": stations}
    return None


def _place_docked_trains_variant(base_grid, bg, regions):
    output = base_grid.copy()
    occupied = output != bg
    trains = []
    for spec in _train_specs():
        placed = _place_train_family(output, bg, occupied, regions, spec["size"], spec["copies"])
        if placed is None:
            return None
        trains.append(placed)
        occupied = output != bg
    return output, trains


def _scatter_trains_variant(base_grid, bg, regions, trains):
    H, W = base_grid.shape
    input_grid = base_grid.copy()
    station_mask = np.zeros((H, W), dtype=bool)
    for t in trains:
        for station in t["stations"]:
            for r, c in station:
                input_grid[r, c] = 9
                station_mask[r, c] = True

    valid_ops = []
    for t in trains:
        valid = _BASE._valid_ops_for_patch(t["patch"])
        if not valid:
            return None
        valid_ops.append(valid)
    ops = [random.choice(valid) for valid in valid_ops]
    if all(op == (False, 0) for op in ops):
        candidates = [op for valid in valid_ops for op in valid if op != (False, 0)]
        if not candidates:
            return None
        idxs = [i for i, valid in enumerate(valid_ops) if any(op != (False, 0) for op in valid)]
        idx = random.choice(idxs)
        ops[idx] = random.choice([op for op in valid_ops[idx] if op != (False, 0)])

    forbidden = _BASE._dilate(base_grid != bg) | _BASE._dilate(station_mask)
    for t, op in zip(trains, ops):
        in_patch = _BASE._inverse_patch(t["patch"], op)
        regs = regions[:]
        random.shuffle(regs)
        all_locs = []
        for reg in regs:
            locs = _BASE._patch_locations(reg, in_patch, bg, forbidden, None, False, 1)
            all_locs.extend((reg, p) for p in locs)
        if not all_locs:
            return None
        _, (r0, c0) = random.choice(all_locs)
        paste(input_grid, in_patch, r0, c0, transparent=bg)
        forbidden |= _BASE._dilate(_mask_for_patch((H, W), in_patch, r0, c0, bg))
    return input_grid


def _gen_attempt():
    layout = _BASE._layout()
    if layout is None:
        return None
    base_grid, bg, regions = layout

    placed = _place_docked_trains_variant(base_grid, bg, regions)
    if placed is None:
        return None
    output, trains = placed

    input_grid = _scatter_trains_variant(base_grid, bg, regions, trains)
    if input_grid is None:
        return None

    counts = Counter(map(int, input_grid.flatten()))
    if counts[bg] != max(counts.values()):
        return None

    repeated_station_groups = 0
    station_sizes = Counter(len(station) for t in trains for station in t["stations"])
    repeated_station_groups = max(station_sizes.values(), default=0)
    if repeated_station_groups < 2:
        return None

    try:
        if np.array_equal(main(input_grid), output):
            return input_grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_BASE.GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 6e4f6532 D8-alt1 input")
